from __future__ import annotations

import csv
import os
import signal
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from pypdf import PdfReader

from app.database.connection import (
    close_connection_pool,
    initialize_connection_pool,
)
from app.worker.job_processor import process_message
from app.worker.redis_consumer import (
    OCR_CONSUMER_GROUP,
    OCR_STREAM_NAME,
    create_consumer_group,
    redis_client,
)
from app.worker.retry_handler import handle_failed_message


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Exactly TWO OCR jobs may be actively processed by this
# benchmark worker.
CONCURRENCY = int(
    os.getenv("BENCHMARK_CONCURRENCY", "2")
)

# Redis waits for new jobs for this amount of time.
REDIS_BLOCK_MS = int(
    os.getenv("BENCHMARK_REDIS_BLOCK_MS", "1000")
)

# Redis will return at most this many jobs in one read.
# We keep it equal to the number of available slots.
REDIS_COUNT = int(
    os.getenv("BENCHMARK_REDIS_COUNT", "2")
)

# Fixed benchmark worker identity.
#
# This is intentionally NOT the hostname because we want
# the worker identity to be obvious in PostgreSQL.
BENCHMARK_WORKER_ID = os.getenv(
    "BENCHMARK_WORKER_ID",
    "benchmark_single_worker",
)

# Output directory.
RESULTS_DIR = Path(
    os.getenv(
        "BENCHMARK_RESULTS_DIR",
        "/app/benchmark_results",
    )
)

CSV_FILE = RESULTS_DIR / "benchmark_worker_jobs.csv"

# Worker stop event.
STOP_EVENT = threading.Event()


# ============================================================
# PROCESS RESOURCE MONITOR
# ============================================================

PROCESS = psutil.Process(os.getpid())

# Used to calculate CPU time per benchmark job.
PROCESS_RESOURCE_LOCK = threading.Lock()


# ============================================================
# METRICS
# ============================================================

RESULTS_LOCK = threading.Lock()

RESULTS: list[dict[str, Any]] = []


# ============================================================
# SIGNAL HANDLING
# ============================================================

def handle_shutdown_signal(
    signum: int,
    frame: Any,
) -> None:

    print()
    print("=" * 70)
    print(
        f"Shutdown signal received: {signum}"
    )
    print(
        "Benchmark worker will stop accepting new jobs."
    )
    print(
        "Currently running OCR jobs will finish."
    )
    print("=" * 70)

    STOP_EVENT.set()


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def get_pdf_page_count(
    file_path: str,
) -> int:

    try:
        reader = PdfReader(file_path)
        return len(reader.pages)

    except Exception as exc:
        print(
            f"Could not read PDF page count: "
            f"{file_path} ({exc})"
        )
        return 0


def get_file_size_bytes(
    file_path: str,
) -> int:

    try:
        return Path(file_path).stat().st_size

    except OSError:
        return 0


def get_process_cpu_time() -> float:
    """
    Return process CPU time.

    This includes user + system CPU time used by the
    benchmark worker process.
    """

    cpu_times = PROCESS.cpu_times()

    return (
        cpu_times.user
        + cpu_times.system
    )


def get_process_rss_mb() -> float:
    """
    Return current benchmark worker RSS in MB.
    """

    return (
        PROCESS.memory_info().rss
        / (1024 * 1024)
    )


# ============================================================
# PROCESS ONE REDIS JOB
# ============================================================

def process_benchmark_message(
    message_id: str,
    data: dict[str, str],
) -> dict[str, Any]:

    job_id = data["job_id"]

    file_path = data["file_path"]

    filename = data.get(
        "filename",
        Path(file_path).name,
    )

    thread_name = threading.current_thread().name

    file_size_bytes = get_file_size_bytes(
        file_path
    )

    file_size_mb = (
        file_size_bytes
        / (1024 * 1024)
    )

    page_count = get_pdf_page_count(
        file_path
    ) if Path(file_path).suffix.lower() == ".pdf" else 1

    started_at = iso_now()

    wall_start = time.perf_counter()

    cpu_start = get_process_cpu_time()

    rss_before_mb = get_process_rss_mb()

    peak_rss_mb = rss_before_mb

    status = "completed"

    error_message = ""

    print()
    print("=" * 75)
    print(
        "BENCHMARK JOB START"
    )
    print("=" * 75)
    print(
        f"Job ID       : {job_id}"
    )
    print(
        f"Redis ID     : {message_id}"
    )
    print(
        f"Filename     : {filename}"
    )
    print(
        f"File size    : {file_size_mb:.3f} MB"
    )
    print(
        f"Pages        : {page_count}"
    )
    print(
        f"Thread       : {thread_name}"
    )
    print(
        f"Worker       : {BENCHMARK_WORKER_ID}"
    )
    print("=" * 75)

    try:

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # We intentionally reuse the REAL application
        # process_message().
        #
        # process_message() calls:
        #
        #     perform_ocr(file_path)
        #
        # perform_ocr() gets the single shared PaddleOCR
        # engine from app.services.ocr_service.
        #
        # Therefore this benchmark does NOT create another
        # OCR engine.
        # ----------------------------------------------------

        process_message(
            message_id=message_id,
            data=data,
        )

    except Exception as exc:

        status = "failed"

        error_message = str(exc)

        print()
        print(
            f"Benchmark job failed: {job_id}"
        )
        print(
            f"Error: {type(exc).__name__}: {exc}"
        )

        # Use the SAME retry/failure mechanism as the
        # production worker.
        try:

            handle_failed_message(
                message_id=message_id,
                data=data,
                exc=exc,
            )

        except Exception as failure_handler_error:

            print(
                "Failure handler itself failed: "
                f"{failure_handler_error}"
            )

    finally:

        wall_time = (
            time.perf_counter()
            - wall_start
        )

        cpu_time = (
            get_process_cpu_time()
            - cpu_start
        )

        rss_after_mb = get_process_rss_mb()

        peak_rss_mb = max(
            peak_rss_mb,
            rss_after_mb,
        )

        finished_at = iso_now()

        result = {
            "job_id": job_id,
            "redis_message_id": message_id,
            "filename": filename,
            "file_path": file_path,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(
                file_size_mb,
                4,
            ),
            "page_count": page_count,
            "worker_id": BENCHMARK_WORKER_ID,
            "thread_name": thread_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_time_seconds": round(
                wall_time,
                3,
            ),
            "cpu_time_seconds": round(
                cpu_time,
                3,
            ),
            "rss_before_mb": round(
                rss_before_mb,
                2,
            ),
            "rss_after_mb": round(
                rss_after_mb,
                2,
            ),
            "peak_rss_mb": round(
                peak_rss_mb,
                2,
            ),
            "status": status,
            "error_message": error_message,
        }

        with RESULTS_LOCK:
            RESULTS.append(result)

        print()
        print("=" * 75)
        print(
            "BENCHMARK JOB FINISHED"
        )
        print("=" * 75)
        print(
            f"Job ID       : {job_id}"
        )
        print(
            f"Filename     : {filename}"
        )
        print(
            f"Status       : {status}"
        )
        print(
            f"Thread       : {thread_name}"
        )
        print(
            f"Wall time    : {wall_time:.2f}s"
        )
        print(
            f"CPU time     : {cpu_time:.2f}s"
        )
        print(
            f"RSS before   : {rss_before_mb:.2f} MB"
        )
        print(
            f"RSS after    : {rss_after_mb:.2f} MB"
        )
        print(
            f"Error        : {error_message}"
        )
        print("=" * 75)

        return result


# ============================================================
# SAVE RESULTS
# ============================================================

def write_results() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_LOCK:

        rows = list(RESULTS)

    if not rows:
        print(
            "No benchmark results to write."
        )
        return

    fieldnames = [
        "job_id",
        "redis_message_id",
        "filename",
        "file_path",
        "file_size_bytes",
        "file_size_mb",
        "page_count",
        "worker_id",
        "thread_name",
        "started_at",
        "finished_at",
        "wall_time_seconds",
        "cpu_time_seconds",
        "rss_before_mb",
        "rss_after_mb",
        "peak_rss_mb",
        "status",
        "error_message",
    ]

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda item: item["filename"],
        ):
            writer.writerow(row)

    print()
    print(
        f"Benchmark CSV written to: {CSV_FILE}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary() -> None:

    with RESULTS_LOCK:
        rows = list(RESULTS)

    if not rows:
        return

    completed = [
        row
        for row in rows
        if row["status"] == "completed"
    ]

    failed = [
        row
        for row in rows
        if row["status"] == "failed"
    ]

    total_mb = sum(
        row["file_size_mb"]
        for row in rows
    )

    total_wall = sum(
        row["wall_time_seconds"]
        for row in rows
    )

    average_wall = (
        total_wall / len(rows)
        if rows
        else 0.0
    )

    max_peak_rss = max(
        row["peak_rss_mb"]
        for row in rows
    )

    print()
    print()
    print("=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)

    print(
        f"Worker                    : "
        f"{BENCHMARK_WORKER_ID}"
    )

    print(
        f"Executor slots            : "
        f"{CONCURRENCY}"
    )

    print(
        f"PaddleOCR engines        : 1"
    )

    print(
        f"Total jobs processed      : "
        f"{len(rows)}"
    )

    print(
        f"Completed                 : "
        f"{len(completed)}"
    )

    print(
        f"Failed                    : "
        f"{len(failed)}"
    )

    print(
        f"Total PDF data            : "
        f"{total_mb:.3f} MB"
    )

    print(
        f"Average job wall time     : "
        f"{average_wall:.2f}s"
    )

    print(
        f"Peak benchmark RSS        : "
        f"{max_peak_rss:.2f} MB"
    )

    print(
        f"Results CSV               : "
        f"{CSV_FILE}"
    )

    print("=" * 75)


# ============================================================
# MAIN REDIS WORKER
# ============================================================

def main() -> None:

    print()
    print("=" * 75)
    print("OCR BENCHMARK WORKER")
    print("=" * 75)
    print(
        f"Worker ID             : {BENCHMARK_WORKER_ID}"
    )
    print(
        f"Redis stream          : {OCR_STREAM_NAME}"
    )
    print(
        f"Redis consumer group  : {OCR_CONSUMER_GROUP}"
    )
    print(
        f"Executor slots        : {CONCURRENCY}"
    )
    print(
        f"PaddleOCR engines     : 1"
    )
    print(
        f"Redis block           : {REDIS_BLOCK_MS} ms"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Signal handling
    # --------------------------------------------------------

    signal.signal(
        signal.SIGINT,
        handle_shutdown_signal,
    )

    signal.signal(
        signal.SIGTERM,
        handle_shutdown_signal,
    )

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    initialize_connection_pool()

    executor = ThreadPoolExecutor(
        max_workers=CONCURRENCY,
        thread_name_prefix="benchmark-ocr",
    )

    active_futures: dict[
        Future,
        tuple[str, dict[str, str]],
    ] = {}

    try:

        # ----------------------------------------------------
        # Redis consumer group
        # ----------------------------------------------------

        create_consumer_group()

        print()
        print(
            "Benchmark worker is waiting for Redis jobs..."
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT call:
        #
        #     recover_pending_messages()
        #
        # here.
        #
        # This benchmark should process only NEW jobs created
        # by the current 10-user Locust test.
        #
        # We don't want the benchmark worker to unexpectedly
        # take old production jobs.
        # ----------------------------------------------------

        while not STOP_EVENT.is_set():

            # ------------------------------------------------
            # Remove completed futures
            # ------------------------------------------------

            completed_futures = [
                future
                for future in active_futures
                if future.done()
            ]

            for future in completed_futures:

                message_id, data = active_futures.pop(
                    future
                )

                try:
                    future.result()

                except Exception as exc:

                    print(
                        "Unexpected benchmark future error: "
                        f"{exc}"
                    )

            # ------------------------------------------------
            # Determine available worker slots
            # ------------------------------------------------

            available_slots = (
                CONCURRENCY
                - len(active_futures)
            )

            if available_slots <= 0:

                done, _ = wait(
                    active_futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:

                    message_id, data = active_futures.pop(
                        future
                    )

                    try:
                        future.result()

                    except Exception as exc:

                        print(
                            "Benchmark future error: "
                            f"{exc}"
                        )

                continue

            # ------------------------------------------------
            # Read NEW Redis jobs
            # ------------------------------------------------

            response = redis_client.xreadgroup(
                groupname=OCR_CONSUMER_GROUP,
                consumername=BENCHMARK_WORKER_ID,
                streams={
                    OCR_STREAM_NAME: ">"
                },
                count=available_slots,
                block=REDIS_BLOCK_MS,
            )

            if not response:
                continue

            # ------------------------------------------------
            # Submit jobs to ThreadPoolExecutor
            # ------------------------------------------------

            for stream_name, messages in response:

                for message_id, data in messages:

                    if (
                        len(active_futures)
                        >= CONCURRENCY
                    ):
                        break

                    future = executor.submit(
                        process_benchmark_message,
                        message_id,
                        data,
                    )

                    active_futures[future] = (
                        message_id,
                        data,
                    )

                    print(
                        f"Redis job received: "
                        f"{message_id}"
                    )

                    print(
                        f"Active OCR slots: "
                        f"{len(active_futures)}/"
                        f"{CONCURRENCY}"
                    )

    finally:

        print()
        print(
            "Stopping benchmark worker..."
        )

        # ----------------------------------------------------
        # Wait for active OCR jobs.
        # ----------------------------------------------------

        if active_futures:

            print(
                f"Waiting for "
                f"{len(active_futures)} active OCR job(s)..."
            )

            wait(
                active_futures,
            )

            for future in list(
                active_futures
            ):

                try:
                    future.result()

                except Exception as exc:

                    print(
                        f"Active job error: {exc}"
                    )

        executor.shutdown(
            wait=True
        )

        write_results()

        print_summary()

        close_connection_pool()

        print(
            "Benchmark worker stopped."
        )


if __name__ == "__main__":
    main()