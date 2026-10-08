from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from pypdf import PdfReader

from app.database.connection import (
    close_connection_pool,
    get_db_connection,
    initialize_connection_pool,
)
from app.services.ocr_service import perform_ocr


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/home/mahesh/Documents/OCR_API")

PDF_DIR = PROJECT_ROOT / "test_pdf"

RESULTS_DIR = Path(
    os.getenv(
        "BENCHMARK_RESULTS_DIR",
        str(PROJECT_ROOT / "benchmark_results"),
    )
)

TEMP_DIR = RESULTS_DIR / "temp_pdfs"

CSV_FILE = RESULTS_DIR / "benchmark_worker_jobs.csv"
SUMMARY_FILE = RESULTS_DIR / "benchmark_summary.json"


# ============================================================
# BENCHMARK SETTINGS
# ============================================================

WORKER_ID = os.getenv(
    "BENCHMARK_WORKER_ID",
    "benchmark_single_worker",
)

QUEUE_MAX_SIZE = int(
    os.getenv("BENCHMARK_QUEUE_SIZE", "5")
)

MAX_RETRIES = int(
    os.getenv("BENCHMARK_MAX_RETRIES", "3")
)

OCR_CONCURRENCY = 1

EXPECTED_PDF_COUNT = 10


# ============================================================
# GLOBAL PROCESS STATE
# ============================================================

PROCESS = psutil.Process(os.getpid())

RESULTS_LOCK = threading.Lock()

RESULTS: list[dict[str, Any]] = []

QUEUE: asyncio.Queue | None = None

EVENT_LOOP: asyncio.AbstractEventLoop | None = None

WORKER_THREAD: threading.Thread | None = None

WORKER_STARTED = threading.Event()

SHUTDOWN_EVENT = threading.Event()


# ============================================================
# JOB MODEL
# ============================================================

@dataclass
class BenchmarkJob:
    job_id: str
    filename: str
    content_type: str
    temp_file_path: str
    file_size_bytes: int
    page_count: int
    submitted_at: float
    source_pdf_path: str


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_page_count(path: Path) -> int:
    reader = PdfReader(str(path))
    return len(reader.pages)


def get_rss_mb() -> float:
    return PROCESS.memory_info().rss / (1024 * 1024)


# ============================================================
# PEAK RSS MONITOR
# ============================================================

class PeakMemoryMonitor:
    """
    Samples process RSS while one OCR job is running.

    Because benchmark OCR concurrency is exactly 1,
    one monitor is sufficient.
    """

    def __init__(self, interval: float = 0.10) -> None:
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = PROCESS.memory_info().rss

                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss

            except Exception:
                pass

            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self.peak_rss_bytes = PROCESS.memory_info().rss

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            name="benchmark-memory-monitor",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        final_rss = PROCESS.memory_info().rss

        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            final_rss,
        )

        return self.peak_rss_bytes / (1024 * 1024)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def create_job_in_database(
    job_id: str,
    filename: str,
    content_type: str,
    file_path: str,
) -> None:

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ocr_jobs (
                    id,
                    filename,
                    content_type,
                    file_path,
                    status,
                    attempt_count,
                    worker_id
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    'queued',
                    0,
                    %s
                )
                """,
                (
                    job_id,
                    filename,
                    content_type,
                    file_path,
                    WORKER_ID,
                ),
            )

        conn.commit()


def mark_job_processing(job_id: str) -> int:
    """
    Atomically increment attempt_count and mark the job processing.

    Returns the current attempt number.
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'processing',
                    attempt_count = attempt_count + 1,
                    worker_id = %s,
                    started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                    updated_at = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE id = %s
                RETURNING attempt_count
                """,
                (
                    WORKER_ID,
                    job_id,
                ),
            )

            row = cursor.fetchone()

        conn.commit()

    if row is None:
        raise RuntimeError(
            f"Benchmark job {job_id} does not exist"
        )

    return int(row[0])


def mark_job_retrying(
    job_id: str,
    error_message: str,
) -> None:

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'retrying',
                    error_message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    error_message[:5000],
                    job_id,
                ),
            )

        conn.commit()


def mark_job_completed(job_id: str) -> None:

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'completed',
                    error_message = NULL,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (job_id,),
            )

        conn.commit()


def mark_job_failed(
    job_id: str,
    error_message: str,
) -> None:

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ocr_jobs
                SET
                    status = 'failed',
                    error_message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    error_message[:5000],
                    job_id,
                ),
            )

        conn.commit()


def save_ocr_result(
    job: BenchmarkJob,
    result: dict[str, Any],
    metrics: dict[str, Any],
) -> None:

    extracted_text = str(
        result.get("text", "")
    )

    confidence = float(
        result.get("confidence", 0.0)
    )

    result_json = {
        "pages": result.get("pages", []),
        "page_count": result.get(
            "page_count",
            job.page_count,
        ),
        "benchmark_metrics": metrics,
    }

    with get_db_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO ocr_results (
                    job_id,
                    filename,
                    content_type,
                    extracted_text,
                    confidence,
                    result_json
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb
                )
                ON CONFLICT (job_id)
                DO UPDATE SET
                    extracted_text = EXCLUDED.extracted_text,
                    confidence = EXCLUDED.confidence,
                    result_json = EXCLUDED.result_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    job.job_id,
                    job.filename,
                    job.content_type,
                    extracted_text,
                    confidence,
                    json.dumps(result_json),
                ),
            )

        conn.commit()


# ============================================================
# FILE CLEANUP
# ============================================================

def delete_temp_pdf(job: BenchmarkJob) -> None:
    path = Path(job.temp_file_path)

    try:
        if path.exists():
            path.unlink()

    except Exception as exc:
        print(
            f"[WARNING] Could not delete temporary PDF "
            f"{path}: {exc}"
        )


# ============================================================
# ONE OCR JOB
# ============================================================

def process_one_job(job: BenchmarkJob) -> dict[str, Any]:
    """
    Execute exactly one OCR job.

    This function is synchronous because perform_ocr()
    is synchronous and CPU-heavy.

    It is called through asyncio.to_thread().
    """

    attempt = mark_job_processing(job.job_id)

    processing_start_monotonic = time.perf_counter()

    processing_start_cpu = (
        time.process_time()
    )

    rss_before_mb = get_rss_mb()

    memory_monitor = PeakMemoryMonitor(
        interval=0.10
    )

    memory_monitor.start()

    error_message: str | None = None

    try:
        result = perform_ocr(
            job.temp_file_path
        )

        processing_end_monotonic = (
            time.perf_counter()
        )

        processing_end_cpu = (
            time.process_time()
        )

        wall_time_seconds = (
            processing_end_monotonic
            - processing_start_monotonic
        )

        cpu_time_seconds = (
            processing_end_cpu
            - processing_start_cpu
        )

        peak_rss_mb = memory_monitor.stop()

        cpu_percent = 0.0

        if wall_time_seconds > 0:
            cpu_percent = (
                cpu_time_seconds
                / wall_time_seconds
                * 100.0
            )

        metrics = {
            "worker_id": WORKER_ID,
            "attempt": attempt,

            "file_name": job.filename,
            "file_size_bytes": job.file_size_bytes,
            "file_size_mb": round(
                job.file_size_bytes / (1024 * 1024),
                4,
            ),

            "page_count": job.page_count,

            "submitted_at": datetime.fromtimestamp(
                job.submitted_at,
                tz=timezone.utc,
            ).isoformat(),

            "processing_started_at": utc_now(),

            "wall_time_seconds": round(
                wall_time_seconds,
                4,
            ),

            "cpu_time_seconds": round(
                cpu_time_seconds,
                4,
            ),

            "cpu_percent": round(
                cpu_percent,
                2,
            ),

            "rss_before_mb": round(
                rss_before_mb,
                2,
            ),

            "peak_rss_mb": round(
                peak_rss_mb,
                2,
            ),

            "rss_after_mb": round(
                get_rss_mb(),
                2,
            ),

            "queue_wait_seconds": round(
                processing_start_monotonic
                - job.submitted_at,
                4,
            ),

            "ocr_status": "completed",
        }

        save_ocr_result(
            job,
            result,
            metrics,
        )

        mark_job_completed(
            job.job_id
        )

        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": "completed",
            "attempt": attempt,
            **metrics,
        }

    except Exception as exc:

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        try:
            peak_rss_mb = memory_monitor.stop()
        except Exception:
            peak_rss_mb = get_rss_mb()

        if attempt < MAX_RETRIES:

            mark_job_retrying(
                job.job_id,
                error_message,
            )

            return {
                "job_id": job.job_id,
                "filename": job.filename,
                "status": "retrying",
                "attempt": attempt,
                "error": error_message,
                "peak_rss_mb": round(
                    peak_rss_mb,
                    2,
                ),
            }

        mark_job_failed(
            job.job_id,
            error_message,
        )

        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "status": "failed",
            "attempt": attempt,
            "error": error_message,
            "peak_rss_mb": round(
                peak_rss_mb,
                2,
            ),
        }


# ============================================================
# ASYNC QUEUE WORKER
# ============================================================

async def queue_worker() -> None:
    """
    The ONLY benchmark worker.

    OCR concurrency is intentionally exactly 1.
    """

    global QUEUE

    if QUEUE is None:
        raise RuntimeError(
            "Benchmark queue has not been initialized"
        )

    print(
        f"[BENCHMARK WORKER] Started: {WORKER_ID}"
    )

    print(
        f"[BENCHMARK WORKER] Queue size: "
        f"{QUEUE_MAX_SIZE}"
    )

    print(
        f"[BENCHMARK WORKER] OCR concurrency: "
        f"{OCR_CONCURRENCY}"
    )

    while not SHUTDOWN_EVENT.is_set():

        job = await QUEUE.get()

        try:

            print(
                f"\n[WORKER] Processing "
                f"{job.filename}"
            )

            result = await asyncio.to_thread(
                process_one_job,
                job,
            )

            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            if result["status"] == "retrying":

                print(
                    f"[RETRY] {job.filename} "
                    f"attempt {result['attempt']} "
                    f"failed"
                )

                await QUEUE.put(job)

                continue

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            with RESULTS_LOCK:
                RESULTS.append(result)

            print(
                f"[WORKER] {job.filename} -> "
                f"{result['status']}"
            )

            if result["status"] == "completed":

                print(
                    f"  OCR time : "
                    f"{result.get('wall_time_seconds', 0):.2f}s"
                )

                print(
                    f"  CPU time : "
                    f"{result.get('cpu_time_seconds', 0):.2f}s"
                )

                print(
                    f"  CPU %    : "
                    f"{result.get('cpu_percent', 0):.2f}%"
                )

                print(
                    f"  Peak RSS : "
                    f"{result.get('peak_rss_mb', 0):.2f} MB"
                )

        finally:

            delete_temp_pdf(job)

            QUEUE.task_done()


# ============================================================
# ASYNCIO BACKGROUND LOOP
# ============================================================

def _run_async_loop() -> None:
    global EVENT_LOOP
    global QUEUE

    EVENT_LOOP = asyncio.new_event_loop()

    asyncio.set_event_loop(EVENT_LOOP)

    QUEUE = asyncio.Queue(
        maxsize=QUEUE_MAX_SIZE
    )

    EVENT_LOOP.create_task(
        queue_worker()
    )

    WORKER_STARTED.set()

    try:
        EVENT_LOOP.run_forever()

    finally:

        pending = asyncio.all_tasks(
            EVENT_LOOP
        )

        for task in pending:
            task.cancel()

        EVENT_LOOP.close()


def start_benchmark_worker() -> None:
    """
    Start exactly one background asyncio worker.
    """

    global WORKER_THREAD

    if WORKER_THREAD is not None:
        return

    ensure_directories()

    initialize_connection_pool()

    SHUTDOWN_EVENT.clear()

    WORKER_THREAD = threading.Thread(
        target=_run_async_loop,
        name="benchmark-async-worker",
        daemon=True,
    )

    WORKER_THREAD.start()

    if not WORKER_STARTED.wait(timeout=10):
        raise RuntimeError(
            "Benchmark asyncio worker failed to start"
        )

    print(
        "[BENCHMARK] Asyncio worker ready"
    )


# ============================================================
# SUBMIT JOB
# ============================================================

def submit_job(
    source_pdf: str | Path,
) -> dict[str, Any]:
    """
    Submit a PDF to the benchmark queue.

    The actual PDF bytes are copied to disk.

    Only metadata/path is placed into asyncio.Queue.
    """

    if EVENT_LOOP is None:
        raise RuntimeError(
            "Benchmark worker is not running"
        )

    if QUEUE is None:
        raise RuntimeError(
            "Benchmark queue is not initialized"
        )

    source_path = Path(source_pdf)

    if not source_path.exists():
        raise FileNotFoundError(
            source_path
        )

    job_id = str(uuid.uuid4())

    filename = source_path.name

    temp_file = (
        TEMP_DIR
        / f"{job_id}_{filename}"
    )

    shutil.copy2(
        source_path,
        temp_file,
    )

    file_size_bytes = (
        temp_file.stat().st_size
    )

    page_count = get_pdf_page_count(
        temp_file
    )

    submitted_at = time.perf_counter()

    create_job_in_database(
        job_id=job_id,
        filename=filename,
        content_type="application/pdf",
        file_path=str(temp_file),
    )

    job = BenchmarkJob(
        job_id=job_id,
        filename=filename,
        content_type="application/pdf",
        temp_file_path=str(temp_file),
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        submitted_at=submitted_at,
        source_pdf_path=str(source_path),
    )

    # --------------------------------------------------------
    # asyncio.Queue.put() is thread-safe through
    # run_coroutine_threadsafe().
    #
    # If queue is full, this future waits until a slot
    # becomes available.
    # --------------------------------------------------------

    future = asyncio.run_coroutine_threadsafe(
        QUEUE.put(job),
        EVENT_LOOP,
    )

    future.result()

    return {
        "job_id": job_id,
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "page_count": page_count,
        "submitted_at": submitted_at,
    }


# ============================================================
# WAIT FOR JOB RESULT
# ============================================================

def get_result(
    job_id: str,
) -> dict[str, Any] | None:

    with RESULTS_LOCK:

        for result in RESULTS:

            if result.get("job_id") == job_id:
                return result

    return None


def wait_for_result(
    job_id: str,
    timeout: float = 1800.0,
) -> dict[str, Any]:

    deadline = (
        time.monotonic()
        + timeout
    )

    while time.monotonic() < deadline:

        result = get_result(job_id)

        if result is not None:
            return result

        time.sleep(0.05)

    raise TimeoutError(
        f"Timed out waiting for benchmark job "
        f"{job_id}"
    )


# ============================================================
# RESULT FILES
# ============================================================

def write_results_csv() -> None:

    ensure_directories()

    with RESULTS_LOCK:
        rows = list(RESULTS)

    if not rows:
        return

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def write_summary() -> dict[str, Any]:

    with RESULTS_LOCK:
        rows = list(RESULTS)

    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
    ]

    failed = [
        row
        for row in rows
        if row.get("status") == "failed"
    ]

    total_wall = sum(
        row.get(
            "wall_time_seconds",
            0.0,
        )
        for row in completed
    )

    total_cpu = sum(
        row.get(
            "cpu_time_seconds",
            0.0,
        )
        for row in completed
    )

    total_pages = sum(
        row.get(
            "page_count",
            0,
        )
        for row in completed
    )

    total_bytes = sum(
        row.get(
            "file_size_bytes",
            0,
        )
        for row in completed
    )

    summary = {
        "worker_id": WORKER_ID,
        "queue_max_size": QUEUE_MAX_SIZE,
        "ocr_concurrency": OCR_CONCURRENCY,
        "max_retries": MAX_RETRIES,

        "total_jobs": len(rows),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),

        "total_pages": total_pages,

        "total_input_mb": round(
            total_bytes / (1024 * 1024),
            4,
        ),

        "total_ocr_wall_time_seconds": round(
            total_wall,
            4,
        ),

        "total_cpu_time_seconds": round(
            total_cpu,
            4,
        ),

        "average_document_time_seconds": (
            round(
                total_wall / len(completed),
                4,
            )
            if completed
            else 0.0
        ),

        "average_page_time_seconds": (
            round(
                total_wall / total_pages,
                4,
            )
            if total_pages
            else 0.0
        ),

        "average_cpu_percent": (
            round(
                sum(
                    row.get(
                        "cpu_percent",
                        0.0,
                    )
                    for row in completed
                )
                / len(completed),
                2,
            )
            if completed
            else 0.0
        ),

        "average_peak_rss_mb": (
            round(
                sum(
                    row.get(
                        "peak_rss_mb",
                        0.0,
                    )
                    for row in completed
                )
                / len(completed),
                2,
            )
            if completed
            else 0.0
        ),

        "max_peak_rss_mb": (
            round(
                max(
                    row.get(
                        "peak_rss_mb",
                        0.0,
                    )
                    for row in completed
                ),
                2,
            )
            if completed
            else 0.0
        ),

        "generated_at": utc_now(),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown_benchmark_worker() -> None:

    global WORKER_THREAD

    SHUTDOWN_EVENT.set()

    if EVENT_LOOP is not None:

        EVENT_LOOP.call_soon_threadsafe(
            EVENT_LOOP.stop
        )

    if WORKER_THREAD is not None:

        WORKER_THREAD.join(
            timeout=5
        )

        WORKER_THREAD = None

    try:
        write_results_csv()
        write_summary()

    except Exception as exc:

        print(
            f"[WARNING] Could not write "
            f"benchmark results: {exc}"
        )

    try:
        close_connection_pool()

    except Exception as exc:

        print(
            f"[WARNING] Could not close "
            f"PostgreSQL pool: {exc}"
        )


# ============================================================
# DIRECT TEST MODE
# ============================================================

def run_direct_benchmark() -> None:
    """
    Optional direct test.

    This is NOT Locust.

    It processes the 10 PDFs through the same
    benchmark queue architecture.
    """

    ensure_directories()

    start_benchmark_worker()

    pdfs = sorted(
        PDF_DIR.glob("ocr_test_*.pdf")
    )

    if len(pdfs) != EXPECTED_PDF_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_PDF_COUNT} PDFs, "
            f"found {len(pdfs)}"
        )

    submitted_jobs = []

    print()
    print("=" * 70)
    print("DIRECT BENCHMARK")
    print("=" * 70)
    print(
        f"PDFs              : {len(pdfs)}"
    )
    print(
        f"Queue capacity     : {QUEUE_MAX_SIZE}"
    )
    print(
        f"OCR concurrency    : {OCR_CONCURRENCY}"
    )
    print(
        f"Max retries        : {MAX_RETRIES}"
    )
    print(
        f"PaddleOCR engines  : 1"
    )
    print(
        f"Worker             : {WORKER_ID}"
    )
    print("=" * 70)

    for pdf in pdfs:

        info = submit_job(pdf)

        submitted_jobs.append(info)

        print(
            f"[SUBMIT] {pdf.name} "
            f"job_id={info['job_id']}"
        )

    for info in submitted_jobs:

        result = wait_for_result(
            info["job_id"]
        )

        print(
            f"[RESULT] "
            f"{result['filename']} -> "
            f"{result['status']}"
        )

    summary = write_summary()

    write_results_csv()

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Total jobs       : "
        f"{summary['total_jobs']}"
    )

    print(
        f"Completed        : "
        f"{summary['completed_jobs']}"
    )

    print(
        f"Failed           : "
        f"{summary['failed_jobs']}"
    )

    print(
        f"Total pages      : "
        f"{summary['total_pages']}"
    )

    print(
        f"Total OCR time   : "
        f"{summary['total_ocr_wall_time_seconds']:.2f}s"
    )

    print(
        f"Average document : "
        f"{summary['average_document_time_seconds']:.2f}s"
    )

    print(
        f"Average page     : "
        f"{summary['average_page_time_seconds']:.2f}s"
    )

    print(
        f"Average CPU      : "
        f"{summary['average_cpu_percent']:.2f}%"
    )

    print(
        f"Average peak RSS : "
        f"{summary['average_peak_rss_mb']:.2f} MB"
    )

    print(
        f"Max peak RSS     : "
        f"{summary['max_peak_rss_mb']:.2f} MB"
    )

    print()
    print(
        f"CSV     : {CSV_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )


if __name__ == "__main__":

    try:
        run_direct_benchmark()

    finally:
        shutdown_benchmark_worker()