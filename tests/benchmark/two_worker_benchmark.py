from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import queue
import resource
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import psutil
import redis
from pypdf import PdfReader

from app.core.config import get_settings
from app.services.ocr_service import get_ocr_engine, perform_ocr


# ============================================================
# CONFIGURATION
# ============================================================

PDF_DIR = Path(
    os.getenv(
        "BENCHMARK_PDF_DIR",
        "/home/mahesh/Documents/OCR_API/test_pdf",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "BENCHMARK_OUTPUT_DIR",
        "benchmark_results",
    )
)

PDF_COUNT = 10

STREAM_PREFIX = "ocr_benchmark_jobs"
GROUP_PREFIX = "ocr_benchmark_workers"

REDIS_BLOCK_MS = 1000


# ============================================================
# TIME
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# REDIS
# ============================================================

def create_redis_client() -> redis.Redis:
    """
    Create a Redis connection for the benchmark.

    The benchmark runs outside Docker by default, so Redis
    defaults to localhost:6379.

    Environment variables can override this.
    """

    settings = get_settings()

    redis_host = os.getenv(
        "BENCHMARK_REDIS_HOST",
        "localhost",
    )

    redis_port = int(
        os.getenv(
            "BENCHMARK_REDIS_PORT",
            "6379",
        )
    )

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
    )


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def collect_documents() -> list[dict]:
    """
    Discover benchmark PDFs and collect metadata before OCR.
    """

    pdf_paths = sorted(
        PDF_DIR.glob("ocr_test_*.pdf")
    )[:PDF_COUNT]

    if len(pdf_paths) != PDF_COUNT:
        raise RuntimeError(
            f"Expected {PDF_COUNT} PDFs in "
            f"{PDF_DIR}, but found {len(pdf_paths)}."
        )

    documents = []

    for pdf_path in pdf_paths:

        file_size_bytes = pdf_path.stat().st_size

        reader = PdfReader(
            str(pdf_path)
        )

        page_count = len(
            reader.pages
        )

        documents.append(
            {
                "job_id": str(uuid.uuid4()),
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "size_bytes": file_size_bytes,
                "size_mb": (
                    file_size_bytes
                    / (1024 * 1024)
                ),
                "page_count": page_count,
            }
        )

    return documents


# ============================================================
# BENCHMARK WORKER
# ============================================================

def benchmark_worker(
    worker_number: int,
    stream_name: str,
    consumer_group: str,
    result_queue: mp.Queue,
    ready_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:

    worker_id = (
        f"benchmark-worker-{worker_number}"
    )

    redis_client = create_redis_client()

    process = psutil.Process(
        os.getpid()
    )

    try:

        # ----------------------------------------------------
        # Create Redis consumer group
        # ----------------------------------------------------

        try:

            redis_client.xgroup_create(
                name=stream_name,
                groupname=consumer_group,
                id="0",
                mkstream=True,
            )

        except redis.exceptions.ResponseError as exc:

            if "BUSYGROUP" not in str(exc):
                raise

        # ----------------------------------------------------
        # PaddleOCR warm-up
        # ----------------------------------------------------

        warmup_start = time.perf_counter()

        get_ocr_engine()

        warmup_seconds = (
            time.perf_counter()
            - warmup_start
        )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "warmup_seconds": warmup_seconds,
            }
        )

        # ----------------------------------------------------
        # Benchmark loop
        # ----------------------------------------------------

        while not stop_event.is_set():

            messages = redis_client.xreadgroup(
                groupname=consumer_group,
                consumername=worker_id,
                streams={
                    stream_name: ">"
                },
                count=1,
                block=REDIS_BLOCK_MS,
            )

            if not messages:
                continue

            for _, entries in messages:

                for message_id, data in entries:

                    if stop_event.is_set():
                        break

                    job_id = data["job_id"]
                    file_path = data["path"]

                    # ------------------------------------------------
                    # BEFORE OCR METRICS
                    # ------------------------------------------------

                    start_time = utc_now()

                    wall_start = time.perf_counter()

                    cpu_start = time.process_time()

                    cpu_percent_samples = []

                    # Prime psutil CPU measurement.
                    process.cpu_percent(
                        interval=None
                    )

                    ram_before_mb = (
                        process.memory_info().rss
                        / (1024 * 1024)
                    )

                    peak_rss_mb = ram_before_mb

                    status = "success"
                    error_message = ""

                    confidence = 0.0
                    page_count = int(
                        data.get(
                            "page_count",
                            0,
                        )
                    )

                    try:

                        # ------------------------------------------------
                        # OCR
                        # ------------------------------------------------

                        ocr_result = perform_ocr(
                            file_path
                        )

                        confidence = float(
                            ocr_result.get(
                                "confidence",
                                0.0,
                            )
                        )

                        page_count = int(
                            ocr_result.get(
                                "page_count",
                                page_count,
                            )
                        )

                        # ------------------------------------------------
                        # CPU SAMPLE
                        # ------------------------------------------------

                        cpu_percent = process.cpu_percent(
                            interval=None
                        )

                        cpu_percent_samples.append(
                            cpu_percent
                        )

                        # ------------------------------------------------
                        # RAM SAMPLE
                        # ------------------------------------------------

                        current_rss_mb = (
                            process.memory_info().rss
                            / (1024 * 1024)
                        )

                        peak_rss_mb = max(
                            peak_rss_mb,
                            current_rss_mb,
                        )

                    except Exception as exc:

                        status = "failed"

                        error_message = (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                    # ------------------------------------------------
                    # FINAL METRICS
                    # ------------------------------------------------

                    wall_time_seconds = (
                        time.perf_counter()
                        - wall_start
                    )

                    cpu_time_seconds = (
                        time.process_time()
                        - cpu_start
                    )

                    ram_after_mb = (
                        process.memory_info().rss
                        / (1024 * 1024)
                    )

                    peak_rss_kb = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss

                    linux_peak_rss_mb = (
                        peak_rss_kb / 1024
                    )

                    peak_rss_mb = max(
                        peak_rss_mb,
                        linux_peak_rss_mb,
                    )

                    # CPU percentage over the OCR operation.
                    #
                    # This is derived from process CPU time divided
                    # by wall-clock time and normalized by CPU count.
                    #
                    # For a multi-threaded process this represents
                    # approximate CPU utilization relative to the
                    # available logical CPUs.

                    cpu_count = psutil.cpu_count(
                        logical=True
                    ) or 1

                    cpu_avg_percent = (
                        (
                            cpu_time_seconds
                            / wall_time_seconds
                        )
                        * 100
                        / cpu_count
                        if wall_time_seconds > 0
                        else 0.0
                    )

                    # Also capture the instantaneous process sample
                    # when available.

                    cpu_sample_max = max(
                        cpu_percent_samples,
                        default=0.0,
                    )

                    result = {
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "filename": data[
                            "filename"
                        ],
                        "size_bytes": int(
                            data["size_bytes"]
                        ),
                        "size_mb": float(
                            data["size_mb"]
                        ),
                        "page_count": page_count,
                        "start_time": start_time,
                        "end_time": utc_now(),
                        "ocr_time_seconds": (
                            wall_time_seconds
                        ),
                        "cpu_time_seconds": (
                            cpu_time_seconds
                        ),
                        "cpu_avg_percent": (
                            cpu_avg_percent
                        ),
                        "cpu_sample_max_percent": (
                            cpu_sample_max
                        ),
                        "ram_before_mb": (
                            ram_before_mb
                        ),
                        "ram_peak_mb": (
                            peak_rss_mb
                        ),
                        "ram_after_mb": (
                            ram_after_mb
                        ),
                        "ram_delta_mb": (
                            ram_after_mb
                            - ram_before_mb
                        ),
                        "confidence": confidence,
                        "status": status,
                        "error": error_message,
                        "redis_message_id": (
                            message_id
                        ),
                    }

                    # ------------------------------------------------
                    # ACK
                    # ------------------------------------------------

                    redis_client.xack(
                        stream_name,
                        consumer_group,
                        message_id,
                    )

                    # ------------------------------------------------
                    # Return result
                    # ------------------------------------------------

                    result_queue.put(
                        {
                            "type": "result",
                            "result": result,
                        }
                    )

    finally:

        redis_client.close()


# ============================================================
# RUN ONE SCENARIO
# ============================================================

def run_scenario(
    worker_count: int,
    documents: list[dict],
) -> dict:

    run_id = uuid.uuid4().hex[:10]

    stream_name = (
        f"{STREAM_PREFIX}_"
        f"{worker_count}w_"
        f"{run_id}"
    )

    consumer_group = (
        f"{GROUP_PREFIX}_"
        f"{run_id}"
    )

    result_queue = mp.Queue()

    ready_queue = mp.Queue()

    stop_event = mp.Event()

    processes = []

    # --------------------------------------------------------
    # Start workers
    # --------------------------------------------------------

    for worker_number in range(
        1,
        worker_count + 1,
    ):

        process = mp.Process(
            target=benchmark_worker,
            args=(
                worker_number,
                stream_name,
                consumer_group,
                result_queue,
                ready_queue,
                stop_event,
            ),
        )

        process.start()

        processes.append(process)

    # --------------------------------------------------------
    # Wait for warm-up
    # --------------------------------------------------------

    warmups = []

    for _ in range(worker_count):

        try:

            warmup = ready_queue.get(
                timeout=600
            )

        except queue.Empty:

            raise RuntimeError(
                "Benchmark worker did not "
                "finish PaddleOCR warm-up "
                "within 10 minutes."
            )

        warmups.append(warmup)

    # --------------------------------------------------------
    # Add jobs
    # --------------------------------------------------------

    redis_client = create_redis_client()

    try:

        for document in documents:

            redis_client.xadd(
                stream_name,
                {
                    "job_id": document[
                        "job_id"
                    ],
                    "filename": document[
                        "filename"
                    ],
                    "path": document[
                        "path"
                    ],
                    "size_bytes": str(
                        document[
                            "size_bytes"
                        ]
                    ),
                    "size_mb": str(
                        document[
                            "size_mb"
                        ]
                    ),
                    "page_count": str(
                        document[
                            "page_count"
                        ]
                    ),
                },
            )

        # ----------------------------------------------------
        # Benchmark starts AFTER jobs are queued
        # ----------------------------------------------------

        benchmark_start_timestamp = utc_now()

        benchmark_start = time.perf_counter()

        results = []

        while len(results) < len(documents):

            message = result_queue.get(
                timeout=1800
            )

            if message["type"] == "result":

                results.append(
                    message["result"]
                )

        benchmark_wall_time = (
            time.perf_counter()
            - benchmark_start
        )

        benchmark_end_timestamp = utc_now()

    finally:

        stop_event.set()

        # ----------------------------------------------------
        # Stop workers
        # ----------------------------------------------------

        for process in processes:

            process.join(
                timeout=5
            )

        for process in processes:

            if process.is_alive():

                process.terminate()

                process.join()

        # ----------------------------------------------------
        # Delete benchmark stream
        # ----------------------------------------------------

        try:

            redis_client.delete(
                stream_name
            )

            redis_client.xgroup_destroy(
                stream_name,
                consumer_group,
            )

        except Exception:
            pass

        redis_client.close()

    results.sort(
        key=lambda item: item[
            "filename"
        ]
    )

    return {
        "run_id": run_id,
        "workers": worker_count,
        "stream": stream_name,
        "consumer_group": consumer_group,
        "benchmark_start": benchmark_start_timestamp,
        "benchmark_end": benchmark_end_timestamp,
        "total_wall_time_seconds": (
            benchmark_wall_time
        ),
        "warmups": warmups,
        "results": results,
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    benchmark: dict,
) -> dict:

    successful = [
        result
        for result in benchmark["results"]
        if result["status"] == "success"
    ]

    total_size_mb = sum(
        result["size_mb"]
        for result in successful
    )

    total_pages = sum(
        result["page_count"]
        for result in successful
    )

    total_ocr_time = sum(
        result["ocr_time_seconds"]
        for result in successful
    )

    benchmark_wall_time = (
        benchmark[
            "total_wall_time_seconds"
        ]
    )

    return {
        "workers": benchmark["workers"],
        "documents": len(
            benchmark["results"]
        ),
        "successful": len(
            successful
        ),
        "failed": (
            len(benchmark["results"])
            - len(successful)
        ),
        "total_size_mb": total_size_mb,
        "total_pages": total_pages,
        "total_ocr_time_seconds": (
            total_ocr_time
        ),
        "benchmark_wall_time_seconds": (
            benchmark_wall_time
        ),
        "average_ocr_time_per_pdf": (
            mean(
                [
                    result[
                        "ocr_time_seconds"
                    ]
                    for result in successful
                ]
            )
            if successful
            else 0.0
        ),
        "average_time_per_mb": (
            total_ocr_time / total_size_mb
            if total_size_mb
            else 0.0
        ),
        "average_time_per_page": (
            total_ocr_time / total_pages
            if total_pages
            else 0.0
        ),
        "throughput_mb_per_second": (
            total_size_mb
            / benchmark_wall_time
            if benchmark_wall_time
            else 0.0
        ),
        "throughput_pages_per_second": (
            total_pages
            / benchmark_wall_time
            if benchmark_wall_time
            else 0.0
        ),
        "average_confidence": (
            mean(
                [
                    result[
                        "confidence"
                    ]
                    for result in successful
                ]
            )
            if successful
            else 0.0
        ),
        "maximum_peak_rss_mb": max(
            [
                result[
                    "ram_peak_mb"
                ]
                for result in successful
            ],
            default=0.0,
        ),
        "average_cpu_percent": (
            mean(
                [
                    result[
                        "cpu_avg_percent"
                    ]
                    for result in successful
                ]
            )
            if successful
            else 0.0
        ),
        "maximum_cpu_percent": max(
            [
                result[
                    "cpu_sample_max_percent"
                ]
                for result in successful
            ],
            default=0.0,
        ),
    }


# ============================================================
# PER-WORKER SUMMARY
# ============================================================

def calculate_worker_summaries(
    results: list[dict],
) -> list[dict]:

    worker_ids = sorted(
        {
            result["worker_id"]
            for result in results
        }
    )

    summaries = []

    for worker_id in worker_ids:

        worker_results = [
            result
            for result in results
            if result["worker_id"]
            == worker_id
        ]

        successful = [
            result
            for result in worker_results
            if result["status"]
            == "success"
        ]

        total_size_mb = sum(
            result["size_mb"]
            for result in successful
        )

        total_pages = sum(
            result["page_count"]
            for result in successful
        )

        total_ocr_time = sum(
            result["ocr_time_seconds"]
            for result in successful
        )

        summaries.append(
            {
                "worker_id": worker_id,
                "documents_processed": (
                    len(worker_results)
                ),
                "successful": len(
                    successful
                ),
                "failed": (
                    len(worker_results)
                    - len(successful)
                ),
                "total_size_mb": (
                    total_size_mb
                ),
                "total_pages": (
                    total_pages
                ),
                "total_ocr_time_seconds": (
                    total_ocr_time
                ),
                "average_time_per_pdf": (
                    mean(
                        [
                            result[
                                "ocr_time_seconds"
                            ]
                            for result in successful
                        ]
                    )
                    if successful
                    else 0.0
                ),
                "average_time_per_mb": (
                    total_ocr_time
                    / total_size_mb
                    if total_size_mb
                    else 0.0
                ),
                "average_time_per_page": (
                    total_ocr_time
                    / total_pages
                    if total_pages
                    else 0.0
                ),
                "throughput_mb_per_sec": (
                    total_size_mb
                    / total_ocr_time
                    if total_ocr_time
                    else 0.0
                ),
                "throughput_pages_per_sec": (
                    total_pages
                    / total_ocr_time
                    if total_ocr_time
                    else 0.0
                ),
                "maximum_peak_rss_mb": max(
                    [
                        result[
                            "ram_peak_mb"
                        ]
                        for result in successful
                    ],
                    default=0.0,
                ),
                "average_cpu_percent": (
                    mean(
                        [
                            result[
                                "cpu_avg_percent"
                            ]
                            for result in successful
                        ]
                    )
                    if successful
                    else 0.0
                ),
                "maximum_cpu_percent": max(
                    [
                        result[
                            "cpu_sample_max_percent"
                        ]
                        for result in successful
                    ],
                    default=0.0,
                ),
            }
        )

    return summaries


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    benchmark: dict,
) -> None:

    scenario_name = (
        f"{benchmark['workers']}_worker"
    )

    output_directory = (
        OUTPUT_DIR
        / scenario_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Per-document CSV
    # --------------------------------------------------------

    csv_path = (
        output_directory
        / "per_document.csv"
    )

    if benchmark["results"]:

        fields = list(
            benchmark["results"][0].keys()
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()

            writer.writerows(
                benchmark["results"]
            )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "run_id": benchmark["run_id"],
        "workers": benchmark["workers"],
        "stream": benchmark["stream"],
        "consumer_group": benchmark[
            "consumer_group"
        ],
        "benchmark_start": benchmark[
            "benchmark_start"
        ],
        "benchmark_end": benchmark[
            "benchmark_end"
        ],
        "total_wall_time_seconds": (
            benchmark[
                "total_wall_time_seconds"
            ]
        ),
        "warmups": benchmark["warmups"],
        "overall": (
            calculate_overall_metrics(
                benchmark
            )
        ),
        "worker_summaries": (
            calculate_worker_summaries(
                benchmark["results"]
            )
        ),
    }

    json_path = (
        output_directory
        / "summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    benchmark: dict,
) -> None:

    metrics = (
        calculate_overall_metrics(
            benchmark
        )
    )

    print()
    print("=" * 70)
    print(
        f"{benchmark['workers']}-WORKER BENCHMARK"
    )
    print("=" * 70)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    print()
    print("PER DOCUMENT")
    print("-" * 70)

    for result in benchmark["results"]:

        print(
            f"{result['filename']} | "
            f"{result['size_mb']:.3f} MB | "
            f"{result['page_count']} pages | "
            f"OCR {result['ocr_time_seconds']:.3f}s | "
            f"CPU {result['cpu_avg_percent']:.1f}% | "
            f"RAM peak {result['ram_peak_mb']:.1f} MB | "
            f"{result['status']}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    mp.set_start_method(
        "spawn",
        force=True,
    )

    print()
    print("=" * 70)
    print("OCR SINGLE-WORKER BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Find PDFs
    # --------------------------------------------------------

    documents = collect_documents()

    print()
    print(
        f"Found {len(documents)} PDFs:"
    )

    for document in documents:

        print(
            f"  {document['filename']} | "
            f"{document['size_mb']:.3f} MB | "
            f"{document['page_count']} pages"
        )

    # --------------------------------------------------------
    # ONLY ONE WORKER
    # --------------------------------------------------------

    benchmark = run_scenario(
        worker_count=1,
        documents=documents,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        benchmark
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        benchmark
    )

    print()
    print("=" * 70)
    print(
        "BENCHMARK COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_DIR}/1_worker"
    )


if __name__ == "__main__":
    main()