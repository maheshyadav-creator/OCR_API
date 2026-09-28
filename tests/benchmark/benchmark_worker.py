from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from pypdf import PdfReader

from app.services.ocr_service import get_ocr_engine, perform_ocr


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/home/mahesh/Documents/OCR_API")
PDF_DIR = PROJECT_ROOT / "test_pdf"
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

BENCHMARK_LIMIT = int(os.getenv("BENCHMARK_LIMIT", "10"))

# This benchmark intentionally uses ONE Python process / ONE OCR engine.
WORKER_ID = "worker_1"
WORKER_COUNT = 1

CSV_PATH = RESULTS_DIR / "per_document.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
BENCHMARK_PATH = RESULTS_DIR / "benchmark.json"


# ============================================================
# HELPERS
# ============================================================

def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def mb(value: float) -> float:
    """Convert bytes to MB."""
    return value / (1024 * 1024)


def get_process() -> psutil.Process:
    """Return the current Python process."""
    return psutil.Process(os.getpid())


def get_logical_cpu_count() -> int:
    """Return the number of logical CPUs visible to this process."""
    return psutil.cpu_count(logical=True) or 1


def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Read PDF metadata and return page count.

    This happens OUTSIDE the OCR timing.
    """
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def get_pdf_size(pdf_path: Path) -> int:
    """Return PDF size in bytes."""
    return pdf_path.stat().st_size


def discover_pdfs() -> list[Path]:
    """
    Find benchmark PDFs.

    Only files matching ocr_test_*.pdf are included.
    """
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {PDF_DIR}"
        )

    pdfs = sorted(PDF_DIR.glob("ocr_test_*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"No benchmark PDFs found in: {PDF_DIR}"
        )

    return pdfs[:BENCHMARK_LIMIT]


def safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_ocr_engine() -> float:
    """
    Initialize the cached PaddleOCR engine.

    Model initialization is measured separately and is NOT included
    in per-document OCR wall time.
    """
    print()
    print("=" * 70)
    print("INITIALIZING PADDLEOCR")
    print("=" * 70)

    start = time.perf_counter()

    get_ocr_engine()

    elapsed = time.perf_counter() - start

    print(f"Model initialization completed in: {elapsed:.3f} sec")

    return elapsed


# ============================================================
# SINGLE DOCUMENT BENCHMARK
# ============================================================

def benchmark_document(
    pdf_path: Path,
    document_number: int,
) -> dict[str, Any]:
    """
    Run OCR on one PDF and collect benchmark metrics.

    Important:
    - PDF size/page count are collected before OCR timing.
    - OCR engine is already initialized.
    - OCR is executed sequentially.
    - No Redis.
    - No PostgreSQL.
    - No API request.
    """

    process = get_process()

    size_bytes = get_pdf_size(pdf_path)
    size_kb = size_bytes / 1024
    size_mb = mb(size_bytes)

    try:
        page_count = get_pdf_page_count(pdf_path)
        page_count_input = page_count
    except Exception as exc:
        print(
            f"WARNING: Could not read page count for "
            f"{pdf_path.name}: {exc}"
        )
        page_count = None
        page_count_input = None

    print()
    print("-" * 70)
    print(f"DOCUMENT {document_number}")
    print("-" * 70)
    print(f"File       : {pdf_path.name}")
    print(f"Size       : {size_mb:.3f} MB")
    print(f"Pages      : {page_count}")
    print(f"Worker     : {WORKER_ID}")

    # --------------------------------------------------------
    # Memory before OCR
    # --------------------------------------------------------

    ram_before_bytes = process.memory_info().rss
    ram_before_mb = mb(ram_before_bytes)

    # --------------------------------------------------------
    # CPU time before OCR
    # --------------------------------------------------------

    cpu_before = process.cpu_times()
    cpu_before_seconds = cpu_before.user + cpu_before.system

    # --------------------------------------------------------
    # OCR timing starts HERE
    # --------------------------------------------------------

    start_timestamp = utc_timestamp()
    wall_start = time.perf_counter()

    status = "success"
    error_message = ""
    result: dict[str, Any] = {}

    try:
        result = perform_ocr(str(pdf_path))

    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"

    wall_end = time.perf_counter()
    end_timestamp = utc_timestamp()

    # --------------------------------------------------------
    # OCR timing ends HERE
    # --------------------------------------------------------

    wall_time_seconds = wall_end - wall_start

    cpu_after = process.cpu_times()
    cpu_after_seconds = cpu_after.user + cpu_after.system

    cpu_time_seconds = cpu_after_seconds - cpu_before_seconds

    # Number of CPU cores effectively consumed by this process
    # during the OCR interval.
    cpu_cores_used = (
        cpu_time_seconds / wall_time_seconds
        if wall_time_seconds > 0
        else 0.0
    )

    logical_cpu_count = get_logical_cpu_count()

    # Percentage of TOTAL MACHINE logical CPU capacity.
    cpu_machine_capacity_percent = (
        cpu_cores_used / logical_cpu_count * 100
        if logical_cpu_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Memory after OCR
    # --------------------------------------------------------

    ram_after_bytes = process.memory_info().rss
    ram_after_mb = mb(ram_after_bytes)

    ram_delta_mb = ram_after_mb - ram_before_mb

    # --------------------------------------------------------
    # Lifetime process peak RSS
    # --------------------------------------------------------

    try:
        peak_rss_bytes = resource_peak_rss_bytes()
        peak_rss_mb = mb(peak_rss_bytes)
    except Exception:
        peak_rss_mb = None

    # --------------------------------------------------------
    # OCR result metrics
    # --------------------------------------------------------

    confidence = None
    text_characters = 0

    if status == "success" and isinstance(result, dict):
        confidence = safe_float(result.get("confidence"))

        extracted_text = result.get("text", "")

        if extracted_text is not None:
            text_characters = len(str(extracted_text))

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(f"OCR wall time       : {wall_time_seconds:.3f} sec")
    print(f"CPU time            : {cpu_time_seconds:.3f} sec")
    print(f"CPU cores used      : {cpu_cores_used:.2f}")
    print(
        f"CPU machine usage   : "
        f"{cpu_machine_capacity_percent:.2f}%"
    )
    print(f"RAM before          : {ram_before_mb:.2f} MB")
    print(f"RAM after           : {ram_after_mb:.2f} MB")

    if peak_rss_mb is not None:
        print(f"Peak RSS            : {peak_rss_mb:.2f} MB")

    if confidence is not None:
        print(f"Confidence          : {confidence:.4f}")

    print(f"Text characters     : {text_characters}")
    print(f"Status              : {status}")

    if error_message:
        print(f"Error               : {error_message}")

    # --------------------------------------------------------
    # Return one complete CSV-compatible record
    # --------------------------------------------------------

    return {
        "document_number": document_number,
        "filename": pdf_path.name,
        "file_path": str(pdf_path),
        "file_type": "application/pdf",
        "size_bytes": size_bytes,
        "size_kb": round(size_kb, 3),
        "size_mb": round(size_mb, 6),
        "page_count": page_count,
        "page_count_input": page_count_input,
        "worker_id": WORKER_ID,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "ocr_wall_time_seconds": round(wall_time_seconds, 6),
        "cpu_process_time_seconds": round(cpu_time_seconds, 6),
        "cpu_cores_used": round(cpu_cores_used, 6),
        "cpu_machine_capacity_percent": round(
            cpu_machine_capacity_percent,
            6,
        ),
        "cpu_avg_percent": round(
            cpu_machine_capacity_percent,
            6,
        ),
        "logical_cpu_count": logical_cpu_count,
        "ram_before_mb": round(ram_before_mb, 3),
        "ram_after_mb": round(ram_after_mb, 3),
        "peak_rss_mb": (
            round(peak_rss_mb, 3)
            if peak_rss_mb is not None
            else None
        ),
        "ram_delta_mb": round(ram_delta_mb, 3),
        "confidence": (
            round(confidence, 6)
            if confidence is not None
            else None
        ),
        "text_characters": text_characters,
        "status": status,
        "error": error_message,
    }


# ============================================================
# PEAK RSS
# ============================================================

def resource_peak_rss_bytes() -> int:
    """
    Return lifetime maximum RSS for this process.

    Linux reports ru_maxrss in KB.
    """
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)

    # Linux: KB
    return int(usage.ru_maxrss * 1024)


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(records: list[dict[str, Any]]) -> None:
    """Write per-document benchmark results to CSV."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "document_number",
        "filename",
        "file_path",
        "file_type",
        "size_bytes",
        "size_kb",
        "size_mb",
        "page_count",
        "page_count_input",
        "worker_id",
        "start_timestamp",
        "end_timestamp",
        "ocr_wall_time_seconds",
        "cpu_process_time_seconds",
        "cpu_cores_used",
        "cpu_machine_capacity_percent",
        "cpu_avg_percent",
        "logical_cpu_count",
        "ram_before_mb",
        "ram_after_mb",
        "peak_rss_mb",
        "ram_delta_mb",
        "confidence",
        "text_characters",
        "status",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(
    records: list[dict[str, Any]],
    model_initialization_seconds: float,
    benchmark_wall_time_seconds: float,
    benchmark_start_timestamp: str,
    benchmark_end_timestamp: str,
) -> dict[str, Any]:

    successful = [
        record
        for record in records
        if record["status"] == "success"
    ]

    failed = [
        record
        for record in records
        if record["status"] != "success"
    ]

    # --------------------------------------------------------
    # Input totals
    # --------------------------------------------------------

    total_size_bytes = sum(
        int(record["size_bytes"])
        for record in records
    )

    total_size_mb = mb(total_size_bytes)

    total_pages = sum(
        int(record["page_count"])
        for record in records
        if record["page_count"] is not None
    )

    # --------------------------------------------------------
    # Successful OCR totals
    # --------------------------------------------------------

    successful_size_bytes = sum(
        int(record["size_bytes"])
        for record in successful
    )

    successful_size_mb = mb(successful_size_bytes)

    successful_pages = sum(
        int(record["page_count"])
        for record in successful
        if record["page_count"] is not None
    )

    total_ocr_wall_time = sum(
        float(record["ocr_wall_time_seconds"])
        for record in successful
    )

    total_cpu_time = sum(
        float(record["cpu_process_time_seconds"])
        for record in successful
    )

    # --------------------------------------------------------
    # Averages
    # --------------------------------------------------------

    successful_count = len(successful)

    if successful_count:
        average_time_per_document = (
            total_ocr_wall_time / successful_count
        )

        average_time_per_mb = (
            total_ocr_wall_time / successful_size_mb
            if successful_size_mb > 0
            else 0.0
        )

        average_time_per_page = (
            total_ocr_wall_time / successful_pages
            if successful_pages > 0
            else 0.0
        )

        average_cpu_time_seconds = (
            total_cpu_time / successful_count
        )

        average_cpu_machine_capacity_percent = (
            sum(
                float(
                    record["cpu_machine_capacity_percent"]
                )
                for record in successful
            )
            / successful_count
        )

        average_ram_before_mb = (
            sum(
                float(record["ram_before_mb"])
                for record in successful
            )
            / successful_count
        )

        average_ram_after_mb = (
            sum(
                float(record["ram_after_mb"])
                for record in successful
            )
            / successful_count
        )

        average_ram_delta_mb = (
            sum(
                float(record["ram_delta_mb"])
                for record in successful
            )
            / successful_count
        )

        confidence_values = [
            float(record["confidence"])
            for record in successful
            if record["confidence"] is not None
        ]

        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )

    else:
        average_time_per_document = 0.0
        average_time_per_mb = 0.0
        average_time_per_page = 0.0
        average_cpu_time_seconds = 0.0
        average_cpu_machine_capacity_percent = 0.0
        average_ram_before_mb = 0.0
        average_ram_after_mb = 0.0
        average_ram_delta_mb = 0.0
        average_confidence = None

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    documents_per_second = (
        successful_count / benchmark_wall_time_seconds
        if benchmark_wall_time_seconds > 0
        else 0.0
    )

    mb_per_second = (
        successful_size_mb / benchmark_wall_time_seconds
        if benchmark_wall_time_seconds > 0
        else 0.0
    )

    pages_per_second = (
        successful_pages / benchmark_wall_time_seconds
        if benchmark_wall_time_seconds > 0
        else 0.0
    )

    gb_per_hour = (
        mb_per_second * 3600 / 1024
    )

    # --------------------------------------------------------
    # Peak RSS
    # --------------------------------------------------------

    peak_values = [
        float(record["peak_rss_mb"])
        for record in successful
        if record["peak_rss_mb"] is not None
    ]

    maximum_peak_rss_mb = (
        max(peak_values)
        if peak_values
        else None
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    return {
        "benchmark_type": "single_process_sequential_ocr",
        "worker_count": WORKER_COUNT,
        "worker_id": WORKER_ID,
        "logical_cpu_count": get_logical_cpu_count(),

        "pdf_directory": str(PDF_DIR),
        "results_directory": str(RESULTS_DIR),

        "benchmark_start_timestamp": benchmark_start_timestamp,
        "benchmark_end_timestamp": benchmark_end_timestamp,

        "documents_requested": len(records),
        "documents": len(records),
        "successful": successful_count,
        "failed": len(failed),

        "success_rate_percent": (
            successful_count / len(records) * 100
            if records
            else 0.0
        ),

        "total_input_size_mb": total_size_mb,
        "successful_size_mb": successful_size_mb,

        "total_pages": total_pages,
        "successful_pages": successful_pages,

        "total_ocr_wall_time_seconds": total_ocr_wall_time,
        "benchmark_wall_time_seconds": benchmark_wall_time_seconds,

        "model_initialization_seconds": model_initialization_seconds,

        "average_time_per_document_seconds": (
            average_time_per_document
        ),

        "average_time_per_mb": average_time_per_mb,
        "average_time_per_page": average_time_per_page,

        "documents_per_second": documents_per_second,
        "mb_per_second": mb_per_second,
        "pages_per_second": pages_per_second,
        "gb_per_hour": gb_per_hour,

        "total_cpu_time_seconds": total_cpu_time,
        "average_cpu_time_seconds": average_cpu_time_seconds,

        "average_cpu_machine_capacity_percent": (
            average_cpu_machine_capacity_percent
        ),

        # Backward-compatible field name.
        "average_cpu_percent": (
            average_cpu_machine_capacity_percent
        ),

        "average_ram_before_mb": average_ram_before_mb,
        "average_ram_after_mb": average_ram_after_mb,
        "average_ram_delta_mb": average_ram_delta_mb,

        "maximum_peak_rss_mb": maximum_peak_rss_mb,

        "average_confidence": average_confidence,

        "measurement_notes": {
            "ocr_wall_time_excludes_model_initialization": True,
            "ocr_processing_is_sequential": True,
            "redis_used": False,
            "postgres_used": False,
            "api_used": False,
            "cpu_machine_capacity_percent_definition": (
                "Process CPU time divided by OCR wall time, "
                "normalized by logical CPU count."
            ),
            "cpu_cores_used_definition": (
                "Process CPU seconds divided by OCR wall seconds."
            ),
            "peak_rss_definition": (
                "Lifetime maximum RSS of this Python process "
                "as reported by Linux getrusage."
            ),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("OCR SINGLE-WORKER BASELINE BENCHMARK")
    print("=" * 70)

    print(f"Project       : {PROJECT_ROOT}")
    print(f"PDF directory : {PDF_DIR}")
    print(f"Results       : {RESULTS_DIR}")
    print(f"Worker        : {WORKER_ID}")
    print(f"Documents     : {BENCHMARK_LIMIT}")
    print(f"Logical CPUs  : {get_logical_cpu_count()}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Discover documents
    # --------------------------------------------------------

    pdfs = discover_pdfs()

    print()
    print("Documents selected:")

    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"  {index:02d}. {pdf_path.name}")

    # --------------------------------------------------------
    # Initialize model ONCE
    # --------------------------------------------------------

    model_initialization_seconds = initialize_ocr_engine()

    # --------------------------------------------------------
    # Start benchmark timing
    # --------------------------------------------------------

    benchmark_start_timestamp = utc_timestamp()
    benchmark_start = time.perf_counter()

    records: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Sequential processing
    # --------------------------------------------------------

    for document_number, pdf_path in enumerate(
        pdfs,
        start=1,
    ):

        record = benchmark_document(
            pdf_path=pdf_path,
            document_number=document_number,
        )

        records.append(record)

    # --------------------------------------------------------
    # Benchmark end
    # --------------------------------------------------------

    benchmark_end = time.perf_counter()
    benchmark_end_timestamp = utc_timestamp()

    benchmark_wall_time_seconds = (
        benchmark_end - benchmark_start
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    write_csv(records)

    summary = calculate_summary(
        records=records,
        model_initialization_seconds=(
            model_initialization_seconds
        ),
        benchmark_wall_time_seconds=(
            benchmark_wall_time_seconds
        ),
        benchmark_start_timestamp=(
            benchmark_start_timestamp
        ),
        benchmark_end_timestamp=(
            benchmark_end_timestamp
        ),
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    benchmark_data = {
        "summary": summary,
        "documents": records,
    }

    with BENCHMARK_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark_data,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"Documents       : "
        f"{summary['documents']}"
    )

    print(
        f"Successful      : "
        f"{summary['successful']}"
    )

    print(
        f"Failed          : "
        f"{summary['failed']}"
    )

    print(
        f"Total input     : "
        f"{summary['total_input_size_mb']:.3f} MB"
    )

    print(
        f"Total pages     : "
        f"{summary['total_pages']}"
    )

    print(
        f"OCR wall time   : "
        f"{summary['total_ocr_wall_time_seconds']:.3f} sec"
    )

    print(
        f"Benchmark time  : "
        f"{summary['benchmark_wall_time_seconds']:.3f} sec"
    )

    print(
        f"Avg/doc         : "
        f"{summary['average_time_per_document_seconds']:.3f} sec"
    )

    print(
        f"Avg/page        : "
        f"{summary['average_time_per_page']:.3f} sec"
    )

    print(
        f"MB/sec          : "
        f"{summary['mb_per_second']:.6f}"
    )

    print(
        f"Pages/sec       : "
        f"{summary['pages_per_second']:.6f}"
    )

    print(
        f"GB/hour         : "
        f"{summary['gb_per_hour']:.6f}"
    )

    print(
        f"CPU time        : "
        f"{summary['total_cpu_time_seconds']:.3f} sec"
    )

    print(
        f"Avg CPU usage   : "
        f"{summary['average_cpu_machine_capacity_percent']:.2f}% "
        f"of total machine capacity"
    )

    if summary["maximum_peak_rss_mb"] is not None:
        print(
            f"Max peak RSS    : "
            f"{summary['maximum_peak_rss_mb']:.2f} MB"
        )

    if summary["average_confidence"] is not None:
        print(
            f"Avg confidence  : "
            f"{summary['average_confidence']:.4f}"
        )

    print()
    print("Output files:")
    print(f"  {CSV_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {BENCHMARK_PATH}")
    print()


if __name__ == "__main__":
    main()
