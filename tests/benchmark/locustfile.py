from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task
from locust.exception import StopUser


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        "/home/mahesh/Documents/OCR_API",
    )
)

PDF_DIR = PROJECT_ROOT / "test_pdf"

# EXACTLY 10 users
EXPECTED_USERS = 10

UPLOAD_PATH = "/ocr"

STATUS_PATH = "/ocr/{job_id}"

FILE_FIELD = "files"

POLL_INTERVAL_SECONDS = float(
    os.getenv(
        "POLL_INTERVAL_SECONDS",
        "2",
    )
)

REQUEST_TIMEOUT = float(
    os.getenv(
        "REQUEST_TIMEOUT",
        "1800",
    )
)

RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

LOCUST_RESULTS_FILE = (
    RESULTS_DIR
    / "locust_10_users_results.csv"
)


# ============================================================
# PDF DISCOVERY
# ============================================================

def load_pdfs() -> list[Path]:
    """
    Load exactly the first 10 PDFs.

    The benchmark will NEVER use:
        - PDF 11+
        - random PDFs
        - duplicate PDFs
    """

    pdfs = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if len(pdfs) != EXPECTED_USERS:
        raise RuntimeError(
            f"STRICT BENCHMARK ERROR: "
            f"Expected exactly {EXPECTED_USERS} PDFs, "
            f"but found {len(pdfs)} in {PDF_DIR}"
        )

    # Extra duplicate filename protection.
    filenames = [
        pdf.name
        for pdf in pdfs
    ]

    if len(filenames) != len(set(filenames)):
        raise RuntimeError(
            "STRICT BENCHMARK ERROR: "
            "Duplicate PDF filenames detected."
        )

    print()
    print("=" * 70)
    print("LOCUST PDF BENCHMARK")
    print("=" * 70)
    print(
        f"Expected users : {EXPECTED_USERS}"
    )
    print(
        f"PDF directory  : {PDF_DIR}"
    )
    print(
        f"PDFs discovered: {len(pdfs)}"
    )
    print()
    print("PDF assignment:")
    print("-" * 70)

    for index, pdf in enumerate(
        pdfs,
        start=1,
    ):
        print(
            f"User {index:02d} -> "
            f"{pdf.name}"
        )

    print("=" * 70)
    print()

    return pdfs


PDFS = load_pdfs()


# ============================================================
# GLOBAL ASSIGNMENT STATE
# ============================================================

assignment_lock = threading.Lock()

next_pdf_index = 0

assigned_pdfs: set[str] = set()


def get_next_pdf() -> Path:
    """
    Give exactly one unique PDF to each Locust user.

    Example:

        User 1  -> ocr_test_01.pdf
        User 2  -> ocr_test_02.pdf
        ...
        User 10 -> ocr_test_10.pdf

    Once all 10 PDFs have been assigned, any additional
    Locust user causes the benchmark to fail immediately.
    """

    global next_pdf_index

    with assignment_lock:

        # ----------------------------------------------------
        # NEVER allow more than 10 assignments.
        # ----------------------------------------------------

        if next_pdf_index >= EXPECTED_USERS:

            raise RuntimeError(
                "STRICT BENCHMARK ERROR: "
                f"More than {EXPECTED_USERS} Locust users "
                "attempted to receive a PDF."
            )

        pdf = PDFS[next_pdf_index]

        next_pdf_index += 1

        # ----------------------------------------------------
        # Duplicate protection.
        # ----------------------------------------------------

        if pdf.name in assigned_pdfs:

            raise RuntimeError(
                "STRICT BENCHMARK ERROR: "
                f"Duplicate PDF assignment detected: "
                f"{pdf.name}"
            )

        assigned_pdfs.add(
            pdf.name
        )

        return pdf


# ============================================================
# RESULTS
# ============================================================

results_lock = threading.Lock()

results: list[dict[str, Any]] = []


def save_results() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_lock:
        rows = list(results)

    if not rows:
        return

    with LOCUST_RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "user_id",
            "filename",
            "job_id",
            "upload_start",
            "upload_end",
            "end_to_end_seconds",
            "final_status",
            "attempt_count",
            "queue_wait_seconds",
            "processing_seconds",
            "error_message",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"Locust results written to: "
        f"{LOCUST_RESULTS_FILE}"
    )


# ============================================================
# LOCUST USER
# ============================================================

class OcrUser(HttpUser):

    wait_time = between(
        0,
        0,
    )

    def on_start(self) -> None:
        """
        Called once when this Locust user starts.

        Each user gets exactly ONE unique PDF.
        """

        self.pdf_path = get_next_pdf()

        print(
            f"[USER ASSIGNMENT] "
            f"{self.pdf_path.name}"
        )

    @task
    def upload_one_pdf(self) -> None:
        """
        Upload exactly ONE PDF.

        After this task finishes, StopUser is raised.

        Therefore this user can NEVER upload a second PDF.
        """

        pdf_path = self.pdf_path

        print()
        print("=" * 70)
        print(
            f"Uploading: {pdf_path.name}"
        )
        print("=" * 70)

        upload_start = time.perf_counter()

        job_id = None

        final_status = "failed"

        attempt_count = 0

        queue_wait_seconds = ""

        processing_seconds = ""

        error_message = ""

        try:

            # ------------------------------------------------
            # SUBMIT EXACTLY ONE PDF
            # ------------------------------------------------

            with pdf_path.open(
                "rb"
            ) as pdf_file:

                response = self.client.post(
                    UPLOAD_PATH,
                    files={
                        FILE_FIELD: (
                            pdf_path.name,
                            pdf_file,
                            "application/pdf",
                        )
                    },
                    name="POST /ocr",
                    timeout=REQUEST_TIMEOUT,
                )

            upload_end = time.perf_counter()

            if response.status_code != 202:

                error_message = (
                    f"Upload failed: "
                    f"HTTP {response.status_code}: "
                    f"{response.text}"
                )

                raise RuntimeError(
                    error_message
                )

            body = response.json()

            jobs = body.get(
                "jobs",
                [],
            )

            # ------------------------------------------------
            # API MUST RETURN EXACTLY ONE JOB.
            # ------------------------------------------------

            if len(jobs) != 1:

                raise RuntimeError(
                    "STRICT BENCHMARK ERROR: "
                    f"Expected exactly 1 job, "
                    f"but API returned {len(jobs)} jobs."
                )

            job_id = jobs[0]["job_id"]

            print(
                f"Uploaded: {pdf_path.name}"
            )

            print(
                f"Job ID: {job_id}"
            )

            # ------------------------------------------------
            # WAIT FOR THIS EXACT JOB
            # ------------------------------------------------

            while True:

                time.sleep(
                    POLL_INTERVAL_SECONDS
                )

                status_response = self.client.get(
                    STATUS_PATH.format(
                        job_id=job_id
                    ),
                    name="GET /ocr/{job_id}",
                    timeout=REQUEST_TIMEOUT,
                )

                if status_response.status_code != 200:

                    raise RuntimeError(
                        "Status request failed: "
                        f"HTTP "
                        f"{status_response.status_code}"
                    )

                status_data = (
                    status_response.json()
                )

                current_status = (
                    status_data.get(
                        "status"
                    )
                )

                attempt_count = int(
                    status_data.get(
                        "attempt_count",
                        0,
                    )
                )

                created_at = (
                    status_data.get(
                        "created_at"
                    )
                )

                started_at = (
                    status_data.get(
                        "started_at"
                    )
                )

                completed_at = (
                    status_data.get(
                        "completed_at"
                    )
                )

                if (
                    current_status
                    == "completed"
                ):

                    final_status = "completed"

                    if (
                        created_at
                        and started_at
                    ):
                        queue_wait_seconds = (
                            "calculated_from_server"
                        )

                    if (
                        started_at
                        and completed_at
                    ):
                        processing_seconds = (
                            "calculated_from_server"
                        )

                    break

                if (
                    current_status
                    == "failed"
                ):

                    final_status = "failed"

                    error_message = (
                        status_data.get(
                            "error_message",
                            "",
                        )
                        or ""
                    )

                    break

                print(
                    f"{pdf_path.name}: "
                    f"{current_status}"
                )

        except Exception as exc:

            final_status = "failed"

            error_message = str(exc)

            print(
                f"ERROR processing "
                f"{pdf_path.name}: "
                f"{exc}"
            )

        finally:

            upload_end = time.perf_counter()

            end_to_end_seconds = (
                upload_end
                - upload_start
            )

            result = {
                "user_id": str(
                    getattr(
                        self,
                        "user_id",
                        "",
                    )
                ),
                "filename": pdf_path.name,
                "job_id": job_id or "",
                "upload_start": upload_start,
                "upload_end": upload_end,
                "end_to_end_seconds": round(
                    end_to_end_seconds,
                    3,
                ),
                "final_status": final_status,
                "attempt_count": attempt_count,
                "queue_wait_seconds": queue_wait_seconds,
                "processing_seconds": processing_seconds,
                "error_message": error_message,
            }

            with results_lock:
                results.append(result)

            print()
            print(
                f"{pdf_path.name} finished: "
                f"{final_status}"
            )

            print(
                f"End-to-end: "
                f"{end_to_end_seconds:.2f}s"
            )

            # =================================================
            # CRITICAL:
            #
            # This user is permanently stopped after ONE PDF.
            # =================================================

            raise StopUser()
