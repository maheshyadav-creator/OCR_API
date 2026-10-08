import os
import threading
import time
import uuid
from pathlib import Path

import psycopg2

from locust import HttpUser, between, events, task


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/home/mahesh/Documents/OCR_API"
)

PDF_DIR = PROJECT_ROOT / "test_pdf"

PDFS = sorted(
    PDF_DIR.glob("ocr_test_*.pdf")
)

PDFS_PER_USER = int(
    os.getenv("PDFS_PER_USER", "5")
)

LOCUST_TOTAL_USERS = int(
    os.getenv("LOCUST_TOTAL_USERS", "1")
)


# ============================================================
# DATABASE MONITOR CONFIGURATION
# ============================================================

DB_HOST = os.getenv(
    "LOCUST_DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "LOCUST_DB_PORT",
        "5432",
    )
)

DB_NAME = os.getenv(
    "LOCUST_DB_NAME",
    "ocr_db",
)

DB_USER = os.getenv(
    "LOCUST_DB_USER",
    "ocr_user",
)

DB_PASSWORD = os.getenv(
    "LOCUST_DB_PASSWORD",
    "ocr_password",
)

MONITOR_INTERVAL_SECONDS = float(
    os.getenv(
        "LOCUST_MONITOR_INTERVAL",
        "5",
    )
)

# ============================================================
# PER-RUN STATE
# ============================================================

# IMPORTANT:
#
# These values are reset at every Locust test_start event.
#
# This means:
#
# START
#   -> fresh counters
#
# STOP
# START
#   -> fresh counters again
#
# No stale state from the previous run.

LOCUST_RUN_ID = ""

_next_user_number = 0
_next_pdf_index = 0

_uploaded_jobs = 0

_test_finished = False

_monitor_started = False
_monitor_lock = threading.Lock()
# ============================================================
# LOCKS
# ============================================================

_user_assignment_lock = threading.Lock()

_pdf_assignment_lock = threading.Lock()

_upload_counter_lock = threading.Lock()

_test_finish_lock = threading.Lock()


# ============================================================
# LOCUST TEST START
# ============================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Reset ALL benchmark state whenever a new Locust
    test is started.

    This is important because the Locust web UI can
    start multiple tests without restarting the Locust
    process.
    """

    global LOCUST_RUN_ID
    global _next_user_number
    global _next_pdf_index
    global _uploaded_jobs
    global _test_finished

    LOCUST_RUN_ID = (
        f"LOCUST-{uuid.uuid4().hex[:12].upper()}"
    )

    _next_user_number = 0
    _next_pdf_index = 0
    _uploaded_jobs = 0
    _test_finished = False

    print()
    print("=" * 80)
    print("LOCUST TEST STARTED")
    print(f"RUN ID              : {LOCUST_RUN_ID}")
    print(f"TOTAL USERS         : {LOCUST_TOTAL_USERS}")
    print(f"PDFs PER USER       : {PDFS_PER_USER}")
    print(
        f"EXPECTED TOTAL JOBS : "
        f"{LOCUST_TOTAL_USERS * PDFS_PER_USER}"
    )
    print("=" * 80)
    print()


# ============================================================
# LOCUST TEST STOP
# ============================================================

@events.test_stop.add_listener
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Print final Locust upload-side information.
    """

    expected_jobs = (
        LOCUST_TOTAL_USERS
        * PDFS_PER_USER
    )

    print()
    print("=" * 80)
    print("LOCUST TEST FINISHED")
    print(f"RUN ID              : {LOCUST_RUN_ID}")
    print(
        f"EXPECTED UPLOADS    : "
        f"{expected_jobs}"
    )
    print(
        f"ACTUAL UPLOADS      : "
        f"{_uploaded_jobs}"
    )
    print("=" * 80)
    print()


# ============================================================
# PDF ASSIGNMENT
# ============================================================

def get_next_pdf() -> Path:
    """
    Assign exactly one PDF from the global PDF sequence.

    Assignment is shared across Locust users so that the
    same PDF is not assigned twice during one run.
    """

    global _next_pdf_index

    with _pdf_assignment_lock:

        if not PDFS:
            raise RuntimeError(
                f"No PDFs found in {PDF_DIR}"
            )

        pdf = PDFS[
            _next_pdf_index % len(PDFS)
        ]

        _next_pdf_index += 1

        return pdf


# ============================================================
# USER NUMBER ASSIGNMENT
# ============================================================

def get_next_user_number() -> int:
    """
    Assign a stable sequential number to each Locust user.
    """

    global _next_user_number

    with _user_assignment_lock:

        _next_user_number += 1

        return _next_user_number


# ============================================================
# UPLOAD COUNTER
# ============================================================

def increment_uploaded_jobs() -> int:
    """
    Increment the number of successfully submitted HTTP
    upload requests.

    Returns the new total.
    """

    global _uploaded_jobs

    with _upload_counter_lock:

        _uploaded_jobs += 1

        return _uploaded_jobs
# ============================================================
# OCR PIPELINE MONITOR
# ============================================================

def monitor_ocr_pipeline(environment) -> None:
    """
    Monitor the asynchronous OCR pipeline directly through
    PostgreSQL.

    This does NOT use self.client.

    Therefore these monitoring operations do NOT appear as
    additional Locust HTTP requests.

    The monitor continues until every expected job reaches
    a terminal state:

        completed
        failed
    """

    expected_jobs = (
        LOCUST_TOTAL_USERS
        * PDFS_PER_USER
    )

    print()
    print("=" * 80)
    print("OCR PIPELINE MONITOR STARTED")
    print(f"RUN ID              : {LOCUST_RUN_ID}")
    print(f"EXPECTED JOBS       : {expected_jobs}")
    print(
        f"CHECK INTERVAL      : "
        f"{MONITOR_INTERVAL_SECONDS}s"
    )
    print("=" * 80)
    print()

    last_state = None

    while True:

        try:

            connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=5,
            )

            try:

                with connection.cursor() as cursor:

                    cursor.execute(
                        """
                        SELECT
                            COUNT(*) AS total_jobs,

                            COUNT(*) FILTER (
                                WHERE status = 'completed'
                            ) AS completed_jobs,

                            COUNT(*) FILTER (
                                WHERE status = 'processing'
                            ) AS processing_jobs,

                            COUNT(*) FILTER (
                                WHERE status = 'queued'
                            ) AS queued_jobs,

                            COUNT(*) FILTER (
                                WHERE status = 'retrying'
                            ) AS retrying_jobs,

                            COUNT(*) FILTER (
                                WHERE status = 'failed'
                            ) AS failed_jobs

                        FROM ocr_jobs

                        WHERE locust_run_id = %s
                        """,
                        (
                            LOCUST_RUN_ID,
                        ),
                    )

                    row = cursor.fetchone()

            finally:

                connection.close()

            (
                total_jobs,
                completed_jobs,
                processing_jobs,
                queued_jobs,
                retrying_jobs,
                failed_jobs,
            ) = row

            state = (
                total_jobs,
                completed_jobs,
                processing_jobs,
                queued_jobs,
                retrying_jobs,
                failed_jobs,
            )

            # ------------------------------------------------
            # Only print when something changed.
            # ------------------------------------------------

            if state != last_state:

                print()
                print(
                    f"[OCR PIPELINE] "
                    f"RUN {LOCUST_RUN_ID}"
                )

                print(
                    f"  Uploaded/Jobs : "
                    f"{total_jobs}/{expected_jobs}"
                )

                print(
                    f"  Completed     : "
                    f"{completed_jobs}/{expected_jobs}"
                )

                print(
                    f"  Processing    : "
                    f"{processing_jobs}"
                )

                print(
                    f"  Queued        : "
                    f"{queued_jobs}"
                )

                print(
                    f"  Retrying      : "
                    f"{retrying_jobs}"
                )

                print(
                    f"  Failed        : "
                    f"{failed_jobs}"
                )

                last_state = state

            # ------------------------------------------------
            # Terminal condition.
            #
            # Every job must be either:
            #
            # completed OR failed
            #
            # This means there is no queued, processing or
            # retrying work left.
            # ------------------------------------------------

            terminal_jobs = (
                completed_jobs
                + failed_jobs
            )

            if (
                total_jobs == expected_jobs
                and terminal_jobs == expected_jobs
            ):

                print()
                print("=" * 80)
                print("OCR PIPELINE FINISHED")
                print(f"RUN ID              : {LOCUST_RUN_ID}")
                print(
                    f"TOTAL JOBS          : "
                    f"{total_jobs}"
                )
                print(
                    f"COMPLETED           : "
                    f"{completed_jobs}"
                )
                print(
                    f"FAILED              : "
                    f"{failed_jobs}"
                )
                print(
                    f"PROCESSING          : "
                    f"{processing_jobs}"
                )
                print(
                    f"QUEUED              : "
                    f"{queued_jobs}"
                )
                print(
                    f"RETRYING            : "
                    f"{retrying_jobs}"
                )
                print("=" * 80)
                print()

                environment.runner.quit()

                return

        except Exception as exc:

            print(
                f"[OCR PIPELINE MONITOR] "
                f"Database check failed: {exc}"
            )

        time.sleep(
            MONITOR_INTERVAL_SECONDS
        )

# ============================================================
# FINISH TEST
# ============================================================
# ============================================================
# FINISH UPLOAD PHASE
# ============================================================

def finish_test_if_complete(environment) -> None:
    """
    Called after every successful upload.

    Once all expected PDFs have been submitted, we DO NOT
    stop Locust immediately.

    Instead we start a background PostgreSQL monitor.

    The monitor waits for Celery/PyMuPDF4LLM/PaddleOCR to
    finish all jobs.
    """

    global _test_finished
    global _monitor_started

    expected_jobs = (
        LOCUST_TOTAL_USERS
        * PDFS_PER_USER
    )

    with _test_finish_lock:

        if _uploaded_jobs < expected_jobs:
            return

        _test_finished = True

        print()
        print("=" * 80)
        print("ALL EXPECTED UPLOADS SUBMITTED")
        print(f"RUN ID           : {LOCUST_RUN_ID}")
        print(
            f"EXPECTED UPLOADS : "
            f"{expected_jobs}"
        )
        print(
            f"ACTUAL UPLOADS   : "
            f"{_uploaded_jobs}"
        )
        print("=" * 80)
        print()

        # ----------------------------------------------------
        # Start OCR pipeline monitor only once.
        # ----------------------------------------------------

        if _monitor_started:
            return

        _monitor_started = True

        monitor_thread = threading.Thread(
            target=monitor_ocr_pipeline,
            args=(environment,),
            daemon=True,
            name="ocr-pipeline-monitor",
        )

        monitor_thread.start()



# ============================================================
# LOCUST USER
# ============================================================

class OcrApiUser(HttpUser):

    # No artificial delay between uploads.
    wait_time = between(0, 0)

    # --------------------------------------------------------
    # USER START
    # --------------------------------------------------------

    def on_start(self) -> None:

        # ----------------------------------------------------
        # Assign stable user number.
        # ----------------------------------------------------

        self.user_number = (
            get_next_user_number()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The number of users is determined by the environment
        # variable, NOT by how many users have already started.
        #
        # Example:
        #
        # LOCUST_TOTAL_USERS=4
        #
        # Every user knows:
        #
        # TOTAL USERS = 4
        # ----------------------------------------------------

        self.total_users = (
            LOCUST_TOTAL_USERS
        )

        # ----------------------------------------------------
        # Assign exactly PDFS_PER_USER PDFs.
        # ----------------------------------------------------

        self.pdf_paths: list[Path] = []

        for _ in range(PDFS_PER_USER):

            self.pdf_paths.append(
                get_next_pdf()
            )

        self.current_pdf_index = 0

        self.finished_work = False

        # ----------------------------------------------------
        # Print user information.
        # ----------------------------------------------------

        expected_jobs = (
            self.total_users
            * PDFS_PER_USER
        )

        print()
        print("-" * 70)
        print("LOCUST USER STARTED")
        print(f"RUN ID              : {LOCUST_RUN_ID}")
        print(
            f"USER                : "
            f"{self.user_number}/{self.total_users}"
        )
        print(
            f"PDFs per user       : "
            f"{PDFS_PER_USER}"
        )
        print(
            f"EXPECTED TOTAL JOBS : "
            f"{expected_jobs}"
        )
        print("-" * 70)
        print()

    # ========================================================
    # UPLOAD ONE PDF
    # ========================================================

    @task
    def upload_pdf(self) -> None:

        # ----------------------------------------------------
        # If this user has completed its workload, DO NOT STOP
        # THE USER.
        #
        # IMPORTANT:
        #
        # self.stop()
        #
        # was the cause of Locust creating users 5,6,7...
        #
        # Instead we simply return.
        #
        # The user remains alive and does not generate another
        # upload because finished_work=True.
        # ----------------------------------------------------

        if self.finished_work:
            return

        # ----------------------------------------------------
        # Safety check.
        # ----------------------------------------------------

        if (
            self.current_pdf_index
            >= len(self.pdf_paths)
        ):

            self.finished_work = True

            return

        # ----------------------------------------------------
        # Select next PDF.
        # ----------------------------------------------------

        pdf_number = (
            self.current_pdf_index + 1
        )

        pdf_path = self.pdf_paths[
            self.current_pdf_index
        ]

        self.current_pdf_index += 1

        # ----------------------------------------------------
        # Expected total jobs.
        # ----------------------------------------------------

        expected_jobs = (
            self.total_users
            * PDFS_PER_USER
        )

        # ----------------------------------------------------
        # Print upload information.
        # ----------------------------------------------------

        print(
            f"[LOCUST RUN {LOCUST_RUN_ID}] "
            f"USER {self.user_number}/"
            f"{self.total_users} "
            f"PDF {pdf_number}/"
            f"{PDFS_PER_USER} "
            f"-> {pdf_path.name}"
        )

        # ----------------------------------------------------
        # Headers sent to FastAPI.
        # ----------------------------------------------------

        headers = {

            "X-Locust-Run-ID":
                LOCUST_RUN_ID,

            "X-Locust-User-ID":
                str(id(self)),

            "X-Locust-User-Number":
                str(self.user_number),

            "X-Locust-Total-Users":
                str(self.total_users),

            "X-Locust-PDF-Number":
                str(pdf_number),

            "X-Locust-PDFs-Per-User":
                str(PDFS_PER_USER),

            "X-Locust-Expected-Jobs":
                str(expected_jobs),
        }

        # ----------------------------------------------------
        # Upload PDF.
        # ----------------------------------------------------

        with pdf_path.open("rb") as pdf_file:

            response = self.client.post(
                "/ocr",
                files={
                    "files": (
                        pdf_path.name,
                        pdf_file,
                        "application/pdf",
                    )
                },
                headers=headers,
                name="OCR Upload",
                timeout=60,
            )

        # ----------------------------------------------------
        # Process response.
        # ----------------------------------------------------

        if response.ok:

            # ------------------------------------------------
            # Count this upload.
            #
            # This is an HTTP submission count.
            #
            # It is NOT yet an OCR-completed count because
            # your API is asynchronous.
            # ------------------------------------------------

            total_uploaded = (
                increment_uploaded_jobs()
            )

            # ------------------------------------------------
            # Print API response.
            # ------------------------------------------------

            try:

                response_data = (
                    response.json()
                )

                jobs = response_data.get(
                    "jobs",
                    [],
                )

                if jobs:

                    job_id = jobs[0].get(
                        "job_id",
                        "unknown",
                    )

                    print(
                        f"[LOCUST RUN {LOCUST_RUN_ID}] "
                        f"USER {self.user_number}/"
                        f"{self.total_users} "
                        f"PDF {pdf_number}/"
                        f"{PDFS_PER_USER} "
                        f"JOB {job_id} "
                        f"STATUS queued "
                        f"TOTAL UPLOADED "
                        f"{total_uploaded}/"
                        f"{expected_jobs}"
                    )

            except Exception:

                print(
                    f"[LOCUST RUN {LOCUST_RUN_ID}] "
                    f"Upload succeeded but response "
                    f"could not be parsed."
                )

            # ------------------------------------------------
            # Mark this user as finished after its assigned
            # PDFs have all been uploaded.
            # ------------------------------------------------

            if (
                self.current_pdf_index
                >= len(self.pdf_paths)
            ):

                self.finished_work = True

                print(
                    f"[LOCUST RUN {LOCUST_RUN_ID}] "
                    f"USER {self.user_number}/"
                    f"{self.total_users} "
                    f"FINISHED "
                    f"{PDFS_PER_USER} PDFs"
                )

            # ------------------------------------------------
            # If ALL expected uploads are done, stop the
            # ENTIRE Locust test.
            # ------------------------------------------------

            finish_test_if_complete(
                self.environment
            )

        else:

            print(
                f"[LOCUST RUN {LOCUST_RUN_ID}] "
                f"USER {self.user_number}/"
                f"{self.total_users} "
                f"PDF {pdf_number}/"
                f"{PDFS_PER_USER} "
                f"UPLOAD FAILED "
                f"HTTP {response.status_code}"
            )
