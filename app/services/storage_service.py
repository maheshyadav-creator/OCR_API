from pathlib import Path
from uuid import UUID

from fastapi import UploadFile


# --------------------------------------------------
# Root directory where uploaded files are stored
# --------------------------------------------------

STORAGE_ROOT = Path("/app/storage")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
# --------------------------------------------------
# Save uploaded file
# --------------------------------------------------

async def save_uploaded_file(
    job_id: UUID,
    filename: str,
    file: UploadFile,
    max_size_bytes: int,
) -> tuple[str, int]:
    """
    Save an uploaded file to disk in chunks.

    Returns:
        tuple[str, int]:
            - saved file path
            - total file size in bytes
    """

    # ----------------------------------------------
    # 1. Create jobs directory
    # ----------------------------------------------

    jobs_directory = STORAGE_ROOT / "jobs"

    jobs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # 2. Get original file extension
    # ----------------------------------------------

    suffix = Path(filename).suffix.lower()

    if not suffix:
        suffix = ".bin"

    # ----------------------------------------------
    # 3. Create unique file path
    # ----------------------------------------------

    file_path = jobs_directory / f"{job_id}{suffix}"

    total_size = 0

    try:

        # ------------------------------------------
        # 4. Open destination file
        # ------------------------------------------

        with file_path.open("wb") as output_file:

            # --------------------------------------
            # 5. Read uploaded file in chunks
            # --------------------------------------

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                # ----------------------------------
                # 6. Update total size
                # ----------------------------------

                total_size += len(chunk)

                # ----------------------------------
                # 7. Check maximum allowed size
                # ----------------------------------

                if total_size > max_size_bytes:
                    raise ValueError("FILE_TOO_LARGE")

                # ----------------------------------
                # 8. Write chunk to disk
                # ----------------------------------

                output_file.write(chunk)

        return str(file_path), total_size

    except Exception:

        # ------------------------------------------
        # Delete partially written file if failed
        # ------------------------------------------

        file_path.unlink(missing_ok=True)

        raise