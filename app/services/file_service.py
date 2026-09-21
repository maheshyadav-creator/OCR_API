from pathlib import Path

from fastapi import HTTPException, UploadFile, status




ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".pdf",
}


def validate_uploaded_file(file: UploadFile) -> str:
    """
    Validate the uploaded file extension.

    Returns:
        The normalized file extension.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have an extension.",
        )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported file type. "
                "Supported formats: "
                "JPG, JPEG, PNG, BMP, WEBP, TIFF and PDF."
            ),
        )

    return extension