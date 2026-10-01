import io
import os

from datasets import load_dataset
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ID = "HuggingFaceFW/ocr-annotations"

OUTPUT_DIR = "/home/mahesh/Documents/OCR_API/test_pdf"

TARGET_PDFS = 10

MIN_PAGES = 3
MAX_PAGES = 4


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("OCR PDF DOWNLOADER")
print("=" * 60)

print(f"Dataset : {DATASET_ID}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Target  : {TARGET_PDFS} PDFs")
print(f"Pages   : {MIN_PAGES}-{MAX_PAGES}")
print()


# ============================================================
# LOAD PUBLIC HUGGING FACE DATASET
# ============================================================

print("Loading Hugging Face dataset...")
print()

dataset = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
)

print("Dataset loaded successfully.")
print()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

downloaded = 0
checked = 0


for item in dataset:

    # Stop after getting 10 PDFs
    if downloaded >= TARGET_PDFS:
        break

    checked += 1

    print(f"Checking document #{checked}...")

    # --------------------------------------------------------
    # Get PDF
    # --------------------------------------------------------

    pdf_data = item.get("pdf")

    if pdf_data is None:
        print("  No PDF found. Skipping.")
        continue


    # --------------------------------------------------------
    # Handle PDF data
    # --------------------------------------------------------

    if isinstance(pdf_data, bytes):

        pdf_bytes = pdf_data

    elif isinstance(pdf_data, dict):

        if pdf_data.get("bytes") is not None:

            pdf_bytes = pdf_data["bytes"]

        elif pdf_data.get("path") is not None:

            try:

                with open(pdf_data["path"], "rb") as file:
                    pdf_bytes = file.read()

            except Exception as error:

                print(f"  Could not read PDF: {error}")
                continue

        else:

            print("  PDF has no bytes/path. Skipping.")
            continue

    else:

        print(
            f"  Unexpected PDF type: "
            f"{type(pdf_data)}"
        )

        continue


    # --------------------------------------------------------
    # Check page count
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        page_count = len(reader.pages)

    except Exception as error:

        print(
            f"  Could not read PDF: {error}"
        )

        continue


    print(
        f"  Pages: {page_count}"
    )


    # --------------------------------------------------------
    # Accept ONLY 3 or 4 page PDFs
    # --------------------------------------------------------

    if not (MIN_PAGES <= page_count <= MAX_PAGES):

        print("  Skipping - not 3 or 4 pages.")
        continue


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    downloaded += 1

    filename = f"ocr_test_{downloaded:02d}.pdf"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    try:

        with open(output_path, "wb") as file:
            file.write(pdf_bytes)

    except Exception as error:

        print(
            f"  Could not save PDF: {error}"
        )

        downloaded -= 1
        continue


    print(
        f"  ✓ SAVED: {filename} "
        f"({page_count} pages)"
    )

    print()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(
    f"PDFs downloaded : {downloaded}/{TARGET_PDFS}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print()

if downloaded == TARGET_PDFS:

    print(
        "✓ Successfully downloaded "
        "10 PDFs with 3-4 pages each."
    )

else:

    print(
        f"⚠ Only {downloaded} suitable PDFs "
        f"were found."
    )