
import io
import os

from datasets import load_dataset
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ID = "HuggingFaceFW/ocr-annotations"

OUTPUT_DIR = "/home/mahesh/Documents//test_pdf"

# We want 50 PDFs TOTAL in the folder.
TARGET_PDFS = 50

# Only accept PDFs with 3 or 4 pages.
MIN_PAGES = 3
MAX_PAGES = 4


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK EXISTING PDFs
# ============================================================

existing_pdfs = sorted(
    filename
    for filename in os.listdir(OUTPUT_DIR)
    if filename.startswith("ocr_test_")
    and filename.endswith(".pdf")
)

existing_count = len(existing_pdfs)

remaining_pdfs = TARGET_PDFS - existing_count


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OCR PDF DOWNLOADER")
print("=" * 70)

print(f"Dataset        : {DATASET_ID}")
print(f"Output         : {OUTPUT_DIR}")
print(f"Existing PDFs  : {existing_count}")
print(f"Target PDFs    : {TARGET_PDFS}")
print(f"Need to add    : {max(remaining_pdfs, 0)}")
print(f"Pages allowed  : {MIN_PAGES}-{MAX_PAGES}")
print()


# ============================================================
# ALREADY COMPLETE?
# ============================================================

if remaining_pdfs <= 0:

    print("=" * 70)
    print("DOWNLOAD NOT REQUIRED")
    print("=" * 70)

    print(
        f"The output directory already contains "
        f"{existing_count} PDF files."
    )

    print(
        f"Target is {TARGET_PDFS} PDFs."
    )

    print()
    print("Nothing to download.")

    raise SystemExit(0)


# ============================================================
# LOAD PUBLIC HUGGING FACE DATASET
# ============================================================

print("=" * 70)
print("LOADING HUGGING FACE DATASET")
print("=" * 70)
print()

print("Loading dataset...")
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

    # --------------------------------------------------------
    # Stop when we have enough NEW PDFs.
    # --------------------------------------------------------

    if downloaded >= remaining_pdfs:
        break

    checked += 1

    print("=" * 70)
    print(f"Checking document #{checked}")
    print("=" * 70)


    # --------------------------------------------------------
    # GET PDF FROM DATASET
    # --------------------------------------------------------

    pdf_data = item.get("pdf")

    if pdf_data is None:

        print("No PDF found. Skipping.")
        print()

        continue


    # --------------------------------------------------------
    # HANDLE PDF DATA
    # --------------------------------------------------------

    if isinstance(pdf_data, bytes):

        pdf_bytes = pdf_data


    elif isinstance(pdf_data, dict):

        # --------------------------------------------
        # PDF returned directly as bytes
        # --------------------------------------------

        if pdf_data.get("bytes") is not None:

            pdf_bytes = pdf_data["bytes"]


        # --------------------------------------------
        # PDF returned as a local path
        # --------------------------------------------

        elif pdf_data.get("path") is not None:

            try:

                with open(
                    pdf_data["path"],
                    "rb",
                ) as file:

                    pdf_bytes = file.read()

            except Exception as error:

                print(
                    f"Could not read PDF path: {error}"
                )

                print()

                continue


        else:

            print(
                "PDF has no bytes/path. Skipping."
            )

            print()

            continue


    else:

        print(
            "Unexpected PDF type: "
            f"{type(pdf_data)}"
        )

        print()

        continue


    # --------------------------------------------------------
    # VALIDATE PDF PAGE COUNT
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        page_count = len(reader.pages)

    except Exception as error:

        print(
            f"Could not read PDF: {error}"
        )

        print()

        continue


    print(
        f"Pages found: {page_count}"
    )


    # --------------------------------------------------------
    # ACCEPT ONLY 3-4 PAGE PDFs
    # --------------------------------------------------------

    if not (
        MIN_PAGES
        <= page_count
        <= MAX_PAGES
    ):

        print(
            f"Skipping - PDF has {page_count} pages."
        )

        print(
            f"Required: {MIN_PAGES}-{MAX_PAGES} pages."
        )

        print()

        continue


    # --------------------------------------------------------
    # CALCULATE NEXT FILE NUMBER
    # --------------------------------------------------------

    file_number = existing_count + downloaded + 1

    filename = (
        f"ocr_test_{file_number:02d}.pdf"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------
    # Never overwrite an existing PDF.
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print(
            f"File already exists: {filename}"
        )

        print(
            "Skipping to avoid overwriting."
        )

        print()

        continue


    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    try:

        with open(
            output_path,
            "wb",
        ) as file:

            file.write(pdf_bytes)

    except Exception as error:

        print(
            f"Could not save PDF: {error}"
        )

        print()

        continue


    # --------------------------------------------------------
    # UPDATE COUNTERS
    # --------------------------------------------------------

    downloaded += 1


    # --------------------------------------------------------
    # PRINT SUCCESS
    # --------------------------------------------------------

    file_size_mb = (
        len(pdf_bytes)
        / (1024 * 1024)
    )

    print()
    print(
        f"✓ SAVED: {filename}"
    )

    print(
        f"  Pages : {page_count}"
    )

    print(
        f"  Size  : {file_size_mb:.3f} MB"
    )

    print(
        f"  Progress: "
        f"{downloaded}/{remaining_pdfs} new PDFs"
    )

    print(
        f"  Total: "
        f"{existing_count + downloaded}/{TARGET_PDFS} PDFs"
    )

    print()


# ============================================================
# FINAL RESULT
# ============================================================

final_pdfs = sorted(
    filename
    for filename in os.listdir(OUTPUT_DIR)
    if filename.startswith("ocr_test_")
    and filename.endswith(".pdf")
)

final_count = len(final_pdfs)


print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"Previously existing : {existing_count}"
)

print(
    f"Newly downloaded    : {downloaded}"
)

print(
    f"Total PDFs          : {final_count}/{TARGET_PDFS}"
)

print(
    f"Output directory    : {OUTPUT_DIR}"
)

print()


# ============================================================
# VERIFY FINAL RESULT
# ============================================================

if final_count >= TARGET_PDFS:

    print(
        f"✓ SUCCESS: {final_count} PDFs are available."
    )

    print(
        f"✓ Files are stored in:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

else:

    print(
        f"⚠ WARNING: Only {final_count} PDFs "
        f"are available."
    )

    print(
        f"Required: {TARGET_PDFS}"
    )

    print(
        f"Missing : {TARGET_PDFS - final_count}"
    )

print()


# ============================================================
# PRINT FILE LIST
# ============================================================

print("=" * 70)
print("PDF FILES")
print("=" * 70)

for filename in final_pdfs:

    print(filename)

print()
