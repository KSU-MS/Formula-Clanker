'''
So I had Claude fix an issue and it went ahead and commented the code too. oops...
'''
from pdf2image import convert_from_path
import pytesseract
import filetype
from pdf2image.pdf2image import pdfinfo_from_path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pymupdf4llm
import pymupdf4llm.helpers.document_layout as dl
import os

# -----------------------------
# Safety Patch for pymupdf4llm
# -----------------------------
_original_list_item_to_md = dl.list_item_to_md

def safe_list_item_to_md(textlines, level):
    if not textlines:
        return ""
    return _original_list_item_to_md(textlines, level)

dl.list_item_to_md = safe_list_item_to_md


# -----------------------------
# Worker for OCR (runs in parallel)
# -----------------------------
def ocr_page(args):
    """
    Worker function for multiprocessing pool.
    args: (filepath, page_number, dpi)
    Returns: (page_number, text_or_error)
    """
    file, page_number, dpi = args
    try:
        # convert only the requested page
        imgs = convert_from_path(
            file,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            fmt='ppm'  # use a light-weight format to reduce I/O overhead
        )
        if not imgs:
            return page_number, f"[ERROR page {page_number}: no image produced]"
        img = imgs[0]
        text = pytesseract.image_to_string(img)
        return page_number, text
    except Exception as e:
        return page_number, f"[ERROR page {page_number}: {repr(e)}]"


# -----------------------------
# Main Extract Function
# -----------------------------
def main(file, format="text", workers=None, dpi_text=800, dpi_md=800):
    """
    file: path to PDF
    format: "text" or "markdown"
    workers: number of parallel workers (defaults to cpu_count())
    dpi_text: dpi for text extraction (higher dpi -> better OCR, more CPU/RAM)
    dpi_md: dpi for fallback markdown OCR
    """
    if workers is None:
        workers = max(1, cpu_count() - 0)  # allow tuning here

    kind = filetype.guess(file)
    if kind is None:
        print(f"Couldn't guess file type for '{file}'")
        return "--unsupported file--"
    print(f"file '{file}' is a '{kind.extension}' file")

    if kind.extension != "pdf":
        print(f"unsupported file type '{kind.extension}'!")
        return "--unsupported file--"

    # -----------------------------
    # TEXT mode (parallel OCR)
    # -----------------------------
    if format == "text":
        info = pdfinfo_from_path(file)
        total_pages = info.get("Pages", 0)
        if total_pages == 0:
            print("PDF has zero pages or couldn't read page count.")
            return ""

        print(f"Using {workers} worker(s) for OCR; DPI={dpi_text}")
        args = [(file, page, dpi_text) for page in range(1, total_pages + 1)]

        texts = [None] * total_pages
        with Pool(workers) as pool:
            for page_num, text in tqdm(pool.imap_unordered(ocr_page, args),
                                        total=total_pages,
                                        desc="OCR Progress",
                                        unit="page"):
                texts[page_num - 1] = text

        return "\n".join(texts)

    # -----------------------------
    # MARKDOWN mode
    # -----------------------------
    elif format == "markdown":
        # First, try the library's to_markdown with its native signature
        try:
            print("Attempting high-quality pymupdf4llm.to_markdown() (library default)...")
            markdown = pymupdf4llm.to_markdown(file)  # removed unsupported kwargs
            print("High-quality extraction succeeded.")
            return markdown

        except Exception as e:
            print(f"High-quality markdown extraction failed: {repr(e)}")
            print("Falling back to multiprocess page-by-page OCR -> markdown...")

            info = pdfinfo_from_path(file)
            total_pages = info.get("Pages", 0)
            if total_pages == 0:
                print("PDF has zero pages or couldn't read page count.")
                return ""

            print(f"Using {workers} worker(s) for fallback OCR; DPI={dpi_md}")
            args = [(file, page, dpi_md) for page in range(1, total_pages + 1)]

            results = [None] * total_pages
            with Pool(workers) as pool:
                for page_num, text in tqdm(pool.imap_unordered(ocr_page, args),
                                            total=total_pages,
                                            desc="OCR Markdown Fallback",
                                            unit="page"):
                    # keep order by placing into results at index page_num-1
                    results[page_num - 1] = text

            # assemble markdown with simple page separators
            md_blocks = []
            for i, page_text in enumerate(results, start=1):
                md_blocks.append(f"# Page {i}\n\n{page_text}\n\n---\n")

            return "\n".join(md_blocks)

    else:
        print(f"Unknown format '{format}'")
        return "--unknown-format--"