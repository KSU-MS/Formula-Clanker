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
import glob

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
def main(file_or_folder, format="text", workers=None, dpi_text=800, dpi_md=800, output_dir=None):
    """
    file_or_folder: path to PDF or folder containing PDFs
    format: "text" or "markdown"
    workers: number of parallel workers (defaults to cpu_count())
    dpi_text: dpi for text extraction (higher dpi -> better OCR, more CPU/RAM)
    dpi_md: dpi for fallback markdown OCR
    output_dir: directory to save extracted text/markdown files (optional)
    """
    if workers is None:
        workers = max(1, cpu_count() - 0)  # allow tuning here

    # Check if input is a folder or single file
    if os.path.isdir(file_or_folder):
        # Process all PDFs in the folder and subfolders
        pdf_files = []
        for root, dirs, files in os.walk(file_or_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            print(f"No PDF files found in '{file_or_folder}' or its subfolders")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF file
        results = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            try:
                result = process_single_pdf(pdf_file, format, workers, dpi_text, dpi_md)
                results.append((pdf_file, result))
                
                # Save to file if output_dir is specified
                if output_dir:
                    save_result_to_file(pdf_file, result, format, output_dir)
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {repr(e)}")
                results.append((pdf_file, f"[ERROR] {repr(e)}"))
        
        return results
        
    else:
        # Single file processing (original behavior)
        return process_single_pdf(file_or_folder, format, workers, dpi_text, dpi_md)


def process_single_pdf(file, format="text", workers=None, dpi_text=800, dpi_md=800):
    """Process a single PDF file"""
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
            print("Attempting high-quality pymupdf4llm.to_markdown() (library default).")
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


def save_result_to_file(input_file, result, format, output_dir):
    """Save extraction result to a file preserving directory structure"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Get the relative path from the input folder to preserve directory structure
    # This requires knowing the base directory of the input folder
    # For simplicity, we'll use the directory structure relative to the input folder
    
    # Create a relative path from the input folder to the current file
    # This preserves the folder structure in the output directory
    if os.path.isdir(input_file):
        # If input is a directory, we're saving to the output directory directly
        output_file = os.path.join(output_dir, f"{filename}.{format}")
    else:
        # If input is a file, we preserve the path structure
        # Get the base directory of the input folder
        input_base = os.path.abspath(input_file)
        while not os.path.isdir(input_base):
            input_base = os.path.dirname(input_base)
        
        # Get the relative path from input folder to current file
        rel_path = os.path.relpath(input_file, input_base)
        # Remove the filename from the path to get just the directory structure
        rel_dir = os.path.dirname(rel_path)
        
        # Create the output directory structure
        if rel_dir:
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
            output_file = os.path.join(output_subdir, f"{filename}.{format}")
        else:
            output_file = os.path.join(output_dir, f"{filename}.{format}")
    
    # Write result to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Saved result to {output_file}")
    except Exception as e:
        print(f"Error saving to file {output_file}: {repr(e)}")


# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__":
    import sys
    
    # Default values
    format = "text"
    workers = None
    dpi_text = 800
    dpi_md = 800
    output_dir = None
    input_path = None
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--format" and i + 1 < len(sys.argv):
            format = sys.argv[i + 1]
            i += 2
        elif arg == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
            i += 2
        elif arg == "--dpi-text" and i + 1 < len(sys.argv):
            dpi_text = int(sys.argv[i + 1])
            i += 2
        elif arg == "--dpi-md" and i + 1 < len(sys.argv):
            dpi_md = int(sys.argv[i + 1])
            i += 2
        elif arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            input_path = arg
            i += 1
    
    # Validate input
    if not input_path:
        print("Usage: python extractText.py <input_path> [options]")
        print("Options:")
        print("  --format <text|markdown>     Output format (default: text)")
        print("  --workers <number>           Number of parallel workers (default: CPU count)")
        print("  --dpi-text <number>          DPI for text extraction (default: 800)")
        print("  --dpi-md <number>            DPI for markdown fallback (default: 800)")
        print("  --output-dir <path>          Directory to save output files")
        sys.exit(1)
    
    # Process the input
    results = main(input_path, format, workers, dpi_text, dpi_md, output_dir)
    
    # Print summary for folder processing
    if os.path.isdir(input_path) and results:
        print(f"\nProcessed {len(results)} PDF files:")
        for pdf_file, result in results:
            if result.startswith("[ERROR]"):
                print(f"  {pdf_file}: ERROR")
            else:
                print(f"  {pdf_file}: SUCCESS")