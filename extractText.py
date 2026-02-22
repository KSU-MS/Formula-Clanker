from pdf2image import convert_from_path
import pytesseract
import filetype
from pdf2image.pdf2image import pdfinfo_from_path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pymupdf4llm
import pymupdf4llm.helpers.document_layout as dl
import os
import sys

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
# Get page count with fallback
# -----------------------------
def get_page_count(file_path):
    """Get page count with fallback for when pdfinfo fails"""
    try:
        # Try the original method first
        info = pdfinfo_from_path(file_path)
        if 'Pages' in info:
            return int(info['Pages'])
    except Exception as e:
        print(f"Warning: Could not get page count using pdfinfo: {e}")
    
    # Fallback: try to get page count using pymupdf
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception as e:
        print(f"Warning: Could not get page count using PyMuPDF: {e}")
    
    # Final fallback: try to get page count using pdfinfo command
    try:
        import subprocess
        result = subprocess.run(['pdfinfo', file_path], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                return int(line.split(':')[1].strip())
    except Exception as e:
        print(f"Warning: Could not get page count using pdfinfo command: {e}")
    
    # If all methods fail, assume 1 page
    print(f"Warning: Could not determine page count for {file_path}, assuming 1 page")
    return 1


# -----------------------------
# Save result to file with directory structure preservation
# -----------------------------
def save_result_to_file(input_file, result, format, output_dir, input_base_dir):
    """Save extraction result to file preserving directory structure"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Calculate relative path from input base directory to current file
    rel_path = os.path.relpath(input_file, input_base_dir)
    
    # Get the directory part of the relative path
    rel_dir = os.path.dirname(rel_path)
    
    # Create output subdirectory if needed
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
# Main Extract Function
# -----------------------------
def main(file_or_folder, format="markdown", workers=None, dpi_text=800, dpi_md=800, output_dir=None):
    """
    file_or_folder: path to PDF or folder containing PDFs
    format: "text" or "markdown"
    workers: number of parallel workers (defaults to cpu_count())
    dpi_text: DPI for text extraction
    dpi_md: DPI for markdown extraction
    output_dir: directory to save extracted files
    """
    if workers is None:
        workers = max(1, cpu_count() - 0)  # allow tuning

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
                    save_result_to_file(pdf_file, result, format, output_dir, file_or_folder)
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {repr(e)}")
                results.append((pdf_file, f"[ERROR] {repr(e)}"))
        
        return results
        
    else:
        # Single file processing (original behavior)
        return process_single_pdf(file_or_folder, format, workers, dpi_text, dpi_md)


def process_single_pdf(file, format="markdown", workers=None, dpi_text=800, dpi_md=800):
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
    if format == "txt":
        try:
            page_count = get_page_count(file)
        except Exception as e:
            print(f"Error getting page count for {file}: {e}")
            page_count = 1  # fallback
            
        if page_count <= 0:
            print(f"Warning: PDF {file} has no pages")
            return ""
            
        print(f"Using {workers} worker(s) for OCR; DPI={dpi_text}")
        args = [(file, page, dpi_text) for page in range(1, page_count + 1)]

        texts = [None] * page_count
        with Pool(workers) as pool:
            for page_num, text in tqdm(pool.imap_unordered(ocr_page, args),
                                        total=page_count,
                                        desc="OCR Progress",
                                        unit="page"):
                texts[page_num - 1] = text

        return "\n".join(texts)

    # -----------------------------
    # MARKDOWN mode
    # -----------------------------
    elif format == "md":
        # First, try the library's to_markdown with its native signature
        try:
            print("Attempting high-quality pymupdf4llm.to_markdown() (library default).")
            markdown = pymupdf4llm.to_markdown(file)  # removed unsupported kwargs
            print("High-quality extraction succeeded.")
            return markdown
        except Exception as e:
            print(f"High-quality markdown extraction failed: {e}")
            print("Falling back to multiprocess page-by-page OCR -> markdown...")

            try:
                page_count = get_page_count(file)
            except Exception as e:
                print(f"Error getting page count for {file}: {e}")
                page_count = 1  # fallback
                
            if page_count <= 0:
                print(f"Warning: PDF {file} has no pages")
                return ""
                
            print(f"Using {workers} worker(s) for fallback OCR; DPI={dpi_md}")
            args = [(file, page, dpi_md) for page in range(1, page_count + 1)]

            results = [None] * page_count
            with Pool(workers) as pool:
                for page_num, text in tqdm(pool.imap_unordered(ocr_page, args),
                                            total=page_count,
                                            desc="OCR Progress",
                                            unit="page"):
                    results[page_num - 1] = text

            # assemble markdown with simple page separators
            md_blocks = []
            for i, page_text in enumerate(results, start=1):
                md_blocks.append(f"# Page {i}\n\n{page_text}\n\n---\n")

            return "\n".join(md_blocks)

    else:
        print(f"Unknown format '{format}'")
        return "--unknown-format--"


# -----------------------------
# Command-line interface
# -----------------------------
if __name__ == "__main__":
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
        print("Usage: python extract_text.py <input_path> [options]")
        print("Options:")
        print("  --format <text|markdown>     Output format (default: text)")
        print("  --workers <number>           Number of parallel workers (default: CPU count)")
        print("  --dpi-text <number>          DPI for text extraction (default: 800)")
        print("  --dpi-md <number>            DPI for markdown extraction (default: 800)")
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