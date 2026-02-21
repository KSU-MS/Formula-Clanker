#!/usr/bin/env python3
"""
PDF to Markdown Converter
Converts all PDF files in a folder and its subfolders into Markdown format.
"""

import os
import sys
import argparse
from pathlib import Path
import pdfplumber
from markdown_it import MarkdownIt
import re

def pdf_to_markdown(pdf_path):
    """
    Convert a PDF file to Markdown format.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Markdown content
    """
    markdown_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Process each page
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    # Add page header
                    markdown_content.append(f"## Page {i+1}")
                    markdown_content.append("")
                    
                    # Process text to improve Markdown formatting
                    lines = text.split('\n')
                    processed_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Handle headers (detect based on font size and formatting)
                        if len(line) > 50 and line.isupper():
                            processed_lines.append(f"# {line}")
                        elif len(line) > 30 and line.isupper():
                            processed_lines.append(f"## {line}")
                        else:
                            processed_lines.append(line)
                    
                    markdown_content.extend(processed_lines)
                    markdown_content.append("")  # Add blank line after page
                    
    except Exception as e:
        return f"Error processing {pdf_path}: {str(e)}"
    
    return "\n".join(markdown_content)

def create_markdown_from_pdf(pdf_path, output_dir):
    """
    Convert a single PDF to Markdown and save to output directory.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get filename without extension
    filename = Path(pdf_path).stem
    markdown_filename = f"{filename}.md"
    markdown_path = os.path.join(output_dir, markdown_filename)
    
    # Convert PDF to Markdown
    markdown_content = pdf_to_markdown(pdf_path)
    
    # Write to file
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Converted: {pdf_path} -> {markdown_path}")

def convert_folder(input_folder, output_folder):
    """
    Convert all PDFs in a folder and its subfolders to Markdown.
    
    Args:
        input_folder (str): Path to the input folder
        output_folder (str): Path to the output folder
    """
    # Ensure input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("No PDF files found in the folder and subfolders.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert.")
    
    # Convert each PDF
    converted_count = 0
    for pdf_file in pdf_files:
        try:
            create_markdown_from_pdf(pdf_file, output_folder)
            converted_count += 1
        except Exception as e:
            print(f"Failed to convert {pdf_file}: {str(e)}")
    
    print(f"Conversion complete. {converted_count} files converted.")

def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(description='Convert PDF files to Markdown')
    parser.add_argument('input_folder', help='Path to the folder containing PDF files')
    parser.add_argument('-o', '--output', default='markdown_output', 
                       help='Output folder for Markdown files (default: markdown_output)')
    
    args = parser.parse_args()
    
    convert_folder(args.input_folder, args.output)

if __name__ == "__main__":
    main()