# chunk.py
"""
Unified Chunking Script
Processes Discord text exports and Markdown files for vector embedding
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path

# ============================================================================
# DISCORD CHUNKING FUNCTIONS
# ============================================================================

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string into datetime object.
    
    Args:
        timestamp_str (str): Timestamp in format [YYYY-MM-DD HH:MM:SS UTC]
    
    Returns:
        datetime: Parsed datetime object
    """
    # Remove brackets and parse
    timestamp_str = timestamp_str.strip('[]')
    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S UTC')

def chunk_lines_to_json(input_file_path, output_file_path=None):
    """
    Read lines from a Discord text file and convert messages into JSON objects
    without grouping by time.
    
    Args:
        input_file_path (str): Path to the input text file
        output_file_path (str): Path to the output JSON file (optional)
    """
    # Get the filename without path
    filename = os.path.basename(input_file_path)
    
    # Load metadata if it exists
    metadata_path = input_file_path.replace('.txt', '_metadata.json')
    metadata_map = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as mf:
                metadata_map = json.load(mf)
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
    
    # Read all lines from the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Parse messages with timestamps
    messages = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # Match timestamp pattern [YYYY-MM-DD HH:MM:SS UTC] username: message content
        # Updated to handle usernames with special characters like #
        timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)\] (.*?): (.*)'
        match = re.match(timestamp_pattern, line)
        
        if match:
            timestamp_str, username, content = match.groups()
            try:
                timestamp = parse_timestamp(timestamp_str)
                message_obj = {
                    "line_number": line_num,
                    "timestamp": timestamp,
                    "username": username,
                    "content": content,
                    "source_file": filename
                }
                # Add Discord IDs if available in metadata
                if str(line_num) in metadata_map:
                    message_obj["discord_info"] = metadata_map[str(line_num)]
                messages.append(message_obj)
            except ValueError as e:
                print(f"Warning: Could not parse timestamp in line {line_num}: {e}")
        else:
            # If line doesn't match timestamp pattern, treat as a standalone message
            message_obj = {
                "line_number": line_num,
                "timestamp": None,
                "username": None,
                "content": line,
                "source_file": filename
            }
            # Add Discord IDs if available in metadata
            if str(line_num) in metadata_map:
                message_obj["discord_info"] = metadata_map[str(line_num)]
            messages.append(message_obj)
    
    # Determine output file path if not provided
    if output_file_path is None:
        # Create output filename based on input filename
        name_without_ext = os.path.splitext(filename)[0]
        output_file_path = f"{name_without_ext}_messages.json"
    
    # Ensure discord_jsons directory exists
    output_dir = "discord_jsons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the output file
    full_output_path = os.path.join(output_dir, output_file_path)
    
    # Write to JSON file
    with open(full_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(messages, json_file, indent=2, ensure_ascii=False, default=str)
    
    print(f"Successfully processed {len(messages)} messages from '{filename}'")
    print(f"Output saved to '{full_output_path}'")

def process_discord_directory(directory_path):
    """
    Process all .txt files in the given directory (Discord exports).
    
    Args:
        directory_path (str): Path to the directory containing text files
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))]
    
    if not txt_files:
        print(f"No .txt files found in '{directory_path}'")
        return
    
    print(f"Found {len(txt_files)} Discord text files to process:")
    for txt_file in txt_files:
        print(f"  - {txt_file}")
    
    # Process each text file
    for txt_file in txt_files:
        input_file_path = os.path.join(directory_path, txt_file)
        try:
            chunk_lines_to_json(input_file_path)
        except Exception as e:
            print(f"Error processing '{txt_file}': {e}")

# ============================================================================
# MARKDOWN CHUNKING FUNCTIONS
# ============================================================================

def get_file_hash(filepath):
    """Calculate MD5 hash of a file's content."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def chunk_markdown_file(file_path, chunk_size=500, overlap=50):
    """
    Split a markdown file into chunks with configurable overlap.
    
    Args:
        file_path (str): Path to the markdown file
        chunk_size (int): Target number of characters per chunk
        overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of chunk dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    chunks = []
    chunk_index = 0
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # Add paragraph to current chunk
        test_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        # If adding this paragraph exceeds chunk size, save current chunk and start new one
        if len(test_chunk) > chunk_size and current_chunk:
            chunks.append({
                'chunk_index': chunk_index,
                'content': current_chunk.strip(),
                'char_count': len(current_chunk)
            })
            chunk_index += 1
            
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk = test_chunk
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'chunk_index': chunk_index,
            'content': current_chunk.strip(),
            'char_count': len(current_chunk)
        })
    
    return chunks

def process_markdown_files(input_dir, output_dir='chunked_markdown', cache_file='chunk_cache.json'):
    """
    Process all markdown files in a directory and save chunks as JSON.
    
    Args:
        input_dir (str): Directory containing markdown files
        output_dir (str): Directory to save chunked JSON files
        cache_file (str): Cache file for tracking processed files
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load or create cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    # Find all markdown files
    markdown_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    
    if not markdown_files:
        print(f"No markdown files found in '{input_dir}'")
        return
    
    print(f"Found {len(markdown_files)} markdown files to process")
    
    processed_count = 0
    skipped_count = 0
    
    for md_file in markdown_files:
        # Calculate file hash
        current_hash = get_file_hash(md_file)
        if current_hash is None:
            continue
        
        # Check if file has been processed and hasn't changed
        file_key = os.path.relpath(md_file, input_dir)
        if file_key in cache and cache[file_key] == current_hash:
            print(f"Skipping '{file_key}' (no changes)")
            skipped_count += 1
            continue
        
        # Process the file
        print(f"Processing '{file_key}'")
        chunks = chunk_markdown_file(md_file)
        
        if not chunks:
            print(f"  No content to chunk in '{file_key}'")
            continue
        
        # Get relative path for file identification
        rel_path = os.path.relpath(md_file, input_dir)
        file_stem = Path(md_file).stem
        
        # Save each chunk as a separate JSON file
        for chunk in chunks:
            chunk_id = f"{file_stem}_chunk_{chunk['chunk_index']}"
            output_filename = f"{chunk_id}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            chunk_data = {
                'chunk_id': chunk_id,
                'chunk_index': chunk['chunk_index'],
                'content': chunk['content'],
                'metadata': {
                    'file_name': Path(md_file).name,
                    'file_path': rel_path,
                    'char_count': chunk['char_count'],
                    'source_type': 'markdown'
                },
                'original_message': {
                    'content': chunk['content'],
                }
            }
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Error saving chunk {chunk_id}: {e}")
                continue
        
        # Update cache
        cache[file_key] = current_hash
        processed_count += 1
        print(f"  Created {len(chunks)} chunks")
    
    # Save updated cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    print(f"\nMarkdown processing complete:")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Output directory: {output_dir}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Unified Chunking Script - Processes Discord exports and Markdown files")
        print("\nUsage:")
        print("  Discord:  python chunk.py discord <discord_directory>")
        print("  Markdown: python chunk.py markdown <markdown_directory> [output_directory]")
        print("  Both:     python chunk.py both <discord_dir> <markdown_dir> [markdown_output]")
        print("\nExamples:")
        print("  python chunk.py discord 'discord_exports/KSU Motorsports'")
        print("  python chunk.py markdown markdown_output chunked_markdown")
        print("  python chunk.py both 'discord_exports/KSU Motorsports' markdown_output chunked_markdown")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == 'discord':
        if len(sys.argv) < 3:
            print("Error: Discord mode requires a directory path")
            print("Usage: python chunk.py discord <directory>")
            sys.exit(1)
        discord_dir = sys.argv[2]
        if not os.path.exists(discord_dir):
            print(f"Error: Discord directory '{discord_dir}' does not exist")
            sys.exit(1)
        process_discord_directory(discord_dir)
    
    elif mode == 'markdown':
        if len(sys.argv) < 3:
            print("Error: Markdown mode requires a directory path")
            print("Usage: python chunk.py markdown <input_dir> [output_dir]")
            sys.exit(1)
        md_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'chunked_markdown'
        if not os.path.exists(md_dir):
            print(f"Error: Markdown directory '{md_dir}' does not exist")
            sys.exit(1)
        process_markdown_files(md_dir, output_dir)
    
    elif mode == 'both':
        if len(sys.argv) < 4:
            print("Error: Both mode requires two directory paths")
            print("Usage: python chunk.py both <discord_dir> <markdown_dir> [markdown_output]")
            sys.exit(1)
        discord_dir = sys.argv[2]
        md_dir = sys.argv[3]
        markdown_output = sys.argv[4] if len(sys.argv) > 4 else 'chunked_markdown'
        
        if not os.path.exists(discord_dir):
            print(f"Error: Discord directory '{discord_dir}' does not exist")
            sys.exit(1)
        if not os.path.exists(md_dir):
            print(f"Error: Markdown directory '{md_dir}' does not exist")
            sys.exit(1)
        
        print("=" * 60)
        print("Processing Discord exports and Markdown files")
        print("=" * 60)
        process_discord_directory(discord_dir)
        print("\n" + "=" * 60)
        process_markdown_files(md_dir, markdown_output)
        print("=" * 60)
    
    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Valid modes: discord, markdown, both")
        sys.exit(1)

if __name__ == "__main__":
    main()
