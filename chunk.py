# chunk.py
import json
import os
import re
import sys
from datetime import datetime

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
    Read lines from a text file and convert messages into JSON objects
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
        # Updated to be more flexible with potential whitespace and special characters in usernames
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

def process_directory(directory_path):
    """
    Process all .txt files in the given directory.
    
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
    
    print(f"Found {len(txt_files)} text files to process:")
    for txt_file in txt_files:
        print(f"  - {txt_file}")
    
    # Process each text file
    for txt_file in txt_files:
        input_file_path = os.path.join(directory_path, txt_file)
        try:
            chunk_lines_to_json(input_file_path)
        except Exception as e:
            print(f"Error processing '{txt_file}': {e}")

def main():
    # Default directory path
    directory_path = "discord_exports/KSU Motorsports"
    
    # Check if arguments were provided
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    
    # Process all text files in the directory
    process_directory(directory_path)

if __name__ == "__main__":
    main()