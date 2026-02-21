# create_vectors.py
import json
import os
import sys
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
import time

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

def load_chunked_files(directory_path, cache_file='file_cache.json'):
    """
    Load all chunked JSON files from the directory.
    
    Args:
        directory_path (str): Path to directory containing chunked JSON files
        cache_file (str): Path to cache file storing file hashes
    
    Returns:
        list: List of all messages from all chunked files
    """
    all_messages = []
    
    # Get all .json files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('_messages.json') and os.path.isfile(os.path.join(directory_path, f))]
    
    if not json_files:
        print(f"No chunked JSON files found in '{directory_path}'")
        return all_messages
    
    print(f"Found {len(json_files)} chunked files to process:")
    
    # Load or create cache
    file_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                file_cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache file: {e}")
    
    # Check which files need to be processed
    files_to_process = []
    files_to_skip = []
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        current_hash = get_file_hash(file_path)
        
        if current_hash is None:
            files_to_process.append(json_file)
            continue
            
        if json_file in file_cache and file_cache[json_file] == current_hash:
            files_to_skip.append(json_file)
        else:
            files_to_process.append(json_file)
    
    if files_to_skip:
        print(f"Skipping {len(files_to_skip)} files (no changes detected):")
        for f in files_to_skip:
            print(f"  - {f}")
    
    if files_to_process:
        print(f"Processing {len(files_to_process)} files:")
        for json_file in files_to_process:
            print(f"  - {json_file}")
            try:
                with open(os.path.join(directory_path, json_file), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Each file contains a list of messages
                    for item in data:
                        # If it's a grouped message (has 'messages' key), flatten it
                        if 'messages' in item and isinstance(item['messages'], list):
                            for msg in item['messages']:
                                all_messages.append(msg)
                        else:
                            all_messages.append(item)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    # Update cache with new hashes
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        current_hash = get_file_hash(file_path)
        if current_hash is not None:
            file_cache[json_file] = current_hash
    
    # Save updated cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(file_cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache file: {e}")
    
    return all_messages

def create_vectors(messages, model_name='all-mpnet-base-v2', output_path='vectors.pkl'):
    """
    Create vector embeddings for messages using sentence transformers.
    
    Args:
        messages (list): List of message objects
        model_name (str): Name of the sentence transformer model to use
        output_path (str): Path to save the vector database
    
    Returns:
        tuple: (embeddings, message_metadata)
    """
    print(f"Loading sentence transformer model: {model_name}")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Extract message content for vectorization
    message_contents = []
    message_metadata = []
    
    for msg in messages:
        content = msg.get('content', '')
        if content:
            message_contents.append(content)
            metadata_entry = {
                'line_number': msg.get('line_number'),
                'timestamp': msg.get('timestamp'),
                'username': msg.get('username'),
                'source_file': msg.get('source_file'),
                'original_message': msg
            }
            # Include discord_info if available
            if 'discord_info' in msg:
                metadata_entry['discord_info'] = msg['discord_info']
            message_metadata.append(metadata_entry)
    
    print(f"Vectorizing {len(message_contents)} messages...")
    
    # Create embeddings with progress bar
    if message_contents:
        embeddings = []
        # Use tqdm for progress bar
        for content in tqdm(message_contents, desc="Creating embeddings"):
            embedding = model.encode(content)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"Created embeddings of shape: {embeddings.shape}")
    else:
        print("No content to vectorize")
        embeddings = np.array([])
    
    # Save vector database
    vector_database = {
        'model_name': model_name,
        'embeddings': embeddings,
        'message_metadata': message_metadata,
        'model': model
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(vector_database, f)
    
    print(f"Vectors saved to '{output_path}'")
    return embeddings, message_metadata

def main():
    # Default directory path for chunked files
    directory_path = "discord_jsons"
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return
    
    # Load chunked files
    print("Loading chunked files...")
    messages = load_chunked_files(directory_path)
    
    if not messages:
        print("No messages found to vectorize.")
        return
    
    print(f"Total messages loaded: {len(messages)}")
    
    # Create vectors using a more powerful model
    print("Creating vectors with enhanced model...")
    embeddings, message_metadata = create_vectors(
        messages, 
        model_name='all-mpnet-base-v2',  # More powerful model
        output_path='vectors.pkl'
    )

if __name__ == "__main__":
    main()