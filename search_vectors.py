# search_vectors.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_vector_database(file_path='vectors.pkl'):
    """
    Load the vector database from pickle file.
    
    Args:
        file_path (str): Path to the pickle file containing vectors
    
    Returns:
        dict: Vector database with embeddings, metadata, and model
    """
    if not os.path.exists(file_path):
        print(f"Error: Vector database file '{file_path}' not found.")
        return None
    
    with open(file_path, 'rb') as f:
        vector_database = pickle.load(f)
    
    print(f"Loaded vector database with {len(vector_database['embeddings'])} embeddings")
    return vector_database

def search_vectors(query, vector_database, top_k=10):
    """
    Search for similar vectors to the query.
    
    Args:
        query (str): Search query
        vector_database (dict): Loaded vector database
        top_k (int): Number of top results to return
    
    Returns:
        list: Top k similar messages with scores
    """
    # Load the model used for vectorization
    model = vector_database['model']
    
    # Create embedding for the query
    query_embedding = model.encode([query])
    
    # Get all embeddings
    embeddings = vector_database['embeddings']
    
    if len(embeddings) == 0:
        print("No embeddings found in database")
        return []
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Get top k results with scores
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include results with some similarity
            results.append({
                'score': similarities[idx],
                'metadata': vector_database['message_metadata'][idx],
                'content': vector_database['message_metadata'][idx]['original_message']['content']
            })
    
    return results

def display_results(results):
    """
    Display search results in a formatted way.
    
    Args:
        results (list): List of search results
    """
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Content: {result['content']}")
        print(f"   Username: {result['metadata']['username']}")
        print(f"   Timestamp: {result['metadata']['timestamp']}")
        print(f"   Source File: {result['metadata']['source_file']}")
        print(f"   Line Number: {result['metadata']['line_number']}")
    

def get_surrounding_messages(result, context_lines=5):
    """
    Display surrounding lines from the source file.
    
    Args:
        result (dict): Search result containing metadata
        context_lines (int): Number of lines to show before and after
    """
    source_file = 'discord_exports/KSU Motorsports/' + result['metadata']['source_file']
    line_number = result['metadata']['line_number']
    
    if not os.path.exists(source_file):
        print(f"Error: Source file '{source_file}' not found.")
        return
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Calculate start and end line numbers
        start_line = max(0, line_number - context_lines - 1)
        end_line = min(len(lines), line_number + context_lines)
        
        print(f"\nSurrounding context for line {line_number} in {source_file}:")
        print("-" * 80)
        
        # Print context lines
        for i in range(start_line, end_line):
            line_num = i + 1
            marker = ">>> " if line_num == line_number else "    "
            print(f"{line_num:4d}: {marker}{lines[i].rstrip()}")
            
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    # Load vector database
    vector_database = load_vector_database('vectors.pkl')
    
    if not vector_database:
        return
    
    print("Vector Search Program")
    print("=" * 50)
    
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            print("Please enter a search query.")
            continue
        
        print(f"\nSearching for: '{query}'")
        results = search_vectors(query, vector_database, top_k=10)
        display_results(results)
        
        # Ask if user wants to see surrounding context
        if results:
            show_context = input("\nShow surrounding context for any result? (y/n): ").strip().lower()
            if show_context in ['y', 'yes']:
                try:
                    result_num = int(input("Enter result number: ")) - 1
                    if 0 <= result_num < len(results):
                        get_surrounding_messages(results[result_num])
                    else:
                        print("Invalid result number.")
                except ValueError:
                    print("Please enter a valid number.")

if __name__ == "__main__":
    main()