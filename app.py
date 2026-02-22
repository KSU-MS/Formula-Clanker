# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

app = Flask(__name__)

# Load vector database
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
    
    try:
        with open(file_path, 'rb') as f:
            vector_database = pickle.load(f)
        
        print(f"Loaded vector database with {len(vector_database['embeddings'])} embeddings")
        return vector_database
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
        return None

# Search vectors function
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
    try:
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
                    'score': float(similarities[idx]),  # Convert to Python float
                    'metadata': vector_database['message_metadata'][idx],
                    'content': vector_database['message_metadata'][idx]['original_message']['content']
                })
        
        return results
    except Exception as e:
        print(f"Error in search_vectors: {str(e)}")
        return []

# Load database once at startup
print("Loading vector database...")
vector_database = load_vector_database('vectors.pkl')
if vector_database is None:
    print("Warning: Vector database not loaded. Search functionality will be limited.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    print("Search endpoint called")
    if vector_database is None:
        print("Vector database not loaded")
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"Search data received: {data}")
        
        query = data.get('query', '')
        top_k = int(data.get('top_k', 10))
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"Searching for query: {query}")
        results = search_vectors(query, vector_database, top_k)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_result = {
                'score': result['score'],
                'content': result['content'],
                'username': result['metadata']['username'],
                'timestamp': result['metadata']['timestamp'],
                'source_file': result['metadata']['source_file'],
                'line_number': result['metadata']['line_number']
            }
            # Add Discord link information if available
            if 'discord_info' in result['metadata']:
                discord_info = result['metadata']['discord_info']
                formatted_result['discord_link'] = f"https://discord.com/channels/{discord_info['guild_id']}/{discord_info['channel_id']}/{discord_info['message_id']}"
            formatted_results.append(formatted_result)
        
        print(f"Found {len(formatted_results)} results")
        return jsonify({'results': formatted_results})
        
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/context', methods=['POST'])
def get_context():
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        source_file = data.get('source_file', '')
        line_number = int(data.get('line_number', 0))
        context_lines = int(data.get('context_lines', 5))
        
        if not source_file or line_number <= 0:
            return jsonify({'error': 'Invalid source file or line number'}), 400
        
        # Construct full file path
        full_path = 'discord_exports/KSU Motorsports/' + source_file
        
        if not os.path.exists(full_path):
            return jsonify({'error': f'Source file not found: {full_path}'}), 404
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Calculate start and end line numbers
            start_line = max(0, line_number - context_lines - 1)
            end_line = min(len(lines), line_number + context_lines)
            
            # Get context lines
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                marker = ">>> " if line_num == line_number else "   "
                context_lines_list.append({
                    'line_number': line_num,
                    'content': lines[i].rstrip(),
                    'is_target': line_num == line_number
                })
            
            return jsonify({
                'source_file': source_file,
                'line_number': line_number,
                'context': context_lines_list
            })
            
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in context endpoint: {str(e)}")
        return jsonify({'error': f'Context error: {str(e)}'}), 500

if __name__ == '__main__':
    # Change to listen on all interfaces to make it accessible from other machines
    print("Starting Flask app on all interfaces...")
    app.run(debug=True, host='0.0.0.0', port=5000)