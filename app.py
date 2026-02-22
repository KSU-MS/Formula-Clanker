# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datetime import datetime

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

def get_available_filters(vector_database):
    """
    Extract available filter options from the vector database.
    
    Returns:
        dict: Dictionary with years, channels, and source types
    """
    years = set()
    channels = set()
    source_types = set()
    
    for metadata in vector_database['message_metadata']:
        # Extract year from timestamp
        if metadata.get('timestamp'):
            try:
                if isinstance(metadata['timestamp'], str):
                    ts = datetime.fromisoformat(metadata['timestamp'].replace(' UTC', '+00:00'))
                else:
                    ts = metadata['timestamp']
                years.add(ts.year)
            except:
                pass
        
        # Extract channel from discord_info
        if 'discord_info' in metadata:
            try:
                # Discord channel names are typically in the format of channel_ID
                # We can use the channel_id as the channel identifier
                channel_id = metadata['discord_info'].get('channel_id', 'Unknown')
                channels.add(str(channel_id))
            except:
                pass
        
        # Determine source type
        if 'file_path' in metadata:
            source_types.add('markdown')
        elif 'discord_info' in metadata:
            source_types.add('discord')
    
    return {
        'years': sorted(list(years), reverse=True),
        'channels': sorted(list(channels)),
        'source_types': sorted(list(source_types))
    }

def filter_results(results, filters):
    """
    Filter search results based on selected filters.
    
    Args:
        results (list): List of search results
        filters (dict): Dictionary with filter parameters (year, channel, source_type)
    
    Returns:
        list: Filtered results
    """
    filtered = results
    
    # Filter by year
    if filters.get('year'):
        year = int(filters['year'])
        filtered = [r for r in filtered if _get_year_from_result(r) == year]
    
    # Filter by channel
    if filters.get('channel'):
        channel = str(filters['channel'])
        filtered = [r for r in filtered if _get_channel_from_result(r) == channel]
    
    # Filter by source type
    if filters.get('source_type'):
        source_type = filters['source_type']
        if source_type == 'discord':
            filtered = [r for r in filtered if 'discord_info' in r.get('metadata', {})]
        elif source_type == 'markdown':
            filtered = [r for r in filtered if 'file_path' in r.get('metadata', {})]
    
    return filtered

def _get_year_from_result(result):
    """Extract year from a search result."""
    try:
        metadata = result.get('metadata', {})
        timestamp = metadata.get('timestamp')
        if isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp.replace(' UTC', '+00:00'))
        else:
            ts = timestamp
        return ts.year
    except:
        return None

def _get_channel_from_result(result):
    """Extract channel from a search result."""
    try:
        metadata = result.get('metadata', {})
        if 'discord_info' in metadata:
            return str(metadata['discord_info'].get('channel_id', 'Unknown'))
    except:
        pass
    return None


print("Loading vector database...")
vector_database = load_vector_database('vectors.pkl')
if vector_database is None:
    print("Warning: Vector database not loaded. Search functionality will be limited.")

@app.route('/reload_vectors', methods=['POST'])
def reload_vectors():
    """Reload the vector database from disk"""
    global vector_database
    
    try:
        print("Reloading vector database...")
        new_database = load_vector_database('vectors.pkl')
        
        if new_database is None:
            return jsonify({'error': 'Failed to load vector database'}), 500
        
        vector_database = new_database
        print(f"Vector database reloaded successfully with {len(vector_database['embeddings'])} embeddings")
        return jsonify({
            'success': True,
            'message': f'Vector database reloaded with {len(vector_database["embeddings"])} embeddings'
        })
    except Exception as e:
        print(f"Error reloading vector database: {str(e)}")
        return jsonify({'error': f'Failed to reload: {str(e)}'}), 500

@app.route('/refresh_status', methods=['GET'])
def refresh_status():
    """Check if the vector database is currently being refreshed"""
    is_refreshing = os.path.exists('vectors.refreshing')
    return jsonify({
        'is_refreshing': is_refreshing,
        'database_loaded': vector_database is not None
    })

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
        top_k_raw = data.get('top_k', 10)
        if top_k_raw in [None, 'null', 'None', '']:
            top_k = 10
        else:
            try:
                top_k = int(top_k_raw)
            except (ValueError, TypeError):
                top_k = 10
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"Searching for query: {query}")
        results = search_vectors(query, vector_database, top_k)
        
        # Apply filters
        if filters:
            print(f"Applying filters: {filters}")
            results = filter_results(results, filters)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            metadata = result['metadata']
            formatted_result = {
                'score': result['score'],
                'content': result['content'],
                'username': metadata.get('username', 'Unknown'),
                'timestamp': metadata.get('timestamp', 'Unknown'),
                'source_type': 'discord' if 'discord_info' in metadata else 'markdown'
            }
            
            # Only include source_file and line_number if they're valid (not None)
            if 'source_file' in metadata and 'line_number' in metadata and metadata.get('line_number') is not None:
                formatted_result['source_file'] = metadata.get('source_file')
                formatted_result['line_number'] = metadata.get('line_number')
            
            # For markdown chunks, include chunk_id for context retrieval
            if 'chunk_id' in metadata:
                formatted_result['chunk_id'] = metadata.get('chunk_id')
            
            # Add Discord link information if available
            if 'discord_info' in metadata:
                discord_info = metadata['discord_info']
                formatted_result['discord_link'] = f"https://discord.com/channels/{discord_info['guild_id']}/{discord_info['channel_id']}/{discord_info['message_id']}"
                formatted_result['channel_id'] = discord_info['channel_id']
            
            # Add markdown file information if available
            if 'file_path' in metadata:
                formatted_result['file_path'] = metadata['file_path']
                formatted_result['chunk_id'] = metadata.get('chunk_id', '')
                formatted_result['chunk_index'] = metadata.get('chunk_index', 0)
            
            formatted_results.append(formatted_result)
        
        print(f"Found {len(formatted_results)} results")
        return jsonify({'results': formatted_results})
        
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/filters', methods=['GET'])
def get_filters():
    """Get available filter options"""
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        filters = get_available_filters(vector_database)
        return jsonify(filters)
    except Exception as e:
        print(f"Error in filters endpoint: {str(e)}")
        return jsonify({'error': f'Filters error: {str(e)}'}), 500

@app.route('/context', methods=['POST'])
def get_context():
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        chunk_id = data.get('chunk_id', '')
        source_file = data.get('source_file', '')
        line_number_raw = data.get('line_number', 0)
        context_lines_raw = data.get('context_lines', 5)
        
        if context_lines_raw in [None, 'null', 'None', '']:
            context_lines = 5
        else:
            try:
                context_lines = int(context_lines_raw)
            except (ValueError, TypeError):
                context_lines = 5
        
        # Handle markdown chunks
        if chunk_id:
            # Find the chunk in the vector database
            for idx, metadata in enumerate(vector_database['message_metadata']):
                if metadata.get('chunk_id') == chunk_id:
                    # Return the chunk content directly
                    return jsonify({
                        'chunk_id': chunk_id,
                        'content': metadata['original_message'].get('content', ''),
                        'file_path': metadata.get('file_path', ''),
                        'context_type': 'markdown_chunk'
                    })
            return jsonify({'error': f'Chunk not found: {chunk_id}'}), 404
        
        # Handle Discord messages (original behavior)
        if line_number_raw in [None, 'null', 'None', '']:
            line_number = 0
        else:
            try:
                line_number = int(line_number_raw)
            except (ValueError, TypeError):
                line_number = 0
        
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

@app.route('/file_metadata', methods=['POST'])
def get_file_metadata():
    """Get metadata for a specific file"""
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        # Find all entries for this file
        file_entries = []
        for metadata in vector_database['message_metadata']:
            if 'file_path' in metadata and metadata['file_path'] == file_path:
                file_entries.append({
                    'chunk_id': metadata.get('chunk_id', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'content_preview': metadata['original_message'].get('content', '')[:100] + '...'
                })
        
        return jsonify({
            'file_path': file_path,
            'chunks': file_entries
        })
        
    except Exception as e:
        print(f"Error in file metadata endpoint: {str(e)}")
        return jsonify({'error': f'File metadata error: {str(e)}'}), 500

@app.route('/search_markdown', methods=['POST'])
def search_markdown():
    """Search specifically for markdown content"""
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k_raw = data.get('top_k', 10)
        if top_k_raw in [None, 'null', 'None', '']:
            top_k = 10
        else:
            try:
                top_k = int(top_k_raw)
            except (ValueError, TypeError):
                top_k = 10
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Filter results to only include markdown entries
        results = search_vectors(query, vector_database, top_k)
        markdown_results = []
        
        for result in results:
            metadata = result['metadata']
            # Only include results that have markdown file information
            if 'file_path' in metadata:
                formatted_result = {
                    'score': result['score'],
                    'content': result['content'],
                    'file_path': metadata['file_path'],
                    'chunk_id': metadata.get('chunk_id', ''),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'source_file': metadata.get('source_file', 'Unknown')
                }
                markdown_results.append(formatted_result)
        
        # Apply filters
        if filters:
            markdown_results = filter_results(markdown_results, filters)
        
        return jsonify({'results': markdown_results})
        
    except Exception as e:
        print(f"Error in markdown search endpoint: {str(e)}")
        return jsonify({'error': f'Markdown search error: {str(e)}'}), 500

@app.route('/search_discord', methods=['POST'])
def search_discord():
    """Search specifically for Discord messages"""
    if vector_database is None:
        return jsonify({'error': 'Vector database not loaded'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k_raw = data.get('top_k', 10)
        if top_k_raw in [None, 'null', 'None', '']:
            top_k = 10
        else:
            try:
                top_k = int(top_k_raw)
            except (ValueError, TypeError):
                top_k = 10
        filters = data.get('filters', {})
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Filter results to only include Discord entries
        results = search_vectors(query, vector_database, top_k)
        discord_results = []
        
        for result in results:
            metadata = result['metadata']
            # Only include results that have Discord information
            if 'discord_info' in metadata:
                formatted_result = {
                    'score': result['score'],
                    'content': result['content'],
                    'username': metadata.get('username', 'Unknown'),
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'source_file': metadata.get('source_file', 'Unknown'),
                    'channel_id': metadata['discord_info'].get('channel_id', 'Unknown'),
                    'discord_link': f"https://discord.com/channels/{metadata['discord_info']['guild_id']}/{metadata['discord_info']['channel_id']}/{metadata['discord_info']['message_id']}"
                }
                discord_results.append(formatted_result)
        
        # Apply filters
        if filters:
            discord_results = filter_results(discord_results, filters)
        
        return jsonify({'results': discord_results})
        
    except Exception as e:
        print(f"Error in discord search endpoint: {str(e)}")
        return jsonify({'error': f'Discord search error: {str(e)}'}), 500

if __name__ == '__main__':
    # Change to listen on all interfaces to make it accessible from other machines
    print("Starting Flask app on all interfaces...")
    app.run(debug=True, host='0.0.0.0', port=5000)