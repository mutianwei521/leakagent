from flask import Flask, render_template, request, jsonify, send_from_directory
import sys
import os
import json

# Ensure the agent can be imported
sys.path.append(os.getcwd())

try:
    from mm_wds_agent import chain_with_history
except ImportError as e:
    print(f"Error importing agent: {e}")
    sys.exit(1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/topology')
def topology():
    return render_template('topology.html')

import uuid
from werkzeug.utils import secure_filename

# Global session storage (simple dict for demo; use Redis/database in production)
# Format: {session_id: {"active_file": "path/to/file.inp", "title": "New Chat"}}
user_sessions = {}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/sessions/new', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    timestamp = str(uuid.uuid1())
    # Initialize metadata
    meta = {
        "title": "New Conversation", 
        "created_at": timestamp,
        "is_pinned": "0"  # Redis stores strings
    }
    user_sessions[session_id] = meta  # Cache
    
    # Save to Redis if available
    if hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
        try:
            chain_with_history.redis_client.hset(f"session_meta:{session_id}", mapping=meta)
            # Ensure empty list exists for key scanning
            chain_with_history.save_message(session_id, SystemMessage(content="Init")) 
        except Exception as e:
            print(f"Redis meta save error: {e}")

    # Initialize empty history in agent
    chain_with_history.get_session_history(session_id) 
    return jsonify({'success': True, 'conversation_id': session_id})

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    # 1. Delete from memory
    if session_id in user_sessions:
        del user_sessions[session_id]
        
    # 2. Delete from Redis
    if hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
        try:
            chain_with_history.redis_client.delete(f"chat:{session_id}")
            chain_with_history.redis_client.delete(f"session_meta:{session_id}")
        except Exception as e:
            print(f"Error deleting from Redis: {e}")
            
    return jsonify({'success': True})

@app.route('/sessions/<session_id>/pin', methods=['POST'])
def toggle_pin(session_id):
    # Toggle pin status
    # 1. Update memory
    if session_id in user_sessions:
        current = user_sessions[session_id].get("is_pinned", "0")
        new_val = "0" if str(current) == "1" else "1"
        user_sessions[session_id]["is_pinned"] = new_val
    else:
        # If not in memory, check Redis (may have restarted)
        new_val = "1"
        
    # 2. Update Redis
    if hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
        key = f"session_meta:{session_id}"
        # If not in memory, get current pin status first
        if session_id not in user_sessions:
            current = chain_with_history.redis_client.hget(key, "is_pinned") or "0"
            new_val = "0" if str(current) == "1" else "1"
        
        chain_with_history.redis_client.hset(key, "is_pinned", new_val)
        
    return jsonify({'success': True, 'is_pinned': new_val == "1"})

@app.route('/sessions', methods=['GET'])
def list_sessions():
    # Return sidebar session list
    sessions_list = []
    
    # 1. Identify all session IDs (memory + Redis)
    known_ids = set(user_sessions.keys())
    
    if hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
        redis_keys = chain_with_history.redis_client.keys("chat:*")
        redis_ids = [k.replace("chat:", "") for k in redis_keys]
        known_ids.update(redis_ids)
    
    # 2. Build list using metadata
    for sid in known_ids:
        # Try memory first
        meta = user_sessions.get(sid)
        
        # If missing, try Redis
        if not meta and hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
            redis_meta = chain_with_history.redis_client.hgetall(f"session_meta:{sid}")
            if redis_meta:
                 meta = redis_meta
        
        # Default values
        if not meta:
            meta = {"title": f"Conversation {sid[:4]}", "created_at": "", "is_pinned": "0"}
            
        sessions_list.append({
            "id": sid,
            "title": meta.get("title", sid),
            "timestamp": meta.get("created_at", ""),
            "is_pinned": str(meta.get("is_pinned", "0")) == "1"
        })
    
        # Sort: pinned first, then by timestamp descending
        sessions_list.sort(key=lambda x: (not x['is_pinned'], x['timestamp']), reverse=True)
        # Actually, we want Pinned=True items to appear first? No, default sort is False < True.
        # We clean up the logic on the frontend, just return the list here.
        # Wait, timestamp defaults to "", sorting would be weird.
        # For "recent" logic, mainly sort by timestamp descending in Python
        # Anyway, frontend will separate pinned items.
    sessions_list.sort(key=lambda x: x['timestamp'] or "0", reverse=True)

    return jsonify({'success': True, 'conversations': sessions_list})

@app.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    # Retrieve messages via agent method (handles Redis retrieval)
    try:
        messages = chain_with_history.get_session_history(session_id)
        
        # Serialization logic
        serialized = []
        for msg in messages:
            role = "unknown"
            content = msg.content
            if msg.type == 'human': role = 'user'
            elif msg.type == 'ai': role = 'agent'
            elif msg.type == 'system': continue  # Skip system prompts
            elif msg.type == 'tool': continue  # Skip raw tool output
            
            if role != "unknown":
                serialized.append({"role": role, "content": content})
                
        return jsonify({'success': True, 'history': serialized})
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'success': False, 'error': 'Session not found or error loading'})

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat interactions with the context of uploaded files.
    """
    try:
        data = request.json
        user_message = data.get('message')
        session_id = data.get('conversation_id') or 'web_user_default'

        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400

        # Retrieve active file for this session
        user_context = user_sessions.get(session_id, {})
        active_file = user_context.get("active_file")
        
        # Prepare input for agent
        # We append the context of the active file invisibly to the user message
        # Or we could dynamically update the agent's system prompt.
        # Here, we prepend it to the message so the LLM knows the context.
        agent_input = user_message
        if active_file:
             agent_input = f"[Context: Current Active File is '{active_file}']\n{user_message}"

        # Invoke the agent
        response = chain_with_history.invoke(
            {"input": agent_input},
            config={"configurable": {"session_id": session_id}}
        )

        if isinstance(response, dict) and 'output' in response:
            agent_output = response['output']
        else:
            agent_output = str(response)

        # --- Auto-title generation logic ---
        new_title = None
        current_meta = user_sessions.get(session_id)
        
        # If title is default (or missing), generate a title
        if not current_meta or current_meta.get("title") == "New Conversation":
            try:
                # 1. Get history (user + agent reply already in history/Redis)
                history = chain_with_history.get_session_history(session_id)
                
                # 2. Only generate when at least one interaction (user+AI) exists
                if len(history) >= 2:
                    new_title = chain_with_history.summarize_conversation(history)
                    
                    # 3. Update memory
                    if not current_meta:
                        user_sessions[session_id] = {}
                    user_sessions[session_id]["title"] = new_title
                    
                    # 4. Update Redis
                    if hasattr(chain_with_history, 'redis_client') and chain_with_history.redis_client:
                        chain_with_history.redis_client.hset(f"session_meta:{session_id}", "title", new_title)
            except Exception as e:
                print(f"Auto-title failed: {e}")

        return jsonify({
            'success': True,
            'response': agent_output,
            'conversation_id': session_id,
            'downloads': [],
            'new_title': new_title,
            'intent': 'chat',
            'confidence': 1.0
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

from optimization_utils.rag_manager import ingest_inp_file

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads (.inp), save to disk, and update session context.
    Also triggers GraphRAG ingestion (automatic indexing).
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Update session context
            session_id = request.form.get('conversation_id') or 'web_user_default'
            if session_id not in user_sessions:
                user_sessions[session_id] = {}
            user_sessions[session_id]["active_file"] = filepath.replace("\\", "/")

            # --- Trigger GraphRAG automatic ingestion ---
            # Only ingest .inp files into RAG (skip other file types like CSV, images, etc.)
            if filename.lower().endswith('.inp'):
                try:
                    print(f"Auto-ingesting {filename} into GraphRAG...")
                    ingest_inp_file(filepath)
                    rag_msg = "GraphRAG indexing complete."
                except Exception as e:
                    print(f"Auto-ingestion warning: {e}")
                    rag_msg = f"GraphRAG indexing warning: {e}"
            elif filename.lower().endswith('.csv'):
                print(f"CSV file uploaded: {filename} (saved for leak detection inference)")
                rag_msg = "CSV file uploaded successfully. Ready for leak detection inference."
            else:
                print(f"Skipping RAG ingestion for unsupported file: {filename}")
                rag_msg = "File uploaded (unsupported type, RAG ingestion skipped)."

            return jsonify({
                'success': True,
                'filename': filename,
                'content': f"File '{filename}' uploaded successfully. {rag_msg}",
                'full_content': f"File path: {filepath}",
                'path': filepath
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/file_manager')
def file_manager():
    return "File Manager (Not Implemented)", 200

# Serve key visualization outputs locally (heatmaps)
@app.route('/visual_outputs/<path:filename>')
def serve_visual_outputs(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'visual_outputs'), filename)

# Serve sensor placement results locally
@app.route('/sensor_results/<path:filename>')
def serve_sensor_results(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'sensor_results'), filename)

# Serve FCM partition results locally
@app.route('/partition_results/<path:filename>')
def serve_partition_results(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'partition_results'), filename)

# Serve leak detection output files (models, results, templates)
@app.route('/leak_detection_output/<path:filename>')
def serve_leak_detection_output(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'leak_detection_output'), filename)

# API endpoint to list cached RAG network files
RAG_STORAGE_DIR = os.path.join(os.getcwd(), "rag_storage")

@app.route('/api/rag_networks', methods=['GET'])
def list_rag_networks():
    """
    List all cached network files from rag_storage directory.
    Each JSON file contains pre-computed topology and hydraulic data.
    """
    try:
        networks = []

        if os.path.exists(RAG_STORAGE_DIR):
            for filename in os.listdir(RAG_STORAGE_DIR):
                if filename.lower().endswith('.json'):
                    filepath = os.path.join(RAG_STORAGE_DIR, filename)
                    stat = os.stat(filepath)
                    md5 = filename.replace('.json', '')

                    # Read basic metadata from JSON
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        networks.append({
                            'md5': md5,
                            'filename': data.get('filename', md5),
                            'size': stat.st_size,
                            'modified': stat.st_mtime,
                            'stats': data.get('stats', {}),
                            'hydraulics': data.get('hydraulics', {})
                        })
                    except Exception as e:
                        print(f"Error reading RAG file {filename}: {e}")

        # Sort by modified time (newest first)
        networks.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'networks': networks
        })
    except Exception as e:
        print(f"Error listing RAG networks: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'networks': []
        }), 500


@app.route('/api/topology/from_rag/<md5>', methods=['GET'])
def load_rag_topology(md5):
    """
    Load network topology data from a cached RAG storage JSON file.
    Transforms detailed_nodes and detailed_links into the format
    expected by the frontend D3.js visualization.
    """
    try:
        cache_path = os.path.join(RAG_STORAGE_DIR, f"{md5}.json")

        if not os.path.exists(cache_path):
            return jsonify({
                'success': False,
                'error': f'RAG cache file not found: {md5}'
            }), 404

        with open(cache_path, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)

        # Transform detailed_nodes → frontend node format
        nodes = []
        detailed_nodes = rag_data.get('detailed_nodes', {})
        for node_id, node_info in detailed_nodes.items():
            node_data = {
                'id': node_id,
                'type': node_info.get('type', 'Junction').lower(),
                'coordinates': node_info.get('coordinates'),
                'elevation': node_info.get('elevation', 0),
                'base_demand': node_info.get('base_demand', 0),
                'pressure_avg': node_info.get('pressure_avg'),
                'head_avg': node_info.get('head_avg'),
                'demand_avg': node_info.get('demand_avg')
            }
            nodes.append(node_data)

        # Transform detailed_links → frontend link format
        links = []
        detailed_links = rag_data.get('detailed_links', {})
        for link_id, link_info in detailed_links.items():
            link_data = {
                'id': link_id,
                'type': link_info.get('type', 'Pipe').lower(),
                'start_node': link_info.get('start_node'),
                'end_node': link_info.get('end_node'),
                'length': link_info.get('length', 0),
                'diameter': link_info.get('diameter', 0),
                'roughness': link_info.get('roughness', 0),
                'flowrate_avg': link_info.get('flowrate_avg'),
                'velocity_avg': link_info.get('velocity_avg')
            }
            links.append(link_data)

        # Build stats
        stats = rag_data.get('stats', {})
        node_stats = {
            'total': len(nodes),
            'junctions': stats.get('junction_count', sum(1 for n in nodes if n['type'] == 'junction')),
            'reservoirs': stats.get('reservoir_count', sum(1 for n in nodes if n['type'] == 'reservoir')),
            'tanks': stats.get('tank_count', sum(1 for n in nodes if n['type'] == 'tank'))
        }
        link_stats = {
            'total': len(links),
            'pipes': stats.get('pipe_count', sum(1 for l in links if l['type'] == 'pipe')),
            'pumps': stats.get('pump_count', sum(1 for l in links if l['type'] == 'pump')),
            'valves': stats.get('valve_count', sum(1 for l in links if l['type'] == 'valve'))
        }

        return jsonify({
            'success': True,
            'nodes': nodes,
            'links': links,
            'stats': {
                'nodes': node_stats,
                'links': link_stats
            },
            'hydraulics': rag_data.get('hydraulics', {}),
            'filename': rag_data.get('filename', md5)
        })

    except Exception as e:
        print(f"Error loading RAG topology: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




if __name__ == '__main__':
    print("Starting MM-WDS Web Interface on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
