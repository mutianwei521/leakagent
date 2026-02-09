#!/usr/bin/env python3
"""
Web Chat Application with OpenAI API Integration
Supports file upload and text conversation web chat interface
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, send_file, abort
from werkzeug.utils import secure_filename
from agents import HydroSim, PartitionSim, SensorPlacement, LeakDetectionAgent, LLMTaskAnalyzer
from agents.agent_executor import AgentExecutor
import openai
from pathlib import Path
import mimetypes
import re
import threading
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Please change to a secure key

# Configuration
UPLOAD_FOLDER = 'uploads'
DOWNLOADS_FOLDER = 'downloads'  # Download folder
CONVERSATIONS_FOLDER = 'conversations'  # Conversation storage directory
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv', 'inp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# File management configuration
MAX_FILES_COUNT = 100  # Maximum file count
MAX_FOLDER_SIZE = 500 * 1024 * 1024  # Maximum folder size 500MB
FILE_RETENTION_DAYS = 7  # File retention days

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

# Lazy initialize agents to avoid duplicate IntentClassifier initialization
hydro_sim_agent = None
partition_sim_agent = None
sensor_placement_agent = None
leak_detection_agent = None
llm_task_analyzer = None
agent_executor = None

def extract_training_parameters(user_message: str, default_scenarios: int, default_epochs: int) -> tuple:
    """Intelligently extract training parameters"""
    import re

    num_scenarios = default_scenarios
    epochs = default_epochs

    # Pattern for extracting iteration count/training epochs
    epoch_patterns = [
        r'iteration count\s*[=:]\s*(\d+)',
        r'iterations?\s*[=:]\s*(\d+)',
        r'training rounds?\s*[=:]\s*(\d+)',
        r'(\d+)\s*rounds?',
        r'epochs?\s*[=:]\s*(\d+)',
        r'(\d+)\s*epochs?',
        r'epoch\s*[=:]\s*(\d+)'
    ]

    # Pattern for extracting sample count/data groups
    scenario_patterns = [
        r'generate data\s*[=:]\s*(\d+)',
        r'(\d+)\s*data groups?',
        r'(\d+)\s*samples?',
        r'(\d+)\s*scenarios?',
        r'data size\s*[=:]\s*(\d+)',
        r'sample count\s*[=:]\s*(\d+)',
        r'scenario count\s*[=:]\s*(\d+)',
        r'data\s*(\d+)\s*groups?',
        r'samples?\s*(\d+)',
        r'scenarios?\s*(\d+)',
        r'(\d+)\s*data',
        r'total samples\s*[=:]\s*(\d+)',
        r'total data\s*[=:]\s*(\d+)'
    ]

    # Try to match iteration count
    for pattern in epoch_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            epochs = min(int(match.group(1)), 500)
            print(f"Identified iteration count: {epochs} (Matched pattern: {pattern})")
            break

    # Try to match sample count
    for pattern in scenario_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            num_scenarios = min(int(match.group(1)), 2000)  # Increase max limit to 2000
            print(f"Identified sample count: {num_scenarios} (Matched pattern: {pattern})")
            break

    # If no specific pattern matched, use simple number extraction as fallback
    if num_scenarios == default_scenarios and epochs == default_epochs:
        numbers = re.findall(r'\d+', user_message)
        if numbers:
            # If only one number, determine based on context
            if len(numbers) == 1:
                num = int(numbers[0])
                if any(keyword in user_message.lower() for keyword in ['iteration', 'round', 'epoch']):
                    epochs = min(num, 500)
                    print(f"Identified as iteration count based on context: {epochs}")
                elif any(keyword in user_message.lower() for keyword in ['data', 'sample', 'scenario', 'group']):
                    num_scenarios = min(num, 2000)  # Increase max limit to 2000
                    print(f"Identified as sample count based on context: {num_scenarios}")
                else:
                    # Default first number as sample count
                    num_scenarios = min(num, 2000)  # Increase max limit to 2000
                    print(f"Default identified as sample count: {num_scenarios}")
            elif len(numbers) >= 2:
                # Multiple numbers: first as sample count, second as iteration count per original logic
                num_scenarios = min(int(numbers[0]), 2000)  # Increase max limit to 2000
                epochs = min(int(numbers[1]), 500)
                print(f"Multiple number mode: Sample count={num_scenarios}, Iteration count={epochs}")

    return num_scenarios, epochs

def init_agents():
    """Lazy initialization of agents"""
    global hydro_sim_agent, partition_sim_agent, sensor_placement_agent, leak_detection_agent
    global llm_task_analyzer, agent_executor

    if hydro_sim_agent is None:
        print("Initializing agents...")
        hydro_sim_agent = HydroSim()
        partition_sim_agent = PartitionSim()
        sensor_placement_agent = SensorPlacement()
        leak_detection_agent = LeakDetectionAgent()

        # Initialize LLM task analyzer and agent executor
        llm_task_analyzer = LLMTaskAnalyzer()
        agent_executor = AgentExecutor(
            hydro_sim_agent,
            partition_sim_agent,
            sensor_placement_agent,
            leak_detection_agent
        )
        print("Agent initialization complete")

# OpenAI Configuration
openai.api_base = "https://api.chatanywhere.tech"
openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"

# Agent standard phrase mapping
AGENT_STANDARD_PHRASES = {
    "Network analysis": "Analyze network structure and basic information",
    "Network partition": "Divide network into specified number of regions",
    "Outlier detection": "Detect and remove outliers in network partition",
    "Sensor placement": "Optimize pressure monitoring sensor placement in network",
    "Resilience analysis": "Analyze resilience and fault detection capability of sensor placement",
    "Leak model training": "Train machine learning based leak detection model",
    "Leak detection": "Detect network leaks using trained model",
    "Hydraulic simulation": "Perform network hydraulic calculation and simulation analysis",
    "Topology analysis": "Analyze network topology structure and connectivity"
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file_content(filepath):
    """Read file content"""
    try:
        # Try reading in text mode
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try other encodings
            with open(filepath, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return "Cannot read file content (may be binary file)"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def generate_conversation_title(first_message):
    """Generate conversation title based on first message"""
    if not first_message:
        return "New Conversation"

    # Clean message content
    clean_message = re.sub(r'\s+', ' ', first_message.strip())

    # If message is too long, take first 30 characters
    if len(clean_message) > 30:
        return clean_message[:30] + "..."

    return clean_message if clean_message else "New Conversation"

def ensure_conversations_folder():
    """Ensure conversation storage directory exists"""
    os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder, handle numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_jsonify(data, status_code=200):
    """Safe jsonify function, handle numpy data types"""
    try:
        # Use custom encoder to serialize data
        json_str = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
        # Create response
        response = app.response_class(
            json_str,
            mimetype='application/json'
        )
        response.status_code = status_code
        return response
    except Exception as e:
        # If custom serialization fails, fallback to standard jsonify
        print(f"JSON serialization warning: {e}")
        return jsonify({'error': 'JSON serialization failed'}), 500

def save_conversation_to_file(conversation_id, conversation_data):
    """Save single conversation to file"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"Failed to save conversation file: {e}")

def load_conversation_from_file(conversation_id):
    """Load single conversation from file"""
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load conversation file: {e}")
    return None

def save_conversations_index(conversations_dict):
    """Save conversation index"""
    ensure_conversations_folder()
    index_data = {
        'conversations': {
            conv_id: {
                'id': conv_data['id'],
                'title': conv_data['title'],
                'created_at': conv_data['created_at'],
                'updated_at': conv_data['updated_at'],
                'message_count': len(conv_data['messages'])
            }
            for conv_id, conv_data in conversations_dict.items()
        },
        'last_updated': datetime.now().isoformat()
    }

    filepath = os.path.join(CONVERSATIONS_FOLDER, 'conversations_index.json')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"Failed to save conversation index: {e}")

def load_all_conversations():
    """Load all conversations from files"""
    ensure_conversations_folder()
    index_filepath = os.path.join(CONVERSATIONS_FOLDER, 'conversations_index.json')

    if not os.path.exists(index_filepath):
        return {}

    try:
        with open(index_filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        conversations = {}
        for conv_id in index_data['conversations']:
            conv_data = load_conversation_from_file(conv_id)
            if conv_data:
                conversations[conv_id] = conv_data

        return conversations
    except Exception as e:
        print(f"Failed to load conversation history: {e}")
        return {}

def delete_conversation_file(conversation_id):
    """Delete conversation file"""
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Failed to delete conversation file: {e}")

def save_pinned_conversations(pinned_list):
    """Save pinned conversation list"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, 'pinned_conversations.json')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'pinned_conversations': pinned_list,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"Failed to save pinned conversation list: {e}")

def load_pinned_conversations():
    """Load pinned conversation list"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, 'pinned_conversations.json')

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('pinned_conversations', [])
    except Exception as e:
        print(f"Failed to load pinned conversation list: {e}")
        return []

def get_inp_file_from_conversation_history(conversation):
    """Get the most recent .inp file path from conversation history"""
    for msg in reversed(conversation['messages']):
        if msg.get('file_type') == 'inp' and msg.get('file_path'):
            # Check if file still exists
            if os.path.exists(msg['file_path']):
                return msg['file_path']
    return None

def has_inp_file_in_conversation_history(conversation):
    """Check if conversation history contains .inp file"""
    return get_inp_file_from_conversation_history(conversation) is not None

def get_partition_csv_from_conversation_history(conversation):
    """Get the most recent partition CSV file path from conversation history"""
    for msg in reversed(conversation['messages']):
        # Check if this is a partition-related message with CSV file generated
        if (msg.get('intent') == 'partition_analysis' and
            msg.get('csv_info') and
            msg['csv_info'].get('success')):
            csv_path = msg['csv_info']['filepath']
            # Check if file still exists
            if os.path.exists(csv_path):
                return csv_path
    return None

def has_partition_csv_in_conversation_history(conversation):
    """Check if conversation history contains partition CSV file"""
    return get_partition_csv_from_conversation_history(conversation) is not None

def cleanup_old_files():
    """Clean up expired files"""
    try:
        if not os.path.exists(DOWNLOADS_FOLDER):
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=FILE_RETENTION_DAYS)

        files_info = []
        total_size = 0

        # Collect file info
        for filename in os.listdir(DOWNLOADS_FOLDER):
            file_path = os.path.join(DOWNLOADS_FOLDER, filename)
            if os.path.isfile(file_path):
                file_stat = os.stat(file_path)
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                file_size = file_stat.st_size

                files_info.append({
                    'path': file_path,
                    'filename': filename,
                    'mtime': file_time,
                    'size': file_size
                })
                total_size += file_size

        # Sort by modification time (oldest first)
        files_info.sort(key=lambda x: x['mtime'])

        deleted_count = 0

        # Delete expired files
        for file_info in files_info:
            if file_info['mtime'] < cutoff_time:
                try:
                    os.remove(file_info['path'])
                    deleted_count += 1
                    total_size -= file_info['size']
                    print(f"Delete expired files: {file_info['filename']}")
                except Exception as e:
                    print(f"Failed to delete file {file_info['filename']}: {e}")

        # If file count still too high, delete oldest files
        remaining_files = [f for f in files_info if os.path.exists(f['path'])]
        while len(remaining_files) > MAX_FILES_COUNT:
            oldest_file = remaining_files.pop(0)
            try:
                os.remove(oldest_file['path'])
                deleted_count += 1
                total_size -= oldest_file['size']
                print(f"Deleted excess file: {oldest_file['filename']}")
            except Exception as e:
                print(f"Failed to delete file {oldest_file['filename']}: {e}")

        # If folder size still too large, delete oldest files
        remaining_files = [f for f in files_info if os.path.exists(f['path'])]
        while total_size > MAX_FOLDER_SIZE and remaining_files:
            oldest_file = remaining_files.pop(0)
            try:
                os.remove(oldest_file['path'])
                deleted_count += 1
                total_size -= oldest_file['size']
                print(f"Deleted large file: {oldest_file['filename']}")
            except Exception as e:
                print(f"Failed to delete file {oldest_file['filename']}: {e}")

        if deleted_count > 0:
            print(f"File cleanup complete, deleted {deleted_count} files")

    except Exception as e:
        print(f"File cleanup failed: {e}")

def start_file_cleanup_scheduler():
    """Start file cleanup scheduler"""
    def cleanup_worker():
        while True:
            try:
                cleanup_old_files()
                # Run cleanup every hour
                time.sleep(3600)
            except Exception as e:
                print(f"File cleanup scheduler error: {e}")
                time.sleep(3600)

    # Start background thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print("File cleanup scheduler started")

def init_session():
    """Initialize session data"""
    # Only store necessary info in session, avoid session getting too large
    if 'current_conversation_id' not in session:
        session['current_conversation_id'] = None
    # No longer store chat_history in session, load from file on demand
    if 'pinned_conversations' not in session:
        # Load pinned conversation list from file
        session['pinned_conversations'] = load_pinned_conversations()
    # No longer store all conversations in session, load from file on demand

@app.route('/')
def index():
    """Home page"""
    init_session()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        init_session()

        # Get or create conversation ID
        conversation_id = session.get('current_conversation_id')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            session['current_conversation_id'] = conversation_id
            session.modified = True

        filename = secure_filename(file.filename)
        # Separate filename and extension
        name, ext = os.path.splitext(filename)

        # Add timestamp and conversation ID to avoid filename conflicts, keep consistent with agent-generated files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        conversation_prefix = conversation_id[:8]  # Use first 8 characters of conversation ID
        filename = f"uploaded_{conversation_prefix}_{timestamp}_{name}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # Read file content
        content = read_file_content(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'content': content[:2000] + '...' if len(content) > 2000 else content,  # Limit display length
            'full_content': content,
            'conversation_id': conversation_id
        })

    return jsonify({'error': 'File type not supported'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat request"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        file_content = data.get('file_content', '')
        conversation_id = data.get('conversation_id', None)

        if not user_message and not file_content:
            return jsonify({'error': 'Please enter message or upload file'}), 400

        init_session()

        # Initialize agents (lazy initialization)
        init_agents()

        # If no conversation ID specified, try to use current active conversation
        if not conversation_id:
            conversation_id = session.get('current_conversation_id')

        # Get or create conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            current_conversation = {
                'id': conversation_id,
                'title': generate_conversation_title(user_message),
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        else:
            # Load existing conversation from file
            current_conversation = load_conversation_from_file(conversation_id)
            if not current_conversation:
                # Conversation does not exist, create new conversation
                conversation_id = str(uuid.uuid4())
                current_conversation = {
                    'id': conversation_id,
                    'title': generate_conversation_title(user_message),
                    'messages': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }

        # Set current conversation
        session['current_conversation_id'] = conversation_id

        # Clear new conversation flag (user has started sending messages)
        if 'is_new_conversation' in session:
            del session['is_new_conversation']
            session.modified = True

        # Check if it's an .inp file (determined by file content features)
        is_inp_file = False
        inp_file_path = None
        is_csv_file = False
        csv_file_path = None

        if file_content:
            # Check if file content contains EPANET format features
            if ('[JUNCTIONS]' in file_content or '[PIPES]' in file_content or
                '[RESERVOIRS]' in file_content or '[TANKS]' in file_content):
                is_inp_file = True

                # Save as temporary .inp file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                inp_filename = f"temp_network_{conversation_id[:8]}_{timestamp}.inp"
                inp_file_path = os.path.join(UPLOAD_FOLDER, inp_filename)

                with open(inp_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)

            # Check if it's a CSV file (determined by content features)
            elif (',' in file_content and '\n' in file_content):
                # Simple check if it looks like CSV format
                lines = file_content.strip().split('\n')
                if len(lines) > 1:  # At least header row and data row
                    is_csv_file = True

                    # Save as temporary CSV file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"sensor_data_{conversation_id[:8]}_{timestamp}.csv"
                    csv_file_path = os.path.join(UPLOAD_FOLDER, csv_filename)

                    with open(csv_file_path, 'w', encoding='utf-8') as f:
                        f.write(file_content)

        # Priority check CSV inference scenario - if CSV file uploaded and user input contains inference keywords
        skip_llm_analysis = False
        if is_csv_file and csv_file_path:
            message_lower = user_message.lower()
            inference_keywords = ['inference', 'predict', 'analyze', 'detect', 'identify']

            if any(keyword in message_lower for keyword in inference_keywords):
                print(f"Detected CSV inference scenario, skip LLM analysis, enter inference mode directly")
                print(f"   - CSV file: {os.path.basename(csv_file_path)}")
                print(f"   - User message: {user_message}")
                skip_llm_analysis = True

        # Only perform LLM analysis in non-CSV inference scenarios
        task_analysis = None
        if not skip_llm_analysis:
            # New LLM-driven task analysis logic
            print(f"Starting LLM task analysis, user message: {user_message}")

            # Use LLM task analyzer to analyze user intent
            task_analysis = llm_task_analyzer.process(
                user_message,
                conversation_id,
                current_conversation.get('messages', [])
            )

            print(f"LLM task analysis result: {task_analysis}")

        # If analysis successful and need to execute agent task
        if (task_analysis and task_analysis.get('success') and
            task_analysis.get('analysis', {}).get('task_type') in ['single', 'workflow']):

            # Check prerequisites
            prerequisites = task_analysis.get('prerequisites', {})

            # If all prerequisites satisfied, execute task
            if prerequisites.get('all_satisfied', False):
                print("Prerequisites satisfied, starting agent task execution")

                # Use agent executor to execute task
                execution_result = agent_executor.process(
                    task_analysis['execution_plan'],
                    conversation_id,
                    user_message
                )

                if execution_result.get('success'):
                    # Get LLM generated response
                    assistant_message = execution_result['llm_response']

                    # Check if there's network analysis result, add detailed network info if so
                    execution_results = execution_result.get('execution_results', [])
                    for step_result in execution_results:
                        if (step_result.get('step_name') == 'Network analysis' and
                            step_result.get('result') and
                            step_result['result'].get('network_info')):
                            network_info = step_result['result']['network_info']
                            network_details = f"""

## 📊 Detailed network information

### 🏗️ Network structure
- **Total nodes**: {network_info['nodes']['total']} 
  - Junctions: {network_info['nodes']['junctions']} 
  - Reservoirs: {network_info['nodes']['reservoirs']} 
  - Tanks: {network_info['nodes']['tanks']} 

- **Total links**: {network_info['links']['total']} 
  - Pipes: {network_info['links']['pipes']} 
  - Pumps: {network_info['links']['pumps']} 
  - Valves: {network_info['links']['valves']} 

### 📏 Network parameters
- **Total network length**: {network_info['network_stats']['total_length']:.2f} meters
- **Simulation duration**: {network_info['network_stats']['simulation_duration']} seconds
- **Hydraulic timestep**: {network_info['network_stats']['hydraulic_timestep']} seconds
- **Pattern timestep**: {network_info['network_stats']['pattern_timestep']} seconds

### 🎯 Analysis suggestions
Based on above network information, you can perform the following further analysis: 
- 🔄 **Hydraulic simulation**: Calculate node pressure and pipe flow
- 🗂️ **Network partition**: Divide network into management zones
- 📍 **Sensor placement**: Optimize monitoring point locations
- 🔍 **Leak detection**: Train and apply leak detection model
"""
                            assistant_message += network_details
                            break

                    # Collect download file info
                    downloads = []
                    for exec_result in execution_result['execution_results']:
                        if exec_result.get('success') and exec_result.get('result'):
                            agent_result = exec_result['result']

                            # Check CSV file
                            if agent_result.get('csv_info') and agent_result['csv_info'].get('success'):
                                downloads.append({
                                    'type': 'csv',
                                    'step': exec_result['step_name'],
                                    'filename': agent_result['csv_info']['filename'],
                                    'url': agent_result['csv_info']['download_url'],
                                    'size': agent_result['csv_info']['file_size']
                                })

                            # Check visualization image
                            if agent_result.get('visualization'):
                                viz_info = agent_result['visualization']
                                if viz_info.get('filename') and viz_info.get('path'):
                                    viz_filename = viz_info['filename']
                                    viz_url = f"/static_files/{viz_filename}"

                                    try:
                                        viz_size = os.path.getsize(viz_info['path'])
                                    except:
                                        viz_size = 0

                                    downloads.append({
                                        'type': 'image',
                                        'step': exec_result['step_name'],
                                        'filename': viz_filename,
                                        'url': viz_url,
                                        'size': viz_size,
                                        'display_url': viz_url
                                    })

                            # Check model file (special handling for leak detection model training files)
                            if agent_result.get('files'):
                                for file_type, file_info in agent_result['files'].items():
                                    if file_info.get('success'):
                                        # Determine download type based on file type and extension
                                        download_type = file_type
                                        filename = file_info['filename']

                                        # Special handling for leak detection model file types
                                        if exec_result['step_name'] == 'Leak model training':
                                            if filename.endswith('.csv'):
                                                download_type = 'csv'
                                            elif filename.endswith('.pth'):
                                                download_type = 'model'

                                        downloads.append({
                                            'type': download_type,
                                            'step': exec_result['step_name'],
                                            'filename': filename,
                                            'url': file_info['download_url'],
                                            'size': file_info['file_size']
                                        })

                    # Save to conversation history
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': task_analysis['analysis']['standard_phrase'],
                        'confidence': task_analysis['analysis']['confidence'],
                        'task_analysis': task_analysis,
                        'execution_results': execution_result['execution_results']
                    }

                    # Add download info to conversation history
                    if downloads:
                        message_data['downloads'] = downloads

                    # If file uploaded, record file info
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path
                        })
                    elif is_csv_file and csv_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'csv',
                            'file_path': csv_file_path
                        })

                    current_conversation['messages'].append(message_data)
                    current_conversation['updated_at'] = datetime.now().isoformat()

                    # Update conversation title
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == 'New Conversation':
                        current_conversation['title'] = generate_conversation_title(user_message)

                    # Save conversation
                    save_conversation_to_file(conversation_id, current_conversation)
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # Build response data
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': task_analysis['analysis']['standard_phrase'],
                        'confidence': task_analysis['analysis']['confidence'],
                        'task_analysis': task_analysis,
                        'execution_summary': {
                            'total_steps': execution_result['total_steps'],
                            'completed_steps': execution_result['completed_steps'],
                            'execution_results': execution_result['execution_results']
                        }
                    }

                    # Add download file info (if any)
                    if downloads:
                        response_data['downloads'] = downloads

                    print(f"LLM-driven task execution success, returning response")
                    return safe_jsonify(response_data)

                else:
                    # Execution failed, use error message
                    error_message = f"Task execution failed: {execution_result.get('error', 'Unknown error')}"
                    print(f"Task execution failed: {error_message}")

            else:
                # Prerequisites not met, generate prompt message
                missing = prerequisites.get('missing_prerequisites', [])
                missing_text = []

                if 'inp_file' in missing:
                    missing_text.append("Network INP file")
                if 'partition_csv' in missing:
                    missing_text.append("Partition CSV file (need to perform network partition first)")
                if 'trained_model' in missing:
                    missing_text.append("Trained leak detection model")

                error_message = f"Missing required prerequisites: {', '.join(missing_text)}. Please complete related steps first. "
                print(f"Prerequisites not met: {error_message}")

        else:
            # CSV inference scenario, jump directly to simplified inference logic
            print("🎯 Skip LLM analysis, directly enter CSV inference mode")

        # If LLM analysis failed or no need to execute agent task, fallback to original logic
        print("Fallback to original agent processing logic")

        # Check if need to use specific agent processing
        should_use_partition_sim = False
        should_use_hydro_sim = False
        should_use_sensor_placement = False
        should_use_leak_detection = False
        agent_inp_file_path = None
        partition_csv_path = None

        # Process CSV file leak detection - simplified inference mode
        if is_csv_file and csv_file_path:
            print(f"\n" + "="*60)
            print(f"🔍 CSV file upload detected, starting intelligent inference mode...")
            print(f"📂 CSV file: {os.path.basename(csv_file_path)}")
            print(f"🆔 Conversation ID: {conversation_id}")
            print(f"💬 User message: {user_message}")
            print("="*60)

            # Simplified prerequisites check: only need trained model file
            missing_prerequisites = []

            # Check if trained model file exists and perform dimension compatibility check
            model_file_path = None
            if os.path.exists('downloads'):
                # First read CSV file to determine column count
                csv_columns = 0
                try:
                    import pandas as pd
                    df_temp = pd.read_csv(csv_file_path)
                    csv_columns = len(df_temp.columns)
                    print(f"📊 CSV file column count: {csv_columns}")
                except Exception as e:
                    print(f"⚠️ Cannot read CSV file column count: {e}")

                model_files = []
                compatible_models = []

                # Find all model files
                for filename in os.listdir('downloads'):
                    if ('leak_detection_model' in filename and filename.endswith('.pth')):
                        model_files.append(filename)

                print(f"Found {len(model_files)} model files, checking compatibility...")

                # Check model compatibility
                for model_file in model_files:
                    model_path = os.path.join('downloads', model_file)
                    try:
                        import torch
                        checkpoint = torch.load(model_path, map_location='cpu')

                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint

                        # Find first layer weights to determine input dimension
                        first_layer_key = None
                        for key in state_dict.keys():
                            if 'weight' in key and ('fc1' in key or 'linear' in key or '0' in key):
                                first_layer_key = key
                                break

                        if first_layer_key:
                            input_dim = state_dict[first_layer_key].shape[1]
                            print(f"   {model_file}: Input dimension={input_dim}", end="")

                            if csv_columns > 0 and input_dim == csv_columns:
                                compatible_models.append(model_file)
                                print(f" ✅ Compatible")
                            else:
                                print(f" Not compatible (Need {input_dim} columns, CSV has {csv_columns} columns)")
                        else:
                            print(f"   {model_file}: ❓ Cannot determine input dimension")

                    except Exception as e:
                        print(f"   {model_file}: ❌ Check failed: {e}")

                # Select model - strictly based on conversation ID matching
                if compatible_models:
                    selected_model = None
                    conversation_prefix = conversation_id[:8]

                    print(f"🎯 Model selection strategy:")
                    print(f"   - Current conversation ID prefix: {conversation_prefix}")
                    print(f"   - Compatible model count: {len(compatible_models)}")

                    # Only select compatible models for current conversation ID
                    for model in compatible_models:
                        if conversation_prefix in model:
                            selected_model = model
                            print(f"   ✅ Found compatible model for current conversation ID: {model}")
                            break

                    # If current conversation has no corresponding model, return error directly
                    if not selected_model:
                        print(f"   ❌ Current conversation ID {conversation_prefix} No corresponding training model")
                        print(f"   💡 Available compatible model conversation IDs:")
                        available_conversation_ids = set()
                        for model in compatible_models:
                            # Extract conversation ID from model file
                            parts = model.split('_')
                            if len(parts) >= 4:
                                model_conversation_id = parts[3]
                                available_conversation_ids.add(model_conversation_id)

                        for conv_id in sorted(available_conversation_ids):
                            print(f"     - {conv_id}")

                        missing_prerequisites.append(f"Leak detection model for conversation ID {conversation_prefix}")
                        print(f"   Suggestion: Please first train leak detection model for current conversation, or use conversation with corresponding model")
                    else:
                        model_file_path = os.path.join('downloads', selected_model)
                        print(f"🏆 Final selected model: {selected_model}")
                else:
                    missing_prerequisites.append("Compatible leak detection model")
                    print(f"❌ No model compatible with CSV file found")
                    if model_files:
                        print(f"💡 Suggestion: Use corresponding training model or retrain model")
            else:
                missing_prerequisites.append("Leak detection model")
                print(f"❌ downloads directory does not exist")

            # Prerequisites check result
            if missing_prerequisites:
                print(f"❌ Missing prerequisites: {', '.join(missing_prerequisites)}")

                # Check if it's a conversation ID mismatch problem
                conversation_prefix = conversation_id[:8]
                is_conversation_mismatch = any(f"Conversation ID {conversation_prefix}" in item for item in missing_prerequisites)

                if is_conversation_mismatch:
                    detailed_error = f"""
❌ **Cannot find corresponding PTH model file**

**Problem**: Current conversation ID `{conversation_prefix}` No corresponding training model

**Reason**: Inference requires a training model matching the current conversation ID

**Solution**: 
1. **Recommended solution**: Please first train leak model for current conversation
   - Input training command (e.g.:"Leak model training, iterations 100, generate 50 data groups"）
   - Wait for training to complete before performing inference

2. **Alternative solution**: Use existing model conversation for inference
   - Switch to conversation with corresponding training model
   - Upload CSV file in same format for inference

**Technical note**: System uses strict conversation ID matching strategy to ensure inference accuracy and traceability. 
"""
                else:
                    detailed_error = f"""
❌ **Leak detection inference failed**

🚫 **Missing required conditions**

{chr(10).join([f'• {item}' for item in missing_prerequisites])}

📋 **Solution**: 
1. **If no model**: Please complete the following steps first: 
   - Upload network INP file
   - Perform network partition analysis
   - Perform sensor placement optimization
   - Train leak detection model

2. **If model exists**: Please confirm model file is in downloads directory

💡 **Quick check**: Look for files like in downloads directory `leak_detection_model_{conversation_prefix}_*.pth` files
"""

                return jsonify({
                    'response': detailed_error,
                    'conversation_id': conversation_id,
                    'error': True,
                    'error_type': 'missing_model' if is_conversation_mismatch else 'missing_prerequisites',
                    'missing_prerequisites': missing_prerequisites,
                    'conversation_mismatch': is_conversation_mismatch
                })

            # Prerequisites met, perform simplified inference
            if model_file_path:
                try:
                    # Directly perform leak detection inference, simplified flow
                    print(f"Starting leak detection inference...")
                    print(f"   📂 Model file: {os.path.basename(model_file_path)}")
                    print(f"   📊 Sensor data: {os.path.basename(csv_file_path)}")

                    # Directly call inference method, pass conversation_id to read partition file
                    detection_result = leak_detection_agent.detect_leak_from_file(csv_file_path, model_file_path, conversation_id)

                    # Process inference result
                    if detection_result.get('success'):
                        results = detection_result.get('results', [])
                        summary = detection_result.get('summary', {})

                        # Print detailed inference results
                        print("\n" + "="*60)
                        print("Leak detection inference result")
                        print("="*60)

                        print(f"📊 Inference result summary:")
                        print(f"   - Total samples: {summary.get('total_samples', 0)}")
                        print(f"   - Normal samples: {summary.get('normal_samples', 0)}")
                        print(f"   - Anomaly samples: {summary.get('anomaly_samples', 0)}")

                        print(f"\n📋 Detailed inference results:")
                        for result in results:
                            sample_id = result.get('sample_id', 0)
                            status = result.get('status', 'N/A')
                            partition = result.get('partition', None)
                            confidence = result.get('confidence', 0)

                            if status == 'Normal':
                                print(f"   Sample {sample_id}: {status} (Confidence: {confidence:.3f})")
                            else:
                                print(f"   Sample {sample_id}: {status} - Partition {partition} (Confidence: {confidence:.3f})")

                        # Anomaly distribution statistics
                        if summary.get('anomaly_samples', 0) > 0:
                            partition_stats = {}
                            for result in results:
                                if result.get('status') == 'Anomaly':
                                    partition = result.get('partition')
                                    if partition:
                                        partition_stats[partition] = partition_stats.get(partition, 0) + 1

                            if partition_stats:
                                print(f"\nAnomaly distribution statistics:")
                                for partition, count in sorted(partition_stats.items()):
                                    print(f"   - Partition {partition}: {count} anomaly samples")

                        print("="*60)

                        # Generate inference result CSV file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        conversation_prefix = conversation_id[:8]
                        inference_result_filename = f"leak_inference_result_{conversation_prefix}_{timestamp}.csv"
                        inference_result_path = os.path.join(DOWNLOADS_FOLDER, inference_result_filename)

                        # Save inference results to CSV
                        import pandas as pd
                        results_data = []
                        for result in results:
                            results_data.append({
                                'Sample ID': result.get('sample_id', 0),
                                'Detection status': result.get('status', 'N/A'),
                                'Anomaly partition': result.get('partition', '') if result.get('status') == 'Anomaly' else '',
                                'Confidence': f"{result.get('confidence', 0):.4f}",
                                'Confidence percentage': f"{result.get('confidence', 0):.1%}"
                            })

                        df_results = pd.DataFrame(results_data)
                        df_results.to_csv(inference_result_path, index=False, encoding='utf-8-sig')

                        # Add to download file list
                        download_url = f"/download/{inference_result_filename}"
                        file_size = os.path.getsize(inference_result_path) if os.path.exists(inference_result_path) else 0

                        detection_result['download_files'] = [{
                            'filename': inference_result_filename,
                            'path': inference_result_path,
                            'url': download_url,
                            'download_url': download_url,
                            'type': 'csv',
                            'size': file_size,
                            'description': 'Leak detection inference result',
                            'step': 'Leak detection inference',
                            'records_count': len(results)
                        }]

                        print(f"💾 Inference result saved: {inference_result_filename}")
                        print(f"📁 File path: {inference_result_path}")
                        print(f"🔗 Download URL: {download_url}")
                        print(f"📊 File size: {file_size} bytes")

                    if detection_result['success']:
                        # Call GPT with professional prompt generated by agent
                        prompt = leak_detection_agent.build_response_prompt(detection_result, user_message, "detection")

                        # Add inference result file info to prompt
                        if detection_result.get('download_files'):
                            prompt += f"""

📁 **Inference result file generated**

System has generated detailed inference result file containing detection status, anomaly partition and confidence for each sample: 

"""
                            for file_info in detection_result.get('download_files', []):
                                prompt += f"• **{file_info['description']}**: `{file_info['filename']}`\n"

                        prompt += f"""

🎯 **Inference mode description**: System detected existing trained model file, directly performing inference analysis, skipped partition analysis, sensor placement, model training steps, greatly improving processing efficiency. 

📊 **Resources used**:
- Model file: `{os.path.basename(model_file_path)}`
- Sensor data: `{os.path.basename(csv_file_path)}`

Please generate professional leak detection analysis report based on above inference results. 
"""

                        messages = [
                            {"role": "system", "content": "You are a professional water distribution network leak detection expert with rich experience in anomaly detection and fault diagnosis. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                        ]

                        # Add current conversation history messages (last 10 rounds)
                        for msg in current_conversation['messages'][-10:]:
                            messages.append({"role": "user", "content": msg['user']})
                            messages.append({"role": "assistant", "content": msg['assistant']})

                        messages.append({"role": "user", "content": prompt})

                        # Call OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.7
                        )

                        assistant_message = response.choices[0].message.content

                        # Save to current conversation
                        message_data = {
                            'user': user_message,
                            'assistant': assistant_message,
                            'timestamp': datetime.now().isoformat(),
                            'intent': 'leak_detection',
                            'confidence': 0.9,
                            'has_file': True,
                            'file_type': 'csv',
                            'file_path': csv_file_path,
                            'detection_results': detection_result
                        }

                        current_conversation['messages'].append(message_data)

                        # Save to file
                        save_conversation_to_file(conversation_id, current_conversation)
                        # Update conversation index
                        all_conversations = load_all_conversations()
                        all_conversations[conversation_id] = current_conversation
                        save_conversations_index(all_conversations)
                        session.modified = True

                        # Build response
                        response_data = {
                            'success': True,
                            'response': assistant_message,
                            'conversation_id': conversation_id,
                            'intent': 'leak_detection_inference',
                            'confidence': 0.9,
                            'detection_results': detection_result,
                            'inference_mode': True,
                            'model_used': os.path.basename(model_file_path),
                            'workflow_skipped': ['Partition analysis', 'Sensor placement', 'Model training']
                        }

                        # Add download file info
                        download_files = detection_result.get('download_files', [])
                        if download_files:
                            response_data['downloads'] = download_files
                            print(f"📁 Add download files to response: {len(download_files)} files")
                            for i, file_info in enumerate(download_files):
                                print(f"   File {i+1}: {file_info.get('filename', 'N/A')}")
                                print(f"     - URL: {file_info.get('url', 'N/A')}")
                                print(f"     - Size: {file_info.get('size', 'N/A')} bytes")
                        else:
                            print(f"❌ No download files added to response")

                        print(f"📤 Return inference response, containing {len(response_data.get('downloads', []))} download files")
                        return jsonify(response_data)

                    else:
                        # Detection failed
                        full_message = f"Leak detection failed: {detection_result.get('error', 'Unknown error')}\n\nUser question: {user_message}"

                except Exception as e:
                    print(f"Leak detection processing error: {e}")
                    full_message = f"Error processing sensor data: {str(e)}\n\nUser question: {user_message}"

            else:
                # Model file not found
                full_message = f"User uploaded sensor data CSV file, but no corresponding leak detection model found. Please train leak detection model first. \n\nUser question: {user_message}"

        # Determine available inp file path
        elif is_inp_file and inp_file_path:
            # Newly uploaded .inp file
            agent_inp_file_path = inp_file_path
        elif current_conversation and has_inp_file_in_conversation_history(current_conversation):
            # Conversation history has .inp file
            historical_inp_path = get_inp_file_from_conversation_history(current_conversation)
            if historical_inp_path:
                agent_inp_file_path = historical_inp_path

        # If inp file available, determine which agent to use
        if agent_inp_file_path:
            # Define keywords
            partition_keywords = ['Partition', 'cluster', 'FCM', 'fuzzy clustering', 'clustering', 'partition', 'region division', 'network division', 'outlier']
            sensor_keywords = ['sensor', 'monitoring point', 'pressure monitoring', 'sensor', 'monitoring', 'resilience', 'sensitivity', 'placement', 'optimization', 'detection point']
            leak_keywords = ['leak', 'leakage', 'water leak', 'anomaly detection', 'fault detection', 'leak', 'leakage', 'leak detection', 'leak analysis', 'train model', 'leak model']

            # Check if it's a leak detection related request
            if any(keyword in user_message for keyword in leak_keywords):
                should_use_leak_detection = True
            # Check if it's a sensor placement related request
            elif any(keyword in user_message for keyword in sensor_keywords):
                should_use_sensor_placement = True
                # Check if historical partition result exists
                if current_conversation:
                    partition_csv_path = get_partition_csv_from_conversation_history(current_conversation)
            # Check if it's a partition related request
            elif any(keyword in user_message for keyword in partition_keywords):
                should_use_partition_sim = True
            else:
                # Default to use hydraulic simulation agent
                should_use_hydro_sim = True

        # Use PartitionSim agent for processing
        if should_use_partition_sim and agent_inp_file_path:
            try:
                partition_result = partition_sim_agent.process(agent_inp_file_path, user_message, conversation_id)

                if partition_result['success']:
                    # Call GPT with professional prompt generated by agent
                    messages = [
                        {"role": "system", "content": "You are a professional water distribution network partition analysis expert with rich experience in network clustering and partition optimization. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # Add current conversation history messages (last 10 rounds)
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # Use professional prompt generated by agent
                    messages.append({"role": "user", "content": partition_result['prompt']})

                    # Call OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # Save to current conversation
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': partition_result['intent'],
                        'confidence': partition_result['confidence']
                    }

                    # If newly uploaded file, save file info
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path
                        })
                    else:
                        # Conversation using historical file
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    current_conversation['messages'].append(message_data)

                    # Update conversation time
                    current_conversation['updated_at'] = datetime.now().isoformat()

                    # If first message and title is default, update title
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == 'New Conversation':
                        current_conversation['title'] = generate_conversation_title(user_message or "Network partition analysis")

                    # Limit message count per conversation
                    if len(current_conversation['messages']) > 50:
                        current_conversation['messages'] = current_conversation['messages'][-50:]

                    # Save to file
                    save_conversation_to_file(conversation_id, current_conversation)
                    # Update conversation index
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # Build response
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': partition_result['intent'],
                        'confidence': partition_result['confidence'],
                        'partition_info': partition_result.get('partition_info', {})
                    }

                    # If CSV file generated, add download info
                    if partition_result.get('csv_info') and partition_result['csv_info']['success']:
                        response_data['download'] = {
                            'available': True,
                            'filename': partition_result['csv_info']['filename'],
                            'url': partition_result['csv_info']['download_url'],
                            'size': partition_result['csv_info']['file_size'],
                            'records_count': partition_result['csv_info']['records_count']
                        }

                    # If visualization image generated, add display info
                    if partition_result.get('visualization'):
                        response_data['visualization'] = {
                            'available': True,
                            'filename': partition_result['visualization']['filename'],
                            'url': f'/static_files/{partition_result["visualization"]["filename"]}',
                            'download_url': f'/download/{partition_result["visualization"]["filename"]}'
                        }

                    return jsonify(response_data)

                else:
                    # Agent processing failed, use normal method
                    full_message = f"User uploaded network file (.inp format), but encountered problem during partition analysis: {partition_result.get('response', 'Unknown error')}\n\nUser question: {user_message}"

            except Exception as e:
                print(f"PartitionSim agent processing error: {e}")
                full_message = f"User uploaded network file (.inp format), but partition agent encountered error. \n\nUser question: {user_message}"

        # Use SensorPlacement agent for processing
        elif should_use_sensor_placement and agent_inp_file_path:
            try:
                # If no partition result, perform auto partition first
                if not partition_csv_path:
                    print("No historical partition result found, starting auto partition...")
                    partition_result = partition_sim_agent.process(
                        agent_inp_file_path,
                        "Auto partition for sensor placement, divided into 6 partitions",
                        conversation_id
                    )

                    if partition_result['success'] and partition_result.get('csv_info'):
                        partition_csv_path = partition_result['csv_info']['filepath']
                        print(f"Auto partition complete, partition file: {partition_csv_path}")
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Auto partition failed, cannot perform sensor placement: {partition_result.get("response", "Unknown error")}'
                        })
                else:
                    print(f"Using historical partition result: {partition_csv_path}")

                # Perform sensor placement
                sensor_result = sensor_placement_agent.process(
                    agent_inp_file_path,
                    partition_csv_path,
                    user_message,
                    conversation_id
                )

                if sensor_result['success']:
                    # Call GPT with professional prompt generated by agent
                    messages = [
                        {"role": "system", "content": "You are a professional water distribution network sensor placement expert with rich experience in pressure monitoring point optimization and resilience analysis. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # Add current conversation history messages (last 10 rounds)
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # Use professional prompt generated by agent
                    messages.append({"role": "user", "content": sensor_result['prompt']})

                    # Call OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # Save to current conversation
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': 'sensor_placement',
                        'confidence': 0.9
                    }

                    # If newly uploaded file, save file info
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path
                        })
                    else:
                        # Conversation using historical file
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    # Add sensor placement result info
                    if sensor_result.get('csv_info'):
                        message_data['csv_info'] = sensor_result['csv_info']
                        print(message_data['csv_info'])
                    # Add resilience analysis result info
                    if sensor_result.get('resilience_csv_info'):
                        message_data['resilience_csv_info'] = sensor_result['resilience_csv_info']
                        print(message_data['resilience_csv_info'])
                    current_conversation['messages'].append(message_data)

                    # Save to file
                    save_conversation_to_file(conversation_id, current_conversation)
                    # Update conversation index
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # Build response
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': 'sensor_placement',
                        'confidence': 0.9,
                        'sensor_info': sensor_result.get('sensor_info', {})
                    }

                    # If CSV file generated, add download info
                    if sensor_result.get('csv_info') and sensor_result['csv_info']['success']:
                        response_data['download'] = {
                            'available': True,
                            'filename': sensor_result['csv_info']['filename'],
                            'url': sensor_result['csv_info']['download_url'],
                            'size': sensor_result['csv_info']['file_size'],
                            'sensor_count': sensor_result['csv_info']['sensor_count']
                        }

                    # If resilience analysis file generated, add download info
                    if sensor_result.get('resilience_csv_info'):
                        response_data['resilience_download'] = {
                            'available': True,
                            'filename': os.path.basename(sensor_result['resilience_csv_info']),
                            'url': f'/download/{os.path.basename(sensor_result["resilience_csv_info"])}'
                        }

                    # If visualization image generated, add display info
                    if sensor_result.get('visualization'):
                        response_data['visualization'] = {
                            'available': True,
                            'filename': sensor_result['visualization']['filename'],
                            'url': f'/static_files/{sensor_result["visualization"]["filename"]}',
                            'download_url': f'/download/{sensor_result["visualization"]["filename"]}'
                        }

                    return jsonify(response_data)

                else:
                    # Agent processing failed, use normal method
                    full_message = f"User uploaded network file (.inp format), but encountered problem during sensor placement: {sensor_result.get('response', 'Unknown error')}\n\nUser question: {user_message}"

            except Exception as e:
                print(f"SensorPlacement agent processing error: {e}")
                full_message = f"User uploaded network file (.inp format), but sensor placement agent encountered error. \n\nUser question: {user_message}"

        # Use LeakDetectionAgent for processing
        elif should_use_leak_detection and agent_inp_file_path:
            try:
                # Check if it's a training request
                training_keywords = ['train', 'model', 'train', 'model', 'machine learning', 'AI', 'learn']
                is_training_request = any(keyword in user_message for keyword in training_keywords)

                if is_training_request:
                    # Train leak detection model
                    print("Starting leak detection model training...")

                    # Extract training parameters
                    num_scenarios = 50  # Default value
                    epochs = 100  # Default value

                    # Intelligently extract training parameters
                    import re
                    num_scenarios, epochs = extract_training_parameters(user_message, num_scenarios, epochs)
                    print(f"Parsed training parameters: Sample count={num_scenarios}, Iteration count={epochs}")

                    leak_result = leak_detection_agent.train_leak_detection_model(
                        agent_inp_file_path,
                        conversation_id,
                        num_scenarios,
                        epochs
                    )

                    if leak_result['success']:
                        # Call GPT with professional prompt generated by agent
                        prompt = leak_detection_agent.build_response_prompt(leak_result, user_message, "training")

                        messages = [
                            {"role": "system", "content": "You are a professional water distribution network leak detection expert with rich experience in machine learning and anomaly detection. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                        ]

                        # Add current conversation history messages (last 10 rounds)
                        for msg in current_conversation['messages'][-10:]:
                            messages.append({"role": "user", "content": msg['user']})
                            messages.append({"role": "assistant", "content": msg['assistant']})

                        messages.append({"role": "user", "content": prompt})

                        # Call OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.7
                        )

                        assistant_message = response.choices[0].message.content

                        # Save to current conversation
                        message_data = {
                            'user': user_message,
                            'assistant': assistant_message,
                            'timestamp': datetime.now().isoformat(),
                            'intent': 'leak_detection_training',
                            'confidence': 0.9
                        }

                        # If newly uploaded file, save file info
                        if is_inp_file and inp_file_path:
                            message_data.update({
                                'has_file': True,
                                'file_type': 'inp',
                                'file_path': inp_file_path
                            })
                        else:
                            # Conversation using historical file
                            message_data.update({
                                'has_file': False,
                                'uses_historical_file': True,
                                'historical_file_path': agent_inp_file_path
                            })

                        # Add training result info
                        if leak_result.get('files'):
                            message_data['leak_training_files'] = leak_result['files']

                        current_conversation['messages'].append(message_data)

                        # Save to file
                        save_conversation_to_file(conversation_id, current_conversation)
                        # Update conversation index
                        all_conversations = load_all_conversations()
                        all_conversations[conversation_id] = current_conversation
                        save_conversations_index(all_conversations)
                        session.modified = True

                        # Build response
                        response_data = {
                            'success': True,
                            'response': assistant_message,
                            'conversation_id': conversation_id,
                            'intent': 'leak_detection_training',
                            'confidence': 0.9,
                            'model_info': leak_result.get('model_info', {}),
                            'evaluation': leak_result.get('evaluation', {})
                        }

                        # Add download info
                        if leak_result.get('files'):
                            response_data['downloads'] = []
                            for file_type, file_info in leak_result['files'].items():
                                if file_info.get('success'):
                                    response_data['downloads'].append({
                                        'type': file_type,
                                        'filename': file_info['filename'],
                                        'url': file_info['download_url'],
                                        'size': file_info['file_size']
                                    })

                        return jsonify(response_data)

                    else:
                        # Training failed, use normal method
                        full_message = f"Leak detection model training failed: {leak_result.get('error', 'Unknown error')}\n\nUser question: {user_message}"

                else:
                    # Detection request - need to upload sensor data file
                    full_message = f"User wants to perform leak detection. Please remind user that they need: \n1. First train leak detection model\n2. Upload sensor pressure data CSV file for detection\n\nUser question: {user_message}"

            except Exception as e:
                print(f"LeakDetectionAgent processing error: {e}")
                full_message = f"User uploaded network file (.inp format), but leak detection agent encountered error. \n\nUser question: {user_message}"

        # Use HydroSim agent for processing
        elif should_use_hydro_sim and agent_inp_file_path:
            try:
                hydro_result = hydro_sim_agent.process(agent_inp_file_path, user_message, conversation_id)

                if hydro_result['success']:
                    # Call GPT with prompt generated by agent
                    messages = [
                        {"role": "system", "content": "You are a professional water distribution network analysis expert with rich experience in hydraulic calculation and network analysis. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # Add current conversation history messages (last 10 rounds)
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # Add agent generated prompt
                    messages.append({"role": "user", "content": hydro_result['prompt']})

                    # Call OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # Debug info
                    print(f"Debug: hydro_result keys: {list(hydro_result.keys())}")
                    print(f"Debug: network_info exists: {hydro_result.get('network_info') is not None}")
                    if hydro_result.get('network_info'):
                        print(f"Debug: network_info type: {type(hydro_result['network_info'])}")

                    # If network info exists, add detailed network info after response
                    if hydro_result.get('network_info'):
                        network_info = hydro_result['network_info']
                        network_details = f"""

## 📊 Detailed network information

### 🏗️ Network structure
- **Total nodes**: {network_info['nodes']['total']} 
  - Junctions: {network_info['nodes']['junctions']} 
  - Reservoirs: {network_info['nodes']['reservoirs']} 
  - Tanks: {network_info['nodes']['tanks']} 

- **Total links**: {network_info['links']['total']} 
  - Pipes: {network_info['links']['pipes']} 
  - Pumps: {network_info['links']['pumps']} 
  - Valves: {network_info['links']['valves']} 

### 📏 Network parameters
- **Total network length**: {network_info['network_stats']['total_length']:.2f} meters
- **Simulation duration**: {network_info['network_stats']['simulation_duration']} seconds
- **Hydraulic timestep**: {network_info['network_stats']['hydraulic_timestep']} seconds
- **Pattern timestep**: {network_info['network_stats']['pattern_timestep']} seconds

### 🎯 Analysis suggestions
Based on above network information, you can perform the following further analysis: 
- 🔄 **Hydraulic simulation**: Calculate node pressure and pipe flow
- 🗂️ **Network partition**: Divide network into management zones
- 📍 **Sensor placement**: Optimize monitoring point locations
- 🔍 **Leak detection**: Train and apply leak detection model
"""
                        assistant_message += network_details

                    # Save to current conversation
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': hydro_result['intent'],
                        'confidence': hydro_result['confidence']
                    }

                    # If newly uploaded file, save file info
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path  # Use original inp_file_path
                        })
                    else:
                        # Conversation using historical file
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    current_conversation['messages'].append(message_data)

                    # Update conversation time
                    current_conversation['updated_at'] = datetime.now().isoformat()

                    # If first message and title is default, update title
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == 'New Conversation':
                        current_conversation['title'] = generate_conversation_title(user_message or "Network analysis")

                    # Limit message count per conversation
                    if len(current_conversation['messages']) > 50:
                        current_conversation['messages'] = current_conversation['messages'][-50:]

                    # Save to file
                    save_conversation_to_file(conversation_id, current_conversation)
                    # Update conversation index
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # Build response
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': hydro_result['intent'],
                        'confidence': hydro_result['confidence'],
                        'network_info': hydro_result['network_info']
                    }

                    # If CSV file generated, add download info
                    if hydro_result['csv_info'] and hydro_result['csv_info']['success']:
                        response_data['download'] = {
                            'available': True,
                            'filename': hydro_result['csv_info']['filename'],
                            'url': hydro_result['csv_info']['download_url'],
                            'size': hydro_result['csv_info']['file_size'],
                            'records_count': hydro_result['csv_info']['records_count']
                        }

                    return jsonify(response_data)

                else:
                    # Agent processing failed, use normal method
                    full_message = f"User uploaded network file (.inp format), but encountered problem during processing: {hydro_result.get('response', 'Unknown error')}\n\nUser question: {user_message}"

            except Exception as e:
                print(f"HydroSim agent processing error: {e}")
                full_message = f"User uploaded network file (.inp format), but agent encountered error. \n\nUser question: {user_message}"

        else:
            # Regular file processing
            full_message = user_message
            if file_content:
                full_message = f"User uploaded file content: \n\n{file_content}\n\nUser question: {user_message}" if user_message else f"User uploaded file content: \n\n{file_content}\n\nPlease analyze the content of this file. "
        
        # Build message history (OpenAI format)
        messages = [
            {"role": "system", "content": "You are an advanced AI assistant based on GPT-4 with powerful analysis and reasoning capabilities. You can:: \n1. Thoroughly analyze various file content (code, documents, data, etc.)\n2. Provide professional technical advice and solutions\n3. Perform complex logical reasoning and problem solving\n4. Support multi-language communication, but please prefer Chinese responses\n5. Provide detailed, accurate, and useful answers\n\nPlease provide the best help based on user specific needs. Please use the following signature format at the end of your reply: \n\nBest regards, \n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
        ]

        # Add current conversation history messages (last 10 rounds)
        for msg in current_conversation['messages'][-10:]:
            messages.append({"role": "user", "content": msg['user']})
            messages.append({"role": "assistant", "content": msg['assistant']})

        # Add current user message
        messages.append({"role": "user", "content": full_message})
        
        # Call OpenAI API - use GPT-4 latest version
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",  # GPT-4 Turbo latest version, faster and stronger
            messages=messages,
            max_tokens=4000,  # GPT-4 supports longer output
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content

        # Save to current conversation
        current_conversation['messages'].append({
            'user': user_message,
            'assistant': assistant_message,
            'timestamp': datetime.now().isoformat(),
            'has_file': bool(file_content)
        })

        # Update conversation time
        current_conversation['updated_at'] = datetime.now().isoformat()

        # If first message and title is default, update title
        if len(current_conversation['messages']) == 1 and current_conversation['title'] == 'New Conversation':
            current_conversation['title'] = generate_conversation_title(user_message)

        # Limit message count per conversation
        if len(current_conversation['messages']) > 50:
            current_conversation['messages'] = current_conversation['messages'][-50:]

        # Save to file
        save_conversation_to_file(conversation_id, current_conversation)
        # Update conversation index
        all_conversations = load_all_conversations()
        all_conversations[conversation_id] = current_conversation
        save_conversations_index(all_conversations)

        session.modified = True

        return jsonify({
            'success': True,
            'response': assistant_message,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    """Create new conversation"""
    init_session()
    session['current_conversation_id'] = None
    session['is_new_conversation'] = True  # Mark user actively created new conversation
    session.modified = True
    return jsonify({'success': True})

@app.route('/get_conversations')
def get_conversations():
    """Get all conversation list"""
    init_session()
    # Load all conversations from files
    all_conversations = load_all_conversations()
    conversations = list(all_conversations.values())
    pinned_ids = session.get('pinned_conversations', [])

    # Separate pinned and regular conversations
    pinned_conversations = []
    regular_conversations = []

    for conv in conversations:
        # Add pinned mark
        conv['is_pinned'] = conv['id'] in pinned_ids
        if conv['is_pinned']:
            pinned_conversations.append(conv)
        else:
            regular_conversations.append(conv)

    # Pinned conversations sorted by pin time (latest pinned first)
    pinned_conversations.sort(key=lambda x: pinned_ids.index(x['id']) if x['id'] in pinned_ids else 999)
    # Regular conversations sorted by update time, latest first
    regular_conversations.sort(key=lambda x: x['updated_at'], reverse=True)

    # If no current conversation ID, but conversations exist, auto select latest conversation
    # But if user actively created new conversation, don't auto select
    current_conversation_id = session.get('current_conversation_id')
    is_new_conversation = session.get('is_new_conversation', False)

    if not current_conversation_id and conversations and not is_new_conversation:
        # Select latest updated conversation (regardless of pinned or regular)
        latest_conversation = max(conversations, key=lambda x: x['updated_at'])

        current_conversation_id = latest_conversation['id']
        session['current_conversation_id'] = current_conversation_id
        session.modified = True

    return jsonify({
        'pinned_conversations': pinned_conversations,
        'regular_conversations': regular_conversations,
        'current_conversation_id': current_conversation_id,
        'auto_selected': current_conversation_id != session.get('original_current_conversation_id')
    })

@app.route('/load_conversation/<conversation_id>')
def load_conversation(conversation_id):
    """Load specified conversation"""
    init_session()
    # Load conversation from file
    conversation = load_conversation_from_file(conversation_id)
    if conversation:
        session['current_conversation_id'] = conversation_id
        session.modified = True
        return jsonify({
            'success': True,
            'conversation': conversation
        })
    else:
        return jsonify({'error': 'Conversation does not exist'}), 404

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete specified conversation"""
    init_session()
    # Check if conversation exists
    conversation = load_conversation_from_file(conversation_id)
    if conversation:
        # Delete file
        delete_conversation_file(conversation_id)
        # Update index
        all_conversations = load_all_conversations()
        if conversation_id in all_conversations:
            del all_conversations[conversation_id]
        save_conversations_index(all_conversations)

        # If pinned conversation, also remove from pinned list
        if conversation_id in session.get('pinned_conversations', []):
            session['pinned_conversations'].remove(conversation_id)
            save_pinned_conversations(session['pinned_conversations'])

        # If deleted conversation is current, clear current state
        if session.get('current_conversation_id') == conversation_id:
            session['current_conversation_id'] = None
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Conversation does not exist'}), 404

@app.route('/pin_conversation/<conversation_id>', methods=['POST'])
def pin_conversation(conversation_id):
    """Pin conversation"""
    init_session()

    # Check if conversation exists from file
    all_conversations = load_all_conversations()
    if conversation_id in all_conversations:
        pinned_list = session.get('pinned_conversations', [])
        if conversation_id not in pinned_list:
            # Add to beginning of pinned list (most recently pinned first)
            pinned_list.insert(0, conversation_id)
            session['pinned_conversations'] = pinned_list
            save_pinned_conversations(pinned_list)
            session.modified = True
        return jsonify({'success': True, 'is_pinned': True})
    else:
        return jsonify({'error': 'Conversation does not exist'}), 404

@app.route('/unpin_conversation/<conversation_id>', methods=['POST'])
def unpin_conversation(conversation_id):
    """Unpin conversation"""
    init_session()
    pinned_list = session.get('pinned_conversations', [])
    if conversation_id in pinned_list:
        pinned_list.remove(conversation_id)
        session['pinned_conversations'] = pinned_list
        save_pinned_conversations(pinned_list)
        session.modified = True
    return jsonify({'success': True, 'is_pinned': False})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear current chat history (backward compatible)"""
    init_session()
    session['current_conversation_id'] = None
    session.modified = True
    return jsonify({'success': True})

@app.route('/get_history')
def get_history():
    """Get current chat history (backward compatible)"""
    init_session()
    current_id = session.get('current_conversation_id')
    history = []
    if current_id:
        conversation = load_conversation_from_file(current_id)
        if conversation:
            history = conversation['messages']

    return jsonify({
        'history': history,
        'current_conversation_id': current_id
    })

@app.route('/get_current_conversation')
def get_current_conversation():
    """Get complete information of current active conversation"""
    init_session()
    current_id = session.get('current_conversation_id')

    # If no current conversation ID, try to auto select latest conversation
    if not current_id:
        all_conversations = load_all_conversations()
        if all_conversations:
            conversations = list(all_conversations.values())
            latest_conversation = max(conversations, key=lambda x: x['updated_at'])
            current_id = latest_conversation['id']
            session['current_conversation_id'] = current_id
            session.modified = True

    if current_id:
        conversation = load_conversation_from_file(current_id)
        if conversation:
            return jsonify({
                'success': True,
                'conversation': conversation,
                'current_conversation_id': current_id
            })

    return jsonify({
        'success': False,
        'conversation': None,
        'current_conversation_id': None
    })

@app.route('/download/<filename>')
def download_file(filename):
    """Download file (CSV, images, etc.)"""
    try:
        # Secure filename check
        safe_filename = secure_filename(filename)
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)

        # Check if file exists
        if not os.path.exists(file_path):
            abort(404)

        # Determine MIME type and download method based on file type
        mimetype, _ = mimetypes.guess_type(file_path)

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Image file: display directly in browser
            return send_file(
                file_path,
                mimetype=mimetype or 'image/png',
                as_attachment=False  # Don't force download, display in browser
            )
        else:
            # Other files (like CSV): download as attachment
            return send_file(
                file_path,
                as_attachment=True,
                download_name=safe_filename,
                mimetype=mimetype or 'text/csv'
            )
    except Exception as e:
        print(f"Download file error: {e}")
        abort(500)

@app.route('/download_upload/<filename>')
def download_upload_file(filename):
    """Download uploaded file"""
    try:
        # Secure filename check
        safe_filename = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        # Check if file exists
        if not os.path.exists(file_path):
            abort(404)

        # Determine MIME type and download method based on file type
        mimetype, _ = mimetypes.guess_type(file_path)

        # Uploaded files usually downloaded as attachment
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype=mimetype or 'application/octet-stream'
        )
    except Exception as e:
        print(f"Download uploaded file error: {e}")
        abort(500)

@app.route('/static_files/<filename>')
def serve_static_file(filename):
    """Provide static file service (specifically for image display)"""
    try:
        # Secure filename check
        safe_filename = secure_filename(filename)
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)

        # Check if file exists
        if not os.path.exists(file_path):
            abort(404)

        # Only allow image files
        if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
            abort(403)

        # Send image file
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        print(f"Static file service error: {e}")
        abort(500)

@app.route('/file_manager')
def file_manager():
    """File management page"""
    try:
        init_session()
        current_conversation_id = session.get('current_conversation_id')

        # Get filter parameters
        filter_type = request.args.get('filter', 'current')  # 'current', 'all'

        files = []

        # Scan downloads folder (files generated by agents)
        if os.path.exists(DOWNLOADS_FOLDER):
            for filename in os.listdir(DOWNLOADS_FOLDER):
                file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                if os.path.isfile(file_path):
                    # Check if file belongs to current conversation
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # Decide whether to include file based on filter type
                    if filter_type == 'current' and not belongs_to_current:
                        continue

                    file_stat = os.stat(file_path)
                    file_info = {
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'image' if filename.endswith(('.png', '.jpg', '.jpeg')) else 'csv',
                        'url': f'/download/{filename}' if filename.endswith('.csv') else f'/static_files/{filename}',
                        'belongs_to_current': belongs_to_current,
                        'source': 'generated'  # Mark as file generated by agent
                    }
                    files.append(file_info)

        # Scan uploads folder (files uploaded by user)
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    # Check if file belongs to current conversation
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # Decide whether to include file based on filter type
                    if filter_type == 'current' and not belongs_to_current:
                        continue

                    file_stat = os.stat(file_path)
                    file_info = {
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'upload',  # Mark as uploaded file
                        'url': f'/download_upload/{filename}',  # Use new download route
                        'belongs_to_current': belongs_to_current,
                        'source': 'uploaded'  # Mark as file uploaded by user
                    }
                    files.append(file_info)

        # Sort by modification time
        files.sort(key=lambda x: x['modified'], reverse=True)

        return render_template('file_manager.html',
                             files=files,
                             current_conversation_id=current_conversation_id,
                             filter_type=filter_type)
    except Exception as e:
        return f"File manager error: {str(e)}", 500

@app.route('/api/files')
def get_files_api():
    """Get file list API"""
    try:
        init_session()
        current_conversation_id = session.get('current_conversation_id')

        # Get filter parameters
        filter_type = request.args.get('filter', 'current')  # 'current', 'all'

        files = []

        # Scan downloads folder (files generated by agents)
        if os.path.exists(DOWNLOADS_FOLDER):
            for filename in os.listdir(DOWNLOADS_FOLDER):
                file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                if os.path.isfile(file_path):
                    # Check if file belongs to current conversation
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # Decide whether to include file based on filter type
                    if filter_type == 'current' and not belongs_to_current:
                        continue

                    file_stat = os.stat(file_path)
                    file_info = {
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'image' if filename.endswith(('.png', '.jpg', '.jpeg')) else 'csv',
                        'url': f'/download/{filename}' if filename.endswith('.csv') else f'/static_files/{filename}',
                        'download_url': f'/download/{filename}',
                        'belongs_to_current': belongs_to_current,
                        'source': 'generated'
                    }
                    files.append(file_info)

        # Scan uploads folder (files uploaded by user)
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    # Check if file belongs to current conversation
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # Decide whether to include file based on filter type
                    if filter_type == 'current' and not belongs_to_current:
                        continue

                    file_stat = os.stat(file_path)
                    file_info = {
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'upload',
                        'url': f'/download_upload/{filename}',
                        'download_url': f'/download_upload/{filename}',
                        'belongs_to_current': belongs_to_current,
                        'source': 'uploaded'
                    }
                    files.append(file_info)

        # Sort by modification time
        files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'files': files,
            'current_conversation_id': current_conversation_id,
            'filter_type': filter_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete file API"""
    try:
        # Secure filename check
        safe_filename = secure_filename(filename)

        # First try to find in downloads folder
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'File {filename} deleted'})

        # If not in downloads, try to find in uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'File {filename} deleted'})

        # File does not exist
        return jsonify({'success': False, 'error': 'File does not exist'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/topology')
def topology_page():
    """Network topology page"""
    return render_template('topology.html')



@app.route('/api/inp_files')
def get_inp_files():
    """Get all uploaded .inp file list"""
    try:
        inp_files = []

        # Scan uploads directory (user uploaded files)
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.endswith('.inp'):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    file_stat = os.stat(file_path)
                    inp_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'source': 'uploaded'
                    })

        # Scan inpfile directory (example files)
        uploaded_filenames = {f['filename'] for f in inp_files}  # Uploaded filename set
        if os.path.exists('inpfile'):
            for filename in os.listdir('inpfile'):
                if filename.endswith('.inp'):
                    # If same name file already exists in uploads, skip example file (prioritize user uploaded files)
                    if filename in uploaded_filenames:
                        continue

                    file_path = os.path.join('inpfile', filename)
                    file_stat = os.stat(file_path)
                    inp_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'source': 'example'
                    })

        # Sort by modification time
        inp_files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'files': inp_files
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/list_downloads')
def list_downloads():
    """List downloadable files"""
    try:
        files = []
        for filename in os.listdir(DOWNLOADS_FOLDER):
            if filename.endswith('.csv'):
                file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(file_path),
                    'created_time': os.path.getctime(file_path)
                }
                files.append(file_info)

        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv_files')
def get_csv_files():
    """Get all CSV file list"""
    try:
        csv_files = []

        if os.path.exists(DOWNLOADS_FOLDER):
            for filename in os.listdir(DOWNLOADS_FOLDER):
                if filename.endswith('.csv'):
                    file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                    file_stat = os.stat(file_path)
                    csv_files.append({
                        'filename': filename,
                        'path': file_path,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })

        # Sort by modification time
        csv_files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'files': csv_files
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/network_topology/<path:inp_file_path>')
def get_network_topology(inp_file_path):
    """Get network topology structure"""
    try:
        # Securely check file path
        if not (inp_file_path.startswith('uploads/') or inp_file_path.startswith('inpfile/')):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400

        if not os.path.exists(inp_file_path):
            return jsonify({'success': False, 'error': 'File does not exist'}), 404

        # Use HydroSim to parse network
        network_info = hydro_sim_agent.parse_network(inp_file_path)

        if 'error' in network_info:
            return jsonify({'success': False, 'error': network_info['error']}), 500

        return jsonify({
            'success': True,
            'topology': network_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/csv_data/<path:csv_file_path>')
def get_csv_data(csv_file_path):
    """Get CSV data"""
    try:
        # Securely check file path
        if not csv_file_path.startswith('downloads/'):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400

        if not os.path.exists(csv_file_path):
            return jsonify({'success': False, 'error': 'File does not exist'}), 404

        # Read CSV data
        import pandas as pd
        df = pd.read_csv(csv_file_path)

        # Get time step list
        time_steps = sorted(df['Time(hours)'].unique()) if 'Time(hours)' in df.columns else [0]

        # Organize data by time and data type
        organized_data = {}
        for time_step in time_steps:
            time_data = df[df['Time(hours)'] == time_step]
            organized_data[str(time_step)] = {
                'node_pressure': {},
                'node_demand': {},
                'link_flow': {},
                'link_velocity': {}
            }

            # Node pressure data
            pressure_data = time_data[time_data['Data type'] == 'Node pressure']
            for _, row in pressure_data.iterrows():
                if pd.notna(row['Node ID']):
                    organized_data[str(time_step)]['node_pressure'][str(row['Node ID'])] = float(row['Value'])

            # Node demand data
            demand_data = time_data[time_data['Data type'] == 'Node demand']
            for _, row in demand_data.iterrows():
                if pd.notna(row['Node ID']):
                    organized_data[str(time_step)]['node_demand'][str(row['Node ID'])] = float(row['Value'])

            # Pipe flow data
            flow_data = time_data[time_data['Data type'] == 'Pipe flow']
            for _, row in flow_data.iterrows():
                if pd.notna(row['Pipe ID']):
                    organized_data[str(time_step)]['link_flow'][str(row['Pipe ID'])] = float(row['Value'])

            # Pipe velocity data
            velocity_data = time_data[time_data['Data type'] == 'Pipe velocity']
            for _, row in velocity_data.iterrows():
                if pd.notna(row['Pipe ID']):
                    organized_data[str(time_step)]['link_velocity'][str(row['Pipe ID'])] = float(row['Value'])

        # Convert to JSON format
        csv_data = {
            'time_steps': time_steps,
            'data_by_time': organized_data,
            'summary': {
                'total_records': len(df),
                'time_steps_count': len(time_steps),
                'data_types': df['Data type'].value_counts().to_dict() if 'Data type' in df.columns else {},
                'time_range': {
                    'min': float(min(time_steps)),
                    'max': float(max(time_steps))
                },
                'nodes_count': len(df[df['Node ID'].notna()]['Node ID'].unique()) if 'Node ID' in df.columns else 0,
                'links_count': len(df[df['Pipe ID'].notna()]['Pipe ID'].unique()) if 'Pipe ID' in df.columns else 0
            }
        }

        return jsonify({
            'success': True,
            'csv_data': csv_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate_compatibility', methods=['POST'])
def validate_compatibility():
    """Validate compatibility between INP and CSV files"""
    try:
        data = request.get_json()
        inp_file_path = data.get('inp_file_path')
        csv_file_path = data.get('csv_file_path')

        if not inp_file_path or not csv_file_path:
            return jsonify({'success': False, 'error': 'Missing file path'}), 400

        # Get network topology info
        network_info = hydro_sim_agent.parse_network(inp_file_path)
        if 'error' in network_info:
            return jsonify({'success': False, 'error': f'Failed to parse INP file: {network_info["error"]}'}), 500

        # Read CSV data
        import pandas as pd
        df = pd.read_csv(csv_file_path)

        # Validate compatibility
        compatibility = {
            'compatible': True,
            'issues': [],
            'network_nodes': network_info['nodes']['total'],
            'network_links': network_info['links']['total'],
            'csv_records': len(df)
        }

        # Check Node ID in CSV
        if 'Node ID' in df.columns:
            csv_nodes = set(df['Node ID'].dropna().astype(str).unique())
            network_nodes = set([node['id'] for node in network_info['topology']['nodes']]) if 'topology' in network_info else set()

            missing_nodes = csv_nodes - network_nodes
            if missing_nodes:
                compatibility['issues'].append(f'CSV contains nodes not existing in network: {list(missing_nodes)[:5]}')
                compatibility['compatible'] = False

        # Check Pipe ID in CSV
        if 'Pipe ID' in df.columns:
            csv_links = set(df['Pipe ID'].dropna().astype(str).unique())
            network_links = set([link['id'] for link in network_info['topology']['links']]) if 'topology' in network_info else set()

            missing_links = csv_links - network_links
            if missing_links:
                compatibility['issues'].append(f'CSV contains pipes not existing in network: {list(missing_links)[:5]}')
                compatibility['compatible'] = False

        return jsonify({
            'success': True,
            'compatibility': compatibility
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Start file cleanup scheduler
    start_file_cleanup_scheduler()

    # Perform file cleanup once at application startup
    cleanup_old_files()

    print("🚀 LeakAgent Web Chat application starting...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Download folder: {DOWNLOADS_FOLDER}")
    print(f"📁 Conversation storage: {CONVERSATIONS_FOLDER}")
    print(f"\ud83d\udd27 File management: Max {MAX_FILES_COUNT} files, {MAX_FOLDER_SIZE/1024/1024:.0f}MB, Retain {FILE_RETENTION_DAYS} days")
    print("🌐 Access URL: http://localhost:5000")
    print("📊 File manager: http://localhost:5000/file_manager")
    print("Press Ctrl+C to stop server")

    app.run(debug=True, host='0.0.0.0', port=5000)
