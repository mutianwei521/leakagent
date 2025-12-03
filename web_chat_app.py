#!/usr/bin/env python3
"""
Web Chat Application with OpenAI API Integration
支持文件上传和文本对话的网页聊天界面
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
app.secret_key = 'your-secret-key-change-this'  # 请更改为安全的密钥

# 配置
UPLOAD_FOLDER = 'uploads'
DOWNLOADS_FOLDER = 'downloads'  # 下载文件夹
CONVERSATIONS_FOLDER = 'conversations'  # 对话存储目录
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv', 'inp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# 文件管理配置
MAX_FILES_COUNT = 100  # 最大文件数量
MAX_FOLDER_SIZE = 500 * 1024 * 1024  # 最大文件夹大小 500MB
FILE_RETENTION_DAYS = 7  # 文件保留天数

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOADS_FOLDER, exist_ok=True)
os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

# 延迟初始化智能体，避免重复的IntentClassifier初始化
hydro_sim_agent = None
partition_sim_agent = None
sensor_placement_agent = None
leak_detection_agent = None
llm_task_analyzer = None
agent_executor = None

def extract_training_parameters(user_message: str, default_scenarios: int, default_epochs: int) -> tuple:
    """智能提取训练参数"""
    import re

    num_scenarios = default_scenarios
    epochs = default_epochs

    # 提取迭代次数/训练轮数的模式
    epoch_patterns = [
        r'迭代次数为?(\d+)次?',
        r'迭代(\d+)次',
        r'训练(\d+)轮',
        r'(\d+)轮训练',
        r'epochs?\s*[=:为]\s*(\d+)',
        r'(\d+)\s*个?epochs?',
        r'训练轮数\s*[=:为]\s*(\d+)',
        r'(\d+)次迭代',
        r'epoch\s*[=:]\s*(\d+)',
        r'轮数\s*[=:为]\s*(\d+)'
    ]

    # 提取样本数/数据组数的模式
    scenario_patterns = [
        r'生成数据为?(\d+)组',
        r'(\d+)组数据',
        r'(\d+)个样本',
        r'(\d+)个场景',
        r'数据量\s*[=:为]\s*(\d+)',
        r'样本数\s*[=:为]\s*(\d+)',
        r'场景数\s*[=:为]\s*(\d+)',
        r'数据\s*(\d+)组',
        r'样本\s*(\d+)个',
        r'场景\s*(\d+)个',
        r'(\d+)\s*个数据',
        r'(\d+)\s*组样本',
        r'(\d+)组',  # 简化模式：直接匹配"1000组"
        r'总样本数\s*[=:为]\s*(\d+)',
        r'样本总数\s*[=:为]\s*(\d+)',
        r'数据总数\s*[=:为]\s*(\d+)',
        r'生成\s*(\d+)\s*组',
        r'训练数据\s*(\d+)\s*组',
        r'(\d+)\s*个训练样本'
    ]

    # 尝试匹配迭代次数
    for pattern in epoch_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            epochs = min(int(match.group(1)), 500)
            print(f"识别到迭代次数: {epochs} (匹配模式: {pattern})")
            break

    # 尝试匹配样本数
    for pattern in scenario_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            num_scenarios = min(int(match.group(1)), 2000)  # 提高最大限制到2000
            print(f"识别到样本数: {num_scenarios} (匹配模式: {pattern})")
            break

    # 如果没有匹配到特定模式，使用原来的简单数字提取作为备用
    if num_scenarios == default_scenarios and epochs == default_epochs:
        numbers = re.findall(r'\d+', user_message)
        if numbers:
            # 如果只有一个数字，根据上下文判断
            if len(numbers) == 1:
                num = int(numbers[0])
                if any(keyword in user_message.lower() for keyword in ['迭代', '轮', 'epoch']):
                    epochs = min(num, 500)
                    print(f"根据上下文识别为迭代次数: {epochs}")
                elif any(keyword in user_message.lower() for keyword in ['数据', '样本', '场景', '组']):
                    num_scenarios = min(num, 2000)  # 提高最大限制到2000
                    print(f"根据上下文识别为样本数: {num_scenarios}")
                else:
                    # 默认第一个数字作为样本数
                    num_scenarios = min(num, 2000)  # 提高最大限制到2000
                    print(f"默认识别为样本数: {num_scenarios}")
            elif len(numbers) >= 2:
                # 多个数字时，按原来的逻辑：第一个作为样本数，第二个作为迭代次数
                num_scenarios = min(int(numbers[0]), 2000)  # 提高最大限制到2000
                epochs = min(int(numbers[1]), 500)
                print(f"多数字模式: 样本数={num_scenarios}, 迭代次数={epochs}")

    return num_scenarios, epochs

def init_agents():
    """延迟初始化智能体"""
    global hydro_sim_agent, partition_sim_agent, sensor_placement_agent, leak_detection_agent
    global llm_task_analyzer, agent_executor

    if hydro_sim_agent is None:
        print("初始化智能体...")
        hydro_sim_agent = HydroSim()
        partition_sim_agent = PartitionSim()
        sensor_placement_agent = SensorPlacement()
        leak_detection_agent = LeakDetectionAgent()

        # 初始化LLM任务分析器和智能体执行器
        llm_task_analyzer = LLMTaskAnalyzer()
        agent_executor = AgentExecutor(
            hydro_sim_agent,
            partition_sim_agent,
            sensor_placement_agent,
            leak_detection_agent
        )
        print("智能体初始化完成")

# OpenAI 配置
openai.api_base = "https://api.chatanywhere.tech"
openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"

# 智能体标准语句映射
AGENT_STANDARD_PHRASES = {
    "管网分析": "分析管网结构和基本信息",
    "管网分区": "把管网划分为指定数量的区域",
    "离群点检测": "检测和剔除管网分区中的离群点",
    "传感器布置": "在管网中优化布置压力监测传感器",
    "韧性分析": "分析传感器布置的韧性和故障检测能力",
    "漏损模型训练": "训练基于机器学习的漏损检测模型",
    "漏损检测": "使用训练好的模型检测管网漏损",
    "水力仿真": "进行管网水力计算和仿真分析",
    "拓扑分析": "分析管网的拓扑结构和连通性"
}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file_content(filepath):
    """读取文件内容"""
    try:
        # 尝试以文本模式读取
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试其他编码
            with open(filepath, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return "无法读取文件内容（可能是二进制文件）"
    except Exception as e:
        return f"读取文件时出错: {str(e)}"

def generate_conversation_title(first_message):
    """根据第一条消息生成对话标题"""
    if not first_message:
        return "新对话"

    # 清理消息内容
    clean_message = re.sub(r'\s+', ' ', first_message.strip())

    # 如果消息太长，截取前30个字符
    if len(clean_message) > 30:
        return clean_message[:30] + "..."

    return clean_message if clean_message else "新对话"

def ensure_conversations_folder():
    """确保对话存储目录存在"""
    os.makedirs(CONVERSATIONS_FOLDER, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
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
    """安全的jsonify函数，处理numpy数据类型"""
    try:
        # 使用自定义编码器序列化数据
        json_str = json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)
        # 创建响应
        response = app.response_class(
            json_str,
            mimetype='application/json'
        )
        response.status_code = status_code
        return response
    except Exception as e:
        # 如果自定义序列化失败，回退到标准jsonify
        print(f"JSON序列化警告: {e}")
        return jsonify({'error': 'JSON序列化失败'}), 500

def save_conversation_to_file(conversation_id, conversation_data):
    """保存单个对话到文件"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"保存对话文件失败: {e}")

def load_conversation_from_file(conversation_id):
    """从文件加载单个对话"""
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载对话文件失败: {e}")
    return None

def save_conversations_index(conversations_dict):
    """保存对话索引"""
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
        print(f"保存对话索引失败: {e}")

def load_all_conversations():
    """从文件加载所有对话"""
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
        print(f"加载对话历史失败: {e}")
        return {}

def delete_conversation_file(conversation_id):
    """删除对话文件"""
    filepath = os.path.join(CONVERSATIONS_FOLDER, f'conversation_{conversation_id}.json')
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"删除对话文件失败: {e}")

def save_pinned_conversations(pinned_list):
    """保存置顶对话列表"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, 'pinned_conversations.json')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'pinned_conversations': pinned_list,
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"保存置顶对话列表失败: {e}")

def load_pinned_conversations():
    """加载置顶对话列表"""
    ensure_conversations_folder()
    filepath = os.path.join(CONVERSATIONS_FOLDER, 'pinned_conversations.json')

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('pinned_conversations', [])
    except Exception as e:
        print(f"加载置顶对话列表失败: {e}")
        return []

def get_inp_file_from_conversation_history(conversation):
    """从对话历史中获取最近的.inp文件路径"""
    for msg in reversed(conversation['messages']):
        if msg.get('file_type') == 'inp' and msg.get('file_path'):
            # 检查文件是否仍然存在
            if os.path.exists(msg['file_path']):
                return msg['file_path']
    return None

def has_inp_file_in_conversation_history(conversation):
    """检查对话历史中是否包含.inp文件"""
    return get_inp_file_from_conversation_history(conversation) is not None

def get_partition_csv_from_conversation_history(conversation):
    """从对话历史中获取最近的分区CSV文件路径"""
    for msg in reversed(conversation['messages']):
        # 检查是否是分区相关的消息且有CSV文件生成
        if (msg.get('intent') == 'partition_analysis' and
            msg.get('csv_info') and
            msg['csv_info'].get('success')):
            csv_path = msg['csv_info']['filepath']
            # 检查文件是否仍然存在
            if os.path.exists(csv_path):
                return csv_path
    return None

def has_partition_csv_in_conversation_history(conversation):
    """检查对话历史中是否包含分区CSV文件"""
    return get_partition_csv_from_conversation_history(conversation) is not None

def cleanup_old_files():
    """清理过期文件"""
    try:
        if not os.path.exists(DOWNLOADS_FOLDER):
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=FILE_RETENTION_DAYS)

        files_info = []
        total_size = 0

        # 收集文件信息
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

        # 按修改时间排序（最旧的在前）
        files_info.sort(key=lambda x: x['mtime'])

        deleted_count = 0

        # 删除过期文件
        for file_info in files_info:
            if file_info['mtime'] < cutoff_time:
                try:
                    os.remove(file_info['path'])
                    deleted_count += 1
                    total_size -= file_info['size']
                    print(f"删除过期文件: {file_info['filename']}")
                except Exception as e:
                    print(f"删除文件失败 {file_info['filename']}: {e}")

        # 如果文件数量仍然过多，删除最旧的文件
        remaining_files = [f for f in files_info if os.path.exists(f['path'])]
        while len(remaining_files) > MAX_FILES_COUNT:
            oldest_file = remaining_files.pop(0)
            try:
                os.remove(oldest_file['path'])
                deleted_count += 1
                total_size -= oldest_file['size']
                print(f"删除多余文件: {oldest_file['filename']}")
            except Exception as e:
                print(f"删除文件失败 {oldest_file['filename']}: {e}")

        # 如果文件夹大小仍然过大，删除最旧的文件
        remaining_files = [f for f in files_info if os.path.exists(f['path'])]
        while total_size > MAX_FOLDER_SIZE and remaining_files:
            oldest_file = remaining_files.pop(0)
            try:
                os.remove(oldest_file['path'])
                deleted_count += 1
                total_size -= oldest_file['size']
                print(f"删除大文件: {oldest_file['filename']}")
            except Exception as e:
                print(f"删除文件失败 {oldest_file['filename']}: {e}")

        if deleted_count > 0:
            print(f"文件清理完成，删除了 {deleted_count} 个文件")

    except Exception as e:
        print(f"文件清理失败: {e}")

def start_file_cleanup_scheduler():
    """启动文件清理调度器"""
    def cleanup_worker():
        while True:
            try:
                cleanup_old_files()
                # 每小时清理一次
                time.sleep(3600)
            except Exception as e:
                print(f"文件清理调度器错误: {e}")
                time.sleep(3600)

    # 启动后台线程
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print("文件清理调度器已启动")

def init_session():
    """初始化会话数据"""
    # 只在session中存储必要的信息，避免session过大
    if 'current_conversation_id' not in session:
        session['current_conversation_id'] = None
    # 不再在session中存储chat_history，改为按需从文件加载
    if 'pinned_conversations' not in session:
        # 从文件加载置顶对话列表
        session['pinned_conversations'] = load_pinned_conversations()
    # 不再在session中存储所有对话，改为按需从文件加载

@app.route('/')
def index():
    """主页"""
    init_session()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        init_session()

        # 获取或创建对话ID
        conversation_id = session.get('current_conversation_id')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            session['current_conversation_id'] = conversation_id
            session.modified = True

        filename = secure_filename(file.filename)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)

        # 添加时间戳和对话ID避免文件名冲突，保持与智能体生成文件的命名一致
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        conversation_prefix = conversation_id[:8]  # 使用对话ID的前8位
        filename = f"uploaded_{conversation_prefix}_{timestamp}_{name}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # 读取文件内容
        content = read_file_content(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'content': content[:2000] + '...' if len(content) > 2000 else content,  # 限制显示长度
            'full_content': content,
            'conversation_id': conversation_id
        })

    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        file_content = data.get('file_content', '')
        conversation_id = data.get('conversation_id', None)

        if not user_message and not file_content:
            return jsonify({'error': '请输入消息或上传文件'}), 400

        init_session()

        # 初始化智能体（延迟初始化）
        init_agents()

        # 如果没有指定对话ID，尝试使用当前活跃的对话
        if not conversation_id:
            conversation_id = session.get('current_conversation_id')

        # 获取或创建对话
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
            # 从文件加载现有对话
            current_conversation = load_conversation_from_file(conversation_id)
            if not current_conversation:
                # 对话不存在，创建新对话
                conversation_id = str(uuid.uuid4())
                current_conversation = {
                    'id': conversation_id,
                    'title': generate_conversation_title(user_message),
                    'messages': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }

        # 设置当前对话
        session['current_conversation_id'] = conversation_id

        # 清除新对话标志（用户已经开始发送消息）
        if 'is_new_conversation' in session:
            del session['is_new_conversation']
            session.modified = True

        # 检查是否是.inp文件（通过文件内容特征判断）
        is_inp_file = False
        inp_file_path = None
        is_csv_file = False
        csv_file_path = None

        if file_content:
            # 检查文件内容是否包含EPANET格式特征
            if ('[JUNCTIONS]' in file_content or '[PIPES]' in file_content or
                '[RESERVOIRS]' in file_content or '[TANKS]' in file_content):
                is_inp_file = True

                # 保存为临时.inp文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                inp_filename = f"temp_network_{conversation_id[:8]}_{timestamp}.inp"
                inp_file_path = os.path.join(UPLOAD_FOLDER, inp_filename)

                with open(inp_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)

            # 检查是否是CSV文件（通过内容特征判断）
            elif (',' in file_content and '\n' in file_content):
                # 简单检查是否像CSV格式
                lines = file_content.strip().split('\n')
                if len(lines) > 1:  # 至少有标题行和数据行
                    is_csv_file = True

                    # 保存为临时CSV文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"sensor_data_{conversation_id[:8]}_{timestamp}.csv"
                    csv_file_path = os.path.join(UPLOAD_FOLDER, csv_filename)

                    with open(csv_file_path, 'w', encoding='utf-8') as f:
                        f.write(file_content)

        # 优先检查CSV推理场景 - 如果上传了CSV文件且用户输入包含推理关键词
        skip_llm_analysis = False
        if is_csv_file and csv_file_path:
            message_lower = user_message.lower()
            inference_keywords = ['推理', '预测', '分析', '检测', '识别']

            if any(keyword in message_lower for keyword in inference_keywords):
                print(f"🎯 检测到CSV推理场景，跳过LLM分析，直接进入推理模式")
                print(f"   - CSV文件: {os.path.basename(csv_file_path)}")
                print(f"   - 用户消息: {user_message}")
                skip_llm_analysis = True

        # 只有在非CSV推理场景下才进行LLM分析
        task_analysis = None
        if not skip_llm_analysis:
            # 新的LLM驱动的任务分析逻辑
            print(f"开始LLM任务分析，用户消息: {user_message}")

            # 使用LLM任务分析器分析用户意图
            task_analysis = llm_task_analyzer.process(
                user_message,
                conversation_id,
                current_conversation.get('messages', [])
            )

            print(f"LLM任务分析结果: {task_analysis}")

        # 如果分析成功且需要执行智能体任务
        if (task_analysis and task_analysis.get('success') and
            task_analysis.get('analysis', {}).get('task_type') in ['single', 'workflow']):

            # 检查前置条件
            prerequisites = task_analysis.get('prerequisites', {})

            # 如果所有前置条件都满足，执行任务
            if prerequisites.get('all_satisfied', False):
                print("前置条件满足，开始执行智能体任务")

                # 使用智能体执行器执行任务
                execution_result = agent_executor.process(
                    task_analysis['execution_plan'],
                    conversation_id,
                    user_message
                )

                if execution_result.get('success'):
                    # 获取LLM生成的响应
                    assistant_message = execution_result['llm_response']

                    # 检查是否有管网分析结果，如果有则添加详细的管网信息
                    execution_results = execution_result.get('execution_results', [])
                    for step_result in execution_results:
                        if (step_result.get('step_name') == '管网分析' and
                            step_result.get('result') and
                            step_result['result'].get('network_info')):
                            network_info = step_result['result']['network_info']
                            network_details = f"""

## 📊 管网详细信息

### 🏗️ 网络结构
- **节点总数**: {network_info['nodes']['total']} 个
  - 接点: {network_info['nodes']['junctions']} 个
  - 水库: {network_info['nodes']['reservoirs']} 个
  - 水塔: {network_info['nodes']['tanks']} 个

- **管段总数**: {network_info['links']['total']} 个
  - 管道: {network_info['links']['pipes']} 个
  - 水泵: {network_info['links']['pumps']} 个
  - 阀门: {network_info['links']['valves']} 个

### 📏 网络参数
- **管网总长度**: {network_info['network_stats']['total_length']:.2f} 米
- **仿真时长**: {network_info['network_stats']['simulation_duration']} 秒
- **水力时间步长**: {network_info['network_stats']['hydraulic_timestep']} 秒
- **模式时间步长**: {network_info['network_stats']['pattern_timestep']} 秒

### 🎯 分析建议
基于以上管网信息，您可以进行以下进一步分析：
- 🔄 **水力仿真**: 计算节点压力和管段流量
- 🗂️ **管网分区**: 将管网划分为管理区域
- 📍 **传感器布置**: 优化监测点位置
- 🔍 **漏损检测**: 训练和应用漏损检测模型
"""
                            assistant_message += network_details
                            break

                    # 收集下载文件信息
                    downloads = []
                    for exec_result in execution_result['execution_results']:
                        if exec_result.get('success') and exec_result.get('result'):
                            agent_result = exec_result['result']

                            # 检查CSV文件
                            if agent_result.get('csv_info') and agent_result['csv_info'].get('success'):
                                downloads.append({
                                    'type': 'csv',
                                    'step': exec_result['step_name'],
                                    'filename': agent_result['csv_info']['filename'],
                                    'url': agent_result['csv_info']['download_url'],
                                    'size': agent_result['csv_info']['file_size']
                                })

                            # 检查可视化图片
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

                            # 检查模型文件（特别处理漏损检测模型训练的文件）
                            if agent_result.get('files'):
                                for file_type, file_info in agent_result['files'].items():
                                    if file_info.get('success'):
                                        # 根据文件类型和扩展名确定下载类型
                                        download_type = file_type
                                        filename = file_info['filename']

                                        # 特殊处理漏损检测模型的文件类型
                                        if exec_result['step_name'] == '漏损模型训练':
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

                    # 保存到对话历史
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': task_analysis['analysis']['standard_phrase'],
                        'confidence': task_analysis['analysis']['confidence'],
                        'task_analysis': task_analysis,
                        'execution_results': execution_result['execution_results']
                    }

                    # 添加下载信息到对话历史
                    if downloads:
                        message_data['downloads'] = downloads

                    # 如果有文件上传，记录文件信息
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

                    # 更新对话标题
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == '新对话':
                        current_conversation['title'] = generate_conversation_title(user_message)

                    # 保存对话
                    save_conversation_to_file(conversation_id, current_conversation)
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # 构建响应数据
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

                    # 添加下载文件信息（如果有）
                    if downloads:
                        response_data['downloads'] = downloads

                    print(f"LLM驱动的任务执行成功，返回响应")
                    return safe_jsonify(response_data)

                else:
                    # 执行失败，使用错误信息
                    error_message = f"任务执行失败: {execution_result.get('error', '未知错误')}"
                    print(f"任务执行失败: {error_message}")

            else:
                # 前置条件不满足，生成提示信息
                missing = prerequisites.get('missing_prerequisites', [])
                missing_text = []

                if 'inp_file' in missing:
                    missing_text.append("管网INP文件")
                if 'partition_csv' in missing:
                    missing_text.append("分区CSV文件（需要先进行管网分区）")
                if 'trained_model' in missing:
                    missing_text.append("训练好的漏损检测模型")

                error_message = f"缺少必要的前置条件: {', '.join(missing_text)}。请先完成相关步骤。"
                print(f"前置条件不满足: {error_message}")

        else:
            # CSV推理场景，直接跳转到简化推理逻辑
            print("🎯 跳过LLM分析，直接进入CSV推理模式")

        # 如果LLM分析失败或不需要执行智能体任务，回退到原有逻辑
        print("回退到原有的智能体处理逻辑")

        # 检查是否需要使用专门的智能体处理
        should_use_partition_sim = False
        should_use_hydro_sim = False
        should_use_sensor_placement = False
        should_use_leak_detection = False
        agent_inp_file_path = None
        partition_csv_path = None

        # 处理CSV文件的漏损检测 - 简化推理模式
        if is_csv_file and csv_file_path:
            print(f"\n" + "="*60)
            print(f"🔍 检测到CSV文件上传，开始智能推理模式...")
            print(f"📂 CSV文件: {os.path.basename(csv_file_path)}")
            print(f"🆔 对话ID: {conversation_id}")
            print(f"💬 用户消息: {user_message}")
            print("="*60)

            # 简化的前置条件检查：只需要训练好的模型文件
            missing_prerequisites = []

            # 检查是否有训练好的模型文件，并进行维度兼容性检查
            model_file_path = None
            if os.path.exists('downloads'):
                # 首先读取CSV文件确定列数
                csv_columns = 0
                try:
                    import pandas as pd
                    df_temp = pd.read_csv(csv_file_path)
                    csv_columns = len(df_temp.columns)
                    print(f"📊 CSV文件列数: {csv_columns}")
                except Exception as e:
                    print(f"⚠️ 无法读取CSV文件列数: {e}")

                model_files = []
                compatible_models = []

                # 查找所有模型文件
                for filename in os.listdir('downloads'):
                    if ('leak_detection_model' in filename and filename.endswith('.pth')):
                        model_files.append(filename)

                print(f"📋 找到 {len(model_files)} 个模型文件，检查兼容性...")

                # 检查模型兼容性
                for model_file in model_files:
                    model_path = os.path.join('downloads', model_file)
                    try:
                        import torch
                        checkpoint = torch.load(model_path, map_location='cpu')

                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint

                        # 查找第一层的权重来确定输入维度
                        first_layer_key = None
                        for key in state_dict.keys():
                            if 'weight' in key and ('fc1' in key or 'linear' in key or '0' in key):
                                first_layer_key = key
                                break

                        if first_layer_key:
                            input_dim = state_dict[first_layer_key].shape[1]
                            print(f"   {model_file}: 输入维度={input_dim}", end="")

                            if csv_columns > 0 and input_dim == csv_columns:
                                compatible_models.append(model_file)
                                print(f" ✅ 兼容")
                            else:
                                print(f" ❌ 不兼容 (需要{input_dim}列，CSV有{csv_columns}列)")
                        else:
                            print(f"   {model_file}: ❓ 无法确定输入维度")

                    except Exception as e:
                        print(f"   {model_file}: ❌ 检查失败: {e}")

                # 选择模型 - 严格基于对话ID匹配
                if compatible_models:
                    selected_model = None
                    conversation_prefix = conversation_id[:8]

                    print(f"🎯 模型选择策略:")
                    print(f"   - 当前对话ID前缀: {conversation_prefix}")
                    print(f"   - 兼容模型数量: {len(compatible_models)}")

                    # 只选择当前对话ID的兼容模型
                    for model in compatible_models:
                        if conversation_prefix in model:
                            selected_model = model
                            print(f"   ✅ 找到当前对话ID的兼容模型: {model}")
                            break

                    # 如果当前对话没有对应模型，直接返回错误
                    if not selected_model:
                        print(f"   ❌ 当前对话ID {conversation_prefix} 没有对应的训练模型")
                        print(f"   💡 可用的兼容模型对话ID:")
                        available_conversation_ids = set()
                        for model in compatible_models:
                            # 提取模型文件中的对话ID
                            parts = model.split('_')
                            if len(parts) >= 4:
                                model_conversation_id = parts[3]
                                available_conversation_ids.add(model_conversation_id)

                        for conv_id in sorted(available_conversation_ids):
                            print(f"     - {conv_id}")

                        missing_prerequisites.append(f"对话ID {conversation_prefix} 对应的漏损检测模型")
                        print(f"   🔧 建议: 请先为当前对话训练漏损检测模型，或使用有对应模型的对话")
                    else:
                        model_file_path = os.path.join('downloads', selected_model)
                        print(f"🏆 最终选择模型: {selected_model}")
                else:
                    missing_prerequisites.append("兼容的漏损检测模型")
                    print(f"❌ 未找到与CSV文件兼容的模型")
                    if model_files:
                        print(f"💡 建议: 使用对应的训练模型或重新训练模型")
            else:
                missing_prerequisites.append("漏损检测模型")
                print(f"❌ downloads目录不存在")

            # 前置条件检查结果
            if missing_prerequisites:
                print(f"❌ 缺少前置条件: {', '.join(missing_prerequisites)}")

                # 检查是否是对话ID不匹配的问题
                conversation_prefix = conversation_id[:8]
                is_conversation_mismatch = any(f"对话ID {conversation_prefix}" in item for item in missing_prerequisites)

                if is_conversation_mismatch:
                    detailed_error = f"""
❌ **无法找到对应的PTH模型文件**

**问题**：当前对话ID `{conversation_prefix}` 没有对应的训练模型

**原因**：推理功能需要使用与当前对话ID匹配的训练模型

**解决方案**：
1. **推荐方案**：请先为当前对话进行漏损模型训练
   - 输入训练指令（如："漏损模型训练，迭代次数为100次，生成数据为50组"）
   - 等待训练完成后再进行推理

2. **替代方案**：使用已有模型的对话进行推理
   - 切换到有对应训练模型的对话
   - 上传相同格式的CSV文件进行推理

**技术说明**：系统采用严格的对话ID匹配策略，确保推理结果的准确性和可追溯性。
"""
                else:
                    detailed_error = f"""
❌ **漏损检测推理失败**

🚫 **缺少必要条件**

{chr(10).join([f'• {item}' for item in missing_prerequisites])}

📋 **解决方案**：
1. **如果没有模型**: 请先完成以下步骤：
   - 上传管网INP文件
   - 进行管网分区分析
   - 进行传感器布置优化
   - 训练漏损检测模型

2. **如果有模型**: 请确认模型文件在downloads目录中

💡 **快速检查**: 在downloads目录中查找类似 `leak_detection_model_{conversation_prefix}_*.pth` 的文件
"""

                return jsonify({
                    'response': detailed_error,
                    'conversation_id': conversation_id,
                    'error': True,
                    'error_type': 'missing_model' if is_conversation_mismatch else 'missing_prerequisites',
                    'missing_prerequisites': missing_prerequisites,
                    'conversation_mismatch': is_conversation_mismatch
                })

            # 前置条件满足，进行简化推理
            if model_file_path:
                try:
                    # 直接进行漏损检测推理，简化流程
                    print(f"🚀 开始漏损检测推理...")
                    print(f"   📂 模型文件: {os.path.basename(model_file_path)}")
                    print(f"   📊 传感器数据: {os.path.basename(csv_file_path)}")

                    # 直接调用推理方法，传递conversation_id以读取分区文件
                    detection_result = leak_detection_agent.detect_leak_from_file(csv_file_path, model_file_path, conversation_id)

                    # 处理推理结果
                    if detection_result.get('success'):
                        results = detection_result.get('results', [])
                        summary = detection_result.get('summary', {})

                        # 打印详细推理结果
                        print("\n" + "="*60)
                        print("🎯 漏损检测推理结果")
                        print("="*60)

                        print(f"📊 推理结果摘要:")
                        print(f"   - 总样本数: {summary.get('total_samples', 0)}")
                        print(f"   - 正常样本: {summary.get('normal_samples', 0)}")
                        print(f"   - 异常样本: {summary.get('anomaly_samples', 0)}")

                        print(f"\n📋 详细推理结果:")
                        for result in results:
                            sample_id = result.get('sample_id', 0)
                            status = result.get('status', 'N/A')
                            partition = result.get('partition', None)
                            confidence = result.get('confidence', 0)

                            if status == '正常':
                                print(f"   样本{sample_id}: {status} (置信度: {confidence:.3f})")
                            else:
                                print(f"   样本{sample_id}: {status} - 分区{partition} (置信度: {confidence:.3f})")

                        # 异常分布统计
                        if summary.get('anomaly_samples', 0) > 0:
                            partition_stats = {}
                            for result in results:
                                if result.get('status') == '异常':
                                    partition = result.get('partition')
                                    if partition:
                                        partition_stats[partition] = partition_stats.get(partition, 0) + 1

                            if partition_stats:
                                print(f"\n⚠️ 异常分布统计:")
                                for partition, count in sorted(partition_stats.items()):
                                    print(f"   - 分区{partition}: {count}个异常样本")

                        print("="*60)

                        # 生成推理结果CSV文件
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        conversation_prefix = conversation_id[:8]
                        inference_result_filename = f"leak_inference_result_{conversation_prefix}_{timestamp}.csv"
                        inference_result_path = os.path.join(DOWNLOADS_FOLDER, inference_result_filename)

                        # 保存推理结果到CSV
                        import pandas as pd
                        results_data = []
                        for result in results:
                            results_data.append({
                                '样本序号': result.get('sample_id', 0),
                                '检测状态': result.get('status', 'N/A'),
                                '异常分区': result.get('partition', '') if result.get('status') == '异常' else '',
                                '置信度': f"{result.get('confidence', 0):.4f}",
                                '置信度百分比': f"{result.get('confidence', 0):.1%}"
                            })

                        df_results = pd.DataFrame(results_data)
                        df_results.to_csv(inference_result_path, index=False, encoding='utf-8-sig')

                        # 添加到下载文件列表
                        download_url = f"/download/{inference_result_filename}"
                        file_size = os.path.getsize(inference_result_path) if os.path.exists(inference_result_path) else 0

                        detection_result['download_files'] = [{
                            'filename': inference_result_filename,
                            'path': inference_result_path,
                            'url': download_url,
                            'download_url': download_url,
                            'type': 'csv',
                            'size': file_size,
                            'description': '漏损检测推理结果',
                            'step': '漏损检测推理',
                            'records_count': len(results)
                        }]

                        print(f"💾 推理结果已保存: {inference_result_filename}")
                        print(f"📁 文件路径: {inference_result_path}")
                        print(f"🔗 下载URL: {download_url}")
                        print(f"📊 文件大小: {file_size} 字节")

                    if detection_result['success']:
                        # 使用智能体生成的专业prompt调用GPT
                        prompt = leak_detection_agent.build_response_prompt(detection_result, user_message, "detection")

                        # 添加推理结果文件信息到prompt中
                        if detection_result.get('download_files'):
                            prompt += f"""

📁 **推理结果文件已生成**

系统已生成详细的推理结果文件，包含每个样本的检测状态、异常分区和置信度信息：

"""
                            for file_info in detection_result.get('download_files', []):
                                prompt += f"• **{file_info['description']}**: `{file_info['filename']}`\n"

                        prompt += f"""

🎯 **推理模式说明**: 系统检测到已有训练好的模型文件，直接进行推理分析，跳过了分区分析、传感器布置、模型训练等步骤，大幅提升了处理效率。

📊 **使用的资源**:
- 模型文件: `{os.path.basename(model_file_path)}`
- 传感器数据: `{os.path.basename(csv_file_path)}`

请基于以上推理结果生成专业的漏损检测分析报告。
"""

                        messages = [
                            {"role": "system", "content": "你是一个专业的给水管网漏损检测专家，具有丰富的异常检测和故障诊断经验。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                        ]

                        # 添加当前对话的历史消息（最近10轮）
                        for msg in current_conversation['messages'][-10:]:
                            messages.append({"role": "user", "content": msg['user']})
                            messages.append({"role": "assistant", "content": msg['assistant']})

                        messages.append({"role": "user", "content": prompt})

                        # 调用OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.7
                        )

                        assistant_message = response.choices[0].message.content

                        # 保存到当前对话
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

                        # 保存到文件
                        save_conversation_to_file(conversation_id, current_conversation)
                        # 更新对话索引
                        all_conversations = load_all_conversations()
                        all_conversations[conversation_id] = current_conversation
                        save_conversations_index(all_conversations)
                        session.modified = True

                        # 构建响应
                        response_data = {
                            'success': True,
                            'response': assistant_message,
                            'conversation_id': conversation_id,
                            'intent': 'leak_detection_inference',
                            'confidence': 0.9,
                            'detection_results': detection_result,
                            'inference_mode': True,
                            'model_used': os.path.basename(model_file_path),
                            'workflow_skipped': ['分区分析', '传感器布置', '模型训练']
                        }

                        # 添加下载文件信息
                        download_files = detection_result.get('download_files', [])
                        if download_files:
                            response_data['downloads'] = download_files
                            print(f"📁 添加下载文件到响应: {len(download_files)} 个文件")
                            for i, file_info in enumerate(download_files):
                                print(f"   文件{i+1}: {file_info.get('filename', 'N/A')}")
                                print(f"     - URL: {file_info.get('url', 'N/A')}")
                                print(f"     - 大小: {file_info.get('size', 'N/A')} 字节")
                        else:
                            print(f"❌ 没有下载文件添加到响应")

                        print(f"📤 返回推理响应，包含 {len(response_data.get('downloads', []))} 个下载文件")
                        return jsonify(response_data)

                    else:
                        # 检测失败
                        full_message = f"漏损检测失败：{detection_result.get('error', '未知错误')}\n\n用户问题：{user_message}"

                except Exception as e:
                    print(f"漏损检测处理错误: {e}")
                    full_message = f"处理传感器数据时出现错误：{str(e)}\n\n用户问题：{user_message}"

            else:
                # 没有找到模型文件
                full_message = f"用户上传了传感器数据CSV文件，但没有找到对应的漏损检测模型。请先训练漏损检测模型。\n\n用户问题：{user_message}"

        # 确定可用的inp文件路径
        elif is_inp_file and inp_file_path:
            # 新上传的.inp文件
            agent_inp_file_path = inp_file_path
        elif current_conversation and has_inp_file_in_conversation_history(current_conversation):
            # 对话历史中有.inp文件
            historical_inp_path = get_inp_file_from_conversation_history(current_conversation)
            if historical_inp_path:
                agent_inp_file_path = historical_inp_path

        # 如果有可用的inp文件，判断使用哪个智能体
        if agent_inp_file_path:
            # 定义关键词
            partition_keywords = ['分区', '聚类', 'FCM', '模糊聚类', 'clustering', 'partition', '区域划分', '管网划分', '离群点']
            sensor_keywords = ['传感器', '监测点', '压力监测', 'sensor', 'monitoring', '韧性', '敏感度', '布置', '优化', '检测点']
            leak_keywords = ['漏损', '泄漏', '漏水', '异常检测', '故障检测', 'leak', 'leakage', '漏损检测', '漏损分析', '训练模型', '漏损模型']

            # 检查是否是漏损检测相关的请求
            if any(keyword in user_message for keyword in leak_keywords):
                should_use_leak_detection = True
            # 检查是否是传感器布置相关的请求
            elif any(keyword in user_message for keyword in sensor_keywords):
                should_use_sensor_placement = True
                # 检查是否有历史分区结果
                if current_conversation:
                    partition_csv_path = get_partition_csv_from_conversation_history(current_conversation)
            # 检查是否是分区相关的请求
            elif any(keyword in user_message for keyword in partition_keywords):
                should_use_partition_sim = True
            else:
                # 默认使用水力仿真智能体
                should_use_hydro_sim = True

        # 使用PartitionSim智能体处理
        if should_use_partition_sim and agent_inp_file_path:
            try:
                partition_result = partition_sim_agent.process(agent_inp_file_path, user_message, conversation_id)

                if partition_result['success']:
                    # 使用智能体生成的专业prompt调用GPT
                    messages = [
                        {"role": "system", "content": "你是一个专业的给水管网分区分析专家，具有丰富的管网聚类和分区优化经验。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # 添加当前对话的历史消息（最近10轮）
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # 使用智能体生成的专业prompt
                    messages.append({"role": "user", "content": partition_result['prompt']})

                    # 调用OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # 保存到当前对话
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': partition_result['intent'],
                        'confidence': partition_result['confidence']
                    }

                    # 如果是新上传的文件，保存文件信息
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path
                        })
                    else:
                        # 使用历史文件的对话
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    current_conversation['messages'].append(message_data)

                    # 更新对话时间
                    current_conversation['updated_at'] = datetime.now().isoformat()

                    # 如果是第一条消息且标题是默认的，更新标题
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == '新对话':
                        current_conversation['title'] = generate_conversation_title(user_message or "管网分区分析")

                    # 限制每个对话的消息数量
                    if len(current_conversation['messages']) > 50:
                        current_conversation['messages'] = current_conversation['messages'][-50:]

                    # 保存到文件
                    save_conversation_to_file(conversation_id, current_conversation)
                    # 更新对话索引
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # 构建响应
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': partition_result['intent'],
                        'confidence': partition_result['confidence'],
                        'partition_info': partition_result.get('partition_info', {})
                    }

                    # 如果有CSV文件生成，添加下载信息
                    if partition_result.get('csv_info') and partition_result['csv_info']['success']:
                        response_data['download'] = {
                            'available': True,
                            'filename': partition_result['csv_info']['filename'],
                            'url': partition_result['csv_info']['download_url'],
                            'size': partition_result['csv_info']['file_size'],
                            'records_count': partition_result['csv_info']['records_count']
                        }

                    # 如果有可视化图像生成，添加显示信息
                    if partition_result.get('visualization'):
                        response_data['visualization'] = {
                            'available': True,
                            'filename': partition_result['visualization']['filename'],
                            'url': f'/static_files/{partition_result["visualization"]["filename"]}',
                            'download_url': f'/download/{partition_result["visualization"]["filename"]}'
                        }

                    return jsonify(response_data)

                else:
                    # 智能体处理失败，使用普通方式处理
                    full_message = f"用户上传了管网文件(.inp格式)，但分区分析时遇到问题：{partition_result.get('response', '未知错误')}\n\n用户问题：{user_message}"

            except Exception as e:
                print(f"PartitionSim智能体处理错误: {e}")
                full_message = f"用户上传了管网文件(.inp格式)，但分区智能体处理时出现错误。\n\n用户问题：{user_message}"

        # 使用SensorPlacement智能体处理
        elif should_use_sensor_placement and agent_inp_file_path:
            try:
                # 如果没有分区结果，先进行自动分区
                if not partition_csv_path:
                    print("没有找到历史分区结果，开始自动分区...")
                    partition_result = partition_sim_agent.process(
                        agent_inp_file_path,
                        "自动分区用于传感器布置，分成6个分区",
                        conversation_id
                    )

                    if partition_result['success'] and partition_result.get('csv_info'):
                        partition_csv_path = partition_result['csv_info']['filepath']
                        print(f"自动分区完成，分区文件: {partition_csv_path}")
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'自动分区失败，无法进行传感器布置: {partition_result.get("response", "未知错误")}'
                        })
                else:
                    print(f"使用历史分区结果: {partition_csv_path}")

                # 进行传感器布置
                sensor_result = sensor_placement_agent.process(
                    agent_inp_file_path,
                    partition_csv_path,
                    user_message,
                    conversation_id
                )

                if sensor_result['success']:
                    # 使用智能体生成的专业prompt调用GPT
                    messages = [
                        {"role": "system", "content": "你是一个专业的给水管网传感器布置专家，具有丰富的压力监测点优化和韧性分析经验。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # 添加当前对话的历史消息（最近10轮）
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # 使用智能体生成的专业prompt
                    messages.append({"role": "user", "content": sensor_result['prompt']})

                    # 调用OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # 保存到当前对话
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': 'sensor_placement',
                        'confidence': 0.9
                    }

                    # 如果是新上传的文件，保存文件信息
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path
                        })
                    else:
                        # 使用历史文件的对话
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    # 添加传感器布置结果信息
                    if sensor_result.get('csv_info'):
                        message_data['csv_info'] = sensor_result['csv_info']
                        print(message_data['csv_info'])
                    # 添加韧性分析结果信息
                    if sensor_result.get('resilience_csv_info'):
                        message_data['resilience_csv_info'] = sensor_result['resilience_csv_info']
                        print(message_data['resilience_csv_info'])
                    current_conversation['messages'].append(message_data)

                    # 保存到文件
                    save_conversation_to_file(conversation_id, current_conversation)
                    # 更新对话索引
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # 构建响应
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': 'sensor_placement',
                        'confidence': 0.9,
                        'sensor_info': sensor_result.get('sensor_info', {})
                    }

                    # 如果有CSV文件生成，添加下载信息
                    if sensor_result.get('csv_info') and sensor_result['csv_info']['success']:
                        response_data['download'] = {
                            'available': True,
                            'filename': sensor_result['csv_info']['filename'],
                            'url': sensor_result['csv_info']['download_url'],
                            'size': sensor_result['csv_info']['file_size'],
                            'sensor_count': sensor_result['csv_info']['sensor_count']
                        }

                    # 如果有韧性分析文件生成，添加下载信息
                    if sensor_result.get('resilience_csv_info'):
                        response_data['resilience_download'] = {
                            'available': True,
                            'filename': os.path.basename(sensor_result['resilience_csv_info']),
                            'url': f'/download/{os.path.basename(sensor_result["resilience_csv_info"])}'
                        }

                    # 如果有可视化图像生成，添加显示信息
                    if sensor_result.get('visualization'):
                        response_data['visualization'] = {
                            'available': True,
                            'filename': sensor_result['visualization']['filename'],
                            'url': f'/static_files/{sensor_result["visualization"]["filename"]}',
                            'download_url': f'/download/{sensor_result["visualization"]["filename"]}'
                        }

                    return jsonify(response_data)

                else:
                    # 智能体处理失败，使用普通方式处理
                    full_message = f"用户上传了管网文件(.inp格式)，但传感器布置时遇到问题：{sensor_result.get('response', '未知错误')}\n\n用户问题：{user_message}"

            except Exception as e:
                print(f"SensorPlacement智能体处理错误: {e}")
                full_message = f"用户上传了管网文件(.inp格式)，但传感器布置智能体处理时出现错误。\n\n用户问题：{user_message}"

        # 使用LeakDetectionAgent智能体处理
        elif should_use_leak_detection and agent_inp_file_path:
            try:
                # 检查是否是训练请求
                training_keywords = ['训练', '模型', 'train', 'model', '机器学习', 'AI', '学习']
                is_training_request = any(keyword in user_message for keyword in training_keywords)

                if is_training_request:
                    # 训练漏损检测模型
                    print("开始训练漏损检测模型...")

                    # 提取训练参数
                    num_scenarios = 50  # 默认值
                    epochs = 100  # 默认值

                    # 智能提取训练参数
                    import re
                    num_scenarios, epochs = extract_training_parameters(user_message, num_scenarios, epochs)
                    print(f"解析的训练参数: 样本数={num_scenarios}, 迭代次数={epochs}")

                    leak_result = leak_detection_agent.train_leak_detection_model(
                        agent_inp_file_path,
                        conversation_id,
                        num_scenarios,
                        epochs
                    )

                    if leak_result['success']:
                        # 使用智能体生成的专业prompt调用GPT
                        prompt = leak_detection_agent.build_response_prompt(leak_result, user_message, "training")

                        messages = [
                            {"role": "system", "content": "你是一个专业的给水管网漏损检测专家，具有丰富的机器学习和异常检测经验。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                        ]

                        # 添加当前对话的历史消息（最近10轮）
                        for msg in current_conversation['messages'][-10:]:
                            messages.append({"role": "user", "content": msg['user']})
                            messages.append({"role": "assistant", "content": msg['assistant']})

                        messages.append({"role": "user", "content": prompt})

                        # 调用OpenAI API
                        response = openai.ChatCompletion.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            max_tokens=4000,
                            temperature=0.7
                        )

                        assistant_message = response.choices[0].message.content

                        # 保存到当前对话
                        message_data = {
                            'user': user_message,
                            'assistant': assistant_message,
                            'timestamp': datetime.now().isoformat(),
                            'intent': 'leak_detection_training',
                            'confidence': 0.9
                        }

                        # 如果是新上传的文件，保存文件信息
                        if is_inp_file and inp_file_path:
                            message_data.update({
                                'has_file': True,
                                'file_type': 'inp',
                                'file_path': inp_file_path
                            })
                        else:
                            # 使用历史文件的对话
                            message_data.update({
                                'has_file': False,
                                'uses_historical_file': True,
                                'historical_file_path': agent_inp_file_path
                            })

                        # 添加训练结果信息
                        if leak_result.get('files'):
                            message_data['leak_training_files'] = leak_result['files']

                        current_conversation['messages'].append(message_data)

                        # 保存到文件
                        save_conversation_to_file(conversation_id, current_conversation)
                        # 更新对话索引
                        all_conversations = load_all_conversations()
                        all_conversations[conversation_id] = current_conversation
                        save_conversations_index(all_conversations)
                        session.modified = True

                        # 构建响应
                        response_data = {
                            'success': True,
                            'response': assistant_message,
                            'conversation_id': conversation_id,
                            'intent': 'leak_detection_training',
                            'confidence': 0.9,
                            'model_info': leak_result.get('model_info', {}),
                            'evaluation': leak_result.get('evaluation', {})
                        }

                        # 添加下载信息
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
                        # 训练失败，使用普通方式处理
                        full_message = f"漏损检测模型训练失败：{leak_result.get('error', '未知错误')}\n\n用户问题：{user_message}"

                else:
                    # 检测请求 - 需要上传的传感器数据文件
                    full_message = f"用户想要进行漏损检测。请提醒用户需要：\n1. 先训练漏损检测模型\n2. 上传传感器压力数据CSV文件进行检测\n\n用户问题：{user_message}"

            except Exception as e:
                print(f"LeakDetectionAgent智能体处理错误: {e}")
                full_message = f"用户上传了管网文件(.inp格式)，但漏损检测智能体处理时出现错误。\n\n用户问题：{user_message}"

        # 使用HydroSim智能体处理
        elif should_use_hydro_sim and agent_inp_file_path:
            try:
                hydro_result = hydro_sim_agent.process(agent_inp_file_path, user_message, conversation_id)

                if hydro_result['success']:
                    # 使用智能体生成的prompt调用GPT
                    messages = [
                        {"role": "system", "content": "你是一个专业的给水管网分析专家，具有丰富的水力计算和管网分析经验。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
                    ]

                    # 添加当前对话的历史消息（最近10轮）
                    for msg in current_conversation['messages'][-10:]:
                        messages.append({"role": "user", "content": msg['user']})
                        messages.append({"role": "assistant", "content": msg['assistant']})

                    # 添加智能体生成的prompt
                    messages.append({"role": "user", "content": hydro_result['prompt']})

                    # 调用OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    )

                    assistant_message = response.choices[0].message.content

                    # 调试信息
                    print(f"🔍 调试: hydro_result keys: {list(hydro_result.keys())}")
                    print(f"🔍 调试: network_info存在: {hydro_result.get('network_info') is not None}")
                    if hydro_result.get('network_info'):
                        print(f"🔍 调试: network_info类型: {type(hydro_result['network_info'])}")

                    # 如果有管网信息，在回复后添加详细的管网信息
                    if hydro_result.get('network_info'):
                        network_info = hydro_result['network_info']
                        network_details = f"""

## 📊 管网详细信息

### 🏗️ 网络结构
- **节点总数**: {network_info['nodes']['total']} 个
  - 接点: {network_info['nodes']['junctions']} 个
  - 水库: {network_info['nodes']['reservoirs']} 个
  - 水塔: {network_info['nodes']['tanks']} 个

- **管段总数**: {network_info['links']['total']} 个
  - 管道: {network_info['links']['pipes']} 个
  - 水泵: {network_info['links']['pumps']} 个
  - 阀门: {network_info['links']['valves']} 个

### 📏 网络参数
- **管网总长度**: {network_info['network_stats']['total_length']:.2f} 米
- **仿真时长**: {network_info['network_stats']['simulation_duration']} 秒
- **水力时间步长**: {network_info['network_stats']['hydraulic_timestep']} 秒
- **模式时间步长**: {network_info['network_stats']['pattern_timestep']} 秒

### 🎯 分析建议
基于以上管网信息，您可以进行以下进一步分析：
- 🔄 **水力仿真**: 计算节点压力和管段流量
- 🗂️ **管网分区**: 将管网划分为管理区域
- 📍 **传感器布置**: 优化监测点位置
- 🔍 **漏损检测**: 训练和应用漏损检测模型
"""
                        assistant_message += network_details

                    # 保存到当前对话
                    message_data = {
                        'user': user_message,
                        'assistant': assistant_message,
                        'timestamp': datetime.now().isoformat(),
                        'intent': hydro_result['intent'],
                        'confidence': hydro_result['confidence']
                    }

                    # 如果是新上传的文件，保存文件信息
                    if is_inp_file and inp_file_path:
                        message_data.update({
                            'has_file': True,
                            'file_type': 'inp',
                            'file_path': inp_file_path  # 使用原始的inp_file_path
                        })
                    else:
                        # 使用历史文件的对话
                        message_data.update({
                            'has_file': False,
                            'uses_historical_file': True,
                            'historical_file_path': agent_inp_file_path
                        })

                    current_conversation['messages'].append(message_data)

                    # 更新对话时间
                    current_conversation['updated_at'] = datetime.now().isoformat()

                    # 如果是第一条消息且标题是默认的，更新标题
                    if len(current_conversation['messages']) == 1 and current_conversation['title'] == '新对话':
                        current_conversation['title'] = generate_conversation_title(user_message or "管网分析")

                    # 限制每个对话的消息数量
                    if len(current_conversation['messages']) > 50:
                        current_conversation['messages'] = current_conversation['messages'][-50:]

                    # 保存到文件
                    save_conversation_to_file(conversation_id, current_conversation)
                    # 更新对话索引
                    all_conversations = load_all_conversations()
                    all_conversations[conversation_id] = current_conversation
                    save_conversations_index(all_conversations)
                    session.modified = True

                    # 构建响应
                    response_data = {
                        'success': True,
                        'response': assistant_message,
                        'conversation_id': conversation_id,
                        'intent': hydro_result['intent'],
                        'confidence': hydro_result['confidence'],
                        'network_info': hydro_result['network_info']
                    }

                    # 如果有CSV文件生成，添加下载信息
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
                    # 智能体处理失败，使用普通方式处理
                    full_message = f"用户上传了管网文件(.inp格式)，但处理时遇到问题：{hydro_result.get('response', '未知错误')}\n\n用户问题：{user_message}"

            except Exception as e:
                print(f"HydroSim智能体处理错误: {e}")
                full_message = f"用户上传了管网文件(.inp格式)，但智能体处理时出现错误。\n\n用户问题：{user_message}"

        else:
            # 普通文件处理
            full_message = user_message
            if file_content:
                full_message = f"用户上传了文件内容：\n\n{file_content}\n\n用户问题：{user_message}" if user_message else f"用户上传了文件内容：\n\n{file_content}\n\n请分析这个文件的内容。"
        
        # 构建消息历史（OpenAI格式）
        messages = [
            {"role": "system", "content": "你是一个基于GPT-4的高级AI助手，具有强大的分析和推理能力。你可以：\n1. 深入分析各种文件内容（代码、文档、数据等）\n2. 提供专业的技术建议和解决方案\n3. 进行复杂的逻辑推理和问题解决\n4. 支持多语言交流，但请优先使用中文回答\n5. 提供详细、准确、有用的回答\n\n请根据用户的具体需求提供最佳的帮助。请在回复的最后使用以下签名格式：\n\n祝好，\n\nTianwei Mu\nGuangzhou Institute of Industrial Intelligence"}
        ]

        # 添加当前对话的历史消息（最近10轮）
        for msg in current_conversation['messages'][-10:]:
            messages.append({"role": "user", "content": msg['user']})
            messages.append({"role": "assistant", "content": msg['assistant']})

        # 添加当前用户消息
        messages.append({"role": "user", "content": full_message})
        
        # 调用OpenAI API - 使用GPT-4最新版本
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",  # GPT-4 Turbo最新版本，更快更强
            messages=messages,
            max_tokens=4000,  # GPT-4支持更长的输出
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content

        # 保存到当前对话
        current_conversation['messages'].append({
            'user': user_message,
            'assistant': assistant_message,
            'timestamp': datetime.now().isoformat(),
            'has_file': bool(file_content)
        })

        # 更新对话时间
        current_conversation['updated_at'] = datetime.now().isoformat()

        # 如果是第一条消息且标题是默认的，更新标题
        if len(current_conversation['messages']) == 1 and current_conversation['title'] == '新对话':
            current_conversation['title'] = generate_conversation_title(user_message)

        # 限制每个对话的消息数量
        if len(current_conversation['messages']) > 50:
            current_conversation['messages'] = current_conversation['messages'][-50:]

        # 保存到文件
        save_conversation_to_file(conversation_id, current_conversation)
        # 更新对话索引
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
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    """创建新对话"""
    init_session()
    session['current_conversation_id'] = None
    session['is_new_conversation'] = True  # 标记用户主动创建新对话
    session.modified = True
    return jsonify({'success': True})

@app.route('/get_conversations')
def get_conversations():
    """获取所有对话列表"""
    init_session()
    # 从文件加载所有对话
    all_conversations = load_all_conversations()
    conversations = list(all_conversations.values())
    pinned_ids = session.get('pinned_conversations', [])

    # 分离置顶和普通对话
    pinned_conversations = []
    regular_conversations = []

    for conv in conversations:
        # 添加置顶标记
        conv['is_pinned'] = conv['id'] in pinned_ids
        if conv['is_pinned']:
            pinned_conversations.append(conv)
        else:
            regular_conversations.append(conv)

    # 置顶对话按置顶时间排序（最新置顶的在前）
    pinned_conversations.sort(key=lambda x: pinned_ids.index(x['id']) if x['id'] in pinned_ids else 999)
    # 普通对话按更新时间排序，最新的在前
    regular_conversations.sort(key=lambda x: x['updated_at'], reverse=True)

    # 如果没有当前对话ID，但有对话存在，自动选择最近的对话
    # 但是如果用户主动创建了新对话，则不自动选择
    current_conversation_id = session.get('current_conversation_id')
    is_new_conversation = session.get('is_new_conversation', False)

    if not current_conversation_id and conversations and not is_new_conversation:
        # 选择最近更新的对话（不区分置顶和普通）
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
    """加载指定对话"""
    init_session()
    # 从文件加载对话
    conversation = load_conversation_from_file(conversation_id)
    if conversation:
        session['current_conversation_id'] = conversation_id
        session.modified = True
        return jsonify({
            'success': True,
            'conversation': conversation
        })
    else:
        return jsonify({'error': '对话不存在'}), 404

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """删除指定对话"""
    init_session()
    # 检查对话是否存在
    conversation = load_conversation_from_file(conversation_id)
    if conversation:
        # 删除文件
        delete_conversation_file(conversation_id)
        # 更新索引
        all_conversations = load_all_conversations()
        if conversation_id in all_conversations:
            del all_conversations[conversation_id]
        save_conversations_index(all_conversations)

        # 如果是置顶对话，也要从置顶列表中移除
        if conversation_id in session.get('pinned_conversations', []):
            session['pinned_conversations'].remove(conversation_id)
            save_pinned_conversations(session['pinned_conversations'])

        # 如果删除的是当前对话，清空当前状态
        if session.get('current_conversation_id') == conversation_id:
            session['current_conversation_id'] = None
        session.modified = True
        return jsonify({'success': True})
    else:
        return jsonify({'error': '对话不存在'}), 404

@app.route('/pin_conversation/<conversation_id>', methods=['POST'])
def pin_conversation(conversation_id):
    """置顶对话"""
    init_session()

    # 从文件检查对话是否存在
    all_conversations = load_all_conversations()
    if conversation_id in all_conversations:
        pinned_list = session.get('pinned_conversations', [])
        if conversation_id not in pinned_list:
            # 添加到置顶列表的开头（最新置顶的在前）
            pinned_list.insert(0, conversation_id)
            session['pinned_conversations'] = pinned_list
            save_pinned_conversations(pinned_list)
            session.modified = True
        return jsonify({'success': True, 'is_pinned': True})
    else:
        return jsonify({'error': '对话不存在'}), 404

@app.route('/unpin_conversation/<conversation_id>', methods=['POST'])
def unpin_conversation(conversation_id):
    """取消置顶对话"""
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
    """清除当前聊天历史（保持向后兼容）"""
    init_session()
    session['current_conversation_id'] = None
    session.modified = True
    return jsonify({'success': True})

@app.route('/get_history')
def get_history():
    """获取当前聊天历史（保持向后兼容）"""
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
    """获取当前活跃对话的完整信息"""
    init_session()
    current_id = session.get('current_conversation_id')

    # 如果没有当前对话ID，尝试自动选择最近的对话
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
    """下载文件（CSV、图片等）"""
    try:
        # 安全文件名检查
        safe_filename = secure_filename(filename)
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            abort(404)

        # 根据文件类型确定MIME类型和下载方式
        mimetype, _ = mimetypes.guess_type(file_path)

        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # 图片文件：直接在浏览器中显示
            return send_file(
                file_path,
                mimetype=mimetype or 'image/png',
                as_attachment=False  # 不强制下载，在浏览器中显示
            )
        else:
            # 其他文件（如CSV）：作为附件下载
            return send_file(
                file_path,
                as_attachment=True,
                download_name=safe_filename,
                mimetype=mimetype or 'text/csv'
            )
    except Exception as e:
        print(f"下载文件错误: {e}")
        abort(500)

@app.route('/download_upload/<filename>')
def download_upload_file(filename):
    """下载上传的文件"""
    try:
        # 安全文件名检查
        safe_filename = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            abort(404)

        # 根据文件类型确定MIME类型和下载方式
        mimetype, _ = mimetypes.guess_type(file_path)

        # 上传的文件通常作为附件下载
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype=mimetype or 'application/octet-stream'
        )
    except Exception as e:
        print(f"下载上传文件错误: {e}")
        abort(500)

@app.route('/static_files/<filename>')
def serve_static_file(filename):
    """提供静态文件服务（专门用于图片显示）"""
    try:
        # 安全文件名检查
        safe_filename = secure_filename(filename)
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            abort(404)

        # 只允许图片文件
        if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
            abort(403)

        # 发送图片文件
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        print(f"静态文件服务错误: {e}")
        abort(500)

@app.route('/file_manager')
def file_manager():
    """文件管理页面"""
    try:
        init_session()
        current_conversation_id = session.get('current_conversation_id')

        # 获取过滤参数
        filter_type = request.args.get('filter', 'current')  # 'current', 'all'

        files = []

        # 扫描下载文件夹（智能体生成的文件）
        if os.path.exists(DOWNLOADS_FOLDER):
            for filename in os.listdir(DOWNLOADS_FOLDER):
                file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                if os.path.isfile(file_path):
                    # 检查文件是否属于当前对话
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # 根据过滤类型决定是否包含文件
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
                        'source': 'generated'  # 标记为智能体生成的文件
                    }
                    files.append(file_info)

        # 扫描上传文件夹（用户上传的文件）
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    # 检查文件是否属于当前对话
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # 根据过滤类型决定是否包含文件
                    if filter_type == 'current' and not belongs_to_current:
                        continue

                    file_stat = os.stat(file_path)
                    file_info = {
                        'filename': filename,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'upload',  # 标记为上传文件
                        'url': f'/download_upload/{filename}',  # 使用新的下载路由
                        'belongs_to_current': belongs_to_current,
                        'source': 'uploaded'  # 标记为用户上传的文件
                    }
                    files.append(file_info)

        # 按修改时间排序
        files.sort(key=lambda x: x['modified'], reverse=True)

        return render_template('file_manager.html',
                             files=files,
                             current_conversation_id=current_conversation_id,
                             filter_type=filter_type)
    except Exception as e:
        return f"文件管理器错误: {str(e)}", 500

@app.route('/api/files')
def get_files_api():
    """获取文件列表API"""
    try:
        init_session()
        current_conversation_id = session.get('current_conversation_id')

        # 获取过滤参数
        filter_type = request.args.get('filter', 'current')  # 'current', 'all'

        files = []

        # 扫描下载文件夹（智能体生成的文件）
        if os.path.exists(DOWNLOADS_FOLDER):
            for filename in os.listdir(DOWNLOADS_FOLDER):
                file_path = os.path.join(DOWNLOADS_FOLDER, filename)
                if os.path.isfile(file_path):
                    # 检查文件是否属于当前对话
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # 根据过滤类型决定是否包含文件
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

        # 扫描上传文件夹（用户上传的文件）
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    # 检查文件是否属于当前对话
                    belongs_to_current = False
                    if current_conversation_id:
                        conversation_prefix = current_conversation_id[:8]
                        belongs_to_current = conversation_prefix in filename

                    # 根过滤类型决定是否包含文件
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

        # 按修改时间排序
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
    """删除文件API"""
    try:
        # 安全文件名检查
        safe_filename = secure_filename(filename)

        # 先尝试在downloads文件夹中查找
        file_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'文件 {filename} 已删除'})

        # 如果downloads中没有，尝试在uploads文件夹中查找
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': f'文件 {filename} 已删除'})

        # 文件不存在
        return jsonify({'success': False, 'error': '文件不存在'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/topology')
def topology_page():
    """管网拓扑图页面"""
    return render_template('topology.html')



@app.route('/api/inp_files')
def get_inp_files():
    """获取所有上传的.inp文件列表"""
    try:
        inp_files = []

        # 扫描uploads目录（用户上传文件）
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

        # 扫描inpfile目录（示例文件）
        uploaded_filenames = {f['filename'] for f in inp_files}  # 已上传的文件名集合
        if os.path.exists('inpfile'):
            for filename in os.listdir('inpfile'):
                if filename.endswith('.inp'):
                    # 如果同名文件已经在uploads中存在，跳过示例文件（优先显示用户上传的文件）
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

        # 按修改时间排序
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
    """列出可下载的文件"""
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
    """获取所有CSV文件列表"""
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

        # 按修改时间排序
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
    """获取管网拓扑结构"""
    try:
        # 安全检查文件路径
        if not (inp_file_path.startswith('uploads/') or inp_file_path.startswith('inpfile/')):
            return jsonify({'success': False, 'error': '无效的文件路径'}), 400

        if not os.path.exists(inp_file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404

        # 使用HydroSim解析网络
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
    """获取CSV数据"""
    try:
        # 安全检查文件路径
        if not csv_file_path.startswith('downloads/'):
            return jsonify({'success': False, 'error': '无效的文件路径'}), 400

        if not os.path.exists(csv_file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404

        # 读取CSV数据
        import pandas as pd
        df = pd.read_csv(csv_file_path)

        # 获取时间步长列表
        time_steps = sorted(df['时间(小时)'].unique()) if '时间(小时)' in df.columns else [0]

        # 按时间和数据类型组织数据
        organized_data = {}
        for time_step in time_steps:
            time_data = df[df['时间(小时)'] == time_step]
            organized_data[str(time_step)] = {
                'node_pressure': {},
                'node_demand': {},
                'link_flow': {},
                'link_velocity': {}
            }

            # 节点压力数据
            pressure_data = time_data[time_data['数据类型'] == '节点压力']
            for _, row in pressure_data.iterrows():
                if pd.notna(row['节点ID']):
                    organized_data[str(time_step)]['node_pressure'][str(row['节点ID'])] = float(row['数值'])

            # 节点需水量数据
            demand_data = time_data[time_data['数据类型'] == '节点需水量']
            for _, row in demand_data.iterrows():
                if pd.notna(row['节点ID']):
                    organized_data[str(time_step)]['node_demand'][str(row['节点ID'])] = float(row['数值'])

            # 管段流量数据
            flow_data = time_data[time_data['数据类型'] == '管段流量']
            for _, row in flow_data.iterrows():
                if pd.notna(row['管段ID']):
                    organized_data[str(time_step)]['link_flow'][str(row['管段ID'])] = float(row['数值'])

            # 管段流速数据
            velocity_data = time_data[time_data['数据类型'] == '管段流速']
            for _, row in velocity_data.iterrows():
                if pd.notna(row['管段ID']):
                    organized_data[str(time_step)]['link_velocity'][str(row['管段ID'])] = float(row['数值'])

        # 转换为JSON格式
        csv_data = {
            'time_steps': time_steps,
            'data_by_time': organized_data,
            'summary': {
                'total_records': len(df),
                'time_steps_count': len(time_steps),
                'data_types': df['数据类型'].value_counts().to_dict() if '数据类型' in df.columns else {},
                'time_range': {
                    'min': float(min(time_steps)),
                    'max': float(max(time_steps))
                },
                'nodes_count': len(df[df['节点ID'].notna()]['节点ID'].unique()) if '节点ID' in df.columns else 0,
                'links_count': len(df[df['管段ID'].notna()]['管段ID'].unique()) if '管段ID' in df.columns else 0
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
    """验证INP文件和CSV文件的兼容性"""
    try:
        data = request.get_json()
        inp_file_path = data.get('inp_file_path')
        csv_file_path = data.get('csv_file_path')

        if not inp_file_path or not csv_file_path:
            return jsonify({'success': False, 'error': '缺少文件路径'}), 400

        # 获取网络拓扑信息
        network_info = hydro_sim_agent.parse_network(inp_file_path)
        if 'error' in network_info:
            return jsonify({'success': False, 'error': f'解析INP文件失败: {network_info["error"]}'}), 500

        # 读取CSV数据
        import pandas as pd
        df = pd.read_csv(csv_file_path)

        # 验证兼容性
        compatibility = {
            'compatible': True,
            'issues': [],
            'network_nodes': network_info['nodes']['total'],
            'network_links': network_info['links']['total'],
            'csv_records': len(df)
        }

        # 检查CSV中的节点ID
        if '节点ID' in df.columns:
            csv_nodes = set(df['节点ID'].dropna().astype(str).unique())
            network_nodes = set([node['id'] for node in network_info['topology']['nodes']]) if 'topology' in network_info else set()

            missing_nodes = csv_nodes - network_nodes
            if missing_nodes:
                compatibility['issues'].append(f'CSV中包含网络中不存在的节点: {list(missing_nodes)[:5]}')
                compatibility['compatible'] = False

        # 检查CSV中的管段ID
        if '管段ID' in df.columns:
            csv_links = set(df['管段ID'].dropna().astype(str).unique())
            network_links = set([link['id'] for link in network_info['topology']['links']]) if 'topology' in network_info else set()

            missing_links = csv_links - network_links
            if missing_links:
                compatibility['issues'].append(f'CSV中包含网络中不存在的管段: {list(missing_links)[:5]}')
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
    # 启动文件清理调度器
    start_file_cleanup_scheduler()

    # 启动应用时进行一次文件清理
    cleanup_old_files()

    print("🚀 LeakAgent Web Chat 应用启动中...")
    print(f"📁 上传文件夹: {UPLOAD_FOLDER}")
    print(f"📁 下载文件夹: {DOWNLOADS_FOLDER}")
    print(f"📁 对话存储: {CONVERSATIONS_FOLDER}")
    print(f"🔧 文件管理: 最大{MAX_FILES_COUNT}个文件, {MAX_FOLDER_SIZE/1024/1024:.0f}MB, 保留{FILE_RETENTION_DAYS}天")
    print("🌐 访问地址: http://localhost:5000")
    print("📊 文件管理器: http://localhost:5000/file_manager")
    print("按 Ctrl+C 停止服务器")

    app.run(debug=True, host='0.0.0.0', port=5000)
