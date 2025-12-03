"""
LLM任务分析器
负责将自然语言转换为智能体标准语句，并分析任务步骤
"""
import os
import json
import openai
from datetime import datetime
from .base_agent import BaseAgent

class LLMTaskAnalyzer(BaseAgent):
    """基于LLM的任务分析器"""
    
    def __init__(self):
        super().__init__("LLMTaskAnalyzer")
        
        # 设置OpenAI API配置
        openai.api_base = "https://api.chatanywhere.tech"
        openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"
        
        # 智能体标准语句映射
        self.standard_phrases = {
            "管网分析": "分析管网结构和基本信息",
            "管网分区": "把管网划分为指定数量的区域",
            "离群点检测": "检测和剔除管网分区中的离群点",
            "传感器布置": "在管网中优化布置压力监测传感器",
            "韧性分析": "分析传感器布置的韧性和故障检测能力",
            "漏损模型训练": "训练基于机器学习的漏损检测模型",
            "漏损检测": "使用训练好的模型检测管网漏损",
            "模型推理": "使用已训练的模型对传感器数据进行直接推理分析",
            "水力仿真": "进行管网水力计算和仿真分析",
            "拓扑分析": "分析管网的拓扑结构和连通性"
        }
        
        # 工作流程步骤定义
        self.workflow_steps = {
            "complete_workflow": [
                "管网分析",
                "管网分区", 
                "离群点检测",
                "传感器布置",
                "漏损模型训练"
            ],
            "sensor_placement_workflow": [
                "管网分析",
                "管网分区",
                "传感器布置"
            ],
            "leak_detection_workflow": [
                "管网分析",
                "管网分区",
                "传感器布置", 
                "漏损模型训练",
                "漏损检测"
            ]
        }
    
    def analyze_user_intent(self, user_message: str, conversation_history: list = None):
        """分析用户意图并转换为标准语句"""
        try:
            # 构建分析prompt
            analysis_prompt = self._build_analysis_prompt(user_message, conversation_history)
            
            # 调用LLM进行分析
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的给水管网智能体任务分析专家。你需要将用户的自然语言转换为标准的智能体语句，并分析任务步骤。"
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # 解析LLM响应
            llm_response = response.choices[0].message.content
            return self._parse_llm_response(llm_response, user_message)
            
        except Exception as e:
            self.log_error(f"LLM任务分析失败: {e}")
            return self._fallback_analysis(user_message)
    
    def _build_analysis_prompt(self, user_message: str, conversation_history: list = None):
        """构建分析prompt"""
        
        # 标准语句列表
        phrases_text = "\n".join([f"- {key}: {value}" for key, value in self.standard_phrases.items()])
        
        # 工作流程说明
        workflow_text = ""
        for workflow_name, steps in self.workflow_steps.items():
            workflow_text += f"\n{workflow_name}: {' -> '.join(steps)}"
        
        # 对话历史摘要
        history_text = ""
        if conversation_history:
            recent_messages = conversation_history[-5:]  # 最近5条消息
            history_text = "\n对话历史摘要:\n"
            for i, msg in enumerate(recent_messages):
                history_text += f"{i+1}. 用户: {msg.get('user', '')[:100]}...\n"
                history_text += f"   助手: {msg.get('assistant', '')[:100]}...\n"
        
        prompt = f"""
请分析用户的自然语言请求，并完成以下任务：

1. 将用户请求转换为对应的智能体标准语句
2. 分析需要执行的任务步骤
3. 判断是否需要检查前置条件

可用的智能体标准语句：
{phrases_text}

预定义工作流程：
{workflow_text}

用户请求: "{user_message}"
{history_text}

请以JSON格式返回分析结果：
{{
    "standard_phrase": "对应的标准语句",
    "task_type": "任务类型(single/workflow)",
    "required_steps": ["步骤1", "步骤2", ...],
    "prerequisites": {{
        "inp_file": true/false,
        "partition_csv": true/false,
        "trained_model": true/false
    }},
    "parameters": {{
        "num_partitions": 数字或null,
        "num_sensors": 数字或null,
        "other_params": "其他参数"
    }},
    "confidence": 0.0-1.0,
    "explanation": "分析说明"
}}
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str, original_message: str):
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['standard_phrase', 'task_type', 'required_steps', 'prerequisites']
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"缺少必要字段: {field}")
                
                # 添加原始消息
                result['original_message'] = original_message
                result['analysis_time'] = datetime.now().isoformat()
                
                self.log_info(f"LLM分析成功: {result['standard_phrase']}")
                return result
            else:
                raise ValueError("无法从LLM响应中提取JSON")
                
        except Exception as e:
            self.log_error(f"解析LLM响应失败: {e}")
            return self._fallback_analysis(original_message)
    
    def _fallback_analysis(self, user_message: str):
        """备用分析方法"""
        # 简单的关键词匹配
        message_lower = user_message.lower()

        if any(word in message_lower for word in ['分区', '聚类', '划分']):
            standard_phrase = "管网分区"
            steps = ["管网分析", "管网分区"]
        elif any(word in message_lower for word in ['传感器', '监测点', '布置']):
            standard_phrase = "传感器布置"
            steps = ["管网分析", "管网分区", "传感器布置"]
        elif any(word in message_lower for word in ['推理', '预测', '分析数据']):
            standard_phrase = "模型推理"
            steps = ["模型推理"]
        elif any(word in message_lower for word in ['漏损', '泄漏', '检测']):
            if any(word in message_lower for word in ['训练', '模型']):
                standard_phrase = "漏损模型训练"
                steps = ["管网分析", "漏损模型训练"]
            else:
                standard_phrase = "漏损检测"
                steps = ["管网分析", "管网分区", "传感器布置", "漏损模型训练", "漏损检测"]
        else:
            standard_phrase = "管网分析"
            steps = ["管网分析"]
        
        return {
            "standard_phrase": standard_phrase,
            "task_type": "single" if len(steps) == 1 else "workflow",
            "required_steps": steps,
            "prerequisites": {
                "inp_file": standard_phrase != "模型推理",
                "partition_csv": "传感器布置" in steps or "漏损检测" in steps,
                "trained_model": "漏损检测" in steps or standard_phrase == "模型推理"
            },
            "parameters": {},
            "confidence": 0.6,
            "explanation": "使用关键词匹配的备用分析",
            "original_message": user_message,
            "analysis_time": datetime.now().isoformat()
        }
    
    def check_prerequisites(self, analysis_result: dict, conversation_id: str):
        """检查任务前置条件"""
        prerequisites = analysis_result.get('prerequisites', {})
        missing_prerequisites = []
        available_files = {}

        try:
            # 检查INP文件
            if prerequisites.get('inp_file'):
                inp_file = self._find_inp_file(conversation_id)
                if inp_file:
                    available_files['inp_file'] = inp_file
                else:
                    missing_prerequisites.append('inp_file')

            # 智能检查分区CSV文件 - 对于传感器布置任务，总是检查是否存在分区文件
            standard_phrase = analysis_result.get('standard_phrase', '')
            required_steps = analysis_result.get('required_steps', [])

            # 如果是传感器布置相关任务，主动检查分区文件
            if (standard_phrase == '传感器布置' or
                '传感器布置' in required_steps or
                prerequisites.get('partition_csv')):

                partition_csv = self._find_partition_csv(conversation_id)
                if partition_csv:
                    available_files['partition_csv'] = partition_csv
                    self.log_info(f"检测到现有分区文件: {partition_csv}")
                else:
                    # 只有在明确需要分区文件时才标记为缺失
                    if prerequisites.get('partition_csv'):
                        missing_prerequisites.append('partition_csv')
                    self.log_info("未检测到分区文件，将执行完整流程")

            # 检查训练好的模型
            if prerequisites.get('trained_model'):
                model_file = self._find_trained_model(conversation_id)
                if model_file:
                    available_files['trained_model'] = model_file
                else:
                    missing_prerequisites.append('trained_model')

            return {
                'all_satisfied': len(missing_prerequisites) == 0,
                'missing_prerequisites': missing_prerequisites,
                'available_files': available_files,
                'check_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"检查前置条件失败: {e}")
            return {
                'all_satisfied': False,
                'missing_prerequisites': list(prerequisites.keys()),
                'available_files': {},
                'error': str(e),
                'check_time': datetime.now().isoformat()
            }
    
    def _find_inp_file(self, conversation_id: str):
        """查找对话相关的INP文件"""
        # 在uploads目录中查找
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if (conversation_id[:8] in filename and 
                    filename.endswith('.inp') and 
                    os.path.exists(os.path.join(uploads_dir, filename))):
                    return os.path.join(uploads_dir, filename)
        return None
    
    def _find_partition_csv(self, conversation_id: str):
        """查找对话相关的分区CSV文件"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'partition_results' in filename and
                    filename.endswith('.csv') and
                    os.path.exists(os.path.join(downloads_dir, filename))):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_trained_model(self, conversation_id: str):
        """查找对话相关的训练模型"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'leak_detection_model' in filename and
                    filename.endswith('.pth') and
                    os.path.exists(os.path.join(downloads_dir, filename))):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def generate_execution_plan(self, analysis_result: dict, prerequisites_check: dict):
        """生成执行计划"""
        required_steps = analysis_result.get('required_steps', [])
        missing_prerequisites = prerequisites_check.get('missing_prerequisites', [])
        available_files = prerequisites_check.get('available_files', {})
        standard_phrase = analysis_result.get('standard_phrase', '')

        execution_plan = {
            'total_steps': 0,  # 将在最后更新
            'steps': [],
            'estimated_time': 0,
            'plan_time': datetime.now().isoformat(),
            'workflow_type': 'incremental' if 'partition_csv' in available_files else 'complete'
        }

        # 智能流程控制：传感器布置任务的特殊处理
        if standard_phrase == '传感器布置':
            if 'partition_csv' in available_files:
                # 情况1：已有分区文件，直接进行传感器布置
                self.log_info("检测到分区文件，将跳过分区步骤直接进行传感器布置")
                execution_plan['workflow_description'] = "基于现有分区结果进行传感器布置"

                # 只添加传感器布置步骤
                execution_plan['steps'].append({
                    'step_name': '传感器布置',
                    'step_type': 'main',
                    'description': '基于现有分区结果优化布置压力监测传感器',
                    'estimated_minutes': 5,
                    'order': 1,
                    'input_file': available_files.get('inp_file'),
                    'partition_file': available_files['partition_csv']
                })

            else:
                # 情况2：无分区文件，执行完整流程
                self.log_info("未检测到分区文件，将执行完整的分析->分区->传感器布置流程")
                execution_plan['workflow_description'] = "执行完整的管网分析、分区和传感器布置流程"

                # 添加完整流程步骤
                if 'inp_file' in available_files:
                    execution_plan['steps'].extend([
                        {
                            'step_name': '管网分析',
                            'step_type': 'main',
                            'description': '分析管网结构和基本信息',
                            'estimated_minutes': 2,
                            'order': 1,
                            'input_file': available_files['inp_file']
                        },
                        {
                            'step_name': '管网分区',
                            'step_type': 'main',
                            'description': '把管网划分为指定数量的区域',
                            'estimated_minutes': 3,
                            'order': 2,
                            'input_file': available_files['inp_file']
                        },
                        {
                            'step_name': '传感器布置',
                            'step_type': 'main',
                            'description': '在管网中优化布置压力监测传感器',
                            'estimated_minutes': 5,
                            'order': 3,
                            'input_file': available_files['inp_file']
                        }
                    ])
                else:
                    # 缺少INP文件
                    execution_plan['steps'].append({
                        'step_name': '上传INP文件',
                        'step_type': 'prerequisite',
                        'description': '需要上传管网INP文件才能继续',
                        'estimated_minutes': 1
                    })

            # 更新总步骤数和计算总预估时间
            execution_plan['total_steps'] = len(execution_plan['steps'])
            execution_plan['estimated_time'] = sum(step.get('estimated_minutes', 0) for step in execution_plan['steps'])

            return execution_plan

        # 其他任务的原有逻辑
        # 如果有缺失的前置条件，需要先执行前置步骤
        if missing_prerequisites:
            if 'inp_file' in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': '上传INP文件',
                    'step_type': 'prerequisite',
                    'description': '需要上传管网INP文件才能继续',
                    'estimated_minutes': 1
                })

            if 'partition_csv' in missing_prerequisites and 'inp_file' not in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': '管网分区',
                    'step_type': 'prerequisite',
                    'description': '需要先进行管网分区生成CSV文件',
                    'estimated_minutes': 3
                })

            if 'trained_model' in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': '训练漏损模型',
                    'step_type': 'prerequisite',
                    'description': '需要先训练漏损检测模型',
                    'estimated_minutes': 10
                })
        
        # 添加主要执行步骤，跳过已经完成的步骤
        for i, step in enumerate(required_steps):
            # 如果已经有分区文件，跳过管网分区步骤
            if step == '管网分区' and 'partition_csv' in available_files:
                self.log_info(f"跳过步骤 '{step}' - 已存在分区文件: {available_files['partition_csv']}")
                continue

            # 如果已经有训练模型，跳过模型训练步骤
            if step == '漏损模型训练' and 'trained_model' in available_files:
                self.log_info(f"跳过步骤 '{step}' - 已存在训练模型: {available_files['trained_model']}")
                continue

            step_info = {
                'step_name': step,
                'step_type': 'main',
                'description': self.standard_phrases.get(step, step),
                'estimated_minutes': self._estimate_step_time(step),
                'order': i + 1
            }
            
            # 添加可用文件信息
            if step == '管网分析' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '管网分区' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '水力仿真' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '拓扑分析' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '漏损模型训练' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '传感器布置' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == '漏损检测' and 'trained_model' in available_files:
                step_info['input_file'] = available_files['trained_model']
            
            execution_plan['steps'].append(step_info)
        
        # 更新总步骤数和计算总预估时间
        execution_plan['total_steps'] = len(execution_plan['steps'])
        execution_plan['estimated_time'] = sum(step.get('estimated_minutes', 0) for step in execution_plan['steps'])

        return execution_plan
    
    def _estimate_step_time(self, step_name: str):
        """估算步骤执行时间（分钟）"""
        time_estimates = {
            "管网分析": 2,
            "管网分区": 3,
            "离群点检测": 2,
            "传感器布置": 5,
            "韧性分析": 3,
            "漏损模型训练": 10,
            "漏损检测": 2,
            "水力仿真": 3,
            "拓扑分析": 2
        }
        return time_estimates.get(step_name, 3)
    
    def process(self, user_message: str, conversation_id: str, conversation_history: list = None):
        """处理用户消息的主要方法"""
        try:
            # 1. 分析用户意图
            analysis_result = self.analyze_user_intent(user_message, conversation_history)
            
            # 2. 检查前置条件
            prerequisites_check = self.check_prerequisites(analysis_result, conversation_id)
            
            # 3. 生成执行计划
            execution_plan = self.generate_execution_plan(analysis_result, prerequisites_check)
            
            # 4. 组合结果
            result = {
                'success': True,
                'analysis': analysis_result,
                'prerequisites': prerequisites_check,
                'execution_plan': execution_plan,
                'conversation_id': conversation_id,
                'process_time': datetime.now().isoformat()
            }
            
            self.log_info(f"任务分析完成: {analysis_result['standard_phrase']}")
            return result
            
        except Exception as e:
            self.log_error(f"处理任务失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'conversation_id': conversation_id,
                'process_time': datetime.now().isoformat()
            }
