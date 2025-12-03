"""
智能体执行器
负责根据LLM分析结果协调执行不同的智能体
"""
import os
import json
import openai
from datetime import datetime
from .base_agent import BaseAgent

class AgentExecutor(BaseAgent):
    """智能体执行器"""
    
    def __init__(self, hydro_sim, partition_sim, sensor_placement, leak_detection):
        super().__init__("AgentExecutor")
        
        # 注册各个智能体
        self.agents = {
            "管网分析": hydro_sim,
            "管网分区": partition_sim,
            "传感器布置": sensor_placement,
            "漏损模型训练": leak_detection,
            "漏损检测": leak_detection,
            "水力仿真": hydro_sim,
            "拓扑分析": hydro_sim
        }
        
        # 设置OpenAI API配置
        openai.api_base = "https://api.chatanywhere.tech"
        openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"
    
    def execute_step(self, step_name: str, step_info: dict, conversation_id: str, user_message: str):
        """执行单个步骤"""
        try:
            self.log_info(f"开始执行步骤: {step_name}")
            
            # 获取对应的智能体
            agent = self.agents.get(step_name)
            if not agent:
                return {
                    'success': False,
                    'error': f'未找到对应的智能体: {step_name}',
                    'step_name': step_name
                }
            
            # 根据步骤类型执行不同的逻辑
            if step_name == "管网分析":
                return self._execute_network_analysis(agent, step_info, conversation_id, user_message)
            elif step_name == "管网分区":
                return self._execute_partition_analysis(agent, step_info, conversation_id, user_message)
            elif step_name == "传感器布置":
                return self._execute_sensor_placement(agent, step_info, conversation_id, user_message)
            elif step_name == "漏损模型训练":
                return self._execute_leak_training(agent, step_info, conversation_id, user_message)
            elif step_name == "漏损检测":
                return self._execute_leak_detection(agent, step_info, conversation_id, user_message)
            else:
                return self._execute_generic_step(agent, step_info, conversation_id, user_message)
                
        except Exception as e:
            self.log_error(f"执行步骤失败 {step_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': step_name
            }
    
    def _execute_network_analysis(self, agent, step_info, conversation_id, user_message):
        """执行管网分析"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': '缺少INP文件',
                'step_name': '管网分析'
            }
        
        # 调用水力仿真智能体
        result = agent.process(inp_file, user_message, conversation_id)
        
        if result['success']:
            self.log_info("管网分析执行成功")
            return {
                'success': True,
                'step_name': '管网分析',
                'result': result,
                'agent_type': 'hydro_sim'
            }
        else:
            return {
                'success': False,
                'error': result.get('response', '管网分析失败'),
                'step_name': '管网分析'
            }
    
    def _execute_partition_analysis(self, agent, step_info, conversation_id, user_message):
        """执行管网分区"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': '缺少INP文件',
                'step_name': '管网分区'
            }
        
        # 构建分区请求消息
        partition_message = user_message
        if "分区" not in user_message:
            partition_message = f"{user_message} 请进行管网分区分析"
        
        # 调用分区智能体
        result = agent.process(inp_file, partition_message, conversation_id)
        
        if result['success']:
            self.log_info("管网分区执行成功")
            return {
                'success': True,
                'step_name': '管网分区',
                'result': result,
                'agent_type': 'partition_sim',
                'csv_file': result.get('csv_info', {}).get('filepath')
            }
        else:
            return {
                'success': False,
                'error': result.get('response', '管网分区失败'),
                'step_name': '管网分区'
            }
    
    def _execute_sensor_placement(self, agent, step_info, conversation_id, user_message):
        """执行传感器布置"""
        try:
            inp_file = step_info.get('input_file')
            # 优先使用执行计划中指定的分区文件，否则查找现有文件
            partition_csv = step_info.get('partition_file') or self._find_partition_csv(conversation_id)

            self.log_info(f"传感器布置参数检查: inp_file={inp_file}, partition_csv={partition_csv}")

            # 检查是否为基于现有分区的增量执行
            if step_info.get('partition_file'):
                self.log_info(f"使用指定的分区文件: {partition_csv}")
            elif partition_csv:
                self.log_info(f"使用检测到的分区文件: {partition_csv}")

            if not inp_file:
                return {
                    'success': False,
                    'error': '缺少INP文件',
                    'step_name': '传感器布置'
                }

            if not partition_csv:
                return {
                    'success': False,
                    'error': '缺少分区CSV文件，请先进行管网分区',
                    'step_name': '传感器布置'
                }

            # 构建传感器布置请求消息
            sensor_message = user_message
            if "传感器" not in user_message and "监测" not in user_message:
                sensor_message = f"{user_message} 请进行传感器布置优化"

            self.log_info(f"开始调用传感器布置智能体: {sensor_message}")

            # 调用传感器布置智能体
            result = agent.process(inp_file, partition_csv, sensor_message, conversation_id)

            self.log_info(f"传感器布置智能体调用完成，结果: {result.get('success', False)}")

            if result['success']:
                self.log_info("传感器布置执行成功")
                return {
                    'success': True,
                    'step_name': '传感器布置',
                    'result': result,
                    'agent_type': 'sensor_placement',
                    'csv_file': result.get('csv_info', {}).get('filepath')
                }
            else:
                self.log_error(f"传感器布置执行失败: {result.get('response', '未知错误')}")
                return {
                    'success': False,
                    'error': result.get('response', '传感器布置失败'),
                    'step_name': '传感器布置'
                }

        except Exception as e:
            error_msg = f"传感器布置执行异常: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_name': '传感器布置'
            }
    
    def _execute_leak_training(self, agent, step_info, conversation_id, user_message):
        """执行漏损模型训练"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': '缺少INP文件',
                'step_name': '漏损模型训练'
            }
        
        # 提取训练参数
        num_scenarios = 50  # 默认值
        epochs = 100  # 默认值

        # 智能提取训练参数
        import re
        num_scenarios, epochs = self._extract_training_parameters(user_message, num_scenarios, epochs)

        self.log_info(f"解析的训练参数: 样本数={num_scenarios}, 迭代次数={epochs}")

        # 调用漏损检测智能体进行训练
        result = agent.train_leak_detection_model(inp_file, conversation_id, num_scenarios, epochs)
        
        if result['success']:
            self.log_info("漏损模型训练执行成功")
            return {
                'success': True,
                'step_name': '漏损模型训练',
                'result': result,
                'agent_type': 'leak_detection',
                'model_file': result.get('files', {}).get('model', {}).get('filepath')
            }
        else:
            return {
                'success': False,
                'error': result.get('error', '漏损模型训练失败'),
                'step_name': '漏损模型训练'
            }
    
    def _execute_leak_detection(self, agent, step_info, conversation_id, user_message):
        """执行漏损检测"""
        # 查找传感器数据文件和训练好的模型
        sensor_data_file = self._find_sensor_data_csv(conversation_id)
        model_file = self._find_trained_model(conversation_id)
        
        if not sensor_data_file:
            return {
                'success': False,
                'error': '缺少传感器数据CSV文件',
                'step_name': '漏损检测'
            }
        
        if not model_file:
            return {
                'success': False,
                'error': '缺少训练好的漏损检测模型',
                'step_name': '漏损检测'
            }
        
        # 调用漏损检测智能体进行检测
        result = agent.detect_leak_from_file(sensor_data_file, model_file)
        
        if result['success']:
            self.log_info("漏损检测执行成功")
            return {
                'success': True,
                'step_name': '漏损检测',
                'result': result,
                'agent_type': 'leak_detection'
            }
        else:
            return {
                'success': False,
                'error': result.get('error', '漏损检测失败'),
                'step_name': '漏损检测'
            }
    
    def _execute_generic_step(self, agent, step_info, conversation_id, user_message):
        """执行通用步骤"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': '缺少输入文件',
                'step_name': step_info.get('step_name', '未知步骤')
            }
        
        # 调用智能体
        result = agent.process(inp_file, user_message, conversation_id)
        
        return {
            'success': result.get('success', False),
            'step_name': step_info.get('step_name', '未知步骤'),
            'result': result,
            'error': result.get('response') if not result.get('success') else None
        }
    
    def _find_partition_csv(self, conversation_id: str):
        """查找分区CSV文件"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'partition_results' in filename and
                    filename.endswith('.csv')):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_trained_model(self, conversation_id: str):
        """查找训练好的模型"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'leak_detection_model' in filename and
                    filename.endswith('.pth')):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_sensor_data_csv(self, conversation_id: str):
        """查找传感器数据CSV文件"""
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if (conversation_id[:8] in filename and 
                    filename.endswith('.csv')):
                    return os.path.join(uploads_dir, filename)
        return None
    
    def generate_response_with_llm(self, execution_results: list, original_message: str):
        """使用LLM生成综合响应"""
        try:
            # 构建执行结果摘要
            results_summary = ""
            for i, result in enumerate(execution_results):
                step_name = result.get('step_name', f'步骤{i+1}')
                success = result.get('success', False)
                status = "成功" if success else "失败"
                error = result.get('error', '')
                
                results_summary += f"\n{i+1}. {step_name}: {status}"
                if not success and error:
                    results_summary += f" (错误: {error})"
                elif success and result.get('result'):
                    # 添加成功结果的简要信息
                    agent_result = result['result']
                    if agent_result.get('network_info'):
                        results_summary += f" - 管网信息已分析"
                    if agent_result.get('partition_info'):
                        results_summary += f" - 分区结果已生成"
                    if agent_result.get('sensor_info'):
                        results_summary += f" - 传感器布置已完成"
                    if agent_result.get('model_info'):
                        results_summary += f" - 模型训练已完成"
            
            # 构建LLM prompt
            prompt = f"""
用户原始请求: "{original_message}"

执行结果摘要:
{results_summary}

请根据执行结果生成一个专业、详细的回复，包括：
1. 对用户请求的理解和确认
2. 执行过程的说明
3. 主要结果和发现
4. 如果有失败的步骤，提供解决建议
5. 下一步建议（如果适用）

请用专业但易懂的语言回复，重点突出关键信息。

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
            
            # 调用LLM生成响应
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的给水管网分析专家，能够根据智能体执行结果生成清晰、专业的分析报告。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.log_error(f"LLM响应生成失败: {e}")
            # 返回简单的执行摘要
            successful_steps = [r['step_name'] for r in execution_results if r.get('success')]
            failed_steps = [r['step_name'] for r in execution_results if not r.get('success')]
            
            summary = f"执行完成。成功步骤: {', '.join(successful_steps) if successful_steps else '无'}"
            if failed_steps:
                summary += f"。失败步骤: {', '.join(failed_steps)}"
            
            return summary
    
    def process(self, execution_plan: dict, conversation_id: str, user_message: str):
        """执行完整的执行计划"""
        try:
            steps = execution_plan.get('steps', [])
            execution_results = []
            
            self.log_info(f"开始执行计划，共{len(steps)}个步骤")
            
            for i, step in enumerate(steps):
                step_name = step.get('step_name')
                step_type = step.get('step_type', 'main')
                
                # 跳过前置条件步骤（这些应该在UI层处理）
                if step_type == 'prerequisite':
                    self.log_info(f"跳过前置条件步骤: {step_name}")
                    continue
                
                # 执行主要步骤
                result = self.execute_step(step_name, step, conversation_id, user_message)
                execution_results.append(result)
                
                # 如果步骤失败且是关键步骤，可能需要停止执行
                if not result.get('success') and step_name in ['管网分析', '管网分区']:
                    self.log_warning(f"关键步骤失败，停止执行: {step_name}")
                    break
            
            # 生成综合响应
            llm_response = self.generate_response_with_llm(execution_results, user_message)
            
            return {
                'success': True,
                'execution_results': execution_results,
                'llm_response': llm_response,
                'total_steps': len(steps),
                'completed_steps': len(execution_results),
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"执行计划失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }

    def _extract_training_parameters(self, user_message: str, default_scenarios: int, default_epochs: int) -> tuple:
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
                self.log_info(f"识别到迭代次数: {epochs} (匹配模式: {pattern})")
                break

        # 尝试匹配样本数
        for pattern in scenario_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                num_scenarios = min(int(match.group(1)), 2000)  # 提高最大限制到2000
                self.log_info(f"识别到样本数: {num_scenarios} (匹配模式: {pattern})")
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
                        self.log_info(f"根据上下文识别为迭代次数: {epochs}")
                    elif any(keyword in user_message.lower() for keyword in ['数据', '样本', '场景', '组']):
                        num_scenarios = min(num, 2000)  # 提高最大限制到2000
                        self.log_info(f"根据上下文识别为样本数: {num_scenarios}")
                    else:
                        # 默认第一个数字作为样本数
                        num_scenarios = min(num, 2000)  # 提高最大限制到2000
                        self.log_info(f"默认识别为样本数: {num_scenarios}")
                elif len(numbers) >= 2:
                    # 多个数字时，按原来的逻辑：第一个作为样本数，第二个作为迭代次数
                    num_scenarios = min(int(numbers[0]), 2000)  # 提高最大限制到2000
                    epochs = min(int(numbers[1]), 500)
                    self.log_info(f"多数字模式: 样本数={num_scenarios}, 迭代次数={epochs}")

        return num_scenarios, epochs
