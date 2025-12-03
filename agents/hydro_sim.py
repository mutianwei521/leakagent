"""
HydroSim 水力仿真智能体
负责处理.inp文件，进行管网分析和水力计算
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from .base_agent import BaseAgent
from .intent_classifier_fast import FastIntentClassifier as IntentClassifier

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

class HydroSim(BaseAgent):
    """水力仿真智能体"""
    
    def __init__(self):
        super().__init__("HydroSim")

        if not WNTR_AVAILABLE:
            self.log_error("WNTR库未安装，水力计算功能不可用")

        self.intent_classifier = IntentClassifier()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)

        # 缓存机制：避免重复解析同一个文件
        self._network_cache = {}  # {file_path: {network_info, last_modified}}
    
    def parse_network(self, inp_file_path: str):
        """解析管网文件，提取基本信息"""
        if not WNTR_AVAILABLE:
            return {'error': 'WNTR库未安装'}

        try:
            # 检查缓存
            if inp_file_path in self._network_cache:
                file_mtime = os.path.getmtime(inp_file_path)
                cached_data = self._network_cache[inp_file_path]
                if cached_data['last_modified'] == file_mtime:
                    self.log_info(f"使用缓存的管网信息: {inp_file_path}")
                    return cached_data['network_info']

            self.log_info(f"开始解析管网文件: {inp_file_path}")

            # 读取管网文件
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            
            # 提取关键信息
            network_info = {
                'nodes': {
                    'junctions': len(wn.junction_name_list),
                    'reservoirs': len(wn.reservoir_name_list),
                    'tanks': len(wn.tank_name_list),
                    'total': len(wn.node_name_list)
                },
                'links': {
                    'pipes': len(wn.pipe_name_list),
                    'pumps': len(wn.pump_name_list),
                    'valves': len(wn.valve_name_list),
                    'total': len(wn.link_name_list)
                },
                'network_stats': {
                    'total_length': float(sum([wn.get_link(pipe).length for pipe in wn.pipe_name_list])) if len(wn.pipe_name_list) > 0 else 0,
                    'simulation_duration': wn.options.time.duration,
                    'hydraulic_timestep': wn.options.time.hydraulic_timestep,
                    'pattern_timestep': wn.options.time.pattern_timestep
                }
            }
            
            # 添加详细的拓扑信息用于可视化
            network_info['topology'] = self._extract_topology_data(wn)

            self.log_info(f"管网解析完成: {network_info['nodes']['total']}个节点, {network_info['links']['total']}个管段")

            # 更新缓存
            file_mtime = os.path.getmtime(inp_file_path)
            self._network_cache[inp_file_path] = {
                'network_info': network_info,
                'last_modified': file_mtime
            }

            return network_info
            
        except Exception as e:
            error_msg = f"解析管网文件失败: {e}"
            self.log_error(error_msg)
            return {'error': error_msg}

    def _extract_topology_data(self, wn):
        """提取拓扑数据用于可视化"""
        try:
            topology = {
                'nodes': [],
                'links': []
            }

            # 提取节点信息
            for node_name in wn.node_name_list:
                node = wn.get_node(node_name)

                # 确定节点类型
                node_type = 'junction'  # 默认类型
                class_name = type(node).__name__

                # 根据WNTR的类名确定类型
                if 'Reservoir' in class_name:
                    node_type = 'reservoir'
                elif 'Tank' in class_name:
                    node_type = 'tank'
                elif 'Junction' in class_name:
                    node_type = 'junction'
                else:
                    # 尝试其他属性
                    if hasattr(node, '_node_type'):
                        node_type = node._node_type.lower()
                    elif hasattr(node, 'node_type'):
                        node_type = node.node_type.lower()
                    else:
                        # 最后的备用方案
                        class_lower = class_name.lower()
                        if 'reservoir' in class_lower:
                            node_type = 'reservoir'
                        elif 'tank' in class_lower:
                            node_type = 'tank'

                node_data = {
                    'id': node_name,
                    'type': node_type,
                    'coordinates': [node.coordinates[0], node.coordinates[1]] if hasattr(node, 'coordinates') and node.coordinates else [0, 0]
                }

                # 添加节点特定属性
                if hasattr(node, 'elevation'):
                    node_data['elevation'] = float(node.elevation) if node.elevation is not None else 0.0
                if hasattr(node, 'base_demand'):
                    node_data['base_demand'] = float(node.base_demand) if node.base_demand is not None else 0.0
                if hasattr(node, 'head'):
                    node_data['head'] = float(node.head) if node.head is not None else 0.0
                if hasattr(node, 'init_level') and node.init_level is not None:
                    node_data['init_level'] = float(node.init_level)
                if hasattr(node, 'max_level') and node.max_level is not None:
                    node_data['max_level'] = float(node.max_level)
                if hasattr(node, 'min_level') and node.min_level is not None:
                    node_data['min_level'] = float(node.min_level)

                topology['nodes'].append(node_data)

            # 提取管段信息
            for link_name in wn.link_name_list:
                link = wn.get_link(link_name)

                # 确定管段类型
                link_type = 'pipe'  # 默认类型
                class_name = type(link).__name__

                # 根据WNTR的类名确定类型
                if 'Pump' in class_name:
                    link_type = 'pump'
                elif 'Valve' in class_name:
                    link_type = 'valve'
                elif 'Pipe' in class_name:
                    link_type = 'pipe'
                else:
                    # 尝试其他属性
                    if hasattr(link, '_link_type'):
                        link_type = link._link_type.lower()
                    elif hasattr(link, 'link_type'):
                        link_type = link.link_type.lower()
                    else:
                        # 最后的备用方案
                        class_lower = class_name.lower()
                        if 'pump' in class_lower:
                            link_type = 'pump'
                        elif 'valve' in class_lower:
                            link_type = 'valve'

                link_data = {
                    'id': link_name,
                    'type': link_type,
                    'start_node': link.start_node_name,
                    'end_node': link.end_node_name
                }

                # 添加管段特定属性
                if hasattr(link, 'length'):
                    link_data['length'] = float(link.length) if link.length is not None else 0.0
                if hasattr(link, 'diameter'):
                    link_data['diameter'] = float(link.diameter) if link.diameter is not None else 0.0
                if hasattr(link, 'roughness'):
                    link_data['roughness'] = float(link.roughness) if link.roughness is not None else 0.0
                if hasattr(link, 'minor_loss'):
                    link_data['minor_loss'] = float(link.minor_loss) if link.minor_loss is not None else 0.0

                topology['links'].append(link_data)

            return topology

        except Exception as e:
            self.log_error(f"提取拓扑数据失败: {e}")
            return {'nodes': [], 'links': []}
    
    def run_hydraulic_simulation(self, inp_file_path: str):
        """运行水力计算"""
        if not WNTR_AVAILABLE:
            return {'success': False, 'error': 'WNTR库未安装'}
        
        try:
            self.log_info("开始水力计算...")
            
            # 创建网络模型
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            
            # 运行水力计算
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()
            
            # 提取关键数据
            simulation_data = {
                'node_pressure': results.node['pressure'],
                'node_demand': results.node['demand'],
                'link_flowrate': results.link['flowrate'],
                'link_velocity': results.link['velocity']
            }
            
            self.log_info("水力计算完成")
            return {'success': True, 'data': simulation_data}
            
        except Exception as e:
            error_msg = f"水力计算失败: {e}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def save_simulation_to_csv(self, simulation_data: dict, conversation_id: str):
        """保存水力计算结果为CSV文件"""
        try:
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hydraulic_simulation_{conversation_id[:8]}_{timestamp}.csv"
            file_path = os.path.join(self.downloads_folder, filename)

            # 准备数据
            all_data = []

            # 处理节点压力数据
            if 'node_pressure' in simulation_data:
                pressure_df = simulation_data['node_pressure']
                # WNTR的DataFrame结构：行是时间步长，列是节点ID
                for time_idx in pressure_df.index:  # 时间步长在行索引中
                    for node_id in pressure_df.columns:  # 节点ID在列索引中
                        try:
                            # time_idx是时间步长（秒），转换为小时
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # 默认值
                        all_data.append({
                            '时间(小时)': time_hours,
                            '节点ID': str(node_id),  # 确保是字符串
                            '数据类型': '节点压力',
                            '数值': pressure_df.loc[time_idx, node_id],
                            '单位': 'm'
                        })
            
            # 处理节点需水量数据
            if 'node_demand' in simulation_data:
                demand_df = simulation_data['node_demand']
                # WNTR的DataFrame结构：行是时间步长，列是节点ID
                for time_idx in demand_df.index:  # 时间步长在行索引中
                    for node_id in demand_df.columns:  # 节点ID在列索引中
                        try:
                            # time_idx是时间步长（秒），转换为小时
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # 默认值
                        all_data.append({
                            '时间(小时)': time_hours,
                            '节点ID': str(node_id),  # 确保是字符串
                            '数据类型': '节点需水量',
                            '数值': demand_df.loc[time_idx, node_id],
                            '单位': 'L/s'
                        })
            
            # 处理管段流量数据
            if 'link_flowrate' in simulation_data:
                flow_df = simulation_data['link_flowrate']
                # WNTR的DataFrame结构：行是时间步长，列是管段ID
                for time_idx in flow_df.index:  # 时间步长在行索引中
                    for link_id in flow_df.columns:  # 管段ID在列索引中
                        try:
                            # time_idx是时间步长（秒），转换为小时
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # 默认值
                        all_data.append({
                            '时间(小时)': time_hours,
                            '管段ID': str(link_id),  # 确保是字符串
                            '数据类型': '管段流量',
                            '数值': flow_df.loc[time_idx, link_id],
                            '单位': 'L/s'
                        })
            
            # 处理管段流速数据
            if 'link_velocity' in simulation_data:
                velocity_df = simulation_data['link_velocity']
                # WNTR的DataFrame结构：行是时间步长，列是管段ID
                for time_idx in velocity_df.index:  # 时间步长在行索引中
                    for link_id in velocity_df.columns:  # 管段ID在列索引中
                        try:
                            # time_idx是时间步长（秒），转换为小时
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # 默认值
                        all_data.append({
                            '时间(小时)': time_hours,
                            '管段ID': str(link_id),  # 确保是字符串
                            '数据类型': '管段流速',
                            '数值': velocity_df.loc[time_idx, link_id],
                            '单位': 'm/s'
                        })
            
            # 保存为CSV
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                file_size = os.path.getsize(file_path)
                self.log_info(f"CSV文件保存成功: {filename} ({file_size} 字节)")
                
                return {
                    'success': True,
                    'filename': filename,
                    'file_path': file_path,
                    'download_url': f'/download/{filename}',
                    'file_size': file_size,
                    'records_count': len(all_data)
                }
            else:
                return {'success': False, 'error': '没有可保存的数据'}
                
        except Exception as e:
            error_msg = f"保存CSV文件失败: {e}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def build_simulation_prompt(self, network_info: dict, simulation_data: dict, user_message: str, csv_info: dict):
        """构建包含下载链接的水力计算分析prompt"""
        prompt = f"""
你是一个专业的给水管网分析专家。现在需要分析以下管网系统：

管网基本信息：
- 节点总数：{network_info['nodes']['total']} (节点: {network_info['nodes']['junctions']}, 水库: {network_info['nodes']['reservoirs']}, 水塔: {network_info['nodes']['tanks']})
- 管段总数：{network_info['links']['total']} (管道: {network_info['links']['pipes']}, 水泵: {network_info['links']['pumps']}, 阀门: {network_info['links']['valves']})
- 管网总长度：{network_info['network_stats']['total_length']:.2f} 米
- 仿真时长：{network_info['network_stats']['simulation_duration']} 秒

✅ 水力计算已成功完成！

计算结果包含：
- 节点压力分布数据
- 节点需水量数据
- 管段流量数据
- 管段流速数据

📊 详细数据已保存为CSV文件：{csv_info['filename']}
文件大小：{csv_info['file_size']} 字节，共 {csv_info['records_count']} 条记录

用户问题：{user_message}

请基于管网信息和水力计算结果，提供专业的分析和建议。
同时告知用户可以下载详细的计算数据进行进一步分析。

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_analysis_prompt(self, network_info: dict, user_message: str):
        """构建管网结构分析prompt"""
        prompt = f"""
你是一个专业的给水管网分析专家。现在需要分析以下管网系统的结构：

管网基本信息：
- 节点总数：{network_info['nodes']['total']} (接点: {network_info['nodes']['junctions']}, 水库: {network_info['nodes']['reservoirs']}, 水塔: {network_info['nodes']['tanks']})
- 管段总数：{network_info['links']['total']} (管道: {network_info['links']['pipes']}, 水泵: {network_info['links']['pumps']}, 阀门: {network_info['links']['valves']})
- 管网总长度：{network_info['network_stats']['total_length']:.2f} 米

用户问题：{user_message}

请基于管网结构信息，提供专业的分析和建议。
如果用户需要详细的水力计算数据，请建议进行水力计算。

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_general_prompt(self, network_info: dict, user_message: str):
        """构建一般咨询prompt"""
        prompt = f"""
你是一个专业的给水管网分析专家。用户上传了一个管网文件(.inp格式)。

管网基本信息：
- 节点总数：{network_info['nodes']['total']}
- 管段总数：{network_info['links']['total']}
- 管网总长度：{network_info['network_stats']['total_length']:.2f} 米

用户问题：{user_message}

请回答用户的问题，并介绍可以提供的分析功能：
1. 管网结构分析
2. 水力计算和仿真
3. 数据导出和下载

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_error_prompt(self, network_info: dict, user_message: str, error_message: str):
        """构建错误处理prompt"""
        prompt = f"""
你是一个专业的给水管网分析专家。在处理用户请求时遇到了问题。

管网基本信息：
- 节点总数：{network_info['nodes']['total']}
- 管段总数：{network_info['links']['total']}

用户问题：{user_message}

遇到的问题：{error_message}

请向用户说明遇到的问题，并提供可能的解决方案或替代建议。

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def process(self, inp_file_path: str, user_message: str, conversation_id: str):
        """处理管网文件和用户消息的主要方法"""
        self.log_info(f"开始处理管网文件: {inp_file_path}")

        # Step 1: 解析管网文件
        network_info = self.parse_network(inp_file_path)
        if 'error' in network_info:
            return {
                'success': False,
                'response': f"管网文件解析失败: {network_info['error']}",
                'network_info': None,
                'intent': 'error',
                'confidence': 0.0
            }

        # Step 2: 智能意图识别
        intent_result = self.intent_classifier.classify_intent(user_message)
        intent = intent_result['intent']
        confidence = intent_result['confidence']

        self.log_info(f"识别意图: {intent}, 置信度: {confidence:.3f}")

        csv_info = None
        prompt = ""

        # Step 3: 根据意图执行不同操作
        if intent == 'hydraulic_simulation' and confidence > 0.7:
            # 执行水力计算
            simulation_result = self.run_hydraulic_simulation(inp_file_path)

            if simulation_result['success']:
                # 保存CSV文件
                csv_info = self.save_simulation_to_csv(
                    simulation_result['data'],
                    conversation_id
                )

                if csv_info['success']:
                    prompt = self.build_simulation_prompt(
                        network_info,
                        simulation_result['data'],
                        user_message,
                        csv_info
                    )
                else:
                    prompt = self.build_error_prompt(
                        network_info,
                        user_message,
                        f"水力计算成功，但保存CSV文件失败: {csv_info['error']}"
                    )
            else:
                prompt = self.build_error_prompt(
                    network_info,
                    user_message,
                    f"水力计算失败: {simulation_result['error']}"
                )

        elif intent == 'network_analysis' and confidence > 0.6:
            # 结构分析
            prompt = self.build_analysis_prompt(network_info, user_message)

        else:
            # 一般咨询
            prompt = self.build_general_prompt(network_info, user_message)

        return {
            'success': True,
            'prompt': prompt,
            'csv_info': csv_info,
            'network_info': network_info,
            'intent': intent,
            'confidence': confidence
        }
