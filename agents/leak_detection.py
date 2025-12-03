"""
漏损检测智能体 - LeakDetectionAgent
基于压力敏感度分析和机器学习的管网漏损检测系统

主要功能：
1. 数据准备：检查分区和传感器配置，缺失时调用其他智能体
2. 敏感度计算：模拟漏损场景，计算压力敏感度矩阵
3. 数据生成：生成平衡的训练数据集（异常+正常）
4. 模型训练：使用MLP进行漏损检测模型训练
5. 推理预测：对新的传感器数据进行漏损检测

作者：LeakAgent Team
日期：2025-09-18
"""

import os
import sys
import json
import uuid
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 机器学习相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 水网络分析
import wntr
import matplotlib.pyplot as plt
import seaborn as sns

# 基础智能体
from .base_agent import BaseAgent


class LeakDetectionMLP(nn.Module):
    """漏损检测多层感知机模型"""
    
    def __init__(self, input_size: int, num_partitions: int, hidden_sizes: List[int] = [128, 64, 32], num_classes: int = None):
        super(LeakDetectionMLP, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes if num_classes is not None else (num_partitions + 1)  # +1 for normal class (0)
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, self.num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class LeakDetectionAgent(BaseAgent):
    """漏损检测智能体"""
    
    def __init__(self):
        super().__init__("LeakDetectionAgent")
        self.agent_name = "LeakDetectionAgent"
        self.downloads_folder = "downloads"
        self.uploads_folder = "uploads"
        
        # 确保下载文件夹存在
        os.makedirs(self.downloads_folder, exist_ok=True)
        
        # 模型相关
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据缓存
        self.partition_data = None
        self.sensor_data = None
        self.network_model = None
        
        self.log_info(f"漏损检测智能体初始化完成，使用设备: {self.device}")
    
    def check_dependencies(self, conversation_id: str, inp_file_path: str = None) -> Dict[str, Any]:
        """智能检查分区和传感器配置文件，优先复用已有文件"""
        try:
            self.log_info("🔍 智能检查分区和传感器配置文件...")

            # 查找相关文件
            partition_file = None
            sensor_file = None
            sensor_files = []  # 存储所有找到的传感器文件

            # 扫描下载文件夹
            if os.path.exists(self.downloads_folder):
                for filename in os.listdir(self.downloads_folder):
                    if conversation_id[:8] in filename:
                        if 'partition_results' in filename and filename.endswith('.csv'):
                            partition_file = os.path.join(self.downloads_folder, filename)
                            self.log_info(f"✅ 找到分区文件: {os.path.basename(partition_file)}")
                        elif 'sensor_placement' in filename and filename.endswith('.csv'):
                            sensor_file_path = os.path.join(self.downloads_folder, filename)
                            sensor_files.append(sensor_file_path)

            # 选择最新的传感器文件
            if sensor_files:
                # 按文件名排序，选择最新的
                sensor_files.sort()
                sensor_file = sensor_files[-1]
                self.log_info(f"✅ 找到传感器布置文件: {os.path.basename(sensor_file)}")

                # 显示传感器信息
                try:
                    sensor_df = pd.read_csv(sensor_file)
                    if '节点名称' in sensor_df.columns:
                        sensor_nodes = sensor_df['节点名称'].tolist()
                        self.log_info(f"📍 检测到 {len(sensor_nodes)} 个传感器节点: {sensor_nodes}")
                    elif 'Node' in sensor_df.columns:
                        sensor_nodes = sensor_df['Node'].tolist()
                        self.log_info(f"📍 检测到 {len(sensor_nodes)} 个传感器节点: {sensor_nodes}")
                    else:
                        self.log_warning("传感器文件格式异常，无法读取节点信息")
                except Exception as e:
                    self.log_warning(f"读取传感器文件信息失败: {str(e)}")

            result = {
                'partition_file': partition_file,
                'sensor_file': sensor_file,
                'missing_files': [],
                'success': True,
                'reused_files': []
            }

            # 智能处理缺失文件
            missing_files = []

            # 检查传感器文件
            if sensor_file:
                result['reused_files'].append('sensor_placement')
                self.log_info("♻️ 复用已有传感器布置，无需重新生成")
            else:
                missing_files.append('sensor_placement')
                self.log_warning("⚠️ 未找到传感器布置文件")

            # 检查分区文件
            if partition_file:
                result['reused_files'].append('partition_results')
                self.log_info("♻️ 复用已有分区结果，无需重新生成")
            else:
                missing_files.append('partition_results')
                self.log_warning("⚠️ 未找到分区结果文件")

            # 特殊处理：如果有传感器文件但没有分区文件，尝试从传感器文件推断分区信息
            if sensor_file and not partition_file:
                self.log_info("🧠 尝试从传感器文件推断分区信息...")
                inferred_partition = self._infer_partition_from_sensors(sensor_file, conversation_id)
                if inferred_partition.get('success'):
                    result['partition_file'] = inferred_partition['partition_file']
                    if 'partition_results' in missing_files:
                        missing_files.remove('partition_results')
                    result['reused_files'].append('partition_results_inferred')
                    self.log_info("✅ 成功从传感器文件推断分区信息")

            # 只有在真正缺失且提供了INP文件时才生成
            if missing_files and inp_file_path:
                self.log_info(f"🔧 需要生成缺失文件: {missing_files}")

                # 智能生成策略：优先保持已有文件不变
                generated_files = self._generate_missing_files_smart(missing_files, inp_file_path, conversation_id, sensor_file)

                if generated_files.get('success'):
                    # 更新结果
                    if 'partition_results' in missing_files and generated_files.get('partition_file'):
                        result['partition_file'] = generated_files['partition_file']
                        missing_files.remove('partition_results')

                    if 'sensor_placement' in missing_files and generated_files.get('sensor_file'):
                        result['sensor_file'] = generated_files['sensor_file']
                        missing_files.remove('sensor_placement')
                else:
                    self.log_error("自动生成依赖文件失败")
                    result['success'] = False
                    result['error'] = generated_files.get('error', '未知错误')
                    return result

            result['missing_files'] = missing_files

            if missing_files:
                self.log_error(f"❌ 仍然缺失文件: {missing_files}")
                result['success'] = False
                result['error'] = f"缺失必要的配置文件: {missing_files}"
                return result

            # 加载数据
            self.partition_data = pd.read_csv(result['partition_file'])
            self.sensor_data = pd.read_csv(result['sensor_file'])

            self.log_info(f"📂 成功加载分区文件: {os.path.basename(result['partition_file'])}")
            self.log_info(f"📂 成功加载传感器文件: {os.path.basename(result['sensor_file'])}")

            if result['reused_files']:
                self.log_info(f"♻️ 复用的文件: {', '.join(result['reused_files'])}")

            return result

        except Exception as e:
            error_msg = f"检查依赖文件失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def _infer_partition_from_sensors(self, sensor_file: str, conversation_id: str) -> Dict[str, Any]:
        """从传感器文件推断分区信息"""
        try:
            self.log_info("从传感器布置文件推断分区信息...")

            # 读取传感器数据
            sensor_df = pd.read_csv(sensor_file)

            # 检查是否包含分区信息
            partition_col = None
            node_col = None

            # 识别列名
            for col in sensor_df.columns:
                if col in ['分区编号', 'partition', 'Partition']:
                    partition_col = col
                if col in ['节点名称', 'node_id', 'Node']:
                    node_col = col

            if partition_col and node_col:
                # 创建简化的分区文件
                partition_data = []
                for _, row in sensor_df.iterrows():
                    node_id = row[node_col]
                    partition_id = row[partition_col]
                    partition_data.append({
                        'Node': node_id,
                        'Partition': partition_id
                    })

                if partition_data:
                    # 保存推断的分区文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    partition_filename = f"partition_results_{conversation_id[:8]}_{timestamp}_inferred.csv"
                    partition_filepath = os.path.join(self.downloads_folder, partition_filename)

                    partition_df = pd.DataFrame(partition_data)
                    partition_df.to_csv(partition_filepath, index=False)

                    self.log_info(f"✅ 成功推断并保存分区文件: {partition_filename}")
                    return {
                        'success': True,
                        'partition_file': partition_filepath,
                        'method': 'inferred_from_sensors'
                    }

            self.log_warning("传感器文件中未找到分区信息，无法推断")
            return {'success': False, 'error': '传感器文件中未找到分区信息'}

        except Exception as e:
            self.log_error(f"从传感器文件推断分区信息失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_missing_files_smart(self, missing_files: List[str], inp_file_path: str,
                                    conversation_id: str, existing_sensor_file: str = None) -> Dict[str, Any]:
        """智能生成缺失文件，优先保持已有文件不变"""
        try:
            self.log_info("🔧 智能生成缺失文件...")

            from agents.partition_sim import PartitionSim
            from agents.sensor_placement import SensorPlacement

            result = {'success': True}

            # 如果缺少分区文件，调用分区智能体
            if 'partition_results' in missing_files:
                self.log_info("🔧 生成分区配置...")

                partition_agent = PartitionSim()
                partition_result = partition_agent.process(
                    inp_file_path=inp_file_path,
                    user_message="自动分区为3个区域，使用FCM聚类算法",
                    conversation_id=conversation_id
                )

                if partition_result.get('success'):
                    # 查找生成的分区文件
                    partition_file = None
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'partition_results' in filename and
                                filename.endswith('.csv')):
                                partition_file = os.path.join(self.downloads_folder, filename)
                                break

                    if partition_file:
                        result['partition_file'] = partition_file
                        self.log_info("✅ 分区配置生成成功")
                    else:
                        self.log_error("分区配置生成后未找到文件")
                        result['success'] = False
                        result['error'] = "分区配置生成后未找到文件"
                        return result
                else:
                    self.log_error(f"分区配置生成失败: {partition_result.get('response', '未知错误')}")
                    result['success'] = False
                    result['error'] = f"分区配置生成失败: {partition_result.get('response', '未知错误')}"
                    return result

            # 如果缺少传感器文件，调用传感器布置智能体
            if 'sensor_placement' in missing_files:
                self.log_info("🔧 生成传感器配置...")

                # 如果已有传感器文件，说明这是不应该发生的情况
                if existing_sensor_file:
                    self.log_warning("⚠️ 检测到已有传感器文件，但仍在缺失列表中，这可能是逻辑错误")
                    result['sensor_file'] = existing_sensor_file
                    return result

                sensor_agent = SensorPlacement()

                # 确保有分区文件（可能刚刚生成的）
                partition_file = result.get('partition_file')
                if not partition_file:
                    # 重新扫描分区文件
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'partition_results' in filename and
                                filename.endswith('.csv')):
                                partition_file = os.path.join(self.downloads_folder, filename)
                                break

                if not partition_file:
                    self.log_error("传感器布置需要分区文件，但未找到")
                    result['success'] = False
                    result['error'] = "传感器布置需要分区文件，但未找到"
                    return result

                sensor_result = sensor_agent.process(
                    inp_file_path=inp_file_path,
                    partition_csv_path=partition_file,
                    user_message="自动布置传感器，使用遗传算法优化",
                    conversation_id=conversation_id
                )

                if sensor_result.get('success'):
                    # 查找生成的传感器文件
                    sensor_file = None
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'sensor_placement' in filename and
                                filename.endswith('.csv')):
                                sensor_file = os.path.join(self.downloads_folder, filename)
                                break

                    if sensor_file:
                        result['sensor_file'] = sensor_file
                        self.log_info("✅ 传感器配置生成成功")
                    else:
                        self.log_error("传感器配置生成后未找到文件")
                        result['success'] = False
                        result['error'] = "传感器配置生成后未找到文件"
                        return result
                else:
                    self.log_error(f"传感器配置生成失败: {sensor_result.get('response', '未知错误')}")
                    result['success'] = False
                    result['error'] = f"传感器配置生成失败: {sensor_result.get('response', '未知错误')}"
                    return result

            return result

        except Exception as e:
            self.log_error(f"智能生成缺失文件失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_missing_files(self, missing_files: List[str], inp_file_path: str, conversation_id: str) -> Dict[str, Any]:
        """调用其他智能体生成缺失的配置文件（兼容性方法）"""
        self.log_info("⚠️ 使用兼容性方法生成缺失文件，建议使用智能生成方法")
        return self._generate_missing_files_smart(missing_files, inp_file_path, conversation_id)

    def load_network_model(self, inp_file_path: str) -> bool:
        """加载水网络模型"""
        try:
            self.log_info(f"加载水网络模型: {inp_file_path}")
            self.network_model = wntr.network.WaterNetworkModel(inp_file_path)
            
            # 获取网络基本信息
            num_nodes = len(self.network_model.node_name_list)
            num_junctions = len(self.network_model.junction_name_list)
            num_links = len(self.network_model.link_name_list)
            
            self.log_info(f"网络加载成功: {num_nodes}个节点, {num_junctions}个需水节点, {num_links}个管段")
            return True
            
        except Exception as e:
            self.log_error(f"加载网络模型失败: {str(e)}")
            return False
    
    def calculate_centrality(self, demand_nodes: List[str]) -> Dict[str, float]:
        """计算节点的网络中心性"""
        try:
            # 转换为NetworkX图
            G = self.network_model.to_graph().to_undirected()
            
            # 计算各种中心性
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            # 综合中心性分数
            centrality_scores = {}
            for node in demand_nodes:
                if node in G.nodes():
                    score = (
                        degree_centrality.get(node, 0) + 
                        betweenness_centrality.get(node, 0) + 
                        closeness_centrality.get(node, 0)
                    ) / 3
                    centrality_scores[node] = score
                else:
                    centrality_scores[node] = 0
            
            return centrality_scores
            
        except Exception as e:
            self.log_error(f"计算网络中心性失败: {str(e)}")
            return {}
    
    def get_total_demand(self, node_name: str) -> float:
        """获取节点的总需水量"""
        try:
            node = self.network_model.get_node(node_name)
            total_demand = 0
            
            for demand_ts in node.demand_timeseries_list:
                total_demand += abs(demand_ts.base_value)
            
            return total_demand
            
        except Exception as e:
            self.log_error(f"获取节点需水量失败: {str(e)}")
            return 0
    
    def select_critical_nodes(self, num_scenarios: int) -> List[str]:
        """选择关键节点进行漏损模拟"""
        try:
            self.log_info(f"选择 {num_scenarios} 个关键节点进行漏损模拟...")
            
            # 获取需水节点和传感器节点
            demand_nodes = self.network_model.junction_name_list

            # 获取传感器节点，尝试不同的列名
            sensor_nodes = []
            if self.sensor_data is not None:
                if '节点ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['节点ID'].tolist()
                elif '节点名称' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['节点名称'].tolist()
                else:
                    # 如果都没有，尝试第一列
                    sensor_nodes = self.sensor_data.iloc[:, 0].tolist()
            
            # 策略1: 高需水量节点 (按需水量排序，取前50%)
            demand_ranking = sorted(demand_nodes, 
                                  key=lambda x: self.get_total_demand(x), 
                                  reverse=True)
            high_demand_nodes = demand_ranking[:len(demand_ranking)//2]
            
            # 策略2: 网络中心位置节点
            centrality_scores = self.calculate_centrality(demand_nodes)
            central_nodes = sorted(demand_nodes, 
                                 key=lambda x: centrality_scores.get(x, 0), 
                                 reverse=True)[:len(demand_nodes)//2]
            
            # 策略3: 非传感器节点优先
            non_sensor_nodes = [node for node in demand_nodes if node not in sensor_nodes]
            
            # 综合选择：优先选择既是高需水量又是中心位置的非传感器节点
            priority_nodes = list(set(high_demand_nodes) & set(central_nodes) & set(non_sensor_nodes))
            
            # 如果优先节点不够，补充其他关键节点
            if len(priority_nodes) < num_scenarios:
                remaining_critical = list(set(high_demand_nodes + central_nodes) & set(non_sensor_nodes))
                priority_nodes.extend([n for n in remaining_critical if n not in priority_nodes])
            
            # 如果还不够，添加其他非传感器节点
            if len(priority_nodes) < num_scenarios:
                other_nodes = [n for n in non_sensor_nodes if n not in priority_nodes]
                priority_nodes.extend(other_nodes)
            
            selected_nodes = priority_nodes[:num_scenarios]
            
            self.log_info(f"选择了 {len(selected_nodes)} 个关键节点:")
            for i, node in enumerate(selected_nodes[:5]):  # 只显示前5个
                demand = self.get_total_demand(node)
                centrality = centrality_scores.get(node, 0)
                self.log_info(f"  {i+1}. {node} (需水量: {demand:.3f}, 中心性: {centrality:.3f})")
            
            if len(selected_nodes) > 5:
                self.log_info(f"  ... 还有 {len(selected_nodes)-5} 个节点")
            
            return selected_nodes

        except Exception as e:
            error_msg = f"选择关键节点失败: {str(e)}"
            self.log_error(error_msg)
            return []

    def run_hydraulic_simulation(self) -> Optional[wntr.sim.results.SimulationResults]:
        """运行水力仿真"""
        try:
            sim = wntr.sim.EpanetSimulator(self.network_model)
            results = sim.run_sim()
            return results
        except Exception as e:
            self.log_error(f"水力仿真失败: {str(e)}")
            return None

    def get_sensor_pressures(self, results: wntr.sim.results.SimulationResults) -> np.ndarray:
        """提取传感器节点的压力数据"""
        try:
            # 尝试不同的列名
            if '节点ID' in self.sensor_data.columns:
                sensor_nodes = self.sensor_data['节点ID'].tolist()
            elif '节点名称' in self.sensor_data.columns:
                sensor_nodes = self.sensor_data['节点名称'].tolist()
            else:
                # 如果都没有，尝试第一列
                sensor_nodes = self.sensor_data.iloc[:, 0].tolist()

            # 调试信息：显示传感器节点和可用列
            available_columns = list(results.node['pressure'].columns)
            self.log_info(f"传感器节点: {sensor_nodes[:5]}... (共{len(sensor_nodes)}个)")
            self.log_info(f"可用压力列: {available_columns[:10]}... (共{len(available_columns)}个)")

            # 尝试将传感器节点名称转换为字符串格式
            sensor_nodes_str = [str(node) for node in sensor_nodes]

            # 检查哪些传感器节点在仿真结果中存在
            valid_sensors = []
            for sensor in sensor_nodes_str:
                if sensor in available_columns:
                    valid_sensors.append(sensor)
                else:
                    # 尝试不同的格式
                    for col in available_columns:
                        if str(col) == sensor or str(col).strip() == sensor.strip():
                            valid_sensors.append(col)
                            break

            if not valid_sensors:
                self.log_error(f"没有找到匹配的传感器节点")
                self.log_error(f"传感器节点: {sensor_nodes_str}")
                self.log_error(f"可用列样例: {available_columns[:20]}")
                return np.array([])

            self.log_info(f"找到 {len(valid_sensors)} 个有效传感器: {valid_sensors}")

            pressure_data = results.node['pressure'].loc[:, valid_sensors].values
            return pressure_data
        except Exception as e:
            self.log_error(f"提取传感器压力失败: {str(e)}")
            return np.array([])

    def simulate_leak(self, leak_node: str, leak_ratio: float) -> Tuple[np.ndarray, int]:
        """模拟单个节点的漏损场景"""
        try:
            # 保存原始需水量
            node = self.network_model.get_node(leak_node)
            original_demands = []
            for demand_ts in node.demand_timeseries_list:
                original_demands.append(demand_ts.base_value)
                # 增加需水量模拟漏损
                demand_ts.base_value = demand_ts.base_value * (1 + leak_ratio)

            # 运行漏损仿真
            leak_results = self.run_hydraulic_simulation()
            if leak_results is None:
                return np.array([]), 0

            # 获取传感器压力
            leak_pressures = self.get_sensor_pressures(leak_results)

            # 恢复原始需水量
            for i, demand_ts in enumerate(node.demand_timeseries_list):
                demand_ts.base_value = original_demands[i]

            # 确定漏损节点所属分区
            partition_label = self.get_node_partition(leak_node)

            return leak_pressures, partition_label

        except Exception as e:
            self.log_error(f"模拟漏损失败: {str(e)}")
            return np.array([]), 0

    def get_node_partition(self, node_name: str) -> int:
        """获取节点所属分区"""
        try:
            if self.partition_data is not None:
                node_row = self.partition_data[self.partition_data['节点ID'] == node_name]
                if not node_row.empty:
                    return int(node_row.iloc[0]['分区编号'])
            return 1  # 默认分区
        except Exception as e:
            self.log_error(f"获取节点分区失败: {str(e)}")
            return 1

    def calculate_sensitivity_matrix(self, normal_pressures: np.ndarray,
                                   leak_pressures: np.ndarray) -> np.ndarray:
        """计算压力敏感度矩阵"""
        try:
            # 计算压力差的绝对值
            pressure_diff = np.abs(leak_pressures - normal_pressures)

            # 对每个时间步进行归一化（避免除零）
            normalized_diff = np.zeros_like(pressure_diff)

            for t in range(pressure_diff.shape[0]):
                for s in range(pressure_diff.shape[1]):
                    if normal_pressures[t, s] > 1e-6:  # 避免除零
                        normalized_diff[t, s] = pressure_diff[t, s] / normal_pressures[t, s]
                    else:
                        normalized_diff[t, s] = 0

            # 计算时间平均值
            sensitivity_vector = np.mean(normalized_diff, axis=0)

            return sensitivity_vector

        except Exception as e:
            self.log_error(f"计算敏感度矩阵失败: {str(e)}")
            return np.array([])

    def add_sensor_noise(self, pressure_data: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """添加传感器噪声"""
        try:
            # 高斯噪声：均值为0，标准差为压力值的百分比
            noise = np.random.normal(0, pressure_data * noise_level)

            # 确保压力值不为负
            noisy_pressure = np.maximum(pressure_data + noise, 0.1)

            return noisy_pressure

        except Exception as e:
            self.log_error(f"添加传感器噪声失败: {str(e)}")
            return pressure_data

    def generate_training_data(self, num_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成平衡的训练数据集"""
        try:
            self.log_info(f"开始生成 {num_scenarios*2} 个训练样本...")
            self.log_info(f"  - {num_scenarios} 个异常样本 (漏损场景)")
            self.log_info(f"  - {num_scenarios} 个正常样本 (含噪声)")

            # 运行基准仿真
            self.log_info("运行基准水力仿真...")
            normal_results = self.run_hydraulic_simulation()
            if normal_results is None:
                raise Exception("基准仿真失败")

            normal_pressures = self.get_sensor_pressures(normal_results)
            self.log_info(f"基准压力数据形状: {normal_pressures.shape}")

            # 选择关键节点
            critical_nodes = self.select_critical_nodes(num_scenarios)
            if len(critical_nodes) < num_scenarios:
                self.log_warning(f"只找到 {len(critical_nodes)} 个关键节点，少于请求的 {num_scenarios} 个")
                num_scenarios = len(critical_nodes)

            # 生成异常数据
            self.log_info("生成异常数据...")
            anomaly_data = []
            anomaly_labels = []

            # 为了确保有足够的数据，对每个关键节点生成多个漏损场景
            scenarios_per_node = max(1, num_scenarios // len(critical_nodes))
            if scenarios_per_node * len(critical_nodes) < num_scenarios:
                scenarios_per_node += 1

            scenario_count = 0
            for node_idx, leak_node in enumerate(critical_nodes):
                for scenario_idx in range(scenarios_per_node):
                    if scenario_count >= num_scenarios:
                        break

                    scenario_count += 1
                    self.log_info(f"  模拟漏损 {scenario_count}/{num_scenarios}: {leak_node} (场景{scenario_idx+1})")

                    # 随机漏损比例 (10%-30%)
                    leak_ratio = random.uniform(0.1, 0.3)

                    # 模拟漏损
                    leak_pressures, partition_label = self.simulate_leak(leak_node, leak_ratio)

                    if leak_pressures.size > 0:
                        # 计算敏感度向量
                        sensitivity_vector = self.calculate_sensitivity_matrix(normal_pressures, leak_pressures)

                        if sensitivity_vector.size > 0:
                            anomaly_data.append(sensitivity_vector)
                            anomaly_labels.append(partition_label)
                            self.log_info(f"    漏损比例: {leak_ratio:.1%}, 分区: {partition_label}")

                if scenario_count >= num_scenarios:
                    break

            # 生成正常数据
            self.log_info("生成正常数据...")
            normal_data = []
            normal_labels = []

            for i in range(len(anomaly_data)):  # 生成同等数量的正常数据
                # 添加不同水平的噪声
                noise_level = random.uniform(0.01, 0.03)
                noisy_pressures = self.add_sensor_noise(normal_pressures, noise_level)

                # 计算"敏感度"（实际上是噪声向量）
                noise_vector = self.calculate_sensitivity_matrix(normal_pressures, noisy_pressures)

                if noise_vector.size > 0:
                    normal_data.append(noise_vector)
                    normal_labels.append(0)  # 正常标签为0

            # 合并数据
            all_data = np.array(anomaly_data + normal_data)
            all_labels = np.array(anomaly_labels + normal_labels)

            self.log_info(f"数据生成完成:")
            self.log_info(f"  总样本数: {len(all_data)}")
            self.log_info(f"  特征维度: {all_data.shape[1] if len(all_data) > 0 else 0}")
            self.log_info(f"  正常样本: {np.sum(all_labels == 0)}")
            self.log_info(f"  异常样本: {np.sum(all_labels > 0)}")

            # 详细的标签统计
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            self.log_info(f"  标签分布: {dict(zip(unique_labels, counts))}")
            self.log_info(f"  标签范围: [{np.min(all_labels)}, {np.max(all_labels)}]")

            # 修复：不进行标签重映射，直接使用原始分区编号
            # 标签0=正常，标签N=分区N漏损，保持分区编号与标签的直接对应关系
            self.log_info(f"  保持原始标签: 0=正常，1-{np.max(unique_labels[unique_labels > 0]) if len(unique_labels[unique_labels > 0]) > 0 else 0}=对应分区漏损")
            self.log_info(f"  最终标签范围: [{np.min(all_labels)}, {np.max(all_labels)}]")

            return all_data, all_labels

        except Exception as e:
            error_msg = f"生成训练数据失败: {str(e)}"
            self.log_error(error_msg)
            return np.array([]), np.array([])

    def prepare_datasets(self, data: np.ndarray, labels: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练、验证、测试数据集"""
        try:
            # 数据标准化
            data_scaled = self.scaler.fit_transform(data)

            # 检查数据集大小和类别分布
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            min_samples_per_class = np.min(label_counts)
            total_samples = len(data)

            self.log_info(f"数据集分析:")
            self.log_info(f"  总样本数: {total_samples}")
            self.log_info(f"  类别数: {len(unique_labels)}")
            self.log_info(f"  最少样本类别: {min_samples_per_class} 个样本")

            # 如果数据集太小或某些类别样本太少，使用简单分割
            if total_samples < 10 or min_samples_per_class < 2:
                self.log_warning("数据集较小，使用简单分割策略")

                # 简单分割：80% 训练，20% 验证，不设置测试集
                if total_samples >= 5:
                    split_idx = int(0.8 * total_samples)
                    X_train = data_scaled[:split_idx]
                    y_train = labels[:split_idx]
                    X_val = data_scaled[split_idx:]
                    y_val = labels[split_idx:]
                    X_test = X_val  # 验证集同时作为测试集
                    y_test = y_val
                else:
                    # 数据太少，全部用于训练
                    X_train = data_scaled
                    y_train = labels
                    X_val = data_scaled
                    y_val = labels
                    X_test = data_scaled
                    y_test = labels
            else:
                # 正常分层分割
                X_temp, X_test, y_temp, y_test = train_test_split(
                    data_scaled, labels, test_size=0.1, random_state=42, stratify=labels
                )

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
                )

            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)

            # 创建数据加载器，调整batch_size
            batch_size = min(8, len(X_train))  # 小数据集使用小batch_size

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            self.log_info(f"数据集准备完成:")
            self.log_info(f"  训练集: {len(X_train)} 样本")
            self.log_info(f"  验证集: {len(X_val)} 样本")
            self.log_info(f"  测试集: {len(X_test)} 样本")
            self.log_info(f"  批次大小: {batch_size}")

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_msg = f"准备数据集失败: {str(e)}"
            self.log_error(error_msg)
            return None, None, None

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   input_size: int, num_partitions: int, epochs: int = 100, num_classes: int = None) -> Dict[str, Any]:
        """训练漏损检测模型"""
        try:
            self.log_info(f"开始训练漏损检测模型...")
            self.log_info(f"  输入维度: {input_size}")
            self.log_info(f"  分区数量: {num_partitions}")
            self.log_info(f"  训练轮数: {epochs}")

            # 创建模型 - 使用正确的类别数
            if num_classes is None:
                num_classes = num_partitions + 1  # 默认：分区数+1（正常类）

            self.log_info(f"  模型类别数: {num_classes}")
            self.model = LeakDetectionMLP(input_size, num_partitions, num_classes=num_classes).to(self.device)

            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

            # 训练历史
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            best_val_acc = 0
            best_model_state = None

            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()

                # 验证阶段
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                        outputs = self.model(batch_data)
                        loss = criterion(outputs, batch_labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()

                # 计算准确率
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total

                # 记录历史
                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()

                # 学习率调度
                scheduler.step()

                # 每10轮打印一次
                if (epoch + 1) % 10 == 0:
                    self.log_info(f"  Epoch {epoch+1}/{epochs}: "
                                f"Train Loss: {train_losses[-1]:.4f}, "
                                f"Train Acc: {train_acc:.2f}%, "
                                f"Val Loss: {val_losses[-1]:.4f}, "
                                f"Val Acc: {val_acc:.2f}%")

            # 加载最佳模型
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            # 确保所有数据都是JSON可序列化的Python原生类型
            training_history = {
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses],
                'train_accuracies': [float(x) for x in train_accuracies],
                'val_accuracies': [float(x) for x in val_accuracies],
                'best_val_accuracy': float(best_val_acc),
                'final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
                'final_val_loss': float(val_losses[-1]) if val_losses else 0.0
            }

            self.log_info(f"模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")

            return training_history

        except Exception as e:
            error_msg = f"模型训练失败: {str(e)}"
            self.log_error(error_msg)
            return {}

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """评估模型性能"""
        try:
            self.log_info("开始评估模型性能...")

            if self.model is None:
                raise Exception("模型未训练")

            self.model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                    outputs = self.model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

            # 计算评估指标
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )

            # 混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)

            # 确保所有数据都是JSON可序列化的Python原生类型
            evaluation_results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'predictions': [int(x) for x in all_predictions],
                'true_labels': [int(x) for x in all_labels]
            }

            self.log_info(f"模型评估完成:")
            self.log_info(f"  准确率 (Accuracy): {accuracy:.4f}")
            self.log_info(f"  精确率 (Precision): {precision:.4f}")
            self.log_info(f"  召回率 (Recall): {recall:.4f}")
            self.log_info(f"  F1分数 (F1-Score): {f1:.4f}")

            return evaluation_results

        except Exception as e:
            error_msg = f"模型评估失败: {str(e)}"
            self.log_error(error_msg)
            return {}

    def save_model(self, conversation_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """保存训练好的模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"leak_detection_model_{conversation_id[:8]}_{timestamp}.pth"
            model_path = os.path.join(self.downloads_folder, model_filename)

            # 保存模型状态，确保所有数据都是可序列化的
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'scaler_mean': [float(x) for x in self.scaler.mean_],
                'scaler_scale': [float(x) for x in self.scaler.scale_],
                'input_size': int(model_info['input_size']),
                'num_partitions': int(model_info['num_partitions']),
                'num_classes': int(model_info.get('num_classes', model_info['num_partitions'] + 1)),  # 保存实际类别数
                'max_partition': int(model_info.get('max_partition', model_info['num_partitions'])),  # 保存最大分区编号
                'model_info': model_info,
                'timestamp': timestamp
            }

            torch.save(model_state, model_path)

            file_size = os.path.getsize(model_path)

            self.log_info(f"模型已保存: {model_filename} ({file_size} 字节)")

            return {
                'success': True,
                'filename': model_filename,
                'file_path': model_path,
                'file_size': file_size,
                'download_url': f'/download/{model_filename}'
            }

        except Exception as e:
            error_msg = f"保存模型失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def load_model(self, model_path: str) -> bool:
        """加载训练好的模型"""
        try:
            self.log_info(f"加载模型: {model_path}")

            # 加载模型状态，设置weights_only=False以兼容旧版本
            try:
                model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # 兼容旧版本PyTorch
                model_state = torch.load(model_path, map_location=self.device)

            # 重建模型 - 使用保存的实际类别数
            input_size = model_state['input_size']
            num_partitions = model_state['num_partitions']

            # 优先使用保存的类别数，否则使用传统计算方式
            num_classes = model_state.get('num_classes', num_partitions + 1)
            max_partition = model_state.get('max_partition', num_partitions)

            self.model = LeakDetectionMLP(input_size, num_partitions, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(model_state['model_state_dict'])

            # 重建标准化器
            self.scaler.mean_ = np.array(model_state['scaler_mean'])
            self.scaler.scale_ = np.array(model_state['scaler_scale'])

            self.log_info(f"模型加载成功: 输入维度={input_size}, 最大分区编号={max_partition}, 类别数={num_classes}")
            self.log_info("注意: 实际推理时将使用当前对话的分区配置")

            return True

        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            self.log_error(error_msg)
            return False

    def predict_leak(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """预测漏损情况"""
        try:
            if self.model is None:
                raise Exception("模型未加载")

            self.log_info(f"开始漏损检测，输入数据形状: {sensor_data.shape}")

            # 数据预处理
            if len(sensor_data.shape) == 1:
                sensor_data = sensor_data.reshape(1, -1)

            # 标准化
            sensor_data_scaled = self.scaler.transform(sensor_data)

            # 转换为张量
            input_tensor = torch.FloatTensor(sensor_data_scaled).to(self.device)

            # 预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

            # 解析结果
            predictions = predicted.cpu().numpy()
            probs = probabilities.cpu().numpy()

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probs)):
                # 确保转换为Python原生类型，避免JSON序列化错误
                pred_int = int(pred)
                confidence = float(prob[pred_int])

                if pred_int == 0:
                    status = "正常"
                    partition = None
                else:
                    status = "异常"
                    partition = pred_int

                results.append({
                    'sample_id': int(i + 1),
                    'status': status,
                    'partition': partition,
                    'confidence': confidence,
                    'probabilities': [float(p) for p in prob]  # 确保所有概率都是float类型
                })

            self.log_info(f"漏损检测完成，检测到 {len(results)} 个样本")

            return {
                'success': True,
                'results': results,
                'summary': {
                    'total_samples': int(len(results)),
                    'normal_samples': int(sum(1 for r in results if r['status'] == '正常')),
                    'anomaly_samples': int(sum(1 for r in results if r['status'] == '异常'))
                }
            }

        except Exception as e:
            error_msg = f"漏损预测失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def save_training_data(self, data: np.ndarray, labels: np.ndarray,
                          conversation_id: str) -> Dict[str, Any]:
        """保存训练数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leak_training_data_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # 准备数据
            df_data = []
            # 修复列名问题
            if self.sensor_data is not None:
                if '节点ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['节点ID'].tolist()
                elif '节点名称' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['节点名称'].tolist()
                else:
                    sensor_nodes = self.sensor_data.iloc[:, 0].tolist()
            else:
                sensor_nodes = []

            for i, (sample, label) in enumerate(zip(data, labels)):
                # 确保转换为Python原生类型，避免JSON序列化错误
                row = {'样本ID': int(i + 1), '标签': int(label)}

                # 添加传感器数据
                for j, sensor in enumerate(sensor_nodes):
                    if j < len(sample):
                        row[f'传感器_{sensor}'] = float(sample[j])

                df_data.append(row)

            # 保存为CSV
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            file_size = os.path.getsize(filepath)

            self.log_info(f"训练数据已保存: {filename} ({file_size} 字节)")

            return {
                'success': True,
                'filename': filename,
                'file_path': filepath,
                'file_size': file_size,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"保存训练数据失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def save_evaluation_report(self, evaluation_results: Dict[str, Any],
                              training_history: Dict[str, Any],
                              conversation_id: str) -> Dict[str, Any]:
        """保存评估报告"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leak_evaluation_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # 准备报告数据
            report_data = []

            # 基本指标
            report_data.append({
                '指标': '准确率 (Accuracy)',
                '数值': f"{evaluation_results.get('accuracy', 0):.4f}",
                '说明': '正确预测的样本比例'
            })

            report_data.append({
                '指标': '精确率 (Precision)',
                '数值': f"{evaluation_results.get('precision', 0):.4f}",
                '说明': '预测为正例中实际为正例的比例'
            })

            report_data.append({
                '指标': '召回率 (Recall)',
                '数值': f"{evaluation_results.get('recall', 0):.4f}",
                '说明': '实际正例中被正确预测的比例'
            })

            report_data.append({
                '指标': 'F1分数 (F1-Score)',
                '数值': f"{evaluation_results.get('f1_score', 0):.4f}",
                '说明': '精确率和召回率的调和平均'
            })

            # 训练信息
            if training_history:
                report_data.append({
                    '指标': '最佳验证准确率',
                    '数值': f"{training_history.get('best_val_accuracy', 0):.2f}%",
                    '说明': '训练过程中的最佳验证准确率'
                })

            # 保存为CSV
            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            file_size = os.path.getsize(filepath)

            self.log_info(f"评估报告已保存: {filename} ({file_size} 字节)")

            return {
                'success': True,
                'filename': filename,
                'file_path': filepath,
                'file_size': file_size,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"保存评估报告失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def train_leak_detection_model(self, inp_file_path: str, conversation_id: str,
                                  num_scenarios: int = 50, epochs: int = 100) -> Dict[str, Any]:
        """训练漏损检测模型的主接口"""
        try:
            self.log_info("=" * 60)
            self.log_info("开始训练漏损检测模型")
            self.log_info("=" * 60)

            # 1. 智能检查依赖文件
            dependency_check = self.check_dependencies(conversation_id, inp_file_path)
            if not dependency_check.get('success'):
                return dependency_check

            # 显示智能复用信息
            if dependency_check.get('reused_files'):
                reused_files = dependency_check.get('reused_files', [])
                self.log_info("🎯 智能工作流优化:")
                for reused_file in reused_files:
                    if reused_file == 'sensor_placement':
                        self.log_info("   ✅ 复用已有传感器布置，跳过传感器布置步骤")
                    elif reused_file == 'partition_results':
                        self.log_info("   ✅ 复用已有分区结果，跳过分区分析步骤")
                    elif reused_file == 'partition_results_inferred':
                        self.log_info("   ✅ 从传感器文件推断分区信息，跳过分区分析步骤")
                self.log_info("   ⚡ 大幅提升训练效率，直接进入模型训练阶段")

            # 2. 加载网络模型
            if not self.load_network_model(inp_file_path):
                return {'success': False, 'error': '加载网络模型失败'}

            # 3. 生成训练数据
            data, labels = self.generate_training_data(num_scenarios)
            if len(data) == 0:
                return {'success': False, 'error': '生成训练数据失败'}

            # 4. 准备数据集
            train_loader, val_loader, test_loader = self.prepare_datasets(data, labels)
            if train_loader is None:
                return {'success': False, 'error': '准备数据集失败'}

            # 5. 训练模型
            # 获取所有唯一标签并确保标签范围正确
            unique_labels = np.unique(labels)
            max_label = int(np.max(unique_labels))
            min_label = int(np.min(unique_labels))

            self.log_info(f"标签统计: 最小值={min_label}, 最大值={max_label}, 唯一值={unique_labels}")

            # 检查标签是否连续且从0开始
            expected_labels = list(range(min_label, max_label + 1))
            if not all(label in unique_labels for label in expected_labels):
                self.log_warning(f"标签不连续，可能导致训练问题")

            # 模型的类别数应该是最大标签值+1（因为标签从0开始）
            num_classes = max_label + 1

            # 修复：分区数应该是最大分区编号，而不是分区种类数
            # 因为分区编号可能不连续（如1,2,3,4,5,6），而不是从1开始的连续编号
            max_partition = np.max(unique_labels[unique_labels > 0]) if len(unique_labels[unique_labels > 0]) > 0 else 0
            num_partitions = max_partition  # 使用最大分区编号作为分区数
            input_size = data.shape[1]

            self.log_info(f"模型配置: 输入维度={input_size}, 最大分区编号={max_partition}, 类别数={num_classes}")

            # 重新计算标签分布用于日志
            unique_labels_with_counts, counts = np.unique(labels, return_counts=True)
            self.log_info(f"标签分布: {dict(zip(unique_labels_with_counts, counts))}")

            # 最终安全检查：确保所有标签都在[0, num_classes-1]范围内
            if np.any(labels < 0) or np.any(labels >= num_classes):
                error_msg = f"标签超出范围 [0, {num_classes-1}]: 实际范围 [{np.min(labels)}, {np.max(labels)}]"
                self.log_error(error_msg)
                return {'success': False, 'error': error_msg}

            training_history = self.train_model(train_loader, val_loader, input_size, num_partitions, epochs, num_classes)
            if not training_history:
                return {'success': False, 'error': '模型训练失败'}

            # 6. 评估模型
            evaluation_results = self.evaluate_model(test_loader)
            if not evaluation_results:
                return {'success': False, 'error': '模型评估失败'}

            # 7. 保存模型和结果
            # 确保所有数据都是JSON可序列化的Python原生类型
            model_info = {
                'input_size': int(input_size),
                'num_partitions': int(num_partitions),
                'num_scenarios': int(num_scenarios),
                'epochs': int(epochs),
                'evaluation': evaluation_results,
                'training_history': training_history
            }

            # 更新模型信息，包含正确的分区数和类别数
            model_info.update({
                'max_partition': int(max_partition),
                'num_classes': int(num_classes)
            })

            model_save_result = self.save_model(conversation_id, model_info)
            training_data_result = self.save_training_data(data, labels, conversation_id)
            evaluation_report_result = self.save_evaluation_report(evaluation_results, training_history, conversation_id)

            self.log_info("=" * 60)
            self.log_info("漏损检测模型训练完成")
            self.log_info("=" * 60)

            return {
                'success': True,
                'model_info': model_info,
                'evaluation': evaluation_results,
                'training_history': training_history,
                'files': {
                    'model': model_save_result,
                    'training_data': training_data_result,
                    'evaluation_report': evaluation_report_result
                }
            }

        except Exception as e:
            error_msg = f"训练漏损检测模型失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def detect_leak_from_file(self, sensor_file_path: str, model_file_path: str, conversation_id: str = None) -> Dict[str, Any]:
        """从文件读取传感器数据并进行漏损检测"""
        try:
            self.log_info("=" * 60)
            self.log_info("开始漏损检测")
            self.log_info("=" * 60)

            # 1. 加载模型
            if not self.load_model(model_file_path):
                return {'success': False, 'error': '加载模型失败'}

            # 2. 读取分区文件获取实际分区数
            actual_num_partitions = None
            if conversation_id:
                partition_file = self._find_partition_file(conversation_id)
                if partition_file:
                    try:
                        partition_df = pd.read_csv(partition_file)
                        # 获取实际分区数
                        if '分区编号' in partition_df.columns:
                            actual_num_partitions = partition_df['分区编号'].max()
                        elif 'Partition' in partition_df.columns:
                            actual_num_partitions = partition_df['Partition'].max()

                        if actual_num_partitions:
                            self.log_info(f"从分区文件读取实际分区数: {actual_num_partitions}")
                            # 更新模型的分区数信息（用于结果解释）
                            self._actual_num_partitions = actual_num_partitions
                        else:
                            self.log_warning("无法从分区文件确定分区数，使用模型默认值")
                    except Exception as e:
                        self.log_warning(f"读取分区文件失败: {str(e)}，使用模型默认分区数")

            # 3. 读取传感器数据
            self.log_info(f"读取传感器数据: {sensor_file_path}")

            try:
                sensor_df = pd.read_csv(sensor_file_path)
                self.log_info(f"传感器数据形状: {sensor_df.shape}")

                # 提取数值数据（排除ID列等）
                numeric_columns = sensor_df.select_dtypes(include=[np.number]).columns
                sensor_data = sensor_df[numeric_columns].values

                if sensor_data.size == 0:
                    return {'success': False, 'error': '传感器文件中没有数值数据'}

            except Exception as e:
                return {'success': False, 'error': f'读取传感器文件失败: {str(e)}'}

            # 4. 进行预测
            prediction_results = self.predict_leak(sensor_data)
            if not prediction_results['success']:
                return prediction_results

            self.log_info("=" * 60)
            self.log_info("漏损检测完成")
            self.log_info("=" * 60)

            return prediction_results

        except Exception as e:
            error_msg = f"漏损检测失败: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def _find_partition_file(self, conversation_id: str) -> str:
        """查找对话对应的分区文件"""
        try:
            if not os.path.exists(self.downloads_folder):
                return None

            # 查找分区文件
            for filename in os.listdir(self.downloads_folder):
                if (conversation_id[:8] in filename and
                    'partition_results' in filename and
                    filename.endswith('.csv')):
                    partition_file = os.path.join(self.downloads_folder, filename)
                    self.log_info(f"找到分区文件: {os.path.basename(partition_file)}")
                    return partition_file

            self.log_warning(f"未找到对话 {conversation_id[:8]} 的分区文件")
            return None

        except Exception as e:
            self.log_error(f"查找分区文件失败: {str(e)}")
            return None

    def build_response_prompt(self, result: Dict[str, Any], user_message: str,
                             operation_type: str) -> str:
        """构建响应prompt"""
        try:
            if operation_type == "training":
                return self._build_training_prompt(result, user_message)
            elif operation_type == "detection":
                return self._build_detection_prompt(result, user_message)
            else:
                return "操作完成。"

        except Exception as e:
            self.log_error(f"构建响应prompt失败: {str(e)}")
            return "操作完成，但生成响应时出现错误。"

    def _build_training_prompt(self, result: Dict[str, Any], user_message: str) -> str:
        """构建训练响应prompt"""
        if not result.get('success', False):
            return f"""
漏损检测模型训练失败。

错误信息：{result.get('error', '未知错误')}

请检查以下可能的问题：
1. 是否已完成管网分区分析
2. 是否已完成传感器布置
3. 网络文件是否正确
4. 系统资源是否充足

用户请求：{user_message}
"""

        model_info = result.get('model_info', {})
        evaluation = result.get('evaluation', {})
        training_history = result.get('training_history', {})
        files = result.get('files', {})

        # 计算详细统计信息
        total_samples = model_info.get('num_scenarios', 0) * 2  # 正常+异常样本
        normal_samples = model_info.get('num_scenarios', 0)
        anomaly_samples = model_info.get('num_scenarios', 0)

        # 获取分区统计
        num_partitions = model_info.get('num_partitions', 0)
        samples_per_partition = anomaly_samples // max(num_partitions, 1) if num_partitions > 0 else 0

        # 构建性能指标说明
        accuracy = evaluation.get('accuracy', 0)
        precision = evaluation.get('precision', 0)
        recall = evaluation.get('recall', 0)
        f1_score = evaluation.get('f1_score', 0)

        # 安全获取训练历史数据
        final_train_loss = training_history.get('final_train_loss', 0)
        final_val_loss = training_history.get('final_val_loss', 0)
        best_val_accuracy = training_history.get('best_val_accuracy', 0)

        # 性能评级
        def get_performance_grade(score):
            if score >= 0.9: return "优秀 🌟"
            elif score >= 0.8: return "良好 ✅"
            elif score >= 0.7: return "一般 ⚠️"
            else: return "需改进 ❌"

        return f"""
🎉 漏损检测模型训练成功完成！

## 📊 训练数据统计
- **总样本数**: {total_samples} 个 (平衡数据集)
- **正常样本**: {normal_samples} 个 (包含传感器噪声)
- **异常样本**: {anomaly_samples} 个 (分布在 {num_partitions} 个分区)
- **每分区样本**: 约 {samples_per_partition} 个漏损场景
- **传感器数量**: {model_info.get('input_size', 'N/A')} 个
- **训练轮数**: {model_info.get('epochs', 'N/A')} 轮

## 📈 模型性能评估
- **准确率 (Accuracy)**: {accuracy:.4f} ({accuracy*100:.2f}%) - {get_performance_grade(accuracy)}
- **精确率 (Precision)**: {precision:.4f} ({precision*100:.2f}%) - {get_performance_grade(precision)}
- **召回率 (Recall)**: {recall:.4f} ({recall*100:.2f}%) - {get_performance_grade(recall)}
- **F1分数 (F1-Score)**: {f1_score:.4f} ({f1_score*100:.2f}%) - {get_performance_grade(f1_score)}

### 📋 性能指标说明
- **准确率**: 所有预测中正确的比例 (包括正常和异常)
- **精确率**: 预测为异常的样本中真正异常的比例
- **召回率**: 真正异常的样本中被正确识别的比例
- **F1分数**: 精确率和召回率的调和平均数

## 🎯 训练过程
- **最终训练损失**: {final_train_loss:.6f}
- **最终验证损失**: {final_val_loss:.6f}
- **最佳验证准确率**: {best_val_accuracy:.4f}

## 📁 生成文件
以下文件已生成并可供下载：

### 🤖 模型文件
- **文件名**: `{files.get('model', {}).get('filename', 'N/A')}`
- **格式**: PyTorch PTH格式
- **用途**: 用于漏损检测推理

### 📊 训练数据文件
- **文件名**: `{files.get('training_data', {}).get('filename', 'N/A')}`
- **格式**: CSV格式
- **内容**: 包含训练用的传感器压力数据和标签

### 📈 评估报告文件
- **文件名**: `{files.get('evaluation_report', {}).get('filename', 'N/A')}`
- **格式**: CSV格式
- **内容**: 详细的模型性能评估指标和混淆矩阵

## 🚀 下一步操作
模型已准备就绪！您现在可以：
1. **下载模型文件**：点击下方的PTH文件下载按钮
2. **进行漏损检测**：上传传感器压力数据CSV文件
3. **查看详细报告**：下载评估报告了解更多性能细节

用户请求：{user_message}

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

    def _build_detection_prompt(self, result: Dict[str, Any], user_message: str) -> str:
        """构建检测响应prompt"""
        if not result.get('success', False):
            return f"""
漏损检测失败。

错误信息：{result.get('error', '未知错误')}

请检查以下可能的问题：
1. 传感器数据文件格式是否正确
2. 模型文件是否存在且有效
3. 数据维度是否匹配

用户请求：{user_message}
"""

        results = result.get('results', [])
        summary = result.get('summary', {})

        # 统计异常情况和置信度
        anomaly_partitions = {}
        normal_confidences = []
        all_confidences = []

        for r in results:
            all_confidences.append(r['confidence'])
            if r['status'] == '异常':
                partition = r['partition']
                if partition not in anomaly_partitions:
                    anomaly_partitions[partition] = []
                anomaly_partitions[partition].append(r)
            else:
                normal_confidences.append(r['confidence'])

        # 计算统计信息
        total_samples = summary.get('total_samples', 0)
        normal_samples = summary.get('normal_samples', 0)
        anomaly_samples = summary.get('anomaly_samples', 0)

        normal_percentage = (normal_samples / total_samples * 100) if total_samples > 0 else 0
        anomaly_percentage = (anomaly_samples / total_samples * 100) if total_samples > 0 else 0

        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        avg_normal_confidence = np.mean(normal_confidences) if normal_confidences else 0

        prompt = f"""
🎯 **智能漏损检测推理完成**

✅ **推理模式说明**：系统检测到已有训练好的漏损检测模型，直接进行推理分析，无需重复执行分区、传感器布置、模型训练等步骤。

## 📊 检测概况
- **分析样本数**: {total_samples} 个时间点
- **正常状态**: {normal_samples} 个样本 ({normal_percentage:.1f}%)
- **异常状态**: {anomaly_samples} 个样本 ({anomaly_percentage:.1f}%)
- **平均检测置信度**: {avg_confidence:.3f}

## 📈 置信度分析
- **正常状态平均置信度**: {avg_normal_confidence:.3f}
- **整体检测可靠性**: {'高' if avg_confidence > 0.8 else '中等' if avg_confidence > 0.6 else '较低'}

"""

        if anomaly_partitions:
            prompt += "## ⚠️ 检测到漏损异常\n"
            for partition, samples in anomaly_partitions.items():
                avg_confidence = np.mean([s['confidence'] for s in samples])
                max_confidence = max([s['confidence'] for s in samples])
                min_confidence = min([s['confidence'] for s in samples])

                prompt += f"""
### 🚨 分区 {partition} 漏损警报
- **异常样本数**: {len(samples)} 个
- **平均置信度**: {avg_confidence:.3f}
- **最高置信度**: {max_confidence:.3f}
- **最低置信度**: {min_confidence:.3f}
- **严重程度**: {'高' if avg_confidence > 0.8 else '中等' if avg_confidence > 0.6 else '低'}
"""

            prompt += "\n## 🔧 建议措施\n"
            prompt += "1. **立即检查**：对检测到异常的分区进行现场检查\n"
            prompt += "2. **确认漏损**：使用其他检测手段验证漏损位置\n"
            prompt += "3. **制定修复计划**：根据漏损严重程度安排维修\n"
            prompt += "4. **持续监控**：加强对异常分区的监控频率\n"
        else:
            prompt += "## ✅ 未检测到漏损异常\n"
            prompt += f"所有 {total_samples} 个时间点的传感器数据均显示管网运行正常。\n"
            prompt += f"平均检测置信度为 {avg_normal_confidence:.3f}，系统运行稳定。\n"

        prompt += f"""

## 📋 详细结果
"""

        for i, r in enumerate(results[:5]):  # 只显示前5个结果
            status_icon = "✅" if r['status'] == '正常' else "⚠️"
            prompt += f"- 样本 {r['sample_id']}: {status_icon} {r['status']}"
            if r['partition']:
                prompt += f" (分区 {r['partition']})"
            prompt += f" - 置信度: {r['confidence']:.3f}\n"

        if len(results) > 5:
            prompt += f"... 还有 {len(results)-5} 个样本\n"

        prompt += f"""

用户请求：{user_message}

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

        return prompt

    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """实现BaseAgent的抽象方法process"""
        # 这个方法主要用于兼容BaseAgent接口
        # 实际的处理逻辑在train_leak_detection_model和detect_leak_from_file中
        return {
            'success': True,
            'message': '漏损检测智能体已就绪。请使用train_leak_detection_model进行训练或detect_leak_from_file进行检测。'
        }
