"""
SensorPlacement 传感器优化布置智能体
负责基于管网分区结果进行传感器优化布置，考虑韧性和检测效果
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from .base_agent import BaseAgent
from .partition_sim import PartitionSim

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

class SensorPlacement(BaseAgent):
    """传感器优化布置智能体"""
    
    def __init__(self):
        super().__init__("SensorPlacement")
        
        if not WNTR_AVAILABLE:
            self.log_error("WNTR库未安装，传感器布置功能不可用")
        
        self.partition_sim = PartitionSim()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)
        
        # 默认参数（优化版本，减少计算量）
        self.default_params = {
            'demand_ratios': [0.20],  # 减少扰动比例，只使用一个
            'sensitivity_threshold': 0.5,  # 敏感度阈值
            'max_failure_rate': 0.8,  # 最大故障率（允许更多传感器故障）
            'resilience_weight': 0.4,  # 韧性权重
            'coverage_weight': 0.6,  # 覆盖率权重
            'target_coverage': 0.95,  # 目标覆盖率
            'min_sensor_ratio': 0.04,  # 最小传感器比例（节点数的4%）
            'max_sensor_ratio': 0.15,  # 最大传感器比例（节点数的15%）
            'custom_thresholds': {},  # 自定义传感器阈值 {sensor_node: threshold}
            'enable_custom_thresholds': False  # 是否启用自定义阈值
        }
    
    def load_partition_results(self, csv_file_path):
        """从CSV文件加载分区结果"""
        try:
            self.log_info(f"加载分区结果: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            
            # 提取需水节点的分区信息
            demand_nodes_df = df[df['节点类型'] == '需水节点']
            
            partitions = {}
            for _, row in demand_nodes_df.iterrows():
                partition_id = row['分区编号']
                node_id = row['节点ID']
                
                if partition_id > 0:  # 排除未分配节点
                    if partition_id not in partitions:
                        partitions[partition_id] = []
                    partitions[partition_id].append(node_id)
            
            self.log_info(f"成功加载分区结果: {len(partitions)}个分区")
            for partition_id, nodes in partitions.items():
                self.log_info(f"  分区{partition_id}: {len(nodes)}个节点")
            
            return partitions
            
        except Exception as e:
            error_msg = f"加载分区结果失败: {str(e)}"
            self.log_error(error_msg)
            return None
    
    def compute_pressure_sensitivity_matrix(self, inp_file_path, partitions, demand_ratios):
        """计算压力敏感度矩阵"""
        try:
            self.log_info(f"开始计算压力敏感度矩阵，扰动比例: {demand_ratios}")

            # 加载网络模型
            wn = wntr.network.WaterNetworkModel(inp_file_path)

            # 获取所有需水节点
            all_demand_nodes = []
            for nodes in partitions.values():
                all_demand_nodes.extend(nodes)

            self.log_info(f"需水节点总数: {len(all_demand_nodes)}")

            # 检查节点是否存在于网络中
            valid_demand_nodes = []
            for node in all_demand_nodes:
                if node in wn.node_name_list:
                    valid_demand_nodes.append(node)
                else:
                    self.log_info(f"节点 {node} 不存在于网络中，跳过")

            if not valid_demand_nodes:
                raise ValueError("没有找到有效的需水节点")

            self.log_info(f"有效需水节点数: {len(valid_demand_nodes)}")

            # 运行基准仿真
            self.log_info("运行基准仿真...")
            sim = wntr.sim.EpanetSimulator(wn)
            base_results = sim.run_sim()
            base_pressure = base_results.node['pressure'].loc[:, valid_demand_nodes].values

            # 计算总实际需水量（参考cluster_simple.py的做法）
            self.log_info("计算总需水量...")
            total_demand = 0
            for name in valid_demand_nodes:
                # 获取该节点在所有时间步的实际需水量
                node_demands = base_results.node['demand'].loc[:, name]
                # 累加该节点的所有时间步需水量
                total_demand += node_demands.sum()

            self.log_info(f"总需水量: {total_demand:.4f}")

            # 初始化敏感度矩阵
            n_nodes = len(valid_demand_nodes)
            sensitivity_matrix = np.zeros((n_nodes, n_nodes))

            # 对每个需水节点进行扰动
            for i, perturb_node in enumerate(valid_demand_nodes):
                if i % 50 == 0:  # 每50个节点输出一次进度
                    self.log_info(f"处理节点 {i+1}/{n_nodes}: {perturb_node} ({(i+1)/n_nodes*100:.1f}%)")

                # 保存原始需水量
                original_demands = {}
                node = wn.get_node(perturb_node)
                for j, ts in enumerate(node.demand_timeseries_list):
                    original_demands[j] = ts.base_value

                # 对每个扰动比例进行计算
                ratio_sensitivities = []

                for ratio in demand_ratios:
                    # 计算该比例下的平均扰动量（参考cluster_simple.py）
                    delta = total_demand * ratio / len(base_results.node['demand'])

                    # 设置扰动需水量
                    node = wn.get_node(perturb_node)
                    for j, ts in enumerate(node.demand_timeseries_list):
                        if original_demands[j] > 0:
                            # 按比例扰动
                            ts.base_value = original_demands[j] + original_demands[j] * ratio
                        else:
                            # 绝对量扰动（使用基于总需水量的delta）
                            ts.base_value = original_demands[j] + delta

                    # 运行扰动仿真
                    sim = wntr.sim.EpanetSimulator(wn)
                    perturb_results = sim.run_sim()
                    perturb_pressure = perturb_results.node['pressure'].loc[:, valid_demand_nodes].values

                    # 计算压力差
                    pressure_diff = np.abs(perturb_pressure - base_pressure)

                    # 计算时间平均
                    avg_pressure_diff = np.mean(pressure_diff, axis=0)
                    ratio_sensitivities.append(avg_pressure_diff)

                    # 恢复原始需水量
                    node = wn.get_node(perturb_node)
                    for j, ts in enumerate(node.demand_timeseries_list):
                        ts.base_value = original_demands[j]

                # 计算多个扰动比例的平均敏感度
                avg_sensitivity = np.mean(ratio_sensitivities, axis=0)

                # 归一化处理（避免分母为0）
                max_sensitivity = np.max(avg_sensitivity)
                if max_sensitivity > 0:
                    sensitivity_matrix[i, :] = avg_sensitivity / max_sensitivity
                else:
                    sensitivity_matrix[i, :] = 0

            self.log_info("压力敏感度矩阵计算完成")

            # 输出敏感度矩阵的统计信息
            non_zero_count = np.count_nonzero(sensitivity_matrix)
            total_elements = sensitivity_matrix.size
            self.log_info(f"敏感度矩阵统计: 非零元素 {non_zero_count}/{total_elements} ({non_zero_count/total_elements*100:.1f}%)")
            self.log_info(f"敏感度矩阵范围: [{np.min(sensitivity_matrix):.6f}, {np.max(sensitivity_matrix):.6f}]")

            # 更新分区信息，只保留有效节点
            valid_partitions = {}
            for partition_id, nodes in partitions.items():
                valid_nodes = [node for node in nodes if node in valid_demand_nodes]
                if valid_nodes:
                    valid_partitions[partition_id] = valid_nodes

            return {
                'matrix': sensitivity_matrix,
                'nodes': valid_demand_nodes,
                'partitions': valid_partitions
            }

        except Exception as e:
            error_msg = f"计算压力敏感度矩阵失败: {str(e)}"
            self.log_error(error_msg)
            return None
    
    def select_sensors_by_partition(self, sensitivity_data, threshold=0.5):
        """基于分区选择传感器"""
        try:
            self.log_info(f"开始传感器选择，敏感度阈值: {threshold}")
            
            sensitivity_matrix = sensitivity_data['matrix']
            all_nodes = sensitivity_data['nodes']
            partitions = sensitivity_data['partitions']
            
            # 创建节点索引映射
            node_to_index = {node: i for i, node in enumerate(all_nodes)}
            
            selected_sensors = {}
            
            for partition_id, partition_nodes in partitions.items():
                self.log_info(f"处理分区{partition_id}: {len(partition_nodes)}个节点")
                
                # 获取分区内节点的索引
                partition_indices = [node_to_index[node] for node in partition_nodes if node in node_to_index]
                
                if len(partition_indices) < 2:
                    self.log_info(f"分区{partition_id}节点数少于2个，跳过")
                    continue
                
                # 计算每个节点的影响力（能检测到的节点数）
                influence_scores = {}
                for i, node_idx in enumerate(partition_indices):
                    node_name = all_nodes[node_idx]
                    
                    # 只考虑同分区内的敏感度
                    partition_sensitivities = sensitivity_matrix[node_idx, partition_indices]
                    detectable_count = np.sum(partition_sensitivities > threshold)
                    
                    influence_scores[node_name] = {
                        'index': node_idx,
                        'detectable_count': detectable_count,
                        'avg_sensitivity': np.mean(partition_sensitivities)
                    }
                
                # 动态确定传感器数量以保证韧性
                partition_size = len(partition_nodes)
                min_sensors = max(2, int(partition_size * self.default_params['min_sensor_ratio']))
                max_sensors = min(10, max(3, int(partition_size * self.default_params['max_sensor_ratio'])))
                target_coverage = self.default_params['target_coverage']

                self.log_info(f"分区{partition_id}动态传感器范围: {min_sensors}-{max_sensors}个")

                # 贪心算法选择传感器
                uncovered_indices = set(partition_indices)
                selected_sensors[partition_id] = []

                # 第一阶段：基于覆盖率选择传感器
                while (uncovered_indices and
                       len(selected_sensors[partition_id]) < max_sensors and
                       len(uncovered_indices) / len(partition_indices) > (1 - target_coverage)):

                    best_sensor = None
                    best_coverage = 0

                    for node_name, info in influence_scores.items():
                        if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                            continue  # 已选择的传感器

                        node_idx = info['index']
                        # 计算能覆盖多少未覆盖的节点
                        partition_sensitivities = sensitivity_matrix[node_idx, list(uncovered_indices)]
                        coverage = np.sum(partition_sensitivities > threshold)

                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_sensor = {
                                'node': node_name,
                                'index': node_idx,
                                'coverage': coverage,
                                'influence_score': info['detectable_count'],
                                'avg_sensitivity': info['avg_sensitivity']
                            }

                    if best_sensor is None or best_coverage == 0:
                        break

                    selected_sensors[partition_id].append(best_sensor)

                    # 更新未覆盖节点
                    sensor_idx = best_sensor['index']
                    covered_indices = []
                    for idx in uncovered_indices:
                        if sensitivity_matrix[sensor_idx, idx] > threshold:
                            covered_indices.append(idx)

                    for idx in covered_indices:
                        uncovered_indices.discard(idx)

                # 第二阶段：确保达到最小传感器数量（韧性保证）
                while len(selected_sensors[partition_id]) < min_sensors:
                    best_sensor = None
                    max_diversity = 0

                    for node_name, info in influence_scores.items():
                        if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                            continue

                        node_idx = info['index']
                        # 计算与已有传感器的多样性（距离）
                        diversity_score = 0
                        for existing_sensor in selected_sensors[partition_id]:
                            existing_idx = existing_sensor['index']
                            # 使用敏感度差异作为距离度量
                            distance = 1 - sensitivity_matrix[node_idx, existing_idx]
                            diversity_score += distance

                        # 平均多样性分数
                        if len(selected_sensors[partition_id]) > 0:
                            diversity_score /= len(selected_sensors[partition_id])

                        if diversity_score > max_diversity:
                            max_diversity = diversity_score
                            best_sensor = {
                                'node': node_name,
                                'index': node_idx,
                                'coverage': 0,
                                'influence_score': info['detectable_count'],
                                'avg_sensitivity': info['avg_sensitivity']
                            }

                    if best_sensor is not None:
                        selected_sensors[partition_id].append(best_sensor)
                    else:
                        # 如果没有更多节点，随机选择剩余节点
                        remaining_nodes = [node for node in partition_nodes
                                         if node not in [s['node'] for s in selected_sensors[partition_id]]]
                        if remaining_nodes:
                            node_name = remaining_nodes[0]
                            if node_name in influence_scores:
                                info = influence_scores[node_name]
                                selected_sensors[partition_id].append({
                                    'node': node_name,
                                    'index': info['index'],
                                    'coverage': 0,
                                    'influence_score': info['detectable_count'],
                                    'avg_sensitivity': info['avg_sensitivity']
                                })
                        else:
                            break
                
                self.log_info(f"分区{partition_id}选择了{len(selected_sensors[partition_id])}个传感器")
            
            return selected_sensors

        except Exception as e:
            error_msg = f"传感器选择失败: {str(e)}"
            self.log_error(error_msg)
            return None

    def evaluate_resilience(self, selected_sensors, sensitivity_data, threshold=0.5):
        """评估传感器布置的韧性 - 详细版本"""
        try:
            self.log_info("开始详细韧性评估")

            sensitivity_matrix = sensitivity_data['matrix']
            all_nodes = sensitivity_data['nodes']
            partitions = sensitivity_data['partitions']

            # 创建节点索引映射
            node_to_index = {node: i for i, node in enumerate(all_nodes)}

            resilience_results = {}

            for partition_id, sensors in selected_sensors.items():
                partition_nodes = partitions[partition_id]
                partition_indices = [node_to_index[node] for node in partition_nodes if node in node_to_index]

                self.log_info(f"评估分区{partition_id}的韧性: {len(sensors)}个传感器, {len(partition_nodes)}个节点")

                # 详细场景分析
                detailed_scenarios = []

                # 1. 全部传感器正常工作
                all_sensor_indices = [s['index'] for s in sensors]
                all_sensor_nodes = [s['node'] for s in sensors]
                full_detected_count, full_coverage_rate = self._calculate_detailed_detection(
                    all_sensor_indices, partition_indices, sensitivity_matrix, threshold, all_sensor_nodes
                )

                detailed_scenarios.append({
                    'scenario_type': '全部传感器正常',
                    'failed_sensors': [],
                    'remaining_sensors': [s['node'] for s in sensors],
                    'detected_nodes': full_detected_count,
                    'total_nodes': len(partition_nodes),
                    'coverage_rate': full_coverage_rate,
                    'coverage_percentage': f"{full_coverage_rate*100:.1f}%",
                    'threshold_used': threshold
                })

                # 2. 传感器故障场景（1到M-1个故障）
                total_failure_coverage = 0.0
                failure_scenario_count = 0

                for failure_count in range(1, len(sensors)):
                    failure_combinations = list(combinations(range(len(sensors)), failure_count))

                    for failed_indices in failure_combinations:
                        # 确定失效和剩余的传感器
                        failed_sensors = [sensors[i]['node'] for i in failed_indices]
                        remaining_sensor_indices = [
                            sensors[i]['index'] for i in range(len(sensors))
                            if i not in failed_indices
                        ]
                        remaining_sensors = [
                            sensors[i]['node'] for i in range(len(sensors))
                            if i not in failed_indices
                        ]

                        # 计算剩余传感器的检测能力
                        detected_count, coverage_rate = self._calculate_detailed_detection(
                            remaining_sensor_indices, partition_indices, sensitivity_matrix, threshold, remaining_sensors
                        )

                        total_failure_coverage += coverage_rate
                        failure_scenario_count += 1

                        detailed_scenarios.append({
                            'scenario_type': f'{failure_count}个传感器失效',
                            'failed_sensors': failed_sensors,
                            'remaining_sensors': remaining_sensors,
                            'detected_nodes': detected_count,
                            'total_nodes': len(partition_nodes),
                            'coverage_rate': coverage_rate,
                            'coverage_percentage': f"{coverage_rate*100:.1f}%",
                            'threshold_used': threshold
                        })

                # 计算平均韧性分数（只考虑故障场景）
                avg_failure_resilience = total_failure_coverage / failure_scenario_count if failure_scenario_count > 0 else 0.0

                resilience_results[partition_id] = {
                    'detailed_scenarios': detailed_scenarios,
                    'resilience_score': avg_failure_resilience,
                    'sensor_count': len(sensors),
                    'full_coverage_rate': full_coverage_rate,
                    'avg_failure_coverage': avg_failure_resilience,
                    'total_scenarios': len(detailed_scenarios),
                    'failure_scenarios': failure_scenario_count
                }

                self.log_info(f"分区{partition_id}韧性分数: {avg_failure_resilience:.4f}")

            return resilience_results

        except Exception as e:
            error_msg = f"韧性评估失败: {str(e)}"
            self.log_error(error_msg)
            return None

    def _calculate_detection_rate(self, sensor_indices, target_indices, sensitivity_matrix, threshold):
        """计算检测率"""
        if not sensor_indices:
            return 0.0

        detected_count = 0
        for target_idx in target_indices:
            # 检查是否有任何传感器能检测到该节点
            for sensor_idx in sensor_indices:
                if sensitivity_matrix[sensor_idx, target_idx] > threshold:
                    detected_count += 1
                    break

        return detected_count / len(target_indices) if target_indices else 0.0

    def _calculate_detailed_detection(self, sensor_indices, target_indices, sensitivity_matrix, threshold, sensor_nodes=None):
        """计算详细检测信息：返回检测到的节点数和覆盖率"""
        if not sensor_indices or not target_indices:
            return 0, 0.0

        detected_count = 0
        for target_idx in target_indices:
            # 检查是否有任何传感器能检测到该节点
            for i, sensor_idx in enumerate(sensor_indices):
                # 获取该传感器的阈值（支持自定义阈值）
                sensor_threshold = threshold
                if (self.default_params.get('enable_custom_thresholds', False) and
                    sensor_nodes and i < len(sensor_nodes)):
                    sensor_node = sensor_nodes[i]
                    sensor_threshold = self.default_params.get('custom_thresholds', {}).get(sensor_node, threshold)

                if sensitivity_matrix[sensor_idx, target_idx] > sensor_threshold:
                    detected_count += 1
                    break

        coverage_rate = detected_count / len(target_indices)
        return detected_count, coverage_rate

    def set_custom_sensor_thresholds(self, custom_thresholds):
        """设置自定义传感器阈值

        Args:
            custom_thresholds (dict): {sensor_node: threshold} 格式的字典
        """
        self.default_params['custom_thresholds'] = custom_thresholds
        self.default_params['enable_custom_thresholds'] = bool(custom_thresholds)
        self.log_info(f"设置自定义传感器阈值: {custom_thresholds}")

    def _calculate_resilience_score(self, scenario_results):
        """计算韧性分数"""
        # 加权平均不同故障场景的检测率
        total_score = 0.0
        total_weight = 0.0

        for scenario, result in scenario_results.items():
            if 'no_failure' in scenario:
                weight = 0.5  # 无故障情况权重
            else:
                failure_count = int(scenario.split('_')[0])
                weight = 1.0 / (failure_count + 1)  # 故障越多权重越小

            total_score += result['detection_rate'] * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def optimize_sensor_placement(self, sensitivity_data, max_iterations=10):
        """优化传感器布置"""
        try:
            self.log_info("开始传感器布置优化")

            best_solution = None
            best_score = 0.0

            # 尝试不同的敏感度阈值（减少数量以加快计算）
            thresholds = [0.4, 0.5, 0.6]

            for threshold in thresholds:
                self.log_info(f"尝试阈值: {threshold}")

                # 选择传感器
                selected_sensors = self.select_sensors_by_partition(sensitivity_data, threshold)
                if not selected_sensors:
                    continue

                # 评估韧性
                resilience_results = self.evaluate_resilience(selected_sensors, sensitivity_data, threshold)
                if not resilience_results:
                    continue

                # 计算综合评分
                total_score = self._calculate_total_score(selected_sensors, resilience_results)

                self.log_info(f"阈值{threshold}的综合评分: {total_score:.4f}")

                if total_score > best_score:
                    best_score = total_score
                    best_solution = {
                        'sensors': selected_sensors,
                        'resilience': resilience_results,
                        'threshold': threshold,
                        'score': total_score
                    }

            if best_solution:
                self.log_info(f"最优解: 阈值={best_solution['threshold']}, 评分={best_solution['score']:.4f}")

            return best_solution

        except Exception as e:
            error_msg = f"传感器布置优化失败: {str(e)}"
            self.log_error(error_msg)
            return None

    def _calculate_total_score(self, selected_sensors, resilience_results):
        """计算综合评分"""
        # 传感器数量惩罚（降低惩罚力度）
        total_sensors = sum(len(sensors) for sensors in selected_sensors.values())
        sensor_penalty = total_sensors * 0.001  # 每个传感器扣0.001分

        # 韧性分数
        resilience_scores = [r['resilience_score'] for r in resilience_results.values()]
        avg_resilience = np.mean(resilience_scores) if resilience_scores else 0.0

        # 覆盖率分数（基于传感器数量和分区覆盖）
        partition_count = len(selected_sensors)
        coverage_score = min(1.0, total_sensors / (partition_count * 2))  # 理想情况每分区2个传感器

        # 综合评分：韧性权重40%，覆盖率权重60%
        resilience_weight = self.default_params['resilience_weight']
        coverage_weight = self.default_params['coverage_weight']

        total_score = (avg_resilience * resilience_weight +
                      coverage_score * coverage_weight -
                      sensor_penalty)

        return max(0.001, total_score)  # 确保最小分数为0.001

    def _save_detailed_resilience_analysis(self, solution, conversation_id):
        """保存详细的韧性分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resilience_analysis_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            analysis_data = []

            for partition_id, resilience_info in solution['resilience'].items():
                detailed_scenarios = resilience_info.get('detailed_scenarios', [])

                for scenario in detailed_scenarios:
                    analysis_data.append({
                        '分区编号': partition_id,
                        '场景类型': scenario['scenario_type'],
                        '失效传感器': ', '.join(scenario['failed_sensors']) if scenario['failed_sensors'] else '无',
                        '剩余传感器': ', '.join(scenario['remaining_sensors']),
                        '检测到节点数': scenario['detected_nodes'],
                        '总节点数': scenario['total_nodes'],
                        '覆盖率': f"{scenario['coverage_rate']:.4f}",
                        '覆盖百分比': scenario['coverage_percentage'],
                        '敏感度阈值': scenario['threshold_used'],
                        '分区传感器总数': resilience_info['sensor_count'],
                        '分区平均韧性': f"{resilience_info['resilience_score']:.4f}"
                    })

            # 创建DataFrame并保存
            df = pd.DataFrame(analysis_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            self.log_info(f"详细韧性分析已保存到: {filepath}")
            return filepath

        except Exception as e:
            self.log_error(f"保存韧性分析失败: {str(e)}")
            return None

    def save_sensor_results(self, solution, inp_file_path, conversation_id):
        """保存传感器布置结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_placement_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # 加载网络模型获取坐标
            wn = wntr.network.WaterNetworkModel(inp_file_path)

            # 准备数据
            results_data = []
            sensor_id = 1

            for partition_id, sensors in solution['sensors'].items():
                for sensor in sensors:
                    node_name = sensor['node']

                    # 获取节点坐标
                    try:
                        coord = wn.get_node(node_name).coordinates
                        if coord is None:
                            coord = (0, 0)
                    except:
                        coord = (0, 0)

                    # 获取韧性信息
                    resilience_info = solution['resilience'].get(partition_id, {})
                    resilience_score = resilience_info.get('resilience_score', 0.0)

                    results_data.append({
                        '传感器ID': f'S{sensor_id:03d}',
                        '节点名称': node_name,
                        '分区编号': partition_id,
                        'X坐标': coord[0],
                        'Y坐标': coord[1],
                        '影响力分数': sensor['influence_score'],
                        '平均敏感度': f"{sensor['avg_sensitivity']:.4f}",
                        '覆盖节点数': sensor['coverage'],
                        '分区韧性分数': f"{resilience_score:.4f}",
                        '敏感度阈值': solution['threshold']
                    })
                    sensor_id += 1

            # 创建DataFrame并保存
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            # 保存详细韧性分析
            self._save_detailed_resilience_analysis(solution, conversation_id)

            # 生成统计报告
            stats = self._generate_statistics(solution)

            file_size = os.path.getsize(filepath)

            self.log_info(f"传感器布置结果已保存到: {filepath}")

            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'sensor_count': len(results_data),
                'statistics': stats,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"保存传感器结果失败: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_statistics(self, solution):
        """生成统计信息"""
        stats = {
            'total_sensors': 0,
            'partitions': len(solution['sensors']),
            'avg_resilience': 0.0,
            'threshold': solution['threshold'],
            'total_score': solution['score']
        }

        # 计算总传感器数
        for sensors in solution['sensors'].values():
            stats['total_sensors'] += len(sensors)

        # 计算平均韧性
        if solution['resilience']:
            resilience_scores = [r['resilience_score'] for r in solution['resilience'].values()]
            stats['avg_resilience'] = np.mean(resilience_scores)

        # 分区详情
        stats['partition_details'] = {}
        for partition_id, sensors in solution['sensors'].items():
            resilience_info = solution['resilience'].get(partition_id, {})
            stats['partition_details'][partition_id] = {
                'sensor_count': len(sensors),
                'resilience_score': resilience_info.get('resilience_score', 0.0)
            }

        return stats

    def generate_visualization(self, solution, inp_file_path, conversation_id):
        """生成传感器布置可视化图"""
        try:
            # 设置matplotlib使用英文字体
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # 加载网络模型
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            G = wn.to_graph().to_undirected()

            # 准备节点位置
            pos = {}
            layout = None

            for node in G.nodes():
                try:
                    coord = wn.get_node(node).coordinates
                    if coord is None or coord == (0, 0):
                        if layout is None:
                            import networkx as nx
                            layout = nx.spring_layout(G, seed=42)
                        coord = layout.get(node, (0, 0))
                except:
                    if layout is None:
                        import networkx as nx
                        layout = nx.spring_layout(G, seed=42)
                    coord = layout.get(node, (0, 0))
                pos[node] = coord

            # 创建图形
            plt.figure(figsize=(15, 12))

            # 绘制网络边
            import networkx as nx
            nx.draw_networkx_edges(G, pos=pos, alpha=0.3, width=0.5, edge_color='gray')

            # 绘制普通节点
            all_nodes = list(G.nodes())
            nx.draw_networkx_nodes(G, pos=pos, nodelist=all_nodes,
                                 node_color='lightblue', node_size=20, alpha=0.6)

            # 绘制传感器节点
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
            sensor_nodes_by_partition = {}

            for partition_id, sensors in solution['sensors'].items():
                sensor_nodes = [s['node'] for s in sensors]
                sensor_nodes_by_partition[partition_id] = sensor_nodes

                color = colors[partition_id % len(colors)]
                nx.draw_networkx_nodes(G, pos=pos, nodelist=sensor_nodes,
                                     node_color=color, node_size=100, alpha=0.8,
                                     label=f'Partition {partition_id} Sensors')

            # 添加图例和标题
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Sensor Placement Results\n'
                     f'Total Sensors: {sum(len(s) for s in solution["sensors"].values())}, '
                     f'Threshold: {solution["threshold"]}, '
                     f'Score: {solution["score"]:.4f}')
            plt.axis('off')

            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"sensor_placement_viz_{conversation_id[:8]}_{timestamp}.png"
            viz_path = os.path.join(self.downloads_folder, viz_filename)

            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log_info(f"传感器布置可视化图已保存到: {viz_path}")

            return viz_path

        except Exception as e:
            error_msg = f"生成可视化图失败: {str(e)}"
            self.log_error(error_msg)
            return None

    def process(self, inp_file_path, partition_csv_path, user_message, conversation_id):
        """主处理函数"""
        try:
            self.log_info(f"开始传感器优化布置: {user_message}")
            self.log_info(f"输入文件: {inp_file_path}")
            self.log_info(f"分区文件: {partition_csv_path}")

            # Step 1: 加载分区结果
            self.log_info("Step 1: 加载分区结果...")
            partitions = self.load_partition_results(partition_csv_path)
            if not partitions:
                self.log_error("分区结果加载失败")
                return {
                    'success': False,
                    'response': "分区结果加载失败",
                    'error': "无法加载分区CSV文件"
                }

            self.log_info(f"分区加载成功，共{len(partitions)}个分区")

            # Step 2: 计算压力敏感度矩阵
            self.log_info("Step 2: 计算压力敏感度矩阵...")
            sensitivity_data = self.compute_pressure_sensitivity_matrix(
                inp_file_path, partitions, self.default_params['demand_ratios']
            )

            if not sensitivity_data:
                self.log_error("压力敏感度矩阵计算失败")
                return {
                    'success': False,
                    'response': "压力敏感度矩阵计算失败",
                    'error': "敏感度计算过程中出现错误"
                }

            self.log_info("敏感度矩阵计算成功")

            # Step 3: 优化传感器布置
            self.log_info("Step 3: 优化传感器布置...")
            solution = self.optimize_sensor_placement(sensitivity_data)

            if not solution:
                self.log_error("传感器布置优化失败")
                return {
                    'success': False,
                    'response': "传感器布置优化失败",
                    'error': "优化过程中出现错误"
                }

            self.log_info("传感器布置优化成功")

            # Step 4: 保存结果
            save_result = self.save_sensor_results(solution, inp_file_path, conversation_id)

            # Step 5: 生成可视化
            viz_path = self.generate_visualization(solution, inp_file_path, conversation_id)

            # Step 6: 生成分析报告
            stats = save_result.get('statistics', {})

            response_text = f"""
传感器优化布置完成！

📊 **布置概况**
- 总传感器数: {stats.get('total_sensors', 0)}
- 分区数量: {stats.get('partitions', 0)}
- 敏感度阈值: {stats.get('threshold', 0.5)}
- 综合评分: {stats.get('total_score', 0.0):.4f}

📈 **韧性分析**
- 平均韧性分数: {stats.get('avg_resilience', 0.0):.4f}
- 扰动比例: {self.default_params['demand_ratios']}

🎯 **分区详情**
"""

            for partition_id, details in stats.get('partition_details', {}).items():
                response_text += f"- 分区{partition_id}: {details['sensor_count']}个传感器 (韧性: {details['resilience_score']:.4f})\n"

            response_text += f"""
✅ 传感器布置优化策略：
1. 基于压力敏感度矩阵进行传感器选择
2. 考虑多种故障场景的韧性评估
3. 确保每个分区至少2个传感器
4. 优化检测覆盖率和传感器数量的平衡

📁 结果文件已保存，包含详细的传感器位置和性能指标
"""

            # 生成专业prompt用于GPT分析
            prompt = self._build_sensor_placement_prompt(solution, stats, user_message, save_result)

            result = {
                'success': True,
                'response': response_text,
                'prompt': prompt,  # 添加专业prompt用于GPT分析
                'solution': solution,
                'statistics': stats
            }

            # 添加文件下载信息
            if save_result['success']:
                result['csv_info'] = save_result

            # 添加韧性分析文件信息
            resilience_csv_path = self._save_detailed_resilience_analysis(solution, conversation_id)
            if resilience_csv_path:
                result['resilience_csv_info'] = resilience_csv_path

            if viz_path:
                result['visualization'] = {
                    'filename': os.path.basename(viz_path),
                    'path': viz_path
                }

            return result

        except Exception as e:
            error_msg = f"传感器优化布置失败: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'response': error_msg,
                'error': str(e)
            }

    def _build_sensor_placement_prompt(self, solution, stats, user_message, save_result):
        """构建传感器布置分析的专业prompt"""

        # 获取传感器详细信息
        sensors_info = []
        for partition_id, sensors in solution['sensors'].items():
            for sensor in sensors:
                sensors_info.append(f"分区{partition_id}: 节点{sensor['node']} (敏感度: {sensor.get('avg_sensitivity', 0):.4f})")

        # 获取韧性分析详情
        resilience_details = []
        for partition_id, resilience_data in solution['resilience'].items():
            resilience_details.append(f"分区{partition_id}: 韧性分数{resilience_data['resilience_score']:.4f}, {resilience_data['sensor_count']}个传感器")

        prompt = f"""
用户请求: {user_message}

## 传感器优化布置分析报告

### 📊 布置概况
- **总传感器数**: {stats.get('total_sensors', 0)}个
- **分区数量**: {stats.get('partitions', 0)}个
- **最优敏感度阈值**: {stats.get('threshold', 0.5)}
- **综合评分**: {stats.get('total_score', 0.0):.4f}
- **平均韧性分数**: {stats.get('avg_resilience', 0.0):.4f}

### 🎯 传感器布置详情
{chr(10).join(sensors_info)}

### 📈 韧性分析结果
{chr(10).join(resilience_details)}

### 🔧 技术参数
- **扰动比例**: {self.default_params['demand_ratios']}
- **韧性权重**: {self.default_params['resilience_weight']}
- **覆盖率权重**: {self.default_params['coverage_weight']}

### 📁 生成文件
- **传感器布置结果**: {save_result.get('filename', 'N/A')}
- **文件大小**: {save_result.get('file_size', 0)} bytes
- **记录数**: {save_result.get('sensor_count', 0)}条

### 🎯 优化策略说明
1. **压力敏感度分析**: 基于需水量扰动计算各节点间的压力敏感度矩阵
2. **分区内优化**: 在每个分区内选择影响力最大的节点作为传感器位置
3. **韧性评估**: 考虑传感器故障场景，确保系统在部分传感器失效时仍能正常工作
4. **多目标平衡**: 在检测覆盖率、韧性和传感器数量之间找到最优平衡

请基于以上技术分析，为用户提供专业的传感器布置方案解读和建议。重点说明：
1. 传感器布置的科学性和合理性
2. 韧性设计的重要性和效果
3. 各分区传感器配置的特点
4. 实际应用中的注意事项和建议

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

        return prompt
