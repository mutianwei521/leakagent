"""
PartitionSim 管网分区智能体
负责处理.inp文件，进行管网FCM聚类分区和离群点检测
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .base_agent import BaseAgent
from .intent_classifier_fast import FastIntentClassifier as IntentClassifier

try:
    import wntr
    import skfuzzy as fuzz
    import networkx as nx
    WNTR_AVAILABLE = True
    SKFUZZY_AVAILABLE = True
except ImportError as e:
    WNTR_AVAILABLE = False
    SKFUZZY_AVAILABLE = False

class PartitionSim(BaseAgent):
    """管网分区智能体"""
    
    def __init__(self):
        super().__init__("PartitionSim")

        if not WNTR_AVAILABLE:
            self.log_error("WNTR库未安装，管网分析功能不可用")
        if not SKFUZZY_AVAILABLE:
            self.log_error("scikit-fuzzy库未安装，FCM聚类功能不可用")

        self.intent_classifier = IntentClassifier()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)

        # 缓存机制：避免重复计算敏感度矩阵
        self._sensitivity_cache = {}  # {file_path: {matrix, last_modified}}
        
        # 默认参数
        self.default_params = {
            'k': 5,  # 默认分区数
            'm': 1.5,  # FCM模糊度参数
            'error': 1e-6,  # 收敛阈值
            'maxiter': 1000,  # 最大迭代次数
            'perturb_rate': 0.1,  # 扰动率
            'k_nearest': 10,  # KNN参数
            'outliers_detection': True,  # 是否进行离群点检测
            'seed': 42  # 随机种子
        }
    
    def parse_user_intent(self, user_message: str):
        """解析用户意图和参数"""
        intent_result = self.intent_classifier.classify_intent(user_message)

        # 提取分区相关参数
        params = self.default_params.copy()

        # 提取分区数量 - 支持中文和英文
        k_patterns = [
            # 英文格式
            r'partition\s+into\s+(\d+)\s+regions?',
            r'partition\s+into\s+(\d+)\s+areas?',
            r'(\d+)\s+regions?',
            r'(\d+)\s+partitions?',
            # 中文格式
            r'分成?(\d+)个?分区',
            r'分成?(\d+)个?区域?',
            r'分成?(\d+)个?区',
            r'(\d+)个?分区',
            r'(\d+)个?区域?',
            r'(\d+)个?区',
            r'k\s*=\s*(\d+)',
            r'聚类数\s*[：:]\s*(\d+)',
            r'分区数\s*[：:]\s*(\d+)',
            r'分区数目\s*[：:为]\s*(\d+)',
            r'分区数目为(\d+)'
        ]

        for pattern in k_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['k'] = int(match.group(1))
                self.logger.info(f"[PartitionSim] 解析到分区数量: {params['k']} (匹配模式: {pattern})")
                break
        
        # 提取FCM参数
        m_patterns = [
            r'm\s*=\s*([\d.]+)',
            r'模糊度\s*[：:=]\s*([\d.]+)',
            r'模糊参数\s*[：:=]\s*([\d.]+)',
            r'模糊度=([\d.]+)'
        ]
        
        for pattern in m_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['m'] = float(match.group(1))
                self.logger.info(f"[PartitionSim] 解析到模糊度参数: {params['m']} (匹配模式: {pattern})")
                break
        
        # 提取扰动率
        perturb_patterns = [
            r'扰动率\s*[：:]\s*([\d.]+)',
            r'扰动率([\d.]+)',
            r'perturb[_\s]*rate\s*[：:=]\s*([\d.]+)'
        ]
        
        for pattern in perturb_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['perturb_rate'] = float(match.group(1))
                self.logger.info(f"[PartitionSim] 解析到扰动率: {params['perturb_rate']} (匹配模式: {pattern})")
                break
        
        # 检测是否需要离群点处理
        outlier_disable_keywords = [
            '不检测离群点', '不处理离群点', '跳过离群点', '不要离群点检测',
            '不进行离群点检测', '禁用离群点检测', '关闭离群点检测',
            '不剔除异常点', '不处理异常点', '跳过异常点', '不要异常点检测',
            'no outlier', 'skip outlier', 'disable outlier'
        ]

        outlier_enable_keywords = [
            '检测离群点', '处理离群点', '离群点检测', '进行离群点检测',
            '启用离群点检测', '开启离群点检测', '剔除异常点', '处理异常点',
            '异常点检测', '异常点剔除', 'outlier detection', 'remove outlier'
        ]

        if any(keyword in user_message for keyword in outlier_disable_keywords):
            params['outliers_detection'] = False
            self.logger.info(f"[PartitionSim] 解析到禁用离群点检测")
        elif any(keyword in user_message for keyword in outlier_enable_keywords):
            params['outliers_detection'] = True
            self.logger.info(f"[PartitionSim] 解析到启用离群点检测")
        
        return {
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'params': params
        }
    
    def parse_network(self, inp_file_path: str):
        """解析管网文件，提取基本信息"""
        if not WNTR_AVAILABLE:
            return {'error': 'WNTR库未安装'}

        try:
            # 检查缓存
            if inp_file_path in getattr(self, '_network_cache', {}):
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

            self.log_info(f"管网解析完成: {network_info['nodes']['total']}个节点, {network_info['links']['total']}个管段")

            # 初始化缓存
            if not hasattr(self, '_network_cache'):
                self._network_cache = {}

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

    def load_network(self, inp_file_path: str):
        """加载水网络模型"""
        if not WNTR_AVAILABLE:
            return None, {'error': 'WNTR库未安装'}

        try:
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            self.log_info(f"加载网络: 节点={len(wn.node_name_list)}, "
                         f"需水节点={len(wn.junction_name_list)}, "
                         f"管段={len(wn.link_name_list)}")
            return wn, None
        except Exception as e:
            error_msg = f"加载网络文件失败: {str(e)}"
            self.log_error(error_msg)
            return None, {'error': error_msg}
    
    def normalize_matrix(self, S):
        """对敏感度矩阵进行标准化和归一化处理"""
        # 标准化：减去均值，除以标准差
        S_mean = np.mean(S, axis=0)
        S_std = np.std(S, axis=0)
        # 添加一个小的阈值，避免除以0
        epsilon = 1e-10
        S_std = np.where(S_std == 0, epsilon, S_std)
        S_std = (S - S_mean) / S_std
        
        # 归一化：将值映射到[0,1]区间
        S_min = np.min(S_std)
        S_max = np.max(S_std)
        S_n = (S_std - S_min) / (S_max - S_min)
        
        return S_n
    
    def compute_sensitivity_matrix(self, inp_file_path: str, perturb_rate: float):
        """计算敏感度矩阵"""
        if not WNTR_AVAILABLE:
            return None, None, {'error': 'WNTR库未安装'}
        
        try:
            # 检查缓存
            cache_key = f"{inp_file_path}_{perturb_rate}"
            if cache_key in self._sensitivity_cache:
                file_mtime = os.path.getmtime(inp_file_path)
                cached_data = self._sensitivity_cache[cache_key]
                if cached_data['last_modified'] == file_mtime:
                    self.log_info(f"使用缓存的敏感度矩阵")
                    return cached_data['nodes'], cached_data['demands'], cached_data['matrix']
            
            self.log_info(f"开始计算敏感度矩阵，扰动率: {perturb_rate}")
            
            # 加载基线网络模型
            wn0, error = self.load_network(inp_file_path)
            if error:
                return None, None, error
            
            # 运行基线仿真获取压力
            sim = wntr.sim.EpanetSimulator(wn0)
            res = sim.run_sim()
            
            # 获取所有节点和需水节点列表
            node_list = wn0.node_name_list
            demand_nodes = wn0.junction_name_list

            # 计算总实际需水量
            total_demand = 0
            for name in demand_nodes:
                node_demands = res.node['demand'].loc[:, name]
                total_demand += node_demands.sum()

            # 初始化敏感度矩阵
            S = np.zeros((len(demand_nodes), len(demand_nodes)))
            # 重新加载网络用于扰动仿真
            wn, _ = self.load_network(inp_file_path)
            
            # 获取基线压力
            base_p = res.node['pressure'].loc[:, demand_nodes].values
            # 计算平均扰动量
            delta = total_demand * perturb_rate / len(res.node['demand'])

            # 对每个需水节点进行扰动
            for j, name in enumerate(demand_nodes):
                self.log_info(f"处理节点 {j+1}/{len(demand_nodes)}: {name}")
                
                # 获取该节点的需水时间序列
                ts_list = wn.get_node(name).demand_timeseries_list
                # 保存原始需水量
                orig_values = [d.base_value for d in ts_list]
                
                # 对每个时间序列进行扰动
                for d in ts_list:
                    if d.base_value > 0:
                        d.base_value = d.base_value + d.base_value * perturb_rate
                    else:
                        d.base_value = d.base_value + delta
                
                # 运行扰动后仿真
                sim = wntr.sim.EpanetSimulator(wn)
                res_pert = sim.run_sim()
                
                # 获取扰动后的压力
                pert_p = res_pert.node['pressure'].loc[:, demand_nodes].values
                
                # 计算当前扰动节点的压力差
                current_node_p_diff = np.abs(pert_p[:, j] - base_p[:, j])
                
                # 计算敏感度
                with np.errstate(divide='ignore', invalid='ignore'):
                    S[:, j] = np.mean(np.where(current_node_p_diff[:, np.newaxis] != 0,
                                              np.abs(pert_p - base_p) / current_node_p_diff[:, np.newaxis],
                                              0), axis=0)
                
                # 恢复原始需水量
                for d, orig in zip(ts_list, orig_values):
                    d.base_value = orig
            
            # 缓存结果
            self._sensitivity_cache[cache_key] = {
                'nodes': node_list,
                'demands': demand_nodes,
                'matrix': S,
                'last_modified': os.path.getmtime(inp_file_path)
            }
            
            self.log_info(f"敏感度矩阵计算完成，矩阵大小: {S.shape}")
            return node_list, demand_nodes, S
            
        except Exception as e:
            error_msg = f"计算敏感度矩阵失败: {str(e)}"
            self.log_error(error_msg)
            return None, None, {'error': error_msg}

    def perform_fcm_clustering(self, S_normalized, params):
        """执行FCM聚类"""
        if not SKFUZZY_AVAILABLE:
            return None, None, {'error': 'scikit-fuzzy库未安装'}

        try:
            self.log_info(f"开始FCM聚类，参数: k={params['k']}, m={params['m']}")

            # 设置随机种子
            np.random.seed(params['seed'])

            # 执行FCM聚类
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data=S_normalized.T,      # 输入数据矩阵，需要转置
                c=params['k'],            # 聚类数量
                m=params['m'],            # 模糊度参数
                error=params['error'],    # 收敛阈值
                maxiter=params['maxiter'], # 最大迭代次数
                init=None,                # 初始聚类中心
                seed=params['seed']       # 随机种子
            )

            # 获取初始标签（从1开始）
            raw_labels = np.argmax(u, axis=0) + 1

            self.log_info(f"FCM聚类完成，收敛迭代次数: {p}, 模糊分割系数: {fpc:.4f}")

            return raw_labels, {
                'centers': cntr,
                'membership': u,
                'iterations': p,
                'fpc': fpc,
                'objective_function': jm
            }, None

        except Exception as e:
            error_msg = f"FCM聚类失败: {str(e)}"
            self.log_error(error_msg)
            return None, None, {'error': error_msg}

    def check_connectivity(self, node_connections, cluster_nodes):
        """使用Warshall算法检查节点连通性"""
        n = len(cluster_nodes)
        adj_matrix = np.zeros((n, n), dtype=int)

        # 填充邻接矩阵
        for i, node1 in enumerate(cluster_nodes):
            for j, node2 in enumerate(cluster_nodes):
                if i == j:
                    adj_matrix[i, j] = 1
                else:
                    # 检查两个节点是否直接相连
                    mask1 = (node_connections[:, 0] == node1) & (node_connections[:, 1] == node2)
                    mask2 = (node_connections[:, 0] == node2) & (node_connections[:, 1] == node1)
                    if np.any(mask1) or np.any(mask2):
                        adj_matrix[i, j] = 1

        # 使用Warshall算法计算传递闭包
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    adj_matrix[i, j] = adj_matrix[i, j] or (adj_matrix[i, k] and adj_matrix[k, j])

        return adj_matrix

    def find_connected_components(self, connect_matrix):
        """找出所有连通分量"""
        n = len(connect_matrix)
        visited = np.zeros(n, dtype=bool)
        components = []

        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        neighbors = np.where(connect_matrix[node, :] == 1)[0]
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                components.append(component)

        return components

    def assign_unassigned_nodes_by_nearest_neighbor(self, wn, nodes, demands, labels, params):
        """将未分配节点分配到最近邻分区"""

        # 找到未分配的需水节点
        unassigned_indices = []
        for i, demand_node in enumerate(demands):
            if labels[i] == 0:
                unassigned_indices.append(i)

        if len(unassigned_indices) == 0:
            return labels

        self.log_info(f"开始为{len(unassigned_indices)}个未分配需水节点分配最近邻分区")

        # 获取节点坐标
        node_coords = {}
        layout = None  # 用于没有坐标的节点

        for node_name in nodes:
            try:
                coord = wn.get_node(node_name).coordinates
                if coord is None or coord == (0, 0):
                    # 如果没有坐标，使用网络布局
                    if layout is None:
                        G = wn.to_graph().to_undirected()
                        layout = nx.spring_layout(G, seed=params['seed'])
                    coord = layout.get(node_name, (0, 0))
            except:
                if layout is None:
                    G = wn.to_graph().to_undirected()
                    layout = nx.spring_layout(G, seed=params['seed'])
                coord = layout.get(node_name, (0, 0))
            node_coords[node_name] = coord

        # 创建已分配节点的分区信息
        assigned_nodes_by_partition = {}
        for i, demand_node in enumerate(demands):
            if labels[i] > 0:
                partition = labels[i]
                if partition not in assigned_nodes_by_partition:
                    assigned_nodes_by_partition[partition] = []
                assigned_nodes_by_partition[partition].append((demand_node, node_coords[demand_node]))

        # 为每个未分配节点找到最近的分区
        labels_copy = labels.copy()

        for unassigned_idx in unassigned_indices:
            unassigned_node = demands[unassigned_idx]
            unassigned_coord = node_coords[unassigned_node]

            min_distance = float('inf')
            nearest_partition = 1  # 默认分区

            # 遍历所有分区，找到最近的节点
            for partition, nodes_in_partition in assigned_nodes_by_partition.items():
                for assigned_node, assigned_coord in nodes_in_partition:
                    # 计算欧氏距离
                    distance = np.sqrt((unassigned_coord[0] - assigned_coord[0])**2 +
                                     (unassigned_coord[1] - assigned_coord[1])**2)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_partition = partition

            # 分配到最近的分区
            labels_copy[unassigned_idx] = nearest_partition

            self.log_info(f"节点{unassigned_node}分配到分区{nearest_partition}，最近距离: {min_distance:.4f}")

            # 更新分区信息，以便后续节点可以考虑这个新分配的节点
            if nearest_partition not in assigned_nodes_by_partition:
                assigned_nodes_by_partition[nearest_partition] = []
            assigned_nodes_by_partition[nearest_partition].append((unassigned_node, unassigned_coord))

        return labels_copy

    def remove_outliers_iteratively(self, wn, nodes, demands, raw_labels, params):
        """迭代处理两类离群点"""
        if not params['outliers_detection']:
            self.log_info("跳过离群点检测")
            return raw_labels

        self.log_info("开始迭代离群点检测")

        # 创建完整的标签数组
        all_labels = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                all_labels[i] = raw_labels[idx]
            else:
                all_labels[i] = 0

        # 获取节点连接关系
        node_connections = []
        for link in wn.links():
            node1 = link[1].start_node_name
            node2 = link[1].end_node_name
            node_connections.append([nodes.index(node1), nodes.index(node2)])
        node_connections = np.array(node_connections)

        number_iter = 0
        max_iterations = 10

        while number_iter < max_iterations:
            # 检查是否还有标签为0的点
            zero_count = np.sum(all_labels == 0)
            if zero_count == 0:
                break

            number_iter += 1
            self.log_info(f"离群点检测迭代 {number_iter}, 剩余未分配节点: {zero_count}")

            # 处理第一类离群点：基于邻居节点标签的一致性
            for i, node in enumerate(nodes):
                if all_labels[i] != 99999:  # 排除特殊标记
                    # 获取当前节点的所有连接节点
                    connected_nodes = []
                    for conn in node_connections:
                        if conn[0] == i:
                            connected_nodes.append(conn[1])
                        elif conn[1] == i:
                            connected_nodes.append(conn[0])
                    connected_nodes = np.array(connected_nodes)

                    if len(connected_nodes) > 0:
                        # 获取邻居节点的唯一标签
                        neighbor_labels = np.unique(all_labels[connected_nodes])
                        # 计算每个标签出现的次数
                        label_counts = np.array([np.sum(all_labels[connected_nodes] == label) for label in neighbor_labels])
                        # 找到出现次数最多的值
                        max_count = np.max(label_counts)
                        # 获取所有达到最大次数的标签
                        max_labels = neighbor_labels[label_counts == max_count]
                        # 如果0在最大次数标签中，且还有其他标签，则移除0
                        if 0 in max_labels and len(max_labels) > 1:
                            max_labels = max_labels[max_labels != 0]
                        # 选择第一个非0的标签（如果存在）
                        if len(max_labels) > 0:
                            all_labels[i] = max_labels[0]
                        else:
                            all_labels[i] = 0

            # 处理第二类离群点：基于空间距离和连通性
            for cluster in range(1, int(np.max(all_labels)) + 1):
                cluster_nodes = np.where(all_labels == cluster)[0]
                if len(cluster_nodes) <= 1:
                    continue

                # 获取节点的坐标和高度
                coordinates = []
                elevations = []
                for node_idx in cluster_nodes:
                    node = wn.get_node(nodes[node_idx])
                    try:
                        coord = node.coordinates
                        elev = node.elevation
                    except:
                        coord = (0, 0)
                        elev = 0
                    coordinates.append(coord)
                    elevations.append(elev)

                # 构建特征矩阵 [x, y, elevation]
                features = np.column_stack([coordinates, elevations])

                # 计算欧氏距离矩阵
                dist_matrix = np.zeros((len(cluster_nodes), len(cluster_nodes)))
                for i in range(len(cluster_nodes)):
                    for j in range(len(cluster_nodes)):
                        dist_matrix[i, j] = np.linalg.norm(features[i] - features[j])

                # 计算每个节点的KNN距离
                knn_distances = []
                for i in range(len(cluster_nodes)):
                    distances = dist_matrix[i, :]
                    distances = distances[distances > 0]
                    k = min(params['k_nearest'], len(distances))
                    if k > 0:
                        knn_dist = np.mean(np.sort(distances)[:k])
                        knn_distances.append(knn_dist)
                    else:
                        knn_distances.append(0)

                knn_distances = np.array(knn_distances)

                # 计算统计量并标记离群点
                if len(knn_distances) > 0:
                    mean_dist = np.mean(knn_distances)
                    std_dist = np.std(knn_distances)

                    # 标记距离离群点
                    outliers = (knn_distances <= mean_dist - 3 * std_dist) | (knn_distances >= mean_dist + 3 * std_dist)
                    all_labels[cluster_nodes[outliers]] = 0

                # 检查连通性
                connect_matrix = self.check_connectivity(node_connections, cluster_nodes)
                components = self.find_connected_components(connect_matrix)

                if len(components) > 1:
                    # 选择最大的连通分量作为主区
                    main_component = max(components, key=len)
                    # 将不在主区中的节点标记为离群点
                    outliers = np.setdiff1d(np.arange(len(cluster_nodes)), main_component)
                    all_labels[cluster_nodes[outliers]] = 0

        # 检查是否有分区被完全消除，如果有则恢复最大的连通分量
        original_partitions = set(raw_labels)
        current_partitions = set(all_labels[all_labels > 0])

        lost_partitions = original_partitions - current_partitions
        if lost_partitions:
            self.log_info(f"检测到被完全消除的分区: {sorted(lost_partitions)}")

            # 对于每个被消除的分区，恢复其最大连通分量
            for lost_partition in lost_partitions:
                # 找到原本属于这个分区的节点
                original_nodes = []
                for i, node in enumerate(nodes):
                    if node in demands:
                        idx = demands.index(node)
                        if raw_labels[idx] == lost_partition:
                            original_nodes.append(i)

                if original_nodes:
                    # 检查这些节点的连通性
                    if len(original_nodes) > 1:
                        # 构建连通性矩阵
                        connect_matrix = self.check_connectivity(node_connections, original_nodes)
                        components = self.find_connected_components(connect_matrix)

                        if components:
                            # 恢复最大的连通分量
                            main_component = max(components, key=len)
                            for local_idx in main_component:
                                global_idx = original_nodes[local_idx]
                                all_labels[global_idx] = lost_partition

                            self.log_info(f"恢复分区{lost_partition}的最大连通分量: {len(main_component)}个节点")
                    else:
                        # 只有一个节点，直接恢复
                        all_labels[original_nodes[0]] = lost_partition
                        self.log_info(f"恢复分区{lost_partition}的单个节点")

        # 更新raw_labels
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                raw_labels[idx] = all_labels[i]

        # 最终验证分区数量
        final_partitions = len(set(raw_labels[raw_labels > 0]))
        expected_partitions = params['k']

        if final_partitions != expected_partitions:
            self.log_info(f"⚠️ 分区数量不匹配: 期望{expected_partitions}个，实际{final_partitions}个")
        else:
            self.log_info(f"✅ 分区数量验证通过: {final_partitions}个分区")

        # 检查未分配节点数量
        unassigned_count = np.sum(raw_labels == 0)
        if unassigned_count > 0:
            self.log_info(f"检测到{unassigned_count}个未分配节点，开始最近邻分配")
            # 进行最近邻分配
            final_labels = self.assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, raw_labels, params)

            # 验证最近邻分配结果
            final_unassigned = np.sum(final_labels == 0)
            if final_unassigned == 0:
                self.log_info("✅ 所有节点已通过最近邻分配成功分配到分区")
            else:
                self.log_info(f"⚠️ 最近邻分配后仍有{final_unassigned}个节点未分配")

            self.log_info(f"离群点检测和最近邻分配完成，迭代次数: {number_iter}")
            return final_labels
        else:
            self.log_info("✅ 所有节点已分配，无需最近邻分配")
            self.log_info(f"离群点检测完成，迭代次数: {number_iter}")
            return raw_labels

    def identify_boundary_pipes(self, wn, nodes, demands, labels):
        """识别边界管道 - 管道两端节点属于不同分区"""
        try:
            # 创建完整的标签数组
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # 创建节点到索引的映射
            node_to_idx = {node: i for i, node in enumerate(nodes)}

            boundary_pipes = []
            non_boundary_pipes = []

            # 遍历所有管段
            for link in wn.links():
                link_obj = link[1]
                start_node = link_obj.start_node_name
                end_node = link_obj.end_node_name

                # 获取两端节点的分区标签
                if start_node in node_to_idx and end_node in node_to_idx:
                    start_idx = node_to_idx[start_node]
                    end_idx = node_to_idx[end_node]
                    start_label = all_labels[start_idx]
                    end_label = all_labels[end_idx]

                    # 判断是否为边界管道
                    if start_label != end_label and start_label > 0 and end_label > 0:
                        boundary_pipes.append((start_node, end_node))
                    else:
                        non_boundary_pipes.append((start_node, end_node))

            self.log_info(f"识别到{len(boundary_pipes)}条边界管道，{len(non_boundary_pipes)}条非边界管道")
            return boundary_pipes, non_boundary_pipes

        except Exception as e:
            error_msg = f"识别边界管道失败: {str(e)}"
            self.log_error(error_msg)
            return [], []

    def generate_partition_visualization(self, wn, nodes, demands, labels, params, save_path=None):
        """生成分区可视化图"""
        try:
            # 设置matplotlib使用英文字体，避免中文乱码
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # 创建无向图
            G = wn.to_graph().to_undirected()

            # 准备节点位置
            pos = {}
            layout = None
            for n in G.nodes():
                try:
                    coord = wn.get_node(n).coordinates
                except:
                    if layout is None:
                        layout = nx.spring_layout(G, seed=params['seed'])
                    coord = layout[n]
                pos[n] = coord

            # 创建完整的标签数组
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # 绘制网络分区
            plt.figure(figsize=(12, 10))

            # 绘制边
            nx.draw_networkx_edges(G, pos=pos, alpha=0.9, width=0.8)

            # 绘制节点
            scatter = nx.draw_networkx_nodes(
                G, pos=pos,
                nodelist=nodes,
                node_color=all_labels,
                cmap=plt.get_cmap("tab10", params['k']+1),
                vmin=0, vmax=params['k'],
                node_size=30
            )

            # 添加图例（使用英文）
            legend_labels = ['Unassigned'] + [f'Partition {i}' for i in range(1, params['k']+1)]
            plt.legend(scatter.legend_elements()[0], legend_labels,
                      title="Node Type",
                      loc='upper right',
                      bbox_to_anchor=(1, 1))

            plt.title(f"Water Network Partitioning Results (K={params['k']}, Fuzziness={params['m']})")
            plt.axis("off")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log_info(f"分区图已保存到: {save_path}")

            return save_path

        except Exception as e:
            error_msg = f"生成可视化图失败: {str(e)}"
            self.log_error(error_msg)
            return None

    def generate_boundary_pipes_visualization(self, wn, nodes, demands, labels, params, save_path=None):
        """生成边界管道可视化图 - 突出显示边界管道"""
        try:
            # 设置matplotlib使用英文字体
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # 加载网络模型
            wn = wntr.network.WaterNetworkModel(wn.name) if hasattr(wn, 'name') else wn
            G = wn.to_graph().to_undirected()

            # 准备节点位置
            pos = {}
            layout = None

            for node in G.nodes():
                try:
                    coord = wn.get_node(node).coordinates
                    if coord is None or coord == (0, 0):
                        if layout is None:
                            layout = nx.spring_layout(G, seed=params['seed'])
                        coord = layout.get(node, (0, 0))
                except:
                    if layout is None:
                        layout = nx.spring_layout(G, seed=params['seed'])
                    coord = layout.get(node, (0, 0))
                pos[node] = coord

            # 创建完整的标签数组
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # 识别边界管道
            boundary_pipes, non_boundary_pipes = self.identify_boundary_pipes(wn, nodes, demands, labels)
            boundary_count = len(boundary_pipes)

            # 创建图形 - 使用与sensor_placement.py相同的风格
            plt.figure(figsize=(15, 12))

            # 绘制非边界管道（淡化但更深）
            nx.draw_networkx_edges(G, pos=pos, edgelist=non_boundary_pipes,
                                  alpha=0.4, width=0.5, edge_color='gray')

            # 绘制边界管道（红色，加粗）
            nx.draw_networkx_edges(G, pos=pos, edgelist=boundary_pipes,
                                  alpha=0.9, width=2.5, edge_color='red')

            # 绘制普通节点（淡化）
            all_nodes = list(G.nodes())
            nx.draw_networkx_nodes(G, pos=pos, nodelist=all_nodes,
                                 node_color='lightblue', node_size=20, alpha=0.5)

            # 绘制分区节点（按分区着色，覆盖普通节点）
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            for partition_id in range(1, params['k'] + 1):
                partition_nodes = [nodes[i] for i in range(len(nodes)) if all_labels[i] == partition_id]
                if partition_nodes:
                    color = colors[partition_id % len(colors)]
                    nx.draw_networkx_nodes(G, pos=pos, nodelist=partition_nodes,
                                         node_color=color, node_size=30, alpha=0.7,
                                         label=f'Partition {partition_id}')

            # 添加图例和标题
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Water Network Boundary Pipes Analysis\n'
                     f'Total Boundary Pipes: {boundary_count}, '
                     f'Partitions: {params["k"]}, '
                     f'Fuzziness: {params["m"]}')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.log_info(f"边界管道可视化图已保存到: {save_path}")

            return save_path, boundary_count

        except Exception as e:
            error_msg = f"生成边界管道可视化图失败: {str(e)}"
            self.log_error(error_msg)
            return None, 0

    def save_partition_results(self, nodes, demands, labels, params, clustering_info, conversation_id):
        """保存分区结果到CSV文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"partition_results_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # 准备数据
            results_data = []

            # 添加需水节点的分区信息
            for i, node_id in enumerate(demands):
                results_data.append({
                    '节点ID': node_id,
                    '节点类型': '需水节点',
                    '分区编号': int(labels[i]),
                    '分区名称': f'分区{int(labels[i])}' if labels[i] > 0 else '未分配'
                })

            # 添加非需水节点信息
            for node_id in nodes:
                if node_id not in demands:
                    results_data.append({
                        '节点ID': node_id,
                        '节点类型': '非需水节点',
                        '分区编号': 0,
                        '分区名称': '非需水节点'
                    })

            # 创建DataFrame并保存
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            # 计算统计信息
            partition_stats = {}
            for i in range(1, params['k'] + 1):
                count = np.sum(labels == i)
                partition_stats[f'分区{i}'] = count

            unassigned_count = np.sum(labels == 0)
            if unassigned_count > 0:
                partition_stats['未分配'] = unassigned_count

            file_size = os.path.getsize(filepath)

            self.log_info(f"分区结果已保存到: {filepath}")

            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'records_count': len(results_data),
                'partition_stats': partition_stats,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"保存分区结果失败: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def build_partition_prompt(self, network_info: dict, partition_result: dict, user_message: str, csv_info: dict = None):
        """构建包含网络信息和分区结果的专业分析prompt"""
        prompt = f"""
你是一个专业的给水管网分区分析专家。现在需要分析以下管网系统的分区结果：

管网基本信息：
- 节点总数：{network_info['nodes']['total']} (节点: {network_info['nodes']['junctions']}, 水库: {network_info['nodes']['reservoirs']}, 水塔: {network_info['nodes']['tanks']})
- 管段总数：{network_info['links']['total']} (管道: {network_info['links']['pipes']}, 水泵: {network_info['links']['pumps']}, 阀门: {network_info['links']['valves']})
- 管网总长度：{network_info['network_stats']['total_length']:.2f} 米
- 仿真时长：{network_info['network_stats']['simulation_duration']} 秒

✅ 管网分区分析已成功完成！

分区分析结果：
{partition_result['response']}

分区技术参数：
- FCM聚类算法，模糊度参数 m = {partition_result['parameters']['m']}
- 敏感度矩阵扰动率：{partition_result['parameters']['perturb_rate']}
- 收敛阈值：{partition_result['parameters']['error']}
- 最大迭代次数：{partition_result['parameters']['maxiter']}
- 离群点检测：{'已启用' if partition_result['parameters']['outliers_detection'] else '未启用'}

分区质量指标：
- 模糊分割系数 (FPC)：{partition_result['partition_info']['fpc']:.4f}
- 聚类收敛迭代次数：{partition_result['partition_info']['iterations']}
"""

        if csv_info and csv_info['success']:
            prompt += f"""
📊 详细分区数据已保存为CSV文件：{csv_info['filename']}
文件大小：{csv_info['file_size']} 字节，共 {csv_info['records_count']} 条记录
"""

        prompt += f"""
用户问题：{user_message}

请基于管网基本信息和分区分析结果，提供专业的分析和建议，包括：
1. 分区结果的合理性评估
2. 分区质量的技术分析（基于FPC值和分区分布）
3. 可能的优化建议
4. 工程应用价值和意义
5. 如有必要，建议进一步的分析方向

同时告知用户可以下载详细的分区数据进行进一步分析。

请在回复的最后使用以下签名格式：

祝好，

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def process(self, inp_file_path: str, user_message: str, conversation_id: str):
        """主处理函数"""
        try:
            self.log_info(f"开始处理管网分区请求: {user_message}")

            # Step 1: 解析管网文件，获取基本信息
            network_info = self.parse_network(inp_file_path)
            if 'error' in network_info:
                return {
                    'success': False,
                    'response': f"管网文件解析失败: {network_info['error']}",
                    'intent': 'partition_analysis',
                    'confidence': 0.0
                }

            # Step 2: 解析用户意图和参数
            intent_result = self.parse_user_intent(user_message)
            params = intent_result['params']

            self.log_info(f"解析参数: {params}")

            # Step 3: 加载网络模型
            wn, error = self.load_network(inp_file_path)
            if error:
                return {
                    'success': False,
                    'response': f"加载网络文件失败: {error['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # 计算敏感度矩阵
            nodes, demands, S = self.compute_sensitivity_matrix(inp_file_path, params['perturb_rate'])
            if isinstance(S, dict) and 'error' in S:
                return {
                    'success': False,
                    'response': f"计算敏感度矩阵失败: {S['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # 标准化敏感度矩阵
            S_normalized = self.normalize_matrix(S)

            # 执行FCM聚类
            raw_labels, clustering_info, error = self.perform_fcm_clustering(S_normalized, params)
            if error:
                return {
                    'success': False,
                    'response': f"FCM聚类失败: {error['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # 离群点检测和处理
            refined_labels = self.remove_outliers_iteratively(wn, nodes, demands, raw_labels, params)

            # 生成可视化图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"partition_viz_{conversation_id[:8]}_{timestamp}.png"
            viz_path = os.path.join(self.downloads_folder, viz_filename)

            viz_result = self.generate_partition_visualization(wn, nodes, demands, refined_labels, params, viz_path)

            # 生成边界管道可视化图
            boundary_viz_filename = f"boundary_pipes_viz_{conversation_id[:8]}_{timestamp}.png"
            boundary_viz_path = os.path.join(self.downloads_folder, boundary_viz_filename)
            boundary_viz_result, boundary_pipe_count = self.generate_boundary_pipes_visualization(
                wn, nodes, demands, refined_labels, params, boundary_viz_path
            )

            # 获取边界管道信息用于报告
            boundary_pipes, non_boundary_pipes = self.identify_boundary_pipes(wn, nodes, demands, refined_labels)

            # 保存分区结果
            save_result = self.save_partition_results(nodes, demands, refined_labels, params, clustering_info, conversation_id)

            # 生成分析报告
            total_nodes = len(nodes)
            demand_nodes_count = len(demands)
            partition_distribution = {}
            for i in range(1, params['k'] + 1):
                count = int(np.sum(refined_labels == i))  # 转换为Python int
                partition_distribution[i] = count

            unassigned_count = int(np.sum(refined_labels == 0))  # 转换为Python int

            response_text = f"""
管网分区分析完成！

📊 **分区概况**
- 总节点数: {total_nodes}
- 需水节点数: {demand_nodes_count}
- 分区数量: {params['k']}
- 模糊度参数: {params['m']}
- 扰动率: {params['perturb_rate']}

📈 **聚类质量**
- 模糊分割系数 (FPC): {clustering_info['fpc']:.4f}
- 收敛迭代次数: {clustering_info['iterations']}

🎯 **分区分布**
"""
            for i in range(1, params['k'] + 1):
                count = partition_distribution[i]
                percentage = (count / demand_nodes_count) * 100
                response_text += f"- 分区{i}: {count}个节点 ({percentage:.1f}%)\n"

            if unassigned_count > 0:
                percentage = (unassigned_count / demand_nodes_count) * 100
                response_text += f"- 未分配: {unassigned_count}个节点 ({percentage:.1f}%)\n"

            if params['outliers_detection']:
                response_text += f"\n✅ 已进行离群点检测和处理"
            else:
                response_text += f"\n⚠️ 未进行离群点检测"

            # 添加边界管道信息
            response_text += f"\n\n🔴 **边界管道分析**\n"
            response_text += f"- 边界管道总数: {boundary_pipe_count}\n"
            response_text += f"- 边界管道占比: {(boundary_pipe_count / (boundary_pipe_count + len(non_boundary_pipes)) * 100):.1f}% (共{boundary_pipe_count + len(non_boundary_pipes)}条管道)"

            # 构建专业分析prompt
            prompt = self.build_partition_prompt(
                network_info,
                {
                    'response': response_text,
                    'partition_info': {
                        'total_nodes': total_nodes,
                        'demand_nodes': demand_nodes_count,
                        'k': params['k'],
                        'partition_distribution': partition_distribution,
                        'unassigned_count': unassigned_count,
                        'fpc': float(clustering_info['fpc']),  # 转换为Python float
                        'iterations': int(clustering_info['iterations'])  # 转换为Python int
                    },
                    'parameters': params
                },
                user_message,
                save_result if save_result['success'] else None
            )

            result = {
                'success': True,
                'response': response_text,
                'prompt': prompt,  # 添加专业prompt用于GPT分析
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'partition_info': {
                    'total_nodes': total_nodes,
                    'demand_nodes': demand_nodes_count,
                    'k': params['k'],
                    'partition_distribution': partition_distribution,
                    'unassigned_count': unassigned_count,
                    'fpc': float(clustering_info['fpc']),  # 转换为Python float
                    'iterations': int(clustering_info['iterations'])  # 转换为Python int
                },
                'parameters': params,
                'network_info': network_info  # 添加网络信息
            }

            # 添加文件下载信息
            if save_result['success']:
                result['csv_info'] = save_result

            if viz_result:
                result['visualization'] = {
                    'filename': viz_filename,
                    'path': viz_path
                }

            if boundary_viz_result:
                result['boundary_visualization'] = {
                    'filename': boundary_viz_filename,
                    'path': boundary_viz_result,
                    'boundary_pipe_count': boundary_pipe_count
                }

            return result

        except Exception as e:
            error_msg = f"处理管网分区请求时出错: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'response': error_msg,
                'intent': 'partition_analysis',
                'confidence': 0.0
            }
