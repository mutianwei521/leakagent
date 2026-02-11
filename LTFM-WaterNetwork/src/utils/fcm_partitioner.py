# -*- coding: utf-8 -*-
"""
FCM分区模块
实现基于压力灵敏度和管长权重的FCM管网分区算法
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False


class FCMPartitioner:
    """基于FCM的管网分区器"""
    
    def __init__(self, n_clusters: int = 5, m: float = 2.0, 
                 max_iter: int = 100, error: float = 1e-5):
        """
        初始化FCM分区器
        
        Args:
            n_clusters: 分区数量
            m: 模糊指数
            max_iter: 最大迭代次数
            error: 收敛误差
        """
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.error = float(error)
        self.cluster_centers = None
        self.membership_matrix = None
        self.partition_labels = None
        
    def prepare_features(self, adjacency_matrix: np.ndarray,
                        node_weights: np.ndarray,
                        pipe_lengths: Dict[str, float],
                        pipe_names: List[str],
                        node_names: List[str]) -> np.ndarray:
        """
        准备FCM聚类特征
        
        Args:
            adjacency_matrix: 邻接矩阵
            node_weights: 节点权重（压力灵敏度平均值）
            pipe_lengths: 管道长度字典
            pipe_names: 管道名称列表
            node_names: 节点名称列表
            
        Returns:
            np.ndarray: 特征矩阵 [n_nodes, n_features]
        """
        try:
            n_nodes = len(node_names)
            features = []

            # 数据类型检查和转换
            logger.debug(f"输入参数检查:")
            logger.debug(f"  adjacency_matrix 形状: {adjacency_matrix.shape}, 类型: {adjacency_matrix.dtype}")
            logger.debug(f"  node_weights 形状: {node_weights.shape}, 类型: {node_weights.dtype}")
            logger.debug(f"  node_names 数量: {len(node_names)}")
            logger.debug(f"  pipe_names 数量: {len(pipe_names)}")

            # 确保 node_weights 是数值类型
            if not np.issubdtype(node_weights.dtype, np.number):
                logger.error(f"node_weights 包含非数值数据，类型: {node_weights.dtype}")
                return np.array([])

            # 检查是否包含 NaN 或 inf
            if np.any(np.isnan(node_weights)) or np.any(np.isinf(node_weights)):
                logger.warning("node_weights 包含 NaN 或 inf 值，将替换为0")
                node_weights = np.nan_to_num(node_weights, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. 节点度（连接数）
            node_degrees = np.sum(adjacency_matrix, axis=1)

            # 2. 节点权重统计特征
            avg_node_weights = np.mean(node_weights, axis=1)  # 平均灵敏度
            max_node_weights = np.max(node_weights, axis=1)   # 最大灵敏度
            std_node_weights = np.std(node_weights, axis=1)   # 灵敏度标准差
            
            # 3. 邻居节点特征
            neighbor_avg_weights = np.zeros(n_nodes)
            neighbor_degrees = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                neighbors = np.where(adjacency_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_avg_weights[i] = np.mean(avg_node_weights[neighbors])
                    neighbor_degrees[i] = np.mean(node_degrees[neighbors])
            
            # 4. 管道长度特征（连接到该节点的管道平均长度）
            avg_pipe_lengths = np.zeros(n_nodes)
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}
            
            # 构建节点到管道的映射
            for pipe_name in pipe_names:
                # 这里需要从EPANET处理器获取管道连接信息
                # 暂时使用邻接矩阵信息
                pass
            
            # 组合所有特征
            feature_matrix = np.column_stack([
                node_degrees,           # 节点度
                avg_node_weights,       # 平均压力灵敏度
                max_node_weights,       # 最大压力灵敏度
                std_node_weights,       # 压力灵敏度标准差
                neighbor_avg_weights,   # 邻居平均权重
                neighbor_degrees,       # 邻居平均度
                avg_pipe_lengths        # 平均管道长度
            ])

            # 检查特征矩阵的有效性
            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                logger.warning("特征矩阵包含 NaN 或 inf 值，将替换为0")
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # 检查特征矩阵是否为空或全零
            if feature_matrix.size == 0:
                logger.error("特征矩阵为空")
                return np.array([])

            if np.all(feature_matrix == 0):
                logger.warning("特征矩阵全为零，添加小的随机扰动")
                feature_matrix += np.random.normal(0, 1e-6, feature_matrix.shape)

            # 标准化特征
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)

            logger.info(f"特征矩阵准备完成: {feature_matrix.shape}")
            logger.debug(f"特征矩阵统计: min={np.min(feature_matrix):.6f}, max={np.max(feature_matrix):.6f}")
            return feature_matrix
            
        except Exception as e:
            logger.error(f"准备特征失败: {e}")
            return np.array([])
    
    def partition_network(self, feature_matrix: np.ndarray) -> bool:
        """
        使用FCM对网络进行分区
        
        Args:
            feature_matrix: 特征矩阵
            
        Returns:
            bool: 分区是否成功
        """
        try:
            if feature_matrix.size == 0:
                logger.error("特征矩阵为空")
                return False

            # 检查特征矩阵的有效性
            if not np.issubdtype(feature_matrix.dtype, np.number):
                logger.error(f"特征矩阵包含非数值数据，类型: {feature_matrix.dtype}")
                return False

            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                logger.error("特征矩阵包含 NaN 或 inf 值")
                return False

            # 检查分区数是否合理
            n_samples = feature_matrix.shape[0]
            if self.n_clusters >= n_samples:
                logger.error(f"分区数 {self.n_clusters} 大于等于样本数 {n_samples}")
                return False

            logger.debug(f"开始FCM聚类: 样本数={n_samples}, 特征数={feature_matrix.shape[1]}, 分区数={self.n_clusters}")

            # 转置特征矩阵以适应skfuzzy的输入格式
            data = feature_matrix.T

            # 确保参数类型正确
            n_clusters = int(self.n_clusters)
            m = float(self.m)
            error = float(self.error)
            maxiter = int(self.max_iter)

            logger.debug(f"FCM参数: n_clusters={n_clusters}, m={m}, error={error}, maxiter={maxiter}")

            # 执行FCM聚类
            self.cluster_centers, self.membership_matrix, _, _, _, _, _ = fuzz.cluster.cmeans(
                data, n_clusters, m, error=error, maxiter=maxiter
            )
            
            # 获取每个节点的分区标签（隶属度最大的簇）
            self.partition_labels = np.argmax(self.membership_matrix, axis=0)
            
            # 计算聚类质量指标
            silhouette_avg = silhouette_score(feature_matrix, self.partition_labels)
            
            logger.info(f"FCM分区完成: {self.n_clusters}个分区, 轮廓系数: {silhouette_avg:.3f}")
            
            # 输出分区统计信息
            for i in range(self.n_clusters):
                cluster_size = np.sum(self.partition_labels == i)
                logger.info(f"分区 {i}: {cluster_size} 个节点")
            
            return True
            
        except Exception as e:
            logger.error(f"FCM分区失败: {e}")
            return False
    
    def get_partition_subgraphs(self, graph: nx.Graph, 
                               node_names: List[str]) -> List[nx.Graph]:
        """
        根据分区结果获取子图
        
        Args:
            graph: 原始网络图
            node_names: 节点名称列表
            
        Returns:
            List[nx.Graph]: 分区子图列表
        """
        try:
            if self.partition_labels is None:
                logger.error("未找到分区结果")
                return []
            
            subgraphs = []
            
            for cluster_id in range(self.n_clusters):
                # 获取当前分区的节点
                cluster_nodes = [node_names[i] for i in range(len(node_names)) 
                               if self.partition_labels[i] == cluster_id]
                
                # 创建子图
                subgraph = graph.subgraph(cluster_nodes).copy()
                subgraphs.append(subgraph)
                
                logger.debug(f"分区 {cluster_id} 子图: {len(subgraph.nodes)} 节点, {len(subgraph.edges)} 边")
            
            return subgraphs
            
        except Exception as e:
            logger.error(f"获取分区子图失败: {e}")
            return []
    
    def optimize_partition_number(self, feature_matrix: np.ndarray,
                                 min_clusters: int = 2, max_clusters: int = 10) -> int:
        """
        优化分区数量
        
        Args:
            feature_matrix: 特征矩阵
            min_clusters: 最小分区数
            max_clusters: 最大分区数
            
        Returns:
            int: 最优分区数
        """
        try:
            best_score = -1
            best_n_clusters = self.n_clusters
            scores = []
            
            logger.info(f"优化分区数量: {min_clusters}-{max_clusters}")
            
            for n in range(min_clusters, max_clusters + 1):
                # 临时设置分区数
                original_n_clusters = self.n_clusters
                self.n_clusters = n
                
                # 执行分区
                if self.partition_network(feature_matrix):
                    # 计算轮廓系数
                    score = silhouette_score(feature_matrix, self.partition_labels)
                    scores.append((n, score))
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n
                    
                    logger.info(f"分区数 {n}: 轮廓系数 {score:.3f}")
                
                # 恢复原始分区数
                self.n_clusters = original_n_clusters
            
            # 设置最优分区数
            self.n_clusters = best_n_clusters
            logger.info(f"最优分区数: {best_n_clusters}, 轮廓系数: {best_score:.3f}")
            
            return best_n_clusters
            
        except Exception as e:
            logger.error(f"优化分区数量失败: {e}")
            return self.n_clusters
    
    def get_partition_info(self) -> Dict:
        """
        获取分区信息
        
        Returns:
            Dict: 分区信息字典
        """
        if self.partition_labels is None:
            return {}
        
        info = {
            'n_clusters': self.n_clusters,
            'partition_labels': self.partition_labels.copy(),
            'cluster_sizes': [np.sum(self.partition_labels == i) for i in range(self.n_clusters)],
            'membership_matrix': self.membership_matrix.copy() if self.membership_matrix is not None else None,
            'cluster_centers': self.cluster_centers.copy() if self.cluster_centers is not None else None
        }
        
        return info
    
    def check_connectivity(self, node_connections: np.ndarray, cluster_nodes: np.ndarray) -> np.ndarray:
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

    def find_connected_components(self, connect_matrix: np.ndarray) -> List[List[int]]:
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

    def remove_outliers_iteratively(self, wn, nodes: List[str], demands: List[str],
                                     raw_labels: np.ndarray, k_nearest: int = 10,
                                     outliers_detection: bool = True, seed: int = 42,
                                     output_dir: str = None) -> np.ndarray:
        """
        迭代处理两类离群点

        Args:
            wn: WNTR网络对象
            nodes: 所有节点名称列表
            demands: 需求节点名称列表
            raw_labels: 原始标签数组
            k_nearest: KNN参数
            outliers_detection: 是否进行离群点检测
            seed: 随机种子
            output_dir: 输出目录（用于保存可视化图像）

        Returns:
            处理后的标签数组
        """
        if not WNTR_AVAILABLE:
            logger.error("WNTR库未安装，无法进行离群点检测")
            return raw_labels

        if not outliers_detection:
            logger.info("跳过离群点检测")
            return raw_labels

        logger.info("开始迭代离群点检测")

        # 创建完整的标签数组
        all_labels = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                all_labels[i] = raw_labels[idx]
            else:
                all_labels[i] = 0

        # 保存初始分区可视化
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_partition(wn, nodes, all_labels,
                                    os.path.join(output_dir, 'partition_0_initial.png'),
                                    'Initial Partition (Before Outlier Removal)')

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
            logger.info(f"离群点检测迭代 {number_iter}, 剩余未分配节点: {zero_count}")

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
                    k = min(k_nearest, len(distances))
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

            # 保存当前迭代的可视化
            if output_dir:
                self.visualize_partition(wn, nodes, all_labels,
                                        os.path.join(output_dir, f'partition_{number_iter}_iteration.png'),
                                        f'Partition After Iteration {number_iter}')

        # 检查是否有分区被完全消除，如果有则恢复最大的连通分量
        original_partitions = set(raw_labels)
        current_partitions = set(all_labels[all_labels > 0])

        lost_partitions = original_partitions - current_partitions
        if lost_partitions:
            logger.info(f"检测到被完全消除的分区: {sorted(lost_partitions)}")

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

                            logger.info(f"恢复分区{lost_partition}的最大连通分量: {len(main_component)}个节点")
                    else:
                        # 只有一个节点，直接恢复
                        all_labels[original_nodes[0]] = lost_partition
                        logger.info(f"恢复分区{lost_partition}的单个节点")

        # 更新raw_labels
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                raw_labels[idx] = all_labels[i]

        # 最终验证分区数量
        final_partitions = len(set(raw_labels[raw_labels > 0]))
        expected_partitions = self.n_clusters

        if final_partitions != expected_partitions:
            logger.info(f"⚠️ 分区数量不匹配: 期望{expected_partitions}个，实际{final_partitions}个")
        else:
            logger.info(f"✅ 分区数量验证通过: {final_partitions}个分区")

        # 检查未分配节点数量
        unassigned_count = np.sum(raw_labels == 0)
        if unassigned_count > 0:
            logger.info(f"检测到{unassigned_count}个未分配节点，开始最近邻分配")
            # 进行最近邻分配
            final_labels = self.assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, raw_labels, k_nearest, seed)

            # 验证最近邻分配结果
            final_unassigned = np.sum(final_labels == 0)
            if final_unassigned == 0:
                logger.info("✅ 所有节点已通过最近邻分配成功分配到分区")
            else:
                logger.info(f"⚠️ 最近邻分配后仍有{final_unassigned}个节点未分配")

            logger.info(f"离群点检测和最近邻分配完成，迭代次数: {number_iter}")

            # 保存最终结果可视化
            if output_dir:
                # 创建完整的标签数组用于可视化
                final_all_labels = np.zeros(len(nodes))
                for i, node in enumerate(nodes):
                    if node in demands:
                        idx = demands.index(node)
                        final_all_labels[i] = final_labels[idx]
                self.visualize_partition(wn, nodes, final_all_labels,
                                        os.path.join(output_dir, 'partition_final.png'),
                                        'Final Partition (After All Processing)')

            return final_labels
        else:
            logger.info("✅ 所有节点已分配，无需最近邻分配")
            logger.info(f"离群点检测完成，迭代次数: {number_iter}")

            # 保存最终结果可视化
            if output_dir:
                self.visualize_partition(wn, nodes, all_labels,
                                        os.path.join(output_dir, 'partition_final.png'),
                                        'Final Partition (After All Processing)')

            return raw_labels

    def assign_unassigned_nodes_by_nearest_neighbor(self, wn, nodes: List[str], demands: List[str],
                                                     labels: np.ndarray, k_nearest: int = 10,
                                                     seed: int = 42) -> np.ndarray:
        """将未分配节点分配到最近邻分区"""
        if not WNTR_AVAILABLE:
            logger.error("WNTR库未安装，无法进行最近邻分配")
            return labels

        # 找到未分配的需水节点
        unassigned_indices = []
        for i, demand_node in enumerate(demands):
            if labels[i] == 0:
                unassigned_indices.append(i)

        if len(unassigned_indices) == 0:
            return labels

        logger.info(f"开始为{len(unassigned_indices)}个未分配需水节点分配最近邻分区")

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
                        layout = nx.spring_layout(G, seed=seed)
                    coord = layout.get(node_name, (0, 0))
            except:
                if layout is None:
                    G = wn.to_graph().to_undirected()
                    layout = nx.spring_layout(G, seed=seed)
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

            logger.info(f"节点{unassigned_node}分配到分区{nearest_partition}，最近距离: {min_distance:.4f}")

            # 更新分区信息，以便后续节点可以考虑这个新分配的节点
            if nearest_partition not in assigned_nodes_by_partition:
                assigned_nodes_by_partition[nearest_partition] = []
            assigned_nodes_by_partition[nearest_partition].append((unassigned_node, unassigned_coord))

        return labels_copy

    def visualize_partition(self, wn, nodes: List[str], labels: np.ndarray,
                           output_path: str, title: str = "Network Partition"):
        """
        可视化分区结果（使用WNTR绘图，保持原始坐标，Nature期刊风格）

        Args:
            wn: WNTR网络对象
            nodes: 节点名称列表
            labels: 节点标签数组
            output_path: 输出图像路径
            title: 图像标题
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端

            # Nature期刊风格设置
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['xtick.major.width'] = 1.2
            plt.rcParams['ytick.major.width'] = 1.2

            # 创建图形，增大尺寸
            fig, ax = plt.subplots(figsize=(12, 10))

            # 为每个分区分配高对比度颜色（Nature期刊常用配色）
            unique_labels = np.unique(labels)
            # 使用高对比度的颜色：红、蓝、绿、橙、紫、棕等
            distinct_colors = [
                '#E41A1C',  # 红色
                '#377EB8',  # 蓝色
                '#4DAF4A',  # 绿色
                '#FF7F00',  # 橙色
                '#984EA3',  # 紫色
                '#A65628',  # 棕色
                '#F781BF',  # 粉色
                '#999999',  # 灰色
                '#66C2A5',  # 青绿色
                '#FC8D62',  # 浅橙色
            ]

            color_map = {}
            for i, label in enumerate(unique_labels):
                if label == 0:
                    continue  # 跳过未分配标签
                color_idx = (int(label) - 1) % len(distinct_colors)
                color_map[label] = distinct_colors[color_idx]

            # 使用WNTR绘制网络（只绘制管道）
            wntr.graphics.plot_network(
                wn,
                node_size=0,  # 不绘制节点
                link_width=1.5,
                add_colorbar=False,
                ax=ax
            )

            # 按分区分组绘制节点
            partition_nodes = {}
            unassigned_nodes = []

            for i, node_name in enumerate(nodes):
                label = labels[i]
                if label == 0:
                    unassigned_nodes.append(node_name)
                else:
                    if label not in partition_nodes:
                        partition_nodes[label] = []
                    partition_nodes[label].append(node_name)

            # 绘制已分配的节点（不显示节点标签）
            for label, node_list in partition_nodes.items():
                color = color_map[label]

                # 获取节点坐标
                node_coords = []
                for node_name in node_list:
                    node = wn.get_node(node_name)
                    node_coords.append(node.coordinates)

                if node_coords:
                    x_coords = [coord[0] for coord in node_coords]
                    y_coords = [coord[1] for coord in node_coords]

                    ax.scatter(x_coords, y_coords,
                             c=color,
                             s=300,  # 节点大小
                             edgecolors='black',
                             linewidths=1.5,
                             zorder=3,
                             alpha=0.9,
                             label=f'Partition {int(label)}')

            # 绘制未分配的节点
            if unassigned_nodes:
                node_coords = []
                for node_name in unassigned_nodes:
                    node = wn.get_node(node_name)
                    node_coords.append(node.coordinates)

                if node_coords:
                    x_coords = [coord[0] for coord in node_coords]
                    y_coords = [coord[1] for coord in node_coords]

                    ax.scatter(x_coords, y_coords,
                             c='white',
                             marker='X',
                             s=250,
                             edgecolors='#D62728',  # 深红色
                             linewidths=2.5,
                             zorder=4,
                             label='Unassigned')

            # 设置标题（Nature风格：简洁）
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # 设置坐标轴标签（Nature风格）
            ax.set_xlabel('X Coordinate (ft)', fontsize=12)
            ax.set_ylabel('Y Coordinate (ft)', fontsize=12)

            # 添加网格（Nature风格：细线）
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')

            # 设置背景颜色（Nature风格：白色）
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            # 设置坐标轴样式
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_linewidth(1.2)
            ax.spines['right'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.spines['left'].set_linewidth(1.2)

            # 添加图例（放大，避免遮挡）
            legend = ax.legend(
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),  # 放在图的右侧，稍远一些
                fontsize=12,
                frameon=True,
                fancybox=False,  # Nature风格：方形边框
                shadow=False,    # Nature风格：无阴影
                # title='Partitions',
                # title_fontsize=13,
                markerscale=0.5,  # 放大图例中的标记，避免遮挡
                handletextpad=1.0,  # 增加标记和文字之间的间距
                borderpad=1.2,      # 增加图例内边距
                labelspacing=1.2    # 增加标签之间的间距
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.2)

            # 调整布局，为图例留出更多空间
            plt.tight_layout(rect=[0, 0, 0.80, 1])

            # 保存图像（Nature期刊要求：高分辨率）
            plt.savefig(output_path, dpi=600, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       pad_inches=0.1)
            plt.close()

            logger.info(f"分区可视化已保存到: {output_path}")
            return True

        except Exception as e:
            logger.error(f"可视化分区失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_partition_results(self, output_dir: str, node_names: List[str]) -> bool:
        """
        保存分区结果

        Args:
            output_dir: 输出目录
            node_names: 节点名称列表

        Returns:
            bool: 保存是否成功
        """
        try:
            import os
            import pandas as pd

            os.makedirs(output_dir, exist_ok=True)

            if self.partition_labels is not None:
                # 保存分区标签
                partition_df = pd.DataFrame({
                    'node_name': node_names,
                    'partition_id': self.partition_labels
                })
                partition_df.to_csv(os.path.join(output_dir, 'fcm_partition.csv'), index=False)

                # 保存隶属度矩阵
                if self.membership_matrix is not None:
                    membership_df = pd.DataFrame(
                        self.membership_matrix.T,
                        columns=[f'cluster_{i}' for i in range(self.n_clusters)],
                        index=node_names
                    )
                    membership_df.to_csv(os.path.join(output_dir, 'fcm_membership.csv'))

                logger.info(f"FCM分区结果已保存到 {output_dir}")
                return True

            return False

        except Exception as e:
            logger.error(f"保存分区结果失败: {e}")
            return False
