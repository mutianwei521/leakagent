# -*- coding: utf-8 -*-
"""
Graph2Vec编码器模块
实现拓扑图到Embedding的转换，作为Visual Encoder
优化版：使用直接数值特征编码替代Word2Vec哈希编码
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
from scipy import stats
import hashlib
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Graph2VecEncoder(nn.Module):
    """Graph2Vec编码器，将图结构转换为嵌入向量（优化版）"""
    
    def __init__(self, embedding_dim: int = 128, wl_iterations: int = 3,
                 workers: int = 4, epochs: int = 10, min_count: int = 5,
                 learning_rate: float = 0.025):
        """
        初始化Graph2Vec编码器
        
        Args:
            embedding_dim: 嵌入维度
            wl_iterations: Weisfeiler-Lehman迭代次数（保留接口兼容性）
            workers: 并行工作数（保留接口兼容性）
            epochs: 训练轮数（保留接口兼容性）
            min_count: 最小词频（保留接口兼容性）
            learning_rate: 学习率（保留接口兼容性）
        """
        super(Graph2VecEncoder, self).__init__()

        # 确保参数类型正确
        self.embedding_dim = int(embedding_dim)
        self.wl_iterations = int(wl_iterations)
        self.workers = int(workers)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.learning_rate = float(learning_rate)
        
        self.model = None  # 保留兼容性，不再使用 Word2Vec
        self.vocabulary = set()
        self.graph_embeddings = {}
        
        # 拓扑特征缓存：同一图结构只计算一次
        self._topology_cache = {}
        
        # 可学习的特征投影层：将原始特征投影到 embedding_dim
        self._raw_feature_dim = None  # 将在第一次编码时确定
        self._projection = None  # 延迟初始化
        
        # 特征归一化统计（在 train_model 时计算）
        self._feature_mean = None
        self._feature_std = None
        
        # 冻结参数（不训练Graph2Vec）
        self.freeze_parameters = True
        self._trained = False
        
    def _get_graph_hash(self, graph: nx.Graph) -> str:
        """
        计算图的拓扑哈希（不含权重），用于缓存拓扑特征
        """
        # 使用节点集合 + 边集合的哈希
        nodes_str = str(sorted(graph.nodes()))
        edges_str = str(sorted(graph.edges()))
        return hashlib.md5(f"{nodes_str}_{edges_str}".encode()).hexdigest()
    
    def _extract_topology_features(self, graph: nx.Graph) -> np.ndarray:
        """
        提取图的拓扑特征（与权重无关，可缓存）
        
        Args:
            graph: NetworkX图
            
        Returns:
            np.ndarray: 拓扑特征向量
        """
        graph_hash = self._get_graph_hash(graph)
        
        if graph_hash in self._topology_cache:
            return self._topology_cache[graph_hash]
        
        features = []
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        
        # 基本图统计 (5维)
        features.append(n_nodes)
        features.append(n_edges)
        features.append(nx.density(graph))
        features.append(1.0 if nx.is_connected(graph) else 0.0)
        features.append(len(list(nx.connected_components(graph))))
        
        # 路径特征 (2维)
        try:
            if nx.is_connected(graph):
                features.append(nx.average_shortest_path_length(graph))
                features.append(nx.diameter(graph))
            else:
                features.append(0.0)
                features.append(0.0)
        except Exception:
            features.append(0.0)
            features.append(0.0)
        
        # 聚类系数 (1维)
        features.append(nx.average_clustering(graph))
        
        # 度分布特征 (6维)
        degrees = np.array([graph.degree(node) for node in graph.nodes()])
        if len(degrees) > 0:
            features.append(np.mean(degrees))
            features.append(np.std(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
            features.append(np.median(degrees))
            # 度分布偏度
            features.append(float(stats.skew(degrees)) if len(degrees) > 2 else 0.0)
        else:
            features.extend([0.0] * 6)
        
        # 中心性特征 (4维)
        try:
            betweenness = list(nx.betweenness_centrality(graph).values())
            features.append(np.mean(betweenness))
            features.append(np.max(betweenness))
        except Exception:
            features.extend([0.0, 0.0])
        
        try:
            closeness = list(nx.closeness_centrality(graph).values())
            features.append(np.mean(closeness))
            features.append(np.max(closeness))
        except Exception:
            features.extend([0.0, 0.0])
        
        # 三角形特征 (2维)
        try:
            triangles = list(nx.triangles(graph).values())
            features.append(sum(triangles) / (3 * max(1, n_nodes)))  # 归一化三角形密度
            features.append(np.mean(triangles) if triangles else 0.0)
        except Exception:
            features.extend([0.0, 0.0])
        
        topo_features = np.array(features, dtype=np.float64)
        self._topology_cache[graph_hash] = topo_features
        return topo_features
    
    def _extract_weight_features(self, graph: nx.Graph, 
                                  node_weights: Optional[Dict] = None) -> np.ndarray:
        """
        提取基于节点权重的特征（每个样本不同）
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            
        Returns:
            np.ndarray: 权重特征向量
        """
        features = []
        
        if node_weights is None or len(node_weights) == 0:
            # 无权重时返回全零特征
            return np.zeros(50, dtype=np.float64)
        
        # 获取图中节点的权重
        weights = np.array([node_weights.get(node, 0.0) for node in graph.nodes()], dtype=np.float64)
        
        # 1. 基本统计 (8维)
        features.append(np.mean(weights))
        features.append(np.std(weights))
        features.append(np.max(weights))
        features.append(np.min(weights))
        features.append(np.median(weights))
        features.append(np.sum(np.abs(weights)))
        features.append(float(stats.skew(weights)) if len(weights) > 2 else 0.0)
        features.append(float(stats.kurtosis(weights)) if len(weights) > 3 else 0.0)
        
        # 2. 分位数特征 (5维)
        for q in [10, 25, 75, 90, 95]:
            features.append(np.percentile(weights, q))
        
        # 3. 权重分布直方图 (15维)
        if np.max(np.abs(weights)) > 1e-10:
            hist, _ = np.histogram(weights, bins=15, density=True)
            features.extend(hist.tolist())
        else:
            features.extend([0.0] * 15)
        
        # 4. 正/负/零统计 (3维)
        n_total = max(1, len(weights))
        features.append(np.sum(weights > 0.01) / n_total)   # 正值比例
        features.append(np.sum(weights < -0.01) / n_total)  # 负值比例
        features.append(np.sum(np.abs(weights) <= 0.01) / n_total)  # 近零比例
        
        # 5. 异常值特征 (3维)
        if np.std(weights) > 1e-10:
            z_scores = np.abs((weights - np.mean(weights)) / np.std(weights))
            features.append(np.sum(z_scores > 2.0) / n_total)  # 2-sigma 外的比例
            features.append(np.sum(z_scores > 3.0) / n_total)  # 3-sigma 外的比例
            features.append(np.max(z_scores))                   # 最大Z分数
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 6. 拓扑-权重交互特征 (10维)
        degrees = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float64)
        
        if len(degrees) > 0 and len(weights) > 0:
            # 高度节点的平均权重
            degree_threshold = np.percentile(degrees, 75) if len(degrees) > 3 else np.max(degrees)
            high_degree_mask = degrees >= degree_threshold
            low_degree_mask = degrees <= np.percentile(degrees, 25) if len(degrees) > 3 else degrees <= np.min(degrees)
            
            features.append(np.mean(weights[high_degree_mask]) if np.sum(high_degree_mask) > 0 else 0.0)
            features.append(np.mean(weights[low_degree_mask]) if np.sum(low_degree_mask) > 0 else 0.0)
            features.append(np.std(weights[high_degree_mask]) if np.sum(high_degree_mask) > 1 else 0.0)
            
            # 度-权重相关性
            if np.std(degrees) > 1e-10 and np.std(weights) > 1e-10:
                features.append(float(np.corrcoef(degrees, weights)[0, 1]))
            else:
                features.append(0.0)
            
            # 邻居权重特征
            neighbor_weight_diffs = []
            for node in graph.nodes():
                node_weight = node_weights.get(node, 0.0)
                for neighbor in graph.neighbors(node):
                    neighbor_weight = node_weights.get(neighbor, 0.0)
                    neighbor_weight_diffs.append(abs(node_weight - neighbor_weight))
            
            if neighbor_weight_diffs:
                nwd = np.array(neighbor_weight_diffs)
                features.append(np.mean(nwd))   # 平均邻居权重差
                features.append(np.std(nwd))    # 邻居权重差的标准差
                features.append(np.max(nwd))    # 最大邻居权重差
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # 权重空间自相关 (Moran's I 简化版)
            if len(weights) > 1:
                w_mean = np.mean(weights)
                w_dev = weights - w_mean
                numerator = 0.0
                denominator = np.sum(w_dev ** 2)
                n_edges_counted = 0
                
                for u, v in graph.edges():
                    try:
                        u_idx = list(graph.nodes()).index(u)
                        v_idx = list(graph.nodes()).index(v)
                        numerator += w_dev[u_idx] * w_dev[v_idx]
                        n_edges_counted += 1
                    except (ValueError, IndexError):
                        continue
                
                if denominator > 1e-10 and n_edges_counted > 0:
                    morans_i = (len(weights) * numerator) / (n_edges_counted * denominator)
                    features.append(float(morans_i))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # 权重能量（L2范数）
            features.append(np.linalg.norm(weights))
            # 权重熵
            abs_weights = np.abs(weights) + 1e-10
            probs = abs_weights / np.sum(abs_weights)
            features.append(float(stats.entropy(probs)))
        else:
            features.extend([0.0] * 10)
        
        # 确保固定长度 (6维 padding/truncation to reach 50)
        result = np.array(features[:50], dtype=np.float64)
        if len(result) < 50:
            result = np.pad(result, (0, 50 - len(result)))
        
        return result
    
    def _encode_graph_numerical(self, graph: nx.Graph,
                                 node_weights: Optional[Dict] = None) -> np.ndarray:
        """
        使用直接数值特征编码图（高效版本）
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            
        Returns:
            np.ndarray: 特征向量
        """
        # 拓扑特征（缓存）
        topo_features = self._extract_topology_features(graph)
        
        # 权重特征（每样本不同）
        weight_features = self._extract_weight_features(graph, node_weights)
        
        # 拼接
        raw_features = np.concatenate([topo_features, weight_features])
        
        return raw_features
    
    def train_model(self, graphs: List[nx.Graph],
                   node_weights_list: Optional[List[Dict]] = None) -> bool:
        """
        训练Graph2Vec模型（计算特征归一化统计量并初始化投影层）
        
        Args:
            graphs: 图列表
            node_weights_list: 节点权重列表
            
        Returns:
            bool: 训练是否成功
        """
        try:
            logger.info(f"开始训练Graph2Vec编码器（数值特征模式）")
            
            # 预计算所有图的拓扑特征（缓存）
            for graph in graphs:
                self._extract_topology_features(graph)
            logger.info(f"拓扑特征已缓存: {len(self._topology_cache)} 个唯一图结构")
            
            # 计算特征归一化统计量
            all_features = []
            for i, graph in enumerate(graphs):
                node_weights = node_weights_list[i] if node_weights_list else None
                features = self._encode_graph_numerical(graph, node_weights)
                all_features.append(features)
            
            all_features = np.stack(all_features)
            self._raw_feature_dim = all_features.shape[1]
            
            # 计算均值和标准差用于归一化
            self._feature_mean = np.mean(all_features, axis=0)
            self._feature_std = np.std(all_features, axis=0)
            self._feature_std[self._feature_std < 1e-10] = 1.0  # 避免除零
            
            # 初始化线性投影层
            self._projection = nn.Linear(self._raw_feature_dim, self.embedding_dim)
            nn.init.xavier_uniform_(self._projection.weight)
            nn.init.zeros_(self._projection.bias)
            
            self._trained = True
            self.model = True  # 兼容性标记：表示模型已训练
            
            logger.info(f"Graph2Vec编码器训练完成: 原始特征维度={self._raw_feature_dim}, "
                        f"输出维度={self.embedding_dim}")
            
            # 如果设置为冻结参数，则不允许进一步训练
            if self.freeze_parameters:
                if self._projection is not None:
                    for param in self._projection.parameters():
                        param.requires_grad = False
                logger.info("Graph2Vec参数已冻结")
            
            return True
            
        except Exception as e:
            logger.error(f"训练Graph2Vec模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def encode_graph(self, graph: nx.Graph, 
                    node_weights: Optional[Dict] = None,
                    graph_id: Optional[str] = None) -> torch.Tensor:
        """
        编码单个图
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            graph_id: 图ID（用于缓存）
            
        Returns:
            torch.Tensor: 图嵌入向量
        """
        try:
            if not self._trained:
                logger.error("模型未训练")
                return torch.zeros(self.embedding_dim)
            
            # 检查缓存
            if graph_id and graph_id in self.graph_embeddings:
                return self.graph_embeddings[graph_id]
            
            # 提取数值特征
            raw_features = self._encode_graph_numerical(graph, node_weights)
            
            # 归一化
            if self._feature_mean is not None:
                raw_features = (raw_features - self._feature_mean) / self._feature_std
            
            # 转为张量并投影到 embedding_dim
            features_tensor = torch.tensor(raw_features, dtype=torch.float32)
            
            if self._projection is not None:
                with torch.no_grad():
                    embedding = self._projection(features_tensor)
            else:
                # 回退：截断或填充
                if len(raw_features) >= self.embedding_dim:
                    embedding = features_tensor[:self.embedding_dim]
                else:
                    embedding = torch.zeros(self.embedding_dim)
                    embedding[:len(raw_features)] = features_tensor
            
            # 缓存结果
            if graph_id:
                self.graph_embeddings[graph_id] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"编码图失败: {e}")
            return torch.zeros(self.embedding_dim)
    
    def encode_graphs(self, graphs: List[nx.Graph],
                     node_weights_list: Optional[List[Dict]] = None,
                     graph_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        批量编码图
        
        Args:
            graphs: 图列表
            node_weights_list: 节点权重列表
            graph_ids: 图ID列表
            
        Returns:
            torch.Tensor: 图嵌入矩阵 [n_graphs, embedding_dim]
        """
        try:
            embeddings = []
            
            for i, graph in enumerate(graphs):
                node_weights = node_weights_list[i] if node_weights_list else None
                graph_id = graph_ids[i] if graph_ids else None
                
                embedding = self.encode_graph(graph, node_weights, graph_id)
                embeddings.append(embedding)
            
            return torch.stack(embeddings)
            
        except Exception as e:
            logger.error(f"批量编码图失败: {e}")
            return torch.zeros(len(graphs), self.embedding_dim)
    
    def forward(self, graphs: Union[nx.Graph, List[nx.Graph]],
               node_weights: Optional[Union[Dict, List[Dict]]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            graphs: 图或图列表
            node_weights: 节点权重
            
        Returns:
            torch.Tensor: 图嵌入
        """
        if isinstance(graphs, list):
            return self.encode_graphs(graphs, node_weights)
        else:
            return self.encode_graph(graphs, node_weights).unsqueeze(0)
    
    def save_model(self, model_path: str) -> bool:
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            state = {
                'embedding_dim': self.embedding_dim,
                'raw_feature_dim': self._raw_feature_dim,
                'feature_mean': self._feature_mean,
                'feature_std': self._feature_std,
                'topology_cache': self._topology_cache,
                'trained': self._trained,
            }
            if self._projection is not None:
                state['projection_state_dict'] = self._projection.state_dict()
            
            # 保存为 .pth 格式
            save_path = model_path.replace('.model', '.pth') if model_path.endswith('.model') else model_path
            torch.save(state, save_path)
            logger.info(f"Graph2Vec模型已保存到 {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            # 尝试加载新格式
            load_path = model_path.replace('.model', '.pth') if model_path.endswith('.model') else model_path
            state = torch.load(load_path, weights_only=False)
            
            self.embedding_dim = state['embedding_dim']
            self._raw_feature_dim = state['raw_feature_dim']
            self._feature_mean = state['feature_mean']
            self._feature_std = state['feature_std']
            self._topology_cache = state.get('topology_cache', {})
            self._trained = state['trained']
            self.model = True  # 兼容性标记
            
            if 'projection_state_dict' in state and self._raw_feature_dim is not None:
                self._projection = nn.Linear(self._raw_feature_dim, self.embedding_dim)
                self._projection.load_state_dict(state['projection_state_dict'])
            
            logger.info(f"Graph2Vec模型已从 {load_path} 加载")
            
            if self.freeze_parameters and self._projection is not None:
                for param in self._projection.parameters():
                    param.requires_grad = False
                    
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
