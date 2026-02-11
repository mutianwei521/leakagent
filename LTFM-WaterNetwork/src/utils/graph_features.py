#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图特征提取器
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from loguru import logger


class GraphFeatureExtractor:
    """图特征提取器，提取丰富的拓扑特征"""
    
    def __init__(self):
        """初始化特征提取器"""
        pass
    
    def extract_node_features(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        提取节点级别的特征
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            
        Returns:
            Dict[str, np.ndarray]: 特征字典
        """
        features = {}
        
        # 基本拓扑特征
        features['degree'] = np.array([graph.degree(node) for node in graph.nodes()])
        features['clustering'] = np.array([nx.clustering(graph, node) for node in graph.nodes()])
        
        # 中心性特征
        try:
            degree_centrality = nx.degree_centrality(graph)
            features['degree_centrality'] = np.array([degree_centrality[node] for node in graph.nodes()])
            
            closeness_centrality = nx.closeness_centrality(graph)
            features['closeness_centrality'] = np.array([closeness_centrality[node] for node in graph.nodes()])
            
            betweenness_centrality = nx.betweenness_centrality(graph)
            features['betweenness_centrality'] = np.array([betweenness_centrality[node] for node in graph.nodes()])
            
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            features['eigenvector_centrality'] = np.array([eigenvector_centrality[node] for node in graph.nodes()])
            
        except Exception as e:
            logger.warning(f"中心性计算失败: {e}")
            # 使用默认值
            n_nodes = len(graph.nodes())
            features['degree_centrality'] = np.ones(n_nodes) / n_nodes
            features['closeness_centrality'] = np.ones(n_nodes) / n_nodes
            features['betweenness_centrality'] = np.zeros(n_nodes)
            features['eigenvector_centrality'] = np.ones(n_nodes) / n_nodes
        
        # 节点权重特征
        if node_weights:
            node_list = list(graph.nodes())
            weights = np.array([node_weights.get(node, 0.0) for node in node_list])
            features['node_weights'] = weights
            features['weight_normalized'] = self._normalize_features(weights)
        else:
            features['node_weights'] = np.zeros(len(graph.nodes()))
            features['weight_normalized'] = np.zeros(len(graph.nodes()))
        
        # 邻居特征
        features['neighbor_count'] = features['degree']
        features['neighbor_weight_sum'] = self._calculate_neighbor_weight_sum(graph, node_weights)
        
        return features
    
    def extract_graph_features(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> np.ndarray:
        """
        提取图级别的特征
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            
        Returns:
            np.ndarray: 图特征向量
        """
        features = []
        
        # 基本图统计
        features.append(len(graph.nodes()))  # 节点数
        features.append(len(graph.edges()))  # 边数
        features.append(nx.density(graph))   # 密度
        
        # 连通性特征
        features.append(1.0 if nx.is_connected(graph) else 0.0)
        features.append(len(list(nx.connected_components(graph))))  # 连通分量数
        
        # 路径特征
        try:
            if nx.is_connected(graph):
                features.append(nx.average_shortest_path_length(graph))
                features.append(nx.diameter(graph))
            else:
                features.append(0.0)
                features.append(0.0)
        except:
            features.append(0.0)
            features.append(0.0)
        
        # 聚类系数
        features.append(nx.average_clustering(graph))
        
        # 度分布特征
        degrees = [graph.degree(node) for node in graph.nodes()]
        if degrees:
            features.append(np.mean(degrees))
            features.append(np.std(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 权重特征
        if node_weights:
            weights = list(node_weights.values())
            if weights:
                features.append(np.mean(weights))
                features.append(np.std(weights))
                features.append(np.max(weights))
                features.append(np.min(weights))
                features.append(np.sum(weights))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def create_enhanced_embedding(self, graph: nx.Graph, node_weights: Dict[str, float] = None, 
                                base_embedding: np.ndarray = None) -> np.ndarray:
        """
        创建增强的图嵌入，结合拓扑特征和基础嵌入
        
        Args:
            graph: NetworkX图
            node_weights: 节点权重字典
            base_embedding: 基础嵌入（如Graph2Vec）
            
        Returns:
            np.ndarray: 增强的图嵌入
        """
        # 提取图级别特征
        graph_features = self.extract_graph_features(graph, node_weights)
        
        # 提取节点级别特征并聚合
        node_features = self.extract_node_features(graph, node_weights)
        
        # 聚合节点特征
        aggregated_features = []
        for feature_name, feature_values in node_features.items():
            if len(feature_values) > 0:
                aggregated_features.extend([
                    np.mean(feature_values),
                    np.std(feature_values),
                    np.max(feature_values),
                    np.min(feature_values)
                ])
            else:
                aggregated_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 组合所有特征
        enhanced_features = np.concatenate([
            graph_features,
            np.array(aggregated_features)
        ])
        
        # 如果有基础嵌入，则结合
        if base_embedding is not None:
            enhanced_features = np.concatenate([base_embedding, enhanced_features])
        
        return enhanced_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        if len(features) == 0:
            return features
        
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val - min_val == 0:
            return np.zeros_like(features)
        
        return (features - min_val) / (max_val - min_val)
    
    def _calculate_neighbor_weight_sum(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> np.ndarray:
        """计算每个节点的邻居权重和"""
        if not node_weights:
            return np.zeros(len(graph.nodes()))
        
        neighbor_sums = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbor_weight_sum = sum(node_weights.get(neighbor, 0.0) for neighbor in neighbors)
            neighbor_sums.append(neighbor_weight_sum)
        
        return np.array(neighbor_sums)
    
    def extract_subgraph_features(self, subgraphs: List[nx.Graph], 
                                node_weights_list: List[Dict[str, float]] = None) -> List[np.ndarray]:
        """
        提取子图特征
        
        Args:
            subgraphs: 子图列表
            node_weights_list: 节点权重列表
            
        Returns:
            List[np.ndarray]: 子图特征列表
        """
        features_list = []
        
        for i, subgraph in enumerate(subgraphs):
            node_weights = node_weights_list[i] if node_weights_list else None
            features = self.extract_graph_features(subgraph, node_weights)
            features_list.append(features)
        
        return features_list
