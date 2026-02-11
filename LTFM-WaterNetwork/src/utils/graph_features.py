#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph Feature Extractor
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from loguru import logger


class GraphFeatureExtractor:
    """Graph Feature Extractor, extracts rich topological features"""
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_node_features(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """
        Extract node-level features
        
        Args:
            graph: NetworkX Graph
            node_weights: Node weights dictionary
            
        Returns:
            Dict[str, np.ndarray]: Feature dictionary
        """
        features = {}
        
        # Basic topological features
        features['degree'] = np.array([graph.degree(node) for node in graph.nodes()])
        features['clustering'] = np.array([nx.clustering(graph, node) for node in graph.nodes()])
        
        # Centrality features
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
            logger.warning(f"Centrality calculation failed: {e}")
            # Use default values
            n_nodes = len(graph.nodes())
            features['degree_centrality'] = np.ones(n_nodes) / n_nodes
            features['closeness_centrality'] = np.ones(n_nodes) / n_nodes
            features['betweenness_centrality'] = np.zeros(n_nodes)
            features['eigenvector_centrality'] = np.ones(n_nodes) / n_nodes
        
        # Node weight features
        if node_weights:
            node_list = list(graph.nodes())
            weights = np.array([node_weights.get(node, 0.0) for node in node_list])
            features['node_weights'] = weights
            features['weight_normalized'] = self._normalize_features(weights)
        else:
            features['node_weights'] = np.zeros(len(graph.nodes()))
            features['weight_normalized'] = np.zeros(len(graph.nodes()))
        
        # Neighbor features
        features['neighbor_count'] = features['degree']
        features['neighbor_weight_sum'] = self._calculate_neighbor_weight_sum(graph, node_weights)
        
        return features
    
    def extract_graph_features(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> np.ndarray:
        """
        Extract graph-level features
        
        Args:
            graph: NetworkX Graph
            node_weights: Node weights dictionary
            
        Returns:
            np.ndarray: Graph feature vector
        """
        features = []
        
        # Basic graph statistics
        features.append(len(graph.nodes()))  # Number of nodes
        features.append(len(graph.edges()))  # Number of edges
        features.append(nx.density(graph))   # Density
        
        # Connectivity features
        features.append(1.0 if nx.is_connected(graph) else 0.0)
        features.append(len(list(nx.connected_components(graph))))  # Number of connected components
        
        # Path features
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
        
        # Clustering coefficient
        features.append(nx.average_clustering(graph))
        
        # Degree distribution features
        degrees = [graph.degree(node) for node in graph.nodes()]
        if degrees:
            features.append(np.mean(degrees))
            features.append(np.std(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Weight features
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
        Create enhanced graph embedding, combining topological features and base embedding
        
        Args:
            graph: NetworkX Graph
            node_weights: Node weights dictionary
            base_embedding: Base embedding (e.g., Graph2Vec)
            
        Returns:
            np.ndarray: Enhanced graph embedding
        """
        # Extract graph-level features
        graph_features = self.extract_graph_features(graph, node_weights)
        
        # Extract node-level features and aggregate
        node_features = self.extract_node_features(graph, node_weights)
        
        # Aggregate node features
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
        
        # Combine all features
        enhanced_features = np.concatenate([
            graph_features,
            np.array(aggregated_features)
        ])
        
        # If base embedding exists, combine it
        if base_embedding is not None:
            enhanced_features = np.concatenate([base_embedding, enhanced_features])
        
        return enhanced_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features"""
        if len(features) == 0:
            return features
        
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val - min_val == 0:
            return np.zeros_like(features)
        
        return (features - min_val) / (max_val - min_val)
    
    def _calculate_neighbor_weight_sum(self, graph: nx.Graph, node_weights: Dict[str, float] = None) -> np.ndarray:
        """Calculate the sum of neighbor weights for each node"""
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
        Extract subgraph features
        
        Args:
            subgraphs: List of subgraphs
            node_weights_list: List of node weights
            
        Returns:
            List[np.ndarray]: List of subgraph features
        """
        features_list = []
        
        for i, subgraph in enumerate(subgraphs):
            node_weights = node_weights_list[i] if node_weights_list else None
            features = self.extract_graph_features(subgraph, node_weights)
            features_list.append(features)
        
        return features_list
