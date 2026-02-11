# -*- coding: utf-8 -*-
"""
Graph2Vec Encoder Module
Implements topology graph to Embedding conversion, serving as Visual Encoder
Optimized version: Uses direct numerical feature encoding instead of Word2Vec hash encoding
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

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Graph2VecEncoder(nn.Module):
    """Graph2Vec Encoder, converts graph structure to embedding vector (Optimized version)"""
    
    def __init__(self, embedding_dim: int = 128, wl_iterations: int = 3,
                 workers: int = 4, epochs: int = 10, min_count: int = 5,
                 learning_rate: float = 0.025):
        """
        Initialize Graph2Vec Encoder
        
        Args:
            embedding_dim: Embedding dimension
            wl_iterations: Weisfeiler-Lehman iterations (Keep interface compatibility)
            workers: Number of parallel workers (Keep interface compatibility)
            epochs: Training epochs (Keep interface compatibility)
            min_count: Minimum word frequency (Keep interface compatibility)
            learning_rate: Learning rate (Keep interface compatibility)
        """
        super(Graph2VecEncoder, self).__init__()

        # Ensure parameter types are correct
        self.embedding_dim = int(embedding_dim)
        self.wl_iterations = int(wl_iterations)
        self.workers = int(workers)
        self.epochs = int(epochs)
        self.min_count = int(min_count)
        self.learning_rate = float(learning_rate)
        
        self.model = None  # Keep compatibility, no longer using Word2Vec
        self.vocabulary = set()
        self.graph_embeddings = {}
        
        # Topology feature cache: Calculate only once for the same graph structure
        self._topology_cache = {}
        
        # Learnable feature projection layer: Project raw features to embedding_dim
        self._raw_feature_dim = None  # Will be determined at first encoding
        self._projection = None  # Lazy initialization
        
        # Feature normalization statistics (calculated during train_model)
        self._feature_mean = None
        self._feature_std = None
        
        # Freeze parameters (Do not train Graph2Vec)
        self.freeze_parameters = True
        self._trained = False
        
    def _get_graph_hash(self, graph: nx.Graph) -> str:
        """
        Calculate graph topology hash (without weights), used for caching topology features
        """
        # Use hash of node set + edge set
        nodes_str = str(sorted(graph.nodes()))
        edges_str = str(sorted(graph.edges()))
        return hashlib.md5(f"{nodes_str}_{edges_str}".encode()).hexdigest()
    
    def _extract_topology_features(self, graph: nx.Graph) -> np.ndarray:
        """
        Extract graph topology features (independent of weights, can be cached)
        
        Args:
            graph: NetworkX graph
            
        Returns:
            np.ndarray: Topology feature vector
        """
        graph_hash = self._get_graph_hash(graph)
        
        if graph_hash in self._topology_cache:
            return self._topology_cache[graph_hash]
        
        features = []
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        
        # Basic graph statistics (5 dims)
        features.append(n_nodes)
        features.append(n_edges)
        features.append(nx.density(graph))
        features.append(1.0 if nx.is_connected(graph) else 0.0)
        features.append(len(list(nx.connected_components(graph))))
        
        # Path features (2 dims)
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
        
        # Clustering coefficient (1 dim)
        features.append(nx.average_clustering(graph))
        
        # Degree distribution features (6 dims)
        degrees = np.array([graph.degree(node) for node in graph.nodes()])
        if len(degrees) > 0:
            features.append(np.mean(degrees))
            features.append(np.std(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
            features.append(np.median(degrees))
            # Degree distribution skewness
            features.append(float(stats.skew(degrees)) if len(degrees) > 2 else 0.0)
        else:
            features.extend([0.0] * 6)
        
        # Centrality features (4 dims)
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
        
        # Triangle features (2 dims)
        try:
            triangles = list(nx.triangles(graph).values())
            features.append(sum(triangles) / (3 * max(1, n_nodes)))  # Normalized triangle density
            features.append(np.mean(triangles) if triangles else 0.0)
        except Exception:
            features.extend([0.0, 0.0])
        
        topo_features = np.array(features, dtype=np.float64)
        self._topology_cache[graph_hash] = topo_features
        return topo_features
    
    def _extract_weight_features(self, graph: nx.Graph, 
                                  node_weights: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features based on node weights (different for each sample)
        
        Args:
            graph: NetworkX graph
            node_weights: Node weight dictionary
            
        Returns:
            np.ndarray: Weight feature vector
        """
        features = []
        
        if node_weights is None or len(node_weights) == 0:
            # Return all-zero features if no weights
            return np.zeros(50, dtype=np.float64)
        
        # Get weights of nodes in the graph
        weights = np.array([node_weights.get(node, 0.0) for node in graph.nodes()], dtype=np.float64)
        
        # 1. Basic statistics (8 dims)
        features.append(np.mean(weights))
        features.append(np.std(weights))
        features.append(np.max(weights))
        features.append(np.min(weights))
        features.append(np.median(weights))
        features.append(np.sum(np.abs(weights)))
        features.append(float(stats.skew(weights)) if len(weights) > 2 else 0.0)
        features.append(float(stats.kurtosis(weights)) if len(weights) > 3 else 0.0)
        
        # 2. Quantile features (5 dims)
        for q in [10, 25, 75, 90, 95]:
            features.append(np.percentile(weights, q))
        
        # 3. Weight distribution histogram (15 dims)
        if np.max(np.abs(weights)) > 1e-10:
            hist, _ = np.histogram(weights, bins=15, density=True)
            features.extend(hist.tolist())
        else:
            features.extend([0.0] * 15)
        
        # 4. Positive/Negative/Zero statistics (3 dims)
        n_total = max(1, len(weights))
        features.append(np.sum(weights > 0.01) / n_total)   # Positive ratio
        features.append(np.sum(weights < -0.01) / n_total)  # Negative ratio
        features.append(np.sum(np.abs(weights) <= 0.01) / n_total)  # Near-zero ratio
        
        # 5. Outlier features (3 dims)
        if np.std(weights) > 1e-10:
            z_scores = np.abs((weights - np.mean(weights)) / np.std(weights))
            features.append(np.sum(z_scores > 2.0) / n_total)  # Ratio outside 2-sigma
            features.append(np.sum(z_scores > 3.0) / n_total)  # Ratio outside 3-sigma
            features.append(np.max(z_scores))                   # Max Z-score
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 6. Topology-Weight interaction features (10 dims)
        degrees = np.array([graph.degree(node) for node in graph.nodes()], dtype=np.float64)
        
        if len(degrees) > 0 and len(weights) > 0:
            # Average weight of high-degree nodes
            degree_threshold = np.percentile(degrees, 75) if len(degrees) > 3 else np.max(degrees)
            high_degree_mask = degrees >= degree_threshold
            low_degree_mask = degrees <= np.percentile(degrees, 25) if len(degrees) > 3 else degrees <= np.min(degrees)
            
            features.append(np.mean(weights[high_degree_mask]) if np.sum(high_degree_mask) > 0 else 0.0)
            features.append(np.mean(weights[low_degree_mask]) if np.sum(low_degree_mask) > 0 else 0.0)
            features.append(np.std(weights[high_degree_mask]) if np.sum(high_degree_mask) > 1 else 0.0)
            
            # Degree-Weight correlation
            if np.std(degrees) > 1e-10 and np.std(weights) > 1e-10:
                features.append(float(np.corrcoef(degrees, weights)[0, 1]))
            else:
                features.append(0.0)
            
            # Neighbor weight features
            neighbor_weight_diffs = []
            for node in graph.nodes():
                node_weight = node_weights.get(node, 0.0)
                for neighbor in graph.neighbors(node):
                    neighbor_weight = node_weights.get(neighbor, 0.0)
                    neighbor_weight_diffs.append(abs(node_weight - neighbor_weight))
            
            if neighbor_weight_diffs:
                nwd = np.array(neighbor_weight_diffs)
                features.append(np.mean(nwd))   # Mean neighbor weight difference
                features.append(np.std(nwd))    # Std neighbor weight difference
                features.append(np.max(nwd))    # Max neighbor weight difference
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Weight spatial autocorrelation (Simplified Moran's I)
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
            
            # Weight energy (L2 norm)
            features.append(np.linalg.norm(weights))
            # Weight entropy
            abs_weights = np.abs(weights) + 1e-10
            probs = abs_weights / np.sum(abs_weights)
            features.append(float(stats.entropy(probs)))
        else:
            features.extend([0.0] * 10)
        
        # Ensure fixed length (6 dim padding/truncation to reach 50)
        result = np.array(features[:50], dtype=np.float64)
        if len(result) < 50:
            result = np.pad(result, (0, 50 - len(result)))
        
        return result
    
    def _encode_graph_numerical(self, graph: nx.Graph,
                                 node_weights: Optional[Dict] = None) -> np.ndarray:
        """
        Encode graph using direct numerical features (Efficient version)
        
        Args:
            graph: NetworkX graph
            node_weights: Node weight dictionary
            
        Returns:
            np.ndarray: Feature vector
        """
        # Topology features (cached)
        topo_features = self._extract_topology_features(graph)
        
        # Weight features (different per sample)
        weight_features = self._extract_weight_features(graph, node_weights)
        
        # Concatenate
        raw_features = np.concatenate([topo_features, weight_features])
        
        return raw_features
    
    def train_model(self, graphs: List[nx.Graph],
                   node_weights_list: Optional[List[Dict]] = None) -> bool:
        """
        Train Graph2Vec model (Calculate feature normalization statistics and initialize projection layer)
        
        Args:
            graphs: List of graphs
            node_weights_list: List of node weights
            
        Returns:
            bool: Whether training was successful
        """
        try:
            logger.info(f"Start training Graph2Vec encoder (Numerical feature mode)")
            
            # Precompute topology features for all graphs (Cache)
            for graph in graphs:
                self._extract_topology_features(graph)
            logger.info(f"Topology features cached: {len(self._topology_cache)} unique graph structures")
            
            # Calculate feature normalization statistics
            all_features = []
            for i, graph in enumerate(graphs):
                node_weights = node_weights_list[i] if node_weights_list else None
                features = self._encode_graph_numerical(graph, node_weights)
                all_features.append(features)
            
            all_features = np.stack(all_features)
            self._raw_feature_dim = all_features.shape[1]
            
            # Calculate mean and std for normalization
            self._feature_mean = np.mean(all_features, axis=0)
            self._feature_std = np.std(all_features, axis=0)
            self._feature_std[self._feature_std < 1e-10] = 1.0  # Avoid division by zero
            
            # Initialize linear projection layer
            self._projection = nn.Linear(self._raw_feature_dim, self.embedding_dim)
            nn.init.xavier_uniform_(self._projection.weight)
            nn.init.zeros_(self._projection.bias)
            
            self._trained = True
            self.model = True  # Compatibility flag: Indicates model is trained
            
            logger.info(f"Graph2Vec encoder training completed: Raw feature dim={self._raw_feature_dim}, "
                        f"Output dim={self.embedding_dim}")
            
            # If freeze parameters is set, do not allow further training
            if self.freeze_parameters:
                if self._projection is not None:
                    for param in self._projection.parameters():
                        param.requires_grad = False
                logger.info("Graph2Vec parameters frozen")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train Graph2Vec model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def encode_graph(self, graph: nx.Graph, 
                    node_weights: Optional[Dict] = None,
                    graph_id: Optional[str] = None) -> torch.Tensor:
        """
        Encode a single graph
        
        Args:
            graph: NetworkX graph
            node_weights: Node weight dictionary
            graph_id: Graph ID (for caching)
            
        Returns:
            torch.Tensor: Graph embedding vector
        """
        try:
            if not self._trained:
                logger.error("Model not trained")
                return torch.zeros(self.embedding_dim)
            
            # Check cache
            if graph_id and graph_id in self.graph_embeddings:
                return self.graph_embeddings[graph_id]
            
            # Extract numerical features
            raw_features = self._encode_graph_numerical(graph, node_weights)
            
            # Normalization
            if self._feature_mean is not None:
                raw_features = (raw_features - self._feature_mean) / self._feature_std
            
            # Convert to tensor and project to embedding_dim
            features_tensor = torch.tensor(raw_features, dtype=torch.float32)
            
            if self._projection is not None:
                with torch.no_grad():
                    embedding = self._projection(features_tensor)
            else:
                # Fallback: Truncate or pad
                if len(raw_features) >= self.embedding_dim:
                    embedding = features_tensor[:self.embedding_dim]
                else:
                    embedding = torch.zeros(self.embedding_dim)
                    embedding[:len(raw_features)] = features_tensor
            
            # Cache result
            if graph_id:
                self.graph_embeddings[graph_id] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode graph: {e}")
            return torch.zeros(self.embedding_dim)
    
    def encode_graphs(self, graphs: List[nx.Graph],
                     node_weights_list: Optional[List[Dict]] = None,
                     graph_ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        Batch encode graphs
        
        Args:
            graphs: List of graphs
            node_weights_list: List of node weights
            graph_ids: List of graph IDs
            
        Returns:
            torch.Tensor: Graph embedding matrix [n_graphs, embedding_dim]
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
            logger.error(f"Failed to batch encode graphs: {e}")
            return torch.zeros(len(graphs), self.embedding_dim)
    
    def forward(self, graphs: Union[nx.Graph, List[nx.Graph]],
               node_weights: Optional[Union[Dict, List[Dict]]] = None) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            graphs: Graph or list of graphs
            node_weights: Node weights
            
        Returns:
            torch.Tensor: Graph embedding
        """
        if isinstance(graphs, list):
            return self.encode_graphs(graphs, node_weights)
        else:
            return self.encode_graph(graphs, node_weights).unsqueeze(0)
    
    def save_model(self, model_path: str) -> bool:
        """
        Save model
        
        Args:
            model_path: Model save path
            
        Returns:
            bool: Whether saving was successful
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
            
            # Save as .pth format
            save_path = model_path.replace('.model', '.pth') if model_path.endswith('.model') else model_path
            torch.save(state, save_path)
            logger.info(f"Graph2Vec model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load model
        
        Args:
            model_path: Model path
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            # Try to load new format
            load_path = model_path.replace('.model', '.pth') if model_path.endswith('.model') else model_path
            state = torch.load(load_path, weights_only=False)
            
            self.embedding_dim = state['embedding_dim']
            self._raw_feature_dim = state['raw_feature_dim']
            self._feature_mean = state['feature_mean']
            self._feature_std = state['feature_std']
            self._topology_cache = state.get('topology_cache', {})
            self._trained = state['trained']
            self.model = True  # Compatibility flag
            
            if 'projection_state_dict' in state and self._raw_feature_dim is not None:
                self._projection = nn.Linear(self._raw_feature_dim, self.embedding_dim)
                self._projection.load_state_dict(state['projection_state_dict'])
            
            logger.info(f"Graph2Vec model loaded from {load_path}")
            
            if self.freeze_parameters and self._projection is not None:
                for param in self._projection.parameters():
                    param.requires_grad = False
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
