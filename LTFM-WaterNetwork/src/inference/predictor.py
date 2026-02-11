# -*- coding: utf-8 -*-
"""
Inference Module
Implements real-time anomaly detection inference flow based on saved partition information
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import pandas as pd
import os

from ..models.ltfm_model import LTFMModel
from ..models.graph2vec_encoder import Graph2VecEncoder
from ..data.epanet_handler import EPANETHandler


class LTFMPredictor:
    """LTFM Anomaly Detection Predictor"""

    def __init__(self, config: Dict):
        """
        Initialize Predictor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')

        # Model components
        self.graph2vec_encoder = None
        self.ltfm_model = None

        # Partition information (loaded from file saved during training)
        self.partition_labels = None
        self.partition_node_names = None
        self.n_clusters = None

        # Network information
        self.epanet_handler = None
        self.network_graph = None
        self.normal_pressure = None
        
    def load_models(self, graph2vec_path: str, ltfm_checkpoint_path: str, partition_info_path: str) -> bool:
        """
        Load trained models and partition information

        Args:
            graph2vec_path: Graph2Vec model path
            ltfm_checkpoint_path: LTFM checkpoint path
            partition_info_path: Partition information file path (fcm_partition.csv)

        Returns:
            bool: Whether loading was successful
        """
        try:
            # Load partition information
            if not self.load_partition_info(partition_info_path):
                logger.error("Failed to load partition information")
                return False

            # Load Graph2Vec encoder
            self.graph2vec_encoder = Graph2VecEncoder(
                embedding_dim=int(self.config['graph2vec']['dimensions']),
                wl_iterations=3,
                workers=int(self.config['graph2vec']['workers']),
                epochs=int(self.config['graph2vec']['epochs']),
                min_count=int(self.config['graph2vec']['min_count']),
                learning_rate=float(self.config['graph2vec']['learning_rate'])
            ).to(self.device)

            if not self.graph2vec_encoder.load_model(graph2vec_path):
                logger.error("Failed to load Graph2Vec model")
                return False

            # Load LTFM model (use partition count obtained from partition information)
            self.ltfm_model = LTFMModel(
                graph2vec_dim=int(self.config['graph2vec']['dimensions']),
                embed_dim=int(self.config['ltfm']['embedding_dim']),
                num_heads=int(self.config['ltfm']['num_heads']),
                num_layers=int(self.config['ltfm']['num_layers']),
                num_regions=self.n_clusters,  # Use loaded partition count
                hidden_dim=int(self.config['ltfm']['hidden_dim']),
                dropout=float(self.config['ltfm']['dropout'])
            ).to(self.device)

            # Load checkpoint
            if os.path.exists(ltfm_checkpoint_path):
                checkpoint = torch.load(ltfm_checkpoint_path, map_location=self.device)
                self.ltfm_model.load_state_dict(checkpoint['ltfm_model_state_dict'])
                logger.info("LTFM model loaded successfully")
            else:
                logger.error(f"LTFM checkpoint file does not exist: {ltfm_checkpoint_path}")
                return False

            # Set to evaluation mode
            self.ltfm_model.eval()

            logger.info(f"Model loading completed (Partitions: {self.n_clusters})")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def load_partition_info(self, partition_info_path: str) -> bool:
        """
        Load partition information saved during training

        Args:
            partition_info_path: Partition information file path (fcm_partition.csv)

        Returns:
            bool: Whether loading was successful
        """
        try:
            if not os.path.exists(partition_info_path):
                logger.error(f"Partition information file does not exist: {partition_info_path}")
                return False

            # Read partition information
            partition_df = pd.read_csv(partition_info_path)
            self.partition_node_names = partition_df['node_name'].tolist()
            self.partition_labels = partition_df['partition_id'].tolist()
            self.n_clusters = len(set(self.partition_labels))

            logger.info(f"✓ Partition information loaded successfully: {len(self.partition_node_names)} nodes, {self.n_clusters} partitions")

            # Print partition statistics
            partition_counts = {}
            for label in self.partition_labels:
                partition_counts[label] = partition_counts.get(label, 0) + 1

            for p, count in sorted(partition_counts.items()):
                logger.info(f"  Partition {p}: {count} nodes")

            return True

        except Exception as e:
            logger.error(f"Failed to load partition information: {e}")
            return False

    def get_partitions_from_saved_info(self) -> Tuple[List[List[str]], List[nx.Graph]]:
        """
        Build partitions and subgraphs from saved partition information

        Returns:
            Tuple[List[List[str]], List[nx.Graph]]: (Partition node lists, Partition subgraph lists)
        """
        try:
            if self.partition_labels is None or self.partition_node_names is None:
                logger.error("Partition information not loaded")
                return [], []

            # Organize partition results
            partitions = [[] for _ in range(self.n_clusters)]
            for node_name, label in zip(self.partition_node_names, self.partition_labels):
                partitions[label].append(node_name)

            # Create subgraphs
            subgraphs = []
            for partition in partitions:
                if partition:  # Ensure partition is not empty
                    subgraph = self.network_graph.subgraph(partition).copy()
                    subgraphs.append(subgraph)
                else:
                    # Create empty graph
                    subgraphs.append(nx.Graph())

            logger.debug(f"Using saved partition information: {len(partitions)} partitions")
            return partitions, subgraphs

        except Exception as e:
            logger.error(f"Failed to build partitions: {e}")
            return [], []
    
    def initialize_network(self, epanet_file: str) -> bool:
        """
        Initialize network
        
        Args:
            epanet_file: EPANET file path
            
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Initialize EPANET handler
            self.epanet_handler = EPANETHandler(epanet_file)
            
            if not self.epanet_handler.load_network():
                return False
            
            # Get network graph
            self.network_graph = self.epanet_handler.get_network_graph()
            
            # Calculate normal pressure data as baseline
            if not self.epanet_handler.run_hydraulic_simulation(
                self.config['hydraulic']['simulation_hours'],
                self.config['hydraulic']['time_step']
            ):
                return False
            
            self.normal_pressure = self.epanet_handler.get_pressure_data()
            
            logger.info("Network initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Network initialization failed: {e}")
            return False

    def predict_anomaly(self, current_pressure: pd.DataFrame) -> Dict:
        """
        Predict anomaly

        Args:
            current_pressure: Current pressure data [Time, Node]

        Returns:
            Dict: Prediction result dictionary
        """
        try:
            if self.ltfm_model is None or self.graph2vec_encoder is None:
                logger.error("Model not loaded")
                return {}

            if self.normal_pressure is None:
                logger.error("Normal pressure baseline not set")
                return {}

            # Calculate pressure difference (current pressure - normal pressure)
            pressure_diff = self.normal_pressure - current_pressure

            # Calculate node weights (mean of pressure difference)
            node_weights = {}
            for node in self.epanet_handler.node_names:
                if node in pressure_diff.columns:
                    node_weights[node] = pressure_diff[node].mean()
                else:
                    node_weights[node] = 0.0

            # Use partition information saved during training (do not re-partition)
            partitions, subgraphs = self.get_partitions_from_saved_info()

            # Build region weights
            region_weights = []
            for partition in partitions:
                region_weight = {node: node_weights.get(node, 0.0) for node in partition}
                region_weights.append(region_weight)

            # Encode graph
            with torch.no_grad():
                # Global graph embedding
                global_embedding = self.graph2vec_encoder.encode_graph(
                    self.network_graph, node_weights
                ).to(self.device).unsqueeze(0)  # Add batch dimension

                # Region graph embeddings
                region_embeddings = []
                for subgraph, region_weight in zip(subgraphs, region_weights):
                    if len(subgraph.nodes()) > 0:
                        region_embedding = self.graph2vec_encoder.encode_graph(subgraph, region_weight)
                    else:
                        region_embedding = torch.zeros(self.config['graph2vec']['dimensions'])
                    region_embeddings.append(region_embedding.to(self.device).unsqueeze(0))

                # Predict
                global_pred, region_pred = self.ltfm_model.predict(
                    global_embedding, region_embeddings,
                    threshold=self.config['inference']['threshold']
                )

                # Get scores
                global_score, region_scores = self.ltfm_model(global_embedding, region_embeddings)
                global_prob = torch.sigmoid(global_score).item()

                region_probs = []
                if region_scores:
                    for score in region_scores:
                        prob = torch.sigmoid(score).item()
                        region_probs.append(prob)

            # Build result
            result = {
                'global_anomaly': bool(global_pred.item()),
                'global_probability': global_prob,
                'anomaly_region': int(region_pred.item()) if global_pred.item() else 0,
                'region_probabilities': region_probs,
                'partitions': partitions,
                'pressure_differences': node_weights,
                'timestamp': pd.Timestamp.now()
            }

            # Add confidence evaluation
            if global_prob > self.config['inference']['confidence_threshold']:
                result['confidence'] = 'high'
            elif global_prob > 0.3:
                result['confidence'] = 'medium'
            else:
                result['confidence'] = 'low'

            return result

        except Exception as e:
            logger.error(f"Anomaly prediction failed: {e}")
            return {}

    def batch_predict(self, pressure_data_list: List[pd.DataFrame]) -> List[Dict]:
        """
        Batch prediction

        Args:
            pressure_data_list: List of pressure data

        Returns:
            List[Dict]: List of prediction results
        """
        try:
            results = []
            for i, pressure_data in enumerate(pressure_data_list):
                logger.info(f"Predicting sample {i+1}/{len(pressure_data_list)}")
                result = self.predict_anomaly(pressure_data)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return []

    def save_prediction_results(self, results: List[Dict], output_path: str) -> bool:
        """
        Save prediction results

        Args:
            results: List of prediction results
            output_path: Output path

        Returns:
            bool: Whether saving was successful
        """
        try:
            if not results:
                logger.warning("Prediction results are empty")
                return False

            # Convert to DataFrame
            df_data = []
            for i, result in enumerate(results):
                row = {
                    'sample_id': i,
                    'global_anomaly': result.get('global_anomaly', False),
                    'global_probability': result.get('global_probability', 0.0),
                    'anomaly_region': result.get('anomaly_region', 0),
                    'confidence': result.get('confidence', 'unknown'),
                    'timestamp': result.get('timestamp', pd.Timestamp.now())
                }

                # Add region probabilities
                region_probs = result.get('region_probabilities', [])
                for j, prob in enumerate(region_probs):
                    row[f'region_{j}_probability'] = prob

                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

            logger.info(f"Prediction results saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save prediction results: {e}")
            return False

    def get_network_status(self) -> Dict:
        """
        Get network status information

        Returns:
            Dict: Network status dictionary
        """
        try:
            if self.epanet_handler is None:
                return {}

            network_info = self.epanet_handler.get_network_info()

            status = {
                'network_loaded': self.network_graph is not None,
                'models_loaded': self.ltfm_model is not None and self.graph2vec_encoder is not None,
                'normal_pressure_available': self.normal_pressure is not None,
                'n_nodes': network_info.get('n_nodes', 0),
                'n_pipes': network_info.get('n_pipes', 0),
                'n_partitions': self.n_clusters or 0,
                'device': str(self.device)
            }

            return status

        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {}
