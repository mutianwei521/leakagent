# -*- coding: utf-8 -*-
"""
Training Module
Implements model training workflow, including loss function calculation and backpropagation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
import os
import networkx as nx
import scipy
import scipy.sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..models.ltfm_model import LTFMModel
from ..models.graph2vec_encoder import Graph2VecEncoder
from ..models.node_localizer import NodeLocalizer
from ..data.epanet_handler import EPANETHandler
from ..data.sensitivity_analyzer import SensitivityAnalyzer
from ..utils.fcm_partitioner import FCMPartitioner


def custom_collate_fn(batch):
    """
    Custom collate function for batch processing containing NetworkX graphs

    Args:
        batch: List of batch data

    Returns:
        List: Original batch data (no collate operation)
    """
    return batch


class PrecomputedDataset(Dataset):
    """Precomputed embedding dataset: Stores tensors directly to avoid repeated encoding"""

    def __init__(self, global_embeddings: torch.Tensor, 
                 region_embeddings: List[torch.Tensor],
                 global_labels: torch.Tensor, 
                 region_labels: torch.Tensor):
        """
        Args:
            global_embeddings: [N, embed_dim]
            region_embeddings: List of [N, embed_dim], length = num_regions
            global_labels: [N]
            region_labels: [N]
        """
        self.global_embeddings = global_embeddings
        self.region_embeddings = region_embeddings
        self.global_labels = global_labels
        self.region_labels = region_labels
    
    def __len__(self):
        return self.global_embeddings.shape[0]
    
    def __getitem__(self, idx):
        return {
            'global_emb': self.global_embeddings[idx],
            'region_embs': [r[idx] for r in self.region_embeddings],
            'global_label': self.global_labels[idx],
            'region_label': self.region_labels[idx]
        }


def precomputed_collate_fn(batch):
    """Collate function for PrecomputedDataset"""
    global_embs = torch.stack([b['global_emb'] for b in batch])
    num_regions = len(batch[0]['region_embs'])
    region_embs = [
        torch.stack([b['region_embs'][i] for b in batch])
        for i in range(num_regions)
    ]
    global_labels = torch.stack([b['global_label'] for b in batch])
    region_labels = torch.stack([b['region_label'] for b in batch])
    return global_embs, region_embs, global_labels, region_labels


class WaterNetworkDataset(Dataset):
    """Water Network Dataset (Compatible with legacy interface)"""

    def __init__(self, scenarios: List[Dict]):
        """
        Initialize dataset

        Args:
            scenarios: List of scenarios, each scenario contains:
                - global_graph: Global graph
                - region_graphs: List of region graphs
                - global_weights: Global node weights
                - region_weights: List of region node weights
                - global_label: Global label (0 normal, 1 anomaly)
                - region_label: Region label (0 normal, >0 anomaly region index)
        """
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]


class LTFMTrainer:
    """LTFM Model Trainer"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        
        # Model components
        self.graph2vec_encoder = None
        self.ltfm_model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training status
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_composite_score = float('-inf')  # Composite score
        self.patience_counter = 0
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def initialize_models(self):
        """Initialize models"""
        try:
            # Graph2Vec Encoder
            self.graph2vec_encoder = Graph2VecEncoder(
                embedding_dim=int(self.config['graph2vec']['dimensions']),
                wl_iterations=3,
                workers=int(self.config['graph2vec']['workers']),
                epochs=int(self.config['graph2vec']['epochs']),
                min_count=int(self.config['graph2vec']['min_count']),
                learning_rate=float(self.config['graph2vec']['learning_rate'])
            ).to(self.device)
            
            # LTFM Model
            self.ltfm_model = LTFMModel(
                graph2vec_dim=int(self.config['graph2vec']['dimensions']),
                embed_dim=int(self.config['ltfm']['embedding_dim']),
                num_heads=int(self.config['ltfm']['num_heads']),
                num_layers=int(self.config['ltfm']['num_layers']),
                num_regions=int(self.config['fcm']['n_clusters']),
                hidden_dim=int(self.config['ltfm']['hidden_dim']),
                dropout=float(self.config['ltfm']['dropout'])
            ).to(self.device)
            
            # Optimizer - Use AdamW and improved parameters
            self.optimizer = optim.AdamW(
                self.ltfm_model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=float(self.config['training']['weight_decay']),
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # Learning rate scheduler - Use ReduceLROnPlateau (More suitable for our task)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )

            # Gradient clipping parameter
            self.max_grad_norm = 1.0
            
            logger.info("Model initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def generate_training_scenarios(self, epanet_handler: EPANETHandler,
                                  sensitivity_analyzer: SensitivityAnalyzer,
                                  fcm_partitioner: FCMPartitioner,
                                  n_scenarios: int = 1000) -> List[Dict]:
        """
        Generate training scenarios
        
        Args:
            epanet_handler: EPANET handler
            sensitivity_analyzer: Sensitivity analyzer
            fcm_partitioner: FCM partitioner
            n_scenarios: Number of scenarios
            
        Returns:
            List[Dict]: List of training scenarios
        """
        try:
            scenarios = []
            node_names = epanet_handler.node_names
            n_nodes = len(node_names)
            
            # Get network graph and partition info
            network_graph = epanet_handler.get_network_graph()
            partition_info = fcm_partitioner.get_partition_info()
            subgraphs = fcm_partitioner.get_partition_subgraphs(network_graph, node_names)
            
            logger.info(f"Start generating {n_scenarios} training scenarios")
            
            # Generate normal scenarios - Increase diversity
            n_normal = n_scenarios // 2
            for i in tqdm(range(n_normal), desc="Generating normal scenarios"):
                # Normal scenario: Add small natural variations
                normal_weights = {}
                for node in node_names:
                    # Add small random variations, simulate natural fluctuations during normal operation
                    # Variation range: -0.05 to +0.05 (Small compared to anomaly scenarios)
                    natural_variation = np.random.uniform(-0.05, 0.05)
                    normal_weights[node] = natural_variation

                # Generate slightly different weights for each region
                region_weights = []
                for subgraph in subgraphs:
                    region_weight = {}
                    for node in node_names:
                        if node in subgraph.nodes():
                            # Small variations for correlated nodes within the region
                            base_variation = normal_weights[node]
                            region_variation = base_variation + np.random.uniform(-0.02, 0.02)
                            region_weight[node] = region_variation
                        else:
                            region_weight[node] = normal_weights[node]
                    region_weights.append(region_weight)

                scenario = {
                    'global_graph': network_graph,
                    'region_graphs': subgraphs,
                    'global_weights': normal_weights,
                    'region_weights': region_weights,
                    'global_label': 0,  # Normal
                    'region_label': 0,  # Normal
                    'anomaly_node_idx': -1,  # No anomaly node in normal scenario
                    'anomaly_node_name': None,
                    'node_names': node_names
                }
                scenarios.append(scenario)
            
            # Generate anomaly scenarios - Enhanced version
            n_anomaly = n_scenarios - n_normal
            anomaly_count = 0
            max_attempts = n_anomaly * 3  # Max attempts 3 times

            for attempt in tqdm(range(max_attempts), desc="Generating anomaly scenarios"):
                if anomaly_count >= n_anomaly:
                    break

                # Randomly select anomaly node
                anomaly_node_idx = np.random.randint(0, n_nodes)
                anomaly_node = node_names[anomaly_node_idx]

                # Get sensitivity features of the anomaly node
                sensitivity_features = sensitivity_analyzer.get_sensitivity_features(anomaly_node_idx)

                if not sensitivity_features:
                    logger.debug(f"Sensitivity features for node {anomaly_node} are empty, skipping")
                    continue

                # Build node weights (pressure difference)
                pressure_sensitivity = sensitivity_features['avg_pressure_sensitivity']

                # Check if pressure sensitivity is valid
                if len(pressure_sensitivity) != n_nodes:
                    logger.debug(f"Pressure sensitivity length mismatch for node {anomaly_node}, skipping")
                    continue

                # Data augmentation: Add noise and transformation
                augmented_pressure_sensitivity = self._augment_pressure_sensitivity(
                    pressure_sensitivity, anomaly_node_idx
                )

                global_weights = {node_names[j]: augmented_pressure_sensitivity[j] for j in range(n_nodes)}

                # Determine anomaly region - Fix label encoding
                partition_labels = partition_info['partition_labels']
                # Anomaly region label: 0=normal, 1=region 0 anomaly, 2=region 1 anomaly, 3=region 2 anomaly...
                anomaly_region = partition_labels[anomaly_node_idx] + 1  # +1 because 0 represents normal

                # Build region weights - Enhance features of the anomaly region
                region_weights = []
                for region_id, subgraph in enumerate(subgraphs):
                    region_weight = {}
                    for node in subgraph.nodes():
                        node_idx = node_names.index(node)
                        weight = augmented_pressure_sensitivity[node_idx]

                        # If it is an anomaly region, significantly enhance weights
                        if partition_labels[node_idx] == (anomaly_region - 1):
                            weight *= np.random.uniform(3.0, 5.0)  # Significantly enhance anomaly region weights
                        else:
                            # Non-anomaly region, appropriately reduce weights
                            weight *= np.random.uniform(0.3, 0.7)

                        region_weight[node] = weight
                    region_weights.append(region_weight)

                scenario = {
                    'global_graph': network_graph,
                    'region_graphs': subgraphs,
                    'global_weights': global_weights,
                    'region_weights': region_weights,
                    'global_label': 1,  # Anomaly
                    'region_label': anomaly_region,  # Anomaly region index
                    'anomaly_node_idx': anomaly_node_idx,
                    'anomaly_node_name': anomaly_node,
                    'node_names': node_names,
                    'partition_labels': partition_labels
                }
                scenarios.append(scenario)
                anomaly_count += 1

                # Generate multiple variants (Increase data diversity)
                variants_to_generate = min(2, n_anomaly - anomaly_count)  # Generate at most 2 variants per base scenario
                for _ in range(variants_to_generate):
                    if anomaly_count >= n_anomaly:
                        break

                    variant_scenario = self._create_scenario_variant(
                        scenario, node_names, partition_labels, anomaly_node_idx
                    )
                    if variant_scenario:
                        scenarios.append(variant_scenario)
                        anomaly_count += 1

            logger.info(f"Successfully generated {anomaly_count} anomaly scenarios (Target: {n_anomaly})")
            
            logger.info(f"Training scenario generation completed: {len(scenarios)} scenarios")
            return scenarios

        except Exception as e:
            logger.error(f"Failed to generate training scenarios: {e}")
            return []

    def _augment_pressure_sensitivity(self, pressure_sensitivity: np.ndarray, anomaly_node_idx: int) -> np.ndarray:
        """
        Augment pressure sensitivity data

        Args:
            pressure_sensitivity: Original pressure sensitivity
            anomaly_node_idx: Anomaly node index

        Returns:
            np.ndarray: Augmented pressure sensitivity
        """
        augmented = pressure_sensitivity.copy()

        # 1. Add Gaussian noise
        noise_level = 0.05  # 5% noise
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented = augmented + noise

        # 2. Enhance impact of the anomaly node
        anomaly_boost = np.random.uniform(1.5, 3.0)
        augmented[anomaly_node_idx] *= anomaly_boost

        # 3. Randomly scale some nodes
        scale_mask = np.random.random(augmented.shape) < 0.3
        scale_factors = np.random.uniform(0.8, 1.2, augmented.shape)
        augmented[scale_mask] *= scale_factors[scale_mask]

        # 4. Ensure numerical stability
        augmented = np.clip(augmented, -10, 10)

        return augmented

    def _create_scenario_variant(self, base_scenario: Dict, node_names: List[str],
                               partition_labels: np.ndarray, anomaly_node_idx: int) -> Optional[Dict]:
        """
        创建场景变体

        Args:
            base_scenario: 基础场景
            node_names: 节点名称列表
            partition_labels: 分区标签
            anomaly_node_idx: 异常节点索引

        Returns:
            Optional[Dict]: 变体场景或None
        """
        try:
            variant = base_scenario.copy()
            # Retain node-level information like anomaly_node_idx
            variant['anomaly_node_idx'] = base_scenario.get('anomaly_node_idx', -1)
            variant['anomaly_node_name'] = base_scenario.get('anomaly_node_name', None)
            variant['node_names'] = base_scenario.get('node_names', node_names)
            variant['partition_labels'] = base_scenario.get('partition_labels', partition_labels)
            variant_type = np.random.choice(['perturbation', 'intensity', 'multi_node', 'temporal'])

            # Variant 1: Weight perturbation
            if variant_type == 'perturbation':
                perturbed_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    # Add Gaussian noise perturbation
                    noise_level = 0.15  # Increase noise level
                    perturbation = np.random.normal(0, abs(weight) * noise_level)
                    perturbed_global_weights[node] = weight + perturbation
                variant['global_weights'] = perturbed_global_weights

                # Synchronously update region weights
                perturbed_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    perturbed_region = {}
                    for node, weight in region_weight.items():
                        perturbation = np.random.normal(0, abs(weight) * noise_level)
                        perturbed_region[node] = weight + perturbation
                    perturbed_region_weights.append(perturbed_region)
                variant['region_weights'] = perturbed_region_weights

            # Variant 2: Anomaly intensity adjustment
            elif variant_type == 'intensity':
                intensity_factor = np.random.uniform(0.5, 2.0)  # Expand intensity range
                adjusted_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    adjusted_global_weights[node] = weight * intensity_factor
                variant['global_weights'] = adjusted_global_weights

                adjusted_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    adjusted_region = {}
                    for node, weight in region_weight.items():
                        adjusted_region[node] = weight * intensity_factor
                    adjusted_region_weights.append(adjusted_region)
                variant['region_weights'] = adjusted_region_weights

            # Variant 3: Multi-node anomaly (Multiple nodes in the same region)
            elif variant_type == 'multi_node':
                anomaly_region_idx = partition_labels[anomaly_node_idx]
                # Find other nodes in the same region
                same_region_nodes = [i for i, label in enumerate(partition_labels)
                                   if label == anomaly_region_idx and i != anomaly_node_idx]

                if same_region_nodes:
                    # Randomly select 1-2 additional nodes
                    additional_nodes = np.random.choice(
                        same_region_nodes,
                        size=min(2, len(same_region_nodes)),
                        replace=False
                    )

                    # Enhance weights of these nodes
                    enhanced_global_weights = base_scenario['global_weights'].copy()
                    for node_idx in additional_nodes:
                        node_name = node_names[node_idx]
                        if node_name in enhanced_global_weights:
                            enhanced_global_weights[node_name] *= np.random.uniform(2.0, 4.0)

                    variant['global_weights'] = enhanced_global_weights

                    # Synchronously update region weights
                    enhanced_region_weights = []
                    for region_weight in base_scenario['region_weights']:
                        enhanced_region = region_weight.copy()
                        for node_idx in additional_nodes:
                            node_name = node_names[node_idx]
                            if node_name in enhanced_region:
                                enhanced_region[node_name] *= np.random.uniform(2.0, 4.0)
                        enhanced_region_weights.append(enhanced_region)
                    variant['region_weights'] = enhanced_region_weights

            # Variant 4: Temporal variation simulation (via weight gradient change)
            elif variant_type == 'temporal':
                temporal_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    # Simulate temporal variation: Add periodic change
                    phase = np.random.uniform(0, 2 * np.pi)
                    amplitude = abs(weight) * 0.3
                    temporal_factor = 1 + amplitude * np.sin(phase)
                    temporal_global_weights[node] = weight * temporal_factor
                variant['global_weights'] = temporal_global_weights

                # Synchronously update region weights
                temporal_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    temporal_region = {}
                    for node, weight in region_weight.items():
                        phase = np.random.uniform(0, 2 * np.pi)
                        amplitude = abs(weight) * 0.3
                        temporal_factor = 1 + amplitude * np.sin(phase)
                        temporal_region[node] = weight * temporal_factor
                    temporal_region_weights.append(temporal_region)
                variant['region_weights'] = temporal_region_weights

            return variant

        except Exception as e:
            logger.debug(f"Failed to create scenario variant: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate training scenarios: {e}")
            return []
    
    def precompute_embeddings(self, scenarios: List[Dict]) -> Tuple[
        torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """
        Precompute embeddings for all scenarios, complete encoding in one go
        
        Args:
            scenarios: List of scenarios
            
        Returns:
            Tuple: (Global embeddings, List of region embeddings, Global labels, Region labels)
        """
        logger.info(f"Precomputing embeddings for {len(scenarios)} scenarios...")
        
        global_embeddings = []
        num_regions = len(scenarios[0]['region_graphs'])
        region_embeddings_list = [[] for _ in range(num_regions)]
        global_labels = []
        region_labels = []
        
        for i, scenario in enumerate(tqdm(scenarios, desc="Precomputing embeddings")):
            # Encode global graph
            global_embedding = self.graph2vec_encoder.encode_graph(
                scenario['global_graph'], scenario['global_weights']
            )
            global_embeddings.append(global_embedding)
            
            # Encode region graphs
            for j, (region_graph, region_weight) in enumerate(
                zip(scenario['region_graphs'], scenario['region_weights'])
            ):
                region_embedding = self.graph2vec_encoder.encode_graph(
                    region_graph, region_weight
                )
                region_embeddings_list[j].append(region_embedding)
            
            # Labels
            global_labels.append(scenario['global_label'])
            region_labels.append(scenario['region_label'])
        
        # Convert to tensor
        global_emb_tensor = torch.stack(global_embeddings)
        region_emb_tensors = [
            torch.stack(region_embs) for region_embs in region_embeddings_list
        ]
        global_labels_tensor = torch.tensor(global_labels, dtype=torch.float32)
        region_labels_tensor = torch.tensor(region_labels, dtype=torch.long)
        
        logger.info(f"Precomputation completed: Global embeddings {global_emb_tensor.shape}, "
                    f"{num_regions} Region embeddings {region_emb_tensors[0].shape}")
        
        # Verify embedding diversity
        emb_std = global_emb_tensor.std(dim=0).mean().item()
        logger.info(f"Global embedding std (averaged across samples): {emb_std:.6f}")
        if emb_std < 1e-6:
            logger.warning("⚠️ Embeddings have almost no diversity! The encoder might not be capturing features correctly")
        
        return global_emb_tensor, region_emb_tensors, global_labels_tensor, region_labels_tensor

    def prepare_batch_data(self, batch: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor], 
                                                           torch.Tensor, torch.Tensor]:
        """
        Prepare batch data (Compatible with legacy interface, but precompute_embeddings is recommended)
        
        Args:
            batch: Batch data
            
        Returns:
            Tuple: (Global graph embeddings, List of region graph embeddings, Global labels, Region labels)
        """
        try:
            global_embeddings = []
            region_embeddings_list = [[] for _ in range(len(batch[0]['region_graphs']))]
            global_labels = []
            region_labels = []
            
            for scenario in batch:
                # Encode global graph
                global_embedding = self.graph2vec_encoder.encode_graph(
                    scenario['global_graph'], scenario['global_weights']
                )
                global_embeddings.append(global_embedding)
                
                # Encode region graphs
                for i, (region_graph, region_weight) in enumerate(
                    zip(scenario['region_graphs'], scenario['region_weights'])
                ):
                    region_embedding = self.graph2vec_encoder.encode_graph(region_graph, region_weight)
                    region_embeddings_list[i].append(region_embedding)
                
                # Labels
                global_labels.append(scenario['global_label'])
                region_labels.append(scenario['region_label'])
            
            # Convert to tensor
            global_embeddings_tensor = torch.stack(global_embeddings).to(self.device)
            region_embeddings_tensors = [
                torch.stack(region_embs).to(self.device) 
                for region_embs in region_embeddings_list
            ]
            global_labels_tensor = torch.tensor(global_labels, dtype=torch.float32).to(self.device)
            region_labels_tensor = torch.tensor(region_labels, dtype=torch.long).to(self.device)
            
            return global_embeddings_tensor, region_embeddings_tensors, global_labels_tensor, region_labels_tensor
            
        except Exception as e:
            logger.error(f"Failed to prepare batch data: {e}")
            return None, None, None, None

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch (Supports precomputed and legacy modes)

        Args:
            dataloader: Data loader

        Returns:
            Tuple[float, float]: (Average loss, Accuracy)
        """
        self.ltfm_model.train()
        total_loss = 0.0
        total_samples = 0
        correct_global = 0
        correct_region = 0

        for batch in tqdm(dataloader, desc=f"Training Epoch {self.current_epoch}"):
            # Determine if it's precomputed mode or legacy mode
            if isinstance(batch, (tuple, list)) and len(batch) == 4 and isinstance(batch[0], torch.Tensor):
                # Precomputed mode: batch is already (global_embs, region_embs, global_labels, region_labels)
                global_embs, region_embs, global_labels, region_labels = batch
                global_embs = global_embs.to(self.device)
                region_embs = [r.to(self.device) for r in region_embs]
                global_labels = global_labels.to(self.device)
                region_labels = region_labels.to(self.device)
            else:
                # Legacy mode: Needs encoding
                global_embs, region_embs, global_labels, region_labels = self.prepare_batch_data(batch)

            if global_embs is None:
                continue

            # Forward propagation
            self.optimizer.zero_grad()
            global_scores, region_scores = self.ltfm_model(global_embs, region_embs)

            # Calculate loss
            loss = self.ltfm_model.compute_loss(global_scores, region_scores, global_labels, region_labels)

            # Backward propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ltfm_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            batch_size = global_embs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Calculate accuracy
            with torch.no_grad():
                global_pred, region_pred = self.ltfm_model.predict(global_embs, region_embs)
                correct_global += (global_pred.squeeze() == global_labels).sum().item()
                correct_region += (region_pred == region_labels).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        global_acc = correct_global / total_samples if total_samples > 0 else 0.0
        region_acc = correct_region / total_samples if total_samples > 0 else 0.0
        avg_acc = (global_acc + region_acc) / 2

        return avg_loss, avg_acc

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Validate one epoch

        Args:
            dataloader: Validation data loader

        Returns:
            Tuple[float, float, Dict]: (Average loss, Accuracy, Detailed metrics)
        """
        self.ltfm_model.eval()
        total_loss = 0.0
        total_samples = 0

        all_global_preds = []
        all_global_labels = []
        all_region_preds = []
        all_region_labels = []
        all_global_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Determine if it's precomputed mode or legacy mode
                if isinstance(batch, (tuple, list)) and len(batch) == 4 and isinstance(batch[0], torch.Tensor):
                    global_embs, region_embs, global_labels, region_labels = batch
                    global_embs = global_embs.to(self.device)
                    region_embs = [r.to(self.device) for r in region_embs]
                    global_labels = global_labels.to(self.device)
                    region_labels = region_labels.to(self.device)
                else:
                    global_embs, region_embs, global_labels, region_labels = self.prepare_batch_data(batch)

                if global_embs is None:
                    continue

                # Forward propagation
                global_scores, region_scores = self.ltfm_model(global_embs, region_embs)

                # Calculate loss
                loss = self.ltfm_model.compute_loss(global_scores, region_scores, global_labels, region_labels)

                # Predict
                global_pred, region_pred = self.ltfm_model.predict(global_embs, region_embs)

                # Statistics
                batch_size = global_embs.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Collect prediction results
                global_pred_list = global_pred.cpu().numpy().flatten().tolist()
                global_labels_list = global_labels.cpu().numpy().flatten().tolist()
                region_pred_list = region_pred.cpu().numpy().flatten().tolist()
                region_labels_list = region_labels.cpu().numpy().flatten().tolist()
                global_scores_list = torch.sigmoid(global_scores).cpu().numpy().flatten().tolist()

                all_global_preds.extend(global_pred_list)
                all_global_labels.extend(global_labels_list)
                all_region_preds.extend(region_pred_list)
                all_region_labels.extend(region_labels_list)
                all_global_scores.extend(global_scores_list)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Add debug info
        if len(all_global_labels) > 0:
            logger.info(f"Validation set prediction statistics:")
            logger.info(f"  Global label distribution: Normal={all_global_labels.count(0)}, Anomaly={all_global_labels.count(1)}")
            global_preds_int = [int(p) for p in all_global_preds]
            logger.info(f"  Global prediction distribution: Normal={global_preds_int.count(0)}, Anomaly={global_preds_int.count(1)}")
            if all_global_scores:
                logger.info(f"  Global score range: [{min(all_global_scores):.4f}, {max(all_global_scores):.4f}]")
            else:
                logger.info("  Global score range: []")
            logger.info(f"  Region label unique values: {set(all_region_labels)}")
            logger.info(f"  Region prediction unique values: {set(all_region_preds)}")

            if len(set(all_global_preds)) == 1:
                logger.warning("⚠️  All global predictions are identical! Model might not have learned useful features")

            if len(set(all_region_preds)) == 1:
                logger.warning("⚠️  All region predictions are identical! Model might not have learned useful features")

        # Calculate detailed metrics
        metrics = self.calculate_metrics(
            all_global_preds, all_global_labels,
            all_region_preds, all_region_labels,
            all_global_scores
        )

        avg_acc = (metrics['global_acc'] + metrics['region_acc']) / 2

        return avg_loss, avg_acc, metrics

    def calculate_metrics(self, global_preds: List, global_labels: List,
                         region_preds: List, region_labels: List,
                         global_scores: List) -> Dict:
        """
        Calculate evaluation metrics

        Args:
            global_preds: Global predictions
            global_labels: Global labels
            region_preds: Region predictions
            region_labels: Region labels
            global_scores: Global scores

        Returns:
            Dict: Dictionary of evaluation metrics
        """
        try:
            metrics = {}

            # Global anomaly detection metrics
            global_preds = np.array(global_preds)
            global_labels = np.array(global_labels)
            global_scores = np.array(global_scores)

            metrics['global_acc'] = accuracy_score(global_labels, global_preds)
            metrics['global_precision'] = precision_score(global_labels, global_preds, zero_division=0)
            metrics['global_recall'] = recall_score(global_labels, global_preds, zero_division=0)
            metrics['global_f1'] = f1_score(global_labels, global_preds, zero_division=0)

            unique_labels = np.unique(global_labels)
            if len(unique_labels) > 1:
                metrics['global_auc'] = roc_auc_score(global_labels, global_scores)
            else:
                metrics['global_auc'] = 0.0
                logger.debug(f"AUC set to 0.0 because there is only one label type: {unique_labels}")

            # Regional localization metrics
            region_preds = np.array(region_preds)
            region_labels = np.array(region_labels)

            metrics['region_acc'] = accuracy_score(region_labels, region_preds)

            # Regional localization accuracy considering only anomaly samples
            anomaly_mask = global_labels == 1
            if np.sum(anomaly_mask) > 0:
                anomaly_region_acc = accuracy_score(
                    region_labels[anomaly_mask],
                    region_preds[anomaly_mask]
                )
                metrics['anomaly_region_acc'] = anomaly_region_acc
            else:
                metrics['anomaly_region_acc'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}

    def train(self, train_scenarios: List[Dict], val_scenarios: List[Dict], skip_stage1: bool = False) -> bool:
        """
        Train model (Use precomputed embeddings for acceleration)

        Args:
            train_scenarios: Training scenarios
            val_scenarios: Validation scenarios
            skip_stage1: Whether to skip Stage 1 training

        Returns:
            bool: Whether training was successful
        """
        try:
            # Precompute all embeddings (Done once to avoid repeated encoding every epoch)
            logger.info("=" * 60)
            logger.info("Stage 1: Precompute embeddings (Execute only once)")
            logger.info("=" * 60)
            
            train_global, train_regions, train_glabels, train_rlabels = \
                self.precompute_embeddings(train_scenarios)
            val_global, val_regions, val_glabels, val_rlabels = \
                self.precompute_embeddings(val_scenarios)
            
            # [NEW] Skip Stage 1 Logic
            if skip_stage1:
                logger.info("Detected request to skip Stage 1...")
                checkpoint_path = os.path.join(self.config['data']['output_dir'], 'checkpoints', 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    logger.info(f"Loading existing model: {checkpoint_path}")
                    try:
                        self.load_checkpoint(checkpoint_path)
                        logger.info("✅ Model loaded successfully, skipping Stage 1 training")
                        
                        # Important: Must save precomputed embeddings for Stage 2 use
                        self._precomputed_train = (train_global, train_regions, train_glabels, train_rlabels)
                        self._precomputed_val = (val_global, val_regions, val_glabels, val_rlabels)
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load model: {e}. Continuing training...")
                else:
                    logger.warning(f"Checkpoint {checkpoint_path} not found. Continuing training...")

            # Create precomputed dataset
            train_dataset = PrecomputedDataset(
                train_global, train_regions, train_glabels, train_rlabels
            )
            val_dataset = PrecomputedDataset(
                val_global, val_regions, val_glabels, val_rlabels
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=0,  # Windows compatibility
                collate_fn=precomputed_collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=0,
                collate_fn=precomputed_collate_fn
            )

            # Check dataset label distribution
            train_global_labels = [s['global_label'] for s in train_scenarios]
            val_global_labels = [s['global_label'] for s in val_scenarios]
            train_region_labels = [s['region_label'] for s in train_scenarios]
            val_region_labels = [s['region_label'] for s in val_scenarios]

            logger.info(f"Start training: {len(train_scenarios)} training samples, {len(val_scenarios)} validation samples")
            logger.info(f"Training set label distribution - Normal: {train_global_labels.count(0)}, Anomaly: {train_global_labels.count(1)}")
            logger.info(f"Validation set label distribution - Normal: {val_global_labels.count(0)}, Anomaly: {val_global_labels.count(1)}")
            logger.info(f"Training set region label distribution: {set(train_region_labels)}")
            logger.info(f"Validation set region label distribution: {set(val_region_labels)}")

            # Training loop
            for epoch in range(self.config['training']['epochs']):
                self.current_epoch = epoch + 1

                # Train
                train_loss, train_acc = self.train_epoch(train_loader)

                # Validate
                val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)

                # Update learning rate
                self.scheduler.step(val_loss)

                # Record history
                self.train_history['train_loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['train_acc'].append(train_acc)
                self.train_history['val_acc'].append(val_acc)

                # Log
                logger.info(
                    f"Epoch {self.current_epoch}/{self.config['training']['epochs']} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                logger.info(
                    f"Val Metrics - Global AUC: {val_metrics.get('global_auc', 0):.4f}, "
                    f"Global F1: {val_metrics.get('global_f1', 0):.4f}, "
                    f"Anomaly Region Acc: {val_metrics.get('anomaly_region_acc', 0):.4f}"
                )

                # Improved early stopping check - Consider multiple metrics comprehensively
                # Calculate composite score: Loss + Region Accuracy
                region_acc = val_metrics.get('anomaly_region_acc', 0.0)
                global_auc = val_metrics.get('global_auc', 0.0)

                # Composite score: Lower loss is better, higher accuracy is better
                composite_score = -val_loss + 2.0 * region_acc + 1.0 * global_auc

                if composite_score > self.best_composite_score:
                    self.best_composite_score = composite_score
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                    logger.info(f"New best model - Composite Score: {composite_score:.4f}, Region Accuracy: {region_acc:.4f}")
                else:
                    self.patience_counter += 1

                # Dynamically adjust early stopping patience
                patience = self.config['training']['early_stopping_patience']
                if region_acc > 0.1:  # If region accuracy improves, increase patience
                    patience = patience + 5

                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered, stopped at epoch {self.current_epoch}")
                    logger.info(f"Best composite score: {self.best_composite_score:.4f}")
                    break

                # Setup periodic checkpoint saving
                if self.current_epoch % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')

            logger.info("LTFM training completed")
            
            # Save precomputed embeddings for NodeLocalizer use
            self._precomputed_train = (train_global, train_regions, train_glabels, train_rlabels)
            self._precomputed_val = (val_global, val_regions, val_glabels, val_rlabels)
            
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def train_node_localizer(self, train_scenarios: List[Dict], 
                              val_scenarios: List[Dict],
                              n_epochs: int = 100,
                              lr: float = 0.001) -> bool:
        """
        Stage 2: Train NodeLocalizer model
        Use frozen LTFM intermediate features + sensitivity vectors -> Predict leakage nodes
        Includes data augmentation (20x Gaussian noise) to solve insufficient sample problem
        
        Args:
            train_scenarios: Training scenarios (containing anomaly_node_idx)
            val_scenarios: Validation scenarios
            n_epochs: Number of epochs
            lr: Learning rate
            
        Returns:
            bool: Whether training was successful
        """
        try:
            logger.info("=" * 60)
            logger.info("Stage 2: Train NodeLocalizer (Node-level leakage localization)")
            logger.info("=" * 60)
            
            # 1. Filter anomaly scenarios (Only anomaly scenarios have leakage node labels)
            train_anomaly = [s for s in train_scenarios if s['global_label'] == 1]
            val_anomaly = [s for s in val_scenarios if s['global_label'] == 1]
            
            if not train_anomaly or not val_anomaly:
                logger.error("No anomaly scenarios for training NodeLocalizer")
                return False
            
            logger.info(f"Anomaly scenarios: Train={len(train_anomaly)}, Val={len(val_anomaly)}")
            
            # 2. Get node information
            node_names = train_anomaly[0].get('node_names', [])
            n_nodes = len(node_names)
            network_graph = train_anomaly[0]['global_graph']
            
            if n_nodes == 0:
                logger.error("Unable to get node name information")
                return False
            
            logger.info(f"Number of network nodes: {n_nodes}")
            
            # 3. Use precomputed embeddings to extract intermediate features (Fast mode)
            logger.info("Using precomputed embeddings to extract regional features (Skipping graph encoding)...")
            self.ltfm_model.eval()
            
            # Try to use cached precomputed embeddings
            if hasattr(self, '_precomputed_train') and hasattr(self, '_precomputed_val'):
                train_global, train_regions, train_glabels, train_rlabels = self._precomputed_train
                val_global, val_regions, val_glabels, val_rlabels = self._precomputed_val
                logger.info("✅ Reusing precomputed embeddings from LTFM training stage")
            else:
                logger.info("⚠️ No cached embeddings, recomputing...")
                train_global, train_regions, train_glabels, train_rlabels = \
                    self.precompute_embeddings(train_scenarios)
                val_global, val_regions, val_glabels, val_rlabels = \
                    self.precompute_embeddings(val_scenarios)
            
            # Filter indices of anomaly scenarios
            train_anomaly_indices = [i for i, s in enumerate(train_scenarios) if s['global_label'] == 1]
            val_anomaly_indices = [i for i, s in enumerate(val_scenarios) if s['global_label'] == 1]
            
            train_features, train_sens, train_labels, train_masks = self._extract_node_features_fast(
                train_scenarios, train_anomaly_indices,
                train_global, train_regions, train_rlabels, node_names
            )
            val_features, val_sens, val_labels, val_masks = self._extract_node_features_fast(
                val_scenarios, val_anomaly_indices,
                val_global, val_regions, val_rlabels, node_names
            )
            
            if train_features is None:
                logger.error("Feature extraction failed")
                return False
            
            # Statistics on region constraints
            avg_region_size = train_masks.float().sum(dim=1).mean().item()
            logger.info(f"Original training features: region_feat={train_features.shape}, "
                        f"sens={train_sens.shape}, labels={train_labels.shape}")
            logger.info(f"Region constraint: On average, each sample is classified within {avg_region_size:.0f}/{n_nodes} nodes")
            
            # 4. Data augmentation: Add Gaussian noise to create 40x more training samples
            augment_factor = 40
            logger.info(f"Data augmentation: {augment_factor}x Gaussian noise amplification...")
            
            aug_features_list = [train_features]
            aug_sens_list = [train_sens]
            aug_labels_list = [train_labels]
            aug_masks_list = [train_masks]
            
            feat_std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
            sens_std = train_sens.std(dim=0, keepdim=True).clamp(min=1e-6)
            
            for aug_i in range(augment_factor):
                # 递增噪声强度: 0.05 ~ 0.5
                noise_scale = 0.05 + 0.45 * (aug_i / max(augment_factor - 1, 1))
                
                feat_noise = torch.randn_like(train_features) * feat_std * noise_scale
                sens_noise = torch.randn_like(train_sens) * sens_std * noise_scale
                
                # 增强特征
                aug_feat = train_features + feat_noise
                aug_sens = train_sens + sens_noise
                
                # [NEW] Sensitivity Masking
                # DISABLE: Masking breaks the physics bias alignment
                # Randomly set 15% of sensitivity values to 0 -> Cancelled
                # mask_prob = 0.15
                # sens_mask = torch.rand_like(aug_sens) > mask_prob
                # aug_sens = aug_sens * sens_mask.float()
                
                aug_features_list.append(aug_feat)
                aug_sens_list.append(aug_sens)
                aug_labels_list.append(train_labels.clone())
                aug_masks_list.append(train_masks.clone())  # Region mask remains unchanged
            
            train_features_aug = torch.cat(aug_features_list, dim=0)
            train_sens_aug = torch.cat(aug_sens_list, dim=0)
            train_labels_aug = torch.cat(aug_labels_list, dim=0)
            train_masks_aug = torch.cat(aug_masks_list, dim=0)
            
            logger.info(f"Augmented training samples: {train_features_aug.shape[0]} "
                        f"(Original {train_features.shape[0]} × {augment_factor + 1})")
            
            # 5. Initialize NodeLocalizer (Read parameters from config)
            nl_config = self.config.get('node_localizer', {})
            hidden_dim = int(nl_config.get('hidden_dim', 256))
            dropout = float(nl_config.get('dropout', 0.1))
            epochs = int(nl_config.get('epochs', n_epochs))
            lr = float(nl_config.get('learning_rate', lr))
            
            logger.info(f"NodeLocalizer Config: hidden_dim={hidden_dim}, dropout={dropout}, epochs={epochs}, lr={lr}")

            embed_dim = self.ltfm_model.embed_dim
            self.node_localizer = NodeLocalizer(
                region_dim=embed_dim,
                n_nodes=n_nodes,
                hidden_dim=hidden_dim,
                dropout=dropout
            ).to(self.device)
            
            optimizer = optim.Adam(self.node_localizer.parameters(), lr=lr, weight_decay=1e-4) # Increase weight_decay to prevent overfitting
            criterion = nn.CrossEntropyLoss()
            
            # 6. Create DataLoader (Including region mask)
            train_dataset = torch.utils.data.TensorDataset(
                train_features_aug, train_sens_aug, train_labels_aug, train_masks_aug
            )
            val_dataset = torch.utils.data.TensorDataset(
                val_features, val_sens, val_labels, val_masks
            )
            
            batch_size = min(512, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # OneCycleLR for better convergence
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr * 3, 
                steps_per_epoch=len(train_loader),
                epochs=epochs
            )
            
            # 7. Training loop
            best_node_acc_hop = 0.0
            patience_counter = 0
            
            for epoch in range(epochs):
                # Train
                self.node_localizer.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0
                
                for feat_batch, sens_batch, label_batch, mask_batch in train_loader:
                    feat_batch = feat_batch.to(self.device)
                    sens_batch = sens_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device) 
                    
                    optimizer.zero_grad()
                    node_scores = self.node_localizer(feat_batch, sens_batch)
                    
                    # Region constraint: Set logits of nodes outside the region to -inf
                    node_scores = node_scores.masked_fill(~mask_batch, -1e9)
                    loss = criterion(node_scores, label_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.node_localizer.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item() * feat_batch.shape[0]
                    preds = node_scores.argmax(dim=-1)
                    total_correct += (preds == label_batch).sum().item()
                    total_samples += feat_batch.shape[0]
                
                train_loss = total_loss / total_samples
                train_acc = total_correct / total_samples
                
                # Validate
                self.node_localizer.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for feat_batch, sens_batch, label_batch, mask_batch in val_loader:
                        feat_batch = feat_batch.to(self.device)
                        sens_batch = sens_batch.to(self.device)
                        label_batch = label_batch.to(self.device)
                        mask_batch = mask_batch.to(self.device)
                        
                        node_scores = self.node_localizer(feat_batch, sens_batch)
                        
                        # Region constraint
                        node_scores = node_scores.masked_fill(~mask_batch, -1e9)
                        loss = criterion(node_scores, label_batch)
                        val_loss += loss.item() * feat_batch.shape[0]
                        
                        preds = node_scores.argmax(dim=-1)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(label_batch.cpu().tolist())
                
                val_loss = val_loss / len(val_dataset)
                
                # Calculate node accuracy (Exact + Neighbor tolerance)
                node_acc_exact, node_acc_hop = self.calculate_node_accuracy(
                    all_preds, all_labels, node_names, network_graph
                )
                
                # Print log every 5 epochs or when there is progress
                should_log = (epoch + 1) % 5 == 0 or epoch == 0 or node_acc_hop > best_node_acc_hop
                if should_log:
                    logger.info(
                    f"NodeLocalizer Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Node Acc (exact): {node_acc_exact:.4f}, "
                    f"Node Acc (±1 hop): {node_acc_hop:.4f}"
                )
                
                # Early stopping
                if node_acc_hop > best_node_acc_hop:
                    best_node_acc_hop = node_acc_hop
                    patience_counter = 0
                    # Save best NodeLocalizer
                    self._save_node_localizer('best_node_localizer.pth')
                    logger.info(f"New best NodeLocalizer - Node Acc (±1 hop): {node_acc_hop:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= 25:
                    logger.info(f"NodeLocalizer early stopping, stopped at epoch {epoch+1}")
                    break
            
            logger.info(f"NodeLocalizer training completed - Best Node Acc (±1 hop): {best_node_acc_hop:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"NodeLocalizer training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_node_features_fast(self, scenarios: List[Dict], 
                                      anomaly_indices: List[int],
                                      global_embs: torch.Tensor,
                                      region_embs_list: List[torch.Tensor],
                                      region_labels: torch.Tensor,
                                      node_names: List[str]):
        """
        Fast node feature extraction: Reuse precomputed embeddings + Batch LTFM forward propagation
        
        Args:
            scenarios: All scenarios list
            anomaly_indices: Indices of anomaly scenarios
            global_embs: Precomputed global embeddings [N, dim]
            region_embs_list: List of precomputed region embeddings
            region_labels: Region labels [N]
            node_names: List of node names
            
        Returns:
            Tuple: (region_features, sensitivity_vectors, node_labels, region_masks)
                   region_masks: [n_samples, n_nodes] bool, True=node inside anomaly region
        """
        try:
            n_nodes = len(node_names)
            batch_size = 128
            
            # 1. Extract embeddings of anomaly scenarios
            anom_global = global_embs[anomaly_indices]  # [n_anomaly, dim]
            anom_regions = [r[anomaly_indices] for r in region_embs_list]  # List of [n_anomaly, dim]
            anom_rlabels = region_labels[anomaly_indices]  # [n_anomaly]
            
            n_anomaly = anom_global.shape[0]
            
            # 2. Batch LTFM forward propagation to extract intermediate features
            all_region_feats = []
            
            with torch.no_grad():
                for start in range(0, n_anomaly, batch_size):
                    end = min(start + batch_size, n_anomaly)
                    
                    batch_global = anom_global[start:end].to(self.device)
                    batch_regions = [r[start:end].to(self.device) for r in anom_regions]
                    batch_rlabels = anom_rlabels[start:end]
                    
                    _, _, intermediate_feats = self.ltfm_model(
                        batch_global, batch_regions, return_features=True
                    )
                    # intermediate_feats: List of [batch, embed_dim], one per region
                    
                    # Get features of the anomaly region for each sample
                    for i in range(end - start):
                        rlabel = batch_rlabels[i].item()
                        if rlabel > 0 and rlabel <= len(intermediate_feats):
                            feat = intermediate_feats[rlabel - 1][i]  # [embed_dim]
                        else:
                            feat = torch.stack([f[i] for f in intermediate_feats]).mean(0)
                        all_region_feats.append(feat.cpu())
            
            # 3. Build sensitivity vectors + Node labels + Region masks
            sensitivity_vectors = []
            node_labels = []
            region_masks = []
            
            for idx in anomaly_indices:
                scenario = scenarios[idx]
                anomaly_node_idx = scenario.get('anomaly_node_idx', -1)
                if anomaly_node_idx < 0:
                    continue
                
                sens_vector = torch.zeros(n_nodes)
                for j, name in enumerate(node_names):
                    if name in scenario['global_weights']:
                        sens_vector[j] = scenario['global_weights'][name]
                sensitivity_vectors.append(sens_vector)
                node_labels.append(anomaly_node_idx)
                
                # Build region mask: Only allow nodes inside the anomaly region
                pl = scenario.get('partition_labels', None)
                rlabel = scenario.get('region_label', 0)
                if pl is not None and rlabel > 0:
                    # region_label is 1-indexed, partition_labels is 0-indexed
                    region_idx = rlabel - 1
                    mask = torch.tensor([pl[j] == region_idx for j in range(n_nodes)], dtype=torch.bool)
                else:
                    # fallback: No constraint (all nodes are optional)
                    mask = torch.ones(n_nodes, dtype=torch.bool)
                region_masks.append(mask)
            
            if not all_region_feats:
                return None, None, None, None
            
            region_features = torch.stack(all_region_feats)
            sensitivity_vectors = torch.stack(sensitivity_vectors)
            node_labels = torch.tensor(node_labels, dtype=torch.long)
            region_masks_tensor = torch.stack(region_masks)
            
            logger.info(f"Fast feature extraction completed: {region_features.shape[0]} samples")
            return region_features, sensitivity_vectors, node_labels, region_masks_tensor
            
        except Exception as e:
            logger.error(f"Fast feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def calculate_node_accuracy(self, preds: List[int], labels: List[int],
                                 node_names: List[str], 
                                 network_graph: nx.Graph) -> Tuple[float, float]:
        """
        Calculate node-level accuracy (Exact + Neighbor tolerance ±1 hop)
        
        Args:
            preds: List of predicted node indices
            labels: List of true node indices
            node_names: List of node names
            network_graph: Network graph (used to find neighbors)
            
        Returns:
            Tuple[float, float]: (Exact accuracy, Neighbor tolerance accuracy)
        """
        if not preds or not labels:
            return 0.0, 0.0
        
        exact_correct = 0
        hop_correct = 0
        total = len(preds)
        
        for pred_idx, true_idx in zip(preds, labels):
            # Exact match
            if pred_idx == true_idx:
                exact_correct += 1
                hop_correct += 1
                continue
            
            # Neighbor tolerance: Check if the predicted node is within ±1 hop of the true node
            try:
                true_node_name = node_names[true_idx]
                pred_node_name = node_names[pred_idx]
                
                if true_node_name in network_graph and pred_node_name in network_graph:
                    neighbors = set(network_graph.neighbors(true_node_name))
                    if pred_node_name in neighbors:
                        hop_correct += 1
            except (IndexError, KeyError):
                pass
        
        exact_acc = exact_correct / total
        hop_acc = hop_correct / total
        
        return exact_acc, hop_acc
    
    def _save_node_localizer(self, filename: str):
        """Save NodeLocalizer model"""
        try:
            checkpoint_dir = os.path.join(self.config['data']['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            filepath = os.path.join(checkpoint_dir, filename)
            torch.save({
                'model_state_dict': self.node_localizer.state_dict(),
                'n_nodes': self.node_localizer.n_nodes,
                'region_dim': self.node_localizer.region_dim,
                'hidden_dim': self.node_localizer.hidden_dim
            }, filepath)
            logger.info(f"NodeLocalizer saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save NodeLocalizer: {e}")

    def save_checkpoint(self, filename: str) -> bool:
        """
        Save checkpoint

        Args:
            filename: Filename

        Returns:
            bool: Whether saving was successful
        """
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'ltfm_model_state_dict': self.ltfm_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'train_history': self.train_history,
                'config': self.config
            }

            # Use output directory from config
            output_dir = self.config['data'].get('output_dir', 'output')
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, filename: str) -> bool:
        """
        Load checkpoint

        Args:
            filename: Filename

        Returns:
            bool: Whether loading was successful
        """
        try:
            # Use output directory from config
            output_dir = self.config['data'].get('output_dir', 'output')
            checkpoint_path = os.path.join(output_dir, 'checkpoints', filename)

            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.ltfm_model.load_state_dict(checkpoint['ltfm_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_history = checkpoint['train_history']

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
