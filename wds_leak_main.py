#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water Distribution Network Leak Detection System
Integrates LTFM leak detection with existing partition and sensor placement results.

This script performs:
1. Load water network from EPANET INP file
2. Compute pressure sensitivity matrix
3. Load external partition results (FCM or Louvain community detection)
4. Load sensor placement results (optional, for inference)
5. Train Graph2Vec encoder + LTFM model + NodeLocalizer
6. Run inference for anomaly detection and localization

Usage:
  # Train with FCM partition
  python wds_leak_main.py --mode train --inp dataset/Exa7.inp \
    --partition partition_results/fcm_partition_summary.json --num-partitions 5

  # Train with Louvain partition
  python wds_leak_main.py --mode train --inp dataset/Exa7.inp \
    --partition partition_results/partition_summary.json --num-partitions 5

  # Inference
  python wds_leak_main.py --mode inference --inp dataset/Exa7.inp \
    --partition partition_results/fcm_partition_summary.json --num-partitions 5 \
    --sensor sensor_results/sensor_placement_20260210_091604.csv
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger

# Add LTFM-WaterNetwork/src to Python path
LTFM_DIR = os.path.join(os.path.dirname(__file__), 'LTFM-WaterNetwork')
sys.path.insert(0, LTFM_DIR)

from src.data.epanet_handler import EPANETHandler
from src.data.sensitivity_analyzer import SensitivityAnalyzer
from src.utils.fcm_partitioner import FCMPartitioner
from src.models.graph2vec_encoder import Graph2VecEncoder
from src.training.trainer import LTFMTrainer
from src.inference.predictor import LTFMPredictor


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    'data': {
        'output_dir': 'leak_detection_output/',
        'log_dir': 'leak_detection_output/logs/',
    },
    'hydraulic': {
        'simulation_hours': 24,
        'time_step': 1,
        'demand_reduction_ratio': 0.03,
    },
    'graph2vec': {
        'dimensions': 128,
        'workers': 4,
        'epochs': 10,
        'min_count': 5,
        'learning_rate': 0.025,
    },
    'ltfm': {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'dropout': 0.3,
    },
    'node_localizer': {
        'hidden_dim': 128,
        'dropout': 0.5,
        'epochs': 100,
        'learning_rate': 0.0001,
    },
    'training': {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 50,
        'weight_decay': 0.001,
        'early_stopping_patience': 20,
    },
    'inference': {
        'threshold': 0.5,
        'confidence_threshold': 0.8,
    },
    'device': {
        'use_cuda': True,
        'gpu_id': 0,
    },
    'logging': {
        'level': 'INFO',
        'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    },
    'fcm': {
        'n_clusters': 5,
        'max_iter': 100,
        'error': 1e-5,
        'm': 1.5,
        'k_nearest': 10,
        'outliers_detection': False,
        'seed': 42,
    },
}


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration. If a YAML file is specified, merge it with defaults.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                for section, values in user_config.items():
                    if section in config and isinstance(values, dict):
                        config[section].update(values)
                    else:
                        config[section] = values
            logger.info(f"Loaded config from: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}, using defaults")
    else:
        logger.info("Using default configuration")

    return config


def setup_logging(config: Dict):
    """Setup loguru logging."""
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = config.get('logging', {}).get('format',
                    '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    log_dir = config.get('data', {}).get('log_dir', 'leak_detection_output/logs/')

    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=log_level, format=log_format)
    logger.add(
        os.path.join(log_dir, 'leak_detection.log'),
        level=log_level,
        format=log_format,
        rotation='1 day',
        retention='30 days'
    )


# ============================================================
# Partition & Sensor Loaders
# ============================================================

def load_external_partitions(partition_json_path: str,
                             num_partitions: int = None) -> Dict[str, int]:
    """
    Load external partition results from JSON (supports both FCM and Louvain formats).

    Supported formats:
      - FCM: {"num_clusters": {"node_assignments": {"node": cluster_id}, ...}}
      - Louvain: {"num_communities": {"num_communities": N, "resolution": R,
                   "node_assignments": {"node": cluster_id}}}

    Args:
        partition_json_path: Path to partition JSON file
        num_partitions: Desired number of partitions (selects the closest key)

    Returns:
        dict: {node_name: cluster_id} mapping
    """
    with open(partition_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keys are string representations of partition counts
    available_keys = list(data.keys())
    logger.info(f"Available partition keys: {available_keys}")

    if num_partitions is not None:
        # Try exact match first
        key = str(num_partitions)
        if key not in data:
            # Find closest key
            int_keys = [int(k) for k in available_keys if k.isdigit()]
            if not int_keys:
                raise ValueError(f"No valid partition keys found in {partition_json_path}")
            closest = min(int_keys, key=lambda x: abs(x - num_partitions))
            key = str(closest)
            logger.warning(f"Partition {num_partitions} not found, using closest: {key}")
    else:
        key = available_keys[0]
        logger.info(f"No partition number specified, using first key: {key}")

    partition_data = data[key]
    node_assignments = partition_data.get('node_assignments', {})

    if not node_assignments:
        raise ValueError(f"No node_assignments found for key={key}")

    n_clusters = len(set(node_assignments.values()))
    logger.info(f"Loaded partition: key={key}, nodes={len(node_assignments)}, "
                f"clusters={n_clusters}")

    return node_assignments


def load_sensor_placement(sensor_csv_path: str) -> pd.DataFrame:
    """
    Load sensor placement results from CSV.

    Expected columns: Sensor_ID, Node_Name, Partition_ID, X, Y, Resilience

    Args:
        sensor_csv_path: Path to sensor placement CSV

    Returns:
        pd.DataFrame: Sensor placement data
    """
    df = pd.read_csv(sensor_csv_path)
    logger.info(f"Loaded {len(df)} sensors from {sensor_csv_path}")
    logger.info(f"  Sensors per partition: "
                f"{df.groupby('Partition_ID').size().to_dict()}")
    return df


def build_partition_labels_and_subgraphs(
        node_assignments: Dict[str, int],
        node_names: List[str],
        network_graph: nx.Graph
) -> Tuple[np.ndarray, List[nx.Graph], int]:
    """
    Convert external partition node_assignments into:
      - partition_labels array (aligned with node_names order)
      - subgraph list (one per cluster)
      - number of clusters

    Args:
        node_assignments: {node_name: cluster_id} from external partition
        node_names: Ordered list of demand node names from EPANET
        network_graph: Full NetworkX graph of the network

    Returns:
        (partition_labels, subgraphs, n_clusters)
    """
    # Normalize cluster IDs to start from 0
    unique_clusters = sorted(set(node_assignments.values()))
    cluster_remap = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    n_clusters = len(unique_clusters)

    # Build partition_labels array aligned with node_names
    partition_labels = np.zeros(len(node_names), dtype=int)
    unassigned = []
    for i, node in enumerate(node_names):
        if node in node_assignments:
            partition_labels[i] = cluster_remap[node_assignments[node]]
        else:
            unassigned.append(node)
            partition_labels[i] = 0  # Default to cluster 0

    if unassigned:
        logger.warning(f"{len(unassigned)} demand nodes not in partition assignments, "
                       f"defaulting to cluster 0. Sample: {unassigned[:5]}")

    # Build subgraphs
    subgraphs = []
    for cluster_id in range(n_clusters):
        cluster_nodes = [node_names[i] for i in range(len(node_names))
                         if partition_labels[i] == cluster_id]
        # Only include nodes that exist in the graph
        valid_nodes = [n for n in cluster_nodes if n in network_graph.nodes()]
        subgraph = network_graph.subgraph(valid_nodes).copy()
        subgraphs.append(subgraph)
        logger.info(f"  Partition {cluster_id}: {len(subgraph.nodes)} nodes, "
                    f"{len(subgraph.edges)} edges")

    return partition_labels, subgraphs, n_clusters


def inject_partition_into_fcm(fcm_partitioner: FCMPartitioner,
                              partition_labels: np.ndarray,
                              n_clusters: int):
    """
    Inject external partition labels into an FCMPartitioner instance,
    making it compatible with LTFM Trainer's expected interface.

    Args:
        fcm_partitioner: FCMPartitioner instance
        partition_labels: Array of cluster assignments
        n_clusters: Number of clusters
    """
    fcm_partitioner.partition_labels = partition_labels
    fcm_partitioner.n_clusters = n_clusters
    # Create dummy membership matrix (hard assignment)
    n_nodes = len(partition_labels)
    membership = np.zeros((n_clusters, n_nodes))
    for i, label in enumerate(partition_labels):
        membership[label, i] = 1.0
    fcm_partitioner.membership_matrix = membership
    fcm_partitioner.cluster_centers = None

    logger.info(f"Injected external partition into FCMPartitioner: "
                f"{n_clusters} clusters, {n_nodes} nodes")


# ============================================================
# Train Mode
# ============================================================

def train_mode(config: Dict, args) -> bool:
    """
    Training mode: compute sensitivity, load external partition,
    train Graph2Vec + LTFM + NodeLocalizer.
    """
    try:
        logger.info("=" * 60)
        logger.info("LEAK DETECTION - TRAINING MODE")
        logger.info("=" * 60)

        # ---- Step 1: Load EPANET network ----
        logger.info("\n[Step 1/7] Loading EPANET network...")
        epanet_file = args.inp
        epanet_handler = EPANETHandler(epanet_file)

        if not epanet_handler.load_network():
            logger.error("Failed to load EPANET network")
            return False

        node_names = epanet_handler.node_names
        logger.info(f"  Network: {len(node_names)} demand nodes")

        # ---- Step 2: Pressure sensitivity analysis ----
        logger.info("\n[Step 2/7] Computing pressure sensitivity...")
        sensitivity_analyzer = SensitivityAnalyzer(epanet_handler)

        if not sensitivity_analyzer.calculate_normal_pressure(
                config['hydraulic']['simulation_hours'],
                config['hydraulic']['time_step']):
            logger.error("Normal pressure calculation failed")
            return False

        if not sensitivity_analyzer.calculate_sensitivity_matrix(
                config['hydraulic']['demand_reduction_ratio'],
                config['hydraulic']['simulation_hours'],
                config['hydraulic']['time_step']):
            logger.error("Sensitivity matrix calculation failed")
            return False

        sensitivity_analyzer.normalize_sensitivity_matrix()
        node_weights = sensitivity_analyzer.calculate_node_weights()

        # Save sensitivity data
        sensitivity_analyzer.save_sensitivity_data(config['data']['output_dir'])
        logger.info("  Sensitivity matrix computed and saved")

        # ---- Step 3: Load external partition results ----
        logger.info("\n[Step 3/7] Loading external partition results...")
        node_assignments = load_external_partitions(
            args.partition, args.num_partitions
        )

        network_graph = epanet_handler.get_network_graph()
        partition_labels, subgraphs, n_clusters = \
            build_partition_labels_and_subgraphs(
                node_assignments, node_names, network_graph
            )

        # Create FCMPartitioner adapter with injected partition
        config['fcm']['n_clusters'] = n_clusters
        fcm_partitioner = FCMPartitioner(
            n_clusters=n_clusters,
            m=config['fcm']['m'],
            max_iter=config['fcm']['max_iter'],
            error=config['fcm']['error']
        )
        inject_partition_into_fcm(fcm_partitioner, partition_labels, n_clusters)

        # Save partition results for inference use
        fcm_partitioner.save_partition_results(
            config['data']['output_dir'], node_names
        )

        logger.info(f"  Partition: {n_clusters} clusters loaded from external source")

        # ---- Step 4: Train Graph2Vec ----
        logger.info("\n[Step 4/7] Training Graph2Vec encoder...")
        all_graphs = [network_graph] + subgraphs

        graph2vec_encoder = Graph2VecEncoder(
            embedding_dim=int(config['graph2vec']['dimensions']),
            workers=int(config['graph2vec']['workers']),
            epochs=int(config['graph2vec']['epochs']),
            min_count=int(config['graph2vec']['min_count']),
            learning_rate=float(config['graph2vec']['learning_rate'])
        )

        if not graph2vec_encoder.train_model(all_graphs):
            logger.error("Graph2Vec training failed")
            return False

        graph2vec_path = os.path.join(config['data']['output_dir'],
                                       'graph2vec_model.pth')
        graph2vec_encoder.save_model(graph2vec_path)
        logger.info(f"  Graph2Vec model saved to {graph2vec_path}")

        # ---- Step 5: Generate training scenarios ----
        logger.info("\n[Step 5/7] Generating training scenarios...")
        trainer = LTFMTrainer(config)

        if not trainer.initialize_models():
            logger.error("LTFM model initialization failed")
            return False

        # Set pre-trained Graph2Vec
        trainer.graph2vec_encoder = graph2vec_encoder

        # Generate scenarios using external partition
        n_scenarios = args.n_scenarios or 1000
        train_scenarios = trainer.generate_training_scenarios(
            epanet_handler, sensitivity_analyzer, fcm_partitioner,
            n_scenarios=n_scenarios
        )

        # Stratified train/val split
        normal = [s for s in train_scenarios if s['global_label'] == 0]
        anomaly = [s for s in train_scenarios if s['global_label'] == 1]

        logger.info(f"  Normal scenarios: {len(normal)}, "
                    f"Anomaly scenarios: {len(anomaly)}")

        n_split = int(len(normal) * 0.8)
        a_split = int(len(anomaly) * 0.8)

        import random
        random.shuffle(normal)
        random.shuffle(anomaly)

        train_data = normal[:n_split] + anomaly[:a_split]
        val_data = normal[n_split:] + anomaly[a_split:]
        random.shuffle(train_data)
        random.shuffle(val_data)

        logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        # ---- Step 6: Train LTFM model ----
        logger.info("\n[Step 6/7] Training LTFM model...")
        if not trainer.train(train_data, val_data,
                             skip_stage1=args.skip_stage1):
            logger.error("LTFM model training failed")
            return False

        # ---- Step 7: Train NodeLocalizer ----
        logger.info("\n[Step 7/7] Training NodeLocalizer (node-level localization)...")
        if not trainer.train_node_localizer(train_data, val_data):
            logger.warning("NodeLocalizer training failed, "
                           "but LTFM model is trained successfully")

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"  Output directory: {config['data']['output_dir']}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Training mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Inference Mode
# ============================================================

def inference_mode(config: Dict, args) -> bool:
    """
    Inference mode: load trained models and partition info,
    predict anomalies using pressure data.
    """
    try:
        logger.info("=" * 60)
        logger.info("LEAK DETECTION - INFERENCE MODE")
        logger.info("=" * 60)

        # ---- Step 1: Initialize predictor ----
        logger.info("\n[Step 1/4] Initializing predictor...")
        predictor = LTFMPredictor(config)

        # Load models
        graph2vec_path = args.graph2vec_model or os.path.join(
            config['data']['output_dir'], 'graph2vec_model.pth')
        ltfm_path = args.ltfm_checkpoint or os.path.join(
            config['data']['output_dir'], 'checkpoints', 'best_model.pth')
        partition_info_path = os.path.join(
            config['data']['output_dir'], 'fcm_partition.csv')

        if not predictor.load_models(graph2vec_path, ltfm_path,
                                     partition_info_path):
            logger.error("Failed to load models")
            return False

        # ---- Step 2: Initialize network ----
        logger.info("\n[Step 2/4] Initializing EPANET network...")
        epanet_file = args.inp
        if not predictor.initialize_network(epanet_file):
            logger.error("Network initialization failed")
            return False

        status = predictor.get_network_status()
        logger.info(f"  Network status: {status}")

        # ---- Step 3: Load sensor placement (optional) ----
        sensor_nodes = None
        if args.sensor:
            logger.info("\n[Step 3/4] Loading sensor placement...")
            sensor_df = load_sensor_placement(args.sensor)
            sensor_nodes = sensor_df['Node_Name'].astype(str).tolist()
            logger.info(f"  Monitoring nodes (sensors): {sensor_nodes}")
        else:
            logger.info("\n[Step 3/4] No sensor file specified, "
                        "using all nodes for monitoring")

        # ---- Step 4: Run prediction ----
        logger.info("\n[Step 4/4] Running anomaly detection...")

        if args.test_data:
            # Batch prediction from CSV
            logger.info(f"  Loading test data: {args.test_data}")
            test_pressure = pd.read_csv(args.test_data, index_col=0,
                                        encoding='utf-8')

            # If sensor nodes specified, filter columns
            if sensor_nodes:
                available = [n for n in sensor_nodes
                             if n in test_pressure.columns]
                if available:
                    logger.info(f"  Using {len(available)} sensor nodes "
                                f"out of {len(test_pressure.columns)}")
                    # Note: prediction still needs all nodes,
                    # sensor filtering is for display/reporting

            results = predictor.batch_predict([test_pressure])

            # Save results
            output_path = args.output or os.path.join(
                config['data']['output_dir'], 'prediction_results.csv')
            predictor.save_prediction_results(results, output_path)

            # Display results
            for i, result in enumerate(results):
                _print_prediction_result(result, i, sensor_nodes)

        else:
            # Demo mode with normal pressure data
            logger.info("  Demo mode: predicting with normal pressure data")
            result = predictor.predict_anomaly(predictor.normal_pressure)
            _print_prediction_result(result, 0, sensor_nodes)

        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE COMPLETED")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Inference mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _print_prediction_result(result: Dict, idx: int,
                              sensor_nodes: Optional[List[str]] = None):
    """Pretty-print a single prediction result."""
    if not result:
        logger.warning(f"  Sample {idx}: No result")
        return

    is_anomaly = result.get('global_anomaly', False)
    probability = result.get('global_probability', 0)
    region = result.get('anomaly_region', 0)
    confidence = result.get('confidence', 0)

    status_icon = "🚨" if is_anomaly else "✅"
    logger.info(f"\n  {status_icon} Sample {idx}:")
    logger.info(f"    Anomaly detected: {is_anomaly}")
    logger.info(f"    Probability: {probability:.4f}")
    logger.info(f"    Anomaly region: {region}")
    logger.info(f"    Confidence: {confidence}")

    if 'node_prediction' in result:
        logger.info(f"    Predicted leak node: {result['node_prediction']}")

    if 'partitions' in result:
        logger.info(f"    Partition count: {len(result['partitions'])}")
        for i, partition in enumerate(result['partitions']):
            n_nodes = len(partition) if isinstance(partition, list) else partition
            logger.info(f"      Region {i}: {n_nodes} nodes")

    if sensor_nodes:
        logger.info(f"    Sensor nodes used: {len(sensor_nodes)}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='WDS Leak Detection System '
                    '(LTFM with external partition & sensor placement)',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--mode', choices=['train', 'inference'],
                        default='train',
                        help='Run mode: train or inference')
    parser.add_argument('--config', default=None,
                        help='Optional YAML config file path '
                             '(overrides defaults)')
    parser.add_argument('--inp', required=True,
                        help='EPANET INP network file path')
    parser.add_argument('--partition', required=True,
                        help='Partition JSON file path '
                             '(from FCM or Louvain)')
    parser.add_argument('--num-partitions', type=int, default=None,
                        help='Number of partitions to use '
                             '(selects closest available)')

    # Training arguments
    parser.add_argument('--n-scenarios', type=int, default=None,
                        help='Number of training scenarios '
                             '(default: 1000)')
    parser.add_argument('--skip-stage1', action='store_true',
                        help='Skip LTFM training (Stage 1), '
                             'only train NodeLocalizer (Stage 2)')

    # Inference arguments
    parser.add_argument('--sensor', default=None,
                        help='Sensor placement CSV file path')
    parser.add_argument('--graph2vec-model', default=None,
                        help='Graph2Vec model path '
                             '(default: output/graph2vec_model.pth)')
    parser.add_argument('--ltfm-checkpoint', default=None,
                        help='LTFM checkpoint path '
                             '(default: output/checkpoints/best_model.pth)')
    parser.add_argument('--test-data', default=None,
                        help='Test pressure data CSV path')
    parser.add_argument('--output', default=None,
                        help='Output file path for prediction results')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)

    # Create output directories
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['data']['output_dir'], 'checkpoints'),
                exist_ok=True)

    # Log startup info
    logger.info("WDS Leak Detection System")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  INP file: {args.inp}")
    logger.info(f"  Partition file: {args.partition}")
    logger.info(f"  Num partitions: {args.num_partitions or 'auto'}")
    if args.sensor:
        logger.info(f"  Sensor file: {args.sensor}")

    # Run
    if args.mode == 'train':
        success = train_mode(config, args)
    else:
        success = inference_mode(config, args)

    if success:
        logger.info("Program completed successfully")
    else:
        logger.error("Program failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
