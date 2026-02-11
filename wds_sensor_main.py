"""
Water Distribution Network Sensor Placement Optimization
Based on partition results from wds_partition_main.py

This script performs:
1. Load water network from EPANET INP file
2. Load partition results from partition_summary.json
3. Compute pressure sensitivity matrix for each partition
4. Select optimal sensor locations based on coverage and resilience
5. Evaluate sensor placement resilience under failure scenarios
6. Save results and generate visualization

Author: Based on sensor_placement.py
"""
import os
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

import wntr


def setup_logger(name: str):
    """Setup logger for the sensor placement module"""
    logger = logging.getLogger(f"sensor.{name}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger("SensorPlacement")


def load_partition_results(partition_summary_path: str, num_partitions: int = None):
    """
    Load partition results from partition_summary.json
    
    Args:
        partition_summary_path: Path to partition_summary.json
        num_partitions: Desired number of partitions (if None, use the first available)
    
    Returns:
        dict: {partition_id: [list of node IDs]}
    """
    logger.info(f"Loading partition results from: {partition_summary_path}")
    
    with open(partition_summary_path, 'r') as f:
        summary = json.load(f)
    
    # Find available partition counts
    available_partitions = sorted([int(k) for k in summary.keys()])
    logger.info(f"Available partition counts: {available_partitions}")
    
    # Select partition count
    if num_partitions is None:
        # Default to a reasonable partition count (e.g., 5 or first available > 2)
        for n in [5, 4, 3, 2] + available_partitions:
            if n in available_partitions:
                num_partitions = n
                break
    
    if num_partitions not in available_partitions:
        raise ValueError(f"Partition count {num_partitions} not available. "
                        f"Available: {available_partitions}")
    
    logger.info(f"Using {num_partitions} partitions")
    
    # Extract node assignments
    partition_data = summary[str(num_partitions)]
    node_assignments = partition_data['node_assignments']
    
    # Group nodes by partition
    partitions = {}
    for node_id, partition_id in node_assignments.items():
        if partition_id not in partitions:
            partitions[partition_id] = []
        partitions[partition_id].append(node_id)
    
    # Log partition statistics
    for pid, nodes in sorted(partitions.items()):
        logger.info(f"  Partition {pid}: {len(nodes)} nodes")
    
    return partitions


def compute_pressure_sensitivity_matrix(wn, partitions, demand_ratio=0.20):
    """
    Compute pressure sensitivity matrix based on demand perturbation
    
    Args:
        wn: WNTR water network model
        partitions: dict of {partition_id: [node_ids]}
        demand_ratio: Demand perturbation ratio
    
    Returns:
        dict containing sensitivity matrix and node information
    """
    logger.info(f"Computing pressure sensitivity matrix (demand ratio: {demand_ratio})")
    
    # Get all nodes from partitions
    all_nodes = []
    for nodes in partitions.values():
        all_nodes.extend(nodes)
    
    # Filter valid demand nodes (junctions only)
    valid_nodes = []
    for node in all_nodes:
        if node in wn.junction_name_list:
            valid_nodes.append(node)
        else:
            logger.debug(f"Node {node} is not a junction, skipping")
    
    if not valid_nodes:
        raise ValueError("No valid junction nodes found in partitions")
    
    logger.info(f"Valid junction nodes: {len(valid_nodes)}")
    
    # Run baseline simulation
    logger.info("Running baseline simulation...")
    sim = wntr.sim.EpanetSimulator(wn)
    base_results = sim.run_sim()
    base_pressure = base_results.node['pressure'].loc[:, valid_nodes].values
    
    # Calculate total demand for scaling
    total_demand = 0
    for name in valid_nodes:
        node_demands = base_results.node['demand'].loc[:, name]
        total_demand += node_demands.sum()
    
    logger.info(f"Total demand: {total_demand:.4f}")
    
    # Initialize sensitivity matrix
    n_nodes = len(valid_nodes)
    sensitivity_matrix = np.zeros((n_nodes, n_nodes))
    
    # Compute delta for perturbation
    num_timesteps = len(base_results.node['demand'])
    delta = float(total_demand * demand_ratio / num_timesteps)
    
    # Perturb each node and compute sensitivity
    for i, perturb_node in enumerate(valid_nodes):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Processing node {i+1}/{n_nodes}: {perturb_node} "
                       f"({(i+1)/n_nodes*100:.1f}%)")
        
        # Save original demands
        node = wn.get_node(perturb_node)
        original_demands = {}
        for j, ts in enumerate(node.demand_timeseries_list):
            original_demands[j] = ts.base_value
        
        # Apply perturbation
        for j, ts in enumerate(node.demand_timeseries_list):
            orig = original_demands[j]
            # Handle None or non-numeric values
            if orig is None:
                orig = 0.0
            try:
                orig = float(orig)
            except (TypeError, ValueError):
                orig = 0.0
            
            if orig > 0:
                # Proportional perturbation
                ts.base_value = orig * (1 + demand_ratio)
            else:
                # Small absolute perturbation for zero-demand nodes
                ts.base_value = delta
        
        # Run perturbed simulation
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            perturb_results = sim.run_sim()
            perturb_pressure = perturb_results.node['pressure'].loc[:, valid_nodes].values
            
            # Compute pressure difference
            pressure_diff = np.abs(perturb_pressure - base_pressure)
            avg_pressure_diff = np.mean(pressure_diff, axis=0)
            
            # Normalize
            max_diff = np.max(avg_pressure_diff)
            if max_diff > 0:
                sensitivity_matrix[i, :] = avg_pressure_diff / max_diff
            else:
                sensitivity_matrix[i, :] = 0
        except Exception as e:
            logger.warning(f"Simulation failed for node {perturb_node}: {e}")
            sensitivity_matrix[i, :] = 0
        
        # Restore original demands
        node = wn.get_node(perturb_node)
        for j, ts in enumerate(node.demand_timeseries_list):
            orig = original_demands[j]
            if orig is None:
                orig = 0.0
            try:
                ts.base_value = float(orig)
            except (TypeError, ValueError):
                ts.base_value = 0.0
    
    # Log statistics
    non_zero_count = np.count_nonzero(sensitivity_matrix)
    total_elements = sensitivity_matrix.size
    logger.info(f"Sensitivity matrix: {non_zero_count}/{total_elements} non-zero "
               f"({non_zero_count/total_elements*100:.1f}%)")
    logger.info(f"Sensitivity range: [{np.min(sensitivity_matrix):.6f}, "
               f"{np.max(sensitivity_matrix):.6f}]")
    
    # Update partitions to only include valid nodes
    valid_partitions = {}
    for partition_id, nodes in partitions.items():
        valid_partition_nodes = [n for n in nodes if n in valid_nodes]
        if valid_partition_nodes:
            valid_partitions[partition_id] = valid_partition_nodes
    
    return {
        'matrix': sensitivity_matrix,
        'nodes': valid_nodes,
        'partitions': valid_partitions
    }


def select_sensors_by_partition(sensitivity_data, threshold=0.5, 
                                 min_sensor_ratio=0.04, max_sensor_ratio=0.15,
                                 target_coverage=0.95):
    """
    Select optimal sensor locations for each partition
    
    Args:
        sensitivity_data: Output from compute_pressure_sensitivity_matrix
        threshold: Sensitivity threshold for detection
        min_sensor_ratio: Minimum sensors as fraction of partition size
        max_sensor_ratio: Maximum sensors as fraction of partition size
        target_coverage: Target coverage rate
    
    Returns:
        dict: {partition_id: [sensor info dicts]}
    """
    logger.info(f"Selecting sensors (threshold: {threshold}, "
               f"coverage target: {target_coverage})")
    
    sensitivity_matrix = sensitivity_data['matrix']
    all_nodes = sensitivity_data['nodes']
    partitions = sensitivity_data['partitions']
    
    # Create node index mapping
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    selected_sensors = {}
    
    for partition_id, partition_nodes in sorted(partitions.items()):
        logger.info(f"Processing partition {partition_id}: {len(partition_nodes)} nodes")
        
        # Get partition node indices
        partition_indices = [node_to_index[n] for n in partition_nodes 
                           if n in node_to_index]
        
        if len(partition_indices) < 2:
            logger.warning(f"Partition {partition_id} has less than 2 nodes, skipping")
            continue
        
        # Calculate influence scores for each node
        influence_scores = {}
        for node_name in partition_nodes:
            if node_name not in node_to_index:
                continue
            
            node_idx = node_to_index[node_name]
            partition_sensitivities = sensitivity_matrix[node_idx, partition_indices]
            detectable_count = np.sum(partition_sensitivities > threshold)
            
            influence_scores[node_name] = {
                'index': node_idx,
                'detectable_count': detectable_count,
                'avg_sensitivity': np.mean(partition_sensitivities)
            }
        
        # Determine sensor count range
        partition_size = len(partition_nodes)
        min_sensors = max(2, int(partition_size * min_sensor_ratio))
        max_sensors = min(10, max(3, int(partition_size * max_sensor_ratio)))
        
        logger.info(f"  Sensor range: {min_sensors}-{max_sensors}")
        
        # Greedy sensor selection
        uncovered_indices = set(partition_indices)
        selected_sensors[partition_id] = []
        
        # Phase 1: Coverage-based selection
        while (uncovered_indices and 
               len(selected_sensors[partition_id]) < max_sensors and
               len(uncovered_indices) / len(partition_indices) > (1 - target_coverage)):
            
            best_sensor = None
            best_coverage = 0
            
            for node_name, info in influence_scores.items():
                if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                    continue
                
                node_idx = info['index']
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
            
            # Update uncovered nodes
            sensor_idx = best_sensor['index']
            covered = [idx for idx in uncovered_indices 
                      if sensitivity_matrix[sensor_idx, idx] > threshold]
            for idx in covered:
                uncovered_indices.discard(idx)
        
        # Phase 2: Ensure minimum sensor count for resilience
        while len(selected_sensors[partition_id]) < min_sensors:
            best_sensor = None
            max_diversity = 0
            
            for node_name, info in influence_scores.items():
                if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                    continue
                
                node_idx = info['index']
                
                # Calculate diversity (distance from existing sensors)
                diversity_score = 0
                for existing in selected_sensors[partition_id]:
                    existing_idx = existing['index']
                    distance = 1 - sensitivity_matrix[node_idx, existing_idx]
                    diversity_score += distance
                
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
            
            if best_sensor:
                selected_sensors[partition_id].append(best_sensor)
            else:
                break
        
        logger.info(f"  Selected {len(selected_sensors[partition_id])} sensors")
    
    return selected_sensors


def evaluate_resilience(selected_sensors, sensitivity_data, threshold=0.5):
    """
    Evaluate sensor placement resilience under failure scenarios
    
    Args:
        selected_sensors: Output from select_sensors_by_partition
        sensitivity_data: Output from compute_pressure_sensitivity_matrix
        threshold: Sensitivity threshold
    
    Returns:
        dict: Resilience analysis results for each partition
    """
    logger.info("Evaluating sensor placement resilience...")
    
    sensitivity_matrix = sensitivity_data['matrix']
    all_nodes = sensitivity_data['nodes']
    partitions = sensitivity_data['partitions']
    
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    resilience_results = {}
    
    for partition_id, sensors in selected_sensors.items():
        partition_nodes = partitions[partition_id]
        partition_indices = [node_to_index[n] for n in partition_nodes 
                           if n in node_to_index]
        
        logger.info(f"Evaluating partition {partition_id}: "
                   f"{len(sensors)} sensors, {len(partition_nodes)} nodes")
        
        detailed_scenarios = []
        
        # Scenario 1: All sensors working
        all_sensor_indices = [s['index'] for s in sensors]
        all_sensor_nodes = [s['node'] for s in sensors]
        
        full_detected, full_coverage = _calculate_detection(
            all_sensor_indices, partition_indices, sensitivity_matrix, threshold
        )
        
        detailed_scenarios.append({
            'scenario_type': 'All sensors working',
            'failed_sensors': [],
            'remaining_sensors': all_sensor_nodes,
            'detected_nodes': full_detected,
            'total_nodes': len(partition_nodes),
            'coverage_rate': full_coverage,
            'coverage_percentage': f"{full_coverage*100:.1f}%"
        })
        
        # Scenario 2+: Failure scenarios
        total_failure_coverage = 0.0
        failure_scenario_count = 0
        
        for failure_count in range(1, len(sensors)):
            for failed_indices in combinations(range(len(sensors)), failure_count):
                failed = [sensors[i]['node'] for i in failed_indices]
                remaining_indices = [sensors[i]['index'] for i in range(len(sensors))
                                   if i not in failed_indices]
                remaining_nodes = [sensors[i]['node'] for i in range(len(sensors))
                                  if i not in failed_indices]
                
                detected, coverage = _calculate_detection(
                    remaining_indices, partition_indices, sensitivity_matrix, threshold
                )
                
                total_failure_coverage += coverage
                failure_scenario_count += 1
                
                detailed_scenarios.append({
                    'scenario_type': f'{failure_count} sensor(s) failed',
                    'failed_sensors': failed,
                    'remaining_sensors': remaining_nodes,
                    'detected_nodes': detected,
                    'total_nodes': len(partition_nodes),
                    'coverage_rate': coverage,
                    'coverage_percentage': f"{coverage*100:.1f}%"
                })
        
        avg_failure_resilience = (total_failure_coverage / failure_scenario_count 
                                 if failure_scenario_count > 0 else 0.0)
        
        resilience_results[partition_id] = {
            'detailed_scenarios': detailed_scenarios,
            'resilience_score': avg_failure_resilience,
            'sensor_count': len(sensors),
            'full_coverage_rate': full_coverage,
            'avg_failure_coverage': avg_failure_resilience,
            'total_scenarios': len(detailed_scenarios),
            'failure_scenarios': failure_scenario_count
        }
        
        logger.info(f"  Resilience score: {avg_failure_resilience:.4f}")
    
    return resilience_results


def _calculate_detection(sensor_indices, target_indices, sensitivity_matrix, threshold):
    """Helper function to calculate detection count and coverage rate"""
    if not sensor_indices or not target_indices:
        return 0, 0.0
    
    detected_count = 0
    for target_idx in target_indices:
        for sensor_idx in sensor_indices:
            if sensitivity_matrix[sensor_idx, target_idx] > threshold:
                detected_count += 1
                break
    
    coverage_rate = detected_count / len(target_indices)
    return detected_count, coverage_rate


def optimize_sensor_placement(sensitivity_data, thresholds=None):
    """
    Find optimal sensor placement by trying different thresholds
    
    Args:
        sensitivity_data: Output from compute_pressure_sensitivity_matrix
        thresholds: List of thresholds to try
    
    Returns:
        dict: Best solution with sensors, resilience, threshold, and score
    """
    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6]
    
    logger.info(f"Optimizing sensor placement with thresholds: {thresholds}")
    
    best_solution = None
    best_score = 0.0
    
    for threshold in thresholds:
        logger.info(f"Trying threshold: {threshold}")
        
        # Select sensors
        selected_sensors = select_sensors_by_partition(sensitivity_data, threshold)
        if not selected_sensors:
            continue
        
        # Evaluate resilience
        resilience_results = evaluate_resilience(selected_sensors, sensitivity_data, threshold)
        if not resilience_results:
            continue
        
        # Calculate total score
        total_sensors = sum(len(s) for s in selected_sensors.values())
        sensor_penalty = total_sensors * 0.001
        
        resilience_scores = [r['resilience_score'] for r in resilience_results.values()]
        avg_resilience = np.mean(resilience_scores) if resilience_scores else 0.0
        
        # Coverage score
        partition_count = len(selected_sensors)
        coverage_score = min(1.0, total_sensors / (partition_count * 2))
        
        # Combined score (40% resilience, 60% coverage)
        total_score = (avg_resilience * 0.4 + coverage_score * 0.6 - sensor_penalty)
        total_score = max(0.001, total_score)
        
        logger.info(f"  Score: {total_score:.4f}")
        
        if total_score > best_score:
            best_score = total_score
            best_solution = {
                'sensors': selected_sensors,
                'resilience': resilience_results,
                'threshold': threshold,
                'score': total_score
            }
    
    if best_solution:
        logger.info(f"Best solution: threshold={best_solution['threshold']}, "
                   f"score={best_solution['score']:.4f}")
    
    return best_solution


def save_sensor_results(solution, wn, output_dir):
    """
    Save sensor placement results to CSV files
    
    Args:
        solution: Output from optimize_sensor_placement
        wn: WNTR water network model
        output_dir: Output directory
    
    Returns:
        dict: File information
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sensor placement results
    results_data = []
    sensor_id = 1
    
    for partition_id, sensors in solution['sensors'].items():
        for sensor in sensors:
            node_name = sensor['node']
            
            # Get node coordinates
            try:
                coord = wn.get_node(node_name).coordinates
                if coord is None:
                    coord = (0, 0)
            except:
                coord = (0, 0)
            
            resilience_info = solution['resilience'].get(partition_id, {})
            resilience_score = resilience_info.get('resilience_score', 0.0)
            
            results_data.append({
                'Sensor_ID': f'S{sensor_id:03d}',
                'Node_Name': node_name,
                'Partition_ID': partition_id,
                'X_Coord': coord[0],
                'Y_Coord': coord[1],
                'Influence_Score': sensor['influence_score'],
                'Avg_Sensitivity': f"{sensor['avg_sensitivity']:.4f}",
                'Coverage_Nodes': sensor['coverage'],
                'Partition_Resilience': f"{resilience_score:.4f}",
                'Sensitivity_Threshold': solution['threshold']
            })
            sensor_id += 1
    
    # Save main results
    df = pd.DataFrame(results_data)
    sensor_file = os.path.join(output_dir, f'sensor_placement_{timestamp}.csv')
    df.to_csv(sensor_file, index=False, encoding='utf-8-sig')
    logger.info(f"Sensor results saved to: {sensor_file}")
    
    # Save resilience analysis
    resilience_data = []
    for partition_id, resilience_info in solution['resilience'].items():
        for scenario in resilience_info.get('detailed_scenarios', []):
            resilience_data.append({
                'Partition_ID': partition_id,
                'Scenario_Type': scenario['scenario_type'],
                'Failed_Sensors': ', '.join(scenario['failed_sensors']) 
                                 if scenario['failed_sensors'] else 'None',
                'Remaining_Sensors': ', '.join(scenario['remaining_sensors']),
                'Detected_Nodes': scenario['detected_nodes'],
                'Total_Nodes': scenario['total_nodes'],
                'Coverage_Rate': f"{scenario['coverage_rate']:.4f}",
                'Coverage_Percentage': scenario['coverage_percentage']
            })
    
    if resilience_data:
        df_resilience = pd.DataFrame(resilience_data)
        resilience_file = os.path.join(output_dir, f'resilience_analysis_{timestamp}.csv')
        df_resilience.to_csv(resilience_file, index=False, encoding='utf-8-sig')
        logger.info(f"Resilience analysis saved to: {resilience_file}")
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'total_sensors': len(results_data),
        'num_partitions': len(solution['sensors']),
        'threshold': solution['threshold'],
        'score': solution['score'],
        'partition_details': {}
    }
    
    for partition_id, sensors in solution['sensors'].items():
        resilience_info = solution['resilience'].get(partition_id, {})
        summary['partition_details'][str(partition_id)] = {
            'sensor_count': len(sensors),
            'sensor_nodes': [s['node'] for s in sensors],
            'resilience_score': resilience_info.get('resilience_score', 0.0)
        }
    
    summary_file = os.path.join(output_dir, f'sensor_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {summary_file}")
    
    return {
        'sensor_file': sensor_file,
        'resilience_file': resilience_file if resilience_data else None,
        'summary_file': summary_file,
        'sensor_count': len(results_data)
    }


def generate_visualization(solution, wn, output_dir):
    """
    Generate visualization of sensor placement
    
    Args:
        solution: Output from optimize_sensor_placement
        wn: WNTR water network model
        output_dir: Output directory
    
    Returns:
        str: Path to saved visualization
    """
    import networkx as nx
    
    logger.info("Generating sensor placement visualization...")
    
    # Create graph from water network
    G_original = wn.to_graph()
    
    # Create a simple undirected graph without self-loops or parallel edges
    G = nx.Graph()
    
    # Add only junction nodes (exclude tanks, reservoirs which may cause self-loops)
    for node in G_original.nodes():
        if node in wn.junction_name_list:
            G.add_node(node)
    
    # Add edges only between junction nodes, excluding self-loops
    for u, v in G_original.edges():
        if u != v and u in G.nodes() and v in G.nodes():
            G.add_edge(u, v)
    
    # Get node positions directly from WNTR coordinates
    pos = {}
    use_spring_layout = False
    
    for node in G.nodes():
        try:
            coord = wn.get_node(node).coordinates
            if coord is None or coord == (0, 0):
                use_spring_layout = True
                break
            pos[node] = coord
        except:
            use_spring_layout = True
            break
    
    if use_spring_layout:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Draw network edges (only valid edges between different nodes)
    edges_to_draw = [(u, v) for u, v in G.edges() if u != v]
    nx.draw_networkx_edges(G, pos=pos, edgelist=edges_to_draw, 
                          alpha=0.3, width=0.5, edge_color='gray')
    
    # Draw all nodes
    all_nodes = list(G.nodes())
    nx.draw_networkx_nodes(G, pos=pos, nodelist=all_nodes,
                          node_color='lightblue', node_size=20, alpha=0.6)
    
    # Draw sensor nodes by partition
    colors = ['red', 'green', 'blue', 'orange', 'purple', 
             'brown', 'pink', 'olive', 'cyan', 'magenta']
    
    for partition_id, sensors in solution['sensors'].items():
        sensor_nodes = [s['node'] for s in sensors]
        sensor_nodes_in_graph = [n for n in sensor_nodes if n in G.nodes()]
        
        if sensor_nodes_in_graph:
            color = colors[partition_id % len(colors)]
            nx.draw_networkx_nodes(G, pos=pos, nodelist=sensor_nodes_in_graph,
                                  node_color=color, node_size=150, alpha=0.9,
                                  label=f'Partition {partition_id} ({len(sensor_nodes)} sensors)')
    
    # Add legend and title
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    total_sensors = sum(len(s) for s in solution['sensors'].values())
    avg_resilience = np.mean([r['resilience_score'] 
                             for r in solution['resilience'].values()])
    
    plt.title(f'Sensor Placement Results\n'
             f'Total Sensors: {total_sensors}, '
             f'Partitions: {len(solution["sensors"])}, '
             f'Threshold: {solution["threshold"]}, '
             f'Avg Resilience: {avg_resilience:.4f}',
             fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = os.path.join(output_dir, f'sensor_placement_viz_{timestamp}.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Visualization saved to: {viz_path}")
    return viz_path


def print_summary(solution):
    """Print a summary of the sensor placement results"""
    print("\n" + "=" * 70)
    print("SENSOR PLACEMENT OPTIMIZATION RESULTS")
    print("=" * 70)
    
    total_sensors = sum(len(s) for s in solution['sensors'].values())
    print(f"\n📊 Overview:")
    print(f"   Total Sensors: {total_sensors}")
    print(f"   Number of Partitions: {len(solution['sensors'])}")
    print(f"   Sensitivity Threshold: {solution['threshold']}")
    print(f"   Optimization Score: {solution['score']:.4f}")
    
    print(f"\n📈 Partition Details:")
    for partition_id, sensors in sorted(solution['sensors'].items()):
        resilience_info = solution['resilience'].get(partition_id, {})
        resilience_score = resilience_info.get('resilience_score', 0.0)
        full_coverage = resilience_info.get('full_coverage_rate', 0.0)
        
        sensor_nodes = [s['node'] for s in sensors]
        print(f"\n   Partition {partition_id}:")
        print(f"     Sensors ({len(sensors)}): {', '.join(sensor_nodes)}")
        print(f"     Full Coverage: {full_coverage*100:.1f}%")
        print(f"     Resilience Score: {resilience_score:.4f}")
    
    avg_resilience = np.mean([r['resilience_score'] 
                             for r in solution['resilience'].values()])
    print(f"\n🎯 Overall Metrics:")
    print(f"   Average Resilience Score: {avg_resilience:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Water Distribution Network Sensor Placement Optimization'
    )
    parser.add_argument(
        '--inp', '-i',
        type=str,
        default='dataset/Exa7.inp',
        help='Path to EPANET INP file (default: dataset/Exa7.inp)'
    )
    parser.add_argument(
        '--partitions', '-p',
        type=str,
        default='partition_results/partition_summary.json',
        help='Path to partition_summary.json (default: partition_results/partition_summary.json)'
    )
    parser.add_argument(
        '--num-partitions', '-n',
        type=int,
        default=None,
        help='Number of partitions to use (default: auto-select)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sensor_results',
        help='Output directory (default: sensor_results)'
    )
    parser.add_argument(
        '--demand-ratio', '-d',
        type=float,
        default=0.20,
        help='Demand perturbation ratio (default: 0.20)'
    )
    parser.add_argument(
        '--thresholds', '-t',
        type=float,
        nargs='+',
        default=[0.4, 0.5, 0.6],
        help='Sensitivity thresholds to try (default: 0.4 0.5 0.6)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Water Distribution Network Sensor Placement Optimization")
    print("=" * 70)
    
    # ==================== Step 1: Load Network ====================
    print("\n" + "=" * 60)
    print("Step 1: Loading water distribution network...")
    wn = wntr.network.WaterNetworkModel(args.inp)
    print(f"  Network loaded: {args.inp}")
    print(f"  Junctions: {len(wn.junction_name_list)}")
    print(f"  Pipes: {len(wn.pipe_name_list)}")
    print(f"  Tanks: {len(wn.tank_name_list)}")
    print(f"  Reservoirs: {len(wn.reservoir_name_list)}")
    
    # ==================== Step 2: Load Partitions ====================
    print("\n" + "=" * 60)
    print("Step 2: Loading partition results...")
    partitions = load_partition_results(args.partitions, args.num_partitions)
    
    # ==================== Step 3: Compute Sensitivity Matrix ====================
    print("\n" + "=" * 60)
    print("Step 3: Computing pressure sensitivity matrix...")
    print(f"  This may take a while for large networks...")
    sensitivity_data = compute_pressure_sensitivity_matrix(
        wn, partitions, args.demand_ratio
    )
    
    # ==================== Step 4: Optimize Sensor Placement ====================
    print("\n" + "=" * 60)
    print("Step 4: Optimizing sensor placement...")
    solution = optimize_sensor_placement(sensitivity_data, args.thresholds)
    
    if not solution:
        print("ERROR: Sensor placement optimization failed!")
        return
    
    # ==================== Step 5: Save Results ====================
    print("\n" + "=" * 60)
    print("Step 5: Saving results...")
    file_info = save_sensor_results(solution, wn, args.output)
    
    # ==================== Step 6: Generate Visualization ====================
    if not args.no_viz:
        print("\n" + "=" * 60)
        print("Step 6: Generating visualization...")
        viz_path = generate_visualization(solution, wn, args.output)
    
    # ==================== Step 7: Print Summary ====================
    print_summary(solution)
    
    print("\n" + "=" * 60)
    print("Sensor placement optimization completed successfully!")
    print(f"  Results saved in: {args.output}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
