import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import wntr
import hashlib
from datetime import datetime
from itertools import combinations

# Set up logger
logger = logging.getLogger("SensorManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_partition_summary_path(inp_file):
    """
    Determine partition summary file path.
    Prioritize recently modified results (FCM or Louvain).
    """
    candidates = []

    # 1. Check FCM results (in current directory partition_results)
    fcm_path = os.path.join("partition_results", "fcm_partition_summary.json")
    if os.path.exists(fcm_path):
        candidates.append((fcm_path, "FCM", os.path.getmtime(fcm_path)))
    
    # 2. Check Louvain results (in static/partition_results/{md5})
    if os.path.exists(inp_file):
        md5 = compute_md5(inp_file)
        louvain_path = os.path.join("static", "partition_results", md5, "partition_summary.json")
        if os.path.exists(louvain_path):
            candidates.append((louvain_path, "Louvain", os.path.getmtime(louvain_path)))
            
    # Sort by modification time descending (latest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    if candidates:
        fastest = candidates[0]
        logger.info(f"Selected partition set: {fastest[1]} (modified {datetime.fromtimestamp(fastest[2])})")
        return fastest[0], fastest[1]

    # 3. Check generic default (legacy fallback)
    generic_path = os.path.join("partition_results", "partition_summary.json")
    if os.path.exists(generic_path):
        return generic_path, "Legacy"

    return None, None

def load_partition_results(partition_summary_path, num_partitions=None):
    logger.info(f"Loading partition results from: {partition_summary_path}")
    
    with open(partition_summary_path, 'r') as f:
        summary = json.load(f)
    
    available_partitions = sorted([int(k) for k in summary.keys()])
    
    if num_partitions is None:
        # Default strategy: Select 5, or the first available partition > 1
        for n in [5, 4, 3, 2] + available_partitions:
            if n in available_partitions:
                num_partitions = n
                break
    
    if num_partitions not in available_partitions:
        # Fallback to nearest value
        if available_partitions:
            num_partitions = available_partitions[0]
        else:
            raise ValueError("No partitions found in summary file.")
            
    partition_data = summary[str(num_partitions)]
    # Handle different formats (FCM vs Louvain)
    if 'node_assignments' in partition_data:
        node_assignments = partition_data['node_assignments']
    elif 'node_to_community' in partition_data:
        node_assignments = partition_data['node_to_community']
    else:
        # Try to infer
        keys = list(partition_data.keys())
        # If keys look like nodes and values look like integers
        if keys and isinstance(partition_data[keys[0]], int):
             node_assignments = partition_data
        else:
             raise ValueError("Could not find node assignments in partition data.")

    partitions = {}
    for node_id, partition_id in node_assignments.items():
        if partition_id not in partitions:
            partitions[partition_id] = []
        partitions[partition_id].append(node_id)
        
    return partitions, num_partitions

def compute_pressure_sensitivity_matrix(wn, partitions, demand_ratio=0.20):
    # (Adapted from wds_sensor_main.py)
    all_nodes = []
    for nodes in partitions.values():
        all_nodes.extend(nodes)
    
    valid_nodes = [n for n in all_nodes if n in wn.junction_name_list]
    if not valid_nodes:
        raise ValueError("No valid junction nodes found in partitions")
        
    sim = wntr.sim.EpanetSimulator(wn)
    base_results = sim.run_sim()
    base_pressure = base_results.node['pressure'].loc[:, valid_nodes].values
    
    # Calculate delta
    total_demand = 0
    for name in valid_nodes:
        total_demand += base_results.node['demand'].loc[:, name].sum()
    
    num_timesteps = len(base_results.node['demand'])
    delta = float(total_demand * demand_ratio / num_timesteps) if num_timesteps > 0 else 0
    
    n_nodes = len(valid_nodes)
    sensitivity_matrix = np.zeros((n_nodes, n_nodes))
    
    for i, perturb_node in enumerate(valid_nodes):
        node = wn.get_node(perturb_node)
        original_demands = {}
        for j, ts in enumerate(node.demand_timeseries_list):
            original_demands[j] = ts.base_value
            
        # Perturbation
        for j, ts in enumerate(node.demand_timeseries_list):
            orig = original_demands[j]
            try:
                orig_val = float(orig) if orig is not None else 0.0
            except:
                orig_val = 0.0
            
            if orig_val > 0:
                ts.base_value = orig_val * (1 + demand_ratio)
            else:
                ts.base_value = delta
                
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            perturb_results = sim.run_sim()
            perturb_pressure = perturb_results.node['pressure'].loc[:, valid_nodes].values
            
            pressure_diff = np.abs(perturb_pressure - base_pressure)
            avg_pressure_diff = np.mean(pressure_diff, axis=0)
            
            max_diff = np.max(avg_pressure_diff)
            if max_diff > 0:
                sensitivity_matrix[i, :] = avg_pressure_diff / max_diff
        except Exception as e:
            logger.warning(f"Simulation failed for node {perturb_node}: {e}")
            
        # Restore
        for j, ts in enumerate(node.demand_timeseries_list):
             try:
                ts.base_value = float(original_demands[j]) if original_demands[j] is not None else 0.0
             except:
                ts.base_value = 0.0
                
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

def _calculate_detection(sensor_indices, target_indices, sensitivity_matrix, threshold):
    if not sensor_indices or not target_indices:
        return 0, 0.0
    
    detected_count = 0
    for target_idx in target_indices:
        for sensor_idx in sensor_indices:
            if sensitivity_matrix[sensor_idx, target_idx] > threshold:
                detected_count += 1
                break
    return detected_count, detected_count / len(target_indices)

def select_sensors_by_partition(sensitivity_data, threshold=0.5, min_sensor_ratio=0.04, max_sensor_ratio=0.15, target_coverage=0.95):
    sensitivity_matrix = sensitivity_data['matrix']
    all_nodes = sensitivity_data['nodes']
    partitions = sensitivity_data['partitions']
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    selected_sensors = {}
    
    for partition_id, partition_nodes in sorted(partitions.items()):
        partition_indices = [node_to_index[n] for n in partition_nodes if n in node_to_index]
        if len(partition_indices) < 2:
            continue
            
        # Influence score
        influence_scores = {}
        for node_name in partition_nodes:
             if node_name not in node_to_index: continue
             node_idx = node_to_index[node_name]
             partition_sensitivities = sensitivity_matrix[node_idx, partition_indices]
             influence_scores[node_name] = {
                 'index': node_idx,
                 'detectable_count': np.sum(partition_sensitivities > threshold),
                 'avg_sensitivity': np.mean(partition_sensitivities)
             }
             
        partition_size = len(partition_nodes)
        min_sensors = max(2, int(partition_size * min_sensor_ratio))
        max_sensors = min(10, max(3, int(partition_size * max_sensor_ratio)))
        
        selected_sensors[partition_id] = []
        uncovered_indices = set(partition_indices)
        
        # Phase 1: Coverage
        while (uncovered_indices and len(selected_sensors[partition_id]) < max_sensors and 
               len(uncovered_indices) / len(partition_indices) > (1 - target_coverage)):
            
            best_sensor = None
            best_coverage = 0
            
            for node_name, info in influence_scores.items():
                if node_name in [s['node'] for s in selected_sensors[partition_id]]: continue
                
                node_idx = info['index']
                partition_sensitivities = sensitivity_matrix[node_idx, list(uncovered_indices)]
                coverage = np.sum(partition_sensitivities > threshold)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_sensor = {
                        'node': node_name, 'index': node_idx, 'coverage': coverage,
                        'influence_score': info['detectable_count'], 'avg_sensitivity': info['avg_sensitivity']
                    }
                    
            if best_sensor is None or best_coverage == 0: break
            selected_sensors[partition_id].append(best_sensor)
            
            sensor_idx = best_sensor['index']
            covered = [idx for idx in uncovered_indices if sensitivity_matrix[sensor_idx, idx] > threshold]
            for idx in covered: uncovered_indices.discard(idx)
            
        # Phase 2: Diversity/Minimum count
        while len(selected_sensors[partition_id]) < min_sensors:
            best_sensor = None
            max_diversity = 0
            
            for node_name, info in influence_scores.items():
                if node_name in [s['node'] for s in selected_sensors[partition_id]]: continue
                
                node_idx = info['index']
                diversity_score = 0
                for existing in selected_sensors[partition_id]:
                    diversity_score += (1 - sensitivity_matrix[node_idx, existing['index']])
                if len(selected_sensors[partition_id]) > 0:
                    diversity_score /= len(selected_sensors[partition_id])
                
                if diversity_score > max_diversity:
                    max_diversity = diversity_score
                    best_sensor = {
                        'node': node_name, 'index': node_idx, 'coverage': 0,
                        'influence_score': info['detectable_count'], 'avg_sensitivity': info['avg_sensitivity']
                    }
            
            if best_sensor: selected_sensors[partition_id].append(best_sensor)
            else: break
            
    return selected_sensors

def evaluate_resilience(selected_sensors, sensitivity_data, threshold=0.5):
    resilience_results = {}
    sensitivity_matrix = sensitivity_data['matrix']
    node_to_index = {node: i for i, node in enumerate(sensitivity_data['nodes'])}
    
    for partition_id, sensors in selected_sensors.items():
        partition_nodes = sensitivity_data['partitions'][partition_id]
        partition_indices = [node_to_index[n] for n in partition_nodes if n in node_to_index]
        
        all_sensor_indices = [s['index'] for s in sensors]
        full_detected, full_coverage = _calculate_detection(all_sensor_indices, partition_indices, sensitivity_matrix, threshold)
        
        total_failure_coverage = 0.0
        failure_scenario_count = 0
        
        for failure_count in range(1, len(sensors)):
            for failed_indices in combinations(range(len(sensors)), failure_count):
                remaining_indices = [sensors[i]['index'] for i in range(len(sensors)) if i not in failed_indices]
                _, coverage = _calculate_detection(remaining_indices, partition_indices, sensitivity_matrix, threshold)
                total_failure_coverage += coverage
                failure_scenario_count += 1
                
        avg_failure_resilience = (total_failure_coverage / failure_scenario_count if failure_scenario_count > 0 else 0.0)
        resilience_results[partition_id] = {
            'resilience_score': avg_failure_resilience,
            'full_coverage_rate': full_coverage
        }
    return resilience_results

def optimize_sensor_placement(sensitivity_data, thresholds=[0.4, 0.5, 0.6]):
    best_solution = None
    best_score = 0.0
    
    for threshold in thresholds:
        selected_sensors = select_sensors_by_partition(sensitivity_data, threshold)
        if not selected_sensors: continue
        
        resilience_results = evaluate_resilience(selected_sensors, sensitivity_data, threshold)
        if not resilience_results: continue
        
        total_sensors = sum(len(s) for s in selected_sensors.values())
        partition_count = len(selected_sensors)
        
        resilience_scores = [r['resilience_score'] for r in resilience_results.values()]
        avg_resilience = np.mean(resilience_scores) if resilience_scores else 0.0
        coverage_score = min(1.0, total_sensors / (partition_count * 2))
        
        total_score = max(0.001, avg_resilience * 0.4 + coverage_score * 0.6 - (total_sensors * 0.001))
        
        if total_score > best_score:
            best_score = total_score
            best_solution = {
                'sensors': selected_sensors,
                'resilience': resilience_results,
                'threshold': threshold,
                'score': total_score
            }
    return best_solution

def save_and_visualize(solution, wn, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save CSV
    results_data = []
    sensor_id = 1
    for partition_id, sensors in solution['sensors'].items():
        for sensor in sensors:
            coord = wn.get_node(sensor['node']).coordinates
            results_data.append({
                'Sensor_ID': f'S{sensor_id:03d}',
                'Node_Name': sensor['node'],
                'Partition_ID': partition_id,
                'X': coord[0], 'Y': coord[1],
                'Resilience': solution['resilience'].get(partition_id, {}).get('resilience_score', 0)
            })
            sensor_id += 1
    
    csv_file = os.path.join(output_dir, f'sensor_placement_{timestamp}.csv')
    pd.DataFrame(results_data).to_csv(csv_file, index=False)
    
    # 2. Visualize
    plt.figure(figsize=(14, 10))
    G = nx.Graph() # Simple graph
    G_orig = wn.to_graph()
    for u,v in G_orig.edges():
        if u!=v: G.add_edge(u,v)
    for n in wn.junction_name_list:
        if n not in G: G.add_node(n)
        
    pos = {}
    for n in G.nodes():
        try: pos[n] = wn.get_node(n).coordinates
        except: pass
    if not pos: pos = nx.spring_layout(G)
    
    nx.draw_networkx_edges(G, pos=pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(G, pos=pos, node_size=10, node_color='lightgray', alpha=0.5)
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']
    for pid, sensors in solution['sensors'].items():
        snodes = [s['node'] for s in sensors if s['node'] in G]
        col = colors[int(pid) % len(colors)] if str(pid).isdigit() else 'r'
        nx.draw_networkx_nodes(G, pos=pos, nodelist=snodes, node_color=col, node_size=100, label=f'Part {pid}')
        
    plt.legend()
    plt.title(f"Sensor Placement (Score: {solution['score']:.3f})")
    plt.axis('off')
    
    viz_file = os.path.join(output_dir, f'sensor_placement_viz_{timestamp}.png')
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return csv_file, viz_file

def run_sensor_placement_for_agent(inp_file, num_partitions=None):
    try:
        # 1. Load partitions
        summary_path, source_type = get_partition_summary_path(inp_file)
        if not summary_path:
            return {"status": "error", "error": "Partition results not found. Please run partitioning first."}
        
        partitions, used_num_partitions = load_partition_results(summary_path, num_partitions)
        logger.info(f"Loaded {len(partitions)} partitions from {source_type} ({summary_path})")
        
        # 2. Load network
        wn = wntr.network.WaterNetworkModel(inp_file)
        
        # 3. Calculate sensitivity
        sensitivity_data = compute_pressure_sensitivity_matrix(wn, partitions)
        
        # 4. Optimize
        solution = optimize_sensor_placement(sensitivity_data)
        if not solution:
            return {"status": "error", "error": "Optimization failed to find a valid solution."}
            
        # 5. Save and visualize
        csv_file, viz_file = save_and_visualize(solution, wn, "sensor_results")
        
        # 6. Format statistics
        partition_details = {}
        for pid, sensors in solution['sensors'].items():
            partition_details[str(pid)] = {
                'count': len(sensors),
                'resilience': solution['resilience'][pid]['resilience_score'],
                'sensor_nodes': [s['node'] for s in sensors],
                'full_coverage_rate': solution['resilience'][pid]['full_coverage_rate']
            }
            
        return {
            "status": "success",
            "msg": f"Sensor placement optimization complete. Found {sum(len(s) for s in solution['sensors'].values())} optimal locations.",
            "source_type": source_type,
            "used_num_partitions": used_num_partitions,
            "summary": {
                "total_sensors": sum(len(s) for s in solution['sensors'].values()),
                "num_partitions": len(partition_details),
                "threshold": solution['threshold'],
                "score": solution['score'],
                "partition_details": partition_details
            },
            "avg_resilience": solution['score'], # Approximate value
            "sensor_file": csv_file,  # Match Agent key name
            "viz_file": viz_file,
        }
        
    except Exception as e:
        logger.error(f"Sensor placement error: {e}")
        return {"status": "error", "error": str(e)}
