"""
FCM (Fuzzy C-Means) partitioning module for water distribution networks.
Based on pressure sensitivity analysis and fuzzy clustering algorithm.
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import wntr
    import networkx as nx
    import skfuzzy as fuzz
    WNTR_AVAILABLE = True
    SKFUZZY_AVAILABLE = True
except ImportError as e:
    WNTR_AVAILABLE = False
    SKFUZZY_AVAILABLE = False


def setup_logger(name: str):
    """Set up logger for FCM partition module"""
    logger = logging.getLogger(f"fcm.{name}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger("FCMPartition")


# ==================== Default Parameters ====================

DEFAULT_PARAMS = {
    'k': 5,  # Number of partitions
    'm': 1.5,  # FCM fuzziness parameter
    'error': 1e-6,  # Convergence threshold
    'maxiter': 1000,  # Maximum iterations
    'perturb_rate': 0.1,  # Perturbation rate for sensitivity analysis
    'k_nearest': 10,  # KNN parameter for outlier handling
    'outliers_detection': True,  # Enable outlier detection
    'seed': 42  # Random seed
}


# ==================== Core Functions ====================

def normalize_matrix(S):
    """Normalize sensitivity matrix using standardization method"""
    S_mean = np.mean(S, axis=0)
    S_std = np.std(S, axis=0)
    epsilon = 1e-10
    S_std = np.where(S_std == 0, epsilon, S_std)
    S_normalized = (S - S_mean) / S_std
    
    S_min = np.min(S_normalized)
    S_max = np.max(S_normalized)
    S_n = (S_normalized - S_min) / (S_max - S_min)
    
    return S_n


def compute_sensitivity_matrix(wn, perturb_rate: float = 0.1):
    """
    Compute pressure sensitivity matrix based on demand perturbation.
    
    Args:
        wn: WNTR water network model
        perturb_rate: Demand perturbation rate (default 0.1)
    
    Returns:
        tuple: (node_list, demand_node_list, sensitivity_matrix)
    """
    logger.info(f"Computing sensitivity matrix (perturbation rate: {perturb_rate})")
    
    # Clean up base_value for all demand patterns
    # Some INP files may contain non-numeric base values, causing simulation errors
    sanitized_count = 0
    for name in wn.junction_name_list:
        node = wn.get_node(name)
        for ts in node.demand_timeseries_list:
            try:
                # Try converting to float
                if ts.base_value is None:
                    ts.base_value = 0.0
                else:
                    float(ts.base_value)
            except (TypeError, ValueError):
                # If conversion fails, default to 1.0 (conservative assumption)
                sanitized_count += 1
                try:
                    ts.base_value = 1.0
                except:
                    pass # Try as much as possible
                    
    if sanitized_count > 0:
        logger.warning(f"Sanitized {sanitized_count} non-numeric base_values to 1.0")

    # Run baseline simulation
    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()
    
    node_list = wn.node_name_list
    demand_nodes = wn.junction_name_list
    
    # Calculate total demand
    total_demand = 0
    for name in demand_nodes:
        node_demands = res.node['demand'].loc[:, name]
        total_demand += node_demands.sum()
    
    # Initialize sensitivity matrix
    n_nodes = len(demand_nodes)
    S = np.zeros((n_nodes, n_nodes))
    
    # Get baseline pressure
    base_p = res.node['pressure'].loc[:, demand_nodes].values
    delta = float(total_demand * perturb_rate / len(res.node['demand']))
    
    # Perturb each demand node
    for j, name in enumerate(demand_nodes):
        if (j + 1) % 50 == 0 or j == 0:
            logger.info(f"Processing node {j+1}/{n_nodes}: {name} ({(j+1)/n_nodes*100:.1f}%)")
        
        # Get demand timeseries
        ts_list = wn.get_node(name).demand_timeseries_list
        orig_values = []
        for d in ts_list:
            orig_values.append(d.base_value if d.base_value is not None else 0.0)
        
        # Apply perturbation
        for d, orig in zip(ts_list, orig_values):
            try:
                orig_float = float(orig) if orig is not None else 0.0
            except (TypeError, ValueError):
                orig_float = 0.0
            
            if orig_float > 0:
                d.base_value = float(orig_float * (1 + perturb_rate))
            else:
                d.base_value = delta
        
        try:
            # Run perturbed simulation
            sim = wntr.sim.EpanetSimulator(wn)
            res_pert = sim.run_sim()
            pert_p = res_pert.node['pressure'].loc[:, demand_nodes].values
            
            # Calculate pressure difference
            current_node_p_diff = np.abs(pert_p[:, j] - base_p[:, j])
            
            # Calculate sensitivity
            with np.errstate(divide='ignore', invalid='ignore'):
                S[:, j] = np.mean(np.where(current_node_p_diff[:, np.newaxis] != 0,
                                          np.abs(pert_p - base_p) / current_node_p_diff[:, np.newaxis],
                                          0), axis=0)
        except Exception as e:
            logger.warning(f"Simulation failed for node {name}: {e}")
            S[:, j] = 0
        
        # Restore original demand
        for d, orig in zip(ts_list, orig_values):
            try:
                d.base_value = float(orig) if orig is not None else 0.0
            except (TypeError, ValueError):
                d.base_value = 0.0
    
    logger.info(f"Sensitivity matrix computed: shape {S.shape}")
    return node_list, demand_nodes, S


def perform_fcm_clustering(S_normalized, params):
    """
    Perform FCM (Fuzzy C-Means) clustering.
    
    Args:
        S_normalized: Normalized sensitivity matrix
        params: Clustering parameters (k, m, error, maxiter, seed)
    
    Returns:
        tuple: (labels, clustering_info, error)
    """
    if not SKFUZZY_AVAILABLE:
        return None, None, {'error': 'scikit-fuzzy library not installed'}
    
    logger.info(f"Starting FCM clustering: k={params['k']}, m={params['m']}")
    
    np.random.seed(params['seed'])
    
    # Execute FCM clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=S_normalized.T,
        c=params['k'],
        m=params['m'],
        error=params['error'],
        maxiter=params['maxiter'],
        init=None,
        seed=params['seed']
    )
    
    # Get labels (starting from 1)
    labels = np.argmax(u, axis=0) + 1
    
    logger.info(f"FCM clustering complete: iterations={p}, FPC={fpc:.4f}")
    
    return labels, {
        'centers': cntr,
        'membership': u,
        'iterations': p,
        'fpc': fpc,
        'objective_function': jm
    }, None


def check_connectivity(node_connections, cluster_nodes):
    """Check connectivity of nodes using Warshall algorithm"""
    n = len(cluster_nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for i, node1 in enumerate(cluster_nodes):
        for j, node2 in enumerate(cluster_nodes):
            if i == j:
                adj_matrix[i, j] = 1
            else:
                mask1 = (node_connections[:, 0] == node1) & (node_connections[:, 1] == node2)
                mask2 = (node_connections[:, 0] == node2) & (node_connections[:, 1] == node1)
                if np.any(mask1) or np.any(mask2):
                    adj_matrix[i, j] = 1
    
    # Warshall algorithm for transitive closure
    for k in range(n):
        for i in range(n):
            for j in range(n):
                adj_matrix[i, j] = adj_matrix[i, j] or (adj_matrix[i, k] and adj_matrix[k, j])
    
    return adj_matrix


def find_connected_components(connect_matrix):
    """Find all connected components"""
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


def assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, labels, params):
    """Assign unassigned nodes to the nearest partition"""
    unassigned_indices = [i for i, label in enumerate(labels) if label == 0]
    
    if len(unassigned_indices) == 0:
        return labels
    
    logger.info(f"Assigning {len(unassigned_indices)} unassigned nodes using nearest neighbor")
    
    # Get node coordinates
    node_coords = {}
    layout = None
    
    for node_name in nodes:
        try:
            coord = wn.get_node(node_name).coordinates
            if coord is None or coord == (0, 0):
                if layout is None:
                    G = wn.to_graph().to_undirected()
                    layout = nx.spring_layout(G, seed=params['seed'])
                coord = layout.get(node_name, (0, 0))
        except:
            if layout is None:
                G = wn.to_graph().to_undirected()
                layout = nx.spring_layout(G, seed=params['seed'])
            coord = layout.get(node_name, (0, 0))
        node_coords[node_name] = coord
    
    # Build list of assigned nodes by partition
    assigned_nodes_by_partition = {}
    for i, demand_node in enumerate(demands):
        if labels[i] > 0:
            partition = labels[i]
            if partition not in assigned_nodes_by_partition:
                assigned_nodes_by_partition[partition] = []
            assigned_nodes_by_partition[partition].append((demand_node, node_coords[demand_node]))
    
    labels_copy = labels.copy()
    
    for unassigned_idx in unassigned_indices:
        unassigned_node = demands[unassigned_idx]
        unassigned_coord = node_coords[unassigned_node]
        
        min_distance = float('inf')
        nearest_partition = 1
        
        for partition, nodes_in_partition in assigned_nodes_by_partition.items():
            for assigned_node, assigned_coord in nodes_in_partition:
                distance = np.sqrt((unassigned_coord[0] - assigned_coord[0])**2 +
                                 (unassigned_coord[1] - assigned_coord[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_partition = partition
        
        labels_copy[unassigned_idx] = nearest_partition
        
        if nearest_partition not in assigned_nodes_by_partition:
            assigned_nodes_by_partition[nearest_partition] = []
        assigned_nodes_by_partition[nearest_partition].append((unassigned_node, unassigned_coord))
    
    return labels_copy


def remove_outliers_iteratively(wn, nodes, demands, raw_labels, params):
    """Iteratively detect and handle outliers"""
    if not params.get('outliers_detection', True):
        logger.info("Outlier detection disabled")
        return raw_labels
    
    logger.info("Starting iterative outlier detection")
    
    # Create full label array
    all_labels = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        if node in demands:
            idx = list(demands).index(node)
            all_labels[i] = raw_labels[idx]
    
    # Get node connections
    node_connections = []
    for link in wn.links():
        node1 = link[1].start_node_name
        node2 = link[1].end_node_name
        node_connections.append([list(nodes).index(node1), list(nodes).index(node2)])
    node_connections = np.array(node_connections)
    
    max_iterations = 10
    
    for iteration in range(max_iterations):
        zero_count = np.sum(all_labels == 0)
        if zero_count == 0:
            break
        
        logger.info(f"Outlier detection iteration {iteration + 1}, unassigned: {zero_count}")
        
        # Type 1 outliers: Based on neighbor label consistency
        for i, node in enumerate(nodes):
            if all_labels[i] != 99999:
                connected_nodes = []
                for conn in node_connections:
                    if conn[0] == i:
                        connected_nodes.append(conn[1])
                    elif conn[1] == i:
                        connected_nodes.append(conn[0])
                connected_nodes = np.array(connected_nodes)
                
                if len(connected_nodes) > 0:
                    neighbor_labels = np.unique(all_labels[connected_nodes])
                    label_counts = np.array([np.sum(all_labels[connected_nodes] == label) for label in neighbor_labels])
                    max_count = np.max(label_counts)
                    max_labels = neighbor_labels[label_counts == max_count]
                    
                    if 0 in max_labels and len(max_labels) > 1:
                        max_labels = max_labels[max_labels != 0]
                    
                    if len(max_labels) > 0:
                        all_labels[i] = max_labels[0]
                    else:
                        all_labels[i] = 0
        
        # Type 2 outliers: Based on spatial distance and connectivity
        for cluster in range(1, int(np.max(all_labels)) + 1):
            cluster_nodes = np.where(all_labels == cluster)[0]
            if len(cluster_nodes) <= 1:
                continue
            
            # Get coordinates
            coordinates = []
            elevations = []
            for node_idx in cluster_nodes:
                node = wn.get_node(nodes[node_idx])
                try:
                    coord = node.coordinates if node.coordinates else (0, 0)
                    elev = node.elevation if hasattr(node, 'elevation') else 0
                except:
                    coord = (0, 0)
                    elev = 0
                coordinates.append(coord)
                elevations.append(elev)
            
            features = np.column_stack([coordinates, elevations])
            
            # Calculate distance matrix
            dist_matrix = np.zeros((len(cluster_nodes), len(cluster_nodes)))
            for i in range(len(cluster_nodes)):
                for j in range(len(cluster_nodes)):
                    dist_matrix[i, j] = np.linalg.norm(features[i] - features[j])
            
            # Calculate KNN distance
            knn_distances = []
            for i in range(len(cluster_nodes)):
                distances = dist_matrix[i, :]
                distances = distances[distances > 0]
                k = min(params.get('k_nearest', 10), len(distances))
                if k > 0:
                    knn_dist = np.mean(np.sort(distances)[:k])
                    knn_distances.append(knn_dist)
                else:
                    knn_distances.append(0)
            
            knn_distances = np.array(knn_distances)
            
            if len(knn_distances) > 0:
                mean_dist = np.mean(knn_distances)
                std_dist = np.std(knn_distances)
                outliers = (knn_distances <= mean_dist - 3 * std_dist) | (knn_distances >= mean_dist + 3 * std_dist)
                all_labels[cluster_nodes[outliers]] = 0
            
            # Check connectivity
            connect_matrix = check_connectivity(node_connections, cluster_nodes)
            components = find_connected_components(connect_matrix)
            
            if len(components) > 1:
                main_component = max(components, key=len)
                outliers = np.setdiff1d(np.arange(len(cluster_nodes)), main_component)
                all_labels[cluster_nodes[outliers]] = 0
    
    # Update original labels (raw_labels)
    for i, node in enumerate(nodes):
        if node in demands:
            idx = list(demands).index(node)
            raw_labels[idx] = all_labels[i]
    
    # Handle remaining unassigned nodes
    unassigned_count = np.sum(raw_labels == 0)
    if unassigned_count > 0:
        logger.info(f"Assigning {unassigned_count} remaining unassigned nodes")
        raw_labels = assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, raw_labels, params)
    
    return raw_labels


def generate_fcm_visualization(wn, nodes, demands, labels, params, output_path):
    """Generate FCM partition visualization"""
    logger.info("Generating FCM partition visualization...")
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    G = wn.to_graph().to_undirected()
    # Convert to simple graph to avoid artifacts (loops) during multi-edge drawing
    G = nx.Graph(G)
    try:
        G.remove_edges_from(nx.selfloop_edges(G))
    except (TypeError, ValueError):
        pass
    
    # Prepare node positions
    pos = {}
    layout = None
    for n in G.nodes():
        try:
            coord = wn.get_node(n).coordinates
            if coord is None or coord == (0, 0):
                if layout is None:
                    layout = nx.spring_layout(G, seed=params['seed'])
                coord = layout.get(n, (0, 0))
        except:
            if layout is None:
                layout = nx.spring_layout(G, seed=params['seed'])
            coord = layout.get(n, (0, 0))
        pos[n] = coord
    
    # Create full label array
    all_labels = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        if node in demands:
            idx = list(demands).index(node)
            all_labels[i] = labels[idx]
    
    plt.figure(figsize=(14, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos=pos, alpha=0.4, width=0.8, edge_color='gray')
    
    # Draw nodes with partition colors
    scatter = nx.draw_networkx_nodes(
        G, pos=pos,
        nodelist=list(nodes),
        node_color=all_labels,
        cmap=plt.get_cmap("tab10", params['k'] + 1),
        vmin=0, vmax=params['k'],
        node_size=30
    )
    
    # Add legend
    legend_labels = ['Unassigned'] + [f'Partition {i}' for i in range(1, params['k'] + 1)]
    plt.legend(scatter.legend_elements()[0], legend_labels,
              title="Partition",
              loc='upper right',
              bbox_to_anchor=(1, 1))
    
    plt.title(f"FCM Network Partitioning (K={params['k']}, Fuzziness={params['m']})")
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")
    return output_path


def save_fcm_results(nodes, demands, labels, params, clustering_info, output_dir):
    """Save FCM partition results to file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare node assignment information
    node_assignments = {}
    for i, node_id in enumerate(demands):
        node_assignments[node_id] = int(labels[i])
    
    # Build summary
    partition_summary = {
        str(params['k']): {
            'node_assignments': node_assignments,
            'algorithm': 'FCM',
            'parameters': {
                'k': params['k'],
                'm': params['m'],
                'perturb_rate': params['perturb_rate'],
                'outliers_detection': params['outliers_detection']
            },
            'metrics': {
                'fpc': float(clustering_info['fpc']),
                'iterations': int(clustering_info['iterations'])
            }
        }
    }
    
    # Save summary JSON
    summary_file = os.path.join(output_dir, 'fcm_partition_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(partition_summary, f, indent=2, ensure_ascii=False)
    
    # Calculate partition statistics
    partition_stats = {}
    for i in range(1, params['k'] + 1):
        count = int(np.sum(labels == i))
        partition_stats[f'Partition_{i}'] = count
    
    unassigned_count = int(np.sum(labels == 0))
    if unassigned_count > 0:
        partition_stats['Unassigned'] = unassigned_count
    
    logger.info(f"FCM results saved to: {summary_file}")
    
    return {
        'summary_file': summary_file,
        'partition_summary': partition_summary,
        'partition_stats': partition_stats,
        'timestamp': timestamp
    }


# ==================== Agent Main Interface ====================

def run_fcm_partitioning_for_agent(inp_file: str, num_partitions: int = 5, fuzziness: float = 1.5) -> dict:
    """Entry point for FCM partitioning agent integration."""
    if not WNTR_AVAILABLE:
        return {"status": "error", "error": "WNTR library not installed"}
    
    if not SKFUZZY_AVAILABLE:
        return {"status": "error", "error": "scikit-fuzzy library not installed. Install with: pip install scikit-fuzzy"}
    
    output_dir = 'partition_results'
    
    try:
        # Set parameters
        params = DEFAULT_PARAMS.copy()
        params['k'] = num_partitions
        params['m'] = fuzziness
        
        # Load network
        logger.info(f"Loading network: {inp_file}")
        wn = wntr.network.WaterNetworkModel(inp_file)
        
        # Compute sensitivity matrix
        logger.info("Computing sensitivity matrix...")
        nodes, demands, S = compute_sensitivity_matrix(wn, params['perturb_rate'])
        
        # Normalize matrix
        S_normalized = normalize_matrix(S)
        
        # Perform FCM clustering
        logger.info("Performing FCM clustering...")
        labels, clustering_info, error = perform_fcm_clustering(S_normalized, params)
        
        if error:
            return {"status": "error", "error": error['error']}
        
        # Handle outliers
        logger.info("Processing outliers...")
        wn_fresh = wntr.network.WaterNetworkModel(inp_file)
        final_labels = remove_outliers_iteratively(wn_fresh, nodes, demands, labels, params)
        
        # Save results
        logger.info("Saving results...")
        save_result = save_fcm_results(nodes, demands, final_labels, params, clustering_info, output_dir)
        
        # Generate visualization
        timestamp = save_result['timestamp']
        viz_path = os.path.join(output_dir, f'fcm_partition_viz_{timestamp}.png')
        wn_viz = wntr.network.WaterNetworkModel(inp_file)
        generate_fcm_visualization(wn_viz, nodes, demands, final_labels, params, viz_path)
        
        # Build summary message
        total_nodes = len(demands)
        unassigned = int(np.sum(final_labels == 0))
        
        return {
            "status": "success",
            "msg": f"FCM partitioning completed! Divided {total_nodes} nodes into {num_partitions} partitions. "
                   f"FPC: {clustering_info['fpc']:.4f}, Iterations: {clustering_info['iterations']}",
            "summary_json": save_result['summary_file'],
            "viz_file": viz_path,
            "partition_stats": save_result['partition_stats'],
            "metrics": {
                "fpc": clustering_info['fpc'],
                "iterations": clustering_info['iterations']
            },
            "num_partitions": num_partitions,
            "fuzziness": fuzziness
        }
        
    except Exception as e:
        logger.error(f"FCM partitioning failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
