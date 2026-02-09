"""
PartitionSim Network Partition Agent
Responsible for processing .inp files, performing network FCM clustering partition and outlier detection
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .base_agent import BaseAgent
from .intent_classifier_fast import FastIntentClassifier as IntentClassifier

try:
    import wntr
    import skfuzzy as fuzz
    import networkx as nx
    WNTR_AVAILABLE = True
    SKFUZZY_AVAILABLE = True
except ImportError as e:
    WNTR_AVAILABLE = False
    SKFUZZY_AVAILABLE = False

class PartitionSim(BaseAgent):
    """Network partition agent"""
    
    def __init__(self):
        super().__init__("PartitionSim")

        if not WNTR_AVAILABLE:
            self.log_error("WNTR library not installed, network analysis function unavailable")
        if not SKFUZZY_AVAILABLE:
            self.log_error("scikit-fuzzy library not installed, FCM clustering function unavailable")

        self.intent_classifier = IntentClassifier()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)

        # Cache mechanism: avoid repeated sensitivity matrix calculation
        self._sensitivity_cache = {}  # {file_path: {matrix, last_modified}}
        
        # Default parameters
        self.default_params = {
            'k': 5,  # Default partition count
            'm': 1.5,  # FCM fuzziness parameter
            'error': 1e-6,  # Convergence threshold
            'maxiter': 1000,  # Max iteration count
            'perturb_rate': 0.1,  # Perturbation rate
            'k_nearest': 10,  # KNN parameter
            'outliers_detection': True,  # Whether to perform outlier detection
            'seed': 42  # Random seed
        }
    
    def parse_user_intent(self, user_message: str):
        """Parse user intent and parameters"""
        intent_result = self.intent_classifier.classify_intent(user_message)

        # Extract partition-related parameters
        params = self.default_params.copy()

        # Extract partition count
        k_patterns = [
            r'partition\s+into\s+(\d+)\s+regions?',
            r'partition\s+into\s+(\d+)\s+areas?',
            r'(\d+)\s+regions?',
            r'(\d+)\s+partitions?',
            r'k\s*=\s*(\d+)',
            r'partition\s*count\s*[: :]\s*(\d+)',
            r'number\s+of\s+partitions\s*[: :]\s*(\d+)',
            r'partition\s+number\s*[: :]\s*(\d+)'
        ]

        for pattern in k_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['k'] = int(match.group(1))
                self.logger.info(f"[PartitionSim] Parsed partition count: {params['k']} (Matched pattern: {pattern})")
                break
        
        # Extract FCM parameters
        m_patterns = [
            r'm\s*=\s*([\d.]+)',
            r'fuzziness\s*[: :=]\s*([\d.]+)',
            r'fuzzy\s*parameter\s*[: :=]\s*([\d.]+)',
            r'fuzziness\s*=\s*([\d.]+)'
        ]
        
        for pattern in m_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['m'] = float(match.group(1))
                self.logger.info(f"[PartitionSim] Parsed fuzziness parameter: {params['m']} (Matched pattern: {pattern})")
                break
        
        # Extract perturbation rate
        perturb_patterns = [
            r'perturbation\s*rate\s*[: :]\s*([\d.]+)',
            r'perturbation\s*rate([\d.]+)',
            r'perturb[_\s]*rate\s*[: :=]\s*([\d.]+)'
        ]
        
        for pattern in perturb_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                params['perturb_rate'] = float(match.group(1))
                self.logger.info(f"[PartitionSim] Parsed perturbation rate: {params['perturb_rate']} (Matched pattern: {pattern})")
                break
        
        # Detect if outlier processing is needed
        outlier_disable_keywords = [
            'no outlier detection', 'disable outlier detection', 'skip outlier',
            'do not detect outliers', 'no outliers', 'disable outliers',
            'ignore outliers', 'skip outlier detection',
            'no outlier', 'skip outlier', 'disable outlier'
        ]

        outlier_enable_keywords = [
            'detect outliers', 'enable outlier detection', 'process outliers',
            'perform outlier detection', 'check for outliers', 'remove outliers',
            'outlier detection', 'outlier removal'
        ]

        if any(keyword in user_message for keyword in outlier_disable_keywords):
            params['outliers_detection'] = False
            self.logger.info(f"[PartitionSim] Parsed disable outlier detection")
        elif any(keyword in user_message for keyword in outlier_enable_keywords):
            params['outliers_detection'] = True
            self.logger.info(f"[PartitionSim] Parsed enable outlier detection")
        
        return {
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'params': params
        }
    
    def parse_network(self, inp_file_path: str):
        """Parse network file, extract basic information"""
        if not WNTR_AVAILABLE:
            return {'error': 'WNTR library not installed'}

        try:
            # Check cache
            if inp_file_path in getattr(self, '_network_cache', {}):
                file_mtime = os.path.getmtime(inp_file_path)
                cached_data = self._network_cache[inp_file_path]
                if cached_data['last_modified'] == file_mtime:
                    self.log_info(f"Using cached network info: {inp_file_path}")
                    return cached_data['network_info']

            self.log_info(f"Starting to parse network file: {inp_file_path}")

            # Read network file
            wn = wntr.network.WaterNetworkModel(inp_file_path)

            # Extract key information
            network_info = {
                'nodes': {
                    'junctions': len(wn.junction_name_list),
                    'reservoirs': len(wn.reservoir_name_list),
                    'tanks': len(wn.tank_name_list),
                    'total': len(wn.node_name_list)
                },
                'links': {
                    'pipes': len(wn.pipe_name_list),
                    'pumps': len(wn.pump_name_list),
                    'valves': len(wn.valve_name_list),
                    'total': len(wn.link_name_list)
                },
                'network_stats': {
                    'total_length': float(sum([wn.get_link(pipe).length for pipe in wn.pipe_name_list])) if len(wn.pipe_name_list) > 0 else 0,
                    'simulation_duration': wn.options.time.duration,
                    'hydraulic_timestep': wn.options.time.hydraulic_timestep,
                    'pattern_timestep': wn.options.time.pattern_timestep
                }
            }

            self.log_info(f"Network parsing complete: {network_info['nodes']['total']} nodes, {network_info['links']['total']} links")

            # Initialize cache
            if not hasattr(self, '_network_cache'):
                self._network_cache = {}

            # Update cache
            file_mtime = os.path.getmtime(inp_file_path)
            self._network_cache[inp_file_path] = {
                'network_info': network_info,
                'last_modified': file_mtime
            }

            return network_info

        except Exception as e:
            error_msg = f"Parsing network file failed: {e}"
            self.log_error(error_msg)
            return {'error': error_msg}

    def load_network(self, inp_file_path: str):
        """Load water network model"""
        if not WNTR_AVAILABLE:
            return None, {'error': 'WNTR library not installed'}

        try:
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            self.log_info(f"Network loaded: nodes={len(wn.node_name_list)}, "
                         f"junctions={len(wn.junction_name_list)}, "
                         f"links={len(wn.link_name_list)}")
            return wn, None
        except Exception as e:
            error_msg = f"Failed to load network file: {str(e)}"
            self.log_error(error_msg)
            return None, {'error': error_msg}
    
    def normalize_matrix(self, S):
        """Standardize and normalize sensitivity matrix"""
        # Standardization: subtract mean, divide by standard deviation
        S_mean = np.mean(S, axis=0)
        S_std = np.std(S, axis=0)
        # Add small threshold to avoid division by zero
        epsilon = 1e-10
        S_std = np.where(S_std == 0, epsilon, S_std)
        S_std = (S - S_mean) / S_std
        
        # Normalize: map values to [0,1] interval
        S_min = np.min(S_std)
        S_max = np.max(S_std)
        S_n = (S_std - S_min) / (S_max - S_min)
        
        return S_n
    
    def compute_sensitivity_matrix(self, inp_file_path: str, perturb_rate: float):
        """Calculate sensitivity matrix"""
        if not WNTR_AVAILABLE:
            return None, None, {'error': 'WNTR library not installed'}
        
        try:
            # Check cache
            cache_key = f"{inp_file_path}_{perturb_rate}"
            if cache_key in self._sensitivity_cache:
                file_mtime = os.path.getmtime(inp_file_path)
                cached_data = self._sensitivity_cache[cache_key]
                if cached_data['last_modified'] == file_mtime:
                    self.log_info(f"Using cached sensitivity matrix")
                    return cached_data['nodes'], cached_data['demands'], cached_data['matrix']
            
            self.log_info(f"Starting sensitivity matrix calculation, perturbation rate: {perturb_rate}")
            
            # Load baseline network model
            wn0, error = self.load_network(inp_file_path)
            if error:
                return None, None, error
            
            # Run baseline simulation to get pressure
            sim = wntr.sim.EpanetSimulator(wn0)
            res = sim.run_sim()
            
            # Get all nodes and demand nodes list
            node_list = wn0.node_name_list
            demand_nodes = wn0.junction_name_list

            # Calculate total actual demand
            total_demand = 0
            for name in demand_nodes:
                node_demands = res.node['demand'].loc[:, name]
                total_demand += node_demands.sum()

            # Initialize sensitivity matrix
            S = np.zeros((len(demand_nodes), len(demand_nodes)))
            # Reload network for perturbation simulation
            wn, _ = self.load_network(inp_file_path)
            
            # Get baseline pressure
            base_p = res.node['pressure'].loc[:, demand_nodes].values
            # Calculate average perturbation amount
            delta = total_demand * perturb_rate / len(res.node['demand'])

            # Perturb each demand node
            for j, name in enumerate(demand_nodes):
                self.log_info(f"Processing node {j+1}/{len(demand_nodes)}: {name}")
                
                # Get demand time series for this node
                ts_list = wn.get_node(name).demand_timeseries_list
                # Save original demand
                orig_values = [d.base_value for d in ts_list]
                
                # Perturb each time series
                for d in ts_list:
                    if d.base_value > 0:
                        d.base_value = d.base_value + d.base_value * perturb_rate
                    else:
                        d.base_value = d.base_value + delta
                
                # Run perturbed simulation
                sim = wntr.sim.EpanetSimulator(wn)
                res_pert = sim.run_sim()
                
                # Get perturbed pressure
                pert_p = res_pert.node['pressure'].loc[:, demand_nodes].values
                
                # Calculate pressure difference for current perturbed node
                current_node_p_diff = np.abs(pert_p[:, j] - base_p[:, j])
                
                # Calculate sensitivity
                with np.errstate(divide='ignore', invalid='ignore'):
                    S[:, j] = np.mean(np.where(current_node_p_diff[:, np.newaxis] != 0,
                                              np.abs(pert_p - base_p) / current_node_p_diff[:, np.newaxis],
                                              0), axis=0)
                
                # Restore original demand
                for d, orig in zip(ts_list, orig_values):
                    d.base_value = orig
            
            # Cache results
            self._sensitivity_cache[cache_key] = {
                'nodes': node_list,
                'demands': demand_nodes,
                'matrix': S,
                'last_modified': os.path.getmtime(inp_file_path)
            }
            
            self.log_info(f"Sensitivity matrix calculation complete, matrix size: {S.shape}")
            return node_list, demand_nodes, S
            
        except Exception as e:
            error_msg = f"Calculate sensitivity matrix failed: {str(e)}"
            self.log_error(error_msg)
            return None, None, {'error': error_msg}

    def perform_fcm_clustering(self, S_normalized, params):
        """Execute FCM clustering"""
        if not SKFUZZY_AVAILABLE:
            return None, None, {'error': 'scikit-fuzzy library not installed'}

        try:
            self.log_info(f"Starting FCM clustering, parameters: k={params['k']}, m={params['m']}")

            # Set random seed
            np.random.seed(params['seed'])

            # Execute FCM clustering
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data=S_normalized.T,      # Input data matrix, needs transpose
                c=params['k'],            # Number of clusters
                m=params['m'],            # Fuzziness parameter
                error=params['error'],    # Convergence threshold
                maxiter=params['maxiter'], # Max iteration count
                init=None,                # Initial cluster centers
                seed=params['seed']       # Random seed
            )

            # Get initial labels (starting from 1)
            raw_labels = np.argmax(u, axis=0) + 1

            self.log_info(f"FCM clustering complete, convergence iterations: {p}, fuzzy partition coefficient: {fpc:.4f}")

            return raw_labels, {
                'centers': cntr,
                'membership': u,
                'iterations': p,
                'fpc': fpc,
                'objective_function': jm
            }, None

        except Exception as e:
            error_msg = f"FCM clustering failed: {str(e)}"
            self.log_error(error_msg)
            return None, None, {'error': error_msg}

    def check_connectivity(self, node_connections, cluster_nodes):
        """Use Warshall algorithm to check node connectivity"""
        n = len(cluster_nodes)
        adj_matrix = np.zeros((n, n), dtype=int)

        # Fill adjacency matrix
        for i, node1 in enumerate(cluster_nodes):
            for j, node2 in enumerate(cluster_nodes):
                if i == j:
                    adj_matrix[i, j] = 1
                else:
                    # Check if two nodes are directly connected
                    mask1 = (node_connections[:, 0] == node1) & (node_connections[:, 1] == node2)
                    mask2 = (node_connections[:, 0] == node2) & (node_connections[:, 1] == node1)
                    if np.any(mask1) or np.any(mask2):
                        adj_matrix[i, j] = 1

        # Use Warshall algorithm to calculate transitive closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    adj_matrix[i, j] = adj_matrix[i, j] or (adj_matrix[i, k] and adj_matrix[k, j])

        return adj_matrix

    def find_connected_components(self, connect_matrix):
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

    def assign_unassigned_nodes_by_nearest_neighbor(self, wn, nodes, demands, labels, params):
        """Assign unassigned nodes to nearest neighbor partition"""

        # Find unassigned demand nodes
        unassigned_indices = []
        for i, demand_node in enumerate(demands):
            if labels[i] == 0:
                unassigned_indices.append(i)

        if len(unassigned_indices) == 0:
            return labels

        self.log_info(f"Starting to assign {len(unassigned_indices)} unassigned demand nodes to nearest neighbor partition")

        # Get node coordinates
        node_coords = {}
        layout = None  # For nodes without coordinates

        for node_name in nodes:
            try:
                coord = wn.get_node(node_name).coordinates
                if coord is None or coord == (0, 0):
                    # If no coordinates, use network layout
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

        # Create partition information for assigned nodes
        assigned_nodes_by_partition = {}
        for i, demand_node in enumerate(demands):
            if labels[i] > 0:
                partition = labels[i]
                if partition not in assigned_nodes_by_partition:
                    assigned_nodes_by_partition[partition] = []
                assigned_nodes_by_partition[partition].append((demand_node, node_coords[demand_node]))

        # Find nearest partition for each unassigned node
        labels_copy = labels.copy()

        for unassigned_idx in unassigned_indices:
            unassigned_node = demands[unassigned_idx]
            unassigned_coord = node_coords[unassigned_node]

            min_distance = float('inf')
            nearest_partition = 1  # Default partition

            # Traverse all partitions, find nearest node
            for partition, nodes_in_partition in assigned_nodes_by_partition.items():
                for assigned_node, assigned_coord in nodes_in_partition:
                    # Calculate Euclidean distance
                    distance = np.sqrt((unassigned_coord[0] - assigned_coord[0])**2 +
                                     (unassigned_coord[1] - assigned_coord[1])**2)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_partition = partition

            # Assign to nearest partition
            labels_copy[unassigned_idx] = nearest_partition

            self.log_info(f"Node {unassigned_node} assigned to partition {nearest_partition}, nearest distance: {min_distance:.4f}")

            # Update partition information, so subsequent nodes can consider this newly assigned node
            if nearest_partition not in assigned_nodes_by_partition:
                assigned_nodes_by_partition[nearest_partition] = []
            assigned_nodes_by_partition[nearest_partition].append((unassigned_node, unassigned_coord))

        return labels_copy

    def remove_outliers_iteratively(self, wn, nodes, demands, raw_labels, params):
        """Iteratively process two types of outliers"""
        if not params['outliers_detection']:
            self.log_info("Skipping outlier detection")
            return raw_labels

        self.log_info("Starting iterative outlier detection")

        # Create complete label array
        all_labels = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                all_labels[i] = raw_labels[idx]
            else:
                all_labels[i] = 0

        # Get node connection relationships
        node_connections = []
        for link in wn.links():
            node1 = link[1].start_node_name
            node2 = link[1].end_node_name
            node_connections.append([nodes.index(node1), nodes.index(node2)])
        node_connections = np.array(node_connections)

        number_iter = 0
        max_iterations = 10

        while number_iter < max_iterations:
            # Check if there are still nodes with label 0
            zero_count = np.sum(all_labels == 0)
            if zero_count == 0:
                break

            number_iter += 1
            self.log_info(f"Outlier detection iteration {number_iter}, remaining unassigned nodes: {zero_count}")

            # Process first type of outliers: based on neighbor node label consistency
            for i, node in enumerate(nodes):
                if all_labels[i] != 99999:  # Exclude special marks
                    # Get all connected nodes of current node
                    connected_nodes = []
                    for conn in node_connections:
                        if conn[0] == i:
                            connected_nodes.append(conn[1])
                        elif conn[1] == i:
                            connected_nodes.append(conn[0])
                    connected_nodes = np.array(connected_nodes)

                    if len(connected_nodes) > 0:
                        # Get unique labels of neighbor nodes
                        neighbor_labels = np.unique(all_labels[connected_nodes])
                        # Calculate occurrence count for each label
                        label_counts = np.array([np.sum(all_labels[connected_nodes] == label) for label in neighbor_labels])
                        # Find value with highest occurrence count
                        max_count = np.max(label_counts)
                        # Get all labels with max count
                        max_labels = neighbor_labels[label_counts == max_count]
                        # If 0 is in max count labels, and there are other labels, remove 0
                        if 0 in max_labels and len(max_labels) > 1:
                            max_labels = max_labels[max_labels != 0]
                        # Select first non-0 label (if exists)
                        if len(max_labels) > 0:
                            all_labels[i] = max_labels[0]
                        else:
                            all_labels[i] = 0

            # Process second type of outliers: based on spatial distance and connectivity
            for cluster in range(1, int(np.max(all_labels)) + 1):
                cluster_nodes = np.where(all_labels == cluster)[0]
                if len(cluster_nodes) <= 1:
                    continue

                # Get node coordinates and elevations
                coordinates = []
                elevations = []
                for node_idx in cluster_nodes:
                    node = wn.get_node(nodes[node_idx])
                    try:
                        coord = node.coordinates
                        elev = node.elevation
                    except:
                        coord = (0, 0)
                        elev = 0
                    coordinates.append(coord)
                    elevations.append(elev)

                # Build feature matrix [x, y, elevation]
                features = np.column_stack([coordinates, elevations])

                # Calculate Euclidean distance matrix
                dist_matrix = np.zeros((len(cluster_nodes), len(cluster_nodes)))
                for i in range(len(cluster_nodes)):
                    for j in range(len(cluster_nodes)):
                        dist_matrix[i, j] = np.linalg.norm(features[i] - features[j])

                # Calculate KNN distance for each node
                knn_distances = []
                for i in range(len(cluster_nodes)):
                    distances = dist_matrix[i, :]
                    distances = distances[distances > 0]
                    k = min(params['k_nearest'], len(distances))
                    if k > 0:
                        knn_dist = np.mean(np.sort(distances)[:k])
                        knn_distances.append(knn_dist)
                    else:
                        knn_distances.append(0)

                knn_distances = np.array(knn_distances)

                # Calculate statistics and mark outliers
                if len(knn_distances) > 0:
                    mean_dist = np.mean(knn_distances)
                    std_dist = np.std(knn_distances)

                    # Mark distance outliers
                    outliers = (knn_distances <= mean_dist - 3 * std_dist) | (knn_distances >= mean_dist + 3 * std_dist)
                    all_labels[cluster_nodes[outliers]] = 0

                # Check connectivity
                connect_matrix = self.check_connectivity(node_connections, cluster_nodes)
                components = self.find_connected_components(connect_matrix)

                if len(components) > 1:
                    # Select largest connected component as main region
                    main_component = max(components, key=len)
                    # Mark nodes not in main region as outliers
                    outliers = np.setdiff1d(np.arange(len(cluster_nodes)), main_component)
                    all_labels[cluster_nodes[outliers]] = 0

        # Check if any partition is completely eliminated, if so restore largest connected component
        original_partitions = set(raw_labels)
        current_partitions = set(all_labels[all_labels > 0])

        lost_partitions = original_partitions - current_partitions
        if lost_partitions:
            self.log_info(f"Detected completely eliminated partitions: {sorted(lost_partitions)}")

            # For each eliminated partition, restore its largest connected component
            for lost_partition in lost_partitions:
                # Find nodes originally belonging to this partition
                original_nodes = []
                for i, node in enumerate(nodes):
                    if node in demands:
                        idx = demands.index(node)
                        if raw_labels[idx] == lost_partition:
                            original_nodes.append(i)

                if original_nodes:
                    # Check connectivity of these nodes
                    if len(original_nodes) > 1:
                        # Build connectivity matrix
                        connect_matrix = self.check_connectivity(node_connections, original_nodes)
                        components = self.find_connected_components(connect_matrix)

                        if components:
                            # Restore largest connected component
                            main_component = max(components, key=len)
                            for local_idx in main_component:
                                global_idx = original_nodes[local_idx]
                                all_labels[global_idx] = lost_partition

                            self.log_info(f"Restored largest connected component of partition {lost_partition}: {len(main_component)} nodes")
                    else:
                        # Only one node, restore directly
                        all_labels[original_nodes[0]] = lost_partition
                        self.log_info(f"Restored single node of partition {lost_partition}")

        # Update raw_labels
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                raw_labels[idx] = all_labels[i]

        # Final verification of partition count
        final_partitions = len(set(raw_labels[raw_labels > 0]))
        expected_partitions = params['k']

        if final_partitions != expected_partitions:
            self.log_info(f"⚠️ Partition count mismatch: expected {expected_partitions}, actual {final_partitions}")
        else:
            self.log_info(f"✅ Partition count verification passed: {final_partitions} partitions")

        # Check unassigned node count
        unassigned_count = np.sum(raw_labels == 0)
        if unassigned_count > 0:
            self.log_info(f"Detected {unassigned_count} unassigned nodes, starting nearest neighbor assignment")
            # Perform nearest neighbor assignment
            final_labels = self.assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, raw_labels, params)

            # Validate nearest neighbor assignment result
            final_unassigned = np.sum(final_labels == 0)
            if final_unassigned == 0:
                self.log_info("✅ All nodes successfully assigned to partitions via nearest neighbor assignment")
            else:
                self.log_info(f"⚠️ Still {final_unassigned} nodes unassigned after nearest neighbor assignment")

            self.log_info(f"Outlier detection and nearest neighbor assignment complete, iterations: {number_iter}")
            return final_labels
        else:
            self.log_info("✅ All nodes assigned, no need for nearest neighbor assignment")
            self.log_info(f"Outlier detection complete, iterations: {number_iter}")
            return raw_labels

    def identify_boundary_pipes(self, wn, nodes, demands, labels):
        """Identify boundary pipes - pipes with endpoints in different partitions"""
        try:
            # Create complete label array
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # Create node to index mapping
            node_to_idx = {node: i for i, node in enumerate(nodes)}

            boundary_pipes = []
            non_boundary_pipes = []

            # Traverse all links
            for link in wn.links():
                link_obj = link[1]
                start_node = link_obj.start_node_name
                end_node = link_obj.end_node_name

                # Get partition labels of both endpoints
                if start_node in node_to_idx and end_node in node_to_idx:
                    start_idx = node_to_idx[start_node]
                    end_idx = node_to_idx[end_node]
                    start_label = all_labels[start_idx]
                    end_label = all_labels[end_idx]

                    # Determine if it's a boundary pipe
                    if start_label != end_label and start_label > 0 and end_label > 0:
                        boundary_pipes.append((start_node, end_node))
                    else:
                        non_boundary_pipes.append((start_node, end_node))

            self.log_info(f"Identified {len(boundary_pipes)} boundary pipes, {len(non_boundary_pipes)} non-boundary pipes")
            return boundary_pipes, non_boundary_pipes

        except Exception as e:
            error_msg = f"Identify boundary pipes failed: {str(e)}"
            self.log_error(error_msg)
            return [], []

    def generate_partition_visualization(self, wn, nodes, demands, labels, params, save_path=None):
        """Generate partition visualization"""
        try:
            # Set matplotlib to use English font, avoid Chinese garbled text
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # Create undirected graph
            G = wn.to_graph().to_undirected()

            # Prepare node positions
            pos = {}
            layout = None
            for n in G.nodes():
                try:
                    coord = wn.get_node(n).coordinates
                except:
                    if layout is None:
                        layout = nx.spring_layout(G, seed=params['seed'])
                    coord = layout[n]
                pos[n] = coord

            # Create complete label array
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # Draw network partition
            plt.figure(figsize=(12, 10))

            # Draw edges
            nx.draw_networkx_edges(G, pos=pos, alpha=0.9, width=0.8)

            # Draw nodes
            scatter = nx.draw_networkx_nodes(
                G, pos=pos,
                nodelist=nodes,
                node_color=all_labels,
                cmap=plt.get_cmap("tab10", params['k']+1),
                vmin=0, vmax=params['k'],
                node_size=30
            )

            # Add legend (using English)
            legend_labels = ['Unassigned'] + [f'Partition {i}' for i in range(1, params['k']+1)]
            plt.legend(scatter.legend_elements()[0], legend_labels,
                      title="Node Type",
                      loc='upper right',
                      bbox_to_anchor=(1, 1))

            plt.title(f"Water Network Partitioning Results (K={params['k']}, Fuzziness={params['m']})")
            plt.axis("off")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.log_info(f"Partition visualization saved to: {save_path}")

            return save_path

        except Exception as e:
            error_msg = f"Generate visualization failed: {str(e)}"
            self.log_error(error_msg)
            return None

    def generate_boundary_pipes_visualization(self, wn, nodes, demands, labels, params, save_path=None):
        """Generate boundary pipes visualization - highlight boundary pipes"""
        try:
            # Set matplotlib to use English font
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # Load network model
            wn = wntr.network.WaterNetworkModel(wn.name) if hasattr(wn, 'name') else wn
            G = wn.to_graph().to_undirected()

            # Prepare node positions
            pos = {}
            layout = None

            for node in G.nodes():
                try:
                    coord = wn.get_node(node).coordinates
                    if coord is None or coord == (0, 0):
                        if layout is None:
                            layout = nx.spring_layout(G, seed=params['seed'])
                        coord = layout.get(node, (0, 0))
                except:
                    if layout is None:
                        layout = nx.spring_layout(G, seed=params['seed'])
                    coord = layout.get(node, (0, 0))
                pos[node] = coord

            # Create complete label array
            all_labels = np.zeros(len(nodes))
            for i, node in enumerate(nodes):
                if node in demands:
                    idx = demands.index(node)
                    all_labels[i] = labels[idx]

            # Identify boundary pipes
            boundary_pipes, non_boundary_pipes = self.identify_boundary_pipes(wn, nodes, demands, labels)
            boundary_count = len(boundary_pipes)

            # Create figure - using same style as sensor_placement.py
            plt.figure(figsize=(15, 12))

            # Draw non-boundary pipes (faded but darker)
            nx.draw_networkx_edges(G, pos=pos, edgelist=non_boundary_pipes,
                                  alpha=0.4, width=0.5, edge_color='gray')

            # Draw boundary pipes (red, bold)
            nx.draw_networkx_edges(G, pos=pos, edgelist=boundary_pipes,
                                  alpha=0.9, width=2.5, edge_color='red')

            # Draw regular nodes (faded)
            all_nodes = list(G.nodes())
            nx.draw_networkx_nodes(G, pos=pos, nodelist=all_nodes,
                                 node_color='lightblue', node_size=20, alpha=0.5)

            # Draw partition nodes (colored by partition, overlay regular nodes)
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            for partition_id in range(1, params['k'] + 1):
                partition_nodes = [nodes[i] for i in range(len(nodes)) if all_labels[i] == partition_id]
                if partition_nodes:
                    color = colors[partition_id % len(colors)]
                    nx.draw_networkx_nodes(G, pos=pos, nodelist=partition_nodes,
                                         node_color=color, node_size=30, alpha=0.7,
                                         label=f'Partition {partition_id}')

            # Add legend and title
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Water Network Boundary Pipes Analysis\n'
                     f'Total Boundary Pipes: {boundary_count}, '
                     f'Partitions: {params["k"]}, '
                     f'Fuzziness: {params["m"]}')
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.log_info(f"Boundary pipes visualization saved to: {save_path}")

            return save_path, boundary_count

        except Exception as e:
            error_msg = f"Generate boundary pipes visualization failed: {str(e)}"
            self.log_error(error_msg)
            return None, 0

    def save_partition_results(self, nodes, demands, labels, params, clustering_info, conversation_id):
        """Save partition results to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"partition_results_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # Prepare data
            results_data = []

            # Add demand node partition information
            for i, node_id in enumerate(demands):
                results_data.append({
                    'Node ID': node_id,
                    'Node Type': 'Demand Node',
                    'Partition Number': int(labels[i]),
                    'Partition Name': f'Partition{int(labels[i])}' if labels[i] > 0 else 'Unassigned'
                })

            # Add non-demand node information
            for node_id in nodes:
                if node_id not in demands:
                    results_data.append({
                        'Node ID': node_id,
                        'Node Type': 'Non-demand Node',
                        'Partition Number': 0,
                        'Partition Name': 'Non-demand Node'
                    })

            # Create DataFrame and save
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            # Calculate statistics
            partition_stats = {}
            for i in range(1, params['k'] + 1):
                count = np.sum(labels == i)
                partition_stats[f'Partition{i}'] = count

            unassigned_count = np.sum(labels == 0)
            if unassigned_count > 0:
                partition_stats['Unassigned'] = unassigned_count

            file_size = os.path.getsize(filepath)

            self.log_info(f"Partition results saved to: {filepath}")

            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'records_count': len(results_data),
                'partition_stats': partition_stats,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"Save partition results failed: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def build_partition_prompt(self, network_info: dict, partition_result: dict, user_message: str, csv_info: dict = None):
        """Build professional analysis prompt with network info and partition results"""
        prompt = f"""
You are a professional water supply network partition analysis expert. Now you need to analyze the following network system partition results:

Network basic information:
- Total nodes: {network_info['nodes']['total']} (Junctions: {network_info['nodes']['junctions']}, Reservoirs: {network_info['nodes']['reservoirs']}, Tanks: {network_info['nodes']['tanks']})
- Total links: {network_info['links']['total']} (Pipes: {network_info['links']['pipes']}, Pumps: {network_info['links']['pumps']}, Valves: {network_info['links']['valves']})
- Total network length: {network_info['network_stats']['total_length']:.2f} meters
- Simulation duration: {network_info['network_stats']['simulation_duration']} seconds

✅ Network partition analysis completed successfully!

Partition analysis results:
{partition_result['response']}

Partition technical parameters:
- FCM clustering algorithm, fuzziness parameter m = {partition_result['parameters']['m']}
- Sensitivity matrix perturbation rate: {partition_result['parameters']['perturb_rate']}
- Convergence threshold: {partition_result['parameters']['error']}
- Max iteration count: {partition_result['parameters']['maxiter']}
- Outlier detection: {'Enabled' if partition_result['parameters']['outliers_detection'] else 'Disabled'}

Partition quality metrics:
- Fuzzy Partition Coefficient (FPC): {partition_result['partition_info']['fpc']:.4f}
- Clustering convergence iterations: {partition_result['partition_info']['iterations']}
"""

        if csv_info and csv_info['success']:
            prompt += f"""
📊 Detailed partition data saved as CSV file: {csv_info['filename']}
File size: {csv_info['file_size']} bytes, total {csv_info['records_count']} records
"""

        prompt += f"""
User question: {user_message}

Based on the network basic information and partition analysis results, please provide professional analysis and suggestions, including:
1. Reasonableness evaluation of partition results
2. Technical analysis of partition quality (based on FPC value and partition distribution)
3. Possible optimization suggestions
4. Engineering application value and significance
5. If necessary, suggest further analysis directions

Also inform the user that they can download detailed partition data for further analysis.

Please use the following signature format at the end of your reply:

Best regards,

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def process(self, inp_file_path: str, user_message: str, conversation_id: str):
        """Main processing function"""
        try:
            self.log_info(f"Starting to process network partition request: {user_message}")

            # Step 1: Parse network file, get basic information
            network_info = self.parse_network(inp_file_path)
            if 'error' in network_info:
                return {
                    'success': False,
                    'response': f"Network file parsing failed: {network_info['error']}",
                    'intent': 'partition_analysis',
                    'confidence': 0.0
                }

            # Step 2: Parse user intent and parameters
            intent_result = self.parse_user_intent(user_message)
            params = intent_result['params']

            self.log_info(f"Parsed parameters: {params}")

            # Step 3: Load network model
            wn, error = self.load_network(inp_file_path)
            if error:
                return {
                    'success': False,
                    'response': f"Load network file failed: {error['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # Calculate sensitivity matrix
            nodes, demands, S = self.compute_sensitivity_matrix(inp_file_path, params['perturb_rate'])
            if isinstance(S, dict) and 'error' in S:
                return {
                    'success': False,
                    'response': f"Calculate sensitivity matrix failed: {S['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # Standardize sensitivity matrix
            S_normalized = self.normalize_matrix(S)

            # Execute FCM clustering
            raw_labels, clustering_info, error = self.perform_fcm_clustering(S_normalized, params)
            if error:
                return {
                    'success': False,
                    'response': f"FCM clustering failed: {error['error']}",
                    'intent': intent_result['intent'],
                    'confidence': intent_result['confidence']
                }

            # Outlier detection and processing
            refined_labels = self.remove_outliers_iteratively(wn, nodes, demands, raw_labels, params)

            # Generate visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"partition_viz_{conversation_id[:8]}_{timestamp}.png"
            viz_path = os.path.join(self.downloads_folder, viz_filename)

            viz_result = self.generate_partition_visualization(wn, nodes, demands, refined_labels, params, viz_path)

            # Generate boundary pipes visualization
            boundary_viz_filename = f"boundary_pipes_viz_{conversation_id[:8]}_{timestamp}.png"
            boundary_viz_path = os.path.join(self.downloads_folder, boundary_viz_filename)
            boundary_viz_result, boundary_pipe_count = self.generate_boundary_pipes_visualization(
                wn, nodes, demands, refined_labels, params, boundary_viz_path
            )

            # Get boundary pipes information for report
            boundary_pipes, non_boundary_pipes = self.identify_boundary_pipes(wn, nodes, demands, refined_labels)

            # Save partition results
            save_result = self.save_partition_results(nodes, demands, refined_labels, params, clustering_info, conversation_id)

            # Generate analysis report
            total_nodes = len(nodes)
            demand_nodes_count = len(demands)
            partition_distribution = {}
            for i in range(1, params['k'] + 1):
                count = int(np.sum(refined_labels == i))  # Convert to Python int
                partition_distribution[i] = count

            unassigned_count = int(np.sum(refined_labels == 0))  # Convert to Python int

            response_text = f"""
Network partition analysis completed!

📊 **Partition Overview**
- Total nodes: {total_nodes}
- Demand nodes: {demand_nodes_count}
- Partition count: {params['k']}
- Fuzziness parameter: {params['m']}
- Perturbation rate: {params['perturb_rate']}

📈 **Clustering Quality**
- Fuzzy Partition Coefficient (FPC): {clustering_info['fpc']:.4f}
- Convergence iterations: {clustering_info['iterations']}

🎯 **Partition Distribution**
"""
            for i in range(1, params['k'] + 1):
                count = partition_distribution[i]
                percentage = (count / demand_nodes_count) * 100
                response_text += f"- Partition {i}: {count} nodes ({percentage:.1f}%)\n"

            if unassigned_count > 0:
                percentage = (unassigned_count / demand_nodes_count) * 100
                response_text += f"- Unassigned: {unassigned_count} nodes ({percentage:.1f}%)\n"

            if params['outliers_detection']:
                response_text += f"\n✅ Outlier detection and processing completed"
            else:
                response_text += f"\n⚠️ Outlier detection not performed"

            # Add boundary pipes information
            response_text += f"\n\n🔴 **Boundary Pipes Analysis**\n"
            response_text += f"- Total boundary pipes: {boundary_pipe_count}\n"
            response_text += f"- Boundary pipes ratio: {(boundary_pipe_count / (boundary_pipe_count + len(non_boundary_pipes)) * 100):.1f}% (total {boundary_pipe_count + len(non_boundary_pipes)} pipes)"

            # Build professional analysis prompt
            prompt = self.build_partition_prompt(
                network_info,
                {
                    'response': response_text,
                    'partition_info': {
                        'total_nodes': total_nodes,
                        'demand_nodes': demand_nodes_count,
                        'k': params['k'],
                        'partition_distribution': partition_distribution,
                        'unassigned_count': unassigned_count,
                        'fpc': float(clustering_info['fpc']),  # Convert to Python float
                        'iterations': int(clustering_info['iterations'])  # Convert to Python int
                    },
                    'parameters': params
                },
                user_message,
                save_result if save_result['success'] else None
            )

            result = {
                'success': True,
                'response': response_text,
                'prompt': prompt,  # Add professional prompt for GPT analysis
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'partition_info': {
                    'total_nodes': total_nodes,
                    'demand_nodes': demand_nodes_count,
                    'k': params['k'],
                    'partition_distribution': partition_distribution,
                    'unassigned_count': unassigned_count,
                    'fpc': float(clustering_info['fpc']),  # Convert to Python float
                    'iterations': int(clustering_info['iterations'])  # Convert to Python int
                },
                'parameters': params,
                'network_info': network_info  # Add network info
            }

            # Add file download info
            if save_result['success']:
                result['csv_info'] = save_result

            if viz_result:
                result['visualization'] = {
                    'filename': viz_filename,
                    'path': viz_path
                }

            if boundary_viz_result:
                result['boundary_visualization'] = {
                    'filename': boundary_viz_filename,
                    'path': boundary_viz_result,
                    'boundary_pipe_count': boundary_pipe_count
                }

            return result

        except Exception as e:
            error_msg = f"Error processing network partition request: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'response': error_msg,
                'intent': 'partition_analysis',
                'confidence': 0.0
            }
