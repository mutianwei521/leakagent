# -*- coding: utf-8 -*-
"""
FCM Partition Module
Implements FCM network partitioning algorithm based on pressure sensitivity and pipe length weights
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False


class FCMPartitioner:
    """FCM-based Network Partitioner"""
    
    def __init__(self, n_clusters: int = 5, m: float = 2.0, 
                 max_iter: int = 100, error: float = 1e-5):
        """
        Initialize FCM Partitioner
        
        Args:
            n_clusters: Number of clusters
            m: Fuzziness exponent
            max_iter: Maximum iterations
            error: Convergence error
        """
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.error = float(error)
        self.cluster_centers = None
        self.membership_matrix = None
        self.partition_labels = None
        
    def prepare_features(self, adjacency_matrix: np.ndarray,
                        node_weights: np.ndarray,
                        pipe_lengths: Dict[str, float],
                        pipe_names: List[str],
                        node_names: List[str]) -> np.ndarray:
        """
        Prepare FCM clustering features
        
        Args:
            adjacency_matrix: Adjacency matrix
            node_weights: Node weights (Average pressure sensitivity)
            pipe_lengths: Dictionary of pipe lengths
            pipe_names: List of pipe names
            node_names: List of node names
            
        Returns:
            np.ndarray: Feature matrix [n_nodes, n_features]
        """
        try:
            n_nodes = len(node_names)
            features = []

            # Data type check and conversion
            logger.debug(f"Input parameter check:")
            logger.debug(f"  adjacency_matrix shape: {adjacency_matrix.shape}, type: {adjacency_matrix.dtype}")
            logger.debug(f"  node_weights shape: {node_weights.shape}, type: {node_weights.dtype}")
            logger.debug(f"  node_names count: {len(node_names)}")
            logger.debug(f"  pipe_names count: {len(pipe_names)}")

            # Ensure node_weights is numeric type
            if not np.issubdtype(node_weights.dtype, np.number):
                logger.error(f"node_weights contains non-numeric data, type: {node_weights.dtype}")
                return np.array([])

            # Check for NaN or inf
            if np.any(np.isnan(node_weights)) or np.any(np.isinf(node_weights)):
                logger.warning("node_weights contains NaN or inf values, replacing with 0")
                node_weights = np.nan_to_num(node_weights, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. Node degree (Number of connections)
            node_degrees = np.sum(adjacency_matrix, axis=1)

            # 2. Node weight statistical features
            avg_node_weights = np.mean(node_weights, axis=1)  # Average sensitivity
            max_node_weights = np.max(node_weights, axis=1)   # Max sensitivity
            std_node_weights = np.std(node_weights, axis=1)   # Sensitivity standard deviation
            
            # 3. Neighbor node features
            neighbor_avg_weights = np.zeros(n_nodes)
            neighbor_degrees = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                neighbors = np.where(adjacency_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_avg_weights[i] = np.mean(avg_node_weights[neighbors])
                    neighbor_degrees[i] = np.mean(node_degrees[neighbors])
            
            # 4. Pipe length features (Average length of pipes connected to this node)
            avg_pipe_lengths = np.zeros(n_nodes)
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}
            
            # Build node-to-pipe mapping
            for pipe_name in pipe_names:
                # Need to get pipe connection info from EPANET handler here
                # Temporarily using adjacency matrix info
                pass
            
            # Combine all features
            feature_matrix = np.column_stack([
                node_degrees,           # Node degree
                avg_node_weights,       # Average pressure sensitivity
                max_node_weights,       # Max pressure sensitivity
                std_node_weights,       # Pressure sensitivity std dev
                neighbor_avg_weights,   # Neighbor average weight
                neighbor_degrees,       # Neighbor average degree
                avg_pipe_lengths        # Average pipe length
            ])

            # Check validity of feature matrix
            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                logger.warning("Feature matrix contains NaN or inf values, replacing with 0")
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Check if feature matrix is empty or all zeros
            if feature_matrix.size == 0:
                logger.error("Feature matrix is empty")
                return np.array([])

            if np.all(feature_matrix == 0):
                logger.warning("Feature matrix is all zeros, adding small random perturbation")
                feature_matrix += np.random.normal(0, 1e-6, feature_matrix.shape)

            # Normalize features
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)

            logger.info(f"Feature matrix preparation completed: {feature_matrix.shape}")
            logger.debug(f"Feature matrix stats: min={np.min(feature_matrix):.6f}, max={np.max(feature_matrix):.6f}")
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            return np.array([])
    
    def partition_network(self, feature_matrix: np.ndarray) -> bool:
        """
        Partition network using FCM
        
        Args:
            feature_matrix: Feature matrix
            
        Returns:
            bool: Whether partitioning was successful
        """
        try:
            if feature_matrix.size == 0:
                logger.error("Feature matrix is empty")
                return False

            # Check validity of feature matrix
            if not np.issubdtype(feature_matrix.dtype, np.number):
                logger.error(f"Feature matrix contains non-numeric data, type: {feature_matrix.dtype}")
                return False

            if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
                logger.error("Feature matrix contains NaN or inf values")
                return False

            # Check if partition number is reasonable
            n_samples = feature_matrix.shape[0]
            if self.n_clusters >= n_samples:
                logger.error(f"Number of clusters {self.n_clusters} is greater than or equal to number of samples {n_samples}")
                return False

            logger.debug(f"Start FCM clustering: samples={n_samples}, features={feature_matrix.shape[1]}, clusters={self.n_clusters}")

            # Transpose feature matrix to adapt to skfuzzy input format
            data = feature_matrix.T

            # Ensure parameter types are correct
            n_clusters = int(self.n_clusters)
            m = float(self.m)
            error = float(self.error)
            maxiter = int(self.max_iter)

            logger.debug(f"FCM parameters: n_clusters={n_clusters}, m={m}, error={error}, maxiter={maxiter}")

            # Execute FCM clustering
            self.cluster_centers, self.membership_matrix, _, _, _, _, _ = fuzz.cluster.cmeans(
                data, n_clusters, m, error=error, maxiter=maxiter
            )
            
            # Get partition label for each node (Cluster with max membership)
            self.partition_labels = np.argmax(self.membership_matrix, axis=0)
            
            # Calculate clustering quality metrics
            silhouette_avg = silhouette_score(feature_matrix, self.partition_labels)
            
            logger.info(f"FCM partitioning completed: {self.n_clusters} clusters, Silhouette Score: {silhouette_avg:.3f}")
            
            # Output partition statistics
            for i in range(self.n_clusters):
                cluster_size = np.sum(self.partition_labels == i)
                logger.info(f"Cluster {i}: {cluster_size} nodes")
            
            return True
            
        except Exception as e:
            logger.error(f"FCM partitioning failed: {e}")
            return False
    
    def get_partition_subgraphs(self, graph: nx.Graph, 
                               node_names: List[str]) -> List[nx.Graph]:
        """
        Get subgraphs based on partition results
        
        Args:
            graph: Original network graph
            node_names: List of node names
            
        Returns:
            List[nx.Graph]: List of partition subgraphs
        """
        try:
            if self.partition_labels is None:
                logger.error("Partition results not found")
                return []
            
            subgraphs = []
            
            for cluster_id in range(self.n_clusters):
                # Get nodes of current partition
                cluster_nodes = [node_names[i] for i in range(len(node_names)) 
                               if self.partition_labels[i] == cluster_id]
                
                # Create subgraph
                subgraph = graph.subgraph(cluster_nodes).copy()
                subgraphs.append(subgraph)
                
                logger.debug(f"Partition {cluster_id} subgraph: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
            
            return subgraphs
            
        except Exception as e:
            logger.error(f"Failed to get partition subgraphs: {e}")
            return []
    
    def optimize_partition_number(self, feature_matrix: np.ndarray,
                                 min_clusters: int = 2, max_clusters: int = 10) -> int:
        """
        Optimize partition number
        
        Args:
            feature_matrix: Feature matrix
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            
        Returns:
            int: Optimal number of clusters
        """
        try:
            best_score = -1
            best_n_clusters = self.n_clusters
            scores = []
            
            logger.info(f"Optimizing partition number: {min_clusters}-{max_clusters}")
            
            for n in range(min_clusters, max_clusters + 1):
                # Temporarily set partition number
                original_n_clusters = self.n_clusters
                self.n_clusters = n
                
                # Execute partitioning
                if self.partition_network(feature_matrix):
                    # Calculate Silhouette Score
                    score = silhouette_score(feature_matrix, self.partition_labels)
                    scores.append((n, score))
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n
                    
                    logger.info(f"Partition count {n}: Silhouette Score {score:.3f}")
                
                # Restore original partition number
                self.n_clusters = original_n_clusters
            
            # Set optimal partition number
            self.n_clusters = best_n_clusters
            logger.info(f"Optimal partition number: {best_n_clusters}, Silhouette Score: {best_score:.3f}")
            
            return best_n_clusters
            
        except Exception as e:
            logger.error(f"Failed to optimize partition number: {e}")
            return self.n_clusters
    
    def get_partition_info(self) -> Dict:
        """
        Get partition info
        
        Returns:
            Dict: Partition info dictionary
        """
        if self.partition_labels is None:
            return {}
        
        info = {
            'n_clusters': self.n_clusters,
            'partition_labels': self.partition_labels.copy(),
            'cluster_sizes': [np.sum(self.partition_labels == i) for i in range(self.n_clusters)],
            'membership_matrix': self.membership_matrix.copy() if self.membership_matrix is not None else None,
            'cluster_centers': self.cluster_centers.copy() if self.cluster_centers is not None else None
        }
        
        return info
    
    def check_connectivity(self, node_connections: np.ndarray, cluster_nodes: np.ndarray) -> np.ndarray:
        """Check node connectivity using Warshall algorithm"""
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

    def find_connected_components(self, connect_matrix: np.ndarray) -> List[List[int]]:
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

    def remove_outliers_iteratively(self, wn, nodes: List[str], demands: List[str],
                                     raw_labels: np.ndarray, k_nearest: int = 10,
                                     outliers_detection: bool = True, seed: int = 42,
                                     output_dir: str = None) -> np.ndarray:
        """
        Iteratively handle two types of outliers

        Args:
            wn: WNTR network object
            nodes: List of all node names
            demands: List of demand node names
            raw_labels: Raw label array
            k_nearest: KNN parameter
            outliers_detection: Whether to perform outlier detection
            seed: Random seed
            output_dir: Output directory (for saving visualizations)

        Returns:
            Processed label array
        """
        if not WNTR_AVAILABLE:
            logger.error("WNTR library not installed, cannot perform outlier detection")
            return raw_labels

        if not outliers_detection:
            logger.info("Skipping outlier detection")
            return raw_labels

        logger.info("Starting iterative outlier detection")

        # Create complete label array
        all_labels = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                all_labels[i] = raw_labels[idx]
            else:
                all_labels[i] = 0

        # Save initial partition visualization
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_partition(wn, nodes, all_labels,
                                    os.path.join(output_dir, 'partition_0_initial.png'),
                                    'Initial Partition (Before Outlier Removal)')

        # Get node connections
        node_connections = []
        for link in wn.links():
            node1 = link[1].start_node_name
            node2 = link[1].end_node_name
            node_connections.append([nodes.index(node1), nodes.index(node2)])
        node_connections = np.array(node_connections)

        number_iter = 0
        max_iterations = 10

        while number_iter < max_iterations:
            # Check if there are still points with label 0
            zero_count = np.sum(all_labels == 0)
            if zero_count == 0:
                break

            number_iter += 1
            logger.info(f"Outlier detection iteration {number_iter}, remaining unassigned nodes: {zero_count}")

            # Handle Type 1 Outliers: Based on consistency of neighbor labels
            for i, node in enumerate(nodes):
                if all_labels[i] != 99999:  # Exclude special markers
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
                        # Calculate count of each label
                        label_counts = np.array([np.sum(all_labels[connected_nodes] == label) for label in neighbor_labels])
                        # Find value with max count
                        max_count = np.max(label_counts)
                        # Get all labels reaching max count
                        max_labels = neighbor_labels[label_counts == max_count]
                        # If 0 is in max labels and there are other labels, remove 0
                        if 0 in max_labels and len(max_labels) > 1:
                            max_labels = max_labels[max_labels != 0]
                        # Select first non-zero label (if exists)
                        if len(max_labels) > 0:
                            all_labels[i] = max_labels[0]
                        else:
                            all_labels[i] = 0

            # Handle Type 2 Outliers: Based on spatial distance and connectivity
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
                    k = min(k_nearest, len(distances))
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
                    # Select largest connected component as main area
                    main_component = max(components, key=len)
                    # Mark nodes not in main area as outliers
                    outliers = np.setdiff1d(np.arange(len(cluster_nodes)), main_component)
                    all_labels[cluster_nodes[outliers]] = 0

            # Save visualization of current iteration
            if output_dir:
                self.visualize_partition(wn, nodes, all_labels,
                                        os.path.join(output_dir, f'partition_{number_iter}_iteration.png'),
                                        f'Partition After Iteration {number_iter}')

        # Check if any partition was completely eliminated, restore largest connected component if so
        original_partitions = set(raw_labels)
        current_partitions = set(all_labels[all_labels > 0])

        lost_partitions = original_partitions - current_partitions
        if lost_partitions:
            logger.info(f"Detected completely eliminated partitions: {sorted(lost_partitions)}")

            # Restore max connected component for each eliminated partition
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

                            logger.info(f"Restored largest connected component of partition {lost_partition}: {len(main_component)} nodes")
                    else:
                        # Only one node, restore directly
                        all_labels[original_nodes[0]] = lost_partition
                        logger.info(f"Restored single node of partition {lost_partition}")

        # Update raw_labels
        for i, node in enumerate(nodes):
            if node in demands:
                idx = demands.index(node)
                raw_labels[idx] = all_labels[i]

        # Final verification of partition count
        final_partitions = len(set(raw_labels[raw_labels > 0]))
        expected_partitions = self.n_clusters

        if final_partitions != expected_partitions:
            logger.info(f"⚠️ Partition count mismatch: Expected {expected_partitions}, Actual {final_partitions}")
        else:
            logger.info(f"✅ Partition count verification passed: {final_partitions} partitions")

        # Check unassigned node count
        unassigned_count = np.sum(raw_labels == 0)
        if unassigned_count > 0:
            logger.info(f"Detected {unassigned_count} unassigned nodes, starting nearest neighbor assignment")
            # Perform nearest neighbor assignment
            final_labels = self.assign_unassigned_nodes_by_nearest_neighbor(wn, nodes, demands, raw_labels, k_nearest, seed)

            # Verify nearest neighbor assignment results
            final_unassigned = np.sum(final_labels == 0)
            if final_unassigned == 0:
                logger.info("✅ All nodes successfully assigned to partitions via nearest neighbor assignment")
            else:
                logger.info(f"⚠️ After nearest neighbor assignment, {final_unassigned} nodes remain unassigned")

            logger.info(f"Outlier detection and nearest neighbor assignment completed, iterations: {number_iter}")

            # Save final result visualization
            if output_dir:
                # Create complete label array for visualization
                final_all_labels = np.zeros(len(nodes))
                for i, node in enumerate(nodes):
                    if node in demands:
                        idx = demands.index(node)
                        final_all_labels[i] = final_labels[idx]
                self.visualize_partition(wn, nodes, final_all_labels,
                                        os.path.join(output_dir, 'partition_final.png'),
                                        'Final Partition (After All Processing)')

            return final_labels
        else:
            logger.info("✅ All nodes assigned, no nearest neighbor assignment needed")
            logger.info(f"Outlier detection completed, iterations: {number_iter}")

            # Save final result visualization
            if output_dir:
                self.visualize_partition(wn, nodes, all_labels,
                                        os.path.join(output_dir, 'partition_final.png'),
                                        'Final Partition (After All Processing)')

            return raw_labels

    def assign_unassigned_nodes_by_nearest_neighbor(self, wn, nodes: List[str], demands: List[str],
                                                     labels: np.ndarray, k_nearest: int = 10,
                                                     seed: int = 42) -> np.ndarray:
        """Assign unassigned nodes to nearest partition"""
        if not WNTR_AVAILABLE:
            logger.error("WNTR library not installed, cannot perform nearest neighbor assignment")
            return labels

        # Find unassigned demand nodes
        unassigned_indices = []
        for i, demand_node in enumerate(demands):
            if labels[i] == 0:
                unassigned_indices.append(i)

        if len(unassigned_indices) == 0:
            return labels

        logger.info(f"Starting nearest partition assignment for {len(unassigned_indices)} unassigned demand nodes")

        # Get node coordinates
        node_coords = {}
        layout = None  # Used for nodes without coordinates

        for node_name in nodes:
            try:
                coord = wn.get_node(node_name).coordinates
                if coord is None or coord == (0, 0):
                    # If no coordinates, use network layout
                    if layout is None:
                        G = wn.to_graph().to_undirected()
                        layout = nx.spring_layout(G, seed=seed)
                    coord = layout.get(node_name, (0, 0))
            except:
                if layout is None:
                    G = wn.to_graph().to_undirected()
                    layout = nx.spring_layout(G, seed=seed)
                coord = layout.get(node_name, (0, 0))
            node_coords[node_name] = coord

        # Create partition info for assigned nodes
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
            nearest_partition = 1  # 默认分区

            # Iterate all partitions to find nearest node
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

            logger.info(f"Node {unassigned_node} assigned to partition {nearest_partition}, nearest distance: {min_distance:.4f}")

            # Update partition info so subsequent nodes can consider this newly assigned node
            if nearest_partition not in assigned_nodes_by_partition:
                assigned_nodes_by_partition[nearest_partition] = []
            assigned_nodes_by_partition[nearest_partition].append((unassigned_node, unassigned_coord))

        return labels_copy

    def visualize_partition(self, wn, nodes: List[str], labels: np.ndarray,
                           output_path: str, title: str = "Network Partition"):
        """
        Visualize partition results (Use WNTR plotting, keep original coordinates, Nature journal style)

        Args:
            wn: WNTR network object
            nodes: List of node names
            labels: Label array
            output_path: Output image path
            title: Image title
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend

            # Nature journal style settings
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['xtick.major.width'] = 1.2
            plt.rcParams['ytick.major.width'] = 1.2

            # Create figure, increase size
            fig, ax = plt.subplots(figsize=(12, 10))

            # Assign high contrast colors for each partition (Nature journal common colors)
            unique_labels = np.unique(labels)
            # Use high contrast colors: red, blue, green, orange, purple, brown, etc.
            distinct_colors = [
                '#E41A1C',  # Red
                '#377EB8',  # Blue
                '#4DAF4A',  # Green
                '#FF7F00',  # Orange
                '#984EA3',  # Purple
                '#A65628',  # Brown
                '#F781BF',  # Pink
                '#999999',  # Grey
                '#66C2A5',  # Teal
                '#FC8D62',  # Light orange
            ]

            color_map = {}
            for i, label in enumerate(unique_labels):
                if label == 0:
                    continue  # Skip unassigned labels
                color_idx = (int(label) - 1) % len(distinct_colors)
                color_map[label] = distinct_colors[color_idx]

            # Plot network using WNTR (Only plot pipes)
            wntr.graphics.plot_network(
                wn,
                node_size=0,  # Do not plot nodes
                link_width=1.5,
                add_colorbar=False,
                ax=ax
            )

            # Group nodes by partition
            partition_nodes = {}
            unassigned_nodes = []

            for i, node_name in enumerate(nodes):
                label = labels[i]
                if label == 0:
                    unassigned_nodes.append(node_name)
                else:
                    if label not in partition_nodes:
                        partition_nodes[label] = []
                    partition_nodes[label].append(node_name)

            # Plot assigned nodes (Do not show node labels)
            for label, node_list in partition_nodes.items():
                color = color_map[label]

                # Get node coordinates
                node_coords = []
                for node_name in node_list:
                    node = wn.get_node(node_name)
                    node_coords.append(node.coordinates)

                if node_coords:
                    x_coords = [coord[0] for coord in node_coords]
                    y_coords = [coord[1] for coord in node_coords]

                    ax.scatter(x_coords, y_coords,
                             c=color,
                             s=300,  # Node size
                             edgecolors='black',
                             linewidths=1.5,
                             zorder=3,
                             alpha=0.9,
                             label=f'Partition {int(label)}')

            # Plot unassigned nodes
            if unassigned_nodes:
                node_coords = []
                for node_name in unassigned_nodes:
                    node = wn.get_node(node_name)
                    node_coords.append(node.coordinates)

                if node_coords:
                    x_coords = [coord[0] for coord in node_coords]
                    y_coords = [coord[1] for coord in node_coords]

                    ax.scatter(x_coords, y_coords,
                             c='white',
                             marker='X',
                             s=250,
                             edgecolors='#D62728',  # Dark red
                             linewidths=2.5,
                             zorder=4,
                             label='Unassigned')

            # Set title (Nature style: concise)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Set axis labels (Nature style)
            ax.set_xlabel('X Coordinate (ft)', fontsize=12)
            ax.set_ylabel('Y Coordinate (ft)', fontsize=12)

            # Add grid (Nature style: thin lines)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')

            # Set background color (Nature style: white)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            # Set axis style
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_linewidth(1.2)
            ax.spines['right'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.spines['left'].set_linewidth(1.2)

            # Add legend (Enlarge to avoid occlusion)
            legend = ax.legend(
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),  # Place on the right side of the plot, slightly further away
                fontsize=12,
                frameon=True,
                fancybox=False,  # Nature style: square border
                shadow=False,    # Nature style: no shadow
                # title='Partitions',
                # title_fontsize=13,
                markerscale=0.5,  # Enlarge markers in legend to avoid occlusion
                handletextpad=1.0,  # Increase spacing between marker and text
                borderpad=1.2,      # Increase legend padding
                labelspacing=1.2    # Increase spacing between labels
            )
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.2)

            # Adjust layout to leave more space for legend
            plt.tight_layout(rect=[0, 0, 0.80, 1])

            # Save image (Nature journal requirement: high resolution)
            plt.savefig(output_path, dpi=600, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       pad_inches=0.1)
            plt.close()

            logger.info(f"Partition visualization saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_partition_results(self, output_dir: str, node_names: List[str]) -> bool:
        """
        Save partition results

        Args:
            output_dir: Output directory
            node_names: List of node names

        Returns:
            bool: Whether saving was successful
        """
        try:
            import os
            import pandas as pd

            os.makedirs(output_dir, exist_ok=True)

            if self.partition_labels is not None:
                # Save partition labels
                partition_df = pd.DataFrame({
                    'node_name': node_names,
                    'partition_id': self.partition_labels
                })
                partition_df.to_csv(os.path.join(output_dir, 'fcm_partition.csv'), index=False)

                # Save membership matrix
                if self.membership_matrix is not None:
                    membership_df = pd.DataFrame(
                        self.membership_matrix.T,
                        columns=[f'cluster_{i}' for i in range(self.n_clusters)],
                        index=node_names
                    )
                    membership_df.to_csv(os.path.join(output_dir, 'fcm_membership.csv'))

                logger.info(f"FCM partition results saved to {output_dir}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to save partition results: {e}")
            return False
