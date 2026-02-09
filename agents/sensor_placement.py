"""
SensorPlacement Sensor Optimization Placement Agent
Responsible for sensor optimization placement based on network partition results, considering resilience and detection effectiveness
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from .base_agent import BaseAgent
from .partition_sim import PartitionSim

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

class SensorPlacement(BaseAgent):
    """Sensor optimization placement agent"""
    
    def __init__(self):
        super().__init__("SensorPlacement")
        
        if not WNTR_AVAILABLE:
            self.log_error("WNTR library not installed, sensor placement function unavailable")
        
        self.partition_sim = PartitionSim()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)
        
        # Default parameters (optimized version, reduced computation)
        self.default_params = {
            'demand_ratios': [0.20],  # Reduced perturbation ratios, only use one
            'sensitivity_threshold': 0.5,  # Sensitivity threshold
            'max_failure_rate': 0.8,  # Max failure rate (allow more sensor failures)
            'resilience_weight': 0.4,  # Resilience weight
            'coverage_weight': 0.6,  # Coverage weight
            'target_coverage': 0.95,  # Target coverage
            'min_sensor_ratio': 0.04,  # Min sensor ratio (4% of node count)
            'max_sensor_ratio': 0.15,  # Max sensor ratio (15% of node count)
            'custom_thresholds': {},  # Custom sensor thresholds {sensor_node: threshold}
            'enable_custom_thresholds': False  # Whether to enable custom thresholds
        }
    
    def load_partition_results(self, csv_file_path):
        """Load partition results from CSV file"""
        try:
            self.log_info(f"Loading partition results: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            
            # Extract partition information for demand nodes
            demand_nodes_df = df[df['Node Type'] == 'Demand Node']
            
            partitions = {}
            for _, row in demand_nodes_df.iterrows():
                partition_id = row['Partition Number']
                node_id = row['Node ID']
                
                if partition_id > 0:  # Exclude unassigned nodes
                    if partition_id not in partitions:
                        partitions[partition_id] = []
                    partitions[partition_id].append(node_id)
            
            self.log_info(f"Successfully loaded partition results: {len(partitions)} partitions")
            for partition_id, nodes in partitions.items():
                self.log_info(f"  Partition{partition_id}: {len(nodes)} nodes")
            
            return partitions
            
        except Exception as e:
            error_msg = f"Load partition results failed: {str(e)}"
            self.log_error(error_msg)
            return None
    
    def compute_pressure_sensitivity_matrix(self, inp_file_path, partitions, demand_ratios):
        """Calculate pressure sensitivity matrix"""
        try:
            self.log_info(f"Starting to calculate pressure sensitivity matrix, perturbation ratios: {demand_ratios}")

            # Load network model
            wn = wntr.network.WaterNetworkModel(inp_file_path)

            # Get all demand nodes
            all_demand_nodes = []
            for nodes in partitions.values():
                all_demand_nodes.extend(nodes)

            self.log_info(f"Total demand nodes: {len(all_demand_nodes)}")

            # Check if nodes exist in network
            valid_demand_nodes = []
            for node in all_demand_nodes:
                if node in wn.node_name_list:
                    valid_demand_nodes.append(node)
                else:
                    self.log_info(f"Node {node} does not exist in network, skipping")

            if not valid_demand_nodes:
                raise ValueError("No valid demand nodes found")

            self.log_info(f"Valid demand node count: {len(valid_demand_nodes)}")

            # Run baseline simulation
            self.log_info("Running baseline simulation...")
            sim = wntr.sim.EpanetSimulator(wn)
            base_results = sim.run_sim()
            base_pressure = base_results.node['pressure'].loc[:, valid_demand_nodes].values

            # Calculate total actual demand (following cluster_simple.py approach)
            self.log_info("Calculating total demand...")
            total_demand = 0
            for name in valid_demand_nodes:
                # Get actual demand at all time steps for this node
                node_demands = base_results.node['demand'].loc[:, name]
                # Accumulate all time step demands for this node
                total_demand += node_demands.sum()

            self.log_info(f"Total demand: {total_demand:.4f}")

            # Initialize sensitivity matrix
            n_nodes = len(valid_demand_nodes)
            sensitivity_matrix = np.zeros((n_nodes, n_nodes))

            # Perturb each demand node
            for i, perturb_node in enumerate(valid_demand_nodes):
                if i % 50 == 0:  # Output progress every 50 nodes
                    self.log_info(f"Processing node {i+1}/{n_nodes}: {perturb_node} ({(i+1)/n_nodes*100:.1f}%)")

                # Save original demand
                original_demands = {}
                node = wn.get_node(perturb_node)
                for j, ts in enumerate(node.demand_timeseries_list):
                    original_demands[j] = ts.base_value

                # Calculate for each perturbation ratio
                ratio_sensitivities = []

                for ratio in demand_ratios:
                    # Calculate average perturbation amount for this ratio (following cluster_simple.py)
                    delta = total_demand * ratio / len(base_results.node['demand'])

                    # Set perturbed demand
                    node = wn.get_node(perturb_node)
                    for j, ts in enumerate(node.demand_timeseries_list):
                        if original_demands[j] > 0:
                            # Proportional perturbation
                            ts.base_value = original_demands[j] + original_demands[j] * ratio
                        else:
                            # Absolute perturbation (using delta based on total demand)
                            ts.base_value = original_demands[j] + delta

                    # Run perturbed simulation
                    sim = wntr.sim.EpanetSimulator(wn)
                    perturb_results = sim.run_sim()
                    perturb_pressure = perturb_results.node['pressure'].loc[:, valid_demand_nodes].values

                    # Calculate pressure difference
                    pressure_diff = np.abs(perturb_pressure - base_pressure)

                    # Calculate time average
                    avg_pressure_diff = np.mean(pressure_diff, axis=0)
                    ratio_sensitivities.append(avg_pressure_diff)

                    # Restore original demand
                    node = wn.get_node(perturb_node)
                    for j, ts in enumerate(node.demand_timeseries_list):
                        ts.base_value = original_demands[j]

                # Calculate average sensitivity for multiple perturbation ratios
                avg_sensitivity = np.mean(ratio_sensitivities, axis=0)

                # Normalization (avoid division by zero)
                max_sensitivity = np.max(avg_sensitivity)
                if max_sensitivity > 0:
                    sensitivity_matrix[i, :] = avg_sensitivity / max_sensitivity
                else:
                    sensitivity_matrix[i, :] = 0

            self.log_info("Pressure sensitivity matrix calculation complete")

            # Output sensitivity matrix statistics
            non_zero_count = np.count_nonzero(sensitivity_matrix)
            total_elements = sensitivity_matrix.size
            self.log_info(f"Sensitivity matrix statistics: non-zero elements {non_zero_count}/{total_elements} ({non_zero_count/total_elements*100:.1f}%)")
            self.log_info(f"Sensitivity matrix range: [{np.min(sensitivity_matrix):.6f}, {np.max(sensitivity_matrix):.6f}]")

            # Update partition info, only retain valid nodes
            valid_partitions = {}
            for partition_id, nodes in partitions.items():
                valid_nodes = [node for node in nodes if node in valid_demand_nodes]
                if valid_nodes:
                    valid_partitions[partition_id] = valid_nodes

            return {
                'matrix': sensitivity_matrix,
                'nodes': valid_demand_nodes,
                'partitions': valid_partitions
            }

        except Exception as e:
            error_msg = f"Calculate pressure sensitivity matrix failed: {str(e)}"
            self.log_error(error_msg)
            return None
    
    def select_sensors_by_partition(self, sensitivity_data, threshold=0.5):
        """Select sensors based on partition"""
        try:
            self.log_info(f"Starting sensor selection, sensitivity threshold: {threshold}")
            
            sensitivity_matrix = sensitivity_data['matrix']
            all_nodes = sensitivity_data['nodes']
            partitions = sensitivity_data['partitions']
            
            # Create node index mapping
            node_to_index = {node: i for i, node in enumerate(all_nodes)}
            
            selected_sensors = {}
            
            for partition_id, partition_nodes in partitions.items():
                self.log_info(f"Processing partition {partition_id}: {len(partition_nodes)} nodes")
                
                # Get indices of nodes within partition
                partition_indices = [node_to_index[node] for node in partition_nodes if node in node_to_index]
                
                if len(partition_indices) < 2:
                    self.log_info(f"Partition {partition_id} has less than 2 nodes, skipping")
                    continue
                
                # Calculate influence score of each node (number of detectable nodes)
                influence_scores = {}
                for i, node_idx in enumerate(partition_indices):
                    node_name = all_nodes[node_idx]
                    
                    # Only consider sensitivity within same partition
                    partition_sensitivities = sensitivity_matrix[node_idx, partition_indices]
                    detectable_count = np.sum(partition_sensitivities > threshold)
                    
                    influence_scores[node_name] = {
                        'index': node_idx,
                        'detectable_count': detectable_count,
                        'avg_sensitivity': np.mean(partition_sensitivities)
                    }
                
                # Dynamically determine sensor count to ensure resilience
                partition_size = len(partition_nodes)
                min_sensors = max(2, int(partition_size * self.default_params['min_sensor_ratio']))
                max_sensors = min(10, max(3, int(partition_size * self.default_params['max_sensor_ratio'])))
                target_coverage = self.default_params['target_coverage']

                self.log_info(f"Partition {partition_id} dynamic sensor range: {min_sensors}-{max_sensors}")

                # Greedy algorithm to select sensors
                uncovered_indices = set(partition_indices)
                selected_sensors[partition_id] = []

                # Phase 1: Select sensors based on coverage
                while (uncovered_indices and
                       len(selected_sensors[partition_id]) < max_sensors and
                       len(uncovered_indices) / len(partition_indices) > (1 - target_coverage)):

                    best_sensor = None
                    best_coverage = 0

                    for node_name, info in influence_scores.items():
                        if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                            continue  # Already selected sensors

                        node_idx = info['index']
                        # Calculate how many uncovered nodes can be covered
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
                    covered_indices = []
                    for idx in uncovered_indices:
                        if sensitivity_matrix[sensor_idx, idx] > threshold:
                            covered_indices.append(idx)

                    for idx in covered_indices:
                        uncovered_indices.discard(idx)

                # Phase 2: Ensure minimum sensor count is reached (resilience guarantee)
                while len(selected_sensors[partition_id]) < min_sensors:
                    best_sensor = None
                    max_diversity = 0

                    for node_name, info in influence_scores.items():
                        if node_name in [s['node'] for s in selected_sensors[partition_id]]:
                            continue

                        node_idx = info['index']
                        # Calculate diversity (distance) with existing sensors
                        diversity_score = 0
                        for existing_sensor in selected_sensors[partition_id]:
                            existing_idx = existing_sensor['index']
                            # Use sensitivity difference as distance metric
                            distance = 1 - sensitivity_matrix[node_idx, existing_idx]
                            diversity_score += distance

                        # Average diversity score
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

                    if best_sensor is not None:
                        selected_sensors[partition_id].append(best_sensor)
                    else:
                        # If no more nodes, randomly select remaining nodes
                        remaining_nodes = [node for node in partition_nodes
                                         if node not in [s['node'] for s in selected_sensors[partition_id]]]
                        if remaining_nodes:
                            node_name = remaining_nodes[0]
                            if node_name in influence_scores:
                                info = influence_scores[node_name]
                                selected_sensors[partition_id].append({
                                    'node': node_name,
                                    'index': info['index'],
                                    'coverage': 0,
                                    'influence_score': info['detectable_count'],
                                    'avg_sensitivity': info['avg_sensitivity']
                                })
                        else:
                            break
                
                self.log_info(f"Partition {partition_id} selected {len(selected_sensors[partition_id])} sensors")
            
            return selected_sensors

        except Exception as e:
            error_msg = f"Sensor selection failed: {str(e)}"
            self.log_error(error_msg)
            return None

    def evaluate_resilience(self, selected_sensors, sensitivity_data, threshold=0.5):
        """Evaluate sensor placement resilience - detailed version"""
        try:
            self.log_info("Starting detailed resilience evaluation")

            sensitivity_matrix = sensitivity_data['matrix']
            all_nodes = sensitivity_data['nodes']
            partitions = sensitivity_data['partitions']

            # Create node index mapping
            node_to_index = {node: i for i, node in enumerate(all_nodes)}

            resilience_results = {}

            for partition_id, sensors in selected_sensors.items():
                partition_nodes = partitions[partition_id]
                partition_indices = [node_to_index[node] for node in partition_nodes if node in node_to_index]

                self.log_info(f"Evaluating partition {partition_id} resilience: {len(sensors)} sensors, {len(partition_nodes)} nodes")

                # Detailed scenario analysis
                detailed_scenarios = []

                # 1. All sensors operating normally
                all_sensor_indices = [s['index'] for s in sensors]
                all_sensor_nodes = [s['node'] for s in sensors]
                full_detected_count, full_coverage_rate = self._calculate_detailed_detection(
                    all_sensor_indices, partition_indices, sensitivity_matrix, threshold, all_sensor_nodes
                )

                detailed_scenarios.append({
                    'scenario_type': 'All sensors normal',
                    'failed_sensors': [],
                    'remaining_sensors': [s['node'] for s in sensors],
                    'detected_nodes': full_detected_count,
                    'total_nodes': len(partition_nodes),
                    'coverage_rate': full_coverage_rate,
                    'coverage_percentage': f"{full_coverage_rate*100:.1f}%",
                    'threshold_used': threshold
                })

                # 2. Sensor failure scenarios (1 to M-1 failures)
                total_failure_coverage = 0.0
                failure_scenario_count = 0

                for failure_count in range(1, len(sensors)):
                    failure_combinations = list(combinations(range(len(sensors)), failure_count))

                    for failed_indices in failure_combinations:
                        # Determine failed and remaining sensors
                        failed_sensors = [sensors[i]['node'] for i in failed_indices]
                        remaining_sensor_indices = [
                            sensors[i]['index'] for i in range(len(sensors))
                            if i not in failed_indices
                        ]
                        remaining_sensors = [
                            sensors[i]['node'] for i in range(len(sensors))
                            if i not in failed_indices
                        ]

                        # Calculate remaining sensors' detection capability
                        detected_count, coverage_rate = self._calculate_detailed_detection(
                            remaining_sensor_indices, partition_indices, sensitivity_matrix, threshold, remaining_sensors
                        )

                        total_failure_coverage += coverage_rate
                        failure_scenario_count += 1

                        detailed_scenarios.append({
                            'scenario_type': f'{failure_count} sensor(s) failed',
                            'failed_sensors': failed_sensors,
                            'remaining_sensors': remaining_sensors,
                            'detected_nodes': detected_count,
                            'total_nodes': len(partition_nodes),
                            'coverage_rate': coverage_rate,
                            'coverage_percentage': f"{coverage_rate*100:.1f}%",
                            'threshold_used': threshold
                        })

                # Calculate average resilience score (only considering failure scenarios)
                avg_failure_resilience = total_failure_coverage / failure_scenario_count if failure_scenario_count > 0 else 0.0

                resilience_results[partition_id] = {
                    'detailed_scenarios': detailed_scenarios,
                    'resilience_score': avg_failure_resilience,
                    'sensor_count': len(sensors),
                    'full_coverage_rate': full_coverage_rate,
                    'avg_failure_coverage': avg_failure_resilience,
                    'total_scenarios': len(detailed_scenarios),
                    'failure_scenarios': failure_scenario_count
                }

                self.log_info(f"Partition {partition_id} resilience score: {avg_failure_resilience:.4f}")

            return resilience_results

        except Exception as e:
            error_msg = f"Resilience evaluation failed: {str(e)}"
            self.log_error(error_msg)
            return None

    def _calculate_detection_rate(self, sensor_indices, target_indices, sensitivity_matrix, threshold):
        """Calculate detection rate"""
        if not sensor_indices:
            return 0.0

        detected_count = 0
        for target_idx in target_indices:
            # Check if any sensor can detect this node
            for sensor_idx in sensor_indices:
                if sensitivity_matrix[sensor_idx, target_idx] > threshold:
                    detected_count += 1
                    break

        return detected_count / len(target_indices) if target_indices else 0.0

    def _calculate_detailed_detection(self, sensor_indices, target_indices, sensitivity_matrix, threshold, sensor_nodes=None):
        """Calculate detailed detection info: returns detected node count and coverage rate"""
        if not sensor_indices or not target_indices:
            return 0, 0.0

        detected_count = 0
        for target_idx in target_indices:
            # Check if any sensor can detect this node
            for i, sensor_idx in enumerate(sensor_indices):
                # Get this sensor's threshold (supports custom thresholds)
                sensor_threshold = threshold
                if (self.default_params.get('enable_custom_thresholds', False) and
                    sensor_nodes and i < len(sensor_nodes)):
                    sensor_node = sensor_nodes[i]
                    sensor_threshold = self.default_params.get('custom_thresholds', {}).get(sensor_node, threshold)

                if sensitivity_matrix[sensor_idx, target_idx] > sensor_threshold:
                    detected_count += 1
                    break

        coverage_rate = detected_count / len(target_indices)
        return detected_count, coverage_rate

    def set_custom_sensor_thresholds(self, custom_thresholds):
        """Set custom sensor thresholds

        Args:
            custom_thresholds (dict): Dictionary in {sensor_node: threshold} format
        """
        self.default_params['custom_thresholds'] = custom_thresholds
        self.default_params['enable_custom_thresholds'] = bool(custom_thresholds)
        self.log_info(f"Set custom sensor thresholds: {custom_thresholds}")

    def _calculate_resilience_score(self, scenario_results):
        """Calculate resilience score"""
        # Weighted average detection rate for different failure scenarios
        total_score = 0.0
        total_weight = 0.0

        for scenario, result in scenario_results.items():
            if 'no_failure' in scenario:
                weight = 0.5  # No failure case weight
            else:
                failure_count = int(scenario.split('_')[0])
                weight = 1.0 / (failure_count + 1)  # More failures = lower weight

            total_score += result['detection_rate'] * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def optimize_sensor_placement(self, sensitivity_data, max_iterations=10):
        """Optimize sensor placement"""
        try:
            self.log_info("Starting sensor placement optimization")

            best_solution = None
            best_score = 0.0

            # Try different sensitivity thresholds (reduced count to speed up calculation)
            thresholds = [0.4, 0.5, 0.6]

            for threshold in thresholds:
                self.log_info(f"Trying threshold: {threshold}")

                # Select sensors
                selected_sensors = self.select_sensors_by_partition(sensitivity_data, threshold)
                if not selected_sensors:
                    continue

                # Evaluate resilience
                resilience_results = self.evaluate_resilience(selected_sensors, sensitivity_data, threshold)
                if not resilience_results:
                    continue

                # Calculate total score
                total_score = self._calculate_total_score(selected_sensors, resilience_results)

                self.log_info(f"Total score for threshold {threshold}: {total_score:.4f}")

                if total_score > best_score:
                    best_score = total_score
                    best_solution = {
                        'sensors': selected_sensors,
                        'resilience': resilience_results,
                        'threshold': threshold,
                        'score': total_score
                    }

            if best_solution:
                self.log_info(f"Optimal solution: threshold={best_solution['threshold']}, score={best_solution['score']:.4f}")

            return best_solution

        except Exception as e:
            error_msg = f"Sensor placement optimization failed: {str(e)}"
            self.log_error(error_msg)
            return None

    def _calculate_total_score(self, selected_sensors, resilience_results):
        """Calculate total score"""
        # Sensor count penalty (reduced penalty weight)
        total_sensors = sum(len(sensors) for sensors in selected_sensors.values())
        sensor_penalty = total_sensors * 0.001  # Deduct 0.001 points per sensor

        # Resilience score
        resilience_scores = [r['resilience_score'] for r in resilience_results.values()]
        avg_resilience = np.mean(resilience_scores) if resilience_scores else 0.0

        # Coverage score (based on sensor count and partition coverage)
        partition_count = len(selected_sensors)
        coverage_score = min(1.0, total_sensors / (partition_count * 2))  # Ideally 2 sensors per partition

        # Total score: resilience weight 40%, coverage weight 60%
        resilience_weight = self.default_params['resilience_weight']
        coverage_weight = self.default_params['coverage_weight']

        total_score = (avg_resilience * resilience_weight +
                      coverage_score * coverage_weight -
                      sensor_penalty)

        return max(0.001, total_score)  # Ensure minimum score is 0.001

    def _save_detailed_resilience_analysis(self, solution, conversation_id):
        """Save detailed resilience analysis results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resilience_analysis_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            analysis_data = []

            for partition_id, resilience_info in solution['resilience'].items():
                detailed_scenarios = resilience_info.get('detailed_scenarios', [])

                for scenario in detailed_scenarios:
                    analysis_data.append({
                        'Partition Number': partition_id,
                        'Scenario Type': scenario['scenario_type'],
                        'Failed Sensors': ', '.join(scenario['failed_sensors']) if scenario['failed_sensors'] else 'None',
                        'Remaining Sensors': ', '.join(scenario['remaining_sensors']),
                        'Detected Node Count': scenario['detected_nodes'],
                        'Total Node Count': scenario['total_nodes'],
                        'Coverage Rate': f"{scenario['coverage_rate']:.4f}",
                        'Coverage Percentage': scenario['coverage_percentage'],
                        'Sensitivity Threshold': scenario['threshold_used'],
                        'Partition Total Sensors': resilience_info['sensor_count'],
                        'Partition Avg Resilience': f"{resilience_info['resilience_score']:.4f}"
                    })

            # Create DataFrame and save
            df = pd.DataFrame(analysis_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            self.log_info(f"Detailed resilience analysis saved to: {filepath}")
            return filepath

        except Exception as e:
            self.log_error(f"Save resilience analysis failed: {str(e)}")
            return None

    def save_sensor_results(self, solution, inp_file_path, conversation_id):
        """Save sensor placement results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensor_placement_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # Load network model to get coordinates
            wn = wntr.network.WaterNetworkModel(inp_file_path)

            # Prepare data
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

                    # Get resilience info
                    resilience_info = solution['resilience'].get(partition_id, {})
                    resilience_score = resilience_info.get('resilience_score', 0.0)

                    results_data.append({
                        'Sensor ID': f'S{sensor_id:03d}',
                        'Node Name': node_name,
                        'Partition Number': partition_id,
                        'X Coordinate': coord[0],
                        'Y Coordinate': coord[1],
                        'Influence Score': sensor['influence_score'],
                        'Avg Sensitivity': f"{sensor['avg_sensitivity']:.4f}",
                        'Covered Node Count': sensor['coverage'],
                        'Partition Resilience Score': f"{resilience_score:.4f}",
                        'Sensitivity Threshold': solution['threshold']
                    })
                    sensor_id += 1

            # Create DataFrame and save
            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            # Save detailed resilience analysis
            self._save_detailed_resilience_analysis(solution, conversation_id)

            # Generate statistics report
            stats = self._generate_statistics(solution)

            file_size = os.path.getsize(filepath)

            self.log_info(f"Sensor placement results saved to: {filepath}")

            return {
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'sensor_count': len(results_data),
                'statistics': stats,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"Save sensor results failed: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_statistics(self, solution):
        """Generate statistics info"""
        stats = {
            'total_sensors': 0,
            'partitions': len(solution['sensors']),
            'avg_resilience': 0.0,
            'threshold': solution['threshold'],
            'total_score': solution['score']
        }

        # Calculate total sensor count
        for sensors in solution['sensors'].values():
            stats['total_sensors'] += len(sensors)

        # Calculate average resilience
        if solution['resilience']:
            resilience_scores = [r['resilience_score'] for r in solution['resilience'].values()]
            stats['avg_resilience'] = np.mean(resilience_scores)

        # Partition details
        stats['partition_details'] = {}
        for partition_id, sensors in solution['sensors'].items():
            resilience_info = solution['resilience'].get(partition_id, {})
            stats['partition_details'][partition_id] = {
                'sensor_count': len(sensors),
                'resilience_score': resilience_info.get('resilience_score', 0.0)
            }

        return stats

    def generate_visualization(self, solution, inp_file_path, conversation_id):
        """Generate sensor placement visualization"""
        try:
            # Set matplotlib to use English fonts
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False

            # Load network model
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            G = wn.to_graph().to_undirected()

            # Prepare node positions
            pos = {}
            layout = None

            for node in G.nodes():
                try:
                    coord = wn.get_node(node).coordinates
                    if coord is None or coord == (0, 0):
                        if layout is None:
                            import networkx as nx
                            layout = nx.spring_layout(G, seed=42)
                        coord = layout.get(node, (0, 0))
                except:
                    if layout is None:
                        import networkx as nx
                        layout = nx.spring_layout(G, seed=42)
                    coord = layout.get(node, (0, 0))
                pos[node] = coord

            # Create figure
            plt.figure(figsize=(15, 12))

            # Draw network edges
            import networkx as nx
            nx.draw_networkx_edges(G, pos=pos, alpha=0.3, width=0.5, edge_color='gray')

            # Draw regular nodes
            all_nodes = list(G.nodes())
            nx.draw_networkx_nodes(G, pos=pos, nodelist=all_nodes,
                                 node_color='lightblue', node_size=20, alpha=0.6)

            # Draw sensor nodes
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
            sensor_nodes_by_partition = {}

            for partition_id, sensors in solution['sensors'].items():
                sensor_nodes = [s['node'] for s in sensors]
                sensor_nodes_by_partition[partition_id] = sensor_nodes

                color = colors[partition_id % len(colors)]
                nx.draw_networkx_nodes(G, pos=pos, nodelist=sensor_nodes,
                                     node_color=color, node_size=100, alpha=0.8,
                                     label=f'Partition {partition_id} Sensors')

            # Add legend and title
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Sensor Placement Results\n'
                     f'Total Sensors: {sum(len(s) for s in solution["sensors"].values())}, '
                     f'Threshold: {solution["threshold"]}, '
                     f'Score: {solution["score"]:.4f}')
            plt.axis('off')

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"sensor_placement_viz_{conversation_id[:8]}_{timestamp}.png"
            viz_path = os.path.join(self.downloads_folder, viz_filename)

            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.log_info(f"Sensor placement visualization saved to: {viz_path}")

            return viz_path

        except Exception as e:
            error_msg = f"Generate visualization failed: {str(e)}"
            self.log_error(error_msg)
            return None

    def process(self, inp_file_path, partition_csv_path, user_message, conversation_id):
        """Main processing function"""
        try:
            self.log_info(f"Starting sensor optimization placement: {user_message}")
            self.log_info(f"Input file: {inp_file_path}")
            self.log_info(f"Partition file: {partition_csv_path}")

            # Step 1: Load partition results
            self.log_info("Step 1: Loading partition results...")
            partitions = self.load_partition_results(partition_csv_path)
            if not partitions:
                self.log_error("Partition results load failed")
                return {
                    'success': False,
                    'response': "Partition results load failed",
                    'error': "Unable to load partition CSV file"
                }

            self.log_info(f"Partition loaded successfully, total {len(partitions)} partitions")

            # Step 2: Calculate pressure sensitivity matrix
            self.log_info("Step 2: Calculating pressure sensitivity matrix...")
            sensitivity_data = self.compute_pressure_sensitivity_matrix(
                inp_file_path, partitions, self.default_params['demand_ratios']
            )

            if not sensitivity_data:
                self.log_error("Pressure sensitivity matrix calculation failed")
                return {
                    'success': False,
                    'response': "Pressure sensitivity matrix calculation failed",
                    'error': "Error occurred during sensitivity calculation"
                }

            self.log_info("Sensitivity matrix calculation successful")

            # Step 3: Optimize sensor placement
            self.log_info("Step 3: Optimizing sensor placement...")
            solution = self.optimize_sensor_placement(sensitivity_data)

            if not solution:
                self.log_error("Sensor placement optimization failed")
                return {
                    'success': False,
                    'response': "Sensor placement optimization failed",
                    'error': "Error occurred during optimization"
                }

            self.log_info("Sensor placement optimization successful")

            # Step 4: Save results
            save_result = self.save_sensor_results(solution, inp_file_path, conversation_id)

            # Step 5: Generate visualization
            viz_path = self.generate_visualization(solution, inp_file_path, conversation_id)

            # Step 6: Generate analysis report
            stats = save_result.get('statistics', {})

            response_text = f"""
Sensor optimization placement complete!

📊 **Placement Overview**
- Total sensors: {stats.get('total_sensors', 0)}
- Partition count: {stats.get('partitions', 0)}
- Sensitivity threshold: {stats.get('threshold', 0.5)}
- Total score: {stats.get('total_score', 0.0):.4f}

📈 **Resilience Analysis**
- Avg resilience score: {stats.get('avg_resilience', 0.0):.4f}
- Perturbation ratios: {self.default_params['demand_ratios']}

🎯 **Partition Details**
"""

            for partition_id, details in stats.get('partition_details', {}).items():
                response_text += f"- Partition {partition_id}: {details['sensor_count']} sensors (resilience: {details['resilience_score']:.4f})\n"

            response_text += f"""
✅ Sensor placement optimization strategy:
1. Sensor selection based on pressure sensitivity matrix
2. Resilience evaluation considering multiple failure scenarios
3. Ensure at least 2 sensors per partition
4. Optimize balance between detection coverage and sensor count

📁 Result files saved, containing detailed sensor locations and performance metrics
"""

            # Generate professional prompt for GPT analysis
            prompt = self._build_sensor_placement_prompt(solution, stats, user_message, save_result)

            result = {
                'success': True,
                'response': response_text,
                'prompt': prompt,  # Add professional prompt for GPT analysis
                'solution': solution,
                'statistics': stats
            }

            # Add file download info
            if save_result['success']:
                result['csv_info'] = save_result

            # Add resilience analysis file info
            resilience_csv_path = self._save_detailed_resilience_analysis(solution, conversation_id)
            if resilience_csv_path:
                result['resilience_csv_info'] = resilience_csv_path

            if viz_path:
                result['visualization'] = {
                    'filename': os.path.basename(viz_path),
                    'path': viz_path
                }

            return result

        except Exception as e:
            error_msg = f"Sensor optimization placement failed: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'response': error_msg,
                'error': str(e)
            }

    def _build_sensor_placement_prompt(self, solution, stats, user_message, save_result):
        """Build professional prompt for sensor placement analysis"""

        # Get sensor detailed info
        sensors_info = []
        for partition_id, sensors in solution['sensors'].items():
            for sensor in sensors:
                sensors_info.append(f"Partition {partition_id}: Node {sensor['node']} (sensitivity: {sensor.get('avg_sensitivity', 0):.4f})")

        # Get resilience analysis details
        resilience_details = []
        for partition_id, resilience_data in solution['resilience'].items():
            resilience_details.append(f"Partition {partition_id}: Resilience score {resilience_data['resilience_score']:.4f}, {resilience_data['sensor_count']} sensors")

        prompt = f"""
User request: {user_message}

## Sensor Optimization Placement Analysis Report

### 📊 Placement Overview
- **Total sensors**: {stats.get('total_sensors', 0)}
- **Partition count**: {stats.get('partitions', 0)}
- **Optimal sensitivity threshold**: {stats.get('threshold', 0.5)}
- **Total score**: {stats.get('total_score', 0.0):.4f}
- **Avg resilience score**: {stats.get('avg_resilience', 0.0):.4f}

### 🎯 Sensor Placement Details
{chr(10).join(sensors_info)}

### 📈 Resilience Analysis Results
{chr(10).join(resilience_details)}

### 🔧 Technical Parameters
- **Perturbation ratios**: {self.default_params['demand_ratios']}
- **Resilience weight**: {self.default_params['resilience_weight']}
- **Coverage weight**: {self.default_params['coverage_weight']}

### 📁 Generated Files
- **Sensor placement results**: {save_result.get('filename', 'N/A')}
- **File size**: {save_result.get('file_size', 0)} bytes
- **Record count**: {save_result.get('sensor_count', 0)} records

### 🎯 Optimization Strategy Description
1. **Pressure sensitivity analysis**: Calculate pressure sensitivity matrix between nodes based on demand perturbation
2. **Partition optimization**: Select nodes with maximum influence within each partition as sensor locations
3. **Resilience evaluation**: Consider sensor failure scenarios to ensure system works normally with partial sensor failures
4. **Multi-objective balance**: Find optimal balance between detection coverage, resilience and sensor count

Based on the above technical analysis, please provide professional interpretation and suggestions for the sensor placement solution. Focus on:
1. Scientific validity and reasonableness of sensor placement
2. Importance and effectiveness of resilience design
3. Characteristics of sensor configuration in each partition
4. Practical application considerations and suggestions

Please use the following signature format at the end of your reply:

Best regards,

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

        return prompt
