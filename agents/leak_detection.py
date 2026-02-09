"""
Leak detection agent - LeakDetectionAgent
Water network leak detection system based on pressure sensitivity analysis and machine learning

Main functions:
1. Data preparation: Check partition and sensor configuration, call other agents when missing
2. Sensitivity calculation: Simulate leak scenarios, calculate pressure sensitivity matrix
3. Data generation: Generate balanced training dataset (Anomaly + Normal)
4. Model training: Use MLP for leak detection model training
5. Inference prediction: Perform leak detection on new sensor data

Author: LeakAgent Team
Date: 2025-09-18
"""

import os
import sys
import json
import uuid
import random
import logging
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Machine learning related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Water network analysis
import wntr
import matplotlib.pyplot as plt
import seaborn as sns

# Base agent
from .base_agent import BaseAgent

class LeakDetectionMLP(nn.Module):
    """Leak detection multi-layer perceptron model"""
    
    def __init__(self, input_size: int, num_partitions: int, hidden_sizes: List[int] = [128, 64, 32], num_classes: int = None):
        super(LeakDetectionMLP, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes if num_classes is not None else (num_partitions + 1)  # +1 for normal class (0)
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class LeakDetectionAgent(BaseAgent):
    """Leak detection agent"""
    
    def __init__(self):
        super().__init__("LeakDetectionAgent")
        self.agent_name = "LeakDetectionAgent"
        self.downloads_folder = "downloads"
        self.uploads_folder = "uploads"
        
        # Ensure download folder exists
        os.makedirs(self.downloads_folder, exist_ok=True)
        
        # Model related
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data cache
        self.partition_data = None
        self.sensor_data = None
        self.network_model = None
        
        self.log_info(f"Leak detection agent initialization complete, using device: {self.device}")
    
    def check_dependencies(self, conversation_id: str, inp_file_path: str = None) -> Dict[str, Any]:
        """Intelligently check partition and sensor config files, prioritize reusing existing files"""
        try:
            self.log_info("🔍 Intelligently checking partition and sensor config files...")

            # Find related files
            partition_file = None
            sensor_file = None
            sensor_files = []  # Store all found sensor files

            # Scan download folder
            if os.path.exists(self.downloads_folder):
                for filename in os.listdir(self.downloads_folder):
                    if conversation_id[:8] in filename:
                        if 'partition_results' in filename and filename.endswith('.csv'):
                            partition_file = os.path.join(self.downloads_folder, filename)
                            self.log_info(f"✅ Found partition file: {os.path.basename(partition_file)}")
                        elif 'sensor_placement' in filename and filename.endswith('.csv'):
                            sensor_file_path = os.path.join(self.downloads_folder, filename)
                            sensor_files.append(sensor_file_path)

            # Select latest sensor file
            if sensor_files:
                # Sort by filename, select the latest
                sensor_files.sort()
                sensor_file = sensor_files[-1]
                self.log_info(f"✅ Found sensor placement file: {os.path.basename(sensor_file)}")

                # Display sensor info
                try:
                    sensor_df = pd.read_csv(sensor_file)
                    if 'Node ID' in sensor_df.columns:
                        sensor_nodes = sensor_df['Node ID'].tolist()
                        self.log_info(f"📍 Detected {len(sensor_nodes)} sensor nodes: {sensor_nodes}")
                    elif 'Node' in sensor_df.columns:
                        sensor_nodes = sensor_df['Node'].tolist()
                        self.log_info(f"📍 Detected {len(sensor_nodes)} sensor nodes: {sensor_nodes}")
                    else:
                        self.log_warning("Sensor file format anomaly, unable to read node info")
                except Exception as e:
                    self.log_warning(f"Failed to read sensor file info: {str(e)}")

            result = {
                'partition_file': partition_file,
                'sensor_file': sensor_file,
                'missing_files': [],
                'success': True,
                'reused_files': []
            }

            # Intelligently handle missing files
            missing_files = []

            # Check sensor file
            if sensor_file:
                result['reused_files'].append('sensor_placement')
                self.log_info("♻️ Reusing existing sensor placement, no need to regenerate")
            else:
                missing_files.append('sensor_placement')
                self.log_warning("⚠️ Sensor placement file not found")

            # Check partition file
            if partition_file:
                result['reused_files'].append('partition_results')
                self.log_info("♻️ Reusing existing partition results, no need to regenerate")
            else:
                missing_files.append('partition_results')
                self.log_warning("⚠️ Partition results file not found")

            # Special handling: if there is sensor file but no partition file, try to infer partition info from sensor file
            if sensor_file and not partition_file:
                self.log_info("🧠 Trying to infer partition info from sensor file...")
                inferred_partition = self._infer_partition_from_sensors(sensor_file, conversation_id)
                if inferred_partition.get('success'):
                    result['partition_file'] = inferred_partition['partition_file']
                    if 'partition_results' in missing_files:
                        missing_files.remove('partition_results')
                    result['reused_files'].append('partition_results_inferred')
                    self.log_info("✅ Successfully inferred partition info from sensor file")

            # Only generate when truly missing and INP file is provided
            if missing_files and inp_file_path:
                self.log_info(f"🔧 Need to generate missing files: {missing_files}")

                # Intelligent generation strategy: prioritize keeping existing files unchanged
                generated_files = self._generate_missing_files_smart(missing_files, inp_file_path, conversation_id, sensor_file)

                if generated_files.get('success'):
                    # Update results
                    if 'partition_results' in missing_files and generated_files.get('partition_file'):
                        result['partition_file'] = generated_files['partition_file']
                        missing_files.remove('partition_results')

                    if 'sensor_placement' in missing_files and generated_files.get('sensor_file'):
                        result['sensor_file'] = generated_files['sensor_file']
                        missing_files.remove('sensor_placement')
                else:
                    self.log_error("Auto-generate dependency files failed")
                    result['success'] = False
                    result['error'] = generated_files.get('error', 'Unknown error')
                    return result

            result['missing_files'] = missing_files

            if missing_files:
                self.log_error(f"❌ Still missing files: {missing_files}")
                result['success'] = False
                result['error'] = f"Missing required config files: {missing_files}"
                return result

            # Load data
            self.partition_data = pd.read_csv(result['partition_file'])
            self.sensor_data = pd.read_csv(result['sensor_file'])

            self.log_info(f"📂 Successfully loaded partition file: {os.path.basename(result['partition_file'])}")
            self.log_info(f"📂 Successfully loaded sensor file: {os.path.basename(result['sensor_file'])}")

            if result['reused_files']:
                self.log_info(f"♻️ Reused files: {', '.join(result['reused_files'])}")

            return result

        except Exception as e:
            error_msg = f"Check dependency files failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def _infer_partition_from_sensors(self, sensor_file: str, conversation_id: str) -> Dict[str, Any]:
        """Infer partition info from sensor file"""
        try:
            self.log_info("Inferring partition info from sensor placement file...")

            # Read sensor data
            sensor_df = pd.read_csv(sensor_file)

            # Check if it contains partition info
            partition_col = None
            node_col = None

            # Identify column names
            for col in sensor_df.columns:
                if col in ['Partition Number', 'partition', 'Partition']:
                    partition_col = col
                if col in ['Node ID', 'node_id', 'Node', 'node']:
                    node_col = col

            if partition_col and node_col:
                # Create simplified partition file
                partition_data = []
                for _, row in sensor_df.iterrows():
                    node_id = row[node_col]
                    partition_id = row[partition_col]
                    partition_data.append({
                        'Node': node_id,
                        'Partition': partition_id
                    })

                if partition_data:
                    # Save inferred partition file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    partition_filename = f"partition_results_{conversation_id[:8]}_{timestamp}_inferred.csv"
                    partition_filepath = os.path.join(self.downloads_folder, partition_filename)

                    partition_df = pd.DataFrame(partition_data)
                    partition_df.to_csv(partition_filepath, index=False)

                    self.log_info(f"✅ Successfully inferred and saved partition file: {partition_filename}")
                    return {
                        'success': True,
                        'partition_file': partition_filepath,
                        'method': 'inferred_from_sensors'
                    }

            self.log_warning("Partition info not found in sensor file, cannot infer")
            return {'success': False, 'error': 'Partition info not found in sensor file'}

        except Exception as e:
            self.log_error(f"Failed to infer partition info from sensor file: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_missing_files_smart(self, missing_files: List[str], inp_file_path: str,
                                    conversation_id: str, existing_sensor_file: str = None) -> Dict[str, Any]:
        """Intelligently generate missing files, prioritize keeping existing files unchanged"""
        try:
            self.log_info("🔧 Intelligently generating missing files...")

            from agents.partition_sim import PartitionSim
            from agents.sensor_placement import SensorPlacement

            result = {'success': True}

            # If missing partition file, call partition agent
            if 'partition_results' in missing_files:
                self.log_info("🔧 Generating partition config...")

                partition_agent = PartitionSim()
                partition_result = partition_agent.process(
                    inp_file_path=inp_file_path,
                    user_message="Auto partition into 3 regions, using FCM clustering algorithm",
                    conversation_id=conversation_id
                )

                if partition_result.get('success'):
                    # Find generated partition file
                    partition_file = None
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'partition_results' in filename and
                                filename.endswith('.csv')):
                                partition_file = os.path.join(self.downloads_folder, filename)
                                break

                    if partition_file:
                        result['partition_file'] = partition_file
                        self.log_info("✅ Partition config generated successfully")
                    else:
                        self.log_error("Partition file not found after generation")
                        result['success'] = False
                        result['error'] = "Partition file not found after generation"
                        return result
                else:
                    self.log_error(f"Partition config generation failed: {partition_result.get('response', 'Unknown error')}")
                    result['success'] = False
                    result['error'] = f"Partition config generation failed: {partition_result.get('response', 'Unknown error')}"
                    return result

            # If missing sensor file, call sensor placement agent
            if 'sensor_placement' in missing_files:
                self.log_info("🔧 Generating sensor config...")

                # If sensor file already exists, this shouldn't happen
                if existing_sensor_file:
                    self.log_warning("⚠️ Detected existing sensor file, but still in missing list, this may be a logic error")
                    result['sensor_file'] = existing_sensor_file
                    return result

                sensor_agent = SensorPlacement()

                # Ensure partition file exists (may have just been generated)
                partition_file = result.get('partition_file')
                if not partition_file:
                    # Rescan for partition file
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'partition_results' in filename and
                                filename.endswith('.csv')):
                                partition_file = os.path.join(self.downloads_folder, filename)
                                break

                if not partition_file:
                    self.log_error("Sensor placement requires partition file, but not found")
                    result['success'] = False
                    result['error'] = "Sensor placement requires partition file, but not found"
                    return result

                sensor_result = sensor_agent.process(
                    inp_file_path=inp_file_path,
                    partition_csv_path=partition_file,
                    user_message="Auto place sensors, using genetic algorithm optimization",
                    conversation_id=conversation_id
                )

                if sensor_result.get('success'):
                    # Find generated sensor file
                    sensor_file = None
                    if os.path.exists(self.downloads_folder):
                        for filename in os.listdir(self.downloads_folder):
                            if (conversation_id[:8] in filename and
                                'sensor_placement' in filename and
                                filename.endswith('.csv')):
                                sensor_file = os.path.join(self.downloads_folder, filename)
                                break

                    if sensor_file:
                        result['sensor_file'] = sensor_file
                        self.log_info("✅ Sensor config generated successfully")
                    else:
                        self.log_error("Sensor file not found after generation")
                        result['success'] = False
                        result['error'] = "Sensor file not found after generation"
                        return result
                else:
                    self.log_error(f"Sensor config generation failed: {sensor_result.get('response', 'Unknown error')}")
                    result['success'] = False
                    result['error'] = f"Sensor config generation failed: {sensor_result.get('response', 'Unknown error')}"
                    return result

            return result

        except Exception as e:
            self.log_error(f"Intelligently generate missing files failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_missing_files(self, missing_files: List[str], inp_file_path: str, conversation_id: str) -> Dict[str, Any]:
        """Call other agents to generate missing config files (Compatibility method)"""
        self.log_info("⚠️ Using compatibility method to generate missing files, recommend using intelligent generation method")
        return self._generate_missing_files_smart(missing_files, inp_file_path, conversation_id)

    def load_network_model(self, inp_file_path: str) -> bool:
        """Load water network model"""
        try:
            self.log_info(f"Loading water network model: {inp_file_path}")
            self.network_model = wntr.network.WaterNetworkModel(inp_file_path)
            
            # Get network basic info
            num_nodes = len(self.network_model.node_name_list)
            num_junctions = len(self.network_model.junction_name_list)
            num_links = len(self.network_model.link_name_list)
            
            self.log_info(f"Network loaded successfully: {num_nodes} nodes, {num_junctions} demand nodes, {num_links} links")
            return True
            
        except Exception as e:
            self.log_error(f"Load network model failed: {str(e)}")
            return False
    
    def calculate_centrality(self, demand_nodes: List[str]) -> Dict[str, float]:
        """Calculate node network centrality"""
        try:
            # Convert to NetworkX graph
            G = self.network_model.to_graph().to_undirected()
            
            # Calculate various centralities
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            # Combined centrality score
            centrality_scores = {}
            for node in demand_nodes:
                if node in G.nodes():
                    score = (
                        degree_centrality.get(node, 0) + 
                        betweenness_centrality.get(node, 0) + 
                        closeness_centrality.get(node, 0)
                    ) / 3
                    centrality_scores[node] = score
                else:
                    centrality_scores[node] = 0
            
            return centrality_scores
            
        except Exception as e:
            self.log_error(f"Calculate network centrality failed: {str(e)}")
            return {}
    
    def get_total_demand(self, node_name: str) -> float:
        """Get node's total demand"""
        try:
            node = self.network_model.get_node(node_name)
            total_demand = 0
            
            for demand_ts in node.demand_timeseries_list:
                total_demand += abs(demand_ts.base_value)
            
            return total_demand
            
        except Exception as e:
            self.log_error(f"Get node demand failed: {str(e)}")
            return 0
    
    def select_critical_nodes(self, num_scenarios: int) -> List[str]:
        """Select critical nodes for leak simulation"""
        try:
            self.log_info(f"Selecting {num_scenarios} critical nodes for leak simulation...")
            
            # Get demand nodes and sensor nodes
            demand_nodes = self.network_model.junction_name_list

            # Get sensor nodes, try different column names
            sensor_nodes = []
            if self.sensor_data is not None:
                if 'Node ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['Node ID'].tolist()
                elif 'Node ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['Node ID'].tolist()
                else:
                    # If neither exists, try first column
                    sensor_nodes = self.sensor_data.iloc[:, 0].tolist()
            
            # Strategy 1: High demand nodes (sorted by demand, take top 50%)
            demand_ranking = sorted(demand_nodes, 
                                  key=lambda x: self.get_total_demand(x), 
                                  reverse=True)
            high_demand_nodes = demand_ranking[:len(demand_ranking)//2]
            
            # Strategy 2: Network central nodes
            centrality_scores = self.calculate_centrality(demand_nodes)
            central_nodes = sorted(demand_nodes, 
                                 key=lambda x: centrality_scores.get(x, 0), 
                                 reverse=True)[:len(demand_nodes)//2]
            
            # Strategy 3: Prioritize non-sensor nodes
            non_sensor_nodes = [node for node in demand_nodes if node not in sensor_nodes]
            
            # Combined selection: Prioritize nodes that are both high demand and central, and non-sensor
            priority_nodes = list(set(high_demand_nodes) & set(central_nodes) & set(non_sensor_nodes))
            
            # If priority nodes not enough, add other critical nodes
            if len(priority_nodes) < num_scenarios:
                remaining_critical = list(set(high_demand_nodes + central_nodes) & set(non_sensor_nodes))
                priority_nodes.extend([n for n in remaining_critical if n not in priority_nodes])
            
            # If still not enough, add other non-sensor nodes
            if len(priority_nodes) < num_scenarios:
                other_nodes = [n for n in non_sensor_nodes if n not in priority_nodes]
                priority_nodes.extend(other_nodes)
            
            selected_nodes = priority_nodes[:num_scenarios]
            
            self.log_info(f"Selected {len(selected_nodes)} critical nodes:")
            for i, node in enumerate(selected_nodes[:5]):  # Only show first 5
                demand = self.get_total_demand(node)
                centrality = centrality_scores.get(node, 0)
                self.log_info(f"  {i+1}. {node} (demand: {demand:.3f}, centrality: {centrality:.3f})")
            
            if len(selected_nodes) > 5:
                self.log_info(f"  ... and {len(selected_nodes)-5} more nodes")
            
            return selected_nodes

        except Exception as e:
            error_msg = f"Select critical nodes failed: {str(e)}"
            self.log_error(error_msg)
            return []

    def run_hydraulic_simulation(self) -> Optional[wntr.sim.results.SimulationResults]:
        """Run hydraulic simulation"""
        try:
            sim = wntr.sim.EpanetSimulator(self.network_model)
            results = sim.run_sim()
            return results
        except Exception as e:
            self.log_error(f"Hydraulic simulation failed: {str(e)}")
            return None

    def get_sensor_pressures(self, results: wntr.sim.results.SimulationResults) -> np.ndarray:
        """Extract sensor node pressure data"""
        try:
            # Try different column names
            if 'Node ID' in self.sensor_data.columns:
                sensor_nodes = self.sensor_data['Node ID'].tolist()
            elif 'Node ID' in self.sensor_data.columns:
                sensor_nodes = self.sensor_data['Node ID'].tolist()
            else:
                # If neither exists, try first column
                sensor_nodes = self.sensor_data.iloc[:, 0].tolist()

            # Debug info: show sensor nodes and available columns
            available_columns = list(results.node['pressure'].columns)
            self.log_info(f"Sensor nodes: {sensor_nodes[:5]}... (total {len(sensor_nodes)})")
            self.log_info(f"Available pressure columns: {available_columns[:10]}... (total {len(available_columns)})")

            # Try converting sensor node names to string format
            sensor_nodes_str = [str(node) for node in sensor_nodes]

            # Check which sensor nodes exist in simulation results
            valid_sensors = []
            for sensor in sensor_nodes_str:
                if sensor in available_columns:
                    valid_sensors.append(sensor)
                else:
                    # Try different formats
                    for col in available_columns:
                        if str(col) == sensor or str(col).strip() == sensor.strip():
                            valid_sensors.append(col)
                            break

            if not valid_sensors:
                self.log_error(f"No matching sensor nodes found")
                self.log_error(f"Sensor nodes: {sensor_nodes_str}")
                self.log_error(f"Available column samples: {available_columns[:20]}")
                return np.array([])

            self.log_info(f"Found {len(valid_sensors)} valid sensors: {valid_sensors}")

            pressure_data = results.node['pressure'].loc[:, valid_sensors].values
            return pressure_data
        except Exception as e:
            self.log_error(f"Extract sensor pressure failed: {str(e)}")
            return np.array([])

    def simulate_leak(self, leak_node: str, leak_ratio: float) -> Tuple[np.ndarray, int]:
        """Simulate single node leak scenario"""
        try:
            # Save original demands
            node = self.network_model.get_node(leak_node)
            original_demands = []
            for demand_ts in node.demand_timeseries_list:
                original_demands.append(demand_ts.base_value)
                # Increase demand to simulate leak
                demand_ts.base_value = demand_ts.base_value * (1 + leak_ratio)

            # Run leak simulation
            leak_results = self.run_hydraulic_simulation()
            if leak_results is None:
                return np.array([]), 0

            # Get sensor pressures
            leak_pressures = self.get_sensor_pressures(leak_results)

            # Restore original demands
            for i, demand_ts in enumerate(node.demand_timeseries_list):
                demand_ts.base_value = original_demands[i]

            # Determine which partition the leak node belongs to
            partition_label = self.get_node_partition(leak_node)

            return leak_pressures, partition_label

        except Exception as e:
            self.log_error(f"Simulate leak failed: {str(e)}")
            return np.array([]), 0

    def get_node_partition(self, node_name: str) -> int:
        """Get node's partition"""
        try:
            if self.partition_data is not None:
                node_row = self.partition_data[self.partition_data['Node ID'] == node_name]
                if not node_row.empty:
                    return int(node_row.iloc[0]['Partition Number'])
            return 1  # Default partition
        except Exception as e:
            self.log_error(f"Get node partition failed: {str(e)}")
            return 1

    def calculate_sensitivity_matrix(self, normal_pressures: np.ndarray,
                                   leak_pressures: np.ndarray) -> np.ndarray:
        """Calculate pressure sensitivity matrix"""
        try:
            # Calculate absolute value of pressure difference
            pressure_diff = np.abs(leak_pressures - normal_pressures)

            # Normalize for each time step (avoid division by zero)
            normalized_diff = np.zeros_like(pressure_diff)

            for t in range(pressure_diff.shape[0]):
                for s in range(pressure_diff.shape[1]):
                    if normal_pressures[t, s] > 1e-6:  # Avoid division by zero
                        normalized_diff[t, s] = pressure_diff[t, s] / normal_pressures[t, s]
                    else:
                        normalized_diff[t, s] = 0

            # Calculate time average
            sensitivity_vector = np.mean(normalized_diff, axis=0)

            return sensitivity_vector

        except Exception as e:
            self.log_error(f"Calculate sensitivity matrix failed: {str(e)}")
            return np.array([])

    def add_sensor_noise(self, pressure_data: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add sensor noise"""
        try:
            # Gaussian noise: mean 0, std as percentage of pressure value
            noise = np.random.normal(0, pressure_data * noise_level)

            # Ensure pressure values are not negative
            noisy_pressure = np.maximum(pressure_data + noise, 0.1)

            return noisy_pressure

        except Exception as e:
            self.log_error(f"Add sensor noise failed: {str(e)}")
            return pressure_data

    def generate_training_data(self, num_scenarios: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate balanced training dataset"""
        try:
            self.log_info(f"Starting to generate {num_scenarios*2} training samples...")
            self.log_info(f"  - {num_scenarios} anomaly samples (leak scenarios)")
            self.log_info(f"  - {num_scenarios} normal samples (with noise)")

            # Run baseline simulation
            self.log_info("Running baseline hydraulic simulation...")
            normal_results = self.run_hydraulic_simulation()
            if normal_results is None:
                raise Exception("Baseline simulation failed")

            normal_pressures = self.get_sensor_pressures(normal_results)
            self.log_info(f"Baseline pressure data shape: {normal_pressures.shape}")

            # Select critical nodes
            critical_nodes = self.select_critical_nodes(num_scenarios)
            if len(critical_nodes) < num_scenarios:
                self.log_warning(f"Only found {len(critical_nodes)} critical nodes, less than requested {num_scenarios}")
                num_scenarios = len(critical_nodes)

            # Generate anomaly data
            self.log_info("Generating anomaly data...")
            anomaly_data = []
            anomaly_labels = []

            # To ensure enough data, generate multiple leak scenarios for each critical node
            scenarios_per_node = max(1, num_scenarios // len(critical_nodes))
            if scenarios_per_node * len(critical_nodes) < num_scenarios:
                scenarios_per_node += 1

            scenario_count = 0
            for node_idx, leak_node in enumerate(critical_nodes):
                for scenario_idx in range(scenarios_per_node):
                    if scenario_count >= num_scenarios:
                        break

                    scenario_count += 1
                    self.log_info(f"  Simulating leak {scenario_count}/{num_scenarios}: {leak_node} (scenario {scenario_idx+1})")

                    # Random leak ratio (10%-30%)
                    leak_ratio = random.uniform(0.1, 0.3)

                    # Simulate leak
                    leak_pressures, partition_label = self.simulate_leak(leak_node, leak_ratio)

                    if leak_pressures.size > 0:
                        # Calculate sensitivity vector
                        sensitivity_vector = self.calculate_sensitivity_matrix(normal_pressures, leak_pressures)

                        if sensitivity_vector.size > 0:
                            anomaly_data.append(sensitivity_vector)
                            anomaly_labels.append(partition_label)
                            self.log_info(f"    Leak ratio: {leak_ratio:.1%}, Partition: {partition_label}")

                if scenario_count >= num_scenarios:
                    break

            # Generate normal data
            self.log_info("Generating normal data...")
            normal_data = []
            normal_labels = []

            for i in range(len(anomaly_data)):  # Generate equal amount of normal data
                # Add different levels of noise
                noise_level = random.uniform(0.01, 0.03)
                noisy_pressures = self.add_sensor_noise(normal_pressures, noise_level)

                # Calculate "sensitivity" (actually noise vector)
                noise_vector = self.calculate_sensitivity_matrix(normal_pressures, noisy_pressures)

                if noise_vector.size > 0:
                    normal_data.append(noise_vector)
                    normal_labels.append(0)  # Normal label is 0

            # Merge data
            all_data = np.array(anomaly_data + normal_data)
            all_labels = np.array(anomaly_labels + normal_labels)

            self.log_info(f"Data generation complete:")
            self.log_info(f"  Total samples: {len(all_data)}")
            self.log_info(f"  Feature dimension: {all_data.shape[1] if len(all_data) > 0 else 0}")
            self.log_info(f"  Normal samples: {np.sum(all_labels == 0)}")
            self.log_info(f"  Anomaly samples: {np.sum(all_labels > 0)}")

            # Detailed label statistics
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            self.log_info(f"  Label distribution: {dict(zip(unique_labels, counts))}")
            self.log_info(f"  Label range: [{np.min(all_labels)}, {np.max(all_labels)}]")

            # Fix: Do not remap labels, use original partition numbers directly
            # Label 0=Normal, Label N=Partition N leak, maintain direct mapping between partition number and label
            self.log_info(f"  Keeping original labels: 0=Normal, 1-{np.max(unique_labels[unique_labels > 0]) if len(unique_labels[unique_labels > 0]) > 0 else 0}=Corresponding partition leak")
            self.log_info(f"  Final label range: [{np.min(all_labels)}, {np.max(all_labels)}]")

            return all_data, all_labels

        except Exception as e:
            error_msg = f"Generate training data failed: {str(e)}"
            self.log_error(error_msg)
            return np.array([]), np.array([])

    def prepare_datasets(self, data: np.ndarray, labels: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, test datasets"""
        try:
            # Data standardization
            data_scaled = self.scaler.fit_transform(data)

            # Check dataset size and class distribution
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            min_samples_per_class = np.min(label_counts)
            total_samples = len(data)

            self.log_info(f"Dataset analysis:")
            self.log_info(f"  Total samples: {total_samples}")
            self.log_info(f"  Number of classes: {len(unique_labels)}")
            self.log_info(f"  Minimum samples per class: {min_samples_per_class} samples")

            # If dataset too small or some classes have too few samples, use simple split
            if total_samples < 10 or min_samples_per_class < 2:
                self.log_warning("Dataset is small, using simple split strategy")

                # Simple split: 80% train, 20% validation, no test set
                if total_samples >= 5:
                    split_idx = int(0.8 * total_samples)
                    X_train = data_scaled[:split_idx]
                    y_train = labels[:split_idx]
                    X_val = data_scaled[split_idx:]
                    y_val = labels[split_idx:]
                    X_test = X_val  # Validation set also serves as test set
                    y_test = y_val
                else:
                    # Too few data, use all for training
                    X_train = data_scaled
                    y_train = labels
                    X_val = data_scaled
                    y_val = labels
                    X_test = data_scaled
                    y_test = labels
            else:
                # Normal stratified split
                X_temp, X_test, y_temp, y_test = train_test_split(
                    data_scaled, labels, test_size=0.1, random_state=42, stratify=labels
                )

                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
                )

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)

            # Create data loaders, adjust batch_size
            batch_size = min(8, len(X_train))  # Small batch_size for small datasets

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            self.log_info(f"Dataset preparation complete:")
            self.log_info(f"  Training set: {len(X_train)} samples")
            self.log_info(f"  Validation set: {len(X_val)} samples")
            self.log_info(f"  Test set: {len(X_test)} samples")
            self.log_info(f"  Batch size: {batch_size}")

            return train_loader, val_loader, test_loader

        except Exception as e:
            error_msg = f"Prepare datasets failed: {str(e)}"
            self.log_error(error_msg)
            return None, None, None

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   input_size: int, num_partitions: int, epochs: int = 100, num_classes: int = None) -> Dict[str, Any]:
        """Train leak detection model"""
        try:
            self.log_info(f"Starting leak detection model training...")
            self.log_info(f"  Input dimension: {input_size}")
            self.log_info(f"  Number of partitions: {num_partitions}")
            self.log_info(f"  Training epochs: {epochs}")

            # Create model - use correct number of classes
            if num_classes is None:
                num_classes = num_partitions + 1  # Default: partitions + 1 (normal class)

            self.log_info(f"  Model classes: {num_classes}")
            self.model = LeakDetectionMLP(input_size, num_partitions, num_classes=num_classes).to(self.device)

            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

            # Training history
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            best_val_acc = 0
            best_model_state = None

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()

                # Validation phase
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                        outputs = self.model(batch_data)
                        loss = criterion(outputs, batch_labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()

                # Calculate accuracy
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total

                # Record history
                train_losses.append(train_loss / len(train_loader))
                val_losses.append(val_loss / len(val_loader))
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()

                # Learning rate scheduling
                scheduler.step()

                # Print every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.log_info(f"  Epoch {epoch+1}/{epochs}: "
                                f"Train Loss: {train_losses[-1]:.4f}, "
                                f"Train Acc: {train_acc:.2f}%, "
                                f"Val Loss: {val_losses[-1]:.4f}, "
                                f"Val Acc: {val_acc:.2f}%")

            # Load best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            # Ensure all data is JSON serializable Python native types
            training_history = {
                'train_losses': [float(x) for x in train_losses],
                'val_losses': [float(x) for x in val_losses],
                'train_accuracies': [float(x) for x in train_accuracies],
                'val_accuracies': [float(x) for x in val_accuracies],
                'best_val_accuracy': float(best_val_acc),
                'final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
                'final_val_loss': float(val_losses[-1]) if val_losses else 0.0
            }

            self.log_info(f"Model training complete, best validation accuracy: {best_val_acc:.2f}%")

            return training_history

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            self.log_error(error_msg)
            return {}

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            self.log_info("Starting model evaluation...")

            if self.model is None:
                raise Exception("Model not trained")

            self.model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                    outputs = self.model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

            # Calculate evaluation metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )

            # Confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)

            # Ensure all data is JSON serializable Python native types
            evaluation_results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'predictions': [int(x) for x in all_predictions],
                'true_labels': [int(x) for x in all_labels]
            }

            self.log_info(f"Model evaluation complete:")
            self.log_info(f"  Accuracy: {accuracy:.4f}")
            self.log_info(f"  Precision: {precision:.4f}")
            self.log_info(f"  Recall: {recall:.4f}")
            self.log_info(f"  F1-Score: {f1:.4f}")

            return evaluation_results

        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            self.log_error(error_msg)
            return {}

    def save_model(self, conversation_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Save trained model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"leak_detection_model_{conversation_id[:8]}_{timestamp}.pth"
            model_path = os.path.join(self.downloads_folder, model_filename)

            # Save model state, ensure all data is serializable
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'scaler_mean': [float(x) for x in self.scaler.mean_],
                'scaler_scale': [float(x) for x in self.scaler.scale_],
                'input_size': int(model_info['input_size']),
                'num_partitions': int(model_info['num_partitions']),
                'num_classes': int(model_info.get('num_classes', model_info['num_partitions'] + 1)),  # Save actual number of classes
                'max_partition': int(model_info.get('max_partition', model_info['num_partitions'])),  # Save max partition number
                'model_info': model_info,
                'timestamp': timestamp
            }

            torch.save(model_state, model_path)

            file_size = os.path.getsize(model_path)

            self.log_info(f"Model saved: {model_filename} ({file_size} bytes)")

            return {
                'success': True,
                'filename': model_filename,
                'file_path': model_path,
                'file_size': file_size,
                'download_url': f'/download/{model_filename}'
            }

        except Exception as e:
            error_msg = f"Save model failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def load_model(self, model_path: str) -> bool:
        """Load trained model"""
        try:
            self.log_info(f"Loading model: {model_path}")

            # Load model state, set weights_only=False for compatibility with older versions
            try:
                model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Compatible with older PyTorch versions
                model_state = torch.load(model_path, map_location=self.device)

            # Rebuild model - use saved actual number of classes
            input_size = model_state['input_size']
            num_partitions = model_state['num_partitions']

            # Prefer saved number of classes, otherwise use traditional calculation
            num_classes = model_state.get('num_classes', num_partitions + 1)
            max_partition = model_state.get('max_partition', num_partitions)

            self.model = LeakDetectionMLP(input_size, num_partitions, num_classes=num_classes).to(self.device)
            self.model.load_state_dict(model_state['model_state_dict'])

            # Rebuild standardizer
            self.scaler.mean_ = np.array(model_state['scaler_mean'])
            self.scaler.scale_ = np.array(model_state['scaler_scale'])

            self.log_info(f"Model loaded successfully: Input dimension={input_size}, Max partition number={max_partition}, Number of classes={num_classes}")
            self.log_info("Note: Actual inference will use current conversation's partition configuration")

            return True

        except Exception as e:
            error_msg = f"Load model failed: {str(e)}"
            self.log_error(error_msg)
            return False

    def predict_leak(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """Predict leak status"""
        try:
            if self.model is None:
                raise Exception("Model not loaded")

            self.log_info(f"Starting leak detection, input data shape: {sensor_data.shape}")

            # Data preprocessing
            if len(sensor_data.shape) == 1:
                sensor_data = sensor_data.reshape(1, -1)

            # Standardization
            sensor_data_scaled = self.scaler.transform(sensor_data)

            # Convert to tensor
            input_tensor = torch.FloatTensor(sensor_data_scaled).to(self.device)

            # Prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

            # Parse results
            predictions = predicted.cpu().numpy()
            probs = probabilities.cpu().numpy()

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probs)):
                # Ensure conversion to Python native types, avoid JSON serialization errors
                pred_int = int(pred)
                confidence = float(prob[pred_int])

                if pred_int == 0:
                    status = "Normal"
                    partition = None
                else:
                    status = "Anomaly"
                    partition = pred_int

                results.append({
                    'sample_id': int(i + 1),
                    'status': status,
                    'partition': partition,
                    'confidence': confidence,
                    'probabilities': [float(p) for p in prob]  # Ensure all probabilities are float type
                })

            self.log_info(f"Leak detection complete, detected {len(results)} samples")

            return {
                'success': True,
                'results': results,
                'summary': {
                    'total_samples': int(len(results)),
                    'normal_samples': int(sum(1 for r in results if r['status'] == 'Normal')),
                    'anomaly_samples': int(sum(1 for r in results if r['status'] == 'Anomaly'))
                }
            }

        except Exception as e:
            error_msg = f"Leak prediction failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def save_training_data(self, data: np.ndarray, labels: np.ndarray,
                          conversation_id: str) -> Dict[str, Any]:
        """Save training data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leak_training_data_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # Prepare data
            df_data = []
            # Fix column name issue
            if self.sensor_data is not None:
                if 'Node ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['Node ID'].tolist()
                elif 'Node ID' in self.sensor_data.columns:
                    sensor_nodes = self.sensor_data['Node ID'].tolist()
                else:
                    sensor_nodes = self.sensor_data.iloc[:, 0].tolist()
            else:
                sensor_nodes = []

            for i, (sample, label) in enumerate(zip(data, labels)):
                # Ensure conversion to Python native types, avoid JSON serialization errors
                row = {'SampleID': int(i + 1), 'Label': int(label)}

                # Add sensor data
                for j, sensor in enumerate(sensor_nodes):
                    if j < len(sample):
                        row[f'Sensor_{sensor}'] = float(sample[j])

                df_data.append(row)

            # Save as CSV
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            file_size = os.path.getsize(filepath)

            self.log_info(f"Training data saved: {filename} ({file_size} bytes)")

            return {
                'success': True,
                'filename': filename,
                'file_path': filepath,
                'file_size': file_size,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"Save training data failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def save_evaluation_report(self, evaluation_results: Dict[str, Any],
                              training_history: Dict[str, Any],
                              conversation_id: str) -> Dict[str, Any]:
        """Save evaluation report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leak_evaluation_{conversation_id[:8]}_{timestamp}.csv"
            filepath = os.path.join(self.downloads_folder, filename)

            # Prepare report data
            report_data = []

            # Basic metrics
            report_data.append({
                'Metric': 'Accuracy',
                'Value': f"{evaluation_results.get('accuracy', 0):.4f}",
                'Description': 'Proportion of correctly predicted samples'
            })

            report_data.append({
                'Metric': 'Precision',
                'Value': f"{evaluation_results.get('precision', 0):.4f}",
                'Description': 'Proportion of actual positives among predicted positives'
            })

            report_data.append({
                'Metric': 'Recall',
                'Value': f"{evaluation_results.get('recall', 0):.4f}",
                'Description': 'Proportion of actual positives correctly predicted'
            })

            report_data.append({
                'Metric': 'F1-Score',
                'Value': f"{evaluation_results.get('f1_score', 0):.4f}",
                'Description': 'Harmonic mean of precision and recall'
            })

            # Training information
            if training_history:
                report_data.append({
                    'Metric': 'Best Validation Accuracy',
                    'Value': f"{training_history.get('best_val_accuracy', 0):.2f}%",
                    'Description': 'Best validation accuracy during training'
                })

            # Save as CSV
            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

            file_size = os.path.getsize(filepath)

            self.log_info(f"Evaluation report saved: {filename} ({file_size} bytes)")

            return {
                'success': True,
                'filename': filename,
                'file_path': filepath,
                'file_size': file_size,
                'download_url': f'/download/{filename}'
            }

        except Exception as e:
            error_msg = f"Save evaluation report failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def train_leak_detection_model(self, inp_file_path: str, conversation_id: str,
                                  num_scenarios: int = 50, epochs: int = 100) -> Dict[str, Any]:
        """Main interface for training leak detection model"""
        try:
            self.log_info("=" * 60)
            self.log_info("Starting leak detection model training")
            self.log_info("=" * 60)

            # 1. Intelligently check dependency files
            dependency_check = self.check_dependencies(conversation_id, inp_file_path)
            if not dependency_check.get('success'):
                return dependency_check

            # Display smart reuse information
            if dependency_check.get('reused_files'):
                reused_files = dependency_check.get('reused_files', [])
                self.log_info("🎯 Smart workflow optimization:")
                for reused_file in reused_files:
                    if reused_file == 'sensor_placement':
                        self.log_info("   ✅ Reusing existing sensor placement, skipping sensor placement step")
                    elif reused_file == 'partition_results':
                        self.log_info("   ✅ Reusing existing partition results, skipping partition analysis step")
                    elif reused_file == 'partition_results_inferred':
                        self.log_info("   ✅ Inferred partition info from sensor file, skipping partition analysis step")
                self.log_info("   ⚡ Significantly improved training efficiency, entering model training phase directly")

            # 2. Load network model
            if not self.load_network_model(inp_file_path):
                return {'success': False, 'error': 'Failed to load network model'}

            # 3. Generate training data
            data, labels = self.generate_training_data(num_scenarios)
            if len(data) == 0:
                return {'success': False, 'error': 'Failed to generate training data'}

            # 4. Prepare datasets
            train_loader, val_loader, test_loader = self.prepare_datasets(data, labels)
            if train_loader is None:
                return {'success': False, 'error': 'Failed to prepare datasets'}

            # 5. Train model
            # Get all unique labels and ensure label range is correct
            unique_labels = np.unique(labels)
            max_label = int(np.max(unique_labels))
            min_label = int(np.min(unique_labels))

            self.log_info(f"Label statistics: min={min_label}, max={max_label}, unique values={unique_labels}")

            # Check if labels are continuous and start from 0
            expected_labels = list(range(min_label, max_label + 1))
            if not all(label in unique_labels for label in expected_labels):
                self.log_warning(f"Labels are not continuous, may cause training issues")

            # Model's number of classes should be max label value + 1 (since labels start from 0)
            num_classes = max_label + 1

            # Fix: Partition count should be max partition number, not number of partition types
            # Because partition numbers may not be continuous (e.g., 1,2,3,4,5,6), not consecutive from 1
            max_partition = np.max(unique_labels[unique_labels > 0]) if len(unique_labels[unique_labels > 0]) > 0 else 0
            num_partitions = max_partition  # Use max partition number as partition count
            input_size = data.shape[1]

            self.log_info(f"Model configuration: Input dimension={input_size}, Max partition number={max_partition}, Number of classes={num_classes}")

            # Recalculate label distribution for logging
            unique_labels_with_counts, counts = np.unique(labels, return_counts=True)
            self.log_info(f"Label distribution: {dict(zip(unique_labels_with_counts, counts))}")

            # Final safety check: ensure all labels are in [0, num_classes-1] range
            if np.any(labels < 0) or np.any(labels >= num_classes):
                error_msg = f"Labels out of range [0, {num_classes-1}]: actual range [{np.min(labels)}, {np.max(labels)}]"
                self.log_error(error_msg)
                return {'success': False, 'error': error_msg}

            training_history = self.train_model(train_loader, val_loader, input_size, num_partitions, epochs, num_classes)
            if not training_history:
                return {'success': False, 'error': 'Model training failed'}

            # 6. Evaluate model
            evaluation_results = self.evaluate_model(test_loader)
            if not evaluation_results:
                return {'success': False, 'error': 'Model evaluation failed'}

            # 7. Save model and results
            # Ensure all data is JSON serializable Python native types
            model_info = {
                'input_size': int(input_size),
                'num_partitions': int(num_partitions),
                'num_scenarios': int(num_scenarios),
                'epochs': int(epochs),
                'evaluation': evaluation_results,
                'training_history': training_history
            }

            # Update model info, including correct partition count and class count
            model_info.update({
                'max_partition': int(max_partition),
                'num_classes': int(num_classes)
            })

            model_save_result = self.save_model(conversation_id, model_info)
            training_data_result = self.save_training_data(data, labels, conversation_id)
            evaluation_report_result = self.save_evaluation_report(evaluation_results, training_history, conversation_id)

            self.log_info("=" * 60)
            self.log_info("Leak detection model training complete")
            self.log_info("=" * 60)

            return {
                'success': True,
                'model_info': model_info,
                'evaluation': evaluation_results,
                'training_history': training_history,
                'files': {
                    'model': model_save_result,
                    'training_data': training_data_result,
                    'evaluation_report': evaluation_report_result
                }
            }

        except Exception as e:
            error_msg = f"Train leak detection model failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def detect_leak_from_file(self, sensor_file_path: str, model_file_path: str, conversation_id: str = None) -> Dict[str, Any]:
        """Read sensor data from file and perform leak detection"""
        try:
            self.log_info("=" * 60)
            self.log_info("Starting leak detection")
            self.log_info("=" * 60)

            # 1. Load model
            if not self.load_model(model_file_path):
                return {'success': False, 'error': 'Failed to load model'}

            # 2. Read partition file to get actual partition count
            actual_num_partitions = None
            if conversation_id:
                partition_file = self._find_partition_file(conversation_id)
                if partition_file:
                    try:
                        partition_df = pd.read_csv(partition_file)
                        # Get actual partition count
                        if 'Partition Number' in partition_df.columns:
                            actual_num_partitions = partition_df['Partition Number'].max()
                        elif 'Partition' in partition_df.columns:
                            actual_num_partitions = partition_df['Partition'].max()

                        if actual_num_partitions:
                            self.log_info(f"Read actual partition count from partition file: {actual_num_partitions}")
                            # Update model's partition count info (for result interpretation)
                            self._actual_num_partitions = actual_num_partitions
                        else:
                            self.log_warning("Cannot determine partition count from partition file, using model default value")
                    except Exception as e:
                        self.log_warning(f"Failed to read partition file: {str(e)}, using model default partition count")

            # 3. Read sensor data
            self.log_info(f"Reading sensor data: {sensor_file_path}")

            try:
                sensor_df = pd.read_csv(sensor_file_path)
                self.log_info(f"Sensor data shape: {sensor_df.shape}")

                # Extract numeric data (exclude ID columns etc.)
                numeric_columns = sensor_df.select_dtypes(include=[np.number]).columns
                sensor_data = sensor_df[numeric_columns].values

                if sensor_data.size == 0:
                    return {'success': False, 'error': 'No numeric data in sensor file'}

            except Exception as e:
                return {'success': False, 'error': f'Failed to read sensor file: {str(e)}'}

            # 4. Perform prediction
            prediction_results = self.predict_leak(sensor_data)
            if not prediction_results['success']:
                return prediction_results

            self.log_info("=" * 60)
            self.log_info("Leak detection complete")
            self.log_info("=" * 60)

            return prediction_results

        except Exception as e:
            error_msg = f"Leak detection failed: {str(e)}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def _find_partition_file(self, conversation_id: str) -> str:
        """Find partition file for the conversation"""
        try:
            if not os.path.exists(self.downloads_folder):
                return None

            # Find partition file
            for filename in os.listdir(self.downloads_folder):
                if (conversation_id[:8] in filename and
                    'partition_results' in filename and
                    filename.endswith('.csv')):
                    partition_file = os.path.join(self.downloads_folder, filename)
                    self.log_info(f"Found partition file: {os.path.basename(partition_file)}")
                    return partition_file

            self.log_warning(f"Partition file not found for conversation {conversation_id[:8]}")
            return None

        except Exception as e:
            self.log_error(f"Failed to find partition file: {str(e)}")
            return None

    def build_response_prompt(self, result: Dict[str, Any], user_message: str,
                             operation_type: str) -> str:
        """Build response prompt"""
        try:
            if operation_type == "training":
                return self._build_training_prompt(result, user_message)
            elif operation_type == "detection":
                return self._build_detection_prompt(result, user_message)
            else:
                return "Operation complete."

        except Exception as e:
            self.log_error(f"Build response prompt failed: {str(e)}")
            return "Operation complete, but an error occurred while generating response."

    def _build_training_prompt(self, result: Dict[str, Any], user_message: str) -> str:
        """Build training response prompt"""
        if not result.get('success', False):
            return f"""
Leak detection model training failed.

Error message: {result.get('error', 'Unknown error')}

Please check the following possible issues:
1. Has network partition analysis been completed
2. Has sensor placement been completed
3. Is the network file correct
4. Are system resources sufficient

User request: {user_message}
"""

        model_info = result.get('model_info', {})
        evaluation = result.get('evaluation', {})
        training_history = result.get('training_history', {})
        files = result.get('files', {})

        # Calculate detailed statistics
        total_samples = model_info.get('num_scenarios', 0) * 2  # Normal + Anomaly samples
        normal_samples = model_info.get('num_scenarios', 0)
        anomaly_samples = model_info.get('num_scenarios', 0)

        # Get partition statistics
        num_partitions = model_info.get('num_partitions', 0)
        samples_per_partition = anomaly_samples // max(num_partitions, 1) if num_partitions > 0 else 0

        # Build performance metrics description
        accuracy = evaluation.get('accuracy', 0)
        precision = evaluation.get('precision', 0)
        recall = evaluation.get('recall', 0)
        f1_score = evaluation.get('f1_score', 0)

        # Safely get training history data
        final_train_loss = training_history.get('final_train_loss', 0)
        final_val_loss = training_history.get('final_val_loss', 0)
        best_val_accuracy = training_history.get('best_val_accuracy', 0)

        # Performance rating
        def get_performance_grade(score):
            if score >= 0.9: return "Excellent 🌟"
            elif score >= 0.8: return "Good ✅"
            elif score >= 0.7: return "Fair ⚠️"
            else: return "Needs Improvement ❌"

        return f"""
🎉 Leak detection model training completed successfully!

## 📊 Training Data Statistics
- **Total samples**: {total_samples} (balanced dataset)
- **Normal samples**: {normal_samples} (with sensor noise)
- **Anomaly samples**: {anomaly_samples} (distributed across {num_partitions} partitions)
- **Samples per partition**: ~{samples_per_partition} leak scenarios
- **Number of sensors**: {model_info.get('input_size', 'N/A')}
- **Training epochs**: {model_info.get('epochs', 'N/A')} epochs

## 📈 Model Performance Evaluation
- **Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%) - {get_performance_grade(accuracy)}
- **Precision**: {precision:.4f} ({precision*100:.2f}%) - {get_performance_grade(precision)}
- **Recall**: {recall:.4f} ({recall*100:.2f}%) - {get_performance_grade(recall)}
- **F1-Score**: {f1_score:.4f} ({f1_score*100:.2f}%) - {get_performance_grade(f1_score)}

### 📋 Performance Metrics Description
- **Accuracy**: Proportion of all predictions that are correct (including Normal and Anomaly)
- **Precision**: Proportion of samples predicted as Anomaly that are truly Anomaly
- **Recall**: Proportion of truly Anomaly samples that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall

## 🎯 Training Process
- **Final training loss**: {final_train_loss:.6f}
- **Final validation loss**: {final_val_loss:.6f}
- **Best validation accuracy**: {best_val_accuracy:.4f}

## 📁 Generated Files
The following files have been generated and are available for download:

### 🤖 Model File
- **Filename**: `{files.get('model', {}).get('filename', 'N/A')}`
- **Format**: PyTorch PTH format
- **Purpose**: For leak detection inference

### 📊 Training Data File
- **Filename**: `{files.get('training_data', {}).get('filename', 'N/A')}`
- **Format**: CSV format
- **Content**: Contains sensor pressure data and labels for training

### 📈 Evaluation Report File
- **Filename**: `{files.get('evaluation_report', {}).get('filename', 'N/A')}`
- **Format**: CSV format
- **Content**: Detailed model performance evaluation metrics and confusion matrix

## 🚀 Next Steps
The model is ready! You can now:
1. **Download model file**: Click the PTH file download button below
2. **Perform leak detection**: Upload sensor pressure data CSV file
3. **View detailed report**: Download evaluation report for more performance details

User request: {user_message}

Please use the following signature format at the end of your reply:

Best regards,

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

    def _build_detection_prompt(self, result: Dict[str, Any], user_message: str) -> str:
        """Build detection response prompt"""
        if not result.get('success', False):
            return f"""
Leak detection failed.

Error message: {result.get('error', 'Unknown error')}

Please check the following possible issues:
1. Is the sensor data file format correct
2. Does the model file exist and is it valid
3. Do the data dimensions match

User request: {user_message}
"""

        results = result.get('results', [])
        summary = result.get('summary', {})

        # Statistics for anomaly situation and confidence
        anomaly_partitions = {}
        normal_confidences = []
        all_confidences = []

        for r in results:
            all_confidences.append(r['confidence'])
            if r['status'] == 'Anomaly':
                partition = r['partition']
                if partition not in anomaly_partitions:
                    anomaly_partitions[partition] = []
                anomaly_partitions[partition].append(r)
            else:
                normal_confidences.append(r['confidence'])

        # Calculate statistics
        total_samples = summary.get('total_samples', 0)
        normal_samples = summary.get('normal_samples', 0)
        anomaly_samples = summary.get('anomaly_samples', 0)

        normal_percentage = (normal_samples / total_samples * 100) if total_samples > 0 else 0
        anomaly_percentage = (anomaly_samples / total_samples * 100) if total_samples > 0 else 0

        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        avg_normal_confidence = np.mean(normal_confidences) if normal_confidences else 0

        prompt = f"""
🎯 **Intelligent Leak Detection Inference Complete**

✅ **Inference mode description**: System detected an existing trained leak detection model, directly performing inference analysis without repeating partition, sensor placement, or model training steps.

## 📊 Detection Overview
- **Analyzed sample count**: {total_samples} time points
- **Normal status**: {normal_samples} samples ({normal_percentage:.1f}%)
- **Anomaly status**: {anomaly_samples} samples ({anomaly_percentage:.1f}%)
- **Average detection confidence**: {avg_confidence:.3f}

## 📈 Confidence Analysis
- **Normal status average confidence**: {avg_normal_confidence:.3f}
- **Overall detection reliability**: {'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'}

"""

        if anomaly_partitions:
            prompt += "## ⚠️ Leak Anomaly Detected\n"
            for partition, samples in anomaly_partitions.items():
                avg_confidence = np.mean([s['confidence'] for s in samples])
                max_confidence = max([s['confidence'] for s in samples])
                min_confidence = min([s['confidence'] for s in samples])

                prompt += f"""
### 🚨 Partition {partition} Leak Alert
- **Anomaly sample count**: {len(samples)}
- **Average confidence**: {avg_confidence:.3f}
- **Highest confidence**: {max_confidence:.3f}
- **Lowest confidence**: {min_confidence:.3f}
- **Severity**: {'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'}
"""

            prompt += "\n## 🔧 Recommended Actions\n"
            prompt += "1. **Immediate inspection**: Conduct on-site inspection of partitions with detected anomalies\n"
            prompt += "2. **Confirm leakage**: Use other detection methods to verify leak location\n"
            prompt += "3. **Develop repair plan**: Schedule repairs based on leak severity\n"
            prompt += "4. **Continuous monitoring**: Increase monitoring frequency for anomaly partitions\n"
        else:
            prompt += "## ✅ No Leak Anomaly Detected\n"
            prompt += f"All {total_samples} time points of sensor data show normal network operation.\n"
            prompt += f"Average detection confidence is {avg_normal_confidence:.3f}, system running stably.\n"

        prompt += f"""

## 📋 Detailed Results
"""

        for i, r in enumerate(results[:5]):  # Only show first 5 results
            status_icon = "✅" if r['status'] == 'Normal' else "⚠️"
            prompt += f"- Sample {r['sample_id']}: {status_icon} {r['status']}"
            if r['partition']:
                prompt += f" (Partition {r['partition']})"
            prompt += f" - Confidence: {r['confidence']:.3f}\n"

        if len(results) > 5:
            prompt += f"... and {len(results)-5} more samples\n"

        prompt += f"""

User request: {user_message}

Please use the following signature format at the end of your reply:

Best regards,

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""

        return prompt

    def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Implement BaseAgent abstract method process"""
        # This method is mainly for compatibility with BaseAgent interface
        # Actual processing logic is in train_leak_detection_model and detect_leak_from_file
        return {
            'success': True,
            'message': 'Leak detection agent is ready. Please use train_leak_detection_model for training or detect_leak_from_file for detection.'
        }
