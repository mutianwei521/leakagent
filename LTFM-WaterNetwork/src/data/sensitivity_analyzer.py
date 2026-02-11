# -*- coding: utf-8 -*-
"""
Pressure Sensitivity Analysis Module
Implements node demand adjustment, pressure difference calculation, and sensitivity matrix generation functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from .epanet_handler import EPANETHandler


class SensitivityAnalyzer:
    """Pressure Sensitivity Analyzer"""
    
    def __init__(self, epanet_handler: EPANETHandler):
        """
        Initialize Sensitivity Analyzer
        
        Args:
            epanet_handler: EPANET handler instance
        """
        self.epanet_handler = epanet_handler
        self.normal_pressure = None
        self.normal_flow = None  # Add storage for normal flow data
        self.sensitivity_matrix = None
        self.flow_sensitivity_matrix = None
        self.node_weights = None
        
    def calculate_normal_pressure(self, duration_hours: int = 24, 
                                time_step_hours: int = 1) -> bool:
        """
        Calculate normal pressure data
        
        Args:
            duration_hours: Simulation duration (hours)
            time_step_hours: Time step (hours)
            
        Returns:
            bool: Whether calculation was successful
        """
        try:
            # Reset network to original state
            self.epanet_handler.reset_network()
            
            # Run hydraulic simulation
            if not self.epanet_handler.run_hydraulic_simulation(duration_hours, time_step_hours):
                return False
            
            # Get normal pressure data
            self.normal_pressure = self.epanet_handler.get_pressure_data()

            if self.normal_pressure.empty:
                logger.error("Failed to get normal pressure data")
                return False

            # Get normal flow data
            self.normal_flow = self.epanet_handler.get_flow_data()

            logger.info(f"Normal pressure data calculation completed: {self.normal_pressure.shape}")
            if not self.normal_flow.empty:
                logger.info(f"Normal flow data calculation completed: {self.normal_flow.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate normal pressure: {e}")
            return False
    
    def calculate_sensitivity_matrix(self, demand_ratio: float = 0.2,
                                   duration_hours: int = 24,
                                   time_step_hours: int = 1) -> bool:
        """
        Calculate pressure sensitivity matrix
        
        Args:
            demand_ratio: Anomalous demand ratio
            duration_hours: Simulation duration (hours)
            time_step_hours: Time step (hours)
            
        Returns:
            bool: Whether calculation was successful
        """
        try:
            if self.normal_pressure is None:
                logger.error("Please calculate normal pressure data first")
                return False
            
            node_names = self.epanet_handler.node_names
            n_nodes = len(node_names)
            n_times = len(self.normal_pressure)

            logger.info(f"Normal pressure data shape: {self.normal_pressure.shape}")
            logger.info(f"Number of nodes: {n_nodes}, Number of time steps: {n_times}")

            # Initialize sensitivity matrix [n_nodes, n_nodes, n_times]
            # sensitivity_matrix[i, j, t] represents pressure sensitivity of node j at time t when node i is anomalous
            self.sensitivity_matrix = np.zeros((n_nodes, n_nodes, n_times))
            self.flow_sensitivity_matrix = np.zeros((n_nodes, len(self.epanet_handler.pipe_names), n_times))

            logger.info(f"Start calculating sensitivity matrix: {n_nodes} nodes, Sensitivity matrix shape: {self.sensitivity_matrix.shape}")
            
            # Iterate through each node, simulate its anomaly
            for i, anomaly_node in enumerate(tqdm(node_names, desc="Calculating node sensitivity")):
                # Reset network
                self.epanet_handler.reset_network()
                
                # Modify demand of current node
                if not self.epanet_handler.modify_node_demand(anomaly_node, demand_ratio):
                    logger.warning(f"Failed to modify demand for node {anomaly_node}, skipping")
                    continue
                
                # Run hydraulic simulation
                if not self.epanet_handler.run_hydraulic_simulation(duration_hours, time_step_hours):
                    logger.warning(f"Hydraulic simulation failed for anomalous node {anomaly_node}, skipping")
                    continue
                
                # Get anomalous pressure data
                anomaly_pressure = self.epanet_handler.get_pressure_data()
                anomaly_flow = self.epanet_handler.get_flow_data()
                
                if anomaly_pressure.empty:
                    logger.warning(f"Anomalous pressure data for node {anomaly_node} is empty, skipping")
                    continue
                
                # Check pressure data shape consistency
                if anomaly_pressure.shape != self.normal_pressure.shape:
                    logger.warning(f"Anomalous pressure data shape mismatch for node {anomaly_node}: "
                                 f"Normal {self.normal_pressure.shape} vs Anomaly {anomaly_pressure.shape}, skipping")
                    continue

                # Calculate absolute pressure difference (sensitivity)
                pressure_diff = np.abs(self.normal_pressure - anomaly_pressure)

                # Store sensitivity data
                for j, sensor_node in enumerate(node_names):
                    if sensor_node in pressure_diff.columns:
                        pressure_values = pressure_diff[sensor_node].values
                        # Ensure data length matches
                        if len(pressure_values) == self.sensitivity_matrix.shape[2]:
                            self.sensitivity_matrix[i, j, :] = pressure_values
                        else:
                            logger.warning(f"Pressure data length mismatch for node {sensor_node}: "
                                         f"Expected {self.sensitivity_matrix.shape[2]}, Actual {len(pressure_values)}")
                            # Pad with zeros or truncate
                            min_len = min(len(pressure_values), self.sensitivity_matrix.shape[2])
                            self.sensitivity_matrix[i, j, :min_len] = pressure_values[:min_len]
                
                # Calculate flow difference
                if not anomaly_flow.empty and self.normal_flow is not None:
                    if not self.normal_flow.empty and anomaly_flow.shape == self.normal_flow.shape:
                        flow_diff = self.normal_flow - anomaly_flow
                        for k, pipe_name in enumerate(self.epanet_handler.pipe_names):
                            if pipe_name in flow_diff.columns:
                                flow_values = flow_diff[pipe_name].values
                                # Ensure data length matches
                                if len(flow_values) == self.flow_sensitivity_matrix.shape[2]:
                                    self.flow_sensitivity_matrix[i, k, :] = flow_values
                                else:
                                    logger.warning(f"Flow data length mismatch for pipe {pipe_name}: "
                                                 f"Expected {self.flow_sensitivity_matrix.shape[2]}, Actual {len(flow_values)}")
                                    # Pad with zeros or truncate
                                    min_len = min(len(flow_values), self.flow_sensitivity_matrix.shape[2])
                                    self.flow_sensitivity_matrix[i, k, :min_len] = flow_values[:min_len]
            
            logger.info("Sensitivity matrix calculation completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate sensitivity matrix: {e}")
            return False
    
    def normalize_sensitivity_matrix(self) -> bool:
        """
        Normalize sensitivity matrix (Standardization + Normalization, handle division by zero)

        Returns:
            bool: Whether normalization was successful
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("Sensitivity matrix not calculated")
                return False

            n_nodes, _, n_times = self.sensitivity_matrix.shape
            epsilon = 1e-6  # Prevent division by zero

            # First perform standardization (Z-score normalization)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    data = self.sensitivity_matrix[i, j, :]

                    # Calculate mean and standard deviation
                    mean_val = np.mean(data)
                    std_val = np.std(data)

                    # Standardization: Handle case where standard deviation is 0
                    if std_val < epsilon:
                        # If std is 0, set to mean (if mean is also 0, keep as 0)
                        self.sensitivity_matrix[i, j, :] = 0.0 if abs(mean_val) < epsilon else 1.0
                    else:
                        self.sensitivity_matrix[i, j, :] = (data - mean_val) / std_val

            # Then perform Min-Max normalization to [0,1] range
            for i in range(n_nodes):
                for j in range(n_nodes):
                    data = self.sensitivity_matrix[i, j, :]

                    min_val = np.min(data)
                    max_val = np.max(data)

                    # Normalization: Handle case where max equals min
                    if abs(max_val - min_val) < epsilon:
                        # If range is 0, set to 0.5 (middle value)
                        self.sensitivity_matrix[i, j, :] = 0.5
                    else:
                        self.sensitivity_matrix[i, j, :] = (data - min_val) / (max_val - min_val + epsilon)

            logger.info("Sensitivity matrix standardization and normalization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to normalize sensitivity matrix: {e}")
            return False
    
    def calculate_node_weights(self) -> np.ndarray:
        """
        Calculate node weight vectors (Time-averaged pressure sensitivity)
        
        Returns:
            np.ndarray: Node weight vector [n_nodes, n_nodes]
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("Sensitivity matrix not calculated")
                return np.array([])
            
            # Calculate time-averaged value as node weight
            self.node_weights = np.mean(self.sensitivity_matrix, axis=2)
            
            logger.info(f"Node weight calculation completed: {self.node_weights.shape}")
            return self.node_weights
            
        except Exception as e:
            logger.error(f"Failed to calculate node weights: {e}")
            return np.array([])
    
    def get_sensitivity_features(self, anomaly_node_idx: int) -> Dict:
        """
        Get sensitivity features for a specified anomalous node
        
        Args:
            anomaly_node_idx: Anomalous node index
            
        Returns:
            Dict: Sensitivity feature dictionary
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("Sensitivity matrix not calculated")
                return {}
            
            features = {
                'pressure_sensitivity': self.sensitivity_matrix[anomaly_node_idx, :, :],
                'avg_pressure_sensitivity': np.mean(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                'max_pressure_sensitivity': np.max(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                'std_pressure_sensitivity': np.std(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1)
            }
            
            if self.flow_sensitivity_matrix is not None:
                features.update({
                    'flow_sensitivity': self.flow_sensitivity_matrix[anomaly_node_idx, :, :],
                    'avg_flow_sensitivity': np.mean(self.flow_sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                    'max_flow_sensitivity': np.max(self.flow_sensitivity_matrix[anomaly_node_idx, :, :], axis=1)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get sensitivity features: {e}")
            return {}
    
    def save_sensitivity_data(self, output_dir: str) -> bool:
        """
        Save sensitivity data
        
        Args:
            output_dir: Output directory
            
        Returns:
            bool: Whether saving was successful
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if self.sensitivity_matrix is not None:
                np.save(os.path.join(output_dir, 'sensitivity_matrix.npy'), self.sensitivity_matrix)
                logger.info(f"Sensitivity matrix saved to {output_dir}/sensitivity_matrix.npy")
            
            if self.node_weights is not None:
                np.save(os.path.join(output_dir, 'node_weights.npy'), self.node_weights)
                logger.info(f"Node weights saved to {output_dir}/node_weights.npy")
            
            if self.normal_pressure is not None:
                self.normal_pressure.to_csv(os.path.join(output_dir, 'normal_pressure.csv'))
                logger.info(f"Normal pressure data saved to {output_dir}/normal_pressure.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sensitivity data: {e}")
            return False
    
    def load_sensitivity_data(self, output_dir: str) -> bool:
        """
        Load sensitivity data
        
        Args:
            output_dir: Data directory
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            import os
            
            sensitivity_file = os.path.join(output_dir, 'sensitivity_matrix.npy')
            if os.path.exists(sensitivity_file):
                self.sensitivity_matrix = np.load(sensitivity_file)
                logger.info("Sensitivity matrix loaded successfully")
            
            weights_file = os.path.join(output_dir, 'node_weights.npy')
            if os.path.exists(weights_file):
                self.node_weights = np.load(weights_file)
                logger.info("Node weights loaded successfully")
            
            pressure_file = os.path.join(output_dir, 'normal_pressure.csv')
            if os.path.exists(pressure_file):
                self.normal_pressure = pd.read_csv(pressure_file, index_col=0, encoding='utf-8')
                logger.info("Normal pressure data loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sensitivity data: {e}")
            return False
