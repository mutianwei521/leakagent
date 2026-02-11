# -*- coding: utf-8 -*-
"""
EPANET Network Model Handler Module
Handles network import, hydraulic simulation, and pressure data acquisition using WNTR library
"""

import numpy as np
import pandas as pd
import wntr
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class EPANETHandler:
    """EPANET Network Model Handler"""
    
    def __init__(self, inp_file: str):
        """
        Initialize EPANET Handler
        
        Args:
            inp_file: EPANET input file path
        """
        self.inp_file = inp_file
        self.wn = None
        self.results = None
        self.node_names = []
        self.pipe_names = []
        self.adjacency_matrix = None
        
    def load_network(self) -> bool:
        """
        Load EPANET network model
        
        Returns:
            bool: Whether loading was successful
        """
        try:
            self.wn = wntr.network.WaterNetworkModel(self.inp_file)
            self.node_names = list(self.wn.junction_name_list)
            self.pipe_names = list(self.wn.pipe_name_list)
            
            # Build adjacency matrix
            self._build_adjacency_matrix()
            
            logger.info(f"Successfully loaded network: {len(self.node_names)} nodes, {len(self.pipe_names)} pipes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
            return False
    
    def _build_adjacency_matrix(self):
        """Build network adjacency matrix"""
        n_nodes = len(self.node_names)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        node_to_idx = {name: idx for idx, name in enumerate(self.node_names)}
        
        for pipe_name in self.pipe_names:
            pipe = self.wn.get_link(pipe_name)
            start_node = pipe.start_node_name
            end_node = pipe.end_node_name
            
            if start_node in node_to_idx and end_node in node_to_idx:
                i = node_to_idx[start_node]
                j = node_to_idx[end_node]
                self.adjacency_matrix[i, j] = 1
                self.adjacency_matrix[j, i] = 1  # Undirected graph
    
    def run_hydraulic_simulation(self, duration_hours: int = 24, 
                                time_step_hours: int = 1) -> bool:
        """
        Run hydraulic simulation
        
        Args:
            duration_hours: Simulation duration (hours)
            time_step_hours: Time step (hours)
            
        Returns:
            bool: Whether simulation was successful
        """
        try:
            # Set simulation parameters
            self.wn.options.time.duration = duration_hours * 3600  # Convert to seconds
            self.wn.options.time.hydraulic_timestep = time_step_hours * 3600
            self.wn.options.time.report_timestep = time_step_hours * 3600
            
            # Run simulation
            sim = wntr.sim.EpanetSimulator(self.wn)
            self.results = sim.run_sim()
            
            logger.info(f"Hydraulic simulation completed: {duration_hours} hours simulation")
            return True
            
        except Exception as e:
            logger.error(f"Hydraulic simulation failed: {e}")
            return False
    
    def get_pressure_data(self) -> pd.DataFrame:
        """
        Get node pressure time series data
        
        Returns:
            pd.DataFrame: Pressure data, rows are time, columns are nodes
        """
        if self.results is None:
            logger.error("Simulation results not found, please run hydraulic simulation first")
            return pd.DataFrame()
        
        try:
            pressure_data = self.results.node['pressure']
            # Only keep pressure data for junction nodes
            junction_pressure = pressure_data[self.node_names]
            return junction_pressure
            
        except Exception as e:
            logger.error(f"Failed to get pressure data: {e}")
            return pd.DataFrame()
    
    def get_flow_data(self) -> pd.DataFrame:
        """
        Get pipe flow time series data
        
        Returns:
            pd.DataFrame: Flow data, rows are time, columns are pipes
        """
        if self.results is None:
            logger.error("Simulation results not found, please run hydraulic simulation first")
            return pd.DataFrame()
        
        try:
            flow_data = self.results.link['flowrate']
            pipe_flow = flow_data[self.pipe_names]
            return pipe_flow
            
        except Exception as e:
            logger.error(f"Failed to get flow data: {e}")
            return pd.DataFrame()
    
    def get_total_network_demand(self) -> float:
        """
        Calculate total network demand

        Returns:
            float: Total demand
        """
        try:
            total_demand = 0.0
            for node_name in self.node_names:
                junction = self.wn.get_node(node_name)
                if junction.demand_timeseries_list:
                    total_demand += abs(junction.demand_timeseries_list[0].base_value)

            logger.debug(f"Total network demand: {total_demand:.3f}")
            return total_demand

        except Exception as e:
            logger.error(f"Failed to calculate total demand: {e}")
            return 0.0

    def modify_node_demand(self, node_name: str, demand_ratio: float = 0.2) -> bool:
        """
        Modify node demand

        Args:
            node_name: Node name
            demand_ratio: Demand ratio (relative to original value)

        Returns:
            bool: Whether modification was successful
        """
        try:
            if node_name not in self.node_names:
                logger.error(f"Node {node_name} does not exist")
                return False

            junction = self.wn.get_node(node_name)
            original_demand = junction.demand_timeseries_list[0].base_value

            # If original demand is 0, use 3% of total network demand
            if abs(original_demand) < 1e-6:  # Use small threshold for float comparison
                total_demand = self.get_total_network_demand()
                new_demand = total_demand * 0.03  # 3%
                logger.info(f"Node {node_name} original demand is 0, using 3% of total network demand: {new_demand:.3f}")
            else:
                new_demand = original_demand * demand_ratio

            # Modify demand
            junction.demand_timeseries_list[0].base_value = new_demand

            logger.debug(f"Node {node_name} demand modified from {original_demand:.3f} to {new_demand:.3f}")
            return True

        except Exception as e:
            logger.error(f"Failed to modify node demand: {e}")
            return False
    
    def reset_network(self):
        """Reset network to original state"""
        try:
            self.wn = wntr.network.WaterNetworkModel(self.inp_file)
            logger.debug("Network reset to original state")
        except Exception as e:
            logger.error(f"Failed to reset network: {e}")
    
    def get_pipe_lengths(self) -> Dict[str, float]:
        """
        Get pipe length information
        
        Returns:
            Dict[str, float]: Mapping from pipe name to length
        """
        pipe_lengths = {}
        try:
            for pipe_name in self.pipe_names:
                pipe = self.wn.get_link(pipe_name)
                pipe_lengths[pipe_name] = pipe.length
            return pipe_lengths
            
        except Exception as e:
            logger.error(f"Failed to get pipe lengths: {e}")
            return {}
    
    def get_network_graph(self) -> nx.Graph:
        """
        Get network graph structure
        
        Returns:
            nx.Graph: NetworkX graph object
        """
        try:
            G = nx.Graph()
            
            # Add nodes
            for node_name in self.node_names:
                G.add_node(node_name)
            
            # Add edges (pipes)
            for pipe_name in self.pipe_names:
                pipe = self.wn.get_link(pipe_name)
                start_node = pipe.start_node_name
                end_node = pipe.end_node_name
                
                if start_node in self.node_names and end_node in self.node_names:
                    G.add_edge(start_node, end_node, 
                             pipe_name=pipe_name, 
                             length=pipe.length)
            
            return G
            
        except Exception as e:
            logger.error(f"Failed to build network graph: {e}")
            return nx.Graph()
    
    def get_network_info(self) -> Dict:
        """
        Get basic network information
        
        Returns:
            Dict: Network information dictionary
        """
        return {
            'n_nodes': len(self.node_names),
            'n_pipes': len(self.pipe_names),
            'node_names': self.node_names.copy(),
            'pipe_names': self.pipe_names.copy(),
            'adjacency_matrix': self.adjacency_matrix.copy() if self.adjacency_matrix is not None else None
        }
