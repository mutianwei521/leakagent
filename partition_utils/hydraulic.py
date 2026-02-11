"""
Hydraulic simulation utility functions for water distribution networks.
"""
import numpy as np
import wntr


def run_hydraulic_simulation(wn):
    """
    Run hydraulic simulation and return results.
    
    Args:
        wn: WNTR Water Network Model
        
    Returns:
        sim_results: Simulation results object
    """
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results


def calculate_average_pressure(results, wn):
    """
    Calculate average pressure for each node across all time steps.
    
    Args:
        results: WNTR simulation results
        wn: WNTR Water Network Model
        
    Returns:
        avg_pressure: Dictionary mapping node names to average pressure values
    """
    # Get pressure data for all nodes across all time steps
    pressure_df = results.node['pressure']
    
    # Calculate average pressure for each node across all time steps
    avg_pressure = pressure_df.mean(axis=0)
    
    # Convert to dictionary for easy access
    avg_pressure_dict = avg_pressure.to_dict()
    
    return avg_pressure_dict

