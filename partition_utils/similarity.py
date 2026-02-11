"""
Build similarity matrix for network partitioning.
Based on linkNode method in references/cd_main.m.
"""
import numpy as np
import networkx as nx


def build_similarity_matrix(wn, avg_pressure):
    """
    Build similarity matrix A, where A[i,j] = (avg_pressure[i] + avg_pressure[j]) / 2
    This follows the method using linkNode in MATLAB program cd_main.m.
    
    Args:
        wn: WNTR Water Network Model
        avg_pressure: Dictionary of average pressure for each node
        
    Returns:
        G: NetworkX graph with edge weights based on pressure similarity
        node_list: List of node names
    """
    # Get all nodes (including tanks, reservoirs) to ensure connectivity in visualization
    junction_names = list(wn.junction_name_list)
    node_list = list(wn.node_name_list)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for node in node_list:
        G.add_node(node)
    
    # Add edges based on pipes/links (similar to linkNode in MATLAB)
    # For each link, add an edge with weight equal to the average pressure of connected nodes
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        # Consider edges between all nodes (including reservoirs/tanks)
        if start_node in node_list and end_node in node_list:
            # Calculate weight as average of node pressures (following MATLAB pattern)
            if start_node in avg_pressure and end_node in avg_pressure:
                weight = (avg_pressure[start_node] + avg_pressure[end_node]) / 2
                # Handle negative pressures by using absolute values
                # Avoid zero values causing errors in some algorithms
                G.add_edge(start_node, end_node, weight=weight)
    
    return G, node_list


def create_network_graph(wn, avg_pressure):
    """
    Create NetworkX graph with location information for partitioning.
    
    Args:
        wn: WNTR Water Network Model
        avg_pressure: Dictionary of average pressure
        
    Returns:
        G: NetworkX graph with weights and location information
        pos: Dictionary of node positions
    """
    G, node_list = build_similarity_matrix(wn, avg_pressure)
    
    # Get node positions from network coordinates
    pos = {}
    for node_name in node_list:
        node = wn.get_node(node_name)
        pos[node_name] = (node.coordinates[0], node.coordinates[1])
    
    return G, pos

