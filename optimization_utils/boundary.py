"""
Boundary pipe identification and configuration configuration module
Identify partition boundary pipes and apply open/closed configurations
"""
import wntr
import copy


def find_boundary_pipes(wn, node_to_community):
    """
    Identify boundary pipes: pipes where the two end nodes belong to different partitions
    Args:
        wn: wntr network model
        node_to_community: Mapping of nodes to partitions {node_name: community_id}
    Returns:
        boundary_pipes: List of boundary pipes [(pipe_name, node1, node2, comm1, comm2), ...]
    """
    boundary_pipes = []  # Store boundary pipe information
    for pipe_name, pipe in wn.pipes():  # Iterate through all pipes
        node1 = pipe.start_node_name  # Start node
        node2 = pipe.end_node_name    # End node
        # Get the partition the node belongs to (default -1 if not in map)
        comm1 = node_to_community.get(node1, -1)
        comm2 = node_to_community.get(node2, -1)
        if comm1 != comm2 and comm1 >= 0 and comm2 >= 0:  # Ends belong to different partitions
            boundary_pipes.append((pipe_name, node1, node2, comm1, comm2))
    return boundary_pipes


def apply_boundary_config(wn_original, boundary_pipes, config):
    """
    Close specified boundary pipes based on configuration
    Args:
        wn_original: Original wntr network model
        boundary_pipes: List of boundary pipes
        config: Binary configuration array (1=Open, 0=Closed)
    Returns:
        wn_modified: Copy of the modified network model
    """
    wn = copy.deepcopy(wn_original)  # Deep copy to avoid modifying the original network
    for i, (pipe_name, _, _, _, _) in enumerate(boundary_pipes):
        if config[i] == 0:  # Close pipe
            pipe = wn.get_link(pipe_name)
            pipe.initial_status = wntr.network.LinkStatus.Closed  # Set initial status to closed
    return wn


def get_boundary_info(boundary_pipes):
    """
    Get summary information of boundary pipes
    Args:
        boundary_pipes: List of boundary pipes
    Returns:
        dict: Summary of boundary pipe information
    """
    return {
        'count': len(boundary_pipes),  # Total number of boundary pipes
        'pipes': [p[0] for p in boundary_pipes],  # List of pipe names
        'connections': [(p[3], p[4]) for p in boundary_pipes]  # Pairs of connected partitions
    }

