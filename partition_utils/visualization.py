"""
Visualization utility functions for network partitioning results.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_distinct_colors(n):
    """Generate n distinct colors for communities."""
    if n <= 10:
        colors = list(mcolors.TABLEAU_COLORS.values())[:n]
    else:
        cmap = plt.cm.get_cmap('tab20', n)
        colors = [cmap(i) for i in range(n)]
    return colors


def plot_partition(G, pos, node_to_community, num_communities, save_path):
    """
    Plot network partition and save to file with legend.

    Args:
        G: NetworkX graph
        pos: Dictionary of node positions
        node_to_community: Dictionary mapping node names to community IDs
        num_communities: Number of communities in this partition
        save_path: File save path
    """
    import networkx as nx
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get distinct colors for communities
    colors = get_distinct_colors(num_communities)

    # Group nodes by community
    community_nodes = {}
    for node in G.nodes():
        comm_id = node_to_community.get(node, 0)
        if comm_id not in community_nodes:
            community_nodes[comm_id] = []
        community_nodes[comm_id].append(node)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')

    # Draw nodes and create legend handles by community
    legend_handles = []
    for comm_id in sorted(community_nodes.keys()):
        nodes = community_nodes[comm_id]
        color = colors[comm_id % len(colors)]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                               node_color=[color], node_size=30, alpha=0.8)
        # Create legend handle
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=8, label=f'Partition {comm_id + 1}')
        legend_handles.append(handle)

    ax.set_title(f'Network Partition: {num_communities} Communities', fontsize=14)
    ax.axis('off')

    # Add legend (adjust columns based on number of communities)
    ncol = min(4, max(1, num_communities // 10 + 1))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=8, ncol=ncol, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_all_partitions_plots(G, pos, unique_partitions, output_dir='partition_plots'):
    """
    Save visualization plots for all unique partitions.
    
    Args:
        G: NetworkX graph
        pos: Dictionary of node positions
        unique_partitions: Dictionary {num_communities: {'node_to_community': dict, 'resolution': float}}
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for num_comm, partition_data in unique_partitions.items():
        node_to_community = partition_data['node_to_community']
        save_path = os.path.join(output_dir, f'partition_{num_comm}_communities.png')
        plot_partition(G, pos, node_to_community, num_comm, save_path)
        print(f"Saved: {save_path}")

