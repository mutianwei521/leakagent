"""
Visual perception module for MM-WDS framework.

This module implements the visual modality (M_Vis) described in Section 2.2.3,
achieving macro-scale spatial pattern recognition through heatmap generation and Vision-Language Model (VLM) integration.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import networkx as nx


# Directory for generated visual outputs
VISUAL_OUTPUT_DIR = os.path.join(os.getcwd(), "visual_outputs")
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)


def normalize_values(values, v_min=None, v_max=None):
    """
    Normalize values to [0, 1] range for color mapping.
    
    Implements Equation 11a from the paper:
    C(P_i) = cmap((P_i - P_min) / (P_max - P_min))
    
    Args:
        values: Array to normalize
        v_min: Optional minimum value (if None, use min(values))
        v_max: Optional maximum value (if None, use max(values))
    
    Returns:
        Normalized values in [0, 1] range
    """
    values = np.array(values)
    if v_min is None:
        v_min = np.min(values)
    if v_max is None:
        v_max = np.max(values)
    
    if v_max - v_min < 1e-10:
        return np.full_like(values, 0.5, dtype=float)
    
    return (values - v_min) / (v_max - v_min)


def calculate_link_widths(velocities, w_min=0.5, w_max=5.0):
    """
    Calculate link (pipe) widths based on flow velocity.
    
    Implements Equation 11b from the paper:
    w_ij = w_min + (w_max - w_min) * |v_ij| / max|v_kl|
    
    Args:
        velocities: Array of flow velocities
        w_min: Minimum line width
        w_max: Maximum line width
    
    Returns:
        Array of line widths
    """
    velocities = np.abs(np.array(velocities))
    v_max = np.max(velocities) if len(velocities) > 0 else 1.0
    
    if v_max < 1e-10:
        return np.full_like(velocities, w_min, dtype=float)
    
    normalized = velocities / v_max
    return w_min + (w_max - w_min) * normalized


def generate_pressure_heatmap(wn, results, save_path=None, title="Pressure Distribution Heatmap"):
    """
    Generate pressure heatmap visualization for water distribution network.
    
    Args:
        wn: WNTR WaterNetworkModel object
        results: WNTR simulation results
        save_path: Path to save the chart (optional)
        title: Chart title
    
    Returns:
        Path to the saved chart, or None if not saved
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get node positions
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    # Extract average pressure
    node_names = list(wn.node_name_list)
    pressures = []
    for node_name in node_names:
        try:
            p = results.node['pressure'][node_name].mean()
            pressures.append(p)
        except:
            pressures.append(0)
    
    pressures = np.array(pressures)
    
    # Normalize pressure for color mapping (Equation 11a)
    p_min, p_max = np.min(pressures), np.max(pressures)
    normalized_pressures = normalize_values(pressures, p_min, p_max)
    
    # Create divergent colormap (RdYlBu - blue for low pressure, red for high pressure)
    cmap = plt.cm.RdYlBu_r  # Reverse so red represents high pressure
    
    # Draw edges first (background)
    G = wn.to_graph()
    edge_positions = []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    
    if edge_positions:
        lc = LineCollection(edge_positions, colors='gray', linewidths=0.5, alpha=0.5)
        ax.add_collection(lc)
    
    # Draw nodes with pressure colors
    x_coords = [pos[n][0] for n in node_names if n in pos]
    y_coords = [pos[n][1] for n in node_names if n in pos]
    valid_pressures = [normalized_pressures[i] for i, n in enumerate(node_names) if n in pos]
    
    scatter = ax.scatter(x_coords, y_coords, c=valid_pressures, cmap=cmap,
                         s=30, alpha=0.8, edgecolors='black', linewidths=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(f'Pressure (m)\n[{p_min:.1f} - {p_max:.1f}]', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path
    else:
        default_path = os.path.join(VISUAL_OUTPUT_DIR, 'pressure_heatmap.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return default_path


def generate_flow_visualization(wn, results, save_path=None, title="Flow Distribution Visualization"):
    """
    Generate flow visualization based on flow velocity line widths.
    
    Implements Equation 11b for link (pipe) width scaling.
    
    Args:
        wn: WNTR WaterNetworkModel object
        results: WNTR simulation results
        save_path: Path to save the chart (optional)
        title: Chart title
    
    Returns:
        Path to the saved chart
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get node positions
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    # Extract link (pipe) velocity and flowrate
    link_data = []
    for link_name, link in wn.links():
        try:
            velocity = abs(results.link['velocity'][link_name].mean())
            flowrate = results.link['flowrate'][link_name].mean()
        except:
            velocity = 0
            flowrate = 0
        
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        if start_node in pos and end_node in pos:
            link_data.append({
                'name': link_name,
                'start': pos[start_node],
                'end': pos[end_node],
                'velocity': velocity,
                'flowrate': flowrate
            })
    
    if link_data:
        velocities = [d['velocity'] for d in link_data]
        widths = calculate_link_widths(velocities, w_min=0.5, w_max=6.0)
        
        # Normalize velocity for color
        v_norm = normalize_values(velocities)
        cmap = plt.cm.YlOrRd
        
        # Draw links (pipes) with varying widths and colors
        for i, d in enumerate(link_data):
            color = cmap(v_norm[i])
            ax.plot([d['start'][0], d['end'][0]], [d['start'][1], d['end'][1]],
                    color=color, linewidth=widths[i], alpha=0.7, solid_capstyle='round')
            
            # Add flow direction arrows for significant flow
            if d['velocity'] > np.mean(velocities):
                mid_x = (d['start'][0] + d['end'][0]) / 2
                mid_y = (d['start'][1] + d['end'][1]) / 2
                dx = (d['end'][0] - d['start'][0]) * 0.1
                dy = (d['end'][1] - d['start'][1]) * 0.1
                if d['flowrate'] < 0:
                    dx, dy = -dx, -dy
                ax.annotate('', xy=(mid_x + dx, mid_y + dy), xytext=(mid_x, mid_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # Draw nodes
    node_x = [pos[n][0] for n in wn.node_name_list if n in pos]
    node_y = [pos[n][1] for n in wn.node_name_list if n in pos]
    ax.scatter(node_x, node_y, c='steelblue', s=20, alpha=0.8, zorder=5)
    
    # Add colorbar for velocity
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(velocities) if velocities else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Velocity (m/s)', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path
    else:
        default_path = os.path.join(VISUAL_OUTPUT_DIR, 'flow_visualization.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return default_path


def generate_combined_heatmap(wn, results, save_path=None, title="Network State Overview"):
    """
    Generate combined visualization showing both pressure and flow.
    
    Args:
        wn: WNTR WaterNetworkModel object
        results: WNTR simulation results
        save_path: Path to save the chart (optional)
        title: Chart title
    
    Returns:
        Path to the saved chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get node positions
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    G = wn.to_graph()
    
    # --- Left Panel: Pressure Heatmap ---
    ax1 = axes[0]
    
    node_names = list(wn.node_name_list)
    pressures = []
    for node_name in node_names:
        try:
            p = results.node['pressure'][node_name].mean()
            pressures.append(p)
        except:
            pressures.append(0)
    
    pressures = np.array(pressures)
    p_min, p_max = np.min(pressures), np.max(pressures)
    normalized_pressures = normalize_values(pressures, p_min, p_max)
    
    # Draw edges
    edge_positions = []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    if edge_positions:
        lc = LineCollection(edge_positions, colors='gray', linewidths=0.5, alpha=0.5)
        ax1.add_collection(lc)
    
    # Draw nodes with pressure colors
    x_coords = [pos[n][0] for n in node_names if n in pos]
    y_coords = [pos[n][1] for n in node_names if n in pos]
    valid_pressures = [normalized_pressures[i] for i, n in enumerate(node_names) if n in pos]
    
    scatter1 = ax1.scatter(x_coords, y_coords, c=valid_pressures, cmap='RdYlBu_r',
                           s=40, alpha=0.8, edgecolors='black', linewidths=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    cbar1.set_label(f'Pressure [{p_min:.1f} - {p_max:.1f}] m', fontsize=10)
    
    ax1.set_title('Pressure Distribution', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # --- Right Panel: Flow Visualization ---
    ax2 = axes[1]
    
    link_data = []
    for link_name, link in wn.links():
        try:
            velocity = abs(results.link['velocity'][link_name].mean())
        except:
            velocity = 0
        
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        if start_node in pos and end_node in pos:
            link_data.append({
                'start': pos[start_node],
                'end': pos[end_node],
                'velocity': velocity
            })
    
    if link_data:
        velocities = [d['velocity'] for d in link_data]
        widths = calculate_link_widths(velocities, w_min=0.5, w_max=6.0)
        v_norm = normalize_values(velocities)
        cmap = plt.cm.YlOrRd
        
        for i, d in enumerate(link_data):
            color = cmap(v_norm[i])
            ax2.plot([d['start'][0], d['end'][0]], [d['start'][1], d['end'][1]],
                     color=color, linewidth=widths[i], alpha=0.7, solid_capstyle='round')
    
    # Draw nodes
    ax2.scatter(x_coords, y_coords, c='steelblue', s=20, alpha=0.8, zorder=5)
    
    if velocities:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(velocities)))
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax2, shrink=0.8)
        cbar2.set_label('Velocity (m/s)', fontsize=10)
    
    ax2.set_title('Flow Distribution', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path
    else:
        default_path = os.path.join(VISUAL_OUTPUT_DIR, 'combined_heatmap.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return default_path


def get_vlm_prompt_template():
    """
    Return VLM prompt template for analyzing network heatmaps.
    
    This implements the VLM prompt engineering part of Section 2.2.3.
    
    Returns:
        Structured prompt string for VLM analysis
    """
    return """Analyze this water distribution network heatmap. Identify:

1. **Pressure anomalies**: Zones with unusually high (red) or low (blue) pressure values. 
   Describe their spatial location (e.g., "northern region", "central zone", "southeast corner").

2. **Critical links**: Pipes that appear to bridge disconnected regions or serve as 
   single points of failure. Look for thin connections between dense clusters.

3. **Dead-end branches**: Tree-like extensions with no loop redundancy that could 
   cause service interruptions if a single pipe fails.

4. **Flow concentration**: Thick lines indicating high-velocity pipes that may be 
   bottlenecks or critical transmission mains.

5. **Pressure gradients**: Areas showing steep color transitions (rapid pressure drop) 
   which may indicate undersized pipes or excessive local demand.

Provide your analysis in the following JSON format:
{
    "pressure_anomalies": [
        {"location": "description", "type": "high/low", "severity": "minor/moderate/severe"}
    ],
    "critical_links": [
        {"location": "description", "vulnerability": "description"}
    ],
    "dead_ends": [
        {"location": "description", "affected_nodes": "estimated count"}
    ],
    "flow_bottlenecks": [
        {"location": "description", "risk_level": "low/medium/high"}
    ],
    "pressure_gradients": [
        {"location": "description", "possible_cause": "description"}
    ],
    "overall_assessment": "Brief summary of network health and priority concerns"
}
"""


def extract_visual_features(wn, results):
    """
    Extract quantitative visual features from network state.
    
    This provides numerical features to complement VLM visual analysis,
    implementing the visual feature extraction part of Section 2.2.3.
    
    Args:
        wn: WNTR WaterNetworkModel object
        results: WNTR simulation results
    
    Returns:
        Dictionary containing extracted visual features
    """
    features = {
        'topological_anomalies': {},
        'pressure_patterns': {},
        'flow_patterns': {},
        'symmetry_metrics': {}
    }
    
    G = wn.to_graph().to_undirected()
    
    # --- Topological Anomalies ---
    # Find articulation points (bridge nodes)
    try:
        articulation_points = list(nx.articulation_points(G))
        features['topological_anomalies']['bridge_nodes'] = articulation_points
        features['topological_anomalies']['bridge_count'] = len(articulation_points)
    except:
        features['topological_anomalies']['bridge_nodes'] = []
        features['topological_anomalies']['bridge_count'] = 0
    
    # Find dead-end nodes (degree 1)
    dead_ends = [n for n in G.nodes() if G.degree(n) == 1]
    features['topological_anomalies']['dead_end_nodes'] = dead_ends
    features['topological_anomalies']['dead_end_count'] = len(dead_ends)
    
    # --- Pressure Patterns ---
    pressures = []
    for node_name in wn.node_name_list:
        try:
            p = results.node['pressure'][node_name].mean()
            pressures.append(p)
        except:
            pass
    
    if pressures:
        pressures = np.array(pressures)
        features['pressure_patterns']['mean'] = float(np.mean(pressures))
        features['pressure_patterns']['std'] = float(np.std(pressures))
        features['pressure_patterns']['min'] = float(np.min(pressures))
        features['pressure_patterns']['max'] = float(np.max(pressures))
        features['pressure_patterns']['range'] = float(np.max(pressures) - np.min(pressures))
        
        # Coefficient of variation (measure of uniformity)
        if np.mean(pressures) > 0:
            features['pressure_patterns']['cv'] = float(np.std(pressures) / np.mean(pressures))
        else:
            features['pressure_patterns']['cv'] = 0.0
    
    # --- Flow Patterns ---
    velocities = []
    for link_name in wn.link_name_list:
        try:
            v = abs(results.link['velocity'][link_name].mean())
            velocities.append(v)
        except:
            pass
    
    if velocities:
        velocities = np.array(velocities)
        features['flow_patterns']['mean_velocity'] = float(np.mean(velocities))
        features['flow_patterns']['max_velocity'] = float(np.max(velocities))
        features['flow_patterns']['velocity_std'] = float(np.std(velocities))
        
        # Identify high flow pipes (top 10%)
        threshold = np.percentile(velocities, 90)
        high_flow_count = np.sum(velocities > threshold)
        features['flow_patterns']['high_flow_pipe_count'] = int(high_flow_count)
        features['flow_patterns']['high_flow_threshold'] = float(threshold)
    
    # --- Symmetry Metrics ---
    # Calculate flow balance using Gini coefficient
    if velocities is not None and len(velocities) > 0:
        sorted_v = np.sort(velocities)
        n = len(sorted_v)
        cumsum = np.cumsum(sorted_v)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_v))) / (n * np.sum(sorted_v)) - (n + 1) / n
        features['symmetry_metrics']['flow_gini_coefficient'] = float(abs(gini))
        features['symmetry_metrics']['flow_balance_score'] = float(1 - abs(gini))  # 1 = Perfect balance
    
    return features


def analyze_network_visually(inp_file_path, output_dir=None):
    """
    Complete visual analysis pipeline for water distribution network.
    
    This is the main entry point for the visual perception module,
    implementing the full workflow described in Section 2.2.3.
    
    Args:
        inp_file_path: Path to EPANET .inp file
        output_dir: Directory for output files (optional)
    
    Returns:
        Dictionary containing:
        - 'heatmap_paths': Paths to generated heatmap images
        - 'visual_features': Extracted quantitative features
        - 'vlm_prompt': Prompt template used for VLM analysis
    """
    import wntr
    
    if output_dir is None:
        output_dir = VISUAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Load network and run simulation
    wn = wntr.network.WaterNetworkModel(inp_file_path)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Generate base filename from input file
    base_name = os.path.splitext(os.path.basename(inp_file_path))[0]
    
    # Generate visualizations
    heatmap_paths = {}
    
    pressure_path = os.path.join(output_dir, f'{base_name}_pressure_heatmap.png')
    heatmap_paths['pressure'] = generate_pressure_heatmap(wn, results, pressure_path,
                                                          f'{base_name} - Pressure Distribution')
    
    flow_path = os.path.join(output_dir, f'{base_name}_flow_visualization.png')
    heatmap_paths['flow'] = generate_flow_visualization(wn, results, flow_path,
                                                        f'{base_name} - Flow Distribution')
    
    combined_path = os.path.join(output_dir, f'{base_name}_combined_heatmap.png')
    heatmap_paths['combined'] = generate_combined_heatmap(wn, results, combined_path,
                                                          f'{base_name} - Network State Overview')
    
    # Extract visual features
    visual_features = extract_visual_features(wn, results)
    
    # Get VLM prompt
    vlm_prompt = get_vlm_prompt_template()
    
    return {
        'heatmap_paths': heatmap_paths,
        'visual_features': visual_features,
        'vlm_prompt': vlm_prompt,
        'network_name': base_name,
        'node_count': wn.num_nodes,
        'link_count': wn.num_links
    }


# Example Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
    else:
        # Default test file
        inp_path = "Exa7.inp"
    
    if os.path.exists(inp_path):
        print(f"Analyzing network: {inp_path}")
        result = analyze_network_visually(inp_path)
        
        print(f"\n=== Visual Analysis Complete ===")
        print(f"Network: {result['network_name']}")
        print(f"Nodes: {result['node_count']}, Links: {result['link_count']}")
        print(f"\nGenerated heatmaps:")
        for name, path in result['heatmap_paths'].items():
            print(f"  - {name}: {path}")
        
        print(f"\nVisual Features:")
        print(f"  - Bridge nodes: {result['visual_features']['topological_anomalies']['bridge_count']}")
        print(f"  - Dead-end nodes: {result['visual_features']['topological_anomalies']['dead_end_count']}")
        print(f"  - Pressure range: {result['visual_features']['pressure_patterns'].get('range', 'N/A'):.2f} m")
        print(f"  - Flow balance score: {result['visual_features']['symmetry_metrics'].get('flow_balance_score', 'N/A'):.3f}")
        
        print(f"\nVLM Prompt Template saved for analysis.")
    else:
        print(f"File not found: {inp_path}")
