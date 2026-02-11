"""
MM-WDS 框架的视觉感知模块。

该模块实现了第 2.2.3 节中描述的视觉模态 (M_Vis)，
通过热图生成和视觉-语言模型 (VLM) 集成来实现宏观尺度的空间模式识别。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import networkx as nx


# 生成的可视化输出目录
VISUAL_OUTPUT_DIR = os.path.join(os.getcwd(), "visual_outputs")
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)


def normalize_values(values, v_min=None, v_max=None):
    """
    将值归一化到 [0, 1] 范围以便进行颜色映射。
    
    实现了论文中的公式 11a：
    C(P_i) = cmap((P_i - P_min) / (P_max - P_min))
    
    参数:
        values: 要归一化的数组
        v_min: 可选的最小值（如果为 None 则使用 min(values)）
        v_max: 可选的最大值（如果为 None 则使用 max(values)）
    
    返回:
        归一化到 [0, 1] 范围的值
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
    根据流速大小计算链接（管道）宽度。
    
    实现了论文中的公式 11b：
    w_ij = w_min + (w_max - w_min) * |v_ij| / max|v_kl|
    
    参数:
        velocities: 流速数组
        w_min: 最小线宽
        w_max: 最大线宽
    
    返回:
        线宽数组
    """
    velocities = np.abs(np.array(velocities))
    v_max = np.max(velocities) if len(velocities) > 0 else 1.0
    
    if v_max < 1e-10:
        return np.full_like(velocities, w_min, dtype=float)
    
    normalized = velocities / v_max
    return w_min + (w_max - w_min) * normalized


def generate_pressure_heatmap(wn, results, save_path=None, title="Pressure Distribution Heatmap"):
    """
    生成供水管网的压力热图可视化。
    
    参数:
        wn: WNTR WaterNetworkModel 对象
        results: WNTR 模拟结果
        save_path: 图表的保存路径（可选）
        title: 图表标题
    
    返回:
        保存的图表路径，如果未保存则返回 None
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 获取节点位置
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    # 提取平均压力
    node_names = list(wn.node_name_list)
    pressures = []
    for node_name in node_names:
        try:
            p = results.node['pressure'][node_name].mean()
            pressures.append(p)
        except:
            pressures.append(0)
    
    pressures = np.array(pressures)
    
    # 为颜色映射归一化压力（公式 11a）
    p_min, p_max = np.min(pressures), np.max(pressures)
    normalized_pressures = normalize_values(pressures, p_min, p_max)
    
    # 创建发散颜色映射（RdYlBu - 蓝色代表低压，红色代表高压）
    cmap = plt.cm.RdYlBu_r  # 反转，使红色代表高压
    
    # 先绘制边（背景）
    G = wn.to_graph()
    edge_positions = []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    
    if edge_positions:
        lc = LineCollection(edge_positions, colors='gray', linewidths=0.5, alpha=0.5)
        ax.add_collection(lc)
    
    # 绘制带有压力颜色的节点
    x_coords = [pos[n][0] for n in node_names if n in pos]
    y_coords = [pos[n][1] for n in node_names if n in pos]
    valid_pressures = [normalized_pressures[i] for i, n in enumerate(node_names) if n in pos]
    
    scatter = ax.scatter(x_coords, y_coords, c=valid_pressures, cmap=cmap,
                         s=30, alpha=0.8, edgecolors='black', linewidths=0.3)
    
    # 添加颜色条
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
    生成基于流速线宽的流量可视化。
    
    实现公式 11b 以进行链接（管道）宽度缩放。
    
    参数:
        wn: WNTR WaterNetworkModel 对象
        results: WNTR 模拟结果
        save_path: 图表的保存路径（可选）
        title: 图表标题
    
    返回:
        保存的图表路径
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 获取节点位置
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    # 提取链接（管道）流速和流量
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
        
        # 为了颜色归一化流速
        v_norm = normalize_values(velocities)
        cmap = plt.cm.YlOrRd
        
        # 绘制具有不同宽度和颜色的链接（管道）
        for i, d in enumerate(link_data):
            color = cmap(v_norm[i])
            ax.plot([d['start'][0], d['end'][0]], [d['start'][1], d['end'][1]],
                    color=color, linewidth=widths[i], alpha=0.7, solid_capstyle='round')
            
            # 为显著流量添加流向箭头
            if d['velocity'] > np.mean(velocities):
                mid_x = (d['start'][0] + d['end'][0]) / 2
                mid_y = (d['start'][1] + d['end'][1]) / 2
                dx = (d['end'][0] - d['start'][0]) * 0.1
                dy = (d['end'][1] - d['start'][1]) * 0.1
                if d['flowrate'] < 0:
                    dx, dy = -dx, -dy
                ax.annotate('', xy=(mid_x + dx, mid_y + dy), xytext=(mid_x, mid_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # 绘制节点
    node_x = [pos[n][0] for n in wn.node_name_list if n in pos]
    node_y = [pos[n][1] for n in wn.node_name_list if n in pos]
    ax.scatter(node_x, node_y, c='steelblue', s=20, alpha=0.8, zorder=5)
    
    # 为流速添加颜色条
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
    生成同时显示压力和流量的组合可视化图形。
    
    参数:
        wn: WNTR WaterNetworkModel 对象
        results: WNTR 模拟结果
        save_path: 图表的保存路径（可选）
        title: 图表标题
    
    返回:
        保存的图表路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 获取节点位置
    pos = {}
    for node_name, node in wn.nodes():
        if hasattr(node, 'coordinates') and node.coordinates is not None:
            pos[node_name] = node.coordinates
        else:
            pos[node_name] = (0, 0)
    
    G = wn.to_graph()
    
    # --- 左面板：压力热图 ---
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
    
    # 绘制边
    edge_positions = []
    for u, v in G.edges():
        if u in pos and v in pos:
            edge_positions.append([pos[u], pos[v]])
    if edge_positions:
        lc = LineCollection(edge_positions, colors='gray', linewidths=0.5, alpha=0.5)
        ax1.add_collection(lc)
    
    # 绘制带有压力的节点
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
    
    # --- 右面板：流量可视化 ---
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
    
    # 绘制节点
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
    返回用于分析管网热图的 VLM 提示词模板。
    
    这实现了第 2.2.3 节中的 VLM 提示词工程部分。
    
    返回:
        用于 VLM 分析的结构化提示词字符串
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
    从管网状态中提取定量视觉特征。
    
    这提供了补充 VLM 视觉分析的数值特征，
    实现了第 2.2.3 节中的视觉特征提取部分。
    
    参数:
        wn: WNTR WaterNetworkModel 对象
        results: WNTR 模拟结果
    
    返回:
        包含提取的视觉特征的字典
    """
    features = {
        'topological_anomalies': {},
        'pressure_patterns': {},
        'flow_patterns': {},
        'symmetry_metrics': {}
    }
    
    G = wn.to_graph().to_undirected()
    
    # --- 拓扑异常 ---
    # 寻找关节点（桥接点）
    try:
        articulation_points = list(nx.articulation_points(G))
        features['topological_anomalies']['bridge_nodes'] = articulation_points
        features['topological_anomalies']['bridge_count'] = len(articulation_points)
    except:
        features['topological_anomalies']['bridge_nodes'] = []
        features['topological_anomalies']['bridge_count'] = 0
    
    # 寻找死端节点（度为 1）
    dead_ends = [n for n in G.nodes() if G.degree(n) == 1]
    features['topological_anomalies']['dead_end_nodes'] = dead_ends
    features['topological_anomalies']['dead_end_count'] = len(dead_ends)
    
    # --- 压力模式 ---
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
        
        # 变异系数（衡量均匀性）
        if np.mean(pressures) > 0:
            features['pressure_patterns']['cv'] = float(np.std(pressures) / np.mean(pressures))
        else:
            features['pressure_patterns']['cv'] = 0.0
    
    # --- 流量模式 ---
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
        
        # 识别高流量管道（前 10%）
        threshold = np.percentile(velocities, 90)
        high_flow_count = np.sum(velocities > threshold)
        features['flow_patterns']['high_flow_pipe_count'] = int(high_flow_count)
        features['flow_patterns']['high_flow_threshold'] = float(threshold)
    
    # --- 对称性指标 ---
    # 使用基尼系数计算流量平衡
    if velocities is not None and len(velocities) > 0:
        sorted_v = np.sort(velocities)
        n = len(sorted_v)
        cumsum = np.cumsum(sorted_v)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_v))) / (n * np.sum(sorted_v)) - (n + 1) / n
        features['symmetry_metrics']['flow_gini_coefficient'] = float(abs(gini))
        features['symmetry_metrics']['flow_balance_score'] = float(1 - abs(gini))  # 1 = 完美平衡
    
    return features


def analyze_network_visually(inp_file_path, output_dir=None):
    """
    供水管网的完整视觉分析流水线。
    
    这是视觉感知模块的主入口点，
    实现了第 2.2.3 节中描述的完整工作流。
    
    参数:
        inp_file_path: EPANET .inp 文件的路径
        output_dir: 输出文件的目录（可选）
    
    返回:
        包含以下内容的字典：
        - 'heatmap_paths': 生成的热图图像路径
        - 'visual_features': 提取的定量特征
        - 'vlm_prompt': 用于 VLM 分析的提示词模板
    """
    import wntr
    
    if output_dir is None:
        output_dir = VISUAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载管网并运行模拟
    wn = wntr.network.WaterNetworkModel(inp_file_path)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # 从输入文件生成基础文件名
    base_name = os.path.splitext(os.path.basename(inp_file_path))[0]
    
    # 生成可视化
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
    
    # 提取视觉特征
    visual_features = extract_visual_features(wn, results)
    
    # 获取 VLM 提示词
    vlm_prompt = get_vlm_prompt_template()
    
    return {
        'heatmap_paths': heatmap_paths,
        'visual_features': visual_features,
        'vlm_prompt': vlm_prompt,
        'network_name': base_name,
        'node_count': wn.num_nodes,
        'link_count': wn.num_links
    }


# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
    else:
        # 默认测试文件
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
