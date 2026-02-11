"""
网络分区结果的可视化工具函数。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_distinct_colors(n):
    """为社区生成 n 种不同的颜色。"""
    if n <= 10:
        colors = list(mcolors.TABLEAU_COLORS.values())[:n]
    else:
        cmap = plt.cm.get_cmap('tab20', n)
        colors = [cmap(i) for i in range(n)]
    return colors


def plot_partition(G, pos, node_to_community, num_communities, save_path):
    """
    绘制网络分区图并保存到带有图例的文件中。

    参数:
        G: NetworkX 图
        pos: 节点位置字典
        node_to_community: 映射节点名称到社区 ID 的字典
        num_communities: 此分区中的社区数量
        save_path: 文件保存路径
    """
    import networkx as nx
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 10))

    # 获取社区的不同颜色
    colors = get_distinct_colors(num_communities)

    # 按社区对节点进行分组
    community_nodes = {}
    for node in G.nodes():
        comm_id = node_to_community.get(node, 0)
        if comm_id not in community_nodes:
            community_nodes[comm_id] = []
        community_nodes[comm_id].append(node)

    # 绘制边
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')

    # 按社区绘制节点并创建图例句柄
    legend_handles = []
    for comm_id in sorted(community_nodes.keys()):
        nodes = community_nodes[comm_id]
        color = colors[comm_id % len(colors)]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                               node_color=[color], node_size=30, alpha=0.8)
        # 创建图例句柄
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=8, label=f'Partition {comm_id + 1}')
        legend_handles.append(handle)

    ax.set_title(f'Network Partition: {num_communities} Communities', fontsize=14)
    ax.axis('off')

    # 添加图例（根据社区数量调整列数）
    ncol = min(4, max(1, num_communities // 10 + 1))
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=8, ncol=ncol, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_all_partitions_plots(G, pos, unique_partitions, output_dir='partition_plots'):
    """
    保存所有唯一分区的可视化图形。
    
    参数:
        G: NetworkX 图
        pos: 节点位置字典
        unique_partitions: {社区数量: {'node_to_community': 字典, 'resolution': 浮点数}} 的字典
        output_dir: 保存图形的目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for num_comm, partition_data in unique_partitions.items():
        node_to_community = partition_data['node_to_community']
        save_path = os.path.join(output_dir, f'partition_{num_comm}_communities.png')
        plot_partition(G, pos, node_to_community, num_comm, save_path)
        print(f"Saved: {save_path}")

