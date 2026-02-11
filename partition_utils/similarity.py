"""
为网络分区构建相似性矩阵。
基于 references/cd_main.m 中的 linkNode 方法。
"""
import numpy as np
import networkx as nx


def build_similarity_matrix(wn, avg_pressure):
    """
    构建相似性矩阵 A，其中 A[i,j] = (avg_pressure[i] + avg_pressure[j]) / 2
    这遵循了 MATLAB 程序 cd_main.m 中使用 linkNode 的方法。
    
    参数:
        wn: WNTR 供水管网模型
        avg_pressure: 每个节点的平均压力字典
        
    返回:
        G: 基于压力相似性带有边权重的 NetworkX 图
        node_list: 节点名称列表
    """
    # 获取所有节点（包括水池、水库）以确保可视化中的连通性
    junction_names = list(wn.junction_name_list)
    node_list = list(wn.node_name_list)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for node in node_list:
        G.add_node(node)
    
    # 基于管道/链接添加边（类似于 MATLAB 中的 linkNode）
    # 对于每个链接，添加一条边，权重为连接节点压力的平均值
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        # 考虑所有节点之间的边（包括水库/水池）
        if start_node in node_list and end_node in node_list:
            # 计算权重为节点压力的平均值（遵循 MATLAB 模式）
            if start_node in avg_pressure and end_node in avg_pressure:
                weight = (avg_pressure[start_node] + avg_pressure[end_node]) / 2
                # 通过使用绝对值处理负压
                # 避免零值导致某些算法出错
                G.add_edge(start_node, end_node, weight=weight)
    
    return G, node_list


def create_network_graph(wn, avg_pressure):
    """
    创建带有位置信息的 NetworkX 图，用于分区。
    
    参数:
        wn: WNTR 供水管网模型
        avg_pressure: 平均压力字典
        
    返回:
        G: 带有权重和位置信息的 NetworkX 图
        pos: 节点位置字典
    """
    G, node_list = build_similarity_matrix(wn, avg_pressure)
    
    # 从网络坐标中获取节点位置
    pos = {}
    for node_name in node_list:
        node = wn.get_node(node_name)
        pos[node_name] = (node.coordinates[0], node.coordinates[1])
    
    return G, pos

