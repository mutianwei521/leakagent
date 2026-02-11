"""
使用 CDlib 中的 Louvain 算法进行网络分区。
"""
import numpy as np
import networkx as nx
from cdlib import algorithms


def normalize_edge_weights(G):
    """
    将边权重归一化到 [0, 1] 范围。

    参数:
        G: 带有边权重的 NetworkX 图

    返回:
        G_normalized: 具有归一化边权重的新图
    """
    G_normalized = G.copy()

    # 获取所有边权重
    weights = [data.get('weight', 1.0) for u, v, data in G.edges(data=True)]

    if len(weights) == 0:
        return G_normalized

    min_weight = min(weights)
    max_weight = max(weights)

    # 将权重归一化到 [0, 1] 范围
    if max_weight > min_weight:
        for u, v, data in G_normalized.edges(data=True):
            original_weight = data.get('weight', 1.0)
            normalized_weight = (original_weight - min_weight) / (max_weight - min_weight)
            # 避免零权重（可能会导致某些算法出现问题）
            G_normalized[u][v]['weight'] = max(normalized_weight, 0.001)

    return G_normalized


def run_louvain_partitioning(G, resolution_range=None, num_iterations=50):
    """
    多次运行不同分辨率的 Louvain 算法，以获得各种分区大小。

    参数:
        G: NetworkX 图
        resolution_range: 分辨率参数的范围元组 (min, max)
        num_iterations: 要尝试的不同分辨率值的数量

    返回:
        all_partitions: (num_communities, partition) 元组的列表
    """
    if resolution_range is None:
        resolution_range = (0.1, 5.0)

    # 在分区之前归一化边权重
    G_normalized = normalize_edge_weights(G)
    print(f"  Edge weights normalized to [0, 1] range")

    all_partitions = []
    resolutions = np.linspace(resolution_range[0], resolution_range[1], num_iterations)

    for res in resolutions:
        try:
            # 在归一化的图上使用当前分辨率运行 Louvain
            partition = algorithms.louvain(G_normalized, weight='weight', resolution=res)
            num_communities = len(partition.communities)
            
            # 存储带有节点分配的分区结果
            node_to_community = {}
            for comm_id, community in enumerate(partition.communities):
                for node in community:
                    node_to_community[node] = comm_id
            
            all_partitions.append((num_communities, node_to_community, res))
        except Exception as e:
            print(f"Warning: Louvain failed at resolution {res}: {e}")
            continue
    
    return all_partitions


def extract_unique_partitions(all_partitions):
    """
    提取唯一的分区编号，为每个分区计数仅保留最后一次出现（迭代中最优的）。
    
    参数:
        all_partitions: (num_communities, node_to_community, resolution) 元组的列表
        
    返回:
        unique_partitions: 字典 {num_communities: node_to_community}
    """
    unique_partitions = {}
    
    # 遍历所有分区，后出现的分区会覆盖先出现的分区
    for num_comm, node_to_comm, res in all_partitions:
        unique_partitions[num_comm] = {
            'node_to_community': node_to_comm,
            'resolution': res
        }
    
    # Sort by number of communities
    sorted_partitions = dict(sorted(unique_partitions.items()))

    return sorted_partitions


def merge_communities_by_connectivity(G, node_to_community, target_num):
    """
    通过合并连接最紧密的社区对，使社区数量达到目标值。

    参数:
        G: NetworkX 图
        node_to_community: 映射节点名称到社区 ID 的字典
        target_num: 目标社区数量

    返回:
        new_node_to_community: 包含合并后社区分配的字典
    """
    current_communities = max(node_to_community.values()) + 1

    if current_communities <= target_num:
        return node_to_community

    # Create a copy to modify
    node_to_comm = node_to_community.copy()

    while current_communities > target_num:
        # 计算社区之间的连接性
        comm_connectivity = {}
        for u, v, data in G.edges(data=True):
            comm_u = node_to_comm.get(u)
            comm_v = node_to_comm.get(v)
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                pair = tuple(sorted([comm_u, comm_v]))
                weight = data.get('weight', 1.0)
                comm_connectivity[pair] = comm_connectivity.get(pair, 0) + weight

        if not comm_connectivity:
            break

        # 找到连接最紧密的一对社区
        best_pair = max(comm_connectivity, key=comm_connectivity.get)
        comm_to_merge, comm_to_keep = best_pair

        # 合并：将 comm_to_merge 中的所有节点重新分配给 comm_to_keep
        for node in node_to_comm:
            if node_to_comm[node] == comm_to_merge:
                node_to_comm[node] = comm_to_keep

        # 重新编号社区以使编号连续
        unique_comms = sorted(set(node_to_comm.values()))
        comm_mapping = {old: new for new, old in enumerate(unique_comms)}
        node_to_comm = {node: comm_mapping[comm] for node, comm in node_to_comm.items()}

        current_communities = len(unique_comms)

    return node_to_comm


def generate_merged_partitions(G, base_partition, target_range=(2, 15)):
    """
    通过从基础分区开始合并，生成具有较少社区的分区。

    参数:
        G: NetworkX 图
        base_partition: 字典 {node: community_id} - 合并的基础分区
        target_range: 目标社区数量的元组 (min, max)

    返回:
        merged_partitions: 字典 {num_communities: {'node_to_community': dict, 'resolution': 'merged'}}
    """
    merged_partitions = {}

    for target_num in range(target_range[0], target_range[1] + 1):
        merged = merge_communities_by_connectivity(G, base_partition, target_num)
        actual_num = len(set(merged.values()))

        if actual_num not in merged_partitions:
            merged_partitions[actual_num] = {
                'node_to_community': merged,
                'resolution': 'merged'
            }

    return merged_partitions


def extract_partitions_with_merge(G, all_partitions, merge_range=(2, 15), target_k=None):
    """
    从 Louvain 结果中提取唯一分区，并为较小的社区计数生成合并分区。
    如果指定了 target_k，还需确保其存在。

    参数:
        G: NetworkX 图
        all_partitions: (num_communities, node_to_community, resolution) 元组的列表
        merge_range: 合并分区目标范围的元组 (min, max)
        target_k: 所需的特定分区计数

    返回:
        unique_partitions: Louvain 分区和合并分区的合集字典
    """
    # Step 1: Extract unique Louvain partitions
    unique_partitions = extract_unique_partitions(all_partitions)
    partition_counts = sorted(unique_partitions.keys())

    print(f"  Louvain partition counts: {partition_counts}")
    print(f"  Louvain range: {min(partition_counts, default=0)} to {max(partition_counts, default=0)} communities")

    # 用于查找合并到目标计数所需的有效基础分区的辅助函数
    def find_best_base(target):
        # 我们需要一个计数大于目标的分区
        candidates = [k for k in partition_counts if k > target]
        if not candidates:
            return None
        # 选择大于目标值的最小计数（即最接近的上方邻居）
        best_k = min(candidates)
        return unique_partitions[best_k]['node_to_community']

    # 第 2 步：为较小的社区计数生成合并分区（默认范围）
    # 修正逻辑：为范围内的每个目标找到合适的基准
    print(f"\n  Generating merged partitions for {merge_range[0]}-{merge_range[1]} communities...")
    
    targets_to_gen = set(range(merge_range[0], merge_range[1] + 1))
    if target_k is not None:
        targets_to_gen.add(target_k)

    merged_partitions = {}
    
    for t_k in sorted(list(targets_to_gen)):
        if t_k in unique_partitions:
            continue # Louvain 结果中已存在
            
        base_partition = find_best_base(t_k)
        if base_partition:
            merged = merge_communities_by_connectivity(G, base_partition, t_k)
            actual_num = len(set(merged.values()))
            if actual_num == t_k:
                 merged_partitions[actual_num] = {
                    'node_to_community': merged,
                    'resolution': 'merged'
                }

    merged_counts = sorted(merged_partitions.keys())
    print(f"  Merged partition counts: {merged_counts}")

    # 第 3 步：合并（如果强制执行，合并结果在重叠处具有优先权，但在这里我们只处理缺失的部分）
    for num_comm, data in merged_partitions.items():
        unique_partitions[num_comm] = data

    # Re-sort
    unique_partitions = dict(sorted(unique_partitions.items()))
    partition_counts = sorted(unique_partitions.keys())

    print(f"  Final partition counts: {partition_counts}")
    if partition_counts:
        print(f"  Final range: {min(partition_counts)} to {max(partition_counts)} communities")

    return unique_partitions
