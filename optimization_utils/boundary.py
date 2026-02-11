"""
边界管道识别与配置模块
识别分区边界管道并应用开/关配置
"""
import wntr
import copy


def find_boundary_pipes(wn, node_to_community):
    """
    识别边界管道：两端节点属于不同分区的管道
    Args:
        wn: wntr网络模型
        node_to_community: 节点到分区的映射 {node_name: community_id}
    Returns:
        boundary_pipes: 边界管道列表 [(pipe_name, node1, node2, comm1, comm2), ...]
    """
    boundary_pipes = []  # 存储边界管道信息
    for pipe_name, pipe in wn.pipes():  # 遍历所有管道
        node1 = pipe.start_node_name  # 起始节点
        node2 = pipe.end_node_name    # 终止节点
        # 获取节点所属分区（如果节点不在映射中，默认-1）
        comm1 = node_to_community.get(node1, -1)
        comm2 = node_to_community.get(node2, -1)
        if comm1 != comm2 and comm1 >= 0 and comm2 >= 0:  # 两端属于不同分区
            boundary_pipes.append((pipe_name, node1, node2, comm1, comm2))
    return boundary_pipes


def apply_boundary_config(wn_original, boundary_pipes, config):
    """
    根据配置关闭指定边界管道
    Args:
        wn_original: 原始wntr网络模型
        boundary_pipes: 边界管道列表
        config: 二进制配置数组 (1=开, 0=关)
    Returns:
        wn_modified: 修改后的网络模型副本
    """
    wn = copy.deepcopy(wn_original)  # 深拷贝避免修改原网络
    for i, (pipe_name, _, _, _, _) in enumerate(boundary_pipes):
        if config[i] == 0:  # 关闭管道
            pipe = wn.get_link(pipe_name)
            pipe.initial_status = wntr.network.LinkStatus.Closed  # 设置初始状态为关闭
    return wn


def get_boundary_info(boundary_pipes):
    """
    获取边界管道摘要信息
    Args:
        boundary_pipes: 边界管道列表
    Returns:
        dict: 边界管道信息摘要
    """
    return {
        'count': len(boundary_pipes),  # 边界管道总数
        'pipes': [p[0] for p in boundary_pipes],  # 管道名称列表
        'connections': [(p[3], p[4]) for p in boundary_pipes]  # 连接的分区对
    }

