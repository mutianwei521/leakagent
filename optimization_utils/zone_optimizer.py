"""
区域优化模块
供代理调用的优化操作包装函数。
"""
import os
import json
import hashlib
import wntr

from optimization_utils.boundary import find_boundary_pipes, get_boundary_info


def compute_md5(file_path):
    """计算文件的 MD5 哈希值。"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_boundary_pipes(inp_file_path, target_k=None):
    """
    查找并返回区域间的边界管道。
    根据 INP 文件的 MD5 哈希值自动加载分区结果。
    
    参数:
        inp_file_path: INP 文件的路径
        target_k: 要分析的特定分区计数（可选）
        
    返回:
        包含边界管道信息的字典
    """
    try:
        # 基于 MD5 查找分区结果
        md5 = compute_md5(inp_file_path)
        base_dir = os.path.join(os.getcwd(), 'static', 'partition_results', md5)
        summary_file = os.path.join(base_dir, 'partition_summary.json')
        
        if not os.path.exists(summary_file):
            return {
                "status": "error",
                "error": "未找到分区结果。请先使用 'network_partitioner' 进行分区。"
            }
        
        # 加载分区数据
        with open(summary_file, 'r') as f:
            partition_data = json.load(f)
        
        available_partitions = list(partition_data.keys())
        
        # 选择要分析的分区
        if target_k is not None:
            target_key = str(target_k)
            if target_key not in partition_data:
                return {
                    "status": "error",
                    "error": f"未找到分区 {target_k}。可用分区：{available_partitions}"
                }
            selected_partition = partition_data[target_key]
        else:
            # 使用第一个（或唯一的）可用分区
            if len(available_partitions) == 1:
                target_key = available_partitions[0]
                selected_partition = partition_data[target_key]
            else:
                return {
                    "status": "error",
                    "error": f"存在多个可用分区：{available_partitions}。请使用 num_partitions 指定其中一个。"
                }
        
        # 获取节点到社区的映射
        node_to_comm = selected_partition.get('node_assignments', {})
        
        # 加载管网模型
        wn = wntr.network.WaterNetworkModel(inp_file_path)
        
        # 查找边界管道
        boundary_pipes = find_boundary_pipes(wn, node_to_comm)
        info = get_boundary_info(boundary_pipes)
        
        # 构建详细响应
        pipe_details = []
        for pipe_name, node1, node2, comm1, comm2 in boundary_pipes:
            pipe = wn.get_link(pipe_name)
            pipe_details.append({
                "pipe": pipe_name,
                "from_node": node1,
                "to_node": node2,
                "zone_from": comm1,
                "zone_to": comm2,
                "diameter_mm": round(pipe.diameter * 1000, 1),
                "length_m": round(pipe.length, 1)
            })
        
        return {
            "status": "success",
            "partition_count": int(target_key),
            "boundary_pipe_count": info['count'],
            "boundary_pipes": pipe_details,
            "zone_connections": info['connections']
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_zone_optimization(inp_file_path, partition_json_path=None, target_k=None, 
                          pop_size=5, n_gen=20):
    """
    针对特定的分区计数运行 NSGA-II 优化。
    
    参数:
        inp_file_path: INP 文件的路径
        partition_json_path: 可选的特定 JSON 路径（如果为 None，则通过 MD5 自动检测）
        target_k: 要优化的特定分区计数（如果存在多个，则必填）
        pop_size: NSGA-II 的种群大小
        n_gen: 迭代次数（代数）
    """
    try:
        from optimization_utils.nsga2 import run_nsga2_optimization
        
        # 1. 定位/加载分区数据
        if partition_json_path is None:
            md5 = compute_md5(inp_file_path)
            base_dir = os.path.join(os.getcwd(), 'static', 'partition_results', md5)
            summary_file = os.path.join(base_dir, 'partition_summary.json')
        else:
            summary_file = partition_json_path
            base_dir = os.path.dirname(summary_file)
            
        if not os.path.exists(summary_file):
             return {"status": "error", "error": "未找到分区摘要文件。请先运行分区。"}
             
        with open(summary_file, 'r') as f:
            partition_data = json.load(f)
            
        # 2. 选择目标分区
        available_partitions = list(partition_data.keys())
        
        if target_k is not None:
            target_key = str(target_k)
            if target_key not in partition_data:
                 return {"status": "error", "error": f"在结果中未找到分区计数 {target_k}。"}
        else:
            if len(available_partitions) == 1:
                target_key = available_partitions[0]
            else:
                return {
                    "status": "error", 
                    "error": f"发现多个分区：{available_partitions}。请通过 num_partitions 指定一个。"
                }
                
        # 3. 为优化准备数据
        selected_partition = partition_data[target_key]
        num_comm = int(target_key)
        node_to_comm = selected_partition.get('node_assignments', {})
        
        wn = wntr.network.WaterNetworkModel(inp_file_path)
        boundary_pipes = find_boundary_pipes(wn, node_to_comm)
        
        if not boundary_pipes:
            return {"status": "error", "error": "此分区未找到边界管道。"}
            
        # 4. 运行 NSGA-II
        print(f"正在对 {target_key} 个分区进行优化，包含 {len(boundary_pipes)} 条边界管道...")
        
        result = run_nsga2_optimization(
            wn, boundary_pipes, node_to_comm, num_comm,
            pop_size=pop_size, n_gen=n_gen
        )
        
        # 5. 保存结果（匹配 partition_X_results.json 的格式）
        opt_filename = f"optimization_{target_key}_zones.json"
        opt_path = os.path.join(base_dir, opt_filename)
        
        save_data = {
            'num_communities': num_comm,
            'boundary_pipes': [p[0] for p in boundary_pipes],
            'best_objectives': result['best_objectives'],
            'best_config': result['best_config'],
            'pareto_front': result['pareto_front'],
            'pareto_solutions': result['pareto_solutions']
        }
        
        with open(opt_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        return {
            "status": "success",
            "msg": f"分区 {target_key} 的优化已完成。",
            "optimization_file": f"static/partition_results/{compute_md5(inp_file_path)}/{opt_filename}",
            "best_objectives": result['best_objectives'],
            "boundary_count": len(boundary_pipes),
            "pareto_count": len(result['pareto_front'])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
