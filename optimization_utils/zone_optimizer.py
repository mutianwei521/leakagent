"""
Zone Optimization Module
Wrapper functions for optimization operations called by the agent.
"""
import os
import json
import hashlib
import wntr

from optimization_utils.boundary import find_boundary_pipes, get_boundary_info


def compute_md5(file_path):
    """Compute MD5 hash of a file."""
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
    Run NSGA-II optimization for a specific partition count.
    
    Args:
        inp_file_path: Path to INP file
        partition_json_path: Optional specific JSON path (if None, auto-detected via MD5)
        target_k: Specific partition count to optimize (required if multiple exist)
        pop_size: Population size for NSGA-II
        n_gen: Number of iterations (generations)
    """
    try:
        from optimization_utils.nsga2 import run_nsga2_optimization
        
        # 1. Locate/Load partition data
        if partition_json_path is None:
            md5 = compute_md5(inp_file_path)
            base_dir = os.path.join(os.getcwd(), 'static', 'partition_results', md5)
            summary_file = os.path.join(base_dir, 'partition_summary.json')
        else:
            summary_file = partition_json_path
            base_dir = os.path.dirname(summary_file)
            
        if not os.path.exists(summary_file):
             return {"status": "error", "error": "Partition summary file not found. Please run partitioning first."}
             
        with open(summary_file, 'r') as f:
            partition_data = json.load(f)
            
        # 2. Select target partition
        available_partitions = list(partition_data.keys())
        
        if target_k is not None:
            target_key = str(target_k)
            if target_key not in partition_data:
                 return {"status": "error", "error": f"Partition count {target_k} not found in results."}
        else:
            if len(available_partitions) == 1:
                target_key = available_partitions[0]
            else:
                return {
                    "status": "error", 
                    "error": f"Multiple partitions found: {available_partitions}. Please specify one via num_partitions."
                }
                
        # 3. Prepare data for optimization
        selected_partition = partition_data[target_key]
        num_comm = int(target_key)
        node_to_comm = selected_partition.get('node_assignments', {})
        
        wn = wntr.network.WaterNetworkModel(inp_file_path)
        boundary_pipes = find_boundary_pipes(wn, node_to_comm)
        
        if not boundary_pipes:
            return {"status": "error", "error": "No boundary pipes found for this partition."}
            
        # 4. Run NSGA-II
        print(f"Optimizing {target_key} partitions, containing {len(boundary_pipes)} boundary pipes...")
        
        result = run_nsga2_optimization(
            wn, boundary_pipes, node_to_comm, num_comm,
            pop_size=pop_size, n_gen=n_gen
        )
        
        # 5. Save results (match partition_X_results.json format)
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
            "msg": f"Optimization for partition {target_key} complete.",
            "optimization_file": f"static/partition_results/{compute_md5(inp_file_path)}/{opt_filename}",
            "best_objectives": result['best_objectives'],
            "boundary_count": len(boundary_pipes),
            "pareto_count": len(result['pareto_front'])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
