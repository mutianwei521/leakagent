import os
import json
import pickle
import wntr
import hashlib
import sys

# 确保可以找到本地模块
sys.path.append(os.getcwd())

from partition_utils.hydraulic import run_hydraulic_simulation, calculate_average_pressure
from partition_utils.similarity import create_network_graph
from partition_utils.partitioning import run_louvain_partitioning, extract_partitions_with_merge
from partition_utils.visualization import save_all_partitions_plots

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run_partitioning_for_agent(inp_file_path, target_k=None):
    """
    运行代理的分区流水线全过程。
    将输出保存到 static/partition_results/{md5}/。
    返回一个包含摘要文本和结果文件列表的字典。
    如果提供了 target_k，则过滤结果，仅显示特定的分区计数（或者显示最接近的/报错）。
    """
    try:
        # 设置路径
        md5 = compute_md5(inp_file_path)
        base_dir = os.path.join(os.getcwd(), 'static', 'partition_results', md5)
        os.makedirs(base_dir, exist_ok=True)
        
        # 检查是否已处理（寻找 summary.json）
        summary_file = os.path.join(base_dir, 'partition_summary.json')
        
        unique_partitions = {}
        # 尝试加载现有摘要，以尽可能避免重新运行
        if os.path.exists(summary_file) and len(os.listdir(base_dir)) > 2:
             print(f"Partition results found for {md5}")
             with open(summary_file, 'r') as f:
                 summary_data = json.load(f)
             # 为过滤逻辑重构简化的 unique_partitions 结构
             for k_str, v in summary_data.items():
                 unique_partitions[int(k_str)] = v

        # 逻辑更新：如果提供了 target_k 但不在缓存/摘要中，我们必须重新运行来生成它（例如强制合并）
        # 或者，如果我们想确保文件仅包含目标（target），我们可能会强制重新运行或更新。
        # 给定用户的要求“仅保存指定的分区”，如果我们发现了一个包含所有内容的摘要，
        # 但用户只需要 15 个分区，我们可能需要覆盖它？
        # 我们采取一种策略：如果需要 target_k，检查我们是否已经拥有它。
        
        need_rerun = False
        if not unique_partitions:
            need_rerun = True
        elif target_k is not None:
            # 如果目标不在缓存中，我们需要运行
            if target_k not in unique_partitions:
                need_rerun = True
            # 注意：如果目标已在缓存中（例如来自之前的“全部”运行），我们可能会将其返回。
            # 但用户说“输出文件应该仅包含目标”。
            # 如果当前文件有 50 个条目，我们可能应该重新保存它并仅包含 1 个？
            # 出于效率考虑，我们只需加载需要的 1 个，清除其他内容并保存？
            else:
                 # 我们已经拥有它。让我们将内存中的字典过滤为仅包含这一个
                 # 如果需要严格遵守要求，我们会覆盖该文件。
                 print(f"Target {target_k} in cache. Filtering output file...")
                 unique_partitions = {target_k: unique_partitions[target_k]}
                 
                 # 仅使用这一个覆盖 JSON
                 summary = {}
                 summary[str(target_k)] = {
                     'num_communities': target_k,
                     'resolution': unique_partitions[target_k].get('resolution', 'cached'),
                     'node_assignments': unique_partitions[target_k]['node_assignments']
                 }
                 with open(summary_file, 'w') as f:
                     json.dump(summary, f, indent=2)

    except Exception as e:
        print(f"Partitioning error (check block): {e}")

    try:
        if need_rerun:
             # --- 流水线执行 ---
             print(f"Running partitioning for {inp_file_path}...")
             
             # 1. 加载并模拟
             wn = wntr.network.WaterNetworkModel(inp_file_path)
             results = run_hydraulic_simulation(wn)
             avg_pressure = calculate_average_pressure(results, wn)
             
             # 2. 构建图
             G, pos = create_network_graph(wn, avg_pressure)
             
             # 3. Louvain 分区
             resolution_range = (0.001, 3.0)
             num_iterations = 100 
             all_partitions = run_louvain_partitioning(G, resolution_range, num_iterations)
             
             # 4. 提取
             # 用户要求：基于 target_k 的动态合并范围
             # 如果设置了 target_k，我们仅需要该分区（为了速度和输出整洁度）。
             # 因此我们将范围设置为 (target_k, target_k)。
             # 如果未设置，我们默认使用 (2, 10) 进行标准概览。
             if target_k is not None:
                 m_range = (target_k, target_k)
             else:
                 m_range = (2, 10)
                 
             unique_partitions = extract_partitions_with_merge(G, all_partitions, merge_range=m_range, target_k=target_k)
             
             # --- 优化：保存前进行过滤 ---
             # （即使提取仅返回一个，这仍然可用作安全保护措施）
             if target_k is not None:
                 if target_k in unique_partitions:
                     unique_partitions = {target_k: unique_partitions[target_k]}
                 else:
                     # 通常在使用强制合并时不会发生这种情况，但作为安全备选
                     all_counts = sorted(list(unique_partitions.keys()))
                     closest_k = min(all_counts, key=lambda x: abs(x - target_k))
                     unique_partitions = {closest_k: unique_partitions[closest_k]}

             # 5. 保存结果
             summary = {}
             for num_comm, data in unique_partitions.items():
                 summary[str(num_comm)] = {
                     'num_communities': num_comm,
                     'resolution': data['resolution'],
                     'node_assignments': data['node_to_community']
                 }
             
             with open(summary_file, 'w') as f:
                 json.dump(summary, f, indent=2)
                 
             # 保存绘图（unique_partitions 已被过滤）
             save_all_partitions_plots(G, pos, unique_partitions, base_dir)
        
        # 6. 格式化返回值
        # 此时 unique_partitions 仅包含我们需要的内容（如果没有指定 target_k 则包含所有内容）
        all_counts = sorted(list(unique_partitions.keys()))
        
        plots = []
        msg = f"Partitioning completed. Found {len(unique_partitions)} solutions."
        
        if target_k is not None:
             if target_k in unique_partitions:
                 target_k_final = target_k
                 msg = f"Successfully partitioned into **{target_k_final}** zones."
             elif all_counts:
                 target_k_final = all_counts[0] # 应该是唯一的一个
                 msg = f"Could not find exact partition for {target_k} zones. Showing closest match: **{target_k_final}** zones."
             else:
                 target_k_final = 0
            
             p_file = f"partition_{target_k_final}_communities.png"
             if os.path.exists(os.path.join(base_dir, p_file)):
                plots = [f"static/partition_results/{md5}/{p_file}"]
        else:
             plots = [f"static/partition_results/{md5}/{f}" for f in os.listdir(base_dir) if f.endswith('.png')]
             plots.sort()

        return {
            "status": "success",
            "msg": msg,
            "summary_json": f"static/partition_results/{md5}/partition_summary.json",
            "plots": plots,
            "partition_counts": all_counts
        }
        
    except Exception as e:
        print(f"Partitioning error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
