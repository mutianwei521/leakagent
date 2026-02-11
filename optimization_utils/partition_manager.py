import os
import json
import pickle
import wntr
import hashlib
import sys

# Ensure local modules can be found
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
    Run the complete partitioning pipeline for the agent.
    Save output to static/partition_results/{md5}/.
    Returns a dictionary containing summary text and a list of result files.
    If target_k is provided, filters results to show only specific partition counts (or show nearest/error).
    """
    try:
        # Set paths
        md5 = compute_md5(inp_file_path)
        base_dir = os.path.join(os.getcwd(), 'static', 'partition_results', md5)
        os.makedirs(base_dir, exist_ok=True)
        
        # Check if already processed (look for summary.json)
        summary_file = os.path.join(base_dir, 'partition_summary.json')
        
        unique_partitions = {}
        # Try to load existing summary to avoid rerun if possible
        if os.path.exists(summary_file) and len(os.listdir(base_dir)) > 2:
             print(f"Partition results found for {md5}")
             with open(summary_file, 'r') as f:
                 summary_data = json.load(f)
             # Reconstruct simplified unique_partitions structure for filtering logic
             for k_str, v in summary_data.items():
                 unique_partitions[int(k_str)] = v

        # Logic update: If target_k is provided but not in cache/summary, we must rerun to generate it (e.g., forced merge)
        # Or, if we want to ensure the file only contains the target, we might force rerun or update.
        # Given user requirement "only save specified partition", if we find a summary containing everything,
        # but user only needs 15 partitions, do we overwrite it?
        # We adopt a strategy: If target_k is needed, check if we already have it.
        
        need_rerun = False
        if not unique_partitions:
            need_rerun = True
        elif target_k is not None:
            # If target is not in cache, we need to run
            if target_k not in unique_partitions:
                need_rerun = True
            # Note: If target is already in cache (e.g. from previous "all" run), we might return it.
            # But user says "output file should only contain target".
            # If current file has 50 entries, we probably should resave it containing only 1?
            # For efficiency, we just load the needed 1, clear others and save?
            else:
                 # We already have it. Let's filter the in-memory dictionary to contain only this one
                 # If we strictly follow requirement, we overwrite the file.
                 print(f"Target {target_k} in cache. Filtering output file...")
                 unique_partitions = {target_k: unique_partitions[target_k]}
                 
                 # Overwrite JSON with only this one
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
             # --- Pipeline Execution ---
             print(f"Running partitioning for {inp_file_path}...")
             
             # 1. Load and simulate
             wn = wntr.network.WaterNetworkModel(inp_file_path)
             results = run_hydraulic_simulation(wn)
             avg_pressure = calculate_average_pressure(results, wn)
             
             # 2. Build graph
             G, pos = create_network_graph(wn, avg_pressure)
             
             # 3. Louvain partitioning
             resolution_range = (0.001, 3.0)
             num_iterations = 100 
             all_partitions = run_louvain_partitioning(G, resolution_range, num_iterations)
             
             # 4. Extract
             # User requirement: Dynamic merge range based on target_k
             # If target_k is set, we only need that partition (for speed and output cleanliness).
             # So we set range to (target_k, target_k).
             # If not set, we default to (2, 10) for standard overview.
             if target_k is not None:
                 m_range = (target_k, target_k)
             else:
                 m_range = (2, 10)
                 
             unique_partitions = extract_partitions_with_merge(G, all_partitions, merge_range=m_range, target_k=target_k)
             
             # --- Optimization: Filter before saving ---
             # (Even if extraction returns only one, this still serves as a safeguard)
             if target_k is not None:
                 if target_k in unique_partitions:
                     unique_partitions = {target_k: unique_partitions[target_k]}
                 else:
                     # Usually this doesn't happen when using forced merge, but as a fallback
                     all_counts = sorted(list(unique_partitions.keys()))
                     closest_k = min(all_counts, key=lambda x: abs(x - target_k))
                     unique_partitions = {closest_k: unique_partitions[closest_k]}

             # 5. Save results
             summary = {}
             for num_comm, data in unique_partitions.items():
                 summary[str(num_comm)] = {
                     'num_communities': num_comm,
                     'resolution': data['resolution'],
                     'node_assignments': data['node_to_community']
                 }
             
             with open(summary_file, 'w') as f:
                 json.dump(summary, f, indent=2)
                 
             # Save plots (unique_partitions already filtered)
             save_all_partitions_plots(G, pos, unique_partitions, base_dir)
        
        # 6. Format return value
        # At this point unique_partitions only contains what we need (contains all if target_k not specified)
        all_counts = sorted(list(unique_partitions.keys()))
        
        plots = []
        msg = f"Partitioning completed. Found {len(unique_partitions)} solutions."
        
        if target_k is not None:
             if target_k in unique_partitions:
                 target_k_final = target_k
                 msg = f"Successfully partitioned into **{target_k_final}** zones."
             elif all_counts:
                 target_k_final = all_counts[0] # Should be the only one
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
