"""
Water Distribution Network Partitioning Optimization
Based on MATLAB code from references/cd_main.m

This script performs:
1. Load water network from EPANET INP file
2. Run hydraulic simulation to get node pressures
3. Calculate average pressure for each node across all time steps
4. Build similarity matrix based on pressure averages
5. Use Louvain algorithm to partition the network
6. Extract unique partition solutions and save results
"""
import os
import json
import pickle
import numpy as np
import wntr

from partition_utils.hydraulic import run_hydraulic_simulation, calculate_average_pressure
from partition_utils.similarity import create_network_graph
from partition_utils.partitioning import run_louvain_partitioning, extract_partitions_with_merge
from partition_utils.visualization import save_all_partitions_plots


def main():
    # ==================== Configuration ====================
    inp_file = 'dataset/Exa7.inp'
    output_dir = 'partition_results'
    plot_dir = 'partition_plots'

    # Louvain parameters
    # Adjust the resolution range to obtain 2 to 50 partitions
    # The smaller the resolution, the fewer partitions; the larger the resolution, the more partitions
    resolution_range = (0.001, 3.0)   # Expanded range: smaller values result in fewer partitions, medium values result in more partitions
    num_iterations = 300              # More iterations to cover a wider range of partition numbers 
    
    # Save range configuration: Only save the results within the specified range of partitions
    # Setting to None means saving all partitions, and setting to (min, max) means only saving the results within the given range
    save_range = (2, 40)  # Only save the results with partition numbers ranging from 2 to 40
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Step 1: Load Network ====================
    print("=" * 60)
    print("Step 1: Loading water distribution network...")
    wn = wntr.network.WaterNetworkModel(inp_file)
    print(f"  Network loaded: {inp_file}")
    print(f"  Junctions: {len(wn.junction_name_list)}")
    print(f"  Pipes: {len(wn.pipe_name_list)}")
    print(f"  Tanks: {len(wn.tank_name_list)}")
    print(f"  Reservoirs: {len(wn.reservoir_name_list)}")
    
    # ==================== Step 2: Hydraulic Simulation ====================
    print("\n" + "=" * 60)
    print("Step 2: Running hydraulic simulation...")
    results = run_hydraulic_simulation(wn)
    print(f"  Simulation completed for {len(results.node['pressure'])} time steps")
    
    # ==================== Step 3: Calculate Average Pressure ====================
    print("\n" + "=" * 60)
    print("Step 3: Calculating average pressure for each node...")
    avg_pressure = calculate_average_pressure(results, wn)
    print(f"  Calculated average pressure for {len(avg_pressure)} nodes")
    
    # Save average pressure data
    pressure_file = os.path.join(output_dir, 'average_pressure.json')
    with open(pressure_file, 'w') as f:
        json.dump(avg_pressure, f, indent=2)
    print(f"  Saved: {pressure_file}")
    
    # ==================== Step 4: Build Similarity Matrix/Graph ====================
    print("\n" + "=" * 60)
    print("Step 4: Building similarity matrix (network graph with pressure weights)...")
    G, pos = create_network_graph(wn, avg_pressure)
    print(f"  Graph nodes: {G.number_of_nodes()}")
    print(f"  Graph edges: {G.number_of_edges()}")
    
    # ==================== Step 5: Run Louvain Partitioning ====================
    print("\n" + "=" * 60)
    print("Step 5: Running Louvain partitioning algorithm...")
    print(f"  Resolution range: {resolution_range}")
    print(f"  Number of iterations: {num_iterations}")
    
    all_partitions = run_louvain_partitioning(G, resolution_range, num_iterations)
    print(f"  Generated {len(all_partitions)} partition solutions")
    
    # ==================== Step 6: Extract Partitions with Merge ====================
    print("\n" + "=" * 60)
    print("Step 6: Extracting partitions (Louvain + merged for small counts)...")
    unique_partitions = extract_partitions_with_merge(G, all_partitions, merge_range=(2, 15))
    
    # ==================== Step 7: Filter and Save Results ====================
    print("\n" + "=" * 60)
    print("Step 7: Saving partitioning results...")

    # Filter partitions by save_range if specified
    if save_range is not None:
        filtered_partitions = {k: v for k, v in unique_partitions.items()
                               if save_range[0] <= k <= save_range[1]}
        print(f"  Save range: {save_range[0]} to {save_range[1]} communities")
        print(f"  Filtered from {len(unique_partitions)} to {len(filtered_partitions)} partitions")
    else:
        filtered_partitions = unique_partitions
        print(f"  Saving all {len(filtered_partitions)} partitions")

    # Save partition data as pickle
    partition_file = os.path.join(output_dir, 'partitions.pkl')
    with open(partition_file, 'wb') as f:
        pickle.dump(filtered_partitions, f)
    print(f"  Saved: {partition_file}")

    # Save partition summary as JSON
    summary = {}
    for num_comm, data in filtered_partitions.items():
        summary[str(num_comm)] = {
            'num_communities': num_comm,
            'resolution': data['resolution'],
            'node_assignments': data['node_to_community']
        }

    summary_file = os.path.join(output_dir, 'partition_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_file}")

    # ==================== Step 8: Generate Visualizations ====================
    print("\n" + "=" * 60)
    print("Step 8: Generating partition visualizations (saving to files)...")
    save_all_partitions_plots(G, pos, filtered_partitions, plot_dir)

    print("\n" + "=" * 60)
    print("Partitioning completed successfully!")
    print(f"  Results saved in: {output_dir}/")
    print(f"  Plots saved in: {plot_dir}/")
    

if __name__ == '__main__':
    main()

