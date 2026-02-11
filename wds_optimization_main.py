"""
Water Distribution Network Multi-Objective Optimization Main Program
Uses NSGA-II to optimize boundary pipe open/close configuration
Objectives: max(FEF), max(HRE), max(MRE), max(NR), min(open_pipes)
"""
import os
import json
import pickle
import wntr

from optimization_utils.boundary import find_boundary_pipes, get_boundary_info
from optimization_utils.nsga2 import run_nsga2_optimization
from optimization_utils.evaluation import select_best_solution


from optimization_utils.cleanup import cleanup

def main():
    try:
        # ==================== Configuration Parameters ====================
        inp_file = 'dataset/Exa7.inp'               # Network file
        partition_file = 'partition_results/partitions.pkl'  # Partition results
        output_dir = 'optimization_results'         # Output directory
        
        # NSGA-II parameters
        pop_size = 5      # Population size
        n_gen = 20         # Number of generations
        hmin, hdes = 10, 30  # Pressure thresholds (m)
        
        # Partition range (only optimize partitions within specified range)
        partition_range = (2, 15)  # Only optimize 2-20 partition configurations
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ==================== Step 1: Load Data ====================
        print("=" * 60)
        print("Step 1: Loading network and partition data...")
        wn = wntr.network.WaterNetworkModel(inp_file)
        print(f"  Network: {inp_file}")
        print(f"  Junctions: {len(wn.junction_name_list)}, Pipes: {len(wn.pipe_name_list)}")
        
        with open(partition_file, 'rb') as f:
            partitions = pickle.load(f)
        print(f"  Loaded {len(partitions)} partition configurations")
        
        # Filter partition range
        partitions = {k: v for k, v in partitions.items() 
                      if partition_range[0] <= k <= partition_range[1]}
        print(f"  Filtered to {len(partitions)} partitions in range {partition_range}")
        
        # ==================== Step 2-6: Optimize Each Partition ====================
        all_results = {}
        for num_comm in sorted(partitions.keys()):
            print("\n" + "=" * 60)
            print(f"Optimizing partition with {num_comm} communities...")
            
            partition_data = partitions[num_comm]
            node_to_comm = partition_data['node_to_community']
            
            # Step 2: Identify boundary pipes
            boundary_pipes = find_boundary_pipes(wn, node_to_comm)
            info = get_boundary_info(boundary_pipes)
            print(f"  Boundary pipes: {info['count']}")
            
            if info['count'] == 0:
                print("  No boundary pipes, skipping optimization")
                all_results[num_comm] = None
                continue
            
            # Step 3-4: NSGA-II optimization
            print(f"  Running NSGA-II (pop={pop_size}, gen={n_gen})...")
            result = run_nsga2_optimization(
                wn, boundary_pipes, node_to_comm, num_comm,
                pop_size=pop_size, n_gen=n_gen, hmin=hmin, hdes=hdes
            )
            
            # Display results
            obj = result['best_objectives']
            print(f"  Best solution:")
            print(f"    FEF={obj['FEF']:.4f}, HRE={obj['HRE']:.4f}, "
                  f"MRE={obj['MRE']:.4f}, NR={obj['NR']:.4f}, Open={obj['open_pipes']}")
            print(f"  Pareto solutions: {len(result['pareto_front'])}")
            
            # Step 5: Save partition results
            result_file = os.path.join(output_dir, f'partition_{num_comm}_results.json')
            save_data = {
                'num_communities': num_comm,
                'boundary_pipes': [p[0] for p in boundary_pipes],
                'best_objectives': result['best_objectives'],
                'best_config': result['best_config'],
                'pareto_front': result['pareto_front'],
                'pareto_solutions': result['pareto_solutions']
            }
            with open(result_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"  Saved: {result_file}")
            
            all_results[num_comm] = result
        
        # ==================== Step 7: Select Global Optimal Solution ====================
        print("\n" + "=" * 60)
        print("Step 7: Selecting global optimal solution...")
        
        best = select_best_solution(all_results)
        if best:
            print(f"  Best partition: {best['num_communities']} communities")
            print(f"  Score: {best['score']:.4f}")
            print(f"  Objectives: FEF={best['objectives']['FEF']:.4f}, "
                  f"HRE={best['objectives']['HRE']:.4f}, "
                  f"MRE={best['objectives']['MRE']:.4f}, "
                  f"NR={best['objectives']['NR']:.4f}, "
                  f"Open={best['objectives']['open_pipes']}")
            
            # Save global optimal solution
            global_file = os.path.join(output_dir, 'global_optimal.json')
            with open(global_file, 'w') as f:
                json.dump(best, f, indent=2)
            print(f"  Saved: {global_file}")
            
        print("\n" + "=" * 60)
        print("Optimization completed!")
        print(f"  Results saved in: {output_dir}/")

    finally:
        # Clean up temporary files regardless of success or failure
        print("\nCleaning up temporary files...")
        cleanup()


if __name__ == '__main__':
    main()

