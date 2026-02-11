"""
NSGA-II Optimization Module - Multi-objective optimization using pymoo library (Parallel Version)
"""
import numpy as np
import warnings
import os
import wntr
from concurrent.futures import ProcessPoolExecutor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .objectives import run_simulation, calculate_fef, calculate_hre, calculate_mre, calculate_nr
from .boundary import apply_boundary_config

# Global variables for Worker initialization to avoid repeated serialization of large objects
_global_wn = None
_global_params = None

def init_worker(wn_filepath, boundary_pipes, node_to_comm, num_comm, hmin, hdes):
    """Worker process initialization function"""
    global _global_wn, _global_params
    # Reload the WNTR model in each process to avoid pickling large objects
    # Or if passing the wn object is too slow, reloading here might be an option.
    # But based on the current structure, passing the wn object might be more direct, as long as it's not huge.
    # To be safe, we accept the file path to reload, or assign if it's an object passed.
    # Considering Exa7.inp is relatively small, passing the object or path directly is fine.
    # Here, for generality, we assume what is passed is the WN itself (if Pickle is fine).
    # If Pickle has issues, the path should be passed.
    # Given Exa7 is small, we try passing the object directly. If too slow, change to reading file.
    # Actually, wntr model pickling is okay.
    # Correction: To avoid pickle issues on win32, it's best to read only once in the worker.
    # But wn has already been read in the main function.
    # Let's assume wn can be transferred via pickle, or use initializer.
    
    _global_wn = wn_filepath # Reuse variable name temporarily, actually likely an object
    _global_params = (boundary_pipes, node_to_comm, num_comm, hmin, hdes)

def check_connectivity(boundary_pipes, config, num_comm):
    """Check if each partition has at least 1 open boundary pipe"""
    partition_open = {i: 0 for i in range(num_comm)}
    for i, pipe_info in enumerate(boundary_pipes):
        c1, c2 = pipe_info[3], pipe_info[4]
        if config[i] == 1:
            partition_open[c1] = partition_open.get(c1, 0) + 1
            partition_open[c2] = partition_open.get(c2, 0) + 1
    for i in range(num_comm):
        if partition_open.get(i, 0) < 1:
            return False
    return True

def create_worker_wn(inp_file):
    # Helper function: if loading from file is needed
    return wntr.network.WaterNetworkModel(inp_file)

def evaluate_single_row(x):
    """Evaluation logic for a single individual (runs in Worker)"""
    global _global_wn, _global_params
    boundary_pipes, node_to_comm, num_comm, hmin, hdes = _global_params
    wn = _global_wn
    
    config = (x > 0.5).astype(int)
    
    # Constraint 1: Connectivity
    is_connected = check_connectivity(boundary_pipes, config, num_comm)
    g1 = 0.0 if is_connected else 1.0
    
    if not is_connected:
        return ([0, 0, 0, 0, float(np.sum(config))], [g1, 1.0, 1.0])
        
    # Run simulation
    # Use process ID as file prefix
    pid = os.getpid()
    file_prefix = f"sim_{pid}"
    
    wn_mod = apply_boundary_config(wn, boundary_pipes, config)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_simulation(wn_mod, file_prefix=file_prefix)
    
    # Constraint 2: Hydraulic feasibility
    # Here we need to bring over the logic of check_hydraulic_feasibility or make it available
    # Simply inline the logic to reduce dependencies
    is_feasible = False
    if results is not None:
        try:
            pressure = results.node['pressure']
            if pressure.min().min() >= -10 and pressure.max().max() <= 500:
                 if not (pressure.isnull().any().any() or np.isinf(pressure.values).any()):
                     is_feasible = True
        except:
            pass
            
    g2 = 0.0 if is_feasible else 1.0
    
    if not is_feasible:
        # Clean up temporary files (if any residue) - run_simulation should overwrite if configured correctly
        return ([0, 0, 0, 0, float(np.sum(config))], [g1, g2, 1.0])
    
    # Calculate objectives
    # FEF: Minimize (no negation needed as NSGA-II minimizes by default)
    # MRE: Minimize (no negation, lower failure probability is better)
    # HRE, NR: Maximize (negate, as NSGA-II minimizes by default)
    fef = calculate_fef(wn_mod, results)  # Minimize FEF
    hre = -calculate_hre(wn_mod, results, hmin, hdes)  # Maximize HRE
    mre = calculate_mre(wn_mod, results, node_to_comm, num_comm)  # Minimize MRE
    nr = -calculate_nr(wn_mod, results)  # Maximize NR
    open_count = float(np.sum(config))
    
    # Constraint 3: FEF must be greater than 0 (avoid invalid entropy solutions)
    # Now fef is positive (directly minimized)
    # Pymoo: G <= 0 is satisfied.
    # We want: FEF > 0  =>  -FEF < 0  =>  g3 = -fef < 0 is satisfied
    if fef < 1e-6:
        g3 = 1.0
    else:
        g3 = 0.0
    
    return ([fef, hre, mre, nr, open_count], [g1, g2, g3])


class WDSPartitionProblemParallel(Problem):
    """Parallel WDS partition optimization problem definition"""

    def __init__(self, pool, n_var):
        # 5 objectives, 3 constraints
        super().__init__(n_var=n_var, n_obj=5, n_constr=3, xl=0, xu=1, vtype=bool)
        self.pool = pool

    def _evaluate(self, X, out, *args, **kwargs):
        # Parallel computation
        # X is (pop_size, n_var)
        # Unpack X into a list
        inputs = [x for x in X]
        
        # Use Pool.map
        results = list(self.pool.map(evaluate_single_row, inputs))
        
        F = []
        G = []
        for res_f, res_g in results:
            F.append(res_f)
            G.append(res_g)
            
        out["F"] = np.array(F)
        out["G"] = np.array(G)


def run_nsga2_optimization(wn, boundary_pipes, node_to_comm, num_comm,
                           pop_size=50, n_gen=100, hmin=10, hdes=30, seed=42):
    
    if len(boundary_pipes) == 0:
        results = run_simulation(wn, file_prefix=f'main_{os.getpid()}')
        return {
            'pareto_front': [[0, 0, 0, 0, 0]],
            'pareto_solutions': [[]],
            'best_config': [],
            'best_objectives': {
                'FEF': calculate_fef(wn, results),
                'HRE': calculate_hre(wn, results, hmin, hdes),
                'MRE': calculate_mre(wn, results, node_to_comm, num_comm),
                'NR': calculate_nr(wn, results),
                'open_pipes': 0
            }
        }
    
    # Get available CPU cores
    max_workers = min(os.cpu_count(), pop_size)
    print(f"  Starting pool with {max_workers} workers...")
    
    # Start process pool using context manager
    # Note: Passing wn object on Windows might be slow, but Exa7 network is small so should be fine.
    with ProcessPoolExecutor(max_workers=max_workers, 
                             initializer=init_worker, 
                             initargs=(wn, boundary_pipes, node_to_comm, num_comm, hmin, hdes)) as pool:
        
        problem = WDSPartitionProblemParallel(pool, n_var=len(boundary_pipes))
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=BitflipMutation(prob=1.0/max(len(boundary_pipes), 1)),
            eliminate_duplicates=True
        )
        
        res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                       seed=seed, verbose=False)
    
    # Extract results - Handle case where no feasible solutions were found
    pareto_F = res.F
    pareto_X = res.X
    
    # If no feasible solutions found, return baseline metrics (all pipes open)
    if pareto_F is None or pareto_X is None or len(pareto_F) == 0:
        print("  Warning: NSGA-II found no feasible solutions. Calculating baseline (all pipes open)...")
        # Calculate baseline metrics with all boundary pipes OPEN
        all_open_config = [1] * len(boundary_pipes)
        wn_baseline = apply_boundary_config(wn, boundary_pipes, all_open_config)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                baseline_results = run_simulation(wn_baseline, file_prefix=f'baseline_{os.getpid()}')
            
            if baseline_results is not None:
                baseline_fef = calculate_fef(wn_baseline, baseline_results)
                baseline_hre = calculate_hre(wn_baseline, baseline_results, hmin, hdes)
                baseline_mre = calculate_mre(wn_baseline, baseline_results, node_to_comm, num_comm)
                baseline_nr = calculate_nr(wn_baseline, baseline_results)
            else:
                baseline_fef = baseline_hre = baseline_mre = baseline_nr = 0.0
        except Exception as e:
            print(f"  Warning: Failed to calculate baseline: {e}")
            baseline_fef = baseline_hre = baseline_mre = baseline_nr = 0.0
        
        return {
            'pareto_front': [],
            'pareto_solutions': [],
            'best_config': all_open_config,
            'best_objectives': {
                'FEF': baseline_fef,
                'HRE': baseline_hre,
                'MRE': baseline_mre,
                'NR': baseline_nr,
                'open_pipes': len(boundary_pipes)
            }
        }
    
    pareto_front = []
    for f in pareto_F:
        # FEF: f[0] is positive (minimized), keep as is
        # HRE, NR: f[1], f[3] are negative (maximized), negate to restore original values
        # MRE: f[2] is positive (minimized), keep as is
        pareto_front.append([f[0], -f[1], f[2], -f[3], f[4]])
    
    if len(pareto_F) > 0:
        # Select best: Minimize FEF and MRE, Maximize HRE and NR, Minimize open_count
        # Score = -FEF + HRE - MRE + NR - 0.1 * open_count
        # In pareto_F: FEF is positive, HRE is negative, MRE is positive, NR is negative
        # Score = -pareto_F[:, 0] + (-pareto_F[:, 1]) - pareto_F[:, 2] + (-pareto_F[:, 3]) - 0.1 * pareto_F[:, 4]
        #       = -pareto_F[:, 0] - pareto_F[:, 1] - pareto_F[:, 2] - pareto_F[:, 3] - 0.1 * pareto_F[:, 4]
        scores = -pareto_F[:, 0] - pareto_F[:, 1] - pareto_F[:, 2] - pareto_F[:, 3] - 0.1 * pareto_F[:, 4]
        best_idx = np.argmax(scores)
        best_config = (pareto_X[best_idx] > 0.5).astype(int).tolist()
        best_row = pareto_front[best_idx]
    else:
        best_config = []
        best_row = [0,0,0,0,0]

    return {
        'pareto_front': pareto_front,
        'pareto_solutions': [(x > 0.5).astype(int).tolist() for x in pareto_X],
        'best_config': best_config,
        'best_objectives': {
            'FEF': best_row[0],
            'HRE': best_row[1],
            'MRE': best_row[2],
            'NR': best_row[3],
            'open_pipes': int(best_row[4])
        }
    }

