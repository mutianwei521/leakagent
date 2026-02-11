"""
NSGA-II优化模块 - 使用pymoo库进行多目标优化 (并行版)
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

# 全局变量用于Worker初始化，避免重复序列化大对象
_global_wn = None
_global_params = None

def init_worker(wn_filepath, boundary_pipes, node_to_comm, num_comm, hmin, hdes):
    """Worker进程初始化函数"""
    global _global_wn, _global_params
    # 在每个进程中重新加载WNTR模型，避免Pickle大对象
    # 或者如果传递wn对象太慢，这里重新加载可能是个选择。
    # 但根据当前结构，传入wn对象可能更直接，只要它不是特别巨大。
    # 为了安全起见，我们接收文件路径重新加载，或者如果是传递对象，则赋值。
    # 考虑到Exa7.inp比较小，直接传递对象或者路径都可以。
    # 这里为了通用性，我们假设传入的是WN本身（如果Pickle没问题）。
    # 如果Pickle有问题，应该传路径。
    # 鉴于Exa7不大，我们尝试直接传对象。如果太慢再改为读文件。
    # 实际上，wntr模型pickle还可以。
    # 修正：为了避免win32下的pickle问题，最好是在worker里只读一次。
    # 但是main函数里已经读了wn。
    # 让我们假设wn可以通过pickle传输，或者使用initializer。
    
    _global_wn = wn_filepath # 这里暂时复用变量名，实际可能是对象
    _global_params = (boundary_pipes, node_to_comm, num_comm, hmin, hdes)

def check_connectivity(boundary_pipes, config, num_comm):
    """检查每个分区是否至少有1条开放的边界管道"""
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
    # 辅助函数：如果需要从文件加载
    return wntr.network.WaterNetworkModel(inp_file)

def evaluate_single_row(x):
    """单个个体的评估逻辑（在Worker中运行）"""
    global _global_wn, _global_params
    boundary_pipes, node_to_comm, num_comm, hmin, hdes = _global_params
    wn = _global_wn
    
    config = (x > 0.5).astype(int)
    
    # 约束1：连通性
    is_connected = check_connectivity(boundary_pipes, config, num_comm)
    g1 = 0.0 if is_connected else 1.0
    
    if not is_connected:
        return ([0, 0, 0, 0, float(np.sum(config))], [g1, 1.0, 1.0])
        
    # 运行仿真
    # 使用进程ID作为文件前缀
    pid = os.getpid()
    file_prefix = f"sim_{pid}"
    
    wn_mod = apply_boundary_config(wn, boundary_pipes, config)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_simulation(wn_mod, file_prefix=file_prefix)
    
    # 约束2：水力可行性
    # 这里我们需要把 check_hydraulic_feasibility 的逻辑搬过来或者可用
    # 简单内联一下逻辑以减少依赖
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
        # 清理临时文件（如果有残留）- run_simulation 如果正确配置应该覆盖
        return ([0, 0, 0, 0, float(np.sum(config))], [g1, g2, 1.0])
    
    # 计算目标
    # FEF: 最小化 (不用取反，因为NSGA-II默认最小化)
    # MRE: 最小化 (不用取反，较低的失效概率更好)
    # HRE, NR: 最大化 (取反，因为NSGA-II默认最小化)
    fef = calculate_fef(wn_mod, results)  # 最小化FEF
    hre = -calculate_hre(wn_mod, results, hmin, hdes)  # 最大化HRE
    mre = calculate_mre(wn_mod, results, node_to_comm, num_comm)  # 最小化MRE
    nr = -calculate_nr(wn_mod, results)  # 最大化NR
    open_count = float(np.sum(config))
    
    # 约束3：FEF必须大于0 (避免无效熵解)
    # 现在fef是正值 (直接最小化)
    # Pymoo: G <= 0 is satisfied.
    # We want: FEF > 0  =>  -FEF < 0  =>  g3 = -fef < 0 is satisfied
    if fef < 1e-6:
        g3 = 1.0
    else:
        g3 = 0.0
    
    return ([fef, hre, mre, nr, open_count], [g1, g2, g3])


class WDSPartitionProblemParallel(Problem):
    """并行的水网分区优化问题定义"""

    def __init__(self, pool, n_var):
        # 5个目标，3个约束
        super().__init__(n_var=n_var, n_obj=5, n_constr=3, xl=0, xu=1, vtype=bool)
        self.pool = pool

    def _evaluate(self, X, out, *args, **kwargs):
        # 并行计算
        # X是 (pop_size, n_var)
        # 将X拆解为列表
        inputs = [x for x in X]
        
        # 使用Pool.map
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
    
    # 获取可用CPU核数
    max_workers = min(os.cpu_count(), pop_size)
    print(f"  Starting pool with {max_workers} workers...")
    
    # 使用上下文管理器启动进程池
    # 注意：Windows下传递wn对象可能很慢，但Exa7网络小应该还好。
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
    
    # 提取结果 - Handle case where no feasible solutions were found
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
        # FEF: f[0] 是正值 (最小化), 保持原样
        # HRE, NR: f[1], f[3] 是负值 (最大化), 取反恢复原值
        # MRE: f[2] 是正值 (最小化), 保持原样
        pareto_front.append([f[0], -f[1], f[2], -f[3], f[4]])
    
    if len(pareto_F) > 0:
        # 选择最佳: 最小化FEF和MRE, 最大化HRE和NR, 最小化open_count
        # 得分 = -FEF + HRE - MRE + NR - 0.1 * open_count
        # pareto_F中: FEF是正值, HRE是负值, MRE是正值, NR是负值
        # 得分 = -pareto_F[:, 0] + (-pareto_F[:, 1]) - pareto_F[:, 2] + (-pareto_F[:, 3]) - 0.1 * pareto_F[:, 4]
        #      = -pareto_F[:, 0] - pareto_F[:, 1] - pareto_F[:, 2] - pareto_F[:, 3] - 0.1 * pareto_F[:, 4]
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

