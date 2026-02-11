"""
解决方案评估模块 - 评估和选择最优解
"""
import numpy as np


def evaluate_solution(objectives):
    """
    评估单个解决方案的综合得分
    Args:
        objectives: dict，包含FEF, HRE, MRE, NR, open_pipes
    Returns:
        score: 综合评分（越高越好）
    """
    # 加权求和 (权重可调整)
    weights = {'FEF': 0.2, 'HRE': 0.25, 'MRE': 0.25, 'NR': 0.2, 'open_pipes': -0.1}
    score = 0.0
    for key, w in weights.items():
        val = objectives.get(key, 0)
        if key == 'open_pipes':  # 开放管道数越少越好
            score += w * val
        else:
            score += w * val
    return score


def select_best_solution(all_results):
    """
    从所有分区的优化结果中选择全局最优解
    Args:
        all_results: dict, {num_comm: optimization_result, ...}
    Returns:
        best_result: 最优解信息
    """
    best_score = float('-inf')
    best_result = None
    
    for num_comm, result in all_results.items():
        if result is None:
            continue
        # 评估该分区的最佳解
        obj = result.get('best_objectives', {})
        score = evaluate_solution(obj)
        
        if score > best_score:
            best_score = score
            best_result = {
                'num_communities': num_comm,
                'score': score,
                'objectives': obj,
                'best_config': result.get('best_config', []),
                'pareto_front': result.get('pareto_front', [])
            }
    
    return best_result


def normalize_pareto_front(pareto_front):
    """
    归一化Pareto前沿用于可视化
    Args:
        pareto_front: list of [FEF, HRE, MRE, NR, open_pipes]
    Returns:
        normalized: 归一化后的Pareto前沿
    """
    if not pareto_front:
        return []
    
    arr = np.array(pareto_front)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # 避免除零
    normalized = (arr - mins) / ranges
    return normalized.tolist()


def rank_solutions(pareto_front, weights=None):
    """
    对Pareto前沿中的解进行排序
    Args:
        pareto_front: Pareto前沿解
        weights: 目标权重
    Returns:
        ranked_indices: 排序后的索引
    """
    if weights is None:
        weights = [0.2, 0.25, 0.25, 0.2, -0.1]  # FEF, HRE, MRE, NR, open_pipes
    
    scores = []
    for sol in pareto_front:
        score = sum(w * v for w, v in zip(weights, sol))
        scores.append(score)
    
    return np.argsort(scores)[::-1].tolist()  # 降序排序

