"""
Solution evaluation module - Evaluate and select the optimal solution
"""
import numpy as np


def evaluate_solution(objectives):
    """
    Evaluate the comprehensive score of a single solution
    Args:
        objectives: dict, containing FEF, HRE, MRE, NR, open_pipes
    Returns:
        score: Comprehensive score (higher is better)
    """
    # Weighted sum (weights can be adjusted)
    weights = {'FEF': 0.2, 'HRE': 0.25, 'MRE': 0.25, 'NR': 0.2, 'open_pipes': -0.1}
    score = 0.0
    for key, w in weights.items():
        val = objectives.get(key, 0)
        if key == 'open_pipes':  # Fewer open pipes is better
            score += w * val
        else:
            score += w * val
    return score


def select_best_solution(all_results):
    """
    Select the global optimal solution from the optimization results of all partitions
    Args:
        all_results: dict, {num_comm: optimization_result, ...}
    Returns:
        best_result: Best solution information
    """
    best_score = float('-inf')
    best_result = None
    
    for num_comm, result in all_results.items():
        if result is None:
            continue
        # Evaluate the best solution for this partition
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
    Normalize Pareto front for visualization
    Args:
        pareto_front: list of [FEF, HRE, MRE, NR, open_pipes]
    Returns:
        normalized: Normalized Pareto front
    """
    if not pareto_front:
        return []
    
    arr = np.array(pareto_front)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (arr - mins) / ranges
    return normalized.tolist()


def rank_solutions(pareto_front, weights=None):
    """
    Rank solutions in the Pareto front
    Args:
        pareto_front: Pareto front solutions
        weights: Objective weights
    Returns:
        ranked_indices: Ranked indices
    """
    if weights is None:
        weights = [0.2, 0.25, 0.25, 0.2, -0.1]  # FEF, HRE, MRE, NR, open_pipes
    
    scores = []
    for sol in pareto_front:
        score = sum(w * v for w, v in zip(weights, sol))
        scores.append(score)
    
    return np.argsort(scores)[::-1].tolist()  # Sort in descending order

