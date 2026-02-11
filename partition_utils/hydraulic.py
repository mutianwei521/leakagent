"""
供水管网的水力模拟工具函数。
"""
import numpy as np
import wntr


def run_hydraulic_simulation(wn):
    """
    运行水力模拟并返回结果。
    
    参数:
        wn: WNTR 供水管网模型
        
    返回:
        sim_results: 模拟结果对象
    """
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return results


def calculate_average_pressure(results, wn):
    """
    计算每个节点在所有时间步长内的平均压力。
    
    参数:
        results: WNTR 模拟结果
        wn: WNTR 供水管网模型
        
    返回:
        avg_pressure: 节点名称到平均压力值的字典映射
    """
    # 获取所有节点在所有时间步长内的压力数据
    pressure_df = results.node['pressure']
    
    # 计算每个节点在所有时间步长内的平均压力
    avg_pressure = pressure_df.mean(axis=0)
    
    # 转换为字典以方便访问
    avg_pressure_dict = avg_pressure.to_dict()
    
    return avg_pressure_dict

