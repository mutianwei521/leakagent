# -*- coding: utf-8 -*-
"""
EPANET管网模型处理模块
处理WNTR库的管网导入、水力计算、压力数据获取功能
"""

import numpy as np
import pandas as pd
import wntr
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class EPANETHandler:
    """EPANET管网模型处理器"""
    
    def __init__(self, inp_file: str):
        """
        初始化EPANET处理器
        
        Args:
            inp_file: EPANET输入文件路径
        """
        self.inp_file = inp_file
        self.wn = None
        self.results = None
        self.node_names = []
        self.pipe_names = []
        self.adjacency_matrix = None
        
    def load_network(self) -> bool:
        """
        加载EPANET网络模型
        
        Returns:
            bool: 加载是否成功
        """
        try:
            self.wn = wntr.network.WaterNetworkModel(self.inp_file)
            self.node_names = list(self.wn.junction_name_list)
            self.pipe_names = list(self.wn.pipe_name_list)
            
            # 构建邻接矩阵
            self._build_adjacency_matrix()
            
            logger.info(f"成功加载网络: {len(self.node_names)}个节点, {len(self.pipe_names)}条管道")
            return True
            
        except Exception as e:
            logger.error(f"加载网络失败: {e}")
            return False
    
    def _build_adjacency_matrix(self):
        """构建网络邻接矩阵"""
        n_nodes = len(self.node_names)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        node_to_idx = {name: idx for idx, name in enumerate(self.node_names)}
        
        for pipe_name in self.pipe_names:
            pipe = self.wn.get_link(pipe_name)
            start_node = pipe.start_node_name
            end_node = pipe.end_node_name
            
            if start_node in node_to_idx and end_node in node_to_idx:
                i = node_to_idx[start_node]
                j = node_to_idx[end_node]
                self.adjacency_matrix[i, j] = 1
                self.adjacency_matrix[j, i] = 1  # 无向图
    
    def run_hydraulic_simulation(self, duration_hours: int = 24, 
                                time_step_hours: int = 1) -> bool:
        """
        运行水力计算
        
        Args:
            duration_hours: 模拟持续时间（小时）
            time_step_hours: 时间步长（小时）
            
        Returns:
            bool: 计算是否成功
        """
        try:
            # 设置模拟参数
            self.wn.options.time.duration = duration_hours * 3600  # 转换为秒
            self.wn.options.time.hydraulic_timestep = time_step_hours * 3600
            self.wn.options.time.report_timestep = time_step_hours * 3600
            
            # 运行模拟
            sim = wntr.sim.EpanetSimulator(self.wn)
            self.results = sim.run_sim()
            
            logger.info(f"水力计算完成: {duration_hours}小时模拟")
            return True
            
        except Exception as e:
            logger.error(f"水力计算失败: {e}")
            return False
    
    def get_pressure_data(self) -> pd.DataFrame:
        """
        获取节点压力时序数据
        
        Returns:
            pd.DataFrame: 压力数据，行为时间，列为节点
        """
        if self.results is None:
            logger.error("未找到计算结果，请先运行水力计算")
            return pd.DataFrame()
        
        try:
            pressure_data = self.results.node['pressure']
            # 只保留junction节点的压力数据
            junction_pressure = pressure_data[self.node_names]
            return junction_pressure
            
        except Exception as e:
            logger.error(f"获取压力数据失败: {e}")
            return pd.DataFrame()
    
    def get_flow_data(self) -> pd.DataFrame:
        """
        获取管道流量时序数据
        
        Returns:
            pd.DataFrame: 流量数据，行为时间，列为管道
        """
        if self.results is None:
            logger.error("未找到计算结果，请先运行水力计算")
            return pd.DataFrame()
        
        try:
            flow_data = self.results.link['flowrate']
            pipe_flow = flow_data[self.pipe_names]
            return pipe_flow
            
        except Exception as e:
            logger.error(f"获取流量数据失败: {e}")
            return pd.DataFrame()
    
    def get_total_network_demand(self) -> float:
        """
        计算整个管网的总需水量

        Returns:
            float: 总需水量
        """
        try:
            total_demand = 0.0
            for node_name in self.node_names:
                junction = self.wn.get_node(node_name)
                if junction.demand_timeseries_list:
                    total_demand += abs(junction.demand_timeseries_list[0].base_value)

            logger.debug(f"管网总需水量: {total_demand:.3f}")
            return total_demand

        except Exception as e:
            logger.error(f"计算总需水量失败: {e}")
            return 0.0

    def modify_node_demand(self, node_name: str, demand_ratio: float = 0.2) -> bool:
        """
        修改节点需水量

        Args:
            node_name: 节点名称
            demand_ratio: 需水量比例（相对于原始值）

        Returns:
            bool: 修改是否成功
        """
        try:
            if node_name not in self.node_names:
                logger.error(f"节点 {node_name} 不存在")
                return False

            junction = self.wn.get_node(node_name)
            original_demand = junction.demand_timeseries_list[0].base_value

            # 如果原始需水量为0，则使用整个管网总水量的3%
            if abs(original_demand) < 1e-6:  # 考虑浮点数精度，使用小阈值判断是否为0
                total_demand = self.get_total_network_demand()
                new_demand = total_demand * 0.03  # 3%
                logger.info(f"节点 {node_name} 原始需水量为0，使用管网总水量的3%: {new_demand:.3f}")
            else:
                new_demand = original_demand * demand_ratio

            # 修改需水量
            junction.demand_timeseries_list[0].base_value = new_demand

            logger.debug(f"节点 {node_name} 需水量从 {original_demand:.3f} 修改为 {new_demand:.3f}")
            return True

        except Exception as e:
            logger.error(f"修改节点需水量失败: {e}")
            return False
    
    def reset_network(self):
        """重置网络到原始状态"""
        try:
            self.wn = wntr.network.WaterNetworkModel(self.inp_file)
            logger.debug("网络已重置到原始状态")
        except Exception as e:
            logger.error(f"重置网络失败: {e}")
    
    def get_pipe_lengths(self) -> Dict[str, float]:
        """
        获取管道长度信息
        
        Returns:
            Dict[str, float]: 管道名称到长度的映射
        """
        pipe_lengths = {}
        try:
            for pipe_name in self.pipe_names:
                pipe = self.wn.get_link(pipe_name)
                pipe_lengths[pipe_name] = pipe.length
            return pipe_lengths
            
        except Exception as e:
            logger.error(f"获取管道长度失败: {e}")
            return {}
    
    def get_network_graph(self) -> nx.Graph:
        """
        获取网络图结构
        
        Returns:
            nx.Graph: NetworkX图对象
        """
        try:
            G = nx.Graph()
            
            # 添加节点
            for node_name in self.node_names:
                G.add_node(node_name)
            
            # 添加边（管道）
            for pipe_name in self.pipe_names:
                pipe = self.wn.get_link(pipe_name)
                start_node = pipe.start_node_name
                end_node = pipe.end_node_name
                
                if start_node in self.node_names and end_node in self.node_names:
                    G.add_edge(start_node, end_node, 
                             pipe_name=pipe_name, 
                             length=pipe.length)
            
            return G
            
        except Exception as e:
            logger.error(f"构建网络图失败: {e}")
            return nx.Graph()
    
    def get_network_info(self) -> Dict:
        """
        获取网络基本信息
        
        Returns:
            Dict: 网络信息字典
        """
        return {
            'n_nodes': len(self.node_names),
            'n_pipes': len(self.pipe_names),
            'node_names': self.node_names.copy(),
            'pipe_names': self.pipe_names.copy(),
            'adjacency_matrix': self.adjacency_matrix.copy() if self.adjacency_matrix is not None else None
        }
