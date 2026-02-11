# -*- coding: utf-8 -*-
"""
压力灵敏度分析模块
实现节点需水量调整、压力差计算、灵敏度矩阵生成功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from .epanet_handler import EPANETHandler


class SensitivityAnalyzer:
    """压力灵敏度分析器"""
    
    def __init__(self, epanet_handler: EPANETHandler):
        """
        初始化灵敏度分析器
        
        Args:
            epanet_handler: EPANET处理器实例
        """
        self.epanet_handler = epanet_handler
        self.normal_pressure = None
        self.normal_flow = None  # 添加正常流量数据存储
        self.sensitivity_matrix = None
        self.flow_sensitivity_matrix = None
        self.node_weights = None
        
    def calculate_normal_pressure(self, duration_hours: int = 24, 
                                time_step_hours: int = 1) -> bool:
        """
        计算正常情况下的压力数据
        
        Args:
            duration_hours: 模拟持续时间（小时）
            time_step_hours: 时间步长（小时）
            
        Returns:
            bool: 计算是否成功
        """
        try:
            # 重置网络到原始状态
            self.epanet_handler.reset_network()
            
            # 运行水力计算
            if not self.epanet_handler.run_hydraulic_simulation(duration_hours, time_step_hours):
                return False
            
            # 获取正常压力数据
            self.normal_pressure = self.epanet_handler.get_pressure_data()

            if self.normal_pressure.empty:
                logger.error("获取正常压力数据失败")
                return False

            # 获取正常流量数据
            self.normal_flow = self.epanet_handler.get_flow_data()

            logger.info(f"正常压力数据计算完成: {self.normal_pressure.shape}")
            if not self.normal_flow.empty:
                logger.info(f"正常流量数据计算完成: {self.normal_flow.shape}")
            return True
            
        except Exception as e:
            logger.error(f"计算正常压力失败: {e}")
            return False
    
    def calculate_sensitivity_matrix(self, demand_ratio: float = 0.2,
                                   duration_hours: int = 24,
                                   time_step_hours: int = 1) -> bool:
        """
        计算压力灵敏度矩阵
        
        Args:
            demand_ratio: 异常需水量比例
            duration_hours: 模拟持续时间（小时）
            time_step_hours: 时间步长（小时）
            
        Returns:
            bool: 计算是否成功
        """
        try:
            if self.normal_pressure is None:
                logger.error("请先计算正常压力数据")
                return False
            
            node_names = self.epanet_handler.node_names
            n_nodes = len(node_names)
            n_times = len(self.normal_pressure)

            logger.info(f"正常压力数据形状: {self.normal_pressure.shape}")
            logger.info(f"节点数: {n_nodes}, 时间步数: {n_times}")

            # 初始化灵敏度矩阵 [n_nodes, n_nodes, n_times]
            # sensitivity_matrix[i, j, t] 表示节点i异常时，节点j在时间t的压力灵敏度
            self.sensitivity_matrix = np.zeros((n_nodes, n_nodes, n_times))
            self.flow_sensitivity_matrix = np.zeros((n_nodes, len(self.epanet_handler.pipe_names), n_times))

            logger.info(f"开始计算灵敏度矩阵: {n_nodes}个节点, 灵敏度矩阵形状: {self.sensitivity_matrix.shape}")
            
            # 遍历每个节点，模拟其异常情况
            for i, anomaly_node in enumerate(tqdm(node_names, desc="计算节点灵敏度")):
                # 重置网络
                self.epanet_handler.reset_network()
                
                # 修改当前节点的需水量
                if not self.epanet_handler.modify_node_demand(anomaly_node, demand_ratio):
                    logger.warning(f"修改节点 {anomaly_node} 需水量失败，跳过")
                    continue
                
                # 运行水力计算
                if not self.epanet_handler.run_hydraulic_simulation(duration_hours, time_step_hours):
                    logger.warning(f"节点 {anomaly_node} 异常情况水力计算失败，跳过")
                    continue
                
                # 获取异常压力数据
                anomaly_pressure = self.epanet_handler.get_pressure_data()
                anomaly_flow = self.epanet_handler.get_flow_data()
                
                if anomaly_pressure.empty:
                    logger.warning(f"节点 {anomaly_node} 异常压力数据为空，跳过")
                    continue
                
                # 检查压力数据形状一致性
                if anomaly_pressure.shape != self.normal_pressure.shape:
                    logger.warning(f"节点 {anomaly_node} 异常压力数据形状不匹配: "
                                 f"正常{self.normal_pressure.shape} vs 异常{anomaly_pressure.shape}，跳过")
                    continue

                # 计算压力差的绝对值（灵敏度）
                pressure_diff = np.abs(self.normal_pressure - anomaly_pressure)

                # 存储灵敏度数据
                for j, sensor_node in enumerate(node_names):
                    if sensor_node in pressure_diff.columns:
                        pressure_values = pressure_diff[sensor_node].values
                        # 确保数据长度匹配
                        if len(pressure_values) == self.sensitivity_matrix.shape[2]:
                            self.sensitivity_matrix[i, j, :] = pressure_values
                        else:
                            logger.warning(f"节点 {sensor_node} 压力数据长度不匹配: "
                                         f"期望{self.sensitivity_matrix.shape[2]}, 实际{len(pressure_values)}")
                            # 使用零填充或截断
                            min_len = min(len(pressure_values), self.sensitivity_matrix.shape[2])
                            self.sensitivity_matrix[i, j, :min_len] = pressure_values[:min_len]
                
                # 计算流量差
                if not anomaly_flow.empty and self.normal_flow is not None:
                    if not self.normal_flow.empty and anomaly_flow.shape == self.normal_flow.shape:
                        flow_diff = self.normal_flow - anomaly_flow
                        for k, pipe_name in enumerate(self.epanet_handler.pipe_names):
                            if pipe_name in flow_diff.columns:
                                flow_values = flow_diff[pipe_name].values
                                # 确保数据长度匹配
                                if len(flow_values) == self.flow_sensitivity_matrix.shape[2]:
                                    self.flow_sensitivity_matrix[i, k, :] = flow_values
                                else:
                                    logger.warning(f"管道 {pipe_name} 流量数据长度不匹配: "
                                                 f"期望{self.flow_sensitivity_matrix.shape[2]}, 实际{len(flow_values)}")
                                    # 使用零填充或截断
                                    min_len = min(len(flow_values), self.flow_sensitivity_matrix.shape[2])
                                    self.flow_sensitivity_matrix[i, k, :min_len] = flow_values[:min_len]
            
            logger.info("灵敏度矩阵计算完成")
            return True
            
        except Exception as e:
            logger.error(f"计算灵敏度矩阵失败: {e}")
            return False
    
    def normalize_sensitivity_matrix(self) -> bool:
        """
        归一化灵敏度矩阵（标准化 + 归一化，处理分母为0的情况）

        Returns:
            bool: 归一化是否成功
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("灵敏度矩阵未计算")
                return False

            n_nodes, _, n_times = self.sensitivity_matrix.shape
            epsilon = 1e-6  # 防止分母为0

            # 先进行标准化（Z-score normalization）
            for i in range(n_nodes):
                for j in range(n_nodes):
                    data = self.sensitivity_matrix[i, j, :]

                    # 计算均值和标准差
                    mean_val = np.mean(data)
                    std_val = np.std(data)

                    # 标准化：处理标准差为0的情况
                    if std_val < epsilon:
                        # 如果标准差为0，设置为均值（如果均值也为0，则保持为0）
                        self.sensitivity_matrix[i, j, :] = 0.0 if abs(mean_val) < epsilon else 1.0
                    else:
                        self.sensitivity_matrix[i, j, :] = (data - mean_val) / std_val

            # 再进行Min-Max归一化到[0,1]区间
            for i in range(n_nodes):
                for j in range(n_nodes):
                    data = self.sensitivity_matrix[i, j, :]

                    min_val = np.min(data)
                    max_val = np.max(data)

                    # 归一化：处理最大值等于最小值的情况
                    if abs(max_val - min_val) < epsilon:
                        # 如果范围为0，设置为0.5（中间值）
                        self.sensitivity_matrix[i, j, :] = 0.5
                    else:
                        self.sensitivity_matrix[i, j, :] = (data - min_val) / (max_val - min_val + epsilon)

            logger.info("灵敏度矩阵标准化和归一化完成")
            return True

        except Exception as e:
            logger.error(f"归一化灵敏度矩阵失败: {e}")
            return False
    
    def calculate_node_weights(self) -> np.ndarray:
        """
        计算节点权重向量（压力灵敏度时序平均值）
        
        Returns:
            np.ndarray: 节点权重向量 [n_nodes, n_nodes]
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("灵敏度矩阵未计算")
                return np.array([])
            
            # 计算时序平均值作为节点权重
            self.node_weights = np.mean(self.sensitivity_matrix, axis=2)
            
            logger.info(f"节点权重计算完成: {self.node_weights.shape}")
            return self.node_weights
            
        except Exception as e:
            logger.error(f"计算节点权重失败: {e}")
            return np.array([])
    
    def get_sensitivity_features(self, anomaly_node_idx: int) -> Dict:
        """
        获取指定异常节点的灵敏度特征
        
        Args:
            anomaly_node_idx: 异常节点索引
            
        Returns:
            Dict: 灵敏度特征字典
        """
        try:
            if self.sensitivity_matrix is None:
                logger.error("灵敏度矩阵未计算")
                return {}
            
            features = {
                'pressure_sensitivity': self.sensitivity_matrix[anomaly_node_idx, :, :],
                'avg_pressure_sensitivity': np.mean(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                'max_pressure_sensitivity': np.max(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                'std_pressure_sensitivity': np.std(self.sensitivity_matrix[anomaly_node_idx, :, :], axis=1)
            }
            
            if self.flow_sensitivity_matrix is not None:
                features.update({
                    'flow_sensitivity': self.flow_sensitivity_matrix[anomaly_node_idx, :, :],
                    'avg_flow_sensitivity': np.mean(self.flow_sensitivity_matrix[anomaly_node_idx, :, :], axis=1),
                    'max_flow_sensitivity': np.max(self.flow_sensitivity_matrix[anomaly_node_idx, :, :], axis=1)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"获取灵敏度特征失败: {e}")
            return {}
    
    def save_sensitivity_data(self, output_dir: str) -> bool:
        """
        保存灵敏度数据
        
        Args:
            output_dir: 输出目录
            
        Returns:
            bool: 保存是否成功
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if self.sensitivity_matrix is not None:
                np.save(os.path.join(output_dir, 'sensitivity_matrix.npy'), self.sensitivity_matrix)
                logger.info(f"灵敏度矩阵已保存到 {output_dir}/sensitivity_matrix.npy")
            
            if self.node_weights is not None:
                np.save(os.path.join(output_dir, 'node_weights.npy'), self.node_weights)
                logger.info(f"节点权重已保存到 {output_dir}/node_weights.npy")
            
            if self.normal_pressure is not None:
                self.normal_pressure.to_csv(os.path.join(output_dir, 'normal_pressure.csv'))
                logger.info(f"正常压力数据已保存到 {output_dir}/normal_pressure.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"保存灵敏度数据失败: {e}")
            return False
    
    def load_sensitivity_data(self, output_dir: str) -> bool:
        """
        加载灵敏度数据
        
        Args:
            output_dir: 数据目录
            
        Returns:
            bool: 加载是否成功
        """
        try:
            import os
            
            sensitivity_file = os.path.join(output_dir, 'sensitivity_matrix.npy')
            if os.path.exists(sensitivity_file):
                self.sensitivity_matrix = np.load(sensitivity_file)
                logger.info("灵敏度矩阵加载成功")
            
            weights_file = os.path.join(output_dir, 'node_weights.npy')
            if os.path.exists(weights_file):
                self.node_weights = np.load(weights_file)
                logger.info("节点权重加载成功")
            
            pressure_file = os.path.join(output_dir, 'normal_pressure.csv')
            if os.path.exists(pressure_file):
                self.normal_pressure = pd.read_csv(pressure_file, index_col=0, encoding='utf-8')
                logger.info("正常压力数据加载成功")
            
            return True
            
        except Exception as e:
            logger.error(f"加载灵敏度数据失败: {e}")
            return False
