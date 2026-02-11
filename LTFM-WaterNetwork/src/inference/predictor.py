# -*- coding: utf-8 -*-
"""
推理模块
实现基于保存分区信息的实时异常检测推理流程
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from loguru import logger
import pandas as pd
import os

from ..models.ltfm_model import LTFMModel
from ..models.graph2vec_encoder import Graph2VecEncoder
from ..data.epanet_handler import EPANETHandler


class LTFMPredictor:
    """LTFM异常检测预测器"""

    def __init__(self, config: Dict):
        """
        初始化预测器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')

        # 模型组件
        self.graph2vec_encoder = None
        self.ltfm_model = None

        # 分区信息（从训练时保存的文件加载）
        self.partition_labels = None
        self.partition_node_names = None
        self.n_clusters = None

        # 网络信息
        self.epanet_handler = None
        self.network_graph = None
        self.normal_pressure = None
        
    def load_models(self, graph2vec_path: str, ltfm_checkpoint_path: str, partition_info_path: str) -> bool:
        """
        加载训练好的模型和分区信息

        Args:
            graph2vec_path: Graph2Vec模型路径
            ltfm_checkpoint_path: LTFM检查点路径
            partition_info_path: 分区信息文件路径（fcm_partition.csv）

        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载分区信息
            if not self.load_partition_info(partition_info_path):
                logger.error("分区信息加载失败")
                return False

            # 加载Graph2Vec编码器
            self.graph2vec_encoder = Graph2VecEncoder(
                embedding_dim=int(self.config['graph2vec']['dimensions']),
                wl_iterations=3,
                workers=int(self.config['graph2vec']['workers']),
                epochs=int(self.config['graph2vec']['epochs']),
                min_count=int(self.config['graph2vec']['min_count']),
                learning_rate=float(self.config['graph2vec']['learning_rate'])
            ).to(self.device)

            if not self.graph2vec_encoder.load_model(graph2vec_path):
                logger.error("Graph2Vec模型加载失败")
                return False

            # 加载LTFM模型（使用从分区信息中获取的分区数）
            self.ltfm_model = LTFMModel(
                graph2vec_dim=int(self.config['graph2vec']['dimensions']),
                embed_dim=int(self.config['ltfm']['embedding_dim']),
                num_heads=int(self.config['ltfm']['num_heads']),
                num_layers=int(self.config['ltfm']['num_layers']),
                num_regions=self.n_clusters,  # 使用加载的分区数
                hidden_dim=int(self.config['ltfm']['hidden_dim']),
                dropout=float(self.config['ltfm']['dropout'])
            ).to(self.device)

            # 加载检查点
            if os.path.exists(ltfm_checkpoint_path):
                checkpoint = torch.load(ltfm_checkpoint_path, map_location=self.device)
                self.ltfm_model.load_state_dict(checkpoint['ltfm_model_state_dict'])
                logger.info("LTFM模型加载成功")
            else:
                logger.error(f"LTFM检查点文件不存在: {ltfm_checkpoint_path}")
                return False

            # 设置为评估模式
            self.ltfm_model.eval()

            logger.info(f"模型加载完成（分区数: {self.n_clusters}）")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def load_partition_info(self, partition_info_path: str) -> bool:
        """
        加载训练时保存的分区信息

        Args:
            partition_info_path: 分区信息文件路径（fcm_partition.csv）

        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(partition_info_path):
                logger.error(f"分区信息文件不存在: {partition_info_path}")
                return False

            # 读取分区信息
            partition_df = pd.read_csv(partition_info_path)
            self.partition_node_names = partition_df['node_name'].tolist()
            self.partition_labels = partition_df['partition_id'].tolist()
            self.n_clusters = len(set(self.partition_labels))

            logger.info(f"✓ 分区信息加载成功: {len(self.partition_node_names)} 个节点, {self.n_clusters} 个分区")

            # 打印分区统计
            partition_counts = {}
            for label in self.partition_labels:
                partition_counts[label] = partition_counts.get(label, 0) + 1

            for p, count in sorted(partition_counts.items()):
                logger.info(f"  分区 {p}: {count} 个节点")

            return True

        except Exception as e:
            logger.error(f"加载分区信息失败: {e}")
            return False

    def get_partitions_from_saved_info(self) -> Tuple[List[List[str]], List[nx.Graph]]:
        """
        根据保存的分区信息构建分区和子图

        Returns:
            Tuple[List[List[str]], List[nx.Graph]]: (分区节点列表, 分区子图列表)
        """
        try:
            if self.partition_labels is None or self.partition_node_names is None:
                logger.error("分区信息未加载")
                return [], []

            # 组织分区结果
            partitions = [[] for _ in range(self.n_clusters)]
            for node_name, label in zip(self.partition_node_names, self.partition_labels):
                partitions[label].append(node_name)

            # 创建子图
            subgraphs = []
            for partition in partitions:
                if partition:  # 确保分区不为空
                    subgraph = self.network_graph.subgraph(partition).copy()
                    subgraphs.append(subgraph)
                else:
                    # 创建空图
                    subgraphs.append(nx.Graph())

            logger.debug(f"使用保存的分区信息: {len(partitions)} 个分区")
            return partitions, subgraphs

        except Exception as e:
            logger.error(f"构建分区失败: {e}")
            return [], []
    
    def initialize_network(self, epanet_file: str) -> bool:
        """
        初始化网络
        
        Args:
            epanet_file: EPANET文件路径
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 初始化EPANET处理器
            self.epanet_handler = EPANETHandler(epanet_file)
            
            if not self.epanet_handler.load_network():
                return False
            
            # 获取网络图
            self.network_graph = self.epanet_handler.get_network_graph()
            
            # 计算正常压力数据作为基准
            if not self.epanet_handler.run_hydraulic_simulation(
                self.config['hydraulic']['simulation_hours'],
                self.config['hydraulic']['time_step']
            ):
                return False
            
            self.normal_pressure = self.epanet_handler.get_pressure_data()
            
            logger.info("网络初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"网络初始化失败: {e}")
            return False

    def predict_anomaly(self, current_pressure: pd.DataFrame) -> Dict:
        """
        预测异常

        Args:
            current_pressure: 当前压力数据 [时间, 节点]

        Returns:
            Dict: 预测结果字典
        """
        try:
            if self.ltfm_model is None or self.graph2vec_encoder is None:
                logger.error("模型未加载")
                return {}

            if self.normal_pressure is None:
                logger.error("正常压力基准未设置")
                return {}

            # 计算压力差（当前压力 - 正常压力）
            pressure_diff = self.normal_pressure - current_pressure

            # 计算节点权重（压力差的平均值）
            node_weights = {}
            for node in self.epanet_handler.node_names:
                if node in pressure_diff.columns:
                    node_weights[node] = pressure_diff[node].mean()
                else:
                    node_weights[node] = 0.0

            # 使用训练时保存的分区信息（不重新分区）
            partitions, subgraphs = self.get_partitions_from_saved_info()

            # 构建区域权重
            region_weights = []
            for partition in partitions:
                region_weight = {node: node_weights.get(node, 0.0) for node in partition}
                region_weights.append(region_weight)

            # 编码图
            with torch.no_grad():
                # 全局图嵌入
                global_embedding = self.graph2vec_encoder.encode_graph(
                    self.network_graph, node_weights
                ).to(self.device).unsqueeze(0)  # 添加batch维度

                # 区域图嵌入
                region_embeddings = []
                for subgraph, region_weight in zip(subgraphs, region_weights):
                    if len(subgraph.nodes()) > 0:
                        region_embedding = self.graph2vec_encoder.encode_graph(subgraph, region_weight)
                    else:
                        region_embedding = torch.zeros(self.config['graph2vec']['dimensions'])
                    region_embeddings.append(region_embedding.to(self.device).unsqueeze(0))

                # 预测
                global_pred, region_pred = self.ltfm_model.predict(
                    global_embedding, region_embeddings,
                    threshold=self.config['inference']['threshold']
                )

                # 获取得分
                global_score, region_scores = self.ltfm_model(global_embedding, region_embeddings)
                global_prob = torch.sigmoid(global_score).item()

                region_probs = []
                if region_scores:
                    for score in region_scores:
                        prob = torch.sigmoid(score).item()
                        region_probs.append(prob)

            # 构建结果
            result = {
                'global_anomaly': bool(global_pred.item()),
                'global_probability': global_prob,
                'anomaly_region': int(region_pred.item()) if global_pred.item() else 0,
                'region_probabilities': region_probs,
                'partitions': partitions,
                'pressure_differences': node_weights,
                'timestamp': pd.Timestamp.now()
            }

            # 添加置信度评估
            if global_prob > self.config['inference']['confidence_threshold']:
                result['confidence'] = 'high'
            elif global_prob > 0.3:
                result['confidence'] = 'medium'
            else:
                result['confidence'] = 'low'

            return result

        except Exception as e:
            logger.error(f"异常预测失败: {e}")
            return {}

    def batch_predict(self, pressure_data_list: List[pd.DataFrame]) -> List[Dict]:
        """
        批量预测

        Args:
            pressure_data_list: 压力数据列表

        Returns:
            List[Dict]: 预测结果列表
        """
        try:
            results = []
            for i, pressure_data in enumerate(pressure_data_list):
                logger.info(f"预测第 {i+1}/{len(pressure_data_list)} 个样本")
                result = self.predict_anomaly(pressure_data)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return []

    def save_prediction_results(self, results: List[Dict], output_path: str) -> bool:
        """
        保存预测结果

        Args:
            results: 预测结果列表
            output_path: 输出路径

        Returns:
            bool: 保存是否成功
        """
        try:
            if not results:
                logger.warning("预测结果为空")
                return False

            # 转换为DataFrame
            df_data = []
            for i, result in enumerate(results):
                row = {
                    'sample_id': i,
                    'global_anomaly': result.get('global_anomaly', False),
                    'global_probability': result.get('global_probability', 0.0),
                    'anomaly_region': result.get('anomaly_region', 0),
                    'confidence': result.get('confidence', 'unknown'),
                    'timestamp': result.get('timestamp', pd.Timestamp.now())
                }

                # 添加区域概率
                region_probs = result.get('region_probabilities', [])
                for j, prob in enumerate(region_probs):
                    row[f'region_{j}_probability'] = prob

                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

            logger.info(f"预测结果已保存到 {output_path}")
            return True

        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
            return False

    def get_network_status(self) -> Dict:
        """
        获取网络状态信息

        Returns:
            Dict: 网络状态字典
        """
        try:
            if self.epanet_handler is None:
                return {}

            network_info = self.epanet_handler.get_network_info()

            status = {
                'network_loaded': self.network_graph is not None,
                'models_loaded': self.ltfm_model is not None and self.graph2vec_encoder is not None,
                'normal_pressure_available': self.normal_pressure is not None,
                'n_nodes': network_info.get('n_nodes', 0),
                'n_pipes': network_info.get('n_pipes', 0),
                'n_partitions': self.n_clusters or 0,
                'device': str(self.device)
            }

            return status

        except Exception as e:
            logger.error(f"获取网络状态失败: {e}")
            return {}
