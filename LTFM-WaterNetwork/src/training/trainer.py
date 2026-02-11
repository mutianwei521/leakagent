# -*- coding: utf-8 -*-
"""
训练模块
实现模型训练流程，包括损失函数计算和反向传播
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tqdm import tqdm
import os
import networkx as nx
import scipy
import scipy.sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..models.ltfm_model import LTFMModel
from ..models.graph2vec_encoder import Graph2VecEncoder
from ..models.node_localizer import NodeLocalizer
from ..data.epanet_handler import EPANETHandler
from ..data.sensitivity_analyzer import SensitivityAnalyzer
from ..utils.fcm_partitioner import FCMPartitioner


def custom_collate_fn(batch):
    """
    自定义collate函数，用于处理包含NetworkX图的批次数据

    Args:
        batch: 批次数据列表

    Returns:
        List: 原始批次数据（不进行collate操作）
    """
    return batch


class PrecomputedDataset(Dataset):
    """预计算嵌入数据集：直接存储tensor，避免重复编码"""

    def __init__(self, global_embeddings: torch.Tensor, 
                 region_embeddings: List[torch.Tensor],
                 global_labels: torch.Tensor, 
                 region_labels: torch.Tensor):
        """
        Args:
            global_embeddings: [N, embed_dim]
            region_embeddings: List of [N, embed_dim], length = num_regions
            global_labels: [N]
            region_labels: [N]
        """
        self.global_embeddings = global_embeddings
        self.region_embeddings = region_embeddings
        self.global_labels = global_labels
        self.region_labels = region_labels
    
    def __len__(self):
        return self.global_embeddings.shape[0]
    
    def __getitem__(self, idx):
        return {
            'global_emb': self.global_embeddings[idx],
            'region_embs': [r[idx] for r in self.region_embeddings],
            'global_label': self.global_labels[idx],
            'region_label': self.region_labels[idx]
        }


def precomputed_collate_fn(batch):
    """Collate function for PrecomputedDataset"""
    global_embs = torch.stack([b['global_emb'] for b in batch])
    num_regions = len(batch[0]['region_embs'])
    region_embs = [
        torch.stack([b['region_embs'][i] for b in batch])
        for i in range(num_regions)
    ]
    global_labels = torch.stack([b['global_label'] for b in batch])
    region_labels = torch.stack([b['region_label'] for b in batch])
    return global_embs, region_embs, global_labels, region_labels


class WaterNetworkDataset(Dataset):
    """供水管网数据集（兼容旧接口）"""

    def __init__(self, scenarios: List[Dict]):
        """
        初始化数据集

        Args:
            scenarios: 场景列表，每个场景包含：
                - global_graph: 全局图
                - region_graphs: 区域图列表
                - global_weights: 全局节点权重
                - region_weights: 区域节点权重列表
                - global_label: 全局标签（0正常，1异常）
                - region_label: 区域标签（0正常，>0异常区域索引）
        """
        self.scenarios = scenarios

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]


class LTFMTrainer:
    """LTFM模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_cuda', True) else 'cpu')
        
        # 模型组件
        self.graph2vec_encoder = None
        self.ltfm_model = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_composite_score = float('-inf')  # 综合评分
        self.patience_counter = 0
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def initialize_models(self):
        """初始化模型"""
        try:
            # Graph2Vec编码器
            self.graph2vec_encoder = Graph2VecEncoder(
                embedding_dim=int(self.config['graph2vec']['dimensions']),
                wl_iterations=3,
                workers=int(self.config['graph2vec']['workers']),
                epochs=int(self.config['graph2vec']['epochs']),
                min_count=int(self.config['graph2vec']['min_count']),
                learning_rate=float(self.config['graph2vec']['learning_rate'])
            ).to(self.device)
            
            # LTFM模型
            self.ltfm_model = LTFMModel(
                graph2vec_dim=int(self.config['graph2vec']['dimensions']),
                embed_dim=int(self.config['ltfm']['embedding_dim']),
                num_heads=int(self.config['ltfm']['num_heads']),
                num_layers=int(self.config['ltfm']['num_layers']),
                num_regions=int(self.config['fcm']['n_clusters']),
                hidden_dim=int(self.config['ltfm']['hidden_dim']),
                dropout=float(self.config['ltfm']['dropout'])
            ).to(self.device)
            
            # 优化器 - 使用AdamW和改进的参数
            self.optimizer = optim.AdamW(
                self.ltfm_model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=float(self.config['training']['weight_decay']),
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # 学习率调度器 - 使用ReduceLROnPlateau（更适合我们的任务）
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )

            # 梯度裁剪参数
            self.max_grad_norm = 1.0
            
            logger.info("模型初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            return False
    
    def generate_training_scenarios(self, epanet_handler: EPANETHandler,
                                  sensitivity_analyzer: SensitivityAnalyzer,
                                  fcm_partitioner: FCMPartitioner,
                                  n_scenarios: int = 1000) -> List[Dict]:
        """
        生成训练场景
        
        Args:
            epanet_handler: EPANET处理器
            sensitivity_analyzer: 灵敏度分析器
            fcm_partitioner: FCM分区器
            n_scenarios: 场景数量
            
        Returns:
            List[Dict]: 训练场景列表
        """
        try:
            scenarios = []
            node_names = epanet_handler.node_names
            n_nodes = len(node_names)
            
            # 获取网络图和分区信息
            network_graph = epanet_handler.get_network_graph()
            partition_info = fcm_partitioner.get_partition_info()
            subgraphs = fcm_partitioner.get_partition_subgraphs(network_graph, node_names)
            
            logger.info(f"开始生成 {n_scenarios} 个训练场景")
            
            # 生成正常场景 - 增加多样性
            n_normal = n_scenarios // 2
            for i in tqdm(range(n_normal), desc="生成正常场景"):
                # 正常场景：添加小幅度的自然变化
                normal_weights = {}
                for node in node_names:
                    # 添加小幅度的随机变化，模拟正常运行时的自然波动
                    # 变化范围：-0.05 到 +0.05（相对于异常场景的变化幅度很小）
                    natural_variation = np.random.uniform(-0.05, 0.05)
                    normal_weights[node] = natural_variation

                # 为每个区域生成略有不同的权重
                region_weights = []
                for subgraph in subgraphs:
                    region_weight = {}
                    for node in node_names:
                        if node in subgraph.nodes():
                            # 区域内节点有相关性的小变化
                            base_variation = normal_weights[node]
                            region_variation = base_variation + np.random.uniform(-0.02, 0.02)
                            region_weight[node] = region_variation
                        else:
                            region_weight[node] = normal_weights[node]
                    region_weights.append(region_weight)

                scenario = {
                    'global_graph': network_graph,
                    'region_graphs': subgraphs,
                    'global_weights': normal_weights,
                    'region_weights': region_weights,
                    'global_label': 0,  # 正常
                    'region_label': 0,  # 正常
                    'anomaly_node_idx': -1,  # 正常场景无漏损节点
                    'anomaly_node_name': None,
                    'node_names': node_names
                }
                scenarios.append(scenario)
            
            # 生成异常场景 - 增强版
            n_anomaly = n_scenarios - n_normal
            anomaly_count = 0
            max_attempts = n_anomaly * 3  # 最多尝试3倍的次数

            for attempt in tqdm(range(max_attempts), desc="生成异常场景"):
                if anomaly_count >= n_anomaly:
                    break

                # 随机选择异常节点
                anomaly_node_idx = np.random.randint(0, n_nodes)
                anomaly_node = node_names[anomaly_node_idx]

                # 获取异常节点的灵敏度特征
                sensitivity_features = sensitivity_analyzer.get_sensitivity_features(anomaly_node_idx)

                if not sensitivity_features:
                    logger.debug(f"节点 {anomaly_node} 灵敏度特征为空，跳过")
                    continue

                # 构建节点权重（压力差）
                pressure_sensitivity = sensitivity_features['avg_pressure_sensitivity']

                # 检查压力灵敏度是否有效
                if len(pressure_sensitivity) != n_nodes:
                    logger.debug(f"节点 {anomaly_node} 压力灵敏度长度不匹配，跳过")
                    continue

                # 数据增强：添加噪声和变换
                augmented_pressure_sensitivity = self._augment_pressure_sensitivity(
                    pressure_sensitivity, anomaly_node_idx
                )

                global_weights = {node_names[j]: augmented_pressure_sensitivity[j] for j in range(n_nodes)}

                # 确定异常区域 - 修复标签编码
                partition_labels = partition_info['partition_labels']
                # 异常区域标签：0=正常，1=区域0异常，2=区域1异常，3=区域2异常...
                anomaly_region = partition_labels[anomaly_node_idx] + 1  # +1因为0表示正常

                # 构建区域权重 - 增强异常区域的特征
                region_weights = []
                for region_id, subgraph in enumerate(subgraphs):
                    region_weight = {}
                    for node in subgraph.nodes():
                        node_idx = node_names.index(node)
                        weight = augmented_pressure_sensitivity[node_idx]

                        # 如果是异常区域，大幅增强权重
                        if partition_labels[node_idx] == (anomaly_region - 1):
                            weight *= np.random.uniform(3.0, 5.0)  # 大幅增强异常区域权重
                        else:
                            # 非异常区域，适当降低权重
                            weight *= np.random.uniform(0.3, 0.7)

                        region_weight[node] = weight
                    region_weights.append(region_weight)

                scenario = {
                    'global_graph': network_graph,
                    'region_graphs': subgraphs,
                    'global_weights': global_weights,
                    'region_weights': region_weights,
                    'global_label': 1,  # 异常
                    'region_label': anomaly_region,  # 异常区域索引
                    'anomaly_node_idx': anomaly_node_idx,
                    'anomaly_node_name': anomaly_node,
                    'node_names': node_names,
                    'partition_labels': partition_labels
                }
                scenarios.append(scenario)
                anomaly_count += 1

                # 生成多种变体（增加数据多样性）
                variants_to_generate = min(2, n_anomaly - anomaly_count)  # 每个基础场景最多生成2个变体
                for _ in range(variants_to_generate):
                    if anomaly_count >= n_anomaly:
                        break

                    variant_scenario = self._create_scenario_variant(
                        scenario, node_names, partition_labels, anomaly_node_idx
                    )
                    if variant_scenario:
                        scenarios.append(variant_scenario)
                        anomaly_count += 1

            logger.info(f"成功生成 {anomaly_count} 个异常场景（目标: {n_anomaly}）")
            
            logger.info(f"训练场景生成完成: {len(scenarios)} 个场景")
            return scenarios

        except Exception as e:
            logger.error(f"训练场景生成失败: {e}")
            return []

    def _augment_pressure_sensitivity(self, pressure_sensitivity: np.ndarray, anomaly_node_idx: int) -> np.ndarray:
        """
        对压力灵敏度进行数据增强

        Args:
            pressure_sensitivity: 原始压力灵敏度
            anomaly_node_idx: 异常节点索引

        Returns:
            np.ndarray: 增强后的压力灵敏度
        """
        augmented = pressure_sensitivity.copy()

        # 1. 添加高斯噪声
        noise_level = 0.05  # 5%的噪声
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented = augmented + noise

        # 2. 增强异常节点的影响
        anomaly_boost = np.random.uniform(1.5, 3.0)
        augmented[anomaly_node_idx] *= anomaly_boost

        # 3. 随机缩放部分节点
        scale_mask = np.random.random(augmented.shape) < 0.3
        scale_factors = np.random.uniform(0.8, 1.2, augmented.shape)
        augmented[scale_mask] *= scale_factors[scale_mask]

        # 4. 确保数值稳定性
        augmented = np.clip(augmented, -10, 10)

        return augmented

    def _create_scenario_variant(self, base_scenario: Dict, node_names: List[str],
                               partition_labels: np.ndarray, anomaly_node_idx: int) -> Optional[Dict]:
        """
        创建场景变体

        Args:
            base_scenario: 基础场景
            node_names: 节点名称列表
            partition_labels: 分区标签
            anomaly_node_idx: 异常节点索引

        Returns:
            Optional[Dict]: 变体场景或None
        """
        try:
            variant = base_scenario.copy()
            # 保留anomaly_node_idx等节点级信息
            variant['anomaly_node_idx'] = base_scenario.get('anomaly_node_idx', -1)
            variant['anomaly_node_name'] = base_scenario.get('anomaly_node_name', None)
            variant['node_names'] = base_scenario.get('node_names', node_names)
            variant['partition_labels'] = base_scenario.get('partition_labels', partition_labels)
            variant_type = np.random.choice(['perturbation', 'intensity', 'multi_node', 'temporal'])

            # 变体1：权重扰动
            if variant_type == 'perturbation':
                perturbed_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    # 添加高斯噪声扰动
                    noise_level = 0.15  # 增加噪声水平
                    perturbation = np.random.normal(0, abs(weight) * noise_level)
                    perturbed_global_weights[node] = weight + perturbation
                variant['global_weights'] = perturbed_global_weights

                # 同步更新区域权重
                perturbed_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    perturbed_region = {}
                    for node, weight in region_weight.items():
                        perturbation = np.random.normal(0, abs(weight) * noise_level)
                        perturbed_region[node] = weight + perturbation
                    perturbed_region_weights.append(perturbed_region)
                variant['region_weights'] = perturbed_region_weights

            # 变体2：异常强度调整
            elif variant_type == 'intensity':
                intensity_factor = np.random.uniform(0.5, 2.0)  # 扩大强度范围
                adjusted_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    adjusted_global_weights[node] = weight * intensity_factor
                variant['global_weights'] = adjusted_global_weights

                adjusted_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    adjusted_region = {}
                    for node, weight in region_weight.items():
                        adjusted_region[node] = weight * intensity_factor
                    adjusted_region_weights.append(adjusted_region)
                variant['region_weights'] = adjusted_region_weights

            # 变体3：多节点异常（同一区域内的多个节点）
            elif variant_type == 'multi_node':
                anomaly_region_idx = partition_labels[anomaly_node_idx]
                # 找到同一区域的其他节点
                same_region_nodes = [i for i, label in enumerate(partition_labels)
                                   if label == anomaly_region_idx and i != anomaly_node_idx]

                if same_region_nodes:
                    # 随机选择1-2个额外节点
                    additional_nodes = np.random.choice(
                        same_region_nodes,
                        size=min(2, len(same_region_nodes)),
                        replace=False
                    )

                    # 增强这些节点的权重
                    enhanced_global_weights = base_scenario['global_weights'].copy()
                    for node_idx in additional_nodes:
                        node_name = node_names[node_idx]
                        if node_name in enhanced_global_weights:
                            enhanced_global_weights[node_name] *= np.random.uniform(2.0, 4.0)

                    variant['global_weights'] = enhanced_global_weights

                    # 同步更新区域权重
                    enhanced_region_weights = []
                    for region_weight in base_scenario['region_weights']:
                        enhanced_region = region_weight.copy()
                        for node_idx in additional_nodes:
                            node_name = node_names[node_idx]
                            if node_name in enhanced_region:
                                enhanced_region[node_name] *= np.random.uniform(2.0, 4.0)
                        enhanced_region_weights.append(enhanced_region)
                    variant['region_weights'] = enhanced_region_weights

            # 变体4：时序变化模拟（通过权重梯度变化）
            elif variant_type == 'temporal':
                temporal_global_weights = {}
                for node, weight in base_scenario['global_weights'].items():
                    # 模拟时序变化：添加周期性变化
                    phase = np.random.uniform(0, 2 * np.pi)
                    amplitude = abs(weight) * 0.3
                    temporal_factor = 1 + amplitude * np.sin(phase)
                    temporal_global_weights[node] = weight * temporal_factor
                variant['global_weights'] = temporal_global_weights

                # 同步更新区域权重
                temporal_region_weights = []
                for region_weight in base_scenario['region_weights']:
                    temporal_region = {}
                    for node, weight in region_weight.items():
                        phase = np.random.uniform(0, 2 * np.pi)
                        amplitude = abs(weight) * 0.3
                        temporal_factor = 1 + amplitude * np.sin(phase)
                        temporal_region[node] = weight * temporal_factor
                    temporal_region_weights.append(temporal_region)
                variant['region_weights'] = temporal_region_weights

            return variant

        except Exception as e:
            logger.debug(f"创建场景变体失败: {e}")
            return None
            
        except Exception as e:
            logger.error(f"生成训练场景失败: {e}")
            return []
    
    def precompute_embeddings(self, scenarios: List[Dict]) -> Tuple[
        torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """
        预计算所有场景的嵌入，一次性完成编码
        
        Args:
            scenarios: 场景列表
            
        Returns:
            Tuple: (全局嵌入, 区域嵌入列表, 全局标签, 区域标签)
        """
        logger.info(f"预计算 {len(scenarios)} 个场景的嵌入...")
        
        global_embeddings = []
        num_regions = len(scenarios[0]['region_graphs'])
        region_embeddings_list = [[] for _ in range(num_regions)]
        global_labels = []
        region_labels = []
        
        for i, scenario in enumerate(tqdm(scenarios, desc="预计算嵌入")):
            # 编码全局图
            global_embedding = self.graph2vec_encoder.encode_graph(
                scenario['global_graph'], scenario['global_weights']
            )
            global_embeddings.append(global_embedding)
            
            # 编码区域图
            for j, (region_graph, region_weight) in enumerate(
                zip(scenario['region_graphs'], scenario['region_weights'])
            ):
                region_embedding = self.graph2vec_encoder.encode_graph(
                    region_graph, region_weight
                )
                region_embeddings_list[j].append(region_embedding)
            
            # 标签
            global_labels.append(scenario['global_label'])
            region_labels.append(scenario['region_label'])
        
        # 转换为张量
        global_emb_tensor = torch.stack(global_embeddings)
        region_emb_tensors = [
            torch.stack(region_embs) for region_embs in region_embeddings_list
        ]
        global_labels_tensor = torch.tensor(global_labels, dtype=torch.float32)
        region_labels_tensor = torch.tensor(region_labels, dtype=torch.long)
        
        logger.info(f"预计算完成: 全局嵌入 {global_emb_tensor.shape}, "
                    f"{num_regions} 个区域嵌入 {region_emb_tensors[0].shape}")
        
        # 验证嵌入有差异性
        emb_std = global_emb_tensor.std(dim=0).mean().item()
        logger.info(f"全局嵌入标准差（跨样本平均）: {emb_std:.6f}")
        if emb_std < 1e-6:
            logger.warning("⚠️ 嵌入几乎没有差异！编码器可能未正确捕获特征差异")
        
        return global_emb_tensor, region_emb_tensors, global_labels_tensor, region_labels_tensor

    def prepare_batch_data(self, batch: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor], 
                                                           torch.Tensor, torch.Tensor]:
        """
        准备批次数据（兼容旧接口，但建议使用 precompute_embeddings）
        
        Args:
            batch: 批次数据
            
        Returns:
            Tuple: (全局图嵌入, 区域图嵌入列表, 全局标签, 区域标签)
        """
        try:
            global_embeddings = []
            region_embeddings_list = [[] for _ in range(len(batch[0]['region_graphs']))]
            global_labels = []
            region_labels = []
            
            for scenario in batch:
                # 编码全局图
                global_embedding = self.graph2vec_encoder.encode_graph(
                    scenario['global_graph'], scenario['global_weights']
                )
                global_embeddings.append(global_embedding)
                
                # 编码区域图
                for i, (region_graph, region_weight) in enumerate(
                    zip(scenario['region_graphs'], scenario['region_weights'])
                ):
                    region_embedding = self.graph2vec_encoder.encode_graph(region_graph, region_weight)
                    region_embeddings_list[i].append(region_embedding)
                
                # 标签
                global_labels.append(scenario['global_label'])
                region_labels.append(scenario['region_label'])
            
            # 转换为张量
            global_embeddings_tensor = torch.stack(global_embeddings).to(self.device)
            region_embeddings_tensors = [
                torch.stack(region_embs).to(self.device) 
                for region_embs in region_embeddings_list
            ]
            global_labels_tensor = torch.tensor(global_labels, dtype=torch.float32).to(self.device)
            region_labels_tensor = torch.tensor(region_labels, dtype=torch.long).to(self.device)
            
            return global_embeddings_tensor, region_embeddings_tensors, global_labels_tensor, region_labels_tensor
            
        except Exception as e:
            logger.error(f"准备批次数据失败: {e}")
            return None, None, None, None

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch（支持预计算和旧模式）

        Args:
            dataloader: 数据加载器

        Returns:
            Tuple[float, float]: (平均损失, 准确率)
        """
        self.ltfm_model.train()
        total_loss = 0.0
        total_samples = 0
        correct_global = 0
        correct_region = 0

        for batch in tqdm(dataloader, desc=f"训练 Epoch {self.current_epoch}"):
            # 判断是预计算模式还是旧模式
            if isinstance(batch, (tuple, list)) and len(batch) == 4 and isinstance(batch[0], torch.Tensor):
                # 预计算模式：batch 已经是 (global_embs, region_embs, global_labels, region_labels)
                global_embs, region_embs, global_labels, region_labels = batch
                global_embs = global_embs.to(self.device)
                region_embs = [r.to(self.device) for r in region_embs]
                global_labels = global_labels.to(self.device)
                region_labels = region_labels.to(self.device)
            else:
                # 旧模式：需要编码
                global_embs, region_embs, global_labels, region_labels = self.prepare_batch_data(batch)

            if global_embs is None:
                continue

            # 前向传播
            self.optimizer.zero_grad()
            global_scores, region_scores = self.ltfm_model(global_embs, region_embs)

            # 计算损失
            loss = self.ltfm_model.compute_loss(global_scores, region_scores, global_labels, region_labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ltfm_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            batch_size = global_embs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 计算准确率
            with torch.no_grad():
                global_pred, region_pred = self.ltfm_model.predict(global_embs, region_embs)
                correct_global += (global_pred.squeeze() == global_labels).sum().item()
                correct_region += (region_pred == region_labels).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        global_acc = correct_global / total_samples if total_samples > 0 else 0.0
        region_acc = correct_region / total_samples if total_samples > 0 else 0.0
        avg_acc = (global_acc + region_acc) / 2

        return avg_loss, avg_acc

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """
        验证一个epoch

        Args:
            dataloader: 验证数据加载器

        Returns:
            Tuple[float, float, Dict]: (平均损失, 准确率, 详细指标)
        """
        self.ltfm_model.eval()
        total_loss = 0.0
        total_samples = 0

        all_global_preds = []
        all_global_labels = []
        all_region_preds = []
        all_region_labels = []
        all_global_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证"):
                # 判断是预计算模式还是旧模式
                if isinstance(batch, (tuple, list)) and len(batch) == 4 and isinstance(batch[0], torch.Tensor):
                    global_embs, region_embs, global_labels, region_labels = batch
                    global_embs = global_embs.to(self.device)
                    region_embs = [r.to(self.device) for r in region_embs]
                    global_labels = global_labels.to(self.device)
                    region_labels = region_labels.to(self.device)
                else:
                    global_embs, region_embs, global_labels, region_labels = self.prepare_batch_data(batch)

                if global_embs is None:
                    continue

                # 前向传播
                global_scores, region_scores = self.ltfm_model(global_embs, region_embs)

                # 计算损失
                loss = self.ltfm_model.compute_loss(global_scores, region_scores, global_labels, region_labels)

                # 预测
                global_pred, region_pred = self.ltfm_model.predict(global_embs, region_embs)

                # 统计
                batch_size = global_embs.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # 收集预测结果
                global_pred_list = global_pred.cpu().numpy().flatten().tolist()
                global_labels_list = global_labels.cpu().numpy().flatten().tolist()
                region_pred_list = region_pred.cpu().numpy().flatten().tolist()
                region_labels_list = region_labels.cpu().numpy().flatten().tolist()
                global_scores_list = torch.sigmoid(global_scores).cpu().numpy().flatten().tolist()

                all_global_preds.extend(global_pred_list)
                all_global_labels.extend(global_labels_list)
                all_region_preds.extend(region_pred_list)
                all_region_labels.extend(region_labels_list)
                all_global_scores.extend(global_scores_list)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # 添加调试信息
        if len(all_global_labels) > 0:
            logger.info(f"验证集预测统计:")
            logger.info(f"  全局标签分布: 正常={all_global_labels.count(0)}, 异常={all_global_labels.count(1)}")
            global_preds_int = [int(p) for p in all_global_preds]
            logger.info(f"  全局预测分布: 正常={global_preds_int.count(0)}, 异常={global_preds_int.count(1)}")
            if all_global_scores:
                logger.info(f"  全局得分范围: [{min(all_global_scores):.4f}, {max(all_global_scores):.4f}]")
            else:
                logger.info("  全局得分范围: []")
            logger.info(f"  区域标签唯一值: {set(all_region_labels)}")
            logger.info(f"  区域预测唯一值: {set(all_region_preds)}")

            if len(set(all_global_preds)) == 1:
                logger.warning("⚠️  所有全局预测都相同！模型可能没有学到有用特征")

            if len(set(all_region_preds)) == 1:
                logger.warning("⚠️  所有区域预测都相同！模型可能没有学到有用特征")

        # 计算详细指标
        metrics = self.calculate_metrics(
            all_global_preds, all_global_labels,
            all_region_preds, all_region_labels,
            all_global_scores
        )

        avg_acc = (metrics['global_acc'] + metrics['region_acc']) / 2

        return avg_loss, avg_acc, metrics

    def calculate_metrics(self, global_preds: List, global_labels: List,
                         region_preds: List, region_labels: List,
                         global_scores: List) -> Dict:
        """
        计算评估指标

        Args:
            global_preds: 全局预测
            global_labels: 全局标签
            region_preds: 区域预测
            region_labels: 区域标签
            global_scores: 全局得分

        Returns:
            Dict: 评估指标字典
        """
        try:
            metrics = {}

            # 全局异常检测指标
            global_preds = np.array(global_preds)
            global_labels = np.array(global_labels)
            global_scores = np.array(global_scores)

            metrics['global_acc'] = accuracy_score(global_labels, global_preds)
            metrics['global_precision'] = precision_score(global_labels, global_preds, zero_division=0)
            metrics['global_recall'] = recall_score(global_labels, global_preds, zero_division=0)
            metrics['global_f1'] = f1_score(global_labels, global_preds, zero_division=0)

            unique_labels = np.unique(global_labels)
            if len(unique_labels) > 1:
                metrics['global_auc'] = roc_auc_score(global_labels, global_scores)
            else:
                metrics['global_auc'] = 0.0
                logger.debug(f"AUC设为0.0，因为标签只有一种类型: {unique_labels}")

            # 区域定位指标
            region_preds = np.array(region_preds)
            region_labels = np.array(region_labels)

            metrics['region_acc'] = accuracy_score(region_labels, region_preds)

            # 只考虑异常样本的区域定位准确率
            anomaly_mask = global_labels == 1
            if np.sum(anomaly_mask) > 0:
                anomaly_region_acc = accuracy_score(
                    region_labels[anomaly_mask],
                    region_preds[anomaly_mask]
                )
                metrics['anomaly_region_acc'] = anomaly_region_acc
            else:
                metrics['anomaly_region_acc'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            return {}

    def train(self, train_scenarios: List[Dict], val_scenarios: List[Dict], skip_stage1: bool = False) -> bool:
        """
        训练模型（使用预计算嵌入加速）

        Args:
            train_scenarios: 训练场景
            val_scenarios: 验证场景
            skip_stage1: 是否跳过Stage 1训练

        Returns:
            bool: 训练是否成功
        """
        try:
            # 预计算所有嵌入（一次性完成，避免每个epoch重复编码）
            logger.info("=" * 60)
            logger.info("步骤1: 预计算嵌入（仅执行一次）")
            logger.info("=" * 60)
            
            train_global, train_regions, train_glabels, train_rlabels = \
                self.precompute_embeddings(train_scenarios)
            val_global, val_regions, val_glabels, val_rlabels = \
                self.precompute_embeddings(val_scenarios)
            
            # [NEW] Skip Stage 1 Logic
            if skip_stage1:
                logger.info("侦测到跳过Stage 1请求...")
                checkpoint_path = os.path.join(self.config['data']['output_dir'], 'checkpoints', 'best_model.pth')
                if os.path.exists(checkpoint_path):
                    logger.info(f"加载已有模型: {checkpoint_path}")
                    try:
                        self.load_checkpoint(checkpoint_path)
                        logger.info("✅ 模型加载成功，跳过Stage 1训练")
                        
                        # 重要: 必须保存预计算嵌入供 Stage 2 使用
                        self._precomputed_train = (train_global, train_regions, train_glabels, train_rlabels)
                        self._precomputed_val = (val_global, val_regions, val_glabels, val_rlabels)
                        return True
                    except Exception as e:
                        logger.warning(f"加载模型失败: {e}. 将继续进行训练...")
                else:
                    logger.warning(f"未找到检查点 {checkpoint_path}. 将继续进行训练...")

            # 创建预计算数据集
            train_dataset = PrecomputedDataset(
                train_global, train_regions, train_glabels, train_rlabels
            )
            val_dataset = PrecomputedDataset(
                val_global, val_regions, val_glabels, val_rlabels
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=0,  # Windows兼容性
                collate_fn=precomputed_collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=0,
                collate_fn=precomputed_collate_fn
            )

            # 检查数据集标签分布
            train_global_labels = [s['global_label'] for s in train_scenarios]
            val_global_labels = [s['global_label'] for s in val_scenarios]
            train_region_labels = [s['region_label'] for s in train_scenarios]
            val_region_labels = [s['region_label'] for s in val_scenarios]

            logger.info(f"开始训练: {len(train_scenarios)} 训练样本, {len(val_scenarios)} 验证样本")
            logger.info(f"训练集标签分布 - 正常: {train_global_labels.count(0)}, 异常: {train_global_labels.count(1)}")
            logger.info(f"验证集标签分布 - 正常: {val_global_labels.count(0)}, 异常: {val_global_labels.count(1)}")
            logger.info(f"训练集区域标签分布: {set(train_region_labels)}")
            logger.info(f"验证集区域标签分布: {set(val_region_labels)}")

            # 训练循环
            for epoch in range(self.config['training']['epochs']):
                self.current_epoch = epoch + 1

                # 训练
                train_loss, train_acc = self.train_epoch(train_loader)

                # 验证
                val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)

                # 更新学习率
                self.scheduler.step(val_loss)

                # 记录历史
                self.train_history['train_loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['train_acc'].append(train_acc)
                self.train_history['val_acc'].append(val_acc)

                # 日志
                logger.info(
                    f"Epoch {self.current_epoch}/{self.config['training']['epochs']} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                logger.info(
                    f"Val Metrics - Global AUC: {val_metrics.get('global_auc', 0):.4f}, "
                    f"Global F1: {val_metrics.get('global_f1', 0):.4f}, "
                    f"Anomaly Region Acc: {val_metrics.get('anomaly_region_acc', 0):.4f}"
                )

                # 改进的早停检查 - 综合考虑多个指标
                # 计算综合评分：损失 + 区域准确率
                region_acc = val_metrics.get('anomaly_region_acc', 0.0)
                global_auc = val_metrics.get('global_auc', 0.0)

                # 综合评分：损失越低越好，准确率越高越好
                composite_score = -val_loss + 2.0 * region_acc + 1.0 * global_auc

                if composite_score > self.best_composite_score:
                    self.best_composite_score = composite_score
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                    logger.info(f"新的最佳模型 - 综合评分: {composite_score:.4f}, 区域准确率: {region_acc:.4f}")
                else:
                    self.patience_counter += 1

                # 动态调整早停耐心值
                patience = self.config['training']['early_stopping_patience']
                if region_acc > 0.1:  # 如果区域准确率有改善，增加耐心
                    patience = patience + 5

                if self.patience_counter >= patience:
                    logger.info(f"早停触发，在第 {self.current_epoch} 轮停止训练")
                    logger.info(f"最佳综合评分: {self.best_composite_score:.4f}")
                    break

                # 定期保存检查点
                if self.current_epoch % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pth')

            logger.info("LTFM训练完成")
            
            # 保存预计算嵌入供 NodeLocalizer 使用
            self._precomputed_train = (train_global, train_regions, train_glabels, train_rlabels)
            self._precomputed_val = (val_global, val_regions, val_glabels, val_rlabels)
            
            return True

        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False

    def train_node_localizer(self, train_scenarios: List[Dict], 
                              val_scenarios: List[Dict],
                              n_epochs: int = 100,
                              lr: float = 0.001) -> bool:
        """
        Stage 2: 训练NodeLocalizer模型
        使用冻结的LTFM中间特征 + 灵敏度向量 → 预测漏损节点
        包含数据增强(20x Gaussian noise)以解决样本不足问题
        
        Args:
            train_scenarios: 训练场景（包含anomaly_node_idx）
            val_scenarios: 验证场景
            n_epochs: 训练轮数
            lr: 学习率
            
        Returns:
            bool: 训练是否成功
        """
        try:
            logger.info("=" * 60)
            logger.info("Stage 2: 训练NodeLocalizer（节点级漏损定位）")
            logger.info("=" * 60)
            
            # 1. 筛选异常场景（只有异常场景有漏损节点标签）
            train_anomaly = [s for s in train_scenarios if s['global_label'] == 1]
            val_anomaly = [s for s in val_scenarios if s['global_label'] == 1]
            
            if not train_anomaly or not val_anomaly:
                logger.error("没有异常场景用于训练NodeLocalizer")
                return False
            
            logger.info(f"异常场景: 训练={len(train_anomaly)}, 验证={len(val_anomaly)}")
            
            # 2. 获取节点信息
            node_names = train_anomaly[0].get('node_names', [])
            n_nodes = len(node_names)
            network_graph = train_anomaly[0]['global_graph']
            
            if n_nodes == 0:
                logger.error("无法获取节点名称信息")
                return False
            
            logger.info(f"网络节点数: {n_nodes}")
            
            # 3. 使用预计算嵌入提取中间特征（快速模式）
            logger.info("使用预计算嵌入提取区域特征（跳过图编码）...")
            self.ltfm_model.eval()
            
            # 尝试使用已缓存的预计算嵌入
            if hasattr(self, '_precomputed_train') and hasattr(self, '_precomputed_val'):
                train_global, train_regions, train_glabels, train_rlabels = self._precomputed_train
                val_global, val_regions, val_glabels, val_rlabels = self._precomputed_val
                logger.info("✅ 复用LTFM训练阶段的预计算嵌入")
            else:
                logger.info("⚠️ 无缓存嵌入，重新计算...")
                train_global, train_regions, train_glabels, train_rlabels = \
                    self.precompute_embeddings(train_scenarios)
                val_global, val_regions, val_glabels, val_rlabels = \
                    self.precompute_embeddings(val_scenarios)
            
            # 筛选异常场景的索引
            train_anomaly_indices = [i for i, s in enumerate(train_scenarios) if s['global_label'] == 1]
            val_anomaly_indices = [i for i, s in enumerate(val_scenarios) if s['global_label'] == 1]
            
            train_features, train_sens, train_labels, train_masks = self._extract_node_features_fast(
                train_scenarios, train_anomaly_indices,
                train_global, train_regions, train_rlabels, node_names
            )
            val_features, val_sens, val_labels, val_masks = self._extract_node_features_fast(
                val_scenarios, val_anomaly_indices,
                val_global, val_regions, val_rlabels, node_names
            )
            
            if train_features is None:
                logger.error("特征提取失败")
                return False
            
            # 统计区域约束信息
            avg_region_size = train_masks.float().sum(dim=1).mean().item()
            logger.info(f"原始训练特征: region_feat={train_features.shape}, "
                        f"sens={train_sens.shape}, labels={train_labels.shape}")
            logger.info(f"区域约束: 平均每个样本只在 {avg_region_size:.0f}/{n_nodes} 个节点中分类")
            
            # 4. 数据增强: 添加高斯噪声创建40x更多训练样本
            augment_factor = 40
            logger.info(f"数据增强: {augment_factor}x 高斯噪声扩增...")
            
            aug_features_list = [train_features]
            aug_sens_list = [train_sens]
            aug_labels_list = [train_labels]
            aug_masks_list = [train_masks]
            
            feat_std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
            sens_std = train_sens.std(dim=0, keepdim=True).clamp(min=1e-6)
            
            for aug_i in range(augment_factor):
                # 递增噪声强度: 0.05 ~ 0.5
                noise_scale = 0.05 + 0.45 * (aug_i / max(augment_factor - 1, 1))
                
                feat_noise = torch.randn_like(train_features) * feat_std * noise_scale
                sens_noise = torch.randn_like(train_sens) * sens_std * noise_scale
                
                # 增强特征
                aug_feat = train_features + feat_noise
                aug_sens = train_sens + sens_noise
                
                # [NEW] 灵敏度掩码增强 (Sensitivity Masking)
                # DISABLE: Masking breaks the physics bias alignment
                # 随机将 15% 的灵敏度数值置为0 -> 取消
                # mask_prob = 0.15
                # sens_mask = torch.rand_like(aug_sens) > mask_prob
                # aug_sens = aug_sens * sens_mask.float()
                
                aug_features_list.append(aug_feat)
                aug_sens_list.append(aug_sens)
                aug_labels_list.append(train_labels.clone())
                aug_masks_list.append(train_masks.clone())  # 区域掩码不变
            
            train_features_aug = torch.cat(aug_features_list, dim=0)
            train_sens_aug = torch.cat(aug_sens_list, dim=0)
            train_labels_aug = torch.cat(aug_labels_list, dim=0)
            train_masks_aug = torch.cat(aug_masks_list, dim=0)
            
            logger.info(f"增强后训练样本: {train_features_aug.shape[0]} "
                        f"(原始 {train_features.shape[0]} × {augment_factor + 1})")
            
            # 5. 初始化NodeLocalizer（从配置读取参数）
            nl_config = self.config.get('node_localizer', {})
            hidden_dim = int(nl_config.get('hidden_dim', 256))
            dropout = float(nl_config.get('dropout', 0.1))
            epochs = int(nl_config.get('epochs', n_epochs))
            lr = float(nl_config.get('learning_rate', lr))
            
            logger.info(f"NodeLocalizer配置: hidden_dim={hidden_dim}, dropout={dropout}, epochs={epochs}, lr={lr}")

            embed_dim = self.ltfm_model.embed_dim
            self.node_localizer = NodeLocalizer(
                region_dim=embed_dim,
                n_nodes=n_nodes,
                hidden_dim=hidden_dim,
                dropout=dropout
            ).to(self.device)
            
            optimizer = optim.Adam(self.node_localizer.parameters(), lr=lr, weight_decay=1e-4) # 增加weight_decay防止过拟合
            criterion = nn.CrossEntropyLoss()
            
            # 6. 创建DataLoader（包含区域掩码）
            train_dataset = torch.utils.data.TensorDataset(
                train_features_aug, train_sens_aug, train_labels_aug, train_masks_aug
            )
            val_dataset = torch.utils.data.TensorDataset(
                val_features, val_sens, val_labels, val_masks
            )
            
            batch_size = min(512, len(train_dataset))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # OneCycleLR for better convergence
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr * 3, 
                steps_per_epoch=len(train_loader),
                epochs=epochs
            )
            
            # 7. 训练循环
            best_node_acc_hop = 0.0
            patience_counter = 0
            
            for epoch in range(epochs):
                # 训练
                self.node_localizer.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0
                
                for feat_batch, sens_batch, label_batch, mask_batch in train_loader:
                    feat_batch = feat_batch.to(self.device)
                    sens_batch = sens_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device) 
                    
                    optimizer.zero_grad()
                    node_scores = self.node_localizer(feat_batch, sens_batch)
                    
                    # 区域约束: 非区域内节点的logit设为-inf
                    node_scores = node_scores.masked_fill(~mask_batch, -1e9)
                    loss = criterion(node_scores, label_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.node_localizer.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item() * feat_batch.shape[0]
                    preds = node_scores.argmax(dim=-1)
                    total_correct += (preds == label_batch).sum().item()
                    total_samples += feat_batch.shape[0]
                
                train_loss = total_loss / total_samples
                train_acc = total_correct / total_samples
                
                # 验证
                self.node_localizer.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for feat_batch, sens_batch, label_batch, mask_batch in val_loader:
                        feat_batch = feat_batch.to(self.device)
                        sens_batch = sens_batch.to(self.device)
                        label_batch = label_batch.to(self.device)
                        mask_batch = mask_batch.to(self.device)
                        
                        node_scores = self.node_localizer(feat_batch, sens_batch)
                        
                        # 区域约束
                        node_scores = node_scores.masked_fill(~mask_batch, -1e9)
                        loss = criterion(node_scores, label_batch)
                        val_loss += loss.item() * feat_batch.shape[0]
                        
                        preds = node_scores.argmax(dim=-1)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(label_batch.cpu().tolist())
                
                val_loss = val_loss / len(val_dataset)
                
                # 计算节点准确率（精确 + 邻居容差）
                node_acc_exact, node_acc_hop = self.calculate_node_accuracy(
                    all_preds, all_labels, node_names, network_graph
                )
                
                # 每5轮或有进展时打印日志
                should_log = (epoch + 1) % 5 == 0 or epoch == 0 or node_acc_hop > best_node_acc_hop
                if should_log:
                    logger.info(
                    f"NodeLocalizer Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Node Acc (exact): {node_acc_exact:.4f}, "
                    f"Node Acc (±1 hop): {node_acc_hop:.4f}"
                )
                
                # 早停
                if node_acc_hop > best_node_acc_hop:
                    best_node_acc_hop = node_acc_hop
                    patience_counter = 0
                    # 保存最优NodeLocalizer
                    self._save_node_localizer('best_node_localizer.pth')
                    logger.info(f"新的最佳NodeLocalizer - Node Acc (±1 hop): {node_acc_hop:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= 25:
                    logger.info(f"NodeLocalizer早停，在第 {epoch+1} 轮停止")
                    break
            
            logger.info(f"NodeLocalizer训练完成 - 最佳 Node Acc (±1 hop): {best_node_acc_hop:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"NodeLocalizer训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_node_features_fast(self, scenarios: List[Dict], 
                                      anomaly_indices: List[int],
                                      global_embs: torch.Tensor,
                                      region_embs_list: List[torch.Tensor],
                                      region_labels: torch.Tensor,
                                      node_names: List[str]):
        """
        快速提取节点特征：复用预计算嵌入 + 批量LTFM前向传播
        
        Args:
            scenarios: 全部场景列表
            anomaly_indices: 异常场景的索引
            global_embs: 预计算的全局嵌入 [N, dim]
            region_embs_list: 预计算的区域嵌入列表
            region_labels: 区域标签 [N]
            node_names: 节点名称列表
            
        Returns:
            Tuple: (region_features, sensitivity_vectors, node_labels, region_masks)
                   region_masks: [n_samples, n_nodes] bool, True=node在异常区域内
        """
        try:
            n_nodes = len(node_names)
            batch_size = 128
            
            # 1. 提取异常场景的嵌入
            anom_global = global_embs[anomaly_indices]  # [n_anomaly, dim]
            anom_regions = [r[anomaly_indices] for r in region_embs_list]  # List of [n_anomaly, dim]
            anom_rlabels = region_labels[anomaly_indices]  # [n_anomaly]
            
            n_anomaly = anom_global.shape[0]
            
            # 2. 批量LTFM前向传播提取中间特征
            all_region_feats = []
            
            with torch.no_grad():
                for start in range(0, n_anomaly, batch_size):
                    end = min(start + batch_size, n_anomaly)
                    
                    batch_global = anom_global[start:end].to(self.device)
                    batch_regions = [r[start:end].to(self.device) for r in anom_regions]
                    batch_rlabels = anom_rlabels[start:end]
                    
                    _, _, intermediate_feats = self.ltfm_model(
                        batch_global, batch_regions, return_features=True
                    )
                    # intermediate_feats: List of [batch, embed_dim], one per region
                    
                    # 每个样本取其异常区域的特征
                    for i in range(end - start):
                        rlabel = batch_rlabels[i].item()
                        if rlabel > 0 and rlabel <= len(intermediate_feats):
                            feat = intermediate_feats[rlabel - 1][i]  # [embed_dim]
                        else:
                            feat = torch.stack([f[i] for f in intermediate_feats]).mean(0)
                        all_region_feats.append(feat.cpu())
            
            # 3. 构建灵敏度向量 + 节点标签 + 区域掩码
            sensitivity_vectors = []
            node_labels = []
            region_masks = []
            
            for idx in anomaly_indices:
                scenario = scenarios[idx]
                anomaly_node_idx = scenario.get('anomaly_node_idx', -1)
                if anomaly_node_idx < 0:
                    continue
                
                sens_vector = torch.zeros(n_nodes)
                for j, name in enumerate(node_names):
                    if name in scenario['global_weights']:
                        sens_vector[j] = scenario['global_weights'][name]
                sensitivity_vectors.append(sens_vector)
                node_labels.append(anomaly_node_idx)
                
                # 构建区域掩码: 只允许异常区域内的节点
                pl = scenario.get('partition_labels', None)
                rlabel = scenario.get('region_label', 0)
                if pl is not None and rlabel > 0:
                    # region_label是1-indexed, partition_labels是0-indexed
                    region_idx = rlabel - 1
                    mask = torch.tensor([pl[j] == region_idx for j in range(n_nodes)], dtype=torch.bool)
                else:
                    # fallback: 不约束（所有节点都可选）
                    mask = torch.ones(n_nodes, dtype=torch.bool)
                region_masks.append(mask)
            
            if not all_region_feats:
                return None, None, None, None
            
            region_features = torch.stack(all_region_feats)
            sensitivity_vectors = torch.stack(sensitivity_vectors)
            node_labels = torch.tensor(node_labels, dtype=torch.long)
            region_masks_tensor = torch.stack(region_masks)
            
            logger.info(f"快速特征提取完成: {region_features.shape[0]} 个样本")
            return region_features, sensitivity_vectors, node_labels, region_masks_tensor
            
        except Exception as e:
            logger.error(f"快速特征提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None
    
    def calculate_node_accuracy(self, preds: List[int], labels: List[int],
                                 node_names: List[str], 
                                 network_graph: nx.Graph) -> Tuple[float, float]:
        """
        计算节点级准确率（精确 + 邻居容差±1 hop）
        
        Args:
            preds: 预测的节点索引列表
            labels: 真实的节点索引列表  
            node_names: 节点名称列表
            network_graph: 网络图（用于查找邻居）
            
        Returns:
            Tuple[float, float]: (精确准确率, 邻居容差准确率)
        """
        if not preds or not labels:
            return 0.0, 0.0
        
        exact_correct = 0
        hop_correct = 0
        total = len(preds)
        
        for pred_idx, true_idx in zip(preds, labels):
            # 精确匹配
            if pred_idx == true_idx:
                exact_correct += 1
                hop_correct += 1
                continue
            
            # 邻居容差：检查预测节点是否为真实节点的±1 hop邻居
            try:
                true_node_name = node_names[true_idx]
                pred_node_name = node_names[pred_idx]
                
                if true_node_name in network_graph and pred_node_name in network_graph:
                    neighbors = set(network_graph.neighbors(true_node_name))
                    if pred_node_name in neighbors:
                        hop_correct += 1
            except (IndexError, KeyError):
                pass
        
        exact_acc = exact_correct / total
        hop_acc = hop_correct / total
        
        return exact_acc, hop_acc
    
    def _save_node_localizer(self, filename: str):
        """保存NodeLocalizer模型"""
        try:
            checkpoint_dir = os.path.join(self.config['data']['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            filepath = os.path.join(checkpoint_dir, filename)
            torch.save({
                'model_state_dict': self.node_localizer.state_dict(),
                'n_nodes': self.node_localizer.n_nodes,
                'region_dim': self.node_localizer.region_dim,
                'hidden_dim': self.node_localizer.hidden_dim
            }, filepath)
            logger.info(f"NodeLocalizer已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存NodeLocalizer失败: {e}")

    def save_checkpoint(self, filename: str) -> bool:
        """
        保存检查点

        Args:
            filename: 文件名

        Returns:
            bool: 保存是否成功
        """
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'ltfm_model_state_dict': self.ltfm_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'train_history': self.train_history,
                'config': self.config
            }

            # 使用配置中的输出目录
            output_dir = self.config['data'].get('output_dir', 'output')
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"检查点已保存: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            return False

    def load_checkpoint(self, filename: str) -> bool:
        """
        加载检查点

        Args:
            filename: 文件名

        Returns:
            bool: 加载是否成功
        """
        try:
            # 使用配置中的输出目录
            output_dir = self.config['data'].get('output_dir', 'output')
            checkpoint_path = os.path.join(output_dir, 'checkpoints', filename)

            if not os.path.exists(checkpoint_path):
                logger.error(f"检查点文件不存在: {checkpoint_path}")
                return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.ltfm_model.load_state_dict(checkpoint['ltfm_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_history = checkpoint['train_history']

            logger.info(f"检查点已加载: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False
