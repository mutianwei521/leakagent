# -*- coding: utf-8 -*-
"""
LTFM模型核心模块
实现Adaptor层、LTFM层、VT SELECTOR层的完整架构
基于"Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import math


class AdaptorLayer(nn.Module):
    """增强的适配器层，将Graph2Vec嵌入适配到LTFM输入格式"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """
        初始化适配器层

        Args:
            input_dim: 输入维度（Graph2Vec嵌入维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout率
        """
        super(AdaptorLayer, self).__init__()

        # 确保参数类型正确
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        dropout = float(dropout)

        # 输入投影 - 增强版
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多层感知机块 - 增加深度
        self.mlp_blocks = nn.ModuleList([
            self._make_mlp_block(hidden_dim, dropout) for _ in range(3)
        ])

        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 输出投影 - 增强版
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 残差连接的投影层（当输入输出维度不同时）
        if input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _make_mlp_block(self, dim: int, dropout: float) -> nn.Module:
        """创建MLP块"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        增强的前向传播

        Args:
            x: 输入张量 [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: 适配后的特征 [batch_size, output_dim] 或 [batch_size, seq_len, output_dim]
        """
        # 保存原始输入用于残差连接
        residual_input = self.residual_projection(x)

        # 输入投影
        x = self.input_projection(x)

        # 多层MLP处理
        for mlp_block in self.mlp_blocks:
            residual = x
            x = mlp_block(x)
            x = x + residual  # 残差连接

        # 特征增强
        enhanced = self.feature_enhancer(x)
        x = x + enhanced  # 残差连接

        # 输出投影
        x = self.output_projection(x)

        # 全局残差连接（如果维度匹配）
        if x.shape == residual_input.shape:
            x = x + residual_input

        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        # 确保参数类型正确
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        dropout = float(dropout)

        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（支持cross-attention，query和key/value可以有不同的seq_len）
        
        Args:
            query: 查询张量 [batch_size, q_len, embed_dim]
            key: 键张量 [batch_size, kv_len, embed_dim]
            value: 值张量 [batch_size, kv_len, embed_dim]
            mask: 注意力掩码
            
        Returns:
            torch.Tensor: 注意力输出 [batch_size, q_len, embed_dim]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]
        
        # 线性投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class LTFMLayer(nn.Module):
    """LTFM层：Look-Twice Feature Matching层"""
    
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        """
        初始化LTFM层

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏维度
            dropout: Dropout率
        """
        super(LTFMLayer, self).__init__()

        # 确保参数类型正确
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        
        # 第一次观察：全局注意力
        self.global_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        
        # 第二次观察：局部注意力
        self.local_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.local_norm1 = nn.LayerNorm(embed_dim)
        self.local_norm2 = nn.LayerNorm(embed_dim)
        
        # 特征匹配 - 使用自定义MultiHeadAttention避免PyTorch 2.7兼容性问题
        self.feature_matching = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.matching_norm = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # 融合层
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, global_features: torch.Tensor, 
                local_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            global_features: 全局特征 [batch_size, seq_len, embed_dim]
            local_features: 局部特征 [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: LTFM输出 [batch_size, seq_len, embed_dim]
        """
        # 第一次观察：全局特征处理
        global_attn = self.global_attention(global_features, global_features, global_features)
        global_features = self.global_norm1(global_features + global_attn)
        
        global_ffn = self.ffn(global_features)
        global_features = self.global_norm2(global_features + global_ffn)
        
        # 第二次观察：局部特征处理
        local_attn = self.local_attention(local_features, local_features, local_features)
        local_features = self.local_norm1(local_features + local_attn)
        
        local_ffn = self.ffn(local_features)
        local_features = self.local_norm2(local_features + local_ffn)
        
        # 特征匹配：全局和局部特征交互
        matched_features = self.feature_matching(
            global_features, local_features, local_features
        )
        matched_features = self.matching_norm(global_features + matched_features)
        
        # 特征融合
        fused_features = torch.cat([global_features, local_features], dim=-1)
        fused_features = self.fusion(fused_features)
        fused_features = self.fusion_norm(fused_features)
        
        # 残差连接
        output = matched_features + fused_features
        
        return output


class VTSelectorLayer(nn.Module):
    """改进的VT Selector层：视觉-文本选择器层"""

    def __init__(self, embed_dim: int, num_regions: int, dropout: float = 0.1, num_heads: int = 8):
        """
        初始化VT Selector层

        Args:
            embed_dim: 嵌入维度
            num_regions: 区域数量
            dropout: Dropout率
            num_heads: 多头注意力头数
        """
        super(VTSelectorLayer, self).__init__()

        # 确保参数类型正确
        embed_dim = int(embed_dim)
        num_regions = int(num_regions)
        dropout = float(dropout)
        num_heads = int(num_heads)

        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.num_heads = num_heads

        # 全局特征处理 - 增强版（使用LayerNorm替代BatchNorm以支持batch_size=1）
        self.global_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)  # 较小的dropout
        )

        # 区域特征处理 - 增强版（使用LayerNorm替代BatchNorm以支持batch_size=1）
        self.region_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)  # 较小的dropout
        )

        # 多头注意力机制 - 使用自定义MultiHeadAttention避免PyTorch 2.7兼容性问题
        self.multihead_attention = MultiHeadAttention(
            embed_dim, num_heads, dropout
        )

        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, num_regions + 1, embed_dim))

        # 交叉注意力 - 使用自定义MultiHeadAttention
        self.cross_attention = MultiHeadAttention(
            embed_dim, num_heads, dropout
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 简化的分类头 - 全局（使用LayerNorm替代BatchNorm以支持batch_size=1）
        self.global_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        # 简化的分类头 - 区域（共享权重但有区域特定的偏置）
        self.shared_region_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 区域特定的最终分类层 - 简化版
        self.region_final_layers = nn.ModuleList([
            nn.Linear(embed_dim // 2, 1) for _ in range(num_regions)
        ])

        # 输出缩放参数 - 让模型输出更有区分度
        self.global_output_scale = nn.Parameter(torch.tensor(2.0))
        self.region_output_scale = nn.Parameter(torch.tensor(3.0))  # 区域输出更大的缩放

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 输入维度适配器（处理增强特征的维度变化）
        self.input_adapter = nn.Linear(embed_dim, embed_dim)  # 这会在运行时动态调整

        # 改进的参数初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)  # 小的正偏置
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        # 特别初始化分类层 - 让输出有更大的初始方差
        for classifier in self.region_final_layers:
            if hasattr(classifier, '0'):  # Sequential中的Linear层
                linear_layer = classifier[0]
                nn.init.normal_(linear_layer.weight, mean=0.0, std=0.5)
                nn.init.uniform_(linear_layer.bias, -0.5, 0.5)

        # 全局分类器初始化
        for module in self.global_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.3)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.3, 0.3)
        
    def forward(self, global_features: torch.Tensor,
                region_features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        改进的前向传播

        Args:
            global_features: 全局特征 [batch_size, embed_dim]
            region_features: 区域特征列表，每个元素为 [batch_size, embed_dim]

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (全局得分, 区域得分列表)
        """
        batch_size = global_features.shape[0]

        # 1. 处理全局特征
        processed_global = self.global_processor(global_features)

        # 2. 处理区域特征
        processed_regions = []
        for region_feat in region_features:
            processed_region = self.region_processor(region_feat)
            processed_regions.append(processed_region)

        # 3. 构建序列：[全局特征, 区域特征1, 区域特征2, ...]
        if processed_regions:
            # 堆叠所有特征
            all_features = torch.stack([processed_global] + processed_regions, dim=1)  # [batch_size, num_regions+1, embed_dim]

            # 添加位置编码
            seq_len = all_features.shape[1]
            pos_encoding = self.position_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
            all_features = all_features + pos_encoding

            # 4. 自注意力机制 - 让特征之间相互交互
            attended_features = self.multihead_attention(
                all_features, all_features, all_features
            )
            attended_features = self.layer_norm(attended_features + all_features)  # 残差连接

            # 5. 交叉注意力 - 全局特征作为query，区域特征作为key和value
            global_query = attended_features[:, 0:1, :]  # [batch_size, 1, embed_dim]
            region_kv = attended_features[:, 1:, :]      # [batch_size, num_regions, embed_dim]

            if region_kv.shape[1] > 0:
                cross_attended = self.cross_attention(
                    global_query, region_kv, region_kv
                )

                # 融合全局和交叉注意力特征
                enhanced_global = torch.cat([
                    attended_features[:, 0, :],  # 原始全局特征
                    cross_attended.squeeze(1)    # 交叉注意力特征
                ], dim=-1)
                enhanced_global = self.feature_fusion(enhanced_global)
            else:
                enhanced_global = attended_features[:, 0, :]

            # 6. 增强区域特征
            enhanced_regions = []
            for i in range(len(processed_regions)):
                if i < attended_features.shape[1] - 1:
                    # 结合原始区域特征和注意力增强特征
                    region_feat = attended_features[:, i + 1, :]

                    # 添加全局上下文
                    global_context = enhanced_global.unsqueeze(1).expand(-1, 1, -1)
                    region_with_context = torch.cat([region_feat.unsqueeze(1), global_context], dim=-1)
                    region_with_context = self.feature_fusion(region_with_context.squeeze(1))

                    enhanced_regions.append(region_with_context)
        else:
            enhanced_global = processed_global
            enhanced_regions = []

        # 7. 分类预测 - 应用输出缩放
        global_score = self.global_classifier(enhanced_global)
        global_score = global_score * self.global_output_scale  # 缩放全局输出

        # 8. 区域分类 - 使用共享分类器 + 区域特定层
        region_scores = []
        for i, enhanced_region in enumerate(enhanced_regions):
            if i < len(self.region_final_layers):
                # 共享特征提取
                shared_features = self.shared_region_classifier(enhanced_region)
                shared_features = self.dropout(shared_features)

                # 区域特定分类 - 应用输出缩放
                region_score = self.region_final_layers[i](shared_features)
                region_score = region_score * self.region_output_scale  # 缩放区域输出
                region_scores.append(region_score)

        return global_score, region_scores


class LTFMModel(nn.Module):
    """完整的LTFM模型"""

    def __init__(self, graph2vec_dim: int = 128, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, num_regions: int = 5,
                 hidden_dim: int = 512, dropout: float = 0.1):
        """
        初始化LTFM模型

        Args:
            graph2vec_dim: Graph2Vec嵌入维度
            embed_dim: 模型嵌入维度
            num_heads: 注意力头数
            num_layers: LTFM层数
            num_regions: 区域数量
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super(LTFMModel, self).__init__()

        # 确保参数类型正确
        graph2vec_dim = int(graph2vec_dim)
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        num_regions = int(num_regions)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)

        self.embed_dim = embed_dim
        self.num_regions = num_regions

        # Adaptor层
        self.global_adaptor = AdaptorLayer(graph2vec_dim, hidden_dim, embed_dim, dropout)
        self.region_adaptors = nn.ModuleList([
            AdaptorLayer(graph2vec_dim, hidden_dim, embed_dim, dropout)
            for _ in range(num_regions)
        ])

        # LTFM层
        self.ltfm_layers = nn.ModuleList([
            LTFMLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # VT Selector层
        self.vt_selector = VTSelectorLayer(embed_dim, num_regions, dropout, num_heads)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, global_graph_embedding: torch.Tensor,
                region_graph_embeddings: List[torch.Tensor],
                return_features: bool = False):
        """
        前向传播

        Args:
            global_graph_embedding: 全局图嵌入 [batch_size, graph2vec_dim]
            region_graph_embeddings: 区域图嵌入列表，每个元素为 [batch_size, graph2vec_dim]
            return_features: 是否返回中间特征（供NodeLocalizer使用）

        Returns:
            如果 return_features=False: (全局异常得分, 区域异常得分列表)
            如果 return_features=True: (全局异常得分, 区域异常得分列表, 区域特征列表)
        """
        batch_size = global_graph_embedding.shape[0]

        # Adaptor层处理
        global_features = self.global_adaptor(global_graph_embedding)  # [batch_size, embed_dim]

        region_features = []
        for i, region_embedding in enumerate(region_graph_embeddings):
            if i < len(self.region_adaptors):
                region_feat = self.region_adaptors[i](region_embedding)
                region_features.append(region_feat)

        # 添加序列维度和位置编码
        global_features = global_features.unsqueeze(1) + self.pos_encoding  # [batch_size, 1, embed_dim]
        region_features_seq = []
        for region_feat in region_features:
            region_feat_seq = region_feat.unsqueeze(1) + self.pos_encoding
            region_features_seq.append(region_feat_seq)

        # LTFM层处理
        for ltfm_layer in self.ltfm_layers:
            # 对每个区域特征与全局特征进行LTFM处理
            processed_regions = []
            for region_feat_seq in region_features_seq:
                processed_region = ltfm_layer(global_features, region_feat_seq)
                processed_regions.append(processed_region)

            # 更新全局特征（使用所有区域特征的平均）
            if processed_regions:
                avg_region_features = torch.stack([r.squeeze(1) for r in processed_regions]).mean(0).unsqueeze(1)
                global_features = ltfm_layer(global_features, avg_region_features)

            region_features_seq = processed_regions

        # 移除序列维度
        global_features = global_features.squeeze(1)  # [batch_size, embed_dim]
        region_features = [r.squeeze(1) for r in region_features_seq]  # List of [batch_size, embed_dim]

        # VT Selector层
        global_score, region_scores = self.vt_selector(global_features, region_features)

        if return_features:
            return global_score, region_scores, region_features
        return global_score, region_scores

    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal Loss实现，用于解决类别不平衡问题

        Args:
            inputs: 预测logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
            alpha: 平衡因子
            gamma: 聚焦参数

        Returns:
            torch.Tensor: Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def compute_loss(self, global_score: torch.Tensor, region_scores: List[torch.Tensor],
                    global_labels: torch.Tensor, region_labels: torch.Tensor) -> torch.Tensor:
        """
        计算改进的损失函数

        Args:
            global_score: 全局异常得分 [batch_size, 1]
            region_scores: 区域异常得分列表
            global_labels: 全局标签 [batch_size] (0表示正常，1表示异常)
            region_labels: 区域标签 [batch_size] (0表示正常，>0表示异常区域索引)

        Returns:
            torch.Tensor: 总损失
        """
        # 1. 全局异常检测损失 - 使用Focal Loss for binary classification
        global_labels_binary = (global_labels > 0).float()
        # 使用squeeze(-1)只压缩最后一维，保持batch维度
        global_probs = torch.sigmoid(global_score.squeeze(-1))
        global_bce = F.binary_cross_entropy(global_probs, global_labels_binary, reduction='none')

        # 计算focal weight
        pt = global_labels_binary * global_probs + (1 - global_labels_binary) * (1 - global_probs)
        focal_weight = (1 - pt) ** 2.0  # gamma=2.0
        global_focal_loss = (focal_weight * global_bce).mean()

        # 2. 区域异常定位损失 - 使用加权Focal Loss
        if region_scores:
            # 将区域得分堆叠
            region_scores_tensor = torch.cat(region_scores, dim=1)  # [batch_size, num_regions]

            # 添加正常类别（索引0）的得分
            normal_scores = torch.zeros(region_scores_tensor.shape[0], 1,
                                      device=region_scores_tensor.device)
            all_scores = torch.cat([normal_scores, region_scores_tensor], dim=1)

            # 使用Focal Loss处理区域分类
            region_focal_loss = self.focal_loss(all_scores, region_labels.long(),
                                               alpha=2.0, gamma=2.0)  # 增加区域检测权重

            # 3. 添加区域一致性损失 - 鼓励异常区域有更高的得分
            consistency_loss = 0.0
            if len(region_scores) > 1:
                # 计算异常样本的区域得分方差，鼓励异常区域突出
                anomaly_mask = global_labels > 0
                if anomaly_mask.sum() > 0:
                    anomaly_region_scores = region_scores_tensor[anomaly_mask]
                    if anomaly_region_scores.shape[0] > 0:
                        # 鼓励异常区域得分的差异性（使用正值损失）
                        region_std = torch.std(anomaly_region_scores, dim=1)
                        # 使用 1/(std+epsilon) 来鼓励更大的方差，避免负数
                        epsilon = 1e-6
                        consistency_loss = 1.0 / (region_std.mean() + epsilon)
        else:
            region_focal_loss = torch.tensor(0.0, device=global_score.device)
            consistency_loss = torch.tensor(0.0, device=global_score.device)

        # 4. 添加区域对比损失 - 强制区域特征差异化
        contrastive_loss = 0.0
        if len(region_scores) > 1 and len(region_scores_tensor) > 0:
            # 对于异常样本，强制异常区域得分远高于其他区域
            anomaly_mask = global_labels > 0
            if anomaly_mask.sum() > 0:
                anomaly_region_scores = region_scores_tensor[anomaly_mask]
                anomaly_region_labels = region_labels[anomaly_mask] - 1  # 转换为0-based索引

                if anomaly_region_scores.shape[0] > 0:
                    # 对每个异常样本，其对应区域得分应该比其他区域高
                    for i, (scores, true_region) in enumerate(zip(anomaly_region_scores, anomaly_region_labels)):
                        if 0 <= true_region < len(scores):
                            # 真实异常区域得分
                            true_score = scores[true_region]
                            # 其他区域得分
                            other_scores = torch.cat([scores[:true_region], scores[true_region+1:]])
                            if len(other_scores) > 0:
                                # 对比损失：真实区域得分应该比其他区域高至少margin
                                margin = 2.0
                                max_other_score = torch.max(other_scores)
                                contrastive_loss += torch.relu(max_other_score - true_score + margin)

                contrastive_loss = contrastive_loss / max(1, anomaly_mask.sum())

        # 5. 总损失 - 平衡的权重策略
        total_loss = (
            1.0 * global_focal_loss +      # 全局检测损失
            15.0 * region_focal_loss +     # 区域检测损失（进一步增加权重）
            0.5 * consistency_loss +       # 一致性损失（增加权重）
            8.0 * contrastive_loss         # 对比损失（强制区域差异化）
        )

        return total_loss

    def predict(self, global_graph_embedding: torch.Tensor,
               region_graph_embeddings: List[torch.Tensor],
               threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测异常

        Args:
            global_graph_embedding: 全局图嵌入
            region_graph_embeddings: 区域图嵌入列表
            threshold: 异常检测阈值

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (全局异常预测, 区域异常预测)
        """
        self.eval()
        with torch.no_grad():
            global_score, region_scores = self.forward(global_graph_embedding, region_graph_embeddings)

            # 全局异常预测
            global_pred = (torch.sigmoid(global_score) > threshold).float()

            # 区域异常预测 - 改进版
            if region_scores:
                region_scores_tensor = torch.cat(region_scores, dim=1)
                normal_scores = torch.zeros(region_scores_tensor.shape[0], 1,
                                          device=region_scores_tensor.device)
                all_scores = torch.cat([normal_scores, region_scores_tensor], dim=1)

                # 使用温度缩放平滑预测
                temperature = 2.0  # 温度参数
                scaled_scores = all_scores / temperature

                # 使用softmax获得概率分布
                probs = torch.softmax(scaled_scores, dim=1)

                # 预测：选择概率最高的类别
                region_pred = torch.argmax(probs, dim=1)

                # 添加置信度阈值：如果最高概率太低，预测为正常
                max_probs = torch.max(probs, dim=1)[0]
                confidence_threshold = 0.35  # 置信度阈值
                low_confidence_mask = max_probs < confidence_threshold
                region_pred[low_confidence_mask] = 0  # 低置信度时预测为正常

            else:
                region_pred = torch.zeros(global_pred.shape[0], dtype=torch.long,
                                        device=global_pred.device)

        return global_pred, region_pred
