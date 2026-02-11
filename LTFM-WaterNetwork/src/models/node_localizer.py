# -*- coding: utf-8 -*-
"""
NodeLocalizer - 节点级漏损定位深度学习模型
利用LTFM区域特征 + 灵敏度向量，预测漏损具体节点

Two-stage approach:
  Stage 1: 训练LTFM (全局/区域检测，已完成，参数冻结)
  Stage 2: 训练NodeLocalizer (节点级定位)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional






class NodeLocalizer(nn.Module):
    """
    物理感知多头注意力定位器 (Physics-Informed Multi-Head Attention Localizer)
    
    核心思想: 
    将漏损定位视为"多视角查询-匹配"问题。
    1. Multi-Head Attention: 多视角匹配 ("经验判断")。
    2. Physics Bias: 物理灵敏度约束 ("理论判断")。
    
    回归: 移除动态门控，使用稳健的静态物理偏置。
    """
    
    def __init__(self, region_dim: int, n_nodes: int, hidden_dim: int = 128,
                 dropout: float = 0.3, num_heads: int = 4):
        super(NodeLocalizer, self).__init__()
        
        self.region_dim = region_dim
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 1. 节点嵌入 (Node Bank)
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.02)
        
        # 2. 物理融合层 (Physics Fusion)
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. 查询映射层 (Query Projection)
        self.query_proj = nn.Sequential(
            nn.Linear(region_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. 多头注意力投影
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        
        # 5. 物理偏置系数 (可学习)
        self.physics_bias_weight = nn.Parameter(torch.tensor(12.0))
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, region_feature: torch.Tensor,
                sensitivity_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_feature: [batch, region_dim]
            sensitivity_vector: [batch, n_nodes]
        Returns:
            logits: [batch, n_nodes]
        """
        batch_size = region_feature.size(0)
        
        # A. 构建 Keys
        expanded_nodes = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_sens = sensitivity_vector.unsqueeze(-1)
        raw_keys = torch.cat([expanded_nodes, expanded_sens], dim=-1)
        keys = self.key_proj(raw_keys) 
        
        # B. 构建 Query
        query = self.query_proj(region_feature).unsqueeze(1)
        
        # C. 多头注意力 (Attention)
        Q = self.W_q(query).view(batch_size, 1, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3) 
        
        K = self.W_k(keys).view(batch_size, self.n_nodes, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 3, 1) 
        
        attn_scores = torch.matmul(Q, K) * self.scale
        attn_logits = attn_scores.mean(dim=1).squeeze(1) 
        
        # D. 物理偏置 (Physics Bias)
        physics_bias = torch.log1p(sensitivity_vector) * self.physics_bias_weight
        
        # 直接相加 (Robust Baseline)
        final_logits = attn_logits + physics_bias
        
        return final_logits
