# -*- coding: utf-8 -*-
"""
NodeLocalizer - Node-level Leakage Localization Deep Learning Model
Uses LTFM regional features + sensitivity vectors to predict specific leakage nodes

Two-stage approach:
  Stage 1: Train LTFM (Global/Regional detection, completed, parameters frozen)
  Stage 2: Train NodeLocalizer (Node-level localization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional






class NodeLocalizer(nn.Module):
    """
    Physics-Informed Multi-Head Attention Localizer
    
    Core Idea:
    Treat leakage localization as a "Multi-view Query-Matching" problem.
    1. Multi-Head Attention: Multi-view matching ("Empirical Judgment").
    2. Physics Bias: Physical sensitivity constraints ("Theoretical Judgment").
    
    Regression: Remove dynamic gating, use robust static physical bias.
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
        
        # 1. Node Embedding (Node Bank)
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.02)
        
        # 2. Physics Fusion Layer (Physics Fusion)
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Query Projection Layer (Query Projection)
        self.query_proj = nn.Sequential(
            nn.Linear(region_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Multi-head Attention Projection
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        
        # 5. Physics Bias Coefficient (Learnable)
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
        
        # A. Build Keys
        expanded_nodes = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        expanded_sens = sensitivity_vector.unsqueeze(-1)
        raw_keys = torch.cat([expanded_nodes, expanded_sens], dim=-1)
        keys = self.key_proj(raw_keys) 
        
        # B. Build Query
        query = self.query_proj(region_feature).unsqueeze(1)
        
        # C. Multi-head Attention
        Q = self.W_q(query).view(batch_size, 1, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3) 
        
        K = self.W_k(keys).view(batch_size, self.n_nodes, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 3, 1) 
        
        attn_scores = torch.matmul(Q, K) * self.scale
        attn_logits = attn_scores.mean(dim=1).squeeze(1) 
        
        # D. Physics Bias
        physics_bias = torch.log1p(sensitivity_vector) * self.physics_bias_weight
        
        # Direct addition (Robust Baseline)
        final_logits = attn_logits + physics_bias
        
        return final_logits
