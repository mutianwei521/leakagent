# -*- coding: utf-8 -*-
"""
LTFM Model Core Module
Implements Adaptor Layer, LTFM Layer, VT SELECTOR Layer architecture
Based on "Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import math


class AdaptorLayer(nn.Module):
    """Enhanced Adaptor Layer, adapting Graph2Vec embedding to LTFM input format"""

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

        # Input projection - Enhanced version
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # MLP blocks - Increase depth
        self.mlp_blocks = nn.ModuleList([
            self._make_mlp_block(hidden_dim, dropout) for _ in range(3)
        ])

        # Feature enhancer module
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Output projection - Enhanced version
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Residual connection projection (when input/output dimensions differ)
        if input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _make_mlp_block(self, dim: int, dropout: float) -> nn.Module:
        """Create MLP block"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward propagation

        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Adapted features [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        # Save original input for residual connection
        residual_input = self.residual_projection(x)

        # Input projection
        x = self.input_projection(x)

        # Multi-layer MLP processing
        for mlp_block in self.mlp_blocks:
            residual = x
            x = mlp_block(x)
            x = x + residual  # Residual connection

        # Feature enhancement
        enhanced = self.feature_enhancer(x)
        x = x + enhanced  # Residual connection

        # Output projection
        x = self.output_projection(x)

        # Global residual connection (if dimensions match)
        if x.shape == residual_input.shape:
            x = x + residual_input

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head Attention Mechanism"""
    
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
        Forward propagation (Supports cross-attention, query and key/value can have different seq_len)
        
        Args:
            query: Query tensor [batch_size, q_len, embed_dim]
            key: Key tensor [batch_size, kv_len, embed_dim]
            value: Value tensor [batch_size, kv_len, embed_dim]
            mask: Attention mask
            
        Returns:
            torch.Tensor: Attention output [batch_size, q_len, embed_dim]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]
        
        # Linear projection
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape to multi-head format
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class LTFMLayer(nn.Module):
    """LTFM Layer: Look-Twice Feature Matching Layer"""
    
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize LTFM Layer

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            hidden_dim: Transformer FFN hidden dimension
            dropout: Dropout rate
        """
        super(LTFMLayer, self).__init__()

        # 确保参数类型正确
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        
        # First Look: Global Attention
        self.global_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        
        # Second Look: Local Attention
        self.local_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.local_norm1 = nn.LayerNorm(embed_dim)
        self.local_norm2 = nn.LayerNorm(embed_dim)
        
        # Feature Matching - Use custom MultiHeadAttention to avoid PyTorch 2.7 compatibility issues
        self.feature_matching = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.matching_norm = nn.LayerNorm(embed_dim)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, global_features: torch.Tensor, 
                local_features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            global_features: Global features [batch_size, seq_len, embed_dim]
            local_features: Local features [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: LTFM output [batch_size, seq_len, embed_dim]
        """
        # First Look: Global feature processing
        global_attn = self.global_attention(global_features, global_features, global_features)
        global_features = self.global_norm1(global_features + global_attn)
        
        global_ffn = self.ffn(global_features)
        global_features = self.global_norm2(global_features + global_ffn)
        
        # Second Look: Local feature processing
        local_attn = self.local_attention(local_features, local_features, local_features)
        local_features = self.local_norm1(local_features + local_attn)
        
        local_ffn = self.ffn(local_features)
        local_features = self.local_norm2(local_features + local_ffn)
        
        # Feature matching: Global and local feature interaction
        matched_features = self.feature_matching(
            global_features, local_features, local_features
        )
        matched_features = self.matching_norm(global_features + matched_features)
        
        # Feature fusion
        fused_features = torch.cat([global_features, local_features], dim=-1)
        fused_features = self.fusion(fused_features)
        fused_features = self.fusion_norm(fused_features)
        
        # Residual connection
        output = matched_features + fused_features
        
        return output


class VTSelectorLayer(nn.Module):
    """Improved VT Selector Layer: Visual-Text Selector Layer"""

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

        # Global feature processing - Enhanced (Use LayerNorm instead of BatchNorm to support batch_size=1)
        self.global_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)  # Smaller dropout
        )

        # Region feature processing - Enhanced (Use LayerNorm instead of BatchNorm to support batch_size=1)
        self.region_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)  # Smaller dropout
        )

        # Multi-head attention - Use custom MultiHeadAttention to avoid PyTorch 2.7 compatibility issues
        self.multihead_attention = MultiHeadAttention(
            embed_dim, num_heads, dropout
        )

        # Position encoding
        self.position_encoding = nn.Parameter(torch.randn(1, num_regions + 1, embed_dim))

        # Cross attention - Use custom MultiHeadAttention
        self.cross_attention = MultiHeadAttention(
            embed_dim, num_heads, dropout
        )

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # Simplified classifier head - Global (Use LayerNorm instead of BatchNorm to support batch_size=1)
        self.global_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        # Simplified classifier head - Region (Shared weights but region-specific bias)
        self.shared_region_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Region-specific final classification layers - Simplified
        self.region_final_layers = nn.ModuleList([
            nn.Linear(embed_dim // 2, 1) for _ in range(num_regions)
        ])

        # Output scaling parameter - Make model output more discriminative
        self.global_output_scale = nn.Parameter(torch.tensor(2.0))
        self.region_output_scale = nn.Parameter(torch.tensor(3.0))  # Larger scale for region output

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Input dimension adapter (Handle dimension change of enhanced features)
        self.input_adapter = nn.Linear(embed_dim, embed_dim)  # This will be dynamically adjusted at runtime

        # Improved parameter initialization
        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)  # Small positive bias
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        # Specifically initialize classifier layers - Allow output to have larger initial variance
        for classifier in self.region_final_layers:
            if hasattr(classifier, '0'):  # Linear layer in Sequential
                linear_layer = classifier[0]
                nn.init.normal_(linear_layer.weight, mean=0.0, std=0.5)
                nn.init.uniform_(linear_layer.bias, -0.5, 0.5)

        # Global classifier initialization
        for module in self.global_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.3)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.3, 0.3)
        
    def forward(self, global_features: torch.Tensor,
                region_features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Improved forward propagation

        Args:
            global_features: Global features [batch_size, embed_dim]
            region_features: List of region features, each element is [batch_size, embed_dim]

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (Global Score, List of Region Scores)
        """
        batch_size = global_features.shape[0]

        # 1. Process global features
        processed_global = self.global_processor(global_features)

        # 2. Process region features
        processed_regions = []
        for region_feat in region_features:
            processed_region = self.region_processor(region_feat)
            processed_regions.append(processed_region)

        # 3. Build sequence: [Global feature, Region feature 1, Region feature 2, ...]
        if processed_regions:
            # Stack all features
            all_features = torch.stack([processed_global] + processed_regions, dim=1)  # [batch_size, num_regions+1, embed_dim]

            # Add position encoding
            seq_len = all_features.shape[1]
            pos_encoding = self.position_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
            all_features = all_features + pos_encoding

            # 4. Self-attention mechanism - Allow features to interact with each other
            attended_features = self.multihead_attention(
                all_features, all_features, all_features
            )
            attended_features = self.layer_norm(attended_features + all_features)  # Residual connection

            # 5. Cross-attention - Global feature as query, region features as key and value
            global_query = attended_features[:, 0:1, :]  # [batch_size, 1, embed_dim]
            region_kv = attended_features[:, 1:, :]      # [batch_size, num_regions, embed_dim]

            if region_kv.shape[1] > 0:
                cross_attended = self.cross_attention(
                    global_query, region_kv, region_kv
                )

                # Fuse global and cross-attention features
                enhanced_global = torch.cat([
                    attended_features[:, 0, :],  # Original global feature
                    cross_attended.squeeze(1)    # Cross-attention feature
                ], dim=-1)
                enhanced_global = self.feature_fusion(enhanced_global)
            else:
                enhanced_global = attended_features[:, 0, :]

            # 6. Enhance region features
            enhanced_regions = []
            for i in range(len(processed_regions)):
                if i < attended_features.shape[1] - 1:
                    # Combine original region feature and attention enhanced feature
                    region_feat = attended_features[:, i + 1, :]

                    # Add global context
                    global_context = enhanced_global.unsqueeze(1).expand(-1, 1, -1)
                    region_with_context = torch.cat([region_feat.unsqueeze(1), global_context], dim=-1)
                    region_with_context = self.feature_fusion(region_with_context.squeeze(1))

                    enhanced_regions.append(region_with_context)
        else:
            enhanced_global = processed_global
            enhanced_regions = []

        # 7. Classification prediction - Apply output scaling
        global_score = self.global_classifier(enhanced_global)
        global_score = global_score * self.global_output_scale  # Scale global output

        # 8. Region classification - Use shared classifier + region-specific layers
        region_scores = []
        for i, enhanced_region in enumerate(enhanced_regions):
            if i < len(self.region_final_layers):
                # Shared feature extraction
                shared_features = self.shared_region_classifier(enhanced_region)
                shared_features = self.dropout(shared_features)

                # Region-specific classification - Apply output scaling
                region_score = self.region_final_layers[i](shared_features)
                region_score = region_score * self.region_output_scale  # Scale region output
                region_scores.append(region_score)

        return global_score, region_scores


class LTFMModel(nn.Module):
    """Complete LTFM Model"""

    def __init__(self, graph2vec_dim: int = 128, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, num_regions: int = 5,
                 hidden_dim: int = 512, dropout: float = 0.1):
        """
        Initialize LTFM Model

        Args:
            graph2vec_dim: Graph2Vec embedding dimension
            embed_dim: Model embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of LTFM layers
            num_regions: Number of regions
            hidden_dim: Hidden dimension
            dropout: Dropout rate
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

        # Adaptor Layer
        self.global_adaptor = AdaptorLayer(graph2vec_dim, hidden_dim, embed_dim, dropout)
        self.region_adaptors = nn.ModuleList([
            AdaptorLayer(graph2vec_dim, hidden_dim, embed_dim, dropout)
            for _ in range(num_regions)
        ])

        # LTFM Layer
        self.ltfm_layers = nn.ModuleList([
            LTFMLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # VT Selector Layer
        self.vt_selector = VTSelectorLayer(embed_dim, num_regions, dropout, num_heads)

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, global_graph_embedding: torch.Tensor,
                region_graph_embeddings: List[torch.Tensor],
                return_features: bool = False):
        """
        Forward propagation

        Args:
            global_graph_embedding: Global graph embedding [batch_size, graph2vec_dim]
            region_graph_embeddings: List of region graph embeddings, each element is [batch_size, graph2vec_dim]
            return_features: Whether to return intermediate features (for NodeLocalizer use)

        Returns:
            If return_features=False: (Global anomaly score, List of region anomaly scores)
            If return_features=True: (Global anomaly score, List of region anomaly scores, List of region features)
        """
        batch_size = global_graph_embedding.shape[0]

        # Adaptor Layer processing
        global_features = self.global_adaptor(global_graph_embedding)  # [batch_size, embed_dim]

        region_features = []
        for i, region_embedding in enumerate(region_graph_embeddings):
            if i < len(self.region_adaptors):
                region_feat = self.region_adaptors[i](region_embedding)
                region_features.append(region_feat)

        # Add sequence dimension and position encoding
        global_features = global_features.unsqueeze(1) + self.pos_encoding  # [batch_size, 1, embed_dim]
        region_features_seq = []
        for region_feat in region_features:
            region_feat_seq = region_feat.unsqueeze(1) + self.pos_encoding
            region_features_seq.append(region_feat_seq)

        # LTFM Layer processing
        for ltfm_layer in self.ltfm_layers:
            # Apply LTFM processing to each region feature and global feature
            processed_regions = []
            for region_feat_seq in region_features_seq:
                processed_region = ltfm_layer(global_features, region_feat_seq)
                processed_regions.append(processed_region)

            # Update global features (use average of all region features)
            if processed_regions:
                avg_region_features = torch.stack([r.squeeze(1) for r in processed_regions]).mean(0).unsqueeze(1)
                global_features = ltfm_layer(global_features, avg_region_features)

            region_features_seq = processed_regions

        # Remove sequence dimension
        global_features = global_features.squeeze(1)  # [batch_size, embed_dim]
        region_features = [r.squeeze(1) for r in region_features_seq]  # List of [batch_size, embed_dim]

        # VT Selector Layer
        global_score, region_scores = self.vt_selector(global_features, region_features)

        if return_features:
            return global_score, region_scores, region_features
        return global_score, region_scores

    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal Loss implementation, used to address class imbalance

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: True labels [batch_size]
            alpha: Balancing factor
            gamma: Focusing parameter

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
        Calculate improved loss function

        Args:
            global_score: Global anomaly score [batch_size, 1]
            region_scores: List of region anomaly scores
            global_labels: Global labels [batch_size] (0 for normal, 1 for anomaly)
            region_labels: Region labels [batch_size] (0 for normal, >0 for anomaly region index)

        Returns:
            torch.Tensor: Total loss
        """
        # 1. Global anomaly detection loss - Use Focal Loss for binary classification
        global_labels_binary = (global_labels > 0).float()
        # Use squeeze(-1) to compress only the last dimension, keeping batch dimension
        global_probs = torch.sigmoid(global_score.squeeze(-1))
        global_bce = F.binary_cross_entropy(global_probs, global_labels_binary, reduction='none')

        # Calculate focal weight
        pt = global_labels_binary * global_probs + (1 - global_labels_binary) * (1 - global_probs)
        focal_weight = (1 - pt) ** 2.0  # gamma=2.0
        global_focal_loss = (focal_weight * global_bce).mean()

        # 2. Region anomaly localization loss - Use Weighted Focal Loss
        if region_scores:
            # Stack region scores
            region_scores_tensor = torch.cat(region_scores, dim=1)  # [batch_size, num_regions]

            # Add score for normal class (index 0)
            normal_scores = torch.zeros(region_scores_tensor.shape[0], 1,
                                      device=region_scores_tensor.device)
            all_scores = torch.cat([normal_scores, region_scores_tensor], dim=1)

            # Use Focal Loss for region classification
            region_focal_loss = self.focal_loss(all_scores, region_labels.long(),
                                               alpha=2.0, gamma=2.0)  # Increase region detection weight

            # 3. Add region consistency loss - Encourage higher scores for anomalous regions
            consistency_loss = 0.0
            if len(region_scores) > 1:
                # Calculate variance of region scores for anomalous samples, encourage anomalous regions to stand out
                anomaly_mask = global_labels > 0
                if anomaly_mask.sum() > 0:
                    anomaly_region_scores = region_scores_tensor[anomaly_mask]
                    if anomaly_region_scores.shape[0] > 0:
                        # Encourage diversity in region scores (use positive value loss)
                        region_std = torch.std(anomaly_region_scores, dim=1)
                        # Use 1/(std+epsilon) to encourage larger variance, avoid negative numbers
                        epsilon = 1e-6
                        consistency_loss = 1.0 / (region_std.mean() + epsilon)
        else:
            region_focal_loss = torch.tensor(0.0, device=global_score.device)
            consistency_loss = torch.tensor(0.0, device=global_score.device)

        # 4. Add region contrastive loss - Force region feature differentiation
        contrastive_loss = 0.0
        if len(region_scores) > 1 and len(region_scores_tensor) > 0:
            # For anomalous samples, force anomalous region score to be much higher than other regions
            anomaly_mask = global_labels > 0
            if anomaly_mask.sum() > 0:
                anomaly_region_scores = region_scores_tensor[anomaly_mask]
                anomaly_region_labels = region_labels[anomaly_mask] - 1  # Convert to 0-based index

                if anomaly_region_scores.shape[0] > 0:
                    # For each anomalous sample, its corresponding region score should be higher than other regions
                    for i, (scores, true_region) in enumerate(zip(anomaly_region_scores, anomaly_region_labels)):
                        if 0 <= true_region < len(scores):
                            # True anomaly region score
                            true_score = scores[true_region]
                            # Other region scores
                            other_scores = torch.cat([scores[:true_region], scores[true_region+1:]])
                            if len(other_scores) > 0:
                                # Contrastive loss: True region score should be higher than others by at least margin
                                margin = 2.0
                                max_other_score = torch.max(other_scores)
                                contrastive_loss += torch.relu(max_other_score - true_score + margin)

                contrastive_loss = contrastive_loss / max(1, anomaly_mask.sum())

        # 5. Total loss - Balanced weight strategy
        total_loss = (
            1.0 * global_focal_loss +      # Global detection loss
            15.0 * region_focal_loss +     # Region detection loss (Further increased weight)
            0.5 * consistency_loss +       # Consistency loss (Increased weight)
            8.0 * contrastive_loss         # Contrastive loss (Force region differentiation)
        )

        return total_loss

    def predict(self, global_graph_embedding: torch.Tensor,
               region_graph_embeddings: List[torch.Tensor],
               threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly

        Args:
            global_graph_embedding: Global graph embedding
            region_graph_embeddings: List of region graph embeddings
            threshold: Anomaly detection threshold

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Global anomaly prediction, Region anomaly prediction)
        """
        self.eval()
        with torch.no_grad():
            global_score, region_scores = self.forward(global_graph_embedding, region_graph_embeddings)

            # Global anomaly prediction
            global_pred = (torch.sigmoid(global_score) > threshold).float()

            # Region anomaly prediction - Improved version
            if region_scores:
                region_scores_tensor = torch.cat(region_scores, dim=1)
                normal_scores = torch.zeros(region_scores_tensor.shape[0], 1,
                                          device=region_scores_tensor.device)
                all_scores = torch.cat([normal_scores, region_scores_tensor], dim=1)

                # Use temperature scaling to smooth predictions
                temperature = 2.0  # Temperature parameter
                scaled_scores = all_scores / temperature

                # Use softmax to get probability distribution
                probs = torch.softmax(scaled_scores, dim=1)

                # Predict: Select class with highest probability
                region_pred = torch.argmax(probs, dim=1)

                # Add confidence threshold: If highest probability is too low, predict as normal
                max_probs = torch.max(probs, dim=1)[0]
                confidence_threshold = 0.35  # Confidence threshold
                low_confidence_mask = max_probs < confidence_threshold
                region_pred[low_confidence_mask] = 0  # Predict as normal when confidence is low

            else:
                region_pred = torch.zeros(global_pred.shape[0], dtype=torch.long,
                                        device=global_pred.device)

        return global_pred, region_pred
