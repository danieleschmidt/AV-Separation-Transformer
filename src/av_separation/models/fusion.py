import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class DynamicTimeWarping(nn.Module):
    def __init__(self, feature_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Linear(feature_dim, feature_dim)
        
    def compute_similarity_matrix(self, audio_features, video_features):
        audio_proj = self.projection(audio_features)
        video_proj = self.projection(video_features)
        
        audio_norm = F.normalize(audio_proj, dim=-1)
        video_norm = F.normalize(video_proj, dim=-1)
        
        similarity = torch.matmul(audio_norm, video_norm.transpose(-2, -1))
        similarity = similarity / self.temperature
        
        return similarity
    
    def forward(self, audio_features, video_features):
        batch_size = audio_features.size(0)
        audio_len = audio_features.size(1)
        video_len = video_features.size(1)
        
        similarity = self.compute_similarity_matrix(audio_features, video_features)
        
        dp = torch.full((batch_size, audio_len + 1, video_len + 1), 
                       float('-inf')).to(audio_features.device)
        dp[:, 0, 0] = 0
        
        for i in range(1, audio_len + 1):
            for j in range(1, video_len + 1):
                cost = similarity[:, i-1, j-1]
                
                dp[:, i, j] = cost + torch.max(
                    torch.stack([
                        dp[:, i-1, j],
                        dp[:, i, j-1],
                        dp[:, i-1, j-1]
                    ], dim=0),
                    dim=0
                )[0]
        
        alignment_score = dp[:, audio_len, video_len]
        
        path_prob = F.softmax(similarity, dim=-1)
        aligned_video = torch.matmul(path_prob, video_features)
        
        return aligned_video, alignment_score


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class CrossModalFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.model.fusion_dim
        
        self.audio_to_video_attn = CrossModalAttention(
            dim, config.model.fusion_heads, config.model.dropout
        )
        self.video_to_audio_attn = CrossModalAttention(
            dim, config.model.fusion_heads, config.model.dropout
        )
        
        self.audio_self_attn = nn.MultiheadAttention(
            dim, config.model.fusion_heads, 
            dropout=config.model.dropout, batch_first=True
        )
        self.video_self_attn = nn.MultiheadAttention(
            dim, config.model.fusion_heads,
            dropout=config.model.dropout, batch_first=True
        )
        
        self.audio_norm1 = nn.LayerNorm(dim)
        self.audio_norm2 = nn.LayerNorm(dim)
        self.audio_norm3 = nn.LayerNorm(dim)
        
        self.video_norm1 = nn.LayerNorm(dim)
        self.video_norm2 = nn.LayerNorm(dim)
        self.video_norm3 = nn.LayerNorm(dim)
        
        self.audio_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(config.model.dropout)
        )
        
        self.video_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(config.model.dropout)
        )
        
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        audio_cross, _ = self.audio_to_video_attn(audio_features, video_features, video_mask)
        audio_features = self.audio_norm1(audio_features + audio_cross)
        
        video_cross, _ = self.video_to_audio_attn(video_features, audio_features, audio_mask)
        video_features = self.video_norm1(video_features + video_cross)
        
        audio_self, _ = self.audio_self_attn(
            audio_features, audio_features, audio_features,
            key_padding_mask=audio_mask
        )
        audio_features = self.audio_norm2(audio_features + audio_self)
        
        video_self, _ = self.video_self_attn(
            video_features, video_features, video_features,
            key_padding_mask=video_mask
        )
        video_features = self.video_norm2(video_features + video_self)
        
        audio_features = self.audio_norm3(audio_features + self.audio_ffn(audio_features))
        video_features = self.video_norm3(video_features + self.video_ffn(video_features))
        
        return audio_features, video_features


class CrossModalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.audio_projection = nn.Linear(
            config.model.audio_encoder_dim,
            config.model.fusion_dim
        )
        self.video_projection = nn.Linear(
            config.model.video_encoder_dim,
            config.model.fusion_dim
        )
        
        self.dtw = DynamicTimeWarping(config.model.fusion_dim)
        
        self.modality_embedding = nn.Parameter(
            torch.randn(2, 1, 1, config.model.fusion_dim) * 0.02
        )
        
        self.fusion_layers = nn.ModuleList([
            CrossModalFusionLayer(config)
            for _ in range(config.model.fusion_layers)
        ])
        
        self.output_projection = nn.Linear(
            config.model.fusion_dim * 2,
            config.model.decoder_dim
        )
        
        self.layer_norm = nn.LayerNorm(config.model.decoder_dim)
        
    def forward(self, audio_features, video_features, audio_mask=None, video_mask=None):
        audio_features = self.audio_projection(audio_features)
        video_features = self.video_projection(video_features)
        
        audio_features = audio_features + self.modality_embedding[0]
        video_features = video_features + self.modality_embedding[1]
        
        aligned_video, alignment_score = self.dtw(audio_features, video_features)
        
        # Handle sequence length mismatch by interpolating
        if video_features.size(1) != aligned_video.size(1):
            aligned_video = F.interpolate(
                aligned_video.transpose(1, 2),
                size=video_features.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        video_features = 0.5 * video_features + 0.5 * aligned_video
        
        # Align sequence lengths before fusion layers
        audio_len, video_len = audio_features.size(1), video_features.size(1)
        target_len = min(audio_len, video_len)
        
        if audio_len != target_len:
            audio_features = F.interpolate(
                audio_features.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        if video_len != target_len:
            video_features = F.interpolate(
                video_features.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        for layer in self.fusion_layers:
            audio_features, video_features = layer(
                audio_features, video_features,
                audio_mask, video_mask
            )
        
        fused_features = torch.cat([audio_features, video_features], dim=-1)
        fused_features = self.output_projection(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features, alignment_score