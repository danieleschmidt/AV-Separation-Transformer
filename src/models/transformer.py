"""Multi-modal transformer model for audio-visual speech separation."""

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention block
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class CrossModalAttention(nn.Module):
    """Cross-modal attention between audio and video features."""
    
    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        d_model: int,
        nhead: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Project inputs to common dimension
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        
        # Cross-attention layers
        self.audio_to_video_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.video_to_audio_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.audio_norm = LayerNorm(d_model)
        self.video_norm = LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)  # [B, T, d_model]
        video_proj = self.video_proj(video_features)  # [B, T, d_model]
        
        # Cross-modal attention
        # Audio attending to video
        audio_enhanced, audio_to_video_weights = self.audio_to_video_attn(
            audio_proj, video_proj, video_proj
        )
        audio_enhanced = self.audio_norm(audio_proj + audio_enhanced)
        
        # Video attending to audio
        video_enhanced, video_to_audio_weights = self.video_to_audio_attn(
            video_proj, audio_proj, audio_proj
        )
        video_enhanced = self.video_norm(video_proj + video_enhanced)
        
        # Concatenate and project
        fused = torch.cat([audio_enhanced, video_enhanced], dim=-1)
        output = self.output_proj(fused)
        
        attention_weights = {
            'audio_to_video': audio_to_video_weights,
            'video_to_audio': video_to_audio_weights
        }
        
        return output, attention_weights


class SpeakerDecoder(nn.Module):
    """Decoder that generates speaker-specific masks."""
    
    def __init__(
        self,
        d_model: int,
        num_speakers: int,
        nhead: int,
        num_layers: int,
        freq_bins: int = 80,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_speakers = num_speakers
        self.freq_bins = freq_bins
        
        # Speaker query embeddings
        self.speaker_queries = nn.Parameter(
            torch.randn(num_speakers, d_model) * 0.02
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to frequency bins
        self.mask_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, freq_bins),
            nn.Sigmoid()  # Masks should be in [0, 1]
        )
        
    def forward(
        self,
        memory: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = memory.shape
        
        # Expand speaker queries for batch
        tgt = self.speaker_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_speakers, d_model]
        
        # Decode speaker-specific representations
        speaker_repr = self.decoder(tgt, memory)  # [B, num_speakers, d_model]
        
        # Generate frequency masks for each time step
        masks = []
        for t in range(seq_len):
            time_masks = self.mask_projection(speaker_repr)  # [B, num_speakers, freq_bins]
            masks.append(time_masks)
        
        # Stack time dimension
        masks = torch.stack(masks, dim=3)  # [B, num_speakers, freq_bins, seq_len]
        
        return masks


class TransformerModel(nn.Module):
    """Complete audio-visual speech separation transformer.
    
    This model implements the full pipeline:
    1. Audio and video feature encoding
    2. Cross-modal attention fusion
    3. Transformer encoding
    4. Speaker-specific decoding
    
    Args:
        num_speakers: Number of speakers to separate
        audio_dim: Dimension of audio features
        video_dim: Dimension of video features
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_speakers: int = 2,
        audio_dim: int = 80,
        video_dim: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        max_length: int = 5000
    ):
        super().__init__()
        
        self.num_speakers = num_speakers
        self.d_model = d_model
        
        # Input projections
        self.audio_input_proj = nn.Linear(audio_dim, d_model)
        self.video_input_proj = nn.Linear(video_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalAttention(
            audio_dim, video_dim, d_model, nhead, dropout
        )
        
        # Transformer encoder
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            )
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Speaker decoder
        self.speaker_decoder = SpeakerDecoder(
            d_model, num_speakers, nhead, num_layers // 2, audio_dim, dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio_features: [B, freq_bins, time] or [B, time, freq_bins]
            video_features: [B, video_dim, time] or [B, time, video_dim]
            return_intermediate: Whether to return intermediate features
            
        Returns:
            Separated spectrograms: [B, num_speakers, freq_bins, time]
        """
        # Ensure correct input format [B, time, features]
        if audio_features.size(1) != video_features.size(1):
            # Transpose if needed
            if audio_features.size(2) == video_features.size(1):
                audio_features = audio_features.transpose(1, 2)
            elif video_features.size(2) == audio_features.size(1):
                video_features = video_features.transpose(1, 2)
        
        batch_size, seq_len, _ = audio_features.shape
        
        intermediates = {} if return_intermediate else None
        
        # Cross-modal fusion
        fused_features, attention_weights = self.cross_modal_fusion(
            audio_features, video_features
        )
        
        if return_intermediate:
            intermediates['cross_modal_attention'] = attention_weights
            intermediates['fused_features'] = fused_features
        
        # Add positional encoding
        fused_features = self.pos_encoder(fused_features)
        
        # Transformer encoding
        encoded = fused_features
        for i, encoder_layer in enumerate(self.encoder):
            encoded = encoder_layer(encoded)
            if return_intermediate:
                intermediates[f'encoder_layer_{i}'] = encoded
        
        # Speaker-specific decoding
        speaker_masks = self.speaker_decoder(encoded)
        
        # Apply masks to original audio features
        # Expand audio features for each speaker
        audio_expanded = audio_features.unsqueeze(1).expand(
            -1, self.num_speakers, -1, -1
        )  # [B, num_speakers, time, freq_bins]
        
        # Transpose to match mask dimensions
        audio_expanded = audio_expanded.transpose(2, 3)  # [B, num_speakers, freq_bins, time]
        
        # Apply speaker masks
        separated = audio_expanded * speaker_masks
        
        if return_intermediate:
            intermediates['speaker_masks'] = speaker_masks
            intermediates['separated_spectrograms'] = separated
            return separated, intermediates
        
        return separated
    
    def get_attention_weights(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization."""
        with torch.no_grad():
            _, intermediates = self.forward(
                audio_features, video_features, return_intermediate=True
            )
        
        return intermediates.get('cross_modal_attention', {})
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def compress_model(self, compression_ratio: float = 0.5) -> 'TransformerModel':
        """Create a compressed version of the model.
        
        Args:
            compression_ratio: Target compression ratio (0.5 = 50% smaller)
            
        Returns:
            Compressed model instance
        """
        compressed_d_model = int(self.d_model * compression_ratio)
        compressed_layers = max(1, int(len(self.encoder) * compression_ratio))
        
        compressed_model = TransformerModel(
            num_speakers=self.num_speakers,
            audio_dim=self.audio_input_proj.in_features,
            video_dim=self.video_input_proj.in_features,
            d_model=compressed_d_model,
            nhead=max(1, compressed_d_model // 64),  # Maintain reasonable head size
            num_layers=compressed_layers,
            dropout=0.1
        )
        
        return compressed_model