"""
Audio-Visual Speech Separation Transformer

Architecture:
  AudioEncoder      - 1D Conv + Transformer encodes mixed audio → audio embeddings
  VisualEncoder     - 2D Conv + positional encoding encodes lip frames → visual embeddings
  CrossModalFusion  - Cross-attention fuses audio (query) with visual (key/value)
  SeparationDecoder - Predicts per-speaker STFT masks, applies to mixed spectrogram
  AVSeparationTransformer - End-to-end model combining all components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class AudioEncoder(nn.Module):
    """
    Encodes a mixed audio spectrogram into a sequence of embeddings.

    Input:  (B, freq_bins, T)  — magnitude spectrogram (single channel)
    Output: (B, T, d_model)
    """

    def __init__(self, freq_bins: int = 257, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.freq_bins = freq_bins
        self.d_model = d_model

        # Project frequency bins → d_model via 1D conv over time
        self.input_proj = nn.Sequential(
            nn.Conv1d(freq_bins, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, freq_bins, T)
        h = self.input_proj(x)          # (B, d_model, T)
        h = h.permute(0, 2, 1)          # (B, T, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)         # (B, T, d_model)
        return h


# ---------------------------------------------------------------------------
# VisualEncoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """
    Encodes a sequence of lip-movement frames into visual embeddings.

    Input:  (B, num_frames, H, W)  — grayscale lip-ROI frames
    Output: (B, T_audio, d_model)  — upsampled to match audio time dimension
    """

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 2D Conv backbone for spatial features per frame
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # H/2, W/2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4, W/4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/8, W/8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.frame_proj = nn.Linear(128, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, frames: torch.Tensor, target_len: int) -> torch.Tensor:
        # frames: (B, num_frames, H, W)
        B, N, H, W = frames.shape
        x = frames.view(B * N, 1, H, W)          # treat each frame independently
        feat = self.conv(x).view(B * N, -1)       # (B*N, 128)
        feat = self.frame_proj(feat).view(B, N, self.d_model)  # (B, N, d_model)

        feat = self.pos_enc(feat)
        feat = self.transformer(feat)              # (B, N, d_model)

        # Upsample visual sequence to match audio time dimension
        feat = feat.permute(0, 2, 1)              # (B, d_model, N)
        feat = F.interpolate(feat, size=target_len, mode='linear', align_corners=False)
        feat = feat.permute(0, 2, 1)              # (B, T_audio, d_model)
        return feat


# ---------------------------------------------------------------------------
# CrossModalFusion
# ---------------------------------------------------------------------------

class CrossModalFusion(nn.Module):
    """
    Fuses audio and visual embeddings via cross-attention.

    Audio embeddings act as queries; visual embeddings supply keys and values.
    This lets each audio time step attend to the most relevant lip-movement cues.

    Input:
        audio:  (B, T, d_model)
        visual: (B, T, d_model)
    Output: (B, T, d_model)  — audio enriched with visual context
    """

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        h = audio
        for layer in self.layers:
            h = layer(h, visual)
        return self.norm(h)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # Pre-norm cross-attention: audio queries visual
        normed = self.norm1(audio)
        attn_out, _ = self.cross_attn(normed, visual, visual)
        audio = audio + self.drop(attn_out)
        # Feed-forward
        audio = audio + self.drop(self.ff(self.norm2(audio)))
        return audio


# ---------------------------------------------------------------------------
# SeparationDecoder
# ---------------------------------------------------------------------------

class SeparationDecoder(nn.Module):
    """
    Predicts per-speaker soft masks in the STFT domain.

    Input:  fused embeddings (B, T, d_model)
    Output: masks (B, num_speakers, freq_bins, T)  — values in [0, 1]
    """

    def __init__(self, d_model: int = 256, freq_bins: int = 257,
                 num_speakers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_speakers = num_speakers
        self.freq_bins = freq_bins

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, freq_bins * num_speakers),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # fused: (B, T, d_model)
        B, T, _ = fused.shape
        masks = self.decoder(fused)                          # (B, T, freq_bins * num_speakers)
        masks = masks.view(B, T, self.num_speakers, self.freq_bins)
        masks = masks.permute(0, 2, 3, 1)                   # (B, num_speakers, freq_bins, T)
        masks = torch.sigmoid(masks)
        return masks

    def separate(self, masks: torch.Tensor, mixed_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply masks to mixed spectrogram to get per-speaker spectrograms.

        Args:
            masks:      (B, num_speakers, freq_bins, T)
            mixed_spec: (B, freq_bins, T)
        Returns:
            separated:  (B, num_speakers, freq_bins, T)
        """
        return masks * mixed_spec.unsqueeze(1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AVSeparationTransformer(nn.Module):
    """
    End-to-end audio-visual speech separation transformer.

    Inputs:
        mixed_spec: (B, freq_bins, T)        — magnitude spectrogram of mixed audio
        lip_frames: (B, num_frames, H, W)    — lip-region video frames

    Returns:
        separated:  (B, num_speakers, freq_bins, T)  — per-speaker spectrograms
        masks:      (B, num_speakers, freq_bins, T)  — soft masks (for loss)
    """

    def __init__(
        self,
        freq_bins: int = 257,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_fusion_layers: int = 2,
        num_speakers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            freq_bins=freq_bins, d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, dropout=dropout,
        )
        self.visual_encoder = VisualEncoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_encoder_layers, dropout=dropout,
        )
        self.fusion = CrossModalFusion(
            d_model=d_model, nhead=nhead,
            num_layers=num_fusion_layers, dropout=dropout,
        )
        self.decoder = SeparationDecoder(
            d_model=d_model, freq_bins=freq_bins,
            num_speakers=num_speakers, dropout=dropout,
        )

    def forward(self, mixed_spec: torch.Tensor, lip_frames: torch.Tensor):
        T = mixed_spec.shape[-1]

        audio_emb = self.audio_encoder(mixed_spec)          # (B, T, d_model)
        visual_emb = self.visual_encoder(lip_frames, T)     # (B, T, d_model)
        fused = self.fusion(audio_emb, visual_emb)          # (B, T, d_model)
        masks = self.decoder(fused)                         # (B, num_speakers, F, T)
        separated = self.decoder.separate(masks, mixed_spec)
        return separated, masks


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
