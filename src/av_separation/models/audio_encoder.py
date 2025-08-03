import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class AudioFrontend(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(1, 64, kernel_size=7, stride=1, padding=3),
            ConvBlock(64, 128, kernel_size=5, stride=2, padding=2),
            ConvBlock(128, 256, kernel_size=5, stride=2, padding=2),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = self.freq_pool(x)
        x = rearrange(x, 'b c 1 t -> b t c')
        x = self.projection(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.frontend = AudioFrontend(
            n_mels=config.audio.n_mels,
            hidden_dim=config.model.audio_encoder_dim
        )
        
        self.positional_encoding = PositionalEncoding(config.model.audio_encoder_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.audio_encoder_dim,
            nhead=config.model.audio_encoder_heads,
            dim_feedforward=config.model.audio_encoder_ffn_dim,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.audio_encoder_layers
        )
        
        self.layer_norm = nn.LayerNorm(config.model.audio_encoder_dim)
        
    def forward(self, audio_features, attention_mask=None):
        x = self.frontend(audio_features)
        x = self.positional_encoding(x)
        
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)
        
        return x
    
    def compute_output_shape(self, input_shape):
        batch_size, n_mels, time_frames = input_shape
        
        for _ in range(2):
            time_frames = (time_frames + 3) // 2
        
        return (batch_size, time_frames, self.config.model.audio_encoder_dim)