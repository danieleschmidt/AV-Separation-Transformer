import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class SpeakerQuery(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_speakers = config.model.max_speakers
        self.dim = config.model.decoder_dim
        
        self.speaker_queries = nn.Parameter(
            torch.randn(self.num_speakers, self.dim) * 0.02
        )
        
        self.position_encoding = nn.Parameter(
            torch.randn(1, 500, self.dim) * 0.02
        )
        
    def forward(self, batch_size, seq_len, device):
        queries = repeat(self.speaker_queries, 'n d -> b n l d', 
                        b=batch_size, l=seq_len)
        
        pos_enc = self.position_encoding[:, :seq_len]
        pos_enc = repeat(pos_enc, '1 l d -> b n l d', 
                        b=batch_size, n=self.num_speakers)
        
        queries = queries + pos_enc
        queries = rearrange(queries, 'b n l d -> b (n l) d')
        
        return queries


class MultiScaleSpectrogram(nn.Module):
    def __init__(self, input_dim, n_mels=80, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.n_mels = n_mels
        
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, n_mels),
            )
            for _ in range(n_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        
    def forward(self, x):
        spectrograms = []
        
        for i, projection in enumerate(self.scale_projections):
            scale_spec = projection(x)
            
            if i > 0:
                scale_factor = 2 ** i
                scale_spec = F.interpolate(
                    scale_spec.transpose(1, 2),
                    scale_factor=scale_factor,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            spectrograms.append(scale_spec)
        
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * spec for w, spec in zip(weights, spectrograms))
        
        return combined, spectrograms


class GriffinLim(nn.Module):
    def __init__(self, n_fft=512, hop_length=160, n_iter=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        
    def forward(self, magnitude):
        batch_size, n_frames, n_bins = magnitude.shape
        magnitude = magnitude.transpose(1, 2)
        
        angles = torch.randn_like(magnitude) * 2 * np.pi
        
        for _ in range(self.n_iter):
            complex_spec = magnitude * torch.exp(1j * angles)
            
            waveform = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                length=None
            )
            
            reconstructed = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True
            )
            
            angles = torch.angle(reconstructed)
        
        final_waveform = torch.istft(
            magnitude * torch.exp(1j * angles),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=None
        )
        
        return final_waveform


class SeparationDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_speakers = config.model.max_speakers
        
        self.speaker_query = SpeakerQuery(config)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model.decoder_dim,
            nhead=config.model.decoder_heads,
            dim_feedforward=config.model.decoder_ffn_dim,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.model.decoder_layers
        )
        
        self.multi_scale_spectrogram = MultiScaleSpectrogram(
            config.model.decoder_dim,
            n_mels=config.audio.n_mels,
            n_scales=3
        )
        
        self.magnitude_projection = nn.Sequential(
            nn.Linear(config.model.decoder_dim, config.model.decoder_dim),
            nn.GELU(),
            nn.Linear(config.model.decoder_dim, config.audio.n_fft // 2 + 1),
            nn.ReLU()
        )
        
        self.griffin_lim = GriffinLim(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length
        )
        
        self.post_processor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=9, padding=4),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
            nn.Tanh()
        )
        
        self.speaker_assignment = nn.Linear(
            config.model.decoder_dim,
            self.num_speakers
        )
        
    def forward(self, fused_features, mixture_spectrogram=None):
        batch_size, seq_len, _ = fused_features.shape
        device = fused_features.device
        
        speaker_queries = self.speaker_query(batch_size, seq_len, device)
        
        separated_features = self.transformer_decoder(
            speaker_queries,
            fused_features.unsqueeze(1).expand(-1, self.num_speakers, -1, -1).reshape(
                batch_size, self.num_speakers * seq_len, -1
            )
        )
        
        separated_features = rearrange(
            separated_features, 
            'b (n l) d -> b n l d',
            n=self.num_speakers, 
            l=seq_len
        )
        
        separated_specs = []
        separated_waveforms = []
        speaker_logits = []
        
        for spk_idx in range(self.num_speakers):
            spk_features = separated_features[:, spk_idx]
            
            magnitude = self.magnitude_projection(spk_features)
            
            mel_spec, multi_scale = self.multi_scale_spectrogram(spk_features)
            
            if mixture_spectrogram is not None:
                mask = torch.sigmoid(magnitude)
                magnitude = magnitude * mask + mixture_spectrogram * (1 - mask)
            
            waveform = self.griffin_lim(magnitude)
            
            waveform = self.post_processor(waveform.unsqueeze(1))
            waveform = waveform.squeeze(1)
            
            spk_logits = self.speaker_assignment(spk_features.mean(dim=1))
            
            separated_specs.append(magnitude)
            separated_waveforms.append(waveform)
            speaker_logits.append(spk_logits)
        
        separated_specs = torch.stack(separated_specs, dim=1)
        separated_waveforms = torch.stack(separated_waveforms, dim=1)
        speaker_logits = torch.stack(speaker_logits, dim=1)
        
        return {
            'waveforms': separated_waveforms,
            'spectrograms': separated_specs,
            'speaker_logits': speaker_logits,
            'features': separated_features
        }