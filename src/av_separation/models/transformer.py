import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .fusion import CrossModalFusion
from .decoder import SeparationDecoder


class AVSeparationTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)
        self.fusion = CrossModalFusion(config)
        self.decoder = SeparationDecoder(config)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        mixture_spectrogram: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        audio_features = self.audio_encoder(audio_input, audio_mask)
        
        video_features, detected_faces = self.video_encoder(video_input, speaker_ids)
        
        fused_features, alignment_score = self.fusion(
            audio_features, video_features,
            audio_mask, video_mask
        )
        
        separation_outputs = self.decoder(fused_features, mixture_spectrogram)
        
        outputs = {
            'separated_waveforms': separation_outputs['waveforms'],
            'separated_spectrograms': separation_outputs['spectrograms'],
            'speaker_logits': separation_outputs['speaker_logits'],
            'audio_features': audio_features,
            'video_features': video_features,
            'fused_features': fused_features,
            'alignment_score': alignment_score,
            'detected_faces': detected_faces,
        }
        
        return outputs
    
    def separate(
        self,
        audio_waveform: torch.Tensor,
        video_frames: torch.Tensor,
        num_speakers: Optional[int] = None
    ) -> torch.Tensor:
        
        self.eval()
        with torch.no_grad():
            # Handle case where audio_waveform is actually a spectrogram
            if audio_waveform.dim() == 2:
                audio_spec = audio_waveform
            else:
                audio_spec = self._compute_spectrogram(audio_waveform)
            
            outputs = self.forward(
                audio_spec.unsqueeze(0),
                video_frames.unsqueeze(0)
            )
            
            separated = outputs['separated_waveforms'].squeeze(0)
            
            if num_speakers and num_speakers < self.config.model.max_speakers:
                speaker_scores = outputs['speaker_logits'].mean(dim=-1)
                top_k = torch.topk(speaker_scores.squeeze(0), num_speakers).indices
                separated = separated[top_k]
            
            return separated
    
    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            window=torch.hann_window(self.config.audio.win_length).to(waveform.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        
        mel_filterbank = self._get_mel_filterbank(waveform.device)
        mel_spec = torch.matmul(mel_filterbank, magnitude)
        
        log_mel = torch.log10(mel_spec + 1e-10)
        
        return log_mel
    
    def _get_mel_filterbank(self, device):
        if not hasattr(self, '_mel_filterbank'):
            import librosa
            mel_basis = librosa.filters.mel(
                sr=self.config.audio.sample_rate,
                n_fft=self.config.audio.n_fft,
                n_mels=self.config.audio.n_mels,
                fmin=self.config.audio.f_min,
                fmax=self.config.audio.f_max
            )
            self._mel_filterbank = torch.from_numpy(mel_basis).float().to(device)
        return self._mel_filterbank
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def enable_gradient_checkpointing(self):
        self.config.model.gradient_checkpointing = True
        
        def enable_checkpoint(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        
        self.apply(enable_checkpoint)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config=None, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if config is None:
            config = checkpoint.get('config')
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_pretrained(self, save_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, save_path)