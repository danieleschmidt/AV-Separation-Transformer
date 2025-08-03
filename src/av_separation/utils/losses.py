import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from typing import Optional, Tuple


class SISNRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if estimated.shape != target.shape:
            raise ValueError(f"Shape mismatch: {estimated.shape} vs {target.shape}")
        
        if length is not None:
            mask = self._create_mask(estimated, length)
            estimated = estimated * mask
            target = target * mask
        
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
        
        dot_product = torch.sum(estimated * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        
        scale = dot_product / target_energy
        target_scaled = scale * target
        
        noise = estimated - target_scaled
        
        signal_power = torch.sum(target_scaled ** 2, dim=-1)
        noise_power = torch.sum(noise ** 2, dim=-1) + self.eps
        
        si_snr = 10 * torch.log10(signal_power / noise_power + self.eps)
        
        return -torch.mean(si_snr)
    
    def _create_mask(
        self,
        tensor: torch.Tensor,
        length: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, max_len = tensor.shape[:2]
        mask = torch.arange(max_len, device=tensor.device).unsqueeze(0)
        mask = mask < length.unsqueeze(1)
        return mask.float().unsqueeze(-1)


class PITLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, eps: float = 1e-8):
        super().__init__()
        self.loss_fn = loss_fn
        self.eps = eps
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, n_sources, waveform_len = estimated.shape
        
        if estimated.shape != target.shape:
            raise ValueError(f"Shape mismatch: {estimated.shape} vs {target.shape}")
        
        all_permutations = list(permutations(range(n_sources)))
        
        losses = []
        for perm in all_permutations:
            perm_estimated = estimated[:, list(perm), :]
            loss = self.loss_fn(perm_estimated, target, length)
            losses.append(loss)
        
        losses = torch.stack(losses, dim=0)
        
        min_loss, min_idx = torch.min(losses, dim=0)
        
        best_perm = torch.tensor(
            all_permutations[min_idx],
            dtype=torch.long,
            device=estimated.device
        )
        
        return min_loss, best_perm


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [120, 240, 480],
        win_lengths: list = [480, 960, 1920],
        sc_weight: float = 1.0,
        mag_weight: float = 1.0
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        
        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(win_len), requires_grad=False)
            for win_len in win_lengths
        ])
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        
        total_loss = 0.0
        
        for fft_size, hop_size, win_len, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows
        ):
            est_stft = torch.stft(
                estimated.reshape(-1, estimated.shape[-1]),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_len,
                window=window.to(estimated.device),
                return_complex=True
            )
            
            tgt_stft = torch.stft(
                target.reshape(-1, target.shape[-1]),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_len,
                window=window.to(target.device),
                return_complex=True
            )
            
            est_mag = torch.abs(est_stft)
            tgt_mag = torch.abs(tgt_stft)
            sc_loss = self._spectral_convergence_loss(est_mag, tgt_mag)
            
            mag_loss = F.l1_loss(est_mag, tgt_mag)
            
            total_loss += self.sc_weight * sc_loss + self.mag_weight * mag_loss
        
        return total_loss / len(self.fft_sizes)
    
    def _spectral_convergence_loss(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        
        numerator = torch.norm(estimated - target, p="fro")
        denominator = torch.norm(target, p="fro") + eps
        
        return numerator / denominator


class PerceptualLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=80,
            sample_rate=16000,
            n_stft=513
        )
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        
        est_stft = torch.stft(
            estimated.reshape(-1, estimated.shape[-1]),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        )
        
        tgt_stft = torch.stft(
            target.reshape(-1, target.shape[-1]),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        )
        
        est_mag = torch.abs(est_stft)
        tgt_mag = torch.abs(tgt_stft)
        
        est_mel = self.mel_scale(est_mag)
        tgt_mel = self.mel_scale(tgt_mag)
        
        est_log_mel = torch.log10(est_mel + 1e-10)
        tgt_log_mel = torch.log10(tgt_mel + 1e-10)
        
        return self.loss_fn(est_log_mel, tgt_log_mel)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        si_snr_weight: float = 1.0,
        stft_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        use_pit: bool = True
    ):
        super().__init__()
        
        self.si_snr_weight = si_snr_weight
        self.stft_weight = stft_weight
        self.perceptual_weight = perceptual_weight
        
        self.si_snr_loss = SISNRLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.perceptual_loss = PerceptualLoss()
        
        if use_pit:
            self.pit_loss = PITLoss(self.si_snr_loss)
        else:
            self.pit_loss = None
    
    def forward(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        length: Optional[torch.Tensor] = None
    ) -> dict:
        
        losses = {}
        
        if self.pit_loss:
            si_snr_loss, best_perm = self.pit_loss(estimated, target, length)
            
            batch_idx = torch.arange(estimated.shape[0]).unsqueeze(1)
            estimated_reordered = estimated[batch_idx, best_perm]
        else:
            si_snr_loss = self.si_snr_loss(estimated, target, length)
            estimated_reordered = estimated
        
        losses['si_snr'] = si_snr_loss * self.si_snr_weight
        
        if self.stft_weight > 0:
            stft_loss = self.stft_loss(estimated_reordered, target)
            losses['stft'] = stft_loss * self.stft_weight
        
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(estimated_reordered, target)
            losses['perceptual'] = perceptual_loss * self.perceptual_weight
        
        losses['total'] = sum(losses.values())
        
        return losses