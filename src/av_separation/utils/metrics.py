import numpy as np
import torch
from typing import Union, Optional
import warnings


def compute_si_snr(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> float:
    
    if isinstance(estimated, torch.Tensor):
        estimated = estimated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimated = estimated.flatten()
    target = target.flatten()
    
    if len(estimated) != len(target):
        min_len = min(len(estimated), len(target))
        estimated = estimated[:min_len]
        target = target[:min_len]
    
    target = target - np.mean(target)
    estimated = estimated - np.mean(estimated)
    
    alpha = np.dot(estimated, target) / (np.dot(target, target) + eps)
    target_scaled = alpha * target
    
    noise = estimated - target_scaled
    
    signal_power = np.dot(target_scaled, target_scaled)
    noise_power = np.dot(noise, noise) + eps
    
    si_snr = 10 * np.log10(signal_power / noise_power + eps)
    
    return float(si_snr)


def compute_sdr(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8
) -> float:
    
    if isinstance(estimated, torch.Tensor):
        estimated = estimated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimated = estimated.flatten()
    target = target.flatten()
    
    if len(estimated) != len(target):
        min_len = min(len(estimated), len(target))
        estimated = estimated[:min_len]
        target = target[:min_len]
    
    signal_power = np.dot(target, target)
    
    error = estimated - target
    error_power = np.dot(error, error) + eps
    
    sdr = 10 * np.log10(signal_power / error_power + eps)
    
    return float(sdr)


def compute_pesq(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    mode: str = 'wb'
) -> Optional[float]:
    
    try:
        from pesq import pesq as pesq_score
    except ImportError:
        warnings.warn("PESQ not available. Install with: pip install pesq")
        return None
    
    if isinstance(estimated, torch.Tensor):
        estimated = estimated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimated = estimated.flatten()
    target = target.flatten()
    
    if len(estimated) != len(target):
        min_len = min(len(estimated), len(target))
        estimated = estimated[:min_len]
        target = target[:min_len]
    
    max_val = max(np.abs(estimated).max(), np.abs(target).max())
    if max_val > 1.0:
        estimated = estimated / max_val
        target = target / max_val
    
    try:
        if mode == 'wb' and sample_rate == 16000:
            score = pesq_score(sample_rate, target, estimated, 'wb')
        elif mode == 'nb' and sample_rate == 8000:
            score = pesq_score(sample_rate, target, estimated, 'nb')
        else:
            warnings.warn(f"PESQ mode {mode} with sample rate {sample_rate} not supported")
            return None
        
        return float(score)
    
    except Exception as e:
        warnings.warn(f"PESQ computation failed: {e}")
        return None


def compute_stoi(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    extended: bool = False
) -> Optional[float]:
    
    try:
        from pystoi import stoi as stoi_score
    except ImportError:
        warnings.warn("STOI not available. Install with: pip install pystoi")
        return None
    
    if isinstance(estimated, torch.Tensor):
        estimated = estimated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    estimated = estimated.flatten()
    target = target.flatten()
    
    if len(estimated) != len(target):
        min_len = min(len(estimated), len(target))
        estimated = estimated[:min_len]
        target = target[:min_len]
    
    max_val = max(np.abs(estimated).max(), np.abs(target).max())
    if max_val > 1.0:
        estimated = estimated / max_val
        target = target / max_val
    
    try:
        score = stoi_score(target, estimated, sample_rate, extended=extended)
        return float(score)
    
    except Exception as e:
        warnings.warn(f"STOI computation failed: {e}")
        return None


def compute_mir_eval_metrics(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000
) -> dict:
    
    try:
        import mir_eval.separation as separation
    except ImportError:
        warnings.warn("mir_eval not available. Install with: pip install mir_eval")
        return {}
    
    if isinstance(estimated, torch.Tensor):
        estimated = estimated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    if estimated.ndim == 1:
        estimated = estimated[np.newaxis, :]
    if target.ndim == 1:
        target = target[np.newaxis, :]
    
    min_len = min(estimated.shape[1], target.shape[1])
    estimated = estimated[:, :min_len]
    target = target[:, :min_len]
    
    try:
        sdr, sir, sar, perm = separation.bss_eval_sources(target, estimated)
        
        metrics = {
            'sdr': float(np.mean(sdr)),
            'sir': float(np.mean(sir)),
            'sar': float(np.mean(sar)),
            'permutation': perm.tolist()
        }
        
        return metrics
    
    except Exception as e:
        warnings.warn(f"mir_eval computation failed: {e}")
        return {}


def compute_all_metrics(
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 16000,
    compute_perceptual: bool = True
) -> dict:
    
    metrics = {}
    
    metrics['si_snr'] = compute_si_snr(estimated, target)
    metrics['sdr'] = compute_sdr(estimated, target)
    
    if compute_perceptual:
        pesq = compute_pesq(estimated, target, sample_rate)
        if pesq is not None:
            metrics['pesq'] = pesq
        
        stoi = compute_stoi(estimated, target, sample_rate)
        if stoi is not None:
            metrics['stoi'] = stoi
    
    mir_metrics = compute_mir_eval_metrics(estimated, target, sample_rate)
    if mir_metrics:
        metrics.update({f"mir_{k}": v for k, v in mir_metrics.items() 
                       if k != 'permutation'})
    
    return metrics


def compute_si_snr_improvement(
    mixture: Union[np.ndarray, torch.Tensor],
    estimated: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    
    si_snr_baseline = compute_si_snr(mixture, target)
    si_snr_estimated = compute_si_snr(estimated, target)
    
    si_snri = si_snr_estimated - si_snr_baseline
    
    return float(si_snri)