"""
Losses for audio-visual speech separation.

SI-SNR (Scale-Invariant Signal-to-Noise Ratio) loss is standard for BSS.
Here we operate in the spectrogram domain, so we use spectral SI-SNR
plus a mask regularization term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SNR between estimate and target spectrograms.

    Args:
        estimate: (B, freq_bins, T) or (B, num_speakers, freq_bins, T)
        target:   same shape as estimate
    Returns:
        si_snr: scalar mean SI-SNR (higher is better)
    """
    # Flatten spatial dims
    est = estimate.reshape(estimate.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)

    # Zero-mean
    est = est - est.mean(dim=-1, keepdim=True)
    tgt = tgt - tgt.mean(dim=-1, keepdim=True)

    # Projection of estimate onto target
    dot = (est * tgt).sum(dim=-1, keepdim=True)
    tgt_energy = (tgt * tgt).sum(dim=-1, keepdim=True) + eps
    projection = dot / tgt_energy * tgt

    noise = est - projection
    snr = 10 * torch.log10(
        (projection * projection).sum(dim=-1) /
        ((noise * noise).sum(dim=-1) + eps) + eps
    )
    return snr.mean()


class SeparationLoss(nn.Module):
    """
    Combined separation loss:
      - Permutation-invariant SI-SNR loss across speakers
      - L1 spectrogram reconstruction loss
    """

    def __init__(self, l1_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight

    def forward(
        self,
        separated: torch.Tensor,   # (B, num_speakers, freq_bins, T)
        targets: torch.Tensor,     # (B, num_speakers, freq_bins, T)
    ) -> torch.Tensor:
        B, S = separated.shape[:2]

        # Permutation-invariant: try both speaker orderings, pick the best
        best_loss = None
        for perm in _permutations(S):
            perm_sep = separated[:, perm, :, :]
            snr = si_snr(perm_sep, targets)
            l1 = F.l1_loss(perm_sep, targets)
            loss = -snr + self.l1_weight * l1
            if best_loss is None or loss < best_loss:
                best_loss = loss

        return best_loss


def _permutations(n: int):
    """Yield all permutations of range(n) as lists."""
    if n == 1:
        yield [0]
        return
    if n == 2:
        yield [0, 1]
        yield [1, 0]
        return
    from itertools import permutations as _perms
    yield from (list(p) for p in _perms(range(n)))
