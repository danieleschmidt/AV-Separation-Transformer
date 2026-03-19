#!/usr/bin/env python3
"""
AV-Separation-Transformer Demo

Demonstrates audio-visual speech separation on synthetic data.
Two sine-wave "speakers" are mixed; the model separates them using
both audio features and simulated lip-movement cues.

Measures SNR improvement before and after separation.
"""

import sys
import math
import torch
import numpy as np

# Allow running from repo root
sys.path.insert(0, "src")

from av_separation import AVSeparationTransformer, SyntheticAVDataset
from av_separation.losses import SeparationLoss, si_snr


def snr_db(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-8) -> float:
    """SNR in dB: signal vs. noise arrays."""
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * math.log10(sig_power / (noise_power + eps) + eps)


def evaluate_separation(model, dataset, device, num_eval: int = 20):
    """
    Compute average SNR for mixed signal vs. separated outputs.

    Returns:
        avg_input_snr:  SNR of the mixed signal (lower bound)
        avg_output_snr: SNR of the separated signals (should be higher)
    """
    model.eval()
    input_snrs, output_snrs = [], []

    with torch.no_grad():
        for i in range(min(num_eval, len(dataset))):
            sample = dataset[i]
            mixed = sample["mixed_spec"].unsqueeze(0).to(device)    # (1, F, T)
            frames = sample["lip_frames"].unsqueeze(0).to(device)   # (1, N, H, W)
            targets = sample["clean_specs"].numpy()                  # (S, F, T)

            separated, _ = model(mixed, frames)
            separated = separated.squeeze(0).cpu().numpy()          # (S, F, T)

            mixed_np = sample["mixed_spec"].numpy()                 # (F, T)

            # Input SNR: how much does mixed bleed into each speaker target?
            for s in range(targets.shape[0]):
                noise_in = mixed_np - targets[s]
                input_snrs.append(snr_db(targets[s], noise_in))

            # Output SNR: how close is separated[s] to targets[s]?
            # Permutation-invariant matching
            best_snr = _permutation_snr(separated, targets)
            output_snrs.append(best_snr)

    return np.mean(input_snrs), np.mean(output_snrs)


def _permutation_snr(separated: np.ndarray, targets: np.ndarray) -> float:
    """Find best-permutation average output SNR."""
    S = separated.shape[0]
    best = -1e9
    from itertools import permutations
    for perm in permutations(range(S)):
        snrs = []
        for s, t in zip(perm, range(S)):
            noise = separated[s] - targets[t]
            snrs.append(snr_db(targets[t], noise))
        val = np.mean(snrs)
        if val > best:
            best = val
    return best


def quick_train(model, dataset, device, steps: int = 100, lr: float = 3e-4):
    """Quick few-step training to give the model a chance to learn."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = SeparationLoss(l1_weight=0.5)
    model.train()

    losses = []
    step = 0
    for batch in loader:
        mixed = batch["mixed_spec"].to(device)
        frames = batch["lip_frames"].to(device)
        targets = batch["clean_specs"].to(device)

        optimizer.zero_grad()
        separated, _ = model(mixed, frames)
        loss = criterion(separated, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        step += 1
        if step % 20 == 0:
            print(f"  step {step:4d} | loss {np.mean(losses[-20:]):.4f}")
        if step >= steps:
            break

    return losses


def main():
    print("=" * 60)
    print("  AV-Separation-Transformer Demo")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Dataset ---
    print("\n[1] Building synthetic AV dataset ...")
    dataset = SyntheticAVDataset(
        num_samples=500,
        sample_rate=8000,
        duration=1.0,
        n_fft=512,
        hop_length=128,
        num_frames=25,
        frame_h=32,
        frame_w=32,
        speaker_freqs=(220.0, 440.0),
    )
    sample = dataset[0]
    print(f"    mixed_spec shape : {tuple(sample['mixed_spec'].shape)}")
    print(f"    lip_frames shape : {tuple(sample['lip_frames'].shape)}")
    print(f"    clean_specs shape: {tuple(sample['clean_specs'].shape)}")

    freq_bins, T = sample["mixed_spec"].shape
    num_frames = sample["lip_frames"].shape[0]
    print(f"    freq_bins={freq_bins}, T={T}, num_video_frames={num_frames}")

    # --- Model ---
    print("\n[2] Initialising model ...")
    model = AVSeparationTransformer(
        freq_bins=freq_bins,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_fusion_layers=2,
        num_speakers=2,
        dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {param_count:,}")

    # --- Baseline SNR (untrained) ---
    print("\n[3] Baseline SNR (untrained model) ...")
    in_snr_before, out_snr_before = evaluate_separation(model, dataset, device)
    print(f"    Input  SNR (mixed)     : {in_snr_before:.2f} dB")
    print(f"    Output SNR (untrained) : {out_snr_before:.2f} dB")

    # --- Training ---
    print("\n[4] Quick training (100 steps) ...")
    losses = quick_train(model, dataset, device, steps=100, lr=3e-4)
    print(f"    Final loss: {losses[-1]:.4f}")

    # --- Post-training SNR ---
    print("\n[5] SNR after training ...")
    in_snr_after, out_snr_after = evaluate_separation(model, dataset, device)
    print(f"    Input  SNR (mixed)    : {in_snr_after:.2f} dB")
    print(f"    Output SNR (trained)  : {out_snr_after:.2f} dB")
    improvement = out_snr_after - in_snr_before
    print(f"    SNR improvement       : {improvement:+.2f} dB")

    # --- Forward pass sanity check ---
    print("\n[6] Single forward pass check ...")
    model.eval()
    with torch.no_grad():
        mixed = sample["mixed_spec"].unsqueeze(0).to(device)
        frames = sample["lip_frames"].unsqueeze(0).to(device)
        separated, masks = model(mixed, frames)
    print(f"    separated shape: {tuple(separated.shape)}")
    print(f"    masks shape    : {tuple(masks.shape)}")
    print(f"    masks range    : [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"    ✓ Mask values in [0,1] — {masks.min() >= 0 and masks.max() <= 1}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    if improvement > 0:
        print(f"  ✓ SNR improved by {improvement:.2f} dB after {100} training steps.")
    else:
        print("  ⚠ Model needs more training for positive SNR gain (expected for 100 steps).")
    print("=" * 60)


if __name__ == "__main__":
    main()
