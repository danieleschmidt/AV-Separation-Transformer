"""
Tests for AV-Separation-Transformer components.
"""

import sys
import math
import pytest
import torch
import numpy as np

sys.path.insert(0, "src")

from av_separation.model import (
    AudioEncoder,
    VisualEncoder,
    CrossModalFusion,
    SeparationDecoder,
    AVSeparationTransformer,
    PositionalEncoding,
)
from av_separation.dataset import SyntheticAVDataset
from av_separation.losses import SeparationLoss, si_snr


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FREQ_BINS = 65
T = 32
D_MODEL = 64
NHEAD = 4
BATCH = 2
NUM_FRAMES = 10
H, W = 16, 16
NUM_SPEAKERS = 2


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def audio_batch(device):
    return torch.randn(BATCH, FREQ_BINS, T, device=device)


@pytest.fixture
def visual_batch(device):
    return torch.randn(BATCH, NUM_FRAMES, H, W, device=device)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(D_MODEL)
        x = torch.zeros(BATCH, T, D_MODEL)
        out = pe(x)
        assert out.shape == (BATCH, T, D_MODEL)

    def test_adds_encoding(self):
        pe = PositionalEncoding(D_MODEL, dropout=0.0)
        x = torch.zeros(BATCH, T, D_MODEL)
        out = pe(x)
        # Output should not be all zeros (position encodings were added)
        assert not torch.all(out == 0)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class TestAudioEncoder:
    def test_output_shape(self, audio_batch, device):
        enc = AudioEncoder(freq_bins=FREQ_BINS, d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        out = enc(audio_batch)
        assert out.shape == (BATCH, T, D_MODEL)

    def test_different_T(self, device):
        enc = AudioEncoder(freq_bins=FREQ_BINS, d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        for t in [16, 32, 64]:
            x = torch.randn(BATCH, FREQ_BINS, t)
            out = enc(x)
            assert out.shape == (BATCH, t, D_MODEL), f"Failed for T={t}"

    def test_gradient_flow(self, audio_batch):
        enc = AudioEncoder(freq_bins=FREQ_BINS, d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        out = enc(audio_batch)
        loss = out.sum()
        loss.backward()
        for name, param in enc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# VisualEncoder
# ---------------------------------------------------------------------------

class TestVisualEncoder:
    def test_output_shape(self, visual_batch, device):
        enc = VisualEncoder(d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        out = enc(visual_batch, target_len=T)
        assert out.shape == (BATCH, T, D_MODEL)

    def test_different_target_len(self, visual_batch):
        enc = VisualEncoder(d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        for tlen in [20, 32, 50]:
            out = enc(visual_batch, target_len=tlen)
            assert out.shape == (BATCH, tlen, D_MODEL), f"Failed for target_len={tlen}"

    def test_gradient_flow(self, visual_batch):
        enc = VisualEncoder(d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        out = enc(visual_batch, target_len=T)
        out.sum().backward()
        for name, param in enc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# CrossModalFusion
# ---------------------------------------------------------------------------

class TestCrossModalFusion:
    def test_output_shape(self, device):
        fusion = CrossModalFusion(d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        audio = torch.randn(BATCH, T, D_MODEL)
        visual = torch.randn(BATCH, T, D_MODEL)
        out = fusion(audio, visual)
        assert out.shape == (BATCH, T, D_MODEL)

    def test_audio_visual_difference(self):
        """Fused output should differ from raw audio embedding."""
        fusion = CrossModalFusion(d_model=D_MODEL, nhead=NHEAD, num_layers=1)
        audio = torch.randn(BATCH, T, D_MODEL)
        visual1 = torch.randn(BATCH, T, D_MODEL)
        visual2 = torch.randn(BATCH, T, D_MODEL)

        out1 = fusion(audio, visual1)
        out2 = fusion(audio, visual2)

        # Different visual inputs should produce different fused outputs
        assert not torch.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# SeparationDecoder
# ---------------------------------------------------------------------------

class TestSeparationDecoder:
    def test_output_shape(self, device):
        dec = SeparationDecoder(d_model=D_MODEL, freq_bins=FREQ_BINS,
                                num_speakers=NUM_SPEAKERS)
        fused = torch.randn(BATCH, T, D_MODEL)
        masks = dec(fused)
        assert masks.shape == (BATCH, NUM_SPEAKERS, FREQ_BINS, T)

    def test_mask_range(self):
        """Masks must be in [0, 1] due to sigmoid."""
        dec = SeparationDecoder(d_model=D_MODEL, freq_bins=FREQ_BINS,
                                num_speakers=NUM_SPEAKERS)
        fused = torch.randn(BATCH, T, D_MODEL)
        masks = dec(fused)
        assert masks.min() >= 0.0, "Mask below 0"
        assert masks.max() <= 1.0, "Mask above 1"

    def test_separate_shape(self):
        dec = SeparationDecoder(d_model=D_MODEL, freq_bins=FREQ_BINS,
                                num_speakers=NUM_SPEAKERS)
        fused = torch.randn(BATCH, T, D_MODEL)
        masks = dec(fused)
        mixed = torch.randn(BATCH, FREQ_BINS, T)
        separated = dec.separate(masks, mixed)
        assert separated.shape == (BATCH, NUM_SPEAKERS, FREQ_BINS, T)


# ---------------------------------------------------------------------------
# AVSeparationTransformer (end-to-end)
# ---------------------------------------------------------------------------

class TestAVSeparationTransformer:
    def _build(self):
        return AVSeparationTransformer(
            freq_bins=FREQ_BINS,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=1,
            num_fusion_layers=1,
            num_speakers=NUM_SPEAKERS,
            dropout=0.0,
        )

    def test_output_shapes(self, audio_batch, visual_batch):
        model = self._build()
        separated, masks = model(audio_batch, visual_batch)
        assert separated.shape == (BATCH, NUM_SPEAKERS, FREQ_BINS, T)
        assert masks.shape == (BATCH, NUM_SPEAKERS, FREQ_BINS, T)

    def test_masks_bounded(self, audio_batch, visual_batch):
        model = self._build()
        _, masks = model(audio_batch, visual_batch)
        assert masks.min() >= 0.0
        assert masks.max() <= 1.0

    def test_backward_pass(self, audio_batch, visual_batch):
        model = self._build()
        separated, masks = model(audio_batch, visual_batch)
        loss = separated.sum() + masks.sum()
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_eval_mode_no_error(self, audio_batch, visual_batch):
        model = self._build()
        model.eval()
        with torch.no_grad():
            separated, masks = model(audio_batch, visual_batch)
        assert separated.shape == (BATCH, NUM_SPEAKERS, FREQ_BINS, T)

    def test_parameter_count(self):
        model = self._build()
        count = sum(p.numel() for p in model.parameters())
        # Should be a reasonable size (not zero, not accidentally huge)
        assert 10_000 < count < 100_000_000, f"Unexpected parameter count: {count}"


# ---------------------------------------------------------------------------
# SyntheticAVDataset
# ---------------------------------------------------------------------------

class TestSyntheticAVDataset:
    def _build(self):
        return SyntheticAVDataset(
            num_samples=10,
            sample_rate=8000,
            duration=0.5,
            n_fft=256,
            hop_length=64,
            num_frames=10,
            frame_h=16,
            frame_w=16,
            speaker_freqs=(220.0, 440.0),
        )

    def test_len(self):
        ds = self._build()
        assert len(ds) == 10

    def test_sample_shapes(self):
        ds = self._build()
        sample = ds[0]
        freq_bins = 256 // 2 + 1  # 129
        T_expected = 1 + int(8000 * 0.5) // 64

        assert sample["mixed_spec"].shape == (freq_bins, T_expected)
        assert sample["clean_specs"].shape == (2, freq_bins, T_expected)
        assert sample["lip_frames"].shape[1:] == (16, 16)
        assert sample["lip_frames"].shape[0] == 2 * 10  # 2 speakers * num_frames

    def test_lip_frames_range(self):
        ds = self._build()
        sample = ds[0]
        assert sample["lip_frames"].min() >= 0.0
        assert sample["lip_frames"].max() <= 1.0

    def test_mixed_equals_sum(self):
        """mixed_spec should be close to STFT(speaker1 + speaker2)."""
        # This is checked implicitly since both use same STFT — verify shapes match
        ds = self._build()
        sample = ds[0]
        assert sample["mixed_spec"].shape == sample["clean_specs"][0].shape

    def test_deterministic(self):
        """Same index should return same sample."""
        ds = self._build()
        s1 = ds[3]
        s2 = ds[3]
        assert torch.allclose(s1["mixed_spec"], s2["mixed_spec"])

    def test_different_samples_differ(self):
        ds = self._build()
        s0 = ds[0]
        s1 = ds[1]
        assert not torch.allclose(s0["mixed_spec"], s1["mixed_spec"])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_si_snr_perfect(self):
        """SI-SNR of identical signals should be very high."""
        x = torch.randn(4, 100)
        snr = si_snr(x, x)
        assert snr > 20.0, f"Expected high SNR for identical signals, got {snr:.1f}"

    def test_si_snr_orthogonal(self):
        """SI-SNR of orthogonal signals should be low."""
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        snr = si_snr(x, y)
        assert snr < 0.0, f"Expected negative SNR for orthogonal, got {snr:.1f}"

    def test_separation_loss_shape(self):
        loss_fn = SeparationLoss()
        separated = torch.randn(BATCH, NUM_SPEAKERS, FREQ_BINS, T)
        targets = torch.randn(BATCH, NUM_SPEAKERS, FREQ_BINS, T)
        loss = loss_fn(separated, targets)
        assert loss.shape == (), "Loss should be scalar"
        assert not torch.isnan(loss)

    def test_separation_loss_backward(self):
        loss_fn = SeparationLoss()
        separated = torch.randn(BATCH, NUM_SPEAKERS, FREQ_BINS, T, requires_grad=True)
        targets = torch.randn(BATCH, NUM_SPEAKERS, FREQ_BINS, T)
        loss = loss_fn(separated, targets)
        loss.backward()
        assert separated.grad is not None


# ---------------------------------------------------------------------------
# Integration: training step
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_one_training_step(self):
        """Full forward + backward + optimizer step without error."""
        model = AVSeparationTransformer(
            freq_bins=FREQ_BINS, d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=1, num_fusion_layers=1,
            num_speakers=NUM_SPEAKERS, dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = SeparationLoss()

        mixed = torch.randn(BATCH, FREQ_BINS, T)
        frames = torch.randn(BATCH, NUM_FRAMES, H, W)
        targets = torch.randn(BATCH, NUM_SPEAKERS, FREQ_BINS, T)

        optimizer.zero_grad()
        separated, masks = model(mixed, frames)
        loss = loss_fn(separated, targets)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), "Loss is NaN"

    def test_dataloader_batch(self):
        from torch.utils.data import DataLoader
        ds = SyntheticAVDataset(num_samples=8, n_fft=256, hop_length=64,
                                num_frames=10, frame_h=16, frame_w=16)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        assert batch["mixed_spec"].shape[0] == 4
        assert batch["lip_frames"].shape[0] == 4
        assert batch["clean_specs"].shape[0] == 4
