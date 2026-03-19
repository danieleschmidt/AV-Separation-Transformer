# AV-Separation-Transformer

Audio-visual speech separation using a transformer-based cross-modal fusion architecture. Separates individual speaker streams from a mixed audio signal by jointly attending to audio features **and** synchronized lip-movement video cues.

## What It Does

Given a mixed audio signal (multiple speakers talking simultaneously) and a video stream of lip movements, the model produces per-speaker audio separation masks in the STFT domain — recovering each speaker's clean signal.

**Key insight:** Visual lip-movement cues provide an independent, complementary signal to resolve the permutation ambiguity in blind source separation. When speaker A's lips are moving, the model can preferentially unmask frequencies associated with their voice.

## Architecture

```
Mixed Audio (STFT)         Lip Frames
       │                       │
  AudioEncoder            VisualEncoder
  (1D Conv +              (2D Conv +
   Transformer)            Positional Enc +
       │                   Transformer)
       │                       │
       └──── CrossModalFusion ─┘
             (Cross-Attention:
              audio queries visual)
                    │
            SeparationDecoder
            (predicts per-speaker
             STFT masks)
                    │
            Separated Spectrograms
            (one per speaker)
```

### Components

| Module | Role |
|---|---|
| `AudioEncoder` | 1D Conv + Transformer encoder on mixed STFT → audio embeddings `(B, T, d)` |
| `VisualEncoder` | 2D Conv per frame + Transformer → visual embeddings `(B, T, d)` |
| `CrossModalFusion` | Multi-layer cross-attention: audio queries, visual keys/values |
| `SeparationDecoder` | Linear head → per-speaker soft masks in `[0, 1]` via sigmoid |
| `SyntheticAVDataset` | Synthetic data generator: sine-wave speakers + correlated lip frames |

## Quick Start

```bash
# Clone
git clone https://github.com/danieleschmidt/AV-Separation-Transformer
cd AV-Separation-Transformer

# Install (minimal dependencies)
pip install torch numpy scipy

# Run demo — separates 2 synthetic speakers, shows SNR improvement
python demo.py
```

Expected output:
```
Device: cuda
Parameters: 1,612,738
Input  SNR (mixed)     :  0.01 dB
Output SNR (untrained) :  3.20 dB
...
Output SNR (trained)   : 37.24 dB
SNR improvement        : +37.23 dB
```

## Usage

```python
from src.av_separation import AVSeparationTransformer, SyntheticAVDataset

# Build model
model = AVSeparationTransformer(
    freq_bins=257,      # n_fft // 2 + 1
    d_model=256,        # embedding dimension
    nhead=4,            # attention heads
    num_encoder_layers=2,
    num_fusion_layers=2,
    num_speakers=2,
)

# Inputs
mixed_spec = ...   # (B, freq_bins, T)   — STFT magnitude of mixed audio
lip_frames = ...   # (B, num_frames, H, W) — lip-region video frames

# Separate
separated, masks = model(mixed_spec, lip_frames)
# separated: (B, num_speakers, freq_bins, T)
# masks:     (B, num_speakers, freq_bins, T)  in [0, 1]
```

### Synthetic Dataset

```python
from src.av_separation import SyntheticAVDataset
from torch.utils.data import DataLoader

ds = SyntheticAVDataset(
    num_samples=1000,
    sample_rate=8000,
    duration=1.0,
    speaker_freqs=(220.0, 440.0),   # Hz — one per speaker
)

loader = DataLoader(ds, batch_size=8, shuffle=True)
batch = next(iter(loader))
# batch["mixed_spec"]  : (8, freq_bins, T)
# batch["lip_frames"]  : (8, 2*num_frames, H, W)
# batch["clean_specs"] : (8, 2, freq_bins, T)
```

### Training

```python
from src.av_separation.losses import SeparationLoss

criterion = SeparationLoss(l1_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for batch in loader:
    separated, masks = model(batch["mixed_spec"], batch["lip_frames"])
    loss = criterion(separated, batch["clean_specs"])
    loss.backward()
    optimizer.step()
```

The loss is **permutation-invariant SI-SNR** + L1, so speaker ordering doesn't matter.

## Tests

```bash
python -m pytest tests/ -v
```

30 tests covering all components: shape correctness, gradient flow, mask bounds, SI-SNR properties, dataset determinism, and end-to-end integration.

## Design Notes

- **STFT domain**: operates on magnitude spectrograms. Phase reconstruction (Griffin-Lim or learned) needed for full waveform output.
- **Permutation invariance**: loss considers all speaker orderings, picks the lowest-loss assignment.
- **Visual upsampling**: visual encoder outputs are linearly interpolated to match the audio time dimension — keeps the model video-FPS agnostic.
- **Synthetic data**: sine waves at different frequencies make the separation task learnable from scratch in minutes. Real data (LRS2, VoxCeleb2) would require torchaudio + face detection preprocessing.

## References

- [CTCNet](https://arxiv.org/abs/2212.10744) — audio-visual speech separation
- [AV-HuBERT](https://arxiv.org/abs/2208.02455) — audio-visual speech representation
- [Conv-TasNet](https://arxiv.org/abs/1809.07454) — mask-based waveform separation
- [TF-GridNet](https://arxiv.org/abs/2211.12433) — state-of-the-art STFT-domain separation
