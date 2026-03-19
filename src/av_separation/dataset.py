"""
SyntheticAVDataset

Generates synthetic audio-visual data for training/testing the separation model.

Audio model:
  - Speaker 1: sine wave at freq1 Hz (e.g. 220 Hz)
  - Speaker 2: sine wave at freq2 Hz (e.g. 440 Hz)
  - Mixed audio = sum of both

Video model (lip frames):
  - Each speaker's "lip activity" is a 2D grayscale patch
  - Pixel intensity is correlated with the speaker's instantaneous voice energy
  - This provides a weak but learnable visual cue for separation
"""

import math
import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticAVDataset(Dataset):
    """
    Synthetic audio-visual speech separation dataset.

    Each sample contains:
        mixed_spec  : (freq_bins, T)     — STFT magnitude of mixed signal
        lip_frames  : (num_frames, H, W) — simulated lip-movement ROI frames
        clean_specs : (num_speakers, freq_bins, T)  — ground-truth per-speaker specs
    """

    def __init__(
        self,
        num_samples: int = 1000,
        sample_rate: int = 8000,
        duration: float = 1.0,          # seconds
        n_fft: int = 512,
        hop_length: int = 128,
        num_frames: int = 25,           # video FPS equivalent per clip
        frame_h: int = 32,
        frame_w: int = 32,
        speaker_freqs: tuple = (220.0, 440.0),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.speaker_freqs = speaker_freqs
        self.num_speakers = len(speaker_freqs)
        self.rng = np.random.default_rng(seed)

        # Pre-compute time axis
        self.num_samples_audio = int(sample_rate * duration)
        self.t = np.linspace(0, duration, self.num_samples_audio, endpoint=False)

        # Compute expected STFT output size
        self.freq_bins = n_fft // 2 + 1
        # T_frames = 1 + floor(num_samples_audio / hop_length)
        self.T = 1 + self.num_samples_audio // hop_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx)  # deterministic per index

        # --- Random amplitude per speaker ---
        amps = rng.uniform(0.3, 1.0, size=self.num_speakers)

        # --- Generate clean audio per speaker ---
        clean_audios = []
        for i, (freq, amp) in enumerate(zip(self.speaker_freqs, amps)):
            # Add small random frequency jitter for variety
            freq_jitter = freq * rng.uniform(0.95, 1.05)
            phase = rng.uniform(0, 2 * math.pi)
            signal = amp * np.sin(2 * math.pi * freq_jitter * self.t + phase)
            clean_audios.append(signal.astype(np.float32))

        mixed_audio = sum(clean_audios).astype(np.float32)

        # --- STFT ---
        mixed_spec = self._stft(mixed_audio)       # (freq_bins, T)
        clean_specs = [self._stft(a) for a in clean_audios]  # list of (freq_bins, T)

        # --- Lip frames ---
        # For each speaker, compute frame-level energy
        # Frame i covers audio samples [i * step : (i+1) * step]
        step = self.num_samples_audio // self.num_frames
        lip_frames_all = []
        for spk_audio in clean_audios:
            frames = []
            for fi in range(self.num_frames):
                start = fi * step
                end = min(start + step, self.num_samples_audio)
                energy = float(np.mean(spk_audio[start:end] ** 2))
                # Create a 32x32 "lip patch": center region brightness ∝ energy
                frame = self._make_lip_frame(energy, rng)
                frames.append(frame)
            lip_frames_all.append(np.stack(frames, axis=0))  # (num_frames, H, W)

        # Stack speakers: (num_speakers, num_frames, H, W)
        # For the model we need (num_frames, H, W) — use speaker 0's lip frames as the
        # "guide" signal. The visual cue tells us where speaker 0 is active so the model
        # can use that to separate. (In a real system each speaker would have their own
        # lip-tracking stream.)
        # We concatenate both speakers' frames along frame dimension so the model sees
        # visual activity from all speakers: (2 * num_frames, H, W)
        lip_frames = np.concatenate(lip_frames_all, axis=0)  # (2*num_frames, H, W)

        return {
            "mixed_spec":  torch.from_numpy(mixed_spec),
            "lip_frames":  torch.from_numpy(lip_frames),
            "clean_specs": torch.from_numpy(np.stack(clean_specs, axis=0)),
        }

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT magnitude spectrogram."""
        window = np.hanning(self.n_fft)
        specs = []
        for i in range(self.T):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = np.zeros(self.n_fft, dtype=np.float32)
            chunk = audio[start:end] if end <= len(audio) else audio[start:]
            frame[:len(chunk)] = chunk
            frame *= window
            spectrum = np.abs(np.fft.rfft(frame))
            specs.append(spectrum)
        return np.stack(specs, axis=-1).astype(np.float32)  # (freq_bins, T)

    def _make_lip_frame(self, energy: float, rng: np.random.Generator) -> np.ndarray:
        """
        Create a synthetic lip-movement frame.
        The center region is brighter when the speaker is active.
        """
        frame = np.zeros((self.frame_h, self.frame_w), dtype=np.float32)
        # Lip region: center 50% of the frame
        h_start, h_end = self.frame_h // 4, 3 * self.frame_h // 4
        w_start, w_end = self.frame_w // 4, 3 * self.frame_w // 4

        # Brightness correlated with energy (clamped to [0, 1])
        brightness = min(1.0, energy * 20.0)  # scale factor for visual range
        noise = rng.normal(0, 0.05, (h_end - h_start, w_end - w_start)).astype(np.float32)
        frame[h_start:h_end, w_start:w_end] = np.clip(brightness + noise, 0, 1)
        return frame
