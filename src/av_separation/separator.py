import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import warnings
from .models import AVSeparationTransformer
from .config import SeparatorConfig
from .utils.audio import AudioProcessor
from .utils.video import VideoProcessor


class AVSeparator:
    def __init__(
        self,
        num_speakers: int = 2,
        device: str = None,
        checkpoint: Optional[str] = None,
        config: Optional[SeparatorConfig] = None
    ):
        self.num_speakers = num_speakers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config is None:
            config = SeparatorConfig()
        self.config = config
        self.config.model.max_speakers = max(num_speakers, config.model.max_speakers)
        
        self.model = AVSeparationTransformer(self.config)
        
        if checkpoint:
            self.load_checkpoint(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.audio_processor = AudioProcessor(self.config.audio)
        self.video_processor = VideoProcessor(self.config.video)
        
    def load_checkpoint(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}")
    
    def separate(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_video: bool = False
    ) -> List[np.ndarray]:
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Processing: {input_path}")
        
        audio_waveform, sample_rate = self.audio_processor.load_audio(str(input_path))
        video_frames = self.video_processor.load_video(str(input_path))
        
        audio_chunks = self._chunk_audio(audio_waveform, sample_rate)
        video_chunks = self._chunk_video(video_frames)
        
        separated_chunks = []
        for audio_chunk, video_chunk in zip(audio_chunks, video_chunks):
            separated_chunk = self._process_chunk(audio_chunk, video_chunk)
            separated_chunks.append(separated_chunk)
        
        separated_audio = self._merge_chunks(separated_chunks)
        
        if output_dir:
            self._save_outputs(separated_audio, output_dir, input_path.stem)
        
        return separated_audio
    
    def _chunk_audio(self, waveform: np.ndarray, sample_rate: int) -> List[torch.Tensor]:
        chunk_samples = int(self.config.audio.chunk_duration * sample_rate)
        hop_samples = chunk_samples // 2
        
        chunks = []
        for start in range(0, len(waveform) - chunk_samples + 1, hop_samples):
            chunk = waveform[start:start + chunk_samples]
            chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
            chunks.append(chunk_tensor)
        
        if len(chunks) == 0:
            chunks = [torch.from_numpy(waveform).float().to(self.device)]
        
        return chunks
    
    def _chunk_video(self, frames: np.ndarray) -> List[torch.Tensor]:
        chunk_frames = int(self.config.audio.chunk_duration * self.config.video.fps)
        hop_frames = chunk_frames // 2
        
        chunks = []
        for start in range(0, len(frames) - chunk_frames + 1, hop_frames):
            chunk = frames[start:start + chunk_frames]
            chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
            chunk_tensor = chunk_tensor.permute(0, 3, 1, 2) / 255.0
            chunks.append(chunk_tensor)
        
        if len(chunks) == 0:
            chunk_tensor = torch.from_numpy(frames).float().to(self.device)
            chunk_tensor = chunk_tensor.permute(0, 3, 1, 2) / 255.0
            chunks = [chunk_tensor]
        
        return chunks
    
    def _process_chunk(
        self,
        audio_chunk: torch.Tensor,
        video_chunk: torch.Tensor
    ) -> torch.Tensor:
        
        with torch.no_grad():
            audio_spec = self.model._compute_spectrogram(audio_chunk)
            
            outputs = self.model(
                audio_spec.unsqueeze(0),
                video_chunk.unsqueeze(0)
            )
            
            separated = outputs['separated_waveforms'].squeeze(0)
            
            speaker_scores = outputs['speaker_logits'].mean(dim=-1)
            top_k = torch.topk(speaker_scores.squeeze(0), self.num_speakers).indices
            separated = separated[top_k]
        
        return separated.cpu()
    
    def _merge_chunks(self, chunks: List[torch.Tensor]) -> List[np.ndarray]:
        if len(chunks) == 1:
            return [chunk.numpy() for chunk in chunks[0]]
        
        num_speakers = chunks[0].shape[0]
        merged = []
        
        for spk_idx in range(num_speakers):
            speaker_chunks = [chunk[spk_idx] for chunk in chunks]
            
            overlap_samples = speaker_chunks[0].shape[0] // 2
            merged_waveform = speaker_chunks[0][:overlap_samples]
            
            for i in range(len(speaker_chunks) - 1):
                curr_chunk = speaker_chunks[i]
                next_chunk = speaker_chunks[i + 1]
                
                fade_out = torch.linspace(1, 0, overlap_samples)
                fade_in = torch.linspace(0, 1, overlap_samples)
                
                overlap = (curr_chunk[-overlap_samples:] * fade_out + 
                          next_chunk[:overlap_samples] * fade_in)
                
                merged_waveform = torch.cat([
                    merged_waveform,
                    overlap,
                    next_chunk[overlap_samples:-overlap_samples if i < len(speaker_chunks) - 2 else None]
                ])
            
            merged.append(merged_waveform.numpy())
        
        return merged
    
    def _save_outputs(
        self,
        separated_audio: List[np.ndarray],
        output_dir: Union[str, Path],
        stem: str
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, audio in enumerate(separated_audio):
            output_path = output_dir / f"{stem}_speaker_{i+1}.wav"
            self.audio_processor.save_audio(audio, str(output_path))
            print(f"Saved: {output_path}")
    
    def separate_stream(
        self,
        audio_stream: np.ndarray,
        video_frame: np.ndarray
    ) -> np.ndarray:
        
        audio_tensor = torch.from_numpy(audio_stream).float().to(self.device)
        frame_tensor = torch.from_numpy(video_frame).float().to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        
        with torch.no_grad():
            audio_spec = self.model._compute_spectrogram(audio_tensor)
            
            outputs = self.model(
                audio_spec.unsqueeze(0),
                frame_tensor.unsqueeze(0).unsqueeze(0)
            )
            
            separated = outputs['separated_waveforms'].squeeze(0)
            
            speaker_scores = outputs['speaker_logits'].mean(dim=-1)
            top_k = torch.topk(speaker_scores.squeeze(0), self.num_speakers).indices
            separated = separated[top_k]
        
        return separated.cpu().numpy()
    
    @torch.no_grad()
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance"""
        
        # Create dummy inputs matching model expectations
        dummy_audio = torch.randn(
            1, self.config.audio.n_mels, 
            int(self.config.audio.chunk_duration * self.config.audio.sample_rate / self.config.audio.hop_length)
        ).to(self.device)
        
        dummy_video = torch.randn(
            1, int(self.config.audio.chunk_duration * self.config.video.fps),
            3, *self.config.video.image_size
        ).to(self.device)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        import time
        latencies = []
        
        # Warm-up runs
        for _ in range(10):
            _ = self.model(dummy_audio, dummy_video)
        
        # Benchmark runs
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            _ = self.model(dummy_audio, dummy_video)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'rtf': float(self.config.audio.chunk_duration * 1000 / np.mean(latencies))
        }