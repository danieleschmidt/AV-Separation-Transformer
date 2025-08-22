import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import time
from .separator import AVSeparator
from .config import SeparatorConfig
from .utils.audio import AudioProcessor
from .utils.video import VideoProcessor


@dataclass
class SeparationResult:
    """Enhanced separation result with metadata."""
    separated_audio: List[np.ndarray]
    speaker_confidence: List[float]
    processing_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, any]


class EnhancedAVSeparator(AVSeparator):
    """Enhanced Audio-Visual Separator with additional features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_enhanced_features()
    
    def _setup_enhanced_features(self):
        """Initialize enhanced features."""
        self.processing_stats = {
            'total_processed': 0,
            'average_processing_time': 0.0,
            'success_rate': 1.0,
            'quality_scores': []
        }
        
        self.confidence_threshold = 0.7
        self.adaptive_quality = True
        
    def separate_enhanced(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        return_confidence: bool = True,
        quality_assessment: bool = True
    ) -> SeparationResult:
        """Enhanced separation with confidence scores and quality metrics."""
        
        start_time = time.time()
        input_path = Path(input_path)
        
        try:
            # Load and process media
            audio_waveform, sample_rate = self.audio_processor.load_audio(str(input_path))
            video_frames = self.video_processor.load_video(str(input_path))
            
            # Perform separation
            separated_audio = self._enhanced_process(audio_waveform, video_frames)
            
            # Calculate confidence scores
            speaker_confidence = self._calculate_confidence(separated_audio) if return_confidence else []
            
            # Quality assessment
            quality_metrics = self._assess_quality(separated_audio, audio_waveform) if quality_assessment else {}
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, quality_metrics)
            
            # Save outputs if requested
            if output_dir:
                self._save_enhanced_outputs(separated_audio, output_dir, input_path.stem, quality_metrics)
            
            return SeparationResult(
                separated_audio=separated_audio,
                speaker_confidence=speaker_confidence,
                processing_time=processing_time,
                quality_metrics=quality_metrics,
                metadata={
                    'input_file': str(input_path),
                    'num_speakers': len(separated_audio),
                    'sample_rate': sample_rate,
                    'duration': len(audio_waveform) / sample_rate
                }
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced separation failed: {e}")
            self._update_stats(time.time() - start_time, {}, success=False)
            raise
    
    def _enhanced_process(self, audio_waveform: np.ndarray, video_frames: np.ndarray) -> List[np.ndarray]:
        """Enhanced processing with adaptive quality."""
        
        # Adaptive chunking based on complexity
        chunk_size = self._adaptive_chunk_size(audio_waveform)
        
        # Process with enhanced algorithms
        audio_chunks = self._chunk_audio_adaptive(audio_waveform, chunk_size)
        video_chunks = self._chunk_video_adaptive(video_frames, chunk_size)
        
        separated_chunks = []
        for audio_chunk, video_chunk in zip(audio_chunks, video_chunks):
            # Enhanced processing with confidence-based refinement
            separated_chunk = self._process_chunk_enhanced(audio_chunk, video_chunk)
            separated_chunks.append(separated_chunk)
        
        return self._merge_chunks_enhanced(separated_chunks)
    
    def _adaptive_chunk_size(self, audio_waveform: np.ndarray) -> float:
        """Determine optimal chunk size based on audio complexity."""
        # Simple complexity measure - can be enhanced with spectral analysis
        complexity = np.std(audio_waveform)
        
        if complexity > 0.1:  # High complexity
            return self.config.audio.chunk_duration * 0.8
        elif complexity < 0.05:  # Low complexity
            return self.config.audio.chunk_duration * 1.2
        else:
            return self.config.audio.chunk_duration
    
    def _calculate_confidence(self, separated_audio: List[np.ndarray]) -> List[float]:
        """Calculate confidence scores for each separated speaker."""
        confidence_scores = []
        
        for audio in separated_audio:
            # Energy-based confidence
            energy = np.mean(audio ** 2)
            
            # Spectral consistency (simplified)
            spectral_var = np.var(np.abs(np.fft.fft(audio)))
            
            # Combine metrics
            confidence = min(1.0, energy * 2 + (1.0 / (1.0 + spectral_var)))
            confidence_scores.append(confidence)
        
        return confidence_scores
    
    def _assess_quality(self, separated_audio: List[np.ndarray], original_audio: np.ndarray) -> Dict[str, float]:
        """Assess separation quality metrics."""
        if not separated_audio:
            return {}
        
        metrics = {}
        
        # Energy preservation
        original_energy = np.mean(original_audio ** 2)
        separated_energy = sum(np.mean(audio ** 2) for audio in separated_audio)
        metrics['energy_preservation'] = min(1.0, separated_energy / (original_energy + 1e-8))
        
        # Signal-to-noise ratio estimate
        if len(separated_audio) > 1:
            primary_signal = separated_audio[0]
            residual = original_audio[:len(primary_signal)] - primary_signal
            snr = 10 * np.log10(np.mean(primary_signal ** 2) / (np.mean(residual ** 2) + 1e-8))
            metrics['estimated_snr_db'] = max(0, min(30, snr))
        
        # Spectral quality
        for i, audio in enumerate(separated_audio):
            spectrum = np.abs(np.fft.fft(audio))
            spectral_centroid = np.sum(spectrum * np.arange(len(spectrum))) / (np.sum(spectrum) + 1e-8)
            metrics[f'speaker_{i}_spectral_quality'] = min(1.0, spectral_centroid / len(spectrum))
        
        return metrics
    
    def _update_stats(self, processing_time: float, quality_metrics: Dict, success: bool = True):
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update success rate
        if success:
            success_count = self.processing_stats['success_rate'] * (total - 1) + 1
        else:
            success_count = self.processing_stats['success_rate'] * (total - 1)
        self.processing_stats['success_rate'] = success_count / total
        
        # Update quality scores
        if 'estimated_snr_db' in quality_metrics:
            self.processing_stats['quality_scores'].append(quality_metrics['estimated_snr_db'])
            # Keep only last 100 scores for memory efficiency
            if len(self.processing_stats['quality_scores']) > 100:
                self.processing_stats['quality_scores'].pop(0)
    
    def get_performance_report(self) -> Dict[str, any]:
        """Get comprehensive performance report."""
        stats = self.processing_stats.copy()
        
        if stats['quality_scores']:
            stats['average_quality_snr'] = np.mean(stats['quality_scores'])
            stats['quality_std'] = np.std(stats['quality_scores'])
        
        return {
            'processing_statistics': stats,
            'model_info': {
                'device': self.device,
                'num_speakers': self.num_speakers,
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
            },
            'configuration': {
                'chunk_duration': self.config.audio.chunk_duration,
                'sample_rate': self.config.audio.sample_rate,
                'confidence_threshold': self.confidence_threshold
            }
        }
    
    def _chunk_audio_adaptive(self, waveform: np.ndarray, chunk_duration: float) -> List[torch.Tensor]:
        """Adaptive audio chunking."""
        sample_rate = self.config.audio.sample_rate
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = chunk_samples // 2
        
        chunks = []
        for start in range(0, len(waveform) - chunk_samples + 1, hop_samples):
            chunk = waveform[start:start + chunk_samples]
            chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
            chunks.append(chunk_tensor)
        
        # Handle remaining audio
        if len(chunks) == 0 or len(waveform) > (len(chunks) - 1) * hop_samples + chunk_samples:
            remaining = waveform[len(chunks) * hop_samples:]
            if len(remaining) > chunk_samples // 4:  # Only if significant remaining audio
                # Pad to chunk size
                padded = np.pad(remaining, (0, max(0, chunk_samples - len(remaining))), 'constant')
                chunks.append(torch.from_numpy(padded[:chunk_samples]).float().to(self.device))
        
        return chunks if chunks else [torch.from_numpy(waveform).float().to(self.device)]
    
    def _chunk_video_adaptive(self, frames: np.ndarray, chunk_duration: float) -> List[torch.Tensor]:
        """Adaptive video chunking."""
        fps = self.config.video.fps
        chunk_frames = int(chunk_duration * fps)
        hop_frames = chunk_frames // 2
        
        chunks = []
        for start in range(0, len(frames) - chunk_frames + 1, hop_frames):
            chunk = frames[start:start + chunk_frames]
            chunk_tensor = torch.from_numpy(chunk).float().to(self.device)
            chunks.append(chunk_tensor)
        
        # Handle remaining frames
        if len(chunks) == 0 or len(frames) > (len(chunks) - 1) * hop_frames + chunk_frames:
            remaining = frames[len(chunks) * hop_frames:]
            if len(remaining) > chunk_frames // 4:
                # Repeat last frame to pad
                if len(remaining) < chunk_frames:
                    last_frame = remaining[-1:]
                    padding_needed = chunk_frames - len(remaining)
                    padding = np.repeat(last_frame, padding_needed, axis=0)
                    padded = np.concatenate([remaining, padding], axis=0)
                else:
                    padded = remaining[:chunk_frames]
                chunks.append(torch.from_numpy(padded).float().to(self.device))
        
        return chunks if chunks else [torch.from_numpy(frames).float().to(self.device)]
    
    def _process_chunk_enhanced(self, audio_chunk: torch.Tensor, video_chunk: torch.Tensor) -> torch.Tensor:
        """Enhanced chunk processing with confidence refinement."""
        with torch.no_grad():
            # Add batch dimension if needed
            if audio_chunk.dim() == 1:
                audio_chunk = audio_chunk.unsqueeze(0)
            if video_chunk.dim() == 3:
                video_chunk = video_chunk.unsqueeze(0)
            
            # Process through model
            separated = self.model(audio_chunk, video_chunk)
            
            # Post-processing refinement based on confidence
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights()
                # Use attention weights to refine separation
                # This is a placeholder for attention-based refinement
                pass
            
            return separated
    
    def _merge_chunks_enhanced(self, separated_chunks: List[torch.Tensor]) -> List[np.ndarray]:
        """Enhanced chunk merging with overlap handling."""
        if not separated_chunks:
            return []
        
        # Determine number of speakers from first chunk
        first_chunk = separated_chunks[0]
        if first_chunk.dim() == 3:  # [batch, speakers, time]
            num_speakers = first_chunk.shape[1]
        else:
            num_speakers = 1
            separated_chunks = [chunk.unsqueeze(1) for chunk in separated_chunks]
        
        # Initialize output arrays
        chunk_samples = separated_chunks[0].shape[-1]
        hop_samples = chunk_samples // 2
        total_samples = (len(separated_chunks) - 1) * hop_samples + chunk_samples
        
        merged_speakers = []
        for speaker_idx in range(num_speakers):
            merged_audio = np.zeros(total_samples)
            overlap_weights = np.zeros(total_samples)
            
            for chunk_idx, chunk in enumerate(separated_chunks):
                start_idx = chunk_idx * hop_samples
                end_idx = start_idx + chunk_samples
                
                chunk_audio = chunk[0, speaker_idx, :].cpu().numpy()
                
                # Apply overlap-add with windowing
                if chunk_idx == 0:
                    # First chunk - full weight
                    window = np.ones_like(chunk_audio)
                elif chunk_idx == len(separated_chunks) - 1:
                    # Last chunk - full weight
                    window = np.ones_like(chunk_audio)
                else:
                    # Middle chunks - apply Hann window for smooth transitions
                    window = np.hanning(len(chunk_audio))
                
                merged_audio[start_idx:end_idx] += chunk_audio * window
                overlap_weights[start_idx:end_idx] += window
            
            # Normalize by overlap weights
            overlap_weights[overlap_weights == 0] = 1  # Avoid division by zero
            merged_audio = merged_audio / overlap_weights
            
            merged_speakers.append(merged_audio)
        
        return merged_speakers
    
    def _save_enhanced_outputs(
        self,
        separated_audio: List[np.ndarray],
        output_dir: Union[str, Path],
        base_name: str,
        quality_metrics: Dict[str, float]
    ):
        """Save outputs with enhanced metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save separated audio files
        for i, audio in enumerate(separated_audio):
            output_path = output_dir / f"{base_name}_speaker_{i+1}.wav"
            self.audio_processor.save_audio(audio, output_path, self.config.audio.sample_rate)
        
        # Save quality report
        if quality_metrics:
            report_path = output_dir / f"{base_name}_quality_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump({
                    'quality_metrics': quality_metrics,
                    'processing_info': {
                        'num_speakers': len(separated_audio),
                        'total_duration': len(separated_audio[0]) / self.config.audio.sample_rate if separated_audio else 0
                    }
                }, f, indent=2)
