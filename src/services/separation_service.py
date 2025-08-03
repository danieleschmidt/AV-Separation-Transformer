"""High-level separation service with business logic."""

from typing import Dict, List, Optional, Union, Any
import logging
import time
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from ..models import AudioVisualSeparator
from ..utils import AudioProcessor, VideoProcessor, ModelUtils
from ..utils.validators import validate_separation_request
from ..utils.metrics import SeparationMetrics

logger = logging.getLogger(__name__)


class SeparationService:
    """High-level service for audio-visual speech separation.
    
    This service provides a business logic layer on top of the core
    separation models, handling:
    - Request validation and preprocessing
    - Model management and optimization
    - Performance monitoring and metrics
    - Async processing and scaling
    - Error handling and recovery
    
    Example:
        >>> service = SeparationService(
        ...     model_config={'num_speakers': 2, 'device': 'cuda'},
        ...     enable_metrics=True
        ... )
        >>> result = await service.separate_async(
        ...     audio_path='meeting.wav',
        ...     video_path='meeting.mp4'
        ... )
    """
    
    def __init__(
        self,
        model_config: Optional[Dict] = None,
        enable_metrics: bool = True,
        max_workers: int = 4,
        cache_models: bool = True,
        performance_mode: str = 'balanced'  # 'speed', 'quality', 'balanced'
    ):
        self.model_config = model_config or {
            'num_speakers': 2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_amp': True
        }
        
        self.enable_metrics = enable_metrics
        self.max_workers = max_workers
        self.cache_models = cache_models
        self.performance_mode = performance_mode
        
        # Initialize components
        self._models_cache = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        if enable_metrics:
            self.metrics = SeparationMetrics()
        
        # Load default model
        self.separator = self._get_or_create_model(self.model_config)
        
        logger.info(f"SeparationService initialized with {self.model_config}")
    
    def _get_or_create_model(self, config: Dict) -> AudioVisualSeparator:
        """Get cached model or create new one."""
        config_key = str(sorted(config.items()))
        
        if self.cache_models and config_key in self._models_cache:
            return self._models_cache[config_key]
        
        # Adjust config based on performance mode
        if self.performance_mode == 'speed':
            config = {**config, 'use_amp': True, 'chunk_size': 1.0}
        elif self.performance_mode == 'quality':
            config = {**config, 'use_amp': False, 'chunk_size': 4.0}
        
        model = AudioVisualSeparator(**config)
        
        if self.cache_models:
            self._models_cache[config_key] = model
        
        return model
    
    async def separate_async(
        self,
        audio_input: Union[str, Path, torch.Tensor, np.ndarray],
        video_input: Union[str, Path, torch.Tensor, np.ndarray],
        output_dir: Optional[str] = None,
        return_tensors: bool = False,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """Asynchronously separate audio-visual content.
        
        Args:
            audio_input: Audio file path or tensor
            video_input: Video file path or tensor
            output_dir: Directory to save separated tracks
            return_tensors: Whether to return raw tensors
            quality_check: Whether to perform quality assessment
            
        Returns:
            Dictionary containing separated tracks and metadata
        """
        start_time = time.time()
        
        try:
            # Validate request
            request_data = {
                'audio_input': audio_input,
                'video_input': video_input,
                'num_speakers': self.model_config.get('num_speakers', 2)
            }
            validate_separation_request(request_data)
            
            # Process in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._separate_sync,
                audio_input,
                video_input,
                output_dir,
                return_tensors,
                quality_check
            )
            
            # Add timing information
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['real_time_factor'] = result.get('duration', 0) / processing_time
            
            # Record metrics
            if self.enable_metrics:
                self.metrics.record_separation(
                    processing_time=processing_time,
                    num_speakers=self.model_config.get('num_speakers', 2),
                    quality_score=result.get('quality_metrics', {}).get('si_snr', 0)
                )
            
            logger.info(f"Separation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Async separation failed: {e}")
            if self.enable_metrics:
                self.metrics.record_error(str(e))
            raise
    
    def _separate_sync(
        self,
        audio_input,
        video_input,
        output_dir,
        return_tensors,
        quality_check
    ) -> Dict[str, Any]:
        """Synchronous separation implementation."""
        # Load and preprocess inputs
        audio_data, video_data = self._load_inputs(audio_input, video_input)
        
        # Perform separation
        separated_tracks = self.separator.separate(
            audio_data, 
            video_data,
            return_intermediate=quality_check
        )
        
        if quality_check and isinstance(separated_tracks, tuple):
            tracks, intermediates = separated_tracks
        else:
            tracks = separated_tracks
            intermediates = None
        
        # Prepare result
        result = {
            'num_speakers': len(tracks),
            'duration': self._estimate_duration(audio_data),
            'success': True
        }
        
        # Handle outputs
        if return_tensors:
            result['separated_tracks'] = tracks
        
        if output_dir:
            output_paths = self._save_tracks(tracks, output_dir)
            result['output_files'] = output_paths
        
        # Quality assessment
        if quality_check and intermediates:
            quality_metrics = self._assess_quality(tracks, intermediates)
            result['quality_metrics'] = quality_metrics
        
        return result
    
    def _load_inputs(self, audio_input, video_input):
        """Load and validate audio/video inputs."""
        # Handle file paths
        if isinstance(audio_input, (str, Path)):
            audio_processor = AudioProcessor()
            audio_data = audio_processor.load_audio(audio_input)
        else:
            audio_data = audio_input
        
        if isinstance(video_input, (str, Path)):
            video_processor = VideoProcessor()
            video_data = video_processor.load_video(video_input)
        else:
            video_data = video_input
        
        return audio_data, video_data
    
    def _estimate_duration(self, audio_data) -> float:
        """Estimate audio duration in seconds."""
        if isinstance(audio_data, torch.Tensor):
            # Assume last dimension is time
            samples = audio_data.shape[-1]
            return samples / self.model_config.get('sample_rate', 16000)
        return 0.0
    
    def _save_tracks(self, tracks: List[torch.Tensor], output_dir: str) -> List[str]:
        """Save separated tracks to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for i, track in enumerate(tracks):
            output_path = output_dir / f"speaker_{i:02d}.wav"
            
            # Convert tensor to audio file
            audio_processor = AudioProcessor()
            audio_processor.save_audio(track, output_path)
            
            output_paths.append(str(output_path))
        
        return output_paths
    
    def _assess_quality(self, tracks: List[torch.Tensor], intermediates: Dict) -> Dict:
        """Assess separation quality using various metrics."""
        quality_metrics = {}
        
        # Compute SI-SNR (placeholder implementation)
        if len(tracks) >= 2:
            # This would typically require ground truth for accurate measurement
            # For now, we provide estimated metrics based on separation characteristics
            signal_power = sum(torch.mean(track ** 2) for track in tracks)
            noise_estimate = torch.var(torch.stack(tracks), dim=0).mean()
            
            si_snr_estimate = 10 * torch.log10(signal_power / (noise_estimate + 1e-8))
            quality_metrics['si_snr_estimate'] = float(si_snr_estimate)
        
        # Cross-talk analysis
        if 'speaker_masks' in intermediates:
            masks = intermediates['speaker_masks']
            # Analyze mask overlap (lower is better)
            mask_overlap = torch.mean(torch.min(masks, dim=1)[0])
            quality_metrics['cross_talk_estimate'] = float(mask_overlap)
        
        # Attention analysis
        if 'cross_modal_attention' in intermediates:
            attn_weights = intermediates['cross_modal_attention']
            # Higher attention variance suggests better audio-visual correlation
            if 'audio_to_video' in attn_weights:
                attn_var = torch.var(attn_weights['audio_to_video'])
                quality_metrics['av_correlation'] = float(attn_var)
        
        return quality_metrics
    
    def separate_batch(
        self,
        batch_requests: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Process multiple separation requests in batch.
        
        Args:
            batch_requests: List of request dictionaries
            
        Returns:
            List of separation results
        """
        results = []
        
        for request in batch_requests:
            try:
                result = self._separate_sync(
                    audio_input=request['audio_input'],
                    video_input=request['video_input'],
                    output_dir=request.get('output_dir'),
                    return_tensors=request.get('return_tensors', False),
                    quality_check=request.get('quality_check', True)
                )
                result['request_id'] = request.get('id')
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                results.append({
                    'request_id': request.get('id'),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def optimize_model(
        self,
        optimization_type: str = 'quantization',
        target_device: str = 'cpu'
    ) -> None:
        """Optimize model for specific deployment scenarios.
        
        Args:
            optimization_type: 'quantization', 'pruning', 'distillation'
            target_device: Target deployment device
        """
        logger.info(f"Optimizing model with {optimization_type} for {target_device}")
        
        if optimization_type == 'quantization':
            self._apply_quantization(target_device)
        elif optimization_type == 'pruning':
            self._apply_pruning()
        elif optimization_type == 'distillation':
            self._apply_distillation()
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
    
    def _apply_quantization(self, target_device: str) -> None:
        """Apply model quantization."""
        if target_device == 'cpu':
            # Dynamic quantization for CPU
            quantized_model = torch.quantization.quantize_dynamic(
                self.separator.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            self.separator.model = quantized_model
            logger.info("Applied dynamic quantization for CPU")
        else:
            logger.warning(f"Quantization not implemented for {target_device}")
    
    def _apply_pruning(self) -> None:
        """Apply structured pruning to reduce model size."""
        # This would implement pruning logic
        logger.info("Pruning optimization not yet implemented")
    
    def _apply_distillation(self) -> None:
        """Apply knowledge distillation for model compression."""
        # This would implement distillation logic
        logger.info("Distillation optimization not yet implemented")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        if not self.enable_metrics:
            return {'error': 'Metrics not enabled'}
        
        return self.metrics.get_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform service health check."""
        try:
            # Test with dummy data
            dummy_audio = torch.randn(1, 1, 16000)  # 1 second of audio
            dummy_video = torch.randn(1, 25, 256)   # 1 second of video features
            
            start_time = time.time()
            result = self.separator.separate(dummy_audio, dummy_video)
            processing_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'device': str(self.separator.device),
                'test_processing_time': processing_time,
                'model_info': self.separator.get_model_info()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        
        # Clear model cache
        if self.cache_models:
            for model in self._models_cache.values():
                del model
            self._models_cache.clear()
        
        logger.info("SeparationService cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup