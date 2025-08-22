import asyncio
import threading
import time
from typing import Callable, Optional, Dict, Any, List
import numpy as np
import torch
from queue import Queue, Empty
from dataclasses import dataclass
import logging
from collections import deque
from .enhanced_separation import EnhancedAVSeparator
from .config import SeparatorConfig


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming."""
    buffer_duration_ms: int = 100
    latency_target_ms: int = 50
    chunk_size_ms: int = 20
    max_queue_size: int = 10
    auto_adjust_quality: bool = True
    enable_vad: bool = True  # Voice Activity Detection
    

class RealtimeEngine:
    """Real-time audio-visual separation engine."""
    
    def __init__(
        self,
        separator: EnhancedAVSeparator,
        streaming_config: Optional[StreamingConfig] = None,
        callback: Optional[Callable] = None
    ):
        self.separator = separator
        self.streaming_config = streaming_config or StreamingConfig()
        self.callback = callback
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Threading components
        self.audio_queue = Queue(maxsize=self.streaming_config.max_queue_size)
        self.video_queue = Queue(maxsize=self.streaming_config.max_queue_size)
        self.output_queue = Queue(maxsize=self.streaming_config.max_queue_size)
        
        self.processing_thread = None
        self.is_running = False
        self.stats_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'average_latency_ms': 0.0,
            'dropped_frames': 0,
            'processing_fps': 0.0,
            'quality_adjustments': 0
        }
        
        # Adaptive quality control
        self.latency_buffer = deque(maxlen=100)
        self.last_adjustment_time = 0
        self.quality_level = 1.0  # 0.5 to 1.0
        
        # Voice Activity Detection
        if self.streaming_config.enable_vad:
            self._setup_vad()
    
    def _setup_vad(self):
        """Setup Voice Activity Detection."""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            self.vad_enabled = True
            self.logger.info("Voice Activity Detection enabled")
        except ImportError:
            self.logger.warning("webrtcvad not available, VAD disabled")
            self.vad_enabled = False
    
    def start_streaming(self):
        """Start the real-time streaming engine."""
        if self.is_running:
            self.logger.warning("Streaming already started")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Real-time streaming engine started")
    
    def stop_streaming(self):
        """Stop the real-time streaming engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Clear queues
        self._clear_queues()
        
        self.logger.info("Real-time streaming engine stopped")
    
    def push_audio_frame(self, audio_frame: np.ndarray, timestamp: float):
        """Push an audio frame for processing."""
        try:
            frame_data = {
                'audio': audio_frame,
                'timestamp': timestamp,
                'frame_id': self.performance_stats['frames_processed']
            }
            
            # Voice Activity Detection
            if self.vad_enabled and self._is_speech(audio_frame):
                frame_data['has_speech'] = True
            else:
                frame_data['has_speech'] = False
            
            self.audio_queue.put_nowait(frame_data)
            
        except:
            # Queue full - drop frame
            with self.stats_lock:
                self.performance_stats['dropped_frames'] += 1
            self.logger.debug("Dropped audio frame due to full queue")
    
    def push_video_frame(self, video_frame: np.ndarray, timestamp: float):
        """Push a video frame for processing."""
        try:
            frame_data = {
                'video': video_frame,
                'timestamp': timestamp,
                'frame_id': self.performance_stats['frames_processed']
            }
            self.video_queue.put_nowait(frame_data)
            
        except:
            # Queue full - drop frame
            with self.stats_lock:
                self.performance_stats['dropped_frames'] += 1
            self.logger.debug("Dropped video frame due to full queue")
    
    def get_separated_audio(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get the next separated audio result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        self.logger.info("Starting real-time processing loop")
        
        audio_buffer = []
        video_buffer = []
        last_fps_time = time.time()
        fps_counter = 0
        
        while self.is_running:
            try:
                # Get synchronized frames
                audio_frame = self._get_frame(self.audio_queue)
                video_frame = self._get_frame(self.video_queue)
                
                if audio_frame is None or video_frame is None:
                    continue
                
                # Check if frames are reasonably synchronized
                time_diff = abs(audio_frame['timestamp'] - video_frame['timestamp'])
                if time_diff > 0.1:  # 100ms threshold
                    self.logger.debug(f"Frame sync issue: {time_diff:.3f}s")
                    continue
                
                # Skip processing if no speech detected (VAD)
                if self.vad_enabled and not audio_frame.get('has_speech', True):
                    # Still output silence for consistency
                    silence_output = self._create_silence_output(audio_frame)
                    self._push_output(silence_output)
                    continue
                
                # Process frames
                start_time = time.time()
                result = self._process_synchronized_frames(audio_frame, video_frame)
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update performance stats
                self._update_performance_stats(processing_time)
                
                # Adaptive quality control
                if self.streaming_config.auto_adjust_quality:
                    self._adjust_quality_if_needed(processing_time)
                
                # Push result
                self._push_output(result)
                
                # FPS calculation
                fps_counter += 1
                if time.time() - last_fps_time >= 1.0:
                    with self.stats_lock:
                        self.performance_stats['processing_fps'] = fps_counter
                    fps_counter = 0
                    last_fps_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                continue
        
        self.logger.info("Real-time processing loop ended")
    
    def _get_frame(self, queue: Queue, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        """Get a frame from queue with timeout."""
        try:
            return queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _process_synchronized_frames(
        self, 
        audio_frame: Dict[str, Any], 
        video_frame: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process synchronized audio and video frames."""
        
        # Extract frame data
        audio_data = audio_frame['audio']
        video_data = video_frame['video']
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_data).float().to(self.separator.device)
        video_tensor = torch.from_numpy(video_data).float().to(self.separator.device)
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        if video_tensor.dim() == 3:
            video_tensor = video_tensor.unsqueeze(0)
        
        # Apply quality scaling if needed
        if self.quality_level < 1.0:
            audio_tensor, video_tensor = self._apply_quality_reduction(
                audio_tensor, video_tensor, self.quality_level
            )
        
        # Process through model
        with torch.no_grad():
            separated = self.separator.model(audio_tensor, video_tensor)
        
        # Convert back to numpy
        if separated.dim() == 3:  # [batch, speakers, time]
            separated_audio = [separated[0, i, :].cpu().numpy() for i in range(separated.shape[1])]
        else:
            separated_audio = [separated[0, :].cpu().numpy()]
        
        return {
            'separated_audio': separated_audio,
            'timestamp': audio_frame['timestamp'],
            'frame_id': audio_frame['frame_id'],
            'processing_quality': self.quality_level,
            'has_speech': audio_frame.get('has_speech', True)
        }
    
    def _is_speech(self, audio_frame: np.ndarray) -> bool:
        """Voice Activity Detection."""
        if not self.vad_enabled:
            return True
        
        try:
            # Convert to 16kHz, 16-bit PCM format required by webrtcvad
            if len(audio_frame.shape) > 1:
                audio_frame = np.mean(audio_frame, axis=1)
            
            # Ensure 16kHz sampling rate
            sample_rate = self.separator.config.audio.sample_rate
            if sample_rate != 16000:
                import librosa
                audio_frame = librosa.resample(audio_frame, orig_sr=sample_rate, target_sr=16000)
            
            # Convert to 16-bit PCM
            audio_pcm = (audio_frame * 32767).astype(np.int16).tobytes()
            
            # VAD expects frames of specific duration (10, 20, or 30 ms)
            frame_duration = 30  # ms
            samples_per_frame = 16000 * frame_duration // 1000
            
            # Pad or trim to exact frame size
            if len(audio_pcm) // 2 != samples_per_frame:
                audio_samples = np.frombuffer(audio_pcm, dtype=np.int16)
                if len(audio_samples) < samples_per_frame:
                    audio_samples = np.pad(audio_samples, (0, samples_per_frame - len(audio_samples)))
                else:
                    audio_samples = audio_samples[:samples_per_frame]
                audio_pcm = audio_samples.tobytes()
            
            return self.vad.is_speech(audio_pcm, 16000)
            
        except Exception as e:
            self.logger.debug(f"VAD error: {e}")
            return True  # Default to speech if VAD fails
    
    def _create_silence_output(self, audio_frame: Dict[str, Any]) -> Dict[str, Any]:
        """Create silent output for non-speech frames."""
        audio_length = len(audio_frame['audio'])
        num_speakers = self.separator.num_speakers
        
        silence = [np.zeros(audio_length) for _ in range(num_speakers)]
        
        return {
            'separated_audio': silence,
            'timestamp': audio_frame['timestamp'],
            'frame_id': audio_frame['frame_id'],
            'processing_quality': 1.0,
            'has_speech': False
        }
    
    def _apply_quality_reduction(
        self, 
        audio_tensor: torch.Tensor, 
        video_tensor: torch.Tensor, 
        quality_level: float
    ) -> tuple:
        """Apply quality reduction for performance."""
        
        # Reduce temporal resolution
        if quality_level < 0.8:
            # Skip every other sample/frame
            audio_tensor = audio_tensor[..., ::2]
            video_tensor = video_tensor[..., ::2, :, :]
        
        # Reduce spatial resolution for video
        if quality_level < 0.6:
            # Downsample video
            b, t, h, w = video_tensor.shape
            video_tensor = torch.nn.functional.interpolate(
                video_tensor.view(-1, 1, h, w),
                scale_factor=0.5,
                mode='bilinear'
            ).view(b, t, h//2, w//2)
        
        return audio_tensor, video_tensor
    
    def _update_performance_stats(self, processing_time_ms: float):
        """Update performance statistics."""
        with self.stats_lock:
            stats = self.performance_stats
            stats['frames_processed'] += 1
            
            # Update average latency
            total_frames = stats['frames_processed']
            current_avg = stats['average_latency_ms']
            stats['average_latency_ms'] = (
                (current_avg * (total_frames - 1) + processing_time_ms) / total_frames
            )
        
        # Add to latency buffer for adaptive control
        self.latency_buffer.append(processing_time_ms)
    
    def _adjust_quality_if_needed(self, current_latency_ms: float):
        """Adaptive quality control based on performance."""
        target_latency = self.streaming_config.latency_target_ms
        current_time = time.time()
        
        # Only adjust every 2 seconds to avoid oscillation
        if current_time - self.last_adjustment_time < 2.0:
            return
        
        # Calculate recent average latency
        if len(self.latency_buffer) < 10:
            return
        
        recent_avg_latency = np.mean(list(self.latency_buffer)[-20:])
        
        # Adjust quality based on latency
        if recent_avg_latency > target_latency * 1.5:  # Too slow
            if self.quality_level > 0.5:
                self.quality_level = max(0.5, self.quality_level - 0.1)
                with self.stats_lock:
                    self.performance_stats['quality_adjustments'] += 1
                self.logger.info(f"Reduced quality to {self.quality_level:.1f} due to high latency")
                
        elif recent_avg_latency < target_latency * 0.8:  # Can increase quality
            if self.quality_level < 1.0:
                self.quality_level = min(1.0, self.quality_level + 0.1)
                with self.stats_lock:
                    self.performance_stats['quality_adjustments'] += 1
                self.logger.info(f"Increased quality to {self.quality_level:.1f} due to low latency")
        
        self.last_adjustment_time = current_time
    
    def _push_output(self, result: Dict[str, Any]):
        """Push result to output queue and call callback if provided."""
        try:
            self.output_queue.put_nowait(result)
        except:
            # Queue full - drop oldest result
            try:
                self.output_queue.get_nowait()
                self.output_queue.put_nowait(result)
            except:
                pass
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(result)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def _clear_queues(self):
        """Clear all queues."""
        for queue in [self.audio_queue, self.video_queue, self.output_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.stats_lock:
            stats = self.performance_stats.copy()
        
        stats['current_quality_level'] = self.quality_level
        stats['queue_sizes'] = {
            'audio': self.audio_queue.qsize(),
            'video': self.video_queue.qsize(),
            'output': self.output_queue.qsize()
        }
        
        if len(self.latency_buffer) > 0:
            stats['recent_latency_ms'] = np.mean(list(self.latency_buffer)[-10:])
        
        return stats


class RealtimeWebRTCEngine(RealtimeEngine):
    """WebRTC-specific real-time engine with additional optimizations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.webrtc_optimizations = True
        self.packet_loss_compensation = True
    
    def handle_packet_loss(self, lost_frame_id: int):
        """Handle packet loss by interpolating missing frames."""
        if not self.packet_loss_compensation:
            return
        
        # Simple frame interpolation - can be enhanced
        self.logger.debug(f"Compensating for lost frame: {lost_frame_id}")
        
        # For now, just log the loss - more sophisticated interpolation
        # could be implemented based on previous and next frames
        with self.stats_lock:
            if 'packet_loss_events' not in self.performance_stats:
                self.performance_stats['packet_loss_events'] = 0
            self.performance_stats['packet_loss_events'] += 1
