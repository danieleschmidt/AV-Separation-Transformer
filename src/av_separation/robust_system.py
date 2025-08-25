"""
Generation 2: Robust Audio-Visual Separation System
Enhanced with comprehensive error handling, security, monitoring and reliability features.
"""
import torch
import numpy as np
import logging
import traceback
import hashlib
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import psutil
import threading
from contextlib import contextmanager
import warnings

from .separator import AVSeparator
from .config import SeparatorConfig


@dataclass
class SystemMetrics:
    """System performance and health metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    gpu_memory: Optional[float]
    inference_latency: float
    error_count: int
    uptime: float
    throughput: float


class SecurityManager:
    """Security manager for input validation and threat detection"""
    
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_formats = {'.mp4', '.avi', '.mov', '.wav', '.mp3', '.m4a'}
        self.suspicious_patterns = ['exec', 'eval', '__import__', 'subprocess']
        
    def validate_input_file(self, file_path: str) -> bool:
        """Validate input file for security"""
        try:
            path = Path(file_path)
            
            # Check file exists and is readable
            if not path.exists() or not path.is_file():
                raise ValueError(f"Invalid file path: {file_path}")
            
            # Check file size
            if path.stat().st_size > self.max_file_size:
                raise ValueError(f"File too large: {path.stat().st_size} bytes")
            
            # Check file extension
            if path.suffix.lower() not in self.allowed_formats:
                raise ValueError(f"Unsupported format: {path.suffix}")
            
            return True
            
        except Exception as e:
            logging.error(f"Security validation failed: {e}")
            return False
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data"""
        if isinstance(data, str):
            # Remove potentially dangerous patterns
            for pattern in self.suspicious_patterns:
                if pattern in data.lower():
                    raise ValueError(f"Suspicious pattern detected: {pattern}")
        
        return data


class ErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.backoff_base = 1.0
        
    def handle_error(self, error: Exception, context: str) -> bool:
        """Handle errors with retry logic"""
        error_key = f"{type(error).__name__}:{context}"
        
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        logging.error(f"Error in {context}: {error}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        
        # Check if we should retry
        if self.error_counts[error_key] <= self.max_retries:
            wait_time = self.backoff_base * (2 ** (self.error_counts[error_key] - 1))
            logging.info(f"Retrying in {wait_time}s (attempt {self.error_counts[error_key]})")
            time.sleep(wait_time)
            return True
        
        logging.error(f"Max retries exceeded for {error_key}")
        return False
    
    @contextmanager
    def handle_errors(self, context: str):
        """Context manager for error handling"""
        try:
            yield
        except Exception as e:
            self.handle_error(e, context)
            raise


class HealthMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        self.last_health_check = 0
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        gpu_usage = None
        gpu_memory = None
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_usage()
            except:
                pass
        
        uptime = time.time() - self.start_time
        throughput = self.total_requests / max(uptime, 1.0)
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            inference_latency=0.0,  # Will be updated by benchmark
            error_count=self.error_count,
            uptime=uptime,
            throughput=throughput
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        metrics = self.get_system_metrics()
        
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'metrics': metrics.__dict__,
            'issues': []
        }
        
        # Check for issues
        if metrics.cpu_usage > 90:
            health_status['issues'].append('High CPU usage')
            health_status['status'] = 'warning'
        
        if metrics.memory_usage > 85:
            health_status['issues'].append('High memory usage')
            health_status['status'] = 'warning'
        
        if metrics.error_count > 10:
            health_status['issues'].append('High error rate')
            health_status['status'] = 'critical'
        
        self.last_health_check = time.time()
        return health_status


class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self):
        self.audio_sample_rates = [8000, 16000, 22050, 44100, 48000]
        self.video_fps_range = (1, 120)
        self.max_duration = 300  # 5 minutes
        
    def validate_audio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Validate audio input"""
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be numpy array")
        
        if len(audio.shape) != 1:
            raise ValueError("Audio must be mono (1D array)")
        
        if sample_rate not in self.audio_sample_rates:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        
        duration = len(audio) / sample_rate
        if duration > self.max_duration:
            raise ValueError(f"Audio too long: {duration}s (max: {self.max_duration}s)")
        
        # Check for NaN or infinite values
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or infinite values")
        
        return True
    
    def validate_video(self, video: np.ndarray) -> bool:
        """Validate video input"""
        if not isinstance(video, np.ndarray):
            raise TypeError("Video must be numpy array")
        
        if len(video.shape) != 4:  # (frames, height, width, channels)
            raise ValueError("Video must be 4D array (frames, height, width, channels)")
        
        frames, height, width, channels = video.shape
        
        if channels != 3:
            raise ValueError("Video must have 3 channels (RGB)")
        
        if frames < 1:
            raise ValueError("Video must have at least 1 frame")
        
        # Check for reasonable dimensions
        if height < 32 or width < 32 or height > 2160 or width > 3840:
            raise ValueError(f"Invalid video dimensions: {height}x{width}")
        
        # Check for valid pixel values
        if video.dtype == np.uint8:
            if video.min() < 0 or video.max() > 255:
                raise ValueError("Invalid pixel values for uint8")
        elif video.dtype == np.float32:
            if video.min() < 0 or video.max() > 1:
                raise ValueError("Invalid pixel values for float32")
        
        return True


class RobustAVSeparator:
    """Robust Audio-Visual Separator with comprehensive error handling and security"""
    
    def __init__(self, config: Optional[SeparatorConfig] = None, enable_monitoring: bool = True):
        self.config = config or SeparatorConfig()
        
        # Initialize security and error handling
        self.security_manager = SecurityManager()
        self.error_handler = ErrorHandler()
        self.health_monitor = HealthMonitor() if enable_monitoring else None
        self.input_validator = InputValidator()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize core separator with error handling
        self.separator = None
        self._initialize_separator()
        
        # Performance tracking
        self.request_times = []
        self.lock = threading.Lock()
        
        logging.info("RobustAVSeparator initialized successfully")
    
    def _setup_logging(self):
        """Configure comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('av_separator.log', mode='a')
            ]
        )
        
        # Suppress some verbose warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    def _initialize_separator(self):
        """Initialize core separator with error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.separator = AVSeparator(
                    num_speakers=self.config.model.max_speakers,
                    device=self.config.inference.device,
                    config=self.config
                )
                logging.info(f"Core separator initialized (attempt {attempt + 1})")
                return
                
            except Exception as e:
                logging.error(f"Failed to initialize separator (attempt {attempt + 1}): {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError("Failed to initialize separator after all attempts")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def separate_audio_visual(self, 
                            audio: np.ndarray, 
                            video: np.ndarray, 
                            sample_rate: int = 16000) -> Dict[str, Any]:
        """Robust separation with comprehensive error handling"""
        
        start_time = time.time()
        request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        
        try:
            with self.error_handler.handle_errors(f"separation_request_{request_id}"):
                # Input validation
                logging.info(f"[{request_id}] Starting separation request")
                
                # Security checks
                audio = self.security_manager.sanitize_input(audio)
                video = self.security_manager.sanitize_input(video)
                
                # Input validation
                self.input_validator.validate_audio(audio, sample_rate)
                self.input_validator.validate_video(video)
                
                # Perform separation
                separated_audio = self.separator.separate_stream(audio, video[0])  # Use first frame
                
                # Track performance
                inference_time = time.time() - start_time
                with self.lock:
                    self.request_times.append(inference_time)
                    if self.health_monitor:
                        self.health_monitor.total_requests += 1
                
                logging.info(f"[{request_id}] Separation completed in {inference_time:.2f}s")
                
                return {
                    'request_id': request_id,
                    'separated_audio': separated_audio,
                    'inference_time': inference_time,
                    'num_speakers': separated_audio.shape[0] if separated_audio is not None else 0,
                    'status': 'success'
                }
                
        except Exception as e:
            if self.health_monitor:
                self.health_monitor.error_count += 1
                
            logging.error(f"[{request_id}] Separation failed: {e}")
            
            return {
                'request_id': request_id,
                'separated_audio': None,
                'inference_time': time.time() - start_time,
                'num_speakers': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def benchmark_system(self, num_iterations: int = 10) -> Dict[str, float]:
        """Comprehensive system benchmarking"""
        if not self.separator:
            raise RuntimeError("Separator not initialized")
        
        try:
            # Run core benchmark
            results = self.separator.benchmark(num_iterations)
            
            # Add system metrics
            if self.health_monitor:
                metrics = self.health_monitor.get_system_metrics()
                results.update({
                    'system_cpu_usage': metrics.cpu_usage,
                    'system_memory_usage': metrics.memory_usage,
                    'system_uptime': metrics.uptime
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            return {'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        if not self.health_monitor:
            return {'status': 'monitoring_disabled'}
        
        return self.health_monitor.health_check()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        with self.lock:
            if not self.request_times:
                return {}
            
            times = np.array(self.request_times[-100:])  # Last 100 requests
            
            return {
                'mean_response_time': float(np.mean(times)),
                'median_response_time': float(np.median(times)),
                'p95_response_time': float(np.percentile(times, 95)),
                'p99_response_time': float(np.percentile(times, 99)),
                'total_requests': len(self.request_times),
                'recent_requests': len(times)
            }
    
    def reset_metrics(self):
        """Reset all metrics and counters"""
        with self.lock:
            self.request_times.clear()
            if self.health_monitor:
                self.health_monitor.error_count = 0
                self.health_monitor.total_requests = 0
                self.health_monitor.start_time = time.time()
        
        logging.info("Metrics reset successfully")


def create_robust_system(config: Optional[SeparatorConfig] = None) -> RobustAVSeparator:
    """Factory function to create robust AV separator system"""
    return RobustAVSeparator(config=config, enable_monitoring=True)


if __name__ == "__main__":
    # Demo the robust system
    print("🛡️ Generation 2: Robust Audio-Visual Separation System")
    
    # Create robust system
    robust_separator = create_robust_system()
    
    # Test with dummy data
    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_video = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    
    # Run separation
    result = robust_separator.separate_audio_visual(dummy_audio, dummy_video)
    print(f"✅ Separation result: {result['status']}")
    
    # Get health status
    health = robust_separator.get_health_status()
    print(f"✅ System health: {health['status']}")
    
    # Run benchmark
    benchmark = robust_separator.benchmark_system(num_iterations=3)
    print(f"✅ Benchmark: {benchmark.get('mean_latency_ms', 'N/A')}ms average latency")
    
    print("🌟 Generation 2: ROBUST system operational")