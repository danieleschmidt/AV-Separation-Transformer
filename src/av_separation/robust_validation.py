import numpy as np
import torch
import cv2
from typing import Union, Tuple, Optional, Dict, List, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
import mimetypes
import json
from datetime import datetime
import warnings
from contextlib import contextmanager
import resource
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
import time


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    

class SecurityValidator:
    """Comprehensive security validation for inputs."""
    
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    ALLOWED_MIME_TYPES = {
        'audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/aac', 'audio/ogg',
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm', 'video/x-flv'
    }
    
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_DURATION = 3600  # 1 hour
    MIN_DURATION = 0.1   # 100ms
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.malware_signatures = self._load_malware_signatures()
    
    def _load_malware_signatures(self) -> set:
        """Load known malware signatures (simplified implementation)."""
        # In production, this would load from a comprehensive malware database
        return {
            # Common malicious patterns in audio/video files
            b'\x4d\x5a\x90\x00',  # PE header (executable disguised as media)
            b'\x7f\x45\x4c\x46',  # ELF header
            b'\xca\xfe\xba\xbe',  # Java class file
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Comprehensive file validation."""
        file_path = Path(file_path)
        errors = []
        warnings = []
        metadata = {}
        
        # Basic file checks
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return ValidationResult(False, errors, warnings, metadata)
        
        if not file_path.is_file():
            errors.append(f"Path is not a file: {file_path}")
            return ValidationResult(False, errors, warnings, metadata)
        
        # File size check
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            errors.append(f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE})")
        
        if file_size == 0:
            errors.append("File is empty")
        
        metadata['file_size'] = file_size
        
        # Extension validation
        extension = file_path.suffix.lower()
        if extension not in self.ALLOWED_AUDIO_EXTENSIONS and extension not in self.ALLOWED_VIDEO_EXTENSIONS:
            errors.append(f"Unsupported file extension: {extension}")
        
        # MIME type validation
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type not in self.ALLOWED_MIME_TYPES:
            warnings.append(f"Unexpected MIME type: {mime_type}")
        
        metadata['mime_type'] = mime_type
        metadata['extension'] = extension
        
        # Security checks
        security_result = self._security_scan(file_path)
        errors.extend(security_result['errors'])
        warnings.extend(security_result['warnings'])
        metadata.update(security_result['metadata'])
        
        # Media format validation
        if not errors:  # Only if basic checks pass
            format_result = self._validate_media_format(file_path)
            errors.extend(format_result['errors'])
            warnings.extend(format_result['warnings'])
            metadata.update(format_result['metadata'])
        
        return ValidationResult(len(errors) == 0, errors, warnings, metadata)
    
    def _security_scan(self, file_path: Path) -> Dict[str, List[str]]:
        """Perform security scan on file."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Read first few KB for signature analysis
            with open(file_path, 'rb') as f:
                header = f.read(8192)
            
            # Check for malware signatures
            for signature in self.malware_signatures:
                if signature in header:
                    errors.append("Malicious signature detected")
                    break
            
            # Check for suspicious patterns
            if b'<script' in header.lower():
                warnings.append("Script content detected in media file")
            
            if b'eval(' in header or b'exec(' in header:
                warnings.append("Suspicious code patterns detected")
            
            # File entropy analysis (simplified)
            entropy = self._calculate_entropy(header)
            if entropy > 7.8:  # Very high entropy might indicate encryption/packing
                warnings.append(f"High entropy detected: {entropy:.2f}")
            
            metadata['entropy'] = entropy
            metadata['file_hash'] = hashlib.sha256(header).hexdigest()[:16]
            
        except Exception as e:
            errors.append(f"Security scan failed: {e}")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        data_len = len(data)
        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _validate_media_format(self, file_path: Path) -> Dict[str, List[str]]:
        """Validate media file format and properties."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Try to open with OpenCV first (works for both audio and video)
            cap = cv2.VideoCapture(str(file_path))
            
            if cap.isOpened():
                # Video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if frame_count > 0 and fps > 0:
                    duration = frame_count / fps
                    
                    if duration > self.MAX_DURATION:
                        errors.append(f"Media too long: {duration:.1f}s (max: {self.MAX_DURATION}s)")
                    elif duration < self.MIN_DURATION:
                        errors.append(f"Media too short: {duration:.3f}s (min: {self.MIN_DURATION}s)")
                    
                    metadata.update({
                        'duration': duration,
                        'fps': fps,
                        'width': width,
                        'height': height,
                        'frame_count': int(frame_count)
                    })
                    
                    # Validate video dimensions
                    if width > 4096 or height > 4096:
                        warnings.append(f"Very high resolution: {width}x{height}")
                    elif width < 64 or height < 64:
                        warnings.append(f"Very low resolution: {width}x{height}")
                    
                    # Validate framerate
                    if fps > 120:
                        warnings.append(f"Very high framerate: {fps}")
                    elif fps < 5:
                        warnings.append(f"Very low framerate: {fps}")
                
                cap.release()
            else:
                # Try audio-specific validation
                try:
                    import librosa
                    y, sr = librosa.load(str(file_path), sr=None, duration=1.0)  # Load just 1 second
                    duration_estimate = librosa.get_duration(filename=str(file_path))
                    
                    if duration_estimate > self.MAX_DURATION:
                        errors.append(f"Audio too long: {duration_estimate:.1f}s")
                    elif duration_estimate < self.MIN_DURATION:
                        errors.append(f"Audio too short: {duration_estimate:.3f}s")
                    
                    metadata.update({
                        'duration': duration_estimate,
                        'sample_rate': sr,
                        'is_audio_only': True
                    })
                    
                except Exception as e:
                    errors.append(f"Cannot decode media file: {e}")
        
        except Exception as e:
            errors.append(f"Media format validation failed: {e}")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


class InputSanitizer:
    """Input sanitization and normalization."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def sanitize_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sanitize and normalize audio input."""
        metadata = {'original_shape': audio.shape}
        warnings = []
        
        # Handle different input formats
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            metadata['dtype_converted'] = True
        
        # Normalize to mono if stereo
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
            warnings.append("Converted stereo to mono")
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Clip extreme values
        max_val = np.max(np.abs(audio))
        if max_val > 10.0:  # Unusually large values
            audio = np.clip(audio, -1.0, 1.0)
            warnings.append(f"Clipped extreme values (max was {max_val:.2f})")
        
        # Normalize amplitude
        if max_val > 0:
            audio = audio / max_val
        
        # Remove NaN and inf values
        nan_count = np.sum(np.isnan(audio))
        inf_count = np.sum(np.isinf(audio))
        
        if nan_count > 0 or inf_count > 0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            warnings.append(f"Replaced {nan_count} NaN and {inf_count} inf values")
        
        metadata['warnings'] = warnings
        metadata['final_shape'] = audio.shape
        metadata['final_max'] = float(np.max(np.abs(audio)))
        
        return audio, metadata
    
    def sanitize_video(self, video: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sanitize and normalize video input."""
        metadata = {'original_shape': video.shape}
        warnings = []
        
        # Ensure proper dimensions [T, H, W, C] or [T, H, W]
        if len(video.shape) < 3:
            raise ValueError(f"Invalid video shape: {video.shape}")
        
        # Convert to float32 if needed
        if video.dtype == np.uint8:
            video = video.astype(np.float32) / 255.0
            metadata['normalized_from_uint8'] = True
        elif video.dtype != np.float32:
            video = video.astype(np.float32)
            metadata['dtype_converted'] = True
        
        # Handle grayscale vs RGB
        if len(video.shape) == 3:  # [T, H, W] - grayscale
            video = np.expand_dims(video, axis=-1)  # Add channel dimension
            metadata['added_channel_dim'] = True
        
        # Clip values to valid range
        video = np.clip(video, 0.0, 1.0)
        
        # Remove NaN and inf values
        nan_count = np.sum(np.isnan(video))
        inf_count = np.sum(np.isinf(video))
        
        if nan_count > 0 or inf_count > 0:
            video = np.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
            warnings.append(f"Replaced {nan_count} NaN and {inf_count} inf values")
        
        metadata['warnings'] = warnings
        metadata['final_shape'] = video.shape
        
        return video, metadata


class RobustErrorHandler:
    """Robust error handling with recovery strategies."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'recoverable_errors': 0,
            'fatal_errors': 0,
            'error_types': {},
            'recovery_strategies_used': {}
        }
    
    def retry_with_backoff(self, max_retries: Optional[int] = None):
        """Decorator for retry with exponential backoff."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        self._log_error(e, attempt, func.__name__)
                        
                        if attempt < retries:
                            sleep_time = self.backoff_factor ** attempt
                            self.logger.info(f"Retrying {func.__name__} in {sleep_time:.1f}s (attempt {attempt + 1}/{retries + 1})")
                            time.sleep(sleep_time)
                        else:
                            self._update_error_stats(e, recoverable=False)
                            raise
                
                # This should never be reached, but just in case
                raise last_exception
            
            return wrapper
        return decorator
    
    def handle_with_fallback(self, primary_func, fallback_func, *args, **kwargs):
        """Execute primary function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary function failed: {e}, trying fallback")
            self._update_error_stats(e, recoverable=True)
            return fallback_func(*args, **kwargs)
    
    def _log_error(self, error: Exception, attempt: int, func_name: str):
        """Log error with context."""
        error_type = type(error).__name__
        self.logger.error(
            f"Error in {func_name} (attempt {attempt + 1}): {error_type}: {error}"
        )
    
    def _update_error_stats(self, error: Exception, recoverable: bool = True):
        """Update error statistics."""
        error_type = type(error).__name__
        
        self.error_stats['total_errors'] += 1
        if recoverable:
            self.error_stats['recoverable_errors'] += 1
        else:
            self.error_stats['fatal_errors'] += 1
        
        if error_type not in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] = 0
        self.error_stats['error_types'][error_type] += 1
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report."""
        stats = self.error_stats.copy()
        
        if stats['total_errors'] > 0:
            stats['recovery_rate'] = stats['recoverable_errors'] / stats['total_errors']
            stats['most_common_error'] = max(
                stats['error_types'].items(), 
                key=lambda x: x[1],
                default=('None', 0)
            )[0]
        else:
            stats['recovery_rate'] = 1.0
            stats['most_common_error'] = 'None'
        
        return stats


class ResourceMonitor:
    """Monitor system resources and prevent resource exhaustion."""
    
    def __init__(
        self,
        max_memory_mb: int = 4096,
        max_cpu_percent: float = 80.0,
        check_interval: float = 1.0
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.check_interval = check_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.process = psutil.Process()
        self.monitoring = False
        self._monitor_thread = None
    
    @contextmanager
    def resource_limit_context(self):
        """Context manager for resource monitoring."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                # Check memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                if memory_mb > self.max_memory_mb:
                    self.logger.error(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                    # Could implement automatic cleanup or process termination here
                
                # Check CPU usage
                cpu_percent = self.process.cpu_percent()
                if cpu_percent > self.max_cpu_percent:
                    self.logger.warning(f"High CPU usage: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                break
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            memory_info = self.process.memory_info()
            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'cpu_percent': self.process.cpu_percent(),
                'num_threads': self.process.num_threads(),
                'open_files': len(self.process.open_files())
            }
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {e}")
            return {}


def timeout_handler(timeout_seconds: int):
    """Decorator to add timeout to function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator


class AuditLogger:
    """Comprehensive audit logging for security and compliance."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('audit')
        self.log_file = log_file
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)
    
    def log_access(self, user_id: str, resource: str, action: str, success: bool = True, **kwargs):
        """Log access attempt."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success,
            'additional_info': kwargs
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, json.dumps(log_entry))
    
    def log_processing(self, input_file: str, processing_time: float, success: bool = True, **kwargs):
        """Log processing event."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'processing',
            'input_file': input_file,
            'processing_time': processing_time,
            'success': success,
            'additional_info': kwargs
        }
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(level, json.dumps(log_entry))
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security-related event."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'security',
            'security_event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        level_map = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = level_map.get(severity.lower(), logging.INFO)
        self.logger.log(level, json.dumps(log_entry))
