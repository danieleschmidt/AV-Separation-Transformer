"""
Comprehensive Error Handling and Recovery System
Production-grade error management with graceful degradation.
"""

import torch
import traceback
import logging
import time
from typing import Optional, Dict, Any, Callable, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque


class ErrorSeverity(Enum):
    """Error severity levels for appropriate response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Container for error context information."""
    error_type: str
    severity: ErrorSeverity
    timestamp: float
    stack_trace: str
    input_shape: Optional[tuple]
    device_info: Optional[str]
    memory_usage: Optional[float]
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker OPEN. Service unavailable.")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e


class MemoryManager:
    """
    Intelligent memory management with automatic cleanup.
    """
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.cleanup_threshold = 0.85  # Cleanup at 85% usage
        
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0.0
    
    def get_memory_usage_ratio(self) -> float:
        """Get memory usage as ratio of maximum."""
        current_usage = self.get_gpu_memory_usage()
        return current_usage / self.max_memory_bytes
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def check_and_cleanup(self):
        """Check memory usage and cleanup if necessary."""
        if self.get_memory_usage_ratio() > self.cleanup_threshold:
            self.emergency_cleanup()
            
            # If still over threshold, raise warning
            if self.get_memory_usage_ratio() > 0.95:
                raise MemoryError(f"GPU memory usage critical: {self.get_memory_usage_ratio()*100:.1f}%")


class GracefulDegradation:
    """
    Implement graceful degradation strategies for various failure modes.
    """
    
    def __init__(self):
        self.fallback_strategies = {
            'cuda_oom': self._cuda_oom_fallback,
            'model_load_error': self._model_load_fallback,
            'inference_error': self._inference_fallback,
            'data_corruption': self._data_corruption_fallback
        }
        
        self.performance_counters = defaultdict(int)
        
    def _cuda_oom_fallback(self, func: Callable, *args, **kwargs):
        """Handle CUDA out-of-memory errors."""
        # Try reducing batch size
        if 'batch_size' in kwargs:
            original_batch_size = kwargs['batch_size']
            kwargs['batch_size'] = max(1, original_batch_size // 2)
            
            try:
                torch.cuda.empty_cache()
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Further reduce batch size
                    kwargs['batch_size'] = 1
                    torch.cuda.empty_cache()
                    return func(*args, **kwargs)
                raise
        
        # Try CPU fallback
        if 'device' in kwargs:
            kwargs['device'] = 'cpu'
            return func(*args, **kwargs)
        
        raise RuntimeError("Unable to recover from CUDA OOM")
    
    def _model_load_fallback(self, model_path: str, config: Any):
        """Handle model loading failures."""
        # Try different checkpoint formats
        fallback_paths = [
            model_path.replace('.pth', '.pt'),
            model_path.replace('.pth', '_backup.pth'),
            'models/default_model.pth'
        ]
        
        for path in fallback_paths:
            try:
                return torch.load(path, map_location='cpu')
            except (FileNotFoundError, RuntimeError):
                continue
        
        # Initialize from scratch with warning
        logging.warning("Could not load model, initializing from scratch")
        return None
    
    def _inference_fallback(self, model, audio_input, video_input):
        """Handle inference failures."""
        # Try audio-only inference
        try:
            logging.warning("Video processing failed, falling back to audio-only")
            return model.audio_only_inference(audio_input)
        except:
            pass
        
        # Try video-only inference
        try:
            logging.warning("Audio processing failed, falling back to video-only")
            return model.video_only_inference(video_input)
        except:
            pass
        
        # Return silence as last resort
        logging.error("All inference methods failed, returning silence")
        return torch.zeros_like(audio_input)
    
    def _data_corruption_fallback(self, data: torch.Tensor):
        """Handle corrupted input data."""
        # Check for NaN/Inf values
        if torch.isnan(data).any() or torch.isinf(data).any():
            logging.warning("Data corruption detected, applying fixes")
            
            # Replace NaN/Inf with zeros
            data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            data = torch.where(torch.isinf(data), torch.zeros_like(data), data)
        
        # Check for extreme values
        if torch.abs(data).max() > 100:
            logging.warning("Extreme values detected, applying normalization")
            data = torch.clamp(data, -10, 10)
        
        return data
    
    def apply_fallback(self, error_type: str, func: Callable, *args, **kwargs):
        """Apply appropriate fallback strategy."""
        if error_type in self.fallback_strategies:
            self.performance_counters[f"fallback_{error_type}"] += 1
            return self.fallback_strategies[error_type](func, *args, **kwargs)
        
        raise RuntimeError(f"No fallback strategy for error type: {error_type}")


class RobustErrorHandler:
    """
    Comprehensive error handling system with monitoring and recovery.
    """
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        self.circuit_breaker = CircuitBreaker()
        self.memory_manager = MemoryManager()
        self.graceful_degradation = GracefulDegradation()
        
        self.error_history = deque(maxlen=1000)
        self.recovery_stats = defaultdict(int)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
    def robust_execution(self, func: Callable, *args, **kwargs):
        """
        Execute function with comprehensive error handling.
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                # Memory check before execution
                self.memory_manager.check_and_cleanup()
                
                # Execute with circuit breaker
                result = self.circuit_breaker.call(func, *args, **kwargs)
                
                # Success - record and return
                if attempt > 0:
                    self.recovery_stats['successful_recoveries'] += 1
                    self.logger.info(f"Recovered after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                attempt += 1
                
                # Record error
                error_context = self._create_error_context(e, *args, **kwargs)
                self.error_history.append(error_context)
                
                # Determine error type and apply recovery
                error_type = self._classify_error(e)
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
                    
                    # Apply recovery strategy
                    try:
                        self._apply_recovery_strategy(error_type, e)
                        
                        # Exponential backoff
                        time.sleep(self.backoff_factor ** (attempt - 1))
                        
                        error_context.recovery_attempted = True
                        
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery failed: {recovery_error}")
                        break
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed")
        
        # All retries exhausted - try graceful degradation
        try:
            error_type = self._classify_error(last_exception)
            fallback_result = self.graceful_degradation.apply_fallback(
                error_type, func, *args, **kwargs
            )
            
            self.recovery_stats['fallback_recoveries'] += 1
            self.logger.warning("Graceful degradation applied successfully")
            
            return fallback_result
            
        except Exception as fallback_error:
            # Final failure - log comprehensive error report
            self._log_comprehensive_error_report(last_exception, fallback_error)
            raise last_exception
    
    def _create_error_context(self, error: Exception, *args, **kwargs) -> ErrorContext:
        """Create detailed error context."""
        input_shape = None
        device_info = None
        memory_usage = None
        
        # Extract input shape if available
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_shape = tuple(arg.shape)
                device_info = str(arg.device)
                break
        
        # Get memory usage
        try:
            memory_usage = self.memory_manager.get_gpu_memory_usage()
        except:
            pass
        
        return ErrorContext(
            error_type=type(error).__name__,
            severity=self._determine_severity(error),
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            input_shape=input_shape,
            device_info=device_info,
            memory_usage=memory_usage
        )
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        
        if "out of memory" in error_str or "cuda" in error_str:
            return "cuda_oom"
        elif "no such file" in error_str or "cannot load" in error_str:
            return "model_load_error"
        elif "runtime error" in error_str or "forward" in error_str:
            return "inference_error"
        elif "nan" in error_str or "inf" in error_str:
            return "data_corruption"
        else:
            return "unknown"
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity level."""
        error_str = str(error).lower()
        
        if "out of memory" in error_str or "cuda" in error_str:
            return ErrorSeverity.HIGH
        elif "file not found" in error_str:
            return ErrorSeverity.MEDIUM
        elif "warning" in error_str:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _apply_recovery_strategy(self, error_type: str, error: Exception):
        """Apply specific recovery strategy based on error type."""
        if error_type == "cuda_oom":
            self.memory_manager.emergency_cleanup()
            
        elif error_type == "model_load_error":
            # Clear any cached models
            torch.cuda.empty_cache()
            
        elif error_type == "data_corruption":
            # No specific recovery needed, will be handled in fallback
            pass
            
        elif error_type == "inference_error":
            # Reset model state if possible
            torch.cuda.empty_cache()
        
        self.recovery_stats[f"recovery_{error_type}"] += 1
    
    def _log_comprehensive_error_report(self, primary_error: Exception, 
                                       fallback_error: Exception):
        """Log comprehensive error report for debugging."""
        report = [
            "=" * 80,
            "COMPREHENSIVE ERROR REPORT",
            "=" * 80,
            f"Primary Error: {type(primary_error).__name__}: {primary_error}",
            f"Fallback Error: {type(fallback_error).__name__}: {fallback_error}",
            "",
            "Recent Error History:",
        ]
        
        for i, context in enumerate(list(self.error_history)[-5:]):
            report.append(f"  {i+1}. {context.error_type} at {context.timestamp} (Severity: {context.severity.value})")
        
        report.extend([
            "",
            "Recovery Statistics:",
        ])
        
        for key, value in self.recovery_stats.items():
            report.append(f"  {key}: {value}")
        
        report.extend([
            "",
            "Memory Status:",
            f"  GPU Memory Usage: {self.memory_manager.get_memory_usage_ratio()*100:.1f}%",
            "",
            "Circuit Breaker Status:",
            f"  State: {self.circuit_breaker.state}",
            f"  Failure Count: {self.circuit_breaker.failure_count}",
            "=" * 80
        ])
        
        error_report = "\n".join(report)
        self.logger.error(error_report)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        return {
            'error_count_last_hour': len([
                e for e in self.error_history 
                if time.time() - e.timestamp < 3600
            ]),
            'recovery_stats': dict(self.recovery_stats),
            'memory_usage_ratio': self.memory_manager.get_memory_usage_ratio(),
            'circuit_breaker_state': self.circuit_breaker.state,
            'circuit_breaker_failures': self.circuit_breaker.failure_count
        }


def robust_decorator(max_retries: int = 3, error_handler: Optional[RobustErrorHandler] = None):
    """
    Decorator for adding robust error handling to functions.
    """
    if error_handler is None:
        error_handler = RobustErrorHandler(max_retries=max_retries)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.robust_execution(func, *args, **kwargs)
        return wrapper
    
    return decorator


# Global error handler instance
global_error_handler = RobustErrorHandler()


@robust_decorator(max_retries=5)
def safe_model_inference(model, audio_input: torch.Tensor, 
                        video_input: torch.Tensor) -> torch.Tensor:
    """
    Example of robust model inference with comprehensive error handling.
    """
    return model(audio_input, video_input)


# Example usage patterns
if __name__ == "__main__":
    # Example: Using the robust decorator
    @robust_decorator(max_retries=3)
    def example_function(x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < 0.3:  # Simulate 30% failure rate
            raise RuntimeError("Random failure")
        return x * 2
    
    # Test the robust function
    test_tensor = torch.randn(10, 512)
    try:
        result = example_function(test_tensor)
        print(f"Success: {result.shape}")
    except Exception as e:
        print(f"Final failure: {e}")
    
    # Print health metrics
    metrics = global_error_handler.get_health_metrics()
    print(f"Health metrics: {metrics}")