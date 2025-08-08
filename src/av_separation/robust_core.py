"""
Robust Core System for AV-Separation-Transformer
Enhanced error handling, validation, and resilience
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional, Union, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ErrorSeverity(Enum):
    """Error severity levels for robust error handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    VALIDATION = "validation"
    PROCESSING = "processing"
    SECURITY = "security"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


@dataclass
class RobustError:
    """Comprehensive error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    trace_id: Optional[str] = None
    component: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp,
            'trace_id': self.trace_id,
            'component': self.component
        }


class RobustValidator:
    """Comprehensive input and data validation"""
    
    @staticmethod
    def validate_audio_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audio configuration parameters"""
        errors = []
        
        if 'sample_rate' in config:
            if not isinstance(config['sample_rate'], int) or config['sample_rate'] <= 0:
                errors.append("sample_rate must be positive integer")
        
        if 'n_fft' in config:
            if not isinstance(config['n_fft'], int) or config['n_fft'] <= 0:
                errors.append("n_fft must be positive integer")
                
        if 'n_mels' in config:
            if not isinstance(config['n_mels'], int) or config['n_mels'] <= 0:
                errors.append("n_mels must be positive integer")
                
        if errors:
            raise ValueError(f"Audio configuration validation failed: {', '.join(errors)}")
            
        return config
    
    @staticmethod
    def validate_video_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video configuration parameters"""
        errors = []
        
        if 'fps' in config:
            if not isinstance(config['fps'], int) or config['fps'] <= 0:
                errors.append("fps must be positive integer")
        
        if 'max_faces' in config:
            if not isinstance(config['max_faces'], int) or config['max_faces'] <= 0:
                errors.append("max_faces must be positive integer")
                
        if 'image_size' in config:
            if not isinstance(config['image_size'], (tuple, list)) or len(config['image_size']) != 2:
                errors.append("image_size must be tuple/list of 2 integers")
                
        if errors:
            raise ValueError(f"Video configuration validation failed: {', '.join(errors)}")
            
        return config
    
    @staticmethod
    def validate_file_content(content: bytes, allowed_types: List[str] = None) -> bool:
        """Validate file content and type"""
        if not content:
            raise ValueError("Empty file content")
            
        if len(content) > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("File size exceeds 100MB limit")
            
        # Basic file type detection
        if allowed_types:
            file_signatures = {
                'mp4': [b'\x00\x00\x00\x18ftypmp4', b'\x00\x00\x00\x20ftypmp4'],
                'wav': [b'RIFF', b'WAVE'],
                'avi': [b'RIFF', b'AVI '],
                'mov': [b'\x00\x00\x00\x14ftyp']
            }
            
            valid_signature = False
            for file_type in allowed_types:
                if file_type in file_signatures:
                    for signature in file_signatures[file_type]:
                        if content.startswith(signature):
                            valid_signature = True
                            break
            
            if not valid_signature:
                raise ValueError(f"File type not allowed. Supported: {allowed_types}")
        
        return True


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e


class RetryHandler:
    """Advanced retry mechanism with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry_async(self, func: Callable, *args, **kwargs):
        """Async retry with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
                else:
                    break
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs):
        """Synchronous retry with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    time.sleep(delay)
                else:
                    break
        
        raise last_exception


class RobustLogger:
    """Enhanced logging system with structured logging"""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(f"av_separation.{component}")
        self.error_count = 0
        self.warning_count = 0
        
    def log_error(self, error: RobustError) -> str:
        """Log structured error information"""
        self.error_count += 1
        trace_id = error.trace_id or self._generate_trace_id()
        
        log_entry = {
            'trace_id': trace_id,
            'component': self.component,
            'error': error.to_dict()
        }
        
        self.logger.error(f"Error in {self.component}: {json.dumps(log_entry)}")
        return trace_id
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warning with context"""
        self.warning_count += 1
        log_entry = {
            'component': self.component,
            'message': message,
            'context': context or {}
        }
        
        self.logger.warning(json.dumps(log_entry))
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """Log informational message with context"""
        log_entry = {
            'component': self.component,
            'message': message,
            'context': context or {}
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics"""
        return {
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID for error tracking"""
        return hashlib.md5(f"{time.time()}{self.component}".encode()).hexdigest()[:8]


class HealthChecker:
    """System health monitoring and checks"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = None
        self.health_status = 'unknown'
        
    def register_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_check': None
        }
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        critical_failed = False
        
        for name, check_info in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_info['func']):
                    result = await check_info['func']()
                else:
                    result = check_info['func']()
                    
                results[name] = {
                    'status': 'healthy',
                    'result': result,
                    'timestamp': time.time()
                }
                check_info['last_result'] = True
                
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': time.time()
                }
                check_info['last_result'] = False
                
                if check_info['critical']:
                    critical_failed = True
            
            check_info['last_check'] = time.time()
        
        self.last_check_time = time.time()
        self.health_status = 'critical' if critical_failed else 'healthy'
        
        return {
            'overall_status': self.health_status,
            'checks': results,
            'timestamp': self.last_check_time
        }


def robust_error_handler(
    component: str,
    category: ErrorCategory = ErrorCategory.PROCESSING,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator for comprehensive error handling"""
    def decorator(func: Callable):
        logger = RobustLogger(component)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error = RobustError(
                    category=category,
                    severity=severity,
                    message=str(e),
                    context={
                        'function': func.__name__,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100],
                        'traceback': traceback.format_exc()[:1000]
                    },
                    timestamp=time.time(),
                    component=component
                )
                
                trace_id = logger.log_error(error)
                
                if severity == ErrorSeverity.CRITICAL:
                    raise Exception(f"Critical error in {component} (trace: {trace_id}): {str(e)}")
                else:
                    raise Exception(f"Error in {component} (trace: {trace_id}): {str(e)}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = RobustError(
                    category=category,
                    severity=severity,
                    message=str(e),
                    context={
                        'function': func.__name__,
                        'args': str(args)[:100],
                        'kwargs': str(kwargs)[:100],
                        'traceback': traceback.format_exc()[:1000]
                    },
                    timestamp=time.time(),
                    component=component
                )
                
                trace_id = logger.log_error(error)
                
                if severity == ErrorSeverity.CRITICAL:
                    raise Exception(f"Critical error in {component} (trace: {trace_id}): {str(e)}")
                else:
                    raise Exception(f"Error in {component} (trace: {trace_id}): {str(e)}")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def robust_context(component: str, operation: str):
    """Context manager for robust operation tracking"""
    logger = RobustLogger(component)
    start_time = time.time()
    
    logger.log_info(f"Starting {operation}", {'start_time': start_time})
    
    try:
        yield logger
        duration = time.time() - start_time
        logger.log_info(f"Completed {operation}", {
            'duration': duration,
            'status': 'success'
        })
    except Exception as e:
        duration = time.time() - start_time
        error = RobustError(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            message=str(e),
            context={
                'operation': operation,
                'duration': duration,
                'traceback': traceback.format_exc()[:1000]
            },
            timestamp=time.time(),
            component=component
        )
        logger.log_error(error)
        raise


class RobustConfig:
    """Robust configuration management with validation"""
    
    def __init__(self):
        self.validator = RobustValidator()
        self.logger = RobustLogger('config')
    
    def validate_and_load(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and load configuration with robust error handling"""
        with robust_context('config', 'validation'):
            validated_config = {}
            
            if 'audio' in config_dict:
                validated_config['audio'] = self.validator.validate_audio_config(config_dict['audio'])
            
            if 'video' in config_dict:
                validated_config['video'] = self.validator.validate_video_config(config_dict['video'])
                
            return validated_config


# Health checks for system components
def check_memory_usage() -> Dict[str, float]:
    """Check system memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    except ImportError:
        return {'status': 'psutil not available'}


def check_disk_space() -> Dict[str, float]:
    """Check disk space"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent_used': (disk.used / disk.total) * 100
        }
    except ImportError:
        return {'status': 'psutil not available'}


# Initialize global health checker
global_health_checker = HealthChecker()
global_health_checker.register_check('memory', check_memory_usage, critical=True)
global_health_checker.register_check('disk', check_disk_space, critical=True)

# Global circuit breakers for critical operations
audio_processing_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
video_processing_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
model_inference_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)