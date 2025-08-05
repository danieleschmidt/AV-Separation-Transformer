"""
Comprehensive Logging Configuration for AV-Separation-Transformer
Structured logging, error tracking, and audit trails
"""

import logging
import logging.handlers
import json
import traceback
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading

from pythonjsonlogger import jsonlogger


class StructuredFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional context
    """
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            now = datetime.utcnow().isoformat() + 'Z'
            log_record['timestamp'] = now
        
        # Add level name
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        
        # Add service info
        log_record['service'] = 'av-separation-transformer'
        log_record['version'] = self._get_version()
        
        # Add thread info
        log_record['thread_id'] = threading.get_ident()
        log_record['thread_name'] = threading.current_thread().name
        
        # Add process info
        import os
        log_record['process_id'] = os.getpid()
    
    def _get_version(self) -> str:
        """Get service version"""
        try:
            from ..version import __version__
            return __version__
        except ImportError:
            return 'unknown'


class AuditLogger:
    """
    Specialized logger for audit events
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audit logger
        self.logger = logging.getLogger('av_separation.audit')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add rotating file handler for audit logs
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'audit.log',
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        
        formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(service)s %(version)s %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.propagate = False
    
    def log_separation_request(
        self,
        user_id: str,
        client_ip: str,
        filename: str,
        num_speakers: int,
        file_size: int,
        duration: Optional[float] = None
    ):
        """Log separation request"""
        
        self.logger.info('separation_request', extra={
            'event_type': 'separation_request',
            'user_id': user_id,
            'client_ip': client_ip,
            'filename': filename,
            'num_speakers': num_speakers,
            'file_size': file_size,
            'duration': duration
        })
    
    def log_separation_completion(
        self,
        user_id: str,
        client_ip: str,
        filename: str,
        processing_time: float,
        success: bool,
        error_message: Optional[str] = None
    ):
        """Log separation completion"""
        
        extra_data = {
            'event_type': 'separation_completion',
            'user_id': user_id,
            'client_ip': client_ip,
            'filename': filename,
            'processing_time': processing_time,
            'success': success
        }
        
        if error_message:
            extra_data['error_message'] = error_message
        
        level = logging.INFO if success else logging.ERROR
        message = 'separation_completed' if success else 'separation_failed'
        
        self.logger.log(level, message, extra=extra_data)
    
    def log_authentication_attempt(
        self,
        user_id: Optional[str],
        client_ip: str,
        success: bool,
        auth_method: str,
        failure_reason: Optional[str] = None
    ):
        """Log authentication attempt"""
        
        extra_data = {
            'event_type': 'authentication_attempt',
            'user_id': user_id,
            'client_ip': client_ip,
            'success': success,
            'auth_method': auth_method
        }
        
        if failure_reason:
            extra_data['failure_reason'] = failure_reason
        
        level = logging.INFO if success else logging.WARNING
        message = 'authentication_success' if success else 'authentication_failure'
        
        self.logger.log(level, message, extra=extra_data)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        client_ip: str,
        description: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        
        extra_data = {
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'client_ip': client_ip,
            'description': description
        }
        
        if additional_data:
            extra_data.update(additional_data)
        
        # Map severity to log level
        severity_levels = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = severity_levels.get(severity.lower(), logging.WARNING)
        
        self.logger.log(level, 'security_event', extra=extra_data)
    
    def log_api_access(
        self,
        user_id: str,
        client_ip: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_agent: Optional[str] = None
    ):
        """Log API access"""
        
        extra_data = {
            'event_type': 'api_access',
            'user_id': user_id,
            'client_ip': client_ip,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time
        }
        
        if user_agent:
            extra_data['user_agent'] = user_agent
        
        # Determine log level based on status code
        if status_code < 400:
            level = logging.INFO
        elif status_code < 500:
            level = logging.WARNING
        else:
            level = logging.ERROR
        
        self.logger.log(level, 'api_access', extra=extra_data)


class ErrorTracker:
    """
    Error tracking and reporting system
    """
    
    def __init__(self, log_dir: Path, enable_sentry: bool = False, sentry_dsn: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_sentry = enable_sentry
        
        # Setup error logger
        self.logger = logging.getLogger('av_separation.errors')
        self.logger.setLevel(logging.ERROR)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add rotating file handler for errors
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5
        )
        
        formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(service)s %(version)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.propagate = False
        
        # Setup Sentry if enabled
        if enable_sentry and sentry_dsn:
            self._setup_sentry(sentry_dsn)
    
    def _setup_sentry(self, sentry_dsn: str):
        """Setup Sentry error tracking"""
        
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
            
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[sentry_logging],
                traces_sample_rate=0.1,
                environment='production'  # Set based on your environment
            )
            
            self.logger.info("Sentry error tracking initialized")
            
        except ImportError:
            self.logger.warning("Sentry SDK not available")
    
    def track_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Track error with full context
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
            user_id: User ID if available
            request_id: Request ID if available
        """
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        
        if user_id:
            error_data['user_id'] = user_id
        
        if request_id:
            error_data['request_id'] = request_id
        
        self.logger.error('error_occurred', extra=error_data)
        
        # Send to Sentry if enabled
        if self.enable_sentry:
            try:
                import sentry_sdk
                
                with sentry_sdk.configure_scope() as scope:
                    scope.set_tag("component", context.get('component', 'unknown'))
                    scope.set_context("error_context", context)
                    
                    if user_id:
                        scope.user = {"id": user_id}
                    
                    if request_id:
                        scope.set_tag("request_id", request_id)
                
                sentry_sdk.capture_exception(error)
                
            except Exception as sentry_error:
                self.logger.error(f"Failed to send error to Sentry: {sentry_error}")
    
    def track_performance_issue(
        self,
        operation: str,
        duration: float,
        threshold: float,
        context: Dict[str, Any]
    ):
        """Track performance issues"""
        
        if duration <= threshold:
            return
        
        perf_data = {
            'event_type': 'performance_issue',
            'operation': operation,
            'duration': duration,
            'threshold': threshold,
            'slowdown_factor': duration / threshold,
            'context': context
        }
        
        self.logger.warning('performance_issue', extra=perf_data)


def setup_logging(
    log_dir: str = './logs',
    log_level: str = 'INFO',
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True,
    enable_error_tracking: bool = True,
    enable_sentry: bool = False,
    sentry_dsn: Optional[str] = None
) -> Dict[str, Any]:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_audit: Enable audit logging
        enable_error_tracking: Enable error tracking
        enable_sentry: Enable Sentry error tracking
        sentry_dsn: Sentry DSN for error tracking
        
    Returns:
        Dictionary with logger instances
    """
    
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup root logger
    root_logger = logging.getLogger('av_separation')
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        # Use structured format for console in production, simple format for development
        if log_level.upper() == 'DEBUG':
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = StructuredFormatter(
                '%(timestamp)s %(level)s %(service)s %(name)s %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # File handler
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir_path / 'application.log',
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        file_handler.setLevel(numeric_level)
        
        file_formatter = StructuredFormatter(
            '%(timestamp)s %(level)s %(service)s %(version)s %(name)s %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False
    
    # Setup specialized loggers
    loggers = {'main': root_logger}
    
    # Audit logger
    if enable_audit:
        audit_logger = AuditLogger(log_dir_path)
        loggers['audit'] = audit_logger
    
    # Error tracker
    if enable_error_tracking:
        error_tracker = ErrorTracker(
            log_dir_path,
            enable_sentry=enable_sentry,
            sentry_dsn=sentry_dsn
        )
        loggers['error_tracker'] = error_tracker
    
    # Configure third-party loggers
    _configure_third_party_loggers(numeric_level)
    
    # Log startup message
    root_logger.info('Logging system initialized', extra={
        'log_dir': str(log_dir_path),
        'log_level': log_level,
        'console_enabled': enable_console,
        'file_enabled': enable_file,
        'audit_enabled': enable_audit,
        'error_tracking_enabled': enable_error_tracking,
        'sentry_enabled': enable_sentry
    })
    
    return loggers


def _configure_third_party_loggers(level: int):
    """Configure third-party library loggers"""
    
    # Suppress verbose third-party logs
    third_party_loggers = [
        'urllib3',
        'requests',
        'boto3',
        'botocore',
        'PIL',
        'matplotlib',
        'numba'
    ]
    
    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(max(level, logging.WARNING))
    
    # Special handling for some loggers
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)


class LoggingContext:
    """
    Context manager for adding logging context
    """
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with consistent naming
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    
    return logging.getLogger(f'av_separation.{name}')


# Decorator for automatic error tracking
def track_errors(component: str, operation: str = None):
    """
    Decorator for automatic error tracking
    
    Args:
        component: Component name
        operation: Operation name (defaults to function name)
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            logger = get_logger(component)
            
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                context = {
                    'component': component,
                    'operation': op_name,
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                logger.error(
                    f'Error in {component}.{op_name}',
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'context': context
                    }
                )
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator


# Global logger instances (to be initialized)
_loggers: Optional[Dict[str, Any]] = None


def initialize_logging(**kwargs) -> Dict[str, Any]:
    """Initialize global logging"""
    
    global _loggers
    _loggers = setup_logging(**kwargs)
    return _loggers


def get_audit_logger() -> Optional[AuditLogger]:
    """Get audit logger instance"""
    
    if _loggers and 'audit' in _loggers:
        return _loggers['audit']
    return None


def get_error_tracker() -> Optional[ErrorTracker]:
    """Get error tracker instance"""
    
    if _loggers and 'error_tracker' in _loggers:
        return _loggers['error_tracker']
    return None