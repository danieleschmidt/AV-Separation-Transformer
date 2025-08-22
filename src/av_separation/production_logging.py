import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


class ProductionLogger:
    """Production-ready logging system with structured logging."""
    
    def __init__(self, service_name: str = "av_separation", log_level: str = "INFO"):
        self.service_name = service_name
        self.log_level = getattr(logging, log_level.upper())
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers for production."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = Path(f"logs/{self.service_name}.log")
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log_structured("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log_structured("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log_structured("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self._log_structured("critical", message, **kwargs)
    
    def _log_structured(self, level: str, message: str, **kwargs):
        """Log structured message with additional context."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "message": message,
            "level": level,
            **kwargs
        }
        
        log_method = getattr(self.logger, level)
        log_method(json.dumps(log_data))


# Global logger instance
_logger = None


def get_production_logger(service_name: str = "av_separation") -> ProductionLogger:
    """Get or create production logger instance."""
    global _logger
    if _logger is None:
        _logger = ProductionLogger(service_name)
    return _logger


def log_request(endpoint: str, user_id: str, processing_time: float, status: str = "success"):
    """Log API request with performance metrics."""
    logger = get_production_logger()
    logger.info(
        f"API request completed",
        endpoint=endpoint,
        user_id=user_id,
        processing_time_ms=processing_time * 1000,
        status=status
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    logger = get_production_logger()
    logger.error(
        f"Error occurred: {str(error)}",
        error_type=type(error).__name__,
        context=context or {}
    )


def log_performance_metric(metric_name: str, value: float, **tags):
    """Log performance metric."""
    logger = get_production_logger()
    logger.info(
        f"Performance metric",
        metric_name=metric_name,
        value=value,
        **tags
    )
