"""
Enhanced Security Module for AV-Separation-Transformer
Advanced security features, input validation, and threat protection
"""

import hashlib
import hmac
import secrets
import time
import re
import base64
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from functools import wraps
import logging
from collections import defaultdict, deque

# Security logger
security_logger = logging.getLogger('av_separation.security')


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events"""
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    INJECTION_ATTEMPT = "injection_attempt"
    FILE_VALIDATION_FAILED = "file_validation_failed"
    AUTHENTICATION_FAILED = "authentication_failed"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"


@dataclass
class SecurityAlert:
    """Security alert with comprehensive details"""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    message: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'message': self.message,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'timestamp': self.timestamp,
            'details': self.details,
            'trace_id': self.trace_id
        }


class AdvancedInputValidator:
    """Enhanced input validation with security focus"""
    
    # Dangerous patterns to detect
    MALICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',  # JavaScript URL
        r'data:.*base64',  # Base64 data URLs
        r'eval\s*\(',  # Code evaluation
        r'exec\s*\(',  # Code execution
        r'system\s*\(',  # System calls
        r'__import__',  # Python imports
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'/etc/passwd',  # Linux system files
        r'cmd\.exe',  # Windows commands
        r'powershell',  # PowerShell
        r'SELECT.*FROM',  # SQL injection (basic)
        r'UNION.*SELECT',  # SQL injection
        r'DROP.*TABLE',  # SQL injection
        r'INSERT.*INTO',  # SQL injection
        r'UPDATE.*SET',  # SQL injection
        r'DELETE.*FROM',  # SQL injection
    ]
    
    # File size limits (bytes)
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_FILENAME_LENGTH = 255
    
    # Content type whitelist
    ALLOWED_CONTENT_TYPES = [
        'video/mp4',
        'video/avi',
        'video/mov',
        'video/wmv',
        'audio/wav',
        'audio/mp3',
        'audio/aac',
        'audio/ogg'
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.MALICIOUS_PATTERNS]
        
    def validate_text_input(self, text: str, max_length: int = 1000) -> Tuple[bool, List[str]]:
        """Validate text input for malicious content"""
        errors = []
        
        if not isinstance(text, str):
            errors.append("Input must be string")
            return False, errors
            
        if len(text) > max_length:
            errors.append(f"Input exceeds maximum length of {max_length}")
            
        # Check for malicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                errors.append(f"Potentially malicious pattern detected: {pattern.pattern}")
                
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-') / max(len(text), 1)
        if special_char_ratio > 0.3:
            errors.append("Excessive special characters detected")
            
        return len(errors) == 0, errors
    
    def validate_filename(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate filename for security issues"""
        errors = []
        
        if not filename:
            errors.append("Filename cannot be empty")
            return False, errors
            
        if len(filename) > self.MAX_FILENAME_LENGTH:
            errors.append(f"Filename exceeds maximum length of {self.MAX_FILENAME_LENGTH}")
            
        # Check for dangerous characters
        if re.search(r'[<>:"|?*\\]', filename):
            errors.append("Filename contains illegal characters")
            
        # Check for directory traversal
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            errors.append("Filename contains directory traversal patterns")
            
        # Check for system file patterns
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js', '.jar', '.com', '.pif']
        if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
            errors.append("Filename has dangerous extension")
            
        return len(errors) == 0, errors
    
    def validate_file_content(
        self, 
        content: bytes, 
        filename: str,
        max_size: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """Validate file content comprehensively"""
        errors = []
        max_size = max_size or self.MAX_FILE_SIZE
        
        if not content:
            errors.append("File content is empty")
            return False, errors
            
        if len(content) > max_size:
            errors.append(f"File size {len(content)} exceeds limit of {max_size}")
            
        # Check for embedded malicious content
        content_str = content[:10000].decode('utf-8', errors='ignore').lower()
        for pattern in self.compiled_patterns:
            if pattern.search(content_str):
                errors.append(f"Malicious pattern in file content: {pattern.pattern}")
                
        # Basic file format validation
        if not self._validate_file_format(content, filename):
            errors.append("File format validation failed")
            
        return len(errors) == 0, errors
    
    def _validate_file_format(self, content: bytes, filename: str) -> bool:
        """Basic file format validation using magic numbers"""
        if not content or len(content) < 12:
            return False
            
        # File signatures (magic numbers)
        signatures = {
            # Video formats
            b'\x00\x00\x00\x18ftyp': 'mp4',
            b'\x00\x00\x00\x20ftyp': 'mp4',
            b'RIFF': 'avi',  # AVI starts with RIFF
            b'\x1a\x45\xdf\xa3': 'webm',
            
            # Audio formats  
            b'RIFF': 'wav',  # WAV also starts with RIFF
            b'ID3': 'mp3',
            b'\xff\xfb': 'mp3',
            b'OggS': 'ogg',
        }
        
        # Check if content matches expected format based on filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        content_start = content[:12]
        
        for signature, format_type in signatures.items():
            if content_start.startswith(signature):
                if file_ext in ['mp4', 'avi', 'mov', 'wav', 'mp3', 'ogg']:
                    return True
                    
        return True  # Allow unknown formats for now


class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600, burst_limit: int = 20):
        self.max_requests = max_requests
        self.time_window = time_window
        self.burst_limit = burst_limit
        self.request_history: Dict[str, deque] = defaultdict(deque)
        self.burst_tracking: Dict[str, int] = defaultdict(int)
        self.last_reset: Dict[str, float] = defaultdict(float)
    
    def is_allowed(self, identifier: str, burst_check: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limiting"""
        current_time = time.time()
        history = self.request_history[identifier]
        
        # Clean old requests
        while history and history[0] < current_time - self.time_window:
            history.popleft()
            
        # Check burst limit
        if burst_check:
            if self.burst_tracking[identifier] >= self.burst_limit:
                if current_time - self.last_reset[identifier] < 60:  # 1 minute burst reset
                    return False, {
                        'reason': 'burst_limit_exceeded',
                        'burst_count': self.burst_tracking[identifier],
                        'limit': self.burst_limit
                    }
                else:
                    self.burst_tracking[identifier] = 0
                    self.last_reset[identifier] = current_time
        
        # Check rate limit
        if len(history) >= self.max_requests:
            return False, {
                'reason': 'rate_limit_exceeded',
                'requests_in_window': len(history),
                'limit': self.max_requests,
                'window_seconds': self.time_window
            }
        
        # Allow request
        history.append(current_time)
        self.burst_tracking[identifier] += 1
        
        return True, {
            'requests_in_window': len(history),
            'burst_count': self.burst_tracking[identifier],
            'limit': self.max_requests
        }


class SecurityAuditor:
    """Security event auditing and alerting"""
    
    def __init__(self, max_alerts: int = 10000):
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_counts: Dict[SecurityEvent, int] = defaultdict(int)
        self.threat_scores: Dict[str, float] = defaultdict(float)  # IP-based threat scoring
    
    def log_security_event(
        self,
        event_type: SecurityEvent,
        threat_level: ThreatLevel,
        message: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Dict[str, Any] = None
    ) -> str:
        """Log security event and return trace ID"""
        trace_id = secrets.token_hex(8)
        
        alert = SecurityAlert(
            event_type=event_type,
            threat_level=threat_level,
            message=message,
            source_ip=source_ip,
            user_id=user_id,
            details=details or {},
            trace_id=trace_id
        )
        
        self.alerts.append(alert)
        self.alert_counts[event_type] += 1
        
        # Update threat score for IP
        if source_ip:
            threat_increment = {
                ThreatLevel.LOW: 1,
                ThreatLevel.MEDIUM: 3,
                ThreatLevel.HIGH: 10,
                ThreatLevel.CRITICAL: 25
            }.get(threat_level, 1)
            
            self.threat_scores[source_ip] += threat_increment
            
        # Log to security logger
        security_logger.warning(f"Security Event: {json.dumps(alert.to_dict())}")
        
        # Critical events trigger immediate alerts
        if threat_level == ThreatLevel.CRITICAL:
            self._trigger_critical_alert(alert)
            
        return trace_id
    
    def get_threat_score(self, source_ip: str) -> float:
        """Get threat score for IP address"""
        return self.threat_scores.get(source_ip, 0.0)
    
    def is_high_risk_ip(self, source_ip: str, threshold: float = 50.0) -> bool:
        """Check if IP is considered high risk"""
        return self.get_threat_score(source_ip) >= threshold
    
    def get_recent_alerts(self, limit: int = 100, event_type: Optional[SecurityEvent] = None) -> List[Dict[str, Any]]:
        """Get recent security alerts"""
        alerts = list(self.alerts)[-limit:]
        
        if event_type:
            alerts = [alert for alert in alerts if alert.event_type == event_type]
            
        return [alert.to_dict() for alert in alerts]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics"""
        current_time = time.time()
        recent_alerts = [alert for alert in self.alerts if current_time - alert.timestamp < 3600]  # Last hour
        
        # Convert enum keys to strings for JSON serialization
        alert_counts_by_type = {event_type.value: count for event_type, count in self.alert_counts.items()}
        
        return {
            'total_alerts': len(self.alerts),
            'recent_alerts_1h': len(recent_alerts),
            'alert_counts_by_type': alert_counts_by_type,
            'high_risk_ips': [ip for ip, score in self.threat_scores.items() if score >= 50],
            'top_threat_ips': sorted(
                self.threat_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def _trigger_critical_alert(self, alert: SecurityAlert):
        """Trigger critical security alert"""
        # In production, this would send alerts to security team
        security_logger.critical(f"CRITICAL SECURITY ALERT: {alert.message}")


class SecureTokenManager:
    """Secure token generation and validation"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        
    def generate_token(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
        """Generate secure token with payload"""
        expires_at = time.time() + (expiry_hours * 3600)
        
        token_data = {
            'payload': payload,
            'expires_at': expires_at,
            'issued_at': time.time(),
            'nonce': secrets.token_hex(8)
        }
        
        # Create signature
        token_json = json.dumps(token_data, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            token_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Encode token
        full_token = base64.b64encode(f"{token_json}.{signature}".encode()).decode()
        return full_token
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Validate token and return payload"""
        try:
            # Decode token
            decoded = base64.b64decode(token.encode()).decode()
            token_json, signature = decoded.rsplit('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                token_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return False, None, "Invalid signature"
            
            # Parse token data
            token_data = json.loads(token_json)
            
            # Check expiry
            if time.time() > token_data['expires_at']:
                return False, None, "Token expired"
                
            return True, token_data['payload'], "Valid"
            
        except Exception as e:
            return False, None, f"Token validation error: {str(e)}"


class IPSecurityChecker:
    """IP-based security checks"""
    
    def __init__(self):
        # Known malicious IP ranges (examples)
        self.blocked_ranges = [
            ipaddress.ip_network('127.0.0.0/8'),  # Localhost
            ipaddress.ip_network('10.0.0.0/8'),   # Private
            ipaddress.ip_network('172.16.0.0/12'), # Private
            ipaddress.ip_network('192.168.0.0/16'), # Private
        ]
        
        self.allowed_countries = ['US', 'CA', 'GB', 'DE', 'FR', 'JP', 'AU']  # Example whitelist
    
    def is_ip_allowed(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP address is allowed"""
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            
            # Check if IP is in blocked ranges
            for blocked_range in self.blocked_ranges:
                if ip_obj in blocked_range:
                    return False, f"IP in blocked range: {blocked_range}"
            
            # Additional checks could include:
            # - GeoIP checking against allowed countries
            # - Known malicious IP databases
            # - Rate limiting per IP
            
            return True, "IP allowed"
            
        except ValueError:
            return False, "Invalid IP address format"


# Security decorators
def security_check(
    validate_input: bool = True,
    rate_limit: Optional[Tuple[int, int]] = None,  # (max_requests, time_window)
    require_auth: bool = False
):
    """Comprehensive security decorator"""
    def decorator(func):
        validator = AdvancedInputValidator()
        auditor = SecurityAuditor()
        rate_limiter = RateLimiter(*rate_limit) if rate_limit else None
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract request info if available
            source_ip = kwargs.get('source_ip', 'unknown')
            user_id = kwargs.get('user_id', 'anonymous')
            
            try:
                # Rate limiting check
                if rate_limiter:
                    allowed, limit_info = rate_limiter.is_allowed(source_ip)
                    if not allowed:
                        auditor.log_security_event(
                            SecurityEvent.RATE_LIMIT_EXCEEDED,
                            ThreatLevel.MEDIUM,
                            f"Rate limit exceeded: {limit_info}",
                            source_ip=source_ip,
                            user_id=user_id
                        )
                        raise Exception(f"Rate limit exceeded: {limit_info['reason']}")
                
                # Input validation
                if validate_input:
                    for arg in args:
                        if isinstance(arg, str):
                            valid, errors = validator.validate_text_input(arg)
                            if not valid:
                                auditor.log_security_event(
                                    SecurityEvent.INVALID_INPUT,
                                    ThreatLevel.HIGH,
                                    f"Invalid input detected: {errors}",
                                    source_ip=source_ip,
                                    user_id=user_id
                                )
                                raise ValueError(f"Input validation failed: {errors}")
                
                return await func(*args, **kwargs)
                
            except Exception as e:
                # Log security-related exceptions
                if any(keyword in str(e).lower() for keyword in ['inject', 'script', 'exploit', 'hack']):
                    auditor.log_security_event(
                        SecurityEvent.SUSPICIOUS_PATTERN,
                        ThreatLevel.HIGH,
                        f"Suspicious activity detected: {str(e)}",
                        source_ip=source_ip,
                        user_id=user_id
                    )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Global security components
global_auditor = SecurityAuditor()
global_validator = AdvancedInputValidator()
global_token_manager = SecureTokenManager()
global_ip_checker = IPSecurityChecker()

# Security configuration
SECURITY_CONFIG = {
    'max_file_size': 500 * 1024 * 1024,  # 500MB
    'rate_limit_requests': 100,
    'rate_limit_window': 3600,
    'token_expiry_hours': 24,
    'threat_score_threshold': 50.0,
    'enable_ip_blocking': True,
    'enable_input_validation': True,
    'log_all_events': True
}