import hmac
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from functools import wraps
import time
from collections import defaultdict, deque
import ipaddress
from urllib.parse import urlparse
import re


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_expiry_minutes: int = 60
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 1
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_encryption: bool = True
    audit_log_path: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ['*'])
    blocked_ips: List[str] = field(default_factory=list)
    require_https: bool = True
    

class SecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window_minutes
        )
        
        # Authentication tracking
        self.failed_attempts = defaultdict(int)
        self.lockout_times = defaultdict(lambda: None)
        
        # Encryption setup
        if self.config.enable_encryption:
            self.encryption_manager = EncryptionManager(self.config.secret_key)
        else:
            self.encryption_manager = None
        
        # Security event monitoring
        self.security_events = deque(maxlen=1000)
        
        # Input validation patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # JavaScript
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'__import__',
            r'\.\.[\\/]',  # Path traversal
        ]
        self.pattern_regex = re.compile('|'.join(self.dangerous_patterns), re.IGNORECASE)
    
    def authenticate_request(self, token: str, client_ip: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate incoming request."""
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            self._log_security_event('blocked_ip_attempt', 'high', {
                'client_ip': client_ip,
                'reason': 'IP in blocklist'
            })
            return False, None
        
        # Check rate limiting
        if not self.rate_limiter.allow_request(client_ip):
            self._log_security_event('rate_limit_exceeded', 'medium', {
                'client_ip': client_ip
            })
            return False, None
        
        # Check if client is locked out
        if self._is_locked_out(client_ip):
            self._log_security_event('lockout_attempt', 'medium', {
                'client_ip': client_ip
            })
            return False, None
        
        # Verify JWT token
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=['HS256']
            )
            
            # Reset failed attempts on successful authentication
            if client_ip in self.failed_attempts:
                del self.failed_attempts[client_ip]
                del self.lockout_times[client_ip]
            
            self._log_security_event('authentication_success', 'low', {
                'client_ip': client_ip,
                'user_id': payload.get('user_id', 'unknown')
            })
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            self._handle_auth_failure(client_ip, 'expired_token')
            return False, None
        except jwt.InvalidTokenError:
            self._handle_auth_failure(client_ip, 'invalid_token')
            return False, None
    
    def generate_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(minutes=self.config.jwt_expiry_minutes),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.config.secret_key, algorithm='HS256')
        
        self._log_security_event('token_generated', 'low', {
            'user_id': user_id,
            'expires_at': payload['exp'].isoformat()
        })
        
        return token
    
    def validate_input(self, input_data: str) -> Tuple[bool, List[str]]:
        """Validate input for security threats."""
        threats = []
        
        # Check for dangerous patterns
        matches = self.pattern_regex.findall(input_data)
        if matches:
            threats.append(f"Dangerous patterns detected: {matches}")
        
        # Check for SQL injection patterns (basic)
        sql_patterns = [
            r'\bunion\s+select\b',
            r'\bor\s+1\s*=\s*1\b',
            r'\band\s+1\s*=\s*1\b',
            r'\bdrop\s+table\b',
            r'\bdelete\s+from\b',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                threats.append(f"Potential SQL injection: {pattern}")
                break
        
        # Check input length
        if len(input_data) > 100000:  # 100KB
            threats.append("Input data too large")
        
        # Check for null bytes
        if '\x00' in input_data:
            threats.append("Null bytes detected")
        
        is_safe = len(threats) == 0
        
        if not is_safe:
            self._log_security_event('input_validation_failed', 'medium', {
                'threats': threats,
                'input_length': len(input_data)
            })
        
        return is_safe, threats
    
    def secure_file_upload(self, file_path: Path, user_id: str) -> Tuple[bool, Optional[str]]:
        """Securely handle file upload."""
        
        # Sanitize filename
        safe_filename = self._sanitize_filename(file_path.name)
        if safe_filename != file_path.name:
            self._log_security_event('filename_sanitized', 'low', {
                'original': file_path.name,
                'sanitized': safe_filename,
                'user_id': user_id
            })
        
        # Check file size
        if file_path.stat().st_size > 500 * 1024 * 1024:  # 500MB
            self._log_security_event('oversized_file_upload', 'medium', {
                'file_size': file_path.stat().st_size,
                'user_id': user_id
            })
            return False, "File too large"
        
        # Create secure storage path
        secure_dir = Path(f"secure_uploads/{user_id}")
        secure_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        secure_path = secure_dir / safe_filename
        
        # Move file to secure location
        try:
            file_path.rename(secure_path)
            secure_path.chmod(0o600)  # Read/write for owner only
            
            self._log_security_event('file_upload_success', 'low', {
                'secure_path': str(secure_path),
                'user_id': user_id
            })
            
            return True, str(secure_path)
            
        except Exception as e:
            self._log_security_event('file_upload_error', 'high', {
                'error': str(e),
                'user_id': user_id
            })
            return False, f"Upload failed: {e}"
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is in blocklist."""
        try:
            client_ip = ipaddress.ip_address(ip)
            for blocked_ip in self.config.blocked_ips:
                if '/' in blocked_ip:  # CIDR notation
                    network = ipaddress.ip_network(blocked_ip, strict=False)
                    if client_ip in network:
                        return True
                else:
                    if str(client_ip) == blocked_ip:
                        return True
            return False
        except ValueError:
            # Invalid IP format - block it
            return True
    
    def _is_locked_out(self, client_ip: str) -> bool:
        """Check if client is locked out due to failed attempts."""
        if client_ip not in self.lockout_times:
            return False
        
        lockout_time = self.lockout_times[client_ip]
        if lockout_time is None:
            return False
        
        lockout_duration = timedelta(minutes=self.config.lockout_duration_minutes)
        return datetime.utcnow() < lockout_time + lockout_duration
    
    def _handle_auth_failure(self, client_ip: str, reason: str):
        """Handle authentication failure."""
        self.failed_attempts[client_ip] += 1
        
        self._log_security_event('authentication_failure', 'medium', {
            'client_ip': client_ip,
            'reason': reason,
            'failed_attempts': self.failed_attempts[client_ip]
        })
        
        # Lock out if too many failures
        if self.failed_attempts[client_ip] >= self.config.max_failed_attempts:
            self.lockout_times[client_ip] = datetime.utcnow()
            self._log_security_event('client_locked_out', 'high', {
                'client_ip': client_ip,
                'lockout_duration': self.config.lockout_duration_minutes
            })
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove path separators and dangerous characters
        safe_chars = re.sub(r'[^\w\s.-]', '', filename)
        # Remove leading/trailing dots and spaces
        safe_chars = safe_chars.strip('. ')
        # Limit length
        safe_chars = safe_chars[:255]
        # Ensure it's not empty
        if not safe_chars:
            safe_chars = f"file_{secrets.token_hex(8)}"
        
        return safe_chars
    
    def _log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Log to file if configured
        if self.config.audit_log_path:
            try:
                with open(self.config.audit_log_path, 'a') as f:
                    f.write(json.dumps(event) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to write audit log: {e}")
        
        # Log to standard logger
        level_map = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = level_map.get(severity, logging.INFO)
        self.logger.log(level, f"Security event: {event_type} - {json.dumps(details)}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event['timestamp']) > last_hour
        ]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event['event_type']] += 1
            severity_counts[event['severity']] += 1
        
        return {
            'report_timestamp': now.isoformat(),
            'total_events_last_hour': len(recent_events),
            'event_type_breakdown': dict(event_counts),
            'severity_breakdown': dict(severity_counts),
            'active_lockouts': len([ip for ip, lockout_time in self.lockout_times.items() 
                                  if lockout_time and self._is_locked_out(ip)]),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'failed_attempts_by_ip': dict(self.failed_attempts)
        }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_window: int, window_minutes: int):
        self.requests_per_window = requests_per_window
        self.window_duration = timedelta(minutes=window_minutes)
        self.client_windows = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        with self.lock:
            now = datetime.utcnow()
            window = self.client_windows[client_id]
            
            # Remove old requests outside window
            while window and window[0] < now - self.window_duration:
                window.popleft()
            
            # Check if under limit
            if len(window) < self.requests_per_window:
                window.append(now)
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self.lock:
            active_clients = len(self.client_windows)
            total_requests = sum(len(window) for window in self.client_windows.values())
            
            return {
                'active_clients': active_clients,
                'total_active_requests': total_requests,
                'requests_per_window': self.requests_per_window,
                'window_minutes': self.window_duration.total_seconds() / 60
            }


class EncryptionManager:
    """Handle encryption/decryption operations."""
    
    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self.fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt_for_consistency',  # In production, use random salt per key
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt string data."""
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted)
        return decrypted.decode()
    
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt file in place."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_path: Path, output_path: Path) -> Path:
        """Decrypt file to output path."""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path


def require_authentication(security_manager: SecurityManager):
    """Decorator to require authentication for API endpoints."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request info (this would depend on your web framework)
            # For FastAPI, you might get this from request context
            token = kwargs.get('auth_token') or getattr(args[0], 'headers', {}).get('Authorization', '')
            client_ip = kwargs.get('client_ip', '127.0.0.1')
            
            if token.startswith('Bearer '):
                token = token[7:]
            
            authenticated, user_info = security_manager.authenticate_request(token, client_ip)
            
            if not authenticated:
                raise ValueError("Authentication failed")
            
            # Add user info to kwargs
            kwargs['user_info'] = user_info
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def secure_endpoint(security_manager: SecurityManager, require_https: bool = True):
    """Comprehensive security decorator for API endpoints."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # HTTPS check
            if require_https:
                request_url = kwargs.get('request_url', '')
                if request_url and not request_url.startswith('https://'):
                    raise ValueError("HTTPS required")
            
            # Input validation
            for key, value in kwargs.items():
                if isinstance(value, str):
                    is_safe, threats = security_manager.validate_input(value)
                    if not is_safe:
                        raise ValueError(f"Input validation failed: {threats}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
