"""
Security Utilities for AV-Separation-Transformer
Input validation, rate limiting, authentication, and security hardening
"""

import hashlib
import hmac
import time
import secrets
import jwt
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from pathlib import Path
import logging
from datetime import datetime, timedelta

import numpy as np
import torch
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt


class InputValidator:
    """
    Comprehensive input validation for audio-visual data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation limits
        self.limits = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'max_duration': 300,  # 5 minutes
            'max_sample_rate': 48000,
            'min_sample_rate': 8000,
            'max_video_resolution': (1920, 1080),
            'min_video_resolution': (64, 64),
            'max_fps': 60,
            'min_fps': 1,
            'max_speakers': 6,
            'min_speakers': 1,
            'allowed_audio_formats': {'.wav', '.mp3', '.flac', '.m4a', '.ogg'},
            'allowed_video_formats': {'.mp4', '.avi', '.mov', '.mkv', '.webm'},
            'max_filename_length': 255
        }
    
    def validate_file_upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        """
        Validate uploaded file
        
        Args:
            filename: Name of uploaded file
            content: File content as bytes
            
        Returns:
            Validation result
            
        Raises:
            ValueError: If validation fails
        """
        
        result = {
            'valid': True,
            'warnings': [],
            'file_info': {}
        }
        
        # Validate filename
        if not filename:
            raise ValueError("Filename is required")
        
        if len(filename) > self.limits['max_filename_length']:
            raise ValueError(f"Filename too long (max {self.limits['max_filename_length']} chars)")
        
        # Check for malicious patterns
        malicious_patterns = ['../', '..\\', '<script', '<?php', 'javascript:']
        filename_lower = filename.lower()
        
        for pattern in malicious_patterns:
            if pattern in filename_lower:
                raise ValueError(f"Potentially malicious filename pattern detected: {pattern}")
        
        # Validate file extension
        file_ext = Path(filename).suffix.lower()
        
        all_allowed_formats = (
            self.limits['allowed_audio_formats'] | 
            self.limits['allowed_video_formats']
        )
        
        if file_ext not in all_allowed_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Validate file size
        file_size = len(content)
        if file_size > self.limits['max_file_size']:
            raise ValueError(f"File too large: {file_size} bytes (max {self.limits['max_file_size']})")
        
        if file_size == 0:
            raise ValueError("Empty file")
        
        # Check file magic bytes
        self._validate_file_magic_bytes(content, file_ext)
        
        # Additional format-specific validation
        if file_ext in self.limits['allowed_audio_formats']:
            self._validate_audio_content(content)
        elif file_ext in self.limits['allowed_video_formats']:
            self._validate_video_content(content)
        
        result['file_info'] = {
            'filename': filename,
            'size': file_size,
            'extension': file_ext,
            'mime_type': self._get_mime_type(file_ext)
        }
        
        return result
    
    def _validate_file_magic_bytes(self, content: bytes, expected_ext: str):
        """Validate file magic bytes match extension"""
        
        magic_bytes = {
            '.wav': [b'RIFF', b'WAVE'],
            '.mp3': [b'ID3', b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'],
            '.flac': [b'fLaC'],
            '.mp4': [b'ftyp'],
            '.avi': [b'RIFF', b'AVI '],
            '.mov': [b'ftyp', b'moov'],
            '.mkv': [b'\x1a\x45\xdf\xa3']
        }
        
        if expected_ext in magic_bytes:
            expected_magic = magic_bytes[expected_ext]
            header = content[:12]  # Check first 12 bytes
            
            magic_found = False
            for magic in expected_magic:
                if magic in header:
                    magic_found = True
                    break
            
            if not magic_found:
                raise ValueError(f"File content doesn't match extension {expected_ext}")
    
    def _validate_audio_content(self, content: bytes):
        """Validate audio content"""
        
        try:
            # Basic audio file validation
            # In practice, you might use librosa or soundfile to validate
            if len(content) < 44:  # Minimum WAV header size
                raise ValueError("Audio file too small to be valid")
            
        except Exception as e:
            raise ValueError(f"Invalid audio content: {e}")
    
    def _validate_video_content(self, content: bytes):
        """Validate video content"""
        
        try:
            # Basic video file validation
            if len(content) < 1024:  # Minimum reasonable video file size
                raise ValueError("Video file too small to be valid")
            
        except Exception as e:
            raise ValueError(f"Invalid video content: {e}")
    
    def _get_mime_type(self, file_ext: str) -> str:
        """Get MIME type for file extension"""
        
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        
        return mime_types.get(file_ext, 'application/octet-stream')
    
    def validate_separation_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate separation parameters
        
        Args:
            params: Parameters dictionary
            
        Returns:
            Validated and sanitized parameters
        """
        
        validated = {}
        
        # Number of speakers
        num_speakers = params.get('num_speakers', 2)
        if not isinstance(num_speakers, int):
            raise ValueError("num_speakers must be an integer")
        
        if not (self.limits['min_speakers'] <= num_speakers <= self.limits['max_speakers']):
            raise ValueError(
                f"num_speakers must be between {self.limits['min_speakers']} "
                f"and {self.limits['max_speakers']}"
            )
        
        validated['num_speakers'] = num_speakers
        
        # Configuration overrides
        config_override = params.get('config_override')
        if config_override is not None:
            if not isinstance(config_override, dict):
                raise ValueError("config_override must be a dictionary")
            
            # Validate configuration parameters
            validated['config_override'] = self._validate_config_override(config_override)
        
        # Boolean parameters
        save_video = params.get('save_video', False)
        if not isinstance(save_video, bool):
            raise ValueError("save_video must be a boolean")
        
        validated['save_video'] = save_video
        
        return validated
    
    def _validate_config_override(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration override parameters"""
        
        validated = {}
        
        # Only allow specific configuration overrides for security
        allowed_overrides = {
            'audio': ['sample_rate', 'chunk_duration', 'n_mels'],
            'video': ['fps', 'image_size'],
            'inference': ['batch_size', 'use_fp16']
        }
        
        for section, params in config.items():
            if section not in allowed_overrides:
                continue  # Skip disallowed sections
            
            if not isinstance(params, dict):
                continue
            
            validated_section = {}
            
            for param, value in params.items():
                if param not in allowed_overrides[section]:
                    continue  # Skip disallowed parameters
                
                # Validate specific parameters
                if param == 'sample_rate':
                    if not isinstance(value, int) or not (8000 <= value <= 48000):
                        raise ValueError("Invalid sample_rate")
                    validated_section[param] = value
                
                elif param == 'chunk_duration':
                    if not isinstance(value, (int, float)) or not (0.1 <= value <= 30.0):
                        raise ValueError("Invalid chunk_duration")
                    validated_section[param] = float(value)
                
                elif param == 'n_mels':
                    if not isinstance(value, int) or not (40 <= value <= 128):
                        raise ValueError("Invalid n_mels")
                    validated_section[param] = value
                
                elif param == 'fps':
                    if not isinstance(value, int) or not (1 <= value <= 60):
                        raise ValueError("Invalid fps")
                    validated_section[param] = value
                
                elif param == 'image_size':
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError("Invalid image_size")
                    h, w = value
                    if not (64 <= h <= 1080 and 64 <= w <= 1920):
                        raise ValueError("Invalid image_size dimensions")
                    validated_section[param] = [int(h), int(w)]
                
                elif param == 'batch_size':
                    if not isinstance(value, int) or not (1 <= value <= 32):
                        raise ValueError("Invalid batch_size")
                    validated_section[param] = value
                
                elif param == 'use_fp16':
                    if not isinstance(value, bool):
                        raise ValueError("Invalid use_fp16")
                    validated_section[param] = value
            
            if validated_section:
                validated[section] = validated_section
        
        return validated
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        
        # Remove potentially dangerous characters
        import re
        
        # Keep only alphanumeric, dots, hyphens, underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Limit length
        if len(sanitized) > self.limits['max_filename_length']:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            max_name_len = self.limits['max_filename_length'] - len(ext)
            sanitized = name[:max_name_len] + ext
        
        return sanitized


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds (default: 1 hour)
        """
        
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}  # client_id -> [(timestamp, count), ...]
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, client_id: str, request_cost: int = 1) -> bool:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Client identifier (IP, API key, etc.)
            request_cost: Cost of this request (default: 1)
            
        Returns:
            True if request is allowed, False otherwise
        """
        
        current_time = time.time()
        
        # Clean old requests
        self._cleanup_old_requests(client_id, current_time)
        
        # Get current request count
        current_count = sum(count for _, count in self.requests.get(client_id, []))
        
        # Check if request would exceed limit
        if current_count + request_cost > self.max_requests:
            self.logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Record the request
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id].append((current_time, request_cost))
        
        return True
    
    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Clean up requests outside the time window"""
        
        if client_id in self.requests:
            cutoff_time = current_time - self.time_window
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if timestamp > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_id]:
                del self.requests[client_id]
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        
        current_time = time.time()
        self._cleanup_old_requests(client_id, current_time)
        
        current_count = sum(count for _, count in self.requests.get(client_id, []))
        return max(0, self.max_requests - current_count)


class APIKeyManager:
    """
    API key generation and validation
    """
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.api_keys = {}  # api_key -> {user_id, permissions, created_at, last_used}
        self.logger = logging.getLogger(__name__)
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """
        Generate new API key
        
        Args:
            user_id: User identifier
            permissions: List of permissions (default: ['read'])
            
        Returns:
            Generated API key
        """
        
        if permissions is None:
            permissions = ['read']
        
        # Generate random key
        random_data = secrets.token_bytes(32)
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key,
            random_data + user_id.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine and encode
        api_key = f"av_{random_data.hex()}_{signature[:16]}"
        
        # Store key info
        self.api_keys[api_key] = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow(),
            'last_used': None
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key info if valid, None otherwise
        """
        
        if not api_key or not api_key.startswith('av_'):
            return None
        
        try:
            # Parse key components
            parts = api_key.split('_')
            if len(parts) != 3:
                return None
            
            random_hex = parts[1]
            signature_part = parts[2]
            
            # Get stored key info
            if api_key not in self.api_keys:
                return None
            
            key_info = self.api_keys[api_key]
            
            # Verify signature
            random_data = bytes.fromhex(random_hex)
            expected_signature = hmac.new(
                self.secret_key,
                random_data + key_info['user_id'].encode(),
                hashlib.sha256
            ).hexdigest()[:16]
            
            if not hmac.compare_digest(signature_part, expected_signature):
                return None
            
            # Update last used
            key_info['last_used'] = datetime.utcnow()
            
            return key_info
            
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        
        return False


class JWTManager:
    """
    JWT token management for authentication
    """
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
    
    def create_token(
        self,
        user_id: str,
        permissions: List[str] = None,
        expires_in: int = 3600
    ) -> str:
        """
        Create JWT token
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            expires_in: Token expiration in seconds
            
        Returns:
            JWT token
        """
        
        if permissions is None:
            permissions = ['read']
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token
        
        Args:
            token: JWT token to validate
            
        Returns:
            Token payload if valid, None otherwise
        """
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None


# FastAPI security dependencies
security = HTTPBearer()
input_validator = InputValidator()
rate_limiter = RateLimiter()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency for authentication"""
    
    token = credentials.credentials
    
    # Try API key validation first
    api_key_manager = get_api_key_manager()  # Implementation depends on setup
    if api_key_manager:
        key_info = api_key_manager.validate_api_key(token)
        if key_info:
            return key_info
    
    # Try JWT validation
    jwt_manager = get_jwt_manager()  # Implementation depends on setup
    if jwt_manager:
        payload = jwt_manager.validate_token(token)
        if payload:
            return payload
    
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def check_permissions(required_permissions: List[str]):
    """FastAPI dependency factory for permission checking"""
    
    def permission_checker(current_user: Dict = Depends(get_current_user)):
        user_permissions = current_user.get('permissions', [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permission: {permission}"
                )
        
        return current_user
    
    return permission_checker


def rate_limit(max_requests: int = 100, time_window: int = 3600):
    """FastAPI dependency factory for rate limiting"""
    
    limiter = RateLimiter(max_requests, time_window)
    
    def rate_limit_checker(request: Request):
        client_ip = request.client.host
        
        if not limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(time_window)}
            )
        
        return True
    
    return rate_limit_checker


# Placeholder functions for dependency injection
def get_api_key_manager() -> Optional[APIKeyManager]:
    """Get API key manager instance (implement based on your setup)"""
    return None


def get_jwt_manager() -> Optional[JWTManager]:
    """Get JWT manager instance (implement based on your setup)"""
    return None


def secure_filename(filename: str) -> str:
    """
    Generate secure filename
    
    Args:
        filename: Original filename
        
    Returns:
        Secure filename with timestamp
    """
    
    sanitized = input_validator.sanitize_filename(filename)
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(4)
    
    name, ext = Path(sanitized).stem, Path(sanitized).suffix
    secure_name = f"{name}_{timestamp}_{random_suffix}{ext}"
    
    return secure_name


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def generate_csrf_token() -> str:
    """Generate CSRF token"""
    
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str, expected: str) -> bool:
    """Validate CSRF token"""
    
    return hmac.compare_digest(token, expected)