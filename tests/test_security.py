"""
Comprehensive Security Tests for AV-Separation-Transformer
"""

import pytest
import tempfile
import secrets
from pathlib import Path
from unittest.mock import patch, MagicMock

from av_separation.security import (
    InputValidator, RateLimiter, APIKeyManager, JWTManager,
    secure_filename, hash_password, verify_password
)


class TestInputValidator:
    """Test input validation functionality"""
    
    def setup_method(self):
        self.validator = InputValidator()
    
    def test_valid_wav_file(self):
        """Test validation of valid WAV file"""
        wav_header = b'RIFF\x24\x00\x00\x00WAVE'
        content = wav_header + b'\x00' * 100
        
        result = self.validator.validate_file_upload('test.wav', content)
        assert result['valid'] is True
        assert result['file_info']['extension'] == '.wav'
    
    def test_invalid_extension(self):
        """Test rejection of invalid file extension"""
        content = b'some content'
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.validator.validate_file_upload('test.exe', content)
    
    def test_malicious_filename(self):
        """Test rejection of malicious filename patterns"""
        content = b'RIFF\x24\x00\x00\x00WAVE' + b'\x00' * 100
        
        malicious_names = [
            '../../../etc/passwd.wav',
            'test<script>alert()</script>.wav',
            'test<?php echo "hack"; ?>.wav',
            'javascript:alert().wav'
        ]
        
        for name in malicious_names:
            with pytest.raises(ValueError, match="malicious filename pattern"):
                self.validator.validate_file_upload(name, content)
    
    def test_file_too_large(self):
        """Test rejection of oversized files"""
        large_content = b'\x00' * (101 * 1024 * 1024)  # 101MB
        
        with pytest.raises(ValueError, match="File too large"):
            self.validator.validate_file_upload('test.wav', large_content)
    
    def test_empty_file(self):
        """Test rejection of empty files"""
        with pytest.raises(ValueError, match="Empty file"):
            self.validator.validate_file_upload('test.wav', b'')
    
    def test_magic_byte_mismatch(self):
        """Test rejection when magic bytes don't match extension"""
        # MP3 header with WAV extension
        mp3_content = b'\xff\xfb\x90\x00' + b'\x00' * 100
        
        with pytest.raises(ValueError, match="doesn't match extension"):
            self.validator.validate_file_upload('test.wav', mp3_content)
    
    def test_separation_parameters_validation(self):
        """Test separation parameter validation"""
        # Valid parameters
        valid_params = {
            'num_speakers': 3,
            'save_video': True,
            'config_override': {
                'audio': {'sample_rate': 16000, 'n_mels': 80},
                'video': {'fps': 30}
            }
        }
        
        result = self.validator.validate_separation_parameters(valid_params)
        assert result['num_speakers'] == 3
        assert result['save_video'] is True
        assert 'config_override' in result
    
    def test_invalid_num_speakers(self):
        """Test validation of invalid number of speakers"""
        invalid_params = [
            {'num_speakers': 0},  # Too few
            {'num_speakers': 10},  # Too many
            {'num_speakers': 'two'},  # Wrong type
        ]
        
        for params in invalid_params:
            with pytest.raises(ValueError):
                self.validator.validate_separation_parameters(params)
    
    def test_config_override_security(self):
        """Test that only allowed config overrides are accepted"""
        # Try to override sensitive model parameters
        malicious_config = {
            'model': {'max_speakers': 100},  # Should be filtered out
            'audio': {'sample_rate': 16000},  # Should be allowed
            'training': {'learning_rate': 0.1}  # Should be filtered out
        }
        
        params = {'config_override': malicious_config}
        result = self.validator.validate_separation_parameters(params)
        
        config = result['config_override']
        assert 'model' not in config
        assert 'training' not in config
        assert 'audio' in config
        assert config['audio']['sample_rate'] == 16000
    
    def test_filename_sanitization(self):
        """Test filename sanitization"""
        test_cases = [
            ('normal_file.wav', 'normal_file.wav'),
            ('file with spaces.wav', 'file_with_spaces.wav'),
            ('file/with\\slashes.wav', 'file_with_slashes.wav'),
            ('file<>:"|?*.wav', 'file_________.wav'),
            ('very_long_filename' * 20 + '.wav', None)  # Will be truncated
        ]
        
        for original, expected in test_cases:
            sanitized = self.validator.sanitize_filename(original)
            if expected:
                assert sanitized == expected
            else:
                # Check that long filenames are truncated
                assert len(sanitized) <= self.validator.limits['max_filename_length']
                assert sanitized.endswith('.wav')


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        client_id = 'test_client'
        
        # First 5 requests should be allowed
        for i in range(5):
            assert limiter.is_allowed(client_id) is True
        
        # 6th request should be denied
        assert limiter.is_allowed(client_id) is False
    
    def test_request_cost(self):
        """Test rate limiting with different request costs"""
        limiter = RateLimiter(max_requests=10, time_window=60)
        client_id = 'test_client'
        
        # Use up allowance with high-cost request
        assert limiter.is_allowed(client_id, request_cost=8) is True
        assert limiter.is_allowed(client_id, request_cost=3) is False  # Would exceed limit
        assert limiter.is_allowed(client_id, request_cost=2) is True   # Within limit
    
    def test_multiple_clients(self):
        """Test that rate limiting is per-client"""
        limiter = RateLimiter(max_requests=2, time_window=60)
        
        # Client 1 uses up allowance
        assert limiter.is_allowed('client1') is True
        assert limiter.is_allowed('client1') is True
        assert limiter.is_allowed('client1') is False
        
        # Client 2 should still have allowance
        assert limiter.is_allowed('client2') is True
        assert limiter.is_allowed('client2') is True
        assert limiter.is_allowed('client2') is False
    
    def test_remaining_requests(self):
        """Test getting remaining request count"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        client_id = 'test_client'
        
        assert limiter.get_remaining_requests(client_id) == 5
        
        limiter.is_allowed(client_id, request_cost=2)
        assert limiter.get_remaining_requests(client_id) == 3
        
        limiter.is_allowed(client_id, request_cost=3)
        assert limiter.get_remaining_requests(client_id) == 0
    
    @patch('time.time')
    def test_time_window_cleanup(self, mock_time):
        """Test that old requests are cleaned up"""
        limiter = RateLimiter(max_requests=2, time_window=60)
        client_id = 'test_client'
        
        # At time 0, use up allowance
        mock_time.return_value = 0
        assert limiter.is_allowed(client_id) is True
        assert limiter.is_allowed(client_id) is True
        assert limiter.is_allowed(client_id) is False
        
        # At time 61, requests should be cleaned up
        mock_time.return_value = 61
        assert limiter.is_allowed(client_id) is True


class TestAPIKeyManager:
    """Test API key management"""
    
    def setup_method(self):
        self.manager = APIKeyManager(secret_key='test_secret_key')
    
    def test_api_key_generation(self):
        """Test API key generation"""
        api_key = self.manager.generate_api_key('user123', ['read', 'write'])
        
        assert api_key.startswith('av_')
        assert len(api_key.split('_')) == 3
        assert api_key in self.manager.api_keys
        
        key_info = self.manager.api_keys[api_key]
        assert key_info['user_id'] == 'user123'
        assert key_info['permissions'] == ['read', 'write']
    
    def test_api_key_validation(self):
        """Test API key validation"""
        api_key = self.manager.generate_api_key('user123', ['read'])
        
        # Valid key should return key info
        key_info = self.manager.validate_api_key(api_key)
        assert key_info is not None
        assert key_info['user_id'] == 'user123'
        assert key_info['permissions'] == ['read']
        assert key_info['last_used'] is not None
    
    def test_invalid_api_key(self):
        """Test validation of invalid API keys"""
        invalid_keys = [
            'invalid_key',
            'av_invalid',
            'av_' + 'x' * 64 + '_invalid',
            ''
        ]
        
        for key in invalid_keys:
            assert self.manager.validate_api_key(key) is None
    
    def test_api_key_revocation(self):
        """Test API key revocation"""
        api_key = self.manager.generate_api_key('user123')
        
        # Key should be valid initially
        assert self.manager.validate_api_key(api_key) is not None
        
        # Revoke key
        assert self.manager.revoke_api_key(api_key) is True
        
        # Key should be invalid after revocation
        assert self.manager.validate_api_key(api_key) is None
        
        # Revoking again should return False
        assert self.manager.revoke_api_key(api_key) is False


class TestJWTManager:
    """Test JWT token management"""
    
    def setup_method(self):
        self.manager = JWTManager(secret_key='test_secret_key')
    
    def test_token_creation_and_validation(self):
        """Test JWT token creation and validation"""
        token = self.manager.create_token('user123', ['read', 'write'], expires_in=3600)
        
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts
        
        # Validate token
        payload = self.manager.validate_token(token)
        assert payload is not None
        assert payload['user_id'] == 'user123'
        assert payload['permissions'] == ['read', 'write']
    
    def test_expired_token(self):
        """Test validation of expired token"""
        # Create token that expires immediately
        token = self.manager.create_token('user123', expires_in=-1)
        
        # Should be invalid due to expiration
        payload = self.manager.validate_token(token)
        assert payload is None
    
    def test_invalid_token_signature(self):
        """Test validation with wrong secret key"""
        token = self.manager.create_token('user123')
        
        # Create manager with different secret
        wrong_manager = JWTManager(secret_key='wrong_secret')
        payload = wrong_manager.validate_token(token)
        assert payload is None
    
    def test_malformed_token(self):
        """Test validation of malformed tokens"""
        malformed_tokens = [
            'not.a.jwt',
            'invalid_token',
            '',
            'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature'
        ]
        
        for token in malformed_tokens:
            payload = self.manager.validate_token(token)
            assert payload is None


class TestPasswordSecurity:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test password hashing functionality"""
        password = 'secure_password123'
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        assert hashed.startswith('$2b$')  # bcrypt identifier
    
    def test_password_verification(self):
        """Test password verification"""
        password = 'test_password'
        hashed = hash_password(password)
        
        # Correct password should verify
        assert verify_password(password, hashed) is True
        
        # Wrong password should not verify
        assert verify_password('wrong_password', hashed) is False
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes (due to salt)"""
        password = 'same_password'
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestSecurityUtilities:
    """Test security utility functions"""
    
    def test_secure_filename_generation(self):
        """Test secure filename generation"""
        original_filename = 'test_file.wav'
        secure_name = secure_filename(original_filename)
        
        assert secure_name != original_filename
        assert secure_name.endswith('.wav')
        assert '_' in secure_name  # Should contain timestamp and random suffix
        
        # Should not contain directory traversal
        malicious_filename = '../../../etc/passwd.wav'
        secure_name = secure_filename(malicious_filename)
        assert '..' not in secure_name
        assert '/' not in secure_name
    
    def test_csrf_token_generation(self):
        """Test CSRF token generation and validation"""
        from av_separation.security import generate_csrf_token, validate_csrf_token
        
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        
        # Tokens should be different
        assert token1 != token2
        assert len(token1) > 32
        assert len(token2) > 32
        
        # Valid token should validate
        assert validate_csrf_token(token1, token1) is True
        
        # Different tokens should not validate
        assert validate_csrf_token(token1, token2) is False
        
        # Invalid token should not validate
        assert validate_csrf_token(token1, 'invalid_token') is False


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        # Setup
        api_key_manager = APIKeyManager('test_secret')
        jwt_manager = JWTManager('test_secret')
        rate_limiter = RateLimiter(max_requests=10, time_window=60)
        
        # Generate API key
        api_key = api_key_manager.generate_api_key('user123', ['separate'])
        
        # Validate API key
        key_info = api_key_manager.validate_api_key(api_key)
        assert key_info is not None
        
        # Check rate limiting
        assert rate_limiter.is_allowed('user123') is True
        
        # Create JWT token
        jwt_token = jwt_manager.create_token('user123', ['separate'])
        
        # Validate JWT token
        jwt_payload = jwt_manager.validate_token(jwt_token)
        assert jwt_payload is not None
        assert jwt_payload['user_id'] == 'user123'
    
    def test_security_event_logging(self):
        """Test security event logging integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            from av_separation.logging_config import setup_logging
            
            # Setup logging
            loggers = setup_logging(
                log_dir=temp_dir,
                enable_audit=True
            )
            
            audit_logger = loggers['audit']
            
            # Log security event
            audit_logger.log_security_event(
                event_type='suspicious_activity',
                severity='high',
                client_ip='192.168.1.100',
                description='Multiple failed authentication attempts',
                additional_data={'attempts': 5}
            )
            
            # Check that log file was created
            audit_log_path = Path(temp_dir) / 'audit.log'
            assert audit_log_path.exists()
            
            # Read log content
            log_content = audit_log_path.read_text()
            assert 'suspicious_activity' in log_content
            assert '192.168.1.100' in log_content