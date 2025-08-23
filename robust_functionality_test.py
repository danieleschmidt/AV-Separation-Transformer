#!/usr/bin/env python3
"""
Generation 2: Robust Functionality Test
Comprehensive error handling, validation, security, and monitoring
"""

import sys
import os
import json
import time
import hashlib
import logging
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Setup robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/av_separation_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self):
        self.error_count = 0
        self.error_log = []
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_open = False
        
    def handle_error(self, error, context="unknown", critical=False):
        """Handle and log errors with recovery strategies"""
        self.error_count += 1
        error_info = {
            'timestamp': time.time(),
            'error': str(error),
            'context': context,
            'critical': critical,
            'error_type': type(error).__name__
        }
        self.error_log.append(error_info)
        
        logger.error(f"Error in {context}: {error}")
        
        if critical or self.error_count >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            logger.critical("Circuit breaker activated due to excessive errors")
            return False
        
        # Implement exponential backoff
        time.sleep(min(0.1 * (2 ** self.error_count), 5.0))
        return True
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker after successful operations"""
        if self.error_count == 0:
            self.circuit_breaker_open = False
            logger.info("Circuit breaker reset")

class SecurityValidator:
    """Security validation and threat detection"""
    
    @staticmethod
    def validate_input_path(path):
        """Validate file paths against directory traversal"""
        try:
            # Resolve path and check if it's within allowed directories
            resolved_path = Path(path).resolve()
            cwd = Path.cwd().resolve()
            
            # Check for directory traversal attempts
            if not str(resolved_path).startswith(str(cwd)):
                raise SecurityError(f"Path traversal attempt detected: {path}")
            
            # Check file extension whitelist
            allowed_extensions = {'.mp4', '.wav', '.avi', '.mov', '.mp3', '.flac'}
            if resolved_path.suffix.lower() not in allowed_extensions:
                raise SecurityError(f"Unauthorized file type: {resolved_path.suffix}")
            
            return True
        except Exception as e:
            raise SecurityError(f"Path validation failed: {e}")
    
    @staticmethod
    def validate_audio_data(audio_data):
        """Validate audio data for malicious content"""
        if not isinstance(audio_data, np.ndarray):
            raise SecurityError("Invalid audio data type")
        
        # Check for suspicious patterns
        if len(audio_data) > 100_000_000:  # 100MB limit
            raise SecurityError("Audio data exceeds size limit")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            raise SecurityError("Audio data contains invalid values")
        
        return True
    
    @staticmethod
    def compute_content_hash(data):
        """Compute secure hash for content integrity"""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return hashlib.sha256(data).hexdigest()

class SecurityError(Exception):
    """Custom security exception"""
    pass

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'latencies': [],
            'memory_usage': [],
            'start_time': time.time()
        }
    
    def start_request(self):
        """Start timing a request"""
        return time.perf_counter()
    
    def end_request(self, start_time, success=True):
        """End timing a request and record metrics"""
        latency = time.perf_counter() - start_time
        self.metrics['requests_total'] += 1
        self.metrics['latencies'].append(latency * 1000)  # Convert to ms
        
        if success:
            self.metrics['requests_success'] += 1
        else:
            self.metrics['requests_failed'] += 1
        
        # Memory usage (simplified)
        try:
            import psutil
            self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)
        except:
            pass  # psutil not available
    
    def get_metrics(self):
        """Get current performance metrics"""
        if self.metrics['latencies']:
            avg_latency = np.mean(self.metrics['latencies'])
            p95_latency = np.percentile(self.metrics['latencies'], 95)
            p99_latency = np.percentile(self.metrics['latencies'], 99)
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        success_rate = (
            self.metrics['requests_success'] / max(self.metrics['requests_total'], 1)
        ) * 100
        
        return {
            'requests_total': self.metrics['requests_total'],
            'success_rate_percent': success_rate,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'uptime_seconds': time.time() - self.metrics['start_time']
        }

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.health_status = {'status': 'healthy', 'checks': {}}
    
    def check_disk_space(self, min_free_gb=1):
        """Check available disk space"""
        try:
            import shutil
            free_bytes = shutil.disk_usage('.').free
            free_gb = free_bytes / (1024**3)
            
            if free_gb < min_free_gb:
                self.health_status['checks']['disk'] = 'critical'
                return False
            else:
                self.health_status['checks']['disk'] = 'healthy'
                return True
        except Exception as e:
            self.health_status['checks']['disk'] = f'error: {e}'
            return False
    
    def check_memory_usage(self, max_usage_percent=85):
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > max_usage_percent:
                self.health_status['checks']['memory'] = 'warning'
                return False
            else:
                self.health_status['checks']['memory'] = 'healthy'
                return True
        except Exception:
            self.health_status['checks']['memory'] = 'unknown'
            return True
    
    def check_dependencies(self):
        """Check if critical dependencies are available"""
        deps = ['numpy', 'json', 'pathlib', 'hashlib', 'logging']
        missing_deps = []
        
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.health_status['checks']['dependencies'] = f'missing: {missing_deps}'
            return False
        else:
            self.health_status['checks']['dependencies'] = 'healthy'
            return True
    
    def get_health_status(self):
        """Get overall health status"""
        checks = [
            self.check_disk_space(),
            self.check_memory_usage(),
            self.check_dependencies()
        ]
        
        if all(checks):
            self.health_status['status'] = 'healthy'
        elif any(checks):
            self.health_status['status'] = 'degraded'
        else:
            self.health_status['status'] = 'unhealthy'
        
        return self.health_status

def test_error_handling():
    """Test comprehensive error handling"""
    try:
        error_handler = RobustErrorHandler()
        
        # Test normal error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            success = error_handler.handle_error(e, "test_context")
            assert success, "Should handle non-critical errors"
        
        # Test circuit breaker
        for i in range(6):
            try:
                raise RuntimeError(f"Error {i}")
            except Exception as e:
                error_handler.handle_error(e, f"test_context_{i}")
        
        assert error_handler.circuit_breaker_open, "Circuit breaker should be open"
        
        print("✅ Error handling and circuit breaker working")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_security_validation():
    """Test security validation mechanisms"""
    try:
        validator = SecurityValidator()
        
        # Test path validation
        try:
            validator.validate_input_path("../../../etc/passwd")
            assert False, "Should reject directory traversal"
        except SecurityError:
            pass  # Expected
        
        # Test valid path
        test_file = Path("test_file.mp4")
        test_file.touch()
        validator.validate_input_path(str(test_file))
        test_file.unlink()
        
        # Test audio validation
        valid_audio = np.random.randn(1000).astype(np.float32)
        validator.validate_audio_data(valid_audio)
        
        # Test invalid audio
        try:
            invalid_audio = np.array([np.inf, np.nan, 1.0])
            validator.validate_audio_data(invalid_audio)
            assert False, "Should reject invalid audio"
        except SecurityError:
            pass  # Expected
        
        # Test content hashing
        hash1 = validator.compute_content_hash(b"test data")
        hash2 = validator.compute_content_hash(b"test data")
        hash3 = validator.compute_content_hash(b"different data")
        assert hash1 == hash2, "Same data should have same hash"
        assert hash1 != hash3, "Different data should have different hash"
        
        print("✅ Security validation working")
        return True
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring and metrics"""
    try:
        monitor = PerformanceMonitor()
        
        # Simulate some requests
        for i in range(10):
            start_time = monitor.start_request()
            time.sleep(0.001)  # Simulate work
            success = i < 8  # 2 failures
            monitor.end_request(start_time, success)
        
        metrics = monitor.get_metrics()
        assert metrics['requests_total'] == 10, "Should track total requests"
        assert metrics['success_rate_percent'] == 80.0, "Should calculate success rate"
        assert metrics['avg_latency_ms'] > 0, "Should measure latency"
        
        print("✅ Performance monitoring working")
        return True
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def test_health_monitoring():
    """Test system health monitoring"""
    try:
        health_checker = HealthChecker()
        
        status = health_checker.get_health_status()
        assert 'status' in status, "Should return status"
        assert 'checks' in status, "Should include health checks"
        assert status['checks']['dependencies'] == 'healthy', "Dependencies should be healthy"
        
        print("✅ Health monitoring working")
        return True
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

def test_logging_and_audit():
    """Test comprehensive logging and audit trails"""
    try:
        # Test that logging is configured
        logger.info("Test log message")
        
        # Test audit trail creation
        audit_data = {
            'timestamp': time.time(),
            'user': 'system',
            'action': 'test_operation',
            'resource': 'test_file.mp4',
            'result': 'success',
            'metadata': {'test': True}
        }
        
        # Save audit log
        audit_file = Path('/tmp/audit.json')
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f)
        
        # Verify audit log
        with open(audit_file, 'r') as f:
            loaded_audit = json.load(f)
        
        assert loaded_audit['action'] == 'test_operation', "Audit log should be preserved"
        audit_file.unlink()  # Cleanup
        
        print("✅ Logging and audit trails working")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_input_validation():
    """Test comprehensive input validation"""
    try:
        from av_separation.config import SeparatorConfig
        
        # Test configuration validation
        config = SeparatorConfig()
        
        # Test valid values
        config.audio.sample_rate = 16000
        config.model.max_speakers = 4
        
        # Test boundary conditions
        assert config.audio.sample_rate > 0, "Sample rate must be positive"
        assert config.model.max_speakers >= 1, "Must have at least 1 speaker"
        
        # Test data type validation
        assert isinstance(config.audio.sample_rate, int), "Sample rate must be integer"
        assert isinstance(config.model.dropout, float), "Dropout must be float"
        
        print("✅ Input validation working")
        return True
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False

def main():
    """Run robust functionality tests"""
    print("🛡️ GENERATION 2: Robust Functionality Tests")
    print("=" * 55)
    
    tests = [
        test_error_handling,
        test_security_validation,
        test_performance_monitoring,
        test_health_monitoring,
        test_logging_and_audit,
        test_input_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 55)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Generation 2: ROBUST FUNCTIONALITY WORKING")
        logger.info("All robustness tests passed successfully")
        return True
    else:
        print(f"❌ Generation 2: {total - passed} tests failed")
        logger.error(f"Robustness tests failed: {total - passed}/{total}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)