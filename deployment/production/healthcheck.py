#!/usr/bin/env python3
"""
Health check script for AV-Separation API service.
"""

import sys
import requests
import time
import os
from pathlib import Path


def check_api_health():
    """Check API health endpoint."""
    try:
        host = os.getenv('HOST', '0.0.0.0')
        port = os.getenv('PORT', '8000')
        url = f"http://{host}:{port}/health"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Check critical components
            if (health_data.get('status') == 'healthy' and
                health_data.get('database', {}).get('status') == 'connected' and
                health_data.get('cache', {}).get('status') == 'connected'):
                return True, "API is healthy"
            else:
                return False, f"API unhealthy: {health_data}"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {e}"
    except Exception as e:
        return False, f"Health check error: {e}"


def check_gpu_health():
    """Check GPU availability if CUDA is expected."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Test GPU memory allocation
                test_tensor = torch.randn(100, 100).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                return True, f"GPU healthy: {device_count} devices"
            else:
                return False, "No CUDA devices available"
        else:
            # If CUDA is not available but we're in GPU mode, that's an error
            if os.getenv('WORKER_TYPE') == 'gpu':
                return False, "CUDA not available in GPU worker"
            else:
                return True, "CPU mode - no GPU required"
    
    except ImportError:
        return True, "PyTorch not available - skipping GPU check"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def check_disk_space():
    """Check available disk space."""
    try:
        # Check available space in /app and /tmp
        paths_to_check = ['/app', '/tmp']
        
        for path in paths_to_check:
            if os.path.exists(path):
                stat = os.statvfs(path)
                available_bytes = stat.f_bavail * stat.f_frsize
                available_gb = available_bytes / (1024**3)
                
                if available_gb < 1.0:  # Less than 1GB available
                    return False, f"Low disk space in {path}: {available_gb:.2f}GB"
        
        return True, "Disk space OK"
    
    except Exception as e:
        return False, f"Disk space check failed: {e}"


def check_memory_usage():
    """Check memory usage."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 95:
            return False, f"High memory usage: {memory_percent:.1f}%"
        elif memory_percent > 85:
            return True, f"Memory usage warning: {memory_percent:.1f}%"
        else:
            return True, f"Memory usage OK: {memory_percent:.1f}%"
    
    except ImportError:
        return True, "psutil not available - skipping memory check"
    except Exception as e:
        return False, f"Memory check failed: {e}"


def check_log_files():
    """Check that log files are writable."""
    try:
        log_dir = Path('/app/logs')
        
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = log_dir / 'healthcheck.test'
        test_file.write_text(f"Health check at {time.time()}")
        test_file.unlink()
        
        return True, "Log directory writable"
    
    except Exception as e:
        return False, f"Log directory check failed: {e}"


def main():
    """Run all health checks."""
    checks = [
        ("API Health", check_api_health),
        ("GPU Health", check_gpu_health),
        ("Disk Space", check_disk_space),
        ("Memory Usage", check_memory_usage),
        ("Log Files", check_log_files)
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            results.append(f"{check_name}: {'✓' if passed else '✗'} {message}")
            
            if not passed:
                all_passed = False
        
        except Exception as e:
            results.append(f"{check_name}: ✗ Exception: {e}")
            all_passed = False
    
    # Print results
    for result in results:
        print(result)
    
    # Exit with appropriate code
    if all_passed:
        print("All health checks passed!")
        sys.exit(0)
    else:
        print("Some health checks failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()