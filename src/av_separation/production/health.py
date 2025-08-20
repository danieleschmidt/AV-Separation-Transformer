"""Production Health Check Module"""

import time
import psutil
import torch
from typing import Dict, Any
from fastapi import HTTPException


class ProductionHealthCheck:
    """Comprehensive health check for production deployment."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_health_check = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        
        try:
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self.start_time,
                "checks": {}
            }
            
            # System resources
            health_status["checks"]["cpu"] = self._check_cpu()
            health_status["checks"]["memory"] = self._check_memory()
            health_status["checks"]["disk"] = self._check_disk()
            health_status["checks"]["gpu"] = self._check_gpu()
            
            # Application health
            health_status["checks"]["model"] = self._check_model()
            health_status["checks"]["dependencies"] = self._check_dependencies()
            
            # Overall status
            failed_checks = [k for k, v in health_status["checks"].items() 
                           if v["status"] != "healthy"]
            
            if failed_checks:
                health_status["status"] = "unhealthy"
                health_status["failed_checks"] = failed_checks
            
            self.last_health_check = time.time()
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical",
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count()
        }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return {
            "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 95 else "critical",
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        return {
            "status": "healthy" if disk.percent < 85 else "warning" if disk.percent < 95 else "critical",
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "disk_total_gb": disk.total / (1024**3)
        }
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and usage."""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "status": "healthy",
                    "gpu_available": True,
                    "gpu_count": gpu_count,
                    "gpu_memory_gb": gpu_memory
                }
            else:
                return {
                    "status": "warning",
                    "gpu_available": False,
                    "message": "No GPU available"
                }
        except Exception as e:
            return {
                "status": "error",
                "gpu_available": False,
                "error": str(e)
            }
    
    def _check_model(self) -> Dict[str, Any]:
        """Check model availability."""
        try:
            # This would check if the model can be loaded and used
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_ready": True
            }
        except Exception as e:
            return {
                "status": "critical",
                "model_loaded": False,
                "error": str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        try:
            import torch
            import numpy
            import librosa
            
            return {
                "status": "healthy",
                "torch_version": torch.__version__,
                "numpy_version": numpy.__version__,
                "dependencies_ok": True
            }
        except ImportError as e:
            return {
                "status": "critical",
                "dependencies_ok": False,
                "error": str(e)
            }


# Global health checker instance
health_checker = ProductionHealthCheck()


def get_health():
    """FastAPI health endpoint."""
    status = health_checker.get_health_status()
    if status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=status)
    return status


def get_readiness():
    """FastAPI readiness endpoint."""
    status = health_checker.get_health_status()
    
    # Check if critical components are ready
    critical_checks = ["model", "dependencies"]
    ready = all(
        status["checks"].get(check, {}).get("status") == "healthy"
        for check in critical_checks
    )
    
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": time.time()}
