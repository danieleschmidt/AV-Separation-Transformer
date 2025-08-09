
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import time
import torch
import psutil
import os

app = FastAPI(title="AV-Separation Health Check", version="1.0.0")

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_ok = memory.percent < 90
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_ok = (disk.free / disk.total) > 0.1  # 10% free space
        
        # Overall health
        healthy = gpu_available and memory_ok and disk_ok
        
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "healthy" if healthy else "unhealthy",
                "timestamp": time.time(),
                "checks": {
                    "gpu_available": gpu_available,
                    "gpu_count": gpu_count,
                    "memory_ok": memory_ok,
                    "disk_ok": disk_ok
                },
                "system": {
                    "memory_percent": memory.percent,
                    "disk_free_percent": (disk.free / disk.total) * 100,
                    "environment": os.getenv("ENVIRONMENT", "unknown")
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Quick model availability check
        from av_separation import SeparatorConfig
        config = SeparatorConfig()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "timestamp": time.time(),
                "model_config_loaded": True
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        metrics_text = f"""
# HELP av_separation_memory_usage_percent Memory usage percentage
# TYPE av_separation_memory_usage_percent gauge
av_separation_memory_usage_percent {memory.percent}

# HELP av_separation_cpu_usage_percent CPU usage percentage  
# TYPE av_separation_cpu_usage_percent gauge
av_separation_cpu_usage_percent {cpu_percent}

# HELP av_separation_gpu_available GPU availability
# TYPE av_separation_gpu_available gauge
av_separation_gpu_available {1 if torch.cuda.is_available() else 0}
"""
        
        return Response(
            content=metrics_text.strip(),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
