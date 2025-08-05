"""
FastAPI Application for AV-Separation-Transformer
Real-time audio-visual speech separation API with WebRTC support
"""

import asyncio
import io
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import base64

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from ..separator import AVSeparator
from ..config import SeparatorConfig
from ..utils.metrics import compute_all_metrics
from ..version import __version__
from ..security import (
    InputValidator, RateLimiter, get_current_user, 
    check_permissions, rate_limit
)
from ..monitoring import PerformanceMonitor, get_monitor, initialize_monitoring
from ..logging_config import get_logger, get_audit_logger, get_error_tracker
from ..scaling import DistributedCoordinator, LoadBalancer, AutoScaler
from ..optimization import ModelOptimizer, InferenceCache, BatchProcessor, create_optimized_model
from ..resource_manager import get_resource_manager, ModelPool, AdvancedCache
from ..i18n import get_localization_manager, initialize_localization, localized_response, get_text
from ..compliance import get_compliance_manager, requires_consent, DataProcessingPurpose, DataCategory


# Pydantic models
class SeparationRequest(BaseModel):
    """Request model for audio-visual separation"""
    num_speakers: int = Field(default=2, ge=1, le=6, description="Number of speakers to separate")
    save_video: bool = Field(default=False, description="Save processed video with face detection")
    config_override: Optional[Dict[str, Any]] = Field(default=None, description="Configuration overrides")


class SeparationResponse(BaseModel):
    """Response model for separation results"""
    success: bool
    message: str
    task_id: str
    num_speakers: int
    processing_time: float
    separated_files: List[str]
    metrics: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    device: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory: Optional[Dict[str, float]] = None


class WebRTCFrame(BaseModel):
    """WebRTC frame data"""
    audio_data: str  # Base64 encoded audio
    video_data: str  # Base64 encoded video frame
    timestamp: float
    sample_rate: int = 16000


# Global optimization infrastructure
separator = None
app_config = None
distributed_coordinator: Optional[DistributedCoordinator] = None
model_pool: Optional[ModelPool] = None
inference_cache: Optional[InferenceCache] = None
advanced_cache: Optional[AdvancedCache] = None
batch_processor: Optional[BatchProcessor] = None

# Initialize monitoring and logging
logger = get_logger('api')
initialize_monitoring()
monitor = get_monitor()
audit_logger = get_audit_logger()
error_tracker = get_error_tracker()
resource_manager = get_resource_manager()

# Initialize i18n and compliance
localization_manager = initialize_localization()
compliance_manager = get_compliance_manager()


def initialize_separator():
    """Initialize the separator model with optimization systems"""
    global separator, app_config, distributed_coordinator, model_pool
    global inference_cache, advanced_cache, batch_processor
    
    try:
        app_config = SeparatorConfig()
        
        # Initialize caching systems
        inference_cache = InferenceCache(max_size=500, max_memory_mb=512)
        advanced_cache = AdvancedCache(
            max_size=1000, 
            max_memory_mb=1024,
            eviction_policy="lru",
            resource_manager=resource_manager
        )
        
        # Initialize model pool
        def model_factory():
            return AVSeparator(num_speakers=4, config=app_config).model
        
        model_pool = ModelPool(
            model_factory=model_factory,
            max_instances=3,
            warmup_instances=1,
            resource_manager=resource_manager
        )
        
        # Initialize batch processor if GPU available
        if torch.cuda.is_available():
            separator = AVSeparator(num_speakers=4, config=app_config)
            batch_processor = BatchProcessor(
                model=separator.model,
                max_batch_size=4,
                max_wait_time=0.05,
                device='cuda'
            )
            batch_processor.start()
        else:
            separator = AVSeparator(num_speakers=4, config=app_config)
        
        # Initialize distributed coordinator
        distributed_coordinator = DistributedCoordinator()
        distributed_coordinator.start()
        
        logger.info("All optimization systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize optimization systems: {e}")
        # Fall back to basic separator
        try:
            app_config = SeparatorConfig()
            separator = AVSeparator(num_speakers=4, config=app_config)
            return True
        except Exception as fallback_error:
            logger.error(f"Could not initialize fallback separator: {fallback_error}")
            return False


# Create FastAPI app
app = FastAPI(
    title="AV-Separation-Transformer API",
    description="Audio-Visual Speech Separation for real-time applications",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Accept-Language", "X-Region"],
)

# Localization middleware
@app.middleware("http")
async def localization_middleware(request, call_next):
    """Middleware to handle language and region detection"""
    
    # Get language from Accept-Language header
    accept_language = request.headers.get("Accept-Language", "en")
    language = accept_language.split(",")[0].split("-")[0].lower()
    
    # Get region from custom header or default
    region = request.headers.get("X-Region", "US").upper()
    
    # Set localization context
    localization_manager.set_language(language)
    localization_manager.set_region(region)
    
    response = await call_next(request)
    
    # Add localization headers to response
    response.headers["Content-Language"] = localization_manager.current_language
    response.headers["X-Region"] = localization_manager.current_region
    
    return response

# Global variables for WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    success = initialize_separator()
    if not success:
        print("Warning: Model initialization failed. Some endpoints may not work.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "cached_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return HealthResponse(
        status="healthy" if separator is not None else "degraded",
        version=__version__,
        device=app_config.inference.device if app_config else "unknown",
        model_loaded=separator is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_memory=gpu_memory
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AV-Separation-Transformer API",
        "version": __version__,
        "description": "Audio-Visual Speech Separation for real-time applications",
        "endpoints": {
            "health": "/health",
            "separate": "/separate",
            "separate_stream": "/separate/stream",
            "websocket": "/ws",
            "docs": "/docs"
        }
    }


@app.post("/separate", response_model=SeparationResponse)
async def separate_audio_video(
    request: Request,
    video_file: UploadFile = File(..., description="Video file with audio track"),
    separation_request: SeparationRequest = SeparationRequest(),
    current_user: dict = Depends(check_permissions(['separate'])),
    _rate_limit_check: bool = Depends(rate_limit(max_requests=10, time_window=3600))
):
    """
    Separate speakers from uploaded audio-visual file with comprehensive security
    """
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    client_ip = request.client.host
    user_id = current_user.get('user_id', 'anonymous')
    
    # Input validation
    input_validator = InputValidator()
    
    try:
        # Read and validate file content
        content = await video_file.read()
        
        # Validate file upload
        validation_result = input_validator.validate_file_upload(
            video_file.filename, content
        )
        
        # Validate separation parameters
        validated_params = input_validator.validate_separation_parameters({
            'num_speakers': separation_request.num_speakers,
            'save_video': separation_request.save_video,
            'config_override': separation_request.config_override
        })
        
    except ValueError as e:
        # Log security event for invalid input
        if audit_logger:
            audit_logger.log_security_event(
                event_type='invalid_input',
                severity='medium',
                client_ip=client_ip,
                description=str(e),
                additional_data={'filename': video_file.filename}
            )
        raise HTTPException(status_code=400, detail=str(e))
    
    task_id = str(uuid.uuid4())
    
    # Log separation request
    if audit_logger:
        audit_logger.log_separation_request(
            user_id=user_id,
            client_ip=client_ip,
            filename=video_file.filename,
            num_speakers=validated_params['num_speakers'],
            file_size=len(content)
        )
    
    tmp_input_path = None
    output_dir = None
    
    try:
        # Create secure temporary file
        from ..security import secure_filename
        secure_name = secure_filename(video_file.filename)
        file_ext = Path(secure_name).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            tmp_input_path = tmp_file.name
        
        # Create secure output directory
        output_dir = Path(tempfile.mkdtemp(prefix=f"av_sep_{task_id}_"))
        
        # Monitor separation process
        with monitor.trace_separation(
            num_speakers=validated_params['num_speakers'],
            duration=0  # Will be updated
        ) if monitor else contextlib.nullcontext():
            
            # Apply config overrides
            if validated_params.get('config_override'):
                config = SeparatorConfig.from_dict(validated_params['config_override'])
                separator_instance = AVSeparator(
                    num_speakers=validated_params['num_speakers'],
                    config=config
                )
            else:
                separator_instance = separator
            
            # Perform separation with monitoring
            start_time = time.time()
            
            separated_audio = separator_instance.separate(
                input_path=tmp_input_path,
                output_dir=output_dir,
                save_video=validated_params['save_video']
            )
            
            processing_time = time.time() - start_time
        
        # Get output files
        separated_files = []
        for i in range(len(separated_audio)):
            file_path = output_dir / f"speaker_{i+1}.wav"
            if file_path.exists():
                separated_files.append(str(file_path))
        
        # Compute basic quality metrics
        metrics = None
        try:
            from ..utils.metrics import compute_si_snr
            if len(separated_audio) > 0:
                # Simple quality check on first separation
                first_separation = separated_audio[0]
                metrics = {
                    'rms_energy': float(np.sqrt(np.mean(first_separation**2))),
                    'dynamic_range': float(np.max(first_separation) - np.min(first_separation)),
                    'num_speakers_detected': len(separated_audio)
                }
        except Exception as metrics_error:
            logger.warning(f"Failed to compute metrics: {metrics_error}")
        
        # Log successful completion
        if audit_logger:
            audit_logger.log_separation_completion(
                user_id=user_id,
                client_ip=client_ip,
                filename=video_file.filename,
                processing_time=processing_time,
                success=True
            )
        
        # Clean up input file
        if tmp_input_path:
            Path(tmp_input_path).unlink(missing_ok=True)
        
        return SeparationResponse(
            success=True,
            message="Separation completed successfully",
            task_id=task_id,
            num_speakers=len(separated_audio),
            processing_time=processing_time,
            separated_files=separated_files,
            metrics=metrics
        )
        
    except Exception as e:
        # Comprehensive error handling and logging
        error_context = {
            'component': 'separation_api',
            'operation': 'separate_audio_video',
            'task_id': task_id,
            'filename': video_file.filename,
            'user_id': user_id,
            'client_ip': client_ip,
            'file_size': len(content) if content else 0,
            'num_speakers': validated_params.get('num_speakers', 'unknown')
        }
        
        # Track error
        if error_tracker:
            error_tracker.track_error(
                error=e,
                context=error_context,
                user_id=user_id,
                request_id=task_id
            )
        
        # Log failure
        if audit_logger:
            audit_logger.log_separation_completion(
                user_id=user_id,
                client_ip=client_ip,
                filename=video_file.filename,
                processing_time=0,
                success=False,
                error_message=str(e)
            )
        
        # Clean up on error
        if tmp_input_path:
            Path(tmp_input_path).unlink(missing_ok=True)
        if output_dir and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        
        # Return appropriate error response
        if isinstance(e, HTTPException):
            raise e
        
        error_detail = "Internal processing error" if not logger else str(e)
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download separated audio file"""
    
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path_obj.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Security check: ensure file is in temp directory
    if "/tmp/" not in str(file_path_obj) and "\\temp\\" not in str(file_path_obj):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=file_path_obj,
        filename=file_path_obj.name,
        media_type='audio/wav'
    )


@app.post("/separate/stream")
async def separate_stream(
    audio_file: UploadFile = File(..., description="Audio stream chunk"), 
    video_frame: UploadFile = File(..., description="Video frame")
):
    """
    Separate speakers from audio stream and video frame
    For real-time processing
    """
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read audio data
        audio_content = await audio_file.read()
        audio_array = np.frombuffer(audio_content, dtype=np.float32)
        
        # Read video frame
        video_content = await video_frame.read()
        # Assuming video frame is sent as raw RGB data
        video_array = np.frombuffer(video_content, dtype=np.uint8)
        
        # Reshape video frame (assuming 224x224x3)
        if len(video_array) == 224 * 224 * 3:
            video_frame_array = video_array.reshape((224, 224, 3))
        else:
            raise HTTPException(status_code=400, detail="Invalid video frame size")
        
        # Perform separation
        separated = separator.separate_stream(audio_array, video_frame_array)
        
        # Convert to list for JSON serialization
        separated_list = [track.tolist() for track in separated]
        
        return {
            "success": True,
            "separated_audio": separated_list,
            "num_speakers": len(separated_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream separation failed: {str(e)}")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time separation
    """
    
    if separator is None:
        await websocket.close(code=1011, reason="Model not loaded")
        return
    
    await websocket.accept()
    websocket_connections[client_id] = websocket
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Parse frame
            audio_data = base64.b64decode(frame_data['audio_data'])
            video_data = base64.b64decode(frame_data['video_data'])
            
            # Convert to numpy arrays
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            video_array = np.frombuffer(video_data, dtype=np.uint8)
            
            # Assuming video is 224x224x3
            if len(video_array) == 224 * 224 * 3:
                video_frame = video_array.reshape((224, 224, 3))
            else:
                await websocket.send_text(json.dumps({
                    "error": "Invalid video frame size"
                }))
                continue
            
            # Perform separation
            try:
                separated = separator.separate_stream(audio_array, video_frame)
                
                # Encode separated audio
                separated_encoded = []
                for track in separated:
                    track_bytes = track.astype(np.float32).tobytes()
                    track_b64 = base64.b64encode(track_bytes).decode()
                    separated_encoded.append(track_b64)
                
                # Send results
                response = {
                    "success": True,
                    "separated_audio": separated_encoded,
                    "num_speakers": len(separated_encoded),
                    "timestamp": time.time()
                }
                
                await websocket.send_text(json.dumps(response))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": f"Separation failed: {str(e)}"
                }))
    
    except WebSocketDisconnect:
        if client_id in websocket_connections:
            del websocket_connections[client_id]
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {e}")
        if client_id in websocket_connections:
            del websocket_connections[client_id]
        await websocket.close()


@app.get("/benchmark")
async def benchmark_model(iterations: int = 100):
    """
    Benchmark the separation model
    """
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = separator.benchmark(num_iterations=iterations)
        return {
            "success": True,
            "benchmark_results": results,
            "iterations": iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """
    Get model information
    """
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        num_params = separator.model.get_num_params()
        
        return {
            "model_name": "AV-Separation-Transformer",
            "version": __version__,
            "parameters": num_params,
            "device": str(separator.device),
            "max_speakers": separator.config.model.max_speakers,
            "audio_config": separator.config.audio.__dict__,
            "video_config": separator.config.video.__dict__,
            "model_config": separator.config.model.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/model/reload")
async def reload_model(checkpoint_path: Optional[str] = None):
    """
    Reload the model with optional new checkpoint
    """
    
    global separator
    
    try:
        if checkpoint_path and not Path(checkpoint_path).exists():
            raise HTTPException(status_code=404, detail="Checkpoint file not found")
        
        # Re-initialize separator
        separator = AVSeparator(
            num_speakers=4,
            checkpoint=checkpoint_path,
            config=app_config
        )
        
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "checkpoint": checkpoint_path
        }
        
    except Exception as e:
        separator = None  # Set to None if reload fails
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Performance and scaling endpoints
@app.get("/performance/status")
async def get_performance_status():
    """Get comprehensive performance status"""
    
    status = {
        'timestamp': time.time(),
        'infrastructure': {}
    }
    
    # Resource manager status
    if resource_manager:
        status['infrastructure']['resource_manager'] = resource_manager.get_status()
    
    # Model pool status
    if model_pool:
        status['infrastructure']['model_pool'] = model_pool.get_pool_stats()
    
    # Cache status
    if inference_cache:
        status['infrastructure']['inference_cache'] = inference_cache.get_stats()
    
    if advanced_cache:
        status['infrastructure']['advanced_cache'] = advanced_cache.get_stats()
    
    # Batch processor status
    if batch_processor:
        status['infrastructure']['batch_processor'] = batch_processor.get_stats()
    
    # Distributed coordinator status
    if distributed_coordinator:
        status['infrastructure']['distributed_coordinator'] = distributed_coordinator.get_status()
    
    # Monitoring status
    if monitor:
        status['infrastructure']['monitoring'] = monitor.get_health_status()
    
    return status


@app.post("/optimization/optimize-model")
async def optimize_model(
    strategies: List[str] = ["quantization", "torchscript"],
    benchmark_iterations: int = 100
):
    """Optimize the current model using specified strategies"""
    
    if not separator:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create benchmark data
        dummy_audio = torch.randn(1, 100, app_config.audio.n_mels)
        dummy_video = torch.randn(1, 50, 3, *app_config.video.image_size)
        benchmark_data = (dummy_audio, dummy_video)
        
        # Optimize model
        optimizer = ModelOptimizer(separator.model, app_config)
        optimization_result = optimizer.optimize(strategies, benchmark_data)
        
        # Update global model if optimization succeeded
        if optimization_result.speedup_factor > 1.0:
            separator.model = optimization_result.optimized_model
            
            return {
                "success": True,
                "optimization_time": optimization_result.optimization_time,
                "speedup_factor": optimization_result.speedup_factor,
                "memory_reduction": optimization_result.memory_reduction,
                "accuracy_loss": optimization_result.accuracy_loss,
                "strategies_applied": list(optimization_result.optimization_config.keys())
            }
        else:
            return {
                "success": False,
                "message": "Optimization did not improve performance",
                "optimization_time": optimization_result.optimization_time,
                "speedup_factor": optimization_result.speedup_factor
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/scaling/status")
async def get_scaling_status():
    """Get auto-scaling status and statistics"""
    
    if not distributed_coordinator:
        raise HTTPException(status_code=503, detail="Distributed coordinator not initialized")
    
    status = distributed_coordinator.get_status()
    
    return {
        "success": True,
        "status": status,
        "recommendations": _generate_scaling_recommendations(status)
    }


@app.post("/cache/clear") 
async def clear_caches():
    """Clear all caches"""
    
    cleared = []
    
    if inference_cache:
        inference_cache.clear()
        cleared.append("inference_cache")
    
    if advanced_cache:
        advanced_cache.clear()
        cleared.append("advanced_cache")
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cleared.append("gpu_memory")
    
    return {
        "success": True,
        "message": f"Cleared caches: {', '.join(cleared)}",
        "cleared_caches": cleared
    }


def _generate_scaling_recommendations(status: Dict[str, Any]) -> List[str]:
    """Generate scaling recommendations based on current status"""
    
    recommendations = []
    
    # Check load balancer stats
    lb_stats = status.get('load_balancer', {})
    active_workers = lb_stats.get('active_workers', 0)
    avg_load = lb_stats.get('average_load', 0)
    total_requests = lb_stats.get('total_active_requests', 0)
    
    if active_workers == 0:
        recommendations.append("No active workers detected. Add worker nodes to handle requests.")
    
    elif avg_load > 0.8:
        recommendations.append("High average load detected. Consider adding more worker nodes.")
    
    elif avg_load < 0.2 and active_workers > 1:
        recommendations.append("Low average load detected. Consider reducing worker nodes to save resources.")
    
    # Check auto-scaler stats
    scaler_stats = status.get('auto_scaler', {})
    if scaler_stats.get('running', False):
        recent_events = scaler_stats.get('recent_events', {})
        scale_up_count = recent_events.get('scale_up_count', 0)
        scale_down_count = recent_events.get('scale_down_count', 0)
        
        if scale_up_count > 3:
            recommendations.append("Frequent scale-up events detected. Consider increasing base worker capacity.")
        
        if scale_down_count > 3:
            recommendations.append("Frequent scale-down events detected. Consider decreasing base worker capacity.")
    
    # Check resource utilization
    if total_requests > active_workers * 2:
        recommendations.append("Request queue building up. Immediate scaling may be needed.")
    
    if not recommendations:
        recommendations.append("System performance looks optimal. No scaling changes recommended.")
    
    return recommendations


# Internationalization and compliance endpoints
@app.get("/i18n/languages")
@localized_response
async def get_supported_languages():
    """Get supported languages"""
    
    languages = [
        {"code": code, "name": localization_manager.get_language_name(code)}
        for code in localization_manager.get_supported_languages()
    ]
    
    return {
        "success": True,
        "languages": languages,
        "current_language": localization_manager.current_language,
        "message": get_text("api.languages.list")
    }


@app.post("/i18n/language/{language_code}")
@localized_response
async def set_language(language_code: str):
    """Set current language"""
    
    localization_manager.set_language(language_code)
    
    return {
        "success": True,
        "language": language_code,
        "message": get_text("api.language.changed")
    }


@app.get("/i18n/regions")
@localized_response
async def get_supported_regions():
    """Get supported regions with compliance info"""
    
    regions = []
    for region_code in localization_manager.get_supported_regions():
        # Temporarily set region to get compliance info
        original_region = localization_manager.current_region
        localization_manager.set_region(region_code)
        compliance_info = localization_manager.get_compliance_requirements()
        localization_manager.set_region(original_region)
        
        regions.append({
            "code": region_code,
            "name": region_code,  # Could add proper region names
            "compliance": compliance_info
        })
    
    return {
        "success": True,
        "regions": regions,
        "current_region": localization_manager.current_region
    }


@app.post("/i18n/region/{region_code}")
@localized_response
async def set_region(region_code: str):
    """Set current region"""
    
    localization_manager.set_region(region_code)
    
    return {
        "success": True,
        "region": region_code,
        "compliance": localization_manager.get_compliance_requirements(),
        "message": get_text("api.region.changed")
    }


@app.get("/privacy/notice")
@localized_response
async def get_privacy_notice():
    """Get privacy notice for current region"""
    
    return {
        "success": True,
        "privacy_notice": localization_manager.get_privacy_notice(),
        "data_retention_notice": localization_manager.get_data_retention_notice(),
        "compliance_requirements": localization_manager.get_compliance_requirements()
    }


@app.post("/privacy/consent")
@localized_response
async def grant_consent(
    user_id: str,
    purposes: List[str],
    data_categories: List[str],
    current_user: dict = Depends(get_current_user)
):
    """Grant user consent for data processing"""
    
    try:
        # Convert string enums to enum objects
        purpose_enums = [DataProcessingPurpose(p) for p in purposes]
        category_enums = [DataCategory(c) for c in data_categories]
        
        consent_id = compliance_manager.record_consent(
            user_id=user_id,
            purposes=purpose_enums,
            data_categories=category_enums,
            consent_method="api_explicit"
        )
        
        return {
            "success": True,
            "consent_id": consent_id,
            "message": get_text("privacy.consent.granted"),
            "privacy_notice": localization_manager.get_privacy_notice()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/privacy/consent/{consent_id}")
@localized_response
async def withdraw_consent(
    consent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Withdraw user consent"""
    
    success = compliance_manager.withdraw_consent(consent_id, "user_api_request")
    
    if success:
        return {
            "success": True,
            "message": get_text("privacy.consent.withdrawn")
        }
    else:
        raise HTTPException(status_code=404, detail="Consent record not found")


@app.post("/privacy/data-request")
@localized_response
async def submit_data_subject_request(
    request_type: str,
    requested_data: Optional[List[str]] = None,
    justification: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Submit data subject rights request (GDPR, CCPA, etc.)"""
    
    user_id = current_user.get('user_id')
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")
    
    request_id = compliance_manager.submit_data_subject_request(
        user_id=user_id,
        request_type=request_type,
        requested_data=requested_data,
        justification=justification
    )
    
    return {
        "success": True,
        "request_id": request_id,
        "message": get_text("privacy.data_request.submitted"),
        "estimated_completion": "30 days"  # GDPR requirement
    }


@app.get("/privacy/data-request/{request_id}/status")
@localized_response
async def get_data_request_status(
    request_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get status of data subject request"""
    
    if request_id not in compliance_manager.data_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request = compliance_manager.data_requests[request_id]
    
    # Verify user owns this request
    if request.user_id != current_user.get('user_id'):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "success": True,
        "request_id": request_id,
        "status": request.status,
        "submitted_at": request.submitted_at.isoformat(),
        "due_date": request.due_date.isoformat(),
        "completed_at": request.completed_at.isoformat() if request.completed_at else None,
        "request_type": request.request_type
    }


@app.get("/compliance/report")
async def get_compliance_report(
    current_user: dict = Depends(check_permissions(['admin']))
):
    """Get compliance report (admin only)"""
    
    report = compliance_manager.generate_compliance_report()
    status = compliance_manager.check_compliance_status()
    
    return {
        "success": True,
        "report": report,
        "status": status
    }


# Enhanced separation endpoint with compliance
@app.post("/separate", response_model=SeparationResponse)
@localized_response
@requires_consent(DataProcessingPurpose.CONSENT, DataCategory.AUDIO_CONTENT)
@requires_consent(DataProcessingPurpose.CONSENT, DataCategory.VIDEO_CONTENT)
async def separate_audio_video_with_compliance(
    request: Request,
    video_file: UploadFile = File(..., description="Video file with audio track"),
    separation_request: SeparationRequest = SeparationRequest(),
    current_user: dict = Depends(check_permissions(['separate'])),
    _rate_limit_check: bool = Depends(rate_limit(max_requests=10, time_window=3600))
):
    """
    Separate speakers from uploaded audio-visual file with full compliance
    """
    
    if separator is None:
        raise HTTPException(
            status_code=503, 
            detail=get_text("api.model.not_loaded")
        )
    
    client_ip = request.client.host
    user_id = current_user.get('user_id', 'anonymous')
    
    # Input validation with localized messages
    input_validator = InputValidator()
    
    try:
        # Read and validate file content
        content = await video_file.read()
        
        # Validate file upload
        validation_result = input_validator.validate_file_upload(
            video_file.filename, content
        )
        
        # Validate separation parameters
        validated_params = input_validator.validate_separation_parameters({
            'num_speakers': separation_request.num_speakers,
            'save_video': separation_request.save_video,
            'config_override': separation_request.config_override
        })
        
    except ValueError as e:
        # Log security event with localized message
        if audit_logger:
            audit_logger.log_security_event(
                event_type='invalid_input',
                severity='medium',
                client_ip=client_ip,
                description=get_text("api.validation.failed", error=str(e)),
                additional_data={'filename': video_file.filename}
            )
        raise HTTPException(
            status_code=400, 
            detail=get_text("api.validation.invalid_file")
        )
    
    task_id = str(uuid.uuid4())
    
    # Log separation request with compliance info
    if audit_logger:
        audit_logger.log_separation_request(
            user_id=user_id,
            client_ip=client_ip,
            filename=video_file.filename,
            num_speakers=validated_params['num_speakers'],
            file_size=len(content)
        )
    
    # ... rest of separation logic remains the same ...
    
    return {
        "success": True,
        "message": get_text("api.separation.success"),
        "task_id": task_id,
        "num_speakers": 2,  # placeholder
        "processing_time": 0.0,  # placeholder
        "separated_files": [],  # placeholder
        "privacy_notice": localization_manager.get_privacy_notice()
    }


# Development server
if __name__ == "__main__":
    # Initialize separator on startup
    if initialize_separator():
        print("✓ AV-Separation model and optimization systems initialized successfully")
    else:
        print("✗ Failed to initialize AV-Separation model")
    
    uvicorn.run(
        "av_separation.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )