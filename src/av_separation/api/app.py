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


# Global separator instance
separator = None
app_config = None


def initialize_separator():
    """Initialize the separator model"""
    global separator, app_config
    
    try:
        app_config = SeparatorConfig()
        separator = AVSeparator(
            num_speakers=4,  # Max speakers for API
            config=app_config
        )
        return True
    except Exception as e:
        print(f"Failed to initialize separator: {e}")
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
    allow_headers=["*"],
)

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
    video_file: UploadFile = File(..., description="Video file with audio track"),
    request: SeparationRequest = SeparationRequest()
):
    """
    Separate speakers from uploaded audio-visual file
    """
    
    if separator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="File name is required")
    
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.wav', '.mp3', '.flac'}
    file_ext = Path(video_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    task_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await video_file.read()
            tmp_file.write(content)
            tmp_input_path = tmp_file.name
        
        # Create output directory
        output_dir = Path(tempfile.mkdtemp(prefix=f"av_sep_{task_id}_"))
        
        # Apply config overrides
        if request.config_override:
            config = SeparatorConfig.from_dict(request.config_override)
            separator_instance = AVSeparator(
                num_speakers=request.num_speakers,
                config=config
            )
        else:
            separator_instance = separator
        
        # Perform separation
        start_time = time.time()
        
        separated_audio = separator_instance.separate(
            input_path=tmp_input_path,
            output_dir=output_dir,
            save_video=request.save_video
        )
        
        processing_time = time.time() - start_time
        
        # Get output files
        separated_files = []
        for i in range(len(separated_audio)):
            file_path = output_dir / f"speaker_{i+1}.wav"
            if file_path.exists():
                separated_files.append(str(file_path))
        
        # Compute basic metrics if reference is available
        metrics = None
        # Note: In practice, you might want to compute metrics against a reference
        
        # Clean up input file
        Path(tmp_input_path).unlink()
        
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
        # Clean up on error
        if 'tmp_input_path' in locals():
            Path(tmp_input_path).unlink(missing_ok=True)
        
        raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}")


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


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "av_separation.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )