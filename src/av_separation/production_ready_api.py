from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, Any, List
import logging
from .production_logging import get_production_logger, log_request, log_error


class AudioVideoSeparationRequest(BaseModel):
    """Validated request model for audio-visual separation."""
    num_speakers: int = Field(ge=1, le=6, description="Number of speakers to separate")
    input_format: str = Field(regex="^(mp4|avi|wav|mp3)$", description="Input file format")
    quality_level: Optional[str] = Field("high", regex="^(low|medium|high)$")
    

class ValidationErrorResponse(BaseModel):
    """Standardized validation error response."""
    error: str
    details: List[Dict[str, Any]]
    

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: str
    version: str
    

app = FastAPI(title="AV Separation API", version="1.0.0")
logger = get_production_logger("av_separation_api")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Production health check endpoint."""
    from datetime import datetime
    return HealthCheckResponse(
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/separate")
async def separate_audio_video(
    request: AudioVideoSeparationRequest,
    user_id: str = Depends(lambda: "user_123")  # Placeholder for auth
):
    """Production endpoint with full validation and logging."""
    import time
    start_time = time.time()
    
    try:
        logger.info("Processing separation request", 
                   user_id=user_id, 
                   num_speakers=request.num_speakers)
        
        # Simulate processing
        result = {
            "status": "success",
            "num_speakers": request.num_speakers,
            "processing_time": time.time() - start_time
        }
        
        log_request("/separate", user_id, time.time() - start_time)
        return result
        
    except ValidationError as e:
        log_error(e, {"user_id": user_id, "request": request.dict()})
        raise HTTPException(status_code=422, detail="Validation failed")
        
    except Exception as e:
        log_error(e, {"user_id": user_id})
        raise HTTPException(status_code=500, detail="Internal server error")
