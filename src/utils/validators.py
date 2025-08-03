"""Input validation utilities for audio-visual processing."""

from typing import Any, Dict, Union
import torch
import numpy as np
from pathlib import Path


def validate_audio_input(audio: Union[torch.Tensor, np.ndarray]) -> None:
    """Validate audio input tensor.
    
    Args:
        audio: Audio tensor to validate
        
    Raises:
        ValueError: If audio format is invalid
    """
    if not isinstance(audio, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Audio must be torch.Tensor or numpy.ndarray, got {type(audio)}")
    
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    if audio.dim() < 1 or audio.dim() > 3:
        raise ValueError(f"Audio must be 1D, 2D, or 3D tensor, got {audio.dim()}D")
    
    if audio.size(-1) < 1000:  # Less than ~62ms at 16kHz
        raise ValueError(f"Audio too short: {audio.size(-1)} samples")
    
    if torch.isnan(audio).any() or torch.isinf(audio).any():
        raise ValueError("Audio contains NaN or infinite values")
    
    # Check dynamic range
    if audio.abs().max() < 1e-6:
        raise ValueError("Audio signal too quiet (possible silence)")


def validate_video_input(video: Union[torch.Tensor, np.ndarray]) -> None:
    """Validate video input tensor.
    
    Args:
        video: Video tensor to validate
        
    Raises:
        ValueError: If video format is invalid
    """
    if not isinstance(video, (torch.Tensor, np.ndarray)):
        raise ValueError(f"Video must be torch.Tensor or numpy.ndarray, got {type(video)}")
    
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    
    if video.dim() < 2 or video.dim() > 5:
        raise ValueError(f"Video must be 2D-5D tensor, got {video.dim()}D")
    
    # Check for minimum number of frames
    if video.size(0) < 5:  # At least 5 frames
        raise ValueError(f"Video too short: {video.size(0)} frames")
    
    if torch.isnan(video).any() or torch.isinf(video).any():
        raise ValueError("Video contains NaN or infinite values")


def validate_separation_request(request: Dict[str, Any]) -> None:
    """Validate a separation request.
    
    Args:
        request: Request dictionary to validate
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['audio_input', 'video_input']
    
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate file paths if strings
    for input_type in ['audio_input', 'video_input']:
        input_val = request[input_type]
        if isinstance(input_val, (str, Path)):
            path = Path(input_val)
            if not path.exists():
                raise ValueError(f"{input_type} file does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"{input_type} path is not a file: {path}")
    
    # Validate num_speakers
    if 'num_speakers' in request:
        num_speakers = request['num_speakers']
        if not isinstance(num_speakers, int) or num_speakers < 1 or num_speakers > 6:
            raise ValueError(f"num_speakers must be integer 1-6, got {num_speakers}")


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration.
    
    Args:
        config: Model configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'num_speakers' in config:
        num_speakers = config['num_speakers']
        if not isinstance(num_speakers, int) or num_speakers < 1 or num_speakers > 6:
            raise ValueError(f"num_speakers must be 1-6, got {num_speakers}")
    
    if 'device' in config:
        device = config['device']
        if device not in ['cpu', 'cuda', 'mps']:
            if not device.startswith('cuda:'):
                raise ValueError(f"Invalid device: {device}")
    
    if 'sample_rate' in config:
        sr = config['sample_rate']
        if not isinstance(sr, int) or sr < 8000 or sr > 48000:
            raise ValueError(f"sample_rate must be 8000-48000, got {sr}")
    
    if 'chunk_size' in config:
        chunk_size = config['chunk_size']
        if not isinstance(chunk_size, (int, float)) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive number, got {chunk_size}")


def validate_export_params(
    output_path: str,
    opset_version: int = 17,
    optimize: bool = True
) -> None:
    """Validate ONNX export parameters.
    
    Args:
        output_path: Output file path
        opset_version: ONNX opset version
        optimize: Whether to optimize
        
    Raises:
        ValueError: If parameters are invalid
    """
    output_path = Path(output_path)
    
    # Check output directory exists
    if not output_path.parent.exists():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")
    
    # Check file extension
    if output_path.suffix.lower() != '.onnx':
        raise ValueError(f"Output path must have .onnx extension, got {output_path.suffix}")
    
    # Validate opset version
    if not isinstance(opset_version, int) or opset_version < 9 or opset_version > 20:
        raise ValueError(f"opset_version must be 9-20, got {opset_version}")
    
    # Validate optimize flag
    if not isinstance(optimize, bool):
        raise ValueError(f"optimize must be boolean, got {type(optimize)}")


def validate_streaming_params(
    chunk_size: float,
    overlap: float = 0.0,
    max_latency: float = 1.0
) -> None:
    """Validate streaming processing parameters.
    
    Args:
        chunk_size: Chunk size in seconds
        overlap: Overlap between chunks (0-0.5)
        max_latency: Maximum acceptable latency
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(chunk_size, (int, float)) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    
    if chunk_size > 10.0:
        raise ValueError(f"chunk_size too large for streaming: {chunk_size}s")
    
    if not isinstance(overlap, (int, float)) or overlap < 0 or overlap >= 0.5:
        raise ValueError(f"overlap must be in [0, 0.5), got {overlap}")
    
    if not isinstance(max_latency, (int, float)) or max_latency <= 0:
        raise ValueError(f"max_latency must be positive, got {max_latency}")
    
    if chunk_size > max_latency:
        raise ValueError(f"chunk_size ({chunk_size}) > max_latency ({max_latency})")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and dangerous characters
    unsafe_chars = ['/', '\\', '..', '~', '$', '&', '|', ';', '`']
    
    sanitized = filename
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure reasonable length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'output'
    
    return sanitized