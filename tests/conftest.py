"""
Pytest configuration and fixtures for AV-Separation-Transformer
"""

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Generator
import json

from src.av_separation import AVSeparator, SeparatorConfig
from src.av_separation.models import AVSeparationTransformer


@pytest.fixture(scope="session")
def test_config() -> SeparatorConfig:
    """Create test configuration with reduced parameters for faster testing"""
    
    config = SeparatorConfig()
    
    # Reduce model complexity for testing
    config.model.audio_encoder_layers = 2
    config.model.audio_encoder_dim = 128
    config.model.audio_encoder_ffn_dim = 256
    
    config.model.video_encoder_layers = 2
    config.model.video_encoder_dim = 64
    config.model.video_encoder_ffn_dim = 128
    
    config.model.fusion_layers = 2
    config.model.fusion_dim = 128
    
    config.model.decoder_layers = 2
    config.model.decoder_dim = 128
    config.model.decoder_ffn_dim = 256
    
    config.model.max_speakers = 3
    
    # Reduce audio/video complexity
    config.audio.n_mels = 40
    config.audio.chunk_duration = 2.0
    config.audio.sample_rate = 8000
    
    config.video.image_size = (64, 64)
    config.video.face_size = (32, 32)
    config.video.lip_size = (24, 24)
    config.video.fps = 10
    
    return config


@pytest.fixture(scope="session")
def test_model(test_config: SeparatorConfig) -> AVSeparationTransformer:
    """Create test model instance"""
    
    model = AVSeparationTransformer(test_config)
    model.eval()
    
    return model


@pytest.fixture(scope="session")
def test_separator(test_config: SeparatorConfig) -> AVSeparator:
    """Create test separator instance"""
    
    separator = AVSeparator(
        num_speakers=2,
        device='cpu',  # Force CPU for testing
        config=test_config
    )
    
    return separator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files"""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio_data(test_config: SeparatorConfig) -> Dict[str, np.ndarray]:
    """Generate sample audio data for testing"""
    
    sample_rate = test_config.audio.sample_rate
    duration = 2.0  # 2 seconds
    samples = int(duration * sample_rate)
    
    # Generate synthetic audio signals
    t = np.linspace(0, duration, samples)
    
    # Speaker 1: Lower frequency
    freq1 = 200
    speaker1 = 0.5 * np.sin(2 * np.pi * freq1 * t)
    
    # Speaker 2: Higher frequency
    freq2 = 400
    speaker2 = 0.3 * np.sin(2 * np.pi * freq2 * t)
    
    # Mix the speakers
    mixture = speaker1 + speaker2
    
    # Add some noise
    noise = 0.1 * np.random.randn(samples)
    mixture += noise
    
    return {
        'mixture': mixture.astype(np.float32),
        'speaker1': speaker1.astype(np.float32),
        'speaker2': speaker2.astype(np.float32),
        'sample_rate': sample_rate,
        'duration': duration
    }


@pytest.fixture
def sample_video_data(test_config: SeparatorConfig) -> Dict[str, np.ndarray]:
    """Generate sample video data for testing"""
    
    fps = test_config.video.fps
    duration = 2.0  # 2 seconds
    num_frames = int(duration * fps)
    height, width = test_config.video.image_size
    
    # Generate synthetic video frames
    frames = np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)
    
    # Add some structure to simulate faces
    for i in range(num_frames):
        # Simple rectangle to simulate a face
        face_y = height // 4
        face_x = width // 4
        face_h = height // 2
        face_w = width // 2
        
        # Fill face region with different color
        frames[i, face_y:face_y+face_h, face_x:face_x+face_w, :] = [180, 150, 120]  # Skin tone
        
        # Add eyes
        eye_y = face_y + face_h // 3
        eye1_x = face_x + face_w // 4
        eye2_x = face_x + 3 * face_w // 4
        eye_size = 3
        
        frames[i, eye_y:eye_y+eye_size, eye1_x:eye1_x+eye_size, :] = [0, 0, 0]  # Black eyes
        frames[i, eye_y:eye_y+eye_size, eye2_x:eye2_x+eye_size, :] = [0, 0, 0]
        
        # Add mouth
        mouth_y = face_y + 2 * face_h // 3
        mouth_x = face_x + face_w // 3
        mouth_w = face_w // 3
        mouth_h = 2
        
        frames[i, mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w, :] = [120, 80, 80]  # Mouth
    
    return {
        'frames': frames,
        'fps': fps,
        'duration': duration,
        'num_frames': num_frames
    }


@pytest.fixture
def sample_mixed_data(sample_audio_data: Dict, sample_video_data: Dict) -> Dict[str, Any]:
    """Combine audio and video data for testing"""
    
    return {
        'audio': sample_audio_data,
        'video': sample_video_data
    }


@pytest.fixture
def sample_wav_file(sample_audio_data: Dict, temp_dir: Path) -> Path:
    """Create sample WAV file for testing"""
    
    import soundfile as sf
    
    wav_path = temp_dir / "test_audio.wav"
    sf.write(str(wav_path), sample_audio_data['mixture'], sample_audio_data['sample_rate'])
    
    return wav_path


@pytest.fixture
def sample_video_file(sample_mixed_data: Dict, temp_dir: Path) -> Path:
    """Create sample video file for testing"""
    
    import cv2
    
    video_path = temp_dir / "test_video.mp4"
    
    video_data = sample_mixed_data['video']
    frames = video_data['frames']
    fps = video_data['fps']
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    return video_path


@pytest.fixture
def batch_tensor_data(test_config: SeparatorConfig) -> Dict[str, torch.Tensor]:
    """Generate batch tensor data for model testing"""
    
    batch_size = 2
    
    # Audio tensor: [batch, time, features]
    audio_frames = int(test_config.audio.chunk_duration * 
                      test_config.audio.sample_rate / test_config.audio.hop_length)
    audio_tensor = torch.randn(batch_size, audio_frames, test_config.audio.n_mels)
    
    # Video tensor: [batch, time, channels, height, width]
    video_frames = int(test_config.audio.chunk_duration * test_config.video.fps)
    video_tensor = torch.randn(batch_size, video_frames, 3, *test_config.video.image_size)
    
    return {
        'audio': audio_tensor,
        'video': video_tensor,
        'batch_size': batch_size
    }


@pytest.fixture
def mock_api_client():
    """Mock API client for testing API endpoints"""
    
    from fastapi.testclient import TestClient
    from src.av_separation.api.app import app
    
    # Override separator initialization for testing
    import src.av_separation.api.app as api_app
    
    # Create test separator
    test_config = SeparatorConfig()
    test_config.model.audio_encoder_layers = 1  # Minimal for testing
    test_config.model.video_encoder_layers = 1
    test_config.model.fusion_layers = 1
    test_config.model.decoder_layers = 1
    
    api_app.separator = AVSeparator(
        num_speakers=2,
        device='cpu',
        config=test_config
    )
    
    return TestClient(app)


@pytest.fixture
def test_metrics_data() -> Dict[str, Any]:
    """Generate test data for metrics testing"""
    
    length = 1000
    
    # Generate test signals
    estimated = np.random.randn(length).astype(np.float32)
    target = np.random.randn(length).astype(np.float32)
    
    # Make estimated somewhat similar to target for realistic metrics
    estimated = 0.7 * target + 0.3 * estimated
    
    return {
        'estimated': estimated,
        'target': target,
        'sample_rate': 16000
    }


@pytest.fixture(scope="session")
def device() -> str:
    """Get test device (prefer CPU for consistent testing)"""
    
    return 'cpu'


@pytest.fixture
def config_override() -> Dict[str, Any]:
    """Configuration override for testing"""
    
    return {
        'audio': {
            'sample_rate': 8000,
            'n_mels': 40,
            'chunk_duration': 1.0
        },
        'video': {
            'fps': 5,
            'image_size': [32, 32]
        },
        'model': {
            'max_speakers': 2
        }
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    
    # Auto-mark GPU tests
    for item in items:
        if "gpu" in item.nodeid.lower() or any("gpu" in mark.name for mark in item.iter_markers()):
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))


# Helper functions for tests
def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert tensor has expected shape"""
    
    assert tensor.shape == expected_shape, (
        f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
    """Assert tensor values are in expected range"""
    
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    
    assert min_val <= actual_min, (
        f"{name} minimum value {actual_min} is below expected minimum {min_val}"
    )
    assert actual_max <= max_val, (
        f"{name} maximum value {actual_max} is above expected maximum {max_val}"
    )


def assert_audio_quality(estimated: np.ndarray, target: np.ndarray, min_si_snr: float = -10.0):
    """Assert audio separation quality meets minimum standards"""
    
    from src.av_separation.utils.metrics import compute_si_snr
    
    si_snr = compute_si_snr(estimated, target)
    
    assert si_snr >= min_si_snr, (
        f"Audio quality too low: SI-SNR {si_snr:.2f} dB is below minimum {min_si_snr} dB"
    )


def create_test_checkpoint(model: torch.nn.Module, path: Path) -> Path:
    """Create test checkpoint file"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': None,  # Add config if needed
        'epoch': 1,
        'loss': 0.5
    }
    
    torch.save(checkpoint, path)
    return path