"""AV-Separation-Transformer: Production-ready audio-visual speech separation.

This package provides state-of-the-art audio-visual speech separation using
transformer architectures for real-time video conferencing applications.

Example:
    >>> from av_separation import AudioVisualSeparator
    >>> separator = AudioVisualSeparator(
    ...     num_speakers=2,
    ...     model_path='weights/av_sepnet_base.pth',
    ...     device='cuda'
    ... )
    >>> separated_audio = separator.separate(audio_tensor, video_tensor)
"""

from .models import AudioVisualSeparator, TransformerModel
from .services import SeparationService, WebRTCService
from .utils import AudioProcessor, VideoProcessor, ModelUtils

__version__ = "1.0.0-dev"
__author__ = "Daniel Schmidt"
__email__ = "daniel.schmidt@terragonlabs.com"

__all__ = [
    "AudioVisualSeparator",
    "TransformerModel", 
    "SeparationService",
    "WebRTCService",
    "AudioProcessor",
    "VideoProcessor",
    "ModelUtils",
]