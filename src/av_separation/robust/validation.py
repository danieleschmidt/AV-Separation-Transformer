"""
Comprehensive Input Validation and Sanitization System
Production-grade validation with security and data integrity checks.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
import logging
from pathlib import Path


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    corrected_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioValidator:
    """
    Comprehensive audio input validation and sanitization.
    """
    
    def __init__(self):
        self.min_sample_rate = 8000
        self.max_sample_rate = 48000
        self.min_duration = 0.1  # seconds
        self.max_duration = 300.0  # seconds
        self.max_amplitude = 10.0
        self.acceptable_dtypes = [torch.float32, torch.float64, torch.float16]
        
        self.logger = logging.getLogger(__name__)
    
    def validate_audio_tensor(self, audio: torch.Tensor, 
                             sample_rate: Optional[int] = None) -> ValidationResult:
        """
        Comprehensive audio tensor validation.
        """
        issues = []
        corrected_audio = audio.clone()
        
        # 1. Check tensor properties
        if not isinstance(audio, torch.Tensor):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Input must be torch.Tensor, got {type(audio)}"
            )
        
        # 2. Check dimensions
        if audio.dim() < 1 or audio.dim() > 3:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Audio must be 1D, 2D, or 3D tensor, got {audio.dim()}D"
            )
        
        # 3. Check data type
        if audio.dtype not in self.acceptable_dtypes:
            issues.append("Converting to float32")
            corrected_audio = corrected_audio.float()
        
        # 4. Check for invalid values
        nan_count = torch.isnan(corrected_audio).sum().item()
        inf_count = torch.isinf(corrected_audio).sum().item()
        
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values, replacing with zeros")
            corrected_audio = torch.where(torch.isnan(corrected_audio), 
                                        torch.zeros_like(corrected_audio), 
                                        corrected_audio)
        
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values, clipping")
            corrected_audio = torch.where(torch.isinf(corrected_audio),
                                        torch.sign(corrected_audio) * self.max_amplitude,
                                        corrected_audio)
        
        # 5. Check amplitude range
        max_amplitude = torch.abs(corrected_audio).max().item()
        if max_amplitude > self.max_amplitude:
            issues.append(f"Amplitude {max_amplitude:.2f} exceeds maximum {self.max_amplitude}, normalizing")
            corrected_audio = corrected_audio / max_amplitude * self.max_amplitude * 0.95
        
        # 6. Check duration if sample rate provided
        if sample_rate is not None:
            duration = corrected_audio.shape[-1] / sample_rate
            
            if duration < self.min_duration:
                issues.append(f"Duration {duration:.2f}s below minimum {self.min_duration}s")
                # Pad with silence
                min_samples = int(self.min_duration * sample_rate)
                padding_needed = min_samples - corrected_audio.shape[-1]
                if padding_needed > 0:
                    padding_shape = list(corrected_audio.shape)
                    padding_shape[-1] = padding_needed
                    padding = torch.zeros(*padding_shape, dtype=corrected_audio.dtype, device=corrected_audio.device)
                    corrected_audio = torch.cat([corrected_audio, padding], dim=-1)
            
            elif duration > self.max_duration:
                issues.append(f"Duration {duration:.2f}s exceeds maximum {self.max_duration}s, truncating")
                max_samples = int(self.max_duration * sample_rate)
                corrected_audio = corrected_audio[..., :max_samples]
        
        # 7. Check for DC offset
        dc_offset = torch.mean(corrected_audio, dim=-1, keepdim=True)
        if torch.abs(dc_offset).max() > 0.1:
            issues.append("Removing DC offset")
            corrected_audio = corrected_audio - dc_offset
        
        # 8. Check for clipping
        clipping_threshold = 0.99
        clipped_samples = (torch.abs(corrected_audio) > clipping_threshold).sum().item()
        total_samples = corrected_audio.numel()
        clipping_ratio = clipped_samples / total_samples
        
        if clipping_ratio > 0.01:  # More than 1% clipped
            issues.append(f"High clipping detected: {clipping_ratio*100:.1f}% of samples")
        
        # Determine severity and result
        if not issues:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Audio validation passed",
                corrected_data=corrected_audio
            )
        else:
            severity = ValidationSeverity.WARNING if len(issues) <= 2 else ValidationSeverity.ERROR
            return ValidationResult(
                is_valid=severity != ValidationSeverity.ERROR,
                severity=severity,
                message=f"Audio validation issues: {'; '.join(issues)}",
                corrected_data=corrected_audio,
                metadata={'issues': issues}
            )
    
    def validate_sample_rate(self, sample_rate: int) -> ValidationResult:
        """Validate audio sample rate."""
        if not isinstance(sample_rate, int):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate must be integer, got {type(sample_rate)}"
            )
        
        if sample_rate < self.min_sample_rate:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate} below minimum {self.min_sample_rate}"
            )
        
        if sample_rate > self.max_sample_rate:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Sample rate {sample_rate} exceeds maximum {self.max_sample_rate}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Sample rate validation passed"
        )


class VideoValidator:
    """
    Comprehensive video input validation and sanitization.
    """
    
    def __init__(self):
        self.min_width = 64
        self.max_width = 1920
        self.min_height = 64
        self.max_height = 1080
        self.min_fps = 1.0
        self.max_fps = 60.0
        self.min_frames = 1
        self.max_frames = 9000  # 5 minutes at 30fps
        self.acceptable_dtypes = [torch.float32, torch.float64, torch.uint8]
        
        self.logger = logging.getLogger(__name__)
    
    def validate_video_tensor(self, video: torch.Tensor, 
                             fps: Optional[float] = None) -> ValidationResult:
        """
        Comprehensive video tensor validation.
        """
        issues = []
        corrected_video = video.clone()
        
        # 1. Check tensor properties
        if not isinstance(video, torch.Tensor):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Input must be torch.Tensor, got {type(video)}"
            )
        
        # 2. Check dimensions (expecting BTHWC or THWC format)
        if video.dim() < 3 or video.dim() > 5:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Video must be 3D-5D tensor, got {video.dim()}D"
            )
        
        # 3. Extract dimensions
        if video.dim() == 5:  # BTHWC
            batch_size, num_frames, height, width, channels = video.shape
        elif video.dim() == 4:  # THWC or BCHW
            if video.shape[-1] <= 4:  # Assume THWC
                num_frames, height, width, channels = video.shape
            else:  # Assume BCHW (single frame)
                batch_size, channels, height, width = video.shape
                num_frames = 1
        else:  # 3D - could be HWC or CHW
            if video.shape[0] <= 4:  # Assume CHW
                channels, height, width = video.shape
                num_frames = 1
            else:  # Assume HWC
                height, width, channels = video.shape
                num_frames = 1
        
        # 4. Validate dimensions
        if height < self.min_height or height > self.max_height:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Height {height} outside valid range [{self.min_height}, {self.max_height}]"
            )
        
        if width < self.min_width or width > self.max_width:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Width {width} outside valid range [{self.min_width}, {self.max_width}]"
            )
        
        if num_frames < self.min_frames or num_frames > self.max_frames:
            issues.append(f"Frame count {num_frames} outside recommended range [{self.min_frames}, {self.max_frames}]")
        
        # 5. Check data type and range
        if video.dtype == torch.uint8:
            # Convert to float and normalize
            if torch.max(corrected_video) > 1.0:
                issues.append("Converting uint8 to normalized float32")
                corrected_video = corrected_video.float() / 255.0
        elif video.dtype in [torch.float32, torch.float64]:
            # Check if values are in [0, 1] range
            min_val = torch.min(corrected_video).item()
            max_val = torch.max(corrected_video).item()
            
            if min_val < 0 or max_val > 1:
                if min_val >= -1 and max_val <= 1:
                    # Assume [-1, 1] range, convert to [0, 1]
                    issues.append("Converting from [-1, 1] to [0, 1] range")
                    corrected_video = (corrected_video + 1.0) / 2.0
                else:
                    # Normalize to [0, 1]
                    issues.append(f"Normalizing values from [{min_val:.2f}, {max_val:.2f}] to [0, 1]")
                    corrected_video = (corrected_video - min_val) / (max_val - min_val)
        else:
            issues.append("Converting to float32")
            corrected_video = corrected_video.float()
        
        # 6. Check for invalid values
        nan_count = torch.isnan(corrected_video).sum().item()
        inf_count = torch.isinf(corrected_video).sum().item()
        
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values, replacing with zeros")
            corrected_video = torch.where(torch.isnan(corrected_video),
                                        torch.zeros_like(corrected_video),
                                        corrected_video)
        
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values, clipping")
            corrected_video = torch.clamp(corrected_video, 0, 1)
        
        # 7. Validate FPS if provided
        if fps is not None:
            if fps < self.min_fps or fps > self.max_fps:
                issues.append(f"FPS {fps} outside valid range [{self.min_fps}, {self.max_fps}]")
        
        # 8. Check for potential issues
        # Check if video is too dark/bright
        mean_brightness = torch.mean(corrected_video).item()
        if mean_brightness < 0.05:
            issues.append("Video appears very dark (mean brightness < 0.05)")
        elif mean_brightness > 0.95:
            issues.append("Video appears very bright (mean brightness > 0.95)")
        
        # Check for low contrast
        std_brightness = torch.std(corrected_video).item()
        if std_brightness < 0.05:
            issues.append("Video appears to have low contrast")
        
        # Determine severity and result
        if not issues:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Video validation passed",
                corrected_data=corrected_video
            )
        else:
            severity = ValidationSeverity.WARNING if len(issues) <= 2 else ValidationSeverity.ERROR
            return ValidationResult(
                is_valid=severity != ValidationSeverity.ERROR,
                severity=severity,
                message=f"Video validation issues: {'; '.join(issues)}",
                corrected_data=corrected_video,
                metadata={'issues': issues}
            )


class ConfigValidator:
    """
    Configuration validation and sanitization.
    """
    
    def __init__(self):
        self.required_fields = {
            'model': ['d_model', 'n_heads', 'n_layers'],
            'audio': ['sample_rate', 'hop_length', 'n_fft'],
            'video': ['fps', 'height', 'width']
        }
        
        self.field_ranges = {
            'd_model': (64, 2048),
            'n_heads': (1, 32),
            'n_layers': (1, 24),
            'sample_rate': (8000, 48000),
            'hop_length': (64, 1024),
            'n_fft': (256, 4096),
            'fps': (1, 60),
            'height': (64, 1080),
            'width': (64, 1920)
        }
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary."""
        issues = []
        corrected_config = config.copy()
        
        # Check required sections
        for section, fields in self.required_fields.items():
            if section not in config:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required configuration section: {section}"
                )
            
            # Check required fields in section
            for field in fields:
                if field not in config[section]:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing required field: {section}.{field}"
                    )
                
                # Validate field ranges
                if field in self.field_ranges:
                    value = config[section][field]
                    min_val, max_val = self.field_ranges[field]
                    
                    if not isinstance(value, (int, float)):
                        return ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Field {section}.{field} must be numeric, got {type(value)}"
                        )
                    
                    if value < min_val or value > max_val:
                        # Try to correct if possible
                        corrected_value = max(min_val, min(max_val, value))
                        issues.append(f"Field {section}.{field} value {value} corrected to {corrected_value}")
                        corrected_config[section][field] = corrected_value
        
        # Additional semantic validation
        if 'model' in corrected_config:
            # Ensure d_model is divisible by n_heads
            d_model = corrected_config['model']['d_model']
            n_heads = corrected_config['model']['n_heads']
            
            if d_model % n_heads != 0:
                new_d_model = (d_model // n_heads) * n_heads
                issues.append(f"d_model {d_model} not divisible by n_heads {n_heads}, corrected to {new_d_model}")
                corrected_config['model']['d_model'] = new_d_model
        
        if 'audio' in corrected_config:
            # Ensure hop_length <= n_fft
            hop_length = corrected_config['audio']['hop_length']
            n_fft = corrected_config['audio']['n_fft']
            
            if hop_length > n_fft:
                new_hop_length = n_fft // 2
                issues.append(f"hop_length {hop_length} > n_fft {n_fft}, corrected to {new_hop_length}")
                corrected_config['audio']['hop_length'] = new_hop_length
        
        # Determine result
        if not issues:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Configuration validation passed",
                corrected_data=corrected_config
            )
        else:
            return ValidationResult(
                is_valid=True,  # Issues were corrected
                severity=ValidationSeverity.WARNING,
                message=f"Configuration corrected: {'; '.join(issues)}",
                corrected_data=corrected_config,
                metadata={'issues': issues}
            )


class SecurityValidator:
    """
    Security validation for inputs and file paths.
    """
    
    def __init__(self):
        self.allowed_extensions = {'.wav', '.mp3', '.mp4', '.avi', '.mov', '.pth', '.pt', '.onnx'}
        self.max_file_size = 500 * 1024 * 1024  # 500MB
        self.dangerous_patterns = [
            r'\.\./',  # Directory traversal
            r'__.*__',  # Python magic methods
            r'eval\(',  # Code execution
            r'exec\(',  # Code execution
            r'import\s+',  # Module imports
            r'from\s+.*\s+import',  # Module imports
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def validate_file_path(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate file path for security issues."""
        issues = []
        
        path = Path(file_path)
        
        # Check if path exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"File does not exist: {file_path}"
            )
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"File extension {path.suffix} not allowed"
            )
        
        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"File size {file_size} exceeds maximum {self.max_file_size}"
                )
        except OSError as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Cannot access file: {e}"
            )
        
        # Check for dangerous patterns in path
        path_str = str(path)
        for pattern in self.dangerous_patterns:
            if re.search(pattern, path_str, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Dangerous pattern detected in path: {pattern}"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="File path validation passed"
        )
    
    def validate_string_input(self, input_string: str) -> ValidationResult:
        """Validate string input for dangerous content."""
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Dangerous pattern detected: {pattern}"
                )
        
        # Check for SQL injection patterns
        sql_patterns = [r'union\s+select', r'drop\s+table', r'insert\s+into', r'delete\s+from']
        for pattern in sql_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"SQL injection pattern detected: {pattern}"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="String validation passed"
        )
    
    def compute_input_hash(self, data: Union[torch.Tensor, str, bytes]) -> str:
        """Compute secure hash of input data for integrity checking."""
        if isinstance(data, torch.Tensor):
            # Convert tensor to bytes for hashing
            data_bytes = data.detach().cpu().numpy().tobytes()
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        return hashlib.sha256(data_bytes).hexdigest()


class ComprehensiveValidator:
    """
    Main validation orchestrator combining all validation components.
    """
    
    def __init__(self):
        self.audio_validator = AudioValidator()
        self.video_validator = VideoValidator()
        self.config_validator = ConfigValidator()
        self.security_validator = SecurityValidator()
        
        self.logger = logging.getLogger(__name__)
    
    def validate_inputs(self, audio: Optional[torch.Tensor] = None,
                       video: Optional[torch.Tensor] = None,
                       config: Optional[Dict[str, Any]] = None,
                       file_paths: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """
        Comprehensive validation of all inputs.
        """
        results = {}
        
        # Validate audio
        if audio is not None:
            results['audio'] = self.audio_validator.validate_audio_tensor(audio)
        
        # Validate video
        if video is not None:
            results['video'] = self.video_validator.validate_video_tensor(video)
        
        # Validate configuration
        if config is not None:
            results['config'] = self.config_validator.validate_config(config)
        
        # Validate file paths
        if file_paths is not None:
            for i, path in enumerate(file_paths):
                results[f'file_path_{i}'] = self.security_validator.validate_file_path(path)
        
        return results
    
    def get_corrected_inputs(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Extract corrected inputs from validation results."""
        corrected = {}
        
        for key, result in validation_results.items():
            if result.corrected_data is not None:
                corrected[key] = result.corrected_data
        
        return corrected
    
    def has_critical_issues(self, validation_results: Dict[str, ValidationResult]) -> bool:
        """Check if there are any critical validation issues."""
        return any(
            result.severity == ValidationSeverity.CRITICAL or 
            (result.severity == ValidationSeverity.ERROR and not result.is_valid)
            for result in validation_results.values()
        )
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> str:
        """Generate human-readable validation summary."""
        summary_lines = ["Validation Summary:"]
        
        for key, result in validation_results.items():
            status = "✓" if result.is_valid else "✗"
            summary_lines.append(f"  {status} {key}: {result.message}")
        
        return "\n".join(summary_lines)


# Global validator instance
global_validator = ComprehensiveValidator()


def validate_and_sanitize(audio: Optional[torch.Tensor] = None,
                         video: Optional[torch.Tensor] = None,
                         config: Optional[Dict[str, Any]] = None,
                         file_paths: Optional[List[str]] = None) -> Tuple[Dict[str, Any], str]:
    """
    Convenience function for complete validation and sanitization.
    
    Returns:
        Tuple of (corrected_inputs, validation_summary)
    """
    validation_results = global_validator.validate_inputs(
        audio=audio, video=video, config=config, file_paths=file_paths
    )
    
    if global_validator.has_critical_issues(validation_results):
        raise ValueError("Critical validation issues detected. Processing cannot continue.")
    
    corrected_inputs = global_validator.get_corrected_inputs(validation_results)
    summary = global_validator.get_validation_summary(validation_results)
    
    return corrected_inputs, summary