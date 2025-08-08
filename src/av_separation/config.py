from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import torch
    _torch_available = True
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _torch_available = False
    _default_device = "cpu"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    window_size_ms: int = 25
    hop_size_ms: int = 10
    chunk_duration: float = 4.0


@dataclass
class VideoConfig:
    fps: int = 30
    image_size: Tuple[int, int] = (224, 224)
    face_size: Tuple[int, int] = (96, 96)
    lip_size: Tuple[int, int] = (64, 64)
    detection_confidence: float = 0.8
    tracking_confidence: float = 0.5
    max_faces: int = 6
    temporal_window: int = 5


@dataclass
class ModelConfig:
    audio_encoder_layers: int = 8
    audio_encoder_heads: int = 8
    audio_encoder_dim: int = 512
    audio_encoder_ffn_dim: int = 2048
    
    video_encoder_layers: int = 6
    video_encoder_heads: int = 8
    video_encoder_dim: int = 256
    video_encoder_ffn_dim: int = 1024
    
    fusion_layers: int = 6
    fusion_heads: int = 8
    fusion_dim: int = 512
    
    decoder_layers: int = 8
    decoder_heads: int = 8
    decoder_dim: int = 512
    decoder_ffn_dim: int = 2048
    
    max_speakers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False


@dataclass
class InferenceConfig:
    device: str = _default_device
    batch_size: int = 1
    chunk_size_ms: int = 20
    lookahead_ms: int = 40
    streaming: bool = False
    use_fp16: bool = False
    use_onnx: bool = False
    num_threads: int = 4
    max_latency_ms: int = 50


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    loss_type: str = "si_snr"
    pit_loss: bool = True
    
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    mixed_precision: bool = True
    distributed: bool = False
    num_workers: int = 8
    
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every_n_epochs: int = 5
    val_every_n_epochs: int = 1


@dataclass
class SeparatorConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self):
        return {
            "audio": self.audio.__dict__,
            "video": self.video.__dict__,
            "model": self.model.__dict__,
            "inference": self.inference.__dict__,
            "training": self.training.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        if "audio" in config_dict:
            config.audio = AudioConfig(**config_dict["audio"])
        if "video" in config_dict:
            config.video = VideoConfig(**config_dict["video"])
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "inference" in config_dict:
            config.inference = InferenceConfig(**config_dict["inference"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        return config