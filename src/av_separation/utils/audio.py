import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
from pathlib import Path
import warnings


class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_mels = config.n_mels
        
    def load_audio(
        self, 
        file_path: Union[str, Path],
        target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        
        file_path = Path(file_path)
        target_sr = target_sr or self.sample_rate
        
        try:
            waveform, sr = librosa.load(str(file_path), sr=None, mono=True)
        except Exception as e:
            try:
                waveform, sr = sf.read(str(file_path))
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio: {e}, {e2}")
        
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        waveform = self.normalize_audio(waveform)
        
        return waveform, sr
    
    def save_audio(
        self,
        waveform: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: Optional[int] = None
    ):
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = sample_rate or self.sample_rate
        
        waveform = np.clip(waveform, -1.0, 1.0)
        
        sf.write(str(file_path), waveform, sample_rate)
    
    def compute_spectrogram(
        self,
        waveform: np.ndarray,
        use_mel: bool = True
    ) -> np.ndarray:
        
        stft = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True
        )
        
        magnitude = np.abs(stft)
        
        if use_mel:
            mel_spec = librosa.feature.melspectrogram(
                S=magnitude**2,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                fmin=self.config.f_min,
                fmax=self.config.f_max
            )
            spec = librosa.power_to_db(mel_spec, ref=np.max)
        else:
            spec = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        return spec
    
    def inverse_spectrogram(
        self,
        spectrogram: np.ndarray,
        use_griffin_lim: bool = True,
        n_iter: int = 32
    ) -> np.ndarray:
        
        if spectrogram.shape[0] == self.n_mels:
            db_to_power = librosa.db_to_power(spectrogram, ref=1.0)
            magnitude = librosa.feature.inverse.mel_to_stft(
                db_to_power,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                fmin=self.config.f_min,
                fmax=self.config.f_max
            )
            magnitude = np.sqrt(magnitude)
        else:
            magnitude = librosa.db_to_amplitude(spectrogram, ref=1.0)
        
        if use_griffin_lim:
            waveform = librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True
            )
        else:
            angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
            complex_spec = magnitude * angles
            waveform = librosa.istft(
                complex_spec,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True
            )
        
        return self.normalize_audio(waveform)
    
    def normalize_audio(
        self,
        waveform: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        
        rms = np.sqrt(np.mean(waveform**2))
        if rms > 0:
            target_rms = 10**(target_db / 20)
            waveform = waveform * (target_rms / rms)
        
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val * 0.95
        
        return waveform
    
    def apply_voice_activity_detection(
        self,
        waveform: np.ndarray,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2
    ) -> np.ndarray:
        
        try:
            import webrtcvad
            vad = webrtcvad.Vad(aggressiveness)
            
            if self.sample_rate not in [8000, 16000, 32000, 48000]:
                target_sr = 16000
                waveform_vad = librosa.resample(
                    waveform, 
                    orig_sr=self.sample_rate,
                    target_sr=target_sr
                )
            else:
                waveform_vad = waveform
                target_sr = self.sample_rate
            
            waveform_int16 = (waveform_vad * 32767).astype(np.int16)
            
            frame_len = int(target_sr * frame_duration_ms / 1000)
            frames = [
                waveform_int16[i:i+frame_len]
                for i in range(0, len(waveform_int16)-frame_len+1, frame_len)
            ]
            
            voice_flags = []
            for frame in frames:
                if len(frame) == frame_len:
                    is_speech = vad.is_speech(frame.tobytes(), target_sr)
                    voice_flags.append(is_speech)
            
            voice_flags = np.repeat(voice_flags, frame_len)[:len(waveform)]
            
            waveform[~voice_flags] *= 0.1
            
        except ImportError:
            warnings.warn("WebRTC VAD not available, skipping VAD")
        
        return waveform
    
    def extract_features(
        self,
        waveform: np.ndarray
    ) -> dict:
        
        features = {}
        
        features['mfcc'] = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            waveform,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        
        features['rms'] = librosa.feature.rms(
            y=waveform,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        
        return features