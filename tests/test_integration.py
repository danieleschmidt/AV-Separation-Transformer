"""
Integration Tests for AV-Separation-Transformer
Full end-to-end testing of the separation pipeline
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from fastapi.testclient import TestClient
from fastapi import UploadFile

# Mock torch to avoid dependency issues
import sys
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['cv2'] = MagicMock()

from av_separation.config import SeparatorConfig
from av_separation.api.app import app


class TestAPIIntegration:
    """Integration tests for the FastAPI application"""
    
    def setup_method(self):
        self.client = TestClient(app)
        
        # Create mock separator
        self.mock_separator = MagicMock()
        self.mock_separator.separate.return_value = [
            np.random.randn(16000),  # 1 second of audio at 16kHz
            np.random.randn(16000)   # Second speaker
        ]
        self.mock_separator.benchmark.return_value = {
            'mean_latency_ms': 50.0,
            'p95_latency_ms': 75.0,
            'rtf': 2.5
        }
        
        # Patch the global separator
        with patch('av_separation.api.app.separator', self.mock_separator):
            pass
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "device" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
    
    def test_root_endpoint(self):
        """Test root endpoint information"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "AV-Separation-Transformer API"
        assert "version" in data
        assert "endpoints" in data
        assert "/separate" in data["endpoints"]["separate"]
    
    @patch('av_separation.api.app.separator')
    def test_model_info_endpoint(self, mock_separator):
        """Test model information endpoint"""
        mock_separator.model.get_num_params.return_value = 124000000
        mock_separator.device = 'cuda:0'
        mock_separator.config.model.max_speakers = 4
        mock_separator.config.audio = MagicMock()
        mock_separator.config.video = MagicMock()
        mock_separator.config.model = MagicMock()
        
        # Configure the mocks to have __dict__ attributes
        mock_separator.config.audio.__dict__ = {'sample_rate': 16000, 'n_mels': 80}
        mock_separator.config.video.__dict__ = {'fps': 30, 'image_size': [224, 224]}
        mock_separator.config.model.__dict__ = {'max_speakers': 4, 'dropout': 0.1}
        
        response = self.client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_name"] == "AV-Separation-Transformer"
        assert data["parameters"] == 124000000
        assert data["device"] == "cuda:0"
        assert data["max_speakers"] == 4
    
    @patch('av_separation.api.app.separator')
    def test_benchmark_endpoint(self, mock_separator):
        """Test model benchmarking endpoint"""
        benchmark_results = {
            'mean_latency_ms': 45.2,
            'std_latency_ms': 5.3,
            'p95_latency_ms': 55.1,
            'rtf': 3.2
        }
        mock_separator.benchmark.return_value = benchmark_results
        
        response = self.client.get("/benchmark?iterations=50")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["iterations"] == 50
        assert data["benchmark_results"] == benchmark_results
        
        # Verify separator.benchmark was called with correct arguments
        mock_separator.benchmark.assert_called_once_with(num_iterations=50)
    
    def test_benchmark_no_model(self):
        """Test benchmark endpoint when model is not loaded"""
        with patch('av_separation.api.app.separator', None):
            response = self.client.get("/benchmark")
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    @patch('av_separation.api.app.separator')
    def test_model_reload_endpoint(self, mock_separator):
        """Test model reload endpoint"""
        response = self.client.post("/model/reload")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["message"] == "Model reloaded successfully"
        assert data["checkpoint"] is None
    
    def test_model_reload_with_checkpoint(self):
        """Test model reload with specific checkpoint"""
        with tempfile.NamedTemporaryFile(suffix='.pth') as temp_checkpoint:
            # Create empty checkpoint file
            temp_checkpoint.write(b'fake checkpoint data')
            temp_checkpoint.flush()
            
            response = self.client.post(
                f"/model/reload?checkpoint_path={temp_checkpoint.name}"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["checkpoint"] == temp_checkpoint.name
    
    def test_model_reload_nonexistent_checkpoint(self):
        """Test model reload with nonexistent checkpoint"""
        response = self.client.post("/model/reload?checkpoint_path=/nonexistent/file.pth")
        
        assert response.status_code == 404
        assert "Checkpoint file not found" in response.json()["detail"]


class TestSeparationPipeline:
    """Integration tests for the separation pipeline"""
    
    def setup_method(self):
        self.config = SeparatorConfig()
    
    @patch('av_separation.separator.torch')
    @patch('av_separation.separator.AudioProcessor')
    @patch('av_separation.separator.VideoProcessor')
    def test_separator_initialization(self, mock_video_proc, mock_audio_proc, mock_torch):
        """Test separator initialization"""
        mock_torch.cuda.is_available.return_value = True
        
        from av_separation.separator import AVSeparator
        
        separator = AVSeparator(
            num_speakers=2,
            config=self.config
        )
        
        assert separator.num_speakers == 2
        assert separator.config == self.config
        assert separator.device == 'cuda'
    
    @patch('av_separation.models.transformer.torch')
    def test_model_forward_pass(self, mock_torch):
        """Test model forward pass"""
        from av_separation.models.transformer import AVSeparationTransformer
        
        # Mock torch components
        mock_torch.nn.Module = MagicMock
        mock_torch.nn.Linear = MagicMock
        mock_torch.nn.LayerNorm = MagicMock
        mock_torch.nn.TransformerEncoder = MagicMock
        mock_torch.nn.TransformerDecoder = MagicMock
        
        model = AVSeparationTransformer(self.config)
        
        # Mock input tensors
        batch_size, seq_len = 2, 100
        audio_input = MagicMock()
        audio_input.shape = (batch_size, 80, seq_len)
        video_input = MagicMock()
        video_input.shape = (batch_size, seq_len, 3, 224, 224)
        
        # Test forward pass
        outputs = model.forward(audio_input, video_input)
        
        assert isinstance(outputs, dict)
        expected_keys = [
            'separated_waveforms', 'separated_spectrograms', 
            'speaker_logits', 'audio_features', 'video_features'
        ]
        for key in expected_keys:
            assert key in outputs
    
    @patch('av_separation.utils.audio.librosa')
    @patch('av_separation.utils.audio.sf')
    def test_audio_processing_pipeline(self, mock_sf, mock_librosa):
        """Test audio processing pipeline"""
        from av_separation.utils.audio import AudioProcessor
        
        # Mock librosa functions
        mock_librosa.load.return_value = (np.random.randn(16000), 16000)
        mock_librosa.stft.return_value = np.random.randn(513, 100) + 1j * np.random.randn(513, 100)
        mock_librosa.feature.melspectrogram.return_value = np.random.randn(80, 100)
        
        processor = AudioProcessor(self.config.audio)
        
        # Test audio loading
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            waveform, sr = processor.load_audio(temp_file.name)
            
            assert isinstance(waveform, np.ndarray)
            assert sr == self.config.audio.sample_rate
    
    @patch('av_separation.utils.video.cv2')
    @patch('av_separation.utils.video.mp')
    def test_video_processing_pipeline(self, mock_mp, mock_cv2):
        """Test video processing pipeline"""
        from av_separation.utils.video import VideoProcessor
        
        # Mock OpenCV functions
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 30,
            mock_cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for _ in range(10)
        ] + [(False, None)]
        
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.side_effect = lambda frame, _: frame
        mock_cv2.resize.side_effect = lambda frame, size: np.random.randint(
            0, 255, (*size[::-1], 3), dtype=np.uint8
        )
        
        processor = VideoProcessor(self.config.video)
        
        # Test video loading
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            frames = processor.load_video(temp_file.name, max_frames=10)
            
            assert isinstance(frames, np.ndarray)
            assert frames.shape[0] <= 10  # Max frames limit
            assert frames.shape[-1] == 3   # RGB channels
    
    def test_metrics_computation(self):
        """Test metrics computation"""
        from av_separation.utils.metrics import compute_si_snr, compute_sdr, compute_all_metrics
        
        # Generate test signals
        target = np.random.randn(1000)
        estimated = target + 0.1 * np.random.randn(1000)  # Add some noise
        
        # Test SI-SNR computation
        si_snr = compute_si_snr(estimated, target)
        assert isinstance(si_snr, float)
        assert si_snr > 0  # Should be positive for good separation
        
        # Test SDR computation
        sdr = compute_sdr(estimated, target)
        assert isinstance(sdr, float)
        
        # Test comprehensive metrics
        metrics = compute_all_metrics(
            estimated, target, 
            sample_rate=16000, 
            compute_perceptual=False  # Skip PESQ/STOI to avoid dependencies
        )
        
        assert 'si_snr' in metrics
        assert 'sdr' in metrics
        assert isinstance(metrics['si_snr'], float)
        assert isinstance(metrics['sdr'], float)
    
    def test_loss_functions(self):
        """Test loss function implementations"""
        from av_separation.utils.losses import SISNRLoss, PITLoss, CombinedLoss
        
        # Mock torch tensors
        with patch('av_separation.utils.losses.torch') as mock_torch:
            mock_torch.mean.return_value = MagicMock()
            mock_torch.sum.return_value = MagicMock()
            mock_torch.log10.return_value = MagicMock()
            mock_torch.stack.return_value = MagicMock()
            mock_torch.min.return_value = (MagicMock(), MagicMock())
            
            # Test SI-SNR loss
            si_snr_loss = SISNRLoss()
            
            estimated = MagicMock()
            target = MagicMock()
            estimated.shape = target.shape = (2, 16000)  # Batch size 2, 1 sec audio
            
            loss = si_snr_loss(estimated, target)
            assert loss is not None
            
            # Test PIT loss
            pit_loss = PITLoss(si_snr_loss)
            
            estimated.shape = target.shape = (2, 2, 16000)  # 2 speakers
            loss, perm = pit_loss(estimated, target)
            assert loss is not None
            assert perm is not None
            
            # Test combined loss
            combined_loss = CombinedLoss()
            losses = combined_loss(estimated, target)
            assert isinstance(losses, dict)
            assert 'total' in losses


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def create_mock_wav_file(self, duration_seconds=1):
        """Create a mock WAV file for testing"""
        # WAV header
        wav_header = b'RIFF'
        wav_header += (36 + duration_seconds * 16000 * 2).to_bytes(4, 'little')  # File size
        wav_header += b'WAVE'
        wav_header += b'fmt '
        wav_header += (16).to_bytes(4, 'little')  # Chunk size
        wav_header += (1).to_bytes(2, 'little')   # Audio format (PCM)
        wav_header += (1).to_bytes(2, 'little')   # Num channels
        wav_header += (16000).to_bytes(4, 'little')  # Sample rate
        wav_header += (32000).to_bytes(4, 'little')  # Byte rate
        wav_header += (2).to_bytes(2, 'little')   # Block align
        wav_header += (16).to_bytes(2, 'little')  # Bits per sample
        wav_header += b'data'
        wav_header += (duration_seconds * 16000 * 2).to_bytes(4, 'little')  # Data size
        
        # Add some dummy audio data
        audio_data = np.random.randint(-32768, 32767, duration_seconds * 16000, dtype=np.int16)
        wav_data = wav_header + audio_data.tobytes()
        
        return wav_data
    
    def create_mock_mp4_file(self):
        """Create a mock MP4 file for testing"""
        # Minimal MP4 header
        mp4_header = b'\x00\x00\x00\x20ftypmp41'
        mp4_header += b'\x00\x00\x00\x00mp41isom'
        mp4_header += b'\x00\x00\x02\x00moov'  # Movie header
        mp4_header += b'\x00' * 500  # Padding to make it look like a real file
        
        return mp4_header
    
    @patch('av_separation.api.app.separator')
    @patch('av_separation.api.app.input_validator')
    @patch('av_separation.api.app.audit_logger')
    def test_successful_separation_workflow(self, mock_audit, mock_validator, mock_separator):
        """Test complete successful separation workflow"""
        # Setup mocks
        mock_separator.separate.return_value = [
            np.random.randn(16000),  # Speaker 1
            np.random.randn(16000)   # Speaker 2  
        ]
        
        mock_validator.validate_file_upload.return_value = {
            'valid': True,
            'warnings': [],
            'file_info': {
                'filename': 'test.wav',
                'size': 1000,
                'extension': '.wav',
                'mime_type': 'audio/wav'
            }
        }
        
        mock_validator.validate_separation_parameters.return_value = {
            'num_speakers': 2,
            'save_video': False
        }
        
        # Create test file
        wav_data = self.create_mock_wav_file()
        
        # Make request with authentication bypass
        with patch('av_separation.api.app.check_permissions') as mock_auth:
            mock_auth.return_value = {'user_id': 'test_user', 'permissions': ['separate']}
            
            with patch('av_separation.api.app.rate_limit') as mock_rate_limit:
                mock_rate_limit.return_value = True
                
                response = self.client.post(
                    "/separate",
                    files={"video_file": ("test.wav", wav_data, "audio/wav")},
                    data={"num_speakers": 2, "save_video": False}
                )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["num_speakers"] == 2
        assert "processing_time" in data
        assert "task_id" in data
        assert isinstance(data["separated_files"], list)
    
    @patch('av_separation.api.app.input_validator')
    def test_invalid_file_upload(self, mock_validator):
        """Test handling of invalid file uploads"""
        # Setup validator to reject file
        mock_validator.validate_file_upload.side_effect = ValueError("Invalid file format")
        
        # Create invalid file data
        invalid_data = b'not a valid audio file'
        
        with patch('av_separation.api.app.check_permissions') as mock_auth:
            mock_auth.return_value = {'user_id': 'test_user', 'permissions': ['separate']}
            
            with patch('av_separation.api.app.rate_limit') as mock_rate_limit:
                mock_rate_limit.return_value = True
                
                response = self.client.post(
                    "/separate",
                    files={"video_file": ("invalid.exe", invalid_data, "application/exe")},
                    data={"num_speakers": 2}
                )
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        with patch('av_separation.api.app.rate_limit') as mock_rate_limit:
            # Simulate rate limit exceeded
            from fastapi import HTTPException
            mock_rate_limit.side_effect = HTTPException(status_code=429, detail="Rate limit exceeded")
            
            response = self.client.post("/separate")
            
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
    
    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints"""
        with patch('av_separation.api.app.check_permissions') as mock_auth:
            # Simulate authentication failure
            from fastapi import HTTPException
            mock_auth.side_effect = HTTPException(status_code=401, detail="Invalid credentials")
            
            response = self.client.post("/separate")
            
            assert response.status_code == 401
            assert "Invalid credentials" in response.json()["detail"]
    
    @patch('av_separation.api.app.separator')
    def test_model_not_loaded(self, mock_separator):
        """Test handling when model is not loaded"""
        # Set separator to None
        with patch('av_separation.api.app.separator', None):
            response = self.client.post("/separate")
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    @patch('av_separation.api.app.separator')
    @patch('av_separation.api.app.input_validator')
    @patch('av_separation.api.app.error_tracker')
    def test_processing_error_handling(self, mock_error_tracker, mock_validator, mock_separator):
        """Test handling of processing errors"""
        # Setup mocks
        mock_validator.validate_file_upload.return_value = {
            'valid': True,
            'file_info': {'filename': 'test.wav', 'size': 1000}
        }
        mock_validator.validate_separation_parameters.return_value = {'num_speakers': 2}
        
        # Make separator throw an error
        mock_separator.separate.side_effect = RuntimeError("Processing failed")
        
        wav_data = self.create_mock_wav_file()
        
        with patch('av_separation.api.app.check_permissions') as mock_auth:
            mock_auth.return_value = {'user_id': 'test_user', 'permissions': ['separate']}
            
            with patch('av_separation.api.app.rate_limit') as mock_rate_limit:
                mock_rate_limit.return_value = True
                
                response = self.client.post(
                    "/separate",
                    files={"video_file": ("test.wav", wav_data, "audio/wav")},
                    data={"num_speakers": 2}
                )
        
        assert response.status_code == 500
        
        # Verify error was tracked
        mock_error_tracker.track_error.assert_called_once()
        
        # Get the error tracking call arguments
        call_args = mock_error_tracker.track_error.call_args
        assert isinstance(call_args[1]['error'], RuntimeError)
        assert call_args[1]['context']['component'] == 'separation_api'


@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance and load testing"""
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        from av_separation.monitoring import PerformanceMonitor
        
        monitor = PerformanceMonitor(
            enable_prometheus=False,  # Disable to avoid dependencies
            enable_opentelemetry=False
        )
        
        # Test health status
        health = monitor.get_health_status()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'system' in health
        assert 'service' in health
        
        # Check system stats
        system_stats = health['system']
        assert 'cpu_percent' in system_stats
        assert 'memory_percent' in system_stats
        assert 'memory_used' in system_stats
    
    @patch('av_separation.monitoring.torch')
    def test_gpu_monitoring(self, mock_torch):
        """Test GPU monitoring functionality"""
        from av_separation.monitoring import PerformanceMonitor
        
        # Mock CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.memory_allocated.side_effect = [1024**3, 2*1024**3]  # 1GB, 2GB
        mock_torch.cuda.memory_reserved.side_effect = [1.5*1024**3, 2.5*1024**3]
        
        monitor = PerformanceMonitor(
            enable_prometheus=False,
            enable_opentelemetry=False
        )
        
        # Allow some time for background monitoring to run
        import time
        time.sleep(0.1)
        
        health = monitor.get_health_status()
        gpu_stats = health['system'].get('gpu_memory_used', {})
        
        # GPU stats might be populated by background thread
        # Just verify the structure is correct
        assert isinstance(gpu_stats, dict)
    
    def test_concurrent_requests_simulation(self):
        """Test handling of concurrent requests"""
        from av_separation.security import RateLimiter
        
        limiter = RateLimiter(max_requests=100, time_window=60)
        
        # Simulate concurrent requests from multiple clients
        import threading
        import time
        
        results = []
        
        def make_requests(client_id, num_requests):
            for _ in range(num_requests):
                result = limiter.is_allowed(client_id)
                results.append((client_id, result))
                time.sleep(0.001)  # Small delay
        
        # Create threads for concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=make_requests,
                args=(f'client_{i}', 20)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze results
        client_results = {}
        for client_id, allowed in results:
            if client_id not in client_results:
                client_results[client_id] = {'allowed': 0, 'denied': 0}
            
            if allowed:
                client_results[client_id]['allowed'] += 1
            else:
                client_results[client_id]['denied'] += 1
        
        # Each client should have some requests allowed
        for client_id, stats in client_results.items():
            assert stats['allowed'] > 0
            assert stats['allowed'] <= 20  # Not more than requested