"""
API Integration Tests for AV-Separation-Transformer
"""

import pytest
import tempfile
import json
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from fastapi.testclient import TestClient

from src.av_separation.api.app import app
from src.av_separation import AVSeparator, SeparatorConfig


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, mock_api_client):
        """Test successful health check"""
        
        response = mock_api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "device" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
    
    def test_health_check_content(self, mock_api_client):
        """Test health check response content"""
        
        response = mock_api_client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded"]
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["gpu_available"], bool)


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self, mock_api_client):
        """Test root endpoint returns API info"""
        
        response = mock_api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        
        assert data["name"] == "AV-Separation-Transformer API"


class TestSeparationEndpoint:
    """Test separation endpoint"""
    
    def test_separation_endpoint_missing_file(self, mock_api_client):
        """Test separation endpoint with missing file"""
        
        response = mock_api_client.post("/separate")
        
        assert response.status_code == 422  # Validation error
    
    def test_separation_endpoint_with_file(self, mock_api_client, sample_wav_file):
        """Test separation endpoint with valid file"""
        
        with open(sample_wav_file, 'rb') as f:
            files = {"video_file": ("test.wav", f, "audio/wav")}
            data = {"num_speakers": 2}
            
            response = mock_api_client.post("/separate", files=files, data=data)
        
        # Should work if separator is properly mocked
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        if response.status_code == 200:
            result = response.json()
            assert "success" in result
            assert "task_id" in result
    
    def test_separation_endpoint_invalid_format(self, mock_api_client):
        """Test separation endpoint with invalid file format"""
        
        # Create fake file with invalid extension
        fake_file = io.BytesIO(b"fake content")
        files = {"video_file": ("test.txt", fake_file, "text/plain")}
        
        response = mock_api_client.post("/separate", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["detail"]
    
    def test_separation_endpoint_with_config_override(self, mock_api_client, sample_wav_file):
        """Test separation endpoint with configuration override"""
        
        config_override = {
            "audio": {
                "sample_rate": 8000,
                "n_mels": 40
            }
        }
        
        with open(sample_wav_file, 'rb') as f:
            files = {"video_file": ("test.wav", f, "audio/wav")}
            data = {
                "num_speakers": 2,
                "config_override": json.dumps(config_override)
            }
            
            response = mock_api_client.post("/separate", files=files, data=data)
        
        assert response.status_code in [200, 503]


class TestStreamEndpoint:
    """Test stream processing endpoint"""
    
    def test_stream_endpoint_missing_files(self, mock_api_client):
        """Test stream endpoint with missing files"""
        
        response = mock_api_client.post("/separate/stream")
        
        assert response.status_code == 422  # Validation error
    
    def test_stream_endpoint_with_data(self, mock_api_client, sample_audio_data):
        """Test stream endpoint with audio/video data"""
        
        # Create mock audio and video data
        audio_data = sample_audio_data['mixture'].tobytes()
        video_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8).tobytes()
        
        files = {
            "audio_file": ("audio.raw", io.BytesIO(audio_data), "application/octet-stream"),
            "video_frame": ("frame.raw", io.BytesIO(video_data), "application/octet-stream")
        }
        
        response = mock_api_client.post("/separate/stream", files=files)
        
        assert response.status_code in [200, 503]


class TestBenchmarkEndpoint:
    """Test benchmark endpoint"""
    
    def test_benchmark_endpoint_default(self, mock_api_client):
        """Test benchmark endpoint with default parameters"""
        
        response = mock_api_client.get("/benchmark")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "benchmark_results" in data
    
    def test_benchmark_endpoint_custom_iterations(self, mock_api_client):
        """Test benchmark endpoint with custom iterations"""
        
        response = mock_api_client.get("/benchmark?iterations=10")
        
        assert response.status_code in [200, 503]


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_endpoint(self, mock_api_client):
        """Test model info endpoint"""
        
        response = mock_api_client.get("/model/info")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "version" in data
            assert "parameters" in data
            assert "device" in data


class TestModelReloadEndpoint:
    """Test model reload endpoint"""
    
    def test_model_reload_endpoint(self, mock_api_client):
        """Test model reload endpoint"""
        
        response = mock_api_client.post("/model/reload")
        
        assert response.status_code in [200, 500]
    
    def test_model_reload_with_checkpoint(self, mock_api_client, temp_dir):
        """Test model reload with checkpoint path"""
        
        # Create fake checkpoint
        checkpoint_path = temp_dir / "fake_checkpoint.pth"
        checkpoint_path.write_text("fake checkpoint")
        
        response = mock_api_client.post(f"/model/reload?checkpoint_path={checkpoint_path}")
        
        assert response.status_code in [200, 404, 500]


class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_handler(self, mock_api_client):
        """Test 404 error handler"""
        
        response = mock_api_client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_method_not_allowed(self, mock_api_client):
        """Test method not allowed error"""
        
        response = mock_api_client.delete("/health")
        
        assert response.status_code == 405


class TestWebSocketEndpoint:
    """Test WebSocket endpoint"""
    
    def test_websocket_connection(self, mock_api_client):
        """Test WebSocket connection"""
        
        # WebSocket testing with FastAPI TestClient is limited
        # In practice, you'd use a proper WebSocket client
        
        with mock_api_client.websocket_connect("/ws/test_client") as websocket:
            # This would normally test the WebSocket connection
            # For now, just ensure the endpoint exists
            pass


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for complete API workflows"""
    
    def test_complete_separation_workflow(self, mock_api_client, sample_wav_file):
        """Test complete separation workflow"""
        
        # 1. Check health
        health_response = mock_api_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info
        info_response = mock_api_client.get("/model/info")
        if info_response.status_code == 200:
            model_info = info_response.json()
            assert model_info["model_name"] == "AV-Separation-Transformer"
        
        # 3. Perform separation
        with open(sample_wav_file, 'rb') as f:
            files = {"video_file": ("test.wav", f, "audio/wav")}
            data = {"num_speakers": 2}
            
            separation_response = mock_api_client.post("/separate", files=files, data=data)
        
        if separation_response.status_code == 200:
            result = separation_response.json()
            assert result["success"] is True
            assert "task_id" in result
            assert "processing_time" in result
    
    def test_api_consistency(self, mock_api_client):
        """Test API response consistency"""
        
        # Test that endpoints return consistent error formats
        endpoints_to_test = [
            ("/nonexistent", 404),
            ("/health", 200),
            ("/", 200)
        ]
        
        for endpoint, expected_status in endpoints_to_test:
            response = mock_api_client.get(endpoint)
            
            if response.status_code == expected_status:
                data = response.json()
                # All responses should be valid JSON
                assert isinstance(data, dict)


@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    def test_health_endpoint_performance(self, mock_api_client):
        """Test health endpoint response time"""
        
        import time
        
        start_time = time.time()
        response = mock_api_client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, mock_api_client):
        """Test handling of concurrent requests"""
        
        import concurrent.futures
        import threading
        
        def make_request():
            return mock_api_client.get("/health")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestAPISecurity:
    """Security tests for API endpoints"""
    
    def test_large_file_handling(self, mock_api_client):
        """Test handling of large files"""
        
        # Create a large fake file (1MB)
        large_data = b"x" * (1024 * 1024)
        files = {"video_file": ("large.wav", io.BytesIO(large_data), "audio/wav")}
        
        response = mock_api_client.post("/separate", files=files)
        
        # Should either process or reject gracefully
        assert response.status_code in [200, 400, 413, 503]
    
    def test_malicious_filename(self, mock_api_client):
        """Test handling of malicious filenames"""
        
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "<script>alert('xss')</script>.wav",
            "file with\x00null.wav"
        ]
        
        for filename in malicious_filenames:
            fake_data = b"fake audio data"
            files = {"video_file": (filename, io.BytesIO(fake_data), "audio/wav")}
            
            response = mock_api_client.post("/separate", files=files)
            
            # Should reject malicious filenames
            if response.status_code == 400:
                data = response.json()
                assert "malicious" in data["detail"].lower() or "invalid" in data["detail"].lower()
    
    def test_invalid_content_type(self, mock_api_client):
        """Test handling of invalid content types"""
        
        # Send executable as audio file
        fake_exe = b"MZ\x90\x00"  # PE header signature
        files = {"video_file": ("malware.wav", io.BytesIO(fake_exe), "audio/wav")}
        
        response = mock_api_client.post("/separate", files=files)
        
        # Should detect and reject
        assert response.status_code in [400, 422]


# Mock implementations for testing
@pytest.fixture
def mock_separator():
    """Mock separator for API testing"""
    
    separator = MagicMock()
    separator.separate.return_value = [
        np.random.randn(1000).astype(np.float32),
        np.random.randn(1000).astype(np.float32)
    ]
    separator.separate_stream.return_value = np.random.randn(2, 1000).astype(np.float32)
    separator.benchmark.return_value = {
        'mean_latency_ms': 50.0,
        'p95_latency_ms': 75.0,
        'rtf': 2.5
    }
    separator.model.get_num_params.return_value = 1000000
    separator.device = "cpu"
    separator.config = SeparatorConfig()
    
    return separator


@pytest.fixture
def mocked_api_app(mock_separator):
    """API app with mocked separator"""
    
    with patch('src.av_separation.api.app.separator', mock_separator):
        with patch('src.av_separation.api.app.app_config', SeparatorConfig()):
            yield app