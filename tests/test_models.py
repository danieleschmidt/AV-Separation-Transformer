"""
Unit tests for AV-Separation-Transformer models
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.av_separation.models import (
    AVSeparationTransformer,
    AudioEncoder,
    VideoEncoder,
    CrossModalFusion,
    SeparationDecoder
)
from src.av_separation.config import SeparatorConfig
from tests.conftest import assert_tensor_shape, assert_tensor_range


class TestAudioEncoder:
    """Test AudioEncoder module"""
    
    def test_audio_encoder_init(self, test_config):
        """Test AudioEncoder initialization"""
        
        encoder = AudioEncoder(test_config)
        
        assert encoder.config == test_config
        assert hasattr(encoder, 'frontend')
        assert hasattr(encoder, 'positional_encoding')
        assert hasattr(encoder, 'transformer')
        assert hasattr(encoder, 'layer_norm')
    
    def test_audio_encoder_forward(self, test_config, batch_tensor_data):
        """Test AudioEncoder forward pass"""
        
        encoder = AudioEncoder(test_config)
        audio_input = batch_tensor_data['audio']
        
        with torch.no_grad():
            output = encoder(audio_input)
        
        batch_size = batch_tensor_data['batch_size']
        expected_time_dim = audio_input.shape[1] // 4  # Due to downsampling in frontend
        expected_feature_dim = test_config.model.audio_encoder_dim
        
        assert_tensor_shape(
            output, 
            (batch_size, expected_time_dim, expected_feature_dim),
            "audio encoder output"
        )
    
    def test_audio_encoder_with_mask(self, test_config, batch_tensor_data):
        """Test AudioEncoder with attention mask"""
        
        encoder = AudioEncoder(test_config)
        audio_input = batch_tensor_data['audio']
        batch_size = batch_tensor_data['batch_size']
        
        # Create attention mask
        seq_len = audio_input.shape[1] // 4
        mask = torch.ones(batch_size, seq_len).bool()
        
        with torch.no_grad():
            output = encoder(audio_input, mask)
        
        assert output.shape[0] == batch_size
        assert output.shape[2] == test_config.model.audio_encoder_dim
    
    def test_audio_encoder_compute_output_shape(self, test_config):
        """Test output shape computation"""
        
        encoder = AudioEncoder(test_config)
        
        input_shape = (2, 40, 100)  # batch, n_mels, time
        output_shape = encoder.compute_output_shape(input_shape)
        
        assert len(output_shape) == 3
        assert output_shape[0] == input_shape[0]  # batch size unchanged
        assert output_shape[2] == test_config.model.audio_encoder_dim


class TestVideoEncoder:
    """Test VideoEncoder module"""
    
    def test_video_encoder_init(self, test_config):
        """Test VideoEncoder initialization"""
        
        encoder = VideoEncoder(test_config)
        
        assert encoder.config == test_config
        assert hasattr(encoder, 'face_detector')
        assert hasattr(encoder, 'lip_encoder')
        assert hasattr(encoder, 'transformer')
        assert hasattr(encoder, 'speaker_embedding')
    
    def test_video_encoder_forward(self, test_config, batch_tensor_data):
        """Test VideoEncoder forward pass"""
        
        encoder = VideoEncoder(test_config)
        video_input = batch_tensor_data['video']
        
        with torch.no_grad():
            output, faces = encoder(video_input)
        
        batch_size = batch_tensor_data['batch_size']
        time_dim = video_input.shape[1]
        feature_dim = test_config.model.video_encoder_dim
        
        assert_tensor_shape(
            output,
            (batch_size, time_dim, feature_dim),
            "video encoder output"
        )
        
        assert isinstance(faces, list)
        assert len(faces) == batch_size
    
    def test_video_encoder_with_speaker_ids(self, test_config, batch_tensor_data):
        """Test VideoEncoder with speaker IDs"""
        
        encoder = VideoEncoder(test_config)
        video_input = batch_tensor_data['video']
        batch_size = batch_tensor_data['batch_size']
        
        # Create speaker IDs
        speaker_ids = torch.randint(0, test_config.model.max_speakers, (batch_size,))
        
        with torch.no_grad():
            output, faces = encoder(video_input, speaker_ids)
        
        assert output.shape[0] == batch_size
        assert len(faces) == batch_size
    
    def test_face_detection(self, test_config, sample_video_data):
        """Test face detection functionality"""
        
        encoder = VideoEncoder(test_config)
        
        # Take first frame
        frame = sample_video_data['frames'][0]
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        
        with torch.no_grad():
            faces = encoder.face_detector(frame_tensor.unsqueeze(0))
        
        assert isinstance(faces, list)
        assert len(faces) == 1  # batch size 1
    
    def test_lip_region_extraction(self, test_config, sample_video_data):
        """Test lip region extraction"""
        
        encoder = VideoEncoder(test_config)
        frames = sample_video_data['frames']
        
        # Mock face detection results
        mock_faces = [torch.tensor([[20, 15, 24, 24]]).float()]  # [x, y, w, h]
        
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        
        with torch.no_grad():
            lip_regions = encoder.extract_lip_regions(frames_tensor.unsqueeze(0), mock_faces)
        
        assert lip_regions.shape[0] == 1  # batch size
        assert lip_regions.shape[1] == len(frames)  # time frames


class TestCrossModalFusion:
    """Test CrossModalFusion module"""
    
    def test_fusion_init(self, test_config):
        """Test CrossModalFusion initialization"""
        
        fusion = CrossModalFusion(test_config)
        
        assert hasattr(fusion, 'audio_projection')
        assert hasattr(fusion, 'video_projection')
        assert hasattr(fusion, 'dtw')
        assert hasattr(fusion, 'fusion_layers')
        assert hasattr(fusion, 'output_projection')
    
    def test_fusion_forward(self, test_config):
        """Test CrossModalFusion forward pass"""
        
        fusion = CrossModalFusion(test_config)
        
        batch_size = 2
        audio_seq_len = 25
        video_seq_len = 20
        
        # Create input features
        audio_features = torch.randn(batch_size, audio_seq_len, test_config.model.audio_encoder_dim)
        video_features = torch.randn(batch_size, video_seq_len, test_config.model.video_encoder_dim)
        
        with torch.no_grad():
            fused_features, alignment_score = fusion(audio_features, video_features)
        
        expected_seq_len = audio_seq_len  # Fusion uses audio sequence length
        
        assert_tensor_shape(
            fused_features,
            (batch_size, expected_seq_len, test_config.model.decoder_dim),
            "fused features"
        )
        
        assert_tensor_shape(
            alignment_score,
            (batch_size,),
            "alignment score"
        )
    
    def test_fusion_with_masks(self, test_config):
        """Test CrossModalFusion with attention masks"""
        
        fusion = CrossModalFusion(test_config)
        
        batch_size = 2
        audio_seq_len = 25
        video_seq_len = 20
        
        audio_features = torch.randn(batch_size, audio_seq_len, test_config.model.audio_encoder_dim)
        video_features = torch.randn(batch_size, video_seq_len, test_config.model.video_encoder_dim)
        
        # Create masks
        audio_mask = torch.ones(batch_size, audio_seq_len).bool()
        video_mask = torch.ones(batch_size, video_seq_len).bool()
        
        with torch.no_grad():
            fused_features, alignment_score = fusion(
                audio_features, video_features, audio_mask, video_mask
            )
        
        assert fused_features.shape[0] == batch_size
        assert alignment_score.shape[0] == batch_size
    
    def test_dynamic_time_warping(self, test_config):
        """Test Dynamic Time Warping component"""
        
        fusion = CrossModalFusion(test_config)
        dtw = fusion.dtw
        
        batch_size = 2
        audio_len = 25
        video_len = 20
        feature_dim = test_config.model.fusion_dim
        
        audio_features = torch.randn(batch_size, audio_len, feature_dim)
        video_features = torch.randn(batch_size, video_len, feature_dim)
        
        with torch.no_grad():
            aligned_video, alignment_score = dtw(audio_features, video_features)
        
        # Aligned video should match audio sequence length
        assert_tensor_shape(
            aligned_video,
            (batch_size, audio_len, feature_dim),
            "aligned video"
        )
        
        assert_tensor_shape(
            alignment_score,
            (batch_size,),
            "DTW alignment score"
        )


class TestSeparationDecoder:
    """Test SeparationDecoder module"""
    
    def test_decoder_init(self, test_config):
        """Test SeparationDecoder initialization"""
        
        decoder = SeparationDecoder(test_config)
        
        assert hasattr(decoder, 'speaker_query')
        assert hasattr(decoder, 'transformer_decoder')
        assert hasattr(decoder, 'multi_scale_spectrogram')
        assert hasattr(decoder, 'magnitude_projection')
        assert hasattr(decoder, 'griffin_lim')
        assert hasattr(decoder, 'speaker_assignment')
    
    def test_decoder_forward(self, test_config):
        """Test SeparationDecoder forward pass"""
        
        decoder = SeparationDecoder(test_config)
        
        batch_size = 2
        seq_len = 25
        
        fused_features = torch.randn(batch_size, seq_len, test_config.model.decoder_dim)
        
        with torch.no_grad():
            outputs = decoder(fused_features)
        
        num_speakers = test_config.model.max_speakers
        
        # Check output shapes
        assert 'waveforms' in outputs
        assert 'spectrograms' in outputs
        assert 'speaker_logits' in outputs
        
        waveforms = outputs['waveforms']
        spectrograms = outputs['spectrograms']
        speaker_logits = outputs['speaker_logits']
        
        # Waveforms shape: [batch, speakers, time]
        assert waveforms.shape[0] == batch_size
        assert waveforms.shape[1] == num_speakers
        assert waveforms.shape[2] > 0  # Has time dimension
        
        # Spectrograms shape: [batch, speakers, freq, time]
        assert spectrograms.shape[0] == batch_size
        assert spectrograms.shape[1] == num_speakers
        
        # Speaker logits shape: [batch, speakers, num_speakers]
        assert_tensor_shape(
            speaker_logits,
            (batch_size, num_speakers, num_speakers),
            "speaker logits"
        )
    
    def test_decoder_with_mixture_spectrogram(self, test_config):
        """Test SeparationDecoder with mixture spectrogram"""
        
        decoder = SeparationDecoder(test_config)
        
        batch_size = 2
        seq_len = 25
        
        fused_features = torch.randn(batch_size, seq_len, test_config.model.decoder_dim)
        mixture_spec = torch.randn(batch_size, test_config.audio.n_fft // 2 + 1, seq_len)
        
        with torch.no_grad():
            outputs = decoder(fused_features, mixture_spec)
        
        assert 'waveforms' in outputs
        assert outputs['waveforms'].shape[0] == batch_size
    
    def test_speaker_query_generation(self, test_config):
        """Test speaker query generation"""
        
        decoder = SeparationDecoder(test_config)
        speaker_query = decoder.speaker_query
        
        batch_size = 2
        seq_len = 25
        device = 'cpu'
        
        with torch.no_grad():
            queries = speaker_query(batch_size, seq_len, device)
        
        expected_query_len = test_config.model.max_speakers * seq_len
        
        assert_tensor_shape(
            queries,
            (batch_size, expected_query_len, test_config.model.decoder_dim),
            "speaker queries"
        )
    
    def test_griffin_lim_reconstruction(self, test_config):
        """Test Griffin-Lim phase reconstruction"""
        
        decoder = SeparationDecoder(test_config)
        griffin_lim = decoder.griffin_lim
        
        batch_size = 2
        freq_bins = test_config.audio.n_fft // 2 + 1
        time_frames = 50
        
        # Create magnitude spectrogram
        magnitude = torch.rand(batch_size, time_frames, freq_bins)
        
        with torch.no_grad():
            waveform = griffin_lim(magnitude)
        
        assert waveform.shape[0] == batch_size
        assert waveform.shape[1] > 0  # Has time dimension
        
        # Check waveform is real-valued
        assert torch.all(torch.isreal(waveform))


class TestAVSeparationTransformer:
    """Test complete AVSeparationTransformer model"""
    
    def test_model_init(self, test_config):
        """Test model initialization"""
        
        model = AVSeparationTransformer(test_config)
        
        assert hasattr(model, 'audio_encoder')
        assert hasattr(model, 'video_encoder')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'decoder')
        assert model.config == test_config
    
    def test_model_forward(self, test_model, batch_tensor_data):
        """Test complete model forward pass"""
        
        audio_input = batch_tensor_data['audio']
        video_input = batch_tensor_data['video']
        
        with torch.no_grad():
            outputs = test_model(audio_input, video_input)
        
        # Check all expected outputs are present
        expected_keys = [
            'separated_waveforms',
            'separated_spectrograms',
            'speaker_logits',
            'audio_features',
            'video_features',
            'fused_features',
            'alignment_score',
            'detected_faces'
        ]
        
        for key in expected_keys:
            assert key in outputs, f"Missing output key: {key}"
        
        batch_size = batch_tensor_data['batch_size']
        
        # Check output shapes
        assert outputs['separated_waveforms'].shape[0] == batch_size
        assert outputs['separated_spectrograms'].shape[0] == batch_size
        assert outputs['speaker_logits'].shape[0] == batch_size
        assert outputs['alignment_score'].shape[0] == batch_size
    
    def test_model_with_masks(self, test_model, batch_tensor_data):
        """Test model with attention masks"""
        
        audio_input = batch_tensor_data['audio']
        video_input = batch_tensor_data['video']
        batch_size = batch_tensor_data['batch_size']
        
        # Create masks
        audio_mask = torch.ones(batch_size, audio_input.shape[1] // 4).bool()
        video_mask = torch.ones(batch_size, video_input.shape[1]).bool()
        
        with torch.no_grad():
            outputs = test_model(audio_input, video_input, audio_mask, video_mask)
        
        assert outputs['separated_waveforms'].shape[0] == batch_size
    
    def test_model_with_speaker_ids(self, test_model, batch_tensor_data, test_config):
        """Test model with speaker IDs"""
        
        audio_input = batch_tensor_data['audio']
        video_input = batch_tensor_data['video']
        batch_size = batch_tensor_data['batch_size']
        
        speaker_ids = torch.randint(0, test_config.model.max_speakers, (batch_size,))
        
        with torch.no_grad():
            outputs = test_model(audio_input, video_input, speaker_ids=speaker_ids)
        
        assert outputs['separated_waveforms'].shape[0] == batch_size
    
    def test_model_separate_method(self, test_model, batch_tensor_data):
        """Test model separate method"""
        
        audio_waveform = torch.randn(8000)  # 1 second at 8kHz
        video_frames = batch_tensor_data['video'][0]  # Take first batch item
        
        with torch.no_grad():
            separated = test_model.separate(audio_waveform, video_frames)
        
        assert separated.shape[0] <= test_model.config.model.max_speakers
        assert separated.shape[1] > 0  # Has time dimension
    
    def test_model_spectrogram_computation(self, test_model):
        """Test spectrogram computation"""
        
        waveform = torch.randn(8000)  # 1 second at 8kHz
        
        with torch.no_grad():
            spectrogram = test_model._compute_spectrogram(waveform)
        
        assert spectrogram.shape[0] == test_model.config.audio.n_mels
        assert spectrogram.shape[1] > 0  # Has time frames
    
    def test_model_parameter_count(self, test_model):
        """Test parameter counting"""
        
        num_params = test_model.get_num_params()
        
        assert isinstance(num_params, int)
        assert num_params > 0
        
        # Verify by manual count
        manual_count = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
        assert num_params == manual_count
    
    def test_model_gradient_checkpointing(self, test_model):
        """Test gradient checkpointing"""
        
        test_model.enable_gradient_checkpointing()
        
        assert test_model.config.model.gradient_checkpointing is True
    
    def test_model_save_load_pretrained(self, test_model, temp_dir):
        """Test save and load pretrained functionality"""
        
        # Save model
        save_path = temp_dir / "test_model.pth"
        test_model.save_pretrained(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = AVSeparationTransformer.from_pretrained(
            str(save_path), 
            config=test_model.config
        )
        
        assert loaded_model.config.to_dict() == test_model.config.to_dict()
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            test_model.named_parameters(), 
            loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


@pytest.mark.slow
class TestModelTraining:
    """Test model training capabilities"""
    
    def test_model_backward_pass(self, test_model, batch_tensor_data):
        """Test model can compute gradients"""
        
        test_model.train()
        
        audio_input = batch_tensor_data['audio']
        video_input = batch_tensor_data['video']
        
        # Forward pass
        outputs = test_model(audio_input, video_input)
        
        # Create dummy loss
        loss = outputs['separated_waveforms'].mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in test_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_optimizer_step(self, test_model, batch_tensor_data):
        """Test model parameter updates"""
        
        import torch.optim as optim
        
        test_model.train()
        optimizer = optim.Adam(test_model.parameters(), lr=1e-4)
        
        # Store initial parameters
        initial_params = [param.clone() for param in test_model.parameters()]
        
        audio_input = batch_tensor_data['audio']
        video_input = batch_tensor_data['video']
        
        # Forward pass
        outputs = test_model(audio_input, video_input)
        loss = outputs['separated_waveforms'].mean()
        
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check parameters changed
        for initial_param, current_param in zip(initial_params, test_model.parameters()):
            if current_param.requires_grad:
                assert not torch.allclose(initial_param, current_param), \
                    "Parameters should change after optimization step"


@pytest.mark.gpu
class TestModelGPU:
    """Test model GPU functionality (requires GPU)"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_gpu_forward(self, test_config):
        """Test model forward pass on GPU"""
        
        device = 'cuda'
        model = AVSeparationTransformer(test_config).to(device)
        
        batch_size = 2
        audio_frames = 25
        video_frames = 20
        
        audio_input = torch.randn(batch_size, audio_frames, test_config.audio.n_mels).to(device)
        video_input = torch.randn(batch_size, video_frames, 3, *test_config.video.image_size).to(device)
        
        with torch.no_grad():
            outputs = model(audio_input, video_input)
        
        # Check outputs are on GPU
        assert outputs['separated_waveforms'].device.type == 'cuda'
        assert outputs['separated_spectrograms'].device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_model_cpu_gpu_transfer(self, test_model):
        """Test model transfer between CPU and GPU"""
        
        # Move to GPU
        test_model = test_model.cuda()
        
        # Check all parameters are on GPU
        for param in test_model.parameters():
            assert param.device.type == 'cuda'
        
        # Move back to CPU
        test_model = test_model.cpu()
        
        # Check all parameters are on CPU
        for param in test_model.parameters():
            assert param.device.type == 'cpu'