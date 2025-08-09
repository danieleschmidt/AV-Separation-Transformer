#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from av_separation import SeparatorConfig
from av_separation.models import AVSeparationTransformer

def test_basic_functionality():
    print("=== AV-Separation Transformer Basic Demo ===")
    
    # Create config
    config = SeparatorConfig()
    print(f"✓ Config created with {config.model.max_speakers} max speakers")
    
    # Create model
    model = AVSeparationTransformer(config)
    print(f"✓ Model created with {model.get_num_params():,} parameters")
    
    # Test with dummy data
    batch_size = 2
    seq_len = 100
    audio_dim = config.audio.n_mels
    video_frames = 30
    
    # Create dummy inputs
    dummy_audio = torch.randn(batch_size, audio_dim, seq_len)
    dummy_video = torch.randn(batch_size, video_frames, 3, *config.video.image_size)
    
    print(f"✓ Created dummy audio: {dummy_audio.shape}")
    print(f"✓ Created dummy video: {dummy_video.shape}")
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_audio, dummy_video)
    
    print(f"✓ Forward pass completed")
    print(f"  - Separated waveforms: {outputs['separated_waveforms'].shape}")
    print(f"  - Speaker logits: {outputs['speaker_logits'].shape}")
    print(f"  - Alignment score: {outputs['alignment_score'].shape}")
    
    # Test separation method
    separated = model.separate(dummy_audio[0], dummy_video[0])
    print(f"✓ Separation method: {separated.shape}")
    
    print("\n=== Generation 1 (MAKE IT WORK) Complete ===")
    return True

if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        print(f"✓ Basic functionality test: {'PASSED' if success else 'FAILED'}")
    except Exception as e:
        print(f"✗ Basic functionality test: FAILED - {e}")
        import traceback
        traceback.print_exc()