#!/usr/bin/env python3
"""
Core System Validation Test
Tests the system architecture without external dependencies
"""

def test_import_structure():
    """Test that the module structure is correct"""
    print("🔍 Testing module structure...")
    
    try:
        from src.av_separation.config import SeparatorConfig
        print("✓ Config module imported successfully")
        
        config = SeparatorConfig()
        print("✓ Config object created successfully")
        print(f"  - Audio sample rate: {config.audio.sample_rate}")
        print(f"  - Video FPS: {config.video.fps}")
        print(f"  - Model max speakers: {config.model.max_speakers}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_model_structure():
    """Test model structure without instantiation"""
    print("\n🔍 Testing model structure...")
    
    try:
        # Test imports without instantiation to avoid torch dependency
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        # Import configuration
        from av_separation.config import SeparatorConfig
        config = SeparatorConfig()
        
        # Check that all model files exist and have correct structure
        model_files = [
            'src/av_separation/models/__init__.py',
            'src/av_separation/models/transformer.py',
            'src/av_separation/models/audio_encoder.py',
            'src/av_separation/models/video_encoder.py',
            'src/av_separation/models/fusion.py',
            'src/av_separation/models/decoder.py'
        ]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} missing")
                return False
        
        # Check utility files
        util_files = [
            'src/av_separation/utils/audio.py',
            'src/av_separation/utils/video.py'
        ]
        
        for file_path in util_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} missing")
                return False
        
        print("✓ All core files present")
        return True
        
    except Exception as e:
        print(f"✗ Model structure test failed: {e}")
        return False

def test_architecture_consistency():
    """Test that the architecture components are consistent"""
    print("\n🔍 Testing architecture consistency...")
    
    try:
        from src.av_separation.config import SeparatorConfig
        config = SeparatorConfig()
        
        # Test configuration consistency
        assert config.model.audio_encoder_dim > 0, "Audio encoder dim must be positive"
        assert config.model.video_encoder_dim > 0, "Video encoder dim must be positive"
        assert config.model.fusion_dim > 0, "Fusion dim must be positive"
        assert config.model.decoder_dim > 0, "Decoder dim must be positive"
        assert config.model.max_speakers > 0, "Max speakers must be positive"
        
        print("✓ Configuration dimensions are valid")
        
        # Test audio configuration
        assert config.audio.sample_rate > 0, "Sample rate must be positive"
        assert config.audio.n_fft > 0, "N_FFT must be positive"
        assert config.audio.hop_length > 0, "Hop length must be positive"
        assert config.audio.n_mels > 0, "N_mels must be positive"
        
        print("✓ Audio configuration is valid")
        
        # Test video configuration
        assert config.video.fps > 0, "FPS must be positive"
        assert len(config.video.image_size) == 2, "Image size must be 2D"
        assert len(config.video.face_size) == 2, "Face size must be 2D"
        assert len(config.video.lip_size) == 2, "Lip size must be 2D"
        
        print("✓ Video configuration is valid")
        print("✓ Architecture consistency validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture consistency test failed: {e}")
        return False

def test_generation_status():
    """Check what generation features are already implemented"""
    print("\n🔍 Checking generation implementation status...")
    
    import os
    
    generation_files = {
        "Generation 1 (Core)": [
            "src/av_separation/separator.py",
            "src/av_separation/models/transformer.py",
            "src/av_separation/config.py"
        ],
        "Generation 2 (Robust)": [
            "src/av_separation/robust_core.py",
            "src/av_separation/enhanced_security.py",
            "src/av_separation/monitoring.py"
        ],
        "Generation 3 (Scale)": [
            "src/av_separation/scaling.py",
            "src/av_separation/performance_optimizer.py",
            "src/av_separation/auto_scaler.py"
        ],
        "Generation 4 (Transcendence)": [
            "src/av_separation/quantum_enhanced/",
            "src/av_separation/neuromorphic/",
            "src/av_separation/self_improving/"
        ]
    }
    
    status = {}
    for gen_name, files in generation_files.items():
        implemented = 0
        for file_path in files:
            if os.path.exists(file_path):
                implemented += 1
        
        percentage = (implemented / len(files)) * 100
        status[gen_name] = percentage
        
        if percentage == 100:
            print(f"✓ {gen_name}: {percentage:.0f}% complete")
        elif percentage >= 50:
            print(f"⚠ {gen_name}: {percentage:.0f}% complete")
        else:
            print(f"✗ {gen_name}: {percentage:.0f}% complete")
    
    return status

def main():
    """Run all validation tests"""
    print("🚀 TERRAGON CORE SYSTEM VALIDATION")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(test_import_structure())
    results.append(test_model_structure())
    results.append(test_architecture_consistency())
    generation_status = test_generation_status()
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Core Tests: {passed}/{total} passed")
    
    for gen_name, percentage in generation_status.items():
        print(f"{gen_name}: {percentage:.0f}% implemented")
    
    # Overall status
    print("\n🎯 SYSTEM STATUS")
    print("=" * 50)
    
    if passed == total:
        print("✓ Core system architecture is VALID")
        print("✓ Ready for autonomous enhancement execution")
        
        # Check if we need to implement or enhance
        if all(p >= 90 for p in generation_status.values()):
            print("✓ All generations appear complete - ready for validation")
        else:
            print("⚠ Some generations need implementation/enhancement")
            
        return True
    else:
        print("✗ Core system has issues - fix required before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)