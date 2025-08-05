"""
Model Export Utilities for AV-Separation-Transformer
Supports ONNX and TorchScript export with optimizations
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import onnx
import onnxruntime as ort
from .config import SeparatorConfig


def export_onnx(
    model: nn.Module,
    output_path: str,
    opset_version: int = 17,
    optimize: bool = True,
    quantize: bool = False,
    dynamic_axes: bool = True,
    input_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Export model to ONNX format with optimizations
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        optimize: Apply ONNX optimizations
        quantize: Apply INT8 quantization
        dynamic_axes: Use dynamic batch/sequence dimensions
        input_shape: Custom input shape (batch, channels, height, width)
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    # Default input shapes for AV separation
    if input_shape is None:
        batch_size = 1
        audio_shape = (batch_size, 80, 250)  # mel spectrogram
        video_shape = (batch_size, 120, 3, 224, 224)  # video frames
    else:
        audio_shape, video_shape = input_shape
    
    # Create dummy inputs
    dummy_audio = torch.randn(*audio_shape).to(device)
    dummy_video = torch.randn(*video_shape).to(device)
    dummy_inputs = (dummy_audio, dummy_video)
    
    # Define dynamic axes for variable sequence lengths
    if dynamic_axes:
        dynamic_axes_dict = {
            'audio_input': {0: 'batch_size', 2: 'audio_time'},
            'video_input': {0: 'batch_size', 1: 'video_time'},
            'separated_waveforms': {0: 'batch_size', 2: 'output_time'},
            'speaker_logits': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    # Export to ONNX
    print(f"Exporting model to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_input', 'video_input'],
        output_names=['separated_waveforms', 'separated_spectrograms', 'speaker_logits'],
        dynamic_axes=dynamic_axes_dict,
        verbose=False
    )
    
    # Verify the exported model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except Exception as e:
        warnings.warn(f"ONNX model validation failed: {e}")
    
    # Apply optimizations
    if optimize:
        print("Applying ONNX optimizations...")
        optimized_path = output_path.replace('.onnx', '_optimized.onnx')
        _optimize_onnx_model(output_path, optimized_path)
        
        # Replace original with optimized version
        Path(optimized_path).rename(output_path)
        print("✓ ONNX optimizations applied")
    
    # Apply quantization
    if quantize:
        print("Applying INT8 quantization...")
        quantized_path = output_path.replace('.onnx', '_quantized.onnx')
        _quantize_onnx_model(output_path, quantized_path, dummy_inputs)
        print(f"✓ Quantized model saved: {quantized_path}")
    
    # Test inference
    print("Testing ONNX inference...")
    try:
        _test_onnx_inference(output_path, dummy_inputs)
        print("✓ ONNX inference test passed")
    except Exception as e:
        warnings.warn(f"ONNX inference test failed: {e}")
    
    print(f"✓ ONNX export completed: {output_path}")


def export_torchscript(
    model: nn.Module,
    output_path: str,
    optimize: bool = True,
    input_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Export model to TorchScript format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        optimize: Apply TorchScript optimizations
        input_shape: Custom input shape
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    # Default input shapes
    if input_shape is None:
        batch_size = 1
        audio_shape = (batch_size, 80, 250)
        video_shape = (batch_size, 120, 3, 224, 224)
    else:
        audio_shape, video_shape = input_shape
    
    # Create dummy inputs
    dummy_audio = torch.randn(*audio_shape).to(device)
    dummy_video = torch.randn(*video_shape).to(device)
    
    print(f"Exporting model to TorchScript: {output_path}")
    
    try:
        # Try tracing first (usually works better for inference)
        scripted_model = torch.jit.trace(model, (dummy_audio, dummy_video))
        print("✓ Model traced successfully")
    except Exception as e:
        print(f"Tracing failed ({e}), trying scripting...")
        try:
            scripted_model = torch.jit.script(model)
            print("✓ Model scripted successfully")
        except Exception as e2:
            raise RuntimeError(f"Both tracing and scripting failed: {e}, {e2}")
    
    # Apply optimizations
    if optimize:
        print("Applying TorchScript optimizations...")
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        print("✓ TorchScript optimizations applied")
    
    # Save the model
    scripted_model.save(output_path)
    
    # Test loading and inference
    print("Testing TorchScript inference...")
    try:
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            _ = loaded_model(dummy_audio, dummy_video)
        
        print("✓ TorchScript inference test passed")
    except Exception as e:
        warnings.warn(f"TorchScript inference test failed: {e}")
    
    print(f"✓ TorchScript export completed: {output_path}")


def _optimize_onnx_model(input_path: str, output_path: str) -> None:
    """Apply ONNX Runtime optimizations"""
    try:
        from onnxruntime.tools import optimizer
        
        # Create optimization configuration
        opt_config = optimizer.OptimizationConfig()
        opt_config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt_config.optimized_model_filepath = output_path
        
        # Apply optimizations
        optimizer.optimize_model(
            input_path,
            opt_config
        )
        
    except ImportError:
        warnings.warn("onnxruntime.tools not available, skipping optimizations")
    except Exception as e:
        warnings.warn(f"ONNX optimization failed: {e}")


def _quantize_onnx_model(
    input_path: str, 
    output_path: str, 
    dummy_inputs: Tuple[torch.Tensor, ...]
) -> None:
    """Apply INT8 quantization to ONNX model"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Convert dummy inputs to numpy
        dummy_data = []
        for tensor in dummy_inputs:
            dummy_data.append(tensor.cpu().numpy())
        
        # Apply dynamic quantization
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
            optimize_model=True
        )
        
    except ImportError:
        warnings.warn("onnxruntime.quantization not available, skipping quantization")
    except Exception as e:
        warnings.warn(f"ONNX quantization failed: {e}")


def _test_onnx_inference(model_path: str, dummy_inputs: Tuple[torch.Tensor, ...]) -> None:
    """Test ONNX model inference"""
    
    # Create inference session
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Prepare inputs
    input_dict = {}
    for i, (input_tensor, input_meta) in enumerate(zip(dummy_inputs, session.get_inputs())):
        input_dict[input_meta.name] = input_tensor.cpu().numpy()
    
    # Run inference
    outputs = session.run(None, input_dict)
    
    # Validate outputs
    if len(outputs) == 0:
        raise RuntimeError("No outputs from ONNX model")
    
    for i, output in enumerate(outputs):
        if output.size == 0:
            raise RuntimeError(f"Empty output {i} from ONNX model")


def create_mobile_config() -> SeparatorConfig:
    """Create optimized configuration for mobile deployment"""
    
    config = SeparatorConfig()
    
    # Reduce model complexity
    config.model.audio_encoder_layers = 4
    config.model.audio_encoder_dim = 256
    config.model.audio_encoder_ffn_dim = 1024
    
    config.model.video_encoder_layers = 3
    config.model.video_encoder_dim = 128
    config.model.video_encoder_ffn_dim = 512
    
    config.model.fusion_layers = 3
    config.model.fusion_dim = 256
    
    config.model.decoder_layers = 4
    config.model.decoder_dim = 256
    config.model.decoder_ffn_dim = 1024
    
    # Optimize for inference
    config.model.max_speakers = 2
    config.model.dropout = 0.0
    config.model.use_flash_attention = False
    
    # Reduce audio complexity
    config.audio.n_mels = 64
    config.audio.chunk_duration = 2.0
    
    # Reduce video complexity
    config.video.image_size = (112, 112)
    config.video.face_size = (64, 64)
    config.video.lip_size = (48, 48)
    config.video.max_faces = 2
    
    return config


def export_for_mobile(
    model: nn.Module,
    output_dir: str,
    platforms: list = ['ios', 'android']
) -> Dict[str, str]:
    """
    Export model for mobile platforms
    
    Args:
        model: PyTorch model to export
        output_dir: Directory to save mobile models
        platforms: List of target platforms
        
    Returns:
        Dictionary mapping platform to model path
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_models = {}
    
    # Create mobile-optimized config
    mobile_config = create_mobile_config()
    
    if 'ios' in platforms:
        print("Exporting for iOS (CoreML)...")
        try:
            import coremltools as ct
            
            # Trace model for CoreML
            model.eval()
            device = next(model.parameters()).device
            
            dummy_audio = torch.randn(1, 64, 125).to(device)  # Reduced size
            dummy_video = torch.randn(1, 60, 3, 112, 112).to(device)
            
            traced_model = torch.jit.trace(model, (dummy_audio, dummy_video))
            
            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[
                    ct.TensorType(name="audio_input", shape=dummy_audio.shape),
                    ct.TensorType(name="video_input", shape=dummy_video.shape)
                ],
                minimum_deployment_target=ct.target.iOS15
            )
            
            ios_path = output_dir / "av_sepnet_mobile.mlmodel"
            coreml_model.save(str(ios_path))
            exported_models['ios'] = str(ios_path)
            print(f"✓ iOS model exported: {ios_path}")
            
        except ImportError:
            warnings.warn("coremltools not available, skipping iOS export")
        except Exception as e:
            warnings.warn(f"iOS export failed: {e}")
    
    if 'android' in platforms:
        print("Exporting for Android (TensorFlow Lite)...")
        try:
            # Export optimized ONNX first
            onnx_path = output_dir / "av_sepnet_mobile.onnx"
            export_onnx(
                model,
                str(onnx_path),
                optimize=True,
                quantize=True,
                input_shape=((1, 64, 125), (1, 60, 3, 112, 112))
            )
            
            # Convert ONNX to TensorFlow Lite via tf2onnx
            tflite_path = output_dir / "av_sepnet_mobile.tflite"
            _convert_onnx_to_tflite(str(onnx_path), str(tflite_path))
            
            exported_models['android'] = str(tflite_path)
            print(f"✓ Android model exported: {tflite_path}")
            
        except Exception as e:
            warnings.warn(f"Android export failed: {e}")
    
    return exported_models


def _convert_onnx_to_tflite(onnx_path: str, tflite_path: str) -> None:
    """Convert ONNX model to TensorFlow Lite"""
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_model_path = onnx_path.replace('.onnx', '_tf')
        tf_rep.export_graph(tf_model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
    except ImportError:
        raise ImportError("tensorflow and onnx-tf required for TensorFlow Lite export")


def benchmark_exported_models(model_paths: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Benchmark exported models across different formats
    
    Args:
        model_paths: Dictionary mapping format name to model path
        
    Returns:
        Dictionary with benchmark results for each format
    """
    
    results = {}
    
    for format_name, model_path in model_paths.items():
        print(f"Benchmarking {format_name} model...")
        
        try:
            if format_name == 'onnx':
                result = _benchmark_onnx(model_path)
            elif format_name == 'torchscript':
                result = _benchmark_torchscript(model_path)
            else:
                print(f"Benchmarking not implemented for {format_name}")
                continue
                
            results[format_name] = result
            print(f"✓ {format_name}: {result['mean_latency_ms']:.2f}ms avg latency")
            
        except Exception as e:
            print(f"✗ {format_name} benchmark failed: {e}")
    
    return results


def _benchmark_onnx(model_path: str, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark ONNX model performance"""
    
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Create dummy inputs
    dummy_audio = np.random.randn(1, 80, 250).astype(np.float32)
    dummy_video = np.random.randn(1, 120, 3, 224, 224).astype(np.float32)
    
    input_dict = {
        session.get_inputs()[0].name: dummy_audio,
        session.get_inputs()[1].name: dummy_video
    }
    
    # Warm up
    for _ in range(10):
        _ = session.run(None, input_dict)
    
    # Benchmark
    import time
    latencies = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(None, input_dict)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99)
    }


def _benchmark_torchscript(model_path: str, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark TorchScript model performance"""
    
    model = torch.jit.load(model_path)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create dummy inputs
    dummy_audio = torch.randn(1, 80, 250).to(device)
    dummy_video = torch.randn(1, 120, 3, 224, 224).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_audio, dummy_video)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_audio, dummy_video)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99)
    }