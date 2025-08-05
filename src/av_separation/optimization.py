"""
Performance Optimization for AV-Separation-Transformer
Model optimization, caching, batching, and inference acceleration
"""

import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import pickle
import logging

import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast
import numpy as np


@dataclass
class OptimizationResult:
    """Result of model optimization"""
    
    optimized_model: nn.Module
    optimization_time: float
    memory_reduction: float
    speedup_factor: float
    accuracy_loss: float
    optimization_config: Dict[str, Any]


class ModelOptimizer:
    """
    Comprehensive model optimization system
    """
    
    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization strategies
        self.strategies = {
            'quantization': self._apply_quantization,
            'pruning': self._apply_pruning,
            'distillation': self._apply_distillation,
            'tensorrt': self._apply_tensorrt,
            'torchscript': self._apply_torchscript,
            'onnx': self._apply_onnx_optimization
        }
    
    def optimize(
        self,
        strategies: List[str] = ['quantization', 'torchscript'],
        benchmark_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> OptimizationResult:
        """
        Apply optimization strategies to model
        
        Args:
            strategies: List of optimization strategies to apply
            benchmark_data: Data for benchmarking optimization impact
            
        Returns:
            OptimizationResult with optimized model and metrics
        """
        
        start_time = time.time()
        original_model = self.model
        optimized_model = self.model
        
        # Benchmark original model
        if benchmark_data:
            original_metrics = self._benchmark_model(original_model, benchmark_data)
        else:
            original_metrics = None
        
        optimization_config = {}
        
        # Apply optimization strategies sequentially
        for strategy in strategies:
            if strategy in self.strategies:
                self.logger.info(f"Applying {strategy} optimization...")
                
                try:
                    optimized_model, strategy_config = self.strategies[strategy](optimized_model)
                    optimization_config[strategy] = strategy_config
                    
                    self.logger.info(f"{strategy} optimization completed")
                    
                except Exception as e:
                    self.logger.error(f"{strategy} optimization failed: {e}")
                    continue
            else:
                self.logger.warning(f"Unknown optimization strategy: {strategy}")
        
        optimization_time = time.time() - start_time
        
        # Benchmark optimized model
        if benchmark_data and original_metrics:
            optimized_metrics = self._benchmark_model(optimized_model, benchmark_data)
            
            memory_reduction = (
                original_metrics['memory_mb'] - optimized_metrics['memory_mb']
            ) / original_metrics['memory_mb']
            
            speedup_factor = original_metrics['inference_time'] / optimized_metrics['inference_time']
            
            # Placeholder for accuracy comparison
            accuracy_loss = 0.0  # Would need validation data to compute
            
        else:
            memory_reduction = 0.0
            speedup_factor = 1.0
            accuracy_loss = 0.0
        
        result = OptimizationResult(
            optimized_model=optimized_model,
            optimization_time=optimization_time,
            memory_reduction=memory_reduction,
            speedup_factor=speedup_factor,
            accuracy_loss=accuracy_loss,
            optimization_config=optimization_config
        )
        
        self.logger.info(
            f"Optimization completed: {speedup_factor:.2f}x speedup, "
            f"{memory_reduction:.1%} memory reduction"
        )
        
        return result
    
    def _benchmark_model(
        self,
        model: nn.Module,
        benchmark_data: Tuple[torch.Tensor, torch.Tensor],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        
        audio_input, video_input = benchmark_data
        device = next(model.parameters()).device
        
        # Move inputs to correct device
        audio_input = audio_input.to(device)
        video_input = video_input.to(device)
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(audio_input, video_input)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark inference time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(audio_input, video_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.time() - start_time) / num_iterations
        
        # Measure memory usage
        if device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(device) / 1024**2
        else:
            memory_mb = 0.0  # CPU memory measurement would need psutil
        
        return {
            'inference_time': inference_time,
            'memory_mb': memory_mb
        }
    
    def _apply_quantization(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply dynamic quantization"""
        
        try:
            # Dynamic quantization for CPU inference
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            config = {
                'method': 'dynamic',
                'dtype': 'qint8',
                'target_modules': ['Linear', 'Conv1d', 'Conv2d', 'LSTM', 'GRU']
            }
            
            return quantized_model, config
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return model, {}
    
    def _apply_pruning(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply structured pruning"""
        
        try:
            import torch.nn.utils.prune as prune
            
            # Apply magnitude-based pruning to linear layers
            modules_to_prune = []
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    modules_to_prune.append((module, 'weight'))
            
            # Apply global unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.2  # Prune 20% of parameters
            )
            
            # Make pruning permanent
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
            
            config = {
                'method': 'global_unstructured',
                'pruning_ratio': 0.2,
                'criterion': 'L1'
            }
            
            return model, config
            
        except ImportError:
            self.logger.error("Pruning requires torch.nn.utils.prune")
            return model, {}
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model, {}
    
    def _apply_distillation(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply knowledge distillation (placeholder)"""
        
        # Knowledge distillation requires training data and student model
        # This is a placeholder for the actual implementation
        
        self.logger.info("Knowledge distillation requires separate training process")
        
        config = {
            'method': 'knowledge_distillation',
            'status': 'not_implemented'
        }
        
        return model, config
    
    def _apply_tensorrt(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply TensorRT optimization"""
        
        try:
            import torch_tensorrt
            
            # Compile model with TensorRT
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        shape=[1, 100, self.config.audio.n_mels],
                        dtype=torch.float32
                    ),
                    torch_tensorrt.Input(
                        shape=[1, 50, 3, *self.config.video.image_size],
                        dtype=torch.float32
                    )
                ],
                enabled_precisions={torch.float32, torch.half}
            )
            
            config = {
                'method': 'tensorrt',
                'precision': ['fp32', 'fp16'],
                'optimization_level': 3
            }
            
            return compiled_model, config
            
        except ImportError:
            self.logger.error("TensorRT optimization requires torch-tensorrt")
            return model, {}
        except Exception as e:
            self.logger.error(f"TensorRT optimization failed: {e}")
            return model, {}
    
    def _apply_torchscript(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply TorchScript optimization"""
        
        try:
            model.eval()
            
            # Create dummy inputs
            dummy_audio = torch.randn(1, 100, self.config.audio.n_mels)
            dummy_video = torch.randn(1, 50, 3, *self.config.video.image_size)
            
            # Try tracing first
            try:
                scripted_model = torch.jit.trace(model, (dummy_audio, dummy_video))
                method = 'trace'
            except Exception:
                # Fall back to scripting
                scripted_model = torch.jit.script(model)
                method = 'script'
            
            # Optimize for inference
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            config = {
                'method': 'torchscript',
                'compilation_method': method,
                'optimized_for_inference': True
            }
            
            return scripted_model, config
            
        except Exception as e:
            self.logger.error(f"TorchScript optimization failed: {e}")
            return model, {}
    
    def _apply_onnx_optimization(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply ONNX optimization (placeholder)"""
        
        # ONNX optimization would require exporting to ONNX and loading back
        # This is a placeholder for the actual implementation
        
        self.logger.info("ONNX optimization requires separate export/import process")
        
        config = {
            'method': 'onnx',
            'status': 'not_implemented'
        }
        
        return model, config


class InferenceCache:
    """
    LRU cache for inference results with memory management
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 1024):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.usage_order = []
        self.current_memory_mb = 0.0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def _compute_key(self, audio_input: torch.Tensor, video_input: torch.Tensor) -> str:
        """Compute cache key from inputs"""
        
        # Create hash from tensor data
        audio_hash = hashlib.md5(audio_input.cpu().numpy().tobytes()).hexdigest()[:16]
        video_hash = hashlib.md5(video_input.cpu().numpy().tobytes()).hexdigest()[:16]
        
        return f"{audio_hash}_{video_hash}"
    
    def _estimate_memory_mb(self, data: Any) -> float:
        """Estimate memory usage of cached data"""
        
        try:
            serialized = pickle.dumps(data)
            return len(serialized) / 1024**2
        except Exception:
            return 1.0  # Default estimate
    
    def get(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """Get cached inference result"""
        
        key = self._compute_key(audio_input, video_input)
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.usage_order.remove(key)
                self.usage_order.append(key)
                
                return self.cache[key]['result']
        
        return None
    
    def put(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        result: Dict[str, Any]
    ):
        """Cache inference result"""
        
        key = self._compute_key(audio_input, video_input)
        memory_estimate = self._estimate_memory_mb(result)
        
        with self.lock:
            # Check if we need to evict items
            while (
                len(self.cache) >= self.max_size or
                self.current_memory_mb + memory_estimate > self.max_memory_mb
            ):
                if not self.usage_order:
                    break
                
                # Evict least recently used item
                lru_key = self.usage_order.pop(0)
                if lru_key in self.cache:
                    evicted_memory = self.cache[lru_key]['memory_mb']
                    del self.cache[lru_key]
                    self.current_memory_mb -= evicted_memory
            
            # Add new item
            self.cache[key] = {
                'result': result,
                'memory_mb': memory_estimate
            }
            self.usage_order.append(key)
            self.current_memory_mb += memory_estimate
            
            self.logger.debug(f"Cached result for key {key[:8]}... ({memory_estimate:.2f} MB)")
    
    def clear(self):
        """Clear all cached items"""
        
        with self.lock:
            self.cache.clear()
            self.usage_order.clear()
            self.current_memory_mb = 0.0
            
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.current_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'utilization': len(self.cache) / self.max_size,
                'memory_utilization': self.current_memory_mb / self.max_memory_mb
            }


class BatchProcessor:
    """
    Dynamic batch processing for improved throughput
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
        device: str = 'cuda'
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.request_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'avg_batch_size': 0.0,
            'avg_latency': 0.0
        }
    
    def start(self):
        """Start batch processing thread"""
        
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processing thread"""
        
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("Batch processor stopped")
    
    def process_async(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        callback: Callable[[Dict[str, Any]], None],
        request_id: str = None
    ):
        """Submit request for asynchronous batch processing"""
        
        request = {
            'audio_input': audio_input,
            'video_input': video_input,
            'callback': callback,
            'request_id': request_id or f"req_{time.time()}",
            'timestamp': time.time()
        }
        
        self.request_queue.put(request)
        self.stats['total_requests'] += 1
    
    def _process_batches(self):
        """Main batch processing loop"""
        
        while self.running:
            try:
                # Collect requests for batching
                batch_requests = []
                deadline = time.time() + self.max_wait_time
                
                # Get first request (blocking)
                try:
                    first_request = self.request_queue.get(timeout=self.max_wait_time)
                    batch_requests.append(first_request)
                except queue.Empty:
                    continue
                
                # Collect additional requests until batch is full or deadline reached
                while (
                    len(batch_requests) < self.max_batch_size and
                    time.time() < deadline
                ):
                    try:
                        remaining_time = max(0, deadline - time.time())
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                
                # Process batch
                if batch_requests:
                    self._process_batch(batch_requests)
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
    
    def _process_batch(self, batch_requests: List[Dict[str, Any]]):
        """Process a batch of requests"""
        
        start_time = time.time()
        batch_size = len(batch_requests)
        
        try:
            # Prepare batch inputs
            audio_inputs = []
            video_inputs = []
            
            for request in batch_requests:
                audio_inputs.append(request['audio_input'])
                video_inputs.append(request['video_input'])
            
            # Stack inputs into batches
            batch_audio = torch.stack(audio_inputs).to(self.device)
            batch_video = torch.stack(video_inputs).to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                batch_outputs = self.model(batch_audio, batch_video)
            
            # Split outputs and send to callbacks
            for i, request in enumerate(batch_requests):
                # Extract individual result from batch
                individual_output = {}
                for key, value in batch_outputs.items():
                    if isinstance(value, torch.Tensor):
                        individual_output[key] = value[i]
                    elif isinstance(value, list) and len(value) > i:
                        individual_output[key] = value[i]
                    else:
                        individual_output[key] = value
                
                # Call callback
                try:
                    request['callback'](individual_output)
                except Exception as e:
                    self.logger.error(f"Callback error for {request['request_id']}: {e}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['batched_requests'] += batch_size
            
            # Update running averages
            total_batches = self.stats['batched_requests'] / self.max_batch_size
            self.stats['avg_batch_size'] = (
                self.stats['avg_batch_size'] * (total_batches - 1) + batch_size
            ) / total_batches
            
            self.stats['avg_latency'] = (
                self.stats['avg_latency'] * (total_batches - 1) + processing_time
            ) / total_batches
            
            self.logger.debug(
                f"Processed batch of {batch_size} requests in {processing_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Batch inference error: {e}")
            
            # Send error to all callbacks
            for request in batch_requests:
                try:
                    request['callback']({'error': str(e)})
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        
        return {
            'queue_size': self.request_queue.qsize(),
            'running': self.running,
            **self.stats
        }


class PerformanceProfiler:
    """
    Performance profiling and optimization recommendations
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
    
    def profile_inference(
        self,
        audio_input: torch.Tensor,
        video_input: torch.Tensor,
        num_iterations: int = 100,
        profile_name: str = "default"
    ) -> Dict[str, Any]:
        """Profile model inference performance"""
        
        device = next(self.model.parameters()).device
        audio_input = audio_input.to(device)
        video_input = video_input.to(device)
        
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(audio_input, video_input)
        
        # Profile with torch.profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ] if device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.model(audio_input, video_input)
        
        # Analyze results
        profile_data = self._analyze_profile(profiler, num_iterations)
        self.profiles[profile_name] = profile_data
        
        return profile_data
    
    def _analyze_profile(self, profiler, num_iterations: int) -> Dict[str, Any]:
        """Analyze profiler results"""
        
        # Get key averages
        key_averages = profiler.key_averages(group_by_stack_n=5)
        
        # Extract important metrics
        cpu_time_total = sum(item.cpu_time_total for item in key_averages) / 1000  # Convert to ms
        cuda_time_total = sum(item.cuda_time_total for item in key_averages) / 1000
        
        # Top operations by time
        top_cpu_ops = sorted(
            [(item.key, item.cpu_time_total / 1000) for item in key_averages],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_cuda_ops = sorted(
            [(item.key, item.cuda_time_total / 1000) for item in key_averages],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Memory usage
        memory_usage = {}
        for item in key_averages:
            if item.cpu_memory_usage > 0 or item.cuda_memory_usage > 0:
                memory_usage[item.key] = {
                    'cpu_memory_mb': item.cpu_memory_usage / 1024**2,
                    'cuda_memory_mb': item.cuda_memory_usage / 1024**2
                }
        
        profile_data = {
            'num_iterations': num_iterations,
            'avg_cpu_time_ms': cpu_time_total / num_iterations,
            'avg_cuda_time_ms': cuda_time_total / num_iterations,
            'total_cpu_time_ms': cpu_time_total,
            'total_cuda_time_ms': cuda_time_total,
            'top_cpu_operations': top_cpu_ops,
            'top_cuda_operations': top_cuda_ops,
            'memory_usage': memory_usage
        }
        
        return profile_data
    
    def generate_optimization_recommendations(
        self,
        profile_name: str = "default"
    ) -> List[str]:
        """Generate optimization recommendations based on profiling"""
        
        if profile_name not in self.profiles:
            return ["No profile data available. Run profile_inference() first."]
        
        profile = self.profiles[profile_name]
        recommendations = []
        
        # Check for common performance issues
        
        # 1. GPU utilization
        cpu_time = profile['avg_cpu_time_ms']
        cuda_time = profile['avg_cuda_time_ms']
        
        if cuda_time > 0 and cpu_time > cuda_time * 2:
            recommendations.append(
                "CPU bottleneck detected. Consider optimizing data preprocessing "
                "or using more efficient data loading."
            )
        
        # 2. Memory usage
        memory_ops = profile['memory_usage']
        high_memory_ops = [
            op for op, usage in memory_ops.items()
            if usage['cuda_memory_mb'] > 100
        ]
        
        if high_memory_ops:
            recommendations.append(
                f"High memory usage operations detected: {high_memory_ops[:3]}. "
                "Consider gradient checkpointing or reducing batch size."
            )
        
        # 3. Slow operations
        top_ops = profile['top_cuda_operations'] if cuda_time > 0 else profile['top_cpu_operations']
        if top_ops:
            slowest_op = top_ops[0]
            if slowest_op[1] > cuda_time * 0.3:  # If single op takes >30% of time
                recommendations.append(
                    f"Operation '{slowest_op[0]}' is taking {slowest_op[1]:.2f}ms "
                    f"({slowest_op[1]/cuda_time*100:.1f}% of total time). "
                    "Consider optimizing this operation."
                )
        
        # 4. Inference time
        total_inference_time = max(cpu_time, cuda_time)
        if total_inference_time > 100:  # >100ms
            recommendations.append(
                f"Inference time is {total_inference_time:.2f}ms. "
                "Consider model quantization, pruning, or TensorRT optimization."
            )
        
        if not recommendations:
            recommendations.append("Model performance looks good! No major issues detected.")
        
        return recommendations
    
    def save_profile_report(self, profile_name: str, output_path: str):
        """Save detailed profile report"""
        
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        profile = self.profiles[profile_name]
        recommendations = self.generate_optimization_recommendations(profile_name)
        
        report = {
            'profile_name': profile_name,
            'model_info': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            },
            'performance_metrics': profile,
            'optimization_recommendations': recommendations
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Profile report saved to {output_path}")


# Factory functions for easy instantiation
def create_optimized_model(
    model: nn.Module,
    config: Any,
    strategies: List[str] = ['quantization', 'torchscript'],
    benchmark_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> OptimizationResult:
    """Create optimized model with specified strategies"""
    
    optimizer = ModelOptimizer(model, config)
    return optimizer.optimize(strategies, benchmark_data)


def create_inference_cache(max_size: int = 1000, max_memory_mb: int = 1024) -> InferenceCache:
    """Create inference cache with specified limits"""
    
    return InferenceCache(max_size, max_memory_mb)


def create_batch_processor(
    model: nn.Module,
    max_batch_size: int = 8,
    max_wait_time: float = 0.1,
    device: str = 'cuda'
) -> BatchProcessor:
    """Create batch processor with specified parameters"""
    
    processor = BatchProcessor(model, max_batch_size, max_wait_time, device)
    processor.start()
    return processor