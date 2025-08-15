"""
Advanced Performance Optimization Engine
High-performance inference with intelligent caching and resource management.
"""

import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import psutil
import gc
from pathlib import Path
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import wraps, lru_cache
import logging


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time: float
    memory_usage: float
    gpu_utilization: float
    cache_hit_rate: float
    throughput: float  # samples per second
    latency_p95: float
    energy_consumption: float


class IntelligentCache:
    """
    Advanced caching system with LRU eviction and performance-based retention.
    """
    
    def __init__(self, max_size_gb: float = 2.0, ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_gb * 1024**3
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def _compute_key(self, data: torch.Tensor, params: Dict[str, Any]) -> str:
        """Compute cache key from input data and parameters."""
        # Hash tensor shape and data type
        tensor_info = f"{data.shape}_{data.dtype}_{data.device}"
        
        # Hash parameters
        param_str = str(sorted(params.items()))
        
        # Create combined hash
        combined = f"{tensor_info}_{param_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _get_size(self, obj: Any) -> int:
        """Estimate memory size of cached object."""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        else:
            return len(pickle.dumps(obj))
    
    def _evict_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def _evict_lru(self):
        """Evict least recently used entries to free memory."""
        current_size = sum(self._get_size(data) for data, _ in self.cache.values())
        
        while current_size > self.max_size_bytes and self.cache:
            # Find LRU entry considering both recency and frequency
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.access_times[k] * (1 + self.access_counts[k] * 0.1)
            )
            
            # Remove LRU entry
            data, _ = self.cache[lru_key]
            current_size -= self._get_size(data)
            
            del self.cache[lru_key]
            del self.access_times[lru_key]
            del self.access_counts[lru_key]
    
    def get(self, data: torch.Tensor, params: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Retrieve cached result if available."""
        with self.lock:
            key = self._compute_key(data, params)
            
            if key in self.cache:
                cached_data, timestamp = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]
                    self.miss_count += 1
                    return None
                
                # Update access information
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.hit_count += 1
                return cached_data.clone() if isinstance(cached_data, torch.Tensor) else cached_data
            
            self.miss_count += 1
            return None
    
    def put(self, data: torch.Tensor, params: Dict[str, Any], result: torch.Tensor):
        """Cache computation result."""
        with self.lock:
            key = self._compute_key(data, params)
            current_time = time.time()
            
            # Store result
            self.cache[key] = (result.clone() if isinstance(result, torch.Tensor) else result, current_time)
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            
            # Cleanup expired entries
            self._evict_expired()
            
            # Evict LRU entries if necessary
            self._evict_lru()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.hit_count = 0
            self.miss_count = 0


class TensorJITCompiler:
    """
    JIT compilation for performance-critical operations.
    """
    
    def __init__(self):
        self.compiled_functions = {}
        self.logger = logging.getLogger(__name__)
    
    def compile_model(self, model: nn.Module, example_inputs: Tuple[torch.Tensor, ...]) -> jit.ScriptModule:
        """Compile model using TorchScript for optimized inference."""
        try:
            # Trace the model
            model.eval()
            with torch.no_grad():
                traced_model = jit.trace(model, example_inputs)
            
            # Optimize for inference
            traced_model = jit.optimize_for_inference(traced_model)
            
            self.logger.info("Model successfully compiled with TorchScript")
            return traced_model
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed: {e}. Using original model.")
            return model
    
    def compile_function(self, func: Callable, name: str = None) -> Callable:
        """Compile function using TorchScript."""
        if name is None:
            name = func.__name__
        
        if name in self.compiled_functions:
            return self.compiled_functions[name]
        
        try:
            compiled_func = jit.script(func)
            self.compiled_functions[name] = compiled_func
            self.logger.info(f"Function '{name}' compiled with TorchScript")
            return compiled_func
        except Exception as e:
            self.logger.warning(f"Function compilation failed for '{name}': {e}")
            return func


class BatchProcessor:
    """
    Intelligent batch processing for optimal GPU utilization.
    """
    
    def __init__(self, max_batch_size: int = 32, target_utilization: float = 0.85):
        self.max_batch_size = max_batch_size
        self.target_utilization = target_utilization
        self.optimal_batch_sizes = {}
        self.processing_times = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def find_optimal_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Dynamically find optimal batch size for given input shape."""
        cache_key = f"{input_shape}"
        
        if cache_key in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[cache_key]
        
        # Binary search for optimal batch size
        low, high = 1, self.max_batch_size
        optimal_size = 1
        
        model.eval()
        with torch.no_grad():
            while low <= high:
                batch_size = (low + high) // 2
                
                try:
                    # Create test batch
                    test_batch = torch.randn(batch_size, *input_shape[1:], device=next(model.parameters()).device)
                    
                    # Measure memory usage before
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        memory_before = torch.cuda.memory_allocated()
                    
                    # Test inference
                    start_time = time.time()
                    _ = model(test_batch, test_batch)  # Assuming dual input
                    inference_time = time.time() - start_time
                    
                    # Measure memory usage after
                    if torch.cuda.is_available():
                        memory_after = torch.cuda.memory_allocated()
                        memory_used = memory_after - memory_before
                        
                        # Check if we're within target utilization
                        total_memory = torch.cuda.get_device_properties(0).total_memory
                        utilization = memory_used / total_memory
                        
                        if utilization <= self.target_utilization:
                            optimal_size = batch_size
                            low = batch_size + 1
                        else:
                            high = batch_size - 1
                    else:
                        # For CPU, use timing-based optimization
                        throughput = batch_size / inference_time
                        if batch_size == 1 or throughput > getattr(self, 'best_throughput', 0):
                            self.best_throughput = throughput
                            optimal_size = batch_size
                            low = batch_size + 1
                        else:
                            high = batch_size - 1
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        high = batch_size - 1
                    else:
                        break
        
        self.optimal_batch_sizes[cache_key] = optimal_size
        self.logger.info(f"Optimal batch size for {input_shape}: {optimal_size}")
        
        return optimal_size
    
    def process_batch(self, model: nn.Module, inputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        """Process inputs in optimal batches."""
        if not inputs:
            return []
        
        # Determine optimal batch size
        sample_shape = inputs[0][0].shape
        optimal_batch_size = self.find_optimal_batch_size(model, sample_shape)
        
        results = []
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(inputs), optimal_batch_size):
                batch_inputs = inputs[i:i + optimal_batch_size]
                
                # Stack inputs
                audio_batch = torch.stack([inp[0] for inp in batch_inputs])
                video_batch = torch.stack([inp[1] for inp in batch_inputs])
                
                # Process batch
                start_time = time.time()
                batch_output = model(audio_batch, video_batch)
                processing_time = time.time() - start_time
                
                # Record processing time
                self.processing_times[len(batch_inputs)].append(processing_time)
                
                # Split output back to individual results
                if isinstance(batch_output, torch.Tensor):
                    for j in range(batch_output.shape[0]):
                        results.append(batch_output[j])
                else:
                    results.extend(batch_output)
        
        return results


class MemoryOptimizer:
    """
    Advanced memory optimization and management.
    """
    
    def __init__(self, memory_threshold: float = 0.9):
        self.memory_threshold = memory_threshold
        self.gradient_checkpointing = False
        self.mixed_precision = False
        
        self.logger = logging.getLogger(__name__)
    
    def enable_gradient_checkpointing(self, model: nn.Module):
        """Enable gradient checkpointing to reduce memory usage."""
        def apply_gradient_checkpointing(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            
            for child in module.children():
                apply_gradient_checkpointing(child)
        
        apply_gradient_checkpointing(model)
        self.gradient_checkpointing = True
        self.logger.info("Gradient checkpointing enabled")
    
    def enable_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """Enable mixed precision training/inference."""
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            self.mixed_precision = True
            self.logger.info("Mixed precision enabled")
            return scaler
        else:
            self.logger.warning("Mixed precision not available (CUDA required)")
            return None
    
    def optimize_model_memory(self, model: nn.Module):
        """Apply various memory optimizations to model."""
        # Fuse operations where possible
        if hasattr(torch.jit, 'fuse'):
            try:
                model = torch.jit.fuse(model)
                self.logger.info("Operation fusion applied")
            except:
                pass
        
        # Convert to half precision if on GPU
        if torch.cuda.is_available() and next(model.parameters()).device.type == 'cuda':
            model = model.half()
            self.logger.info("Model converted to half precision")
        
        return model
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        # Python garbage collection
        gc.collect()
        
        # PyTorch memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Log memory status
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            self.logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        memory_info = {}
        
        # System memory
        sys_memory = psutil.virtual_memory()
        memory_info['system_used_gb'] = sys_memory.used / 1024**3
        memory_info['system_percent'] = sys_memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_cached_gb'] = torch.cuda.memory_reserved() / 1024**3
            memory_info['gpu_percent'] = (torch.cuda.memory_allocated() / 
                                        torch.cuda.get_device_properties(0).total_memory * 100)
        
        return memory_info


class PerformanceProfiler:
    """
    Comprehensive performance profiling and optimization recommendations.
    """
    
    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.recommendations = []
        self.logger = logging.getLogger(__name__)
    
    def profile_inference(self, model: nn.Module, inputs: Tuple[torch.Tensor, ...], 
                         num_runs: int = 100) -> PerformanceMetrics:
        """Profile model inference performance."""
        model.eval()
        
        inference_times = []
        memory_usages = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(*inputs)
        
        # Actual profiling runs
        with torch.no_grad():
            for _ in range(num_runs):
                # Clear cache and collect garbage
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Measure memory before
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                # Time inference
                start_time = time.time()
                output = model(*inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                
                # Measure memory after
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_usage = (memory_after - memory_before) / 1024**2  # MB
                
                inference_times.append(inference_time)
                memory_usages.append(memory_usage)
        
        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        avg_memory_usage = np.mean(memory_usages)
        latency_p95 = np.percentile(inference_times, 95)
        throughput = 1.0 / avg_inference_time
        
        # GPU utilization (approximate)
        gpu_utilization = min(100.0, (avg_memory_usage / 1000) * 50)  # Rough estimate
        
        return PerformanceMetrics(
            inference_time=avg_inference_time,
            memory_usage=avg_memory_usage,
            gpu_utilization=gpu_utilization,
            cache_hit_rate=0.0,  # Will be set by cache
            throughput=throughput,
            latency_p95=latency_p95,
            energy_consumption=0.0  # Would need specialized hardware to measure
        )
    
    def generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []
        
        # Inference time recommendations
        if metrics.inference_time > 0.1:  # 100ms
            recommendations.append("Consider model quantization or pruning to reduce inference time")
            recommendations.append("Enable TorchScript compilation for faster execution")
        
        # Memory usage recommendations
        if metrics.memory_usage > 1000:  # 1GB
            recommendations.append("Enable gradient checkpointing to reduce memory usage")
            recommendations.append("Consider using mixed precision training")
            recommendations.append("Implement model sharding for large models")
        
        # GPU utilization recommendations
        if metrics.gpu_utilization < 50:
            recommendations.append("Increase batch size to improve GPU utilization")
            recommendations.append("Consider using multiple GPU streams")
        elif metrics.gpu_utilization > 95:
            recommendations.append("Reduce batch size to prevent out-of-memory errors")
        
        # Latency recommendations
        if metrics.latency_p95 > metrics.inference_time * 2:
            recommendations.append("High latency variance detected - check for memory allocation issues")
            recommendations.append("Consider using tensor memory pools")
        
        # Throughput recommendations
        if metrics.throughput < 10:  # Less than 10 samples/second
            recommendations.append("Low throughput detected - consider model optimization")
            recommendations.append("Enable multi-threading for data loading")
        
        return recommendations


class HighPerformanceEngine:
    """
    Main performance optimization engine integrating all optimization components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.cache = IntelligentCache(
            max_size_gb=self.config.get('cache_size_gb', 2.0),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        self.jit_compiler = TensorJITCompiler()
        self.batch_processor = BatchProcessor(
            max_batch_size=self.config.get('max_batch_size', 32)
        )
        self.memory_optimizer = MemoryOptimizer()
        self.profiler = PerformanceProfiler()
        
        self.optimized_models = {}
        self.performance_history = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self, model: nn.Module, example_inputs: Tuple[torch.Tensor, ...],
                      optimization_level: str = "balanced") -> nn.Module:
        """
        Comprehensively optimize model for inference.
        
        Args:
            model: Model to optimize
            example_inputs: Example inputs for tracing/profiling
            optimization_level: "fast", "balanced", or "memory_efficient"
        """
        model_id = id(model)
        
        if model_id in self.optimized_models:
            return self.optimized_models[model_id]
        
        self.logger.info(f"Optimizing model with level: {optimization_level}")
        
        # 1. Profile baseline performance
        baseline_metrics = self.profiler.profile_inference(model, example_inputs)
        self.logger.info(f"Baseline performance - Time: {baseline_metrics.inference_time:.3f}s, "
                        f"Memory: {baseline_metrics.memory_usage:.1f}MB")
        
        # 2. Apply optimizations based on level
        optimized_model = model
        
        if optimization_level in ["fast", "balanced"]:
            # JIT compilation
            optimized_model = self.jit_compiler.compile_model(optimized_model, example_inputs)
            
            # Memory optimizations
            optimized_model = self.memory_optimizer.optimize_model_memory(optimized_model)
        
        if optimization_level in ["memory_efficient", "balanced"]:
            # Enable gradient checkpointing
            self.memory_optimizer.enable_gradient_checkpointing(optimized_model)
        
        if optimization_level == "fast":
            # Mixed precision for speed
            self.memory_optimizer.enable_mixed_precision()
        
        # 3. Profile optimized performance
        optimized_metrics = self.profiler.profile_inference(optimized_model, example_inputs)
        self.logger.info(f"Optimized performance - Time: {optimized_metrics.inference_time:.3f}s, "
                        f"Memory: {optimized_metrics.memory_usage:.1f}MB")
        
        # 4. Calculate improvement
        time_improvement = (baseline_metrics.inference_time - optimized_metrics.inference_time) / baseline_metrics.inference_time * 100
        memory_improvement = (baseline_metrics.memory_usage - optimized_metrics.memory_usage) / baseline_metrics.memory_usage * 100
        
        self.logger.info(f"Performance improvements - Time: {time_improvement:.1f}%, Memory: {memory_improvement:.1f}%")
        
        # 5. Generate and log recommendations
        recommendations = self.profiler.generate_recommendations(optimized_metrics)
        for rec in recommendations:
            self.logger.info(f"Recommendation: {rec}")
        
        # Cache optimized model
        self.optimized_models[model_id] = optimized_model
        
        return optimized_model
    
    def cached_inference(self, model: nn.Module, audio_input: torch.Tensor,
                        video_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform inference with intelligent caching.
        """
        # Check cache first
        cache_params = {
            'model_id': id(model),
            'audio_shape': audio_input.shape,
            'video_shape': video_input.shape,
            **kwargs
        }
        
        # Try to get from cache
        cached_result = self.cache.get(audio_input, cache_params)
        if cached_result is not None:
            return cached_result
        
        # Perform inference
        model.eval()
        with torch.no_grad():
            result = model(audio_input, video_input)
        
        # Cache result
        self.cache.put(audio_input, cache_params, result)
        
        return result
    
    def batch_inference(self, model: nn.Module, 
                       inputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Perform optimized batch inference.
        """
        return self.batch_processor.process_batch(model, inputs)
    
    async def async_inference(self, model: nn.Module, audio_input: torch.Tensor,
                             video_input: torch.Tensor) -> torch.Tensor:
        """
        Asynchronous inference for concurrent processing.
        """
        loop = asyncio.get_event_loop()
        
        # Run inference in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor, 
                self.cached_inference, 
                model, audio_input, video_input
            )
            result = await future
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        memory_info = self.memory_optimizer.monitor_memory_usage()
        cache_stats = {
            'hit_rate': self.cache.get_hit_rate(),
            'cache_size': len(self.cache.cache),
            'hit_count': self.cache.hit_count,
            'miss_count': self.cache.miss_count
        }
        
        return {
            'memory_info': memory_info,
            'cache_stats': cache_stats,
            'optimized_models': len(self.optimized_models),
            'optimal_batch_sizes': dict(self.batch_processor.optimal_batch_sizes)
        }
    
    def cleanup(self):
        """Perform comprehensive cleanup."""
        self.cache.clear()
        self.memory_optimizer.cleanup_memory()
        self.optimized_models.clear()
        
        self.logger.info("Performance engine cleanup completed")


# Global performance engine instance
global_performance_engine = HighPerformanceEngine()


def performance_optimized(optimization_level: str = "balanced"):
    """
    Decorator for automatic performance optimization of inference functions.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract model and inputs
            if len(args) >= 3:
                model, audio_input, video_input = args[0], args[1], args[2]
                
                # Optimize model if not already optimized
                optimized_model = global_performance_engine.optimize_model(
                    model, (audio_input, video_input), optimization_level
                )
                
                # Use cached inference
                return global_performance_engine.cached_inference(
                    optimized_model, audio_input, video_input, **kwargs
                )
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Create example model and inputs
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 256)
        
        def forward(self, audio, video):
            return self.linear(audio + video)
    
    model = ExampleModel()
    audio = torch.randn(4, 512)
    video = torch.randn(4, 512)
    
    # Optimize model
    engine = HighPerformanceEngine()
    optimized_model = engine.optimize_model(model, (audio, video))
    
    # Perform cached inference
    result = engine.cached_inference(optimized_model, audio, video)
    
    # Get performance report
    report = engine.get_performance_report()
    print(f"Performance report: {report}")