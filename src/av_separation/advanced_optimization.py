import torch
import torch.nn as nn
import torch.jit
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import logging
from queue import Queue, Empty
from collections import deque, defaultdict
import psutil
import asyncio
from functools import lru_cache, wraps
import pickle
import hashlib
from pathlib import Path
import json
import gc
from contextlib import contextmanager
import resource

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings."""
    enable_compilation: bool = True
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_tensorrt: bool = TRT_AVAILABLE
    enable_onnx: bool = ONNX_AVAILABLE
    cache_size_mb: int = 1024
    batch_optimization: bool = True
    async_processing: bool = True
    memory_pool_size_mb: int = 2048
    enable_gpu_memory_pooling: bool = True
    

class AdvancedOptimizer:
    """Advanced model optimization system."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance monitoring
        self.performance_metrics = {
            'inference_times': deque(maxlen=1000),
            'throughput_fps': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_applied': []
        }
        
        # Caching system
        self.inference_cache = InferenceCache(self.config.cache_size_mb)
        
        # Memory management
        self.memory_manager = MemoryManager(self.config.memory_pool_size_mb)
        
        # Async processing
        if self.config.async_processing:
            self.async_executor = ThreadPoolExecutor(max_workers=4)
        
        # Optimization registry
        self.optimization_registry = {}
        
    def optimize_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply comprehensive model optimizations."""
        optimized_model = model
        
        # 1. Torch JIT Compilation
        if self.config.enable_compilation:
            optimized_model = self._apply_torch_jit(optimized_model, sample_input)
        
        # 2. Quantization
        if self.config.enable_quantization:
            optimized_model = self._apply_quantization(optimized_model)
        
        # 3. Pruning
        if self.config.enable_pruning:
            optimized_model = self._apply_pruning(optimized_model)
        
        # 4. TensorRT optimization
        if self.config.enable_tensorrt and TRT_AVAILABLE:
            optimized_model = self._apply_tensorrt(optimized_model, sample_input)
        
        # 5. ONNX conversion for inference
        if self.config.enable_onnx and ONNX_AVAILABLE:
            self._prepare_onnx_model(optimized_model, sample_input)
        
        self.logger.info(f"Applied optimizations: {self.performance_metrics['optimizations_applied']}")
        return optimized_model
    
    def _apply_torch_jit(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply Torch JIT compilation."""
        try:
            model.eval()
            with torch.no_grad():
                traced_model = torch.jit.trace(model, sample_input)
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            self.performance_metrics['optimizations_applied'].append('torch_jit')
            self.logger.info("Applied Torch JIT compilation")
            return traced_model
        
        except Exception as e:
            self.logger.warning(f"Torch JIT compilation failed: {e}")
            return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            self.performance_metrics['optimizations_applied'].append('quantization')
            self.logger.info("Applied dynamic quantization")
            return quantized_model
        
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply structured pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply pruning to linear layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
            
            self.performance_metrics['optimizations_applied'].append('pruning')
            self.logger.info(f"Applied {sparsity*100}% pruning")
            return model
        
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def _apply_tensorrt(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply TensorRT optimization."""
        if not TRT_AVAILABLE:
            self.logger.warning("TensorRT not available")
            return model
        
        try:
            import torch_tensorrt
            
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=[sample_input],
                enabled_precisions={torch.float, torch.half},
                workspace_size=1 << 30,  # 1GB
                max_batch_size=8
            )
            
            self.performance_metrics['optimizations_applied'].append('tensorrt')
            self.logger.info("Applied TensorRT optimization")
            return compiled_model
        
        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
            return model
    
    def _prepare_onnx_model(self, model: nn.Module, sample_input: torch.Tensor):
        """Convert model to ONNX for optimized inference."""
        if not ONNX_AVAILABLE:
            return
        
        try:
            onnx_path = Path("optimized_model.onnx")
            
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Create optimized ONNX runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            self.onnx_session = session
            self.performance_metrics['optimizations_applied'].append('onnx')
            self.logger.info("Prepared ONNX model for inference")
            
        except Exception as e:
            self.logger.warning(f"ONNX conversion failed: {e}")
    
    @contextmanager
    def optimized_inference_context(self):
        """Context manager for optimized inference."""
        # Pre-allocate memory
        self.memory_manager.prepare_inference_memory()
        
        # Set optimal thread count
        original_threads = torch.get_num_threads()
        optimal_threads = min(psutil.cpu_count(), 8)
        torch.set_num_threads(optimal_threads)
        
        # Enable optimizations
        with torch.inference_mode():
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                yield
        
        # Cleanup
        torch.set_num_threads(original_threads)
        self.memory_manager.cleanup_inference_memory()
    
    def cached_inference(self, model: nn.Module, input_data: torch.Tensor, cache_key: str = None) -> torch.Tensor:
        """Perform cached inference."""
        
        if cache_key is None:
            cache_key = self._generate_cache_key(input_data)
        
        # Check cache
        cached_result = self.inference_cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return cached_result
        
        self.performance_metrics['cache_misses'] += 1
        
        # Perform inference
        start_time = time.time()
        
        with self.optimized_inference_context():
            if hasattr(self, 'onnx_session') and self.config.enable_onnx:
                result = self._onnx_inference(input_data)
            else:
                result = model(input_data)
        
        inference_time = time.time() - start_time
        self.performance_metrics['inference_times'].append(inference_time)
        
        # Cache result
        self.inference_cache.put(cache_key, result.detach().clone())
        
        return result
    
    def _onnx_inference(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform ONNX inference."""
        input_np = input_data.cpu().numpy()
        outputs = self.onnx_session.run(None, {'input': input_np})
        return torch.from_numpy(outputs[0]).to(input_data.device)
    
    def _generate_cache_key(self, input_data: torch.Tensor) -> str:
        """Generate cache key for input data."""
        # Create hash of tensor properties and first/last elements
        key_data = {
            'shape': tuple(input_data.shape),
            'dtype': str(input_data.dtype),
            'device': str(input_data.device),
            'first_elements': input_data.flatten()[:10].tolist(),
            'last_elements': input_data.flatten()[-10:].tolist(),
            'mean': float(input_data.mean()),
            'std': float(input_data.std())
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def async_batch_inference(
        self, 
        model: nn.Module, 
        batch_inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Perform asynchronous batch inference."""
        if not self.config.async_processing:
            return [self.cached_inference(model, inp) for inp in batch_inputs]
        
        # Submit inference tasks
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.async_executor, self.cached_inference, model, inp)
            for inp in batch_inputs
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        inference_times = list(self.performance_metrics['inference_times'])
        
        report = {
            'optimizations_applied': self.performance_metrics['optimizations_applied'],
            'inference_statistics': {
                'total_inferences': len(inference_times),
                'average_time_ms': np.mean(inference_times) * 1000 if inference_times else 0,
                'p50_time_ms': np.percentile(inference_times, 50) * 1000 if inference_times else 0,
                'p95_time_ms': np.percentile(inference_times, 95) * 1000 if inference_times else 0,
                'p99_time_ms': np.percentile(inference_times, 99) * 1000 if inference_times else 0,
            },
            'cache_statistics': {
                'cache_hits': self.performance_metrics['cache_hits'],
                'cache_misses': self.performance_metrics['cache_misses'],
                'hit_rate': (
                    self.performance_metrics['cache_hits'] / 
                    max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                ),
                'cache_size_mb': self.inference_cache.get_size_mb()
            },
            'memory_statistics': self.memory_manager.get_stats()
        }
        
        return report


class InferenceCache:
    """High-performance inference cache."""
    
    def __init__(self, max_size_mb: int):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.current_size = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached result."""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key].clone()
            return None
    
    def put(self, key: str, value: torch.Tensor):
        """Cache inference result."""
        with self.lock:
            # Calculate size
            item_size = value.numel() * value.element_size()
            
            # Evict if necessary
            while self.current_size + item_size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            # Store item
            self.cache[key] = value.clone()
            self.access_times[key] = time.time()
            self.current_size += item_size
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        item = self.cache.pop(lru_key)
        del self.access_times[lru_key]
        
        item_size = item.numel() * item.element_size()
        self.current_size -= item_size
    
    def get_size_mb(self) -> float:
        """Get current cache size in MB."""
        with self.lock:
            return self.current_size / (1024 * 1024)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size = 0


class MemoryManager:
    """Advanced memory management system."""
    
    def __init__(self, pool_size_mb: int):
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.memory_pools = {}
        self.allocation_stats = defaultdict(int)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if torch.cuda.is_available() and CUPY_AVAILABLE:
            self._setup_gpu_memory_pool()
    
    def _setup_gpu_memory_pool(self):
        """Setup GPU memory pool using CuPy."""
        try:
            # Set up CuPy memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=self.pool_size_bytes)
            
            self.gpu_memory_pool = mempool
            self.logger.info(f"Setup GPU memory pool: {self.pool_size_bytes // (1024**2)}MB")
        except Exception as e:
            self.logger.warning(f"GPU memory pool setup failed: {e}")
    
    def prepare_inference_memory(self):
        """Pre-allocate memory for inference."""
        try:
            # Pre-allocate common tensor sizes
            common_shapes = [
                (1, 512, 128),   # Audio features
                (1, 256, 64, 64), # Video features
                (1, 1024),       # Embeddings
            ]
            
            self.preallocated_tensors = {}
            for shape in common_shapes:
                if torch.cuda.is_available():
                    tensor = torch.empty(shape, device='cuda')
                else:
                    tensor = torch.empty(shape)
                self.preallocated_tensors[shape] = tensor
                
        except Exception as e:
            self.logger.warning(f"Memory pre-allocation failed: {e}")
    
    def cleanup_inference_memory(self):
        """Clean up inference memory."""
        try:
            # Clear pre-allocated tensors
            if hasattr(self, 'preallocated_tensors'):
                self.preallocated_tensors.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'system_memory': {
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            stats['gpu_memory'] = {
                'allocated_gb': gpu_memory['allocated_bytes.all.current'] / (1024**3),
                'reserved_gb': gpu_memory['reserved_bytes.all.current'] / (1024**3),
                'max_allocated_gb': gpu_memory['allocated_bytes.all.peak'] / (1024**3)
            }
        
        return stats


class AdaptiveScaling:
    """Intelligent auto-scaling system."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        # Metrics for scaling decisions
        self.request_queue = Queue()
        self.response_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=20)
        
        # Scaling thresholds
        self.scale_up_threshold_ms = 100  # Average response time
        self.scale_down_threshold_ms = 50
        self.queue_threshold = 10  # Queue length for immediate scaling
        
        self.scaling_lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up."""
        if self.current_workers >= self.max_workers:
            return False
        
        # Check response times
        if len(self.response_times) >= 10:
            avg_response = np.mean(list(self.response_times)[-10:])
            if avg_response > self.scale_up_threshold_ms / 1000:
                return True
        
        # Check queue length
        if self.request_queue.qsize() > self.queue_threshold:
            return True
        
        return False
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down."""
        if self.current_workers <= self.min_workers:
            return False
        
        # Check response times
        if len(self.response_times) >= 20:
            avg_response = np.mean(list(self.response_times)[-20:])
            if avg_response < self.scale_down_threshold_ms / 1000:
                # Also check that queue is not growing
                if self.request_queue.qsize() < 2:
                    return True
        
        return False
    
    def scale_up(self):
        """Scale up workers."""
        with self.scaling_lock:
            if self.current_workers < self.max_workers:
                self.current_workers += 1
                self.logger.info(f"Scaled up to {self.current_workers} workers")
    
    def scale_down(self):
        """Scale down workers."""
        with self.scaling_lock:
            if self.current_workers > self.min_workers:
                self.current_workers -= 1
                self.logger.info(f"Scaled down to {self.current_workers} workers")
    
    def record_request_metrics(self, response_time: float):
        """Record request metrics for scaling decisions."""
        self.response_times.append(response_time)
        
        # Calculate throughput
        current_time = time.time()
        if len(self.throughput_history) == 0:
            self.throughput_history.append((current_time, 1))
        else:
            last_time, last_count = self.throughput_history[-1]
            if current_time - last_time >= 1.0:  # New second
                self.throughput_history.append((current_time, 1))
            else:
                self.throughput_history[-1] = (last_time, last_count + 1)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics and recommendations."""
        metrics = {
            'current_workers': self.current_workers,
            'queue_size': self.request_queue.qsize(),
            'should_scale_up': self.should_scale_up(),
            'should_scale_down': self.should_scale_down()
        }
        
        if self.response_times:
            metrics['avg_response_time_ms'] = np.mean(list(self.response_times)) * 1000
            metrics['p95_response_time_ms'] = np.percentile(list(self.response_times), 95) * 1000
        
        if len(self.throughput_history) >= 2:
            recent_throughput = [count for _, count in list(self.throughput_history)[-10:]]
            metrics['avg_throughput_rps'] = np.mean(recent_throughput)
        
        return metrics


@contextmanager
def nullcontext():
    """Null context manager for Python < 3.7 compatibility."""
    yield


def performance_profiler(func):
    """Decorator for performance profiling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger = logging.getLogger('performance_profiler')
            logger.info(
                f"{func.__name__}: {execution_time:.3f}s, "
                f"memory_delta: {memory_delta / (1024**2):.1f}MB"
            )
            
            return result
            
        except Exception as e:
            logger = logging.getLogger('performance_profiler')
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper
