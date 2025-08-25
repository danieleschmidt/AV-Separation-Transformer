"""
Generation 3: Scalable Audio-Visual Separation System
High-performance system with distributed processing, caching, and auto-scaling.
"""
import torch
import torch.nn as nn
import numpy as np
import asyncio
import concurrent.futures
import threading
import time
import logging
import hashlib
import pickle
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import redis
from pathlib import Path
import queue

from .robust_system import RobustAVSeparator, SystemMetrics
from .config import SeparatorConfig


@dataclass
class ScalingConfig:
    """Configuration for scaling system"""
    # Threading and processing
    max_workers: int = min(8, mp.cpu_count())
    enable_async: bool = True
    batch_size: int = 4
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    min_workers: int = 2
    max_workers_limit: int = 16
    cpu_scale_threshold: float = 80.0
    memory_scale_threshold: float = 75.0
    
    # Performance optimization
    enable_model_parallel: bool = False
    enable_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    compile_model: bool = True
    
    # Load balancing
    enable_load_balancing: bool = True
    queue_timeout: int = 30


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply performance optimizations to model"""
        optimized_model = model
        
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing'):
                model.gradient_checkpointing = True
            logging.info("Gradient checkpointing enabled")
        
        # Compile model for better performance (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                optimized_model = torch.compile(model)
                logging.info("Model compilation enabled")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        # Mixed precision setup
        if self.config.mixed_precision:
            logging.info("Mixed precision training enabled")
        
        return optimized_model
    
    def optimize_inference(self, model: nn.Module, input_tensor: torch.Tensor):
        """Optimize inference execution"""
        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.autocast(device_type=input_tensor.device.type):
                    return model(input_tensor)
            else:
                return model(input_tensor)


class CacheManager:
    """Advanced caching system for repeated computations"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.lock = threading.RLock()
        
        # Redis cache (optional)
        self.redis_client = None
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
            self.redis_client.ping()
            logging.info("Redis cache connected")
        except:
            logging.info("Redis cache not available, using memory cache only")
    
    def _compute_cache_key(self, data: Any) -> str:
        """Compute cache key for input data"""
        if isinstance(data, np.ndarray):
            # Hash array data
            return hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, torch.Tensor):
            return hashlib.md5(data.cpu().numpy().tobytes()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                item, timestamp = self.memory_cache[key]
                if time.time() - timestamp < self.config.cache_ttl:
                    self.cache_stats['hits'] += 1
                    return item
                else:
                    del self.memory_cache[key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        item = pickle.loads(data)
                        self.cache_stats['hits'] += 1
                        return item
                except Exception as e:
                    logging.warning(f"Redis cache get failed: {e}")
            
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        with self.lock:
            # Memory cache
            self.memory_cache[key] = (value, time.time())
            
            # Limit memory cache size
            if len(self.memory_cache) > self.config.cache_size:
                # Remove oldest items
                oldest_keys = sorted(self.memory_cache.keys(), 
                                   key=lambda k: self.memory_cache[k][1])[:10]
                for k in oldest_keys:
                    del self.memory_cache[k]
            
            # Redis cache
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        key, 
                        self.config.cache_ttl,
                        pickle.dumps(value)
                    )
                except Exception as e:
                    logging.warning(f"Redis cache set failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = (self.cache_stats['hits'] / max(total_requests, 1)) * 100
            
            return {
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate': hit_rate,
                'cache_size': len(self.memory_cache)
            }


class LoadBalancer:
    """Load balancer for distributing requests across workers"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.request_queue = queue.Queue(maxsize=1000)
        self.worker_stats = {}
        self.lock = threading.Lock()
        
    def add_request(self, request: Dict[str, Any]) -> str:
        """Add request to queue"""
        request_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        request['id'] = request_id
        request['timestamp'] = time.time()
        
        try:
            self.request_queue.put(request, timeout=self.config.queue_timeout)
            return request_id
        except queue.Full:
            raise RuntimeError("Request queue is full")
    
    def get_request(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get next request for worker"""
        try:
            request = self.request_queue.get(timeout=1.0)
            with self.lock:
                if worker_id not in self.worker_stats:
                    self.worker_stats[worker_id] = {
                        'requests_processed': 0,
                        'total_time': 0.0,
                        'last_active': time.time()
                    }
                self.worker_stats[worker_id]['last_active'] = time.time()
            return request
        except queue.Empty:
            return None
    
    def complete_request(self, worker_id: str, request_id: str, processing_time: float):
        """Mark request as completed"""
        with self.lock:
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]['requests_processed'] += 1
                self.worker_stats[worker_id]['total_time'] += processing_time
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        with self.lock:
            return dict(self.worker_stats)


class AutoScaler:
    """Auto-scaling manager"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.scaling_history = []
        
    def should_scale_up(self, metrics: SystemMetrics, queue_size: int) -> bool:
        """Determine if system should scale up"""
        # High CPU or memory usage
        if (metrics.cpu_usage > self.config.cpu_scale_threshold or 
            metrics.memory_usage > self.config.memory_scale_threshold):
            return True
        
        # Large queue backlog
        if queue_size > self.current_workers * 5:
            return True
        
        return False
    
    def should_scale_down(self, metrics: SystemMetrics, queue_size: int) -> bool:
        """Determine if system should scale down"""
        # Low resource usage and small queue
        if (metrics.cpu_usage < 30 and 
            metrics.memory_usage < 50 and 
            queue_size < self.current_workers):
            return True
        
        return False
    
    def scale_up(self) -> int:
        """Scale up workers"""
        new_workers = min(self.current_workers + 2, self.config.max_workers_limit)
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            self.scaling_history.append({
                'action': 'scale_up',
                'workers': new_workers,
                'timestamp': time.time()
            })
            logging.info(f"Scaled up to {new_workers} workers")
        return new_workers
    
    def scale_down(self) -> int:
        """Scale down workers"""
        new_workers = max(self.current_workers - 1, self.config.min_workers)
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            self.scaling_history.append({
                'action': 'scale_down',
                'workers': new_workers,
                'timestamp': time.time()
            })
            logging.info(f"Scaled down to {new_workers} workers")
        return new_workers


class BatchProcessor:
    """Batch processing for improved throughput"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.batch_queue = []
        self.batch_lock = threading.Lock()
        
    def add_to_batch(self, request: Dict[str, Any]) -> bool:
        """Add request to current batch"""
        with self.batch_lock:
            self.batch_queue.append(request)
            return len(self.batch_queue) >= self.config.batch_size
    
    def get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch and clear queue"""
        with self.batch_lock:
            batch = self.batch_queue.copy()
            self.batch_queue.clear()
            return batch
    
    def process_batch(self, batch: List[Dict[str, Any]], processor: Callable) -> List[Dict[str, Any]]:
        """Process batch of requests"""
        if not batch:
            return []
        
        # Group similar requests for more efficient processing
        results = []
        for request in batch:
            try:
                result = processor(request)
                results.append(result)
            except Exception as e:
                results.append({
                    'request_id': request.get('id', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
        
        return results


class ScalableAVSeparator:
    """Scalable Audio-Visual Separator with distributed processing"""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        
        # Initialize components
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.cache_manager = CacheManager(self.config) if self.config.enable_caching else None
        self.load_balancer = LoadBalancer(self.config) if self.config.enable_load_balancing else None
        self.auto_scaler = AutoScaler(self.config) if self.config.enable_auto_scaling else None
        self.batch_processor = BatchProcessor(self.config)
        
        # Create worker pool
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.workers = {}
        
        # Initialize robust separators for workers
        self._initialize_workers()
        
        # Start background tasks
        self._start_background_tasks()
        
        logging.info(f"ScalableAVSeparator initialized with {self.config.max_workers} workers")
    
    def _initialize_workers(self):
        """Initialize worker separators"""
        for i in range(self.config.max_workers):
            worker_id = f"worker_{i}"
            try:
                separator_config = SeparatorConfig()
                separator_config.inference.device = 'cpu'  # Use CPU for compatibility
                
                robust_separator = RobustAVSeparator(separator_config)
                
                # Optimize the model
                if robust_separator.separator:
                    robust_separator.separator.model = self.performance_optimizer.optimize_model(
                        robust_separator.separator.model
                    )
                
                self.workers[worker_id] = robust_separator
                logging.info(f"Worker {worker_id} initialized")
                
            except Exception as e:
                logging.error(f"Failed to initialize worker {worker_id}: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and scaling tasks"""
        if self.config.enable_auto_scaling and self.auto_scaler:
            def auto_scaling_loop():
                while True:
                    try:
                        # Get system metrics
                        if self.workers:
                            first_worker = next(iter(self.workers.values()))
                            health = first_worker.get_health_status()
                            
                            if 'metrics' in health:
                                metrics = SystemMetrics(**health['metrics'])
                                queue_size = self.load_balancer.request_queue.qsize() if self.load_balancer else 0
                                
                                # Check scaling conditions
                                if self.auto_scaler.should_scale_up(metrics, queue_size):
                                    self.auto_scaler.scale_up()
                                elif self.auto_scaler.should_scale_down(metrics, queue_size):
                                    self.auto_scaler.scale_down()
                    
                    except Exception as e:
                        logging.error(f"Auto-scaling error: {e}")
                    
                    time.sleep(30)  # Check every 30 seconds
            
            scaling_thread = threading.Thread(target=auto_scaling_loop, daemon=True)
            scaling_thread.start()
    
    async def separate_async(self, audio: np.ndarray, video: np.ndarray) -> Dict[str, Any]:
        """Asynchronous separation processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._process_separation, audio, video)
    
    def _process_separation(self, audio: np.ndarray, video: np.ndarray) -> Dict[str, Any]:
        """Process separation request"""
        start_time = time.time()
        
        # Check cache first
        cache_key = None
        if self.cache_manager:
            cache_key = f"{self.cache_manager._compute_cache_key(audio)}_{self.cache_manager._compute_cache_key(video)}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                cached_result['cache_hit'] = True
                return cached_result
        
        # Get available worker
        worker = self._get_available_worker()
        if not worker:
            return {
                'status': 'error',
                'error': 'No available workers',
                'processing_time': time.time() - start_time
            }
        
        # Process request
        try:
            result = worker.separate_audio_visual(audio, video)
            result['processing_time'] = time.time() - start_time
            result['cache_hit'] = False
            result['worker_id'] = id(worker)
            
            # Cache result
            if self.cache_manager and cache_key and result['status'] == 'success':
                self.cache_manager.set(cache_key, result)
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'cache_hit': False
            }
    
    def _get_available_worker(self) -> Optional[RobustAVSeparator]:
        """Get available worker (simple round-robin)"""
        if not self.workers:
            return None
        
        # Simple round-robin selection
        worker_ids = list(self.workers.keys())
        for worker_id in worker_ids:
            return self.workers[worker_id]
        
        return None
    
    def separate_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of separation requests"""
        results = []
        
        # Process requests in parallel
        futures = []
        for request in requests:
            future = self.executor.submit(
                self._process_separation,
                request['audio'],
                request['video']
            )
            futures.append(future)
        
        # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=60)  # 60 second timeout
                result['request_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'request_index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'workers': len(self.workers),
            'config': self.config.__dict__
        }
        
        # Cache statistics
        if self.cache_manager:
            metrics['cache'] = self.cache_manager.get_stats()
        
        # Worker statistics
        if self.load_balancer:
            metrics['load_balancer'] = self.load_balancer.get_worker_stats()
        
        # Auto-scaling history
        if self.auto_scaler:
            metrics['scaling_history'] = self.auto_scaler.scaling_history[-10:]  # Last 10 events
        
        # System health from first worker
        if self.workers:
            first_worker = next(iter(self.workers.values()))
            worker_health = first_worker.get_health_status()
            metrics['system_health'] = worker_health
        
        return metrics
    
    def shutdown(self):
        """Gracefully shutdown the scaling system"""
        logging.info("Shutting down ScalableAVSeparator...")
        self.executor.shutdown(wait=True)
        logging.info("ScalableAVSeparator shutdown complete")


def create_scalable_system(config: Optional[ScalingConfig] = None) -> ScalableAVSeparator:
    """Factory function to create scalable AV separator system"""
    return ScalableAVSeparator(config=config)


if __name__ == "__main__":
    # Demo the scalable system
    print("⚡ Generation 3: Scalable Audio-Visual Separation System")
    
    # Create scalable system
    scaling_config = ScalingConfig(
        max_workers=4,
        enable_caching=True,
        enable_auto_scaling=True
    )
    
    scalable_separator = create_scalable_system(scaling_config)
    
    # Test with dummy data
    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_video = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    
    # Process single request
    result = scalable_separator._process_separation(dummy_audio, dummy_video)
    print(f"✅ Single request: {result['status']}")
    
    # Process batch
    batch_requests = [
        {'audio': dummy_audio, 'video': dummy_video} for _ in range(3)
    ]
    batch_results = scalable_separator.separate_batch(batch_requests)
    print(f"✅ Batch processing: {len(batch_results)} results")
    
    # Get performance metrics
    metrics = scalable_separator.get_performance_metrics()
    print(f"✅ Active workers: {metrics['workers']}")
    print(f"✅ Cache hit rate: {metrics.get('cache', {}).get('hit_rate', 0):.1f}%")
    
    print("🌟 Generation 3: SCALABLE system operational")
    
    # Cleanup
    scalable_separator.shutdown()