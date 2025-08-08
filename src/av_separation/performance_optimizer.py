"""
Advanced Performance Optimization System
Generation 3: MAKE IT SCALE - High-performance computing optimizations
"""

import asyncio
import time
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import heapq
import hashlib
import json
import weakref
import gc
from contextlib import contextmanager
from functools import lru_cache, wraps

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

# Performance metrics
@dataclass 
class PerformanceMetrics:
    """Comprehensive performance metrics tracking"""
    component: str
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'operation': self.operation,
            'duration': self.duration,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'timestamp': self.timestamp
        }


class AdvancedCache:
    """High-performance caching with LRU, TTL, and memory management"""
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_memory_mb: int = 256,
        ttl_seconds: int = 3600,
        eviction_policy: str = "lru"
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.memory_usage = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache with hit/miss tracking"""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if key in self.cache:
                if key in self.expiry_times and current_time > self.expiry_times[key]:
                    self._remove_key(key)
                    self.miss_count += 1
                    return None, False
                
                # Update access time for LRU
                self.access_times[key] = current_time
                self.hit_count += 1
                return self.cache[key], True
            
            self.miss_count += 1
            return None, False
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put item in cache with size and memory management"""
        with self._lock:
            current_time = time.time()
            
            # Calculate value size
            value_size = self._calculate_size(value)
            
            # Check memory constraints
            if value_size > self.max_memory_bytes:
                return False  # Value too large
            
            # Remove existing key if present
            if key in self.cache:
                self._remove_key(key)
            
            # Ensure cache has space
            while (len(self.cache) >= self.max_size or 
                   self.memory_usage + value_size > self.max_memory_bytes):
                if not self._evict_one():
                    return False  # Could not make space
            
            # Add to cache
            self.cache[key] = value
            self.access_times[key] = current_time
            self.memory_usage += value_size
            
            # Set expiry
            ttl = ttl or self.ttl_seconds
            self.expiry_times[key] = current_time + ttl
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache"""
        with self._lock:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
            self.memory_usage = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count
            }
    
    def _remove_key(self, key: str) -> None:
        """Remove key and update memory usage"""
        if key in self.cache:
            value_size = self._calculate_size(self.cache[key])
            del self.cache[key]
            self.access_times.pop(key, None)
            self.expiry_times.pop(key, None)
            self.memory_usage -= value_size
    
    def _evict_one(self) -> bool:
        """Evict one item based on policy"""
        if not self.cache:
            return False
            
        current_time = time.time()
        
        # First, try to evict expired items
        for key in list(self.cache.keys()):
            if key in self.expiry_times and current_time > self.expiry_times[key]:
                self._remove_key(key)
                self.eviction_count += 1
                return True
        
        # Then evict based on policy
        if self.eviction_policy == "lru":
            # Evict least recently used
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            self._remove_key(oldest_key)
            self.eviction_count += 1
            return True
        
        return False
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, (list, tuple)):
            return sum(self._calculate_size(item) for item in value) + 64
        elif isinstance(value, dict):
            return sum(self._calculate_size(k) + self._calculate_size(v) 
                      for k, v in value.items()) + 64
        elif _numpy_available and isinstance(value, np.ndarray):
            return value.nbytes
        else:
            # Fallback: use string representation length
            return len(str(value))


class BatchProcessor:
    """High-performance batch processing with dynamic batching"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_queue_size: int = 1000,
        num_workers: int = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())
        
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.results = {}
        self.running = False
        self.worker_tasks = []
        
        # Performance metrics
        self.processed_batches = 0
        self.total_items = 0
        self.total_time = 0.0
        self.error_count = 0
        
    async def start(self):
        """Start batch processing workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start batch collector
        collector_task = asyncio.create_task(self._batch_collector())
        self.worker_tasks.append(collector_task)
        
    async def stop(self):
        """Stop batch processing"""
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
    
    async def process_async(
        self, 
        item: Any, 
        processor_func: Callable,
        timeout: float = 30.0
    ) -> Any:
        """Submit item for batch processing"""
        if not self.running:
            await self.start()
        
        # Create result future
        item_id = id(item)
        future = asyncio.Future()
        self.results[item_id] = future
        
        # Submit to queue
        try:
            await asyncio.wait_for(
                self.queue.put((item, item_id, processor_func)), 
                timeout=1.0
            )
        except asyncio.TimeoutError:
            self.results.pop(item_id, None)
            raise Exception("Batch queue full")
        
        # Wait for result
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.results.pop(item_id, None)
            raise Exception("Batch processing timeout")
        finally:
            self.results.pop(item_id, None)
    
    async def _batch_collector(self):
        """Collect items into batches and process"""
        current_batch = []
        batch_functions = {}
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Collect items for batch
                while (len(current_batch) < self.batch_size and 
                       time.time() - last_batch_time < self.max_wait_time):
                    try:
                        item, item_id, func = await asyncio.wait_for(
                            self.queue.get(), timeout=0.01
                        )
                        current_batch.append((item, item_id))
                        
                        # Group by function
                        func_id = id(func)
                        if func_id not in batch_functions:
                            batch_functions[func_id] = (func, [])
                        batch_functions[func_id][1].append((item, item_id))
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batches if we have items
                if current_batch:
                    await self._process_batches(batch_functions)
                    current_batch.clear()
                    batch_functions.clear()
                    last_batch_time = time.time()
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception as e:
                self.error_count += 1
                # Handle errors gracefully
                for _, item_id in current_batch:
                    if item_id in self.results and not self.results[item_id].done():
                        self.results[item_id].set_exception(e)
                
                current_batch.clear()
                batch_functions.clear()
    
    async def _process_batches(self, batch_functions: Dict[int, Tuple[Callable, List]]):
        """Process all batches"""
        start_time = time.time()
        
        try:
            # Process each function's batch
            for func_id, (func, items) in batch_functions.items():
                batch_items = [item for item, _ in items]
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        results = await func(batch_items)
                    else:
                        # Run in executor for CPU-bound functions
                        loop = asyncio.get_event_loop()
                        results = await loop.run_in_executor(None, func, batch_items)
                    
                    # Set results
                    for (_, item_id), result in zip(items, results):
                        if item_id in self.results and not self.results[item_id].done():
                            self.results[item_id].set_result(result)
                
                except Exception as e:
                    # Set exceptions for all items in batch
                    for _, item_id in items:
                        if item_id in self.results and not self.results[item_id].done():
                            self.results[item_id].set_exception(e)
            
            # Update metrics
            self.processed_batches += len(batch_functions)
            self.total_items += sum(len(items) for _, (_, items) in batch_functions.items())
            self.total_time += time.time() - start_time
            
        except Exception as e:
            # Handle processing errors
            for func_id, (func, items) in batch_functions.items():
                for _, item_id in items:
                    if item_id in self.results and not self.results[item_id].done():
                        self.results[item_id].set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        avg_batch_time = self.total_time / self.processed_batches if self.processed_batches > 0 else 0
        throughput = self.total_items / self.total_time if self.total_time > 0 else 0
        
        return {
            'processed_batches': self.processed_batches,
            'total_items': self.total_items,
            'total_time': self.total_time,
            'avg_batch_time': avg_batch_time,
            'throughput_items_per_sec': throughput,
            'error_count': self.error_count,
            'queue_size': self.queue.qsize(),
            'active_results': len(self.results)
        }


class ConnectionPool:
    """High-performance connection pooling for database/API connections"""
    
    def __init__(
        self,
        create_connection: Callable,
        max_connections: int = 20,
        min_connections: int = 5,
        max_idle_time: int = 300,
        connection_timeout: int = 30
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        
        self.pool = deque()
        self.active_connections = set()
        self.connection_times = {}
        self.lock = asyncio.Lock()
        
        # Statistics
        self.total_created = 0
        self.total_reused = 0
        self.total_closed = 0
        
    async def get_connection(self):
        """Get connection from pool or create new one"""
        async with self.lock:
            current_time = time.time()
            
            # Remove expired connections
            while self.pool:
                conn = self.pool[0]
                if current_time - self.connection_times.get(id(conn), 0) > self.max_idle_time:
                    self.pool.popleft()
                    await self._close_connection(conn)
                else:
                    break
            
            # Try to reuse existing connection
            if self.pool:
                conn = self.pool.popleft()
                self.active_connections.add(conn)
                self.total_reused += 1
                return conn
            
            # Create new connection if under limit
            if len(self.active_connections) < self.max_connections:
                conn = await self._create_connection()
                self.active_connections.add(conn)
                self.total_created += 1
                return conn
            
            raise Exception("Connection pool exhausted")
    
    async def return_connection(self, conn):
        """Return connection to pool"""
        async with self.lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                
                # Add to pool if under min connections or pool is empty
                if len(self.pool) < self.min_connections:
                    self.pool.append(conn)
                    self.connection_times[id(conn)] = time.time()
                else:
                    await self._close_connection(conn)
    
    async def _create_connection(self):
        """Create new connection"""
        return await asyncio.wait_for(
            self.create_connection(),
            timeout=self.connection_timeout
        )
    
    async def _close_connection(self, conn):
        """Close connection"""
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
            self.total_closed += 1
        except Exception:
            pass  # Ignore close errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'pool_size': len(self.pool),
            'active_connections': len(self.active_connections),
            'max_connections': self.max_connections,
            'total_created': self.total_created,
            'total_reused': self.total_reused,
            'total_closed': self.total_closed
        }


class PerformanceProfiler:
    """Advanced performance profiling and monitoring"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics = deque(maxlen=max_metrics)
        self.component_stats = defaultdict(list)
        self.active_operations = {}
        
    @contextmanager
    def profile(self, component: str, operation: str):
        """Context manager for profiling operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        operation_id = f"{component}:{operation}:{id(threading.current_thread())}"
        
        self.active_operations[operation_id] = {
            'component': component,
            'operation': operation,
            'start_time': start_time,
            'start_memory': start_memory
        }
        
        try:
            yield
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Create performance metric
            metric = PerformanceMetrics(
                component=component,
                operation=operation,
                duration=duration,
                memory_usage=memory_delta,
                cpu_usage=0.0,  # Would need psutil for accurate CPU usage
                throughput=1.0 / duration if duration > 0 else 0,
                error_rate=0.0,
                timestamp=end_time
            )
            
            self.metrics.append(metric)
            self.component_stats[component].append(metric)
            
        except Exception as e:
            # Record error metrics
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetrics(
                component=component,
                operation=operation,
                duration=duration,
                memory_usage=0,
                cpu_usage=0,
                throughput=0,
                error_rate=1.0,
                timestamp=end_time
            )
            
            self.metrics.append(metric)
            self.component_stats[component].append(metric)
            raise
        finally:
            self.active_operations.pop(operation_id, None)
    
    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get statistics for specific component"""
        metrics = self.component_stats.get(component, [])
        if not metrics:
            return {}
        
        durations = [m.duration for m in metrics]
        error_rate = sum(m.error_rate for m in metrics) / len(metrics)
        
        return {
            'total_operations': len(metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'error_rate': error_rate,
            'operations_per_sec': len(metrics) / (time.time() - metrics[0].timestamp) if metrics else 0
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        if not self.metrics:
            return {}
        
        all_durations = [m.duration for m in self.metrics]
        all_errors = [m.error_rate for m in self.metrics]
        
        return {
            'total_operations': len(self.metrics),
            'avg_duration': sum(all_durations) / len(all_durations),
            'error_rate': sum(all_errors) / len(all_errors),
            'active_operations': len(self.active_operations),
            'components': list(self.component_stats.keys())
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0


# Performance decorators
def cached_result(ttl: int = 3600, max_size: int = 128):
    """Decorator for caching function results"""
    cache = AdvancedCache(max_size=max_size, ttl_seconds=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result, hit = cache.get(cache_key)
            if hit:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator


def performance_monitored(component: str):
    """Decorator for monitoring function performance"""
    profiler = PerformanceProfiler()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(component, func.__name__):
                return func(*args, **kwargs)
        
        wrapper.profiler = profiler
        return wrapper
    return decorator


# Global performance infrastructure
global_cache = AdvancedCache(max_size=5000, max_memory_mb=512)
global_batch_processor = BatchProcessor(batch_size=64, max_wait_time=0.05)
global_profiler = PerformanceProfiler(max_metrics=50000)

# Utility functions
def optimize_memory():
    """Force garbage collection and memory optimization"""
    gc.collect()
    if _numpy_available:
        # Additional numpy memory cleanup could be added here
        pass


async def benchmark_async_function(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """Benchmark async function performance"""
    durations = []
    
    for _ in range(iterations):
        start_time = time.time()
        await func()
        duration = time.time() - start_time
        durations.append(duration)
    
    return {
        'mean_duration': sum(durations) / len(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'total_duration': sum(durations),
        'iterations': iterations
    }


def benchmark_function(func: Callable, iterations: int = 100) -> Dict[str, float]:
    """Benchmark function performance"""
    durations = []
    
    for _ in range(iterations):
        start_time = time.time()
        func()
        duration = time.time() - start_time
        durations.append(duration)
    
    return {
        'mean_duration': sum(durations) / len(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'total_duration': sum(durations),
        'iterations': iterations
    }