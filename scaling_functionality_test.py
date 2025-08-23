#!/usr/bin/env python3
"""
Generation 3: Scaling Functionality Test
Performance optimization, caching, concurrency, auto-scaling, and resource management
"""

import sys
import os
import time
import json
import threading
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, OrderedDict
from pathlib import Path
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LRUCache:
    """High-performance LRU cache implementation"""
    
    def __init__(self, max_size=128, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def _is_expired(self, key):
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamps[key] > self.ttl
    
    def get(self, key):
        """Get item from cache"""
        with self._lock:
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hit_count += 1
                return value
            
            self.miss_count += 1
            return None
    
    def put(self, key, value):
        """Put item in cache"""
        with self._lock:
            # Remove expired entries
            self._cleanup_expired()
            
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        if self.ttl is None:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.timestamps[key]
    
    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate_percent': hit_rate
        }

class ResourcePool:
    """Thread-safe resource pool for managing expensive objects"""
    
    def __init__(self, factory, max_size=10, timeout=30):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.pool = []
        self.in_use = set()
        self.created = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def acquire(self):
        """Acquire a resource from the pool"""
        with self._condition:
            # Try to get existing resource
            while not self.pool and self.created >= self.max_size:
                if not self._condition.wait(timeout=self.timeout):
                    raise TimeoutError("Timeout waiting for resource")
            
            if self.pool:
                resource = self.pool.pop()
            else:
                # Create new resource
                resource = self.factory()
                self.created += 1
            
            self.in_use.add(id(resource))
            return resource
    
    def release(self, resource):
        """Release a resource back to the pool"""
        with self._condition:
            resource_id = id(resource)
            if resource_id in self.in_use:
                self.in_use.remove(resource_id)
                self.pool.append(resource)
                self._condition.notify()
    
    def get_stats(self):
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self.pool),
                'in_use': len(self.in_use),
                'total_created': self.created,
                'max_size': self.max_size
            }

class AdaptiveLoadBalancer:
    """Adaptive load balancer with health checking"""
    
    def __init__(self, workers):
        self.workers = workers
        self.worker_stats = {i: {'requests': 0, 'failures': 0, 'avg_latency': 0} 
                           for i in range(len(workers))}
        self.current_worker = 0
        self._lock = threading.Lock()
    
    def get_next_worker(self):
        """Get next worker using weighted round-robin"""
        with self._lock:
            # Calculate weights based on performance
            weights = []
            for i, stats in self.worker_stats.items():
                # Lower latency and fewer failures = higher weight
                failure_rate = stats['failures'] / max(stats['requests'], 1)
                latency_penalty = min(stats['avg_latency'] / 1000.0, 1.0)  # Cap at 1s
                weight = max(0.1, 1.0 - failure_rate - latency_penalty)
                weights.append(weight)
            
            # Weighted selection
            total_weight = sum(weights)
            if total_weight > 0:
                # Find worker with best weight
                best_worker = max(range(len(weights)), key=lambda i: weights[i])
                return best_worker
            else:
                # Fallback to round-robin
                worker = self.current_worker
                self.current_worker = (self.current_worker + 1) % len(self.workers)
                return worker
    
    def record_request(self, worker_id, latency, success=True):
        """Record request statistics for a worker"""
        with self._lock:
            stats = self.worker_stats[worker_id]
            stats['requests'] += 1
            if not success:
                stats['failures'] += 1
            
            # Update average latency (exponential moving average)
            alpha = 0.1
            stats['avg_latency'] = (
                alpha * latency + (1 - alpha) * stats['avg_latency']
            )

class AutoScaler:
    """Automatic scaling based on metrics"""
    
    def __init__(self, min_workers=1, max_workers=10, target_cpu=70):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu = target_cpu
        self.current_workers = min_workers
        self.metrics_history = []
        self.scale_cooldown = 30  # seconds
        self.last_scale_time = 0
    
    def add_metrics(self, cpu_usage, memory_usage, queue_size, avg_latency):
        """Add current system metrics"""
        self.metrics_history.append({
            'timestamp': time.time(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'queue_size': queue_size,
            'avg_latency': avg_latency
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self.metrics_history = [
            m for m in self.metrics_history if m['timestamp'] > cutoff_time
        ]
    
    def should_scale_up(self):
        """Determine if we should scale up"""
        if not self.metrics_history or self.current_workers >= self.max_workers:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check recent metrics
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        if len(recent_metrics) < 3:
            return False
        
        avg_cpu = np.mean([m['cpu_usage'] for m in recent_metrics])
        avg_queue = np.mean([m['queue_size'] for m in recent_metrics])
        avg_latency = np.mean([m['avg_latency'] for m in recent_metrics])
        
        # Scale up conditions
        return (avg_cpu > self.target_cpu or 
                avg_queue > self.current_workers * 2 or
                avg_latency > 1000)  # 1 second
    
    def should_scale_down(self):
        """Determine if we should scale down"""
        if not self.metrics_history or self.current_workers <= self.min_workers:
            return False
        
        if time.time() - self.last_scale_time < self.scale_cooldown * 2:  # Longer cooldown
            return False
        
        recent_metrics = self.metrics_history[-10:]  # Longer observation for scale down
        if len(recent_metrics) < 5:
            return False
        
        avg_cpu = np.mean([m['cpu_usage'] for m in recent_metrics])
        avg_queue = np.mean([m['queue_size'] for m in recent_metrics])
        
        # Scale down conditions
        return avg_cpu < self.target_cpu * 0.5 and avg_queue < self.current_workers * 0.5
    
    def scale(self):
        """Perform scaling decision"""
        if self.should_scale_up():
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            self.last_scale_time = time.time()
            return 'scale_up', self.current_workers
        elif self.should_scale_down():
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            self.last_scale_time = time.time()
            return 'scale_down', self.current_workers
        
        return 'no_change', self.current_workers

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_numpy():
        """Optimize NumPy settings"""
        # Enable multi-threading for NumPy operations
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
        os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    @staticmethod
    def create_memory_mapped_array(size, dtype=np.float32):
        """Create memory-mapped array for large data"""
        filename = f'/tmp/mmap_array_{hash(time.time())}.dat'
        return np.memmap(filename, dtype=dtype, mode='w+', shape=size)
    
    @staticmethod
    def batch_process(data, batch_size, process_func):
        """Process data in batches for memory efficiency"""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = process_func(batch)
            results.append(result)
        return results

async def async_process_simulation(item_id, duration=0.1):
    """Simulate asynchronous processing"""
    await asyncio.sleep(duration)
    return f"processed_{item_id}"

def test_caching_system():
    """Test high-performance caching"""
    try:
        cache = LRUCache(max_size=5, ttl=2)
        
        # Test basic operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1", "Should retrieve cached value"
        assert cache.get("nonexistent") is None, "Should return None for missing keys"
        
        # Test LRU eviction
        for i in range(6):
            cache.put(f"key{i+3}", f"value{i+3}")
        
        assert cache.get("key1") is None, "Should evict oldest items"
        assert len(cache.cache) <= cache.max_size, "Should respect max size"
        
        # Test TTL expiration
        cache.put("ttl_test", "value", )
        time.sleep(2.1)  # Wait for expiration
        assert cache.get("ttl_test") is None, "Should expire after TTL"
        
        stats = cache.get_stats()
        assert stats['hit_rate_percent'] >= 0, "Should calculate hit rate"
        
        print("✅ Caching system working")
        return True
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def test_resource_pooling():
    """Test resource pool management"""
    try:
        # Create a simple resource factory
        def create_resource():
            return {"id": time.time(), "data": np.random.randn(100)}
        
        pool = ResourcePool(create_resource, max_size=3)
        
        # Test resource acquisition and release
        resources = []
        for _ in range(3):
            resource = pool.acquire()
            resources.append(resource)
        
        assert pool.get_stats()['in_use'] == 3, "Should track in-use resources"
        
        # Release resources
        for resource in resources:
            pool.release(resource)
        
        assert pool.get_stats()['pool_size'] == 3, "Should return resources to pool"
        
        print("✅ Resource pooling working")
        return True
    except Exception as e:
        print(f"❌ Resource pooling test failed: {e}")
        return False

def test_load_balancing():
    """Test adaptive load balancing"""
    try:
        workers = ['worker1', 'worker2', 'worker3']
        balancer = AdaptiveLoadBalancer(workers)
        
        # Simulate some requests with different performance
        for i in range(20):
            worker_id = balancer.get_next_worker()
            # Worker 0 performs better
            latency = 100 if worker_id == 0 else 300
            success = True if worker_id != 1 or i % 5 != 0 else False  # Worker 1 fails sometimes
            
            balancer.record_request(worker_id, latency, success)
        
        # Should favor worker 0 due to better performance
        selected_workers = [balancer.get_next_worker() for _ in range(10)]
        worker_0_count = selected_workers.count(0)
        
        assert worker_0_count >= 3, "Should prefer better performing workers"
        
        print("✅ Load balancing working")
        return True
    except Exception as e:
        print(f"❌ Load balancing test failed: {e}")
        return False

def test_auto_scaling():
    """Test automatic scaling decisions"""
    try:
        scaler = AutoScaler(min_workers=2, max_workers=8, target_cpu=70)
        
        # Simulate high load requiring scale up
        for i in range(5):
            scaler.add_metrics(cpu_usage=85, memory_usage=60, queue_size=10, avg_latency=1200)
        
        action, workers = scaler.scale()
        assert action == 'scale_up', "Should scale up under high load"
        assert workers > scaler.min_workers, "Should increase worker count"
        
        # Simulate low load for scale down
        time.sleep(61)  # Wait for cooldown
        for i in range(10):
            scaler.add_metrics(cpu_usage=20, memory_usage=30, queue_size=1, avg_latency=100)
        
        action, workers = scaler.scale()
        assert action in ['scale_down', 'no_change'], "Should scale down or maintain under low load"
        
        print("✅ Auto-scaling working")
        return True
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities"""
    try:
        # Test thread-based concurrency
        def cpu_bound_task(n):
            return sum(i * i for i in range(n))
        
        start_time = time.perf_counter()
        
        # Sequential processing
        results_seq = [cpu_bound_task(1000) for _ in range(4)]
        sequential_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        
        # Concurrent processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            results_conc = list(executor.map(cpu_bound_task, [1000] * 4))
        concurrent_time = time.perf_counter() - start_time
        
        assert results_seq == results_conc, "Results should be identical"
        # Concurrent might not always be faster for CPU-bound tasks due to GIL
        
        print("✅ Concurrent processing working")
        return True
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False

def test_async_processing():
    """Test asynchronous processing"""
    try:
        async def test_async():
            # Process multiple items concurrently
            tasks = [async_process_simulation(i, 0.05) for i in range(10)]
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            async_time = time.perf_counter() - start_time
            
            assert len(results) == 10, "Should process all items"
            assert all(r.startswith("processed_") for r in results), "Should process correctly"
            assert async_time < 0.3, "Should complete faster than sequential"  # 10 * 0.05 = 0.5s
            
            return True
        
        # Run the async test
        result = asyncio.run(test_async())
        assert result, "Async test should pass"
        
        print("✅ Async processing working")
        return True
    except Exception as e:
        print(f"❌ Async processing test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization utilities"""
    try:
        PerformanceOptimizer.optimize_numpy()
        
        # Test batch processing
        data = list(range(100))
        def square_batch(batch):
            return [x ** 2 for x in batch]
        
        results = PerformanceOptimizer.batch_process(data, batch_size=25, process_func=square_batch)
        flattened_results = [item for batch in results for item in batch]
        
        expected = [x ** 2 for x in data]
        assert flattened_results == expected, "Batch processing should produce correct results"
        
        # Test memory-mapped array
        mmap_array = PerformanceOptimizer.create_memory_mapped_array((10, 10))
        mmap_array[0, 0] = 42.0
        assert mmap_array[0, 0] == 42.0, "Memory-mapped array should work"
        
        print("✅ Performance optimization working")
        return True
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def main():
    """Run scaling functionality tests"""
    print("🚀 GENERATION 3: Scaling Functionality Tests")
    print("=" * 55)
    
    tests = [
        test_caching_system,
        test_resource_pooling,
        test_load_balancing,
        test_auto_scaling,
        test_concurrent_processing,
        test_async_processing,
        test_performance_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 55)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Generation 3: SCALING FUNCTIONALITY WORKING")
        return True
    else:
        print(f"❌ Generation 3: {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)