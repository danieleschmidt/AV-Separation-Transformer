"""
Advanced Resource Management for AV-Separation-Transformer
GPU memory optimization, model lifecycle, and resource pooling
"""

import time
import threading
import weakref
import gc
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import queue

import torch
import torch.nn as nn
import numpy as np
import psutil


class ResourceType(Enum):
    """Types of managed resources"""
    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    MODEL_INSTANCE = "model_instance"
    THREAD_POOL = "thread_pool"
    FILE_HANDLE = "file_handle"


@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    
    max_gpu_memory_mb: float = 4096.0
    max_cpu_memory_mb: float = 8192.0
    max_model_instances: int = 3
    max_thread_pools: int = 2
    max_file_handles: int = 100
    
    # Thresholds for cleanup
    gpu_cleanup_threshold: float = 0.8  # 80%
    cpu_cleanup_threshold: float = 0.8
    model_cleanup_threshold: float = 0.9


@dataclass
class ResourceUsage:
    """Current resource usage"""
    
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    model_instances: int = 0
    thread_pools: int = 0
    file_handles: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'gpu_memory_mb': self.gpu_memory_mb,
            'cpu_memory_mb': self.cpu_memory_mb,
            'model_instances': self.model_instances,
            'thread_pools': self.thread_pools,
            'file_handles': self.file_handles
        }


class ResourceManager:
    """
    Centralized resource management system
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.usage = ResourceUsage()
        
        # Resource tracking
        self.active_resources: Dict[str, Dict[str, Any]] = {
            resource_type.value: {}
            for resource_type in ResourceType
        }
        
        # Cleanup callbacks
        self.cleanup_callbacks: Dict[ResourceType, List[Callable]] = {
            resource_type: []
            for resource_type in ResourceType
        }
        
        # Locks for thread safety
        self.locks: Dict[ResourceType, threading.RLock] = {
            resource_type: threading.RLock()
            for resource_type in ResourceType
        }
        
        self.global_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Periodic cleanup
        self.cleanup_interval = 60  # seconds
        self.cleanup_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'cleanup_events': 0,
            'oom_events': 0,
            'last_cleanup': 0
        }
    
    def start(self):
        """Start resource manager"""
        
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        
        self.logger.info("Resource manager started")
    
    def stop(self):
        """Stop resource manager"""
        
        self.running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        # Force cleanup of all resources
        self._force_cleanup_all()
        
        self.logger.info("Resource manager stopped")
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        resource_id: str,
        resource_data: Any,
        size_estimate: float = 0.0,
        cleanup_callback: Optional[Callable] = None
    ) -> bool:
        """
        Allocate a resource with automatic management
        
        Args:
            resource_type: Type of resource
            resource_id: Unique identifier for the resource
            resource_data: The actual resource data
            size_estimate: Estimated size in MB
            cleanup_callback: Function to call when resource is cleaned up
            
        Returns:
            True if allocation succeeded, False otherwise
        """
        
        with self.locks[resource_type]:
            # Check if we can allocate this resource
            if not self._can_allocate(resource_type, size_estimate):
                # Try cleanup first
                self._cleanup_resources(resource_type)
                
                if not self._can_allocate(resource_type, size_estimate):
                    self.stats['oom_events'] += 1
                    self.logger.warning(
                        f"Cannot allocate {resource_type.value} resource {resource_id} "
                        f"(size: {size_estimate:.2f} MB)"
                    )
                    return False
            
            # Allocate resource
            resource_info = {
                'data': resource_data,
                'size_mb': size_estimate,
                'allocated_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'cleanup_callback': cleanup_callback,
                'weak_ref': weakref.ref(resource_data) if hasattr(resource_data, '__weakref__') else None
            }
            
            self.active_resources[resource_type.value][resource_id] = resource_info
            self._update_usage(resource_type, size_estimate)
            self.stats['total_allocations'] += 1
            
            self.logger.debug(
                f"Allocated {resource_type.value} resource {resource_id} "
                f"(size: {size_estimate:.2f} MB)"
            )
            
            return True
    
    def deallocate_resource(self, resource_type: ResourceType, resource_id: str) -> bool:
        """Deallocate a specific resource"""
        
        with self.locks[resource_type]:
            resources = self.active_resources[resource_type.value]
            
            if resource_id not in resources:
                return False
            
            resource_info = resources[resource_id]
            
            # Call cleanup callback if provided
            if resource_info['cleanup_callback']:
                try:
                    resource_info['cleanup_callback'](resource_info['data'])
                except Exception as e:
                    self.logger.error(f"Cleanup callback failed for {resource_id}: {e}")
            
            # Update usage
            self._update_usage(resource_type, -resource_info['size_mb'])
            
            # Remove from tracking
            del resources[resource_id]
            self.stats['total_deallocations'] += 1
            
            self.logger.debug(f"Deallocated {resource_type.value} resource {resource_id}")
            
            return True
    
    def get_resource(self, resource_type: ResourceType, resource_id: str) -> Optional[Any]:
        """Get a resource and update access statistics"""
        
        with self.locks[resource_type]:
            resources = self.active_resources[resource_type.value]
            
            if resource_id not in resources:
                return None
            
            resource_info = resources[resource_id]
            
            # Check if resource is still valid (using weak reference)
            if resource_info['weak_ref'] and resource_info['weak_ref']() is None:
                # Resource was garbage collected
                self.deallocate_resource(resource_type, resource_id)
                return None
            
            # Update access statistics
            resource_info['last_accessed'] = time.time()
            resource_info['access_count'] += 1
            
            return resource_info['data']
    
    def register_cleanup_callback(self, resource_type: ResourceType, callback: Callable):
        """Register a cleanup callback for a resource type"""
        
        self.cleanup_callbacks[resource_type].append(callback)
    
    def _can_allocate(self, resource_type: ResourceType, size_estimate: float) -> bool:
        """Check if resource can be allocated"""
        
        if resource_type == ResourceType.GPU_MEMORY:
            return (self.usage.gpu_memory_mb + size_estimate) <= self.limits.max_gpu_memory_mb
        
        elif resource_type == ResourceType.CPU_MEMORY:
            return (self.usage.cpu_memory_mb + size_estimate) <= self.limits.max_cpu_memory_mb
        
        elif resource_type == ResourceType.MODEL_INSTANCE:
            return self.usage.model_instances < self.limits.max_model_instances
        
        elif resource_type == ResourceType.THREAD_POOL:
            return self.usage.thread_pools < self.limits.max_thread_pools
        
        elif resource_type == ResourceType.FILE_HANDLE:
            return self.usage.file_handles < self.limits.max_file_handles
        
        return True
    
    def _update_usage(self, resource_type: ResourceType, size_delta: float):
        """Update resource usage statistics"""
        
        if resource_type == ResourceType.GPU_MEMORY:
            self.usage.gpu_memory_mb += size_delta
        
        elif resource_type == ResourceType.CPU_MEMORY:
            self.usage.cpu_memory_mb += size_delta
        
        elif resource_type == ResourceType.MODEL_INSTANCE:
            self.usage.model_instances += int(size_delta)
        
        elif resource_type == ResourceType.THREAD_POOL:
            self.usage.thread_pools += int(size_delta)
        
        elif resource_type == ResourceType.FILE_HANDLE:
            self.usage.file_handles += int(size_delta)
    
    def _cleanup_resources(self, resource_type: ResourceType):
        """Cleanup resources of specific type"""
        
        with self.locks[resource_type]:
            resources = self.active_resources[resource_type.value]
            current_time = time.time()
            
            # Sort by last access time (least recently used first)
            sorted_resources = sorted(
                resources.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            resources_cleaned = 0
            
            # Remove old or unused resources
            for resource_id, resource_info in sorted_resources:
                # Check if resource should be cleaned up
                should_cleanup = (
                    # Haven't been accessed in 10 minutes
                    current_time - resource_info['last_accessed'] > 600 or
                    # Weak reference is dead
                    (resource_info['weak_ref'] and resource_info['weak_ref']() is None)
                )
                
                if should_cleanup:
                    self.deallocate_resource(resource_type, resource_id)
                    resources_cleaned += 1
                    
                    # Stop if we've cleaned enough
                    if self._get_utilization(resource_type) < 0.7:  # Below 70%
                        break
            
            # Call registered cleanup callbacks
            for callback in self.cleanup_callbacks[resource_type]:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Cleanup callback failed: {e}")
            
            if resources_cleaned > 0:
                self.logger.info(
                    f"Cleaned up {resources_cleaned} {resource_type.value} resources"
                )
            
            self.stats['cleanup_events'] += 1
    
    def _get_utilization(self, resource_type: ResourceType) -> float:
        """Get current utilization ratio for resource type"""
        
        if resource_type == ResourceType.GPU_MEMORY:
            return self.usage.gpu_memory_mb / self.limits.max_gpu_memory_mb
        
        elif resource_type == ResourceType.CPU_MEMORY:
            return self.usage.cpu_memory_mb / self.limits.max_cpu_memory_mb
        
        elif resource_type == ResourceType.MODEL_INSTANCE:
            return self.usage.model_instances / self.limits.max_model_instances
        
        elif resource_type == ResourceType.THREAD_POOL:
            return self.usage.thread_pools / self.limits.max_thread_pools
        
        elif resource_type == ResourceType.FILE_HANDLE:
            return self.usage.file_handles / self.limits.max_file_handles
        
        return 0.0
    
    def _periodic_cleanup(self):
        """Periodic cleanup thread"""
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check each resource type
                for resource_type in ResourceType:
                    utilization = self._get_utilization(resource_type)
                    
                    # Determine cleanup threshold
                    if resource_type == ResourceType.GPU_MEMORY:
                        threshold = self.limits.gpu_cleanup_threshold
                    elif resource_type == ResourceType.CPU_MEMORY:
                        threshold = self.limits.cpu_cleanup_threshold
                    elif resource_type == ResourceType.MODEL_INSTANCE:
                        threshold = self.limits.model_cleanup_threshold
                    else:
                        threshold = 0.8
                    
                    # Cleanup if over threshold
                    if utilization > threshold:
                        self._cleanup_resources(resource_type)
                
                # Force garbage collection periodically
                if current_time - self.stats.get('last_gc', 0) > 300:  # Every 5 minutes
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.stats['last_gc'] = current_time
                
                self.stats['last_cleanup'] = current_time
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Periodic cleanup error: {e}")
                time.sleep(self.cleanup_interval)
    
    def _force_cleanup_all(self):
        """Force cleanup of all resources"""
        
        for resource_type in ResourceType:
            resource_ids = list(self.active_resources[resource_type.value].keys())
            for resource_id in resource_ids:
                self.deallocate_resource(resource_type, resource_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current resource manager status"""
        
        with self.global_lock:
            utilization = {}
            for resource_type in ResourceType:
                utilization[resource_type.value] = {
                    'utilization': self._get_utilization(resource_type),
                    'count': len(self.active_resources[resource_type.value])
                }
            
            return {
                'limits': {
                    'max_gpu_memory_mb': self.limits.max_gpu_memory_mb,
                    'max_cpu_memory_mb': self.limits.max_cpu_memory_mb,
                    'max_model_instances': self.limits.max_model_instances,
                    'max_thread_pools': self.limits.max_thread_pools,
                    'max_file_handles': self.limits.max_file_handles
                },
                'usage': self.usage.to_dict(),
                'utilization': utilization,
                'stats': self.stats,
                'running': self.running
            }


class ModelPool:
    """
    Pool of model instances for efficient resource utilization
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        max_instances: int = 3,
        warmup_instances: int = 1,
        resource_manager: Optional[ResourceManager] = None
    ):
        self.model_factory = model_factory
        self.max_instances = max_instances
        self.warmup_instances = warmup_instances
        self.resource_manager = resource_manager
        
        self.available_models = queue.Queue()
        self.in_use_models: Dict[str, Dict[str, Any]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize with warmup models
        self._warmup_models()
    
    def _warmup_models(self):
        """Create initial model instances"""
        
        for i in range(self.warmup_instances):
            try:
                model = self._create_model_instance(f"warmup_{i}")
                if model:
                    self.available_models.put(model)
                    self.logger.debug(f"Created warmup model instance {i}")
            except Exception as e:
                self.logger.error(f"Failed to create warmup model {i}: {e}")
    
    def _create_model_instance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Create a new model instance"""
        
        try:
            model = self.model_factory()
            
            # Estimate model size
            model_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1024**2
            
            model_instance = {
                'id': model_id,
                'model': model,
                'created_at': time.time(),
                'usage_count': 0,
                'total_inference_time': 0.0,
                'size_mb': model_size_mb
            }
            
            # Register with resource manager
            if self.resource_manager:
                success = self.resource_manager.allocate_resource(
                    ResourceType.MODEL_INSTANCE,
                    model_id,
                    model_instance,
                    size_estimate=model_size_mb,
                    cleanup_callback=self._cleanup_model_callback
                )
                
                if not success:
                    return None
            
            self.model_stats[model_id] = {
                'created_at': time.time(),
                'usage_count': 0,
                'total_inference_time': 0.0,
                'avg_inference_time': 0.0
            }
            
            return model_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create model instance {model_id}: {e}")
            return None
    
    def _cleanup_model_callback(self, model_instance: Dict[str, Any]):
        """Cleanup callback for model instances"""
        
        model_id = model_instance['id']
        
        # Move model to CPU and clear GPU memory
        if hasattr(model_instance['model'], 'cpu'):
            model_instance['model'].cpu()
        
        # Clear from stats
        if model_id in self.model_stats:
            del self.model_stats[model_id]
        
        self.logger.debug(f"Cleaned up model instance {model_id}")
    
    @contextmanager
    def get_model(self, timeout: float = 30.0):
        """
        Get a model instance from the pool
        
        Args:
            timeout: Maximum time to wait for available model
            
        Yields:
            Model instance dictionary
        """
        
        model_instance = None
        checkout_time = time.time()
        
        try:
            # Try to get available model
            try:
                model_instance = self.available_models.get(timeout=timeout)
            except queue.Empty:
                # No available models, try to create new one
                with self.lock:
                    if len(self.in_use_models) < self.max_instances:
                        model_id = f"dynamic_{int(time.time())}"
                        model_instance = self._create_model_instance(model_id)
                    
                    if not model_instance:
                        raise RuntimeError("No available models and cannot create new instance")
            
            # Track as in-use
            with self.lock:
                self.in_use_models[model_instance['id']] = {
                    'instance': model_instance,
                    'checkout_time': checkout_time,
                    'checkout_thread': threading.current_thread().ident
                }
            
            yield model_instance['model']
            
        finally:
            # Return model to pool
            if model_instance:
                with self.lock:
                    # Update statistics
                    usage_time = time.time() - checkout_time
                    model_instance['usage_count'] += 1
                    model_instance['total_inference_time'] += usage_time
                    
                    if model_instance['id'] in self.model_stats:
                        stats = self.model_stats[model_instance['id']]
                        stats['usage_count'] += 1
                        stats['total_inference_time'] += usage_time
                        stats['avg_inference_time'] = (
                            stats['total_inference_time'] / stats['usage_count']
                        )
                    
                    # Remove from in-use tracking
                    if model_instance['id'] in self.in_use_models:
                        del self.in_use_models[model_instance['id']]
                    
                    # Return to available pool
                    self.available_models.put(model_instance)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get model pool statistics"""
        
        with self.lock:
            total_models = self.available_models.qsize() + len(self.in_use_models)
            
            # Calculate aggregate statistics
            total_usage = sum(stats['usage_count'] for stats in self.model_stats.values())
            total_inference_time = sum(stats['total_inference_time'] for stats in self.model_stats.values())
            avg_inference_time = total_inference_time / max(total_usage, 1)
            
            return {
                'max_instances': self.max_instances,
                'total_instances': total_models,
                'available_instances': self.available_models.qsize(),
                'in_use_instances': len(self.in_use_models),
                'total_usage_count': total_usage,
                'avg_inference_time': avg_inference_time,
                'model_stats': dict(self.model_stats),
                'utilization': len(self.in_use_models) / max(total_models, 1)
            }
    
    def cleanup(self):
        """Cleanup all model instances"""
        
        with self.lock:
            # Clear available models
            while not self.available_models.empty():
                model_instance = self.available_models.get()
                if self.resource_manager:
                    self.resource_manager.deallocate_resource(
                        ResourceType.MODEL_INSTANCE,
                        model_instance['id']
                    )
            
            # Clear in-use models (warning: this may cause issues)
            for model_info in self.in_use_models.values():
                if self.resource_manager:
                    self.resource_manager.deallocate_resource(
                        ResourceType.MODEL_INSTANCE,
                        model_info['instance']['id']
                    )
            
            self.in_use_models.clear()
            self.model_stats.clear()
        
        self.logger.info("Model pool cleaned up")


class AdvancedCache:
    """
    Advanced caching system with multiple eviction policies
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 1024.0,
        eviction_policy: str = "lru",
        ttl_seconds: Optional[float] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds
        self.resource_manager = resource_manager
        
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.creation_times = {}
        self.sizes_mb = {}
        
        self.current_memory_mb = 0.0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_mb': 0.0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            # Check TTL
            if self.ttl_seconds:
                age = time.time() - self.creation_times[key]
                if age > self.ttl_seconds:
                    self._evict_item(key)
                    self.stats['misses'] += 1
                    return None
            
            # Update access statistics
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            self.stats['hits'] += 1
            return self.cache[key]
    
    def put(self, key: str, value: Any, size_mb: Optional[float] = None) -> bool:
        """Put item in cache"""
        
        if size_mb is None:
            size_mb = self._estimate_size_mb(value)
        
        with self.lock:
            # Check if item already exists
            if key in self.cache:
                self._evict_item(key)
            
            # Check if we need to make space
            while (
                len(self.cache) >= self.max_size or
                self.current_memory_mb + size_mb > self.max_memory_mb
            ):
                if not self._evict_lru():
                    # Cannot make space
                    return False
            
            # Add item
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] = 1
            self.creation_times[key] = current_time
            self.sizes_mb[key] = size_mb
            self.current_memory_mb += size_mb
            
            # Register with resource manager
            if self.resource_manager:
                self.resource_manager.allocate_resource(
                    ResourceType.CPU_MEMORY,
                    f"cache_{key}",
                    value,
                    size_estimate=size_mb
                )
            
            return True
    
    def _estimate_size_mb(self, value: Any) -> float:
        """Estimate memory size of value"""
        
        try:
            import sys
            import pickle
            
            # Try pickle size first
            try:
                pickled = pickle.dumps(value)
                return len(pickled) / 1024**2
            except Exception:
                pass
            
            # Fall back to sys.getsizeof
            return sys.getsizeof(value) / 1024**2
            
        except Exception:
            return 1.0  # Default estimate
    
    def _evict_item(self, key: str):
        """Evict specific item from cache"""
        
        if key in self.cache:
            size_mb = self.sizes_mb[key]
            
            # Cleanup resource manager registration
            if self.resource_manager:
                self.resource_manager.deallocate_resource(
                    ResourceType.CPU_MEMORY,
                    f"cache_{key}"
                )
            
            # Remove from all tracking
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.creation_times[key]
            del self.sizes_mb[key]
            
            self.current_memory_mb -= size_mb
            self.stats['evictions'] += 1
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        
        if not self.cache:
            return False
        
        if self.eviction_policy == "lru":
            # Least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        elif self.eviction_policy == "lfu":
            # Least frequently used
            lru_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        
        elif self.eviction_policy == "fifo":
            # First in, first out
            lru_key = min(self.creation_times.keys(), key=lambda k: self.creation_times[k])
        
        elif self.eviction_policy == "largest":
            # Largest item first
            lru_key = max(self.sizes_mb.keys(), key=lambda k: self.sizes_mb[k])
        
        else:
            # Default to LRU
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        self._evict_item(lru_key)
        return True
    
    def clear(self):
        """Clear all cache items"""
        
        with self.lock:
            keys_to_evict = list(self.cache.keys())
            for key in keys_to_evict:
                self._evict_item(key)
            
            self.current_memory_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.current_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'eviction_policy': self.eviction_policy,
                'memory_utilization': self.current_memory_mb / self.max_memory_mb,
                'size_utilization': len(self.cache) / self.max_size
            }


# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
        _global_resource_manager.start()
    
    return _global_resource_manager


def initialize_resource_management(
    limits: Optional[ResourceLimits] = None
) -> ResourceManager:
    """Initialize global resource management"""
    
    global _global_resource_manager
    
    if _global_resource_manager:
        _global_resource_manager.stop()
    
    _global_resource_manager = ResourceManager(limits)
    _global_resource_manager.start()
    
    return _global_resource_manager


def cleanup_resources():
    """Cleanup all managed resources"""
    
    if _global_resource_manager:
        _global_resource_manager.stop()