"""Caching implementation for models, features, and computed results."""

import pickle
import hashlib
import time
import logging
from typing import Any, Optional, Dict, Union, List
from pathlib import Path
import threading
from abc import ABC, abstractmethod

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False

import torch
import numpy as np

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCache(BaseCache):
    """In-memory cache implementation with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._expiry_times = {}
        self._lock = threading.RLock()
    
    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove expired items first
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._expiry_times.items()
                if expiry and expiry < current_time
            ]
            
            for key in expired_keys:
                self._remove_key(key)
            
            # If still full, remove LRU items
            while len(self._cache) >= self.max_size:
                lru_key = min(self._access_times, key=self._access_times.get)
                self._remove_key(lru_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all internal structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiry_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check expiry
            expiry = self._expiry_times.get(key)
            if expiry and time.time() > expiry:
                self._remove_key(key)
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        with self._lock:
            self._evict_if_needed()
            
            current_time = time.time()
            self._cache[key] = value
            self._access_times[key] = current_time
            
            # Set expiry time
            ttl = ttl or self.default_ttl
            self._expiry_times[key] = current_time + ttl if ttl > 0 else None
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for expiry in self._expiry_times.values()
                if expiry and expiry < current_time
            )
            
            return {
                'type': 'memory',
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_items': expired_count,
                'hit_ratio': getattr(self, '_hit_ratio', 0.0)
            }


class RedisCache(BaseCache):
    """Redis-based distributed cache implementation."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = 'av_sep:'
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required for RedisCache")
        
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We handle binary data
        )
        
        # Test connection
        try:
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        try:
            data = self.client.get(self._make_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        try:
            data = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            
            if ttl > 0:
                return self.client.setex(self._make_key(key), ttl, data)
            else:
                return self.client.set(self._make_key(key), data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            return self.client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries with our prefix."""
        try:
            keys = self.client.keys(f"{self.key_prefix}*")
            if keys:
                return self.client.delete(*keys) > 0
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return self.client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            info = self.client.info()
            return {
                'type': 'redis',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_ratio': info.get('keyspace_hits', 0) / max(1, 
                    info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)
                )
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {'type': 'redis', 'error': str(e)}


class CacheManager:
    """High-level cache manager with multiple backends and intelligent strategies.
    
    Supports automatic serialization of PyTorch tensors, NumPy arrays,
    and other ML objects with size-aware caching policies.
    
    Example:
        >>> cache = CacheManager('memory', max_size=1000)
        >>> model_output = separator.separate(audio, video)
        >>> cache.set_model_output('session_123', model_output)
        >>> cached_output = cache.get_model_output('session_123')
    """
    
    def __init__(
        self,
        backend: str = 'memory',
        max_size: int = 1000,
        default_ttl: int = 3600,
        enable_compression: bool = True,
        max_object_size: int = 100 * 1024 * 1024,  # 100MB
        **backend_kwargs
    ):
        self.backend_type = backend
        self.enable_compression = enable_compression
        self.max_object_size = max_object_size
        
        # Initialize backend
        if backend == 'memory':
            self.backend = MemoryCache(max_size, default_ttl)
        elif backend == 'redis':
            self.backend = RedisCache(default_ttl=default_ttl, **backend_kwargs)
        else:
            raise ValueError(f"Unsupported cache backend: {backend}")
        
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.Lock()
        
        logger.info(f"CacheManager initialized with {backend} backend")
    
    def _serialize_object(self, obj: Any) -> bytes:
        """Serialize object with special handling for ML objects."""
        if isinstance(obj, torch.Tensor):
            # Convert to CPU and numpy for serialization
            data = {
                'type': 'torch.Tensor',
                'data': obj.detach().cpu().numpy(),
                'dtype': str(obj.dtype),
                'device': str(obj.device)
            }
        elif isinstance(obj, np.ndarray):
            data = {
                'type': 'numpy.ndarray',
                'data': obj,
                'dtype': str(obj.dtype)
            }
        else:
            data = obj
        
        serialized = pickle.dumps(data)
        
        # Apply compression if enabled and beneficial
        if self.enable_compression and len(serialized) > 1024:
            try:
                import gzip
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # Only if 20%+ reduction
                    return b'GZIP:' + compressed
            except ImportError:
                pass
        
        return serialized
    
    def _deserialize_object(self, data: bytes) -> Any:
        """Deserialize object with special handling for ML objects."""
        # Check for compression
        if data.startswith(b'GZIP:'):
            import gzip
            data = gzip.decompress(data[5:])
        
        obj_data = pickle.loads(data)
        
        # Reconstruct ML objects
        if isinstance(obj_data, dict) and 'type' in obj_data:
            if obj_data['type'] == 'torch.Tensor':
                tensor = torch.from_numpy(obj_data['data'])
                # Note: Device conversion should be handled by caller
                return tensor
            elif obj_data['type'] == 'numpy.ndarray':
                return obj_data['data']
        
        return obj_data
    
    def _check_object_size(self, obj: Any) -> bool:
        """Check if object size is within limits."""
        try:
            serialized = self._serialize_object(obj)
            return len(serialized) <= self.max_object_size
        except Exception:
            return False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with hit/miss tracking."""
        try:
            cached_data = self.backend.get(key)
            
            with self._lock:
                if cached_data is not None:
                    self._hit_count += 1
                    return self._deserialize_object(cached_data)
                else:
                    self._miss_count += 1
                    return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            with self._lock:
                self._miss_count += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with size checks."""
        try:
            if not self._check_object_size(value):
                logger.warning(f"Object too large for cache: {key}")
                return False
            
            serialized = self._serialize_object(value)
            return self.backend.set(key, serialized, ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_model_output(
        self,
        session_id: str,
        audio_hash: str,
        video_hash: str,
        model_config: Dict[str, Any]
    ) -> Optional[List[torch.Tensor]]:
        """Get cached model separation output."""
        key = self._generate_key(
            'model_output',
            session_id,
            audio_hash,
            video_hash,
            **model_config
        )
        return self.get(key)
    
    def set_model_output(
        self,
        session_id: str,
        audio_hash: str,
        video_hash: str,
        model_config: Dict[str, Any],
        output: List[torch.Tensor],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache model separation output."""
        key = self._generate_key(
            'model_output',
            session_id,
            audio_hash,
            video_hash,
            **model_config
        )
        return self.set(key, output, ttl)
    
    def get_features(
        self,
        feature_type: str,
        input_hash: str,
        processor_config: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Get cached audio or video features."""
        key = self._generate_key(
            f'{feature_type}_features',
            input_hash,
            **processor_config
        )
        return self.get(key)
    
    def set_features(
        self,
        feature_type: str,
        input_hash: str,
        processor_config: Dict[str, Any],
        features: torch.Tensor,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache audio or video features."""
        key = self._generate_key(
            f'{feature_type}_features',
            input_hash,
            **processor_config
        )
        return self.set(key, features, ttl)
    
    def invalidate_session(self, session_id: str) -> int:
        """Invalidate all cache entries for a session."""
        # This is a simplified implementation
        # In practice, you'd need to track keys by session
        count = 0
        if hasattr(self.backend, 'scan_iter'):  # Redis
            pattern = f"*{session_id}*"
            for key in self.backend.client.scan_iter(match=pattern):
                if self.backend.client.delete(key):
                    count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        backend_stats = self.backend.get_stats()
        
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_ratio = self._hit_count / max(1, total_requests)
        
        return {
            **backend_stats,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_ratio': hit_ratio,
            'total_requests': total_requests,
            'compression_enabled': self.enable_compression,
            'max_object_size': self.max_object_size
        }
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        return self.backend.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            test_key = 'health_check'
            test_value = {'test': True, 'timestamp': time.time()}
            
            # Test write
            if not self.set(test_key, test_value, ttl=60):
                return {'status': 'unhealthy', 'error': 'Failed to write test value'}
            
            # Test read
            cached_value = self.get(test_key)
            if cached_value != test_value:
                return {'status': 'unhealthy', 'error': 'Failed to read test value'}
            
            # Cleanup
            self.backend.delete(test_key)
            
            return {
                'status': 'healthy',
                **self.get_stats()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


class DistributedCache:
    """Distributed cache with multiple backends for high availability."""
    
    def __init__(self, backends: List[CacheManager]):
        self.backends = backends
        self.primary = backends[0] if backends else None
        
        if not self.primary:
            raise ValueError("At least one cache backend required")
    
    def get(self, key: str) -> Optional[Any]:
        """Get from first available backend."""
        for backend in self.backends:
            try:
                value = backend.get(key)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(f"Backend {backend.backend_type} failed: {e}")
                continue
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set to all available backends."""
        success_count = 0
        for backend in self.backends:
            try:
                if backend.set(key, value, ttl):
                    success_count += 1
            except Exception as e:
                logger.warning(f"Backend {backend.backend_type} failed: {e}")
        
        return success_count > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats from all backends."""
        return {
            f'backend_{i}': backend.get_stats()
            for i, backend in enumerate(self.backends)
        }