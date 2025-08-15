"""
Distributed Inference Engine for Multi-GPU and Multi-Node Scaling
Advanced parallelization strategies with intelligent load balancing.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import asyncio
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
import logging
from collections import defaultdict, deque
import socket
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib


class DistributionStrategy(Enum):
    """Distribution strategies for model deployment."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class NodeRole(Enum):
    """Roles for distributed nodes."""
    MASTER = "master"
    WORKER = "worker"
    PARAMETER_SERVER = "parameter_server"
    CACHE_SERVER = "cache_server"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    role: NodeRole
    host: str
    port: int
    gpu_count: int
    memory_gb: float
    cpu_cores: int
    status: str
    load: float
    last_heartbeat: float


@dataclass
class InferenceRequest:
    """Container for inference request."""
    request_id: str
    audio_data: torch.Tensor
    video_data: torch.Tensor
    priority: int
    timestamp: float
    callback: Optional[Callable] = None


@dataclass
class InferenceResult:
    """Container for inference result."""
    request_id: str
    result: torch.Tensor
    processing_time: float
    node_id: str
    timestamp: float


class DistributedCache:
    """
    Redis-based distributed caching system for inference results.
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 ttl_seconds: int = 3600, max_size_mb: int = 1000):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            self.redis_client.ping()  # Test connection
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.local_cache = {}  # Fallback local cache
        
        self.logger = logging.getLogger(__name__)
        
        if self.redis_available:
            self.logger.info("Connected to Redis for distributed caching")
        else:
            self.logger.warning("Redis not available, using local cache fallback")
    
    def _compute_key(self, audio_data: torch.Tensor, video_data: torch.Tensor, 
                    model_id: str) -> str:
        """Compute cache key for inputs."""
        audio_hash = hashlib.md5(audio_data.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        video_hash = hashlib.md5(video_data.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        return f"av_cache:{model_id}:{audio_hash}:{video_hash}"
    
    def get(self, audio_data: torch.Tensor, video_data: torch.Tensor, 
           model_id: str) -> Optional[torch.Tensor]:
        """Get cached result."""
        key = self._compute_key(audio_data, video_data, model_id)
        
        if self.redis_available:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return pickle.loads(cached_data)
            except Exception as e:
                self.logger.warning(f"Redis get error: {e}")
        
        # Fallback to local cache
        return self.local_cache.get(key)
    
    def put(self, audio_data: torch.Tensor, video_data: torch.Tensor,
           model_id: str, result: torch.Tensor):
        """Store result in cache."""
        key = self._compute_key(audio_data, video_data, model_id)
        serialized_result = pickle.dumps(result)
        
        # Check size limit
        if len(serialized_result) > self.max_size_mb * 1024 * 1024:
            self.logger.warning("Result too large for caching")
            return
        
        if self.redis_available:
            try:
                self.redis_client.setex(key, self.ttl_seconds, serialized_result)
                return
            except Exception as e:
                self.logger.warning(f"Redis put error: {e}")
        
        # Fallback to local cache
        self.local_cache[key] = result
        
        # Simple size management for local cache
        if len(self.local_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.local_cache.keys())[:100]
            for k in keys_to_remove:
                del self.local_cache[k]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        if self.redis_available:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Redis invalidate error: {e}")
        
        # Local cache invalidation
        keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
        for k in keys_to_remove:
            del self.local_cache[k]


class LoadBalancer:
    """
    Intelligent load balancer for distributing inference requests.
    """
    
    def __init__(self):
        self.nodes = {}
        self.request_queue = deque()
        self.node_loads = defaultdict(float)
        self.node_performance = defaultdict(lambda: {"total_requests": 0, "total_time": 0.0})
        
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node_info: NodeInfo):
        """Register a new node."""
        self.nodes[node_info.node_id] = node_info
        self.logger.info(f"Registered node: {node_info.node_id} ({node_info.role.value})")
    
    def unregister_node(self, node_id: str):
        """Unregister a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Unregistered node: {node_id}")
    
    def update_node_load(self, node_id: str, load: float):
        """Update node load information."""
        self.node_loads[node_id] = load
        if node_id in self.nodes:
            self.nodes[node_id].load = load
    
    def select_best_node(self, request: InferenceRequest) -> Optional[str]:
        """Select best node for processing request using intelligent routing."""
        available_workers = [
            node for node in self.nodes.values() 
            if node.role == NodeRole.WORKER and node.status == "ready"
        ]
        
        if not available_workers:
            return None
        
        # Score nodes based on multiple factors
        best_node = None
        best_score = float('inf')
        
        for node in available_workers:
            # Calculate composite score
            load_score = self.node_loads.get(node.node_id, 0.5)
            
            # Performance score (average processing time)
            perf_data = self.node_performance[node.node_id]
            if perf_data["total_requests"] > 0:
                avg_time = perf_data["total_time"] / perf_data["total_requests"]
                perf_score = avg_time / 0.1  # Normalize to 100ms baseline
            else:
                perf_score = 1.0  # Default for new nodes
            
            # Resource score (GPU count and memory)
            resource_score = 1.0 / (node.gpu_count * node.memory_gb / 100)
            
            # Priority boost for high-priority requests
            priority_factor = 1.0 if request.priority <= 5 else 0.8
            
            # Composite score (lower is better)
            composite_score = (load_score * 0.4 + perf_score * 0.3 + 
                             resource_score * 0.3) * priority_factor
            
            if composite_score < best_score:
                best_score = composite_score
                best_node = node.node_id
        
        return best_node
    
    def record_completion(self, node_id: str, processing_time: float):
        """Record completion of request on node."""
        self.node_performance[node_id]["total_requests"] += 1
        self.node_performance[node_id]["total_time"] += processing_time
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        stats = {
            "total_nodes": len(self.nodes),
            "worker_nodes": len([n for n in self.nodes.values() if n.role == NodeRole.WORKER]),
            "average_load": np.mean(list(self.node_loads.values())) if self.node_loads else 0,
            "node_performance": {}
        }
        
        for node_id, perf_data in self.node_performance.items():
            if perf_data["total_requests"] > 0:
                stats["node_performance"][node_id] = {
                    "requests": perf_data["total_requests"],
                    "avg_time": perf_data["total_time"] / perf_data["total_requests"]
                }
        
        return stats


class ModelSharding:
    """
    Model sharding for large models across multiple GPUs/nodes.
    """
    
    def __init__(self, model: torch.nn.Module, num_shards: int):
        self.model = model
        self.num_shards = num_shards
        self.shards = []
        self.shard_devices = []
        
        self.logger = logging.getLogger(__name__)
        
        self._create_shards()
    
    def _create_shards(self):
        """Create model shards."""
        # Get model parameters
        params = list(self.model.parameters())
        param_count = len(params)
        params_per_shard = param_count // self.num_shards
        
        # Create shards
        for i in range(self.num_shards):
            start_idx = i * params_per_shard
            end_idx = (i + 1) * params_per_shard if i < self.num_shards - 1 else param_count
            
            # Create shard with subset of parameters
            shard = torch.nn.ModuleDict()
            
            # This is a simplified sharding - in practice, you'd need more sophisticated
            # model partitioning based on the actual model architecture
            for j, (name, module) in enumerate(self.model.named_modules()):
                if start_idx <= j < end_idx:
                    shard[name] = module
            
            # Assign to GPU if available
            device = f"cuda:{i}" if torch.cuda.is_available() and i < torch.cuda.device_count() else "cpu"
            shard = shard.to(device)
            
            self.shards.append(shard)
            self.shard_devices.append(device)
            
            self.logger.info(f"Created shard {i} on device {device}")
    
    def forward_sharded(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Forward pass using sharded model."""
        # This is a simplified implementation
        # In practice, you'd need sophisticated communication between shards
        
        intermediate_results = []
        
        for i, shard in enumerate(self.shards):
            # Move inputs to shard device
            shard_inputs = tuple(inp.to(self.shard_devices[i]) for inp in inputs)
            
            # Process with shard
            with torch.no_grad():
                shard_output = shard(shard_inputs[0])  # Simplified
                intermediate_results.append(shard_output)
        
        # Combine results (this would be model-specific)
        # For demonstration, we'll just average the outputs
        combined = torch.stack(intermediate_results).mean(dim=0)
        
        return combined


class DataParallelism:
    """
    Data parallelism for distributing batches across multiple devices.
    """
    
    def __init__(self, model: torch.nn.Module, device_ids: List[int] = None):
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.num_devices = len(self.device_ids)
        
        if self.num_devices > 1:
            # Use DataParallel for multi-GPU
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        else:
            self.model = model
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized data parallelism with {self.num_devices} devices")
    
    def forward_parallel(self, audio_batch: torch.Tensor, 
                        video_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass with data parallelism."""
        # Move to primary device
        if self.num_devices > 1:
            primary_device = f"cuda:{self.device_ids[0]}"
            audio_batch = audio_batch.to(primary_device)
            video_batch = video_batch.to(primary_device)
        
        # Forward pass (DataParallel handles distribution automatically)
        with torch.no_grad():
            result = self.model(audio_batch, video_batch)
        
        return result


class PipelineParallelism:
    """
    Pipeline parallelism for processing different stages on different devices.
    """
    
    def __init__(self, model_stages: List[torch.nn.Module], device_ids: List[int]):
        self.stages = model_stages
        self.device_ids = device_ids
        self.num_stages = len(model_stages)
        
        # Move each stage to its assigned device
        for i, (stage, device_id) in enumerate(zip(self.stages, device_ids)):
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            stage.to(device)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Stage {i} assigned to device {device}")
    
    def forward_pipeline(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Forward pass using pipeline parallelism."""
        # Process through pipeline stages
        current_data = inputs
        
        for i, stage in enumerate(self.stages):
            # Move data to stage device
            device = f"cuda:{self.device_ids[i]}" if torch.cuda.is_available() else "cpu"
            
            if isinstance(current_data, tuple):
                current_data = tuple(data.to(device) for data in current_data)
            else:
                current_data = current_data.to(device)
            
            # Process with current stage
            with torch.no_grad():
                current_data = stage(current_data)
        
        return current_data


class DistributedInferenceEngine:
    """
    Main distributed inference engine orchestrating all components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.distributed_cache = DistributedCache(
            redis_host=self.config.get('redis_host', 'localhost'),
            redis_port=self.config.get('redis_port', 6379)
        )
        self.load_balancer = LoadBalancer()
        
        # Node management
        self.node_id = self.config.get('node_id', socket.gethostname())
        self.node_role = NodeRole(self.config.get('role', 'worker'))
        self.master_host = self.config.get('master_host', 'localhost')
        self.master_port = self.config.get('master_port', 29500)
        
        # Request processing
        self.request_queue = asyncio.Queue()
        self.result_callbacks = {}
        self.processing_active = False
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize based on role
        if self.node_role == NodeRole.MASTER:
            self._init_master()
        elif self.node_role == NodeRole.WORKER:
            self._init_worker()
    
    def _init_master(self):
        """Initialize master node."""
        self.logger.info("Initializing as master node")
        
        # Register self as master
        master_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.MASTER,
            host=socket.gethostname(),
            port=self.master_port,
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            memory_gb=self._get_memory_gb(),
            cpu_cores=mp.cpu_count(),
            status="ready",
            load=0.0,
            last_heartbeat=time.time()
        )
        
        self.load_balancer.register_node(master_info)
    
    def _init_worker(self):
        """Initialize worker node."""
        self.logger.info("Initializing as worker node")
        
        # Register self as worker
        worker_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.WORKER,
            host=socket.gethostname(),
            port=0,  # Dynamic port assignment
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            memory_gb=self._get_memory_gb(),
            cpu_cores=mp.cpu_count(),
            status="ready",
            load=0.0,
            last_heartbeat=time.time()
        )
        
        self.load_balancer.register_node(worker_info)
    
    def _get_memory_gb(self) -> float:
        """Get available memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0  # Default fallback
    
    def distribute_model(self, model: torch.nn.Module, 
                        strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL) -> Any:
        """Distribute model based on strategy."""
        if strategy == DistributionStrategy.DATA_PARALLEL:
            device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
            return DataParallelism(model, device_ids)
        
        elif strategy == DistributionStrategy.MODEL_PARALLEL:
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            return ModelSharding(model, num_devices)
        
        elif strategy == DistributionStrategy.PIPELINE_PARALLEL:
            # This would require model-specific stage definition
            # For demonstration, we'll split into 2 stages
            stages = [model.encoder, model.decoder]  # Assuming model has these components
            device_ids = list(range(min(2, torch.cuda.device_count())))
            return PipelineParallelism(stages, device_ids)
        
        else:
            return model
    
    async def submit_request(self, audio_data: torch.Tensor, video_data: torch.Tensor,
                           priority: int = 5) -> str:
        """Submit inference request and return request ID."""
        request_id = f"{self.node_id}_{int(time.time() * 1000)}_{self.request_count}"
        self.request_count += 1
        
        request = InferenceRequest(
            request_id=request_id,
            audio_data=audio_data,
            video_data=video_data,
            priority=priority,
            timestamp=time.time()
        )
        
        await self.request_queue.put(request)
        self.logger.info(f"Submitted request: {request_id}")
        
        return request_id
    
    async def process_request(self, request: InferenceRequest, 
                            model: torch.nn.Module) -> InferenceResult:
        """Process individual inference request."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.distributed_cache.get(
                request.audio_data, request.video_data, str(id(model))
            )
            
            if cached_result is not None:
                processing_time = time.time() - start_time
                
                return InferenceResult(
                    request_id=request.request_id,
                    result=cached_result,
                    processing_time=processing_time,
                    node_id=self.node_id,
                    timestamp=time.time()
                )
            
            # Perform inference
            model.eval()
            with torch.no_grad():
                result = model(request.audio_data, request.video_data)
            
            # Cache result
            self.distributed_cache.put(
                request.audio_data, request.video_data, str(id(model)), result
            )
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Update load balancer
            self.load_balancer.record_completion(self.node_id, processing_time)
            
            return InferenceResult(
                request_id=request.request_id,
                result=result,
                processing_time=processing_time,
                node_id=self.node_id,
                timestamp=time.time()
            )
        
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            raise
    
    async def start_processing(self, model: torch.nn.Module):
        """Start processing requests from queue."""
        self.processing_active = True
        self.logger.info("Started distributed inference processing")
        
        while self.processing_active:
            try:
                # Get request from queue (with timeout)
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                # Process request
                result = await self.process_request(request, model)
                
                # Handle result (callback or store)
                if request.callback:
                    request.callback(result)
                else:
                    self.result_callbacks[request.request_id] = result
                
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def stop_processing(self):
        """Stop processing requests."""
        self.processing_active = False
        self.logger.info("Stopped distributed inference processing")
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResult]:
        """Get result for request ID."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_callbacks:
                result = self.result_callbacks.pop(request_id)
                return result
            
            time.sleep(0.1)
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = (self.total_processing_time / self.request_count 
                             if self.request_count > 0 else 0)
        
        return {
            "node_id": self.node_id,
            "role": self.node_role.value,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "average_processing_time": avg_processing_time,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count),
            "load_balancer_stats": self.load_balancer.get_load_statistics()
        }


# Global distributed engine instance
global_distributed_engine = DistributedInferenceEngine()


def distributed_inference(strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL):
    """
    Decorator for enabling distributed inference on functions.
    """
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Extract model and inputs
            if len(args) >= 3:
                model, audio_input, video_input = args[0], args[1], args[2]
                
                # Distribute model
                distributed_model = global_distributed_engine.distribute_model(model, strategy)
                
                # Submit request
                request_id = await global_distributed_engine.submit_request(
                    audio_input, video_input
                )
                
                # Get result
                result = global_distributed_engine.get_result(request_id)
                
                if result:
                    return result.result
                else:
                    # Fallback to local processing
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # Run async wrapper in event loop
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except:
                # Fallback to original function
                return func(*args, **kwargs)
        
        return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Create example model
    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(512, 256)
            self.decoder = torch.nn.Linear(256, 512)
        
        def forward(self, audio, video):
            encoded = self.encoder(audio + video)
            return self.decoder(encoded)
    
    # Initialize distributed engine
    engine = DistributedInferenceEngine({'role': 'master'})
    
    # Create and distribute model
    model = ExampleModel()
    distributed_model = engine.distribute_model(model, DistributionStrategy.DATA_PARALLEL)
    
    # Example inference
    audio = torch.randn(4, 512)
    video = torch.randn(4, 512)
    
    async def run_inference():
        request_id = await engine.submit_request(audio, video)
        result = engine.get_result(request_id)
        if result:
            print(f"Inference completed: {result.processing_time:.3f}s")
    
    # Run example
    asyncio.run(run_inference())