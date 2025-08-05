"""
Auto-scaling and Load Balancing for AV-Separation-Transformer
Horizontal scaling, load balancing, and resource management
"""

import time
import threading
import queue
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

import torch
import numpy as np
import psutil


class WorkerStatus(Enum):
    """Worker node status"""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class WorkerNode:
    """Worker node information"""
    
    node_id: str
    host: str
    port: int
    status: WorkerStatus
    current_load: float
    max_capacity: int
    active_requests: int
    last_heartbeat: float
    gpu_memory_mb: float
    cpu_percent: float
    model_loaded: bool
    capabilities: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class ScalingMetrics:
    """Scaling decision metrics"""
    
    avg_queue_length: float
    avg_response_time: float
    cpu_utilization: float
    memory_utilization: float
    active_workers: int
    total_requests_per_minute: float
    error_rate: float
    timestamp: float


class LoadBalancer:
    """
    Intelligent load balancer with multiple strategies
    """
    
    def __init__(self, strategy: str = "least_connections"):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin_select,
            "least_connections": self._least_connections_select,
            "weighted_response_time": self._weighted_response_time_select,
            "resource_aware": self._resource_aware_select,
            "consistent_hashing": self._consistent_hash_select
        }
        
        # Round robin counter
        self._round_robin_counter = 0
        
        # Consistent hashing ring
        self._hash_ring = {}
        self._rebuild_hash_ring()
    
    def register_worker(self, worker: WorkerNode):
        """Register a new worker node"""
        
        with self.lock:
            self.workers[worker.node_id] = worker
            self._rebuild_hash_ring()
            
            self.logger.info(f"Registered worker {worker.node_id} at {worker.host}:{worker.port}")
    
    def unregister_worker(self, node_id: str):
        """Unregister a worker node"""
        
        with self.lock:
            if node_id in self.workers:
                worker = self.workers.pop(node_id)
                self._rebuild_hash_ring()
                
                self.logger.info(f"Unregistered worker {node_id}")
    
    def update_worker_status(self, node_id: str, status_update: Dict[str, Any]):
        """Update worker status"""
        
        with self.lock:
            if node_id in self.workers:
                worker = self.workers[node_id]
                
                # Update fields
                for key, value in status_update.items():
                    if hasattr(worker, key):
                        if key == 'status' and isinstance(value, str):
                            worker.status = WorkerStatus(value)
                        else:
                            setattr(worker, key, value)
                
                worker.last_heartbeat = time.time()
    
    def select_worker(
        self,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkerNode]:
        """Select best worker for request"""
        
        with self.lock:
            # Filter available workers
            available_workers = [
                worker for worker in self.workers.values()
                if worker.status in [WorkerStatus.IDLE, WorkerStatus.BUSY] and
                worker.model_loaded and
                time.time() - worker.last_heartbeat < 30  # 30 second timeout
            ]
            
            if not available_workers:
                return None
            
            # Apply load balancing strategy
            strategy_func = self.strategies.get(self.strategy, self._least_connections_select)
            selected_worker = strategy_func(available_workers, request_data)
            
            # Update worker load
            if selected_worker:
                selected_worker.active_requests += 1
                self._update_worker_status_by_load(selected_worker)
            
            return selected_worker
    
    def release_worker(self, node_id: str):
        """Release worker after request completion"""
        
        with self.lock:
            if node_id in self.workers:
                worker = self.workers[node_id]
                worker.active_requests = max(0, worker.active_requests - 1)
                self._update_worker_status_by_load(worker)
    
    def _update_worker_status_by_load(self, worker: WorkerNode):
        """Update worker status based on current load"""
        
        load_ratio = worker.active_requests / max(worker.max_capacity, 1)
        
        if load_ratio == 0:
            worker.status = WorkerStatus.IDLE
        elif load_ratio < 0.8:
            worker.status = WorkerStatus.BUSY
        else:
            worker.status = WorkerStatus.OVERLOADED
        
        worker.current_load = load_ratio
    
    def _round_robin_select(
        self,
        workers: List[WorkerNode],
        request_data: Optional[Dict[str, Any]] = None
    ) -> WorkerNode:
        """Round robin selection"""
        
        self._round_robin_counter = (self._round_robin_counter + 1) % len(workers)
        return workers[self._round_robin_counter]
    
    def _least_connections_select(
        self,
        workers: List[WorkerNode],
        request_data: Optional[Dict[str, Any]] = None
    ) -> WorkerNode:
        """Select worker with least active connections"""
        
        return min(workers, key=lambda w: w.active_requests)
    
    def _weighted_response_time_select(
        self,
        workers: List[WorkerNode],
        request_data: Optional[Dict[str, Any]] = None
    ) -> WorkerNode:
        """Select worker based on weighted response time"""
        
        # Calculate weights based on recent response times
        # This would require tracking response times per worker
        
        # For now, fall back to least connections
        return self._least_connections_select(workers, request_data)
    
    def _resource_aware_select(
        self,
        workers: List[WorkerNode],
        request_data: Optional[Dict[str, Any]] = None
    ) -> WorkerNode:
        """Select worker based on resource availability"""
        
        def resource_score(worker: WorkerNode) -> float:
            # Higher score is better
            cpu_score = max(0, 100 - worker.cpu_percent) / 100
            memory_score = max(0, 1 - worker.gpu_memory_mb / 8000)  # Assume 8GB GPU
            load_score = max(0, 1 - worker.current_load)
            
            return (cpu_score * 0.3 + memory_score * 0.4 + load_score * 0.3)
        
        return max(workers, key=resource_score)
    
    def _consistent_hash_select(
        self,
        workers: List[WorkerNode],
        request_data: Optional[Dict[str, Any]] = None
    ) -> WorkerNode:
        """Select worker using consistent hashing"""
        
        if not request_data:
            return self._least_connections_select(workers, request_data)
        
        # Create hash from request data
        request_str = json.dumps(request_data, sort_keys=True)
        request_hash = int(hashlib.md5(request_str.encode()).hexdigest(), 16)
        
        # Find closest worker in hash ring
        if not self._hash_ring:
            return self._least_connections_select(workers, request_data)
        
        ring_positions = sorted(self._hash_ring.keys())
        
        # Find first position >= request_hash
        for position in ring_positions:
            if position >= request_hash:
                node_id = self._hash_ring[position]
                if node_id in self.workers:
                    worker = self.workers[node_id]
                    if worker in workers:  # Worker is available
                        return worker
        
        # Wrap around to first position
        if ring_positions:
            node_id = self._hash_ring[ring_positions[0]]
            if node_id in self.workers:
                worker = self.workers[node_id]
                if worker in workers:
                    return worker
        
        # Fall back to least connections
        return self._least_connections_select(workers, request_data)
    
    def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring"""
        
        self._hash_ring = {}
        
        # Add multiple virtual nodes per worker for better distribution
        virtual_nodes_per_worker = 100
        
        for worker in self.workers.values():
            for i in range(virtual_nodes_per_worker):
                virtual_key = f"{worker.node_id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self._hash_ring[hash_value] = worker.node_id
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        
        with self.lock:
            total_workers = len(self.workers)
            active_workers = sum(1 for w in self.workers.values() 
                               if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY])
            total_requests = sum(w.active_requests for w in self.workers.values())
            avg_load = np.mean([w.current_load for w in self.workers.values()]) if self.workers else 0
            
            return {
                'strategy': self.strategy,
                'total_workers': total_workers,
                'active_workers': active_workers,
                'total_active_requests': total_requests,
                'average_load': avg_load,
                'workers': {node_id: worker.to_dict() for node_id, worker in self.workers.items()}
            }


class AutoScaler:
    """
    Automatic scaling based on metrics and thresholds
    """
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        min_workers: int = 1,
        max_workers: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scale_up_cooldown: int = 300,  # 5 minutes
        scale_down_cooldown: int = 600   # 10 minutes
    ):
        self.load_balancer = load_balancer
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_events: List[Dict[str, Any]] = []
        
        self.running = False
        self.monitoring_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], None]] = None
        self.scale_down_callback: Optional[Callable[[List[str]], None]] = None
    
    def set_scale_up_callback(self, callback: Callable[[int], None]):
        """Set callback for scaling up (adding workers)"""
        self.scale_up_callback = callback
    
    def set_scale_down_callback(self, callback: Callable[[List[str]], None]):
        """Set callback for scaling down (removing workers)"""
        self.scale_down_callback = callback
    
    def start_monitoring(self, monitoring_interval: int = 60):
        """Start automatic scaling monitoring"""
        
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(monitoring_interval,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic scaling monitoring"""
        
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self, monitoring_interval: int):
        """Main monitoring loop"""
        
        while self.running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 2 hours)
                cutoff_time = time.time() - 7200
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                
                # Make scaling decision
                self._evaluate_scaling_decision(metrics)
                
                time.sleep(monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(monitoring_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        
        stats = self.load_balancer.get_worker_stats()
        
        # Calculate metrics
        active_workers = stats['active_workers']
        total_requests = stats['total_active_requests']
        avg_load = stats['average_load']
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Calculate request rate (requests per minute)
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics_history
            if current_time - m.timestamp < 60  # Last minute
        ]
        
        if recent_metrics:
            request_rate = np.mean([m.total_requests_per_minute for m in recent_metrics])
        else:
            request_rate = 0.0
        
        return ScalingMetrics(
            avg_queue_length=total_requests,
            avg_response_time=0.0,  # Would need to track response times
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            active_workers=active_workers,
            total_requests_per_minute=request_rate,
            error_rate=0.0,  # Would need to track errors
            timestamp=current_time
        )
    
    def _evaluate_scaling_decision(self, metrics: ScalingMetrics):
        """Evaluate whether to scale up or down"""
        
        current_time = time.time()
        
        # Check if we need to scale up
        should_scale_up = (
            metrics.active_workers < self.max_workers and
            (
                metrics.avg_queue_length / max(metrics.active_workers, 1) > self.scale_up_threshold or
                metrics.cpu_utilization > 80 or
                metrics.memory_utilization > 85
            ) and
            current_time - self.last_scale_up > self.scale_up_cooldown
        )
        
        # Check if we can scale down
        should_scale_down = (
            metrics.active_workers > self.min_workers and
            metrics.avg_queue_length / max(metrics.active_workers, 1) < self.scale_down_threshold and
            metrics.cpu_utilization < 50 and
            metrics.memory_utilization < 60 and
            current_time - self.last_scale_down > self.scale_down_cooldown
        )
        
        if should_scale_up:
            self._scale_up(metrics)
        elif should_scale_down:
            self._scale_down(metrics)
    
    def _scale_up(self, metrics: ScalingMetrics):
        """Scale up by adding workers"""
        
        # Determine how many workers to add
        load_per_worker = metrics.avg_queue_length / max(metrics.active_workers, 1)
        workers_needed = max(1, int(load_per_worker / self.scale_up_threshold))
        workers_to_add = min(workers_needed, self.max_workers - metrics.active_workers)
        
        if workers_to_add > 0:
            self.logger.info(f"Scaling up: adding {workers_to_add} workers")
            
            # Record scaling event
            event = {
                'type': 'scale_up',
                'timestamp': time.time(),
                'workers_added': workers_to_add,
                'trigger_metrics': asdict(metrics),
                'reason': f"Load per worker: {load_per_worker:.2f}, threshold: {self.scale_up_threshold}"
            }
            self.scaling_events.append(event)
            
            # Call scale up callback
            if self.scale_up_callback:
                try:
                    self.scale_up_callback(workers_to_add)
                    self.last_scale_up = time.time()
                except Exception as e:
                    self.logger.error(f"Scale up callback failed: {e}")
    
    def _scale_down(self, metrics: ScalingMetrics):
        """Scale down by removing workers"""
        
        # Determine how many workers to remove
        load_per_worker = metrics.avg_queue_length / max(metrics.active_workers, 1)
        optimal_workers = max(self.min_workers, int(metrics.avg_queue_length / self.scale_up_threshold))
        workers_to_remove = metrics.active_workers - optimal_workers
        workers_to_remove = max(1, min(workers_to_remove, metrics.active_workers - self.min_workers))
        
        if workers_to_remove > 0:
            # Select workers to remove (prefer idle workers)
            stats = self.load_balancer.get_worker_stats()
            worker_candidates = []
            
            for node_id, worker_info in stats['workers'].items():
                if worker_info['status'] == 'idle':
                    worker_candidates.append(node_id)
            
            # If not enough idle workers, select least loaded ones
            if len(worker_candidates) < workers_to_remove:
                busy_workers = [
                    (node_id, worker_info['active_requests'])
                    for node_id, worker_info in stats['workers'].items()
                    if worker_info['status'] == 'busy'
                ]
                busy_workers.sort(key=lambda x: x[1])  # Sort by active requests
                
                needed = workers_to_remove - len(worker_candidates)
                worker_candidates.extend([node_id for node_id, _ in busy_workers[:needed]])
            
            workers_to_remove_list = worker_candidates[:workers_to_remove]
            
            if workers_to_remove_list:
                self.logger.info(f"Scaling down: removing {len(workers_to_remove_list)} workers")
                
                # Record scaling event
                event = {
                    'type': 'scale_down',
                    'timestamp': time.time(),
                    'workers_removed': len(workers_to_remove_list),
                    'worker_ids': workers_to_remove_list,
                    'trigger_metrics': asdict(metrics),
                    'reason': f"Load per worker: {load_per_worker:.2f}, threshold: {self.scale_down_threshold}"
                }
                self.scaling_events.append(event)
                
                # Call scale down callback
                if self.scale_down_callback:
                    try:
                        self.scale_down_callback(workers_to_remove_list)
                        self.last_scale_down = time.time()
                    except Exception as e:
                        self.logger.error(f"Scale down callback failed: {e}")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        
        recent_events = [
            event for event in self.scaling_events
            if time.time() - event['timestamp'] < 3600  # Last hour
        ]
        
        scale_up_events = [e for e in recent_events if e['type'] == 'scale_up']
        scale_down_events = [e for e in recent_events if e['type'] == 'scale_down']
        
        return {
            'running': self.running,
            'config': {
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'scale_up_threshold': self.scale_up_threshold,
                'scale_down_threshold': self.scale_down_threshold,
                'scale_up_cooldown': self.scale_up_cooldown,
                'scale_down_cooldown': self.scale_down_cooldown
            },
            'recent_events': {
                'scale_up_count': len(scale_up_events),
                'scale_down_count': len(scale_down_events),
                'total_workers_added': sum(e['workers_added'] for e in scale_up_events),
                'total_workers_removed': sum(e['workers_removed'] for e in scale_down_events)
            },
            'current_metrics': self.metrics_history[-1].to_dict() if self.metrics_history else None,
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down
        }


class DistributedCoordinator:
    """
    Coordinator for distributed AV separation processing
    """
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(self.load_balancer)
        self.request_queue = asyncio.Queue()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Setup auto-scaling callbacks
        self.auto_scaler.set_scale_up_callback(self._scale_up_workers)
        self.auto_scaler.set_scale_down_callback(self._scale_down_workers)
    
    async def process_request(
        self,
        request_id: str,
        audio_data: np.ndarray,
        video_data: np.ndarray,
        num_speakers: int = 2
    ) -> Dict[str, Any]:
        """Process separation request using distributed workers"""
        
        start_time = time.time()
        
        try:
            # Select worker
            worker = self.load_balancer.select_worker({
                'request_id': request_id,
                'num_speakers': num_speakers,
                'audio_length': len(audio_data),
                'video_frames': len(video_data)
            })
            
            if not worker:
                raise RuntimeError("No available workers")
            
            # Track request
            self.active_requests[request_id] = {
                'worker_id': worker.node_id,
                'start_time': start_time,
                'num_speakers': num_speakers
            }
            
            # Send request to worker (placeholder - would use actual RPC)
            result = await self._send_request_to_worker(
                worker, request_id, audio_data, video_data, num_speakers
            )
            
            # Release worker
            self.load_balancer.release_worker(worker.node_id)
            
            # Clean up tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'result': result,
                'worker_id': worker.node_id,
                'processing_time': processing_time
            }
            
        except Exception as e:
            # Release worker if assigned
            if request_id in self.active_requests:
                worker_id = self.active_requests[request_id]['worker_id']
                self.load_balancer.release_worker(worker_id)
                del self.active_requests[request_id]
            
            self.logger.error(f"Request {request_id} failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def _send_request_to_worker(
        self,
        worker: WorkerNode,
        request_id: str,
        audio_data: np.ndarray,
        video_data: np.ndarray,
        num_speakers: int
    ) -> Dict[str, Any]:
        """Send separation request to worker (placeholder implementation)"""
        
        # This would implement actual RPC communication with workers
        # For now, simulate processing time and return mock result
        
        await asyncio.sleep(0.1)  # Simulate network + processing time
        
        return {
            'separated_audio': [audio_data / 2, audio_data / 2],  # Mock separation
            'num_speakers_detected': num_speakers,
            'processing_node': worker.node_id
        }
    
    def _scale_up_workers(self, num_workers: int):
        """Scale up callback - add new workers"""
        
        # This would implement actual worker spawning
        # For now, simulate by creating mock workers
        
        for i in range(num_workers):
            worker_id = f"worker_{int(time.time())}_{i}"
            worker = WorkerNode(
                node_id=worker_id,
                host="localhost",
                port=8000 + len(self.load_balancer.workers),
                status=WorkerStatus.IDLE,
                current_load=0.0,
                max_capacity=4,
                active_requests=0,
                last_heartbeat=time.time(),
                gpu_memory_mb=1000.0,
                cpu_percent=20.0,
                model_loaded=True,
                capabilities=["audio_visual_separation"]
            )
            
            self.load_balancer.register_worker(worker)
            
            self.logger.info(f"Added worker {worker_id}")
    
    def _scale_down_workers(self, worker_ids: List[str]):
        """Scale down callback - remove workers"""
        
        for worker_id in worker_ids:
            self.load_balancer.unregister_worker(worker_id)
            self.logger.info(f"Removed worker {worker_id}")
    
    def start(self):
        """Start the distributed coordinator"""
        
        self.auto_scaler.start_monitoring()
        self.logger.info("Distributed coordinator started")
    
    def stop(self):
        """Stop the distributed coordinator"""
        
        self.auto_scaler.stop_monitoring()
        self.logger.info("Distributed coordinator stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of distributed system"""
        
        return {
            'load_balancer': self.load_balancer.get_worker_stats(),
            'auto_scaler': self.auto_scaler.get_scaling_stats(),
            'active_requests': len(self.active_requests),
            'request_details': self.active_requests
        }