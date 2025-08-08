"""
Auto-Scaling System for AV-Separation-Transformer
Intelligent resource scaling and load balancing
"""

import asyncio
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger('av_separation.autoscaler')


class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"


class ResourceType(Enum):
    """Types of resources to scale"""
    CPU_WORKERS = "cpu_workers"
    MEMORY_POOLS = "memory_pools"
    CONNECTION_POOLS = "connection_pools"
    CACHE_SIZE = "cache_size"
    BATCH_SIZE = "batch_size"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_usage: float
    memory_usage: float
    queue_length: int
    response_time: float
    error_rate: float
    throughput: float
    active_connections: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'queue_length': self.queue_length,
            'response_time': self.response_time,
            'error_rate': self.error_rate,
            'throughput': self.throughput,
            'active_connections': self.active_connections,
            'timestamp': self.timestamp
        }


@dataclass
class ScalingRule:
    """Rule for automatic scaling decisions"""
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    cooldown_seconds: int = 300
    min_value: int = 1
    max_value: int = 100
    enabled: bool = True


class MetricsCollector:
    """Collects and aggregates system metrics for scaling decisions"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = {}
        self._lock = threading.Lock()
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics to history"""
        with self._lock:
            self.metrics_history.append(metrics)
            self.current_metrics = metrics.to_dict()
    
    def get_recent_metrics(self, seconds: int = 300) -> List[ScalingMetrics]:
        """Get metrics from recent time window"""
        with self._lock:
            cutoff_time = time.time() - seconds
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, seconds: int = 300) -> Optional[Dict[str, float]]:
        """Get average metrics over time window"""
        recent = self.get_recent_metrics(seconds)
        if not recent:
            return None
            
        return {
            'cpu_usage': statistics.mean(m.cpu_usage for m in recent),
            'memory_usage': statistics.mean(m.memory_usage for m in recent),
            'queue_length': statistics.mean(m.queue_length for m in recent),
            'response_time': statistics.mean(m.response_time for m in recent),
            'error_rate': statistics.mean(m.error_rate for m in recent),
            'throughput': statistics.mean(m.throughput for m in recent),
            'active_connections': statistics.mean(m.active_connections for m in recent)
        }
    
    def get_percentile_metrics(self, percentile: int = 95, seconds: int = 300) -> Optional[Dict[str, float]]:
        """Get percentile metrics over time window"""
        recent = self.get_recent_metrics(seconds)
        if not recent:
            return None
            
        def safe_percentile(values):
            if not values:
                return 0.0
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]
        
        return {
            'cpu_usage': safe_percentile([m.cpu_usage for m in recent]),
            'memory_usage': safe_percentile([m.memory_usage for m in recent]),
            'queue_length': safe_percentile([m.queue_length for m in recent]),
            'response_time': safe_percentile([m.response_time for m in recent]),
            'error_rate': safe_percentile([m.error_rate for m in recent]),
            'throughput': safe_percentile([m.throughput for m in recent]),
            'active_connections': safe_percentile([m.active_connections for m in recent])
        }


class ScalingEngine:
    """Core scaling decision engine"""
    
    def __init__(self):
        self.rules: List[ScalingRule] = []
        self.last_scaling_action = {}  # resource_type -> timestamp
        self.current_resource_levels = {}  # resource_type -> current_value
        self.scaling_history = deque(maxlen=1000)
        
        # Initialize default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules"""
        self.rules = [
            # CPU worker scaling
            ScalingRule(
                resource_type=ResourceType.CPU_WORKERS,
                metric_name='cpu_usage',
                threshold_up=75.0,
                threshold_down=25.0,
                scale_up_factor=1.5,
                scale_down_factor=0.8,
                cooldown_seconds=180,
                min_value=2,
                max_value=32
            ),
            # Memory pool scaling
            ScalingRule(
                resource_type=ResourceType.MEMORY_POOLS,
                metric_name='memory_usage',
                threshold_up=80.0,
                threshold_down=30.0,
                scale_up_factor=1.3,
                scale_down_factor=0.8,
                cooldown_seconds=240,
                min_value=256,  # MB
                max_value=4096  # MB
            ),
            # Connection pool scaling
            ScalingRule(
                resource_type=ResourceType.CONNECTION_POOLS,
                metric_name='active_connections',
                threshold_up=80.0,  # % of max connections
                threshold_down=20.0,
                scale_up_factor=1.4,
                scale_down_factor=0.7,
                cooldown_seconds=120,
                min_value=5,
                max_value=100
            ),
            # Batch size optimization
            ScalingRule(
                resource_type=ResourceType.BATCH_SIZE,
                metric_name='response_time',
                threshold_up=100.0,  # ms
                threshold_down=20.0,   # ms
                scale_up_factor=0.8,   # Reduce batch size if response time high
                scale_down_factor=1.2, # Increase batch size if response time low
                cooldown_seconds=60,
                min_value=8,
                max_value=128
            )
        ]
        
        # Initialize current resource levels
        for rule in self.rules:
            self.current_resource_levels[rule.resource_type] = rule.min_value
    
    def add_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.rules.append(rule)
        if rule.resource_type not in self.current_resource_levels:
            self.current_resource_levels[rule.resource_type] = rule.min_value
    
    def evaluate_scaling(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate all rules and return scaling actions"""
        actions = []
        current_time = time.time()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            # Check cooldown period
            last_action_time = self.last_scaling_action.get(rule.resource_type, 0)
            if current_time - last_action_time < rule.cooldown_seconds:
                continue
            
            # Get current metric value
            metric_value = metrics.get(rule.metric_name, 0)
            current_level = self.current_resource_levels.get(rule.resource_type, rule.min_value)
            
            action = self._evaluate_rule(rule, metric_value, current_level)
            
            if action['action'] != ScalingAction.NO_ACTION:
                actions.append(action)
                self.last_scaling_action[rule.resource_type] = current_time
                
                # Update current level
                self.current_resource_levels[rule.resource_type] = action['new_value']
                
                # Record in history
                self.scaling_history.append({
                    'timestamp': current_time,
                    'resource_type': rule.resource_type.value,
                    'action': action['action'].value,
                    'old_value': current_level,
                    'new_value': action['new_value'],
                    'trigger_metric': rule.metric_name,
                    'metric_value': metric_value
                })
        
        return actions
    
    def _evaluate_rule(
        self, 
        rule: ScalingRule, 
        metric_value: float, 
        current_level: Union[int, float]
    ) -> Dict[str, Any]:
        """Evaluate single scaling rule"""
        
        # Special handling for connection pools (percentage-based)
        if rule.resource_type == ResourceType.CONNECTION_POOLS:
            # Convert to percentage of current level
            if current_level > 0:
                metric_percentage = (metric_value / current_level) * 100
            else:
                metric_percentage = 0
            comparison_value = metric_percentage
        else:
            comparison_value = metric_value
        
        # Determine scaling action
        if comparison_value > rule.threshold_up:
            # Scale up
            if rule.resource_type == ResourceType.BATCH_SIZE:
                # For batch size, "scaling up" means reducing size
                new_value = max(rule.min_value, int(current_level * rule.scale_up_factor))
            else:
                new_value = min(rule.max_value, int(current_level * rule.scale_up_factor))
            
            if new_value != current_level:
                return {
                    'action': ScalingAction.SCALE_UP,
                    'resource_type': rule.resource_type,
                    'old_value': current_level,
                    'new_value': new_value,
                    'reason': f'{rule.metric_name} ({metric_value:.2f}) > threshold ({rule.threshold_up})'
                }
        
        elif comparison_value < rule.threshold_down:
            # Scale down
            if rule.resource_type == ResourceType.BATCH_SIZE:
                # For batch size, "scaling down" means increasing size
                new_value = min(rule.max_value, int(current_level * rule.scale_down_factor))
            else:
                new_value = max(rule.min_value, int(current_level * rule.scale_down_factor))
            
            if new_value != current_level:
                return {
                    'action': ScalingAction.SCALE_DOWN,
                    'resource_type': rule.resource_type,
                    'old_value': current_level,
                    'new_value': new_value,
                    'reason': f'{rule.metric_name} ({metric_value:.2f}) < threshold ({rule.threshold_down})'
                }
        
        return {
            'action': ScalingAction.NO_ACTION,
            'resource_type': rule.resource_type,
            'old_value': current_level,
            'new_value': current_level,
            'reason': f'{rule.metric_name} ({metric_value:.2f}) within thresholds'
        }
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent scaling history"""
        return list(self.scaling_history)[-limit:]
    
    def get_current_resource_levels(self) -> Dict[str, Union[int, float]]:
        """Get current resource levels"""
        return dict(self.current_resource_levels)


class LoadBalancer:
    """Intelligent load balancing across workers"""
    
    def __init__(self, initial_workers: int = 4):
        self.workers = []
        self.worker_stats = defaultdict(dict)
        self.round_robin_index = 0
        self._lock = threading.Lock()
        
        # Initialize workers
        for i in range(initial_workers):
            self.add_worker(f"worker_{i}")
    
    def add_worker(self, worker_id: str):
        """Add new worker to pool"""
        with self._lock:
            if worker_id not in self.workers:
                self.workers.append(worker_id)
                self.worker_stats[worker_id] = {
                    'active_requests': 0,
                    'total_requests': 0,
                    'total_errors': 0,
                    'avg_response_time': 0.0,
                    'last_activity': time.time()
                }
    
    def remove_worker(self, worker_id: str):
        """Remove worker from pool"""
        with self._lock:
            if worker_id in self.workers:
                self.workers.remove(worker_id)
                # Keep stats for historical purposes
    
    def get_next_worker(self, strategy: str = "least_connections") -> Optional[str]:
        """Get next worker using specified strategy"""
        with self._lock:
            if not self.workers:
                return None
            
            if strategy == "round_robin":
                worker = self.workers[self.round_robin_index]
                self.round_robin_index = (self.round_robin_index + 1) % len(self.workers)
                return worker
            
            elif strategy == "least_connections":
                # Choose worker with least active connections
                best_worker = min(
                    self.workers,
                    key=lambda w: self.worker_stats[w]['active_requests']
                )
                return best_worker
            
            elif strategy == "weighted_response_time":
                # Choose worker with best response time
                active_workers = [
                    w for w in self.workers 
                    if self.worker_stats[w]['total_requests'] > 0
                ]
                
                if not active_workers:
                    return self.workers[0]
                
                best_worker = min(
                    active_workers,
                    key=lambda w: self.worker_stats[w]['avg_response_time']
                )
                return best_worker
            
            else:
                # Default to round robin
                return self.get_next_worker("round_robin")
    
    def record_request_start(self, worker_id: str):
        """Record request start for worker"""
        with self._lock:
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]['active_requests'] += 1
                self.worker_stats[worker_id]['total_requests'] += 1
                self.worker_stats[worker_id]['last_activity'] = time.time()
    
    def record_request_end(self, worker_id: str, response_time: float, error: bool = False):
        """Record request completion for worker"""
        with self._lock:
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats['active_requests'] = max(0, stats['active_requests'] - 1)
                
                if error:
                    stats['total_errors'] += 1
                
                # Update average response time
                current_avg = stats['avg_response_time']
                total_requests = stats['total_requests']
                if total_requests > 1:
                    # Exponential moving average
                    stats['avg_response_time'] = (current_avg * 0.9) + (response_time * 0.1)
                else:
                    stats['avg_response_time'] = response_time
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers"""
        with self._lock:
            return dict(self.worker_stats)
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across workers"""
        with self._lock:
            if not self.workers:
                return {}
            
            total_requests = sum(
                self.worker_stats[w]['active_requests'] 
                for w in self.workers
            )
            
            if total_requests == 0:
                return {w: 0.0 for w in self.workers}
            
            return {
                w: (self.worker_stats[w]['active_requests'] / total_requests) * 100
                for w in self.workers
            }


class AutoScaler:
    """Main auto-scaling coordinator"""
    
    def __init__(
        self,
        metrics_window: int = 300,
        evaluation_interval: int = 30,
        enable_predictive_scaling: bool = True
    ):
        self.metrics_collector = MetricsCollector()
        self.scaling_engine = ScalingEngine()
        self.load_balancer = LoadBalancer()
        
        self.metrics_window = metrics_window
        self.evaluation_interval = evaluation_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        
        self.running = False
        self.evaluation_task = None
        
        # Callbacks for scaling actions
        self.scaling_callbacks = {}
        
    def register_scaling_callback(
        self, 
        resource_type: ResourceType, 
        callback: Callable
    ):
        """Register callback for scaling actions"""
        self.scaling_callbacks[resource_type] = callback
    
    async def start(self):
        """Start auto-scaling system"""
        if self.running:
            return
            
        self.running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop auto-scaling system"""
        self.running = False
        
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaler stopped")
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add new metrics for scaling evaluation"""
        self.metrics_collector.add_metrics(metrics)
    
    async def _evaluation_loop(self):
        """Main evaluation loop"""
        while self.running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in scaling evaluation: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_scaling(self):
        """Evaluate current metrics and execute scaling actions"""
        # Get current average metrics
        avg_metrics = self.metrics_collector.get_average_metrics(self.metrics_window)
        if not avg_metrics:
            return
        
        # Add predictive metrics if enabled
        if self.enable_predictive_scaling:
            predicted_metrics = self._predict_future_load(avg_metrics)
            avg_metrics.update(predicted_metrics)
        
        # Evaluate scaling rules
        scaling_actions = self.scaling_engine.evaluate_scaling(avg_metrics)
        
        # Execute scaling actions
        for action in scaling_actions:
            await self._execute_scaling_action(action)
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute scaling action"""
        resource_type = action['resource_type']
        callback = self.scaling_callbacks.get(resource_type)
        
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(action)
                else:
                    callback(action)
                
                logger.info(f"Executed scaling action: {action}")
            except Exception as e:
                logger.error(f"Failed to execute scaling action {action}: {e}")
        else:
            logger.warning(f"No callback registered for resource type: {resource_type}")
    
    def _predict_future_load(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Simple load prediction based on trends"""
        # Get metrics from different time windows
        recent_metrics = self.metrics_collector.get_average_metrics(60)  # 1 minute
        older_metrics = self.metrics_collector.get_average_metrics(600)   # 10 minutes
        
        if not recent_metrics or not older_metrics:
            return {}
        
        predicted = {}
        
        # Simple linear trend prediction
        for metric, current_value in current_metrics.items():
            recent_value = recent_metrics.get(metric, current_value)
            older_value = older_metrics.get(metric, current_value)
            
            # Calculate trend
            if older_value != 0:
                trend = (recent_value - older_value) / older_value
                # Predict 5 minutes ahead
                predicted_value = current_value * (1 + trend * 0.5)
                predicted[f"predicted_{metric}"] = max(0, predicted_value)
        
        return predicted
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler status"""
        avg_metrics = self.metrics_collector.get_average_metrics(self.metrics_window)
        
        return {
            'running': self.running,
            'current_metrics': avg_metrics,
            'resource_levels': self.scaling_engine.get_current_resource_levels(),
            'scaling_history': self.scaling_engine.get_scaling_history(10),
            'worker_stats': self.load_balancer.get_worker_stats(),
            'load_distribution': self.load_balancer.get_load_distribution(),
            'registered_callbacks': list(self.scaling_callbacks.keys())
        }


# Example usage functions
async def example_cpu_scaling_callback(action: Dict[str, Any]):
    """Example callback for CPU worker scaling"""
    resource_type = action['resource_type']
    old_value = action['old_value']
    new_value = action['new_value']
    
    if resource_type == ResourceType.CPU_WORKERS:
        if new_value > old_value:
            # Scale up: add workers
            logger.info(f"Scaling up CPU workers from {old_value} to {new_value}")
            # Implementation would add actual workers here
        else:
            # Scale down: remove workers
            logger.info(f"Scaling down CPU workers from {old_value} to {new_value}")
            # Implementation would remove workers here


# Global auto-scaler instance
global_autoscaler = AutoScaler(
    metrics_window=300,
    evaluation_interval=30,
    enable_predictive_scaling=True
)