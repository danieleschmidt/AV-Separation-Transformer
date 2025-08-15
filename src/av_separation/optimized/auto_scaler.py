"""
Intelligent Auto-Scaling System for Dynamic Resource Management
Kubernetes-native scaling with predictive load balancing and cost optimization.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
import logging
from collections import deque, defaultdict
import psutil
import requests
from pathlib import Path
import yaml


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class ScalingMetrics:
    """Container for scaling decision metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    request_rate: float
    queue_length: int
    response_time: float
    current_replicas: int
    target_replicas: int
    predicted_load: float


@dataclass
class ScalingDecision:
    """Container for scaling decision information."""
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    current_replicas: int
    target_replicas: int
    confidence: float
    reason: str
    cost_impact: float


class LoadPredictor:
    """
    Predictive load analysis using time series forecasting.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
        # Simple moving averages for trend detection
        self.short_term_window = 10  # 10 minutes
        self.long_term_window = 60   # 1 hour
        
        self.logger = logging.getLogger(__name__)
    
    def add_datapoint(self, timestamp: float, load: float):
        """Add new load datapoint."""
        self.load_history.append(load)
        self.time_history.append(timestamp)
    
    def predict_load(self, horizon_minutes: int = 15) -> Tuple[float, float]:
        """
        Predict load for given time horizon.
        
        Returns:
            Tuple of (predicted_load, confidence)
        """
        if len(self.load_history) < self.short_term_window:
            return np.mean(self.load_history) if self.load_history else 0.0, 0.5
        
        # Convert to numpy arrays
        loads = np.array(list(self.load_history))
        times = np.array(list(self.time_history))
        
        # Calculate moving averages
        short_term_avg = np.mean(loads[-self.short_term_window:])
        long_term_avg = np.mean(loads[-min(len(loads), self.long_term_window):])
        
        # Detect trend
        if len(loads) >= 5:
            recent_trend = np.polyfit(range(5), loads[-5:], 1)[0]
        else:
            recent_trend = 0
        
        # Simple linear extrapolation
        current_load = loads[-1]
        trend_factor = recent_trend * horizon_minutes
        
        # Seasonal adjustment (basic daily pattern)
        current_hour = time.localtime().tm_hour
        seasonal_factor = self._get_seasonal_factor(current_hour)
        
        # Combine predictions
        predicted_load = (
            current_load * 0.4 +          # Current load weight
            short_term_avg * 0.3 +        # Short-term trend
            long_term_avg * 0.2 +         # Long-term baseline
            trend_factor * 0.1            # Trend extrapolation
        ) * seasonal_factor
        
        # Calculate confidence based on load stability
        load_variance = np.var(loads[-min(len(loads), 20):])
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + load_variance)))
        
        return max(0.0, predicted_load), confidence
    
    def _get_seasonal_factor(self, hour: int) -> float:
        """Get seasonal adjustment factor based on hour of day."""
        # Simple business hours pattern
        if 9 <= hour <= 17:
            return 1.2  # Business hours - higher load
        elif 22 <= hour or hour <= 6:
            return 0.6  # Night hours - lower load
        else:
            return 0.9  # Off-peak hours
    
    def detect_anomaly(self, current_load: float) -> Tuple[bool, float]:
        """
        Detect if current load is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if len(self.load_history) < 20:
            return False, 0.0
        
        loads = np.array(list(self.load_history))
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        if std_load == 0:
            return False, 0.0
        
        # Z-score based anomaly detection
        z_score = abs(current_load - mean_load) / std_load
        is_anomaly = z_score > 2.5  # 2.5 standard deviations
        
        return is_anomaly, z_score


class CostOptimizer:
    """
    Cost optimization for cloud resource scaling decisions.
    """
    
    def __init__(self):
        # Example cost per hour for different instance types
        self.instance_costs = {
            'cpu_small': 0.05,    # $0.05/hour
            'cpu_medium': 0.10,   # $0.10/hour
            'cpu_large': 0.20,    # $0.20/hour
            'gpu_small': 0.50,    # $0.50/hour
            'gpu_large': 1.50,    # $1.50/hour
        }
        
        # Performance coefficients (requests per hour per instance)
        self.performance_coefficients = {
            'cpu_small': 1000,
            'cpu_medium': 2500,
            'cpu_large': 5000,
            'gpu_small': 3000,
            'gpu_large': 8000,
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_cost_efficiency(self, instance_type: str, replicas: int, 
                                 expected_load: float) -> float:
        """Calculate cost efficiency for given configuration."""
        total_cost = self.instance_costs[instance_type] * replicas
        total_capacity = self.performance_coefficients[instance_type] * replicas
        
        if total_capacity == 0:
            return float('inf')
        
        # Cost per unit of load handled
        cost_efficiency = total_cost / min(total_capacity, expected_load)
        
        return cost_efficiency
    
    def recommend_instance_mix(self, expected_load: float, budget_limit: float = None) -> Dict[str, int]:
        """
        Recommend optimal instance mix for expected load.
        
        Returns:
            Dictionary mapping instance type to recommended count
        """
        best_config = {}
        best_cost_efficiency = float('inf')
        
        # Try different configurations
        for instance_type in self.instance_costs:
            capacity_per_instance = self.performance_coefficients[instance_type]
            min_instances = max(1, int(np.ceil(expected_load / capacity_per_instance)))
            
            # Try configurations with some buffer
            for replicas in range(min_instances, min_instances + 3):
                total_cost = self.instance_costs[instance_type] * replicas
                
                if budget_limit and total_cost > budget_limit:
                    continue
                
                cost_efficiency = self.calculate_cost_efficiency(instance_type, replicas, expected_load)
                
                if cost_efficiency < best_cost_efficiency:
                    best_cost_efficiency = cost_efficiency
                    best_config = {instance_type: replicas}
        
        return best_config
    
    def calculate_scaling_cost_impact(self, current_config: Dict[str, int], 
                                    new_config: Dict[str, int], duration_hours: float = 1.0) -> float:
        """Calculate cost impact of scaling decision."""
        current_cost = sum(
            self.instance_costs[instance_type] * count * duration_hours
            for instance_type, count in current_config.items()
        )
        
        new_cost = sum(
            self.instance_costs[instance_type] * count * duration_hours
            for instance_type, count in new_config.items()
        )
        
        return new_cost - current_cost


class KubernetesScaler:
    """
    Kubernetes-native scaling operations.
    """
    
    def __init__(self, namespace: str = "default", deployment_name: str = "av-separation"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.api_server = None
        self.auth_token = None
        
        self.logger = logging.getLogger(__name__)
        
        # Load Kubernetes configuration
        self._load_k8s_config()
    
    def _load_k8s_config(self):
        """Load Kubernetes configuration."""
        try:
            # Try to load from service account token (in-cluster)
            token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
            if token_path.exists():
                with open(token_path) as f:
                    self.auth_token = f.read().strip()
                
                self.api_server = "https://kubernetes.default.svc"
                self.logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                # Try to load from kubeconfig
                import os
                kubeconfig_path = os.path.expanduser("~/.kube/config")
                if Path(kubeconfig_path).exists():
                    # For simplicity, assume kubectl proxy is running
                    self.api_server = "http://localhost:8001"
                    self.logger.info("Using kubectl proxy for Kubernetes API")
                else:
                    self.logger.warning("No Kubernetes configuration found")
        
        except Exception as e:
            self.logger.error(f"Failed to load Kubernetes configuration: {e}")
    
    def get_current_replicas(self) -> int:
        """Get current number of replicas for deployment."""
        if not self.api_server:
            return 1  # Default fallback
        
        try:
            url = f"{self.api_server}/apis/apps/v1/namespaces/{self.namespace}/deployments/{self.deployment_name}"
            headers = {}
            
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("spec", {}).get("replicas", 1)
            else:
                self.logger.warning(f"Failed to get replicas: {response.status_code}")
                return 1
        
        except Exception as e:
            self.logger.error(f"Error getting current replicas: {e}")
            return 1
    
    def scale_deployment(self, target_replicas: int) -> bool:
        """Scale deployment to target number of replicas."""
        if not self.api_server:
            self.logger.warning("No Kubernetes API server configured")
            return False
        
        try:
            url = f"{self.api_server}/apis/apps/v1/namespaces/{self.namespace}/deployments/{self.deployment_name}/scale"
            headers = {"Content-Type": "application/json"}
            
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            # Patch request to update replicas
            patch_data = {
                "spec": {
                    "replicas": target_replicas
                }
            }
            
            response = requests.patch(url, headers=headers, json=patch_data, verify=False, timeout=10)
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Successfully scaled deployment to {target_replicas} replicas")
                return True
            else:
                self.logger.error(f"Failed to scale deployment: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error scaling deployment: {e}")
            return False
    
    def get_pod_metrics(self) -> Dict[str, float]:
        """Get metrics for pods in deployment."""
        # This would integrate with Kubernetes metrics server
        # For now, return mock metrics
        return {
            'cpu_utilization': np.random.uniform(20, 80),
            'memory_utilization': np.random.uniform(30, 70),
            'request_rate': np.random.uniform(10, 100)
        }


class IntelligentAutoScaler:
    """
    Main auto-scaling system integrating all components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.load_predictor = LoadPredictor()
        self.cost_optimizer = CostOptimizer()
        self.k8s_scaler = KubernetesScaler(
            namespace=self.config.get('namespace', 'default'),
            deployment_name=self.config.get('deployment_name', 'av-separation')
        )
        
        # Scaling parameters
        self.min_replicas = self.config.get('min_replicas', 1)
        self.max_replicas = self.config.get('max_replicas', 10)
        self.target_cpu_utilization = self.config.get('target_cpu_utilization', 70.0)
        self.target_memory_utilization = self.config.get('target_memory_utilization', 80.0)
        self.scale_up_threshold = self.config.get('scale_up_threshold', 80.0)
        self.scale_down_threshold = self.config.get('scale_down_threshold', 50.0)
        
        # Scaling behavior
        self.scale_up_cooldown = self.config.get('scale_up_cooldown', 300)  # 5 minutes
        self.scale_down_cooldown = self.config.get('scale_down_cooldown', 600)  # 10 minutes
        
        # State tracking
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.metrics_history = deque(maxlen=1000)
        self.scaling_decisions = deque(maxlen=100)
        
        # Control flags
        self.auto_scaling_enabled = True
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Get Kubernetes metrics
        k8s_metrics = self.k8s_scaler.get_pod_metrics()
        
        # GPU metrics (if available)
        gpu_utilization = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = gpu_info.gpu
        except:
            pass
        
        # Current replicas
        current_replicas = self.k8s_scaler.get_current_replicas()
        
        # Predict load and calculate target replicas
        predicted_load, _ = self.load_predictor.predict_load()
        target_replicas = self._calculate_target_replicas(cpu_percent, memory_percent, predicted_load)
        
        metrics = ScalingMetrics(
            timestamp=current_time,
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            gpu_utilization=gpu_utilization,
            request_rate=k8s_metrics.get('request_rate', 0),
            queue_length=0,  # Would need queue metrics
            response_time=0.1,  # Would need response time metrics
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            predicted_load=predicted_load
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        self.load_predictor.add_datapoint(current_time, cpu_percent)
        
        return metrics
    
    def _calculate_target_replicas(self, cpu_utilization: float, memory_utilization: float,
                                  predicted_load: float) -> int:
        """Calculate target number of replicas based on metrics."""
        # CPU-based scaling
        cpu_target = max(1, int(np.ceil(cpu_utilization / self.target_cpu_utilization)))
        
        # Memory-based scaling
        memory_target = max(1, int(np.ceil(memory_utilization / self.target_memory_utilization)))
        
        # Load-based scaling
        load_target = max(1, int(np.ceil(predicted_load / 50.0)))  # Assume 50% base load per replica
        
        # Take the maximum requirement
        target_replicas = max(cpu_target, memory_target, load_target)
        
        # Apply min/max constraints
        target_replicas = max(self.min_replicas, min(self.max_replicas, target_replicas))
        
        return target_replicas
    
    def make_scaling_decision(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision based on metrics."""
        current_time = time.time()
        current_replicas = metrics.current_replicas
        target_replicas = metrics.target_replicas
        
        # Check if scaling is needed
        if current_replicas == target_replicas:
            return None
        
        # Determine scaling direction
        if target_replicas > current_replicas:
            direction = ScalingDirection.UP
            cooldown_period = self.scale_up_cooldown
            last_scale_time = self.last_scale_up
        else:
            direction = ScalingDirection.DOWN
            cooldown_period = self.scale_down_cooldown
            last_scale_time = self.last_scale_down
        
        # Check cooldown period
        if current_time - last_scale_time < cooldown_period:
            return None
        
        # Determine primary trigger
        trigger = ScalingTrigger.CPU_UTILIZATION
        confidence = 0.7
        reason = f"CPU utilization: {metrics.cpu_utilization:.1f}%"
        
        if metrics.memory_utilization > self.scale_up_threshold:
            trigger = ScalingTrigger.MEMORY_UTILIZATION
            reason = f"Memory utilization: {metrics.memory_utilization:.1f}%"
        
        # Check for anomalies
        is_anomaly, anomaly_score = self.load_predictor.detect_anomaly(metrics.cpu_utilization)
        if is_anomaly:
            confidence = min(0.9, confidence + anomaly_score * 0.1)
            reason += f" (anomaly detected: {anomaly_score:.2f})"
        
        # Cost impact calculation
        current_config = {'cpu_medium': current_replicas}
        new_config = {'cpu_medium': target_replicas}
        cost_impact = self.cost_optimizer.calculate_scaling_cost_impact(current_config, new_config)
        
        decision = ScalingDecision(
            timestamp=current_time,
            direction=direction,
            trigger=trigger,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            confidence=confidence,
            reason=reason,
            cost_impact=cost_impact
        )
        
        return decision
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        if not self.auto_scaling_enabled:
            self.logger.info("Auto-scaling is disabled, skipping scaling decision")
            return False
        
        success = self.k8s_scaler.scale_deployment(decision.target_replicas)
        
        if success:
            # Update last scale time
            if decision.direction == ScalingDirection.UP:
                self.last_scale_up = decision.timestamp
            else:
                self.last_scale_down = decision.timestamp
            
            self.logger.info(
                f"Scaled {decision.direction.value}: {decision.current_replicas} -> {decision.target_replicas} "
                f"(Trigger: {decision.trigger.value}, Confidence: {decision.confidence:.2f}, "
                f"Cost impact: ${decision.cost_impact:.2f})"
            )
        
        # Record decision
        self.scaling_decisions.append(decision)
        
        return success
    
    def start_monitoring(self):
        """Start continuous monitoring and auto-scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and auto-scaling."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Make scaling decision
                decision = self.make_scaling_decision(metrics)
                
                if decision:
                    self.execute_scaling_decision(decision)
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        if not self.metrics_history:
            return {"status": "no_metrics"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "monitoring_active": self.monitoring_active,
            "current_replicas": latest_metrics.current_replicas,
            "target_replicas": latest_metrics.target_replicas,
            "cpu_utilization": latest_metrics.cpu_utilization,
            "memory_utilization": latest_metrics.memory_utilization,
            "predicted_load": latest_metrics.predicted_load,
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "recent_decisions": len(self.scaling_decisions)
        }
    
    def generate_scaling_report(self) -> str:
        """Generate comprehensive scaling report."""
        status = self.get_scaling_status()
        
        report_lines = [
            "=== AUTO-SCALING REPORT ===",
            f"Status: {'Active' if status['monitoring_active'] else 'Inactive'}",
            f"Current Replicas: {status.get('current_replicas', 'Unknown')}",
            f"Target Replicas: {status.get('target_replicas', 'Unknown')}",
            f"CPU Utilization: {status.get('cpu_utilization', 0):.1f}%",
            f"Memory Utilization: {status.get('memory_utilization', 0):.1f}%",
            f"Predicted Load: {status.get('predicted_load', 0):.1f}",
            "",
            "Recent Scaling Decisions:",
        ]
        
        for decision in list(self.scaling_decisions)[-5:]:
            report_lines.append(
                f"  - {decision.direction.value.upper()}: {decision.current_replicas} -> {decision.target_replicas} "
                f"({decision.trigger.value}, confidence: {decision.confidence:.2f})"
            )
        
        return "\n".join(report_lines)


# Global auto-scaler instance
global_auto_scaler = IntelligentAutoScaler()


def auto_scale_endpoint(min_replicas: int = 1, max_replicas: int = 10):
    """
    Decorator for endpoints that should trigger auto-scaling monitoring.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Record request for load prediction
            global_auto_scaler.load_predictor.add_datapoint(time.time(), 1.0)
            
            # Execute function
            result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator