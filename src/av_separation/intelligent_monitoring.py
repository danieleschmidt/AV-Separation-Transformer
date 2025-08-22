import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import json
from datetime import datetime, timedelta
import psutil
import numpy as np
from pathlib import Path
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from contextlib import asynccontextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 1  # How long condition must persist
    cooldown_minutes: int = 15  # Minimum time between alerts
    description: str = ""
    actions: List[str] = field(default_factory=list)  # Actions to take


@dataclass
class Alert:
    """An active alert."""
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    alert_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    metrics_retention_hours: int = 24
    alert_check_interval_seconds: int = 30
    enable_prometheus: bool = PROMETHEUS_AVAILABLE
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    email_config: Dict[str, str] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    log_file: Optional[str] = None
    enable_predictive_alerts: bool = True


class IntelligentMonitoringSystem:
    """Comprehensive monitoring system with predictive capabilities."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata = {}
        
        # Alert system
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.last_alert_times = defaultdict(lambda: datetime.min)
        
        # Prometheus integration
        if self.config.enable_prometheus:
            self._setup_prometheus()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Predictive modeling
        self.trend_analyzers = {}
        self.anomaly_detectors = {}
        
        # Performance baselines
        self.baselines = {}
        
        # System metrics
        self.system_metrics_collector = SystemMetricsCollector()
        
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Intelligent monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Intelligent monitoring system stopped")
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a metric value."""
        timestamp = time.time()
        
        metric_entry = {
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {},
            'type': metric_type.value
        }
        
        self.metrics[name].append(metric_entry)
        
        # Store metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                'type': metric_type.value,
                'first_seen': timestamp,
                'tags_seen': set()
            }
        
        # Update Prometheus if available
        if self.config.enable_prometheus and hasattr(self, 'prometheus_metrics'):
            self._update_prometheus_metric(name, value, tags, metric_type)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def get_metric_stats(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a metric over a time period."""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_values = [
            entry['value'] for entry in self.metrics[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'mean': np.mean(recent_values),
            'median': np.median(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99)
        }
    
    def detect_anomalies(self, metric_name: str, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data using statistical analysis."""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_entries = [
            entry for entry in self.metrics[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(recent_entries) < 10:
            return []
        
        values = np.array([entry['value'] for entry in recent_entries])
        timestamps = np.array([entry['timestamp'] for entry in recent_entries])
        
        # Z-score based anomaly detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        z_scores = np.abs((values - mean_val) / std_val)
        anomaly_threshold = 3.0  # 3 standard deviations
        
        anomalies = []
        for i, (z_score, value, timestamp) in enumerate(zip(z_scores, values, timestamps)):
            if z_score > anomaly_threshold:
                anomalies.append({
                    'timestamp': timestamp,
                    'value': value,
                    'z_score': z_score,
                    'severity': 'high' if z_score > 4.0 else 'medium',
                    'deviation_from_mean': abs(value - mean_val)
                })
        
        return anomalies
    
    def predict_metric_trend(
        self, 
        metric_name: str, 
        prediction_minutes: int = 30,
        history_minutes: int = 120
    ) -> Dict[str, Any]:
        """Predict future metric values using linear regression."""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - (history_minutes * 60)
        recent_entries = [
            entry for entry in self.metrics[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(recent_entries) < 10:
            return {}
        
        timestamps = np.array([entry['timestamp'] for entry in recent_entries])
        values = np.array([entry['value'] for entry in recent_entries])
        
        # Normalize timestamps
        start_time = timestamps[0]
        normalized_times = timestamps - start_time
        
        # Simple linear regression
        coeffs = np.polyfit(normalized_times, values, 1)
        slope, intercept = coeffs
        
        # Predict future values
        future_time = prediction_minutes * 60
        predicted_value = slope * future_time + intercept
        
        # Calculate confidence based on recent variance
        recent_variance = np.var(values[-min(20, len(values)):])  # Last 20 points or all
        confidence = max(0.1, min(0.9, 1.0 / (1.0 + recent_variance)))
        
        return {
            'predicted_value': predicted_value,
            'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
            'slope': slope,
            'confidence': confidence,
            'prediction_horizon_minutes': prediction_minutes,
            'current_value': values[-1] if len(values) > 0 else None
        }
    
    def create_baseline(self, metric_name: str, duration_hours: int = 24):
        """Create a performance baseline for a metric."""
        cutoff_time = time.time() - (duration_hours * 3600)
        historical_values = [
            entry['value'] for entry in self.metrics[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(historical_values) < 100:  # Need sufficient data
            self.logger.warning(f"Insufficient data to create baseline for {metric_name}")
            return
        
        baseline = {
            'metric_name': metric_name,
            'created_at': datetime.utcnow(),
            'duration_hours': duration_hours,
            'sample_count': len(historical_values),
            'mean': np.mean(historical_values),
            'std': np.std(historical_values),
            'p25': np.percentile(historical_values, 25),
            'p50': np.percentile(historical_values, 50),
            'p75': np.percentile(historical_values, 75),
            'p95': np.percentile(historical_values, 95),
            'p99': np.percentile(historical_values, 99),
            'min': np.min(historical_values),
            'max': np.max(historical_values)
        }
        
        self.baselines[metric_name] = baseline
        self.logger.info(f"Created baseline for {metric_name}")
    
    def compare_to_baseline(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """Compare current value to established baseline."""
        if metric_name not in self.baselines:
            return {'error': 'No baseline available'}
        
        baseline = self.baselines[metric_name]
        
        # Calculate percentile ranking
        if current_value <= baseline['p25']:
            percentile_range = 'bottom_quartile'
        elif current_value <= baseline['p50']:
            percentile_range = 'second_quartile'
        elif current_value <= baseline['p75']:
            percentile_range = 'third_quartile'
        elif current_value <= baseline['p95']:
            percentile_range = 'top_quartile'
        else:
            percentile_range = 'outlier_high'
        
        # Calculate deviation
        deviation_from_mean = abs(current_value - baseline['mean']) / baseline['std']
        
        return {
            'percentile_range': percentile_range,
            'deviation_sigma': deviation_from_mean,
            'is_anomaly': deviation_from_mean > 2.0,
            'baseline_mean': baseline['mean'],
            'baseline_std': baseline['std'],
            'current_value': current_value,
            'baseline_age_hours': (datetime.utcnow() - baseline['created_at']).total_seconds() / 3600
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alert rules
                self._check_alert_rules()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Update predictive models
                self._update_predictive_models()
                
                time.sleep(self.config.alert_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.record_metric('system.cpu_percent', cpu_percent)
            self.record_metric('system.memory_percent', memory.percent)
            self.record_metric('system.memory_available_gb', memory.available / (1024**3))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk_percent', disk.percent)
            
            # GPU metrics if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    self.record_metric(f'gpu.{i}.memory_allocated_gb', memory_allocated)
                    self.record_metric(f'gpu.{i}.memory_reserved_gb', memory_reserved)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if necessary."""
        current_time = datetime.utcnow()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Get recent metric values
                recent_stats = self.get_metric_stats(
                    rule.metric_name, 
                    rule.duration_minutes
                )
                
                if not recent_stats:
                    continue
                
                current_value = recent_stats['mean']
                
                # Check condition
                condition_met = False
                if rule.condition == 'gt' and current_value > rule.threshold:
                    condition_met = True
                elif rule.condition == 'lt' and current_value < rule.threshold:
                    condition_met = True
                elif rule.condition == 'eq' and abs(current_value - rule.threshold) < 0.001:
                    condition_met = True
                elif rule.condition == 'ne' and abs(current_value - rule.threshold) >= 0.001:
                    condition_met = True
                
                # Handle alert
                if condition_met:
                    self._trigger_alert(rule, current_value)
                else:
                    self._resolve_alert(rule_name)
                    
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_key = rule.name
        current_time = datetime.utcnow()
        
        # Check cooldown period
        if current_time - self.last_alert_times[alert_key] < timedelta(minutes=rule.cooldown_minutes):
            return
        
        # Create or update alert
        if alert_key not in self.active_alerts:
            alert = Alert(
                rule=rule,
                triggered_at=current_time,
                current_value=current_value
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            self.last_alert_times[alert_key] = current_time
            
            # Execute alert actions
            self._execute_alert_actions(alert)
            
            self.logger.warning(
                f"ALERT TRIGGERED: {rule.name} - {rule.description} "
                f"(current: {current_value}, threshold: {rule.threshold})"
            )
        else:
            # Update existing alert
            self.active_alerts[alert_key].current_value = current_value
    
    def _resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved_at = datetime.utcnow()
            
            self.logger.info(f"ALERT RESOLVED: {alert.rule.name}")
            
            del self.active_alerts[alert_key]
    
    def _execute_alert_actions(self, alert: Alert):
        """Execute actions for an alert."""
        for action in alert.rule.actions:
            try:
                if action == 'email' and self.config.enable_email_alerts:
                    self._send_email_alert(alert)
                elif action == 'webhook' and self.config.enable_webhook_alerts:
                    self._send_webhook_alert(alert)
                elif action == 'log':
                    self._log_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Failed to execute alert action {action}: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        if not self.config.email_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_config['from']
            msg['To'] = self.config.email_config['to']
            msg['Subject'] = f"[{alert.rule.severity.value.upper()}] {alert.rule.name}"
            
            body = f"""
            Alert: {alert.rule.name}
            Severity: {alert.rule.severity.value}
            Description: {alert.rule.description}
            Current Value: {alert.current_value}
            Threshold: {alert.rule.threshold}
            Triggered At: {alert.triggered_at}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_config['smtp_server'])
            if 'username' in self.config.email_config:
                server.login(
                    self.config.email_config['username'],
                    self.config.email_config['password']
                )
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert."""
        if not self.config.webhook_url or not REQUESTS_AVAILABLE:
            return
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'rule_name': alert.rule.name,
                'severity': alert.rule.severity.value,
                'description': alert.rule.description,
                'current_value': alert.current_value,
                'threshold': alert.rule.threshold,
                'triggered_at': alert.triggered_at.isoformat(),
                'metric_name': alert.rule.metric_name
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        if not self.config.log_file:
            return
        
        try:
            alert_data = {
                'timestamp': alert.triggered_at.isoformat(),
                'alert_id': alert.alert_id,
                'rule_name': alert.rule.name,
                'severity': alert.rule.severity.value,
                'current_value': alert.current_value,
                'threshold': alert.rule.threshold
            }
            
            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metric data."""
        cutoff_time = time.time() - (self.config.metrics_retention_hours * 3600)
        
        for metric_name, entries in self.metrics.items():
            # Remove old entries
            while entries and entries[0]['timestamp'] < cutoff_time:
                entries.popleft()
    
    def _update_predictive_models(self):
        """Update predictive models for key metrics."""
        # This is a simplified version - in production, you'd use more sophisticated models
        key_metrics = ['system.cpu_percent', 'system.memory_percent', 'response_time_ms']
        
        for metric_name in key_metrics:
            if metric_name in self.metrics and len(self.metrics[metric_name]) > 50:
                try:
                    # Create or update baseline
                    if metric_name not in self.baselines:
                        self.create_baseline(metric_name, duration_hours=12)
                    
                    # Detect anomalies
                    anomalies = self.detect_anomalies(metric_name, window_minutes=60)
                    if anomalies:
                        self.logger.info(f"Detected {len(anomalies)} anomalies in {metric_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error updating predictive model for {metric_name}: {e}")
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics = {
            'counters': {},
            'gauges': {},
            'histograms': {}
        }
    
    def _update_prometheus_metric(self, name: str, value: float, tags: Dict, metric_type: MetricType):
        """Update Prometheus metric."""
        if not hasattr(self, 'prometheus_metrics'):
            return
        
        try:
            if metric_type == MetricType.COUNTER:
                if name not in self.prometheus_metrics['counters']:
                    self.prometheus_metrics['counters'][name] = Counter(
                        name, f'Counter for {name}', list(tags.keys()), registry=self.prometheus_registry
                    )
                self.prometheus_metrics['counters'][name].labels(**tags).inc(value)
                
            elif metric_type == MetricType.GAUGE:
                if name not in self.prometheus_metrics['gauges']:
                    self.prometheus_metrics['gauges'][name] = Gauge(
                        name, f'Gauge for {name}', list(tags.keys()), registry=self.prometheus_registry
                    )
                self.prometheus_metrics['gauges'][name].labels(**tags).set(value)
                
        except Exception as e:
            self.logger.error(f"Error updating Prometheus metric {name}: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in exposition format."""
        if hasattr(self, 'prometheus_registry'):
            return generate_latest(self.prometheus_registry)
        return ""
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        # Active alerts
        active_alerts_data = [
            {
                'rule_name': alert.rule.name,
                'severity': alert.rule.severity.value,
                'current_value': alert.current_value,
                'threshold': alert.rule.threshold,
                'triggered_at': alert.triggered_at.isoformat(),
                'duration_minutes': (datetime.utcnow() - alert.triggered_at).total_seconds() / 60
            }
            for alert in self.active_alerts.values()
        ]
        
        # System health overview
        system_health = {}
        for metric in ['system.cpu_percent', 'system.memory_percent', 'system.disk_percent']:
            stats = self.get_metric_stats(metric, duration_minutes=5)
            if stats:
                system_health[metric] = {
                    'current': stats['mean'],
                    'status': 'healthy' if stats['mean'] < 80 else 'warning' if stats['mean'] < 95 else 'critical'
                }
        
        # Performance trends
        performance_trends = {}
        for metric in ['response_time_ms', 'throughput_rps', 'error_rate']:
            trend = self.predict_metric_trend(metric, prediction_minutes=30)
            if trend:
                performance_trends[metric] = trend
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_alerts': active_alerts_data,
            'system_health': system_health,
            'performance_trends': performance_trends,
            'total_metrics': len(self.metrics),
            'monitoring_uptime_hours': (time.time() - getattr(self, 'start_time', time.time())) / 3600
        }


class SystemMetricsCollector:
    """Collect system-level metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_cpu_times = None
        self.last_network_io = None
    
    def collect_all_metrics(self) -> Dict[str, float]:
        """Collect all available system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['system.cpu_percent'] = cpu_percent
            
            cpu_times = psutil.cpu_times()
            if self.last_cpu_times:
                idle_delta = cpu_times.idle - self.last_cpu_times.idle
                total_delta = sum(cpu_times) - sum(self.last_cpu_times)
                if total_delta > 0:
                    cpu_usage = 100 * (1.0 - idle_delta / total_delta)
                    metrics['system.cpu_usage_calculated'] = cpu_usage
            self.last_cpu_times = cpu_times
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['system.memory_percent'] = memory.percent
            metrics['system.memory_available_gb'] = memory.available / (1024**3)
            metrics['system.memory_used_gb'] = memory.used / (1024**3)
            
            # Swap metrics
            swap = psutil.swap_memory()
            metrics['system.swap_percent'] = swap.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['system.disk_percent'] = disk.percent
            metrics['system.disk_free_gb'] = disk.free / (1024**3)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if self.last_network_io:
                bytes_sent_delta = network_io.bytes_sent - self.last_network_io.bytes_sent
                bytes_recv_delta = network_io.bytes_recv - self.last_network_io.bytes_recv
                metrics['system.network_bytes_sent_rate'] = bytes_sent_delta
                metrics['system.network_bytes_recv_rate'] = bytes_recv_delta
            self.last_network_io = network_io
            
            # Load average (Unix/Linux only)
            try:
                load_avg = psutil.getloadavg()
                metrics['system.load_avg_1min'] = load_avg[0]
                metrics['system.load_avg_5min'] = load_avg[1]
                metrics['system.load_avg_15min'] = load_avg[2]
            except AttributeError:
                pass  # Not available on Windows
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics


# Convenience functions for common monitoring patterns

def setup_default_alerts(monitoring_system: IntelligentMonitoringSystem):
    """Setup common alert rules."""
    
    # High CPU usage
    monitoring_system.add_alert_rule(AlertRule(
        name="high_cpu_usage",
        metric_name="system.cpu_percent",
        condition="gt",
        threshold=85.0,
        severity=AlertSeverity.HIGH,
        duration_minutes=2,
        description="CPU usage is consistently high",
        actions=["log", "webhook"]
    ))
    
    # High memory usage
    monitoring_system.add_alert_rule(AlertRule(
        name="high_memory_usage",
        metric_name="system.memory_percent",
        condition="gt",
        threshold=90.0,
        severity=AlertSeverity.CRITICAL,
        duration_minutes=1,
        description="Memory usage is critically high",
        actions=["log", "email", "webhook"]
    ))
    
    # High response time
    monitoring_system.add_alert_rule(AlertRule(
        name="high_response_time",
        metric_name="response_time_ms",
        condition="gt",
        threshold=1000.0,
        severity=AlertSeverity.MEDIUM,
        duration_minutes=3,
        description="Response time is consistently high",
        actions=["log", "webhook"]
    ))
    
    # Low throughput
    monitoring_system.add_alert_rule(AlertRule(
        name="low_throughput",
        metric_name="throughput_rps",
        condition="lt",
        threshold=1.0,
        severity=AlertSeverity.MEDIUM,
        duration_minutes=5,
        description="System throughput is low",
        actions=["log"]
    ))
