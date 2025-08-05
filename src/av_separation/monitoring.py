"""
Monitoring and Telemetry for AV-Separation-Transformer
Comprehensive performance monitoring with Prometheus metrics and OpenTelemetry
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import logging

import torch
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, push_to_gateway
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(
        self,
        service_name: str = "av-separation-transformer",
        enable_prometheus: bool = True,
        enable_opentelemetry: bool = True,
        pushgateway_url: Optional[str] = None,
        jaeger_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.enable_prometheus = enable_prometheus
        self.enable_opentelemetry = enable_opentelemetry
        self.pushgateway_url = pushgateway_url
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring systems
        if enable_prometheus:
            self._setup_prometheus_metrics()
        
        if enable_opentelemetry:
            self._setup_opentelemetry(jaeger_endpoint)
        
        # System monitoring
        self._setup_system_monitoring()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        self.registry = CollectorRegistry()
        
        # Separation metrics
        self.separation_requests = Counter(
            'av_separation_requests_total',
            'Total number of separation requests',
            ['status', 'num_speakers'],
            registry=self.registry
        )
        
        self.separation_duration = Histogram(
            'av_separation_duration_seconds',
            'Duration of separation requests',
            ['num_speakers'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.separation_latency = Histogram(
            'av_separation_latency_seconds',
            'End-to-end latency of separation',
            ['stage'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference_time = Histogram(
            'model_inference_duration_seconds',
            'Model inference time',
            buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            registry=self.registry
        )
        
        self.model_batch_size = Histogram(
            'model_batch_size',
            'Batch size used for inference',
            buckets=[1, 2, 4, 8, 16, 32],
            registry=self.registry
        )
        
        # Audio/Video processing metrics
        self.audio_processing_time = Histogram(
            'audio_processing_duration_seconds',
            'Audio processing time',
            ['operation'],
            registry=self.registry
        )
        
        self.video_processing_time = Histogram(
            'video_processing_duration_seconds',
            'Video processing time',
            ['operation'],
            registry=self.registry
        )
        
        # System metrics
        self.gpu_memory_usage = Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage',
            ['device'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        # Quality metrics
        self.separation_quality = Gauge(
            'separation_quality_score',
            'Separation quality score (SI-SNR)',
            ['speaker_id'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'av_separation_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Service info
        self.service_info = Info(
            'av_separation_service_info',
            'Service information',
            registry=self.registry
        )
        
        # Set service info
        import torch
        from ..version import __version__
        
        self.service_info.info({
            'version': __version__,
            'pytorch_version': torch.__version__,
            'cuda_available': str(torch.cuda.is_available()),
            'service_name': self.service_name
        })
    
    def _setup_opentelemetry(self, jaeger_endpoint: Optional[str]):
        """Setup OpenTelemetry tracing and metrics"""
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Setup metrics
        if self.enable_prometheus:
            metric_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        
        self.meter = metrics.get_meter(__name__)
    
    def _setup_system_monitoring(self):
        """Setup system resource monitoring"""
        
        self.system_stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used': 0,
            'memory_available': 0,
            'gpu_memory_used': {},
            'gpu_memory_total': {},
            'disk_usage': 0.0
        }
    
    def _start_background_monitoring(self):
        """Start background thread for system monitoring"""
        
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.system_stats['cpu_percent'] = cpu_percent
                    
                    if self.enable_prometheus:
                        self.cpu_usage.set(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_stats['memory_percent'] = memory.percent
                    self.system_stats['memory_used'] = memory.used
                    self.system_stats['memory_available'] = memory.available
                    
                    if self.enable_prometheus:
                        self.memory_usage.labels(type='used').set(memory.used)
                        self.memory_usage.labels(type='available').set(memory.available)
                    
                    # GPU memory (if available)
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(i)
                            reserved = torch.cuda.memory_reserved(i)
                            
                            self.system_stats['gpu_memory_used'][i] = allocated
                            
                            if self.enable_prometheus:
                                self.gpu_memory_usage.labels(device=f'cuda:{i}').set(allocated)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.system_stats['disk_usage'] = disk.percent
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
                
                time.sleep(30)  # Update every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    @contextmanager
    def trace_separation(self, num_speakers: int, duration: float):
        """Context manager for tracing separation requests"""
        
        start_time = time.time()
        status = "success"
        
        # Start OpenTelemetry span
        span = None
        if self.enable_opentelemetry:
            span = self.tracer.start_span("separation_request")
            span.set_attribute("num_speakers", num_speakers)
            span.set_attribute("duration", duration)
        
        try:
            yield
            
        except Exception as e:
            status = "error"
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
            raise
        
        finally:
            # Record metrics
            processing_time = time.time() - start_time
            
            if self.enable_prometheus:
                self.separation_requests.labels(
                    status=status, 
                    num_speakers=num_speakers
                ).inc()
                
                self.separation_duration.labels(
                    num_speakers=num_speakers
                ).observe(processing_time)
            
            if span:
                span.set_attribute("processing_time", processing_time)
                span.end()
    
    @contextmanager
    def trace_component(self, component: str, operation: str):
        """Context manager for tracing individual components"""
        
        start_time = time.time()
        
        span = None
        if self.enable_opentelemetry:
            span = self.tracer.start_span(f"{component}_{operation}")
        
        try:
            yield
            
        except Exception as e:
            if self.enable_prometheus:
                self.error_count.labels(
                    error_type=type(e).__name__,
                    component=component
                ).inc()
            
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
            raise
        
        finally:
            processing_time = time.time() - start_time
            
            # Record component-specific metrics
            if self.enable_prometheus:
                if component == "audio":
                    self.audio_processing_time.labels(operation=operation).observe(processing_time)
                elif component == "video":
                    self.video_processing_time.labels(operation=operation).observe(processing_time)
                elif component == "model":
                    self.model_inference_time.observe(processing_time)
            
            if span:
                span.set_attribute("processing_time", processing_time)
                span.end()
    
    def record_separation_quality(self, speaker_id: int, si_snr: float):
        """Record separation quality metrics"""
        
        if self.enable_prometheus:
            self.separation_quality.labels(speaker_id=speaker_id).set(si_snr)
    
    def record_batch_size(self, batch_size: int):
        """Record model batch size"""
        
        if self.enable_prometheus:
            self.model_batch_size.observe(batch_size)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'system': self.system_stats.copy(),
            'service': {
                'name': self.service_name,
                'uptime': time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
        }
        
        # Check for issues
        if self.system_stats['cpu_percent'] > 90:
            health['status'] = 'degraded'
            health['issues'] = health.get('issues', [])
            health['issues'].append('High CPU usage')
        
        if self.system_stats['memory_percent'] > 90:
            health['status'] = 'degraded'
            health['issues'] = health.get('issues', [])
            health['issues'].append('High memory usage')
        
        return health
    
    def push_metrics(self):
        """Push metrics to Pushgateway if configured"""
        
        if self.pushgateway_url and self.enable_prometheus:
            try:
                push_to_gateway(
                    self.pushgateway_url,
                    job=self.service_name,
                    registry=self.registry
                )
            except Exception as e:
                self.logger.error(f"Failed to push metrics: {e}")


# Decorator for automatic monitoring
def monitor_function(
    component: str,
    operation: str,
    monitor: Optional[PerformanceMonitor] = None
):
    """
    Decorator to automatically monitor function performance
    
    Args:
        component: Component name (e.g., 'audio', 'video', 'model')
        operation: Operation name (e.g., 'load', 'process', 'inference')
        monitor: PerformanceMonitor instance
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if monitor:
                with monitor.trace_component(component, operation):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class AlertManager:
    """
    Alert management system for critical issues
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.webhook_url = webhook_url
        self.email_config = email_config
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'error_rate': 0.1,  # 10% error rate
            'latency_p99': 5.0,  # 5 second P99 latency
        }
        
        # Alert cooldown (prevent spam)
        self.alert_cooldown = {}
        self.cooldown_period = 300  # 5 minutes
    
    def check_and_alert(self, metrics: Dict[str, Any]):
        """Check metrics and send alerts if thresholds are exceeded"""
        
        current_time = time.time()
        
        # CPU usage alert
        if metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            self._send_alert_if_not_in_cooldown(
                'high_cpu',
                f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                current_time
            )
        
        # Memory usage alert
        if metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
            self._send_alert_if_not_in_cooldown(
                'high_memory',
                f"High memory usage: {metrics['memory_percent']:.1f}%",
                current_time
            )
        
        # GPU memory alerts
        for device, usage in metrics.get('gpu_memory_used', {}).items():
            if device in metrics.get('gpu_memory_total', {}):
                total = metrics['gpu_memory_total'][device]
                percent = (usage / total) * 100 if total > 0 else 0
                
                if percent > self.thresholds['gpu_memory_percent']:
                    self._send_alert_if_not_in_cooldown(
                        f'high_gpu_memory_{device}',
                        f"High GPU memory usage on {device}: {percent:.1f}%",
                        current_time
                    )
    
    def _send_alert_if_not_in_cooldown(self, alert_type: str, message: str, current_time: float):
        """Send alert if not in cooldown period"""
        
        last_alert_time = self.alert_cooldown.get(alert_type, 0)
        
        if current_time - last_alert_time > self.cooldown_period:
            self._send_alert(alert_type, message)
            self.alert_cooldown[alert_type] = current_time
    
    def _send_alert(self, alert_type: str, message: str):
        """Send alert via configured channels"""
        
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Send webhook alert
        if self.webhook_url:
            self._send_webhook_alert(alert_type, message)
        
        # Send email alert
        if self.email_config:
            self._send_email_alert(alert_type, message)
    
    def _send_webhook_alert(self, alert_type: str, message: str):
        """Send alert via webhook"""
        
        try:
            import requests
            
            payload = {
                'alert_type': alert_type,
                'message': message,
                'timestamp': time.time(),
                'service': 'av-separation-transformer'
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_email_alert(self, alert_type: str, message: str):
        """Send alert via email"""
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"AV-Separation Alert: {alert_type}"
            
            body = f"""
            Alert Type: {alert_type}
            Message: {message}
            Timestamp: {time.ctime()}
            Service: av-separation-transformer
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from'], self.email_config['to'], text)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> Optional[PerformanceMonitor]:
    """Get global monitor instance"""
    return _global_monitor


def initialize_monitoring(
    service_name: str = "av-separation-transformer",
    enable_prometheus: bool = True,
    enable_opentelemetry: bool = True,
    pushgateway_url: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None
) -> PerformanceMonitor:
    """Initialize global monitoring"""
    
    global _global_monitor
    
    _global_monitor = PerformanceMonitor(
        service_name=service_name,
        enable_prometheus=enable_prometheus,
        enable_opentelemetry=enable_opentelemetry,
        pushgateway_url=pushgateway_url,
        jaeger_endpoint=jaeger_endpoint
    )
    
    return _global_monitor