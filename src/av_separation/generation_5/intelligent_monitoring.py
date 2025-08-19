"""
GENERATION 5: INTELLIGENT MONITORING
AI-driven system monitoring that understands behavior patterns and predicts issues
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import warnings


@dataclass
class MonitoringAlert:
    """Intelligent monitoring alert"""
    id: str
    timestamp: str
    severity: str  # critical, warning, info
    category: str  # performance, security, resource, behavior
    title: str
    description: str
    metrics: Dict[str, float]
    predicted_impact: str
    recommended_actions: List[str]
    confidence: float
    auto_resolution: Optional[str] = None


@dataclass
class SystemBehaviorPattern:
    """Detected system behavior pattern"""
    pattern_id: str
    pattern_type: str  # periodic, anomalous, trending, stable
    description: str
    frequency: str
    metrics_involved: List[str]
    confidence: float
    prediction_accuracy: float
    first_detected: str
    last_observed: str


class MetricsAnalyzer:
    """Intelligent metrics analysis with pattern recognition"""
    
    def __init__(self):
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.patterns = []
        self.anomaly_detectors = {}
        self.trend_analyzers = {}
        
    def analyze_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze metrics for patterns, anomalies, and trends"""
        
        timestamp = datetime.now()
        
        # Store metrics
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append((timestamp, value))
        
        analysis = {
            'timestamp': timestamp.isoformat(),
            'anomalies': self._detect_anomalies(metrics),
            'trends': self._analyze_trends(metrics),
            'patterns': self._identify_patterns(metrics),
            'predictions': self._generate_predictions(metrics),
            'health_score': self._calculate_health_score(metrics)
        }
        
        return analysis
    
    def _detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict]:
        """Detect anomalies using statistical methods and ML"""
        
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if len(self.metric_history[metric_name]) < 10:
                continue  # Need more data for anomaly detection
            
            # Get historical data
            historical_values = [val for _, val in self.metric_history[metric_name]]
            
            # Statistical anomaly detection
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            
            # Z-score based detection
            z_score = abs(current_value - mean_val) / (std_val + 1e-8)
            
            if z_score > 3.0:  # 3-sigma rule
                anomaly = {
                    'metric': metric_name,
                    'current_value': current_value,
                    'expected_range': (mean_val - 2*std_val, mean_val + 2*std_val),
                    'z_score': z_score,
                    'severity': 'high' if z_score > 5 else 'medium',
                    'type': 'statistical'
                }
                anomalies.append(anomaly)
            
            # Trend-based anomaly detection
            if len(historical_values) >= 5:
                recent_trend = self._calculate_trend(historical_values[-5:])
                long_trend = self._calculate_trend(historical_values[-20:] if len(historical_values) >= 20 else historical_values)
                
                if abs(recent_trend - long_trend) > 0.1:  # Significant trend change
                    anomaly = {
                        'metric': metric_name,
                        'current_value': current_value,
                        'recent_trend': recent_trend,
                        'long_trend': long_trend,
                        'severity': 'medium',
                        'type': 'trend_change'
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _analyze_trends(self, current_metrics: Dict[str, float]) -> Dict[str, Dict]:
        """Analyze trends in metrics"""
        
        trends = {}
        
        for metric_name in current_metrics.keys():
            if len(self.metric_history[metric_name]) < 5:
                continue
            
            values = [val for _, val in self.metric_history[metric_name]]
            
            # Short-term trend (last 10 points)
            short_term_values = values[-10:]
            short_term_trend = self._calculate_trend(short_term_values)
            
            # Long-term trend (last 50 points or all available)
            long_term_values = values[-50:]
            long_term_trend = self._calculate_trend(long_term_values)
            
            # Trend classification
            trend_class = self._classify_trend(short_term_trend, long_term_trend)
            
            trends[metric_name] = {
                'short_term_trend': short_term_trend,
                'long_term_trend': long_term_trend,
                'classification': trend_class,
                'stability': self._calculate_stability(values[-20:] if len(values) >= 20 else values),
                'prediction_confidence': self._calculate_prediction_confidence(values)
            }
        
        return trends
    
    def _identify_patterns(self, current_metrics: Dict[str, float]) -> List[SystemBehaviorPattern]:
        """Identify behavioral patterns in metrics"""
        
        patterns = []
        
        for metric_name in current_metrics.keys():
            if len(self.metric_history[metric_name]) < 50:
                continue
            
            values = [val for _, val in self.metric_history[metric_name]]
            timestamps = [ts for ts, _ in self.metric_history[metric_name]]
            
            # Detect periodic patterns
            periodic_pattern = self._detect_periodic_pattern(values, timestamps)
            if periodic_pattern:
                patterns.append(periodic_pattern)
            
            # Detect cyclical patterns
            cyclical_pattern = self._detect_cyclical_pattern(values, timestamps)
            if cyclical_pattern:
                patterns.append(cyclical_pattern)
            
            # Detect threshold patterns
            threshold_pattern = self._detect_threshold_pattern(values, metric_name)
            if threshold_pattern:
                patterns.append(threshold_pattern)
        
        return patterns
    
    def _generate_predictions(self, current_metrics: Dict[str, float]) -> Dict[str, Dict]:
        """Generate predictions for future metric values"""
        
        predictions = {}
        
        for metric_name, current_value in current_metrics.items():
            if len(self.metric_history[metric_name]) < 10:
                continue
            
            values = [val for _, val in self.metric_history[metric_name]]
            
            # Simple linear prediction
            trend = self._calculate_trend(values[-10:])
            
            predictions[metric_name] = {
                'next_5_minutes': current_value + trend * 5,
                'next_15_minutes': current_value + trend * 15,
                'next_hour': current_value + trend * 60,
                'confidence': self._calculate_prediction_confidence(values),
                'trend_strength': abs(trend),
                'prediction_range': (
                    current_value + trend * 60 - np.std(values[-10:]),
                    current_value + trend * 60 + np.std(values[-10:])
                )
            }
        
        return predictions
    
    def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall system health score"""
        
        # Define optimal ranges for different metrics
        optimal_ranges = {
            'cpu_usage': (0.0, 0.7),
            'memory_usage': (0.0, 0.8),
            'latency': (0.0, 50.0),  # ms
            'throughput': (10.0, float('inf')),  # ops/sec
            'error_rate': (0.0, 0.01),
            'accuracy': (0.95, 1.0),
            'response_time': (0.0, 100.0)  # ms
        }
        
        health_scores = []
        
        for metric_name, value in metrics.items():
            if metric_name in optimal_ranges:
                min_val, max_val = optimal_ranges[metric_name]
                
                if min_val <= value <= max_val:
                    score = 1.0
                elif value < min_val:
                    score = value / min_val if min_val > 0 else 0.0
                else:  # value > max_val
                    if max_val == float('inf'):
                        score = 1.0
                    else:
                        score = max(0.0, 1.0 - (value - max_val) / max_val)
                
                health_scores.append(score)
        
        return np.mean(health_scores) if health_scores else 0.5
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _classify_trend(self, short_trend: float, long_trend: float) -> str:
        """Classify trend behavior"""
        
        threshold = 0.01
        
        if abs(short_trend) < threshold and abs(long_trend) < threshold:
            return 'stable'
        elif short_trend > threshold and long_trend > threshold:
            return 'increasing'
        elif short_trend < -threshold and long_trend < -threshold:
            return 'decreasing'
        elif abs(short_trend - long_trend) > threshold:
            return 'changing'
        else:
            return 'fluctuating'
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score (lower variance = higher stability)"""
        if len(values) < 2:
            return 1.0
        
        variance = np.var(values)
        mean_val = np.mean(values)
        
        # Coefficient of variation as stability measure
        cv = variance / (mean_val + 1e-8)
        stability = 1.0 / (1.0 + cv)
        
        return stability
    
    def _calculate_prediction_confidence(self, values: List[float]) -> float:
        """Calculate confidence in predictions based on historical stability"""
        if len(values) < 5:
            return 0.3  # Low confidence with insufficient data
        
        stability = self._calculate_stability(values[-10:])
        data_sufficiency = min(1.0, len(values) / 50.0)
        
        confidence = (stability + data_sufficiency) / 2.0
        return confidence
    
    def _detect_periodic_pattern(self, values: List[float], timestamps: List[datetime]) -> Optional[SystemBehaviorPattern]:
        """Detect periodic patterns in data"""
        
        if len(values) < 20:
            return None
        
        # Simple autocorrelation-based period detection
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(2, len(autocorr) - 2):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3 * max(autocorr):
                peaks.append(i)
        
        if peaks:
            period = peaks[0]  # First significant peak
            confidence = autocorr[period] / max(autocorr)
            
            if confidence > 0.6:  # Strong periodic pattern
                return SystemBehaviorPattern(
                    pattern_id=f"periodic_{hash(str(values[:10]))}",
                    pattern_type="periodic",
                    description=f"Periodic pattern with period {period} measurements",
                    frequency=f"every_{period}_measurements",
                    metrics_involved=[],  # Would be filled with actual metric names
                    confidence=confidence,
                    prediction_accuracy=0.8,  # Estimated
                    first_detected=timestamps[0].isoformat() if timestamps else "",
                    last_observed=timestamps[-1].isoformat() if timestamps else ""
                )
        
        return None
    
    def _detect_cyclical_pattern(self, values: List[float], timestamps: List[datetime]) -> Optional[SystemBehaviorPattern]:
        """Detect cyclical patterns (daily, weekly, etc.)"""
        
        if len(timestamps) < 48:  # Need at least 48 data points
            return None
        
        # Analyze time-based cycles
        time_diffs = [(timestamps[i] - timestamps[0]).total_seconds() / 3600 for i in range(len(timestamps))]  # Hours
        
        # Check for daily patterns (24-hour cycles)
        daily_pattern_strength = self._check_time_cycle(values, time_diffs, 24)
        
        if daily_pattern_strength > 0.7:
            return SystemBehaviorPattern(
                pattern_id=f"daily_{hash(str(values[:10]))}",
                pattern_type="cyclical",
                description="Daily cyclical pattern detected",
                frequency="daily",
                metrics_involved=[],
                confidence=daily_pattern_strength,
                prediction_accuracy=0.75,
                first_detected=timestamps[0].isoformat(),
                last_observed=timestamps[-1].isoformat()
            )
        
        return None
    
    def _detect_threshold_pattern(self, values: List[float], metric_name: str) -> Optional[SystemBehaviorPattern]:
        """Detect threshold-based patterns"""
        
        if len(values) < 20:
            return None
        
        # Calculate percentiles
        p90 = np.percentile(values, 90)
        p10 = np.percentile(values, 10)
        
        # Count threshold crossings
        high_crossings = sum(1 for i in range(1, len(values)) if values[i-1] < p90 <= values[i])
        low_crossings = sum(1 for i in range(1, len(values)) if values[i-1] > p10 >= values[i])
        
        total_crossings = high_crossings + low_crossings
        crossing_frequency = total_crossings / len(values)
        
        if crossing_frequency > 0.1:  # Frequent threshold crossings
            return SystemBehaviorPattern(
                pattern_id=f"threshold_{metric_name}_{hash(str(values[:10]))}",
                pattern_type="threshold",
                description=f"Frequent threshold crossings in {metric_name}",
                frequency=f"{crossing_frequency:.1%}_of_measurements",
                metrics_involved=[metric_name],
                confidence=min(1.0, crossing_frequency * 2),
                prediction_accuracy=0.6,
                first_detected="",  # Would be filled with actual timestamps
                last_observed=""
            )
        
        return None
    
    def _check_time_cycle(self, values: List[float], time_hours: List[float], cycle_hours: float) -> float:
        """Check strength of time-based cyclical pattern"""
        
        if len(values) < cycle_hours:
            return 0.0
        
        # Group values by cycle position
        cycle_positions = [t % cycle_hours for t in time_hours]
        position_values = defaultdict(list)
        
        for pos, val in zip(cycle_positions, values):
            position_values[round(pos)].append(val)
        
        # Calculate consistency across cycles
        position_means = {pos: np.mean(vals) for pos, vals in position_values.items() if len(vals) > 1}
        
        if len(position_means) < cycle_hours * 0.5:  # Need sufficient coverage
            return 0.0
        
        # Measure cyclical strength by comparing variance within vs between positions
        within_variance = np.mean([np.var(vals) for vals in position_values.values() if len(vals) > 1])
        between_variance = np.var(list(position_means.values()))
        
        if within_variance == 0:
            return 1.0 if between_variance > 0 else 0.0
        
        cyclical_strength = between_variance / (between_variance + within_variance)
        return cyclical_strength


class AlertManager:
    """Intelligent alert management system"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = self._initialize_alert_rules()
        self.suppression_rules = {}
        self.escalation_rules = {}
        
    def process_analysis(self, analysis: Dict[str, Any]) -> List[MonitoringAlert]:
        """Process analysis results and generate intelligent alerts"""
        
        new_alerts = []
        
        # Process anomalies
        for anomaly in analysis.get('anomalies', []):
            alert = self._create_anomaly_alert(anomaly, analysis)
            if alert and not self._is_suppressed(alert):
                new_alerts.append(alert)
        
        # Process trend alerts
        for metric_name, trend_data in analysis.get('trends', {}).items():
            alert = self._create_trend_alert(metric_name, trend_data, analysis)
            if alert and not self._is_suppressed(alert):
                new_alerts.append(alert)
        
        # Process prediction alerts
        for metric_name, prediction in analysis.get('predictions', {}).items():
            alert = self._create_prediction_alert(metric_name, prediction, analysis)
            if alert and not self._is_suppressed(alert):
                new_alerts.append(alert)
        
        # Process health score alerts
        health_score = analysis.get('health_score', 1.0)
        if health_score < 0.7:
            alert = self._create_health_alert(health_score, analysis)
            if alert and not self._is_suppressed(alert):
                new_alerts.append(alert)
        
        self.alerts.extend(new_alerts)
        return new_alerts
    
    def _create_anomaly_alert(self, anomaly: Dict, analysis: Dict) -> Optional[MonitoringAlert]:
        """Create alert for detected anomaly"""
        
        metric_name = anomaly['metric']
        severity = 'critical' if anomaly['severity'] == 'high' else 'warning'
        
        # Generate contextual recommendations
        recommendations = self._generate_anomaly_recommendations(anomaly)
        
        # Predict impact
        impact = self._predict_anomaly_impact(anomaly, analysis)
        
        alert = MonitoringAlert(
            id=f"anomaly_{metric_name}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category="performance",
            title=f"Anomaly detected in {metric_name}",
            description=f"Metric {metric_name} shows {anomaly['type']} anomaly. "
                       f"Current value: {anomaly['current_value']:.3f}, "
                       f"Z-score: {anomaly.get('z_score', 0):.2f}",
            metrics={metric_name: anomaly['current_value']},
            predicted_impact=impact,
            recommended_actions=recommendations,
            confidence=min(1.0, anomaly.get('z_score', 0) / 5.0),
            auto_resolution=self._suggest_auto_resolution(anomaly)
        )
        
        return alert
    
    def _create_trend_alert(self, metric_name: str, trend_data: Dict, analysis: Dict) -> Optional[MonitoringAlert]:
        """Create alert for concerning trends"""
        
        classification = trend_data['classification']
        
        # Only alert on concerning trends
        concerning_trends = ['decreasing', 'changing']
        
        if classification not in concerning_trends:
            return None
        
        if trend_data['stability'] > 0.8:  # Stable trends are less concerning
            return None
        
        recommendations = self._generate_trend_recommendations(metric_name, trend_data)
        
        alert = MonitoringAlert(
            id=f"trend_{metric_name}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            severity='warning',
            category='performance',
            title=f"Concerning trend in {metric_name}",
            description=f"Metric {metric_name} shows {classification} trend. "
                       f"Short-term trend: {trend_data['short_term_trend']:.4f}, "
                       f"Stability: {trend_data['stability']:.2f}",
            metrics={f"{metric_name}_trend": trend_data['short_term_trend']},
            predicted_impact=f"Continued trend may lead to performance degradation",
            recommended_actions=recommendations,
            confidence=trend_data['prediction_confidence']
        )
        
        return alert
    
    def _create_prediction_alert(self, metric_name: str, prediction: Dict, analysis: Dict) -> Optional[MonitoringAlert]:
        """Create alert for predicted issues"""
        
        # Check if any predicted values are concerning
        future_values = [
            prediction['next_5_minutes'],
            prediction['next_15_minutes'], 
            prediction['next_hour']
        ]
        
        # Define thresholds for different metrics
        thresholds = {
            'cpu_usage': 0.9,
            'memory_usage': 0.9,
            'latency': 100,
            'error_rate': 0.05
        }
        
        threshold = thresholds.get(metric_name)
        if not threshold:
            return None
        
        concerning_predictions = [v for v in future_values if v > threshold]
        
        if not concerning_predictions or prediction['confidence'] < 0.6:
            return None
        
        alert = MonitoringAlert(
            id=f"prediction_{metric_name}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            severity='warning',
            category='prediction',
            title=f"Predicted issue with {metric_name}",
            description=f"Predicted {metric_name} values may exceed threshold. "
                       f"Next hour prediction: {prediction['next_hour']:.3f}, "
                       f"Threshold: {threshold}",
            metrics={f"{metric_name}_prediction": prediction['next_hour']},
            predicted_impact=f"Performance degradation expected in next hour",
            recommended_actions=self._generate_prediction_recommendations(metric_name, prediction),
            confidence=prediction['confidence']
        )
        
        return alert
    
    def _create_health_alert(self, health_score: float, analysis: Dict) -> MonitoringAlert:
        """Create alert for overall system health"""
        
        severity = 'critical' if health_score < 0.5 else 'warning'
        
        alert = MonitoringAlert(
            id=f"health_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category='system',
            title="System health degraded",
            description=f"Overall system health score: {health_score:.2f}. "
                       f"Multiple metrics showing concerning patterns.",
            metrics={'health_score': health_score},
            predicted_impact="System performance and reliability at risk",
            recommended_actions=[
                "Review all metrics for anomalies",
                "Check resource utilization",
                "Verify system configurations",
                "Consider scaling resources"
            ],
            confidence=0.9
        )
        
        return alert
    
    def _generate_anomaly_recommendations(self, anomaly: Dict) -> List[str]:
        """Generate recommendations for anomaly resolution"""
        
        metric_name = anomaly['metric']
        
        recommendations = {
            'cpu_usage': [
                "Check for runaway processes",
                "Consider scaling CPU resources",
                "Review recent code deployments",
                "Optimize computational algorithms"
            ],
            'memory_usage': [
                "Check for memory leaks",
                "Review memory allocation patterns",
                "Consider increasing memory limits",
                "Optimize data structures"
            ],
            'latency': [
                "Check network connectivity",
                "Review database query performance",
                "Verify caching effectiveness",
                "Optimize processing pipelines"
            ],
            'error_rate': [
                "Review recent logs for error patterns",
                "Check system dependencies",
                "Verify input data quality",
                "Review error handling logic"
            ]
        }
        
        return recommendations.get(metric_name, [
            "Monitor metric closely",
            "Check for correlation with other metrics",
            "Review system logs",
            "Consider temporary scaling"
        ])
    
    def _generate_trend_recommendations(self, metric_name: str, trend_data: Dict) -> List[str]:
        """Generate recommendations for trend issues"""
        
        trend_type = trend_data['classification']
        
        if trend_type == 'decreasing':
            return [
                f"Investigate cause of decreasing {metric_name}",
                "Check for resource constraints",
                "Review recent changes",
                "Consider proactive scaling"
            ]
        elif trend_type == 'changing':
            return [
                f"Monitor {metric_name} for stability",
                "Identify source of variability",
                "Check system load patterns",
                "Review configuration changes"
            ]
        
        return ["Monitor trend continuation", "Investigate root causes"]
    
    def _generate_prediction_recommendations(self, metric_name: str, prediction: Dict) -> List[str]:
        """Generate recommendations for predicted issues"""
        
        return [
            f"Prepare for {metric_name} threshold breach",
            "Consider proactive resource scaling",
            "Review capacity planning",
            "Set up additional monitoring",
            "Prepare contingency procedures"
        ]
    
    def _predict_anomaly_impact(self, anomaly: Dict, analysis: Dict) -> str:
        """Predict impact of detected anomaly"""
        
        severity = anomaly['severity']
        metric_name = anomaly['metric']
        
        impact_map = {
            'cpu_usage': {
                'high': 'Critical performance degradation expected',
                'medium': 'Moderate performance impact likely'
            },
            'memory_usage': {
                'high': 'Risk of out-of-memory errors',
                'medium': 'Increased garbage collection overhead'
            },
            'latency': {
                'high': 'User experience significantly impacted',
                'medium': 'Noticeable response time delays'
            },
            'error_rate': {
                'high': 'Service reliability compromised',
                'medium': 'Intermittent failures expected'
            }
        }
        
        return impact_map.get(metric_name, {}).get(severity, 'Unknown impact')
    
    def _suggest_auto_resolution(self, anomaly: Dict) -> Optional[str]:
        """Suggest automatic resolution for anomaly"""
        
        metric_name = anomaly['metric']
        severity = anomaly['severity']
        
        # Only suggest auto-resolution for safe, well-understood cases
        if severity == 'medium' and metric_name in ['cpu_usage', 'memory_usage']:
            return f"auto_scale_{metric_name}"
        
        return None
    
    def _is_suppressed(self, alert: MonitoringAlert) -> bool:
        """Check if alert should be suppressed"""
        
        # Simple suppression logic - could be more sophisticated
        
        # Suppress duplicate alerts within time window
        recent_alerts = [a for a in self.alerts if 
                        (datetime.now() - datetime.fromisoformat(a.timestamp)).total_seconds() < 300]  # 5 minutes
        
        for recent_alert in recent_alerts:
            if (recent_alert.category == alert.category and 
                recent_alert.title == alert.title):
                return True
        
        return False
    
    def _initialize_alert_rules(self) -> Dict:
        """Initialize intelligent alert rules"""
        
        return {
            'anomaly_threshold': 3.0,  # Z-score threshold
            'trend_stability_threshold': 0.7,
            'prediction_confidence_threshold': 0.6,
            'health_score_threshold': 0.7,
            'suppression_window': 300,  # seconds
            'escalation_threshold': 600  # seconds
        }


class IntelligentMonitor:
    """Main intelligent monitoring system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_analyzer = MetricsAnalyzer()
        self.alert_manager = AlertManager()
        self.monitoring_history = []
        self.active_alerts = []
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # seconds
        self.retention_period = self.config.get('retention_period', 86400 * 7)  # 7 days
        
    def monitor_system(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Main monitoring function"""
        
        print(f"🔍 Intelligent monitoring: analyzing {len(metrics)} metrics...")
        
        # Analyze metrics
        analysis = self.metrics_analyzer.analyze_metrics(metrics)
        
        # Generate alerts
        new_alerts = self.alert_manager.process_analysis(analysis)
        
        # Update active alerts
        self._update_active_alerts(new_alerts)
        
        # Create monitoring report
        report = {
            'timestamp': analysis['timestamp'],
            'metrics': metrics,
            'analysis': analysis,
            'new_alerts': [asdict(alert) for alert in new_alerts],
            'active_alerts_count': len(self.active_alerts),
            'health_score': analysis['health_score'],
            'system_status': self._determine_system_status(analysis, new_alerts)
        }
        
        # Store in history
        self.monitoring_history.append(report)
        
        # Cleanup old data
        self._cleanup_old_data()
        
        return report
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        
        if not self.monitoring_history:
            return {'status': 'no_data', 'message': 'No monitoring data available'}
        
        latest_report = self.monitoring_history[-1]
        
        # Calculate summary statistics
        recent_health_scores = [r['health_score'] for r in self.monitoring_history[-10:]]
        avg_health = np.mean(recent_health_scores)
        health_trend = self._calculate_health_trend()
        
        # Alert statistics
        alert_stats = self._calculate_alert_statistics()
        
        summary = {
            'current_status': latest_report['system_status'],
            'current_health_score': latest_report['health_score'],
            'average_health_score': avg_health,
            'health_trend': health_trend,
            'active_alerts': len(self.active_alerts),
            'alert_statistics': alert_stats,
            'patterns_detected': len(latest_report['analysis'].get('patterns', [])),
            'monitoring_duration': len(self.monitoring_history),
            'last_updated': latest_report['timestamp']
        }
        
        return summary
    
    def _update_active_alerts(self, new_alerts: List[MonitoringAlert]):
        """Update list of active alerts"""
        
        # Add new alerts
        self.active_alerts.extend(new_alerts)
        
        # Remove resolved alerts (simplified logic)
        current_time = datetime.now()
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - datetime.fromisoformat(alert.timestamp)).total_seconds() < 3600  # 1 hour
        ]
    
    def _determine_system_status(self, analysis: Dict, new_alerts: List[MonitoringAlert]) -> str:
        """Determine overall system status"""
        
        health_score = analysis['health_score']
        critical_alerts = len([a for a in new_alerts if a.severity == 'critical'])
        warning_alerts = len([a for a in new_alerts if a.severity == 'warning'])
        
        if critical_alerts > 0 or health_score < 0.5:
            return 'critical'
        elif warning_alerts > 2 or health_score < 0.7:
            return 'warning'
        elif health_score > 0.9:
            return 'excellent'
        else:
            return 'good'
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        current_time = datetime.now()
        
        # Remove old history
        self.monitoring_history = [
            report for report in self.monitoring_history
            if (current_time - datetime.fromisoformat(report['timestamp'])).total_seconds() < self.retention_period
        ]
    
    def _calculate_health_trend(self) -> str:
        """Calculate health score trend"""
        
        if len(self.monitoring_history) < 3:
            return 'insufficient_data'
        
        recent_scores = [r['health_score'] for r in self.monitoring_history[-5:]]
        older_scores = [r['health_score'] for r in self.monitoring_history[-10:-5]]
        
        if not older_scores:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'degrading'
        else:
            return 'stable'
    
    def _calculate_alert_statistics(self) -> Dict[str, Any]:
        """Calculate alert statistics"""
        
        if not self.monitoring_history:
            return {}
        
        # Count alerts by type over recent history
        recent_reports = self.monitoring_history[-10:]
        
        total_alerts = 0
        critical_alerts = 0
        warning_alerts = 0
        categories = defaultdict(int)
        
        for report in recent_reports:
            for alert_data in report.get('new_alerts', []):
                total_alerts += 1
                severity = alert_data['severity']
                category = alert_data['category']
                
                if severity == 'critical':
                    critical_alerts += 1
                elif severity == 'warning':
                    warning_alerts += 1
                
                categories[category] += 1
        
        return {
            'total_recent_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'alert_rate': total_alerts / len(recent_reports),
            'categories': dict(categories)
        }
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        
        export_data = {
            'monitoring_summary': self.get_monitoring_summary(),
            'monitoring_history': self.monitoring_history[-100:],  # Last 100 reports
            'active_alerts': [asdict(alert) for alert in self.active_alerts],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Monitoring data exported to {filepath}")


# Demonstration function
def demonstrate_intelligent_monitoring():
    """Demonstrate intelligent monitoring capabilities"""
    
    print("🤖 GENERATION 5: INTELLIGENT MONITORING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize monitoring system
    monitor = IntelligentMonitor({
        'monitoring_interval': 60,
        'retention_period': 86400
    })
    
    # Simulate various system scenarios
    scenarios = [
        # Normal operation
        {'cpu_usage': 0.3, 'memory_usage': 0.4, 'latency': 25, 'throughput': 50, 'error_rate': 0.001},
        {'cpu_usage': 0.35, 'memory_usage': 0.42, 'latency': 28, 'throughput': 48, 'error_rate': 0.002},
        
        # Gradual degradation
        {'cpu_usage': 0.5, 'memory_usage': 0.6, 'latency': 45, 'throughput': 40, 'error_rate': 0.005},
        {'cpu_usage': 0.7, 'memory_usage': 0.75, 'latency': 65, 'throughput': 35, 'error_rate': 0.01},
        
        # Anomaly
        {'cpu_usage': 0.95, 'memory_usage': 0.9, 'latency': 150, 'throughput': 15, 'error_rate': 0.05},
        
        # Recovery
        {'cpu_usage': 0.4, 'memory_usage': 0.5, 'latency': 30, 'throughput': 45, 'error_rate': 0.002},
        {'cpu_usage': 0.3, 'memory_usage': 0.4, 'latency': 25, 'throughput': 50, 'error_rate': 0.001},
    ]
    
    print("📈 Simulating system monitoring over time...\n")
    
    for i, metrics in enumerate(scenarios):
        print(f"⏱️  Time {i+1}: Monitoring metrics {metrics}")
        
        report = monitor.monitor_system(metrics)
        
        print(f"   📊 Health Score: {report['health_score']:.3f}")
        print(f"   🚨 New Alerts: {len(report['new_alerts'])}")
        print(f"   📈 Status: {report['system_status'].upper()}")
        
        if report['new_alerts']:
            for alert_data in report['new_alerts']:
                print(f"      ⚠️  {alert_data['severity'].upper()}: {alert_data['title']}")
        
        print()
        time.sleep(0.1)  # Simulate time passage
    
    # Show monitoring summary
    summary = monitor.get_monitoring_summary()
    
    print("📋 MONITORING SUMMARY")
    print("-" * 40)
    print(f"Current Status: {summary['current_status'].upper()}")
    print(f"Current Health: {summary['current_health_score']:.3f}")
    print(f"Average Health: {summary['average_health_score']:.3f}")
    print(f"Health Trend: {summary['health_trend'].upper()}")
    print(f"Active Alerts: {summary['active_alerts']}")
    print(f"Patterns Detected: {summary['patterns_detected']}")
    
    if 'alert_statistics' in summary:
        stats = summary['alert_statistics']
        print(f"\nAlert Statistics:")
        print(f"  Total Recent: {stats.get('total_recent_alerts', 0)}")
        print(f"  Critical: {stats.get('critical_alerts', 0)}")
        print(f"  Warning: {stats.get('warning_alerts', 0)}")
        print(f"  Alert Rate: {stats.get('alert_rate', 0):.1f} alerts/report")
    
    # Export monitoring data
    monitor.export_monitoring_data('intelligent_monitoring_data.json')
    
    print("\n✅ Intelligent monitoring demonstration complete!")
    
    return monitor


if __name__ == "__main__":
    # Run demonstration
    demonstrate_intelligent_monitoring()