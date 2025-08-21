"""
Self-Improving AI System for Audio-Visual Separation
Implements online learning, performance optimization, and autonomous improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import time
import logging
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


@dataclass
class SelfImprovingConfig:
    """Configuration for self-improving AI system."""
    learning_rate: float = 1e-5
    momentum: float = 0.9
    adaptation_window: int = 1000
    performance_threshold: float = 0.95
    improvement_threshold: float = 0.02
    memory_size: int = 10000
    update_frequency: int = 100
    min_samples_for_update: int = 50
    confidence_threshold: float = 0.8
    exploration_rate: float = 0.1
    enable_active_learning: bool = True
    enable_curriculum_learning: bool = True
    enable_continual_learning: bool = True
    
    # Online learning parameters
    online_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Performance monitoring
    performance_history_size: int = 1000
    alert_degradation_threshold: float = 0.05
    
    # Model evolution
    architecture_mutation_rate: float = 0.01
    weight_mutation_rate: float = 0.001
    enable_neural_evolution: bool = False


class ExperienceBuffer:
    """Buffer for storing and managing learning experiences."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.metadata = deque(maxlen=max_size)
    
    def add_experience(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        loss: float,
        confidence: float,
        metadata: Dict[str, Any] = None
    ):
        """Add new experience to buffer."""
        experience = {
            'audio': audio.detach().cpu(),
            'video': video.detach().cpu(),
            'target': target.detach().cpu(),
            'prediction': prediction.detach().cpu(),
            'loss': loss,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        # Priority based on loss and novelty
        priority = loss * (1.0 - confidence)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.metadata.append(metadata or {})
    
    def sample_batch(self, batch_size: int, prioritized: bool = True) -> List[Dict[str, Any]]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / (priorities.sum() + 1e-8)
            indices = np.random.choice(
                len(self.buffer), 
                size=batch_size, 
                p=probabilities,
                replace=False
            )
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def get_hard_examples(self, top_k: int = 100) -> List[Dict[str, Any]]:
        """Get the hardest examples based on loss."""
        if len(self.priorities) == 0:
            return []
        
        sorted_indices = np.argsort(self.priorities)[-top_k:]
        return [self.buffer[i] for i in sorted_indices]
    
    def clear_old_experiences(self, max_age_hours: float = 24.0):
        """Remove experiences older than specified age."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        while (self.buffer and 
               self.buffer[0]['timestamp'] < cutoff_time):
            self.buffer.popleft()
            if self.priorities:
                self.priorities.popleft()
            if self.metadata:
                self.metadata.popleft()


class PerformanceMonitor:
    """Monitors model performance and detects degradation."""
    
    def __init__(self, config: SelfImprovingConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_history_size)
        self.metrics_history = defaultdict(lambda: deque(maxlen=config.performance_history_size))
        self.baseline_performance = None
        self.alerts = []
    
    def log_performance(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        timestamp = time.time()
        
        # Store overall performance score
        overall_score = metrics.get('overall_score', 0.0)
        self.performance_history.append((timestamp, overall_score))
        
        # Store individual metrics
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append((timestamp, value))
        
        # Set baseline if not set
        if self.baseline_performance is None and len(self.performance_history) >= 10:
            recent_scores = [score for _, score in list(self.performance_history)[-10:]]
            self.baseline_performance = np.mean(recent_scores)
    
    def detect_performance_degradation(self) -> Dict[str, Any]:
        """Detect if performance has degraded significantly."""
        if len(self.performance_history) < 20 or self.baseline_performance is None:
            return {'degraded': False, 'severity': 0.0, 'metrics': {}}
        
        # Get recent performance
        recent_scores = [score for _, score in list(self.performance_history)[-10:]]
        current_performance = np.mean(recent_scores)
        
        # Calculate degradation
        degradation = (self.baseline_performance - current_performance) / self.baseline_performance
        
        # Check if degradation exceeds threshold
        degraded = degradation > self.config.alert_degradation_threshold
        
        if degraded:
            alert = {
                'timestamp': time.time(),
                'type': 'performance_degradation',
                'severity': degradation,
                'baseline': self.baseline_performance,
                'current': current_performance,
                'degradation_percent': degradation * 100
            }
            self.alerts.append(alert)
            logger.warning(f"Performance degradation detected: {degradation:.2%}")
        
        return {
            'degraded': degraded,
            'severity': degradation,
            'baseline': self.baseline_performance,
            'current': current_performance
        }
    
    def get_performance_trend(self, window: int = 100) -> Dict[str, float]:
        """Analyze performance trend over time."""
        if len(self.performance_history) < window:
            return {'trend': 0.0, 'confidence': 0.0}
        
        recent_data = list(self.performance_history)[-window:]
        timestamps = [t for t, _ in recent_data]
        scores = [s for _, s in recent_data]
        
        # Linear regression to detect trend
        x = np.array(timestamps) - timestamps[0]
        y = np.array(scores)
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
        else:
            slope, r_squared = 0.0, 0.0
        
        return {
            'trend': slope,
            'confidence': r_squared,
            'improving': slope > 0,
            'stable': abs(slope) < 0.001
        }


class OnlineLearningModule:
    """Handles online learning and model updates."""
    
    def __init__(
        self,
        model: nn.Module,
        config: SelfImprovingConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Online optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2
        )
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(config.memory_size)
        
        # Training statistics
        self.update_count = 0
        self.online_losses = []
        self.learning_rates = []
        
        # Active learning
        self.uncertainty_estimator = UncertaintyEstimator()
    
    def process_sample(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        should_learn: bool = True
    ) -> Dict[str, Any]:
        """Process a single sample and potentially learn from it."""
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(audio, video)
            prediction = outputs['separated_spectrograms']
            
            # Estimate confidence/uncertainty
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(outputs)
            confidence = 1.0 - uncertainty
        
        results = {
            'prediction': prediction,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'should_collect': False,
            'should_update': False
        }
        
        if target is not None:
            # Compute loss
            loss = F.mse_loss(prediction, target).item()
            results['loss'] = loss
            
            # Decide whether to collect this experience
            should_collect = (
                confidence < self.config.confidence_threshold or
                loss > np.mean(self.online_losses[-100:]) if self.online_losses else True
            )
            
            if should_collect:
                self.experience_buffer.add_experience(
                    audio, video, target, prediction,
                    loss, confidence
                )
                results['should_collect'] = True
            
            # Online learning
            if should_learn and len(self.experience_buffer.buffer) >= self.config.min_samples_for_update:
                if self.update_count % self.config.update_frequency == 0:
                    self._perform_online_update()
                    results['should_update'] = True
                
                self.update_count += 1
        
        return results
    
    def _perform_online_update(self):
        """Perform online model update using experience buffer."""
        
        # Sample batch from experience buffer
        batch = self.experience_buffer.sample_batch(
            self.config.online_batch_size,
            prioritized=True
        )
        
        if len(batch) < self.config.online_batch_size // 2:
            return
        
        # Prepare batch tensors
        audio_batch = torch.stack([exp['audio'] for exp in batch]).to(self.device)
        video_batch = torch.stack([exp['video'] for exp in batch]).to(self.device)
        target_batch = torch.stack([exp['target'] for exp in batch]).to(self.device)
        
        # Online learning step
        self.model.train()
        
        total_loss = 0.0
        for i in range(0, len(batch), self.config.gradient_accumulation_steps):
            end_idx = min(i + self.config.gradient_accumulation_steps, len(batch))
            
            # Forward pass
            outputs = self.model(
                audio_batch[i:end_idx],
                video_batch[i:end_idx]
            )
            
            # Compute loss
            loss = F.mse_loss(
                outputs['separated_spectrograms'],
                target_batch[i:end_idx]
            )
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
        
        # Update parameters
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Record statistics
        self.online_losses.append(total_loss)
        self.learning_rates.append(self.scheduler.get_last_lr()[0])
        
        self.model.eval()
        
        logger.debug(f"Online update {self.update_count}: loss={total_loss:.4f}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about online learning."""
        return {
            'update_count': self.update_count,
            'buffer_size': len(self.experience_buffer.buffer),
            'avg_loss': np.mean(self.online_losses[-100:]) if self.online_losses else 0.0,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0.0,
            'hard_examples_count': len(self.experience_buffer.get_hard_examples())
        }


class UncertaintyEstimator:
    """Estimates model uncertainty for active learning."""
    
    def __init__(self):
        self.entropy_history = deque(maxlen=1000)
    
    def estimate_uncertainty(self, model_outputs: Dict[str, torch.Tensor]) -> float:
        """Estimate uncertainty from model outputs."""
        
        # Extract attention weights if available
        if 'attention_weights' in model_outputs:
            attention = model_outputs['attention_weights']
            # Entropy of attention as uncertainty measure
            entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1)
            uncertainty = torch.mean(entropy).item()
        else:
            # Use prediction variance as uncertainty proxy
            predictions = model_outputs['separated_spectrograms']
            variance = torch.var(predictions, dim=-1)
            uncertainty = torch.mean(variance).item()
        
        # Normalize uncertainty
        self.entropy_history.append(uncertainty)
        if len(self.entropy_history) > 10:
            mean_entropy = np.mean(self.entropy_history)
            std_entropy = np.std(self.entropy_history) + 1e-8
            normalized_uncertainty = (uncertainty - mean_entropy) / std_entropy
            uncertainty = torch.sigmoid(torch.tensor(normalized_uncertainty)).item()
        
        return uncertainty


class PerformanceOptimizer:
    """Optimizes model performance through various strategies."""
    
    def __init__(self, config: SelfImprovingConfig):
        self.config = config
        self.optimization_history = []
        
    def optimize_architecture(self, model: nn.Module) -> nn.Module:
        """Optimize model architecture dynamically."""
        if not self.config.enable_neural_evolution:
            return model
        
        # Simple architecture mutations
        optimized_model = self._mutate_architecture(model)
        
        return optimized_model
    
    def _mutate_architecture(self, model: nn.Module) -> nn.Module:
        """Apply small mutations to model architecture."""
        # This is a simplified example - real implementation would be more sophisticated
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and np.random.random() < self.config.architecture_mutation_rate:
                # Small chance to modify layer dimensions
                current_features = module.out_features
                
                # Small random change
                change = int(current_features * 0.1 * (np.random.random() - 0.5))
                new_features = max(1, current_features + change)
                
                if new_features != current_features:
                    # Create new layer with different size
                    new_module = nn.Linear(module.in_features, new_features)
                    
                    # Copy weights (truncate or pad as needed)
                    with torch.no_grad():
                        min_features = min(current_features, new_features)
                        new_module.weight[:min_features, :] = module.weight[:min_features, :]
                        new_module.bias[:min_features] = module.bias[:min_features]
                    
                    # Replace module
                    setattr(model, name.split('.')[-1], new_module)
        
        return model
    
    def optimize_hyperparameters(
        self,
        online_learner: OnlineLearningModule,
        performance_monitor: PerformanceMonitor
    ) -> Dict[str, float]:
        """Optimize hyperparameters based on performance."""
        
        # Get recent performance trend
        trend_info = performance_monitor.get_performance_trend()
        
        # Adaptive learning rate
        if not trend_info['improving'] and trend_info['confidence'] > 0.5:
            # Reduce learning rate if not improving
            for param_group in online_learner.optimizer.param_groups:
                param_group['lr'] *= 0.9
        elif trend_info['improving'] and trend_info['confidence'] > 0.7:
            # Increase learning rate if improving confidently
            for param_group in online_learner.optimizer.param_groups:
                param_group['lr'] *= 1.05
        
        # Return new hyperparameters
        return {
            'learning_rate': online_learner.optimizer.param_groups[0]['lr'],
            'trend_improving': trend_info['improving'],
            'trend_confidence': trend_info['confidence']
        }


class SelfImprovingAgent:
    """Main self-improving AI agent that coordinates all components."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SelfImprovingConfig] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config or SelfImprovingConfig()
        self.device = device
        
        # Initialize components
        self.online_learner = OnlineLearningModule(model, self.config, device)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.performance_optimizer = PerformanceOptimizer(self.config)
        
        # State tracking
        self.is_running = False
        self.improvement_thread = None
        self.last_optimization = time.time()
        
        # Statistics
        self.total_samples_processed = 0
        self.improvement_events = []
    
    def start_autonomous_improvement(self):
        """Start autonomous improvement in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.improvement_thread = threading.Thread(
            target=self._autonomous_improvement_loop,
            daemon=True
        )
        self.improvement_thread.start()
        
        logger.info("Started autonomous improvement agent")
    
    def stop_autonomous_improvement(self):
        """Stop autonomous improvement."""
        self.is_running = False
        if self.improvement_thread:
            self.improvement_thread.join(timeout=5.0)
        
        logger.info("Stopped autonomous improvement agent")
    
    def process_and_learn(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process sample and potentially learn from it."""
        
        # Process through online learner
        results = self.online_learner.process_sample(audio, video, target)
        
        # Update performance monitoring
        if target is not None:
            metrics = {
                'loss': results['loss'],
                'confidence': results['confidence'],
                'overall_score': results['confidence']  # Simplified
            }
            self.performance_monitor.log_performance(metrics)
        
        self.total_samples_processed += 1
        
        return results
    
    def _autonomous_improvement_loop(self):
        """Background loop for autonomous improvement."""
        
        while self.is_running:
            try:
                # Check for performance degradation
                degradation = self.performance_monitor.detect_performance_degradation()
                
                if degradation['degraded']:
                    self._handle_performance_degradation(degradation)
                
                # Periodic optimization
                current_time = time.time()
                if current_time - self.last_optimization > 3600:  # Every hour
                    self._perform_periodic_optimization()
                    self.last_optimization = current_time
                
                # Sleep between checks
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in autonomous improvement loop: {e}")
                time.sleep(60)
    
    def _handle_performance_degradation(self, degradation_info: Dict[str, Any]):
        """Handle detected performance degradation."""
        
        logger.warning(f"Handling performance degradation: {degradation_info['severity']:.2%}")
        
        # Strategy 1: Increase learning rate temporarily
        for param_group in self.online_learner.optimizer.param_groups:
            param_group['lr'] *= 2.0
        
        # Strategy 2: Focus on hard examples
        hard_examples = self.online_learner.experience_buffer.get_hard_examples(50)
        if hard_examples:
            self._retrain_on_hard_examples(hard_examples)
        
        # Strategy 3: Architecture optimization (if enabled)
        if self.config.enable_neural_evolution:
            self.model = self.performance_optimizer.optimize_architecture(self.model)
        
        # Record improvement event
        self.improvement_events.append({
            'timestamp': time.time(),
            'type': 'degradation_response',
            'severity': degradation_info['severity'],
            'actions': ['lr_increase', 'hard_examples_retrain']
        })
    
    def _perform_periodic_optimization(self):
        """Perform periodic optimization tasks."""
        
        logger.info("Performing periodic optimization")
        
        # Optimize hyperparameters
        new_params = self.performance_optimizer.optimize_hyperparameters(
            self.online_learner,
            self.performance_monitor
        )
        
        # Clean old experiences
        self.online_learner.experience_buffer.clear_old_experiences()
        
        # Log optimization event
        self.improvement_events.append({
            'timestamp': time.time(),
            'type': 'periodic_optimization',
            'new_params': new_params
        })
    
    def _retrain_on_hard_examples(self, hard_examples: List[Dict[str, Any]]):
        """Retrain model on hard examples."""
        
        if len(hard_examples) < 10:
            return
        
        # Prepare batch
        audio_batch = torch.stack([exp['audio'] for exp in hard_examples]).to(self.device)
        video_batch = torch.stack([exp['video'] for exp in hard_examples]).to(self.device)
        target_batch = torch.stack([exp['target'] for exp in hard_examples]).to(self.device)
        
        # Additional training steps on hard examples
        self.model.train()
        
        for _ in range(5):  # 5 additional steps
            outputs = self.model(audio_batch, video_batch)
            loss = F.mse_loss(outputs['separated_spectrograms'], target_batch)
            
            loss.backward()
            self.online_learner.optimizer.step()
            self.online_learner.optimizer.zero_grad()
        
        self.model.eval()
        
        logger.info(f"Retrained on {len(hard_examples)} hard examples")
    
    def get_improvement_report(self) -> Dict[str, Any]:
        """Get comprehensive improvement report."""
        
        learning_stats = self.online_learner.get_learning_statistics()
        trend_info = self.performance_monitor.get_performance_trend()
        
        return {
            'total_samples_processed': self.total_samples_processed,
            'learning_statistics': learning_stats,
            'performance_trend': trend_info,
            'improvement_events': len(self.improvement_events),
            'recent_improvements': self.improvement_events[-5:],
            'is_improving': trend_info['improving'],
            'performance_alerts': len(self.performance_monitor.alerts),
            'buffer_utilization': learning_stats['buffer_size'] / self.config.memory_size
        }
    
    def save_improvement_state(self, save_path: str):
        """Save the current improvement state."""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save improvement statistics
        report = self.get_improvement_report()
        
        with open(save_path / 'improvement_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save model state
        torch.save(self.model.state_dict(), save_path / 'improved_model.pth')
        
        logger.info(f"Saved improvement state to {save_path}")


def create_self_improving_system(
    base_model: nn.Module,
    config: Optional[SelfImprovingConfig] = None,
    device: str = 'cuda'
) -> SelfImprovingAgent:
    """Create a complete self-improving AI system."""
    
    if config is None:
        config = SelfImprovingConfig()
    
    agent = SelfImprovingAgent(base_model, config, device)
    
    # Start autonomous improvement by default
    agent.start_autonomous_improvement()
    
    return agent