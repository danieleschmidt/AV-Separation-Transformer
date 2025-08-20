"""
Meta-Learning Framework for Audio-Visual Separation
Enables few-shot adaptation to new speakers, accents, and acoustic conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import copy
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning framework."""
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 8
    num_tasks_per_batch: int = 4
    support_shots: int = 5
    query_shots: int = 15
    adaptation_layers: List[str] = None
    freeze_backbone: bool = False
    
    def __post_init__(self):
        if self.adaptation_layers is None:
            self.adaptation_layers = ['decoder', 'fusion']


class TaskDataset:
    """Represents a task for meta-learning (e.g., specific speaker or acoustic condition)."""
    
    def __init__(
        self,
        task_id: str,
        support_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        query_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        task_metadata: Dict[str, Any] = None
    ):
        self.task_id = task_id
        self.support_data = support_data  # (audio, video, target) pairs
        self.query_data = query_data
        self.task_metadata = task_metadata or {}
    
    def get_support_batch(self, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get support set as batched tensors."""
        audio_list, video_list, target_list = zip(*self.support_data)
        
        audio_batch = torch.stack(audio_list).to(device)
        video_batch = torch.stack(video_list).to(device)
        target_batch = torch.stack(target_list).to(device)
        
        return audio_batch, video_batch, target_batch
    
    def get_query_batch(self, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get query set as batched tensors."""
        audio_list, video_list, target_list = zip(*self.query_data)
        
        audio_batch = torch.stack(audio_list).to(device)
        video_batch = torch.stack(video_list).to(device)
        target_batch = torch.stack(target_list).to(device)
        
        return audio_batch, video_batch, target_batch


class MetaModule(nn.Module):
    """Base class for modules that can be meta-learned."""
    
    def __init__(self):
        super().__init__()
        self._meta_parameters = {}
    
    def meta_forward(self, *args, params: Dict[str, torch.Tensor] = None, **kwargs):
        """Forward pass with custom parameters."""
        if params is None:
            return self.forward(*args, **kwargs)
        
        # Temporarily replace parameters
        backup_params = {}
        for name, param in self.named_parameters():
            if name in params:
                backup_params[name] = param.data.clone()
                param.data = params[name]
        
        try:
            output = self.forward(*args, **kwargs)
        finally:
            # Restore original parameters
            for name, param in self.named_parameters():
                if name in backup_params:
                    param.data = backup_params[name]
        
        return output
    
    def get_meta_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters for meta-learning."""
        return {name: param.clone() for name, param in self.named_parameters()}


class MetaAdaptiveEncoder(MetaModule):
    """Encoder that can quickly adapt to new tasks."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Task-specific adaptation layers
        self.task_adaptation = nn.ModuleDict({
            'task_embedding': nn.Linear(hidden_dim, 64),
            'adaptation_gate': nn.Linear(64, hidden_dim),
            'task_modulation': nn.Linear(64, hidden_dim)
        })
    
    def forward(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard encoding
        encoded = self.encoder(x)
        
        # Task-specific adaptation
        if task_context is not None:
            task_emb = self.task_adaptation['task_embedding'](task_context)
            gate = torch.sigmoid(self.task_adaptation['adaptation_gate'](task_emb))
            modulation = self.task_adaptation['task_modulation'](task_emb)
            
            encoded = encoded * gate + modulation
        
        return encoded


class MetaFusionLayer(MetaModule):
    """Fusion layer with meta-learning capabilities."""
    
    def __init__(self, audio_dim: int, video_dim: int, output_dim: int):
        super().__init__()
        
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.video_proj = nn.Linear(video_dim, output_dim)
        
        # Cross-modal attention with meta-adaptation
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
        # Task-specific fusion parameters
        self.meta_fusion = nn.ModuleDict({
            'audio_weight': nn.Parameter(torch.ones(1)),
            'video_weight': nn.Parameter(torch.ones(1)),
            'fusion_bias': nn.Parameter(torch.zeros(output_dim))
        })
    
    def forward(
        self, 
        audio_features: torch.Tensor, 
        video_features: torch.Tensor,
        task_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        
        # Project to common dimension
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)
        
        # Cross-modal attention
        attended_audio, _ = self.cross_attention(audio_proj, video_proj, video_proj)
        attended_video, _ = self.cross_attention(video_proj, audio_proj, audio_proj)
        
        # Meta-weighted fusion
        if task_weights is not None:
            audio_weight = task_weights.get('audio_weight', self.meta_fusion['audio_weight'])
            video_weight = task_weights.get('video_weight', self.meta_fusion['video_weight'])
            fusion_bias = task_weights.get('fusion_bias', self.meta_fusion['fusion_bias'])
        else:
            audio_weight = self.meta_fusion['audio_weight']
            video_weight = self.meta_fusion['video_weight']
            fusion_bias = self.meta_fusion['fusion_bias']
        
        fused = audio_weight * attended_audio + video_weight * attended_video + fusion_bias
        
        return fused


class MAMLSeparator(MetaModule):
    """Model-Agnostic Meta-Learning for audio-visual separation."""
    
    def __init__(self, config, meta_config: MetaLearningConfig):
        super().__init__()
        
        self.config = config
        self.meta_config = meta_config
        
        # Backbone encoders (can be frozen for meta-learning)
        self.audio_encoder = MetaAdaptiveEncoder(
            config.audio.n_mels, 
            config.model.audio_encoder_dim
        )
        
        self.video_encoder = MetaAdaptiveEncoder(
            config.video.face_size[0] * config.video.face_size[1] * 3,
            config.model.video_encoder_dim
        )
        
        # Meta-learnable fusion layer
        self.fusion = MetaFusionLayer(
            config.model.audio_encoder_dim,
            config.model.video_encoder_dim,
            config.model.fusion_dim
        )
        
        # Adaptation decoder
        self.decoder = MetaAdaptiveEncoder(
            config.model.fusion_dim,
            config.model.decoder_dim
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.model.decoder_dim,
            config.model.max_speakers * config.audio.n_mels
        )
        
        # Task context encoder
        self.task_context_encoder = nn.Sequential(
            nn.Linear(config.model.fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(
        self, 
        audio: torch.Tensor, 
        video: torch.Tensor,
        task_context: Optional[torch.Tensor] = None,
        adapted_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Encode modalities
        audio_features = self.audio_encoder(audio, task_context)
        video_features = self.video_encoder(video, task_context)
        
        # Fusion with potential adaptation
        if adapted_params and 'fusion' in adapted_params:
            fusion_weights = {k.replace('fusion.', ''): v 
                            for k, v in adapted_params.items() 
                            if k.startswith('fusion.')}
            fused = self.fusion(audio_features, video_features, fusion_weights)
        else:
            fused = self.fusion(audio_features, video_features)
        
        # Generate task context if not provided
        if task_context is None:
            task_context = self.task_context_encoder(fused.mean(dim=1))
        
        # Decode with adaptation
        decoded = self.decoder(fused, task_context)
        
        # Output projection
        output = self.output_proj(decoded)
        
        B, T, _ = output.shape
        separated = output.view(B, T, self.config.model.max_speakers, -1)
        
        return {
            'separated_spectrograms': separated,
            'task_context': task_context,
            'fused_features': fused
        }
    
    def adapt_to_task(
        self, 
        task: TaskDataset, 
        loss_fn: Callable,
        device: str = 'cuda'
    ) -> Dict[str, torch.Tensor]:
        """Adapt model parameters to a specific task using MAML."""
        
        # Get support set
        support_audio, support_video, support_target = task.get_support_batch(device)
        
        # Create copies of adaptable parameters
        adapted_params = {}
        for name, param in self.named_parameters():
            if any(layer in name for layer in self.meta_config.adaptation_layers):
                adapted_params[name] = param.clone()
        
        # Inner loop: gradient descent on support set
        for step in range(self.meta_config.inner_steps):
            # Forward pass with current adapted parameters
            outputs = self.meta_forward(
                support_audio, support_video,
                params=adapted_params
            )
            
            # Compute loss
            loss = loss_fn(outputs['separated_spectrograms'], support_target)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, adapted_params.values(),
                create_graph=True, retain_graph=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.meta_config.inner_lr * grad
        
        return adapted_params
    
    def meta_forward(
        self, 
        audio: torch.Tensor, 
        video: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with custom parameters for meta-learning."""
        
        if params is None:
            return self.forward(audio, video)
        
        # Backup and replace parameters
        original_params = {}
        for name, param in self.named_parameters():
            if name in params:
                original_params[name] = param.data.clone()
                param.data = params[name]
        
        try:
            outputs = self.forward(audio, video)
        finally:
            # Restore original parameters
            for name, param in self.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        
        return outputs


class MetaLearningFramework:
    """Complete meta-learning framework for audio-visual separation."""
    
    def __init__(
        self, 
        model: MAMLSeparator,
        meta_config: MetaLearningConfig,
        device: str = 'cuda'
    ):
        self.model = model
        self.meta_config = meta_config
        self.device = device
        
        # Meta-optimizer (outer loop)
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=meta_config.outer_lr
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Training statistics
        self.meta_losses = []
        self.adaptation_losses = []
    
    def meta_train_step(self, task_batch: List[TaskDataset]) -> Dict[str, float]:
        """Single meta-training step with batch of tasks."""
        
        meta_loss = 0.0
        adaptation_losses = []
        
        for task in task_batch:
            # Inner loop: adapt to task
            adapted_params = self.model.adapt_to_task(
                task, self.loss_fn, self.device
            )
            
            # Get query set
            query_audio, query_video, query_target = task.get_query_batch(self.device)
            
            # Forward pass on query set with adapted parameters
            query_outputs = self.model.meta_forward(
                query_audio, query_video, adapted_params
            )
            
            # Compute meta-loss
            task_loss = self.loss_fn(
                query_outputs['separated_spectrograms'], 
                query_target
            )
            
            meta_loss += task_loss
            adaptation_losses.append(task_loss.item())
        
        # Average meta-loss
        meta_loss = meta_loss / len(task_batch)
        
        # Outer loop: update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Record statistics
        self.meta_losses.append(meta_loss.item())
        self.adaptation_losses.extend(adaptation_losses)
        
        return {
            'meta_loss': meta_loss.item(),
            'adaptation_loss_mean': np.mean(adaptation_losses),
            'adaptation_loss_std': np.std(adaptation_losses)
        }
    
    def meta_train(
        self, 
        task_generator: Callable[[], List[TaskDataset]],
        num_iterations: int = 1000,
        eval_every: int = 100,
        eval_tasks: Optional[List[TaskDataset]] = None
    ) -> Dict[str, List[float]]:
        """Full meta-training loop."""
        
        logger.info(f"Starting meta-training for {num_iterations} iterations")
        
        training_stats = {
            'meta_losses': [],
            'adaptation_losses': [],
            'eval_losses': []
        }
        
        for iteration in range(num_iterations):
            # Generate batch of tasks
            task_batch = task_generator()
            
            # Meta-training step
            step_stats = self.meta_train_step(task_batch)
            
            training_stats['meta_losses'].append(step_stats['meta_loss'])
            training_stats['adaptation_losses'].append(step_stats['adaptation_loss_mean'])
            
            # Evaluation
            if eval_tasks and (iteration + 1) % eval_every == 0:
                eval_loss = self.meta_evaluate(eval_tasks)
                training_stats['eval_losses'].append(eval_loss)
                
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"Meta Loss: {step_stats['meta_loss']:.4f}, "
                    f"Eval Loss: {eval_loss:.4f}"
                )
        
        return training_stats
    
    def meta_evaluate(self, eval_tasks: List[TaskDataset]) -> float:
        """Evaluate meta-learning performance on test tasks."""
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for task in eval_tasks:
                # Adapt to task
                adapted_params = self.model.adapt_to_task(
                    task, self.loss_fn, self.device
                )
                
                # Evaluate on query set
                query_audio, query_video, query_target = task.get_query_batch(self.device)
                
                query_outputs = self.model.meta_forward(
                    query_audio, query_video, adapted_params
                )
                
                task_loss = self.loss_fn(
                    query_outputs['separated_spectrograms'],
                    query_target
                )
                
                total_loss += task_loss.item()
        
        self.model.train()
        return total_loss / len(eval_tasks)
    
    def few_shot_adapt(
        self, 
        task: TaskDataset,
        num_adaptation_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Quickly adapt to a new task with few examples."""
        
        if num_adaptation_steps is None:
            num_adaptation_steps = self.meta_config.inner_steps
        
        # Create temporary config for adaptation
        temp_config = copy.deepcopy(self.meta_config)
        temp_config.inner_steps = num_adaptation_steps
        
        # Adapt model
        adapted_params = self.model.adapt_to_task(task, self.loss_fn, self.device)
        
        return adapted_params


class FewShotSeparation:
    """Few-shot separation for new speakers or conditions."""
    
    def __init__(self, meta_model: MAMLSeparator, device: str = 'cuda'):
        self.meta_model = meta_model
        self.device = device
        self.adapted_models = {}  # Cache of adapted models
    
    def register_new_speaker(
        self,
        speaker_id: str,
        sample_audio: List[torch.Tensor],
        sample_video: List[torch.Tensor],
        sample_targets: List[torch.Tensor]
    ) -> str:
        """Register a new speaker with few-shot examples."""
        
        # Create task dataset
        support_data = list(zip(sample_audio, sample_video, sample_targets))
        
        # For few-shot, we use the same data for query (in practice, you'd have separate query data)
        query_data = support_data
        
        task = TaskDataset(
            task_id=speaker_id,
            support_data=support_data,
            query_data=query_data,
            task_metadata={'speaker_id': speaker_id, 'num_samples': len(sample_audio)}
        )
        
        # Adapt model to this speaker
        framework = MetaLearningFramework(
            self.meta_model, 
            MetaLearningConfig(inner_steps=10),  # More steps for better adaptation
            self.device
        )
        
        adapted_params = framework.few_shot_adapt(task)
        
        # Cache adapted model
        self.adapted_models[speaker_id] = adapted_params
        
        logger.info(f"Registered new speaker: {speaker_id} with {len(sample_audio)} samples")
        
        return speaker_id
    
    def separate_for_speaker(
        self,
        speaker_id: str,
        audio: torch.Tensor,
        video: torch.Tensor
    ) -> torch.Tensor:
        """Perform separation using speaker-adapted model."""
        
        if speaker_id not in self.adapted_models:
            raise ValueError(f"Speaker {speaker_id} not registered. Call register_new_speaker first.")
        
        adapted_params = self.adapted_models[speaker_id]
        
        with torch.no_grad():
            outputs = self.meta_model.meta_forward(
                audio.to(self.device),
                video.to(self.device),
                adapted_params
            )
        
        return outputs['separated_spectrograms']
    
    def get_registered_speakers(self) -> List[str]:
        """Get list of registered speakers."""
        return list(self.adapted_models.keys())


class TaskAdaptiveModel:
    """Model that adapts to different tasks and conditions dynamically."""
    
    def __init__(self, meta_model: MAMLSeparator):
        self.meta_model = meta_model
        self.task_history = []
        self.adaptation_cache = {}
    
    def detect_task_shift(
        self, 
        audio: torch.Tensor, 
        video: torch.Tensor,
        threshold: float = 0.1
    ) -> bool:
        """Detect if the current input represents a new task/condition."""
        
        with torch.no_grad():
            # Extract features
            audio_features = self.meta_model.audio_encoder(audio)
            video_features = self.meta_model.video_encoder(video)
            
            current_features = torch.cat([
                audio_features.mean(dim=1),
                video_features.mean(dim=1)
            ], dim=-1)
        
        if len(self.task_history) == 0:
            self.task_history.append(current_features)
            return True
        
        # Compare with recent task features
        recent_features = torch.stack(self.task_history[-10:])  # Last 10 tasks
        distances = torch.cdist(current_features, recent_features)
        min_distance = torch.min(distances).item()
        
        # If significantly different from recent tasks, it's a task shift
        is_shift = min_distance > threshold
        
        if is_shift:
            self.task_history.append(current_features)
            # Keep history manageable
            if len(self.task_history) > 100:
                self.task_history = self.task_history[-50:]
        
        return is_shift
    
    def adaptive_inference(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        adaptation_examples: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """Perform inference with potential online adaptation."""
        
        # Check for task shift
        task_shift = self.detect_task_shift(audio, video)
        
        if task_shift and adaptation_examples:
            # Quick adaptation to new task
            support_data = adaptation_examples
            query_data = [(audio, video, torch.zeros_like(audio))]  # Dummy target
            
            task = TaskDataset(
                task_id=f"adaptive_task_{len(self.task_history)}",
                support_data=support_data,
                query_data=query_data
            )
            
            # Fast adaptation
            framework = MetaLearningFramework(
                self.meta_model,
                MetaLearningConfig(inner_steps=3),  # Quick adaptation
                'cuda'
            )
            
            adapted_params = framework.few_shot_adapt(task)
            
            # Use adapted model
            with torch.no_grad():
                outputs = self.meta_model.meta_forward(audio, video, adapted_params)
        else:
            # Use base model
            with torch.no_grad():
                outputs = self.meta_model(audio, video)
        
        return outputs['separated_spectrograms']


def create_meta_learning_pipeline(
    base_config,
    meta_config: Optional[MetaLearningConfig] = None,
    device: str = 'cuda'
) -> Tuple[MAMLSeparator, MetaLearningFramework, FewShotSeparation]:
    """Create complete meta-learning pipeline."""
    
    if meta_config is None:
        meta_config = MetaLearningConfig()
    
    # Create meta-learnable model
    meta_model = MAMLSeparator(base_config, meta_config)
    meta_model.to(device)
    
    # Create training framework
    framework = MetaLearningFramework(meta_model, meta_config, device)
    
    # Create few-shot separation interface
    few_shot = FewShotSeparation(meta_model, device)
    
    return meta_model, framework, few_shot