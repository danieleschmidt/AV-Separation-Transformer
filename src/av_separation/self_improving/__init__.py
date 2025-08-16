"""
🧠 Self-Improving AI Architecture
Advanced meta-learning and continual adaptation capabilities

This module implements cutting-edge self-improving AI capabilities including:
- Meta-learning for rapid task adaptation (MAML)
- Neural Architecture Search (NAS) for auto-optimization
- Continual learning without catastrophic forgetting
- Experience replay and elastic weight consolidation
- Performance monitoring and auto-improvement triggers

Components:
- AdaptationConfig: Configuration for self-improvement
- MetaLearner: Model-Agnostic Meta-Learning implementation
- NeuralArchitectureSearch: Automated architecture optimization
- ContinualLearner: Lifelong learning without forgetting
- SelfImprovingModel: Complete self-improving wrapper

Author: TERRAGON Autonomous SDLC v4.0 - Generation 4 Transcendence
"""

from .adaptive_learning import (
    AdaptationConfig,
    ExperienceMemory,
    MetaLearner,
    NeuralArchitectureSearch,
    ContinualLearner,
    ForgettingDetector,
    SelfImprovingModel,
    create_self_improving_model
)

__all__ = [
    'AdaptationConfig',
    'ExperienceMemory',
    'MetaLearner', 
    'NeuralArchitectureSearch',
    'ContinualLearner',
    'ForgettingDetector',
    'SelfImprovingModel',
    'create_self_improving_model'
]