"""
Optimized Performance Components for AV-Separation
High-performance inference, intelligent caching, and auto-scaling capabilities.
"""

from .performance_engine import (
    HighPerformanceEngine,
    IntelligentCache,
    TensorJITCompiler,
    BatchProcessor,
    MemoryOptimizer,
    PerformanceProfiler,
    PerformanceMetrics,
    performance_optimized,
    global_performance_engine
)

from .auto_scaler import (
    IntelligentAutoScaler,
    LoadPredictor,
    CostOptimizer,
    KubernetesScaler,
    ScalingMetrics,
    ScalingDecision,
    ScalingDirection,
    ScalingTrigger,
    auto_scale_endpoint,
    global_auto_scaler
)

from .distributed_engine import (
    DistributedInferenceEngine,
    ModelSharding,
    DataParallelism,
    PipelineParallelism,
    DistributedCache,
    LoadBalancer,
    distributed_inference
)

__all__ = [
    # Performance Engine
    'HighPerformanceEngine',
    'IntelligentCache',
    'TensorJITCompiler',
    'BatchProcessor',
    'MemoryOptimizer',
    'PerformanceProfiler',
    'PerformanceMetrics',
    'performance_optimized',
    'global_performance_engine',
    
    # Auto Scaler
    'IntelligentAutoScaler',
    'LoadPredictor',
    'CostOptimizer',
    'KubernetesScaler',
    'ScalingMetrics',
    'ScalingDecision',
    'ScalingDirection',
    'ScalingTrigger',
    'auto_scale_endpoint',
    'global_auto_scaler',
    
    # Distributed Engine
    'DistributedInferenceEngine',
    'ModelSharding',
    'DataParallelism',
    'PipelineParallelism',
    'DistributedCache',
    'LoadBalancer',
    'distributed_inference'
]