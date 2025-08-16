"""
🌌 Quantum-Classical Hybrid Architecture
Next-generation fusion of quantum and classical computing for audio-visual processing

This module implements a groundbreaking quantum-classical hybrid architecture that
seamlessly integrates quantum computing advantages with classical neural networks,
creating a future-proof system ready for the quantum computing era.

Components:
- HybridConfig: Configuration for quantum-classical hybrid system
- QuantumCircuitLayer: Variational quantum circuits integrated with neural networks
- QuantumSimulatorBackend: Quantum circuit simulation backend
- NoiseModel: Realistic quantum noise simulation
- HybridQuantumClassicalLayer: Seamless quantum-classical processing layer
- WorkloadBalancer: Intelligent quantum-classical workload distribution
- PerformanceMonitor: Hybrid system performance monitoring and optimization
- QuantumClassicalHybridModel: Complete hybrid model architecture

Features:
- Quantum-classical hybrid processing pipelines
- Variational quantum circuits with classical backpropagation
- Quantum advantage detection and automatic switching
- Classical fallback for quantum unavailability
- Distributed quantum-classical workload balancing
- Quantum error correction and noise resilience
- Future-proof architecture ready for quantum hardware

Quantum Backends Supported:
- Simulator (for development and testing)
- IBM Quantum (planned)
- Google Cirq (planned)
- Amazon Braket (planned)
- Microsoft Azure Quantum (planned)
- Xanadu PennyLane (planned)

Author: TERRAGON Autonomous SDLC v4.0 - Generation 4 Transcendence
"""

from .quantum_classical_fusion import (
    QuantumBackend,
    HybridConfig,
    QuantumCircuitLayer,
    QuantumSimulatorBackend,
    NoiseModel,
    HybridQuantumClassicalLayer,
    WorkloadBalancer,
    PerformanceMonitor,
    QuantumClassicalHybridModel,
    create_hybrid_model,
    benchmark_hybrid_performance
)

__all__ = [
    'QuantumBackend',
    'HybridConfig',
    'QuantumCircuitLayer',
    'QuantumSimulatorBackend',
    'NoiseModel',
    'HybridQuantumClassicalLayer',
    'WorkloadBalancer',
    'PerformanceMonitor',
    'QuantumClassicalHybridModel',
    'create_hybrid_model',
    'benchmark_hybrid_performance'
]