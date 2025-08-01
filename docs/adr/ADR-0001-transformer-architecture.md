# ADR-0001: Multi-Modal Transformer Architecture Choice

## Status
Accepted

## Context

The AV-Separation-Transformer project requires a neural architecture that can effectively fuse audio and visual information for speech separation. Several architectural approaches were considered:

1. **CNN-based Fusion**: Traditional convolutional approaches with concatenation
2. **RNN-based Sequential**: LSTM/GRU with attention mechanisms  
3. **Transformer-based Multi-Modal**: Cross-attention between modalities
4. **Hybrid CNN-Transformer**: CNN frontends with transformer backends

Key requirements:
- Real-time processing capability (<50ms latency)
- High separation quality (>12dB SI-SNR improvement)
- Scalable to multiple speakers (2-6 simultaneous)
- Efficient on both GPU and CPU platforms
- Maintainable and extensible codebase

## Decision

We will use a **Multi-Modal Transformer Architecture** with the following design:

### Audio Pipeline
- 2D CNN frontend for spectrogram processing
- Sinusoidal positional encoding for temporal structure
- Transformer encoder (8 layers, 8 heads, 512-dim)

### Video Pipeline  
- RetinaFace for face detection and tracking
- MobileFaceNet for efficient visual feature extraction
- 3D convolutions for lip motion capture
- Transformer encoder (6 layers, 8 heads, 256-dim)

### Cross-Modal Fusion
- Bidirectional cross-attention mechanism
- Learnable modality embeddings
- Dynamic time warping for temporal alignment

### Separation Decoder
- Transformer decoder with speaker-specific queries
- Multi-scale spectrogram prediction
- Griffin-Lim phase reconstruction

## Consequences

### Positive
- **High Performance**: Transformer attention excels at long-range dependencies
- **Modality Fusion**: Cross-attention naturally handles audio-visual correlation
- **Scalability**: Attention scales well with sequence length and multiple speakers
- **SOTA Results**: Achieves competitive performance on standard benchmarks
- **Flexibility**: Easy to modify for different numbers of speakers
- **Parallelization**: Transformer layers parallelize well on modern hardware

### Negative
- **Computational Cost**: Higher memory and compute requirements than CNNs
- **Training Complexity**: Requires careful initialization and learning rate scheduling
- **Inference Latency**: Self-attention has quadratic complexity in sequence length
- **Memory Usage**: Attention matrices require significant memory
- **Model Size**: Larger parameter count than simpler architectures

### Mitigations
- **Efficient Attention**: Use Flash Attention 2 for memory efficiency
- **Mixed Precision**: FP16 training and inference to reduce memory
- **Model Variants**: Provide lite versions for resource-constrained deployment
- **Quantization**: INT8 post-training quantization for mobile deployment
- **Streaming**: Sliding window attention for real-time processing
- **Knowledge Distillation**: Train smaller student models from large teachers

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Cross-Modal Transformer](https://arxiv.org/abs/2104.07170) - Multi-modal attention mechanisms
- [Looking to Listen](https://arxiv.org/abs/1804.03619) - Audio-visual speech separation
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention implementation
- Internal benchmarking results comparing architectural alternatives