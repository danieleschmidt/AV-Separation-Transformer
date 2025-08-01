# Architecture Documentation

## System Overview

The AV-Separation-Transformer is a production-ready implementation of an audio-visual speech separation system designed for real-time video conferencing applications. The system leverages a multi-modal transformer architecture to separate multiple speakers in challenging acoustic environments.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AV-Separation System                     │
├─────────────────────────────────────────────────────────────┤
│                    Input Processing                         │
├─────────────────────┬───────────────────────────────────────┤
│    Audio Pipeline   │         Video Pipeline               │
│                     │                                       │
│  ┌─────────────┐   │    ┌─────────────┐                   │
│  │   Audio     │   │    │   Video     │                   │
│  │  Encoder    │   │    │  Encoder    │                   │
│  └─────────────┘   │    └─────────────┘                   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     ▼                                       │
│              Cross-Modal Fusion                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Multi-Modal Transformer                   │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │        Audio-Visual Attention               │   │   │
│  │  │  - Bidirectional Cross-Attention           │   │   │
│  │  │  - Temporal Alignment (DTW)                │   │   │
│  │  │  - Learnable Modality Embeddings           │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     ▼                                       │
│              Output Generation                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Separation Decoder                        │   │
│  │  - Multi-scale Spectrogram Prediction              │   │
│  │  - Phase Reconstruction (Griffin-Lim)              │   │
│  │  - Dynamic Speaker Assignment                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Audio Processing Pipeline

#### Audio Encoder
- **Frontend**: 2D CNN with 5 layers [1→64→128→256→512 channels]
- **Positional Encoding**: Sinusoidal encoding for temporal awareness
- **Backend**: Transformer encoder (8 layers, 8 heads, dim=512)
- **Output**: Audio feature embeddings with temporal structure

#### Input Processing
- **Sampling Rate**: 16kHz for speech optimization
- **Window Size**: 25ms Hamming windows
- **Hop Length**: 10ms for low-latency streaming
- **Feature Extraction**: Log-mel spectrograms (80 bins)

### 2. Video Processing Pipeline

#### Face Detection and Tracking
- **Detector**: RetinaFace for robust face detection
- **Tracking**: DeepSORT for multi-speaker tracking
- **ROI Extraction**: Lip region focus with margin expansion

#### Visual Feature Extraction
- **Frontend**: MobileFaceNet for lightweight face encoding
- **3D Convolutions**: Temporal lip motion capture
- **Backend**: Transformer encoder (6 layers, 8 heads, dim=256)
- **Output**: Visual embeddings aligned with audio features

### 3. Cross-Modal Fusion Layer

#### Attention Mechanisms
- **Audio-to-Visual**: Query audio features attend to visual context
- **Visual-to-Audio**: Query visual features attend to audio context
- **Self-Attention**: Within-modality feature refinement

#### Temporal Alignment
- **Dynamic Time Warping**: Handles audio-video synchronization drift
- **Learnable Alignment**: Trainable temporal offset correction
- **Buffering Strategy**: Sliding window for real-time processing

### 4. Separation Architecture

#### Transformer Decoder
- **Layers**: 8 transformer decoder layers
- **Speaker Queries**: Learnable embeddings for each target speaker
- **Cross-Attention**: Queries attend to fused audio-visual features

#### Output Generation
- **Multi-Scale Prediction**: Spectrograms at multiple resolutions
- **Phase Reconstruction**: Griffin-Lim with iterative refinement
- **Post-Processing**: Wiener filtering for artifact reduction

## Data Flow

```
Audio Input (16kHz PCM) ──┐
                          ├──► Feature Extraction ──┐
Video Input (30fps RGB) ──┘                        │
                                                    ▼
                                            Cross-Modal Fusion
                                                    │
                                                    ▼
                                            Separation Decoder
                                                    │
                                                    ▼
                          ┌──── Speaker 1 Audio ◄──┤
                          ├──── Speaker 2 Audio ◄──┤
                          └──── Speaker N Audio ◄──┘
```

## Performance Characteristics

### Latency Targets
- **Batch Processing**: <100ms for offline processing
- **Real-Time Streaming**: <50ms for video conferencing
- **WebRTC Integration**: <30ms for low-latency communications

### Throughput Metrics
- **GPU (RTX 4090)**: 32x real-time processing
- **CPU (Intel i9)**: 4x real-time processing
- **Mobile (Apple M2)**: 1.2x real-time processing

### Memory Requirements
- **Model Parameters**: 124M (base), 356M (large), 892M (XL)
- **Runtime Memory**: 2GB (base), 4GB (large), 8GB (XL)
- **Streaming Buffer**: 512MB for 5-second lookahead

## Deployment Architectures

### 1. Cloud Deployment
```
Load Balancer ──► API Gateway ──► Kubernetes Pods
                                      │
                                      ├─ Model Server (GPU)
                                      ├─ Preprocessing (CPU)
                                      └─ Storage (Redis/S3)
```

### 2. Edge Deployment
```
WebRTC Client ──► Edge Server ──► Local GPU/CPU
                      │
                      ├─ ONNX Runtime
                      ├─ TensorRT Engine
                      └─ Result Cache
```

### 3. Mobile Deployment
```
Mobile App ──► CoreML/TFLite ──► On-Device Inference
                   │
                   ├─ Model Quantization (INT8)
                   ├─ Memory Mapping
                   └─ Background Processing
```

## Security Considerations

### Data Privacy
- **Local Processing**: Edge deployment minimizes data transmission
- **Encryption**: TLS 1.3 for all network communications
- **Data Retention**: Configurable deletion policies

### Model Security
- **Model Signing**: Cryptographic verification of model integrity
- **Runtime Protection**: Memory encryption for model weights
- **Access Control**: Role-based permissions for model updates

### Input Validation
- **Audio Sanitization**: Prevent adversarial audio attacks
- **Video Validation**: Face detection confidence thresholds
- **Rate Limiting**: Prevent resource exhaustion attacks

## Scalability Design

### Horizontal Scaling
- **Stateless Processing**: Each request is independent
- **Load Distribution**: Round-robin with health checks
- **Auto-Scaling**: CPU/GPU utilization-based scaling

### Vertical Scaling
- **Memory Management**: Efficient tensor memory reuse
- **GPU Optimization**: Mixed precision and kernel fusion
- **CPU Optimization**: SIMD vectorization and multi-threading

## Quality Assurance

### Performance Monitoring
- **Latency Tracking**: P95/P99 latency metrics
- **Quality Metrics**: SI-SNR, PESQ, STOI measurement
- **Resource Utilization**: CPU/GPU/Memory monitoring

### A/B Testing Framework
- **Model Versioning**: Seamless model rollout/rollback
- **Quality Comparison**: Side-by-side separation quality
- **User Experience**: Latency and accuracy trade-offs

## Future Architecture Evolution

### Planned Enhancements
- **Multi-Language Support**: Extend beyond English speech
- **Noise Robustness**: Better handling of non-speech audio
- **Speaker Diarization**: Automatic speaker identification
- **Emotion Recognition**: Preserve emotional context in separation

### Technology Roadmap
- **Quantization**: 4-bit and 8-bit model variants
- **Pruning**: Structured sparsity for mobile deployment
- **Knowledge Distillation**: Teacher-student model compression
- **Neural Architecture Search**: Automated model optimization