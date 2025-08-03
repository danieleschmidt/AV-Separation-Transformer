# Contributing to AV-Separation-Transformer

We welcome contributions to the AV-Separation-Transformer project! This guide will help you understand our development process and how to contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Performance Benchmarking](#performance-benchmarking)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.4.0 or higher
- CUDA 12.1+ (for GPU acceleration)
- Git LFS for large model files
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/AV-Separation-Transformer.git
   cd AV-Separation-Transformer
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   pip install -e .
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   python -c "import av_separation; print('Setup successful!')"
   pytest tests/ --tb=short
   ```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for new features
- `feature/feature-name`: Individual feature development
- `hotfix/issue-description`: Critical bug fixes
- `release/version-number`: Release preparation

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   **Commit Message Format:**
   ```
   type(scope): description
   
   [optional body]
   
   [optional footer(s)]
   ```
   
   **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Testing Requirements

### Test Categories

1. **Unit Tests**: Fast, isolated component tests
2. **Integration Tests**: Multi-component interaction tests
3. **Performance Tests**: Latency and throughput benchmarks
4. **End-to-End Tests**: Full pipeline validation

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=av_separation --cov-report=html

# Run performance benchmarks
python benchmarks/separation_benchmark.py
```

### Test Requirements

- **Coverage**: New code must have >90% test coverage
- **Performance**: No regression in key metrics (latency, quality)
- **Memory**: No memory leaks in long-running tests
- **Compatibility**: Tests must pass on all supported platforms

## Code Style Guidelines

### Python Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **Line Length**: Maximum 88 characters (Black default)

### Tools

```bash
# Format code
black av_separation/ tests/
isort av_separation/ tests/

# Lint code
flake8 av_separation/ tests/
mypy av_separation/

# Security check
bandit -r av_separation/
```

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class AudioVisualSeparator:
    """Audio-visual speech separation using transformer architecture.
    
    This class implements the core separation algorithm that combines
    audio spectrograms with visual facial features to separate multiple
    speakers in challenging acoustic environments.
    
    Args:
        num_speakers: Number of target speakers to separate (2-4)
        model_path: Path to pre-trained model weights
        device: Computing device ('cpu', 'cuda', 'mps')
        
    Example:
        >>> separator = AudioVisualSeparator(
        ...     num_speakers=2,
        ...     model_path='weights/av_sepnet_base.pth',
        ...     device='cuda'
        ... )
        >>> separated = separator.separate(audio, video)
    """
    
    def __init__(
        self,
        num_speakers: int,
        model_path: str,
        device: str = 'cpu'
    ) -> None:
        self.num_speakers = num_speakers
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
    def separate(
        self,
        audio: torch.Tensor,
        video: torch.Tensor
    ) -> List[torch.Tensor]:
        """Separate audio into individual speaker tracks.
        
        Args:
            audio: Input audio tensor [batch, channels, time]
            video: Input video tensor [batch, frames, height, width, channels]
            
        Returns:
            List of separated audio tensors, one per speaker
            
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If separation fails
        """
        if audio.dim() != 3:
            raise ValueError(f"Audio must be 3D tensor, got {audio.dim()}D")
            
        # Implementation here...
        return separated_tracks
```

## Pull Request Process

### Before Submitting

1. **Rebase on Latest Main**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run Full Test Suite**
   ```bash
   pytest
   python benchmarks/separation_benchmark.py
   ```

3. **Update Documentation**
   ```bash
   # Generate API docs
   sphinx-build docs/ docs/_build/
   
   # Update README if needed
   # Update CHANGELOG.md
   ```

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance benchmarks run
- [ ] Manual testing completed

## Performance Impact
- Latency: +/- X ms
- Memory: +/- X MB
- Quality: +/- X dB SI-SNR

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: At least 2 reviewers must approve
3. **Performance Review**: Benchmarks must show no regression
4. **Documentation Review**: Docs must be clear and complete
5. **Final Approval**: Maintainer approval required for merge

## Performance Benchmarking

### Benchmarking Requirements

All contributions that affect core algorithms must include performance benchmarks:

```bash
# Run standard benchmarks
python benchmarks/separation_benchmark.py
python benchmarks/latency_benchmark.py
python benchmarks/memory_benchmark.py

# Compare against baseline
python benchmarks/compare_performance.py --baseline main --current feature-branch
```

### Performance Criteria

- **Latency**: <50ms end-to-end on RTX 3080
- **Quality**: >15dB SI-SNR improvement
- **Memory**: <4GB peak usage for base model
- **Throughput**: >4x real-time on CPU

### Reporting Results

Include benchmark results in PR description:

```
Performance Results:
- Latency: 45ms (-3ms improvement)
- SI-SNR: 16.2dB (+0.4dB improvement) 
- Memory: 3.8GB (-0.2GB improvement)
- CPU Throughput: 4.2x real-time (+0.1x improvement)
```

## Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **Tutorials**: Step-by-step guides for common tasks
3. **Architecture Docs**: System design and implementation details
4. **Performance Guides**: Optimization and deployment best practices

### Writing Documentation

```bash
# Build documentation locally
cd docs/
make html
open _build/html/index.html

# Check for documentation issues
make linkcheck
make spelling
```

### Documentation Standards

- **Clarity**: Write for developers with basic ML knowledge
- **Examples**: Include working code examples
- **Accuracy**: Keep docs in sync with code changes
- **Completeness**: Cover all public APIs and major features

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community chat
- **Discord**: Real-time development discussion
- **Email**: daniel.schmidt@terragonlabs.com for sensitive issues

### Getting Help

1. **Search Existing Issues**: Check if your question has been answered
2. **Check Documentation**: Review API docs and tutorials
3. **Ask Questions**: Use GitHub Discussions for help
4. **Report Bugs**: Use GitHub Issues with detailed reproduction steps

### Recognition

We recognize contributors through:

- **Contributors File**: All contributors listed in CONTRIBUTORS.md
- **Release Notes**: Major contributions highlighted in releases
- **Badge System**: GitHub badges for different contribution types
- **Annual Awards**: Recognition for outstanding contributions

## Advanced Development

### Custom Model Development

For developing new model architectures:

1. **Research Phase**: Literature review and feasibility analysis
2. **Prototyping**: Implement and test core algorithms
3. **Integration**: Integrate with existing pipeline
4. **Evaluation**: Comprehensive benchmarking against baselines
5. **Documentation**: Write technical documentation and tutorials

### Performance Optimization

For optimization contributions:

1. **Profile Current Performance**: Identify bottlenecks
2. **Implement Optimizations**: CUDA kernels, quantization, pruning
3. **Validate Accuracy**: Ensure no quality degradation
4. **Benchmark Improvements**: Measure and document gains
5. **Cross-Platform Testing**: Verify improvements across devices

## Questions?

If you have questions about contributing, please:

1. Check our [FAQ](docs/FAQ.md)
2. Search [existing issues](https://github.com/danieleschmidt/AV-Separation-Transformer/issues)
3. Ask in [GitHub Discussions](https://github.com/danieleschmidt/AV-Separation-Transformer/discussions)
4. Email the maintainers: daniel.schmidt@terragonlabs.com

Thank you for contributing to AV-Separation-Transformer! 🎉