# Contributing to AV-Separation-Transformer

We welcome contributions to the AV-Separation-Transformer project! This document provides guidelines for contributing.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Describe the issue clearly with steps to reproduce
- Include system information (OS, Python version, PyTorch version)
- Attach relevant logs or error messages

### Suggesting Features

- Open a GitHub issue with the "enhancement" label
- Describe the feature and its use case
- Discuss implementation approach if you have ideas

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/AV-Separation-Transformer
   cd AV-Separation-Transformer
   git remote add upstream https://github.com/danieleschmidt/AV-Separation-Transformer
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Write tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   pytest tests/
   flake8 src/
   mypy src/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: descriptive commit message"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Maximum line length: 100 characters
- Use type hints for function signatures

### Commit Messages

Follow conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### Testing Requirements

- Write unit tests for new functions
- Maintain >90% code coverage
- Include integration tests for major features
- Test edge cases and error conditions

### Documentation

- Add docstrings to all public functions
- Use Google-style docstrings
- Update README for user-facing changes
- Add examples for new features

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=av_separation --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Performance Benchmarking

Before submitting performance-related changes:

1. Run baseline benchmarks
   ```bash
   python benchmarks/test_latency.py --baseline
   ```

2. Apply your changes

3. Run comparison benchmarks
   ```bash
   python benchmarks/test_latency.py --compare
   ```

4. Include benchmark results in PR description

## Model Contributions

### Adding New Models

1. Implement in `src/av_separation/models/`
2. Follow existing model structure
3. Include configuration in `config.py`
4. Add tests in `tests/models/`
5. Document architecture in docstring

### Pre-trained Weights

- Test thoroughly on standard benchmarks
- Document training configuration
- Provide evaluation metrics
- Upload to approved hosting service
- Update model zoo documentation

## Release Process

1. Update version in `src/av_separation/version.py`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Build and test package
6. Create GitHub release
7. Deploy to PyPI

## Getting Help

- Join our Discord community
- Check existing issues and discussions
- Read the documentation thoroughly
- Ask questions in GitHub Discussions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to AV-Separation-Transformer!