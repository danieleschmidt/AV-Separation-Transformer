#!/bin/bash

# Post-create script for AV-Separation-Transformer development container
set -e

echo "🚀 Setting up AV-Separation-Transformer development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    libopencv-dev \
    build-essential \
    cmake \
    git-lfs \
    htop \
    nvtop \
    tree \
    curl \
    wget

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML/AI dependencies
echo "🧠 Installing ML/AI dependencies..."
pip install \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb

# Install audio/video processing
echo "🎵 Installing audio/video processing libraries..."
pip install \
    librosa \
    soundfile \
    audioread \
    opencv-python \
    av \
    moviepy \
    pydub

# Install ONNX and optimization
echo "⚡ Installing ONNX and optimization tools..."
pip install \
    onnx \
    onnxruntime \
    onnxruntime-gpu \
    openvino \
    tensorrt

# Install WebRTC and networking
echo "🌐 Installing WebRTC and networking..."
pip install \
    aiortc \
    websockets \
    fastapi \
    uvicorn \
    pydantic

# Install development tools
echo "🛠️ Installing development tools..."
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-benchmark \
    pre-commit \
    bandit \
    safety

# Install documentation tools
echo "📚 Installing documentation tools..."
pip install \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autoapi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    data/raw \
    data/processed \
    data/models \
    logs \
    outputs \
    weights \
    experiments \
    .cache

# Setup Git LFS
echo "📦 Setting up Git LFS..."
git lfs install
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.wav"
git lfs track "*.mp4"

# Install pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Setup Jupyter Lab extensions
echo "🪐 Setting up Jupyter Lab..."
jupyter lab --generate-config
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Create default environment file
echo "⚙️ Creating environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || echo "# AV-Separation-Transformer Environment Variables" > .env
fi

# Install the package in development mode
echo "📦 Installing package in development mode..."
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e .
fi

# Setup CUDA environment
echo "🔥 Configuring CUDA environment..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Create helpful aliases
echo "⚡ Setting up helpful aliases..."
cat << EOF >> ~/.bashrc

# AV-Separation-Transformer aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Python aliases
alias python='python3'
alias pip='pip3'
alias pytest='python -m pytest'
alias black='python -m black'
alias isort='python -m isort'
alias flake8='python -m flake8'
alias mypy='python -m mypy'

# Development aliases
alias test='make test'
alias lint='make lint'
alias format='make format'
alias docs='make docs'
alias serve='make serve'

# GPU monitoring
alias gpu='nvidia-smi'
alias gputop='nvtop'

# Project specific
alias train='python scripts/train.py'
alias evaluate='python scripts/evaluate.py'
alias separate='python scripts/separate.py'
EOF

# Display system info
echo "🖥️ System Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
fi

echo "✅ Development environment setup complete!"
echo "🚀 You can now start developing AV-Separation-Transformer"
echo ""
echo "Useful commands:"
echo "  make test     - Run tests"
echo "  make lint     - Run linting"
echo "  make format   - Format code"
echo "  make docs     - Build documentation"
echo "  jupyter lab   - Start Jupyter Lab"
echo "  tensorboard --logdir=logs - Start TensorBoard"
echo ""
echo "Happy coding! 🎉"