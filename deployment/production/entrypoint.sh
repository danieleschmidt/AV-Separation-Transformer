#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate av-separation

# Set default values
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-info}
export ENV=${ENV:-production}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z $host $port; do
        echo "Waiting for $service_name ($host:$port)..."
        sleep 2
    done
    echo "$service_name is ready!"
}

# Wait for dependencies if in production
if [ "$ENV" = "production" ]; then
    # Wait for Redis
    if [ -n "$REDIS_HOST" ]; then
        wait_for_service $REDIS_HOST ${REDIS_PORT:-6379} "Redis"
    fi
    
    # Wait for PostgreSQL
    if [ -n "$POSTGRES_HOST" ]; then
        wait_for_service $POSTGRES_HOST ${POSTGRES_PORT:-5432} "PostgreSQL"
    fi
fi

# Initialize logging
mkdir -p /app/logs

# Set up monitoring
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
mkdir -p $PROMETHEUS_MULTIPROC_DIR

# Command selection
case "$1" in
    "api")
        echo "Starting AV-Separation API server..."
        exec gunicorn \
            --bind $HOST:$PORT \
            --workers $WORKERS \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections 1000 \
            --max-requests 10000 \
            --max-requests-jitter 1000 \
            --timeout 120 \
            --keep-alive 5 \
            --log-level $LOG_LEVEL \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --capture-output \
            --preload \
            av_separation.api.app:app
        ;;
    
    "worker")
        echo "Starting AV-Separation background worker..."
        exec python -m av_separation.worker \
            --log-level $LOG_LEVEL \
            --concurrency ${WORKER_CONCURRENCY:-4}
        ;;
    
    "gpu-worker")
        echo "Starting AV-Separation GPU worker..."
        export CUDA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0}
        exec python -m av_separation.gpu_worker \
            --log-level $LOG_LEVEL \
            --gpu-memory-limit ${GPU_MEMORY_LIMIT:-8000}
        ;;
    
    "scheduler")
        echo "Starting AV-Separation task scheduler..."
        exec python -m av_separation.scheduler \
            --log-level $LOG_LEVEL
        ;;
    
    "migrate")
        echo "Running database migrations..."
        python -m av_separation.database.migrate
        ;;
    
    "shell")
        echo "Starting interactive shell..."
        exec python
        ;;
    
    "test")
        echo "Running tests..."
        exec python -m pytest tests/ -v
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Available commands: api, worker, gpu-worker, scheduler, migrate, shell, test"
        exit 1
        ;;
esac