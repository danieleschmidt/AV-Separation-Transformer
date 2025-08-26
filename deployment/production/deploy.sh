#!/bin/bash
set -e

echo "🚀 Deploying AV Separation System"

# Run quality gates
echo "🔍 Running quality gates..."
python autonomous_quality_gates.py

if [ $? -eq 0 ]; then
    echo "✅ Quality gates passed"
else
    echo "❌ Quality gates failed - aborting"
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t av-separation:latest .

# Deploy with Docker Compose
echo "📦 Starting services..."
docker-compose up -d

echo "✅ Deployment complete!"
echo "🌐 Application: http://localhost:8000"
echo "📊 Prometheus: http://localhost:9090"  
echo "📈 Grafana: http://localhost:3000"
