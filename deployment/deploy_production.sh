#!/bin/bash
# Production Deployment Script for Autonomous SDLC

set -e

echo "🚀 Starting Production Deployment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required but not installed. Aborting." >&2; exit 1; }

# Environment check
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please configure .env file before deployment"
    exit 1
fi

# Build and deploy
echo "Building production images..."
docker-compose -f docker-compose.prod.yml build

echo "Starting services..."
docker-compose -f docker-compose.prod.yml up -d

echo "Waiting for services to be ready..."
sleep 30

# Health check
echo "Performing health checks..."
curl -f http://localhost:8000/health || { echo "Health check failed"; exit 1; }

echo "✅ Deployment completed successfully!"
echo "Services available at:"
echo "  - API: http://localhost:8000"
echo "  - Monitoring: http://localhost:3000"
echo "  - Documentation: http://localhost:8080"
