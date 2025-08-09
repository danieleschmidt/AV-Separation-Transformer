#!/bin/bash
set -euo pipefail

echo "🚀 Starting AV-Separation-Transformer Deployment"

# Build and tag container
echo "Building container..."
docker build -f Dockerfile.prod -t av-separation-transformer:latest .
docker tag av-separation-transformer:latest av-separation-transformer:v1.0.0

# Push to registry (placeholder)
echo "Pushing to container registry..."
# docker push av-separation-transformer:v1.0.0

# Deploy to Kubernetes clusters
echo "Deploying to production clusters..."

for region in us-east-1 eu-west-1 ap-southeast-1; do
    echo "Deploying to $region..."
    
    # Switch to region context
    kubectl config use-context "production-$region"
    
    # Apply configurations
    kubectl apply -f deployment/production/
    
    # Wait for rollout
    kubectl rollout status deployment/av-separation-transformer -n production --timeout=600s
    
    # Verify deployment
    kubectl get pods -n production -l app=av-separation-transformer
    
    echo "✅ Deployment to $region completed"
done

echo "🎉 Global deployment completed successfully!"
