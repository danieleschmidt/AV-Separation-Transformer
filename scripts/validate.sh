#!/bin/bash
set -euo pipefail

echo "🔍 Running Production Deployment Validation"

# Test endpoints
REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")

for region in "${REGIONS[@]}"; do
    echo "Testing $region deployment..."
    
    # Health check
    if curl -f "https://api-$region.av-separation.ai/health" > /dev/null 2>&1; then
        echo "✅ Health check passed for $region"
    else
        echo "❌ Health check failed for $region"
        exit 1
    fi
    
    # Readiness check
    if curl -f "https://api-$region.av-separation.ai/ready" > /dev/null 2>&1; then
        echo "✅ Readiness check passed for $region"
    else
        echo "❌ Readiness check failed for $region"
        exit 1
    fi
done

# Test global load balancer
echo "Testing global load balancer..."
if curl -f "https://api.av-separation.ai/health" > /dev/null 2>&1; then
    echo "✅ Global load balancer operational"
else
    echo "❌ Global load balancer failed"
    exit 1
fi

# Test auto-scaling
echo "Testing auto-scaling configuration..."
for region in "${REGIONS[@]}"; do
    kubectl config use-context "production-$region"
    
    # Check HPA status
    if kubectl get hpa av-separation-hpa -n production > /dev/null 2>&1; then
        echo "✅ Auto-scaling configured for $region"
    else
        echo "❌ Auto-scaling missing for $region"
        exit 1
    fi
done

# Test monitoring
echo "Testing monitoring endpoints..."
if curl -f "https://prometheus.av-separation.ai/api/v1/query?query=up" > /dev/null 2>&1; then
    echo "✅ Prometheus monitoring operational"
else
    echo "❌ Prometheus monitoring failed"
    exit 1
fi

if curl -f "https://grafana.av-separation.ai/api/health" > /dev/null 2>&1; then
    echo "✅ Grafana dashboards operational"
else
    echo "❌ Grafana dashboards failed" 
    exit 1
fi

echo "🎉 All production deployment validation checks passed!"
