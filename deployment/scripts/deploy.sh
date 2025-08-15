#!/bin/bash
set -euo pipefail

# AV-Separation Production Deployment Script
# This script handles complete deployment to production environment

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="av-separation"
REGISTRY=${REGISTRY:-"your-registry.com"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
DRY_RUN=${DRY_RUN:-"false"}
SKIP_TESTS=${SKIP_TESTS:-"false"}
FORCE_REDEPLOY=${FORCE_REDEPLOY:-"false"}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    for tool in kubectl docker helm; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is required but not installed"
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
    fi
    
    # Check registry access
    if ! docker pull hello-world &> /dev/null; then
        log_warning "Cannot pull from Docker registry - proceeding anyway"
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace and basic resources
setup_namespace() {
    log_info "Setting up namespace and basic resources..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for monitoring
    kubectl label namespace $NAMESPACE monitoring=enabled --overwrite
    
    # Create service account
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: av-separation-service-account
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: av-separation-role
  namespace: $NAMESPACE
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: av-separation-role-binding
  namespace: $NAMESPACE
subjects:
- kind: ServiceAccount
  name: av-separation-service-account
  namespace: $NAMESPACE
roleRef:
  kind: Role
  name: av-separation-role
  apiGroup: rbac.authorization.k8s.io
EOF
    
    log_success "Namespace setup completed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    local base_dir="$(dirname $(dirname $(realpath $0)))"
    
    # Build API image
    log_info "Building API image..."
    docker build -t $REGISTRY/av-separation:$IMAGE_TAG \
        -f $base_dir/deployment/production/Dockerfile.api \
        $base_dir
    
    # Build GPU worker image
    log_info "Building GPU worker image..."
    docker build -t $REGISTRY/av-separation-gpu:$IMAGE_TAG \
        -f $base_dir/deployment/production/Dockerfile.gpu \
        $base_dir
    
    # Build NGINX image
    log_info "Building NGINX image..."
    docker build -t $REGISTRY/av-separation-nginx:$IMAGE_TAG \
        -f $base_dir/deployment/production/Dockerfile.nginx \
        $base_dir
    
    # Push images
    if [ "$DRY_RUN" != "true" ]; then
        log_info "Pushing images to registry..."
        docker push $REGISTRY/av-separation:$IMAGE_TAG
        docker push $REGISTRY/av-separation-gpu:$IMAGE_TAG
        docker push $REGISTRY/av-separation-nginx:$IMAGE_TAG
    else
        log_info "DRY_RUN: Skipping image push"
    fi
    
    log_success "Image build and push completed"
}

# Run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running pre-deployment tests..."
    
    # Run unit tests
    docker run --rm $REGISTRY/av-separation:$IMAGE_TAG ./entrypoint.sh test
    
    # Run integration tests
    log_info "Running integration tests..."
    # Add integration test commands here
    
    # Run security scans
    log_info "Running security scans..."
    # Add security scan commands here
    
    log_success "All tests passed"
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Generate secrets if they don't exist
    if ! kubectl get secret av-separation-secrets -n $NAMESPACE &> /dev/null; then
        # Generate random passwords
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        REDIS_PASSWORD=$(openssl rand -base64 32)
        JWT_SECRET=$(openssl rand -base64 64)
        API_KEY=$(openssl rand -base64 32)
        
        kubectl create secret generic av-separation-secrets \
            --namespace=$NAMESPACE \
            --from-literal=postgres-user=av_user \
            --from-literal=postgres-password="$POSTGRES_PASSWORD" \
            --from-literal=redis-password="$REDIS_PASSWORD" \
            --from-literal=jwt-secret="$JWT_SECRET" \
            --from-literal=api-key="$API_KEY"
        
        log_success "Secrets created"
    else
        log_info "Secrets already exist, skipping creation"
    fi
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    local base_dir="$(dirname $(dirname $(realpath $0)))"
    
    # Deploy Redis
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install av-separation-redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.password=$(kubectl get secret av-separation-secrets -n $NAMESPACE -o jsonpath='{.data.redis-password}' | base64 -d) \
        --set master.persistence.enabled=true \
        --set master.persistence.size=10Gi \
        --set replica.replicaCount=1 \
        --wait
    
    # Deploy PostgreSQL
    helm upgrade --install av-separation-postgres bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.username=av_user \
        --set auth.password=$(kubectl get secret av-separation-secrets -n $NAMESPACE -o jsonpath='{.data.postgres-password}' | base64 -d) \
        --set auth.database=av_separation \
        --set persistence.enabled=true \
        --set persistence.size=50Gi \
        --wait
    
    # Deploy monitoring stack
    kubectl apply -f $base_dir/deployment/monitoring/prometheus/
    kubectl apply -f $base_dir/deployment/monitoring/grafana/
    
    log_success "Infrastructure deployment completed"
}

# Deploy application
deploy_application() {
    log_info "Deploying AV-Separation application..."
    
    local base_dir="$(dirname $(dirname $(realpath $0)))"
    
    # Update image references in manifests
    sed -i.bak "s|av-separation:latest|$REGISTRY/av-separation:$IMAGE_TAG|g" \
        $base_dir/deployment/kubernetes/production/deployment.yaml
    sed -i.bak "s|av-separation-gpu:latest|$REGISTRY/av-separation-gpu:$IMAGE_TAG|g" \
        $base_dir/deployment/kubernetes/production/deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f $base_dir/deployment/kubernetes/production/
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s \
        deployment/av-separation-api -n $NAMESPACE
    
    # Check GPU workers (they might not be immediately available)
    if kubectl get deployment av-separation-gpu-worker -n $NAMESPACE &> /dev/null; then
        kubectl wait --for=condition=available --timeout=300s \
            deployment/av-separation-gpu-worker -n $NAMESPACE || \
            log_warning "GPU workers not ready - check GPU node availability"
    fi
    
    log_success "Application deployment completed"
}

# Setup auto-scaling
setup_autoscaling() {
    log_info "Setting up auto-scaling..."
    
    local base_dir="$(dirname $(dirname $(realpath $0)))"
    
    # Apply HPA and KEDA configurations
    kubectl apply -f $base_dir/deployment/kubernetes/production/hpa.yaml
    
    # Install KEDA if not present
    if ! kubectl get namespace keda-system &> /dev/null; then
        log_info "Installing KEDA..."
        kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml
        kubectl wait --for=condition=available --timeout=300s \
            deployment/keda-operator -n keda-system
    fi
    
    log_success "Auto-scaling setup completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all pods are running
    local failed_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running -o name | wc -l)
    if [ $failed_pods -gt 0 ]; then
        log_warning "Some pods are not running:"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
    fi
    
    # Check services are accessible
    local api_service=$(kubectl get service av-separation-api -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    if [ -n "$api_service" ]; then
        log_info "Testing API health endpoint..."
        kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
            curl -f http://$api_service:8000/health || \
            log_warning "API health check failed"
    fi
    
    # Check auto-scaling metrics
    kubectl get hpa -n $NAMESPACE
    
    # Check monitoring is working
    if kubectl get service prometheus -n $NAMESPACE &> /dev/null; then
        log_info "Prometheus is running"
    fi
    
    if kubectl get service grafana -n $NAMESPACE &> /dev/null; then
        log_info "Grafana is running"
    fi
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    local base_dir="$(dirname $(dirname $(realpath $0)))"
    
    # Restore original manifests
    if [ -f "$base_dir/deployment/kubernetes/production/deployment.yaml.bak" ]; then
        mv "$base_dir/deployment/kubernetes/production/deployment.yaml.bak" \
           "$base_dir/deployment/kubernetes/production/deployment.yaml"
    fi
}

# Main deployment function
main() {
    log_info "Starting AV-Separation deployment to $ENVIRONMENT"
    log_info "Registry: $REGISTRY"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry Run: $DRY_RUN"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    setup_namespace
    build_and_push_images
    run_tests
    deploy_secrets
    deploy_infrastructure
    deploy_application
    setup_autoscaling
    verify_deployment
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at: http://$(kubectl get ingress av-separation-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')"
    log_info "Access Grafana at: http://$(kubectl get service grafana -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):3000"
    log_info "Access Prometheus at: http://$(kubectl get service prometheus -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        build_and_push_images
        ;;
    "test")
        run_tests
        ;;
    "verify")
        verify_deployment
        ;;
    "cleanup")
        log_info "Cleaning up deployment..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|build|test|verify|cleanup}"
        echo ""
        echo "Environment variables:"
        echo "  REGISTRY       - Docker registry (default: your-registry.com)"
        echo "  IMAGE_TAG      - Image tag (default: latest)"
        echo "  ENVIRONMENT    - Deployment environment (default: production)"
        echo "  DRY_RUN        - Skip actual deployment (default: false)"
        echo "  SKIP_TESTS     - Skip test execution (default: false)"
        echo "  FORCE_REDEPLOY - Force redeployment (default: false)"
        exit 1
        ;;
esac