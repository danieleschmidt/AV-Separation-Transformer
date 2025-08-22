#!/bin/bash
set -e

# Production Deployment Script for AV Separation System
# This script handles complete production deployment with health checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
KUBE_NAMESPACE="av-separation"
DOCKER_IMAGE="av-separation:production"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Run quality gates
run_quality_gates() {
    log_info "Running production quality gates..."
    
    cd "$PROJECT_ROOT"
    
    if ! python3 production_quality_gates.py; then
        log_error "Quality gates failed. Deployment aborted."
        exit 1
    fi
    
    log_success "Quality gates passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the production image
    docker build -f deployment/production/Dockerfile.production -t "$DOCKER_IMAGE" .
    
    # Tag for registry (if needed)
    # docker tag "$DOCKER_IMAGE" "your-registry.com/$DOCKER_IMAGE"
    # docker push "your-registry.com/$DOCKER_IMAGE"
    
    log_success "Docker image built successfully"
}

# Create namespace and secrets
setup_kubernetes_resources() {
    log_info "Setting up Kubernetes resources..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$KUBE_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply production deployment
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/production-deployment.yaml"
    
    log_success "Kubernetes resources applied"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Apply the deployment configuration
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/production-deployment.yaml"
    
    # Wait for rollout to complete
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/av-separation-api -n "$KUBE_NAMESPACE" --timeout=600s
    
    log_success "Kubernetes deployment completed"
}

# Deploy using Docker Compose
deploy_docker_compose() {
    log_info "Deploying using Docker Compose..."
    
    cd "$PROJECT_ROOT/deployment/production"
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    log_success "Docker Compose deployment completed"
}

# Health check function
health_check() {
    local service_url="$1"
    local retries="$2"
    local interval="$3"
    
    log_info "Performing health check on $service_url"
    
    for i in $(seq 1 "$retries"); do
        if curl -f -s "$service_url/health" > /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check attempt $i/$retries failed, waiting ${interval}s..."
        sleep "$interval"
    done
    
    log_error "Health check failed after $retries attempts"
    return 1
}

# Post-deployment validation
validate_deployment() {
    log_info "Validating deployment..."
    
    if [ "$DEPLOYMENT_ENV" = "kubernetes" ]; then
        # Get service endpoint for Kubernetes
        local service_ip
        service_ip=$(kubectl get service av-separation-service -n "$KUBE_NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        local service_port
        service_port=$(kubectl get service av-separation-service -n "$KUBE_NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
        local service_url="http://$service_ip:$service_port"
        
        # Port forward for health check
        kubectl port-forward service/av-separation-service 8080:80 -n "$KUBE_NAMESPACE" &
        local port_forward_pid=$!
        
        sleep 5
        
        # Perform health check
        if health_check "http://localhost:8080" "$HEALTH_CHECK_RETRIES" "$HEALTH_CHECK_INTERVAL"; then
            log_success "Kubernetes deployment validation passed"
        else
            log_error "Kubernetes deployment validation failed"
            kill $port_forward_pid 2>/dev/null || true
            return 1
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    else
        # Docker Compose health check
        if health_check "http://localhost:8000" "$HEALTH_CHECK_RETRIES" "$HEALTH_CHECK_INTERVAL"; then
            log_success "Docker Compose deployment validation passed"
        else
            log_error "Docker Compose deployment validation failed"
            return 1
        fi
    fi
}

# Rollback function
rollback() {
    log_warning "Performing rollback..."
    
    if [ "$DEPLOYMENT_ENV" = "kubernetes" ]; then
        kubectl rollout undo deployment/av-separation-api -n "$KUBE_NAMESPACE"
        kubectl rollout status deployment/av-separation-api -n "$KUBE_NAMESPACE" --timeout=300s
    else
        cd "$PROJECT_ROOT/deployment/production"
        docker-compose -f docker-compose.production.yml down
        # Restore previous version if available
        # docker-compose -f docker-compose.production.yml up -d
    fi
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Remove temporary files if any
    # rm -f /tmp/deployment_*
}

# Main deployment function
main() {
    log_info "Starting production deployment..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Target namespace: $KUBE_NAMESPACE"
    log_info "Docker image: $DOCKER_IMAGE"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Run quality gates
    run_quality_gates
    
    # Build Docker image
    build_image
    
    # Deploy based on environment
    if [ "$DEPLOYMENT_ENV" = "kubernetes" ]; then
        setup_kubernetes_resources
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    # Validate deployment
    if ! validate_deployment; then
        log_error "Deployment validation failed. Initiating rollback."
        rollback
        exit 1
    fi
    
    log_success "Production deployment completed successfully!"
    log_info "Access the application at:"
    
    if [ "$DEPLOYMENT_ENV" = "kubernetes" ]; then
        log_info "  - Health check: kubectl port-forward service/av-separation-service 8080:80 -n $KUBE_NAMESPACE"
        log_info "  - Then visit: http://localhost:8080/health"
    else
        log_info "  - Health check: http://localhost:8000/health"
        log_info "  - API endpoint: http://localhost:8000/separate"
    fi
    
    # Display deployment information
    log_info "Deployment information:"
    if [ "$DEPLOYMENT_ENV" = "kubernetes" ]; then
        kubectl get pods -n "$KUBE_NAMESPACE" -l app=av-separation-api
        kubectl get services -n "$KUBE_NAMESPACE"
        kubectl get hpa -n "$KUBE_NAMESPACE"
    else
        docker-compose -f "$PROJECT_ROOT/deployment/production/docker-compose.production.yml" ps
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        -n|--namespace)
            KUBE_NAMESPACE="$2"
            shift 2
            ;;
        -i|--image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --skip-quality-gates)
            SKIP_QUALITY_GATES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --env ENV          Deployment environment (kubernetes|docker-compose)"
            echo "  -n, --namespace NS     Kubernetes namespace (default: av-separation)"
            echo "  -i, --image IMAGE      Docker image name (default: av-separation:production)"
            echo "  --skip-quality-gates   Skip quality gate checks"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main "$@"
