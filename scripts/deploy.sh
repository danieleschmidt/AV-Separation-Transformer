#!/bin/bash
# Deployment Script for AV-Separation-Transformer
# Supports Docker, Kubernetes, and cloud deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-development}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps_ok=true
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed"
                deps_ok=false
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed"
                deps_ok=false
            fi
            ;;
        "kubernetes")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                deps_ok=false
            fi
            if ! command -v helm &> /dev/null; then
                log_warning "Helm is not installed (optional)"
            fi
            ;;
        "aws")
            if ! command -v aws &> /dev/null; then
                log_error "AWS CLI is not installed"
                deps_ok=false
            fi
            ;;
        "gcp")
            if ! command -v gcloud &> /dev/null; then
                log_error "gcloud CLI is not installed"
                deps_ok=false
            fi
            ;;
    esac
    
    if [ "$deps_ok" = false ]; then
        log_error "Missing required dependencies. Please install them and retry."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build \
        --target production \
        --tag "av-separation-transformer:${VERSION}" \
        --tag "av-separation-transformer:latest" \
        .
    
    log_success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p logs models tmp
    
    # Set environment variables
    export VERSION
    export ENVIRONMENT
    
    # Deploy based on environment
    if [ "$ENVIRONMENT" = "development" ]; then
        docker-compose --profile development up -d
    else
        docker-compose up -d
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "AV-Separation API is running at http://localhost:8000"
    else
        log_error "Health check failed"
        docker-compose logs av-separation-api
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Check if cluster is accessible
    if ! kubectl cluster-info > /dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/deployment.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/av-separation-api -n av-separation
    
    # Get service information
    kubectl get services -n av-separation
    
    log_success "Kubernetes deployment completed"
}

# Deploy to AWS ECS
deploy_aws() {
    log_info "Deploying to AWS ECS..."
    
    # Check AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    # Build and push to ECR
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local region="${AWS_REGION:-us-west-2}"
    local repository="av-separation-transformer"
    local image_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/${repository}:${VERSION}"
    
    # Login to ECR
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "${account_id}.dkr.ecr.${region}.amazonaws.com"
    
    # Create repository if it doesn't exist
    aws ecr describe-repositories --repository-names "$repository" --region "$region" > /dev/null 2>&1 || \
        aws ecr create-repository --repository-name "$repository" --region "$region"
    
    # Tag and push image
    docker tag "av-separation-transformer:${VERSION}" "$image_uri"
    docker push "$image_uri"
    
    # Update ECS service (assuming service exists)
    local cluster_name="${ECS_CLUSTER:-av-separation-cluster}"
    local service_name="${ECS_SERVICE:-av-separation-service}"
    
    if aws ecs describe-services --cluster "$cluster_name" --services "$service_name" --region "$region" > /dev/null 2>&1; then
        log_info "Updating ECS service..."
        aws ecs update-service \
            --cluster "$cluster_name" \
            --service "$service_name" \
            --force-new-deployment \
            --region "$region"
        
        # Wait for deployment
        aws ecs wait services-stable \
            --cluster "$cluster_name" \
            --services "$service_name" \
            --region "$region"
    else
        log_warning "ECS service not found. Please create the service manually."
    fi
    
    log_success "AWS ECS deployment completed"
}

# Deploy to Google Cloud Run
deploy_gcp() {
    log_info "Deploying to Google Cloud Run..."
    
    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null 2>&1; then
        log_error "Not authenticated with Google Cloud"
        exit 1
    fi
    
    local project_id="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
    local region="${GCP_REGION:-us-central1}"
    local service_name="av-separation-api"
    local image_uri="gcr.io/${project_id}/${service_name}:${VERSION}"
    
    # Build and push to Container Registry
    docker tag "av-separation-transformer:${VERSION}" "$image_uri"
    docker push "$image_uri"
    
    # Deploy to Cloud Run
    gcloud run deploy "$service_name" \
        --image "$image_uri" \
        --platform managed \
        --region "$region" \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --concurrency 80 \
        --max-instances 10 \
        --timeout 300 \
        --port 8000
    
    # Get service URL
    local service_url=$(gcloud run services describe "$service_name" --region "$region" --format="value(status.url)")
    
    log_success "Google Cloud Run deployment completed"
    log_info "Service URL: $service_url"
}

# Health check function
health_check() {
    local url="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$url/health" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Rollback function
rollback() {
    log_warning "Initiating rollback..."
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            docker-compose down
            # Restore previous version if needed
            ;;
        "kubernetes")
            kubectl rollout undo deployment/av-separation-api -n av-separation
            ;;
        "aws")
            # AWS ECS rollback would depend on task definition versions
            log_info "Manual rollback required for AWS ECS"
            ;;
        "gcp")
            # Google Cloud Run keeps previous revisions
            gcloud run services update-traffic av-separation-api \
                --to-revisions LATEST=0,PREVIOUS=100 \
                --region "${GCP_REGION:-us-central1}"
            ;;
    esac
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            docker-compose down --volumes --remove-orphans
            docker system prune -f
            ;;
        "kubernetes")
            kubectl delete -f kubernetes/deployment.yaml || true
            ;;
    esac
}

# Main deployment function
main() {
    log_info "Starting deployment process..."
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
    
    # Set up trap for cleanup on failure
    trap 'log_error "Deployment failed. Running cleanup..."; cleanup; exit 1' ERR
    
    # Check dependencies
    check_dependencies
    
    # Build Docker image if needed
    if [[ "$DEPLOYMENT_TYPE" != "kubernetes" ]] || [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_docker_image
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        "docker")
            deploy_docker
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "aws")
            deploy_aws
            ;;
        "gcp")
            deploy_gcp
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --rollback)
            rollback
            exit 0
            ;;
        --cleanup)
            cleanup
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE          Deployment type (docker, kubernetes, aws, gcp)"
            echo "  -e, --environment ENV    Environment (development, staging, production)"
            echo "  -v, --version VERSION    Version tag"
            echo "  --rollback              Rollback to previous version"
            echo "  --cleanup               Clean up resources"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --type docker --environment development"
            echo "  $0 --type kubernetes --version v1.0.1"
            echo "  $0 --type aws --environment production"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main