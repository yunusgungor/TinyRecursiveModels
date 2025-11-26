#!/bin/bash

# Rollback Deployment Script
# Supports both Docker Compose and Kubernetes rollbacks
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
PLATFORM="compose"
BACKUP_DIR="$ROOT_DIR/backups"
ROLLBACK_STEPS=1

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Rollback deployment to a previous version.

OPTIONS:
    -e, --environment ENV    Environment (development|production) [default: development]
    -p, --platform PLATFORM  Platform (compose|k8s) [default: compose]
    -s, --steps STEPS        Number of versions to rollback [default: 1]
    -v, --version VERSION    Specific version to rollback to
    -h, --help               Show this help message

EXAMPLES:
    # Rollback development environment by 1 version
    $0 -e development

    # Rollback production Kubernetes deployment by 2 versions
    $0 -e production -p k8s -s 2

    # Rollback to specific version
    $0 -e production -p k8s -v v1.2.3

EOF
    exit 1
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -s|--steps)
                ROLLBACK_STEPS="$2"
                shift 2
                ;;
            -v|--version)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                print_message "$RED" "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
        print_message "$RED" "❌ Invalid environment: $ENVIRONMENT"
        print_message "$YELLOW" "   Valid options: development, production"
        exit 1
    fi
    
    if [[ "$PLATFORM" != "compose" && "$PLATFORM" != "k8s" ]]; then
        print_message "$RED" "❌ Invalid platform: $PLATFORM"
        print_message "$YELLOW" "   Valid options: compose, k8s"
        exit 1
    fi
}

# Create backup before rollback
create_backup() {
    print_message "$BLUE" "💾 Creating backup before rollback..."
    
    mkdir -p "$BACKUP_DIR"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/pre_rollback_${timestamp}"
    
    if [[ "$PLATFORM" == "compose" ]]; then
        # Backup Docker Compose state
        if [[ "$ENVIRONMENT" == "production" ]]; then
            docker-compose -f docker-compose.prod.yml ps > "${backup_file}_services.txt"
            
            # Backup database if running
            if docker-compose -f docker-compose.prod.yml ps postgres | grep -q "Up"; then
                print_message "$BLUE" "   Backing up database..."
                docker-compose -f docker-compose.prod.yml exec -T postgres \
                    pg_dump -U postgres trendyol_gift_prod > "${backup_file}_db.sql"
                print_message "$GREEN" "   ✅ Database backup created"
            fi
        else
            docker-compose ps > "${backup_file}_services.txt"
        fi
    else
        # Backup Kubernetes state
        kubectl get all -n trendyol-gift -o yaml > "${backup_file}_k8s_state.yaml"
        
        # Backup database if running
        if kubectl get pods -n trendyol-gift -l app=postgres | grep -q "Running"; then
            print_message "$BLUE" "   Backing up database..."
            kubectl exec -n trendyol-gift deployment/postgres -- \
                pg_dump -U postgres trendyol_gift_prod > "${backup_file}_db.sql"
            print_message "$GREEN" "   ✅ Database backup created"
        fi
    fi
    
    print_message "$GREEN" "✅ Backup created: $backup_file"
}

# Rollback Docker Compose deployment
rollback_compose() {
    print_message "$BLUE" "🔄 Rolling back Docker Compose deployment..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    # Check if there are previous images
    print_message "$BLUE" "   Checking for previous images..."
    
    # Get current image tags
    local backend_image=$(docker-compose -f "$compose_file" images -q backend | head -1)
    local frontend_image=$(docker-compose -f "$compose_file" images -q frontend | head -1)
    
    if [[ -z "$backend_image" && -z "$frontend_image" ]]; then
        print_message "$RED" "❌ No running containers found"
        exit 1
    fi
    
    # Stop current containers
    print_message "$BLUE" "   Stopping current containers..."
    docker-compose -f "$compose_file" down
    
    # If specific version provided, use it
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        print_message "$BLUE" "   Rolling back to version: $ROLLBACK_VERSION"
        
        export VERSION="$ROLLBACK_VERSION"
        docker-compose -f "$compose_file" pull
        docker-compose -f "$compose_file" up -d
    else
        # Otherwise, use previous images
        print_message "$YELLOW" "   ⚠️  Automatic version detection not fully implemented"
        print_message "$YELLOW" "   Please specify version with -v flag or manually restore from backup"
        exit 1
    fi
    
    # Wait for services to be healthy
    print_message "$BLUE" "   Waiting for services to be healthy..."
    sleep 10
    
    # Verify rollback
    verify_compose_deployment
}

# Rollback Kubernetes deployment
rollback_k8s() {
    print_message "$BLUE" "🔄 Rolling back Kubernetes deployment..."
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        print_message "$RED" "❌ kubectl is not configured"
        exit 1
    fi
    
    # Get deployments
    local deployments=$(kubectl get deployments -n trendyol-gift -o name)
    
    if [[ -z "$deployments" ]]; then
        print_message "$RED" "❌ No deployments found in namespace trendyol-gift"
        exit 1
    fi
    
    # Rollback each deployment
    for deployment in $deployments; do
        local dep_name=$(echo "$deployment" | cut -d'/' -f2)
        
        print_message "$BLUE" "   Rolling back $dep_name..."
        
        if [[ -n "$ROLLBACK_VERSION" ]]; then
            # Rollback to specific version
            kubectl set image "$deployment" \
                "*=${DOCKER_REGISTRY}/${dep_name}:${ROLLBACK_VERSION}" \
                -n trendyol-gift
        else
            # Rollback to previous revision
            kubectl rollout undo "$deployment" \
                --to-revision=$(($(kubectl rollout history "$deployment" -n trendyol-gift | wc -l) - ROLLBACK_STEPS)) \
                -n trendyol-gift
        fi
        
        # Wait for rollout to complete
        print_message "$BLUE" "   Waiting for $dep_name rollout to complete..."
        kubectl rollout status "$deployment" -n trendyol-gift --timeout=300s
        
        print_message "$GREEN" "   ✅ $dep_name rolled back successfully"
    done
    
    # Verify rollback
    verify_k8s_deployment
}

# Verify Docker Compose deployment
verify_compose_deployment() {
    print_message "$BLUE" "🏥 Verifying deployment health..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    local max_retries=30
    local retry_count=0
    
    # Check PostgreSQL
    while [ $retry_count -lt $max_retries ]; do
        if docker-compose -f "$compose_file" exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ PostgreSQL is healthy"
            break
        fi
        retry_count=$((retry_count + 1))
        sleep 2
    done
    
    if [ $retry_count -eq $max_retries ]; then
        print_message "$RED" "   ❌ PostgreSQL health check failed"
        return 1
    fi
    
    # Check Backend
    retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ Backend is healthy"
            break
        fi
        retry_count=$((retry_count + 1))
        sleep 2
    done
    
    if [ $retry_count -eq $max_retries ]; then
        print_message "$RED" "   ❌ Backend health check failed"
        return 1
    fi
    
    print_message "$GREEN" "✅ Rollback verification successful"
}

# Verify Kubernetes deployment
verify_k8s_deployment() {
    print_message "$BLUE" "🏥 Verifying deployment health..."
    
    # Check if all pods are ready
    local not_ready=$(kubectl get pods -n trendyol-gift --no-headers | grep -v "Running\|Completed" | wc -l)
    
    if [ $not_ready -gt 0 ]; then
        print_message "$RED" "   ❌ Some pods are not ready"
        kubectl get pods -n trendyol-gift
        return 1
    fi
    
    print_message "$GREEN" "   ✅ All pods are ready"
    
    # Check health endpoints
    print_message "$BLUE" "   Checking health endpoints..."
    
    # Port forward to backend and check health
    kubectl port-forward -n trendyol-gift svc/backend-service 8000:8000 &
    local pf_pid=$!
    sleep 3
    
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_message "$GREEN" "   ✅ Backend health check passed"
    else
        print_message "$RED" "   ❌ Backend health check failed"
        kill $pf_pid 2>/dev/null || true
        return 1
    fi
    
    kill $pf_pid 2>/dev/null || true
    
    print_message "$GREEN" "✅ Rollback verification successful"
}

# Main rollback function
main() {
    parse_args "$@"
    validate_environment
    
    print_message "$BLUE" "=== Deployment Rollback ==="
    print_message "$BLUE" "Environment: $ENVIRONMENT"
    print_message "$BLUE" "Platform: $PLATFORM"
    
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        print_message "$BLUE" "Target Version: $ROLLBACK_VERSION"
    else
        print_message "$BLUE" "Rollback Steps: $ROLLBACK_STEPS"
    fi
    
    print_message "$BLUE" "=========================="
    echo
    
    # Confirm rollback
    read -p "⚠️  This will rollback the deployment. Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_message "$YELLOW" "❌ Rollback cancelled"
        exit 0
    fi
    
    # Create backup
    create_backup
    
    # Perform rollback based on platform
    if [[ "$PLATFORM" == "compose" ]]; then
        rollback_compose
    else
        rollback_k8s
    fi
    
    print_message "$GREEN" ""
    print_message "$GREEN" "✅ Rollback completed successfully!"
    print_message "$GREEN" ""
    print_message "$BLUE" "📊 Current Status:"
    
    if [[ "$PLATFORM" == "compose" ]]; then
        local compose_file="docker-compose.yml"
        if [[ "$ENVIRONMENT" == "production" ]]; then
            compose_file="docker-compose.prod.yml"
        fi
        docker-compose -f "$compose_file" ps
    else
        kubectl get pods -n trendyol-gift
    fi
}

# Run main function
main "$@"
