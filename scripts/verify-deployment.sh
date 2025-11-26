#!/bin/bash

# Deployment Verification Script
# Verifies all pods are ready, checks health endpoints, and can trigger rollback on failure
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
AUTO_ROLLBACK=false
MAX_RETRIES=30
RETRY_INTERVAL=2

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

Verify deployment health and optionally rollback on failure.

OPTIONS:
    -e, --environment ENV    Environment (development|production) [default: development]
    -p, --platform PLATFORM  Platform (compose|k8s) [default: compose]
    -r, --auto-rollback      Automatically rollback on verification failure
    -m, --max-retries NUM    Maximum health check retries [default: 30]
    -i, --interval SECONDS   Retry interval in seconds [default: 2]
    -h, --help               Show this help message

EXAMPLES:
    # Verify development deployment
    $0 -e development

    # Verify production with auto-rollback
    $0 -e production -r

    # Verify Kubernetes deployment
    $0 -e production -p k8s

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
            -r|--auto-rollback)
                AUTO_ROLLBACK=true
                shift
                ;;
            -m|--max-retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            -i|--interval)
                RETRY_INTERVAL="$2"
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

# Verify Docker Compose deployment
verify_compose() {
    print_message "$BLUE" "🏥 Verifying Docker Compose deployment..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    local all_healthy=true
    
    # Check PostgreSQL
    print_message "$BLUE" "   Checking PostgreSQL..."
    local retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if docker-compose -f "$compose_file" exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ PostgreSQL is healthy"
            break
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $MAX_RETRIES ]; then
            print_message "$RED" "   ❌ PostgreSQL health check failed after $MAX_RETRIES attempts"
            docker-compose -f "$compose_file" logs --tail=50 postgres
            all_healthy=false
            break
        fi
        sleep $RETRY_INTERVAL
    done
    
    # Check Redis
    print_message "$BLUE" "   Checking Redis..."
    retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if docker-compose -f "$compose_file" exec -T redis redis-cli ping > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ Redis is healthy"
            break
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $MAX_RETRIES ]; then
            print_message "$RED" "   ❌ Redis health check failed after $MAX_RETRIES attempts"
            docker-compose -f "$compose_file" logs --tail=50 redis
            all_healthy=false
            break
        fi
        sleep $RETRY_INTERVAL
    done
    
    # Check Backend
    print_message "$BLUE" "   Checking Backend..."
    retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ Backend is healthy"
            
            # Get detailed health info
            local health_response=$(curl -s http://localhost:8000/api/health)
            print_message "$BLUE" "   Backend health details: $health_response"
            break
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $MAX_RETRIES ]; then
            print_message "$RED" "   ❌ Backend health check failed after $MAX_RETRIES attempts"
            docker-compose -f "$compose_file" logs --tail=50 backend
            all_healthy=false
            break
        fi
        sleep $RETRY_INTERVAL
    done
    
    # Check Frontend
    print_message "$BLUE" "   Checking Frontend..."
    retry_count=0
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            print_message "$GREEN" "   ✅ Frontend is healthy"
            break
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -eq $MAX_RETRIES ]; then
            print_message "$YELLOW" "   ⚠️  Frontend health check failed after $MAX_RETRIES attempts"
            print_message "$YELLOW" "   This may be normal if frontend is still building"
            docker-compose -f "$compose_file" logs --tail=50 frontend
            # Don't mark as unhealthy for frontend in development
            if [[ "$ENVIRONMENT" == "production" ]]; then
                all_healthy=false
            fi
            break
        fi
        sleep $RETRY_INTERVAL
    done
    
    # Verify all containers are running
    print_message "$BLUE" "   Checking container status..."
    local running_containers=$(docker-compose -f "$compose_file" ps --services --filter "status=running" | wc -l)
    local total_containers=$(docker-compose -f "$compose_file" ps --services | wc -l)
    
    print_message "$BLUE" "   Running containers: $running_containers/$total_containers"
    
    if [ $running_containers -lt $total_containers ]; then
        print_message "$RED" "   ❌ Not all containers are running"
        docker-compose -f "$compose_file" ps
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        print_message "$GREEN" ""
        print_message "$GREEN" "✅ All services are healthy!"
        return 0
    else
        print_message "$RED" ""
        print_message "$RED" "❌ Some services failed health checks"
        return 1
    fi
}

# Verify Kubernetes deployment
verify_k8s() {
    print_message "$BLUE" "🏥 Verifying Kubernetes deployment..."
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        print_message "$RED" "❌ kubectl is not configured"
        return 1
    fi
    
    local all_healthy=true
    
    # Check if namespace exists
    if ! kubectl get namespace trendyol-gift &> /dev/null; then
        print_message "$RED" "❌ Namespace 'trendyol-gift' does not exist"
        return 1
    fi
    
    # Check all pods are ready
    print_message "$BLUE" "   Checking pod status..."
    
    local not_ready=$(kubectl get pods -n trendyol-gift --no-headers 2>/dev/null | grep -v "Running\|Completed" | wc -l)
    
    if [ $not_ready -gt 0 ]; then
        print_message "$RED" "   ❌ Some pods are not ready"
        kubectl get pods -n trendyol-gift
        all_healthy=false
    else
        print_message "$GREEN" "   ✅ All pods are ready"
        kubectl get pods -n trendyol-gift
    fi
    
    # Check deployments
    print_message "$BLUE" "   Checking deployments..."
    
    local deployments=$(kubectl get deployments -n trendyol-gift -o name 2>/dev/null)
    
    for deployment in $deployments; do
        local dep_name=$(echo "$deployment" | cut -d'/' -f2)
        
        print_message "$BLUE" "   Checking $dep_name..."
        
        # Check if deployment is available
        local available=$(kubectl get "$deployment" -n trendyol-gift -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')
        
        if [[ "$available" == "True" ]]; then
            print_message "$GREEN" "   ✅ $dep_name is available"
        else
            print_message "$RED" "   ❌ $dep_name is not available"
            kubectl describe "$deployment" -n trendyol-gift
            all_healthy=false
        fi
        
        # Check replica count
        local desired=$(kubectl get "$deployment" -n trendyol-gift -o jsonpath='{.spec.replicas}')
        local ready=$(kubectl get "$deployment" -n trendyol-gift -o jsonpath='{.status.readyReplicas}')
        
        if [[ "$ready" == "$desired" ]]; then
            print_message "$GREEN" "   ✅ $dep_name has $ready/$desired replicas ready"
        else
            print_message "$RED" "   ❌ $dep_name has $ready/$desired replicas ready"
            all_healthy=false
        fi
    done
    
    # Check health endpoints
    print_message "$BLUE" "   Checking health endpoints..."
    
    # Port forward to backend and check health
    kubectl port-forward -n trendyol-gift svc/backend-service 8000:8000 &
    local pf_pid=$!
    sleep 3
    
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_message "$GREEN" "   ✅ Backend health endpoint is responding"
        
        # Get detailed health info
        local health_response=$(curl -s http://localhost:8000/api/health)
        print_message "$BLUE" "   Backend health details: $health_response"
    else
        print_message "$RED" "   ❌ Backend health endpoint is not responding"
        all_healthy=false
    fi
    
    kill $pf_pid 2>/dev/null || true
    
    # Check HPA status
    print_message "$BLUE" "   Checking HPA status..."
    
    local hpas=$(kubectl get hpa -n trendyol-gift -o name 2>/dev/null)
    
    for hpa in $hpas; do
        local hpa_name=$(echo "$hpa" | cut -d'/' -f2)
        
        print_message "$BLUE" "   Checking $hpa_name..."
        
        local current=$(kubectl get "$hpa" -n trendyol-gift -o jsonpath='{.status.currentReplicas}')
        local desired=$(kubectl get "$hpa" -n trendyol-gift -o jsonpath='{.status.desiredReplicas}')
        
        print_message "$BLUE" "   $hpa_name: $current/$desired replicas"
    done
    
    if [ "$all_healthy" = true ]; then
        print_message "$GREEN" ""
        print_message "$GREEN" "✅ All services are healthy!"
        return 0
    else
        print_message "$RED" ""
        print_message "$RED" "❌ Some services failed health checks"
        return 1
    fi
}

# Trigger rollback
trigger_rollback() {
    print_message "$YELLOW" ""
    print_message "$YELLOW" "⚠️  Deployment verification failed!"
    
    if [ "$AUTO_ROLLBACK" = true ]; then
        print_message "$YELLOW" "🔄 Initiating automatic rollback..."
        
        if [[ "$PLATFORM" == "compose" ]]; then
            "$SCRIPT_DIR/rollback.sh" -e "$ENVIRONMENT" -p compose
        else
            "$SCRIPT_DIR/rollback.sh" -e "$ENVIRONMENT" -p k8s
        fi
    else
        print_message "$YELLOW" "   Auto-rollback is disabled"
        print_message "$YELLOW" "   To rollback manually, run:"
        print_message "$NC" "   ./scripts/rollback.sh -e $ENVIRONMENT -p $PLATFORM"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    print_message "$BLUE" "=== Deployment Verification ==="
    print_message "$BLUE" "Environment: $ENVIRONMENT"
    print_message "$BLUE" "Platform: $PLATFORM"
    print_message "$BLUE" "Auto-rollback: $AUTO_ROLLBACK"
    print_message "$BLUE" "Max retries: $MAX_RETRIES"
    print_message "$BLUE" "Retry interval: ${RETRY_INTERVAL}s"
    print_message "$BLUE" "==============================="
    echo
    
    # Perform verification based on platform
    if [[ "$PLATFORM" == "compose" ]]; then
        if verify_compose; then
            exit 0
        else
            trigger_rollback
            exit 1
        fi
    else
        if verify_k8s; then
            exit 0
        else
            trigger_rollback
            exit 1
        fi
    fi
}

# Run main function
main "$@"
