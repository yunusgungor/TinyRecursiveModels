#!/bin/bash

# Selective Deployment Script
# Detects changed services and deploys only those
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
BASE_COMMIT="HEAD~1"
CURRENT_COMMIT="HEAD"
DRY_RUN=false

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

Detect changed services and deploy only those.

OPTIONS:
    -e, --environment ENV    Environment (development|production) [default: development]
    -p, --platform PLATFORM  Platform (compose|k8s) [default: compose]
    -b, --base COMMIT        Base commit for comparison [default: HEAD~1]
    -c, --current COMMIT     Current commit [default: HEAD]
    -d, --dry-run            Show what would be deployed without deploying
    -h, --help               Show this help message

EXAMPLES:
    # Deploy changed services in development
    $0 -e development

    # Deploy changed services to Kubernetes
    $0 -e production -p k8s

    # Compare specific commits
    $0 -b main -c feature-branch

    # Dry run to see what would be deployed
    $0 -d

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
            -b|--base)
                BASE_COMMIT="$2"
                shift 2
                ;;
            -c|--current)
                CURRENT_COMMIT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
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

# Detect changed files
detect_changed_files() {
    print_message "$BLUE" "🔍 Detecting changed files..."
    print_message "$BLUE" "   Base: $BASE_COMMIT"
    print_message "$BLUE" "   Current: $CURRENT_COMMIT"
    
    # Get list of changed files
    local changed_files=$(git diff --name-only "$BASE_COMMIT" "$CURRENT_COMMIT" 2>/dev/null || echo "")
    
    if [[ -z "$changed_files" ]]; then
        print_message "$YELLOW" "⚠️  No changes detected or not a git repository"
        print_message "$YELLOW" "   Falling back to detecting all services"
        return 1
    fi
    
    echo "$changed_files"
}

# Detect changed services
detect_changed_services() {
    local changed_files="$1"
    local changed_services=()
    
    print_message "$BLUE" "📦 Analyzing changed services..."
    
    # Check if backend files changed
    if echo "$changed_files" | grep -q "^backend/"; then
        changed_services+=("backend")
        print_message "$GREEN" "   ✓ Backend has changes"
    fi
    
    # Check if frontend files changed
    if echo "$changed_files" | grep -q "^frontend/"; then
        changed_services+=("frontend")
        print_message "$GREEN" "   ✓ Frontend has changes"
    fi
    
    # Check if docker-compose files changed (affects all services)
    if echo "$changed_files" | grep -q "docker-compose"; then
        if [[ ! " ${changed_services[@]} " =~ " backend " ]]; then
            changed_services+=("backend")
        fi
        if [[ ! " ${changed_services[@]} " =~ " frontend " ]]; then
            changed_services+=("frontend")
        fi
        print_message "$YELLOW" "   ⚠️  Docker Compose configuration changed - deploying all services"
    fi
    
    # Check if Kubernetes manifests changed
    if echo "$changed_files" | grep -q "^k8s/"; then
        if [[ ! " ${changed_services[@]} " =~ " backend " ]]; then
            changed_services+=("backend")
        fi
        if [[ ! " ${changed_services[@]} " =~ " frontend " ]]; then
            changed_services+=("frontend")
        fi
        print_message "$YELLOW" "   ⚠️  Kubernetes manifests changed - deploying all services"
    fi
    
    # Use build dependency graph for more accurate detection
    if [[ -f "$SCRIPT_DIR/build_dependency_graph.py" ]]; then
        print_message "$BLUE" "   Using dependency graph for accurate detection..."
        
        local affected_services=$(python3 "$SCRIPT_DIR/build_dependency_graph.py" affected $changed_files 2>/dev/null | grep "Services to rebuild:" | cut -d':' -f2 | tr ',' '\n' | xargs)
        
        if [[ -n "$affected_services" ]]; then
            for service in $affected_services; do
                if [[ ! " ${changed_services[@]} " =~ " $service " ]]; then
                    changed_services+=("$service")
                    print_message "$GREEN" "   ✓ $service affected by dependency changes"
                fi
            done
        fi
    fi
    
    echo "${changed_services[@]}"
}

# Deploy services using Docker Compose
deploy_compose() {
    local services=("$@")
    
    if [[ ${#services[@]} -eq 0 ]]; then
        print_message "$YELLOW" "⚠️  No services to deploy"
        return 0
    fi
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    print_message "$BLUE" "🔨 Building changed services..."
    
    for service in "${services[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            print_message "$YELLOW" "   [DRY RUN] Would build: $service"
        else
            print_message "$BLUE" "   Building $service..."
            docker-compose -f "$compose_file" build "$service"
            print_message "$GREEN" "   ✅ $service built"
        fi
    done
    
    print_message "$BLUE" "🚀 Deploying changed services..."
    
    for service in "${services[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            print_message "$YELLOW" "   [DRY RUN] Would deploy: $service"
        else
            print_message "$BLUE" "   Deploying $service..."
            docker-compose -f "$compose_file" up -d --no-deps "$service"
            print_message "$GREEN" "   ✅ $service deployed"
        fi
    done
    
    # Verify deployment
    if [[ "$DRY_RUN" == false ]]; then
        print_message "$BLUE" "🏥 Verifying deployment..."
        sleep 5
        
        for service in "${services[@]}"; do
            if docker-compose -f "$compose_file" ps "$service" | grep -q "Up"; then
                print_message "$GREEN" "   ✅ $service is running"
            else
                print_message "$RED" "   ❌ $service failed to start"
                docker-compose -f "$compose_file" logs --tail=50 "$service"
            fi
        done
    fi
}

# Deploy services to Kubernetes
deploy_k8s() {
    local services=("$@")
    
    if [[ ${#services[@]} -eq 0 ]]; then
        print_message "$YELLOW" "⚠️  No services to deploy"
        return 0
    fi
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        print_message "$RED" "❌ kubectl is not configured"
        exit 1
    fi
    
    print_message "$BLUE" "🚀 Deploying changed services to Kubernetes..."
    
    for service in "${services[@]}"; do
        local manifest_file="k8s/${service}-deployment.yaml"
        
        if [[ ! -f "$manifest_file" ]]; then
            print_message "$YELLOW" "   ⚠️  Manifest not found: $manifest_file"
            continue
        fi
        
        if [[ "$DRY_RUN" == true ]]; then
            print_message "$YELLOW" "   [DRY RUN] Would deploy: $service"
        else
            print_message "$BLUE" "   Deploying $service..."
            kubectl apply -f "$manifest_file"
            
            # Wait for rollout to complete
            print_message "$BLUE" "   Waiting for $service rollout..."
            if kubectl rollout status deployment/$service -n trendyol-gift --timeout=300s; then
                print_message "$GREEN" "   ✅ $service deployed successfully"
            else
                print_message "$RED" "   ❌ $service deployment failed"
                kubectl logs -l app=$service -n trendyol-gift --tail=50
            fi
        fi
    done
    
    # Verify deployment
    if [[ "$DRY_RUN" == false ]]; then
        print_message "$BLUE" "🏥 Verifying deployment..."
        
        for service in "${services[@]}"; do
            local ready_pods=$(kubectl get pods -n trendyol-gift -l app=$service --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
            
            if [[ $ready_pods -gt 0 ]]; then
                print_message "$GREEN" "   ✅ $service has $ready_pods running pod(s)"
            else
                print_message "$RED" "   ❌ $service has no running pods"
            fi
        done
    fi
}

# Main function
main() {
    parse_args "$@"
    
    print_message "$BLUE" "=== Selective Deployment ==="
    print_message "$BLUE" "Environment: $ENVIRONMENT"
    print_message "$BLUE" "Platform: $PLATFORM"
    print_message "$BLUE" "Dry Run: $DRY_RUN"
    print_message "$BLUE" "=========================="
    echo
    
    # Detect changed files
    local changed_files=$(detect_changed_files)
    
    if [[ -z "$changed_files" ]]; then
        print_message "$YELLOW" "⚠️  No changes detected"
        print_message "$YELLOW" "   Run full deployment instead:"
        if [[ "$PLATFORM" == "compose" ]]; then
            print_message "$NC" "   ./scripts/deploy-$ENVIRONMENT.sh"
        else
            print_message "$NC" "   ./scripts/deploy-k8s.sh"
        fi
        exit 0
    fi
    
    print_message "$BLUE" "Changed files:"
    echo "$changed_files" | while read -r file; do
        print_message "$NC" "   - $file"
    done
    echo
    
    # Detect changed services
    local changed_services=($(detect_changed_services "$changed_files"))
    
    if [[ ${#changed_services[@]} -eq 0 ]]; then
        print_message "$GREEN" "✅ No service changes detected"
        print_message "$BLUE" "   Changed files don't affect any services"
        exit 0
    fi
    
    print_message "$GREEN" ""
    print_message "$GREEN" "Services to deploy: ${changed_services[*]}"
    print_message "$GREEN" ""
    
    # Confirm deployment unless dry run
    if [[ "$DRY_RUN" == false ]]; then
        read -p "Continue with deployment? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_message "$YELLOW" "❌ Deployment cancelled"
            exit 0
        fi
    fi
    
    # Deploy based on platform
    if [[ "$PLATFORM" == "compose" ]]; then
        deploy_compose "${changed_services[@]}"
    else
        deploy_k8s "${changed_services[@]}"
    fi
    
    print_message "$GREEN" ""
    print_message "$GREEN" "✅ Selective deployment complete!"
    print_message "$GREEN" ""
    
    if [[ "$DRY_RUN" == false ]]; then
        print_message "$BLUE" "📊 Deployment Summary:"
        print_message "$NC" "   Deployed services: ${changed_services[*]}"
        print_message "$NC" "   Environment: $ENVIRONMENT"
        print_message "$NC" "   Platform: $PLATFORM"
    fi
}

# Run main function
main "$@"
