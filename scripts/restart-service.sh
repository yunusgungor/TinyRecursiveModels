#!/bin/bash

# Selective Service Restart Script
# This script allows restarting individual services without affecting others
# Validates service isolation and ensures no shared state issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if service exists
service_exists() {
    local service=$1
    docker-compose ps --services | grep -q "^${service}$"
}

# Function to get service status
get_service_status() {
    local service=$1
    docker-compose ps --format json "$service" 2>/dev/null | jq -r '.[0].State' 2>/dev/null || echo "not found"
}

# Function to restart a single service
restart_service() {
    local service=$1
    
    print_info "Checking if service '$service' exists..."
    
    if ! service_exists "$service"; then
        print_error "Service '$service' not found in docker-compose.yml"
        print_info "Available services:"
        docker-compose ps --services
        exit 1
    fi
    
    print_info "Current status of '$service': $(get_service_status "$service")"
    
    # Get list of dependent services (services that depend on this one)
    print_info "Checking for dependent services..."
    local dependent_services=$(docker-compose config | grep -A 10 "depends_on:" | grep -B 5 "$service" | grep "^  [a-z]" | cut -d: -f1 | tr -d ' ' | grep -v "$service" || true)
    
    if [ -n "$dependent_services" ]; then
        print_warning "The following services depend on '$service':"
        echo "$dependent_services"
        print_warning "They may experience temporary connection issues during restart"
    fi
    
    # Restart the service
    print_info "Restarting service '$service'..."
    docker-compose restart "$service"
    
    # Wait for health check if available
    print_info "Waiting for service to become healthy..."
    local max_wait=30
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        local health=$(docker inspect --format='{{.State.Health.Status}}' "$(docker-compose ps -q "$service" 2>/dev/null)" 2>/dev/null || echo "none")
        
        if [ "$health" = "healthy" ]; then
            print_info "Service '$service' is healthy!"
            break
        elif [ "$health" = "none" ]; then
            # No health check defined, just check if running
            local status=$(get_service_status "$service")
            if [ "$status" = "running" ]; then
                print_info "Service '$service' is running (no health check defined)"
                break
            fi
        fi
        
        sleep 1
        waited=$((waited + 1))
    done
    
    if [ $waited -ge $max_wait ]; then
        print_warning "Service '$service' did not become healthy within ${max_wait}s"
        print_info "Check logs with: docker-compose logs $service"
    fi
    
    # Verify other services are still running
    print_info "Verifying other services are unaffected..."
    local all_services=$(docker-compose ps --services)
    local affected_count=0
    
    for svc in $all_services; do
        if [ "$svc" != "$service" ]; then
            local status=$(get_service_status "$svc")
            if [ "$status" != "running" ]; then
                print_warning "Service '$svc' is not running (status: $status)"
                affected_count=$((affected_count + 1))
            fi
        fi
    done
    
    if [ $affected_count -eq 0 ]; then
        print_info "All other services remain running - isolation verified!"
    else
        print_warning "$affected_count other service(s) are not running"
    fi
    
    print_info "Service restart complete!"
}

# Main script
if [ $# -eq 0 ]; then
    print_error "Usage: $0 <service-name>"
    print_info "Available services:"
    docker-compose ps --services
    exit 1
fi

SERVICE_NAME=$1

print_info "=== Selective Service Restart ==="
print_info "Target service: $SERVICE_NAME"
print_info "================================="

restart_service "$SERVICE_NAME"
