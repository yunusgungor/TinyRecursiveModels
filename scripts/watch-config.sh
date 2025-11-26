#!/bin/bash

# Configuration Change Detection and Selective Service Restart
# This script watches configuration files and triggers selective service restarts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_debug() {
    if [ "$DEBUG" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Configuration file to service mapping
declare -A CONFIG_SERVICE_MAP=(
    [".env"]="backend frontend"
    [".env.development"]="backend frontend"
    [".env.production"]="backend frontend"
    ["backend/.env"]="backend"
    ["backend/requirements.txt"]="backend"
    ["backend/requirements-dev.txt"]="backend"
    ["frontend/.env"]="frontend"
    ["frontend/package.json"]="frontend"
    ["frontend/package-lock.json"]="frontend"
    ["nginx/nginx.conf"]="nginx"
    ["nginx/conf.d/default.conf"]="nginx"
    ["docker-compose.yml"]="all"
    ["docker-compose.prod.yml"]="all"
)

# Function to get affected services for a config file
get_affected_services() {
    local config_file=$1
    local services=""
    
    # Check exact match first
    if [ -n "${CONFIG_SERVICE_MAP[$config_file]}" ]; then
        services="${CONFIG_SERVICE_MAP[$config_file]}"
    else
        # Check for pattern matches
        for pattern in "${!CONFIG_SERVICE_MAP[@]}"; do
            if [[ "$config_file" == *"$pattern"* ]]; then
                services="${CONFIG_SERVICE_MAP[$pattern]}"
                break
            fi
        done
    fi
    
    echo "$services"
}

# Function to restart affected services
restart_affected_services() {
    local config_file=$1
    local services=$(get_affected_services "$config_file")
    
    if [ -z "$services" ]; then
        print_debug "No services affected by $config_file"
        return
    fi
    
    print_info "Configuration file changed: $config_file"
    print_info "Affected services: $services"
    
    if [ "$services" = "all" ]; then
        print_warning "docker-compose.yml changed - this may require full restart"
        print_info "Run: docker-compose up -d --force-recreate"
        return
    fi
    
    # Restart each affected service
    for service in $services; do
        print_info "Restarting service: $service"
        
        if docker-compose ps --services | grep -q "^${service}$"; then
            docker-compose restart "$service"
            print_info "Service $service restarted successfully"
        else
            print_warning "Service $service not found or not running"
        fi
    done
}

# Function to calculate file checksum
get_file_checksum() {
    local file=$1
    if [ -f "$file" ]; then
        if command -v md5sum &> /dev/null; then
            md5sum "$file" | cut -d' ' -f1
        elif command -v md5 &> /dev/null; then
            md5 -q "$file"
        else
            stat -f%m "$file" 2>/dev/null || stat -c%Y "$file" 2>/dev/null
        fi
    fi
}

# Function to watch configuration files
watch_configs() {
    local watch_interval=${WATCH_INTERVAL:-5}
    
    print_info "=== Configuration Change Watcher ==="
    print_info "Watch interval: ${watch_interval}s"
    print_info "Monitoring configuration files..."
    print_info "Press Ctrl+C to stop"
    print_info "===================================="
    
    # Initialize checksums
    declare -A file_checksums
    
    for config_file in "${!CONFIG_SERVICE_MAP[@]}"; do
        if [ -f "$config_file" ]; then
            file_checksums[$config_file]=$(get_file_checksum "$config_file")
            print_debug "Watching: $config_file"
        fi
    done
    
    # Watch loop
    while true; do
        sleep "$watch_interval"
        
        for config_file in "${!CONFIG_SERVICE_MAP[@]}"; do
            if [ ! -f "$config_file" ]; then
                continue
            fi
            
            local current_checksum=$(get_file_checksum "$config_file")
            local previous_checksum="${file_checksums[$config_file]}"
            
            if [ "$current_checksum" != "$previous_checksum" ]; then
                print_info "Change detected in: $config_file"
                
                # Update checksum
                file_checksums[$config_file]=$current_checksum
                
                # Restart affected services
                restart_affected_services "$config_file"
            fi
        done
    done
}

# Function to list watched files
list_watched_files() {
    print_info "=== Watched Configuration Files ==="
    
    for config_file in "${!CONFIG_SERVICE_MAP[@]}"; do
        local services="${CONFIG_SERVICE_MAP[$config_file]}"
        local exists="❌"
        
        if [ -f "$config_file" ]; then
            exists="✅"
        fi
        
        echo "$exists $config_file → $services"
    done
    
    print_info "===================================="
}

# Function to test configuration change detection
test_config_detection() {
    local test_file=$1
    
    if [ -z "$test_file" ]; then
        print_error "Usage: $0 test <config-file>"
        exit 1
    fi
    
    print_info "Testing configuration change detection for: $test_file"
    
    local services=$(get_affected_services "$test_file")
    
    if [ -z "$services" ]; then
        print_warning "No services configured for $test_file"
        print_info "Add mapping to CONFIG_SERVICE_MAP in script"
    else
        print_info "Affected services: $services"
        
        if [ "$services" = "all" ]; then
            print_warning "This would trigger a full stack restart"
        else
            print_info "Would restart: $services"
        fi
    fi
}

# Main script
case "${1:-watch}" in
    watch)
        watch_configs
        ;;
    list)
        list_watched_files
        ;;
    test)
        test_config_detection "$2"
        ;;
    restart)
        if [ -z "$2" ]; then
            print_error "Usage: $0 restart <config-file>"
            exit 1
        fi
        restart_affected_services "$2"
        ;;
    *)
        echo "Usage: $0 {watch|list|test|restart} [config-file]"
        echo ""
        echo "Commands:"
        echo "  watch              Watch configuration files and auto-restart services"
        echo "  list               List all watched configuration files"
        echo "  test <file>        Test which services would be affected by a file change"
        echo "  restart <file>     Manually trigger restart for services affected by a file"
        echo ""
        echo "Environment variables:"
        echo "  WATCH_INTERVAL     Seconds between checks (default: 5)"
        echo "  DEBUG              Enable debug output (true/false)"
        exit 1
        ;;
esac
