#!/bin/bash
# Script to verify file permissions in Docker containers follow least privilege principle

set -e

echo "🔒 Verifying file permissions in Docker containers..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check permissions
check_permissions() {
    local container_name=$1
    local path=$2
    local expected_max_perms=$3
    local description=$4
    
    echo "Checking: $description"
    echo "  Path: $path"
    
    # Get actual permissions
    actual_perms=$(docker exec "$container_name" find "$path" -type f -exec stat -c '%a' {} \; 2>/dev/null | sort -u)
    
    if [ -z "$actual_perms" ]; then
        echo -e "  ${YELLOW}⚠ Path not found or no files${NC}"
        return
    fi
    
    # Check each permission
    has_violation=false
    for perm in $actual_perms; do
        # Convert to decimal for comparison
        perm_dec=$((8#$perm))
        max_dec=$((8#$expected_max_perms))
        
        if [ $perm_dec -gt $max_dec ]; then
            echo -e "  ${RED}✗ Found excessive permissions: $perm (max allowed: $expected_max_perms)${NC}"
            has_violation=true
        fi
    done
    
    if [ "$has_violation" = false ]; then
        echo -e "  ${GREEN}✓ Permissions OK${NC}"
    fi
    
    echo ""
}

# Function to check for world-writable files
check_world_writable() {
    local container_name=$1
    local path=$2
    
    echo "Checking for world-writable files in: $path"
    
    world_writable=$(docker exec "$container_name" find "$path" -type f -perm -002 2>/dev/null || true)
    
    if [ -n "$world_writable" ]; then
        echo -e "  ${RED}✗ Found world-writable files:${NC}"
        echo "$world_writable"
    else
        echo -e "  ${GREEN}✓ No world-writable files${NC}"
    fi
    
    echo ""
}

# Function to check for unnecessary executable files
check_unnecessary_executables() {
    local container_name=$1
    local path=$2
    local allowed_extensions=$3
    
    echo "Checking for unnecessary executable files in: $path"
    
    # Find executable files that shouldn't be
    unnecessary_exec=$(docker exec "$container_name" sh -c "
        find $path -type f -executable 2>/dev/null | while read file; do
            ext=\${file##*.}
            case \$ext in
                $allowed_extensions) ;;
                *) echo \$file ;;
            esac
        done
    " || true)
    
    if [ -n "$unnecessary_exec" ]; then
        echo -e "  ${YELLOW}⚠ Found executable files (may be unnecessary):${NC}"
        echo "$unnecessary_exec" | head -10
        if [ $(echo "$unnecessary_exec" | wc -l) -gt 10 ]; then
            echo "  ... and more"
        fi
    else
        echo -e "  ${GREEN}✓ No unnecessary executable files${NC}"
    fi
    
    echo ""
}

# Function to verify non-root user
check_user() {
    local container_name=$1
    local expected_user=$2
    
    echo "Checking container user..."
    
    actual_user=$(docker exec "$container_name" whoami 2>/dev/null || echo "unknown")
    
    if [ "$actual_user" = "$expected_user" ]; then
        echo -e "  ${GREEN}✓ Running as $actual_user (non-root)${NC}"
    elif [ "$actual_user" = "root" ]; then
        echo -e "  ${RED}✗ Running as root (security risk)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Running as $actual_user (expected: $expected_user)${NC}"
    fi
    
    echo ""
}

# Check if containers are running
echo "Checking if containers are running..."
if ! docker ps --format '{{.Names}}' | grep -q "backend"; then
    echo -e "${YELLOW}⚠ Backend container not running. Starting containers...${NC}"
    docker-compose up -d
    sleep 5
fi

echo ""
echo "================================"
echo "Backend Container Checks"
echo "================================"
echo ""

BACKEND_CONTAINER=$(docker ps --filter "name=backend" --format "{{.Names}}" | head -1)

if [ -z "$BACKEND_CONTAINER" ]; then
    echo -e "${RED}✗ Backend container not found${NC}"
else
    check_user "$BACKEND_CONTAINER" "appuser"
    
    check_permissions "$BACKEND_CONTAINER" "/app/*.py" "644" "Python source files"
    check_permissions "$BACKEND_CONTAINER" "/app/packages" "644" "Python packages"
    
    check_world_writable "$BACKEND_CONTAINER" "/app"
    
    check_unnecessary_executables "$BACKEND_CONTAINER" "/app" "sh|bash|py"
fi

echo ""
echo "================================"
echo "Frontend Container Checks"
echo "================================"
echo ""

FRONTEND_CONTAINER=$(docker ps --filter "name=frontend" --format "{{.Names}}" | head -1)

if [ -z "$FRONTEND_CONTAINER" ]; then
    echo -e "${RED}✗ Frontend container not found${NC}"
else
    check_user "$FRONTEND_CONTAINER" "nginx"
    
    check_permissions "$FRONTEND_CONTAINER" "/usr/share/nginx/html" "644" "Static HTML/JS/CSS files"
    check_permissions "$FRONTEND_CONTAINER" "/etc/nginx/conf.d/default.conf" "644" "Nginx configuration"
    
    check_world_writable "$FRONTEND_CONTAINER" "/usr/share/nginx/html"
    
    check_unnecessary_executables "$FRONTEND_CONTAINER" "/usr/share/nginx/html" "none"
fi

echo ""
echo "================================"
echo "Summary"
echo "================================"
echo ""
echo "File permission verification complete!"
echo ""
echo "Best practices:"
echo "  • Application files: 644 (rw-r--r--)"
echo "  • Directories: 755 (rwxr-xr-x)"
echo "  • Executables: 755 (rwxr-xr-x)"
echo "  • No world-writable files"
echo "  • Run as non-root user"
echo ""
