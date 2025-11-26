#!/bin/bash
# Verify BuildKit setup and configuration
# This script checks if BuildKit is properly configured and working

# Don't exit on error, we want to collect all results
set +e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Start verification
echo ""
echo "🔍 BuildKit Verification"
echo "========================"

# 1. Check Docker installation
print_header "1. Docker Installation"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    print_success "Docker is installed (version: $DOCKER_VERSION)"
    
    # Check if version is >= 18.09 (simplified check)
    if [ "$DOCKER_VERSION" != "unknown" ]; then
        MAJOR=$(echo "$DOCKER_VERSION" | cut -d. -f1)
        if [ "$MAJOR" -ge 18 ] 2>/dev/null; then
            print_success "Docker version supports BuildKit (>= 18.09)"
        else
            print_warning "Could not verify Docker version compatibility"
        fi
    fi
else
    print_error "Docker is not installed"
fi

# 2. Check BuildKit environment variables
print_header "2. Environment Variables"

if [ "$DOCKER_BUILDKIT" = "1" ]; then
    print_success "DOCKER_BUILDKIT is set to 1"
else
    print_warning "DOCKER_BUILDKIT is not set (current: ${DOCKER_BUILDKIT:-not set})"
    print_info "Run: export DOCKER_BUILDKIT=1"
fi

if [ "$COMPOSE_DOCKER_CLI_BUILD" = "1" ]; then
    print_success "COMPOSE_DOCKER_CLI_BUILD is set to 1"
else
    print_warning "COMPOSE_DOCKER_CLI_BUILD is not set (current: ${COMPOSE_DOCKER_CLI_BUILD:-not set})"
    print_info "Run: export COMPOSE_DOCKER_CLI_BUILD=1"
fi

if [ -n "$BUILDKIT_PROGRESS" ]; then
    print_success "BUILDKIT_PROGRESS is set to: $BUILDKIT_PROGRESS"
else
    print_warning "BUILDKIT_PROGRESS is not set"
fi

# 3. Check Docker config
print_header "3. Docker Configuration"

DOCKER_CONFIG="$HOME/.docker/config.json"
if [ -f "$DOCKER_CONFIG" ]; then
    print_success "Docker config file exists"
    
    if grep -q '"buildkit"' "$DOCKER_CONFIG" 2>/dev/null; then
        print_success "BuildKit is enabled in Docker config"
    else
        print_warning "BuildKit not found in Docker config"
    fi
else
    print_warning "Docker config file not found at $DOCKER_CONFIG"
fi

# 4. Check BuildKit availability
print_header "4. BuildKit Availability"

if docker buildx version &> /dev/null; then
    BUILDX_VERSION=$(docker buildx version 2>/dev/null | awk '{print $2}')
    print_success "docker buildx is available (version: $BUILDX_VERSION)"
else
    print_error "docker buildx is not available"
fi

# 5. Check project files
print_header "5. Project Configuration Files"

if [ -f ".buildkit.env" ]; then
    print_success ".buildkit.env file exists"
else
    print_error ".buildkit.env file not found"
fi

if [ -f ".buildkitconfig.toml" ]; then
    print_success ".buildkitconfig.toml file exists"
else
    print_error ".buildkitconfig.toml file not found"
fi

if [ -f "backend/.dockerignore" ]; then
    print_success "backend/.dockerignore file exists"
else
    print_error "backend/.dockerignore file not found"
fi

if [ -f "frontend/.dockerignore" ]; then
    print_success "frontend/.dockerignore file exists"
else
    print_error "frontend/.dockerignore file not found"
fi

# 6. Check Dockerfiles
print_header "6. Dockerfile Syntax"

if [ -f "backend/Dockerfile" ]; then
    if grep -q "syntax=docker/dockerfile" "backend/Dockerfile"; then
        print_success "backend/Dockerfile has BuildKit syntax directive"
    else
        print_warning "backend/Dockerfile missing BuildKit syntax directive"
        print_info "Add: # syntax=docker/dockerfile:1.4"
    fi
    
    if grep -q "mount=type=cache" "backend/Dockerfile"; then
        print_success "backend/Dockerfile uses cache mounts"
    else
        print_warning "backend/Dockerfile not using cache mounts"
    fi
else
    print_error "backend/Dockerfile not found"
fi

if [ -f "frontend/Dockerfile" ]; then
    if grep -q "syntax=docker/dockerfile" "frontend/Dockerfile"; then
        print_success "frontend/Dockerfile has BuildKit syntax directive"
    else
        print_warning "frontend/Dockerfile missing BuildKit syntax directive"
        print_info "Add: # syntax=docker/dockerfile:1.4"
    fi
    
    if grep -q "mount=type=cache" "frontend/Dockerfile"; then
        print_success "frontend/Dockerfile uses cache mounts"
    else
        print_warning "frontend/Dockerfile not using cache mounts"
    fi
else
    print_error "frontend/Dockerfile not found"
fi

# 7. Test BuildKit functionality
print_header "7. BuildKit Functionality Test"

print_info "Creating test Dockerfile..."
TEST_DIR=$(mktemp -d)
TEST_DOCKERFILE="$TEST_DIR/Dockerfile"

cat > "$TEST_DOCKERFILE" << 'EOF'
# syntax=docker/dockerfile:1.4
FROM alpine:latest
RUN --mount=type=cache,target=/tmp/cache \
    echo "BuildKit cache mount test" > /tmp/cache/test.txt
RUN echo "BuildKit test successful"
EOF

print_info "Running test build..."
if DOCKER_BUILDKIT=1 docker build --progress=plain -f "$TEST_DOCKERFILE" -t buildkit-test "$TEST_DIR" > /dev/null 2>&1; then
    print_success "BuildKit test build successful"
    docker rmi buildkit-test > /dev/null 2>&1 || true
else
    print_error "BuildKit test build failed"
fi

rm -rf "$TEST_DIR"

# 8. Check docker-compose
print_header "8. Docker Compose"

if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose version --short 2>/dev/null || echo "unknown")
    print_success "docker-compose is installed (version: $COMPOSE_VERSION)"
elif docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    print_success "docker compose (plugin) is installed (version: $COMPOSE_VERSION)"
else
    print_warning "docker-compose is not installed"
fi

if [ -f "docker-compose.yml" ]; then
    print_success "docker-compose.yml exists"
    
    if grep -q "cache_from" "docker-compose.yml"; then
        print_success "docker-compose.yml uses cache_from"
    else
        print_warning "docker-compose.yml not using cache_from"
    fi
    
    if grep -q "BUILDKIT_INLINE_CACHE" "docker-compose.yml"; then
        print_success "docker-compose.yml uses BUILDKIT_INLINE_CACHE"
    else
        print_warning "docker-compose.yml not using BUILDKIT_INLINE_CACHE"
    fi
else
    print_error "docker-compose.yml not found"
fi

# Summary
print_header "Summary"

TOTAL=$((PASSED + FAILED + WARNINGS))
echo ""
echo "Results:"
echo "  ✓ Passed:   $PASSED"
echo "  ⚠ Warnings: $WARNINGS"
echo "  ✗ Failed:   $FAILED"
echo "  ━━━━━━━━━━━━━━━━"
echo "  Total:      $TOTAL"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✅ All checks passed! BuildKit is properly configured.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  BuildKit is working but there are some warnings.${NC}"
        echo "   Review the warnings above for optimization opportunities."
        exit 0
    fi
else
    echo -e "${RED}❌ Some checks failed. Please fix the errors above.${NC}"
    echo ""
    echo "Quick fixes:"
    echo "  1. Run: make setup-buildkit"
    echo "  2. Run: source .buildkit.env"
    echo "  3. Restart your terminal"
    exit 1
fi
