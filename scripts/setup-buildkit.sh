#!/bin/bash
# Setup BuildKit for Docker and Docker Compose
# This script configures BuildKit and verifies the installation

set -e

echo "🔧 Setting up BuildKit..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_success "Docker is installed"

# Check Docker version
DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
print_success "Docker version: $DOCKER_VERSION"

# Enable BuildKit in Docker daemon config
DOCKER_CONFIG_DIR="$HOME/.docker"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"

mkdir -p "$DOCKER_CONFIG_DIR"

if [ -f "$DOCKER_CONFIG_FILE" ]; then
    print_warning "Docker config file exists, backing up..."
    cp "$DOCKER_CONFIG_FILE" "$DOCKER_CONFIG_FILE.backup"
fi

# Create or update Docker config with BuildKit enabled
cat > "$DOCKER_CONFIG_FILE" << 'EOF'
{
  "experimental": "enabled",
  "features": {
    "buildkit": true
  }
}
EOF

print_success "Docker config updated with BuildKit enabled"

# Set environment variables for current session
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
export BUILDKIT_PROGRESS=auto
export BUILDKIT_COLORS=1

print_success "BuildKit environment variables set"

# Add to shell profile for persistence
SHELL_PROFILE=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.profile" ]; then
    SHELL_PROFILE="$HOME/.profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if ! grep -q "DOCKER_BUILDKIT" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Enable Docker BuildKit" >> "$SHELL_PROFILE"
        echo "export DOCKER_BUILDKIT=1" >> "$SHELL_PROFILE"
        echo "export COMPOSE_DOCKER_CLI_BUILD=1" >> "$SHELL_PROFILE"
        echo "export BUILDKIT_PROGRESS=auto" >> "$SHELL_PROFILE"
        print_success "BuildKit variables added to $SHELL_PROFILE"
    else
        print_warning "BuildKit variables already in $SHELL_PROFILE"
    fi
fi

# Verify BuildKit is working
echo ""
echo "🧪 Verifying BuildKit installation..."

# Test BuildKit with a simple build
TEST_DOCKERFILE=$(mktemp)
cat > "$TEST_DOCKERFILE" << 'EOF'
# syntax=docker/dockerfile:1.4
FROM alpine:latest
RUN echo "BuildKit test successful"
EOF

if docker build --progress=plain -f "$TEST_DOCKERFILE" -t buildkit-test . > /dev/null 2>&1; then
    print_success "BuildKit is working correctly"
    docker rmi buildkit-test > /dev/null 2>&1 || true
else
    print_error "BuildKit test failed"
    rm "$TEST_DOCKERFILE"
    exit 1
fi

rm "$TEST_DOCKERFILE"

# Check if docker-compose is installed
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose version --short 2>/dev/null || echo "unknown")
    print_success "docker-compose is installed (version: $COMPOSE_VERSION)"
elif docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    print_success "docker compose (plugin) is installed (version: $COMPOSE_VERSION)"
else
    print_warning "docker-compose is not installed"
fi

echo ""
echo "✅ BuildKit setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
echo "   2. Verify with: docker buildx version"
echo "   3. Start building with BuildKit enabled!"
echo ""
echo "💡 Tip: Use 'docker buildx' for advanced BuildKit features"
