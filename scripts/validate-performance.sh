#!/bin/bash
# Performance Validation Wrapper Script
# Validates container infrastructure performance metrics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Container Infrastructure Performance Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Check if BuildKit is enabled
if [ -z "$DOCKER_BUILDKIT" ]; then
    echo -e "${YELLOW}Warning: DOCKER_BUILDKIT not set, enabling it...${NC}"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo -e "${GREEN}✓ BuildKit is enabled${NC}"
echo ""

# Run the Python validation script
python3 scripts/validate-performance.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Performance validation completed successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Performance validation failed${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $exit_code
