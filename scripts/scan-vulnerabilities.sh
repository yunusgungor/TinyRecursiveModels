#!/bin/bash
# Script to scan Docker images and dependencies for vulnerabilities using Trivy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Security Vulnerability Scanner${NC}"
echo ""

# Check if Trivy is installed
if ! command -v trivy &> /dev/null; then
    echo -e "${YELLOW}⚠ Trivy not found. Installing...${NC}"
    
    # Detect OS and install Trivy
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install aquasecurity/trivy/trivy
        else
            echo -e "${RED}✗ Homebrew not found. Please install Trivy manually:${NC}"
            echo "  https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
        sudo apt-get update
        sudo apt-get install trivy
    else
        echo -e "${RED}✗ Unsupported OS. Please install Trivy manually:${NC}"
        echo "  https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Trivy installed${NC}"
echo ""

# Function to scan image
scan_image() {
    local image_name=$1
    local service_name=$2
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Scanning $service_name Image: $image_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Check if image exists
    if ! docker image inspect "$image_name" &> /dev/null; then
        echo -e "${YELLOW}⚠ Image not found. Building...${NC}"
        if [ "$service_name" = "Backend" ]; then
            docker build -t "$image_name" --target production ./backend
        elif [ "$service_name" = "Frontend" ]; then
            docker build -t "$image_name" --target production ./frontend
        fi
    fi
    
    echo -e "${YELLOW}Scanning for vulnerabilities...${NC}"
    trivy image \
        --severity CRITICAL,HIGH,MEDIUM \
        --format table \
        "$image_name"
    
    echo ""
    echo -e "${YELLOW}Scanning for secrets...${NC}"
    trivy image \
        --scanners secret \
        --severity CRITICAL,HIGH,MEDIUM \
        --format table \
        "$image_name"
    
    echo ""
    echo -e "${YELLOW}Scanning for misconfigurations...${NC}"
    trivy image \
        --scanners config \
        --severity CRITICAL,HIGH,MEDIUM \
        --format table \
        "$image_name"
    
    echo ""
    
    # Check for critical vulnerabilities
    echo -e "${YELLOW}Checking for critical vulnerabilities...${NC}"
    if trivy image \
        --severity CRITICAL \
        --exit-code 1 \
        --format json \
        --quiet \
        "$image_name" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ No critical vulnerabilities found${NC}"
    else
        echo -e "${RED}✗ Critical vulnerabilities found!${NC}"
        trivy image \
            --severity CRITICAL \
            --format table \
            "$image_name"
        return 1
    fi
    
    echo ""
}

# Function to scan filesystem/dependencies
scan_dependencies() {
    local path=$1
    local name=$2
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Scanning $name Dependencies${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    trivy fs \
        --severity CRITICAL,HIGH,MEDIUM \
        --format table \
        "$path"
    
    echo ""
    
    # Check for critical vulnerabilities
    echo -e "${YELLOW}Checking for critical dependency vulnerabilities...${NC}"
    if trivy fs \
        --severity CRITICAL \
        --exit-code 1 \
        --format json \
        --quiet \
        "$path" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ No critical vulnerabilities in dependencies${NC}"
    else
        echo -e "${RED}✗ Critical vulnerabilities found in dependencies!${NC}"
        trivy fs \
            --severity CRITICAL \
            --format table \
            "$path"
        return 1
    fi
    
    echo ""
}

# Function to scan repository for secrets
scan_secrets() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Scanning Repository for Secrets${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    trivy fs \
        --scanners secret \
        --severity CRITICAL,HIGH,MEDIUM \
        --format table \
        .
    
    echo ""
    
    # Fail if secrets found
    if trivy fs \
        --scanners secret \
        --severity CRITICAL,HIGH,MEDIUM \
        --exit-code 1 \
        --format json \
        --quiet \
        . > /dev/null 2>&1; then
        echo -e "${GREEN}✓ No secrets found in repository${NC}"
    else
        echo -e "${RED}✗ Secrets found in repository!${NC}"
        echo -e "${RED}Please remove secrets and use BuildKit secrets or environment variables${NC}"
        return 1
    fi
    
    echo ""
}

# Parse command line arguments
SCAN_IMAGES=true
SCAN_DEPS=true
SCAN_SECRETS=true
FAIL_ON_CRITICAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --images-only)
            SCAN_DEPS=false
            SCAN_SECRETS=false
            shift
            ;;
        --deps-only)
            SCAN_IMAGES=false
            SCAN_SECRETS=false
            shift
            ;;
        --secrets-only)
            SCAN_IMAGES=false
            SCAN_DEPS=false
            shift
            ;;
        --fail-on-critical)
            FAIL_ON_CRITICAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --images-only        Scan only Docker images"
            echo "  --deps-only          Scan only dependencies"
            echo "  --secrets-only       Scan only for secrets"
            echo "  --fail-on-critical   Exit with error if critical vulnerabilities found"
            echo "  --help               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Track failures
FAILED=false

# Scan images
if [ "$SCAN_IMAGES" = true ]; then
    if ! scan_image "backend:latest" "Backend"; then
        FAILED=true
    fi
    
    if ! scan_image "frontend:latest" "Frontend"; then
        FAILED=true
    fi
fi

# Scan dependencies
if [ "$SCAN_DEPS" = true ]; then
    if ! scan_dependencies "./backend" "Backend"; then
        FAILED=true
    fi
    
    if ! scan_dependencies "./frontend" "Frontend"; then
        FAILED=true
    fi
fi

# Scan for secrets
if [ "$SCAN_SECRETS" = true ]; then
    if ! scan_secrets; then
        FAILED=true
    fi
fi

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Scan Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$FAILED" = true ]; then
    echo -e "${RED}✗ Security scan completed with issues${NC}"
    echo ""
    echo "Recommendations:"
    echo "  1. Update dependencies to latest secure versions"
    echo "  2. Review and fix critical vulnerabilities"
    echo "  3. Remove any secrets from repository"
    echo "  4. Use BuildKit secrets for build-time secrets"
    echo "  5. Use environment variables for runtime secrets"
    echo ""
    
    if [ "$FAIL_ON_CRITICAL" = true ]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ All security scans passed!${NC}"
    echo ""
fi

echo "For detailed vulnerability information, visit:"
echo "  https://aquasecurity.github.io/trivy/"
echo ""
