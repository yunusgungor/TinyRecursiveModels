.PHONY: help install dev build test lint format clean docker-up docker-down

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements-dev.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installing pre-commit hooks..."
	pre-commit install

dev: ## Start development environment
	docker-compose up -d

build: ## Build all Docker images
	docker-compose build

test: ## Run all tests
	@echo "Running backend tests..."
	cd backend && pytest
	@echo "Running frontend tests..."
	cd frontend && npm test

test-backend: ## Run backend tests only
	cd backend && pytest -v

test-frontend: ## Run frontend tests only
	cd frontend && npm test

lint: ## Run linters
	@echo "Linting backend..."
	cd backend && black --check . && ruff check . && mypy .
	@echo "Linting frontend..."
	cd frontend && npm run lint

format: ## Format code
	@echo "Formatting backend..."
	cd backend && black . && ruff check --fix .
	@echo "Formatting frontend..."
	cd frontend && npm run format

clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-clean: ## Remove Docker containers and volumes
	docker-compose down -v

setup-env: ## Copy example env files
	@echo "Setting up environment files..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env"; fi
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; echo "Created backend/.env"; fi
	@if [ ! -f frontend/.env ]; then cp frontend/.env.example frontend/.env; echo "Created frontend/.env"; fi

init: setup-env install ## Initialize project (setup env + install deps)
	@echo "Project initialized successfully!"
	@echo "Next steps:"
	@echo "  1. Update .env files with your configuration"
	@echo "  2. Run 'make dev' to start development environment"

# BuildKit targets
setup-buildkit: ## Setup and enable BuildKit
	@echo "Setting up BuildKit..."
	@bash scripts/setup-buildkit.sh

buildkit-env: ## Source BuildKit environment variables
	@echo "Loading BuildKit environment..."
	@echo "Run: source .buildkit.env"

buildkit-verify: ## Verify BuildKit installation
	@bash scripts/verify-buildkit.sh

build-optimized: ## Build with BuildKit optimizations
	@echo "Building with BuildKit optimizations..."
	@export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 && docker-compose build --progress=plain

build-backend: ## Build backend with BuildKit
	@echo "Building backend..."
	@export DOCKER_BUILDKIT=1 && docker build --progress=plain -t backend:latest ./backend

build-frontend: ## Build frontend with BuildKit
	@echo "Building frontend..."
	@export DOCKER_BUILDKIT=1 && docker build --progress=plain -t frontend:latest ./frontend

# Security scanning targets
scan: ## Run all security scans
	@bash scripts/scan-vulnerabilities.sh

scan-images: ## Scan Docker images for vulnerabilities
	@bash scripts/scan-vulnerabilities.sh --images-only

scan-deps: ## Scan dependencies for vulnerabilities
	@bash scripts/scan-vulnerabilities.sh --deps-only

scan-secrets: ## Scan repository for secrets
	@bash scripts/scan-vulnerabilities.sh --secrets-only

scan-critical: ## Scan and fail on critical vulnerabilities
	@bash scripts/scan-vulnerabilities.sh --fail-on-critical

verify-permissions: ## Verify file permissions in containers
	@bash scripts/verify-file-permissions.sh

security-check: scan-critical verify-permissions ## Run all security checks
	@echo "All security checks completed!"
