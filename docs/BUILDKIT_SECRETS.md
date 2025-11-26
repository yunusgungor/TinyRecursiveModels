# BuildKit Secret Management Guide

## Overview

This guide explains how to use BuildKit secrets for secure build-time secret management. BuildKit secrets ensure that sensitive data (API keys, tokens, credentials) are never stored in Docker image layers.

## Why BuildKit Secrets?

Traditional approaches like `ARG` or `ENV` in Dockerfiles expose secrets in:
- Image layers (visible with `docker history`)
- Build cache
- Final image

BuildKit secrets provide:
- **Ephemeral access**: Secrets are mounted temporarily during build
- **No layer persistence**: Secrets never appear in image layers
- **Cache safety**: Secrets don't invalidate cache unnecessarily

## Basic Usage

### 1. Using Secrets in Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim as builder

# Mount secret during build (not persisted in image)
RUN --mount=type=secret,id=pip_token \
    PIP_TOKEN=$(cat /run/secrets/pip_token) && \
    pip install --index-url https://${PIP_TOKEN}@private-pypi.example.com/simple/ \
    private-package
```

### 2. Providing Secrets During Build

#### From File
```bash
docker build \
  --secret id=pip_token,src=./secrets/pip_token.txt \
  -t myapp:latest .
```

#### From Environment Variable
```bash
export PIP_TOKEN="secret-token-value"
docker build \
  --secret id=pip_token,env=PIP_TOKEN \
  -t myapp:latest .
```

#### From stdin
```bash
echo "secret-token-value" | docker build \
  --secret id=pip_token \
  -t myapp:latest .
```

## Common Use Cases

### Private Package Registry Authentication

#### Python (pip)
```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .

# Use secret for private PyPI authentication
RUN --mount=type=secret,id=pip_token \
    pip install \
    --index-url https://$(cat /run/secrets/pip_token)@pypi.example.com/simple/ \
    -r requirements.txt \
    --target=/app/packages
```

#### Node.js (npm)
```dockerfile
# syntax=docker/dockerfile:1.4

FROM node:20-alpine as builder

WORKDIR /app
COPY package.json package-lock.json ./

# Use secret for private npm registry
RUN --mount=type=secret,id=npm_token \
    echo "//registry.npmjs.org/:_authToken=$(cat /run/secrets/npm_token)" > .npmrc && \
    npm ci && \
    rm -f .npmrc
```

### Git Repository Access

```dockerfile
# syntax=docker/dockerfile:1.4

FROM alpine:latest as builder

# Install git
RUN apk add --no-cache git

# Clone private repository using SSH key
RUN --mount=type=secret,id=ssh_key,target=/root/.ssh/id_rsa \
    --mount=type=ssh \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git clone git@github.com:private/repo.git /app
```

### API Key for Build-Time Data Fetching

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim as builder

WORKDIR /app

# Fetch configuration from API during build
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) && \
    curl -H "Authorization: Bearer ${API_KEY}" \
    https://api.example.com/config > config.json
```

## Docker Compose Integration

### Development Environment

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      secrets:
        - pip_token
    environment:
      - ENVIRONMENT=development

secrets:
  pip_token:
    file: ./secrets/pip_token.txt
```

### Using Environment Variables

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      secrets:
        - pip_token

secrets:
  pip_token:
    environment: PIP_TOKEN
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build with Secrets

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build with secrets
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          push: true
          tags: myapp:latest
          secrets: |
            "pip_token=${{ secrets.PIP_TOKEN }}"
            "npm_token=${{ secrets.NPM_TOKEN }}"
```

### GitLab CI

```yaml
build:
  image: docker:latest
  services:
    - docker:dind
  variables:
    DOCKER_BUILDKIT: 1
  script:
    - echo "$PIP_TOKEN" | docker build 
      --secret id=pip_token 
      -t myapp:latest .
  only:
    - main
```

## Best Practices

### 1. Never Use ARG for Secrets

❌ **Bad** - Secret exposed in image layers:
```dockerfile
ARG API_KEY
RUN curl -H "Authorization: Bearer ${API_KEY}" ...
```

✅ **Good** - Secret not persisted:
```dockerfile
RUN --mount=type=secret,id=api_key \
    curl -H "Authorization: Bearer $(cat /run/secrets/api_key)" ...
```

### 2. Clean Up Temporary Files

Even with secrets, ensure temporary files are cleaned:
```dockerfile
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) && \
    curl -H "Authorization: Bearer ${API_KEY}" https://api.example.com/data > data.json && \
    # Process data.json \
    rm -f data.json  # Clean up if it contains sensitive info
```

### 3. Use Multi-Stage Builds

Combine secrets with multi-stage builds to ensure secrets are only in builder stages:

```dockerfile
# syntax=docker/dockerfile:1.4

# Builder stage - uses secrets
FROM python:3.10-slim as builder

RUN --mount=type=secret,id=pip_token \
    pip install --index-url https://$(cat /run/secrets/pip_token)@pypi.example.com/simple/ \
    private-package \
    --target=/app/packages

# Production stage - no secrets, only compiled packages
FROM python:3.10-slim as production

COPY --from=builder /app/packages /app/packages
# Secret never reaches this stage
```

### 4. Validate Secret Presence

Add validation to fail fast if secrets are missing:

```dockerfile
RUN --mount=type=secret,id=api_key \
    if [ ! -f /run/secrets/api_key ]; then \
        echo "Error: api_key secret not provided" && exit 1; \
    fi && \
    API_KEY=$(cat /run/secrets/api_key) && \
    # Use API_KEY
```

### 5. Use Required Flag

Make secrets required to prevent silent failures:

```dockerfile
# This will fail the build if secret is not provided
RUN --mount=type=secret,id=api_key,required=true \
    curl -H "Authorization: Bearer $(cat /run/secrets/api_key)" ...
```

## Security Verification

### Verify Secrets Not in Image

After building, verify secrets are not in the image:

```bash
# Build image
docker build --secret id=api_key,src=./secret.txt -t myapp:latest .

# Check image history (should not show secret)
docker history myapp:latest

# Inspect image layers (should not contain secret)
docker save myapp:latest | tar -xO | grep -a "secret-value" || echo "Secret not found (good!)"

# Run container and check filesystem
docker run --rm myapp:latest find / -name "*secret*" 2>/dev/null
```

### Automated Secret Scanning

Use tools to scan for accidentally committed secrets:

```bash
# Install gitleaks
brew install gitleaks

# Scan repository
gitleaks detect --source . --verbose

# Scan Docker image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --scanners secret myapp:latest
```

## Troubleshooting

### Secret Not Found During Build

**Error**: `cat: /run/secrets/api_key: No such file or directory`

**Solutions**:
1. Ensure BuildKit is enabled: `export DOCKER_BUILDKIT=1`
2. Verify secret ID matches: `--secret id=api_key` in build command
3. Check secret source exists: file path or environment variable

### Secret Appears in Build Output

**Issue**: Secret value visible in build logs

**Solution**: Avoid echoing or printing secret values:
```dockerfile
# Bad - prints secret
RUN --mount=type=secret,id=api_key \
    echo "Using key: $(cat /run/secrets/api_key)"

# Good - uses secret without printing
RUN --mount=type=secret,id=api_key \
    API_KEY=$(cat /run/secrets/api_key) && \
    curl -H "Authorization: Bearer ${API_KEY}" ... > /dev/null 2>&1
```

### Docker Compose Secret Not Working

**Issue**: Secret not available in build

**Solution**: Ensure Docker Compose version supports build secrets (v3.8+):
```yaml
version: '3.8'  # Minimum version for build secrets

services:
  app:
    build:
      context: .
      secrets:
        - my_secret

secrets:
  my_secret:
    file: ./secret.txt
```

## Examples in This Project

### Backend Private Package Installation

See `backend/Dockerfile` for example of using secrets with pip:
```bash
# Build with private PyPI token
docker build \
  --secret id=pip_token,src=./secrets/pip_token.txt \
  --target production \
  -t backend:latest \
  ./backend
```

### Frontend Private npm Registry

See `frontend/Dockerfile` for example of using secrets with npm:
```bash
# Build with npm token
docker build \
  --secret id=npm_token,env=NPM_TOKEN \
  --target production \
  -t frontend:latest \
  ./frontend
```

## References

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Docker Build Secrets](https://docs.docker.com/build/building/secrets/)
- [BuildKit Syntax Reference](https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md)
