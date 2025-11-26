# Service Isolation Guide

## Overview

This document describes the service isolation architecture implemented in the container infrastructure. Each service is designed to restart independently without affecting other services.

## Service Isolation Principles

### 1. No Shared State Between Services

Each service maintains its own state:

- **PostgreSQL**: State in `postgres_data` named volume
- **Redis**: State in `redis_data` named volume  
- **Backend**: Stateless application, state in databases
- **Frontend**: Stateless application, node_modules in isolated volume
- **Nginx**: Stateless proxy with read-only config

### 2. Independent Restart Capability

All services can be restarted individually:

```bash
# Restart a single service
docker-compose restart backend

# Or use the helper script
./scripts/restart-service.sh backend
```

### 3. Health Check Dependencies

Services use health checks to ensure dependencies are ready:

```yaml
depends_on:
  postgres:
    condition: service_healthy
  redis:
    condition: service_healthy
```

This ensures:
- Backend waits for databases to be healthy before starting
- Restarting backend doesn't restart databases
- Databases can restart without restarting backend (backend will reconnect)

## Service Restart Behavior

### PostgreSQL Restart

**Impact**: Minimal
- Backend connections will be temporarily interrupted
- Backend will automatically reconnect when PostgreSQL is healthy
- No data loss (state persisted in volume)
- Other services unaffected

**Command**:
```bash
docker-compose restart postgres
```

### Redis Restart

**Impact**: Minimal
- Cache will be temporarily unavailable
- Backend will handle cache misses gracefully
- No permanent data loss (AOF persistence enabled)
- Other services unaffected

**Command**:
```bash
docker-compose restart redis
```

### Backend Restart

**Impact**: Moderate
- API temporarily unavailable
- Frontend will show connection errors
- Database connections preserved
- No impact on databases or other services

**Command**:
```bash
docker-compose restart backend
```

### Frontend Restart

**Impact**: Minimal
- Development server restarts
- Browser will reconnect automatically (HMR)
- No impact on backend or databases

**Command**:
```bash
docker-compose restart frontend
```

### Nginx Restart

**Impact**: Minimal (production only)
- Proxy temporarily unavailable
- Upstream services continue running
- Connections re-established immediately

**Command**:
```bash
docker-compose restart nginx
```

## Verification

### Testing Service Isolation

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **Verify all services are running**:
   ```bash
   docker-compose ps
   ```

3. **Restart a single service**:
   ```bash
   ./scripts/restart-service.sh backend
   ```

4. **Verify other services remain running**:
   ```bash
   docker-compose ps
   ```

Expected result: Only the restarted service shows a recent restart time.

### Monitoring Service Status

```bash
# Check all service statuses
docker-compose ps

# Check specific service logs
docker-compose logs -f backend

# Check service health
docker inspect --format='{{.State.Health.Status}}' trendyol-gift-backend
```

## Configuration for Isolation

### Volume Isolation

Each service uses dedicated volumes:

```yaml
volumes:
  postgres_data:      # PostgreSQL only
  redis_data:         # Redis only
  backend-packages:   # Backend pip cache only
  frontend-node-modules:  # Frontend npm cache only
```

### Network Isolation

All services share a bridge network but:
- No shared file systems (except read-only mounts)
- No shared memory
- No shared process namespaces

### Resource Isolation

Each service has defined resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

This ensures:
- One service can't consume all resources
- Service restarts don't affect resource allocation of others

## Troubleshooting

### Service Won't Restart

1. Check logs:
   ```bash
   docker-compose logs backend
   ```

2. Check dependencies:
   ```bash
   docker-compose ps postgres redis
   ```

3. Verify health checks:
   ```bash
   docker inspect --format='{{.State.Health}}' trendyol-gift-postgres
   ```

### Dependent Services Affected

If restarting a service affects dependents:

1. Check if dependent services have proper reconnection logic
2. Verify health checks are configured correctly
3. Check for shared volumes (should be none except read-only)

### State Loss After Restart

If state is lost after restart:

1. Verify named volumes are used (not bind mounts for state)
2. Check volume persistence:
   ```bash
   docker volume ls
   docker volume inspect tinyrecursivemodels_postgres_data
   ```

## Best Practices

1. **Always use health checks** for services with dependencies
2. **Use named volumes** for persistent state
3. **Avoid shared writable volumes** between services
4. **Implement reconnection logic** in application code
5. **Test service isolation** regularly
6. **Monitor restart counts** to detect issues

## Configuration Change Detection

The project includes automated configuration change detection that triggers selective service restarts.

### Using the Configuration Watcher

**Start the watcher**:
```bash
python scripts/config_watcher.py watch
```

**List watched files**:
```bash
python scripts/config_watcher.py list
```

**Test which services would be affected**:
```bash
python scripts/config_watcher.py test backend/.env
```

**Manually trigger restart for a config file**:
```bash
python scripts/config_watcher.py restart backend/requirements.txt
```

### Watched Configuration Files

The watcher monitors these files and restarts affected services:

- `.env`, `.env.development`, `.env.production` → backend, frontend
- `backend/.env` → backend
- `backend/requirements.txt`, `backend/requirements-dev.txt` → backend
- `frontend/.env` → frontend
- `frontend/package.json`, `frontend/package-lock.json` → frontend
- `nginx/nginx.conf`, `nginx/conf.d/default.conf` → nginx
- `docker-compose.yml`, `docker-compose.prod.yml` → all services (manual restart required)

### Configuration

Set environment variables to customize behavior:

```bash
# Watch interval in seconds (default: 5)
export WATCH_INTERVAL=10

# Enable debug output
export DEBUG=true

# Start watcher
python scripts/config_watcher.py watch
```

## Related Documentation

- [Docker Compose Configuration](../docker-compose.yml)
- [Health Check Configuration](./HEALTH_CHECKS.md)
- [Development Workflow](./DEVELOPER_GUIDE.md)
- [Configuration Watcher Script](../scripts/config_watcher.py)
