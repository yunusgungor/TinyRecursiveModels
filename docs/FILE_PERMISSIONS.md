# File Permission Security Guide

## Overview

This guide explains how to implement least privilege file permissions in Docker containers to minimize security risks. Proper file permissions prevent unauthorized access and limit the impact of potential security breaches.

## Least Privilege Principle

The least privilege principle states that:
- Users and processes should have only the minimum permissions necessary to perform their tasks
- Files should be readable/writable/executable only by those who need that access
- Default permissions should be restrictive, not permissive

## Permission Notation

### Numeric (Octal) Notation
```
4 = read (r)
2 = write (w)
1 = execute (x)

Format: [owner][group][others]
Example: 644 = rw-r--r--
  6 (4+2) = owner can read and write
  4 = group can read
  4 = others can read
```

### Symbolic Notation
```
u = user/owner
g = group
o = others
a = all

r = read
w = write
x = execute
X = execute only if file is directory or already executable

Example: u=rw,g=r,o= means owner can read/write, group can read, others have no access
```

## Recommended Permissions

### Application Files

| File Type | Permissions | Numeric | Reason |
|-----------|-------------|---------|--------|
| Python source (.py) | rw-r----- | 640 | Owner can modify, group can read, no public access |
| JavaScript source (.js) | rw-r----- | 640 | Same as above |
| Configuration files | rw-r----- | 640 | Prevent unauthorized modification |
| Static HTML/CSS | r--r----- | 440 | Read-only, no write or execute needed |
| Shell scripts (.sh) | rwxr-x--- | 750 | Executable by owner and group only |
| Log files | rw-r----- | 640 | Owner writes, group reads |
| Data files | rw------- | 600 | Owner only access for sensitive data |

### Directories

| Directory Type | Permissions | Numeric | Reason |
|----------------|-------------|---------|--------|
| Application root | rwxr-x--- | 750 | Owner can modify, group can read/execute |
| Static assets | rwxr-x--- | 750 | Same as above |
| Temporary directories | rwx------ | 700 | Owner only access |
| Log directories | rwxr-x--- | 750 | Owner writes, group reads |

### Package Directories

| Directory Type | Permissions | Numeric | Reason |
|----------------|-------------|---------|--------|
| Python packages | r-xr-x--- | 550 | Read and execute only, no write |
| Node modules | r-xr-x--- | 550 | Same as above |

## Implementation in Dockerfiles

### Backend (Python) Example

```dockerfile
FROM python:3.10-slim as production

WORKDIR /app

# Copy application files
COPY --from=builder /app/packages /app/packages
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    # Set minimal permissions
    # u=rwX: owner can read, write, and execute directories
    # g=rX: group can read and execute directories
    # o=: others have no access
    chmod -R u=rwX,g=rX,o= /app && \
    # Python files should not be executable
    find /app -type f -name "*.py" -exec chmod u=rw,g=r,o= {} \; && \
    # Shell scripts should be executable
    find /app -type f -name "*.sh" -exec chmod u=rwx,g=rx,o= {} \; && \
    # Packages should be read-only
    chmod -R u=rX,g=rX,o= /app/packages

USER appuser
```

### Frontend (Nginx) Example

```dockerfile
FROM nginx:alpine as production

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN chown -R nginx:nginx /usr/share/nginx/html && \
    # Static files should be readable only
    find /usr/share/nginx/html -type f -exec chmod u=r,g=r,o= {} \; && \
    find /usr/share/nginx/html -type d -exec chmod u=rx,g=rx,o= {} \; && \
    # Config should be readable only
    chmod u=r,g=r,o= /etc/nginx/conf.d/default.conf && \
    # Cache and log directories need write permissions
    chmod u=rwx,g=rx,o= /var/cache/nginx && \
    chmod u=rwx,g=rx,o= /var/log/nginx

USER nginx
```

## Common Permission Issues

### Issue 1: World-Writable Files

**Problem**: Files with permissions like 666 or 777 can be modified by anyone.

```bash
# Bad - world-writable
chmod 666 config.json  # rw-rw-rw-
chmod 777 script.sh    # rwxrwxrwx
```

**Solution**: Remove write permissions for others.

```bash
# Good - restricted access
chmod 640 config.json  # rw-r-----
chmod 750 script.sh    # rwxr-x---
```

### Issue 2: Unnecessary Execute Permissions

**Problem**: Non-executable files (like .py, .js, .json) have execute permission.

```bash
# Bad - unnecessary execute permission
chmod 755 app.py       # rwxr-xr-x
```

**Solution**: Remove execute permission from non-executable files.

```bash
# Good - no execute permission
chmod 644 app.py       # rw-r--r--
```

### Issue 3: Overly Permissive Package Directories

**Problem**: Package directories are writable, allowing modification of dependencies.

```bash
# Bad - packages can be modified
chmod -R 777 /app/packages
```

**Solution**: Make packages read-only after installation.

```bash
# Good - packages are read-only
chmod -R 555 /app/packages  # r-xr-xr-x
```

### Issue 4: Running as Root

**Problem**: Container runs as root user, giving full system access.

```dockerfile
# Bad - runs as root
FROM python:3.10-slim
COPY . /app
CMD ["python", "app.py"]
```

**Solution**: Create and use a non-root user.

```dockerfile
# Good - runs as non-root
FROM python:3.10-slim
COPY . /app
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R u=rwX,g=rX,o= /app
USER appuser
CMD ["python", "app.py"]
```

## Verification

### Manual Verification

Check permissions in a running container:

```bash
# Check file permissions
docker exec backend ls -la /app

# Find world-writable files (should return nothing)
docker exec backend find /app -type f -perm -002

# Find files with excessive permissions
docker exec backend find /app -type f -perm -777

# Check current user
docker exec backend whoami  # Should not be root
```

### Automated Verification

Use the provided script:

```bash
./scripts/verify-file-permissions.sh
```

This script checks:
- Container runs as non-root user
- No world-writable files
- Appropriate permissions on application files
- No unnecessary executable permissions

### Property-Based Testing

Run property tests to verify permissions:

```bash
cd backend
pytest tests/property/test_security_properties.py::TestFilePermissionProperties -v
```

## Security Scanning

### Trivy Scanner

Scan images for permission issues:

```bash
# Scan for misconfigurations including permissions
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --scanners config backend:latest

# Scan for secrets and sensitive files
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --scanners secret backend:latest
```

### Docker Bench Security

Run Docker security best practices check:

```bash
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -v /var/lib:/var/lib \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /etc:/etc \
  --label docker_bench_security \
  docker/docker-bench-security
```

## Best Practices Summary

### DO:
✅ Run containers as non-root users
✅ Use minimal permissions (640 for files, 750 for directories)
✅ Remove execute permissions from non-executable files
✅ Make package directories read-only
✅ Set restrictive permissions before switching to non-root user
✅ Use `chmod -R u=rwX,g=rX,o=` for recursive permission setting
✅ Verify permissions in CI/CD pipeline

### DON'T:
❌ Run containers as root
❌ Use 777 or 666 permissions
❌ Make application files world-writable
❌ Give execute permissions to data files
❌ Allow write access to package directories
❌ Use overly permissive default permissions
❌ Forget to verify permissions after build

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Verify file permissions
  run: |
    docker-compose up -d
    sleep 5
    
    # Check backend permissions
    docker exec backend find /app -type f -perm -002 > world_writable.txt
    if [ -s world_writable.txt ]; then
      echo "Found world-writable files:"
      cat world_writable.txt
      exit 1
    fi
    
    # Check user
    USER=$(docker exec backend whoami)
    if [ "$USER" = "root" ]; then
      echo "Container running as root!"
      exit 1
    fi
    
    echo "✓ File permissions OK"
```

## Troubleshooting

### Permission Denied Errors

If you encounter permission denied errors:

1. **Check current user**: `docker exec container whoami`
2. **Check file ownership**: `docker exec container ls -la /path/to/file`
3. **Check file permissions**: `docker exec container stat -c '%a %n' /path/to/file`

### Application Can't Write Logs

If application can't write to log directory:

```dockerfile
# Ensure log directory has write permissions for app user
RUN mkdir -p /app/logs && \
    chown appuser:appuser /app/logs && \
    chmod u=rwx,g=rx,o= /app/logs
```

### Nginx Can't Access Files

If nginx can't serve files:

```dockerfile
# Ensure nginx user owns and can read files
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    find /usr/share/nginx/html -type f -exec chmod u=r,g=r,o= {} \; && \
    find /usr/share/nginx/html -type d -exec chmod u=rx,g=rx,o= {} \;
```

## References

- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [Linux File Permissions](https://www.linux.com/training-tutorials/understanding-linux-file-permissions/)
