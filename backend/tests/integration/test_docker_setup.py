"""
Integration tests for Docker setup
Tests container builds, Docker Compose, and health checks
"""

import subprocess
import time
import requests
import pytest
import os
from pathlib import Path

# Get project root directory (parent of backend)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class TestDockerSetup:
    """Test Docker configuration and setup"""

    @pytest.fixture(scope="class")
    def docker_available(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @pytest.fixture(scope="class")
    def docker_compose_available(self):
        """Check if Docker Compose is available"""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def test_docker_installed(self, docker_available):
        """Test that Docker is installed and running"""
        assert docker_available, "Docker is not installed or not running"

    def test_docker_compose_installed(self, docker_compose_available):
        """Test that Docker Compose is installed"""
        assert docker_compose_available, "Docker Compose is not installed"

    def test_backend_dockerfile_exists(self):
        """Test that backend Dockerfile exists"""
        dockerfile_path = PROJECT_ROOT / "backend" / "Dockerfile"
        assert dockerfile_path.exists(), "Backend Dockerfile not found"

    def test_frontend_dockerfile_exists(self):
        """Test that frontend Dockerfile exists"""
        dockerfile_path = PROJECT_ROOT / "frontend" / "Dockerfile"
        assert dockerfile_path.exists(), "Frontend Dockerfile not found"

    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"

    def test_docker_compose_prod_file_exists(self):
        """Test that docker-compose.prod.yml exists"""
        compose_file = PROJECT_ROOT / "docker-compose.prod.yml"
        assert compose_file.exists(), "docker-compose.prod.yml not found"

    def test_backend_dockerfile_has_stages(self):
        """Test that backend Dockerfile has multi-stage build"""
        dockerfile_path = PROJECT_ROOT / "backend" / "Dockerfile"
        with open(dockerfile_path, "r") as f:
            content = f.read()
            assert "FROM" in content, "Dockerfile missing FROM instruction"
            assert "as development" in content or "AS development" in content, \
                "Dockerfile missing development stage"
            assert "as production" in content or "AS production" in content, \
                "Dockerfile missing production stage"

    def test_frontend_dockerfile_has_stages(self):
        """Test that frontend Dockerfile has multi-stage build"""
        dockerfile_path = PROJECT_ROOT / "frontend" / "Dockerfile"
        with open(dockerfile_path, "r") as f:
            content = f.read()
            assert "FROM" in content, "Dockerfile missing FROM instruction"
            assert "as development" in content or "AS development" in content, \
                "Dockerfile missing development stage"
            assert "as production" in content or "AS production" in content, \
                "Dockerfile missing production stage"

    def test_docker_compose_has_required_services(self):
        """Test that docker-compose.yml has all required services"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_file, "r") as f:
            content = f.read()
            required_services = ["postgres", "redis", "backend", "frontend"]
            for service in required_services:
                assert service in content, f"Service {service} not found in docker-compose.yml"

    def test_docker_compose_has_health_checks(self):
        """Test that docker-compose.yml has health checks for databases"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_file, "r") as f:
            content = f.read()
            assert "healthcheck" in content, "No health checks found in docker-compose.yml"

    def test_docker_compose_has_networks(self):
        """Test that docker-compose.yml defines networks"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_file, "r") as f:
            content = f.read()
            assert "networks:" in content, "No networks defined in docker-compose.yml"

    def test_docker_compose_has_volumes(self):
        """Test that docker-compose.yml defines volumes"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_file, "r") as f:
            content = f.read()
            assert "volumes:" in content, "No volumes defined in docker-compose.yml"

    def test_nginx_config_exists(self):
        """Test that nginx configuration exists"""
        nginx_conf = PROJECT_ROOT / "nginx" / "nginx.conf"
        nginx_prod_conf = PROJECT_ROOT / "nginx" / "nginx.prod.conf"
        assert nginx_conf.exists(), "nginx.conf not found"
        assert nginx_prod_conf.exists(), "nginx.prod.conf not found"

    def test_kubernetes_manifests_exist(self):
        """Test that Kubernetes manifests exist"""
        k8s_files = [
            "k8s/namespace.yaml",
            "k8s/configmap.yaml",
            "k8s/secrets.yaml",
            "k8s/postgres-deployment.yaml",
            "k8s/redis-deployment.yaml",
            "k8s/backend-deployment.yaml",
            "k8s/frontend-deployment.yaml",
            "k8s/ingress.yaml",
            "k8s/hpa.yaml"
        ]
        for k8s_file in k8s_files:
            file_path = PROJECT_ROOT / k8s_file
            assert file_path.exists(), f"Kubernetes manifest {k8s_file} not found"

    def test_environment_files_exist(self):
        """Test that environment configuration files exist"""
        env_files = [
            ".env.development",
            ".env.staging",
            ".env.production"
        ]
        for env_file in env_files:
            file_path = PROJECT_ROOT / env_file
            assert file_path.exists(), f"Environment file {env_file} not found"

    def test_deployment_scripts_exist(self):
        """Test that deployment scripts exist"""
        scripts = [
            "scripts/deploy-dev.sh",
            "scripts/deploy-prod.sh",
            "scripts/deploy-k8s.sh"
        ]
        for script in scripts:
            script_path = PROJECT_ROOT / script
            assert script_path.exists(), f"Deployment script {script} not found"

    def test_deployment_scripts_executable(self):
        """Test that deployment scripts are executable"""
        scripts = [
            "scripts/deploy-dev.sh",
            "scripts/deploy-prod.sh",
            "scripts/deploy-k8s.sh"
        ]
        for script in scripts:
            script_path = PROJECT_ROOT / script
            assert os.access(script_path, os.X_OK), f"Deployment script {script} is not executable"

    def test_deployment_documentation_exists(self):
        """Test that deployment documentation exists"""
        deployment_md = PROJECT_ROOT / "DEPLOYMENT.md"
        k8s_readme = PROJECT_ROOT / "k8s" / "README.md"
        assert deployment_md.exists(), "DEPLOYMENT.md not found"
        assert k8s_readme.exists(), "k8s/README.md not found"


@pytest.mark.skipif(
    not os.getenv("RUN_DOCKER_TESTS"),
    reason="Docker tests require RUN_DOCKER_TESTS=1 environment variable"
)
class TestDockerBuild:
    """Test Docker image builds (requires Docker)"""

    def test_backend_image_builds(self):
        """Test that backend Docker image builds successfully"""
        result = subprocess.run(
            [
                "docker", "build",
                "-t", "trendyol-gift-backend-test:latest",
                "--target", "development",
                "-f", "backend/Dockerfile",
                "backend/"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        assert result.returncode == 0, f"Backend image build failed: {result.stderr}"

    def test_frontend_image_builds(self):
        """Test that frontend Docker image builds successfully"""
        result = subprocess.run(
            [
                "docker", "build",
                "-t", "trendyol-gift-frontend-test:latest",
                "--target", "development",
                "-f", "frontend/Dockerfile",
                "frontend/"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        assert result.returncode == 0, f"Frontend image build failed: {result.stderr}"


@pytest.mark.skipif(
    not os.getenv("RUN_DOCKER_COMPOSE_TESTS"),
    reason="Docker Compose tests require RUN_DOCKER_COMPOSE_TESTS=1 environment variable"
)
class TestDockerCompose:
    """Test Docker Compose setup (requires Docker Compose)"""

    @pytest.fixture(scope="class", autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for Docker Compose tests"""
        # Setup: Stop any running containers
        subprocess.run(
            ["docker-compose", "down"],
            capture_output=True,
            timeout=60
        )
        
        yield
        
        # Teardown: Stop containers
        subprocess.run(
            ["docker-compose", "down"],
            capture_output=True,
            timeout=60
        )

    def test_docker_compose_config_valid(self):
        """Test that docker-compose.yml is valid"""
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, f"docker-compose.yml is invalid: {result.stderr}"

    def test_docker_compose_up(self):
        """Test that docker-compose up starts all services"""
        # Start services
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            timeout=300
        )
        assert result.returncode == 0, f"docker-compose up failed: {result.stderr}"

        # Wait for services to start
        time.sleep(30)

        # Check services are running
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, "docker-compose ps failed"
        assert "postgres" in result.stdout, "PostgreSQL service not running"
        assert "redis" in result.stdout, "Redis service not running"

    def test_postgres_health_check(self):
        """Test PostgreSQL health check"""
        max_retries = 30
        for i in range(max_retries):
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "postgres", "pg_isready", "-U", "postgres"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                break
            time.sleep(2)
        else:
            pytest.fail("PostgreSQL health check failed after 30 retries")

    def test_redis_health_check(self):
        """Test Redis health check"""
        max_retries = 30
        for i in range(max_retries):
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "redis", "redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and "PONG" in result.stdout:
                break
            time.sleep(2)
        else:
            pytest.fail("Redis health check failed after 30 retries")

    def test_backend_health_check(self):
        """Test backend health check endpoint"""
        max_retries = 60
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/api/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        else:
            pytest.fail("Backend health check failed after 60 retries")

    def test_frontend_accessible(self):
        """Test that frontend is accessible"""
        max_retries = 60
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:3000", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        else:
            pytest.fail("Frontend not accessible after 60 retries")
