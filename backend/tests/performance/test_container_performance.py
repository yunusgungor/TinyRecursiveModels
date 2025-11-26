"""
Performance tests for container infrastructure
Tests build times, image sizes, hot reload, and cache efficiency
Requirements: 1.1, 2.1, 2.4, 3.1, 3.2, 4.2, 9.1
"""

import pytest
import subprocess
import time
import os
from pathlib import Path
from typing import Tuple, Optional


# Performance targets from design document
COLD_BUILD_TARGET = 600  # 10 minutes
WARM_BUILD_TARGET = 120  # 2 minutes
INCREMENTAL_BUILD_TARGET = 30  # 30 seconds
BACKEND_IMAGE_SIZE_TARGET = 200 * 1024 * 1024  # 200MB
FRONTEND_IMAGE_SIZE_TARGET = 50 * 1024 * 1024  # 50MB
HOT_RELOAD_TARGET = 2  # 2 seconds
CACHE_HIT_RATE_TARGET = 0.95  # 95%


class TestBuildPerformance:
    """Test suite for build time performance (Task 14.1)"""
    
    def run_docker_build(self, context: str, dockerfile: str, target: str, 
                        use_cache: bool = True, tag: str = None) -> Tuple[float, int]:
        """Helper to run docker build and measure time"""
        cmd = [
            "docker", "build",
            "-f", dockerfile,
            "--target", target,
            context
        ]
        
        if not use_cache:
            cmd.insert(2, "--no-cache")
        
        if tag:
            cmd.extend(["-t", tag])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        return end_time - start_time, result.returncode
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_backend_cold_build_time(self):
        """
        Test cold build time for backend
        Requirement 1.1: Container System should start all services within 5 minutes
        Requirement 2.1: Changes should be applied within 2 seconds with hot reload
        """
        build_time, returncode = self.run_docker_build(
            context="./backend",
            dockerfile="./backend/Dockerfile",
            target="production",
            use_cache=False,
            tag="backend:cold-build-test"
        )
        
        assert returncode == 0, "Cold build failed"
        assert build_time < COLD_BUILD_TARGET, (
            f"Cold build time {build_time:.2f}s exceeds target {COLD_BUILD_TARGET}s"
        )
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_backend_warm_build_time(self):
        """
        Test warm build time for backend (with cache)
        Requirement 2.1: Changes should be applied within 2 seconds
        """
        # First build to populate cache
        self.run_docker_build(
            context="./backend",
            dockerfile="./backend/Dockerfile",
            target="production",
            use_cache=True,
            tag="backend:warm-build-test-1"
        )
        
        # Second build should use cache
        build_time, returncode = self.run_docker_build(
            context="./backend",
            dockerfile="./backend/Dockerfile",
            target="production",
            use_cache=True,
            tag="backend:warm-build-test-2"
        )
        
        assert returncode == 0, "Warm build failed"
        assert build_time < WARM_BUILD_TARGET, (
            f"Warm build time {build_time:.2f}s exceeds target {WARM_BUILD_TARGET}s"
        )
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_backend_incremental_build_time(self):
        """
        Test incremental build time (code change only)
        Requirement 2.4: Only dependency layer should rebuild when dependencies change
        """
        # Initial build
        self.run_docker_build(
            context="./backend",
            dockerfile="./backend/Dockerfile",
            target="production",
            use_cache=True,
            tag="backend:incremental-test-1"
        )
        
        # Make a small code change
        test_file = Path("./backend/app/__init__.py")
        original_content = test_file.read_text() if test_file.exists() else ""
        
        try:
            # Add a comment
            with open(test_file, "a") as f:
                f.write(f"\n# Test comment {time.time()}\n")
            
            # Rebuild with code change
            build_time, returncode = self.run_docker_build(
                context="./backend",
                dockerfile="./backend/Dockerfile",
                target="production",
                use_cache=True,
                tag="backend:incremental-test-2"
            )
            
            assert returncode == 0, "Incremental build failed"
            assert build_time < INCREMENTAL_BUILD_TARGET, (
                f"Incremental build time {build_time:.2f}s exceeds target {INCREMENTAL_BUILD_TARGET}s"
            )
        finally:
            # Restore original file
            if original_content:
                test_file.write_text(original_content)
            elif test_file.exists():
                test_file.unlink()


class TestImageSizes:
    """Test suite for image size validation (Task 14.2)"""
    
    def get_image_size(self, image_name: str) -> Optional[int]:
        """Get Docker image size in bytes"""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_name, "--format={{.Size}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
        except subprocess.CalledProcessError:
            return None
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_backend_image_size(self):
        """
        Test backend production image size
        Requirement 3.1: Backend image should be under 200MB
        """
        # Build production image
        result = subprocess.run(
            [
                "docker", "build",
                "-f", "./backend/Dockerfile",
                "--target", "production",
                "-t", "backend:size-test",
                "./backend"
            ],
            capture_output=True
        )
        
        assert result.returncode == 0, "Backend build failed"
        
        size = self.get_image_size("backend:size-test")
        assert size is not None, "Failed to get backend image size"
        
        size_mb = size / (1024 * 1024)
        target_mb = BACKEND_IMAGE_SIZE_TARGET / (1024 * 1024)
        
        assert size < BACKEND_IMAGE_SIZE_TARGET, (
            f"Backend image size {size_mb:.2f}MB exceeds target {target_mb:.0f}MB"
        )
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_frontend_image_size(self):
        """
        Test frontend production image size
        Requirement 3.2: Frontend image should be under 50MB
        """
        # Build production image
        result = subprocess.run(
            [
                "docker", "build",
                "-f", "./frontend/Dockerfile",
                "--target", "production",
                "-t", "frontend:size-test",
                "./frontend"
            ],
            capture_output=True
        )
        
        assert result.returncode == 0, "Frontend build failed"
        
        size = self.get_image_size("frontend:size-test")
        assert size is not None, "Failed to get frontend image size"
        
        size_mb = size / (1024 * 1024)
        target_mb = FRONTEND_IMAGE_SIZE_TARGET / (1024 * 1024)
        
        assert size < FRONTEND_IMAGE_SIZE_TARGET, (
            f"Frontend image size {size_mb:.2f}MB exceeds target {target_mb:.0f}MB"
        )


class TestHotReloadPerformance:
    """Test suite for hot reload performance (Task 14.3)"""
    
    @pytest.mark.skip(reason="Requires running docker-compose environment")
    def test_backend_hot_reload_latency(self):
        """
        Test hot reload latency for backend
        Requirement 2.1: Changes should be applied within 2 seconds
        Requirement 2.2: Frontend changes should reflect immediately
        
        Note: This test requires manual validation with docker-compose running
        """
        # This is a placeholder for manual testing
        # To test manually:
        # 1. Run: docker-compose up -d
        # 2. Make a code change in backend/app/main.py
        # 3. Observe reload time in logs
        # 4. Verify reload completes in < 2 seconds
        pass


class TestCacheEfficiency:
    """Test suite for cache efficiency (Task 14.4)"""
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS") == "1",
        reason="Slow test skipped"
    )
    def test_dependency_cache_hit_rate(self):
        """
        Test cache hit rate for unchanged dependencies
        Requirement 4.2: Dependency layer should be cached when unchanged
        Requirement 9.1: Dependencies should be cached when unchanged
        """
        # Build 1: Initial build
        result1 = subprocess.run(
            [
                "docker", "build",
                "-f", "./backend/Dockerfile",
                "--target", "production",
                "-t", "backend:cache-test-1",
                "./backend"
            ],
            capture_output=True,
            text=True
        )
        
        assert result1.returncode == 0, "Initial build failed"
        
        # Build 2: Rebuild without changes
        result2 = subprocess.run(
            [
                "docker", "build",
                "-f", "./backend/Dockerfile",
                "--target", "production",
                "-t", "backend:cache-test-2",
                "./backend"
            ],
            capture_output=True,
            text=True
        )
        
        assert result2.returncode == 0, "Cached build failed"
        
        # Analyze cache usage
        output = result2.stdout + result2.stderr
        cache_lines = [line for line in output.split('\n') 
                      if 'CACHED' in line or 'cache' in line.lower()]
        total_steps = len([line for line in output.split('\n') 
                          if line.strip().startswith('Step') or '#' in line])
        
        if total_steps > 0:
            cached_steps = len(cache_lines)
            cache_hit_rate = cached_steps / total_steps
            
            assert cache_hit_rate >= CACHE_HIT_RATE_TARGET, (
                f"Cache hit rate {cache_hit_rate:.2%} is below target {CACHE_HIT_RATE_TARGET:.2%}"
            )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
