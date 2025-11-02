"""
Unit tests for hardware detection module.

Tests hardware detection accuracy, memory and CPU specification parsing,
and platform capability detection for MacBook optimization.
"""

import pytest
import platform
import sys
from unittest.mock import Mock, patch, MagicMock
from macbook_optimization.hardware_detection import (
    CPUSpecs,
    MemorySpecs,
    PlatformCapabilities,
    HardwareDetector
)


class TestCPUSpecs:
    """Test CPUSpecs dataclass."""
    
    def test_cpu_specs_creation(self):
        """Test CPUSpecs dataclass creation."""
        specs = CPUSpecs(
            cores=4,
            threads=8,
            architecture="x86_64",
            features=["avx", "avx2", "sse4_1"],
            base_frequency=2.4,
            max_frequency=3.8,
            brand="Intel Core i5",
            model="Intel Core i5-8259U"
        )
        
        assert specs.cores == 4
        assert specs.threads == 8
        assert specs.architecture == "x86_64"
        assert "avx" in specs.features
        assert specs.base_frequency == 2.4
        assert specs.max_frequency == 3.8


class TestMemorySpecs:
    """Test MemorySpecs dataclass."""
    
    def test_memory_specs_creation(self):
        """Test MemorySpecs dataclass creation."""
        specs = MemorySpecs(
            total_memory=8589934592,  # 8GB in bytes
            available_memory=4294967296,  # 4GB in bytes
            memory_type="LPDDR3",
            memory_speed=2133
        )
        
        assert specs.total_memory == 8589934592
        assert specs.available_memory == 4294967296
        assert specs.memory_type == "LPDDR3"
        assert specs.memory_speed == 2133


class TestPlatformCapabilities:
    """Test PlatformCapabilities dataclass."""
    
    def test_platform_capabilities_creation(self):
        """Test PlatformCapabilities dataclass creation."""
        caps = PlatformCapabilities(
            has_mkl=True,
            has_accelerate=True,
            torch_version="1.12.0",
            python_version="3.9.0",
            macos_version="12.6.0",
            optimal_dtype="float32",
            supports_avx=True,
            supports_avx2=True
        )
        
        assert caps.has_mkl is True
        assert caps.has_accelerate is True
        assert caps.torch_version == "1.12.0"
        assert caps.supports_avx is True


class TestHardwareDetector:
    """Test HardwareDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HardwareDetector()
    
    @patch('psutil.cpu_count')
    @patch('cpuinfo.get_cpu_info')
    @patch('psutil.cpu_freq')
    @patch('platform.machine')
    def test_detect_cpu_specs(self, mock_machine, mock_cpu_freq, mock_cpu_info, mock_cpu_count):
        """Test CPU specification detection."""
        # Mock return values
        mock_cpu_count.side_effect = [4, 8]  # cores, threads
        mock_machine.return_value = "x86_64"
        mock_cpu_info.return_value = {
            'flags': ['avx', 'avx2', 'sse4_1', 'sse4_2', 'fma'],
            'hz_advertised_friendly': '2.4000 GHz',
            'brand_raw': 'Intel Core i5-8259U'
        }
        mock_cpu_freq.return_value = Mock(current=2400.0, max=3800.0)
        
        specs = self.detector.detect_cpu_specs()
        
        assert specs.cores == 4
        assert specs.threads == 8
        assert specs.architecture == "x86_64"
        assert "avx" in specs.features
        assert "avx2" in specs.features
        assert specs.base_frequency == 2.4
        assert specs.max_frequency == 3.8
        assert "Intel" in specs.brand
    
    @patch('psutil.virtual_memory')
    @patch('subprocess.run')
    @patch('platform.system')
    def test_detect_memory_specs_macos(self, mock_system, mock_subprocess, mock_virtual_memory):
        """Test memory specification detection on macOS."""
        # Mock return values
        mock_system.return_value = "Darwin"
        mock_virtual_memory.return_value = Mock(
            total=8589934592,  # 8GB
            available=4294967296,  # 4GB
            used=4294967296,
            percent=50.0
        )
        
        # Mock system_profiler output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        Memory:
            Type: LPDDR3
            Speed: 2133 MHz
        """
        mock_subprocess.return_value = mock_result
        
        specs = self.detector.detect_memory_specs()
        
        assert specs.total_memory == 8589934592
        assert specs.available_memory == 4294967296
        assert specs.memory_type == "LPDDR3"
        assert specs.memory_speed == 2133
    
    @patch('psutil.virtual_memory')
    @patch('platform.system')
    def test_detect_memory_specs_non_macos(self, mock_system, mock_virtual_memory):
        """Test memory specification detection on non-macOS systems."""
        # Mock return values
        mock_system.return_value = "Linux"
        mock_virtual_memory.return_value = Mock(
            total=8589934592,
            available=4294967296,
            used=4294967296,
            percent=50.0
        )
        
        specs = self.detector.detect_memory_specs()
        
        assert specs.total_memory == 8589934592
        assert specs.available_memory == 4294967296
        assert specs.memory_type == "Unknown"
        assert specs.memory_speed is None
    
    @patch('platform.system')
    @patch('platform.mac_ver')
    def test_detect_platform_capabilities_macos_with_mkl(self, mock_mac_ver, mock_system):
        """Test platform capabilities detection on macOS with MKL."""
        # Mock return values
        mock_system.return_value = "Darwin"
        mock_mac_ver.return_value = ("12.6.0", "", "")
        
        # Mock torch import and MKL availability
        mock_torch = Mock()
        mock_torch.__version__ = "1.12.0"
        mock_torch.backends.mkl.is_available.return_value = True
        
        # Mock CPU specs for AVX detection
        with patch.object(self.detector, 'detect_cpu_specs') as mock_cpu_specs, \
             patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('sys.version_info', Mock(major=3, minor=9, micro=0)):
            mock_cpu_specs.return_value = CPUSpecs(
                cores=4, threads=8, architecture="x86_64",
                features=["avx", "avx2"], base_frequency=2.4,
                max_frequency=3.8, brand="Intel", model="Intel"
            )
            
            caps = self.detector.detect_platform_capabilities()
            
            assert caps.has_mkl is True
            assert caps.has_accelerate is True
            assert caps.torch_version == "1.12.0"
            assert caps.python_version == "3.9.0"
            assert caps.macos_version == "12.6.0"
            assert caps.optimal_dtype == "float32"
            assert caps.supports_avx is True
            assert caps.supports_avx2 is True
    
    @patch('platform.system')
    def test_detect_platform_capabilities_non_macos(self, mock_system):
        """Test platform capabilities detection on non-macOS systems."""
        mock_system.return_value = "Linux"
        
        # Mock torch import
        mock_torch = Mock()
        mock_torch.__version__ = "1.12.0"
        mock_torch.backends.mkl.is_available.return_value = False
        
        # Mock CPU specs
        with patch.object(self.detector, 'detect_cpu_specs') as mock_cpu_specs, \
             patch.dict('sys.modules', {'torch': mock_torch}), \
             patch('sys.version_info', Mock(major=3, minor=9, micro=0)):
            mock_cpu_specs.return_value = CPUSpecs(
                cores=4, threads=8, architecture="x86_64",
                features=[], base_frequency=2.4,
                max_frequency=3.8, brand="Intel", model="Intel"
            )
            
            caps = self.detector.detect_platform_capabilities()
            
            assert caps.has_accelerate is False
            assert caps.macos_version == "N/A"
            assert caps.supports_avx is False
            assert caps.supports_avx2 is False
    
    def test_get_optimal_worker_count(self):
        """Test optimal worker count calculation."""
        # Mock hardware specs
        with patch.object(self.detector, 'detect_cpu_specs') as mock_cpu_specs, \
             patch.object(self.detector, 'detect_memory_specs') as mock_memory_specs:
            
            mock_cpu_specs.return_value = CPUSpecs(
                cores=4, threads=8, architecture="x86_64",
                features=[], base_frequency=2.4,
                max_frequency=3.8, brand="Intel", model="Intel"
            )
            mock_memory_specs.return_value = MemorySpecs(
                total_memory=8589934592,  # 8GB
                available_memory=6442450944,  # 6GB available
                memory_type="LPDDR3",
                memory_speed=2133
            )
            
            worker_count = self.detector.get_optimal_worker_count()
            
            # Should be between 1 and 4 (min of cores, memory-based limit, and max 4)
            assert 1 <= worker_count <= 4
    
    def test_get_hardware_summary(self):
        """Test hardware summary generation."""
        # Mock all detection methods
        with patch.object(self.detector, 'detect_cpu_specs') as mock_cpu_specs, \
             patch.object(self.detector, 'detect_memory_specs') as mock_memory_specs, \
             patch.object(self.detector, 'detect_platform_capabilities') as mock_platform_caps, \
             patch.object(self.detector, 'get_optimal_worker_count') as mock_workers:
            
            mock_cpu_specs.return_value = CPUSpecs(
                cores=4, threads=8, architecture="x86_64",
                features=["avx", "avx2"], base_frequency=2.4,
                max_frequency=3.8, brand="Intel Core i5", model="Intel Core i5"
            )
            mock_memory_specs.return_value = MemorySpecs(
                total_memory=8589934592,
                available_memory=6442450944,
                memory_type="LPDDR3",
                memory_speed=2133
            )
            mock_platform_caps.return_value = PlatformCapabilities(
                has_mkl=True, has_accelerate=True, torch_version="1.12.0",
                python_version="3.9.0", macos_version="12.6.0",
                optimal_dtype="float32", supports_avx=True, supports_avx2=True
            )
            mock_workers.return_value = 4
            
            summary = self.detector.get_hardware_summary()
            
            # Check structure and key values
            assert "cpu" in summary
            assert "memory" in summary
            assert "platform" in summary
            assert "optimization" in summary
            
            assert summary["cpu"]["cores"] == 4
            assert summary["cpu"]["threads"] == 8
            assert summary["memory"]["total_gb"] == 8.0
            assert summary["platform"]["has_mkl"] is True
            assert summary["optimization"]["optimal_workers"] == 4
    
    def test_cpu_specs_with_missing_frequency_info(self):
        """Test CPU specs detection when frequency information is missing."""
        with patch('psutil.cpu_count') as mock_cpu_count, \
             patch('cpuinfo.get_cpu_info') as mock_cpu_info, \
             patch('psutil.cpu_freq') as mock_cpu_freq, \
             patch('platform.machine') as mock_machine:
            
            mock_cpu_count.side_effect = [4, 8]
            mock_machine.return_value = "x86_64"
            mock_cpu_info.return_value = {
                'flags': ['avx'],
                'brand_raw': 'Intel Core i5'
            }
            mock_cpu_freq.return_value = None  # No frequency info
            
            specs = self.detector.detect_cpu_specs()
            
            # Should use fallback values
            assert specs.base_frequency == 2.4
            assert specs.max_frequency == 3.8
    
    def test_memory_specs_with_subprocess_error(self):
        """Test memory specs detection when subprocess fails."""
        with patch('psutil.virtual_memory') as mock_virtual_memory, \
             patch('subprocess.run') as mock_subprocess, \
             patch('platform.system') as mock_system:
            
            mock_system.return_value = "Darwin"
            mock_virtual_memory.return_value = Mock(
                total=8589934592,
                available=4294967296,
                used=4294967296,
                percent=50.0
            )
            mock_subprocess.side_effect = Exception("Command failed")
            
            specs = self.detector.detect_memory_specs()
            
            # Should handle error gracefully
            assert specs.memory_type == "Unknown"
            assert specs.memory_speed is None
    
    def test_platform_capabilities_without_torch(self):
        """Test platform capabilities detection when PyTorch is not available."""
        with patch('platform.system') as mock_system, \
             patch('platform.mac_ver') as mock_mac_ver, \
             patch('sys.version_info', Mock(major=3, minor=9, micro=0)), \
             patch.object(self.detector, 'detect_cpu_specs') as mock_cpu_specs:
            
            mock_system.return_value = "Darwin"
            mock_mac_ver.return_value = ("12.6.0", "", "")
            mock_cpu_specs.return_value = CPUSpecs(
                cores=4, threads=8, architecture="x86_64",
                features=[], base_frequency=2.4,
                max_frequency=3.8, brand="Intel", model="Intel"
            )
            
            # Mock ImportError for torch
            def mock_import(name, *args, **kwargs):
                if name == 'torch':
                    raise ImportError("No module named 'torch'")
                # For other imports, use the real import
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                caps = self.detector.detect_platform_capabilities()
                
                assert caps.has_mkl is False
                assert caps.torch_version == "Not installed"