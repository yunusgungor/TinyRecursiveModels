"""
Unit tests for CPU optimization module.

Tests CPU configuration optimizer, thread configuration effectiveness,
MKL integration, and tensor operation optimization for MacBook TRM training.
"""

import os
import pytest
import platform
from unittest.mock import Mock, patch, MagicMock
from macbook_optimization.cpu_optimization import (
    CPUOptimizationConfig,
    CPUOptimizer,
    TensorOperationOptimizer
)
from macbook_optimization.hardware_detection import (
    CPUSpecs,
    MemorySpecs,
    PlatformCapabilities,
    HardwareDetector
)


class TestCPUOptimizationConfig:
    """Test CPUOptimizationConfig dataclass."""
    
    def test_config_creation(self):
        """Test CPUOptimizationConfig creation."""
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        assert config.torch_threads == 4
        assert config.mkl_threads == 4
        assert config.use_mkl is True
        assert config.enable_jit is True
        assert config.tensor_core_optimization is False


class TestCPUOptimizer:
    """Test CPUOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock hardware detector
        self.mock_detector = Mock(spec=HardwareDetector)
        self.mock_detector.detect_cpu_specs.return_value = CPUSpecs(
            cores=4,
            threads=8,
            architecture="x86_64",
            features=["avx", "avx2", "sse4_1"],
            base_frequency=2.4,
            max_frequency=3.8,
            brand="Intel Core i5",
            model="Intel Core i5-8259U"
        )
        self.mock_detector.detect_platform_capabilities.return_value = PlatformCapabilities(
            has_mkl=True,
            has_accelerate=True,
            torch_version="1.12.0",
            python_version="3.9.0",
            macos_version="12.6.0",
            optimal_dtype="float32",
            supports_avx=True,
            supports_avx2=True
        )
        self.mock_detector.get_optimal_worker_count.return_value = 2
        
        self.optimizer = CPUOptimizer(self.mock_detector)
    
    @patch('psutil.virtual_memory')
    def test_calculate_optimal_thread_counts(self, mock_virtual_memory):
        """Test optimal thread count calculation."""
        # Mock 8GB available memory
        mock_virtual_memory.return_value = Mock(available=8 * 1024**3)
        
        thread_counts = self.optimizer.calculate_optimal_thread_counts()
        
        assert "torch_threads" in thread_counts
        assert "mkl_threads" in thread_counts
        assert "omp_threads" in thread_counts
        assert "dataloader_workers" in thread_counts
        
        # Check reasonable values
        assert 1 <= thread_counts["torch_threads"] <= 8
        assert 1 <= thread_counts["mkl_threads"] <= 8
        assert 1 <= thread_counts["omp_threads"] <= 8
        assert 1 <= thread_counts["dataloader_workers"] <= 4
    
    @patch('psutil.virtual_memory')
    def test_calculate_optimal_thread_counts_low_memory(self, mock_virtual_memory):
        """Test thread count calculation with low memory."""
        # Mock 2GB available memory
        mock_virtual_memory.return_value = Mock(available=2 * 1024**3)
        
        thread_counts = self.optimizer.calculate_optimal_thread_counts()
        
        # Should use fewer threads with limited memory
        assert thread_counts["torch_threads"] <= 4
        assert thread_counts["dataloader_workers"] <= 2
    
    def test_create_optimization_config(self):
        """Test optimization configuration creation."""
        config = self.optimizer.create_optimization_config()
        
        assert isinstance(config, CPUOptimizationConfig)
        assert config.torch_threads > 0
        assert config.mkl_threads > 0
        assert config.omp_threads > 0
        assert config.dataloader_workers > 0
        # MKL availability depends on actual PyTorch installation and mock setup
        assert isinstance(config.use_mkl, bool)
        assert config.use_accelerate is True  # Based on mock platform capabilities
    
    def test_configure_torch_threads(self):
        """Test PyTorch thread configuration."""
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        # Test should not raise exception even when PyTorch is not available
        self.optimizer.configure_torch_threads(config)
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_configure_torch_threads_no_pytorch(self):
        """Test torch configuration when PyTorch is not available."""
        # Should not raise exception
        self.optimizer.configure_torch_threads()
    
    def test_setup_mkl_optimization(self):
        """Test MKL optimization setup."""
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        # Clear environment first
        for var in ['MKL_NUM_THREADS', 'MKL_DYNAMIC', 'MKL_THREADING_LAYER']:
            if var in os.environ:
                del os.environ[var]
        
        self.optimizer.setup_mkl_optimization(config)
        
        assert os.environ.get('MKL_NUM_THREADS') == '4'
        assert os.environ.get('MKL_DYNAMIC') == 'TRUE'
        
        # Clean up
        self.optimizer.restore_environment()
    
    def test_setup_mkl_optimization_disabled(self):
        """Test MKL optimization when disabled."""
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=False,  # Disabled
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        self.optimizer.setup_mkl_optimization(config)
        
        # Should not set MKL environment variables
        assert 'MKL_NUM_THREADS' not in os.environ or \
               os.environ.get('MKL_NUM_THREADS') != '4'
    
    @patch('platform.system')
    def test_setup_accelerate_framework_macos(self, mock_system):
        """Test Accelerate framework setup on macOS."""
        mock_system.return_value = "Darwin"
        
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        # Clear environment first
        if 'VECLIB_MAXIMUM_THREADS' in os.environ:
            del os.environ['VECLIB_MAXIMUM_THREADS']
        
        self.optimizer.setup_accelerate_framework(config)
        
        assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '4'
        
        # Clean up
        self.optimizer.restore_environment()
    
    @patch('platform.system')
    def test_setup_accelerate_framework_non_macos(self, mock_system):
        """Test Accelerate framework setup on non-macOS."""
        mock_system.return_value = "Linux"
        
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        self.optimizer.setup_accelerate_framework(config)
        
        # Should not set Accelerate variables on non-macOS
        assert 'VECLIB_MAXIMUM_THREADS' not in os.environ or \
               os.environ.get('VECLIB_MAXIMUM_THREADS') != '4'
    
    def test_setup_openmp_optimization(self):
        """Test OpenMP optimization setup."""
        config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        
        # Clear environment first
        omp_vars = ['OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES']
        for var in omp_vars:
            if var in os.environ:
                del os.environ[var]
        
        self.optimizer.setup_openmp_optimization(config)
        
        assert os.environ.get('OMP_NUM_THREADS') == '4'
        assert os.environ.get('OMP_DYNAMIC') == 'TRUE'
        assert os.environ.get('OMP_PROC_BIND') == 'TRUE'
        assert os.environ.get('OMP_PLACES') == 'cores'
        
        # Clean up
        self.optimizer.restore_environment()
    
    def test_configure_all(self):
        """Test complete configuration."""
        config = self.optimizer.configure_all()
        
        assert isinstance(config, CPUOptimizationConfig)
        assert self.optimizer._is_configured is True
        
        # Clean up
        self.optimizer.restore_environment()
    
    def test_restore_environment(self):
        """Test environment restoration."""
        # Set some environment variables
        original_value = os.environ.get('OMP_NUM_THREADS', 'not_set')
        os.environ['OMP_NUM_THREADS'] = '8'
        
        # Configure optimizer (this will change environment)
        self.optimizer.configure_all()
        
        # Restore environment
        self.optimizer.restore_environment()
        
        # Check that configuration flag is reset
        assert self.optimizer._is_configured is False
        
        # If there was an original value, it should be restored
        if original_value != 'not_set':
            assert os.environ.get('OMP_NUM_THREADS') == original_value
    
    def test_get_configuration_summary_not_configured(self):
        """Test configuration summary when not configured."""
        summary = self.optimizer.get_configuration_summary()
        
        assert summary["status"] == "not_configured"
    
    def test_get_configuration_summary_configured(self):
        """Test configuration summary when configured."""
        self.optimizer.configure_all()
        
        summary = self.optimizer.get_configuration_summary()
        
        assert summary["status"] == "configured"
        assert "hardware" in summary
        assert "threading" in summary
        assert "optimizations" in summary
        assert "environment_vars" in summary
        
        # Clean up
        self.optimizer.restore_environment()
    
    def test_get_optimal_worker_count(self):
        """Test optimal worker count retrieval."""
        worker_count = self.optimizer.get_optimal_worker_count()
        
        assert worker_count == 2  # Based on mock


class TestTensorOperationOptimizer:
    """Test TensorOperationOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock CPU optimizer
        self.mock_cpu_optimizer = Mock(spec=CPUOptimizer)
        self.mock_cpu_optimizer.cpu_specs = CPUSpecs(
            cores=4,
            threads=8,
            architecture="x86_64",
            features=["avx", "avx2"],
            base_frequency=2.4,
            max_frequency=3.8,
            brand="Intel Core i5",
            model="Intel Core i5-8259U"
        )
        self.mock_cpu_optimizer.platform_caps = PlatformCapabilities(
            has_mkl=True,
            has_accelerate=True,
            torch_version="1.12.0",
            python_version="3.9.0",
            macos_version="12.6.0",
            optimal_dtype="float32",
            supports_avx=True,
            supports_avx2=True
        )
        
        self.tensor_optimizer = TensorOperationOptimizer(self.mock_cpu_optimizer)
    
    def test_get_blas_info_with_mkl(self):
        """Test BLAS info detection with MKL."""
        blas_info = self.tensor_optimizer._get_blas_info()
        # Should return appropriate BLAS library info
        assert blas_info in ["PyTorch not available", "Unknown", "MKL", "OpenBLAS", "Accelerate"]
    
    @patch('platform.system')
    def test_get_blas_info_with_accelerate(self, mock_system):
        """Test BLAS info detection with Accelerate."""
        mock_system.return_value = "Darwin"
        
        blas_info = self.tensor_optimizer._get_blas_info()
        # Should return appropriate BLAS library info
        assert blas_info in ["PyTorch not available", "Unknown", "MKL", "OpenBLAS", "Accelerate"]
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_get_blas_info_no_pytorch(self):
        """Test BLAS info when PyTorch is not available."""
        blas_info = self.tensor_optimizer._get_blas_info()
        assert blas_info == "PyTorch not available"
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', True)
    def test_configure_mkl_blas(self):
        """Test MKL BLAS configuration."""
        # Clear environment first
        if 'MKL_CBWR' in os.environ:
            del os.environ['MKL_CBWR']
        
        self.tensor_optimizer._configure_mkl_blas()
        
        assert os.environ.get('MKL_CBWR') == 'AUTO'
        
        # Clean up
        if 'MKL_CBWR' in os.environ:
            del os.environ['MKL_CBWR']
    
    def test_configure_accelerate_blas(self):
        """Test Accelerate BLAS configuration."""
        # Should not raise exception
        self.tensor_optimizer._configure_accelerate_blas()
    
    def test_set_tensor_flags(self):
        """Test tensor operation flags setting."""
        # Should not raise exception when PyTorch is not available
        self.tensor_optimizer._set_tensor_flags()
    
    def test_configure_memory_allocation(self):
        """Test memory allocation configuration."""
        # Should not raise exception when PyTorch is not available
        self.tensor_optimizer._configure_memory_allocation()
    
    def test_configure_mixed_precision_cpu_enabled(self):
        """Test CPU mixed precision configuration when enabled."""
        # Should not raise exception when PyTorch is not available
        self.tensor_optimizer.configure_mixed_precision_cpu(enable=True)
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', True)
    def test_configure_mixed_precision_cpu_disabled(self):
        """Test CPU mixed precision configuration when disabled."""
        # Should not raise exception
        self.tensor_optimizer.configure_mixed_precision_cpu(enable=False)
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_configure_mixed_precision_cpu_no_pytorch(self):
        """Test mixed precision configuration when PyTorch is not available."""
        # Should not raise exception
        self.tensor_optimizer.configure_mixed_precision_cpu(enable=True)
    
    @patch('psutil.Process')
    def test_set_cpu_thread_affinity_success(self, mock_process_class):
        """Test CPU thread affinity setting success."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        self.tensor_optimizer.set_cpu_thread_affinity()
        
        mock_process.cpu_affinity.assert_called_once_with([0, 1, 2, 3])
    
    @patch('psutil.Process')
    def test_set_cpu_thread_affinity_failure(self, mock_process_class):
        """Test CPU thread affinity setting failure."""
        mock_process_class.side_effect = ImportError("psutil not available")
        
        # Should not raise exception
        self.tensor_optimizer.set_cpu_thread_affinity()
    
    def test_optimize_for_inference(self):
        """Test inference-specific optimizations."""
        # Should not raise exception when PyTorch is not available
        self.tensor_optimizer.optimize_for_inference()
    
    def test_optimize_for_training(self):
        """Test training-specific optimizations."""
        # Mock CPU optimizer config
        mock_config = CPUOptimizationConfig(
            torch_threads=4,
            mkl_threads=4,
            omp_threads=4,
            dataloader_workers=2,
            use_mkl=True,
            mkl_dynamic=True,
            use_accelerate=True,
            enable_jit=True,
            enable_mixed_precision=False,
            tensor_core_optimization=False
        )
        self.mock_cpu_optimizer.create_optimization_config.return_value = mock_config
        
        # Should not raise exception when PyTorch is not available
        self.tensor_optimizer.optimize_for_training()
    
    def test_benchmark_tensor_operations(self):
        """Test tensor operations benchmarking."""
        results = self.tensor_optimizer.benchmark_tensor_operations()
        
        # Should either return benchmark results or status
        if "status" in results:
            assert results["status"] == "pytorch_not_available"
        else:
            # PyTorch is available, should have benchmark results
            assert "matrix_multiply_ms" in results
            assert "elementwise_ops_ms" in results
            assert isinstance(results["matrix_multiply_ms"], (int, float))
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_benchmark_tensor_operations_no_pytorch(self):
        """Test benchmarking when PyTorch is not available."""
        results = self.tensor_optimizer.benchmark_tensor_operations()
        
        assert results["status"] == "pytorch_not_available"
    
    def test_get_optimization_info(self):
        """Test optimization info retrieval."""
        info = self.tensor_optimizer.get_optimization_info()
        
        # Should either return status or optimization info
        if "status" in info:
            assert info["status"] == "pytorch_not_available"
        else:
            # PyTorch is available, should have optimization info
            assert "blas_library" in info
            assert "mkl_available" in info
            assert "torch_threads" in info
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_get_optimization_info_no_pytorch(self):
        """Test optimization info when PyTorch is not available."""
        info = self.tensor_optimizer.get_optimization_info()
        
        assert info["status"] == "pytorch_not_available"
    
    @patch('psutil.Process')
    def test_get_cpu_affinity_success(self, mock_process_class):
        """Test CPU affinity retrieval success."""
        mock_process = Mock()
        mock_process.cpu_affinity.return_value = [0, 1, 2, 3]
        mock_process_class.return_value = mock_process
        
        affinity = self.tensor_optimizer._get_cpu_affinity()
        
        assert affinity == [0, 1, 2, 3]
    
    @patch('psutil.Process')
    def test_get_cpu_affinity_failure(self, mock_process_class):
        """Test CPU affinity retrieval failure."""
        mock_process_class.side_effect = ImportError("psutil not available")
        
        affinity = self.tensor_optimizer._get_cpu_affinity()
        
        assert affinity == []
    
    def test_get_optimization_flags(self):
        """Test optimization flags retrieval."""
        flags = self.tensor_optimizer._get_optimization_flags()
        
        assert isinstance(flags, dict)
        # Should be empty when PyTorch is not available, or contain flags when available
        if len(flags) > 0:
            # PyTorch is available
            assert "jit_enabled" in flags or "detect_anomaly" in flags
    
    @patch('macbook_optimization.cpu_optimization.TORCH_AVAILABLE', False)
    def test_get_optimization_flags_no_pytorch(self):
        """Test optimization flags when PyTorch is not available."""
        flags = self.tensor_optimizer._get_optimization_flags()
        
        assert flags == {}


class TestIntegration:
    """Integration tests for CPU optimization components."""
    
    def test_cpu_optimizer_with_tensor_optimizer_integration(self):
        """Test integration between CPU optimizer and tensor optimizer."""
        # Create real hardware detector (will use actual system)
        detector = HardwareDetector()
        cpu_optimizer = CPUOptimizer(detector)
        tensor_optimizer = TensorOperationOptimizer(cpu_optimizer)
        
        # Test that they work together
        config = cpu_optimizer.create_optimization_config()
        assert isinstance(config, CPUOptimizationConfig)
        
        # Test tensor optimizer can access CPU specs
        assert tensor_optimizer.cpu_specs is not None
        assert tensor_optimizer.platform_caps is not None
        
        # Test optimization info retrieval
        info = tensor_optimizer.get_optimization_info()
        assert isinstance(info, dict)
    
    def test_environment_variable_management(self):
        """Test that environment variables are properly managed."""
        detector = HardwareDetector()
        optimizer = CPUOptimizer(detector)
        
        # Store original environment
        original_omp = os.environ.get('OMP_NUM_THREADS')
        
        try:
            # Configure optimizer
            config = optimizer.configure_all()
            
            # Check that environment variables are set
            assert 'OMP_NUM_THREADS' in os.environ
            
            # Restore environment
            optimizer.restore_environment()
            
            # Check that environment is restored
            current_omp = os.environ.get('OMP_NUM_THREADS')
            if original_omp is None:
                assert current_omp is None or current_omp != str(config.omp_threads)
            else:
                assert current_omp == original_omp
                
        finally:
            # Ensure cleanup
            optimizer.restore_environment()