"""
CPU optimization module for MacBook TRM training.

This module provides CPU-specific optimizations for training Tiny Recursive Models
on Intel-based MacBook hardware, including thread configuration, MKL integration,
and tensor operation optimization.
"""

import os
import platform
import psutil
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError, RuntimeError, Exception) as e:
    TORCH_AVAILABLE = False
    warnings.warn(f"PyTorch not available ({type(e).__name__}: {str(e)[:100]}...). CPU optimization will be limited.")

from .hardware_detection import HardwareDetector, CPUSpecs, PlatformCapabilities


logger = logging.getLogger(__name__)


@dataclass
class CPUOptimizationConfig:
    """Configuration for CPU optimization settings."""
    # Thread configuration
    torch_threads: int
    mkl_threads: int
    omp_threads: int
    dataloader_workers: int
    
    # MKL settings
    use_mkl: bool
    mkl_dynamic: bool
    
    # Accelerate framework (macOS)
    use_accelerate: bool
    
    # Performance settings
    enable_jit: bool
    enable_mixed_precision: bool
    tensor_core_optimization: bool


class CPUOptimizer:
    """CPU configuration optimizer for MacBook TRM training."""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        """Initialize CPU optimizer with hardware detection."""
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.cpu_specs = self.hardware_detector.detect_cpu_specs()
        self.platform_caps = self.hardware_detector.detect_platform_capabilities()
        self._original_env = {}
        self._is_configured = False
        
    def calculate_optimal_thread_counts(self) -> Dict[str, int]:
        """Calculate optimal thread counts for different operations."""
        # Base thread count on physical cores
        physical_cores = self.cpu_specs.cores
        logical_cores = self.cpu_specs.threads
        
        # Conservative threading for memory-constrained systems
        memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # PyTorch threads: use physical cores, but limit based on memory
        torch_threads = min(physical_cores, max(1, int(memory_gb * 2)))
        
        # MKL threads: slightly more aggressive, can use hyperthreading
        mkl_threads = min(logical_cores, max(1, int(memory_gb * 2.5)))
        
        # OpenMP threads: conservative for stability
        omp_threads = min(physical_cores, max(1, int(memory_gb * 1.5)))
        
        # DataLoader workers: very conservative to avoid memory issues
        dataloader_workers = min(4, max(1, physical_cores // 2))
        
        logger.info(f"Calculated thread counts - PyTorch: {torch_threads}, "
                   f"MKL: {mkl_threads}, OMP: {omp_threads}, "
                   f"DataLoader: {dataloader_workers}")
        
        return {
            "torch_threads": torch_threads,
            "mkl_threads": mkl_threads,
            "omp_threads": omp_threads,
            "dataloader_workers": dataloader_workers
        }
    
    def create_optimization_config(self) -> CPUOptimizationConfig:
        """Create CPU optimization configuration based on hardware."""
        thread_counts = self.calculate_optimal_thread_counts()
        
        # Determine MKL usage
        use_mkl = self.platform_caps.has_mkl and TORCH_AVAILABLE
        if use_mkl:
            try:
                use_mkl = torch.backends.mkl.is_available()
            except AttributeError:
                use_mkl = False
        
        # Determine Accelerate framework usage (macOS only)
        use_accelerate = (self.platform_caps.has_accelerate and 
                         platform.system() == "Darwin")
        
        # Mixed precision: only enable if beneficial on CPU
        # Generally not recommended for CPU training, but can be useful
        # for inference or specific operations
        enable_mixed_precision = False
        
        # JIT compilation can help with repeated operations
        enable_jit = TORCH_AVAILABLE
        
        return CPUOptimizationConfig(
            torch_threads=thread_counts["torch_threads"],
            mkl_threads=thread_counts["mkl_threads"],
            omp_threads=thread_counts["omp_threads"],
            dataloader_workers=thread_counts["dataloader_workers"],
            use_mkl=use_mkl,
            mkl_dynamic=True,  # Allow dynamic thread adjustment
            use_accelerate=use_accelerate,
            enable_jit=enable_jit,
            enable_mixed_precision=enable_mixed_precision,
            tensor_core_optimization=False  # Not applicable for CPU
        )
    
    def configure_torch_threads(self, config: Optional[CPUOptimizationConfig] = None) -> None:
        """Configure PyTorch threading for optimal CPU performance."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping torch thread configuration.")
            return
            
        if config is None:
            config = self.create_optimization_config()
        
        # Set PyTorch thread count
        torch.set_num_threads(config.torch_threads)
        logger.info(f"Set PyTorch threads to {config.torch_threads}")
        
        # Configure inter-op and intra-op parallelism
        try:
            torch.set_num_interop_threads(max(1, config.torch_threads // 2))
        except RuntimeError as e:
            if "cannot set number of interop threads after parallel work has started" in str(e):
                logger.warning("Cannot set interop threads after PyTorch has started parallel work")
            else:
                raise
        
        # Enable/disable JIT if requested
        if hasattr(torch.jit, 'set_fusion_strategy'):
            if config.enable_jit:
                torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
            else:
                torch.jit.set_fusion_strategy([])
    
    def setup_mkl_optimization(self, config: Optional[CPUOptimizationConfig] = None) -> None:
        """Set up Intel MKL optimization."""
        if config is None:
            config = self.create_optimization_config()
        
        if not config.use_mkl:
            logger.info("MKL optimization disabled or not available")
            return
        
        # Store original environment variables
        mkl_vars = ['MKL_NUM_THREADS', 'MKL_DYNAMIC', 'MKL_THREADING_LAYER']
        for var in mkl_vars:
            if var in os.environ:
                self._original_env[var] = os.environ[var]
        
        # Set MKL environment variables
        os.environ['MKL_NUM_THREADS'] = str(config.mkl_threads)
        os.environ['MKL_DYNAMIC'] = 'TRUE' if config.mkl_dynamic else 'FALSE'
        
        # Use Intel threading layer if available
        if self.platform_caps.supports_avx2:
            os.environ['MKL_THREADING_LAYER'] = 'INTEL'
        
        logger.info(f"Configured MKL with {config.mkl_threads} threads, "
                   f"dynamic={'enabled' if config.mkl_dynamic else 'disabled'}")
    
    def setup_accelerate_framework(self, config: Optional[CPUOptimizationConfig] = None) -> None:
        """Set up macOS Accelerate framework optimization."""
        if config is None:
            config = self.create_optimization_config()
        
        if not config.use_accelerate:
            logger.info("Accelerate framework optimization disabled or not available")
            return
        
        if platform.system() != "Darwin":
            logger.warning("Accelerate framework only available on macOS")
            return
        
        # Set environment variables for Accelerate framework
        accelerate_vars = ['VECLIB_MAXIMUM_THREADS']
        for var in accelerate_vars:
            if var in os.environ:
                self._original_env[var] = os.environ[var]
        
        # Configure Accelerate threading
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(config.torch_threads)
        
        logger.info(f"Configured Accelerate framework with {config.torch_threads} threads")
    
    def setup_openmp_optimization(self, config: Optional[CPUOptimizationConfig] = None) -> None:
        """Set up OpenMP optimization."""
        if config is None:
            config = self.create_optimization_config()
        
        # Store original OpenMP environment variables
        omp_vars = ['OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES']
        for var in omp_vars:
            if var in os.environ:
                self._original_env[var] = os.environ[var]
        
        # Set OpenMP environment variables
        os.environ['OMP_NUM_THREADS'] = str(config.omp_threads)
        os.environ['OMP_DYNAMIC'] = 'TRUE'  # Allow dynamic adjustment
        os.environ['OMP_PROC_BIND'] = 'TRUE'  # Bind threads to cores
        os.environ['OMP_PLACES'] = 'cores'  # Use physical cores
        
        logger.info(f"Configured OpenMP with {config.omp_threads} threads")
    
    def configure_all(self, config: Optional[CPUOptimizationConfig] = None) -> CPUOptimizationConfig:
        """Configure all CPU optimizations."""
        if config is None:
            config = self.create_optimization_config()
        
        logger.info("Configuring CPU optimizations for MacBook TRM training")
        logger.info(f"Hardware: {self.cpu_specs.brand} - {self.cpu_specs.cores} cores, "
                   f"{self.cpu_specs.threads} threads")
        
        # Apply all optimizations
        self.setup_openmp_optimization(config)
        self.setup_mkl_optimization(config)
        self.setup_accelerate_framework(config)
        self.configure_torch_threads(config)
        
        self._is_configured = True
        logger.info("CPU optimization configuration complete")
        
        return config
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of DataLoader workers."""
        return self.hardware_detector.get_optimal_worker_count()
    
    def restore_environment(self) -> None:
        """Restore original environment variables."""
        for var, value in self._original_env.items():
            os.environ[var] = value
        
        # Remove variables that weren't originally set
        vars_to_remove = []
        for var in ['MKL_NUM_THREADS', 'MKL_DYNAMIC', 'MKL_THREADING_LAYER',
                   'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OMP_DYNAMIC',
                   'OMP_PROC_BIND', 'OMP_PLACES']:
            if var not in self._original_env and var in os.environ:
                vars_to_remove.append(var)
        
        for var in vars_to_remove:
            del os.environ[var]
        
        self._original_env.clear()
        self._is_configured = False
        logger.info("Restored original environment variables")
    
    def get_configuration_summary(self) -> Dict:
        """Get summary of current CPU optimization configuration."""
        if not self._is_configured:
            return {"status": "not_configured"}
        
        config = self.create_optimization_config()
        
        return {
            "status": "configured",
            "hardware": {
                "cpu_brand": self.cpu_specs.brand,
                "cores": self.cpu_specs.cores,
                "threads": self.cpu_specs.threads,
                "features": self.cpu_specs.features,
            },
            "threading": {
                "torch_threads": config.torch_threads,
                "mkl_threads": config.mkl_threads,
                "omp_threads": config.omp_threads,
                "dataloader_workers": config.dataloader_workers,
            },
            "optimizations": {
                "mkl_enabled": config.use_mkl,
                "accelerate_enabled": config.use_accelerate,
                "jit_enabled": config.enable_jit,
                "mixed_precision": config.enable_mixed_precision,
            },
            "environment_vars": {
                var: os.environ.get(var, "not_set")
                for var in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
            }
        }


class TensorOperationOptimizer:
    """Optimizer for tensor operations on CPU."""
    
    def __init__(self, cpu_optimizer: CPUOptimizer):
        """Initialize with CPU optimizer."""
        self.cpu_optimizer = cpu_optimizer
        self.cpu_specs = cpu_optimizer.cpu_specs
        self.platform_caps = cpu_optimizer.platform_caps
    
    def configure_blas_libraries(self) -> None:
        """Configure BLAS libraries for optimal performance."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping BLAS configuration.")
            return
        
        # Check which BLAS library PyTorch is using
        blas_info = self._get_blas_info()
        logger.info(f"PyTorch BLAS backend: {blas_info}")
        
        # Configure based on available BLAS
        if self.platform_caps.has_mkl:
            self._configure_mkl_blas()
        elif self.platform_caps.has_accelerate:
            self._configure_accelerate_blas()
        else:
            logger.warning("No optimized BLAS library detected")
    
    def _get_blas_info(self) -> str:
        """Get information about BLAS library being used."""
        if not TORCH_AVAILABLE:
            return "PyTorch not available"
        
        try:
            # Try to get BLAS info from torch
            if hasattr(torch, 'show_config'):
                config = torch.show_config()
                if 'BLAS' in config:
                    return "MKL" if "MKL" in config else "OpenBLAS"
            
            # Fallback: check backends
            if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
                return "MKL"
            elif platform.system() == "Darwin":
                return "Accelerate"
            else:
                return "OpenBLAS"
        except Exception as e:
            logger.warning(f"Could not determine BLAS library: {e}")
            return "Unknown"
    
    def _configure_mkl_blas(self) -> None:
        """Configure Intel MKL BLAS settings."""
        logger.info("Configuring Intel MKL BLAS")
        
        # MKL-specific optimizations
        if 'MKL_CBWR' not in os.environ:
            # Conditional Numerical Reproducibility
            os.environ['MKL_CBWR'] = 'AUTO'
        
        # Enable MKL verbose mode for debugging (optional)
        # os.environ['MKL_VERBOSE'] = '1'
    
    def _configure_accelerate_blas(self) -> None:
        """Configure macOS Accelerate BLAS settings."""
        logger.info("Configuring macOS Accelerate BLAS")
        
        # Accelerate-specific optimizations
        # Most optimizations are handled automatically by the framework
        pass
    
    def optimize_tensor_operations(self) -> None:
        """Apply tensor operation optimizations."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping tensor optimizations.")
            return
        
        # Configure BLAS libraries
        self.configure_blas_libraries()
        
        # Set tensor operation flags
        self._set_tensor_flags()
        
        # Configure memory allocation
        self._configure_memory_allocation()
    
    def _set_tensor_flags(self) -> None:
        """Set PyTorch tensor operation flags."""
        if not TORCH_AVAILABLE:
            return
            
        # Enable/disable specific optimizations based on hardware
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = False  # Not using CUDA
        
        # Enable deterministic operations for reproducibility
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)  # Allow non-deterministic for speed
        
        # Configure autograd
        torch.autograd.set_detect_anomaly(False)  # Disable for performance
    
    def _configure_memory_allocation(self) -> None:
        """Configure memory allocation for tensor operations."""
        if not TORCH_AVAILABLE:
            return
            
        # Set memory allocation strategy
        if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
            # MKL memory pool settings
            pass
        
        # Configure garbage collection
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive GC for memory-constrained systems
    
    def configure_mixed_precision_cpu(self, enable: bool = False) -> None:
        """Configure CPU-specific mixed precision settings."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping mixed precision configuration.")
            return
        
        if enable:
            logger.info("Enabling CPU mixed precision (experimental)")
            # CPU mixed precision is limited, but we can optimize specific operations
            
            # Enable autocast for CPU (PyTorch 1.10+)
            try:
                if hasattr(torch.cpu.amp, 'autocast'):
                    logger.info("CPU autocast available")
                else:
                    logger.warning("CPU autocast not available in this PyTorch version")
            except (AttributeError, NameError):
                logger.warning("CPU mixed precision not supported")
        else:
            logger.info("Mixed precision disabled for CPU training")
    
    def set_cpu_thread_affinity(self) -> None:
        """Set CPU thread affinity for optimal performance."""
        try:
            import psutil
            
            # Get current process
            process = psutil.Process()
            
            # Set CPU affinity to use all available cores
            available_cpus = list(range(self.cpu_specs.cores))
            process.cpu_affinity(available_cpus)
            
            logger.info(f"Set CPU affinity to cores: {available_cpus}")
        except (ImportError, AttributeError, psutil.AccessDenied) as e:
            logger.warning(f"Could not set CPU affinity: {e}")
    
    def optimize_for_inference(self) -> None:
        """Apply optimizations specifically for inference workloads."""
        if not TORCH_AVAILABLE:
            return
        
        logger.info("Applying inference-specific optimizations")
        
        # Enable JIT optimizations for inference
        torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        
        # Set inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            logger.info("Inference mode available")
        
        # Optimize for single-threaded inference if needed
        if self.cpu_specs.cores <= 2:
            torch.set_num_threads(1)
            logger.info("Using single-threaded mode for inference on low-core system")
    
    def optimize_for_training(self) -> None:
        """Apply optimizations specifically for training workloads."""
        if not TORCH_AVAILABLE:
            return
        
        logger.info("Applying training-specific optimizations")
        
        # Use all available cores for training
        config = self.cpu_optimizer.create_optimization_config()
        torch.set_num_threads(config.torch_threads)
        
        # Enable autograd optimizations
        torch.autograd.set_detect_anomaly(False)  # Disable for performance
        
        # Configure gradient computation
        if hasattr(torch.backends, 'opt_einsum'):
            torch.backends.opt_einsum.enabled = True
    
    def benchmark_tensor_operations(self) -> Dict:
        """Benchmark basic tensor operations to validate optimizations."""
        if not TORCH_AVAILABLE:
            return {"status": "pytorch_not_available"}
        
        import time
        
        logger.info("Benchmarking tensor operations")
        
        # Create test tensors
        size = 1000
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        results = {}
        
        # Matrix multiplication benchmark
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        mm_time = (time.time() - start_time) / 10
        results['matrix_multiply_ms'] = mm_time * 1000
        
        # Element-wise operations benchmark
        start_time = time.time()
        for _ in range(100):
            c = a + b
            c = torch.relu(c)
        elementwise_time = (time.time() - start_time) / 100
        results['elementwise_ops_ms'] = elementwise_time * 1000
        
        # Convolution benchmark (if applicable)
        try:
            conv_input = torch.randn(1, 3, 224, 224)
            conv_layer = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            
            start_time = time.time()
            for _ in range(10):
                output = conv_layer(conv_input)
            conv_time = (time.time() - start_time) / 10
            results['convolution_ms'] = conv_time * 1000
        except Exception as e:
            results['convolution_ms'] = f"Error: {e}"
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def get_optimization_info(self) -> Dict:
        """Get information about tensor operation optimizations."""
        if not TORCH_AVAILABLE:
            return {"status": "pytorch_not_available"}
        
        return {
            "blas_library": self._get_blas_info(),
            "mkl_available": self.platform_caps.has_mkl,
            "accelerate_available": self.platform_caps.has_accelerate,
            "avx_support": self.platform_caps.supports_avx,
            "avx2_support": self.platform_caps.supports_avx2,
            "torch_threads": torch.get_num_threads() if TORCH_AVAILABLE else 0,
            "interop_threads": torch.get_num_interop_threads() if TORCH_AVAILABLE else 0,
            "cpu_affinity": self._get_cpu_affinity(),
            "optimization_flags": self._get_optimization_flags(),
        }
    
    def _get_cpu_affinity(self) -> list:
        """Get current CPU affinity."""
        try:
            import psutil
            process = psutil.Process()
            return process.cpu_affinity()
        except (ImportError, AttributeError):
            return []
    
    def _get_optimization_flags(self) -> Dict:
        """Get current optimization flags."""
        if not TORCH_AVAILABLE:
            return {}
        
        flags = {}
        
        # Check JIT settings
        try:
            flags['jit_enabled'] = torch.jit.is_scripting()
        except:
            flags['jit_enabled'] = False
        
        # Check autograd settings
        try:
            flags['detect_anomaly'] = torch.is_anomaly_enabled()
        except:
            flags['detect_anomaly'] = False
        
        return flags