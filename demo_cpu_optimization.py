#!/usr/bin/env python3
"""
Demo script for MacBook CPU optimization module.

This script demonstrates how to use the CPU optimization module
to configure optimal settings for TRM training on MacBook hardware.
"""

import logging
from macbook_optimization import (
    HardwareDetector,
    CPUOptimizer,
    TensorOperationOptimizer,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate CPU optimization functionality."""
    print("=== MacBook CPU Optimization Demo ===\n")
    
    # 1. Hardware Detection
    print("1. Detecting hardware specifications...")
    detector = HardwareDetector()
    hardware_summary = detector.get_hardware_summary()
    
    print(f"CPU: {hardware_summary['cpu']['brand']}")
    print(f"Cores: {hardware_summary['cpu']['cores']} physical, {hardware_summary['cpu']['threads']} logical")
    print(f"Memory: {hardware_summary['memory']['total_gb']} GB total, {hardware_summary['memory']['available_gb']} GB available")
    print(f"Platform: {hardware_summary['platform']['os']} {hardware_summary['platform']['os_version']}")
    print(f"PyTorch: {hardware_summary['platform']['torch_version']}")
    print(f"MKL Available: {hardware_summary['platform']['has_mkl']}")
    print(f"Accelerate Available: {hardware_summary['platform']['has_accelerate']}")
    print()
    
    # 2. CPU Optimization Configuration
    print("2. Creating CPU optimization configuration...")
    cpu_optimizer = CPUOptimizer(detector)
    config = cpu_optimizer.create_optimization_config()
    
    print(f"Recommended PyTorch threads: {config.torch_threads}")
    print(f"Recommended MKL threads: {config.mkl_threads}")
    print(f"Recommended OpenMP threads: {config.omp_threads}")
    print(f"Recommended DataLoader workers: {config.dataloader_workers}")
    print(f"Use MKL: {config.use_mkl}")
    print(f"Use Accelerate: {config.use_accelerate}")
    print(f"Enable JIT: {config.enable_jit}")
    print()
    
    # 3. Apply CPU Optimizations
    print("3. Applying CPU optimizations...")
    try:
        applied_config = cpu_optimizer.configure_all()
        print("✓ CPU optimizations applied successfully")
        
        # Show configuration summary
        summary = cpu_optimizer.get_configuration_summary()
        if summary["status"] == "configured":
            print("Configuration Summary:")
            print(f"  - PyTorch threads: {summary['threading']['torch_threads']}")
            print(f"  - MKL threads: {summary['threading']['mkl_threads']}")
            print(f"  - OpenMP threads: {summary['threading']['omp_threads']}")
            print(f"  - DataLoader workers: {summary['threading']['dataloader_workers']}")
            print(f"  - MKL enabled: {summary['optimizations']['mkl_enabled']}")
            print(f"  - Accelerate enabled: {summary['optimizations']['accelerate_enabled']}")
        print()
        
    except Exception as e:
        print(f"✗ Error applying optimizations: {e}")
        print()
    
    # 4. Tensor Operation Optimization
    print("4. Setting up tensor operation optimization...")
    tensor_optimizer = TensorOperationOptimizer(cpu_optimizer)
    
    try:
        tensor_optimizer.optimize_tensor_operations()
        print("✓ Tensor operations optimized")
        
        # Get optimization info
        opt_info = tensor_optimizer.get_optimization_info()
        if "status" not in opt_info or opt_info["status"] != "pytorch_not_available":
            print("Tensor Optimization Info:")
            print(f"  - BLAS library: {opt_info.get('blas_library', 'Unknown')}")
            print(f"  - AVX support: {opt_info.get('avx_support', False)}")
            print(f"  - AVX2 support: {opt_info.get('avx2_support', False)}")
            if 'torch_threads' in opt_info:
                print(f"  - PyTorch threads: {opt_info['torch_threads']}")
        else:
            print("  - PyTorch not available, tensor optimizations limited")
        print()
        
    except Exception as e:
        print(f"✗ Error optimizing tensor operations: {e}")
        print()
    
    # 5. Benchmark (if PyTorch is available)
    print("5. Running tensor operation benchmark...")
    try:
        benchmark_results = tensor_optimizer.benchmark_tensor_operations()
        if "status" not in benchmark_results:
            print("Benchmark Results:")
            print(f"  - Matrix multiplication: {benchmark_results.get('matrix_multiply_ms', 'N/A'):.2f} ms")
            print(f"  - Element-wise operations: {benchmark_results.get('elementwise_ops_ms', 'N/A'):.2f} ms")
            if 'convolution_ms' in benchmark_results and isinstance(benchmark_results['convolution_ms'], (int, float)):
                print(f"  - Convolution: {benchmark_results['convolution_ms']:.2f} ms")
        else:
            print("  - Benchmark skipped (PyTorch not available)")
        print()
        
    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        print()
    
    # 6. Cleanup
    print("6. Cleaning up...")
    try:
        cpu_optimizer.restore_environment()
        print("✓ Environment restored")
    except Exception as e:
        print(f"✗ Error restoring environment: {e}")
    
    print("\n=== Demo Complete ===")
    print("\nTo use these optimizations in your training script:")
    print("1. Import: from macbook_optimization import CPUOptimizer, HardwareDetector")
    print("2. Create: optimizer = CPUOptimizer(HardwareDetector())")
    print("3. Configure: config = optimizer.configure_all()")
    print("4. Train with optimized settings!")


if __name__ == "__main__":
    main()