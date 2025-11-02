#!/usr/bin/env python3
"""
Demonstration script for MacBook TRM training optimization infrastructure.

This script shows how to use the hardware detection, resource monitoring,
and configuration management components for MacBook-optimized TRM training.
"""

import time
from macbook_optimization import (
    HardwareDetector,
    ResourceMonitor,
    MacBookConfigManager
)


def main():
    print("=== MacBook TRM Training Optimization Demo ===\n")
    
    # 1. Hardware Detection
    print("1. Hardware Detection")
    print("-" * 20)
    detector = HardwareDetector()
    
    # Get detailed hardware summary
    hardware_summary = detector.get_hardware_summary()
    
    print(f"CPU: {hardware_summary['cpu']['brand']}")
    print(f"Cores: {hardware_summary['cpu']['cores']} cores, {hardware_summary['cpu']['threads']} threads")
    print(f"Base Frequency: {hardware_summary['cpu']['base_frequency_ghz']:.1f} GHz")
    print(f"Features: {', '.join(hardware_summary['cpu']['features'])}")
    print()
    
    print(f"Memory: {hardware_summary['memory']['total_gb']:.1f}GB {hardware_summary['memory']['type']}")
    print(f"Available: {hardware_summary['memory']['available_gb']:.1f}GB")
    if hardware_summary['memory']['speed_mhz']:
        print(f"Speed: {hardware_summary['memory']['speed_mhz']} MHz")
    print()
    
    print(f"Platform: {hardware_summary['platform']['os']} {hardware_summary['platform']['os_version']}")
    print(f"Python: {hardware_summary['platform']['python_version']}")
    print(f"PyTorch: {hardware_summary['platform']['torch_version']}")
    print(f"Intel MKL: {'Yes' if hardware_summary['platform']['has_mkl'] else 'No'}")
    print(f"macOS Accelerate: {'Yes' if hardware_summary['platform']['has_accelerate'] else 'No'}")
    print(f"AVX Support: {'Yes' if hardware_summary['platform']['supports_avx'] else 'No'}")
    print()
    
    # 2. Configuration Management
    print("2. Optimal Configuration Detection")
    print("-" * 35)
    config_manager = MacBookConfigManager()
    optimal_config = config_manager.detect_optimal_config()
    
    config_summary = config_manager.get_config_summary(optimal_config)
    
    print("Training Configuration:")
    print(f"  Batch Size: {config_summary['training']['batch_size']}")
    print(f"  Gradient Accumulation: {config_summary['training']['gradient_accumulation_steps']} steps")
    print(f"  Effective Batch Size: {config_summary['training']['effective_batch_size']}")
    print(f"  Learning Rate: {config_summary['training']['learning_rate']:.2e}")
    print()
    
    print("Hardware Configuration:")
    print(f"  Workers: {config_summary['hardware']['num_workers']}")
    print(f"  PyTorch Threads: {config_summary['hardware']['torch_threads']}")
    print(f"  Memory Limit: {config_summary['hardware']['memory_limit_mb']} MB")
    print(f"  Intel MKL: {'Enabled' if config_summary['hardware']['use_mkl'] else 'Disabled'}")
    print()
    
    print("Monitoring Configuration:")
    print(f"  Memory Monitoring: {'Enabled' if config_summary['monitoring']['memory_monitoring'] else 'Disabled'}")
    print(f"  Thermal Monitoring: {'Enabled' if config_summary['monitoring']['thermal_monitoring'] else 'Disabled'}")
    print(f"  Monitoring Interval: {config_summary['monitoring']['monitoring_interval']}s")
    print()
    
    # Validate configuration
    validation = config_manager.validate_config(optimal_config)
    if validation['valid']:
        print("✅ Configuration is valid for current hardware")
    else:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
        print("Suggestions:")
        for suggestion in validation['suggestions']:
            print(f"   - {suggestion}")
    print()
    
    # 3. Resource Monitoring Demo
    print("3. Resource Monitoring Demo")
    print("-" * 27)
    monitor = ResourceMonitor()
    
    print("Starting resource monitoring for 5 seconds...")
    monitor.start_monitoring(interval=1.0)
    
    # Monitor for a few seconds
    for i in range(5):
        time.sleep(1)
        current = monitor.get_current_snapshot()
        print(f"  [{i+1}s] Memory: {current.memory.percent_used:.1f}% | "
              f"CPU: {current.cpu.percent_total:.1f}% | "
              f"Thermal: {current.thermal.thermal_state}")
    
    monitor.stop_monitoring()
    
    # Get monitoring summary
    summary = monitor.get_resource_summary()
    print("\nResource Summary:")
    print(f"  Current Memory Usage: {summary['current']['memory_used_percent']:.1f}%")
    print(f"  Available Memory: {summary['current']['memory_available_gb']:.1f}GB")
    print(f"  Current CPU Usage: {summary['current']['cpu_percent']:.1f}%")
    print(f"  Thermal State: {summary['current']['thermal_state']}")
    
    if summary['alerts']['memory_pressure']:
        print("  ⚠️  Memory pressure detected!")
    if summary['alerts']['thermal_throttling']:
        print("  🔥 Thermal throttling likely!")
    
    print()
    
    # 4. Configuration Templates
    print("4. Configuration Templates")
    print("-" * 25)
    
    # Show templates for different MacBook configurations
    configs = [
        ("MacBook Air 8GB", 8, 4),
        ("MacBook Pro 16GB", 16, 8),
        ("MacBook Pro 32GB", 32, 10)
    ]
    
    for name, memory_gb, cpu_cores in configs:
        template = config_manager.create_config_template(memory_gb, cpu_cores)
        print(f"{name}:")
        print(f"  Batch Size: {template.batch_size}")
        print(f"  Memory Limit: {template.memory_limit_mb}MB")
        print(f"  Workers: {template.num_workers}")
        print(f"  Checkpoint Interval: {template.checkpoint_interval}")
        print()
    
    print("=== Demo Complete ===")
    print("\nTo use this infrastructure in your TRM training:")
    print("1. Import the components: from macbook_optimization import HardwareDetector, ResourceMonitor, MacBookConfigManager")
    print("2. Detect optimal configuration: config = MacBookConfigManager().detect_optimal_config()")
    print("3. Start resource monitoring: monitor = ResourceMonitor(); monitor.start_monitoring()")
    print("4. Use the configuration in your training script")


if __name__ == "__main__":
    main()