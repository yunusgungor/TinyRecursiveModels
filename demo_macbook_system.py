#!/usr/bin/env python3
"""
Simple demo of the MacBook training system capabilities.
"""

import sys
import time
from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager, MemoryConfig
from macbook_optimization.cpu_optimization import CPUOptimizer
from macbook_optimization.resource_monitoring import ResourceMonitor
from macbook_optimization.training_config_adapter import TrainingConfigAdapter
from macbook_optimization.dataset_management import DatasetManager, DatasetManagementConfig

def demo_macbook_system():
    """Demonstrate MacBook training system capabilities."""
    
    print("🍎 MacBook Training System Demo")
    print("=" * 50)
    
    # 1. Hardware Detection
    print("\n1. Hardware Detection:")
    hardware_detector = HardwareDetector()
    hardware_summary = hardware_detector.get_hardware_summary()
    
    print(f"   CPU: {hardware_summary['cpu']['brand']}")
    print(f"   Cores: {hardware_summary['cpu']['cores']}")
    print(f"   Memory: {hardware_summary['memory']['available_gb']:.1f}GB available")
    print(f"   Platform: {hardware_summary['platform']['os']} {hardware_summary['platform']['os_version']}")
    
    # 2. CPU Optimization
    print("\n2. CPU Optimization:")
    cpu_optimizer = CPUOptimizer(hardware_detector)
    cpu_config = cpu_optimizer.configure_all()
    
    print(f"   PyTorch threads: {cpu_config.torch_threads}")
    print(f"   MKL threads: {cpu_config.mkl_threads}")
    print(f"   OpenMP threads: {cpu_config.omp_threads}")
    print(f"   DataLoader workers: {cpu_config.dataloader_workers}")
    
    # 3. Memory Management
    print("\n3. Memory Management:")
    memory_manager = MemoryManager()
    memory_stats = memory_manager.monitor_memory_usage()
    
    print(f"   Total memory: {memory_stats.total_mb:.0f}MB")
    print(f"   Used memory: {memory_stats.used_mb:.0f}MB ({memory_stats.percent_used:.1f}%)")
    print(f"   Available memory: {memory_stats.available_mb:.0f}MB")
    
    # 4. Resource Monitoring
    print("\n4. Resource Monitoring:")
    resource_monitor = ResourceMonitor()
    resource_monitor.start_monitoring(interval=1.0)
    
    print("   Starting resource monitoring...")
    time.sleep(3)  # Monitor for 3 seconds
    
    snapshot = resource_monitor.get_current_snapshot()
    print(f"   CPU usage: {snapshot.cpu.percent_total:.1f}%")
    print(f"   Memory usage: {snapshot.memory.percent_used:.1f}%")
    temp = snapshot.thermal.cpu_temperature
    print(f"   Temperature: {temp if temp else 'N/A'}°C")
    
    resource_monitor.stop_monitoring()
    
    # 5. Configuration Adaptation
    print("\n5. Configuration Adaptation:")
    config_adapter = TrainingConfigAdapter(hardware_detector)
    
    # Sample base configuration
    base_config = {
        'global_batch_size': 768,
        'lr': 1e-4,
        'epochs': 100,
        'arch': {
            'hidden_size': 512,
            'num_layers': 12,
            'num_heads': 8
        }
    }
    
    dataset_size = 10000
    config_result = config_adapter.create_hardware_appropriate_config(base_config, dataset_size)
    
    print(f"   Original batch size: {base_config['global_batch_size']}")
    print(f"   Adapted batch size: {config_result.training_params.batch_size}")
    print(f"   Gradient accumulation: {config_result.training_params.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config_result.training_params.effective_batch_size}")
    print(f"   Memory limit: {config_result.training_params.memory_limit_mb}MB")
    
    # 6. Dataset Management
    print("\n6. Dataset Management:")
    dataset_config = DatasetManagementConfig(
        max_dataset_memory_mb=800.0,
        streaming_threshold_mb=400.0,
        cache_threshold_mb=200.0,
        chunk_size_mb=50.0,
        enable_caching=True,
        auto_fallback_streaming=True
    )
    dataset_manager = DatasetManager(dataset_config, memory_manager)
    
    # Analyze our demo dataset
    dataset_paths = ['data/arc-demo']
    analysis = dataset_manager.analyze_dataset_requirements(dataset_paths, "train")
    
    print(f"   Dataset size: {analysis['total_size_mb']:.1f}MB")
    print(f"   Recommended strategy: {analysis['recommended_strategy']}")
    print(f"   Memory utilization: {analysis['memory_utilization_percent']:.1f}%")
    
    # 7. Summary
    print("\n7. System Summary:")
    print("   ✅ Hardware detection: Working")
    print("   ✅ CPU optimization: Configured")
    print("   ✅ Memory management: Active")
    print("   ✅ Resource monitoring: Functional")
    print("   ✅ Configuration adaptation: Applied")
    print("   ✅ Dataset management: Ready")
    
    print(f"\n🎉 MacBook training system is fully operational!")
    print(f"   Optimized for: {hardware_summary['cpu']['brand']}")
    print(f"   Memory available: {hardware_summary['memory']['available_gb']:.1f}GB")
    print(f"   CPU cores: {hardware_summary['cpu']['cores']}")
    print(f"   Recommended batch size: {config_result.training_params.batch_size}")
    
    # Cleanup
    cpu_optimizer.restore_environment()
    
    return True

if __name__ == "__main__":
    try:
        success = demo_macbook_system()
        if success:
            print("\n✨ Demo completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        sys.exit(1)