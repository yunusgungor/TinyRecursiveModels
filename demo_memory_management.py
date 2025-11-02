#!/usr/bin/env python3
"""
Demonstration of the MacBook memory management system for TRM training.

This script shows how to use the memory management components to optimize
training on memory-constrained MacBook hardware.
"""

import time
from macbook_optimization import (
    MemoryConfig,
    MemoryManager,
    GradientAccumulationConfig,
    GradientAccumulator,
    DataLoadingConfig,
    DataLoadingManager,
)


def demo_memory_management():
    """Demonstrate memory management functionality."""
    print("=== MacBook TRM Training Memory Management Demo ===\n")
    
    # 1. Memory Configuration
    print("1. Setting up memory management configuration...")
    memory_config = MemoryConfig(
        memory_warning_threshold=70.0,
        memory_critical_threshold=80.0,
        memory_emergency_threshold=90.0,
        max_batch_size=32,
        min_batch_size=1,
        safety_margin_mb=500.0
    )
    print(f"   - Warning threshold: {memory_config.memory_warning_threshold}%")
    print(f"   - Critical threshold: {memory_config.memory_critical_threshold}%")
    print(f"   - Max batch size: {memory_config.max_batch_size}")
    print(f"   - Safety margin: {memory_config.safety_margin_mb} MB")
    
    # 2. Memory Manager
    print("\n2. Initializing memory manager...")
    memory_manager = MemoryManager(memory_config)
    
    # Get current memory stats
    memory_stats = memory_manager.monitor_memory_usage()
    print(f"   - Total memory: {memory_stats.total_mb:.1f} MB")
    print(f"   - Available memory: {memory_stats.available_mb:.1f} MB")
    print(f"   - Memory usage: {memory_stats.percent_used:.1f}%")
    
    # 3. Batch Size Calculation
    print("\n3. Calculating optimal batch size for TRM model...")
    model_params = 7_000_000  # 7M parameter TRM model
    sequence_length = 512
    
    batch_recommendation = memory_manager.calculate_optimal_batch_size(
        model_params, sequence_length
    )
    
    print(f"   - Model parameters: {model_params:,}")
    print(f"   - Sequence length: {sequence_length}")
    print(f"   - Recommended batch size: {batch_recommendation.recommended_batch_size}")
    print(f"   - Max safe batch size: {batch_recommendation.max_safe_batch_size}")
    print(f"   - Memory utilization: {batch_recommendation.memory_utilization_percent:.1f}%")
    print(f"   - Reasoning: {batch_recommendation.reasoning}")
    
    if batch_recommendation.warnings:
        print("   - Warnings:")
        for warning in batch_recommendation.warnings:
            print(f"     * {warning}")
    
    # 4. Gradient Accumulation Setup
    print("\n4. Setting up gradient accumulation...")
    grad_config = GradientAccumulationConfig(
        target_batch_size=batch_recommendation.recommended_batch_size * 2,  # Effective larger batch
        max_micro_batch_size=batch_recommendation.recommended_batch_size,
        gradient_clipping=1.0,
        scale_gradients=True
    )
    
    accumulator = GradientAccumulator(grad_config, memory_manager)
    accumulation_info = accumulator.get_accumulation_info()
    
    print(f"   - Target effective batch size: {grad_config.target_batch_size}")
    print(f"   - Micro batch size: {accumulation_info['state']['micro_batch_size']}")
    print(f"   - Accumulation steps: {accumulation_info['state']['accumulation_steps']}")
    print(f"   - Gradient scale: {accumulation_info['state']['gradient_scale']:.3f}")
    
    # 5. Memory Recommendations
    print("\n5. Getting comprehensive memory recommendations...")
    recommendations = memory_manager.get_memory_recommendations(model_params)
    
    print("   Current Memory Status:")
    current_mem = recommendations["current_memory"]
    print(f"     - Used: {current_mem['used_percent']:.1f}%")
    print(f"     - Available: {current_mem['available_gb']:.2f} GB")
    print(f"     - Pressure level: {current_mem['pressure_level']}")
    
    print("   Batch Size Recommendations:")
    batch_info = recommendations["batch_size"]
    print(f"     - Current: {batch_info['current']}")
    print(f"     - Recommended: {batch_info['recommended']}")
    print(f"     - Max safe: {batch_info['max_safe']}")
    print(f"     - Utilization: {batch_info['utilization_percent']:.1f}%")
    
    print("   Action Recommendations:")
    rec_info = recommendations["recommendations"]
    print(f"     - Action: {rec_info['action']}")
    if rec_info['warnings']:
        print("     - Warnings:")
        for warning in rec_info['warnings']:
            print(f"       * {warning}")
    
    # 6. Data Loading Configuration
    print("\n6. Configuring memory-efficient data loading...")
    data_config = DataLoadingConfig(
        max_memory_usage_mb=min(1000.0, memory_stats.available_mb * 0.3),
        use_memory_mapping=True,
        streaming_threshold_mb=500.0,
        pin_memory=False,  # Conservative for MacBook
        prefetch_factor=2
    )
    
    data_manager = DataLoadingManager(memory_manager, data_config)
    print(f"   - Max memory for data loading: {data_config.max_memory_usage_mb:.1f} MB")
    print(f"   - Memory mapping enabled: {data_config.use_memory_mapping}")
    print(f"   - Streaming threshold: {data_config.streaming_threshold_mb} MB")
    print(f"   - Pin memory: {data_config.pin_memory}")
    
    # 7. Memory Monitoring Demo
    print("\n7. Demonstrating memory monitoring...")
    print("   Starting memory monitoring for 5 seconds...")
    
    def memory_pressure_callback(pressure_info):
        if pressure_info.pressure_level != "low":
            print(f"   Memory pressure detected: {pressure_info.pressure_level} "
                  f"({pressure_info.current_usage_percent:.1f}% used)")
    
    memory_manager.add_memory_pressure_callback(memory_pressure_callback)
    
    # Simulate some memory usage changes
    start_time = time.time()
    while time.time() - start_time < 5:
        time.sleep(1)
        current_stats = memory_manager.monitor_memory_usage()
        print(f"   Memory: {current_stats.percent_used:.1f}% used, "
              f"{current_stats.available_mb:.0f} MB available")
    
    # 8. Summary
    print("\n8. Training Configuration Summary:")
    print("   Recommended settings for MacBook TRM training:")
    print(f"   - Batch size: {batch_recommendation.recommended_batch_size}")
    print(f"   - Gradient accumulation steps: {accumulation_info['state']['accumulation_steps']}")
    print(f"   - Effective batch size: {accumulation_info['state']['effective_batch_size']}")
    print(f"   - Memory monitoring: Enabled with {memory_config.memory_warning_threshold}% warning threshold")
    print(f"   - Data loading: Memory-efficient with {data_config.max_memory_usage_mb:.0f} MB limit")
    
    print("\n=== Demo Complete ===")
    print("The memory management system is ready for TRM training on MacBook!")


if __name__ == "__main__":
    try:
        demo_memory_management()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()