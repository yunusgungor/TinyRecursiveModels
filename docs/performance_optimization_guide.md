# MacBook TRM Performance Optimization Guide

This guide provides detailed strategies for optimizing TRM training performance on MacBook hardware, covering memory management, CPU utilization, and dataset handling.

## Table of Contents

1. [Performance Fundamentals](#performance-fundamentals)
2. [Memory Optimization](#memory-optimization)
3. [CPU Optimization](#cpu-optimization)
4. [Dataset Loading Optimization](#dataset-loading-optimization)
5. [Model Architecture Tuning](#model-architecture-tuning)
6. [Thermal Management](#thermal-management)
7. [Benchmarking and Monitoring](#benchmarking-and-monitoring)
8. [Performance Troubleshooting](#performance-troubleshooting)

## Performance Fundamentals

### Understanding MacBook Constraints

MacBook training faces several unique constraints:

1. **Memory Limitations**: 8-32GB RAM shared between system and training
2. **CPU-Only Training**: No dedicated GPU acceleration
3. **Thermal Constraints**: Sustained high CPU usage causes throttling
4. **Power Management**: Battery vs. performance trade-offs

### Performance Metrics

Key metrics to monitor:

- **Samples per second**: Primary throughput metric
- **Memory usage**: Peak and average memory consumption
- **CPU utilization**: Per-core usage and thermal state
- **Training loss convergence**: Quality vs. speed trade-off

### Baseline Performance Expectations

| MacBook Model | RAM | Small Dataset | Medium Dataset | Large Dataset |
|---------------|-----|---------------|----------------|---------------|
| Air 8GB       | 8GB | 5-15 samples/s | 3-8 samples/s | 1-3 samples/s |
| Pro 13" 16GB  | 16GB| 15-30 samples/s| 8-20 samples/s| 3-10 samples/s|
| Pro 16" 32GB  | 32GB| 25-50 samples/s| 15-35 samples/s| 8-20 samples/s|

## Memory Optimization

### 1. Dynamic Batch Sizing

The most critical optimization for MacBook training:

```python
# Automatic batch size calculation
from macbook_optimization.memory_management import MemoryManager

memory_manager = MemoryManager()
optimal_batch_size = memory_manager.calculate_optimal_batch_size(
    model_size_mb=100,  # Estimated model size
    available_memory_mb=6000,  # Available for training
    safety_margin=0.8  # Use 80% of available memory
)
```

**Configuration:**
```yaml
training:
  # Enable dynamic batch sizing
  dynamic_batch_sizing: true
  memory_pressure_threshold: 75.0
  
  # Gradient accumulation for effective larger batches
  gradient_accumulation_steps: 4
  effective_batch_size: 128  # batch_size * accumulation_steps
```

### 2. Memory-Efficient Model Loading

```python
# Load model with memory optimization
def create_memory_efficient_model(config):
    # Use float32 instead of float64 for CPU training
    config['forward_dtype'] = 'float32'
    
    # Enable gradient checkpointing for large models
    config['gradient_checkpointing'] = True
    
    # Reduce embedding dimensions if memory-constrained
    if available_memory_gb < 12:
        config['hidden_size'] = min(config['hidden_size'], 256)
        config['expansion'] = min(config['expansion'], 2)
    
    return create_model(config)
```

### 3. Gradient Accumulation Strategy

```yaml
# Optimize gradient accumulation for memory vs. speed
training:
  # For 8GB MacBook
  batch_size: 8
  gradient_accumulation_steps: 8
  effective_batch_size: 64
  
  # For 16GB MacBook  
  batch_size: 16
  gradient_accumulation_steps: 4
  effective_batch_size: 64
```

### 4. Memory Monitoring and Cleanup

```python
# Implement aggressive memory management
class MemoryOptimizedTrainer:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.gc_interval = 50  # Force GC every 50 batches
        
    def train_step(self, batch_idx, batch):
        # Monitor memory before training step
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        if memory_stats.percent_used > 85:
            # Aggressive cleanup
            self.memory_manager.force_garbage_collection()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Regular cleanup
        if batch_idx % self.gc_interval == 0:
            self.memory_manager.force_garbage_collection()
```

## CPU Optimization

### 1. Intel MKL Integration

```python
# Configure Intel MKL for optimal performance
from macbook_optimization.cpu_optimization import CPUOptimizer

cpu_optimizer = CPUOptimizer()
config = cpu_optimizer.configure_all()

print(f"Configured {config.torch_threads} threads")
print(f"MKL enabled: {config.mkl_enabled}")
print(f"Vectorization: {config.vectorization_enabled}")
```

**Environment Setup:**
```bash
# Install Intel MKL
conda install mkl mkl-include

# Set environment variables
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
```

### 2. Thread Configuration

```yaml
macbook_optimizations:
  # CPU optimization settings
  torch_threads: 4  # Match CPU core count
  use_mkl: true
  optimize_tensor_operations: true
  enable_vectorization: true
  
  # DataLoader optimization
  num_workers: 2  # Conservative for memory
  pin_memory: true  # If sufficient memory
  prefetch_factor: 2
```

### 3. Tensor Operation Optimization

```python
# Optimize tensor operations for CPU
from macbook_optimization.cpu_optimization import TensorOperationOptimizer

tensor_optimizer = TensorOperationOptimizer()
tensor_optimizer.optimize_for_training()

# This configures:
# - BLAS library selection
# - Thread pool management  
# - Memory allocation strategies
# - Vectorization settings
```

### 4. Model Compilation (PyTorch 2.0+)

```python
# Experimental: Compile model for CPU optimization
if hasattr(torch, 'compile'):
    try:
        model = torch.compile(
            model, 
            mode='default',  # or 'reduce-overhead' for CPU
            backend='inductor'
        )
        print("Model compiled successfully")
    except Exception as e:
        print(f"Compilation failed: {e}")
```

## Dataset Loading Optimization

### 1. Streaming vs. In-Memory Loading

```python
# Intelligent dataset loading strategy
from macbook_optimization.dataset_management import DatasetManager

dataset_manager = DatasetManager()

# Analyze dataset requirements
analysis = dataset_manager.analyze_dataset_requirements(
    dataset_paths=['data/my-dataset'],
    split='train'
)

print(f"Dataset size: {analysis['total_size_mb']:.1f}MB")
print(f"Recommended strategy: {analysis['recommended_strategy']}")
print(f"Memory utilization: {analysis['memory_utilization_percent']:.1f}%")
```

**Configuration:**
```yaml
macbook_optimizations:
  # Dataset loading strategy
  streaming_threshold_mb: 200.0  # Stream if dataset > 200MB
  cache_threshold_mb: 100.0      # Cache if < 100MB
  chunk_size_mb: 50.0           # Chunk size for streaming
  
  # Memory management
  enable_caching: true
  auto_fallback_streaming: true
  force_streaming_mode: false   # Set true for large datasets
```

### 2. Data Loading Workers

```python
# Optimize worker count based on memory constraints
def calculate_optimal_workers(available_memory_gb, dataset_size_mb):
    if available_memory_gb <= 8:
        # Conservative for 8GB systems
        return min(2, max(1, available_memory_gb // 4))
    elif available_memory_gb <= 16:
        # Balanced for 16GB systems
        return min(4, max(2, available_memory_gb // 4))
    else:
        # Aggressive for 32GB+ systems
        return min(6, max(4, available_memory_gb // 6))
```

### 3. Preprocessing and Caching

```yaml
# Preprocessing optimization
dataset_preprocessing:
  # Cache preprocessed data
  enable_preprocessing_cache: true
  cache_directory: "cache/preprocessed"
  
  # Parallel preprocessing
  preprocessing_workers: 2
  
  # Memory-efficient preprocessing
  process_in_chunks: true
  chunk_size: 1000
```

## Model Architecture Tuning

### 1. Hardware-Appropriate Model Sizing

```python
# Adjust model size based on available resources
def optimize_model_config(base_config, hardware_specs):
    memory_gb = hardware_specs['memory']['total_gb']
    cpu_cores = hardware_specs['cpu']['cores']
    
    if memory_gb <= 8:
        # Conservative sizing for 8GB
        config = {
            'hidden_size': 256,
            'num_heads': 8,
            'L_layers': 2,
            'expansion': 2,
            'seq_len': 256
        }
    elif memory_gb <= 16:
        # Balanced sizing for 16GB
        config = {
            'hidden_size': 384,
            'num_heads': 12,
            'L_layers': 3,
            'expansion': 4,
            'seq_len': 512
        }
    else:
        # Larger sizing for 32GB+
        config = {
            'hidden_size': 512,
            'num_heads': 16,
            'L_layers': 4,
            'expansion': 4,
            'seq_len': 768
        }
    
    return {**base_config, **config}
```

### 2. Recursive Reasoning Optimization

```yaml
# Optimize recursive reasoning for CPU training
arch:
  # Conservative cycles for CPU training
  H_cycles: 2  # Reduce from 3-4 for speed
  L_cycles: 3  # Reduce from 4-6 for speed
  halt_max_steps: 6  # Limit maximum reasoning steps
  
  # Exploration vs. exploitation
  halt_exploration_prob: 0.1  # Conservative exploration
```

### 3. Sequence Length Optimization

```python
# Dynamic sequence length based on memory pressure
class AdaptiveSequenceLength:
    def __init__(self, base_seq_len=512, min_seq_len=128):
        self.base_seq_len = base_seq_len
        self.min_seq_len = min_seq_len
        self.current_seq_len = base_seq_len
        
    def adjust_for_memory_pressure(self, memory_usage_percent):
        if memory_usage_percent > 85:
            # Reduce sequence length by 25%
            self.current_seq_len = max(
                self.min_seq_len,
                int(self.current_seq_len * 0.75)
            )
        elif memory_usage_percent < 60:
            # Gradually increase back to base
            self.current_seq_len = min(
                self.base_seq_len,
                int(self.current_seq_len * 1.1)
            )
        
        return self.current_seq_len
```

## Thermal Management

### 1. Thermal Monitoring

```python
# Monitor CPU temperature and throttling
from macbook_optimization.resource_monitoring import ResourceMonitor

monitor = ResourceMonitor()
monitor.start_monitoring(interval=2.0)

# Check thermal state
thermal_state = monitor.get_thermal_state()
if thermal_state['is_throttling']:
    print(f"CPU throttling detected: {thermal_state['temperature']}°C")
    # Implement cooling strategy
```

### 2. Adaptive Training Strategy

```yaml
macbook_optimizations:
  # Thermal management
  enable_thermal_monitoring: true
  thermal_throttle_threshold: 85.0  # °C
  cooling_delay_seconds: 3.0
  
  # Adaptive batch sizing based on temperature
  thermal_batch_reduction: true
  thermal_batch_reduction_factor: 0.75
```

### 3. Cooling Strategies

```python
# Implement cooling breaks during training
class ThermalAwareTrainer:
    def __init__(self):
        self.thermal_monitor = ResourceMonitor()
        self.cooling_break_duration = 30  # seconds
        
    def should_take_cooling_break(self):
        thermal_state = self.thermal_monitor.get_thermal_state()
        return (
            thermal_state['temperature'] > 85 or
            thermal_state['is_throttling']
        )
    
    def cooling_break(self):
        print("Taking cooling break...")
        time.sleep(self.cooling_break_duration)
        
        # Reduce batch size temporarily
        self.reduce_batch_size_temporarily()
```

## Benchmarking and Monitoring

### 1. Performance Benchmarking

```python
# Comprehensive performance benchmark
from macbook_optimization.performance_reporting import PerformanceBenchmark

benchmark = PerformanceBenchmark()

# Run benchmark suite
results = benchmark.run_full_benchmark(
    model_config=config,
    dataset_path='data/benchmark',
    duration_minutes=10
)

print(f"Throughput: {results['samples_per_second']:.1f} samples/s")
print(f"Memory efficiency: {results['memory_efficiency']:.1f}%")
print(f"CPU utilization: {results['cpu_utilization']:.1f}%")
```

### 2. Real-time Monitoring

```python
# Real-time performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'samples_per_second': [],
            'memory_usage_mb': [],
            'cpu_usage_percent': [],
            'temperature_celsius': []
        }
    
    def log_metrics(self, step, metrics):
        # Log to wandb, tensorboard, or local file
        if wandb.run:
            wandb.log({
                'performance/samples_per_second': metrics['sps'],
                'performance/memory_usage_mb': metrics['memory_mb'],
                'performance/cpu_usage_percent': metrics['cpu_percent'],
                'performance/temperature': metrics['temperature']
            }, step=step)
```

### 3. Performance Analysis

```python
# Analyze performance bottlenecks
def analyze_performance_bottlenecks(training_metrics):
    analysis = {}
    
    # Memory bottleneck detection
    memory_usage = training_metrics['memory_usage_mb']
    if max(memory_usage) > 0.9 * total_memory_mb:
        analysis['memory_bottleneck'] = True
        analysis['recommendations'].append("Reduce batch size or enable streaming")
    
    # CPU bottleneck detection
    cpu_usage = training_metrics['cpu_usage_percent']
    if max(cpu_usage) < 70:
        analysis['cpu_underutilization'] = True
        analysis['recommendations'].append("Increase worker count or batch size")
    
    # Thermal throttling detection
    if 'thermal_throttling_events' in training_metrics:
        analysis['thermal_issues'] = len(training_metrics['thermal_throttling_events'])
        analysis['recommendations'].append("Improve cooling or reduce workload")
    
    return analysis
```

## Performance Troubleshooting

### 1. Slow Training Diagnosis

```python
# Diagnose slow training performance
def diagnose_slow_training(current_sps, expected_sps):
    if current_sps < expected_sps * 0.5:
        print("Severe performance degradation detected")
        
        # Check common issues
        memory_stats = get_memory_stats()
        if memory_stats['percent_used'] > 90:
            print("Issue: Memory pressure")
            print("Solution: Reduce batch size or enable streaming")
        
        cpu_stats = get_cpu_stats()
        if cpu_stats['utilization'] < 50:
            print("Issue: CPU underutilization")
            print("Solution: Increase workers or check MKL configuration")
        
        thermal_stats = get_thermal_stats()
        if thermal_stats['is_throttling']:
            print("Issue: Thermal throttling")
            print("Solution: Improve cooling or reduce workload")
```

### 2. Memory Leak Detection

```python
# Detect and fix memory leaks
class MemoryLeakDetector:
    def __init__(self):
        self.baseline_memory = None
        self.memory_history = []
        
    def check_for_leaks(self, step):
        current_memory = get_memory_usage_mb()
        self.memory_history.append((step, current_memory))
        
        if len(self.memory_history) > 100:
            # Check for consistent memory growth
            recent_memory = [m for s, m in self.memory_history[-50:]]
            if self.is_memory_growing(recent_memory):
                print("Memory leak detected!")
                self.suggest_fixes()
    
    def suggest_fixes(self):
        print("Potential fixes:")
        print("1. Enable more frequent garbage collection")
        print("2. Check for tensor accumulation in training loop")
        print("3. Clear optimizer state periodically")
```

### 3. Performance Regression Detection

```python
# Detect performance regressions
class PerformanceRegression:
    def __init__(self, baseline_sps):
        self.baseline_sps = baseline_sps
        self.recent_sps = []
        
    def check_regression(self, current_sps):
        self.recent_sps.append(current_sps)
        
        if len(self.recent_sps) > 20:
            avg_recent_sps = sum(self.recent_sps[-20:]) / 20
            
            if avg_recent_sps < self.baseline_sps * 0.8:
                print(f"Performance regression detected!")
                print(f"Baseline: {self.baseline_sps:.1f} sps")
                print(f"Current: {avg_recent_sps:.1f} sps")
                print(f"Degradation: {(1 - avg_recent_sps/self.baseline_sps)*100:.1f}%")
                
                return True
        return False
```

## Advanced Optimization Techniques

### 1. Mixed Precision Training (CPU)

```python
# CPU-compatible mixed precision
def setup_cpu_mixed_precision():
    # Use bfloat16 for forward pass, float32 for gradients
    torch.backends.cpu.enable_mixed_precision = True
    
    # Configure autocast for CPU
    return torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)
```

### 2. Model Parallelism

```python
# Simple model parallelism for large models
class ModelParallelTRM:
    def __init__(self, config):
        # Split model across available CPU cores
        self.num_partitions = min(4, config['L_layers'])
        self.layers_per_partition = config['L_layers'] // self.num_partitions
        
    def forward_parallel(self, x):
        # Implement pipeline parallelism
        for partition in self.partitions:
            x = partition(x)
        return x
```

### 3. Gradient Compression

```python
# Compress gradients to reduce memory usage
class GradientCompression:
    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio
        
    def compress_gradients(self, model):
        for param in model.parameters():
            if param.grad is not None:
                # Top-k sparsification
                grad_flat = param.grad.flatten()
                k = int(len(grad_flat) * self.compression_ratio)
                
                # Keep only top-k gradients
                _, indices = torch.topk(torch.abs(grad_flat), k)
                compressed_grad = torch.zeros_like(grad_flat)
                compressed_grad[indices] = grad_flat[indices]
                
                param.grad = compressed_grad.reshape(param.grad.shape)
```

---

This performance optimization guide provides comprehensive strategies for maximizing TRM training performance on MacBook hardware. Regular monitoring and iterative optimization based on your specific hardware and dataset characteristics will yield the best results.