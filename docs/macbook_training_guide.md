# MacBook TRM Training Guide

This comprehensive guide covers training Tiny Recursive Models (TRM) on MacBook hardware with limited resources. The guide includes setup instructions, optimization strategies, and troubleshooting for common issues.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation and Setup](#installation-and-setup)
3. [Quick Start](#quick-start)
4. [Configuration Guide](#configuration-guide)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [FAQ](#faq)

## System Requirements

### Minimum Requirements
- **MacBook Model**: 2017 or later with Intel processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for checkpoints and datasets
- **macOS**: 10.15 (Catalina) or later
- **Python**: 3.8 or later

### Recommended Requirements
- **MacBook Model**: 2019 or later with Intel i5/i7 processor
- **RAM**: 16GB or more
- **Storage**: 50GB+ free space for large datasets
- **macOS**: 11.0 (Big Sur) or later

### Supported Hardware Configurations

| MacBook Model | RAM | CPU Cores | Expected Performance |
|---------------|-----|-----------|---------------------|
| MacBook Air 2017-2020 | 8GB | 2-4 | Basic training, small datasets |
| MacBook Pro 13" 2017-2020 | 8-16GB | 4 | Good performance, medium datasets |
| MacBook Pro 15"/16" 2017-2020 | 16-32GB | 6-8 | Excellent performance, large datasets |

## Performance Expectations

Training performance varies significantly based on MacBook model, dataset size, and configuration. Here are realistic expectations:

| MacBook Model | RAM | Small Dataset | Medium Dataset | Large Dataset |
|---------------|-----|---------------|----------------|---------------|
| MacBook Air 8GB | 8GB | 2-5 samples/s | 1-3 samples/s | Streaming mode |
| MacBook Pro 13" 16GB | 16GB| 5-15 samples/s | 3-8 samples/s  | 2-5 samples/s |
| MacBook Pro 16" 32GB | 32GB| 10-25 samples/s | 8-15 samples/s  | 5-12 samples/s |

*Performance varies based on CPU model, dataset complexity, and thermal conditions*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd trm-training
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install MacBook-specific optimizations
pip install psutil  # For hardware detection
```

### 3. Verify Installation

```bash
# Test hardware detection
python -c "from macbook_optimization.hardware_detection import HardwareDetector; print(HardwareDetector().get_hardware_summary())"

# Test basic functionality
python examples/macbook_training/scripts/train_small_dataset.py --dry-run --auto-detect
```

### 4. Optional: Install Intel MKL (Recommended)

```bash
# For Intel CPU optimization
conda install mkl mkl-include
# or
pip install mkl
```

## Quick Start

### 1. Prepare Your Dataset

Organize your dataset in one of these formats:

```
data/
├── my-dataset/
│   ├── train/
│   │   ├── puzzle1.json
│   │   ├── puzzle2.json
│   │   └── ...
│   └── test/
│       ├── test1.json
│       └── ...
```

### 2. Choose Configuration Template

Based on your MacBook model and dataset size:

```bash
# For 8GB MacBook with small dataset
cp examples/macbook_training/configs/macbook_8gb/small_dataset.yaml my_config.yaml

# For 16GB MacBook with medium dataset
cp examples/macbook_training/configs/macbook_16gb/medium_dataset.yaml my_config.yaml
```

### 3. Start Training

```bash
# Auto-detect hardware and start training
python examples/macbook_training/scripts/train_small_dataset.py \\
    --auto-detect \\
    --data-path data/my-dataset \\
    --output-dir outputs/my_experiment

# Or use specific configuration
python pretrain_macbook.py --config my_config.yaml
```

### 4. Monitor Progress

Training will display real-time metrics:

```
Step 150/10000 (1.5%) | Loss: 2.3456 | Speed: 12.3 samples/s | Memory: 65.2% (3200MB) | ETA: 45.2min
```

## Configuration Guide

### Hardware-Specific Settings

#### 8GB MacBook Configuration

```yaml
# Memory-constrained settings
global_batch_size: 32
training:
  memory_limit_mb: 4000
  gradient_accumulation_steps: 4
  num_workers: 2
  pin_memory: false
  max_sequence_length: 256

# Model size adjustments
arch:
  hidden_size: 256
  num_heads: 8
  L_layers: 2
  expansion: 2
```

#### 16GB MacBook Configuration

```yaml
# Balanced settings
global_batch_size: 64
training:
  memory_limit_mb: 8000
  gradient_accumulation_steps: 4
  num_workers: 4
  pin_memory: true
  max_sequence_length: 512

# Larger model possible
arch:
  hidden_size: 384
  num_heads: 12
  L_layers: 3
  expansion: 4
```

### Dataset-Specific Settings

#### Small Datasets (< 100MB)

```yaml
macbook_optimizations:
  streaming_threshold_mb: 50.0
  enable_caching: true
  auto_fallback_streaming: false
```

#### Medium Datasets (100MB - 1GB)

```yaml
macbook_optimizations:
  streaming_threshold_mb: 200.0
  enable_caching: true
  auto_fallback_streaming: true
```

#### Large Datasets (> 1GB)

```yaml
macbook_optimizations:
  streaming_threshold_mb: 500.0
  enable_caching: false
  force_streaming_mode: true
```

### CPU Optimization Settings

```yaml
macbook_optimizations:
  # CPU optimization
  use_mkl: true
  optimize_tensor_operations: true
  enable_vectorization: true
  torch_threads: 4  # Adjust based on CPU cores
  
  # Thermal management
  thermal_throttle_threshold: 85.0
  cooling_delay_seconds: 2.0
```

## Performance Optimization

### Memory Optimization

1. **Batch Size Tuning**
   ```bash
   # Start with auto-detection
   python train_script.py --auto-detect
   
   # Fine-tune manually
   python train_script.py --memory-limit 6000
   ```

2. **Gradient Accumulation**
   - Use gradient accumulation to achieve larger effective batch sizes
   - Recommended: 4-8 accumulation steps for MacBook

3. **Sequence Length**
   - Reduce sequence length for memory-constrained systems
   - 256 tokens for 8GB, 512 tokens for 16GB+

### CPU Optimization

1. **Thread Configuration**
   ```python
   # Automatic optimization
   from macbook_optimization.cpu_optimization import CPUOptimizer
   optimizer = CPUOptimizer()
   optimizer.configure_all()
   ```

2. **Intel MKL Integration**
   - Install Intel MKL for optimized BLAS operations
   - 2-3x speedup on Intel CPUs

3. **Model Compilation**
   ```yaml
   # Enable PyTorch 2.0 compilation (experimental)
   compile_model: true
   ```

### Dataset Loading Optimization

1. **Streaming Mode**
   - Automatically enabled for large datasets
   - Reduces memory usage at cost of some speed

2. **Caching Strategy**
   ```yaml
   macbook_optimizations:
     enable_caching: true
     cache_threshold_mb: 100.0
     chunk_size_mb: 50.0
   ```

3. **Worker Processes**
   - 8GB MacBook: 1-2 workers
   - 16GB MacBook: 2-4 workers
   - More workers = faster loading but more memory usage

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] posix_memalign. DefaultCPUAllocator: can't allocate memory
```

**Solutions:**
1. Reduce batch size:
   ```yaml
   global_batch_size: 16  # Reduce from 32
   ```

2. Enable streaming mode:
   ```yaml
   macbook_optimizations:
     force_streaming_mode: true
   ```

3. Reduce sequence length:
   ```yaml
   arch:
     seq_len: 256  # Reduce from 512
   ```

4. Close other applications to free memory

#### 2. Slow Training Speed

**Symptoms:**
- Training speed < 5 samples/second
- High CPU usage but low utilization

**Solutions:**
1. Enable CPU optimizations:
   ```yaml
   macbook_optimizations:
     use_mkl: true
     optimize_tensor_operations: true
   ```

2. Adjust worker count:
   ```yaml
   training:
     num_workers: 2  # Try different values
   ```

3. Check thermal throttling:
   ```bash
   # Monitor CPU temperature
   sudo powermetrics -n 1 -s cpu_power
   ```

#### 3. Thermal Throttling

**Symptoms:**
- Training speed decreases over time
- MacBook gets very hot
- CPU frequency drops

**Solutions:**
1. Enable thermal monitoring:
   ```yaml
   macbook_optimizations:
     enable_thermal_monitoring: true
     thermal_throttle_threshold: 80.0
     cooling_delay_seconds: 3.0
   ```

2. Reduce batch size or add cooling breaks
3. Use external cooling (laptop stand, fan)
4. Train in cooler environment

#### 4. Disk Space Issues

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**
1. Reduce checkpoint frequency:
   ```yaml
   training:
     checkpoint_interval: 1000  # Increase interval
     max_checkpoints_to_keep: 2  # Keep fewer checkpoints
   ```

2. Clean up old checkpoints:
   ```bash
   python -c "from macbook_optimization.checkpoint_management import CheckpointManager; CheckpointManager.cleanup_old_checkpoints('checkpoints/')"
   ```

#### 5. Configuration Errors

**Symptoms:**
```
ValidationError: Invalid configuration for MacBook hardware
```

**Solutions:**
1. Use auto-detection:
   ```bash
   python train_script.py --auto-detect
   ```

2. Validate configuration:
   ```bash
   python train_script.py --dry-run
   ```

3. Check hardware compatibility:
   ```python
   from macbook_optimization.config_validation import ConfigurationValidator
   validator = ConfigurationValidator()
   result = validator.validate_configuration(config)
   print(result.validation_messages)
   ```

### Performance Debugging

#### 1. Memory Profiling

```python
# Enable detailed memory monitoring
from macbook_optimization.memory_management import MemoryManager
memory_manager = MemoryManager()
memory_manager.enable_detailed_monitoring()

# Check memory usage during training
stats = memory_manager.get_memory_summary()
print(f"Peak memory: {stats['tracking']['peak_mb']:.0f}MB")
```

#### 2. CPU Profiling

```python
# Monitor CPU utilization
from macbook_optimization.resource_monitoring import ResourceMonitor
monitor = ResourceMonitor()
monitor.start_monitoring()

# After training
cpu_stats = monitor.get_cpu_statistics()
print(f"Average CPU usage: {cpu_stats['average_usage']:.1f}%")
```

#### 3. Dataset Loading Analysis

```python
# Analyze dataset loading performance
from macbook_optimization.dataset_management import DatasetManager
manager = DatasetManager()
metrics = manager.get_loading_metrics()

for metric in metrics:
    print(f"Dataset: {metric.total_size_mb:.1f}MB, "
          f"Strategy: {metric.loading_strategy}, "
          f"Speed: {metric.loading_speed_mb_per_sec:.1f}MB/s")
```

### Getting Help

1. **Check logs**: Training logs contain detailed error information
2. **Hardware detection**: Run hardware detection to verify system specs
3. **Configuration validation**: Use dry-run mode to validate settings
4. **Community support**: Check GitHub issues for similar problems

## Advanced Usage

### Custom Model Architectures

```python
# Create custom TRM variant for MacBook
from models.recursive_reasoning.trm import TRM

class MacBookTRM(TRM):
    def __init__(self, config):
        # Optimize for CPU training
        config['forward_dtype'] = 'float32'
        config['enable_checkpointing'] = True
        super().__init__(config)
```

### Hyperparameter Tuning

```yaml
# Enable hyperparameter search
hyperparameter_search:
  enabled: true
  strategy: "random"
  num_trials: 10
  
  parameters:
    learning_rate: [1e-5, 5e-4]
    batch_size: [16, 32, 64]
    hidden_size: [256, 384, 512]
```

### Multi-Stage Training

```python
# Stage 1: Small model, fast iteration
stage1_config = load_config('configs/macbook_8gb/small_dataset.yaml')
stage1_config['arch']['hidden_size'] = 128
trainer1 = MacBookTRMTrainer(stage1_config)
trainer1.train(dataset_size)

# Stage 2: Larger model, fine-tuning
stage2_config = load_config('configs/macbook_16gb/medium_dataset.yaml')
stage2_config['load_checkpoint'] = True
trainer2 = MacBookTRMTrainer(stage2_config)
trainer2.train(dataset_size)
```

### Integration with Weights & Biases

```yaml
# Enable W&B logging
use_wandb: true
wandb_project: "macbook-trm-experiments"
wandb_tags: ["macbook", "cpu-training", "trm"]

# Custom metrics
wandb_log_interval: 50
wandb_log_gradients: false  # Disabled for performance
```

## FAQ

### Q: Can I train on MacBook Air?
A: Yes, but performance will be limited. Use 8GB configurations and small datasets. Expect longer training times and potential thermal throttling.

### Q: How long does training take?
A: Depends on dataset size and MacBook model:
- Small dataset (< 100MB): 1-4 hours
- Medium dataset (100MB-1GB): 4-12 hours  
- Large dataset (> 1GB): 12+ hours

### Q: Can I use GPU acceleration?
A: This guide focuses on CPU training. For GPU training on newer MacBooks with Apple Silicon, see the Apple Silicon training guide.

### Q: How much disk space do I need?
A: Plan for:
- Dataset: Original size
- Checkpoints: 2-5GB per checkpoint
- Logs and outputs: 1-2GB
- Total: Dataset size + 10-20GB

### Q: Can I pause and resume training?
A: Yes, training automatically saves checkpoints. Use `--resume checkpoint_path` to continue.

### Q: What if my MacBook overheats?
A: Enable thermal monitoring, reduce batch size, use external cooling, or train in shorter sessions with breaks.

### Q: How do I know if my configuration is optimal?
A: Use the dry-run mode and auto-detection features. Monitor memory usage and training speed during initial runs.

### Q: Can I train multiple models simultaneously?
A: Not recommended on MacBook due to memory constraints. Train models sequentially instead.

### Q: How do I optimize for my specific dataset?
A: Start with the closest template configuration, then adjust batch size, sequence length, and memory limits based on your dataset characteristics.

---

For additional support, check the GitHub repository issues or create a new issue with your specific problem and system configuration.