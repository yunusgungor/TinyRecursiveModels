# MacBook Email Classification Training Performance Optimization Guide

This guide provides advanced techniques for optimizing email classification training performance on different MacBook configurations.

**Requirements: 5.4**

## Table of Contents

1. [Performance Fundamentals](#performance-fundamentals)
2. [Hardware-Specific Optimizations](#hardware-specific-optimizations)
3. [Memory Optimization Strategies](#memory-optimization-strategies)
4. [CPU Optimization Techniques](#cpu-optimization-techniques)
5. [Training Strategy Optimization](#training-strategy-optimization)
6. [Data Pipeline Optimization](#data-pipeline-optimization)
7. [Model Architecture Optimization](#model-architecture-optimization)
8. [Advanced Optimization Techniques](#advanced-optimization-techniques)
9. [Performance Monitoring](#performance-monitoring)
10. [Benchmarking and Profiling](#benchmarking-and-profiling)

## Performance Fundamentals

### Key Performance Metrics

| Metric | Target (8GB) | Target (16GB) | Target (32GB+) | Description |
|--------|--------------|---------------|----------------|-------------|
| Samples/sec | 10-20 | 20-40 | 40-80 | Training throughput |
| Memory Usage | <6GB | <12GB | <24GB | Peak memory consumption |
| CPU Usage | 70-85% | 80-90% | 85-95% | CPU utilization |
| Training Time | 2-4 hours | 1-3 hours | 0.5-2 hours | Time to 95% accuracy |
| Convergence Steps | 8000-12000 | 5000-8000 | 3000-5000 | Steps to target accuracy |

### Performance Bottlenecks

Common bottlenecks and their solutions:

1. **Memory Bandwidth**: Use smaller batches with gradient accumulation
2. **CPU Computation**: Optimize tensor operations and reduce model complexity
3. **Data Loading**: Implement efficient data pipelines and caching
4. **Thermal Throttling**: Monitor temperature and adjust workload intensity

## Hardware-Specific Optimizations

### 8GB MacBook Optimization

**Configuration Profile:**
```yaml
# Optimized for memory-constrained environments
model:
  hidden_size: 256
  num_layers: 2
  max_sequence_length: 256

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  use_mixed_precision: false

hardware:
  memory_limit_mb: 5500
  dynamic_batch_sizing: true
  garbage_collection_frequency: 50
  cpu_threads: 2
```

**Key Strategies:**
- Minimize memory footprint at all costs
- Use streaming data loading
- Frequent garbage collection
- Conservative CPU threading
- Enable all memory optimizations

**Performance Tweaks:**
```python
# Memory-efficient model initialization
import torch
torch.backends.cudnn.benchmark = False  # Reduce memory overhead
torch.backends.cudnn.deterministic = True

# Aggressive memory cleanup
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

### 16GB MacBook Optimization

**Configuration Profile:**
```yaml
# Balanced performance and memory usage
model:
  hidden_size: 384
  num_layers: 2
  max_sequence_length: 512

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  use_mixed_precision: false

hardware:
  memory_limit_mb: 12000
  dynamic_batch_sizing: false
  cpu_threads: 4
```

**Key Strategies:**
- Balance between speed and memory efficiency
- Use moderate model complexity
- Enable hierarchical attention
- Optimize data loading pipeline

**Performance Tweaks:**
```python
# Optimized tensor operations
torch.set_num_threads(4)  # Match CPU cores
torch.set_num_interop_threads(2)

# Efficient data loading
dataloader_kwargs = {
    'num_workers': 2,
    'pin_memory': False,  # Not needed for CPU-only training
    'prefetch_factor': 2
}
```

### 32GB+ MacBook Optimization

**Configuration Profile:**
```yaml
# High-performance configuration
model:
  hidden_size: 512
  num_layers: 3
  max_sequence_length: 768

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  use_mixed_precision: false

hardware:
  memory_limit_mb: 24000
  cpu_threads: 8
  enable_advanced_features: true
```

**Key Strategies:**
- Maximize model complexity and batch sizes
- Enable advanced training features
- Use ensemble methods
- Implement sophisticated attention mechanisms

**Performance Tweaks:**
```python
# High-performance settings
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# Enable optimized BLAS operations
import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
```

## Memory Optimization Strategies

### Memory-Efficient Model Design

**1. Gradient Checkpointing**
```python
# Trade computation for memory
class MemoryEfficientTRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_checkpoint = config.gradient_checkpointing
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)
```

**2. Dynamic Batch Sizing**
```python
class DynamicBatchSizer:
    def __init__(self, initial_batch_size, memory_limit):
        self.batch_size = initial_batch_size
        self.memory_limit = memory_limit
        self.memory_history = []
    
    def adjust_batch_size(self, current_memory):
        if current_memory > self.memory_limit * 0.9:
            self.batch_size = max(1, self.batch_size // 2)
        elif current_memory < self.memory_limit * 0.7:
            self.batch_size = min(16, self.batch_size * 2)
        return self.batch_size
```

**3. Memory-Mapped Datasets**
```python
import mmap
import json

class MemoryMappedDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'rb') as f:
            self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __getitem__(self, idx):
        # Load only the required item, not entire dataset
        pass
```

### Memory Monitoring and Management

**Real-time Memory Monitoring:**
```python
import psutil
import threading
import time

class MemoryMonitor:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.monitoring = False
        self.callbacks = []
    
    def start_monitoring(self):
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _monitor_loop(self):
        while self.monitoring:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.threshold:
                for callback in self.callbacks:
                    callback(memory_percent)
            time.sleep(1)
```

## CPU Optimization Techniques

### Intel MKL Optimization

**Configuration for Intel MacBooks:**
```bash
# Environment variables for optimal performance
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

**PyTorch CPU Optimization:**
```python
import torch

# Optimize for Intel CPUs
torch.set_num_threads(4)  # Match your CPU cores
torch.set_num_interop_threads(2)

# Enable Intel MKL optimizations
torch.backends.mkl.enabled = True
torch.backends.mkldnn.enabled = True
```

### Efficient Tensor Operations

**Vectorized Operations:**
```python
# Efficient attention computation
def efficient_attention(query, key, value, mask=None):
    # Use batch matrix multiplication
    scores = torch.bmm(query, key.transpose(-2, -1))
    
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    
    # Use in-place operations where possible
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.bmm(attention_weights, value)
```

**Memory-Efficient Activations:**
```python
# Use in-place operations
def efficient_gelu(x):
    return x.mul_(torch.sigmoid(1.702 * x))

# Avoid creating intermediate tensors
def efficient_layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps) + bias
```

### Parallel Processing Optimization

**Optimal Worker Configuration:**
```python
def get_optimal_workers(memory_gb, cpu_cores):
    """Determine optimal number of workers based on hardware."""
    if memory_gb <= 8:
        return min(1, cpu_cores // 4)  # Conservative for 8GB
    elif memory_gb <= 16:
        return min(2, cpu_cores // 2)  # Moderate for 16GB
    else:
        return min(4, cpu_cores)       # Aggressive for 32GB+

# Usage
num_workers = get_optimal_workers(16, 8)
dataloader = DataLoader(dataset, num_workers=num_workers)
```

## Training Strategy Optimization

### Adaptive Learning Rate Scheduling

**Performance-Based Scheduling:**
```python
class PerformanceScheduler:
    def __init__(self, optimizer, patience=3, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.wait = 0
    
    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
                self.wait = 0
```

### Curriculum Learning Implementation

**Difficulty-Based Curriculum:**
```python
class EmailCurriculumSampler:
    def __init__(self, dataset, difficulty_fn):
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        self.current_difficulty = 0.3  # Start with easy samples
    
    def get_batch_indices(self, batch_size):
        # Sample based on current difficulty threshold
        difficulties = [self.difficulty_fn(item) for item in self.dataset]
        valid_indices = [i for i, d in enumerate(difficulties) 
                        if d <= self.current_difficulty]
        return random.sample(valid_indices, min(batch_size, len(valid_indices)))
    
    def update_difficulty(self, accuracy):
        if accuracy > 0.8:  # Increase difficulty when performing well
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
```

### Multi-Phase Training Optimization

**Optimized Phase Transitions:**
```python
class OptimizedPhaseTrainer:
    def __init__(self):
        self.phases = [
            {
                'name': 'warmup',
                'steps': 1000,
                'lr_factor': 0.5,
                'batch_size_factor': 0.5,
                'model_complexity': 0.7
            },
            {
                'name': 'main',
                'steps': 4000,
                'lr_factor': 1.0,
                'batch_size_factor': 1.0,
                'model_complexity': 1.0
            },
            {
                'name': 'fine_tune',
                'steps': 1000,
                'lr_factor': 0.1,
                'batch_size_factor': 0.75,
                'model_complexity': 1.0
            }
        ]
    
    def transition_to_phase(self, phase_config, model, optimizer):
        # Adjust learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= phase_config['lr_factor']
        
        # Adjust model complexity if needed
        if phase_config['model_complexity'] < 1.0:
            self._reduce_model_complexity(model, phase_config['model_complexity'])
```

## Data Pipeline Optimization

### Efficient Data Loading

**Optimized DataLoader Configuration:**
```python
class OptimizedEmailDataLoader:
    def __init__(self, dataset, batch_size, memory_gb):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Optimize based on available memory
        if memory_gb <= 8:
            self.num_workers = 0  # Single-threaded for memory efficiency
            self.prefetch_factor = 1
            self.pin_memory = False
        elif memory_gb <= 16:
            self.num_workers = 2
            self.prefetch_factor = 2
            self.pin_memory = False
        else:
            self.num_workers = 4
            self.prefetch_factor = 4
            self.pin_memory = False
    
    def create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            shuffle=True
        )
```

### Caching Strategies

**Intelligent Caching:**
```python
class SmartCache:
    def __init__(self, max_memory_mb=1000):
        self.cache = {}
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self.access_count = {}
    
    def get(self, key, compute_fn):
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        
        value = compute_fn()
        self._maybe_cache(key, value)
        return value
    
    def _maybe_cache(self, key, value):
        value_size = sys.getsizeof(value)
        if self.current_memory + value_size > self.max_memory:
            self._evict_lru()
        
        self.cache[key] = value
        self.current_memory += value_size
        self.access_count[key] = 1
```

### Data Preprocessing Optimization

**Batch Preprocessing:**
```python
class BatchPreprocessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def preprocess_batch(self, emails):
        # Vectorized tokenization
        subjects = [email['subject'] for email in emails]
        bodies = [email['body'] for email in emails]
        
        # Batch tokenization is more efficient
        subject_tokens = self.tokenizer.batch_encode(subjects, max_length=64)
        body_tokens = self.tokenizer.batch_encode(bodies, max_length=self.max_length-64)
        
        # Combine efficiently
        combined_tokens = []
        for subj, body in zip(subject_tokens, body_tokens):
            combined = subj + [self.tokenizer.sep_token_id] + body
            combined_tokens.append(combined[:self.max_length])
        
        return combined_tokens
```

## Model Architecture Optimization

### Efficient Attention Mechanisms

**Sparse Attention for Long Sequences:**
```python
class SparseAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, sparsity_factor=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sparsity_factor = sparsity_factor
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create sparse attention pattern
        sparse_mask = self._create_sparse_mask(seq_len)
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply sparse attention
        attention_output = self._sparse_attention(q, k, v, sparse_mask)
        return attention_output.view(batch_size, seq_len, self.hidden_size)
```

### Parameter Sharing

**Efficient Parameter Sharing:**
```python
class SharedParameterTRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Share parameters across layers for memory efficiency
        self.shared_attention = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads
        )
        self.shared_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        for _ in range(self.config.num_layers):
            # Shared attention layer
            attn_output, _ = self.shared_attention(x, x, x)
            x = self.layer_norm1(x + attn_output)
            
            # Shared FFN layer
            ffn_output = self.shared_ffn(x)
            x = self.layer_norm2(x + ffn_output)
        
        return x
```

### Model Compression Techniques

**Dynamic Model Pruning:**
```python
class DynamicPruner:
    def __init__(self, model, sparsity_target=0.3):
        self.model = model
        self.sparsity_target = sparsity_target
    
    def prune_model(self, importance_scores):
        """Prune model based on importance scores."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                importance = importance_scores.get(name, torch.ones_like(weights))
                
                # Calculate threshold for target sparsity
                flat_importance = importance.flatten()
                threshold = torch.quantile(flat_importance, self.sparsity_target)
                
                # Apply pruning mask
                mask = importance > threshold
                weights.mul_(mask)
```

## Advanced Optimization Techniques

### Gradient Accumulation Optimization

**Smart Gradient Accumulation:**
```python
class SmartGradientAccumulator:
    def __init__(self, target_batch_size, max_memory_mb):
        self.target_batch_size = target_batch_size
        self.max_memory_mb = max_memory_mb
        self.current_batch_size = 1
        self.accumulation_steps = target_batch_size
    
    def update_batch_size(self, current_memory_mb):
        """Dynamically adjust batch size based on memory usage."""
        memory_ratio = current_memory_mb / self.max_memory_mb
        
        if memory_ratio > 0.9:  # High memory usage
            self.current_batch_size = max(1, self.current_batch_size // 2)
        elif memory_ratio < 0.6:  # Low memory usage
            self.current_batch_size = min(
                self.target_batch_size, 
                self.current_batch_size * 2
            )
        
        self.accumulation_steps = self.target_batch_size // self.current_batch_size
        return self.current_batch_size, self.accumulation_steps
```

### Mixed Precision Training (Experimental)

**Note:** Mixed precision may not provide benefits on CPU-only training, but can be useful for future GPU support.

```python
class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def training_step(self, batch):
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch)
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(batch)
            loss = outputs['loss']
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
```

### Ensemble Training Optimization

**Efficient Ensemble Training:**
```python
class EfficientEnsemble:
    def __init__(self, base_config, num_models=3):
        self.num_models = num_models
        self.models = []
        
        for i in range(num_models):
            # Create diverse models with different initializations
            config = self._create_diverse_config(base_config, i)
            model = EmailTRM(config)
            self.models.append(model)
    
    def _create_diverse_config(self, base_config, model_id):
        """Create diverse configurations for ensemble members."""
        config = copy.deepcopy(base_config)
        
        # Vary model architecture slightly
        if model_id == 0:
            config.hidden_size = int(config.hidden_size * 0.9)
        elif model_id == 1:
            config.num_layers = max(1, config.num_layers - 1)
        elif model_id == 2:
            config.dropout_rate = config.dropout_rate * 1.5
        
        return config
    
    def train_ensemble(self, dataloader):
        """Train ensemble models efficiently."""
        for epoch in range(self.num_epochs):
            for batch in dataloader:
                # Train models in sequence to manage memory
                for model in self.models:
                    loss = self._train_single_model(model, batch)
```

## Performance Monitoring

### Real-time Performance Tracking

**Comprehensive Performance Monitor:**
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'samples_per_second': [],
            'memory_usage_mb': [],
            'cpu_usage_percent': [],
            'loss_values': [],
            'accuracy_values': [],
            'step_times': []
        }
        self.start_time = time.time()
    
    def log_step(self, step, loss, accuracy, batch_size):
        current_time = time.time()
        step_time = current_time - getattr(self, 'last_step_time', current_time)
        
        # Calculate samples per second
        samples_per_sec = batch_size / step_time if step_time > 0 else 0
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()
        
        # Store metrics
        self.metrics['samples_per_second'].append(samples_per_sec)
        self.metrics['memory_usage_mb'].append(memory_usage)
        self.metrics['cpu_usage_percent'].append(cpu_usage)
        self.metrics['loss_values'].append(loss)
        self.metrics['accuracy_values'].append(accuracy)
        self.metrics['step_times'].append(step_time)
        
        self.last_step_time = current_time
        
        # Print performance summary every 100 steps
        if step % 100 == 0:
            self._print_performance_summary(step)
    
    def _print_performance_summary(self, step):
        recent_samples_per_sec = np.mean(self.metrics['samples_per_second'][-10:])
        recent_memory = np.mean(self.metrics['memory_usage_mb'][-10:])
        recent_cpu = np.mean(self.metrics['cpu_usage_percent'][-10:])
        
        print(f"Step {step}: "
              f"{recent_samples_per_sec:.1f} samples/sec, "
              f"{recent_memory:.0f}MB memory, "
              f"{recent_cpu:.1f}% CPU")
```

### Automated Performance Optimization

**Auto-tuning System:**
```python
class AutoTuner:
    def __init__(self, initial_config):
        self.config = initial_config
        self.performance_history = []
        self.best_config = initial_config
        self.best_performance = 0
    
    def tune_batch_size(self, current_performance, memory_usage):
        """Automatically tune batch size based on performance and memory."""
        if memory_usage > 0.9:  # High memory usage
            self.config['batch_size'] = max(1, self.config['batch_size'] // 2)
            self.config['gradient_accumulation_steps'] *= 2
        elif memory_usage < 0.6 and current_performance > self.best_performance:
            # Try increasing batch size if performance is good and memory is low
            self.config['batch_size'] = min(16, self.config['batch_size'] * 2)
            self.config['gradient_accumulation_steps'] = max(1, 
                self.config['gradient_accumulation_steps'] // 2)
        
        return self.config
    
    def tune_learning_rate(self, loss_trend):
        """Adjust learning rate based on loss trend."""
        if len(loss_trend) >= 5:
            recent_trend = np.polyfit(range(5), loss_trend[-5:], 1)[0]
            
            if recent_trend > 0:  # Loss increasing
                self.config['learning_rate'] *= 0.8
            elif recent_trend < -0.01:  # Loss decreasing rapidly
                self.config['learning_rate'] *= 1.1
        
        return self.config
```

## Benchmarking and Profiling

### Performance Benchmarking

**Comprehensive Benchmark Suite:**
```python
class MacBookBenchmark:
    def __init__(self):
        self.benchmark_configs = {
            '8gb': {
                'batch_size': 2,
                'hidden_size': 256,
                'sequence_length': 256
            },
            '16gb': {
                'batch_size': 4,
                'hidden_size': 384,
                'sequence_length': 512
            },
            '32gb': {
                'batch_size': 8,
                'hidden_size': 512,
                'sequence_length': 768
            }
        }
    
    def run_benchmark(self, config_name, num_steps=100):
        """Run performance benchmark for specific configuration."""
        config = self.benchmark_configs[config_name]
        
        # Create model and data
        model = self._create_benchmark_model(config)
        dataloader = self._create_benchmark_data(config)
        
        # Warm up
        self._warmup(model, dataloader, 10)
        
        # Benchmark
        start_time = time.time()
        memory_usage = []
        step_times = []
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            step_start = time.time()
            
            # Forward pass
            outputs = model(batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Memory monitoring
            memory_usage.append(psutil.virtual_memory().used / (1024 * 1024))
        
        total_time = time.time() - start_time
        
        return {
            'config': config_name,
            'total_time': total_time,
            'avg_step_time': np.mean(step_times),
            'samples_per_second': config['batch_size'] * num_steps / total_time,
            'peak_memory_mb': max(memory_usage),
            'avg_memory_mb': np.mean(memory_usage)
        }
```

### Profiling Tools

**Custom Profiler:**
```python
class TrainingProfiler:
    def __init__(self):
        self.timers = {}
        self.memory_snapshots = {}
    
    def start_timer(self, name):
        self.timers[name] = time.time()
    
    def end_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            return elapsed
        return 0
    
    def memory_snapshot(self, name):
        self.memory_snapshots[name] = psutil.virtual_memory().used / (1024 * 1024)
    
    def profile_training_step(self, model, batch):
        """Profile a single training step."""
        self.start_timer('total_step')
        self.memory_snapshot('step_start')
        
        # Forward pass
        self.start_timer('forward')
        outputs = model(batch)
        forward_time = self.end_timer('forward')
        
        # Backward pass
        self.start_timer('backward')
        loss = outputs['loss']
        loss.backward()
        backward_time = self.end_timer('backward')
        
        total_time = self.end_timer('total_step')
        self.memory_snapshot('step_end')
        
        return {
            'total_time': total_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'memory_delta': (self.memory_snapshots['step_end'] - 
                           self.memory_snapshots['step_start'])
        }
```

## Performance Optimization Checklist

### Pre-Training Optimization

- [ ] Hardware detection and configuration validation
- [ ] Memory limit configuration based on available RAM
- [ ] CPU thread optimization for your MacBook model
- [ ] Data pipeline optimization (workers, prefetch, caching)
- [ ] Model architecture tuning for hardware constraints

### During Training Optimization

- [ ] Real-time memory monitoring
- [ ] Dynamic batch size adjustment
- [ ] Performance metric tracking
- [ ] Thermal monitoring and throttling prevention
- [ ] Gradient accumulation optimization

### Post-Training Analysis

- [ ] Performance benchmark comparison
- [ ] Memory usage analysis
- [ ] Training efficiency evaluation
- [ ] Model quality vs. performance trade-off analysis
- [ ] Configuration optimization recommendations

## Conclusion

Optimizing email classification training on MacBook requires a holistic approach considering hardware constraints, model architecture, training strategies, and data pipeline efficiency. The key is to:

1. **Start Conservative**: Begin with hardware-appropriate configurations
2. **Monitor Continuously**: Track performance and resource usage in real-time
3. **Optimize Iteratively**: Make incremental improvements based on bottleneck analysis
4. **Balance Trade-offs**: Consider accuracy vs. speed vs. memory usage
5. **Validate Improvements**: Benchmark changes to ensure actual performance gains

Remember that optimal configurations vary significantly between MacBook models, so always validate performance improvements on your specific hardware configuration.