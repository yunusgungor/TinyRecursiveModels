# MacBook TRM Training Troubleshooting Guide

This guide provides solutions to common issues encountered when training TRM models on MacBook hardware.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Memory Issues](#memory-issues)
3. [Performance Problems](#performance-problems)
4. [Configuration Errors](#configuration-errors)
5. [Dataset Loading Issues](#dataset-loading-issues)
6. [Thermal and Hardware Issues](#thermal-and-hardware-issues)
7. [Checkpoint and Resume Issues](#checkpoint-and-resume-issues)
8. [Environment and Dependencies](#environment-and-dependencies)
9. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```bash
# 1. Check system resources
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().total/1e9:.1f}GB available')
print(f'CPU: {psutil.cpu_count()} cores')
print(f'Disk: {psutil.disk_usage(\".\").free/1e9:.1f}GB free')
"

# 2. Test hardware detection
python -c "
from macbook_optimization.hardware_detection import HardwareDetector
detector = HardwareDetector()
print(detector.get_hardware_summary())
"

# 3. Validate configuration
python examples/macbook_training/scripts/train_small_dataset.py --dry-run --auto-detect

# 4. Check dependencies
python -c "
import torch
import yaml
import psutil
print('All dependencies available')
"
```

## Memory Issues

### Issue 1: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:75] posix_memalign
DefaultCPUAllocator: can't allocate memory: you tried to allocate X bytes
```

**Immediate Solutions:**

1. **Reduce Batch Size**
   ```yaml
   # In your config file
   global_batch_size: 16  # Reduce from 32 or 64
   training:
     batch_size: 4  # Physical batch size
     gradient_accumulation_steps: 4  # Maintain effective batch size
   ```

2. **Enable Streaming Mode**
   ```yaml
   macbook_optimizations:
     force_streaming_mode: true
     streaming_threshold_mb: 50.0  # Lower threshold
   ```

3. **Reduce Model Size**
   ```yaml
   arch:
     hidden_size: 256  # Reduce from 384 or 512
     seq_len: 256      # Reduce from 512
     L_layers: 2       # Reduce from 3 or 4
   ```

**Diagnostic Commands:**
```python
# Check current memory usage
from macbook_optimization.memory_management import MemoryManager
memory_manager = MemoryManager()
stats = memory_manager.monitor_memory_usage()
print(f"Memory usage: {stats.percent_used:.1f}% ({stats.used_mb:.0f}MB)")

# Calculate safe batch size
safe_batch_size = memory_manager.calculate_optimal_batch_size(
    model_size_mb=100,  # Estimate your model size
    available_memory_mb=stats.available_mb
)
print(f"Recommended batch size: {safe_batch_size}")
```

### Issue 2: Memory Leaks During Training

**Symptoms:**
- Memory usage continuously increases during training
- Training slows down over time
- Eventually leads to OOM

**Solutions:**

1. **Enable Aggressive Garbage Collection**
   ```yaml
   macbook_optimizations:
     auto_garbage_collection: true
     force_gc_interval: 25  # Force GC every 25 batches
   ```

2. **Clear Optimizer State Periodically**
   ```python
   # In training loop
   if step % 1000 == 0:
       for optimizer in optimizers:
           optimizer.zero_grad(set_to_none=True)
       torch.cuda.empty_cache() if torch.cuda.is_available() else None
   ```

3. **Monitor Memory Growth**
   ```python
   # Add to training loop
   if step % 100 == 0:
       current_memory = psutil.virtual_memory().used / 1e6
       print(f"Step {step}: Memory usage {current_memory:.0f}MB")
   ```

### Issue 3: Memory Fragmentation

**Symptoms:**
- OOM errors despite apparently sufficient memory
- Inconsistent memory allocation failures

**Solutions:**

1. **Use Memory Mapping**
   ```yaml
   macbook_optimizations:
     use_memory_mapping: true
     enable_caching: false  # Disable caching to reduce fragmentation
   ```

2. **Restart Training Periodically**
   ```bash
   # Use checkpoint resumption for long training runs
   python train_script.py --resume checkpoints/latest.pt
   ```

## Performance Problems

### Issue 1: Very Slow Training Speed

**Symptoms:**
- Training speed < 5 samples/second
- CPU usage appears low despite training

**Diagnostic Steps:**

1. **Check CPU Configuration**
   ```python
   from macbook_optimization.cpu_optimization import CPUOptimizer
   optimizer = CPUOptimizer()
   config = optimizer.get_current_config()
   print(f"Torch threads: {config.torch_threads}")
   print(f"MKL enabled: {config.mkl_enabled}")
   ```

2. **Monitor Resource Usage**
   ```python
   from macbook_optimization.resource_monitoring import ResourceMonitor
   monitor = ResourceMonitor()
   monitor.start_monitoring()
   # ... after some training
   stats = monitor.get_cpu_statistics()
   print(f"CPU utilization: {stats['average_usage']:.1f}%")
   ```

**Solutions:**

1. **Enable CPU Optimizations**
   ```yaml
   macbook_optimizations:
     use_mkl: true
     optimize_tensor_operations: true
     enable_vectorization: true
     torch_threads: 4  # Match your CPU core count
   ```

2. **Adjust DataLoader Settings**
   ```yaml
   training:
     num_workers: 2  # Try different values: 0, 1, 2, 4
     pin_memory: true  # If you have sufficient memory
     prefetch_factor: 2
   ```

3. **Install Intel MKL**
   ```bash
   # Using conda
   conda install mkl mkl-include
   
   # Using pip
   pip install mkl
   
   # Verify installation
   python -c "import torch; print(torch.backends.mkl.is_available())"
   ```

### Issue 2: Training Speed Degrades Over Time

**Symptoms:**
- Training starts fast but slows down significantly
- MacBook becomes very hot

**Solutions:**

1. **Enable Thermal Monitoring**
   ```yaml
   macbook_optimizations:
     enable_thermal_monitoring: true
     thermal_throttle_threshold: 80.0  # Lower threshold
     cooling_delay_seconds: 5.0
   ```

2. **Implement Cooling Breaks**
   ```python
   # Add to training loop
   if step % 500 == 0:  # Every 500 steps
       thermal_state = resource_monitor.get_thermal_state()
       if thermal_state['temperature'] > 85:
           print("Taking cooling break...")
           time.sleep(30)  # 30-second break
   ```

3. **Reduce Workload Intensity**
   ```yaml
   # Reduce batch size when hot
   training:
     thermal_batch_reduction: true
     thermal_batch_reduction_factor: 0.75
   ```

### Issue 3: Inconsistent Performance

**Symptoms:**
- Training speed varies significantly between runs
- Performance depends on other running applications

**Solutions:**

1. **Set Process Priority**
   ```bash
   # Run training with higher priority
   nice -n -10 python train_script.py
   ```

2. **Close Other Applications**
   ```bash
   # Check memory usage by other apps
   ps aux --sort=-%mem | head -10
   
   # Close unnecessary applications before training
   ```

3. **Use Consistent Environment**
   ```bash
   # Set consistent environment variables
   export MKL_NUM_THREADS=4
   export OMP_NUM_THREADS=4
   export VECLIB_MAXIMUM_THREADS=4
   ```

## Configuration Errors

### Issue 1: Configuration Validation Failures

**Symptoms:**
```
ValidationError: Configuration incompatible with MacBook hardware
Invalid batch size for available memory
```

**Solutions:**

1. **Use Auto-Detection**
   ```bash
   python train_script.py --auto-detect
   ```

2. **Validate Configuration Manually**
   ```python
   from macbook_optimization.config_validation import ConfigurationValidator
   from macbook_optimization.hardware_detection import HardwareDetector
   
   validator = ConfigurationValidator(HardwareDetector())
   result = validator.validate_configuration(config, auto_correct=True)
   
   if not result.is_valid:
       print("Validation errors:")
       for error in result.validation_errors:
           print(f"  - {error}")
       
       if result.corrected_config:
           print("Using auto-corrected configuration")
           config = result.corrected_config
   ```

3. **Check Hardware Compatibility**
   ```python
   # Verify hardware meets minimum requirements
   hardware_summary = HardwareDetector().get_hardware_summary()
   
   if hardware_summary['memory']['total_gb'] < 8:
       print("Warning: Less than 8GB RAM detected")
       print("Consider using smaller model configurations")
   
   if hardware_summary['cpu']['cores'] < 4:
       print("Warning: Less than 4 CPU cores detected")
       print("Training may be very slow")
   ```

### Issue 2: Architecture Configuration Errors

**Symptoms:**
```
KeyError: 'hidden_size'
AttributeError: 'TRM' object has no attribute 'num_heads'
```

**Solutions:**

1. **Use Complete Architecture Configuration**
   ```yaml
   arch:
     # Required parameters
     hidden_size: 256
     num_heads: 8
     L_layers: 2
     expansion: 4
     
     # TRM-specific parameters
     H_cycles: 2
     L_cycles: 3
     halt_max_steps: 6
     
     # Sequence parameters
     seq_len: 256
     batch_size: 8
     
     # Training parameters
     forward_dtype: "float32"
     embed_scale: true
   ```

2. **Validate Architecture Compatibility**
   ```python
   # Check if architecture parameters are compatible
   def validate_architecture(arch_config):
       required_params = ['hidden_size', 'num_heads', 'L_layers', 'seq_len']
       for param in required_params:
           if param not in arch_config:
               raise ValueError(f"Missing required parameter: {param}")
       
       # Check parameter relationships
       if arch_config['hidden_size'] % arch_config['num_heads'] != 0:
           raise ValueError("hidden_size must be divisible by num_heads")
   ```

## Dataset Loading Issues

### Issue 1: Dataset Not Found

**Symptoms:**
```
FileNotFoundError: Dataset path does not exist: data/my-dataset
OSError: No such file or directory
```

**Solutions:**

1. **Verify Dataset Path**
   ```python
   import os
   dataset_path = "data/my-dataset"
   
   if not os.path.exists(dataset_path):
       print(f"Dataset path does not exist: {dataset_path}")
       print("Available paths:")
       for item in os.listdir("data" if os.path.exists("data") else "."):
           print(f"  - {item}")
   ```

2. **Create Symbolic Links**
   ```bash
   # If dataset is in different location
   ln -s /path/to/actual/dataset data/my-dataset
   ```

3. **Update Configuration**
   ```yaml
   # Use absolute paths if needed
   data_paths: ['/absolute/path/to/dataset']
   ```

### Issue 2: Dataset Loading Timeout

**Symptoms:**
- Training hangs during dataset loading
- No progress for extended periods

**Solutions:**

1. **Reduce Worker Count**
   ```yaml
   training:
     num_workers: 0  # Try single-threaded loading first
   ```

2. **Enable Streaming Mode**
   ```yaml
   macbook_optimizations:
     force_streaming_mode: true
     chunk_size_mb: 25.0  # Smaller chunks
   ```

3. **Check Dataset Integrity**
   ```python
   # Validate dataset files
   def check_dataset_integrity(dataset_path):
       corrupted_files = []
       for root, dirs, files in os.walk(dataset_path):
           for file in files:
               filepath = os.path.join(root, file)
               try:
                   with open(filepath, 'r') as f:
                       f.read(1)  # Try to read first byte
               except Exception as e:
                   corrupted_files.append((filepath, str(e)))
       
       if corrupted_files:
           print(f"Found {len(corrupted_files)} corrupted files")
           for filepath, error in corrupted_files[:5]:  # Show first 5
               print(f"  {filepath}: {error}")
   ```

### Issue 3: Dataset Format Errors

**Symptoms:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**

1. **Validate Dataset Format**
   ```python
   import json
   
   def validate_json_dataset(dataset_path):
       invalid_files = []
       for root, dirs, files in os.walk(dataset_path):
           for file in files:
               if file.endswith('.json'):
                   filepath = os.path.join(root, file)
                   try:
                       with open(filepath, 'r', encoding='utf-8') as f:
                           json.load(f)
                   except Exception as e:
                       invalid_files.append((filepath, str(e)))
       
       return invalid_files
   ```

2. **Handle Encoding Issues**
   ```python
   # Try different encodings
   def safe_read_file(filepath):
       encodings = ['utf-8', 'latin-1', 'cp1252']
       for encoding in encodings:
           try:
               with open(filepath, 'r', encoding=encoding) as f:
                   return f.read()
           except UnicodeDecodeError:
               continue
       raise ValueError(f"Could not decode file: {filepath}")
   ```

## Thermal and Hardware Issues

### Issue 1: CPU Thermal Throttling

**Symptoms:**
- Training speed decreases significantly over time
- MacBook becomes very hot (>85°C)
- Fan runs at maximum speed

**Solutions:**

1. **Monitor Temperature**
   ```python
   # Check CPU temperature (requires additional tools)
   import subprocess
   
   def get_cpu_temperature():
       try:
           # Using powermetrics (requires sudo)
           result = subprocess.run(
               ['sudo', 'powermetrics', '-n', '1', '-s', 'cpu_power'],
               capture_output=True, text=True, timeout=10
           )
           # Parse temperature from output
           for line in result.stdout.split('\n'):
               if 'CPU die temperature' in line:
                   temp = float(line.split(':')[1].strip().replace('C', ''))
                   return temp
       except:
           pass
       return None
   ```

2. **Implement Thermal Management**
   ```yaml
   macbook_optimizations:
     enable_thermal_monitoring: true
     thermal_throttle_threshold: 80.0  # Conservative threshold
     cooling_delay_seconds: 10.0
     thermal_batch_reduction: true
   ```

3. **Physical Cooling Solutions**
   - Use laptop cooling pad
   - Ensure proper ventilation
   - Clean dust from vents
   - Train in cooler environment

### Issue 2: Hardware Detection Failures

**Symptoms:**
```
ImportError: No module named 'psutil'
RuntimeError: Hardware detection failed
```

**Solutions:**

1. **Install Missing Dependencies**
   ```bash
   pip install psutil
   ```

2. **Manual Hardware Configuration**
   ```yaml
   # Override auto-detection
   hardware_profile: "macbook_8gb"  # or macbook_16gb, macbook_32gb
   expected_ram_gb: 8
   expected_cpu_cores: 4
   ```

3. **Fallback Configuration**
   ```python
   # Use safe defaults if detection fails
   def get_safe_hardware_config():
       return {
           'memory_gb': 8,  # Conservative assumption
           'cpu_cores': 4,
           'cpu_threads': 4,
           'has_mkl': False  # Conservative assumption
       }
   ```

## Checkpoint and Resume Issues

### Issue 1: Checkpoint Loading Failures

**Symptoms:**
```
FileNotFoundError: Checkpoint file not found
RuntimeError: Error loading checkpoint: size mismatch
```

**Solutions:**

1. **Verify Checkpoint Path**
   ```python
   import os
   checkpoint_path = "checkpoints/latest.pt"
   
   if not os.path.exists(checkpoint_path):
       print(f"Checkpoint not found: {checkpoint_path}")
       # List available checkpoints
       checkpoint_dir = os.path.dirname(checkpoint_path)
       if os.path.exists(checkpoint_dir):
           checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
           print(f"Available checkpoints: {checkpoints}")
   ```

2. **Handle Model Size Mismatches**
   ```python
   # Load checkpoint with size checking
   def safe_load_checkpoint(model, checkpoint_path):
       checkpoint = torch.load(checkpoint_path, map_location='cpu')
       model_state = checkpoint['model_state_dict']
       
       # Check for size mismatches
       model_dict = model.state_dict()
       mismatched_keys = []
       
       for key in model_state:
           if key in model_dict:
               if model_state[key].shape != model_dict[key].shape:
                   mismatched_keys.append(key)
       
       if mismatched_keys:
           print(f"Size mismatches found: {mismatched_keys}")
           print("Loading compatible parameters only")
           
           # Load only compatible parameters
           compatible_state = {
               k: v for k, v in model_state.items()
               if k in model_dict and v.shape == model_dict[k].shape
           }
           model.load_state_dict(compatible_state, strict=False)
       else:
           model.load_state_dict(model_state)
   ```

### Issue 2: Checkpoint Corruption

**Symptoms:**
```
RuntimeError: PytorchStreamReader failed reading zip archive
EOFError: Ran out of input
```

**Solutions:**

1. **Validate Checkpoint Integrity**
   ```python
   def validate_checkpoint(checkpoint_path):
       try:
           checkpoint = torch.load(checkpoint_path, map_location='cpu')
           required_keys = ['model_state_dict', 'step', 'loss']
           
           for key in required_keys:
               if key not in checkpoint:
                   print(f"Missing key in checkpoint: {key}")
                   return False
           
           print("Checkpoint validation passed")
           return True
       except Exception as e:
           print(f"Checkpoint validation failed: {e}")
           return False
   ```

2. **Use Backup Checkpoints**
   ```python
   # Try loading from multiple checkpoint candidates
   def load_best_available_checkpoint(checkpoint_dir):
       checkpoint_files = sorted([
           f for f in os.listdir(checkpoint_dir) 
           if f.endswith('.pt')
       ], reverse=True)  # Try newest first
       
       for checkpoint_file in checkpoint_files:
           checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
           if validate_checkpoint(checkpoint_path):
               print(f"Loading checkpoint: {checkpoint_file}")
               return torch.load(checkpoint_path, map_location='cpu')
       
       print("No valid checkpoints found")
       return None
   ```

## Environment and Dependencies

### Issue 1: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'macbook_optimization'
ImportError: cannot import name 'HardwareDetector'
```

**Solutions:**

1. **Check Python Path**
   ```python
   import sys
   print("Python path:")
   for path in sys.path:
       print(f"  {path}")
   
   # Add project root to path if needed
   import os
   project_root = os.path.dirname(os.path.abspath(__file__))
   if project_root not in sys.path:
       sys.path.insert(0, project_root)
   ```

2. **Verify Installation**
   ```bash
   # Check if modules are in the right place
   find . -name "macbook_optimization" -type d
   
   # Check Python can import
   python -c "from macbook_optimization import hardware_detection; print('Import successful')"
   ```

### Issue 2: Version Conflicts

**Symptoms:**
```
AttributeError: module 'torch' has no attribute 'compile'
ImportError: This version of PyTorch is not compatible
```

**Solutions:**

1. **Check Versions**
   ```python
   import torch
   import yaml
   import psutil
   
   print(f"PyTorch version: {torch.__version__}")
   print(f"Python version: {sys.version}")
   print(f"psutil version: {psutil.__version__}")
   ```

2. **Update Dependencies**
   ```bash
   # Update to compatible versions
   pip install torch>=1.12.0
   pip install pyyaml>=6.0
   pip install psutil>=5.8.0
   ```

## Advanced Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Enable specific module debugging
logging.getLogger('macbook_optimization').setLevel(logging.DEBUG)
```

### Memory Profiling

```python
# Profile memory usage
import tracemalloc

tracemalloc.start()

# ... run training code ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f}MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f}MB")

tracemalloc.stop()
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile training performance
profiler = cProfile.Profile()
profiler.enable()

# ... run training code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### System Resource Monitoring

```bash
# Monitor system resources during training
# Terminal 1: Start training
python train_script.py

# Terminal 2: Monitor resources
watch -n 2 'ps aux | grep python | head -5; echo ""; free -h; echo ""; df -h .'
```

---

If you encounter issues not covered in this guide, please:

1. Check the GitHub repository issues
2. Run the diagnostic checklist
3. Collect system information and error logs
4. Create a minimal reproduction case
5. Submit a detailed issue report

Remember to include:
- MacBook model and specifications
- macOS version
- Python and PyTorch versions
- Complete error messages
- Configuration files used
- Steps to reproduce the issue