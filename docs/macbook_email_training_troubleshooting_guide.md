# MacBook Email Classification Training Troubleshooting Guide

This guide helps you diagnose and resolve common issues when training email classification models on MacBook hardware.

**Requirements: 5.4**

## Table of Contents

1. [Memory Issues](#memory-issues)
2. [Performance Problems](#performance-problems)
3. [Training Failures](#training-failures)
4. [Configuration Errors](#configuration-errors)
5. [Dataset Issues](#dataset-issues)
6. [Model Quality Problems](#model-quality-problems)
7. [Hardware-Specific Issues](#hardware-specific-issues)
8. [Debugging Tools](#debugging-tools)

## Memory Issues

### Out of Memory (OOM) Errors

**Symptoms:**
- Training crashes with "RuntimeError: CUDA out of memory" or similar
- System becomes unresponsive during training
- Memory usage exceeds available RAM

**Solutions:**

1. **Reduce Batch Size**
   ```bash
   python train_email_classifier_macbook.py \
     --train \
     --batch-size 2 \
     --gradient-accumulation-steps 16
   ```

2. **Enable Dynamic Batch Sizing**
   ```yaml
   hardware:
     dynamic_batch_sizing: true
     memory_pressure_threshold: 0.80
   ```

3. **Reduce Model Complexity**
   ```yaml
   model:
     hidden_size: 256  # Reduce from 512
     num_layers: 2     # Reduce from 3
     max_sequence_length: 256  # Reduce from 512
   ```

4. **Enable Memory Monitoring**
   ```yaml
   hardware:
     enable_memory_monitoring: true
     garbage_collection_frequency: 100
   ```

### Memory Leaks

**Symptoms:**
- Memory usage continuously increases during training
- Training slows down over time
- System performance degrades

**Solutions:**

1. **Enable Checkpoint Memory Cleanup**
   ```yaml
   hardware:
     checkpoint_memory_cleanup: true
   ```

2. **Increase Garbage Collection Frequency**
   ```yaml
   hardware:
     garbage_collection_frequency: 50  # More frequent cleanup
   ```

3. **Use Streaming Data Loading**
   ```yaml
   data:
     streaming_mode: true
     chunk_size: 500
     cache_preprocessed: false
   ```

### Memory Pressure Warnings

**Symptoms:**
- Warnings about high memory usage
- Training becomes unstable
- Frequent garbage collection

**Solutions:**

1. **Lower Memory Limit**
   ```yaml
   hardware:
     memory_limit_mb: 5000  # Reduce from default
   ```

2. **Enable Graceful Degradation**
   ```yaml
   error_handling:
     graceful_degradation: true
     auto_reduce_batch_size: true
   ```

## Performance Problems

### Slow Training Speed

**Symptoms:**
- Training takes much longer than expected
- Low samples per second
- High CPU usage but slow progress

**Solutions:**

1. **Optimize CPU Usage**
   ```yaml
   hardware:
     use_cpu_optimization: true
     cpu_threads: 4  # Adjust based on your MacBook
     num_workers: 2
   ```

2. **Reduce Model Complexity**
   ```yaml
   model:
     hidden_size: 256
     num_layers: 2
   ```

3. **Use Efficient Data Loading**
   ```yaml
   data:
     prefetch_factor: 2
     shuffle_buffer_size: 1000
   ```

4. **Disable Expensive Features**
   ```yaml
   email:
     use_hierarchical_attention: false  # For 8GB MacBooks
   ```

### Thermal Throttling

**Symptoms:**
- Training speed decreases over time
- MacBook becomes very hot
- Fan noise increases significantly

**Solutions:**

1. **Enable Thermal Monitoring**
   ```yaml
   hardware:
     thermal_monitoring: true
   ```

2. **Reduce Training Intensity**
   ```yaml
   training:
     batch_size: 2  # Smaller batches
   hardware:
     cpu_threads: 2  # Fewer threads
   ```

3. **Take Breaks Between Training Sessions**
   ```bash
   # Train in shorter sessions
   python train_email_classifier_macbook.py --max-steps 2000
   # Wait for cooling, then resume
   python train_email_classifier_macbook.py --resume-from-checkpoint
   ```

### High CPU Usage

**Symptoms:**
- CPU usage consistently above 90%
- System becomes unresponsive
- Other applications slow down

**Solutions:**

1. **Limit CPU Threads**
   ```yaml
   hardware:
     cpu_threads: 4  # Reduce from default
     num_workers: 1
   ```

2. **Use Lower Priority Process**
   ```bash
   nice -n 10 python train_email_classifier_macbook.py --train
   ```

## Training Failures

### Training Crashes

**Symptoms:**
- Training stops unexpectedly
- Error messages in logs
- Incomplete model files

**Common Causes and Solutions:**

1. **Memory Issues** (see Memory Issues section)

2. **Corrupted Data**
   ```bash
   # Validate dataset
   python dataset/build_email_dataset.py --validate --input_file emails.json
   ```

3. **Configuration Errors**
   ```bash
   # Validate configuration
   python examples/macbook_training/config_validator.py --config config.yaml
   ```

4. **Disk Space Issues**
   ```bash
   # Check available space
   df -h
   # Clean up old checkpoints
   rm -rf training_output/checkpoints/old_*
   ```

### Training Hangs

**Symptoms:**
- Training stops progressing
- No error messages
- Process appears frozen

**Solutions:**

1. **Check for Deadlocks**
   ```yaml
   data:
     dataloader_num_workers: 0  # Disable multiprocessing
   ```

2. **Enable Timeout Handling**
   ```yaml
   error_handling:
     max_retries: 3
     retry_delay: 5
   ```

3. **Monitor Resource Usage**
   ```bash
   # In another terminal
   top -pid $(pgrep -f train_email_classifier)
   ```

### Checkpoint Loading Failures

**Symptoms:**
- Cannot resume training from checkpoint
- "Checkpoint not found" errors
- Model architecture mismatch

**Solutions:**

1. **Verify Checkpoint Path**
   ```bash
   ls -la training_output/checkpoints/
   ```

2. **Check Model Compatibility**
   ```bash
   python -c "
   import torch
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')
   print(checkpoint.keys())
   "
   ```

3. **Reset Training if Necessary**
   ```bash
   rm -rf training_output/checkpoints/
   python train_email_classifier_macbook.py --train  # Start fresh
   ```

## Configuration Errors

### Invalid Configuration Values

**Symptoms:**
- "Configuration validation failed" errors
- Training won't start
- Parameter out of range warnings

**Solutions:**

1. **Validate Configuration**
   ```bash
   python examples/macbook_training/config_validator.py \
     --config config.yaml \
     --memory-gb 16
   ```

2. **Use Hardware-Appropriate Presets**
   ```bash
   # For 8GB MacBook
   cp examples/macbook_training/configs/macbook_8gb/email_classification.yaml config.yaml
   ```

3. **Generate Recommended Configuration**
   ```bash
   python examples/macbook_training/config_validator.py \
     --recommend \
     --memory-gb 16 \
     --output recommended_config.yaml
   ```

### Missing Required Fields

**Symptoms:**
- "Missing required field" errors
- Configuration parsing failures

**Solutions:**

1. **Check Required Fields**
   ```yaml
   model:
     vocab_size: 5000      # Required
     hidden_size: 512      # Required
     num_layers: 2         # Required
   training:
     batch_size: 4         # Required
     learning_rate: 1e-4   # Required
     max_steps: 5000       # Required
   ```

2. **Use Complete Configuration Template**
   ```bash
   cp examples/macbook_training/configs/macbook_16gb/email_classification.yaml config.yaml
   ```

## Dataset Issues

### Dataset Format Errors

**Symptoms:**
- "Invalid dataset format" errors
- JSON parsing failures
- Missing required fields

**Solutions:**

1. **Validate Dataset Format**
   ```bash
   python -c "
   import json
   with open('emails.json', 'r') as f:
       data = json.load(f)
   print(f'Loaded {len(data)} emails')
   print('Required fields:', data[0].keys())
   "
   ```

2. **Check Required Fields**
   Each email must have:
   - `id`: Unique identifier
   - `subject`: Email subject
   - `body`: Email body
   - `category`: One of 10 predefined categories
   - `sender`: Sender email address

3. **Fix Common Format Issues**
   ```python
   # Fix encoding issues
   import json
   with open('emails.json', 'r', encoding='utf-8') as f:
       data = json.load(f)
   
   # Ensure all required fields exist
   for email in data:
       if 'category' not in email:
           email['category'] = 'Other'
       if 'language' not in email:
           email['language'] = 'en'
   ```

### Category Imbalance

**Symptoms:**
- Poor performance on some categories
- Warnings about category distribution
- Uneven accuracy across categories

**Solutions:**

1. **Check Category Distribution**
   ```python
   import json
   from collections import Counter
   
   with open('emails.json', 'r') as f:
       data = json.load(f)
   
   categories = [email['category'] for email in data]
   print(Counter(categories))
   ```

2. **Balance Dataset**
   ```python
   # Oversample minority classes or undersample majority classes
   from sklearn.utils import resample
   
   # Implementation depends on your specific imbalance
   ```

3. **Use Class Weighting**
   ```yaml
   training:
     use_class_weighting: true
     class_weights: "balanced"
   ```

### Small Dataset Issues

**Symptoms:**
- Overfitting quickly
- Poor generalization
- High validation loss

**Solutions:**

1. **Increase Data Augmentation**
   ```yaml
   email:
     email_augmentation_prob: 0.5  # Increase from default 0.3
   ```

2. **Use Stronger Regularization**
   ```yaml
   training:
     weight_decay: 0.05  # Increase from default 0.01
     dropout_rate: 0.3
   ```

3. **Reduce Model Complexity**
   ```yaml
   model:
     hidden_size: 256  # Smaller model
     num_layers: 2
   ```

## Model Quality Problems

### Low Accuracy

**Symptoms:**
- Final accuracy below target (95%)
- Poor performance on validation set
- High loss values

**Solutions:**

1. **Increase Training Steps**
   ```bash
   python train_email_classifier_macbook.py \
     --max-steps 10000  # Increase from default
   ```

2. **Use Multi-Phase Training**
   ```bash
   python train_email_classifier_macbook.py \
     --strategy multi_phase
   ```

3. **Tune Hyperparameters**
   ```bash
   python train_email_classifier_macbook.py \
     --optimize \
     --num-trials 20
   ```

4. **Check Data Quality**
   - Ensure correct category labels
   - Remove duplicate emails
   - Fix encoding issues

### Overfitting

**Symptoms:**
- High training accuracy, low validation accuracy
- Validation loss increases while training loss decreases
- Poor performance on new data

**Solutions:**

1. **Increase Regularization**
   ```yaml
   training:
     weight_decay: 0.05
     dropout_rate: 0.3
   ```

2. **Early Stopping**
   ```yaml
   targets:
     early_stopping_patience: 3  # Stop if no improvement
   ```

3. **More Data Augmentation**
   ```yaml
   email:
     email_augmentation_prob: 0.4
   ```

### Slow Convergence

**Symptoms:**
- Loss decreases very slowly
- Many training steps needed
- Training takes too long

**Solutions:**

1. **Adjust Learning Rate**
   ```yaml
   training:
     learning_rate: 2e-4  # Increase from 1e-4
   ```

2. **Use Learning Rate Scheduling**
   ```yaml
   training:
     use_lr_scheduler: true
     scheduler_type: "cosine"
   ```

3. **Increase Batch Size**
   ```yaml
   training:
     batch_size: 8  # If memory allows
     gradient_accumulation_steps: 4
   ```

## Hardware-Specific Issues

### Intel MacBook Issues

**Symptoms:**
- Slower training than expected
- High CPU usage
- Thermal throttling

**Solutions:**

1. **Optimize for Intel Architecture**
   ```yaml
   hardware:
     use_cpu_optimization: true
     cpu_threads: 4  # Adjust based on your model
   ```

2. **Use Conservative Settings**
   ```yaml
   training:
     batch_size: 2  # Start small
   model:
     hidden_size: 256  # Conservative size
   ```

### M1/M2 MacBook Compatibility

**Note:** This training system is optimized for Intel MacBooks. For M1/M2 MacBooks:

1. **Use Rosetta 2**
   ```bash
   arch -x86_64 python train_email_classifier_macbook.py
   ```

2. **Install x86_64 Python**
   ```bash
   arch -x86_64 brew install python@3.9
   ```

### Memory Architecture Differences

**8GB vs 16GB vs 32GB MacBooks have different optimal configurations:**

**8GB MacBook:**
- Use smallest possible batch sizes
- Enable all memory optimizations
- Consider streaming data loading

**16GB MacBook:**
- Balanced configuration
- Can use moderate model sizes
- Good performance/memory trade-off

**32GB+ MacBook:**
- Can use larger models and batches
- Enable advanced features
- Consider ensemble training

## Debugging Tools

### Memory Monitoring

```python
# Add to your training script
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

# Call periodically during training
```

### Performance Profiling

```bash
# Profile training performance
python -m cProfile -o profile.stats train_email_classifier_macbook.py --train

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

### Log Analysis

```bash
# Check training logs
tail -f training_output/logs/training.log

# Search for errors
grep -i error training_output/logs/training.log

# Monitor memory usage in logs
grep -i memory training_output/logs/training.log
```

### Configuration Debugging

```bash
# Validate configuration thoroughly
python examples/macbook_training/config_validator.py \
  --config config.yaml \
  --memory-gb $(sysctl hw.memsize | awk '{print $2/1024/1024/1024}') \
  --cpu-cores $(sysctl hw.ncpu | awk '{print $2}')
```

### Dataset Debugging

```python
# Analyze dataset statistics
import json
from collections import Counter

with open('emails.json', 'r') as f:
    data = json.load(f)

print(f"Total emails: {len(data)}")
print(f"Categories: {Counter(email['category'] for email in data)}")
print(f"Languages: {Counter(email.get('language', 'unknown') for email in data)}")

# Check for missing fields
for i, email in enumerate(data):
    required_fields = ['id', 'subject', 'body', 'category']
    missing = [field for field in required_fields if field not in email]
    if missing:
        print(f"Email {i}: missing fields {missing}")
```

## Getting Additional Help

If you continue to experience issues:

1. **Check System Requirements**: Ensure your MacBook meets minimum requirements
2. **Update Dependencies**: Make sure all packages are up to date
3. **Review Logs**: Check detailed error messages in log files
4. **Test with Sample Data**: Try training with the provided sample dataset
5. **Reduce Complexity**: Start with the simplest possible configuration
6. **Monitor Resources**: Use Activity Monitor to check system resource usage

## Common Error Messages and Solutions

| Error Message | Cause | Solution |
|---------------|-------|----------|
| "CUDA out of memory" | Batch size too large | Reduce batch size, enable dynamic sizing |
| "Configuration validation failed" | Invalid config values | Use config validator, check hardware limits |
| "Dataset not found" | Wrong dataset path | Check path, validate dataset structure |
| "Checkpoint loading failed" | Incompatible checkpoint | Delete checkpoints, start fresh training |
| "Memory pressure detected" | High memory usage | Enable memory monitoring, reduce model size |
| "Training hangs" | Deadlock or resource issue | Disable multiprocessing, check system resources |

Remember: Start with conservative settings and gradually increase complexity as you verify stability on your specific MacBook configuration.