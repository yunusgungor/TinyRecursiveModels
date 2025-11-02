# MacBook TRM Training Examples

This directory contains example training configurations and scripts optimized for different MacBook models and use cases.

## Directory Structure

- `configs/` - Configuration templates for different MacBook models and dataset sizes
- `scripts/` - Example training scripts with MacBook optimizations
- `notebooks/` - Jupyter notebooks for interactive training and analysis
- `troubleshooting/` - Common issues and solutions

## Quick Start

1. Choose a configuration template based on your MacBook model:
   - `configs/macbook_8gb/` - For MacBooks with 8GB RAM
   - `configs/macbook_16gb/` - For MacBooks with 16GB RAM
   - `configs/macbook_32gb/` - For MacBooks with 32GB RAM

2. Select a dataset size configuration:
   - `small_dataset.yaml` - For datasets < 100MB
   - `medium_dataset.yaml` - For datasets 100MB - 1GB
   - `large_dataset.yaml` - For datasets > 1GB

3. Run training:
   ```bash
   python examples/macbook_training/scripts/train_small_dataset.py --config configs/macbook_8gb/small_dataset.yaml
   ```

## Configuration Templates

Each configuration template is optimized for specific hardware constraints and includes:
- Memory-appropriate batch sizes
- CPU-optimized settings
- Dataset management strategies
- Checkpoint management
- Progress monitoring

## Performance Expectations

| MacBook Model | RAM | Small Dataset | Medium Dataset | Large Dataset |
|---------------|-----|---------------|----------------|---------------|
| 8GB RAM       | 8GB | 2-5 min/epoch | 5-15 min/epoch | Streaming mode |
| 16GB RAM      | 16GB| 1-3 min/epoch | 3-8 min/epoch  | 8-20 min/epoch |
| 32GB RAM      | 32GB| 1-2 min/epoch | 2-5 min/epoch  | 5-12 min/epoch |

*Performance varies based on CPU model and dataset complexity*