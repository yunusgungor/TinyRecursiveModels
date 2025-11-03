# MacBook Email Classification Training Setup Guide

This guide provides step-by-step instructions for setting up and running email classification training on MacBook hardware using the optimized training pipeline.

**Requirements: 5.4**

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Hardware Detection](#hardware-detection)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration](#configuration)
6. [Training Execution](#training-execution)
7. [Model Evaluation](#model-evaluation)
8. [Production Deployment](#production-deployment)

## Prerequisites

### System Requirements

- **MacBook Models**: Intel-based MacBook with macOS 10.15 or later
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: At least 10GB free space for datasets and models
- **Python**: Python 3.8 or later

### Supported MacBook Configurations

| MacBook Model | Memory | Recommended Batch Size | Max Model Size | Training Time (est.) |
|---------------|--------|----------------------|----------------|---------------------|
| 8GB MacBook   | 8GB    | 2-4                  | 256 hidden     | 2-4 hours          |
| 16GB MacBook  | 16GB   | 4-8                  | 384 hidden     | 1-3 hours          |
| 32GB+ MacBook | 32GB+  | 8-16                 | 512+ hidden    | 0.5-2 hours        |

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd email-classification-macbook-training
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python train_email_classifier_macbook.py --detect-hardware
```

This should display your MacBook's hardware specifications and recommended configuration.

## Hardware Detection

The training system automatically detects your MacBook's hardware capabilities and recommends optimal settings.

### Manual Hardware Detection

```bash
python train_email_classifier_macbook.py --detect-hardware
```

**Example Output:**
```
Hardware detected:
  CPU cores: 8
  Total memory: 16.0GB
  Available memory: 12.3GB
  Platform: macOS 12.6

Recommended configuration: macbook_16gb
  Batch size: 4
  Gradient accumulation: 8
  Max steps: 8000
  Learning rate: 0.0001
  Memory limit: 12000MB
```

### Configuration Validation

Validate your configuration against hardware constraints:

```bash
python examples/macbook_training/config_validator.py \
  --config examples/macbook_training/configs/macbook_16gb/email_classification.yaml \
  --memory-gb 16 \
  --cpu-cores 8
```

## Dataset Preparation

### Dataset Format

Email datasets should be in JSON format with the following structure:

```json
[
  {
    "id": "email_001",
    "subject": "Meeting Tomorrow",
    "body": "Don't forget about our meeting tomorrow at 2 PM.",
    "sender": "manager@company.com",
    "recipient": "user@example.com",
    "category": "Work",
    "language": "en",
    "timestamp": "2024-01-15T14:30:00Z"
  }
]
```

### Required Categories

The system supports 10 email categories:
- Newsletter
- Work
- Personal
- Spam
- Promotional
- Social
- Finance
- Travel
- Shopping
- Other

### Dataset Structure

Organize your dataset as follows:

```
email_dataset/
├── train/
│   └── dataset.json
├── test/
│   └── dataset.json
├── vocab.json
└── categories.json
```

### Create Sample Dataset

For testing purposes, create a sample dataset:

```bash
python train_email_classifier_macbook.py --create-sample-dataset --dataset-path ./sample_emails
```

### Dataset Validation

Validate your dataset format:

```bash
python dataset/build_email_dataset.py --validate --input_file your_emails.json
```

## Configuration

### Using Pre-built Configurations

Choose a configuration based on your MacBook model:

**8GB MacBook:**
```bash
cp examples/macbook_training/configs/macbook_8gb/email_classification.yaml config.yaml
```

**16GB MacBook:**
```bash
cp examples/macbook_training/configs/macbook_16gb/email_classification.yaml config.yaml
```

**32GB+ MacBook:**
```bash
cp examples/macbook_training/configs/macbook_32gb/email_classification.yaml config.yaml
```

### Custom Configuration

Generate a custom configuration for your hardware:

```bash
python examples/macbook_training/config_validator.py \
  --recommend \
  --memory-gb 16 \
  --cpu-cores 8 \
  --dataset-size 5000 \
  --output custom_config.yaml
```

### Configuration Parameters

Key parameters to adjust:

| Parameter | 8GB MacBook | 16GB MacBook | 32GB+ MacBook | Description |
|-----------|-------------|--------------|---------------|-------------|
| `batch_size` | 2 | 4 | 8 | Training batch size |
| `gradient_accumulation_steps` | 16 | 8 | 4 | Gradient accumulation |
| `hidden_size` | 256 | 384 | 512 | Model hidden dimension |
| `max_sequence_length` | 256 | 512 | 768 | Maximum email length |
| `memory_limit_mb` | 5500 | 12000 | 24000 | Memory usage limit |

## Training Execution

### Basic Training

Run training with automatic hardware detection:

```bash
python train_email_classifier_macbook.py \
  --train \
  --dataset-path ./your_email_dataset \
  --output-dir ./training_output
```

### Training with Custom Configuration

```bash
python train_email_classifier_macbook.py \
  --train \
  --dataset-path ./your_email_dataset \
  --output-dir ./training_output \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --max-steps 8000 \
  --strategy multi_phase
```

### Training Strategies

Choose from different training strategies:

**Single Phase (fastest):**
```bash
--strategy single
```

**Multi-Phase (recommended):**
```bash
--strategy multi_phase
```

**Progressive (for complex datasets):**
```bash
--strategy progressive
```

**Curriculum Learning (for diverse datasets):**
```bash
--strategy curriculum
```

### Monitoring Training Progress

Training progress is displayed in real-time:

```
Starting phase 1/3: warmup
Phase description: Warmup phase with lower learning rate
Step 100: loss=2.1234, acc=0.6789, steps/sec=1.23
Step 200: loss=1.9876, acc=0.7234, steps/sec=1.25
...
Phase warmup completed successfully

Starting phase 2/3: main_training
Phase description: Main training phase with full configuration
Step 300: loss=1.7654, acc=0.8123, steps/sec=1.20
...
```

### Training Outputs

Training produces the following outputs:

```
training_output/
├── email_training_20240115_143022_final_model.pt  # Final trained model
├── training_config.json                           # Training configuration
├── email_training_20240115_143022_result.json    # Training results
├── checkpoints/                                   # Training checkpoints
│   ├── checkpoint_step_1000.pt
│   └── checkpoint_step_2000.pt
└── logs/                                          # Training logs
    └── training.log
```

## Model Evaluation

### Automatic Evaluation

Training automatically evaluates the model and reports:

```
TRAINING COMPLETED
================
✓ Training successful!
✓ Final accuracy: 0.9534
✓ Best accuracy: 0.9587
✓ Training time: 127.3 minutes
✓ Phases completed: warmup, main_training, fine_tuning

Per-category performance:
  Newsletter: 0.9623
  Work: 0.9545
  Personal: 0.9678
  Spam: 0.9834
  Promotional: 0.9456
  Social: 0.9234
  Finance: 0.9567
  Travel: 0.9445
  Shopping: 0.9389
  Other: 0.9123
```

### Manual Evaluation

Evaluate a trained model on new data:

```bash
python evaluators/email.py \
  --model-path ./training_output/final_model.pt \
  --dataset-path ./test_emails \
  --output-dir ./evaluation_results
```

### Performance Metrics

The system reports comprehensive metrics:

- **Overall Accuracy**: Percentage of correctly classified emails
- **Per-Category Accuracy**: Accuracy for each of the 10 categories
- **F1 Scores**: Macro, micro, and weighted F1 scores
- **Confusion Matrix**: Detailed classification breakdown
- **Training Efficiency**: Samples per second, memory usage

## Production Deployment

### Model Export

Export the trained model for production use:

```bash
python macbook_optimization/model_export.py \
  --model-path ./training_output/final_model.pt \
  --output-dir ./production_model \
  --format pytorch
```

### Inference API

Deploy the model as a REST API:

```bash
python macbook_optimization/inference_api.py \
  --model-path ./production_model/model.pt \
  --host 0.0.0.0 \
  --port 8000
```

### API Usage

Classify emails using the API:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Meeting Tomorrow",
    "body": "Don't forget about our meeting tomorrow at 2 PM.",
    "sender": "manager@company.com"
  }'
```

**Response:**
```json
{
  "category": "Work",
  "confidence": 0.9567,
  "category_id": 1,
  "processing_time_ms": 23
}
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce batch size: `--batch-size 2`
- Increase gradient accumulation: `--gradient-accumulation-steps 16`
- Enable dynamic batch sizing in configuration

**Slow Training:**
- Reduce model complexity for 8GB MacBooks
- Use streaming data loading for large datasets
- Enable CPU optimization in configuration

**Low Accuracy:**
- Increase training steps: `--max-steps 10000`
- Use multi-phase training strategy
- Ensure balanced dataset across categories

**Configuration Errors:**
- Validate configuration: `python config_validator.py --config config.yaml`
- Use hardware-appropriate presets
- Check dataset format and structure

### Getting Help

1. Check the [troubleshooting guide](troubleshooting_guide.md)
2. Review [performance optimization guide](performance_optimization_guide.md)
3. Validate your configuration and dataset
4. Check system logs for detailed error messages

## Next Steps

After successful training:

1. **Evaluate Performance**: Test on diverse email samples
2. **Optimize Configuration**: Fine-tune parameters for your use case
3. **Deploy to Production**: Set up inference API and monitoring
4. **Monitor Performance**: Track accuracy and resource usage in production
5. **Iterate and Improve**: Collect feedback and retrain as needed

## Additional Resources

- [Performance Optimization Guide](performance_optimization_guide.md)
- [Troubleshooting Guide](troubleshooting_guide.md)
- [API Documentation](api_documentation.md)
- [Configuration Reference](configuration_reference.md)