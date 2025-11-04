# Gmail Dataset Creator - CLI Usage Guide

This guide covers all aspects of using the Gmail Dataset Creator command-line interface.

## Installation

```bash
# Install from source
pip install -e .

# Or install from package
pip install gmail-dataset-creator
```

## Quick Start

### 1. Generate Configuration File

```bash
gmail-dataset-creator --generate-config config.yaml
```

### 2. Set Up Environment

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Download credentials.json from Google Cloud Console
```

### 3. Run Interactive Setup

```bash
gmail-dataset-creator --interactive
```

### 4. Create Dataset

```bash
gmail-dataset-creator --config config.yaml --max-emails 1000
```

## Command Reference

### Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config`, `-c` | Configuration YAML file | `--config config.yaml` |
| `--output`, `-o` | Output directory | `--output ./my_dataset` |
| `--credentials` | Gmail credentials file | `--credentials creds.json` |
| `--gemini-key` | Gemini API key | `--gemini-key sk-...` |

### Dataset Options

| Option | Description | Example |
|--------|-------------|---------|
| `--max-emails` | Maximum emails to process | `--max-emails 1000` |
| `--train-ratio` | Train/test split ratio | `--train-ratio 0.8` |
| `--date-start` | Start date filter | `--date-start 2023-01-01` |
| `--date-end` | End date filter | `--date-end 2023-12-31` |

### Privacy & Security Options

| Option | Description | Example |
|--------|-------------|---------|
| `--anonymize-senders` | Anonymize email addresses | `--anonymize-senders` |
| `--exclude-personal` | Exclude personal emails | `--exclude-personal` |
| `--confidence-threshold` | Min classification confidence | `--confidence-threshold 0.8` |

### Process Control Options

| Option | Description | Example |
|--------|-------------|---------|
| `--resume` | Resume from checkpoint | `--resume` |
| `--interactive`, `-i` | Interactive configuration | `--interactive` |
| `--dry-run` | Show config without processing | `--dry-run` |

### Utility Commands

| Option | Description | Example |
|--------|-------------|---------|
| `--auth-only` | Test authentication only | `--auth-only` |
| `--status` | Show system status | `--status` |
| `--generate-config` | Generate sample config | `--generate-config config.yaml` |

### Logging Options

| Option | Description | Example |
|--------|-------------|---------|
| `--verbose`, `-v` | Enable verbose logging | `--verbose` |
| `--quiet`, `-q` | Suppress non-error output | `--quiet` |
| `--log-file` | Log to file | `--log-file dataset.log` |

## Usage Examples

### Basic Usage

```bash
# Simple dataset creation
gmail-dataset-creator --max-emails 500 --output ./my_dataset

# With date filtering
gmail-dataset-creator --date-start 2023-01-01 --date-end 2023-12-31 --max-emails 1000

# With privacy settings
gmail-dataset-creator --anonymize-senders --exclude-personal --confidence-threshold 0.8
```

### Advanced Usage

```bash
# Full configuration with all options
gmail-dataset-creator \
  --config advanced_config.yaml \
  --max-emails 2000 \
  --train-ratio 0.85 \
  --date-start 2023-01-01 \
  --date-end 2024-01-01 \
  --output ./large_dataset \
  --anonymize-senders \
  --confidence-threshold 0.8 \
  --verbose \
  --log-file creation.log

# Resume interrupted process
gmail-dataset-creator --config config.yaml --resume --verbose

# Test configuration without processing
gmail-dataset-creator --config config.yaml --dry-run
```

### Utility Commands

```bash
# Generate sample configuration
gmail-dataset-creator --generate-config my_config.yaml

# Test authentication
gmail-dataset-creator --auth-only --verbose

# Check system status
gmail-dataset-creator --status

# Interactive setup
gmail-dataset-creator --interactive
```

## Configuration File Format

```yaml
gmail_api:
  credentials_file: "credentials.json"
  token_file: "token.json"
  scopes:
    - "https://www.googleapis.com/auth/gmail.readonly"

gemini_api:
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-pro"
  max_tokens: 1000

dataset:
  output_path: "./gmail_dataset"
  train_ratio: 0.8
  min_emails_per_category: 10
  max_emails_total: 1000

filters:
  date_range: ["2023-01-01", "2023-12-31"]
  exclude_labels: ["TRASH", "SPAM"]
  include_labels: ["INBOX"]
  sender_filters: []

privacy:
  anonymize_senders: true
  exclude_personal: false
  remove_attachments: true
  encrypt_tokens: true
  exclude_sensitive: true
  anonymize_recipients: true
  remove_sensitive_content: true
  exclude_keywords: ["password", "ssn"]
  exclude_domains: ["internal.company.com"]
  min_confidence_threshold: 0.7

security:
  encryption_algorithm: "fernet"
  secure_export: true
  data_retention_days: 30
  secure_cleanup: true
  audit_logging: true
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key | `export GEMINI_API_KEY="your-key"` |
| `GMAIL_CREDENTIALS_PATH` | Gmail credentials file | `export GMAIL_CREDENTIALS_PATH="./creds.json"` |
| `OUTPUT_PATH` | Default output directory | `export OUTPUT_PATH="./datasets"` |
| `MAX_EMAILS` | Default max emails | `export MAX_EMAILS="1000"` |
| `TRAIN_RATIO` | Default train ratio | `export TRAIN_RATIO="0.8"` |

## Interactive Mode

The interactive mode guides you through configuration:

```bash
gmail-dataset-creator --interactive
```

You'll be prompted for:
- Configuration file path
- Output directory
- Gmail credentials file
- Gemini API key
- Maximum number of emails
- Date range filters
- Privacy settings

## Error Handling and Recovery

### Resume Interrupted Process

If the process is interrupted, you can resume:

```bash
gmail-dataset-creator --config config.yaml --resume
```

### Common Issues

1. **Authentication Failed**
   ```bash
   # Test authentication separately
   gmail-dataset-creator --auth-only --verbose
   ```

2. **Configuration Errors**
   ```bash
   # Validate configuration
   gmail-dataset-creator --config config.yaml --dry-run
   ```

3. **API Rate Limits**
   - The system automatically handles rate limits
   - Use `--verbose` to see retry attempts

4. **Low Confidence Classifications**
   - Adjust `--confidence-threshold` to filter results
   - Check logs for flagged emails

## Output Structure

The CLI creates the following output structure:

```
output_directory/
├── train/
│   └── dataset.json
├── test/
│   └── dataset.json
├── categories.json
├── vocab.json
├── stats.json
├── checkpoints/
│   └── process_state.json
└── logs/
    └── gmail_dataset_creator.log
```

## Performance Tips

1. **Batch Processing**: Use reasonable `--max-emails` values (500-2000)
2. **Date Filtering**: Use `--date-start` and `--date-end` to limit scope
3. **Resume Capability**: Use `--resume` for large datasets
4. **Logging**: Use `--log-file` to track progress
5. **Dry Run**: Use `--dry-run` to validate configuration first

## Troubleshooting

### Enable Verbose Logging

```bash
gmail-dataset-creator --verbose --log-file debug.log
```

### Check System Status

```bash
gmail-dataset-creator --status
```

### Validate Configuration

```bash
gmail-dataset-creator --config config.yaml --dry-run
```

### Test Authentication

```bash
gmail-dataset-creator --auth-only --verbose
```

## Integration Examples

### Shell Script

```bash
#!/bin/bash
# create_dataset.sh

set -e

echo "Creating email dataset..."

# Set environment
export GEMINI_API_KEY="your-api-key"

# Create dataset
gmail-dataset-creator \
  --config production_config.yaml \
  --max-emails 5000 \
  --output "./datasets/$(date +%Y%m%d)" \
  --verbose \
  --log-file "dataset_$(date +%Y%m%d).log"

echo "Dataset creation complete!"
```

### Python Integration

```python
import subprocess
import sys

def create_dataset(config_path, output_path, max_emails=1000):
    """Create dataset using CLI."""
    cmd = [
        "gmail-dataset-creator",
        "--config", config_path,
        "--output", output_path,
        "--max-emails", str(max_emails),
        "--verbose"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Dataset created successfully!")
        print(result.stdout)
    else:
        print("Dataset creation failed!")
        print(result.stderr)
        sys.exit(1)

# Usage
create_dataset("config.yaml", "./my_dataset", 500)
```

This CLI provides a comprehensive interface for all Gmail Dataset Creator functionality with extensive configuration options, error handling, and recovery capabilities.