# Gmail Dataset Creator

A Python system for creating high-quality email classification datasets from Gmail data using intelligent categorization via Google's Gemini API.

## Project Structure

```
gmail_dataset_creator/
├── __init__.py                 # Package initialization and exports
├── main.py                     # Main orchestrator class
├── models.py                   # Core data models and interfaces
├── auth/                       # Authentication module
│   ├── __init__.py
│   └── authentication.py      # OAuth2 authentication handler (to be implemented)
├── gmail/                      # Gmail API client module
│   ├── __init__.py
│   └── client.py              # Gmail API client (to be implemented)
├── processing/                 # Email processing and classification
│   ├── __init__.py
│   ├── email_processor.py     # Email content extraction (to be implemented)
│   └── gemini_classifier.py   # Gemini API classification (to be implemented)
├── dataset/                    # Dataset building and export
│   ├── __init__.py
│   └── builder.py             # Dataset builder (to be implemented)
├── config/                     # Configuration management
│   ├── __init__.py
│   └── manager.py             # Configuration manager
└── utils/                      # Utility modules
    ├── __init__.py
    ├── logging.py             # Logging configuration
    └── helpers.py             # Helper functions
```

## Core Components

### Data Models (`models.py`)

- **EmailData**: Core email data structure with validation
- **ClassificationResult**: Email classification results from Gemini API
- **DatasetStats**: Dataset statistics and metadata
- **CATEGORIES**: Email category mappings (newsletter, work, personal, etc.)

### Configuration Management (`config/manager.py`)

- **ConfigManager**: Loads and validates configuration from YAML files and environment variables
- **SystemConfig**: Complete system configuration with nested config objects
- Support for environment variable overrides

### Main Orchestrator (`main.py`)

- **GmailDatasetCreator**: Main class that coordinates all system components
- Handles setup, authentication, and dataset creation workflow
- Provides status monitoring and error handling

### Utilities (`utils/`)

- **Logging**: Structured logging setup with console and file output
- **Helpers**: Common utility functions for validation, formatting, and data processing

## Configuration

The system uses YAML configuration files with environment variable support:

```yaml
gmail_api:
  credentials_file: "credentials.json"
  token_file: "token.json"
  scopes: ["https://www.googleapis.com/auth/gmail.readonly"]

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
  date_range:
    start: "2023-01-01"
    end: "2024-12-31"
  exclude_labels: ["TRASH", "SPAM"]
  include_labels: ["INBOX"]

privacy:
  anonymize_senders: true
  exclude_personal: false
  remove_attachments: true
```

## Usage

### Command Line Interface

```bash
# Basic usage
python create_gmail_dataset.py --config config.yaml

# With custom parameters
python create_gmail_dataset.py --max-emails 500 --output ./my_dataset

# Authentication only
python create_gmail_dataset.py --authenticate-only

# Show system status
python create_gmail_dataset.py --status
```

### Programmatic Usage

```python
from gmail_dataset_creator import GmailDatasetCreator

# Initialize with configuration
creator = GmailDatasetCreator(config_path="config.yaml")

# Setup system
creator.setup()

# Authenticate
creator.authenticate()

# Create dataset
stats = creator.create_dataset(max_emails=1000)

print(f"Created dataset with {stats.total_emails} emails")
```

## Dependencies

See `requirements_gmail_dataset.txt` for complete dependency list:

- google-auth-oauthlib: OAuth2 authentication
- google-api-python-client: Gmail API client
- google-generativeai: Gemini API integration
- beautifulsoup4: HTML content processing
- pyyaml: Configuration file parsing

## Implementation Status

This is the initial project structure setup (Task 1). The following components are implemented:

✅ **Completed:**
- Project directory structure
- Core data models and interfaces
- Configuration management system
- Main orchestrator class structure
- Utility modules (logging, helpers)
- CLI entry point

🚧 **To be implemented in subsequent tasks:**
- OAuth2 authentication handler (Task 2)
- Gmail API client with rate limiting (Task 3)
- Email content processing (Task 4)
- Gemini API classification (Task 5)
- Dataset building and export (Task 6)
- Comprehensive error handling and logging (Task 7)
- Privacy and security features (Task 8)
- Testing suite (Task 10)

## Next Steps

1. Implement OAuth2 authentication system (Task 2)
2. Create Gmail API client with rate limiting (Task 3)
3. Build email content processing pipeline (Task 4)
4. Integrate Gemini API for classification (Task 5)
5. Implement dataset generation and export (Task 6)

Each task builds incrementally on the foundation established in this initial setup.