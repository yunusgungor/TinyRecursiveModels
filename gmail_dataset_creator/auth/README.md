# Gmail API Authentication System

This module provides secure OAuth2 authentication for the Gmail API with encrypted token storage.

## Features

- **OAuth2 Flow**: Complete Google OAuth2 authorization flow
- **Secure Token Storage**: Encrypted storage of refresh tokens using PBKDF2 and Fernet encryption
- **Automatic Token Refresh**: Handles expired tokens automatically
- **Error Handling**: Comprehensive error handling with recovery suggestions
- **Authentication State Management**: Track and validate authentication status

## Setup

### 1. Google Cloud Console Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API
4. Create OAuth2 credentials (Desktop Application)
5. Download the credentials JSON file

### 2. Installation

Install required dependencies:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 cryptography
```

## Usage

### Basic Authentication

```python
from gmail_dataset_creator.auth import AuthenticationHandler, AuthConfig

# Configure authentication
config = AuthConfig(
    credentials_file="path/to/credentials.json",
    token_file="path/to/token.json",
    scopes=['https://www.googleapis.com/auth/gmail.readonly'],
    use_encryption=True
)

# Initialize handler
auth_handler = AuthenticationHandler(config)

# Authenticate
if auth_handler.authenticate():
    print("Authentication successful!")
    credentials = auth_handler.get_credentials()
else:
    print("Authentication failed!")
```

### Configuration Options

- `credentials_file`: Path to Google OAuth2 credentials JSON file
- `token_file`: Path where encrypted tokens will be stored
- `scopes`: List of Gmail API scopes to request
- `use_encryption`: Enable/disable token encryption (default: True)
- `encryption_password`: Optional password for encryption (will prompt if None)

### Security Features

#### Encrypted Token Storage

When `use_encryption=True`, tokens are encrypted using:
- **PBKDF2** key derivation with 100,000 iterations
- **Fernet** symmetric encryption (AES 128 in CBC mode)
- **Random salt** for each encryption key
- **Secure file permissions** (600) for token and key files

#### Token Validation

The system validates:
- Token structure and required fields
- Token expiration and refresh capability
- Encryption integrity
- Authentication state consistency

### Error Handling

The authentication handler provides detailed error information:

```python
try:
    auth_handler.authenticate()
except Exception as e:
    error_info = auth_handler.handle_authentication_error(e)
    print(f"Error: {error_info['error_message']}")
    for suggestion in error_info['recovery_suggestions']:
        print(f"  - {suggestion}")
```

### Authentication State

Check current authentication status:

```python
state = auth_handler.get_auth_state()
print(f"Authenticated: {state['authenticated']}")
print(f"Token exists: {state['token_file_exists']}")
print(f"Encryption enabled: {state['encryption_enabled']}")
```

## Security Considerations

1. **Credentials File**: Keep your `credentials.json` file secure and never commit it to version control
2. **Token Storage**: Encrypted tokens are stored locally with secure file permissions
3. **Password Protection**: Use a strong password for token encryption
4. **Scope Limitation**: Only request necessary Gmail API scopes
5. **Token Revocation**: Use `revoke_access()` to properly revoke tokens when done

## Example

See `examples/auth_example.py` for a complete working example.

## API Reference

### AuthenticationHandler

Main class for handling Gmail API authentication.

#### Methods

- `authenticate() -> bool`: Run OAuth2 flow and authenticate
- `get_credentials() -> Optional[Credentials]`: Get current valid credentials
- `refresh_token() -> bool`: Refresh expired access token
- `is_authenticated() -> bool`: Check if currently authenticated
- `revoke_access() -> bool`: Revoke access and clear tokens
- `get_auth_state() -> Dict[str, Any]`: Get authentication state info
- `validate_stored_token() -> bool`: Validate stored token integrity

### SecureTokenStorage

Handles encrypted storage of OAuth2 tokens.

#### Methods

- `save_token(token_data: Dict[str, Any]) -> bool`: Save encrypted token
- `load_token() -> Optional[Dict[str, Any]]`: Load and decrypt token
- `delete_token() -> bool`: Delete stored token files
- `token_exists() -> bool`: Check if token file exists
- `validate_token_integrity() -> bool`: Validate token can be decrypted

### TokenValidator

Validates OAuth2 token data and structure.

#### Methods

- `is_token_valid(token_data: Dict[str, Any]) -> bool`: Validate token structure
- `is_token_expired(token_data: Dict[str, Any]) -> bool`: Check if token expired
- `needs_refresh(token_data: Dict[str, Any]) -> bool`: Check if refresh needed