"""
Tests for the authentication handler module.

This module contains unit tests for the AuthenticationHandler class,
testing OAuth2 flow, token management, and error handling.
"""

import unittest
import tempfile
import os
import json
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Create proper mock exception classes
class MockRefreshError(Exception):
    """Mock RefreshError for testing."""
    pass

# Create mock classes
class MockCredentials:
    """Mock Credentials class for testing."""
    def __init__(self):
        self.token = None
        self.refresh_token = None
        self.expired = False
        self.valid = True
    
    def refresh(self, request):
        pass
    
    @classmethod
    def from_authorized_user_file(cls, filename, scopes=None):
        """Mock class method for loading credentials from file."""
        return cls()

# Mock Google API modules to avoid import errors
mock_google = MagicMock()
mock_oauth2 = MagicMock()
mock_credentials_module = MagicMock()
mock_credentials_module.Credentials = MockCredentials
mock_oauth2.credentials = mock_credentials_module
mock_google.oauth2 = mock_oauth2

mock_auth = MagicMock()
mock_transport = MagicMock()
mock_requests = MagicMock()
mock_requests.Request = MagicMock()
mock_transport.requests = mock_requests
mock_auth.transport = mock_transport

mock_exceptions = MagicMock()
mock_exceptions.RefreshError = MockRefreshError
mock_auth.exceptions = mock_exceptions
mock_google.auth = mock_auth

mock_oauthlib = MagicMock()
mock_flow = MagicMock()
mock_oauthlib.flow = mock_flow

sys.modules['google'] = mock_google
sys.modules['google.oauth2'] = mock_oauth2
sys.modules['google.oauth2.credentials'] = mock_credentials_module
sys.modules['google.auth'] = mock_auth
sys.modules['google.auth.transport'] = mock_transport
sys.modules['google.auth.transport.requests'] = mock_requests
sys.modules['google.auth.exceptions'] = mock_exceptions
sys.modules['google_auth_oauthlib'] = mock_oauthlib
sys.modules['google_auth_oauthlib.flow'] = mock_flow

# Import the classes for use in tests
Credentials = MockCredentials
RefreshError = MockRefreshError

# Import after mocking
from gmail_dataset_creator.auth.authentication import AuthenticationHandler, AuthConfig


class TestAuthConfig(unittest.TestCase):
    """Test the AuthConfig dataclass."""
    
    def test_auth_config_creation(self):
        """Test AuthConfig creation with default values."""
        config = AuthConfig(
            credentials_file="credentials.json",
            token_file="token.json",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        
        self.assertEqual(config.credentials_file, "credentials.json")
        self.assertEqual(config.token_file, "token.json")
        self.assertEqual(config.scopes, ["https://www.googleapis.com/auth/gmail.readonly"])
        self.assertTrue(config.use_encryption)
        self.assertIsNone(config.encryption_password)
    
    def test_auth_config_custom_values(self):
        """Test AuthConfig creation with custom values."""
        config = AuthConfig(
            credentials_file="custom_creds.json",
            token_file="custom_token.json",
            scopes=["custom.scope"],
            use_encryption=False,
            encryption_password="test_password"
        )
        
        self.assertEqual(config.credentials_file, "custom_creds.json")
        self.assertEqual(config.token_file, "custom_token.json")
        self.assertEqual(config.scopes, ["custom.scope"])
        self.assertFalse(config.use_encryption)
        self.assertEqual(config.encryption_password, "test_password")


class TestAuthenticationHandler(unittest.TestCase):
    """Test the AuthenticationHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.credentials_file = os.path.join(self.temp_dir, "credentials.json")
        self.token_file = os.path.join(self.temp_dir, "token.json")
        
        # Create mock credentials file
        mock_credentials = {
            "installed": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        }
        
        with open(self.credentials_file, 'w') as f:
            json.dump(mock_credentials, f)
        
        # Create test configuration
        self.config = AuthConfig(
            credentials_file=self.credentials_file,
            token_file=self.token_file,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            use_encryption=False  # Disable encryption for simpler testing
        )
        
        self.config_encrypted = AuthConfig(
            credentials_file=self.credentials_file,
            token_file=self.token_file,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            use_encryption=True,
            encryption_password="test_password"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test AuthenticationHandler initialization."""
        handler = AuthenticationHandler(self.config)
        
        self.assertEqual(handler.config, self.config)
        self.assertIsNone(handler.credentials)
        self.assertIsNone(handler.token_storage)  # No encryption
        self.assertIsNotNone(handler.token_validator)
    
    def test_initialization_with_encryption(self):
        """Test AuthenticationHandler initialization with encryption."""
        with patch('gmail_dataset_creator.auth.authentication.SecureTokenStorage') as mock_storage:
            handler = AuthenticationHandler(self.config_encrypted)
            
            self.assertEqual(handler.config, self.config_encrypted)
            self.assertIsNone(handler.credentials)
            mock_storage.assert_called_once_with(
                self.token_file,
                "test_password"
            )
    
    def test_get_credentials_no_credentials(self):
        """Test get_credentials when no credentials exist."""
        handler = AuthenticationHandler(self.config)
        
        result = handler.get_credentials()
        self.assertIsNone(result)
    
    def test_get_credentials_valid_credentials(self):
        """Test get_credentials with valid credentials."""
        handler = AuthenticationHandler(self.config)
        
        # Mock valid credentials
        mock_creds = Mock(spec=Credentials)
        mock_creds.valid = True
        handler.credentials = mock_creds
        
        result = handler.get_credentials()
        self.assertEqual(result, mock_creds)
    
    def test_get_credentials_expired_with_refresh(self):
        """Test get_credentials with expired credentials that can be refreshed."""
        handler = AuthenticationHandler(self.config)
        
        # Mock expired credentials with refresh token
        mock_creds = Mock(spec=Credentials)
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token"
        handler.credentials = mock_creds
        
        with patch.object(handler, 'refresh_token', return_value=True):
            result = handler.get_credentials()
            self.assertEqual(result, mock_creds)
    
    def test_is_authenticated_true(self):
        """Test is_authenticated returns True for valid credentials."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, 'get_credentials') as mock_get_creds:
            mock_creds = Mock()
            mock_creds.valid = True
            mock_get_creds.return_value = mock_creds
            
            result = handler.is_authenticated()
            self.assertTrue(result)
    
    def test_is_authenticated_false(self):
        """Test is_authenticated returns False for no credentials."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, 'get_credentials', return_value=None):
            result = handler.is_authenticated()
            self.assertFalse(result)
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_refresh_token_success(self, mock_request):
        """Test successful token refresh."""
        handler = AuthenticationHandler(self.config)
        
        # Mock credentials with refresh token
        mock_creds = Mock()
        mock_creds.refresh_token = "refresh_token"
        handler.credentials = mock_creds
        
        with patch.object(handler, '_save_credentials'):
            result = handler.refresh_token()
            
            self.assertTrue(result)
            mock_creds.refresh.assert_called_once()
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_refresh_token_no_refresh_token(self, mock_request):
        """Test token refresh with no refresh token."""
        handler = AuthenticationHandler(self.config)
        
        # Mock credentials without refresh token
        mock_creds = Mock()
        mock_creds.refresh_token = None
        handler.credentials = mock_creds
        
        result = handler.refresh_token()
        self.assertFalse(result)
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_refresh_token_refresh_error(self, mock_request):
        """Test token refresh with RefreshError."""
        handler = AuthenticationHandler(self.config)
        
        # Mock credentials with refresh token
        mock_creds = Mock()
        mock_creds.refresh_token = "refresh_token"
        mock_creds.refresh.side_effect = MockRefreshError("Token expired")
        handler.credentials = mock_creds
        
        with patch.object(handler, '_clear_stored_credentials'):
            result = handler.refresh_token()
            
            self.assertFalse(result)
            self.assertIsNone(handler.credentials)
    
    def test_load_existing_credentials_no_file(self):
        """Test loading credentials when token file doesn't exist."""
        handler = AuthenticationHandler(self.config)
        
        result = handler._load_existing_credentials()
        self.assertFalse(result)
    
    def test_load_existing_credentials_success(self):
        """Test successful loading of existing credentials."""
        handler = AuthenticationHandler(self.config)
        
        # Create mock token file
        mock_token_data = {
            "token": "access_token",
            "refresh_token": "refresh_token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client_id",
            "client_secret": "client_secret",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
        }
        
        with open(self.token_file, 'w') as f:
            json.dump(mock_token_data, f)
        
        with patch('gmail_dataset_creator.auth.authentication.Credentials.from_authorized_user_file') as mock_from_file:
            mock_creds = Mock()
            mock_creds.valid = True
            mock_from_file.return_value = mock_creds
            
            result = handler._load_existing_credentials()
            
            self.assertTrue(result)
            self.assertEqual(handler.credentials, mock_creds)
            mock_from_file.assert_called_once_with(self.token_file, self.config.scopes)
    
    def test_load_existing_credentials_with_encryption(self):
        """Test loading credentials with encryption enabled."""
        handler = AuthenticationHandler(self.config_encrypted)
        
        mock_token_data = {
            "token": "access_token",
            "refresh_token": "refresh_token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client_id",
            "client_secret": "client_secret",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
        }
        
        with patch.object(handler, 'token_storage') as mock_storage:
            mock_storage.token_exists.return_value = True
            mock_storage.load_token.return_value = mock_token_data
            
            with patch.object(handler, 'token_validator') as mock_validator:
                mock_validator.is_token_valid.return_value = True
                
                with patch('gmail_dataset_creator.auth.authentication.Credentials') as mock_creds_class:
                    mock_creds = Mock()
                    mock_creds.valid = True
                    mock_creds_class.return_value = mock_creds
                    
                    result = handler._load_existing_credentials()
                    
                    self.assertTrue(result)
                    self.assertEqual(handler.credentials, mock_creds)
    
    @patch('gmail_dataset_creator.auth.authentication.InstalledAppFlow')
    def test_run_oauth_flow_success(self, mock_flow_class):
        """Test successful OAuth flow."""
        handler = AuthenticationHandler(self.config)
        
        # Mock the OAuth flow
        mock_flow = Mock()
        mock_creds = Mock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_class.from_client_secrets_file.return_value = mock_flow
        
        with patch.object(handler, '_save_credentials'):
            result = handler._run_oauth_flow()
            
            self.assertTrue(result)
            self.assertEqual(handler.credentials, mock_creds)
            mock_flow_class.from_client_secrets_file.assert_called_once_with(
                self.credentials_file, 
                self.config.scopes
            )
            mock_flow.run_local_server.assert_called_once_with(port=0)
    
    @patch('gmail_dataset_creator.auth.authentication.InstalledAppFlow')
    def test_run_oauth_flow_no_credentials_file(self, mock_flow_class):
        """Test OAuth flow when credentials file doesn't exist."""
        # Remove the credentials file
        os.remove(self.credentials_file)
        
        handler = AuthenticationHandler(self.config)
        
        result = handler._run_oauth_flow()
        self.assertFalse(result)
    
    @patch('gmail_dataset_creator.auth.authentication.InstalledAppFlow')
    def test_run_oauth_flow_exception(self, mock_flow_class):
        """Test OAuth flow with exception."""
        handler = AuthenticationHandler(self.config)
        
        mock_flow_class.from_client_secrets_file.side_effect = Exception("OAuth error")
        
        result = handler._run_oauth_flow()
        self.assertFalse(result)
    
    def test_authenticate_with_existing_credentials(self):
        """Test authenticate method with existing valid credentials."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, '_load_existing_credentials', return_value=True):
            result = handler.authenticate()
            self.assertTrue(result)
    
    def test_authenticate_with_oauth_flow(self):
        """Test authenticate method falling back to OAuth flow."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, '_load_existing_credentials', return_value=False):
            with patch.object(handler, '_run_oauth_flow', return_value=True):
                result = handler.authenticate()
                self.assertTrue(result)
    
    def test_authenticate_failure(self):
        """Test authenticate method with failure."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, '_load_existing_credentials', side_effect=Exception("Test error")):
            result = handler.authenticate()
            self.assertFalse(result)
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_revoke_access_success(self, mock_request):
        """Test successful access revocation."""
        handler = AuthenticationHandler(self.config)
        
        mock_creds = Mock()
        handler.credentials = mock_creds
        
        with patch.object(handler, '_clear_stored_credentials'):
            result = handler.revoke_access()
            
            self.assertTrue(result)
            mock_creds.revoke.assert_called_once()
            self.assertIsNone(handler.credentials)
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_revoke_access_no_credentials(self, mock_request):
        """Test access revocation with no credentials."""
        handler = AuthenticationHandler(self.config)
        
        with patch.object(handler, '_clear_stored_credentials'):
            result = handler.revoke_access()
            
            self.assertTrue(result)
            self.assertIsNone(handler.credentials)
    
    @patch('gmail_dataset_creator.auth.authentication.Request')
    def test_revoke_access_exception(self, mock_request):
        """Test access revocation with exception."""
        handler = AuthenticationHandler(self.config)
        
        mock_creds = Mock()
        mock_creds.revoke.side_effect = Exception("Revoke error")
        handler.credentials = mock_creds
        
        with patch.object(handler, '_clear_stored_credentials'):
            result = handler.revoke_access()
            
            # Should still return True and clear credentials
            self.assertFalse(result)
            self.assertIsNone(handler.credentials)
    
    def test_save_credentials_standard(self):
        """Test saving credentials to standard file."""
        handler = AuthenticationHandler(self.config)
        
        mock_creds = Mock()
        mock_creds.to_json.return_value = '{"token": "test_token"}'
        handler.credentials = mock_creds
        
        handler._save_credentials()
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.token_file))
        
        # Check file contents
        with open(self.token_file, 'r') as f:
            content = f.read()
            self.assertEqual(content, '{"token": "test_token"}')
    
    def test_save_credentials_encrypted(self):
        """Test saving credentials with encryption."""
        handler = AuthenticationHandler(self.config_encrypted)
        
        mock_creds = Mock()
        mock_creds.token = "access_token"
        mock_creds.refresh_token = "refresh_token"
        mock_creds.token_uri = "token_uri"
        mock_creds.client_id = "client_id"
        mock_creds.client_secret = "client_secret"
        mock_creds.scopes = ["scope1"]
        handler.credentials = mock_creds
        
        with patch.object(handler, 'token_storage') as mock_storage:
            mock_storage.save_token.return_value = True
            
            handler._save_credentials()
            
            mock_storage.save_token.assert_called_once()
    
    def test_clear_stored_credentials_standard(self):
        """Test clearing standard stored credentials."""
        handler = AuthenticationHandler(self.config)
        
        # Create token file
        with open(self.token_file, 'w') as f:
            f.write('{"token": "test"}')
        
        handler._clear_stored_credentials()
        
        # Check that file was removed
        self.assertFalse(os.path.exists(self.token_file))
    
    def test_clear_stored_credentials_encrypted(self):
        """Test clearing encrypted stored credentials."""
        handler = AuthenticationHandler(self.config_encrypted)
        
        with patch.object(handler, 'token_storage') as mock_storage:
            mock_storage.delete_token.return_value = True
            
            handler._clear_stored_credentials()
            
            mock_storage.delete_token.assert_called_once()
    
    def test_get_auth_state(self):
        """Test getting authentication state."""
        handler = AuthenticationHandler(self.config)
        
        mock_creds = Mock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.refresh_token = "refresh_token"
        handler.credentials = mock_creds
        
        with patch.object(handler, 'is_authenticated', return_value=True):
            state = handler.get_auth_state()
            
            expected_keys = [
                'authenticated', 'has_credentials', 'credentials_file_exists',
                'scopes', 'encryption_enabled', 'token_file_exists',
                'credentials_valid', 'credentials_expired', 'has_refresh_token'
            ]
            
            for key in expected_keys:
                self.assertIn(key, state)
            
            self.assertTrue(state['authenticated'])
            self.assertTrue(state['has_credentials'])
            self.assertTrue(state['credentials_file_exists'])
            self.assertFalse(state['encryption_enabled'])
    
    def test_get_auth_state_with_encryption(self):
        """Test getting authentication state with encryption."""
        handler = AuthenticationHandler(self.config_encrypted)
        
        with patch.object(handler, 'token_storage') as mock_storage:
            mock_storage.token_exists.return_value = True
            mock_storage.validate_token_integrity.return_value = True
            
            with patch.object(handler, 'is_authenticated', return_value=False):
                state = handler.get_auth_state()
                
                self.assertTrue(state['encryption_enabled'])
                self.assertTrue(state['token_file_exists'])
                self.assertTrue(state['token_integrity_valid'])
    
    def test_validate_stored_token_standard(self):
        """Test validating stored token with standard storage."""
        handler = AuthenticationHandler(self.config)
        
        # Create token file
        with open(self.token_file, 'w') as f:
            f.write('{"token": "test"}')
        
        result = handler.validate_stored_token()
        self.assertTrue(result)
        
        # Remove token file
        os.remove(self.token_file)
        
        result = handler.validate_stored_token()
        self.assertFalse(result)
    
    def test_validate_stored_token_encrypted(self):
        """Test validating stored token with encryption."""
        handler = AuthenticationHandler(self.config_encrypted)
        
        with patch.object(handler, 'token_storage') as mock_storage:
            mock_storage.validate_token_integrity.return_value = True
            
            result = handler.validate_stored_token()
            self.assertTrue(result)
            
            mock_storage.validate_token_integrity.assert_called_once()
    
    def test_handle_authentication_error_refresh_error(self):
        """Test handling RefreshError."""
        handler = AuthenticationHandler(self.config)
        
        error = MockRefreshError("Token expired")
        error_info = handler.handle_authentication_error(error)
        
        self.assertEqual(error_info['error_type'], 'MockRefreshError')
        self.assertEqual(error_info['error_message'], 'Token expired')
        self.assertIn('The refresh token has expired or been revoked', error_info['recovery_suggestions'][0])
    
    def test_handle_authentication_error_decrypt_error(self):
        """Test handling decryption error."""
        handler = AuthenticationHandler(self.config)
        
        error = Exception("Failed to decrypt token")
        error_info = handler.handle_authentication_error(error)
        
        self.assertEqual(error_info['error_type'], 'Exception')
        self.assertIn('decrypt', error_info['error_message'])
        self.assertIn('decryption failed', error_info['recovery_suggestions'][0])
    
    def test_handle_authentication_error_generic(self):
        """Test handling generic error."""
        handler = AuthenticationHandler(self.config)
        
        error = Exception("Generic error")
        error_info = handler.handle_authentication_error(error)
        
        self.assertEqual(error_info['error_type'], 'Exception')
        self.assertIn('Check internet connection', error_info['recovery_suggestions'][0])


if __name__ == '__main__':
    unittest.main()