"""OAuth2 authentication handler for Gmail API."""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError

from .token_storage import SecureTokenStorage, TokenValidator


@dataclass
class AuthConfig:
    """Configuration for OAuth2 authentication."""
    credentials_file: str
    token_file: str
    scopes: list[str]
    use_encryption: bool = True
    encryption_password: Optional[str] = None


class AuthenticationHandler:
    """Handles OAuth2 authentication with Gmail API."""
    
    # Gmail API scopes
    DEFAULT_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, config: AuthConfig):
        """Initialize the authentication handler.
        
        Args:
            config: Authentication configuration containing credentials file,
                   token file path, and required scopes.
        """
        self.config = config
        self.credentials: Optional[Credentials] = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize secure token storage
        if self.config.use_encryption:
            self.token_storage = SecureTokenStorage(
                self.config.token_file,
                self.config.encryption_password
            )
        else:
            self.token_storage = None
        
        self.token_validator = TokenValidator()
        
        # Ensure token directory exists
        token_path = Path(self.config.token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
    
    def authenticate(self) -> bool:
        """Authenticate with Gmail API using OAuth2 flow.
        
        Returns:
            True if authentication successful, False otherwise.
        """
        try:
            # Try to load existing credentials
            if self._load_existing_credentials():
                self.logger.info("Loaded existing credentials successfully")
                return True
            
            # If no valid credentials exist, run OAuth flow
            self.logger.info("No valid credentials found, starting OAuth flow")
            return self._run_oauth_flow()
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get current credentials.
        
        Returns:
            Current credentials if authenticated, None otherwise.
        """
        if self.credentials and self.credentials.valid:
            return self.credentials
        
        # Try to refresh if we have credentials but they're expired
        if self.credentials and self.credentials.expired and self.credentials.refresh_token:
            if self.refresh_token():
                return self.credentials
        
        return None
    
    def refresh_token(self) -> bool:
        """Refresh expired access token using refresh token.
        
        Returns:
            True if refresh successful, False otherwise.
        """
        if not self.credentials or not self.credentials.refresh_token:
            self.logger.error("No refresh token available")
            return False
        
        try:
            request = Request()
            self.credentials.refresh(request)
            self._save_credentials()
            self.logger.info("Token refreshed successfully")
            return True
            
        except RefreshError as e:
            self.logger.error(f"Token refresh failed: {e}")
            # Clear invalid credentials
            self.credentials = None
            self._clear_stored_credentials()
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during token refresh: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid credentials.
        
        Returns:
            True if authenticated with valid credentials, False otherwise.
        """
        creds = self.get_credentials()
        return creds is not None and creds.valid
    
    def revoke_access(self) -> bool:
        """Revoke access and clear stored credentials.
        
        Returns:
            True if revocation successful, False otherwise.
        """
        try:
            if self.credentials:
                # Revoke the credentials
                request = Request()
                self.credentials.revoke(request)
                self.logger.info("Credentials revoked successfully")
            
            # Clear stored credentials
            self._clear_stored_credentials()
            self.credentials = None
            return True
            
        except Exception as e:
            self.logger.error(f"Error revoking access: {e}")
            # Still clear local credentials even if revocation failed
            self._clear_stored_credentials()
            self.credentials = None
            return False
    
    def _load_existing_credentials(self) -> bool:
        """Load existing credentials from token file.
        
        Returns:
            True if valid credentials loaded, False otherwise.
        """
        try:
            # Load token data using secure storage if enabled
            if self.config.use_encryption and self.token_storage:
                if not self.token_storage.token_exists():
                    return False
                
                token_data = self.token_storage.load_token()
                if not token_data:
                    self.logger.error("Failed to decrypt stored token")
                    return False
                
                # Validate token data
                if not self.token_validator.is_token_valid(token_data):
                    self.logger.error("Invalid token data structure")
                    return False
                
                # Create credentials from token data
                self.credentials = Credentials(
                    token=token_data.get('token'),
                    refresh_token=token_data.get('refresh_token'),
                    token_uri=token_data.get('token_uri'),
                    client_id=token_data.get('client_id'),
                    client_secret=token_data.get('client_secret'),
                    scopes=token_data.get('scopes', self.config.scopes)
                )
            else:
                # Use standard file-based storage
                if not os.path.exists(self.config.token_file):
                    return False
                
                self.credentials = Credentials.from_authorized_user_file(
                    self.config.token_file, 
                    self.config.scopes
                )
            
            # Check if credentials are valid or can be refreshed
            if self.credentials.valid:
                return True
            elif self.credentials.expired and self.credentials.refresh_token:
                return self.refresh_token()
            else:
                self.logger.warning("Stored credentials are invalid and cannot be refreshed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading existing credentials: {e}")
            return False
    
    def _run_oauth_flow(self) -> bool:
        """Run the OAuth2 authorization flow.
        
        Returns:
            True if OAuth flow completed successfully, False otherwise.
        """
        if not os.path.exists(self.config.credentials_file):
            self.logger.error(f"Credentials file not found: {self.config.credentials_file}")
            return False
        
        try:
            # Create the flow using the client secrets file
            flow = InstalledAppFlow.from_client_secrets_file(
                self.config.credentials_file, 
                self.config.scopes
            )
            
            # Run the OAuth flow
            self.logger.info("Starting OAuth2 authorization flow...")
            self.credentials = flow.run_local_server(port=0)
            
            # Save the credentials for future use
            self._save_credentials()
            self.logger.info("OAuth2 flow completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"OAuth flow failed: {e}")
            return False
    
    def _save_credentials(self) -> None:
        """Save credentials to token file."""
        if not self.credentials:
            return
        
        try:
            if self.config.use_encryption and self.token_storage:
                # Save using secure encrypted storage
                token_data = {
                    'token': self.credentials.token,
                    'refresh_token': self.credentials.refresh_token,
                    'token_uri': self.credentials.token_uri,
                    'client_id': self.credentials.client_id,
                    'client_secret': self.credentials.client_secret,
                    'scopes': self.credentials.scopes
                }
                
                if self.token_storage.save_token(token_data):
                    self.logger.debug(f"Credentials saved securely to {self.config.token_file}")
                else:
                    self.logger.error("Failed to save credentials securely")
            else:
                # Use standard file-based storage
                with open(self.config.token_file, 'w') as token_file:
                    token_file.write(self.credentials.to_json())
                self.logger.debug(f"Credentials saved to {self.config.token_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving credentials: {e}")
    
    def _clear_stored_credentials(self) -> None:
        """Clear stored credentials file."""
        try:
            if self.config.use_encryption and self.token_storage:
                # Clear secure storage
                if self.token_storage.delete_token():
                    self.logger.debug("Stored credentials cleared securely")
                else:
                    self.logger.error("Failed to clear stored credentials securely")
            else:
                # Clear standard file
                if os.path.exists(self.config.token_file):
                    os.remove(self.config.token_file)
                    self.logger.debug("Stored credentials cleared")
        except Exception as e:
            self.logger.error(f"Error clearing stored credentials: {e}")
    
    def get_auth_state(self) -> Dict[str, Any]:
        """Get current authentication state information.
        
        Returns:
            Dictionary containing authentication state details.
        """
        state = {
            'authenticated': self.is_authenticated(),
            'has_credentials': self.credentials is not None,
            'credentials_file_exists': os.path.exists(self.config.credentials_file),
            'scopes': self.config.scopes,
            'encryption_enabled': self.config.use_encryption
        }
        
        # Check token file existence based on storage type
        if self.config.use_encryption and self.token_storage:
            state['token_file_exists'] = self.token_storage.token_exists()
            state['token_integrity_valid'] = self.token_storage.validate_token_integrity()
        else:
            state['token_file_exists'] = os.path.exists(self.config.token_file)
        
        if self.credentials:
            state.update({
                'credentials_valid': self.credentials.valid,
                'credentials_expired': self.credentials.expired,
                'has_refresh_token': bool(self.credentials.refresh_token)
            })
        
        return state
    
    def validate_stored_token(self) -> bool:
        """Validate the integrity of stored token.
        
        Returns:
            True if stored token is valid and can be loaded, False otherwise.
        """
        if self.config.use_encryption and self.token_storage:
            return self.token_storage.validate_token_integrity()
        else:
            return os.path.exists(self.config.token_file)
    
    def handle_authentication_error(self, error: Exception) -> Dict[str, Any]:
        """Handle authentication errors and provide recovery suggestions.
        
        Args:
            error: The authentication error that occurred
            
        Returns:
            Dictionary with error details and recovery suggestions.
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'recovery_suggestions': []
        }
        
        if isinstance(error, RefreshError):
            error_info['recovery_suggestions'].extend([
                'The refresh token has expired or been revoked',
                'Run authentication flow again to get new tokens',
                'Check if the application has been removed from Google account'
            ])
        elif 'decrypt' in str(error).lower():
            error_info['recovery_suggestions'].extend([
                'Token decryption failed - password may be incorrect',
                'Try re-entering the encryption password',
                'Consider clearing stored tokens and re-authenticating'
            ])
        else:
            error_info['recovery_suggestions'].extend([
                'Check internet connection',
                'Verify credentials file exists and is valid',
                'Try re-running the authentication flow'
            ])
        
        return error_info