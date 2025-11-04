"""Secure token storage with encryption for OAuth2 credentials."""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import getpass


class SecureTokenStorage:
    """Handles secure storage and retrieval of OAuth2 tokens with encryption."""
    
    def __init__(self, token_file: str, password: Optional[str] = None):
        """Initialize secure token storage.
        
        Args:
            token_file: Path to the token file
            password: Optional password for encryption. If None, will prompt user.
        """
        self.token_file = Path(token_file)
        self.key_file = self.token_file.with_suffix('.key')
        self.logger = logging.getLogger(__name__)
        self._password = password
        self._fernet: Optional[Fernet] = None
        
        # Ensure directory exists
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_password(self) -> str:
        """Get password for encryption/decryption."""
        if self._password:
            return self._password
        
        # Prompt user for password
        return getpass.getpass("Enter password for token encryption: ")
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _get_or_create_fernet(self) -> Fernet:
        """Get or create Fernet encryption instance."""
        if self._fernet:
            return self._fernet
        
        password = self._get_password()
        
        # Check if key file exists
        if self.key_file.exists():
            # Load existing salt
            with open(self.key_file, 'rb') as f:
                salt = f.read()
        else:
            # Generate new salt
            salt = os.urandom(16)
            with open(self.key_file, 'wb') as f:
                f.write(salt)
            # Secure the key file
            os.chmod(self.key_file, 0o600)
        
        key = self._derive_key(password, salt)
        self._fernet = Fernet(key)
        return self._fernet
    
    def save_token(self, token_data: Dict[str, Any]) -> bool:
        """Save token data with encryption.
        
        Args:
            token_data: Token data to save
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            fernet = self._get_or_create_fernet()
            
            # Convert to JSON and encrypt
            json_data = json.dumps(token_data).encode()
            encrypted_data = fernet.encrypt(json_data)
            
            # Write encrypted data to file
            with open(self.token_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure the token file
            os.chmod(self.token_file, 0o600)
            
            self.logger.debug(f"Token saved securely to {self.token_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving token: {e}")
            return False
    
    def load_token(self) -> Optional[Dict[str, Any]]:
        """Load and decrypt token data.
        
        Returns:
            Token data if successful, None otherwise
        """
        if not self.token_file.exists():
            return None
        
        try:
            fernet = self._get_or_create_fernet()
            
            # Read and decrypt data
            with open(self.token_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())
            
            self.logger.debug("Token loaded successfully")
            return token_data
            
        except Exception as e:
            self.logger.error(f"Error loading token: {e}")
            return None
    
    def delete_token(self) -> bool:
        """Delete stored token and key files.
        
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                self.logger.debug("Token file deleted")
            
            if self.key_file.exists():
                self.key_file.unlink()
                self.logger.debug("Key file deleted")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting token files: {e}")
            return False
    
    def token_exists(self) -> bool:
        """Check if token file exists.
        
        Returns:
            True if token file exists, False otherwise
        """
        return self.token_file.exists()
    
    def validate_token_integrity(self) -> bool:
        """Validate that stored token can be decrypted.
        
        Returns:
            True if token is valid and can be decrypted, False otherwise
        """
        try:
            token_data = self.load_token()
            return token_data is not None
        except Exception:
            return False


class TokenValidator:
    """Validates OAuth2 tokens and handles refresh logic."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_token_valid(self, token_data: Dict[str, Any]) -> bool:
        """Check if token data is valid.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        required_fields = ['token', 'refresh_token', 'token_uri', 'client_id', 'client_secret']
        
        for field in required_fields:
            if field not in token_data:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        return True
    
    def is_token_expired(self, token_data: Dict[str, Any]) -> bool:
        """Check if token is expired.
        
        Args:
            token_data: Token data to check
            
        Returns:
            True if token is expired, False otherwise
        """
        # This would typically check the expiry time
        # For now, we'll rely on the Google auth library to handle this
        return False
    
    def needs_refresh(self, token_data: Dict[str, Any]) -> bool:
        """Check if token needs to be refreshed.
        
        Args:
            token_data: Token data to check
            
        Returns:
            True if token needs refresh, False otherwise
        """
        return self.is_token_expired(token_data) and 'refresh_token' in token_data