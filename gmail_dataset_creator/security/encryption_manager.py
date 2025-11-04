"""
Enhanced encryption management for sensitive data.

This module provides comprehensive encryption capabilities for authentication tokens,
dataset files, and other sensitive information with multiple encryption algorithms
and key management features.
"""

import os
import json
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    FERNET = "fernet"
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"


class KeyDerivationFunction(Enum):
    """Supported key derivation functions."""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2
    iterations: int = 100000
    salt_length: int = 32
    key_length: int = 32
    use_hardware_rng: bool = True


class EncryptionManager:
    """
    Advanced encryption manager for sensitive data protection.
    
    Provides multiple encryption algorithms, secure key derivation,
    and comprehensive key management for protecting sensitive data.
    """
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize encryption manager.
        
        Args:
            config: Encryption configuration settings
        """
        self.config = config or EncryptionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Key cache for performance
        self._key_cache: Dict[str, bytes] = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate encryption configuration."""
        if self.config.iterations < 10000:
            raise ValueError("Iterations must be at least 10,000 for security")
        
        if self.config.salt_length < 16:
            raise ValueError("Salt length must be at least 16 bytes")
        
        if self.config.key_length not in [16, 24, 32]:
            raise ValueError("Key length must be 16, 24, or 32 bytes")
    
    def generate_salt(self) -> bytes:
        """
        Generate cryptographically secure salt.
        
        Returns:
            Random salt bytes
        """
        if self.config.use_hardware_rng:
            return secrets.token_bytes(self.config.salt_length)
        else:
            return os.urandom(self.config.salt_length)
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password and salt.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            
        Returns:
            Derived encryption key
        """
        password_bytes = password.encode('utf-8')
        
        if self.config.kdf == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                iterations=self.config.iterations,
                backend=default_backend()
            )
        elif self.config.kdf == KeyDerivationFunction.SCRYPT:
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,      # Block size parameter
                p=1,      # Parallelization parameter
                backend=default_backend()
            )
        else:
            raise ValueError(f"Unsupported KDF: {self.config.kdf}")
        
        return kdf.derive(password_bytes)
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]], 
                    password: str, 
                    salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Encrypt data using specified algorithm.
        
        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable dict)
            password: Password for encryption
            salt: Optional salt (will generate if not provided)
            
        Returns:
            Tuple of (encrypted_data, salt)
        """
        # Convert data to bytes if necessary
        if isinstance(data, dict):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        # Generate salt if not provided
        if salt is None:
            salt = self.generate_salt()
        
        # Derive encryption key
        key = self.derive_key(password, salt)
        
        # Encrypt based on algorithm
        if self.config.algorithm == EncryptionAlgorithm.FERNET:
            encrypted_data = self._encrypt_fernet(data_bytes, key)
        elif self.config.algorithm == EncryptionAlgorithm.AES_256_GCM:
            encrypted_data = self._encrypt_aes_gcm(data_bytes, key)
        elif self.config.algorithm == EncryptionAlgorithm.AES_256_CBC:
            encrypted_data = self._encrypt_aes_cbc(data_bytes, key)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        return encrypted_data, salt
    
    def decrypt_data(self, encrypted_data: bytes, 
                    password: str, 
                    salt: bytes) -> bytes:
        """
        Decrypt data using specified algorithm.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            password: Password for decryption
            salt: Salt used for key derivation
            
        Returns:
            Decrypted data as bytes
        """
        # Derive decryption key
        key = self.derive_key(password, salt)
        
        # Decrypt based on algorithm
        if self.config.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_data, key)
        elif self.config.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data, key)
        elif self.config.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def _encrypt_fernet(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using Fernet algorithm."""
        # Fernet requires base64-encoded 32-byte key
        fernet_key = base64.urlsafe_b64encode(key[:32])
        fernet = Fernet(fernet_key)
        return fernet.encrypt(data)
    
    def _decrypt_fernet(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using Fernet algorithm."""
        fernet_key = base64.urlsafe_b64encode(key[:32])
        fernet = Fernet(fernet_key)
        return fernet.decrypt(encrypted_data)
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        # Generate random IV
        iv = os.urandom(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine IV, tag, and ciphertext
        return iv + encryptor.tag + ciphertext
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        # Extract IV, tag, and ciphertext
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-256-CBC with PKCS7 padding."""
        # Generate random IV
        iv = os.urandom(16)  # 128-bit IV for CBC
        
        # Add PKCS7 padding
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and ciphertext
        return iv + ciphertext
    
    def _decrypt_aes_cbc(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-256-CBC with PKCS7 padding."""
        # Extract IV and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove PKCS7 padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def encrypt_file(self, file_path: str, password: str, 
                    output_path: Optional[str] = None) -> str:
        """
        Encrypt a file.
        
        Args:
            file_path: Path to file to encrypt
            password: Password for encryption
            output_path: Optional output path (defaults to file_path + .enc)
            
        Returns:
            Path to encrypted file
        """
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if output_path is None:
            output_path = str(input_path) + ".enc"
        
        # Read file data
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        # Encrypt data
        encrypted_data, salt = self.encrypt_data(file_data, password)
        
        # Create encrypted file format: [salt][encrypted_data]
        with open(output_path, 'wb') as f:
            f.write(salt)
            f.write(encrypted_data)
        
        self.logger.info(f"File encrypted: {file_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, password: str,
                    output_path: Optional[str] = None) -> str:
        """
        Decrypt a file.
        
        Args:
            encrypted_file_path: Path to encrypted file
            password: Password for decryption
            output_path: Optional output path (defaults to removing .enc extension)
            
        Returns:
            Path to decrypted file
        """
        encrypted_path = Path(encrypted_file_path)
        if not encrypted_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_path}")
        
        if output_path is None:
            if encrypted_file_path.endswith('.enc'):
                output_path = encrypted_file_path[:-4]
            else:
                output_path = encrypted_file_path + ".dec"
        
        # Read encrypted file
        with open(encrypted_path, 'rb') as f:
            salt = f.read(self.config.salt_length)
            encrypted_data = f.read()
        
        # Decrypt data
        decrypted_data = self.decrypt_data(encrypted_data, password, salt)
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        self.logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
        return output_path
    
    def generate_secure_password(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure password.
        
        Args:
            length: Length of password to generate
            
        Returns:
            Secure random password
        """
        if self.config.use_hardware_rng:
            return secrets.token_urlsafe(length)
        else:
            return base64.urlsafe_b64encode(os.urandom(length)).decode('utf-8')[:length]
    
    def hash_data(self, data: Union[str, bytes], 
                 algorithm: str = "sha256") -> str:
        """
        Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, blake2b)
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def verify_data_integrity(self, data: Union[str, bytes], 
                             expected_hash: str,
                             algorithm: str = "sha256") -> bool:
        """
        Verify data integrity using hash comparison.
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm used
            
        Returns:
            True if data integrity is verified, False otherwise
        """
        actual_hash = self.hash_data(data, algorithm)
        return secrets.compare_digest(actual_hash, expected_hash)
    
    def clear_key_cache(self):
        """Clear the key cache for security."""
        self._key_cache.clear()
        self.logger.debug("Key cache cleared")
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """
        Get information about current encryption configuration.
        
        Returns:
            Dictionary with encryption configuration details
        """
        return {
            'algorithm': self.config.algorithm.value,
            'kdf': self.config.kdf.value,
            'iterations': self.config.iterations,
            'salt_length': self.config.salt_length,
            'key_length': self.config.key_length,
            'hardware_rng': self.config.use_hardware_rng
        }