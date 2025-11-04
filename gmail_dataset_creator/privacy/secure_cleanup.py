"""
Secure data cleanup utilities for Gmail Dataset Creator.

This module provides secure deletion and cleanup of temporary data, tokens,
and other sensitive information after processing completion.
"""

import os
import shutil
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CleanupPolicy:
    """Configuration for data cleanup policies."""
    delete_temp_files: bool = True
    delete_raw_emails: bool = False
    delete_tokens_on_completion: bool = False
    secure_delete_passes: int = 3
    retention_days: int = 0  # 0 means delete immediately
    preserve_dataset: bool = True
    preserve_logs: bool = True


class SecureDataCleanup:
    """
    Handles secure cleanup of temporary data and sensitive information.
    
    Provides secure deletion methods and data retention policy enforcement
    to ensure sensitive data is properly removed after processing.
    """
    
    def __init__(self, cleanup_policy: Optional[CleanupPolicy] = None):
        """
        Initialize secure data cleanup.
        
        Args:
            cleanup_policy: Cleanup policy configuration
        """
        self.policy = cleanup_policy or CleanupPolicy()
        self.logger = logging.getLogger(__name__)
        
        # Track files and directories for cleanup
        self.temp_files: Set[Path] = set()
        self.temp_directories: Set[Path] = set()
        self.sensitive_files: Set[Path] = set()
        
        # Cleanup statistics
        self.cleanup_stats = {
            'files_deleted': 0,
            'directories_deleted': 0,
            'bytes_cleaned': 0,
            'secure_deletions': 0,
            'errors': 0
        }
    
    def register_temp_file(self, file_path: str) -> None:
        """
        Register a temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file
        """
        path = Path(file_path)
        self.temp_files.add(path)
        self.logger.debug(f"Registered temp file for cleanup: {path}")
    
    def register_temp_directory(self, dir_path: str) -> None:
        """
        Register a temporary directory for cleanup.
        
        Args:
            dir_path: Path to temporary directory
        """
        path = Path(dir_path)
        self.temp_directories.add(path)
        self.logger.debug(f"Registered temp directory for cleanup: {path}")
    
    def register_sensitive_file(self, file_path: str) -> None:
        """
        Register a sensitive file for secure cleanup.
        
        Args:
            file_path: Path to sensitive file
        """
        path = Path(file_path)
        self.sensitive_files.add(path)
        self.logger.debug(f"Registered sensitive file for cleanup: {path}")
    
    def cleanup_temp_files(self) -> bool:
        """
        Clean up registered temporary files.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        if not self.policy.delete_temp_files:
            self.logger.info("Temp file cleanup disabled by policy")
            return True
        
        success = True
        
        # Clean up temporary files
        for file_path in self.temp_files.copy():
            if self._cleanup_file(file_path):
                self.temp_files.remove(file_path)
            else:
                success = False
        
        # Clean up temporary directories
        for dir_path in self.temp_directories.copy():
            if self._cleanup_directory(dir_path):
                self.temp_directories.remove(dir_path)
            else:
                success = False
        
        return success
    
    def cleanup_sensitive_data(self) -> bool:
        """
        Securely clean up sensitive files.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        success = True
        
        for file_path in self.sensitive_files.copy():
            if self._secure_delete_file(file_path):
                self.sensitive_files.remove(file_path)
            else:
                success = False
        
        return success
    
    def cleanup_tokens(self, token_storage_path: str) -> bool:
        """
        Clean up authentication tokens if policy allows.
        
        Args:
            token_storage_path: Path to token storage
            
        Returns:
            True if cleanup successful, False otherwise
        """
        if not self.policy.delete_tokens_on_completion:
            self.logger.info("Token cleanup disabled by policy")
            return True
        
        token_path = Path(token_storage_path)
        
        # Also clean up key file if it exists
        key_path = token_path.with_suffix('.key')
        
        success = True
        
        if token_path.exists():
            if not self._secure_delete_file(token_path):
                success = False
        
        if key_path.exists():
            if not self._secure_delete_file(key_path):
                success = False
        
        return success
    
    def cleanup_raw_email_data(self, data_directory: str) -> bool:
        """
        Clean up raw email data if policy allows.
        
        Args:
            data_directory: Directory containing raw email data
            
        Returns:
            True if cleanup successful, False otherwise
        """
        if not self.policy.delete_raw_emails:
            self.logger.info("Raw email cleanup disabled by policy")
            return True
        
        data_path = Path(data_directory)
        
        if not data_path.exists():
            return True
        
        # Look for raw email files (typically .json or .pkl files)
        raw_files = []
        for pattern in ['*.json', '*.pkl', '*.raw']:
            raw_files.extend(data_path.glob(pattern))
        
        success = True
        for file_path in raw_files:
            if not self._secure_delete_file(file_path):
                success = False
        
        return success
    
    def apply_retention_policy(self, base_directory: str) -> bool:
        """
        Apply data retention policy to remove old files.
        
        Args:
            base_directory: Base directory to apply retention policy
            
        Returns:
            True if policy applied successfully, False otherwise
        """
        if self.policy.retention_days <= 0:
            return True  # No retention policy
        
        base_path = Path(base_directory)
        if not base_path.exists():
            return True
        
        cutoff_date = datetime.now() - timedelta(days=self.policy.retention_days)
        success = True
        
        # Find files older than retention period
        for file_path in base_path.rglob('*'):
            if file_path.is_file():
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        # Check if file should be preserved
                        if self._should_preserve_file(file_path):
                            continue
                        
                        if not self._cleanup_file(file_path):
                            success = False
                except Exception as e:
                    self.logger.error(f"Error checking file {file_path}: {e}")
                    success = False
        
        return success
    
    def _should_preserve_file(self, file_path: Path) -> bool:
        """
        Check if a file should be preserved based on policy.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file should be preserved, False otherwise
        """
        file_name = file_path.name.lower()
        
        # Preserve dataset files
        if self.policy.preserve_dataset:
            if any(pattern in file_name for pattern in ['dataset', 'train', 'test', 'vocab']):
                return True
        
        # Preserve log files
        if self.policy.preserve_logs:
            if file_name.endswith('.log') or 'log' in file_name:
                return True
        
        return False
    
    def _cleanup_file(self, file_path: Path) -> bool:
        """
        Clean up a single file.
        
        Args:
            file_path: Path to file to clean up
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            file_path.unlink()
            
            self.cleanup_stats['files_deleted'] += 1
            self.cleanup_stats['bytes_cleaned'] += file_size
            
            self.logger.debug(f"Cleaned up file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up file {file_path}: {e}")
            self.cleanup_stats['errors'] += 1
            return False
    
    def _cleanup_directory(self, dir_path: Path) -> bool:
        """
        Clean up a directory and its contents.
        
        Args:
            dir_path: Path to directory to clean up
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if not dir_path.exists():
                return True
            
            # Calculate total size before deletion
            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            
            shutil.rmtree(dir_path)
            
            self.cleanup_stats['directories_deleted'] += 1
            self.cleanup_stats['bytes_cleaned'] += total_size
            
            self.logger.debug(f"Cleaned up directory: {dir_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up directory {dir_path}: {e}")
            self.cleanup_stats['errors'] += 1
            return False
    
    def _secure_delete_file(self, file_path: Path) -> bool:
        """
        Securely delete a file by overwriting it multiple times.
        
        Args:
            file_path: Path to file to securely delete
            
        Returns:
            True if secure deletion successful, False otherwise
        """
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            
            # Perform multiple overwrite passes
            with open(file_path, 'r+b') as f:
                for pass_num in range(self.policy.secure_delete_passes):
                    f.seek(0)
                    
                    # Different patterns for each pass
                    if pass_num == 0:
                        pattern = b'\x00' * 1024  # Zeros
                    elif pass_num == 1:
                        pattern = b'\xFF' * 1024  # Ones
                    else:
                        pattern = os.urandom(1024)  # Random
                    
                    # Overwrite file in chunks
                    bytes_written = 0
                    while bytes_written < file_size:
                        chunk_size = min(1024, file_size - bytes_written)
                        f.write(pattern[:chunk_size])
                        bytes_written += chunk_size
                    
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            
            # Finally delete the file
            file_path.unlink()
            
            self.cleanup_stats['files_deleted'] += 1
            self.cleanup_stats['bytes_cleaned'] += file_size
            self.cleanup_stats['secure_deletions'] += 1
            
            self.logger.debug(f"Securely deleted file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error securely deleting file {file_path}: {e}")
            self.cleanup_stats['errors'] += 1
            return False
    
    def create_temp_directory(self, prefix: str = "gmail_dataset_") -> str:
        """
        Create a temporary directory and register it for cleanup.
        
        Args:
            prefix: Prefix for temporary directory name
            
        Returns:
            Path to created temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.register_temp_directory(temp_dir)
        return temp_dir
    
    def create_temp_file(self, suffix: str = ".tmp", prefix: str = "gmail_dataset_") -> str:
        """
        Create a temporary file and register it for cleanup.
        
        Args:
            suffix: Suffix for temporary file name
            prefix: Prefix for temporary file name
            
        Returns:
            Path to created temporary file
        """
        fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close the file descriptor
        self.register_temp_file(temp_file)
        return temp_file
    
    def perform_full_cleanup(self, 
                           token_storage_path: Optional[str] = None,
                           data_directory: Optional[str] = None,
                           base_directory: Optional[str] = None) -> bool:
        """
        Perform complete cleanup based on policy settings.
        
        Args:
            token_storage_path: Path to token storage
            data_directory: Directory containing raw data
            base_directory: Base directory for retention policy
            
        Returns:
            True if all cleanup operations successful, False otherwise
        """
        success = True
        
        self.logger.info("Starting full cleanup process")
        
        # Clean up temporary files
        if not self.cleanup_temp_files():
            success = False
        
        # Clean up sensitive data
        if not self.cleanup_sensitive_data():
            success = False
        
        # Clean up tokens if specified
        if token_storage_path and not self.cleanup_tokens(token_storage_path):
            success = False
        
        # Clean up raw email data if specified
        if data_directory and not self.cleanup_raw_email_data(data_directory):
            success = False
        
        # Apply retention policy if specified
        if base_directory and not self.apply_retention_policy(base_directory):
            success = False
        
        # Log cleanup statistics
        self._log_cleanup_stats()
        
        if success:
            self.logger.info("Full cleanup completed successfully")
        else:
            self.logger.warning("Full cleanup completed with some errors")
        
        return success
    
    def _log_cleanup_stats(self):
        """Log cleanup statistics."""
        stats = self.cleanup_stats
        self.logger.info(f"Cleanup statistics:")
        self.logger.info(f"  Files deleted: {stats['files_deleted']}")
        self.logger.info(f"  Directories deleted: {stats['directories_deleted']}")
        self.logger.info(f"  Bytes cleaned: {stats['bytes_cleaned']:,}")
        self.logger.info(f"  Secure deletions: {stats['secure_deletions']}")
        if stats['errors'] > 0:
            self.logger.warning(f"  Errors encountered: {stats['errors']}")
    
    def get_cleanup_stats(self) -> Dict[str, int]:
        """
        Get cleanup statistics.
        
        Returns:
            Dictionary of cleanup statistics
        """
        return self.cleanup_stats.copy()