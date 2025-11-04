"""
Secure export utilities for generated datasets.

This module provides secure export options for datasets including encryption,
compression, integrity verification, and secure transfer capabilities.
"""

import os
import json
import gzip
import tarfile
import zipfile
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .encryption_manager import EncryptionManager, EncryptionConfig


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar.gz"


class SecurityLevel(Enum):
    """Security levels for export."""
    BASIC = "basic"          # No encryption, basic integrity checks
    STANDARD = "standard"    # Encryption, integrity verification
    HIGH = "high"           # Strong encryption, multiple integrity checks
    MAXIMUM = "maximum"     # Maximum security with all features


@dataclass
class ExportConfig:
    """Configuration for secure export operations."""
    format: ExportFormat = ExportFormat.JSONL
    compression: CompressionType = CompressionType.GZIP
    security_level: SecurityLevel = SecurityLevel.STANDARD
    encrypt_data: bool = True
    include_metadata: bool = True
    include_checksums: bool = True
    split_large_files: bool = True
    max_file_size_mb: int = 100
    password_protect: bool = True
    remove_source_after_export: bool = False


class SecureExporter:
    """
    Handles secure export of datasets with encryption and integrity verification.
    
    Provides comprehensive export capabilities including multiple formats,
    compression options, encryption, and integrity verification.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize secure exporter.
        
        Args:
            config: Export configuration settings
        """
        self.config = config or ExportConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption manager based on security level
        self.encryption_manager = self._create_encryption_manager()
        
        # Export statistics
        self.export_stats = {
            'files_exported': 0,
            'total_size_bytes': 0,
            'compression_ratio': 0.0,
            'export_time_seconds': 0.0,
            'integrity_checks_passed': 0
        }
    
    def _create_encryption_manager(self) -> Optional[EncryptionManager]:
        """Create encryption manager based on security level."""
        if not self.config.encrypt_data:
            return None
        
        if self.config.security_level == SecurityLevel.BASIC:
            return None  # No encryption for basic level
        
        # Configure encryption based on security level
        if self.config.security_level == SecurityLevel.MAXIMUM:
            encryption_config = EncryptionConfig(
                iterations=200000,  # Higher iterations for maximum security
                salt_length=64,     # Longer salt
                key_length=32       # AES-256
            )
        elif self.config.security_level == SecurityLevel.HIGH:
            encryption_config = EncryptionConfig(
                iterations=150000,
                salt_length=48,
                key_length=32
            )
        else:  # STANDARD
            encryption_config = EncryptionConfig()
        
        return EncryptionManager(encryption_config)
    
    def export_dataset(self, 
                      dataset_path: str,
                      output_path: str,
                      password: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export dataset with security measures.
        
        Args:
            dataset_path: Path to dataset directory or file
            output_path: Output path for exported dataset
            password: Optional password for encryption
            metadata: Optional metadata to include
            
        Returns:
            Dictionary with export results and statistics
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            self._validate_export_inputs(dataset_path, output_path, password)
            
            # Prepare export directory
            export_dir = self._prepare_export_directory(output_path)
            
            # Process dataset files
            processed_files = self._process_dataset_files(dataset_path, export_dir)
            
            # Apply compression if requested
            if self.config.compression != CompressionType.NONE:
                processed_files = self._compress_files(processed_files, export_dir)
            
            # Apply encryption if requested
            if self.config.encrypt_data and self.encryption_manager and password:
                processed_files = self._encrypt_files(processed_files, password)
            
            # Generate metadata and checksums
            export_metadata = self._generate_export_metadata(
                processed_files, metadata, start_time
            )
            
            # Create integrity verification files
            if self.config.include_checksums:
                self._create_integrity_files(processed_files, export_dir)
            
            # Save export metadata
            if self.config.include_metadata:
                self._save_export_metadata(export_metadata, export_dir)
            
            # Clean up source files if requested
            if self.config.remove_source_after_export:
                self._cleanup_source_files(dataset_path)
            
            # Calculate final statistics
            end_time = datetime.now()
            self.export_stats['export_time_seconds'] = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'export_path': str(export_dir),
                'files_exported': processed_files,
                'metadata': export_metadata,
                'statistics': self.export_stats.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'statistics': self.export_stats.copy()
            }
    
    def _validate_export_inputs(self, dataset_path: str, output_path: str, 
                               password: Optional[str]) -> None:
        """Validate export inputs."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        if self.config.encrypt_data and not password:
            if self.config.password_protect:
                raise ValueError("Password required for encrypted export")
        
        # Validate output path is writable
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def _prepare_export_directory(self, output_path: str) -> Path:
        """Prepare export directory."""
        export_dir = Path(output_path)
        
        if export_dir.exists():
            # Create timestamped subdirectory to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = export_dir / f"export_{timestamp}"
        
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir
    
    def _process_dataset_files(self, dataset_path: str, export_dir: Path) -> List[Path]:
        """Process and copy dataset files."""
        dataset_path = Path(dataset_path)
        processed_files = []
        
        if dataset_path.is_file():
            # Single file
            processed_file = self._process_single_file(dataset_path, export_dir)
            processed_files.append(processed_file)
        else:
            # Directory with multiple files
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    processed_file = self._process_single_file(file_path, export_dir)
                    processed_files.append(processed_file)
        
        return processed_files
    
    def _process_single_file(self, file_path: Path, export_dir: Path) -> Path:
        """Process a single file for export."""
        # Determine output filename
        relative_path = file_path.name
        output_path = export_dir / relative_path
        
        # Handle large file splitting if enabled
        if self.config.split_large_files:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                return self._split_large_file(file_path, export_dir)
        
        # Copy file to export directory
        import shutil
        shutil.copy2(file_path, output_path)
        
        # Update statistics
        self.export_stats['files_exported'] += 1
        self.export_stats['total_size_bytes'] += file_path.stat().st_size
        
        return output_path
    
    def _split_large_file(self, file_path: Path, export_dir: Path) -> Path:
        """Split large file into smaller chunks."""
        chunk_size = self.config.max_file_size_mb * 1024 * 1024
        base_name = file_path.stem
        extension = file_path.suffix
        
        chunk_files = []
        chunk_index = 0
        
        with open(file_path, 'rb') as input_file:
            while True:
                chunk_data = input_file.read(chunk_size)
                if not chunk_data:
                    break
                
                chunk_filename = f"{base_name}.part{chunk_index:03d}{extension}"
                chunk_path = export_dir / chunk_filename
                
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)
                
                chunk_files.append(chunk_path)
                chunk_index += 1
        
        # Create manifest file for reassembly
        manifest_path = export_dir / f"{base_name}.manifest"
        manifest_data = {
            'original_file': file_path.name,
            'chunks': [f.name for f in chunk_files],
            'total_size': file_path.stat().st_size,
            'chunk_size': chunk_size
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        self.logger.info(f"Split large file {file_path.name} into {len(chunk_files)} chunks")
        return manifest_path
    
    def _compress_files(self, files: List[Path], export_dir: Path) -> List[Path]:
        """Compress files based on compression type."""
        if self.config.compression == CompressionType.GZIP:
            return self._compress_gzip(files)
        elif self.config.compression == CompressionType.ZIP:
            return self._compress_zip(files, export_dir)
        elif self.config.compression == CompressionType.TAR_GZ:
            return self._compress_tar_gz(files, export_dir)
        else:
            return files
    
    def _compress_gzip(self, files: List[Path]) -> List[Path]:
        """Compress files using gzip."""
        compressed_files = []
        
        for file_path in files:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Remove original file
            file_path.unlink()
            compressed_files.append(compressed_path)
        
        return compressed_files
    
    def _compress_zip(self, files: List[Path], export_dir: Path) -> List[Path]:
        """Compress files into a ZIP archive."""
        zip_path = export_dir / "dataset.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                zipf.write(file_path, file_path.name)
                file_path.unlink()  # Remove original
        
        return [zip_path]
    
    def _compress_tar_gz(self, files: List[Path], export_dir: Path) -> List[Path]:
        """Compress files into a tar.gz archive."""
        tar_path = export_dir / "dataset.tar.gz"
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            for file_path in files:
                tar.add(file_path, arcname=file_path.name)
                file_path.unlink()  # Remove original
        
        return [tar_path]
    
    def _encrypt_files(self, files: List[Path], password: str) -> List[Path]:
        """Encrypt files using encryption manager."""
        if not self.encryption_manager:
            return files
        
        encrypted_files = []
        
        for file_path in files:
            try:
                encrypted_path = self.encryption_manager.encrypt_file(
                    str(file_path), password
                )
                
                # Remove original file
                file_path.unlink()
                encrypted_files.append(Path(encrypted_path))
                
            except Exception as e:
                self.logger.error(f"Failed to encrypt {file_path}: {e}")
                encrypted_files.append(file_path)  # Keep original on error
        
        return encrypted_files
    
    def _generate_export_metadata(self, files: List[Path], 
                                 user_metadata: Optional[Dict[str, Any]],
                                 start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive export metadata."""
        metadata = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'start_time': start_time.isoformat(),
                'format': self.config.format.value,
                'compression': self.config.compression.value,
                'security_level': self.config.security_level.value,
                'encrypted': self.config.encrypt_data,
                'password_protected': self.config.password_protect
            },
            'files': [],
            'statistics': self.export_stats.copy()
        }
        
        # Add file information
        for file_path in files:
            if file_path.exists():
                file_info = {
                    'name': file_path.name,
                    'size_bytes': file_path.stat().st_size,
                    'checksum_sha256': self._calculate_file_checksum(file_path),
                    'modified_time': datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat()
                }
                metadata['files'].append(file_info)
        
        # Add user metadata if provided
        if user_metadata:
            metadata['user_metadata'] = user_metadata
        
        # Add encryption information if applicable
        if self.encryption_manager:
            metadata['encryption_info'] = self.encryption_manager.get_encryption_info()
        
        return metadata
    
    def _calculate_file_checksum(self, file_path: Path, 
                                algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _create_integrity_files(self, files: List[Path], export_dir: Path) -> None:
        """Create integrity verification files."""
        checksums = {}
        
        for file_path in files:
            if file_path.exists():
                checksums[file_path.name] = {
                    'sha256': self._calculate_file_checksum(file_path, 'sha256'),
                    'size': file_path.stat().st_size
                }
        
        # Save checksums file
        checksums_path = export_dir / "checksums.json"
        with open(checksums_path, 'w') as f:
            json.dump(checksums, f, indent=2)
        
        # Create SHA256SUMS file (standard format)
        sha256sums_path = export_dir / "SHA256SUMS"
        with open(sha256sums_path, 'w') as f:
            for filename, info in checksums.items():
                f.write(f"{info['sha256']}  {filename}\n")
        
        self.export_stats['integrity_checks_passed'] += len(checksums)
    
    def _save_export_metadata(self, metadata: Dict[str, Any], 
                             export_dir: Path) -> None:
        """Save export metadata to file."""
        metadata_path = export_dir / "export_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Export metadata saved to {metadata_path}")
    
    def _cleanup_source_files(self, dataset_path: str) -> None:
        """Clean up source files after export."""
        try:
            dataset_path = Path(dataset_path)
            
            if dataset_path.is_file():
                dataset_path.unlink()
            else:
                import shutil
                shutil.rmtree(dataset_path)
            
            self.logger.info(f"Source files cleaned up: {dataset_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup source files: {e}")
    
    def verify_export_integrity(self, export_path: str) -> Dict[str, Any]:
        """
        Verify the integrity of an exported dataset.
        
        Args:
            export_path: Path to exported dataset
            
        Returns:
            Dictionary with verification results
        """
        export_dir = Path(export_path)
        
        if not export_dir.exists():
            return {'success': False, 'error': 'Export directory not found'}
        
        results = {
            'success': True,
            'files_verified': 0,
            'files_failed': 0,
            'errors': []
        }
        
        # Load checksums if available
        checksums_path = export_dir / "checksums.json"
        if checksums_path.exists():
            with open(checksums_path, 'r') as f:
                expected_checksums = json.load(f)
            
            # Verify each file
            for filename, expected_info in expected_checksums.items():
                file_path = export_dir / filename
                
                if not file_path.exists():
                    results['errors'].append(f"Missing file: {filename}")
                    results['files_failed'] += 1
                    continue
                
                # Verify checksum
                actual_checksum = self._calculate_file_checksum(file_path)
                if actual_checksum != expected_info['sha256']:
                    results['errors'].append(f"Checksum mismatch: {filename}")
                    results['files_failed'] += 1
                    continue
                
                # Verify size
                actual_size = file_path.stat().st_size
                if actual_size != expected_info['size']:
                    results['errors'].append(f"Size mismatch: {filename}")
                    results['files_failed'] += 1
                    continue
                
                results['files_verified'] += 1
        
        results['success'] = results['files_failed'] == 0
        return results
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        return self.export_stats.copy()