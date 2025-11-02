"""
Checkpoint management module for MacBook TRM training optimization.

This module provides robust checkpoint saving with resource monitoring,
checkpoint resumption with configuration validation, and checkpoint cleanup
to manage disk space efficiently on MacBook hardware.
"""

import json
import os
import shutil
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager, MemoryConfig
from .config_management import MacBookTrainingConfig
from .resource_monitoring import ResourceMonitor


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    # Basic info
    checkpoint_id: str
    timestamp: datetime
    step: int
    epoch: int
    
    # Training state
    loss: float
    learning_rate: float
    model_config_hash: str
    
    # System info
    memory_usage_mb: float
    disk_usage_mb: float
    training_time_seconds: float
    
    # Configuration
    training_config: Dict[str, Any]
    hardware_summary: Dict[str, Any]
    
    # File info
    checkpoint_path: str
    model_state_size_mb: float
    optimizer_state_size_mb: float
    
    # Validation
    config_compatible: bool
    validation_errors: List[str]


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    # Storage settings
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 3
    min_disk_space_mb: float = 1000.0  # Minimum free disk space
    
    # Saving intervals
    save_interval_steps: int = 500
    save_interval_minutes: float = 30.0  # Also save every 30 minutes
    memory_aware_intervals: bool = True
    
    # Memory-aware saving
    memory_threshold_for_saving: float = 70.0  # Don't save if memory > 70%
    wait_for_memory_cooldown: bool = True
    max_wait_time_seconds: float = 300.0  # Max 5 minutes wait
    
    # Cleanup settings
    auto_cleanup: bool = True
    cleanup_on_low_disk: bool = True
    disk_space_threshold_mb: float = 2000.0  # Cleanup if less than 2GB free
    
    # Validation
    validate_on_load: bool = True
    strict_config_validation: bool = False
    
    # Compression
    compress_checkpoints: bool = True
    compression_level: int = 6  # 0-9, higher = better compression but slower


@dataclass
class CheckpointSaveResult:
    """Result of checkpoint save operation."""
    success: bool
    checkpoint_id: str
    save_path: str
    save_time_seconds: float
    memory_usage_mb: float
    disk_usage_mb: float
    warnings: List[str]
    errors: List[str]


@dataclass
class CheckpointLoadResult:
    """Result of checkpoint load operation."""
    success: bool
    checkpoint_id: str
    metadata: Optional[CheckpointMetadata]
    model_state: Optional[Dict]
    optimizer_state: Optional[Dict]
    config_compatible: bool
    warnings: List[str]
    errors: List[str]


class CheckpointManager:
    """Robust checkpoint management for MacBook TRM training."""
    
    def __init__(self, 
                 config: Optional[CheckpointConfig] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Checkpoint management configuration
            memory_manager: Memory manager for resource monitoring
            logger: Logger instance
        """
        self.config = config or CheckpointConfig()
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logger or logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.last_save_time = 0.0
        self.last_save_step = 0
        self.save_in_progress = False
        
        # Metadata cache
        self._metadata_cache: Dict[str, CheckpointMetadata] = {}
        self._load_metadata_cache()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
    def _load_metadata_cache(self):
        """Load checkpoint metadata cache from disk."""
        cache_file = self.checkpoint_dir / "metadata_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for checkpoint_id, metadata_dict in cache_data.items():
                    # Convert timestamp string back to datetime
                    metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                    self._metadata_cache[checkpoint_id] = CheckpointMetadata(**metadata_dict)
                    
            except Exception as e:
                self.logger.warning(f"Failed to load metadata cache: {e}")
    
    def _save_metadata_cache(self):
        """Save checkpoint metadata cache to disk."""
        cache_file = self.checkpoint_dir / "metadata_cache.json"
        try:
            cache_data = {}
            for checkpoint_id, metadata in self._metadata_cache.items():
                metadata_dict = asdict(metadata)
                # Convert datetime to string for JSON serialization
                metadata_dict['timestamp'] = metadata.timestamp.isoformat()
                cache_data[checkpoint_id] = metadata_dict
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save metadata cache: {e}")
    
    def _generate_checkpoint_id(self, step: int, epoch: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get full path for checkpoint."""
        return self.checkpoint_dir / f"{checkpoint_id}.pt"
    
    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        """Get full path for checkpoint metadata."""
        return self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for compatibility checking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_disk_usage(self, path: Path) -> float:
        """Get disk usage of file or directory in MB."""
        if path.is_file():
            return path.stat().st_size / (1024**2)
        elif path.is_dir():
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024**2)
        return 0.0
    
    def _get_free_disk_space(self) -> float:
        """Get free disk space in MB."""
        statvfs = os.statvfs(self.checkpoint_dir)
        return (statvfs.f_frsize * statvfs.f_bavail) / (1024**2)
    
    def _wait_for_memory_cooldown(self) -> bool:
        """Wait for memory usage to cool down before saving."""
        if not self.config.wait_for_memory_cooldown:
            return True
        
        start_time = time.time()
        while time.time() - start_time < self.config.max_wait_time_seconds:
            memory_stats = self.memory_manager.monitor_memory_usage()
            if memory_stats.percent_used < self.config.memory_threshold_for_saving:
                return True
            
            # Wait a bit and check again
            time.sleep(5.0)
        
        return False  # Timeout
    
    def should_save_checkpoint(self, step: int, force: bool = False) -> Tuple[bool, str]:
        """
        Determine if checkpoint should be saved.
        
        Args:
            step: Current training step
            force: Force saving regardless of conditions
            
        Returns:
            Tuple of (should_save, reason)
        """
        if force:
            return True, "forced"
        
        if self.save_in_progress:
            return False, "save_in_progress"
        
        current_time = time.time()
        
        # Check step interval
        steps_since_last = step - self.last_save_step
        if steps_since_last >= self.config.save_interval_steps:
            return True, f"step_interval ({steps_since_last} steps)"
        
        # Check time interval
        time_since_last = current_time - self.last_save_time
        if time_since_last >= self.config.save_interval_minutes * 60:
            return True, f"time_interval ({time_since_last/60:.1f} minutes)"
        
        # Check memory-aware conditions
        if self.config.memory_aware_intervals:
            memory_stats = self.memory_manager.monitor_memory_usage()
            
            # Save more frequently if memory pressure is high
            if memory_stats.percent_used > 80 and steps_since_last >= self.config.save_interval_steps // 2:
                return True, f"high_memory_pressure ({memory_stats.percent_used:.1f}%)"
        
        return False, "no_trigger"
    
    def save_checkpoint(self,
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       step: int,
                       epoch: int,
                       loss: float,
                       learning_rate: float,
                       training_config: MacBookTrainingConfig,
                       training_time_seconds: float = 0.0,
                       force: bool = False) -> CheckpointSaveResult:
        """
        Save training checkpoint with resource monitoring.
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            learning_rate: Current learning rate
            training_config: Training configuration
            training_time_seconds: Total training time
            force: Force saving regardless of conditions
            
        Returns:
            CheckpointSaveResult with save details
        """
        start_time = time.time()
        warnings = []
        errors = []
        
        # Check if we should save
        should_save, reason = self.should_save_checkpoint(step, force)
        if not should_save:
            return CheckpointSaveResult(
                success=False,
                checkpoint_id="",
                save_path="",
                save_time_seconds=0.0,
                memory_usage_mb=0.0,
                disk_usage_mb=0.0,
                warnings=[f"Skipped save: {reason}"],
                errors=[]
            )
        
        self.save_in_progress = True
        
        try:
            # Check disk space
            free_space = self._get_free_disk_space()
            if free_space < self.config.min_disk_space_mb:
                if self.config.auto_cleanup:
                    self.cleanup_old_checkpoints()
                    free_space = self._get_free_disk_space()
                
                if free_space < self.config.min_disk_space_mb:
                    errors.append(f"Insufficient disk space: {free_space:.1f}MB < {self.config.min_disk_space_mb}MB")
                    return CheckpointSaveResult(
                        success=False,
                        checkpoint_id="",
                        save_path="",
                        save_time_seconds=time.time() - start_time,
                        memory_usage_mb=0.0,
                        disk_usage_mb=0.0,
                        warnings=warnings,
                        errors=errors
                    )
            
            # Wait for memory cooldown if needed
            if self.config.memory_aware_intervals:
                memory_stats = self.memory_manager.monitor_memory_usage()
                if memory_stats.percent_used > self.config.memory_threshold_for_saving:
                    if not self._wait_for_memory_cooldown():
                        warnings.append(f"Saved with high memory usage: {memory_stats.percent_used:.1f}%")
            
            # Generate checkpoint ID and paths
            checkpoint_id = self._generate_checkpoint_id(step, epoch)
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)
            
            # Get current resource usage
            memory_stats = self.memory_manager.monitor_memory_usage()
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'step': step,
                'epoch': epoch,
                'loss': loss,
                'learning_rate': learning_rate,
                'timestamp': datetime.now().isoformat(),
                'training_time_seconds': training_time_seconds,
            }
            
            # Save checkpoint
            if TORCH_AVAILABLE:
                torch.save(checkpoint_data, checkpoint_path)
            else:
                # Fallback for non-PyTorch environments
                import pickle
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            # Calculate file sizes
            checkpoint_size = self._get_disk_usage(checkpoint_path)
            
            # Create metadata
            config_dict = asdict(training_config)
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                model_config_hash=self._calculate_config_hash(config_dict),
                memory_usage_mb=memory_stats.used_mb,
                disk_usage_mb=checkpoint_size,
                training_time_seconds=training_time_seconds,
                training_config=config_dict,
                hardware_summary=training_config.hardware_summary,
                checkpoint_path=str(checkpoint_path),
                model_state_size_mb=checkpoint_size * 0.7,  # Rough estimate
                optimizer_state_size_mb=checkpoint_size * 0.3,  # Rough estimate
                config_compatible=True,
                validation_errors=[]
            )
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                metadata_dict = asdict(metadata)
                metadata_dict['timestamp'] = metadata.timestamp.isoformat()
                json.dump(metadata_dict, f, indent=2)
            
            # Update cache
            self._metadata_cache[checkpoint_id] = metadata
            self._save_metadata_cache()
            
            # Update state
            self.last_save_time = time.time()
            self.last_save_step = step
            
            # Cleanup old checkpoints if needed
            if self.config.auto_cleanup:
                self.cleanup_old_checkpoints()
            
            save_time = time.time() - start_time
            
            self.logger.info(f"Saved checkpoint {checkpoint_id} in {save_time:.2f}s "
                           f"(size: {checkpoint_size:.1f}MB, memory: {memory_stats.percent_used:.1f}%)")
            
            return CheckpointSaveResult(
                success=True,
                checkpoint_id=checkpoint_id,
                save_path=str(checkpoint_path),
                save_time_seconds=save_time,
                memory_usage_mb=memory_stats.used_mb,
                disk_usage_mb=checkpoint_size,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Failed to save checkpoint: {str(e)}")
            self.logger.error(f"Checkpoint save failed: {e}")
            
            return CheckpointSaveResult(
                success=False,
                checkpoint_id="",
                save_path="",
                save_time_seconds=time.time() - start_time,
                memory_usage_mb=0.0,
                disk_usage_mb=0.0,
                warnings=warnings,
                errors=errors
            )
            
        finally:
            self.save_in_progress = False
    
    def load_checkpoint(self, 
                       checkpoint_id: Optional[str] = None,
                       validate_config: bool = True,
                       current_config: Optional[MacBookTrainingConfig] = None) -> CheckpointLoadResult:
        """
        Load checkpoint with configuration validation.
        
        Args:
            checkpoint_id: Specific checkpoint ID to load (latest if None)
            validate_config: Whether to validate configuration compatibility
            current_config: Current training configuration for validation
            
        Returns:
            CheckpointLoadResult with loaded data
        """
        warnings = []
        errors = []
        
        try:
            # Find checkpoint to load
            if checkpoint_id is None:
                checkpoint_id = self.get_latest_checkpoint_id()
                if checkpoint_id is None:
                    errors.append("No checkpoints found")
                    return CheckpointLoadResult(
                        success=False,
                        checkpoint_id="",
                        metadata=None,
                        model_state=None,
                        optimizer_state=None,
                        config_compatible=False,
                        warnings=warnings,
                        errors=errors
                    )
            
            # Get paths
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)
            
            if not checkpoint_path.exists():
                errors.append(f"Checkpoint file not found: {checkpoint_path}")
                return CheckpointLoadResult(
                    success=False,
                    checkpoint_id=checkpoint_id,
                    metadata=None,
                    model_state=None,
                    optimizer_state=None,
                    config_compatible=False,
                    warnings=warnings,
                    errors=errors
                )
            
            # Load metadata
            metadata = None
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata_dict = json.load(f)
                    metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                    metadata = CheckpointMetadata(**metadata_dict)
                except Exception as e:
                    warnings.append(f"Failed to load metadata: {e}")
            
            # Load checkpoint data
            if TORCH_AVAILABLE:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            else:
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            # Validate configuration compatibility
            config_compatible = True
            if validate_config and current_config and metadata:
                current_config_dict = asdict(current_config)
                current_hash = self._calculate_config_hash(current_config_dict)
                
                if current_hash != metadata.model_config_hash:
                    if self.config.strict_config_validation:
                        config_compatible = False
                        errors.append("Configuration mismatch - strict validation failed")
                    else:
                        config_compatible = True
                        warnings.append("Configuration mismatch - proceeding with compatibility mode")
            
            self.logger.info(f"Loaded checkpoint {checkpoint_id} "
                           f"(step: {checkpoint_data.get('step', 'unknown')}, "
                           f"loss: {checkpoint_data.get('loss', 'unknown')})")
            
            return CheckpointLoadResult(
                success=True,
                checkpoint_id=checkpoint_id,
                metadata=metadata,
                model_state=checkpoint_data.get('model_state_dict'),
                optimizer_state=checkpoint_data.get('optimizer_state_dict'),
                config_compatible=config_compatible,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Failed to load checkpoint: {str(e)}")
            self.logger.error(f"Checkpoint load failed: {e}")
            
            return CheckpointLoadResult(
                success=False,
                checkpoint_id=checkpoint_id or "",
                metadata=None,
                model_state=None,
                optimizer_state=None,
                config_compatible=False,
                warnings=warnings,
                errors=errors
            )
    
    def get_latest_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint."""
        if not self._metadata_cache:
            return None
        
        # Sort by timestamp and return the latest
        sorted_checkpoints = sorted(
            self._metadata_cache.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        return sorted_checkpoints[0][0] if sorted_checkpoints else None
    
    def list_checkpoints(self, limit: Optional[int] = None) -> List[CheckpointMetadata]:
        """
        List available checkpoints sorted by timestamp (newest first).
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = sorted(
            self._metadata_cache.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints
    
    def cleanup_old_checkpoints(self) -> Dict[str, Any]:
        """
        Clean up old checkpoints to manage disk space.
        
        Returns:
            Cleanup summary
        """
        cleanup_summary = {
            "removed_count": 0,
            "freed_space_mb": 0.0,
            "remaining_count": 0,
            "errors": []
        }
        
        try:
            # Get all checkpoints sorted by timestamp (oldest first)
            all_checkpoints = sorted(
                self._metadata_cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Keep only the most recent N checkpoints
            checkpoints_to_remove = all_checkpoints[:-self.config.max_checkpoints]
            
            for checkpoint_id, metadata in checkpoints_to_remove:
                try:
                    # Remove checkpoint file
                    checkpoint_path = Path(metadata.checkpoint_path)
                    if checkpoint_path.exists():
                        size_mb = self._get_disk_usage(checkpoint_path)
                        checkpoint_path.unlink()
                        cleanup_summary["freed_space_mb"] += size_mb
                    
                    # Remove metadata file
                    metadata_path = self._get_metadata_path(checkpoint_id)
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    # Remove from cache
                    del self._metadata_cache[checkpoint_id]
                    cleanup_summary["removed_count"] += 1
                    
                    self.logger.info(f"Removed old checkpoint: {checkpoint_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to remove checkpoint {checkpoint_id}: {e}"
                    cleanup_summary["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            cleanup_summary["remaining_count"] = len(self._metadata_cache)
            
            # Update metadata cache
            self._save_metadata_cache()
            
            if cleanup_summary["removed_count"] > 0:
                self.logger.info(f"Cleanup completed: removed {cleanup_summary['removed_count']} checkpoints, "
                               f"freed {cleanup_summary['freed_space_mb']:.1f}MB")
            
        except Exception as e:
            error_msg = f"Checkpoint cleanup failed: {e}"
            cleanup_summary["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return cleanup_summary
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint management summary."""
        total_size_mb = sum(metadata.disk_usage_mb for metadata in self._metadata_cache.values())
        free_space_mb = self._get_free_disk_space()
        
        return {
            "checkpoint_count": len(self._metadata_cache),
            "total_size_mb": total_size_mb,
            "free_disk_space_mb": free_space_mb,
            "latest_checkpoint": self.get_latest_checkpoint_id(),
            "last_save_time": self.last_save_time,
            "last_save_step": self.last_save_step,
            "save_in_progress": self.save_in_progress,
            "config": {
                "max_checkpoints": self.config.max_checkpoints,
                "save_interval_steps": self.config.save_interval_steps,
                "save_interval_minutes": self.config.save_interval_minutes,
                "auto_cleanup": self.config.auto_cleanup,
            }
        }