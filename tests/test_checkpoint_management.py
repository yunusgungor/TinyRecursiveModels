"""
Unit tests for checkpoint management module.

Tests checkpoint saving and loading functionality, configuration compatibility checking,
and checkpoint cleanup and rotation for MacBook optimization.
"""

import pytest
import tempfile
import shutil
import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from macbook_optimization.checkpoint_management import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointMetadata,
    CheckpointSaveResult,
    CheckpointLoadResult,
)
from macbook_optimization.memory_management import MemoryManager
from macbook_optimization.config_management import MacBookTrainingConfig


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""
    
    def test_checkpoint_config_defaults(self):
        """Test CheckpointConfig default values."""
        config = CheckpointConfig()
        
        assert config.checkpoint_dir == "checkpoints"
        assert config.max_checkpoints == 3
        assert config.min_disk_space_mb == 1000.0
        assert config.save_interval_steps == 500
        assert config.save_interval_minutes == 30.0
        assert config.memory_aware_intervals == True
        assert config.memory_threshold_for_saving == 70.0
        assert config.wait_for_memory_cooldown == True
        assert config.max_wait_time_seconds == 300.0
        assert config.auto_cleanup == True
        assert config.cleanup_on_low_disk == True
        assert config.disk_space_threshold_mb == 2000.0
        assert config.validate_on_load == True
        assert config.strict_config_validation == False
        assert config.compress_checkpoints == True
        assert config.compression_level == 6
    
    def test_checkpoint_config_custom_values(self):
        """Test CheckpointConfig with custom values."""
        config = CheckpointConfig(
            checkpoint_dir="custom_checkpoints",
            max_checkpoints=5,
            save_interval_steps=1000,
            memory_threshold_for_saving=80.0,
            auto_cleanup=False
        )
        
        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.max_checkpoints == 5
        assert config.save_interval_steps == 1000
        assert config.memory_threshold_for_saving == 80.0
        assert config.auto_cleanup == False


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""
    
    def test_checkpoint_metadata_creation(self):
        """Test CheckpointMetadata creation."""
        timestamp = datetime.now()
        metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint_001",
            timestamp=timestamp,
            step=1000,
            epoch=5,
            loss=0.5,
            learning_rate=1e-4,
            model_config_hash="abc123",
            memory_usage_mb=2048.0,
            disk_usage_mb=100.0,
            training_time_seconds=3600.0,
            training_config={},
            hardware_summary={},
            checkpoint_path="/path/to/checkpoint.pt",
            model_state_size_mb=70.0,
            optimizer_state_size_mb=30.0,
            config_compatible=True,
            validation_errors=[]
        )
        
        assert metadata.checkpoint_id == "test_checkpoint_001"
        assert metadata.timestamp == timestamp
        assert metadata.step == 1000
        assert metadata.epoch == 5
        assert metadata.loss == 0.5
        assert metadata.learning_rate == 1e-4
        assert metadata.model_config_hash == "abc123"
        assert metadata.memory_usage_mb == 2048.0
        assert metadata.disk_usage_mb == 100.0
        assert metadata.training_time_seconds == 3600.0
        assert metadata.config_compatible == True
        assert len(metadata.validation_errors) == 0


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.monitor_memory_usage.return_value = Mock(
            used_mb=1024.0,
            available_mb=6144.0,
            percent_used=14.3
        )
        return memory_manager
    
    @pytest.fixture
    def checkpoint_config(self, temp_dir):
        """Create test checkpoint configuration."""
        return CheckpointConfig(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            save_interval_steps=100,
            save_interval_minutes=5.0,
            memory_threshold_for_saving=70.0,
            auto_cleanup=False  # Disable auto cleanup for most tests
        )
    
    @pytest.fixture
    def checkpoint_manager(self, checkpoint_config, mock_memory_manager):
        """Create test checkpoint manager."""
        return CheckpointManager(
            config=checkpoint_config,
            memory_manager=mock_memory_manager
        )
    
    @pytest.fixture
    def sample_training_config(self):
        """Create sample training configuration."""
        return MacBookTrainingConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
            num_workers=2,
            pin_memory=False,
            memory_limit_mb=4000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={"cpu": {"cores": 4}, "memory": {"total_gb": 8}}
        )
    
    def test_checkpoint_manager_initialization(self, checkpoint_manager, temp_dir):
        """Test checkpoint manager initialization."""
        assert checkpoint_manager.config.checkpoint_dir == temp_dir
        assert checkpoint_manager.checkpoint_dir.exists()
        assert checkpoint_manager.last_save_time == 0.0
        assert checkpoint_manager.last_save_step == 0
        assert checkpoint_manager.save_in_progress == False
        assert len(checkpoint_manager._metadata_cache) == 0
    
    def test_generate_checkpoint_id(self, checkpoint_manager):
        """Test checkpoint ID generation."""
        checkpoint_id = checkpoint_manager._generate_checkpoint_id(1000, 5)
        
        assert "checkpoint_step_1000_epoch_5" in checkpoint_id
        assert len(checkpoint_id.split("_")) >= 5  # Should include timestamp
    
    def test_calculate_config_hash(self, checkpoint_manager):
        """Test configuration hash calculation."""
        config1 = {"batch_size": 8, "learning_rate": 1e-4}
        config2 = {"batch_size": 8, "learning_rate": 1e-4}
        config3 = {"batch_size": 16, "learning_rate": 1e-4}
        
        hash1 = checkpoint_manager._calculate_config_hash(config1)
        hash2 = checkpoint_manager._calculate_config_hash(config2)
        hash3 = checkpoint_manager._calculate_config_hash(config3)
        
        assert hash1 == hash2  # Same config should have same hash
        assert hash1 != hash3  # Different config should have different hash
        assert len(hash1) == 32  # MD5 hash length
    
    def test_should_save_checkpoint_step_interval(self, checkpoint_manager):
        """Test checkpoint saving based on step interval."""
        checkpoint_manager.last_save_step = 0
        
        # Should save at step interval (100 steps from 0)
        should_save, reason = checkpoint_manager.should_save_checkpoint(100)
        assert should_save == True
        assert "step_interval" in reason
        
        # Should not save before interval (70 steps from 50 = 120, but interval is 100)
        # Disable memory-aware intervals to avoid interference and set recent save time
        checkpoint_manager.config.memory_aware_intervals = False
        checkpoint_manager.last_save_step = 50
        checkpoint_manager.last_save_time = time.time()  # Set to current time to avoid time interval trigger
        should_save, reason = checkpoint_manager.should_save_checkpoint(120)  # 120 - 50 = 70 < 100
        assert should_save == False
        assert reason == "no_trigger"
        
        # Test a case that should save
        checkpoint_manager.last_save_step = 50
        should_save, reason = checkpoint_manager.should_save_checkpoint(150)  # 150 - 50 = 100 >= 100
        assert should_save == True
        assert "step_interval" in reason
    
    def test_should_save_checkpoint_time_interval(self, checkpoint_manager):
        """Test checkpoint saving based on time interval."""
        # Set last save time to 10 minutes ago
        checkpoint_manager.last_save_time = time.time() - 600
        checkpoint_manager.config.save_interval_minutes = 5.0
        
        should_save, reason = checkpoint_manager.should_save_checkpoint(50)
        assert should_save == True
        assert "time_interval" in reason
    
    def test_should_save_checkpoint_force(self, checkpoint_manager):
        """Test forced checkpoint saving."""
        should_save, reason = checkpoint_manager.should_save_checkpoint(50, force=True)
        assert should_save == True
        assert reason == "forced"
    
    def test_should_save_checkpoint_save_in_progress(self, checkpoint_manager):
        """Test checkpoint saving when save is in progress."""
        checkpoint_manager.save_in_progress = True
        
        # Without force, should not save when save is in progress
        should_save, reason = checkpoint_manager.should_save_checkpoint(100, force=False)
        assert should_save == False
        assert reason == "save_in_progress"
        
        # With force, should still save (force overrides save_in_progress check)
        should_save, reason = checkpoint_manager.should_save_checkpoint(100, force=True)
        assert should_save == True
        assert reason == "forced"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_checkpoint_success(self, checkpoint_manager, sample_training_config):
        """Test successful checkpoint saving."""
        # Create mock model and optimizer states
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        result = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=1000,
            epoch=5,
            loss=0.5,
            learning_rate=1e-4,
            training_config=sample_training_config,
            training_time_seconds=3600.0,
            force=True
        )
        
        assert result.success == True
        assert result.checkpoint_id != ""
        assert result.save_path != ""
        assert result.save_time_seconds > 0
        assert result.memory_usage_mb > 0
        assert result.disk_usage_mb > 0
        assert len(result.errors) == 0
        
        # Check that files were created
        checkpoint_path = Path(result.save_path)
        assert checkpoint_path.exists()
        
        metadata_path = checkpoint_path.parent / f"{result.checkpoint_id}_metadata.json"
        assert metadata_path.exists()
        
        # Check metadata cache
        assert result.checkpoint_id in checkpoint_manager._metadata_cache
    
    def test_save_checkpoint_insufficient_disk_space(self, checkpoint_manager, sample_training_config):
        """Test checkpoint saving with insufficient disk space."""
        # Mock insufficient disk space
        with patch.object(checkpoint_manager, '_get_free_disk_space', return_value=500.0):
            result = checkpoint_manager.save_checkpoint(
                model_state={"test": "data"},
                optimizer_state={"test": "data"},
                step=1000,
                epoch=5,
                loss=0.5,
                learning_rate=1e-4,
                training_config=sample_training_config,
                force=True
            )
            
            assert result.success == False
            assert any("Insufficient disk space" in error for error in result.errors)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_load_checkpoint_success(self, checkpoint_manager, sample_training_config):
        """Test successful checkpoint loading."""
        # First save a checkpoint
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        save_result = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=1000,
            epoch=5,
            loss=0.5,
            learning_rate=1e-4,
            training_config=sample_training_config,
            force=True
        )
        
        assert save_result.success == True
        
        # Now load the checkpoint
        load_result = checkpoint_manager.load_checkpoint(
            checkpoint_id=save_result.checkpoint_id,
            validate_config=True,
            current_config=sample_training_config
        )
        
        assert load_result.success == True
        assert load_result.checkpoint_id == save_result.checkpoint_id
        assert load_result.model_state is not None
        assert load_result.optimizer_state is not None
        assert load_result.metadata is not None
        assert load_result.config_compatible == True
        assert len(load_result.errors) == 0
        
        # Check loaded data
        assert "layer.weight" in load_result.model_state
        assert torch.equal(load_result.model_state["layer.weight"], model_state["layer.weight"])
    
    def test_load_checkpoint_not_found(self, checkpoint_manager):
        """Test loading non-existent checkpoint."""
        result = checkpoint_manager.load_checkpoint(checkpoint_id="nonexistent")
        
        assert result.success == False
        assert result.checkpoint_id == "nonexistent"
        assert result.model_state is None
        assert result.optimizer_state is None
        assert result.metadata is None
        assert any("not found" in error for error in result.errors)
    
    def test_load_latest_checkpoint_empty(self, checkpoint_manager):
        """Test loading latest checkpoint when none exist."""
        result = checkpoint_manager.load_checkpoint()
        
        assert result.success == False
        assert any("No checkpoints found" in error for error in result.errors)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_latest_checkpoint_id(self, checkpoint_manager, sample_training_config):
        """Test getting latest checkpoint ID."""
        # Initially no checkpoints
        assert checkpoint_manager.get_latest_checkpoint_id() is None
        
        # Save multiple checkpoints
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        checkpoint_ids = []
        for i in range(3):
            time.sleep(0.1)  # Ensure different timestamps
            result = checkpoint_manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                step=1000 + i * 100,
                epoch=5 + i,
                loss=0.5 - i * 0.1,
                learning_rate=1e-4,
                training_config=sample_training_config,
                force=True
            )
            checkpoint_ids.append(result.checkpoint_id)
        
        # Latest should be the last one saved
        latest_id = checkpoint_manager.get_latest_checkpoint_id()
        assert latest_id == checkpoint_ids[-1]
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_list_checkpoints(self, checkpoint_manager, sample_training_config):
        """Test listing checkpoints."""
        # Initially empty
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Save multiple checkpoints
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        for i in range(3):
            time.sleep(0.1)  # Ensure different timestamps
            checkpoint_manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                step=1000 + i * 100,
                epoch=5 + i,
                loss=0.5 - i * 0.1,
                learning_rate=1e-4,
                training_config=sample_training_config,
                force=True
            )
        
        # List all checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Should be sorted by timestamp (newest first)
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].timestamp >= checkpoints[i + 1].timestamp
        
        # Test limit
        limited_checkpoints = checkpoint_manager.list_checkpoints(limit=2)
        assert len(limited_checkpoints) == 2
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_cleanup_old_checkpoints(self, temp_dir, mock_memory_manager, sample_training_config):
        """Test cleanup of old checkpoints."""
        # Create checkpoint manager with auto cleanup enabled for this test
        config = CheckpointConfig(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            save_interval_steps=100,
            auto_cleanup=True  # Enable auto cleanup for this test
        )
        checkpoint_manager = CheckpointManager(config, mock_memory_manager)
        
        # Save more checkpoints than max_checkpoints
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        checkpoint_ids = []
        for i in range(5):  # More than max_checkpoints (3)
            time.sleep(0.1)  # Ensure different timestamps
            result = checkpoint_manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                step=1000 + i * 100,
                epoch=5 + i,
                loss=0.5 - i * 0.1,
                learning_rate=1e-4,
                training_config=sample_training_config,
                force=True
            )
            checkpoint_ids.append(result.checkpoint_id)
        
        # With auto cleanup enabled, should have max_checkpoints (3) checkpoints
        assert len(checkpoint_manager._metadata_cache) == 3
        
        # Manual cleanup should not remove any more since auto cleanup already happened
        cleanup_summary = checkpoint_manager.cleanup_old_checkpoints()
        
        # Should have removed 0 checkpoints since auto cleanup already handled it
        assert cleanup_summary["removed_count"] == 0
        assert cleanup_summary["remaining_count"] == 3
        assert len(cleanup_summary["errors"]) == 0
        
        # Should keep the 3 most recent checkpoints
        remaining_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(remaining_checkpoints) == 3
        
        # Check that the most recent checkpoints were kept
        remaining_ids = [cp.checkpoint_id for cp in remaining_checkpoints]
        assert checkpoint_ids[-3:] == remaining_ids[::-1]  # Most recent first
    
    def test_get_checkpoint_summary(self, checkpoint_manager):
        """Test checkpoint summary generation."""
        summary = checkpoint_manager.get_checkpoint_summary()
        
        assert "checkpoint_count" in summary
        assert "total_size_mb" in summary
        assert "free_disk_space_mb" in summary
        assert "latest_checkpoint" in summary
        assert "last_save_time" in summary
        assert "last_save_step" in summary
        assert "save_in_progress" in summary
        assert "config" in summary
        
        # Initially empty
        assert summary["checkpoint_count"] == 0
        assert summary["total_size_mb"] == 0
        assert summary["latest_checkpoint"] is None
        assert summary["last_save_time"] == 0.0
        assert summary["last_save_step"] == 0
        assert summary["save_in_progress"] == False
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_config_compatibility_validation(self, checkpoint_manager, sample_training_config):
        """Test configuration compatibility validation."""
        # Save checkpoint with specific config
        model_state = {"layer.weight": torch.randn(10, 10)}
        optimizer_state = {"state": {}, "param_groups": [{"lr": 1e-4}]}
        
        save_result = checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=1000,
            epoch=5,
            loss=0.5,
            learning_rate=1e-4,
            training_config=sample_training_config,
            force=True
        )
        
        # Load with same config (should be compatible)
        load_result = checkpoint_manager.load_checkpoint(
            checkpoint_id=save_result.checkpoint_id,
            validate_config=True,
            current_config=sample_training_config
        )
        
        assert load_result.success == True
        assert load_result.config_compatible == True
        
        # Load with different config (should be incompatible but still load)
        different_config = MacBookTrainingConfig(
            batch_size=16,  # Different batch size
            gradient_accumulation_steps=4,
            num_workers=2,
            pin_memory=False,
            memory_limit_mb=4000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=2e-4,  # Different learning rate
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={"cpu": {"cores": 4}, "memory": {"total_gb": 8}}
        )
        
        load_result_diff = checkpoint_manager.load_checkpoint(
            checkpoint_id=save_result.checkpoint_id,
            validate_config=True,
            current_config=different_config
        )
        
        assert load_result_diff.success == True
        assert load_result_diff.config_compatible == True  # Non-strict validation
        assert len(load_result_diff.warnings) > 0  # Should have compatibility warning
    
    def test_memory_aware_saving(self, checkpoint_manager, sample_training_config):
        """Test memory-aware checkpoint saving."""
        # Mock high memory usage
        checkpoint_manager.memory_manager.monitor_memory_usage.return_value = Mock(
            used_mb=7000.0,
            available_mb=1000.0,
            percent_used=87.5  # Above threshold
        )
        
        # Mock wait for memory cooldown to timeout
        with patch.object(checkpoint_manager, '_wait_for_memory_cooldown', return_value=False):
            result = checkpoint_manager.save_checkpoint(
                model_state={"test": "data"},
                optimizer_state={"test": "data"},
                step=1000,
                epoch=5,
                loss=0.5,
                learning_rate=1e-4,
                training_config=sample_training_config,
                force=True
            )
            
            # Should still save but with warning
            assert result.success == True
            assert any("high memory usage" in warning.lower() for warning in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__])