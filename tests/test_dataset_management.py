"""
Unit tests for dataset management module.

Tests memory-efficient dataset loading strategies, streaming datasets,
preprocessing/caching utilities, and integration with MacBook optimization.
"""

import pytest
import os
import tempfile
import pickle
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    DataLoader = object
    TORCH_AVAILABLE = False

from macbook_optimization.dataset_management import (
    DatasetManagementConfig,
    DatasetMetrics,
    StreamingPuzzleDataset,
    CachedDatasetLoader,
    DatasetManager,
    create_memory_efficient_dataloader,
    preprocess_for_memory_efficiency,
    estimate_dataset_memory_usage
)
from macbook_optimization.memory_management import MemoryManager
from puzzle_dataset import PuzzleDatasetConfig, PuzzleDatasetMetadata
from dataset.common import PuzzleDatasetMetadata


class TestDatasetManagementConfig:
    """Test DatasetManagementConfig dataclass."""
    
    def test_dataset_management_config_defaults(self):
        """Test DatasetManagementConfig default values."""
        config = DatasetManagementConfig()
        
        assert config.max_dataset_memory_mb == 800.0
        assert config.streaming_threshold_mb == 400.0
        assert config.cache_threshold_mb == 200.0
        assert config.chunk_size_mb == 50.0
        assert config.prefetch_chunks == 2
        assert config.enable_caching is True
        assert config.cache_dir == ".cache/datasets"
        assert config.max_cache_size_gb == 2.0
        assert config.cache_compression is True
        assert config.memory_check_interval == 100
        assert config.auto_fallback_streaming is True
    
    def test_dataset_management_config_custom_values(self):
        """Test DatasetManagementConfig with custom values."""
        config = DatasetManagementConfig(
            max_dataset_memory_mb=1600.0,
            streaming_threshold_mb=800.0,
            cache_threshold_mb=400.0,
            chunk_size_mb=100.0,
            enable_caching=False,
            auto_fallback_streaming=False
        )
        
        assert config.max_dataset_memory_mb == 1600.0
        assert config.streaming_threshold_mb == 800.0
        assert config.cache_threshold_mb == 400.0
        assert config.chunk_size_mb == 100.0
        assert config.enable_caching is False
        assert config.auto_fallback_streaming is False


class TestDatasetMetrics:
    """Test DatasetMetrics dataclass."""
    
    def test_dataset_metrics_creation(self):
        """Test DatasetMetrics creation."""
        metrics = DatasetMetrics(
            total_size_mb=150.5,
            loading_strategy="streaming",
            memory_usage_mb=75.2,
            load_time_seconds=2.5,
            cache_hit=True,
            num_samples=1000,
            avg_sample_size_bytes=1024
        )
        
        assert metrics.total_size_mb == 150.5
        assert metrics.loading_strategy == "streaming"
        assert metrics.memory_usage_mb == 75.2
        assert metrics.load_time_seconds == 2.5
        assert metrics.cache_hit is True
        assert metrics.num_samples == 1000
        assert metrics.avg_sample_size_bytes == 1024


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestStreamingPuzzleDataset:
    """Test StreamingPuzzleDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset")
        self.train_dir = os.path.join(self.dataset_path, "train")
        os.makedirs(self.train_dir, exist_ok=True)
        
        # Create test dataset files
        self.create_test_dataset_files()
        
        # Create test configuration
        self.config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[self.dataset_path],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        self.mock_memory_manager = Mock(spec=MemoryManager)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset_files(self):
        """Create test dataset files."""
        # Create metadata
        metadata = {
            "seq_len": 128,
            "vocab_size": 1000,
            "pad_id": 0,
            "ignore_label_id": None,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 100,
            "total_groups": 50,
            "mean_puzzle_examples": 2.0,
            "total_puzzles": 100,
            "sets": ["all"]
        }
        
        with open(os.path.join(self.train_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)
        
        # Create sample data arrays
        num_samples = 100
        seq_len = 128
        
        inputs = np.random.randint(0, 1000, (num_samples, seq_len), dtype=np.int32)
        labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
        puzzle_identifiers = np.arange(num_samples, dtype=np.int32)
        puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
        group_indices = np.arange(51, dtype=np.int32)  # 50 groups + 1
        
        # Save arrays
        np.save(os.path.join(self.train_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(self.train_dir, "all__labels.npy"), labels)
        np.save(os.path.join(self.train_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(self.train_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(self.train_dir, "all__group_indices.npy"), group_indices)
    
    def test_streaming_puzzle_dataset_creation(self):
        """Test StreamingPuzzleDataset creation."""
        dataset = StreamingPuzzleDataset(
            self.config, 
            split="train",
            chunk_size_mb=1.0,
            memory_manager=self.mock_memory_manager
        )
        
        assert dataset.config == self.config
        assert dataset.split == "train"
        assert dataset.chunk_size_bytes == int(1.0 * 1024 * 1024)
        assert dataset.memory_manager == self.mock_memory_manager
        assert dataset.metadata is not None
        assert len(dataset.chunks_info) > 0
    
    def test_streaming_puzzle_dataset_metadata_loading(self):
        """Test metadata loading and merging."""
        dataset = StreamingPuzzleDataset(self.config, split="train")
        
        metadata = dataset.metadata
        assert metadata.seq_len == 128
        assert metadata.vocab_size == 1000
        assert metadata.pad_id == 0
        assert metadata.num_puzzle_identifiers == 100
        assert metadata.total_groups == 50
        assert metadata.total_puzzles == 100
        assert metadata.sets == ["all"]
    
    def test_streaming_puzzle_dataset_chunk_analysis(self):
        """Test chunk analysis functionality."""
        dataset = StreamingPuzzleDataset(
            self.config, 
            split="train",
            chunk_size_mb=0.1  # Very small chunks for testing
        )
        
        # Should have analyzed chunks
        assert len(dataset.chunks_info) > 0
        
        # Check chunk info structure
        chunk_info = dataset.chunks_info[0]
        assert 'dataset_idx' in chunk_info
        assert 'dataset_path' in chunk_info
        assert 'set_name' in chunk_info
        assert 'chunk_idx' in chunk_info
        assert 'total_chunks' in chunk_info
        assert 'inputs_path' in chunk_info
        assert 'labels_path' in chunk_info
        assert 'file_size' in chunk_info
    
    def test_streaming_puzzle_dataset_iteration(self):
        """Test dataset iteration."""
        # Mock memory manager
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 50.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        dataset = StreamingPuzzleDataset(
            self.config, 
            split="train",
            chunk_size_mb=1.0,
            memory_manager=self.mock_memory_manager
        )
        
        # Test iteration
        batches = []
        for i, (set_name, batch, batch_size) in enumerate(dataset):
            batches.append((set_name, batch, batch_size))
            if i >= 2:  # Only test first few batches
                break
        
        assert len(batches) > 0
        
        # Check batch structure
        for set_name, batch, batch_size in batches:
            assert isinstance(set_name, str)
            assert isinstance(batch, dict)
            assert 'inputs' in batch
            assert 'labels' in batch
            assert 'puzzle_identifiers' in batch
            assert batch_size > 0
    
    def test_streaming_puzzle_dataset_memory_monitoring(self):
        """Test memory monitoring during iteration."""
        # Mock high memory usage
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 90.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        dataset = StreamingPuzzleDataset(
            self.config, 
            split="train",
            memory_manager=self.mock_memory_manager
        )
        
        # Iterate through dataset
        for i, (set_name, batch, batch_size) in enumerate(dataset):
            if i >= 1:  # Just test one batch
                break
        
        # Memory monitoring should have been called
        assert self.mock_memory_manager.monitor_memory_usage.called
        
        # Garbage collection should have been triggered due to high memory usage
        assert self.mock_memory_manager.force_garbage_collection.called


class TestCachedDatasetLoader:
    """Test CachedDatasetLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        
        self.config = DatasetManagementConfig(
            enable_caching=True,
            cache_dir=self.cache_dir,
            cache_compression=False  # Disable compression for simpler testing
        )
        
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.loader = CachedDatasetLoader(self.config, self.memory_manager)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cached_dataset_loader_creation(self):
        """Test CachedDatasetLoader creation."""
        assert self.loader.config == self.config
        assert self.loader.memory_manager == self.memory_manager
        assert self.loader.cache_dir == Path(self.cache_dir)
        assert self.loader.cache_dir.exists()
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        dataset_paths = ["/path/to/dataset1", "/path/to/dataset2"]
        split = "train"
        preprocessing_params = {"param1": "value1", "param2": 42}
        
        key1 = self.loader._get_cache_key(dataset_paths, split, preprocessing_params)
        key2 = self.loader._get_cache_key(dataset_paths, split, preprocessing_params)
        
        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length
        
        # Different inputs should produce different keys
        key3 = self.loader._get_cache_key(dataset_paths, "test", preprocessing_params)
        assert key1 != key3
    
    def test_cache_operations(self):
        """Test cache save and load operations."""
        cache_key = "test_cache_key"
        test_data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        
        # Initially not cached
        assert not self.loader._is_cached(cache_key)
        
        # Save to cache
        self.loader._save_to_cache(cache_key, test_data)
        
        # Should now be cached
        assert self.loader._is_cached(cache_key)
        
        # Load from cache
        loaded_data = self.loader._load_from_cache(cache_key)
        assert loaded_data == test_data
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Create multiple cache files
        test_data = {"test": "data"}
        
        for i in range(5):
            cache_key = f"test_key_{i}"
            self.loader._save_to_cache(cache_key, test_data)
            time.sleep(0.01)  # Small delay to ensure different modification times
        
        # Set very small cache size limit
        self.loader.config.max_cache_size_gb = 0.000001  # Very small limit
        
        # Trigger cleanup
        self.loader._cleanup_cache()
        
        # Some files should have been removed
        remaining_files = list(self.loader.cache_dir.glob("*.pkl"))
        assert len(remaining_files) < 5
    
    def test_preprocess_and_cache(self):
        """Test preprocessing and caching functionality."""
        dataset_paths = ["/fake/path/to/dataset"]
        split = "train"
        
        # Mock PuzzleDataset creation
        with patch('macbook_optimization.dataset_management.PuzzleDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_metadata = Mock()
            mock_metadata.seq_len = 128
            mock_metadata.vocab_size = 1000
            mock_dataset.metadata = mock_metadata
            mock_dataset_class.return_value = mock_dataset
            
            cache_key = self.loader.preprocess_and_cache(dataset_paths, split)
            
            # Should return a cache key
            assert isinstance(cache_key, str)
            assert len(cache_key) == 32
            
            # Should be cached now
            assert self.loader._is_cached(cache_key)
            
            # Load cached data
            cached_data = self.loader.load_cached_dataset(cache_key)
            assert cached_data is not None
            assert 'metadata' in cached_data


class TestDatasetManager:
    """Test DatasetManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset")
        self.train_dir = os.path.join(self.dataset_path, "train")
        os.makedirs(self.train_dir, exist_ok=True)
        
        # Create test dataset files
        self.create_test_dataset_files()
        
        self.config = DatasetManagementConfig()
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.manager = DatasetManager(self.config, self.mock_memory_manager)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset_files(self):
        """Create test dataset files."""
        # Create metadata
        metadata = {
            "seq_len": 128,
            "vocab_size": 1000,
            "pad_id": 0,
            "ignore_label_id": None,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 100,
            "total_groups": 50,
            "mean_puzzle_examples": 2.0,
            "total_puzzles": 100,
            "sets": ["all"]
        }
        
        with open(os.path.join(self.train_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)
        
        # Create sample data arrays (small for testing)
        num_samples = 50
        seq_len = 128
        
        inputs = np.random.randint(0, 1000, (num_samples, seq_len), dtype=np.int32)
        labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
        
        # Save arrays
        np.save(os.path.join(self.train_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(self.train_dir, "all__labels.npy"), labels)
    
    def test_dataset_manager_creation(self):
        """Test DatasetManager creation."""
        assert self.manager.config == self.config
        assert self.manager.memory_manager == self.mock_memory_manager
        assert self.manager.dataset_analyzer is not None
        assert self.manager.cached_loader is not None
        assert isinstance(self.manager.loading_metrics, list)
    
    def test_analyze_dataset_requirements(self):
        """Test dataset requirements analysis."""
        # Mock memory manager
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        analysis = self.manager.analyze_dataset_requirements([self.dataset_path], "train")
        
        assert "total_size_mb" in analysis
        assert "available_memory_mb" in analysis
        assert "memory_utilization_percent" in analysis
        assert "recommended_strategy" in analysis
        assert "estimated_memory_usage_mb" in analysis
        assert "dataset_info" in analysis
        assert "can_fit_in_memory" in analysis
        assert "requires_streaming" in analysis
        
        # Check values are reasonable
        assert analysis["total_size_mb"] > 0
        assert analysis["available_memory_mb"] == 4000.0
        assert analysis["memory_utilization_percent"] >= 0
        assert analysis["recommended_strategy"] in ["cached", "memory_mapped", "streaming"]
        assert isinstance(analysis["can_fit_in_memory"], bool)
        assert isinstance(analysis["requires_streaming"], bool)
    
    def test_create_memory_efficient_dataset_standard(self):
        """Test memory-efficient dataset creation (standard strategy)."""
        # Mock memory manager for normal memory conditions
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[self.dataset_path],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        with patch('macbook_optimization.dataset_management.PuzzleDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_metadata = Mock()
            mock_dataset.metadata = mock_metadata
            mock_dataset_class.return_value = mock_dataset
            
            dataset = self.manager.create_memory_efficient_dataset(
                [self.dataset_path], config, "train"
            )
            
            # Should create standard PuzzleDataset for small datasets
            assert dataset == mock_dataset
            
            # Should have recorded metrics
            assert len(self.manager.loading_metrics) > 0
            metrics = self.manager.loading_metrics[-1]
            assert isinstance(metrics, DatasetMetrics)
    
    def test_create_memory_efficient_dataset_streaming(self):
        """Test memory-efficient dataset creation (streaming strategy)."""
        # Mock memory manager for low memory conditions
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 500.0  # Low memory
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[self.dataset_path],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset = self.manager.create_memory_efficient_dataset(
            [self.dataset_path], config, "train", force_strategy="streaming"
        )
        
        # Should create StreamingPuzzleDataset
        assert isinstance(dataset, StreamingPuzzleDataset)
        
        # Should have recorded metrics
        assert len(self.manager.loading_metrics) > 0
        metrics = self.manager.loading_metrics[-1]
        assert metrics.loading_strategy == "streaming"
    
    def test_validate_dataset_memory_constraints(self):
        """Test dataset memory constraints validation."""
        # Mock memory manager
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 2000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        validation = self.manager.validate_dataset_memory_constraints(
            [self.dataset_path], batch_size=16, split="train"
        )
        
        assert "memory_constraints_met" in validation
        assert "total_estimated_memory_mb" in validation
        assert "available_memory_mb" in validation
        assert "memory_utilization_percent" in validation
        assert "batch_size_ok" in validation
        assert "recommendations" in validation
        assert "fallback_to_streaming" in validation
        
        # Check values
        assert validation["available_memory_mb"] == 2000.0
        assert isinstance(validation["memory_constraints_met"], bool)
        assert isinstance(validation["batch_size_ok"], bool)
        assert isinstance(validation["recommendations"], list)
        assert isinstance(validation["fallback_to_streaming"], bool)
    
    def test_create_dataloader_with_fallback_success(self):
        """Test dataloader creation with successful standard loading."""
        # Mock memory manager for good memory conditions
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[self.dataset_path],
            global_batch_size=8,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        with patch('macbook_optimization.dataset_management.PuzzleDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_metadata = Mock()
            mock_dataset.metadata = mock_metadata
            mock_dataset_class.return_value = mock_dataset
            
            dataset, creation_info = self.manager.create_dataloader_with_fallback(
                [self.dataset_path], config, "train"
            )
            
            assert dataset == mock_dataset
            assert creation_info["fallback_used"] is False
            assert creation_info["final_strategy"] == "memory_mapped"
    
    def test_create_dataloader_with_fallback_streaming(self):
        """Test dataloader creation with fallback to streaming."""
        # Mock memory manager for low memory conditions
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 200.0  # Very low memory
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[self.dataset_path],
            global_batch_size=32,  # Large batch size
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset, creation_info = self.manager.create_dataloader_with_fallback(
            [self.dataset_path], config, "train"
        )
        
        # Should fallback to streaming
        assert isinstance(dataset, StreamingPuzzleDataset)
        assert creation_info["fallback_used"] is True
        assert creation_info["final_strategy"] == "streaming"
        
        # Should have triggered garbage collection
        assert self.mock_memory_manager.force_garbage_collection.called
    
    def test_monitor_dataset_memory_usage(self):
        """Test dataset memory usage monitoring."""
        # Mock memory manager
        mock_memory_stats = Mock()
        mock_memory_stats.used_mb = 1500.0
        mock_memory_stats.available_mb = 2500.0
        mock_memory_stats.percent_used = 60.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        # Mock memory recommendations
        mock_recommendations = {"test": "recommendations"}
        self.mock_memory_manager.get_memory_recommendations.return_value = mock_recommendations
        
        stats = self.manager.monitor_dataset_memory_usage("test_dataset")
        
        assert stats["dataset_name"] == "test_dataset"
        assert "timestamp" in stats
        assert stats["memory_usage"]["used_mb"] == 1500.0
        assert stats["memory_usage"]["available_mb"] == 2500.0
        assert stats["memory_usage"]["percent_used"] == 60.0
        assert stats["recommendations"] == mock_recommendations
    
    def test_optimize_batch_size_for_dataset(self):
        """Test batch size optimization for dataset."""
        # Mock memory manager
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        optimization = self.manager.optimize_batch_size_for_dataset(
            [self.dataset_path], initial_batch_size=64, split="train"
        )
        
        assert "initial_batch_size" in optimization
        assert "recommended_batch_size" in optimization
        assert "max_safe_batch_size" in optimization
        assert "memory_per_sample_mb" in optimization
        assert "available_memory_mb" in optimization
        assert "optimization_ratio" in optimization
        assert "memory_utilization_percent" in optimization
        
        # Check values
        assert optimization["initial_batch_size"] == 64
        assert optimization["recommended_batch_size"] > 0
        assert optimization["recommended_batch_size"] <= 32  # MacBook limit
        assert optimization["max_safe_batch_size"] >= optimization["recommended_batch_size"]
        assert optimization["available_memory_mb"] > 0
        assert optimization["optimization_ratio"] > 0
    
    def test_get_loading_metrics(self):
        """Test loading metrics retrieval."""
        # Initially no metrics
        metrics = self.manager.get_loading_metrics()
        assert len(metrics) == 0
        
        # Add some metrics
        test_metrics = DatasetMetrics(
            total_size_mb=100.0,
            loading_strategy="memory_mapped",
            memory_usage_mb=50.0,
            load_time_seconds=1.5,
            cache_hit=False,
            num_samples=1000,
            avg_sample_size_bytes=1024
        )
        self.manager.loading_metrics.append(test_metrics)
        
        # Should return copy of metrics
        metrics = self.manager.get_loading_metrics()
        assert len(metrics) == 1
        assert metrics[0] == test_metrics
        
        # Modifying returned metrics shouldn't affect original
        metrics[0].total_size_mb = 200.0
        original_metrics = self.manager.get_loading_metrics()
        assert original_metrics[0].total_size_mb == 100.0
    
    def test_clear_metrics(self):
        """Test metrics clearing."""
        # Add some metrics
        test_metrics = DatasetMetrics(
            total_size_mb=100.0,
            loading_strategy="memory_mapped",
            memory_usage_mb=50.0,
            load_time_seconds=1.5,
            cache_hit=False,
            num_samples=1000,
            avg_sample_size_bytes=1024
        )
        self.manager.loading_metrics.append(test_metrics)
        
        assert len(self.manager.loading_metrics) == 1
        
        # Clear metrics
        self.manager.clear_metrics()
        
        assert len(self.manager.loading_metrics) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_memory_efficient_dataloader(self):
        """Test create_memory_efficient_dataloader function."""
        dataset_paths = ["/fake/path"]
        batch_size = 16
        
        mock_memory_manager = Mock(spec=MemoryManager)
        
        with patch('macbook_optimization.dataset_management.DatasetManager') as mock_manager_class:
            mock_manager = Mock()
            mock_dataset = Mock()
            mock_creation_info = {"final_strategy": "memory_mapped", "fallback_used": False}
            mock_manager.create_dataloader_with_fallback.return_value = (mock_dataset, mock_creation_info)
            mock_manager_class.return_value = mock_manager
            
            dataset, creation_info = create_memory_efficient_dataloader(
                dataset_paths, batch_size, memory_manager=mock_memory_manager
            )
            
            assert dataset == mock_dataset
            assert creation_info == mock_creation_info
            
            # Check that DatasetManager was created with correct parameters
            mock_manager_class.assert_called_once()
            args, kwargs = mock_manager_class.call_args
            assert kwargs.get('memory_manager') == mock_memory_manager
    
    def test_preprocess_for_memory_efficiency(self):
        """Test preprocess_for_memory_efficiency function."""
        # Mock dataset
        mock_dataset = Mock()
        mock_metadata = Mock()
        mock_metadata.seq_len = 512
        mock_metadata.vocab_size = 5000
        mock_dataset.metadata = mock_metadata
        
        params = {
            'max_seq_len': 256,
            'batch_size': 16
        }
        
        result = preprocess_for_memory_efficiency(mock_dataset, params)
        
        assert "original_metadata" in result
        assert "efficient_params" in result
        assert "preprocessing_applied" in result
        assert "memory_optimizations" in result
        
        # Check efficient parameters
        efficient_params = result["efficient_params"]
        assert efficient_params["seq_len"] == 256  # Capped at max_seq_len
        assert efficient_params["vocab_size"] == 5000
        assert efficient_params["recommended_batch_size"] == 16
        assert efficient_params["memory_mapping_recommended"] is True
        
        # Check optimizations
        optimizations = result["memory_optimizations"]
        assert "sequence_length_capping" in optimizations
        assert "memory_mapping_enabled" in optimizations
        assert "batch_size_optimization" in optimizations
    
    def create_test_dataset_files(self, dataset_path, size_mb=10):
        """Create test dataset files."""
        train_dir = os.path.join(dataset_path, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            "seq_len": 128,
            "vocab_size": 1000,
            "pad_id": 0,
            "ignore_label_id": None,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 100,
            "total_groups": 50,
            "mean_puzzle_examples": 2.0,
            "total_puzzles": 100,
            "sets": ["all"]
        }
        
        with open(os.path.join(train_dir, "dataset.json"), "w") as f:
            json.dump(metadata, f)
        
        # Create data files of specified size
        target_size_bytes = int(size_mb * 1024 * 1024)
        
        # Calculate number of samples needed
        sample_size = 128 * 4  # 128 int32 values
        num_samples = max(10, target_size_bytes // sample_size)
        
        inputs = np.random.randint(0, 1000, (num_samples, 128), dtype=np.int32)
        labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
        
        np.save(os.path.join(train_dir, "all__inputs.npy"), inputs)
        np.save(os.path.join(train_dir, "all__labels.npy"), labels)
    
    def test_estimate_dataset_memory_usage_small(self):
        """Test memory usage estimation for small dataset."""
        dataset_path = os.path.join(self.temp_dir, "small_dataset")
        self.create_test_dataset_files(dataset_path, size_mb=5)
        
        estimate = estimate_dataset_memory_usage([dataset_path], "train")
        
        assert "total_size_mb" in estimate
        assert "estimated_memory_usage_mb" in estimate
        assert "streaming_threshold_mb" in estimate
        assert "requires_streaming" in estimate
        assert "file_breakdown" in estimate
        
        # Check values
        assert estimate["total_size_mb"] > 0
        assert estimate["estimated_memory_usage_mb"] > estimate["total_size_mb"]  # Should include overhead
        assert estimate["streaming_threshold_mb"] == 400.0
        assert estimate["requires_streaming"] is False  # Small dataset
        
        # Check file breakdown
        file_breakdown = estimate["file_breakdown"]
        assert len(file_breakdown) > 0
        for file_path, sizes in file_breakdown.items():
            assert "inputs_mb" in sizes
            assert "labels_mb" in sizes
            assert "total_mb" in sizes
            assert sizes["total_mb"] == sizes["inputs_mb"] + sizes["labels_mb"]
    
    def test_estimate_dataset_memory_usage_large(self):
        """Test memory usage estimation for large dataset."""
        dataset_path = os.path.join(self.temp_dir, "large_dataset")
        self.create_test_dataset_files(dataset_path, size_mb=500)
        
        estimate = estimate_dataset_memory_usage([dataset_path], "train")
        
        assert estimate["total_size_mb"] > 400.0
        assert estimate["requires_streaming"] is True  # Large dataset
        assert estimate["estimated_memory_usage_mb"] > estimate["total_size_mb"]
    
    def test_estimate_dataset_memory_usage_multiple_datasets(self):
        """Test memory usage estimation for multiple datasets."""
        dataset_path1 = os.path.join(self.temp_dir, "dataset1")
        dataset_path2 = os.path.join(self.temp_dir, "dataset2")
        
        self.create_test_dataset_files(dataset_path1, size_mb=50)
        self.create_test_dataset_files(dataset_path2, size_mb=75)
        
        estimate = estimate_dataset_memory_usage([dataset_path1, dataset_path2], "train")
        
        # Should combine sizes from both datasets
        assert estimate["total_size_mb"] > 100.0  # Should be sum of both
        
        # Should have file breakdown for both datasets
        file_breakdown = estimate["file_breakdown"]
        dataset1_files = [k for k in file_breakdown.keys() if "dataset1" in k]
        dataset2_files = [k for k in file_breakdown.keys() if "dataset2" in k]
        
        assert len(dataset1_files) > 0
        assert len(dataset2_files) > 0
    
    def test_estimate_dataset_memory_usage_missing_files(self):
        """Test memory usage estimation with missing files."""
        # Non-existent dataset path
        estimate = estimate_dataset_memory_usage(["/nonexistent/path"], "train")
        
        assert estimate["total_size_mb"] == 0.0
        assert estimate["estimated_memory_usage_mb"] == 0.0
        assert estimate["requires_streaming"] is False
        assert len(estimate["file_breakdown"]) == 0


class TestDatasetManagementIntegration:
    """Integration tests for dataset management system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_dataset(self, dataset_path, num_samples=1000):
        """Create a realistic test dataset."""
        train_dir = os.path.join(dataset_path, "train")
        test_dir = os.path.join(dataset_path, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            "seq_len": 256,
            "vocab_size": 5000,
            "pad_id": 0,
            "ignore_label_id": -100,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": num_samples,
            "total_groups": num_samples // 2,
            "mean_puzzle_examples": 2.0,
            "total_puzzles": num_samples,
            "sets": ["all"]
        }
        
        for split_dir in [train_dir, test_dir]:
            with open(os.path.join(split_dir, "dataset.json"), "w") as f:
                json.dump(metadata, f)
            
            # Create realistic data
            split_samples = num_samples if "train" in split_dir else num_samples // 5
            
            inputs = np.random.randint(0, 5000, (split_samples, 256), dtype=np.int32)
            labels = np.random.randint(0, 10, split_samples, dtype=np.int32)
            puzzle_identifiers = np.arange(split_samples, dtype=np.int32)
            puzzle_indices = np.arange(split_samples + 1, dtype=np.int32)
            group_indices = np.arange(split_samples // 2 + 1, dtype=np.int32)
            
            np.save(os.path.join(split_dir, "all__inputs.npy"), inputs)
            np.save(os.path.join(split_dir, "all__labels.npy"), labels)
            np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
            np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
            np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_end_to_end_dataset_management(self):
        """Test complete dataset management pipeline."""
        # Create realistic dataset
        dataset_path = os.path.join(self.temp_dir, "realistic_dataset")
        self.create_realistic_dataset(dataset_path, num_samples=500)
        
        # Create dataset manager
        config = DatasetManagementConfig(
            max_dataset_memory_mb=200.0,  # Force streaming for testing
            streaming_threshold_mb=100.0,
            enable_caching=True,
            cache_dir=os.path.join(self.temp_dir, "cache")
        )
        
        memory_manager = MemoryManager()
        dataset_manager = DatasetManager(config, memory_manager)
        
        # Analyze dataset
        analysis = dataset_manager.analyze_dataset_requirements([dataset_path], "train")
        
        assert analysis["total_size_mb"] > 0
        assert analysis["recommended_strategy"] in ["cached", "memory_mapped", "streaming"]
        
        # Create memory-efficient dataset
        puzzle_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[dataset_path],
            global_batch_size=16,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset, creation_info = dataset_manager.create_dataloader_with_fallback(
            [dataset_path], puzzle_config, "train"
        )
        
        assert dataset is not None
        assert "final_strategy" in creation_info
        assert "fallback_used" in creation_info
        
        # Test dataset iteration
        if TORCH_AVAILABLE:
            dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
            
            batches_processed = 0
            for set_name, batch, batch_size in dataloader:
                assert isinstance(set_name, str)
                assert isinstance(batch, dict)
                assert "inputs" in batch
                assert "labels" in batch
                assert batch_size > 0
                
                batches_processed += 1
                if batches_processed >= 3:  # Test first few batches
                    break
            
            assert batches_processed > 0
        
        # Check metrics were recorded
        metrics = dataset_manager.get_loading_metrics()
        assert len(metrics) > 0
        
        final_metrics = metrics[-1]
        assert final_metrics.total_size_mb > 0
        assert final_metrics.loading_strategy in ["cached", "memory_mapped", "streaming"]
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure during dataset operations."""
        dataset_path = os.path.join(self.temp_dir, "pressure_test_dataset")
        self.create_realistic_dataset(dataset_path, num_samples=200)
        
        # Mock memory manager with varying memory pressure
        mock_memory_manager = Mock(spec=MemoryManager)
        
        # Start with normal memory
        normal_memory = Mock()
        normal_memory.available_mb = 4000.0
        normal_memory.percent_used = 50.0
        
        # High memory pressure
        high_memory = Mock()
        high_memory.available_mb = 500.0
        high_memory.percent_used = 90.0
        
        # Simulate memory pressure increase
        mock_memory_manager.monitor_memory_usage.side_effect = [normal_memory, high_memory, high_memory]
        
        config = DatasetManagementConfig(auto_fallback_streaming=True)
        dataset_manager = DatasetManager(config, mock_memory_manager)
        
        # Validate constraints with high memory pressure
        validation = dataset_manager.validate_dataset_memory_constraints(
            [dataset_path], batch_size=32, split="train"
        )
        
        # Should recommend fallback due to memory pressure
        assert validation["fallback_to_streaming"] is True
        assert len(validation["recommendations"]) > 0
        
        # Create dataset with fallback
        puzzle_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[dataset_path],
            global_batch_size=32,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset, creation_info = dataset_manager.create_dataloader_with_fallback(
            [dataset_path], puzzle_config, "train"
        )
        
        # Should use streaming strategy due to memory pressure
        assert creation_info["final_strategy"] == "streaming"
        assert creation_info["fallback_used"] is True
        
        # Should have triggered garbage collection
        assert mock_memory_manager.force_garbage_collection.called
    
    def test_batch_size_optimization_integration(self):
        """Test batch size optimization integration."""
        dataset_path = os.path.join(self.temp_dir, "batch_opt_dataset")
        self.create_realistic_dataset(dataset_path, num_samples=300)
        
        memory_manager = MemoryManager()
        dataset_manager = DatasetManager(memory_manager=memory_manager)
        
        # Test batch size optimization
        optimization = dataset_manager.optimize_batch_size_for_dataset(
            [dataset_path], initial_batch_size=64, split="train"
        )
        
        assert optimization["initial_batch_size"] == 64
        assert optimization["recommended_batch_size"] <= 32  # MacBook limit
        assert optimization["recommended_batch_size"] > 0
        assert optimization["max_safe_batch_size"] >= optimization["recommended_batch_size"]
        
        # Use optimized batch size in dataset creation
        puzzle_config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=[dataset_path],
            global_batch_size=optimization["recommended_batch_size"],
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset, creation_info = dataset_manager.create_dataloader_with_fallback(
            [dataset_path], puzzle_config, "train"
        )
        
        assert dataset is not None
        assert creation_info["final_strategy"] in ["cached", "memory_mapped", "streaming"]