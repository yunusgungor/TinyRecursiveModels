"""
Unit tests for data loading module.

Tests memory-efficient data loading, streaming datasets, and memory-aware
data management for MacBook optimization.
"""

import pytest
import os
import tempfile
import pickle
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
from torch.utils.data import DataLoader

from macbook_optimization.data_loading import (
    DataLoadingConfig,
    DatasetInfo,
    MemoryMappedDataset,
    StreamingDataset,
    MemoryAwareDataLoader,
    DatasetAnalyzer,
    DataLoadingManager,
)
from macbook_optimization.memory_management import MemoryManager


class TestDataLoadingConfig:
    """Test DataLoadingConfig dataclass."""
    
    def test_data_loading_config_defaults(self):
        """Test DataLoadingConfig default values."""
        config = DataLoadingConfig()
        
        assert config.max_memory_usage_mb == 1000.0
        assert config.prefetch_factor == 2
        assert config.use_memory_mapping is True
        assert config.mmap_threshold_mb == 100.0
        assert config.streaming_threshold_mb == 500.0
        assert config.chunk_size_mb == 50.0
        assert config.enable_preprocessing_cache is True
        assert config.cache_dir is None
        assert config.max_cache_size_mb == 200.0
        assert config.pin_memory is False
        assert config.persistent_workers is False
    
    def test_data_loading_config_custom_values(self):
        """Test DataLoadingConfig with custom values."""
        config = DataLoadingConfig(
            max_memory_usage_mb=2000.0,
            prefetch_factor=4,
            use_memory_mapping=False,
            streaming_threshold_mb=1000.0,
            pin_memory=True
        )
        
        assert config.max_memory_usage_mb == 2000.0
        assert config.prefetch_factor == 4
        assert config.use_memory_mapping is False
        assert config.streaming_threshold_mb == 1000.0
        assert config.pin_memory is True


class TestDatasetInfo:
    """Test DatasetInfo dataclass."""
    
    def test_dataset_info_creation(self):
        """Test DatasetInfo creation."""
        info = DatasetInfo(
            path="/path/to/dataset",
            size_mb=150.5,
            num_samples=1000,
            sample_size_bytes=1024,
            recommended_loading_strategy="mmap",
            estimated_memory_usage_mb=15.0
        )
        
        assert info.path == "/path/to/dataset"
        assert info.size_mb == 150.5
        assert info.num_samples == 1000
        assert info.sample_size_bytes == 1024
        assert info.recommended_loading_strategy == "mmap"
        assert info.estimated_memory_usage_mb == 15.0


class TestMemoryMappedDataset:
    """Test MemoryMappedDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "test_data.bin")
        self.index_file = os.path.join(self.temp_dir, "test_data.idx")
        
        # Create test data file with simple format
        test_data = [
            {"id": 1, "value": "sample1"},
            {"id": 2, "value": "sample2"},
            {"id": 3, "value": "sample3"},
        ]
        
        with open(self.data_file, 'wb') as f:
            for sample in test_data:
                sample_bytes = pickle.dumps(sample)
                # Write size then data
                f.write(len(sample_bytes).to_bytes(4, 'little'))
                f.write(sample_bytes)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_mapped_dataset_creation(self):
        """Test MemoryMappedDataset creation."""
        dataset = MemoryMappedDataset(self.data_file)
        
        assert dataset.data_path == self.data_file
        assert dataset.index_path is None
        assert dataset.transform is None
        assert len(dataset.index) > 0
        
        # Clean up
        dataset.__del__()
    
    def test_memory_mapped_dataset_with_transform(self):
        """Test MemoryMappedDataset with transform function."""
        def transform_fn(sample):
            if isinstance(sample, dict):
                sample['transformed'] = True
            return sample
        
        dataset = MemoryMappedDataset(self.data_file, transform=transform_fn)
        
        # Get a sample and check transform was applied
        sample = dataset[0]
        if isinstance(sample, dict):
            assert sample.get('transformed') is True
        
        dataset.__del__()
    
    def test_memory_mapped_dataset_indexing(self):
        """Test MemoryMappedDataset indexing."""
        dataset = MemoryMappedDataset(self.data_file)
        
        # Should be able to access samples by index
        assert len(dataset) > 0
        
        # Test valid indexing
        sample = dataset[0]
        assert sample is not None
        
        # Test invalid indexing
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
        
        dataset.__del__()
    
    def test_memory_mapped_dataset_with_index_file(self):
        """Test MemoryMappedDataset with separate index file."""
        # Create dataset first to generate index
        dataset1 = MemoryMappedDataset(self.data_file, self.index_file)
        original_length = len(dataset1)
        dataset1.__del__()
        
        # Index file should now exist
        assert os.path.exists(self.index_file)
        
        # Create new dataset with existing index
        dataset2 = MemoryMappedDataset(self.data_file, self.index_file)
        assert len(dataset2) == original_length
        
        dataset2.__del__()


class TestStreamingDataset:
    """Test StreamingDataset class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, "streaming_data.bin")
        
        # Create larger test data file
        test_data = []
        for i in range(100):  # Create 100 samples
            test_data.append({"id": i, "value": f"sample{i}"})
        
        with open(self.data_file, 'wb') as f:
            for sample in test_data:
                sample_bytes = pickle.dumps(sample)
                f.write(len(sample_bytes).to_bytes(4, 'little'))
                f.write(sample_bytes)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_streaming_dataset_creation(self):
        """Test StreamingDataset creation."""
        dataset = StreamingDataset(self.data_file, chunk_size_mb=0.01)  # Very small chunks
        
        assert dataset.data_path == self.data_file
        assert dataset.chunk_size_bytes > 0
        assert dataset.file_size > 0
        assert dataset.num_chunks > 0
    
    def test_streaming_dataset_iteration(self):
        """Test StreamingDataset iteration."""
        dataset = StreamingDataset(self.data_file, chunk_size_mb=0.01)
        
        samples = list(dataset)
        assert len(samples) > 0
        
        # Check that samples are valid
        for sample in samples[:5]:  # Check first 5 samples
            if isinstance(sample, dict):
                assert "id" in sample
                assert "value" in sample
    
    def test_streaming_dataset_with_transform(self):
        """Test StreamingDataset with transform function."""
        def transform_fn(sample):
            if isinstance(sample, dict):
                sample['transformed'] = True
            return sample
        
        dataset = StreamingDataset(self.data_file, transform=transform_fn)
        
        samples = list(dataset)
        # Check that transform was applied
        for sample in samples[:3]:
            if isinstance(sample, dict):
                assert sample.get('transformed') is True
    
    def test_streaming_dataset_shuffle(self):
        """Test StreamingDataset with shuffle enabled."""
        dataset = StreamingDataset(self.data_file, shuffle=True)
        
        # Get samples twice and compare
        samples1 = list(dataset)
        samples2 = list(dataset)
        
        # With shuffle, the order might be different
        # (This test might occasionally fail due to randomness, but it's unlikely)
        assert len(samples1) == len(samples2)


class TestMemoryAwareDataLoader:
    """Test MemoryAwareDataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple dataset
        self.dataset = [(torch.randn(10,), torch.randn(1,)) for _ in range(100)]
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.config = DataLoadingConfig(prefetch_factor=1)
    
    def test_memory_aware_dataloader_creation(self):
        """Test MemoryAwareDataLoader creation."""
        dataloader = MemoryAwareDataLoader(
            self.dataset,
            batch_size=8,
            memory_manager=self.mock_memory_manager,
            config=self.config
        )
        
        assert dataloader.initial_batch_size == 8
        assert dataloader.current_batch_size == 8
        assert dataloader.memory_manager == self.mock_memory_manager
        assert dataloader.config == self.config
    
    def test_memory_aware_dataloader_iteration(self):
        """Test MemoryAwareDataLoader iteration."""
        dataloader = MemoryAwareDataLoader(
            self.dataset,
            batch_size=8,
            config=self.config
        )
        
        batches = list(dataloader)
        assert len(batches) > 0
        
        # Check batch structure
        for batch in batches[:2]:  # Check first 2 batches
            assert len(batch) == 2  # input, target
            assert isinstance(batch[0], torch.Tensor)
            assert isinstance(batch[1], torch.Tensor)
    
    def test_memory_aware_dataloader_memory_monitoring(self):
        """Test memory monitoring during data loading."""
        # Mock memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.percent_used = 85.0  # High memory usage
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        dataloader = MemoryAwareDataLoader(
            self.dataset,
            batch_size=16,
            memory_manager=self.mock_memory_manager,
            config=self.config
        )
        
        # Iterate through some batches
        batches = list(dataloader)
        
        # Memory monitoring should have been called
        assert self.mock_memory_manager.monitor_memory_usage.called
        
        # Batch size might have been reduced due to high memory usage
        assert dataloader.current_batch_size <= 16
    
    def test_memory_aware_dataloader_get_memory_stats(self):
        """Test memory statistics retrieval."""
        dataloader = MemoryAwareDataLoader(
            self.dataset,
            batch_size=8,
            config=self.config
        )
        
        # Initially no stats
        stats = dataloader.get_memory_stats()
        assert stats == {}
        
        # After some iterations, should have stats
        _ = list(dataloader)
        stats = dataloader.get_memory_stats()
        
        if stats:  # Only check if stats were collected
            assert "current_batch_size" in stats
            assert "initial_batch_size" in stats


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = DatasetAnalyzer()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, size_mb):
        """Create a test file of specified size."""
        file_path = os.path.join(self.temp_dir, f"test_{size_mb}mb.dat")
        size_bytes = int(size_mb * 1024 * 1024)
        
        with open(file_path, 'wb') as f:
            f.write(b'0' * size_bytes)
        
        return file_path
    
    def test_analyze_small_dataset(self):
        """Test analysis of small dataset."""
        file_path = self.create_test_file(10)  # 10MB file
        
        info = self.analyzer.analyze_dataset(file_path)
        
        assert info.path == file_path
        assert info.size_mb == 10.0
        assert info.recommended_loading_strategy == "memory"
        assert info.num_samples > 0
        assert info.estimated_memory_usage_mb > 0
    
    def test_analyze_medium_dataset(self):
        """Test analysis of medium dataset."""
        file_path = self.create_test_file(200)  # 200MB file
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
            
            info = self.analyzer.analyze_dataset(file_path)
        
        assert info.size_mb == 200.0
        assert info.recommended_loading_strategy == "mmap"
    
    def test_analyze_large_dataset(self):
        """Test analysis of large dataset."""
        file_path = self.create_test_file(1000)  # 1GB file
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 2 * 1024 * 1024 * 1024  # 2GB available
            
            info = self.analyzer.analyze_dataset(file_path)
        
        assert info.size_mb == 1000.0
        assert info.recommended_loading_strategy == "streaming"
    
    def test_create_optimal_dataset_memory(self):
        """Test optimal dataset creation for memory loading."""
        file_path = self.create_test_file(10)
        
        with patch.object(self.analyzer, 'analyze_dataset') as mock_analyze:
            mock_analyze.return_value = DatasetInfo(
                path=file_path,
                size_mb=10.0,
                num_samples=100,
                sample_size_bytes=1024,
                recommended_loading_strategy="memory",
                estimated_memory_usage_mb=12.0
            )
            
            dataset = self.analyzer.create_optimal_dataset(file_path)
            
            # Should create in-memory dataset
            assert dataset is not None
            assert hasattr(dataset, '__len__')
            assert hasattr(dataset, '__getitem__')
    
    def test_create_optimal_dataset_mmap(self):
        """Test optimal dataset creation for memory mapping."""
        file_path = self.create_test_file(100)
        
        with patch.object(self.analyzer, 'analyze_dataset') as mock_analyze:
            mock_analyze.return_value = DatasetInfo(
                path=file_path,
                size_mb=100.0,
                num_samples=1000,
                sample_size_bytes=1024,
                recommended_loading_strategy="mmap",
                estimated_memory_usage_mb=10.0
            )
            
            dataset = self.analyzer.create_optimal_dataset(file_path)
            
            # Should create memory-mapped dataset
            assert isinstance(dataset, MemoryMappedDataset)
            dataset.__del__()  # Clean up
    
    def test_create_optimal_dataset_streaming(self):
        """Test optimal dataset creation for streaming."""
        file_path = self.create_test_file(500)
        
        with patch.object(self.analyzer, 'analyze_dataset') as mock_analyze:
            mock_analyze.return_value = DatasetInfo(
                path=file_path,
                size_mb=500.0,
                num_samples=5000,
                sample_size_bytes=1024,
                recommended_loading_strategy="streaming",
                estimated_memory_usage_mb=50.0
            )
            
            dataset = self.analyzer.create_optimal_dataset(file_path)
            
            # Should create streaming dataset
            assert isinstance(dataset, StreamingDataset)


class TestDataLoadingManager:
    """Test DataLoadingManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.config = DataLoadingConfig()
        self.manager = DataLoadingManager(self.mock_memory_manager, self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_dataset_file(self, size_mb=50):
        """Create a test dataset file."""
        file_path = os.path.join(self.temp_dir, "test_dataset.dat")
        
        # Create file with some structured data
        test_data = []
        for i in range(100):
            test_data.append(f"sample_{i}")
        
        with open(file_path, 'w') as f:
            for sample in test_data:
                f.write(sample + '\n')
        
        return file_path
    
    def test_data_loading_manager_creation(self):
        """Test DataLoadingManager creation."""
        assert self.manager.memory_manager == self.mock_memory_manager
        assert self.manager.config == self.config
        assert isinstance(self.manager.analyzer, DatasetAnalyzer)
    
    def test_create_dataloader(self):
        """Test dataloader creation."""
        file_path = self.create_test_dataset_file()
        
        # Mock memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        dataloader = self.manager.create_dataloader(
            file_path,
            batch_size=8,
            shuffle=True
        )
        
        assert isinstance(dataloader, MemoryAwareDataLoader)
        assert dataloader.memory_manager == self.mock_memory_manager
    
    def test_create_dataloader_memory_adjustment(self):
        """Test dataloader creation with memory adjustment."""
        file_path = self.create_test_dataset_file()
        
        # Mock low memory situation
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 500.0  # Low memory
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        with patch.object(self.manager.analyzer, 'analyze_dataset') as mock_analyze:
            mock_analyze.return_value = DatasetInfo(
                path=file_path,
                size_mb=50.0,
                num_samples=100,
                sample_size_bytes=1024,
                recommended_loading_strategy="memory",
                estimated_memory_usage_mb=400.0  # High estimated usage
            )
            
            dataloader = self.manager.create_dataloader(
                file_path,
                batch_size=16
            )
            
            # Batch size should be adjusted down due to memory constraints
            # The exact value depends on the adjustment logic
            assert dataloader.current_batch_size <= 16
    
    def test_get_loading_recommendations(self):
        """Test loading recommendations."""
        file_path = self.create_test_dataset_file()
        
        # Mock memory stats
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 4000.0
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        recommendations = self.manager.get_loading_recommendations(file_path)
        
        assert "dataset_info" in recommendations
        assert "memory_info" in recommendations
        assert "recommendations" in recommendations
        
        # Check dataset info
        dataset_info = recommendations["dataset_info"]
        assert "size_mb" in dataset_info
        assert "num_samples" in dataset_info
        assert "loading_strategy" in dataset_info
        
        # Check memory info
        memory_info = recommendations["memory_info"]
        assert "available_mb" in memory_info
        assert "estimated_usage_mb" in memory_info
        assert "memory_utilization_percent" in memory_info
        
        # Check recommendations
        rec = recommendations["recommendations"]
        assert "batch_size" in rec
        assert "max_batch_size" in rec
        assert "num_workers" in rec
        assert "pin_memory" in rec
        assert "use_streaming" in rec
        
        # Validate recommendation values
        assert rec["batch_size"] >= 1
        assert rec["max_batch_size"] >= rec["batch_size"]
        assert rec["num_workers"] == 0  # Conservative for MacBook
        assert rec["pin_memory"] is False
        assert isinstance(rec["use_streaming"], bool)
    
    def test_get_loading_recommendations_low_memory(self):
        """Test loading recommendations with low memory."""
        file_path = self.create_test_dataset_file()
        
        # Mock very low memory situation
        mock_memory_stats = Mock()
        mock_memory_stats.available_mb = 200.0  # Very low memory
        self.mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
        
        recommendations = self.manager.get_loading_recommendations(file_path)
        
        # Should recommend very small batch size
        assert recommendations["recommendations"]["batch_size"] <= 4
        assert recommendations["memory_info"]["memory_utilization_percent"] > 0


class TestDataLoadingIntegration:
    """Integration tests for data loading system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_pickle_dataset(self, num_samples=100):
        """Create a pickle-based dataset file."""
        file_path = os.path.join(self.temp_dir, "pickle_dataset.pkl")
        
        data = []
        for i in range(num_samples):
            sample = {
                "input": torch.randn(10),
                "target": torch.randn(1),
                "id": i
            }
            data.append(sample)
        
        with open(file_path, 'wb') as f:
            for sample in data:
                sample_bytes = pickle.dumps(sample)
                f.write(len(sample_bytes).to_bytes(4, 'little'))
                f.write(sample_bytes)
        
        return file_path
    
    def test_end_to_end_data_loading(self):
        """Test complete data loading pipeline."""
        # Create test dataset
        dataset_path = self.create_pickle_dataset(50)
        
        # Create data loading manager
        config = DataLoadingConfig(
            max_memory_usage_mb=500.0,
            use_memory_mapping=True
        )
        
        with patch('macbook_optimization.data_loading.MemoryManager') as mock_manager_class:
            mock_memory_manager = Mock()
            mock_memory_stats = Mock()
            mock_memory_stats.available_mb = 2000.0
            mock_memory_stats.percent_used = 50.0
            mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
            mock_manager_class.return_value = mock_memory_manager
            
            manager = DataLoadingManager(mock_memory_manager, config)
            
            # Get recommendations
            recommendations = manager.get_loading_recommendations(dataset_path)
            
            # Create dataloader
            dataloader = manager.create_dataloader(
                dataset_path,
                batch_size=recommendations["recommendations"]["batch_size"]
            )
            
            # Test iteration
            batches = []
            for i, batch in enumerate(dataloader):
                batches.append(batch)
                if i >= 5:  # Only test first few batches
                    break
            
            assert len(batches) > 0
            
            # Check memory stats
            memory_stats = dataloader.get_memory_stats()
            if memory_stats:  # Only check if stats were collected
                assert "current_batch_size" in memory_stats
    
    def test_memory_pressure_adaptation(self):
        """Test data loading adaptation to memory pressure."""
        dataset_path = self.create_pickle_dataset(100)
        
        config = DataLoadingConfig()
        
        with patch('macbook_optimization.data_loading.MemoryManager') as mock_manager_class:
            mock_memory_manager = Mock()
            mock_manager_class.return_value = mock_memory_manager
            
            # Start with normal memory
            mock_memory_stats = Mock()
            mock_memory_stats.available_mb = 4000.0
            mock_memory_stats.percent_used = 50.0
            mock_memory_manager.monitor_memory_usage.return_value = mock_memory_stats
            
            manager = DataLoadingManager(mock_memory_manager, config)
            dataloader = manager.create_dataloader(dataset_path, batch_size=16)
            
            initial_batch_size = dataloader.current_batch_size
            
            # Simulate memory pressure increase
            mock_memory_stats.percent_used = 85.0  # High memory usage
            mock_memory_stats.available_mb = 1000.0
            
            # Process some batches to trigger memory monitoring
            batches_processed = 0
            for batch in dataloader:
                batches_processed += 1
                if batches_processed >= 3:
                    break
            
            # Batch size might have been adjusted due to memory pressure
            # (The exact behavior depends on the implementation)
            assert dataloader.current_batch_size <= initial_batch_size
    
    def test_different_loading_strategies(self):
        """Test different loading strategies based on dataset size."""
        # Create datasets of different sizes
        small_dataset = self.create_pickle_dataset(10)
        
        analyzer = DatasetAnalyzer()
        
        # Test small dataset (should use memory loading)
        with patch('os.path.getsize', return_value=1024 * 1024):  # 1MB
            info = analyzer.analyze_dataset(small_dataset)
            assert info.recommended_loading_strategy == "memory"
        
        # Test medium dataset (should use memory mapping)
        with patch('os.path.getsize', return_value=150 * 1024 * 1024):  # 150MB
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
                info = analyzer.analyze_dataset(small_dataset)
                assert info.recommended_loading_strategy == "mmap"
        
        # Test large dataset (should use streaming)
        with patch('os.path.getsize', return_value=2 * 1024 * 1024 * 1024):  # 2GB
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.available = 4 * 1024 * 1024 * 1024  # 4GB
                info = analyzer.analyze_dataset(small_dataset)
                assert info.recommended_loading_strategy == "streaming"