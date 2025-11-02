"""
Memory-efficient data loading module for MacBook TRM training optimization.

This module provides memory-mapped dataset loading, streaming datasets,
and memory-aware data prefetching for efficient training on memory-constrained
MacBook hardware.
"""

import os
import mmap
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Union, Tuple
import numpy as np
import psutil

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object
    IterableDataset = object
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager


@dataclass
class DataLoadingConfig:
    """Configuration for memory-efficient data loading."""
    # Memory management
    max_memory_usage_mb: float = 1000.0  # Maximum memory for data loading
    prefetch_factor: int = 2  # Number of batches to prefetch
    
    # Memory mapping
    use_memory_mapping: bool = True
    mmap_threshold_mb: float = 100.0  # Use mmap for files larger than this
    
    # Streaming
    streaming_threshold_mb: float = 500.0  # Use streaming for datasets larger than this
    chunk_size_mb: float = 50.0  # Size of chunks for streaming
    
    # Caching
    enable_preprocessing_cache: bool = True
    cache_dir: Optional[str] = None
    max_cache_size_mb: float = 200.0
    
    # Data loading
    pin_memory: bool = False  # Usually False for CPU training
    persistent_workers: bool = False  # Can cause memory issues on MacBook


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    path: str
    size_mb: float
    num_samples: int
    sample_size_bytes: int
    recommended_loading_strategy: str  # "memory", "mmap", "streaming"
    estimated_memory_usage_mb: float


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for efficient large file access."""
    
    def __init__(self, data_path: str, index_path: Optional[str] = None,
                 transform: Optional[callable] = None):
        """
        Initialize memory-mapped dataset.
        
        Args:
            data_path: Path to data file
            index_path: Path to index file (optional)
            transform: Data transformation function
        """
        self.data_path = data_path
        self.index_path = index_path
        self.transform = transform
        
        # Open file and create memory map
        self.file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Load or create index
        self._load_index()
        
    def _load_index(self):
        """Load or create index for fast sample access."""
        if self.index_path and os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
        else:
            # Create index by scanning the file
            self.index = self._create_index()
            if self.index_path:
                with open(self.index_path, 'wb') as f:
                    pickle.dump(self.index, f)
    
    def _create_index(self) -> List[Tuple[int, int]]:
        """Create index of sample positions in the file."""
        # This is a simplified implementation
        # In practice, this would depend on the specific data format
        index = []
        position = 0
        
        # For demonstration, assume fixed-size records
        # In practice, you'd parse the actual file format
        try:
            while position < len(self.mmap):
                # Try to read a sample size indicator
                if position + 4 > len(self.mmap):
                    break
                    
                # Read sample size (assuming 4-byte integer)
                sample_size = int.from_bytes(self.mmap[position:position+4], 'little')
                
                if sample_size <= 0 or position + 4 + sample_size > len(self.mmap):
                    break
                
                index.append((position + 4, sample_size))
                position += 4 + sample_size
                
        except Exception:
            # Fallback: assume all samples are the same size
            if len(self.mmap) > 0:
                estimated_sample_size = min(1024, len(self.mmap) // 100)  # Estimate
                for i in range(0, len(self.mmap), estimated_sample_size):
                    if i + estimated_sample_size <= len(self.mmap):
                        index.append((i, estimated_sample_size))
        
        return index
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Any:
        if idx >= len(self.index):
            raise IndexError(f"Index {idx} out of range")
        
        position, size = self.index[idx]
        
        # Read data from memory map
        data_bytes = self.mmap[position:position + size]
        
        # Deserialize data (format-dependent)
        try:
            # Try pickle first
            data = pickle.loads(data_bytes)
        except Exception:
            try:
                # Try JSON
                data = json.loads(data_bytes.decode('utf-8'))
            except Exception:
                # Fallback to raw bytes
                data = data_bytes
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()


class StreamingDataset(IterableDataset):
    """Streaming dataset for very large datasets that don't fit in memory."""
    
    def __init__(self, data_path: str, chunk_size_mb: float = 50.0,
                 transform: Optional[callable] = None, shuffle: bool = False):
        """
        Initialize streaming dataset.
        
        Args:
            data_path: Path to data file
            chunk_size_mb: Size of chunks to load at once
            transform: Data transformation function
            shuffle: Whether to shuffle data (within chunks)
        """
        self.data_path = data_path
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.transform = transform
        self.shuffle = shuffle
        
        # Get file size
        self.file_size = os.path.getsize(data_path)
        self.num_chunks = (self.file_size + self.chunk_size_bytes - 1) // self.chunk_size_bytes
        
    def _read_chunk(self, chunk_idx: int) -> List[Any]:
        """Read a chunk of data from the file."""
        start_pos = chunk_idx * self.chunk_size_bytes
        end_pos = min(start_pos + self.chunk_size_bytes, self.file_size)
        
        with open(self.data_path, 'rb') as f:
            f.seek(start_pos)
            chunk_data = f.read(end_pos - start_pos)
        
        # Parse chunk data (format-dependent)
        samples = []
        position = 0
        
        while position < len(chunk_data):
            try:
                # Try to read sample size
                if position + 4 > len(chunk_data):
                    break
                
                sample_size = int.from_bytes(chunk_data[position:position+4], 'little')
                if sample_size <= 0 or position + 4 + sample_size > len(chunk_data):
                    break
                
                sample_bytes = chunk_data[position+4:position+4+sample_size]
                
                # Deserialize sample
                try:
                    sample = pickle.loads(sample_bytes)
                except Exception:
                    try:
                        sample = json.loads(sample_bytes.decode('utf-8'))
                    except Exception:
                        sample = sample_bytes
                
                samples.append(sample)
                position += 4 + sample_size
                
            except Exception:
                break
        
        return samples
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over dataset samples."""
        chunk_indices = list(range(self.num_chunks))
        
        if self.shuffle:
            np.random.shuffle(chunk_indices)
        
        for chunk_idx in chunk_indices:
            chunk_samples = self._read_chunk(chunk_idx)
            
            if self.shuffle:
                np.random.shuffle(chunk_samples)
            
            for sample in chunk_samples:
                if self.transform:
                    sample = self.transform(sample)
                yield sample


class MemoryAwareDataLoader:
    """Data loader with memory-aware prefetching and batch management."""
    
    def __init__(self, dataset: Dataset, batch_size: int,
                 memory_manager: Optional[MemoryManager] = None,
                 config: Optional[DataLoadingConfig] = None,
                 **dataloader_kwargs):
        """
        Initialize memory-aware data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            memory_manager: Memory manager for monitoring
            config: Data loading configuration
            **dataloader_kwargs: Additional arguments for DataLoader
        """
        self.dataset = dataset
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size
        self.memory_manager = memory_manager
        self.config = config or DataLoadingConfig()
        
        # Setup DataLoader with memory-conscious settings
        dataloader_kwargs.setdefault('num_workers', 0)  # Conservative for MacBook
        dataloader_kwargs.setdefault('pin_memory', self.config.pin_memory)
        dataloader_kwargs.setdefault('persistent_workers', self.config.persistent_workers)
        dataloader_kwargs.setdefault('prefetch_factor', self.config.prefetch_factor)
        
        self.dataloader_kwargs = dataloader_kwargs
        self._create_dataloader()
        
        # Memory monitoring
        self.memory_usage_history = []
        
    def _create_dataloader(self):
        """Create or recreate the DataLoader with current settings."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.current_batch_size,
            **self.dataloader_kwargs
        )
    
    def _monitor_memory_usage(self):
        """Monitor memory usage during data loading."""
        if self.memory_manager:
            memory_stats = self.memory_manager.monitor_memory_usage()
            self.memory_usage_history.append(memory_stats.percent_used)
            
            # Keep only recent history
            if len(self.memory_usage_history) > 100:
                self.memory_usage_history = self.memory_usage_history[-50:]
            
            # Adjust batch size if memory pressure is high
            if memory_stats.percent_used > 80 and self.current_batch_size > 1:
                new_batch_size = max(1, int(self.current_batch_size * 0.8))
                if new_batch_size != self.current_batch_size:
                    print(f"Reducing batch size due to memory pressure: "
                          f"{self.current_batch_size} -> {new_batch_size}")
                    self.current_batch_size = new_batch_size
                    self._create_dataloader()
    
    def __iter__(self):
        """Iterate over batches with memory monitoring."""
        for batch in self.dataloader:
            self._monitor_memory_usage()
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics for data loading."""
        if not self.memory_usage_history:
            return {}
        
        return {
            "current_batch_size": self.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "avg_memory_usage": np.mean(self.memory_usage_history),
            "max_memory_usage": np.max(self.memory_usage_history),
            "memory_samples": len(self.memory_usage_history),
        }


class DatasetAnalyzer:
    """Analyzer for determining optimal data loading strategy."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize dataset analyzer.
        
        Args:
            memory_manager: Memory manager for memory constraints
        """
        self.memory_manager = memory_manager
    
    def analyze_dataset(self, data_path: str) -> DatasetInfo:
        """
        Analyze dataset and recommend loading strategy.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Dataset information and recommendations
        """
        # Get file size
        file_size_bytes = os.path.getsize(data_path)
        size_mb = file_size_bytes / (1024 * 1024)
        
        # Estimate number of samples (rough heuristic)
        # This would be more accurate with format-specific parsing
        estimated_sample_size = 1024  # 1KB per sample estimate
        num_samples = max(1, file_size_bytes // estimated_sample_size)
        
        # Get available memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if self.memory_manager:
            memory_stats = self.memory_manager.monitor_memory_usage()
            available_memory_mb = memory_stats.available_mb
        
        # Determine loading strategy
        if size_mb < 50:  # Small dataset
            strategy = "memory"
            estimated_memory_usage = size_mb * 1.2  # 20% overhead
        elif size_mb < available_memory_mb * 0.3:  # Medium dataset
            strategy = "mmap"
            estimated_memory_usage = min(size_mb * 0.1, 100)  # Memory mapping overhead
        else:  # Large dataset
            strategy = "streaming"
            estimated_memory_usage = min(100, available_memory_mb * 0.1)  # Streaming buffer
        
        return DatasetInfo(
            path=data_path,
            size_mb=size_mb,
            num_samples=num_samples,
            sample_size_bytes=estimated_sample_size,
            recommended_loading_strategy=strategy,
            estimated_memory_usage_mb=estimated_memory_usage
        )
    
    def create_optimal_dataset(self, data_path: str, 
                             transform: Optional[callable] = None) -> Dataset:
        """
        Create optimal dataset based on analysis.
        
        Args:
            data_path: Path to dataset file
            transform: Data transformation function
            
        Returns:
            Optimized dataset instance
        """
        info = self.analyze_dataset(data_path)
        
        if info.recommended_loading_strategy == "memory":
            # Load entire dataset into memory (for small datasets)
            # This would be a custom implementation based on data format
            return self._create_memory_dataset(data_path, transform)
        
        elif info.recommended_loading_strategy == "mmap":
            # Use memory mapping
            return MemoryMappedDataset(data_path, transform=transform)
        
        else:  # streaming
            # Use streaming dataset
            return StreamingDataset(data_path, transform=transform)
    
    def _create_memory_dataset(self, data_path: str, 
                             transform: Optional[callable] = None) -> Dataset:
        """Create in-memory dataset for small files."""
        # This is a simplified implementation
        # In practice, you'd implement format-specific loading
        
        class InMemoryDataset(Dataset):
            def __init__(self, path: str, transform: Optional[callable] = None):
                self.transform = transform
                self.data = self._load_data(path)
            
            def _load_data(self, path: str) -> List[Any]:
                # Load all data into memory
                # This would be format-specific
                data = []
                try:
                    with open(path, 'rb') as f:
                        while True:
                            try:
                                sample = pickle.load(f)
                                data.append(sample)
                            except EOFError:
                                break
                except Exception:
                    # Fallback: read as text lines
                    with open(path, 'r') as f:
                        data = [line.strip() for line in f if line.strip()]
                
                return data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                if self.transform:
                    sample = self.transform(sample)
                return sample
        
        return InMemoryDataset(data_path, transform)


class DataLoadingManager:
    """High-level manager for memory-efficient data loading."""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None,
                 config: Optional[DataLoadingConfig] = None):
        """
        Initialize data loading manager.
        
        Args:
            memory_manager: Memory manager instance
            config: Data loading configuration
        """
        self.memory_manager = memory_manager
        self.config = config or DataLoadingConfig()
        self.analyzer = DatasetAnalyzer(memory_manager)
        
        # Cache for preprocessed datasets
        self.cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataloader(self, data_path: str, batch_size: int,
                         transform: Optional[callable] = None,
                         **kwargs) -> MemoryAwareDataLoader:
        """
        Create optimal data loader for given dataset.
        
        Args:
            data_path: Path to dataset
            batch_size: Batch size
            transform: Data transformation function
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Memory-aware data loader
        """
        # Analyze dataset
        dataset_info = self.analyzer.analyze_dataset(data_path)
        
        # Adjust batch size based on memory constraints
        if self.memory_manager:
            memory_stats = self.memory_manager.monitor_memory_usage()
            if memory_stats.available_mb < dataset_info.estimated_memory_usage_mb * 2:
                # Reduce batch size if memory is tight
                adjusted_batch_size = max(1, batch_size // 2)
                print(f"Adjusting batch size for memory constraints: {batch_size} -> {adjusted_batch_size}")
                batch_size = adjusted_batch_size
        
        # Create optimal dataset
        dataset = self.analyzer.create_optimal_dataset(data_path, transform)
        
        # Create memory-aware data loader
        return MemoryAwareDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            memory_manager=self.memory_manager,
            config=self.config,
            **kwargs
        )
    
    def get_loading_recommendations(self, data_path: str) -> Dict[str, Any]:
        """
        Get recommendations for loading a dataset.
        
        Args:
            data_path: Path to dataset
            
        Returns:
            Dictionary with loading recommendations
        """
        dataset_info = self.analyzer.analyze_dataset(data_path)
        
        # Get memory constraints
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if self.memory_manager:
            memory_stats = self.memory_manager.monitor_memory_usage()
            available_memory_mb = memory_stats.available_mb
        
        # Calculate recommended batch size
        max_batch_size = max(1, int(available_memory_mb / dataset_info.estimated_memory_usage_mb))
        recommended_batch_size = min(32, max_batch_size)
        
        return {
            "dataset_info": {
                "size_mb": dataset_info.size_mb,
                "num_samples": dataset_info.num_samples,
                "loading_strategy": dataset_info.recommended_loading_strategy,
            },
            "memory_info": {
                "available_mb": available_memory_mb,
                "estimated_usage_mb": dataset_info.estimated_memory_usage_mb,
                "memory_utilization_percent": (dataset_info.estimated_memory_usage_mb / available_memory_mb) * 100,
            },
            "recommendations": {
                "batch_size": recommended_batch_size,
                "max_batch_size": max_batch_size,
                "num_workers": 0,  # Conservative for MacBook
                "pin_memory": False,
                "use_streaming": dataset_info.recommended_loading_strategy == "streaming",
            }
        }