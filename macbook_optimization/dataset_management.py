"""
Dataset management module for MacBook TRM training optimization.

This module provides memory-efficient dataset loading strategies, streaming datasets,
and preprocessing/caching utilities specifically designed for memory-constrained
MacBook hardware training scenarios.
"""

import os
import json
import mmap
import pickle
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Union, Tuple, Callable
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

from .memory_management import MemoryManager, MemoryConfig
from .data_loading import DataLoadingConfig, DatasetAnalyzer
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata


@dataclass
class DatasetManagementConfig:
    """Configuration for dataset management."""
    # Memory constraints
    max_dataset_memory_mb: float = 800.0  # Maximum memory for dataset loading
    streaming_threshold_mb: float = 400.0  # Use streaming above this size
    cache_threshold_mb: float = 200.0  # Cache datasets smaller than this
    
    # Streaming configuration
    chunk_size_mb: float = 50.0  # Size of streaming chunks
    prefetch_chunks: int = 2  # Number of chunks to prefetch
    
    # Caching configuration
    enable_caching: bool = True
    cache_dir: Optional[str] = ".cache/datasets"
    max_cache_size_gb: float = 2.0  # Maximum total cache size
    cache_compression: bool = True  # Compress cached data
    
    # Memory monitoring
    memory_check_interval: int = 100  # Check memory every N batches
    auto_fallback_streaming: bool = True  # Auto-fallback to streaming on OOM


@dataclass
class DatasetMetrics:
    """Metrics for dataset loading performance."""
    total_size_mb: float
    loading_strategy: str
    memory_usage_mb: float
    load_time_seconds: float
    cache_hit: bool
    num_samples: int
    avg_sample_size_bytes: float


class StreamingPuzzleDataset(IterableDataset):
    """Memory-efficient streaming version of PuzzleDataset."""
    
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train",
                 chunk_size_mb: float = 50.0, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize streaming puzzle dataset.
        
        Args:
            config: Puzzle dataset configuration
            split: Dataset split ("train" or "test")
            chunk_size_mb: Size of chunks to stream
            memory_manager: Memory manager for monitoring
        """
        self.config = config
        self.split = split
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.memory_manager = memory_manager
        
        # Load metadata
        self.metadata = self._load_merged_metadata()
        
        # Initialize streaming state
        self.current_chunk = 0
        self.chunks_info = []
        self._analyze_chunks()
        
    def _load_merged_metadata(self) -> PuzzleDatasetMetadata:
        """Load and merge metadata from all dataset paths."""
        # This mirrors the logic from PuzzleDataset
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        
        for dataset_path in self.config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            
            mean_puzzle_examples += current_metadata.mean_puzzle_examples * current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles
        
        return PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )
    
    def _load_metadata(self, dataset_path: str) -> PuzzleDatasetMetadata:
        """Load metadata from a single dataset path."""
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))
    
    def _analyze_chunks(self):
        """Analyze dataset files and determine chunk boundaries."""
        self.chunks_info = []
        
        for dataset_idx, dataset_path in enumerate(self.config.dataset_paths):
            for set_name in self.metadata.sets:
                # Get file paths
                inputs_path = os.path.join(dataset_path, self.split, f"{set_name}__inputs.npy")
                labels_path = os.path.join(dataset_path, self.split, f"{set_name}__labels.npy")
                
                if not os.path.exists(inputs_path):
                    continue
                
                # Get file size
                file_size = os.path.getsize(inputs_path)
                num_chunks = max(1, (file_size + self.chunk_size_bytes - 1) // self.chunk_size_bytes)
                
                for chunk_idx in range(num_chunks):
                    chunk_info = {
                        'dataset_idx': dataset_idx,
                        'dataset_path': dataset_path,
                        'set_name': set_name,
                        'chunk_idx': chunk_idx,
                        'total_chunks': num_chunks,
                        'inputs_path': inputs_path,
                        'labels_path': labels_path,
                        'file_size': file_size
                    }
                    self.chunks_info.append(chunk_info)
    
    def _load_chunk(self, chunk_info: Dict) -> Dict[str, np.ndarray]:
        """Load a specific chunk of data."""
        # Calculate chunk boundaries
        chunk_start = chunk_info['chunk_idx'] * self.chunk_size_bytes
        chunk_end = min(chunk_start + self.chunk_size_bytes, chunk_info['file_size'])
        
        # Load chunk data using memory mapping
        with open(chunk_info['inputs_path'], 'rb') as f:
            # Use memory mapping for efficient access
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Calculate sample boundaries (this is simplified)
                # In practice, you'd need to parse the numpy array structure
                chunk_data = mm[chunk_start:chunk_end]
                
                # For now, load the entire arrays and slice
                # This is a simplification - in practice you'd implement
                # proper chunked loading of numpy arrays
                inputs = np.load(chunk_info['inputs_path'], mmap_mode='r')
                labels = np.load(chunk_info['labels_path'], mmap_mode='r')
                
                # Calculate sample indices for this chunk
                samples_per_chunk = max(1, len(inputs) // chunk_info['total_chunks'])
                start_idx = chunk_info['chunk_idx'] * samples_per_chunk
                end_idx = min(len(inputs), start_idx + samples_per_chunk)
                
                return {
                    'inputs': inputs[start_idx:end_idx],
                    'labels': labels[start_idx:end_idx],
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
    
    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        """Iterate over dataset chunks."""
        for chunk_info in self.chunks_info:
            # Monitor memory usage
            if self.memory_manager:
                memory_stats = self.memory_manager.monitor_memory_usage()
                if memory_stats.percent_used > 85:
                    print(f"High memory usage ({memory_stats.percent_used:.1f}%) - "
                          f"forcing garbage collection")
                    self.memory_manager.force_garbage_collection()
            
            # Load chunk
            chunk_data = self._load_chunk(chunk_info)
            
            # Create batches from chunk
            inputs = chunk_data['inputs']
            labels = chunk_data['labels']
            
            # Simple batching (this would be more sophisticated in practice)
            batch_size = self.config.global_batch_size // self.config.num_replicas
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # Convert to tensors
                if TORCH_AVAILABLE:
                    batch = {
                        'inputs': torch.from_numpy(batch_inputs),
                        'labels': torch.from_numpy(batch_labels),
                        'puzzle_identifiers': torch.zeros(len(batch_inputs), dtype=torch.long)
                    }
                else:
                    batch = {
                        'inputs': batch_inputs,
                        'labels': batch_labels,
                        'puzzle_identifiers': np.zeros(len(batch_inputs), dtype=np.int64)
                    }
                
                yield chunk_info['set_name'], batch, len(batch_inputs)


class CachedDatasetLoader:
    """Dataset loader with preprocessing and caching capabilities."""
    
    def __init__(self, config: DatasetManagementConfig, 
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize cached dataset loader.
        
        Args:
            config: Dataset management configuration
            memory_manager: Memory manager for monitoring
        """
        self.config = config
        self.memory_manager = memory_manager
        
        # Setup cache directory
        if self.config.enable_caching and self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def _get_cache_key(self, dataset_paths: List[str], split: str, 
                      preprocessing_params: Optional[Dict] = None) -> str:
        """Generate cache key for dataset."""
        # Create hash from dataset paths, split, and preprocessing params
        content = {
            'dataset_paths': sorted(dataset_paths),
            'split': split,
            'preprocessing_params': preprocessing_params or {}
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if dataset is cached."""
        if not self.cache_dir:
            return False
        
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                if self.config.cache_compression:
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        pickle.dump(data, gz_f)
                else:
                    pickle.dump(data, f)
            
            print(f"Cached dataset to {cache_path}")
        except Exception as e:
            print(f"Failed to cache dataset: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Any:
        """Load data from cache."""
        if not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'rb') as f:
                if self.config.cache_compression:
                    import gzip
                    with gzip.open(f, 'rb') as gz_f:
                        return pickle.load(gz_f)
                else:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load from cache: {e}")
            return None
    
    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        # Get all cache files with their sizes and modification times
        cache_files = []
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            size = cache_file.stat().st_size
            mtime = cache_file.stat().st_mtime
            cache_files.append((cache_file, size, mtime))
            total_size += size
        
        # Check if cleanup is needed
        max_size_bytes = self.config.max_cache_size_gb * 1024 * 1024 * 1024
        if total_size <= max_size_bytes:
            return
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[2])
        
        # Remove oldest files until under limit
        for cache_file, size, _ in cache_files:
            if total_size <= max_size_bytes:
                break
            
            try:
                cache_file.unlink()
                total_size -= size
                print(f"Removed old cache file: {cache_file}")
            except Exception as e:
                print(f"Failed to remove cache file {cache_file}: {e}")
    
    def preprocess_and_cache(self, dataset_paths: List[str], split: str,
                           preprocessing_fn: Optional[Callable] = None,
                           preprocessing_params: Optional[Dict] = None) -> str:
        """
        Preprocess dataset and cache the result.
        
        Args:
            dataset_paths: List of dataset paths
            split: Dataset split
            preprocessing_fn: Preprocessing function
            preprocessing_params: Parameters for preprocessing
            
        Returns:
            Cache key for the processed dataset
        """
        cache_key = self._get_cache_key(dataset_paths, split, preprocessing_params)
        
        # Check if already cached
        if self._is_cached(cache_key):
            print(f"Dataset already cached with key: {cache_key}")
            return cache_key
        
        # Load and preprocess dataset
        print(f"Preprocessing and caching dataset...")
        
        # Create standard PuzzleDataset for preprocessing
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_paths=dataset_paths,
            global_batch_size=32,  # Temporary batch size for preprocessing
            test_set_mode=(split == "test"),
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        dataset = PuzzleDataset(config, split)
        
        # Apply preprocessing if provided
        if preprocessing_fn:
            processed_data = preprocessing_fn(dataset, preprocessing_params or {})
        else:
            # Default: just load the metadata and basic info
            processed_data = {
                'metadata': dataset.metadata,
                'config': config,
                'preprocessing_params': preprocessing_params
            }
        
        # Cache the processed data
        self._save_to_cache(cache_key, processed_data)
        
        # Cleanup old cache files if needed
        self._cleanup_cache()
        
        return cache_key
    
    def load_cached_dataset(self, cache_key: str) -> Any:
        """Load cached dataset."""
        return self._load_from_cache(cache_key)


class DatasetManager:
    """High-level dataset management for memory-constrained training."""
    
    def __init__(self, config: Optional[DatasetManagementConfig] = None,
                 memory_manager: Optional[MemoryManager] = None):
        """
        Initialize dataset manager.
        
        Args:
            config: Dataset management configuration
            memory_manager: Memory manager for monitoring
        """
        self.config = config or DatasetManagementConfig()
        self.memory_manager = memory_manager or MemoryManager()
        
        # Initialize components
        self.dataset_analyzer = DatasetAnalyzer(self.memory_manager)
        self.cached_loader = CachedDatasetLoader(self.config, self.memory_manager)
        
        # Metrics tracking
        self.loading_metrics: List[DatasetMetrics] = []
    
    def analyze_dataset_requirements(self, dataset_paths: List[str], 
                                   split: str = "train") -> Dict[str, Any]:
        """
        Analyze dataset memory requirements and recommend loading strategy.
        
        Args:
            dataset_paths: List of dataset paths
            split: Dataset split
            
        Returns:
            Analysis results and recommendations
        """
        total_size_mb = 0.0
        dataset_info = []
        
        # Analyze each dataset path
        for dataset_path in dataset_paths:
            for set_name in ["all"]:  # Default set name
                inputs_path = os.path.join(dataset_path, split, f"{set_name}__inputs.npy")
                labels_path = os.path.join(dataset_path, split, f"{set_name}__labels.npy")
                
                if os.path.exists(inputs_path):
                    inputs_size = os.path.getsize(inputs_path) / (1024 * 1024)
                    labels_size = os.path.getsize(labels_path) / (1024 * 1024) if os.path.exists(labels_path) else 0
                    set_size = inputs_size + labels_size
                    total_size_mb += set_size
                    
                    dataset_info.append({
                        'dataset_path': dataset_path,
                        'set_name': set_name,
                        'inputs_size_mb': inputs_size,
                        'labels_size_mb': labels_size,
                        'total_size_mb': set_size
                    })
        
        # Get available memory
        memory_stats = self.memory_manager.monitor_memory_usage()
        available_memory_mb = memory_stats.available_mb
        
        # Determine loading strategy
        if total_size_mb <= self.config.cache_threshold_mb:
            strategy = "cached"
            estimated_memory_usage = total_size_mb * 1.2  # 20% overhead
        elif total_size_mb <= available_memory_mb * 0.4:
            strategy = "memory_mapped"
            estimated_memory_usage = total_size_mb * 0.1  # Memory mapping overhead
        else:
            strategy = "streaming"
            estimated_memory_usage = self.config.chunk_size_mb * self.config.prefetch_chunks
        
        return {
            'total_size_mb': total_size_mb,
            'available_memory_mb': available_memory_mb,
            'memory_utilization_percent': (estimated_memory_usage / available_memory_mb) * 100,
            'recommended_strategy': strategy,
            'estimated_memory_usage_mb': estimated_memory_usage,
            'dataset_info': dataset_info,
            'can_fit_in_memory': total_size_mb <= available_memory_mb * 0.6,
            'requires_streaming': total_size_mb > self.config.streaming_threshold_mb
        }
    
    def create_memory_efficient_dataset(self, dataset_paths: List[str], 
                                      config: PuzzleDatasetConfig,
                                      split: str = "train",
                                      force_strategy: Optional[str] = None) -> Union[PuzzleDataset, StreamingPuzzleDataset]:
        """
        Create memory-efficient dataset based on analysis.
        
        Args:
            dataset_paths: List of dataset paths
            config: Puzzle dataset configuration
            split: Dataset split
            force_strategy: Force specific loading strategy
            
        Returns:
            Optimized dataset instance
        """
        import time
        start_time = time.time()
        
        # Analyze dataset requirements
        analysis = self.analyze_dataset_requirements(dataset_paths, split)
        strategy = force_strategy or analysis['recommended_strategy']
        
        print(f"Dataset analysis: {analysis['total_size_mb']:.1f}MB total, "
              f"strategy: {strategy}")
        
        # Create dataset based on strategy
        if strategy == "streaming" or analysis['requires_streaming']:
            print("Creating streaming dataset for memory efficiency")
            dataset = StreamingPuzzleDataset(
                config=config,
                split=split,
                chunk_size_mb=self.config.chunk_size_mb,
                memory_manager=self.memory_manager
            )
            memory_usage_mb = self.config.chunk_size_mb * self.config.prefetch_chunks
            
        else:
            print("Creating standard dataset with memory monitoring")
            dataset = PuzzleDataset(config, split)
            memory_usage_mb = analysis['estimated_memory_usage_mb']
        
        # Record metrics
        load_time = time.time() - start_time
        metrics = DatasetMetrics(
            total_size_mb=analysis['total_size_mb'],
            loading_strategy=strategy,
            memory_usage_mb=memory_usage_mb,
            load_time_seconds=load_time,
            cache_hit=False,  # Would be determined by caching logic
            num_samples=0,  # Would be determined from metadata
            avg_sample_size_bytes=0  # Would be calculated
        )
        self.loading_metrics.append(metrics)
        
        return dataset
    
    def get_loading_recommendations(self, dataset_path: str) -> Dict[str, Any]:
        """
        Get loading recommendations for a dataset.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Loading recommendations and dataset info
        """
        # Analyze dataset requirements
        analysis = self.analyze_dataset_requirements([dataset_path], "train")
        
        # Calculate recommended batch size based on memory
        memory_stats = self.memory_manager.monitor_memory_usage()
        available_memory_mb = memory_stats.available_mb
        
        # Conservative batch size calculation
        if analysis['total_size_mb'] < 100:
            recommended_batch_size = 8
        elif analysis['total_size_mb'] < 500:
            recommended_batch_size = 4
        else:
            recommended_batch_size = 2
        
        return {
            'dataset_info': {
                'total_size_mb': analysis['total_size_mb'],
                'loading_strategy': analysis['recommended_strategy'],
                'memory_utilization_percent': analysis['memory_utilization_percent']
            },
            'recommendations': {
                'batch_size': recommended_batch_size,
                'use_streaming': analysis['requires_streaming'],
                'enable_caching': analysis['total_size_mb'] < self.config.cache_threshold_mb
            }
        }
    
    def validate_dataset_memory_constraints(self, dataset_paths: List[str],
                                          batch_size: int, split: str = "train") -> Dict[str, Any]:
        """
        Validate that dataset can be loaded with given memory constraints.
        
        Args:
            dataset_paths: List of dataset paths
            batch_size: Intended batch size
            split: Dataset split
            
        Returns:
            Validation results and recommendations
        """
        analysis = self.analyze_dataset_requirements(dataset_paths, split)
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        # Estimate memory usage with batch size
        estimated_batch_memory = batch_size * 1024 * 4  # Rough estimate: 4KB per sample
        total_estimated_memory = analysis['estimated_memory_usage_mb'] + (estimated_batch_memory / (1024 * 1024))
        
        # Check constraints
        memory_ok = total_estimated_memory <= memory_stats.available_mb * 0.8
        batch_size_ok = batch_size <= 64  # Conservative limit for MacBook
        
        recommendations = []
        if not memory_ok:
            recommendations.append(f"Reduce dataset size or use streaming (current: {total_estimated_memory:.1f}MB)")
        if not batch_size_ok:
            recommendations.append(f"Reduce batch size (current: {batch_size}, recommended: <= 32)")
        
        return {
            'memory_constraints_met': memory_ok and batch_size_ok,
            'total_estimated_memory_mb': total_estimated_memory,
            'available_memory_mb': memory_stats.available_mb,
            'memory_utilization_percent': (total_estimated_memory / memory_stats.available_mb) * 100,
            'batch_size_ok': batch_size_ok,
            'recommendations': recommendations,
            'fallback_to_streaming': not memory_ok
        }
    
    def get_loading_metrics(self) -> List[DatasetMetrics]:
        """Get dataset loading performance metrics."""
        return self.loading_metrics.copy()
    
    def clear_metrics(self):
        """Clear loading metrics history."""
        self.loading_metrics.clear()
    
    def create_dataloader_with_fallback(self, dataset_paths: List[str],
                                      config: PuzzleDatasetConfig,
                                      split: str = "train") -> Tuple[Union[PuzzleDataset, StreamingPuzzleDataset], Dict[str, Any]]:
        """
        Create dataloader with automatic fallback to streaming on memory pressure.
        
        Args:
            dataset_paths: List of dataset paths
            config: Puzzle dataset configuration
            split: Dataset split
            
        Returns:
            Tuple of (dataset, creation_info)
        """
        validation = self.validate_dataset_memory_constraints(
            dataset_paths, config.global_batch_size, split
        )
        
        creation_info = {
            'validation': validation,
            'fallback_used': False,
            'final_strategy': 'standard'
        }
        
        try:
            # Try standard dataset first
            if validation['memory_constraints_met'] and not validation['fallback_to_streaming']:
                dataset = self.create_memory_efficient_dataset(
                    dataset_paths, config, split, force_strategy='memory_mapped'
                )
                creation_info['final_strategy'] = 'memory_mapped'
            else:
                raise MemoryError("Preemptive fallback to streaming")
                
        except (MemoryError, RuntimeError) as e:
            print(f"Falling back to streaming dataset due to: {e}")
            dataset = self.create_memory_efficient_dataset(
                dataset_paths, config, split, force_strategy='streaming'
            )
            creation_info['fallback_used'] = True
            creation_info['final_strategy'] = 'streaming'
            
            # Force garbage collection after fallback
            self.memory_manager.force_garbage_collection()
        
        return dataset, creation_info
    
    def monitor_dataset_memory_usage(self, dataset_name: str = "current") -> Dict[str, Any]:
        """
        Monitor memory usage during dataset operations.
        
        Args:
            dataset_name: Name identifier for the dataset
            
        Returns:
            Memory usage statistics
        """
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        return {
            'dataset_name': dataset_name,
            'timestamp': time.time(),
            'memory_usage': {
                'used_mb': memory_stats.used_mb,
                'available_mb': memory_stats.available_mb,
                'percent_used': memory_stats.percent_used
            },
            'recommendations': self.memory_manager.get_memory_recommendations(7000000)  # 7M params
        }
    
    def optimize_batch_size_for_dataset(self, dataset_paths: List[str],
                                      initial_batch_size: int,
                                      split: str = "train") -> Dict[str, Any]:
        """
        Optimize batch size based on dataset characteristics and memory constraints.
        
        Args:
            dataset_paths: List of dataset paths
            initial_batch_size: Initial batch size to optimize
            split: Dataset split
            
        Returns:
            Optimization results with recommended batch size
        """
        analysis = self.analyze_dataset_requirements(dataset_paths, split)
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        # Calculate optimal batch size
        available_memory_mb = memory_stats.available_mb * 0.7  # Use 70% of available memory
        
        # Estimate memory per sample (rough heuristic)
        if analysis['dataset_info']:
            avg_sample_size_mb = analysis['total_size_mb'] / max(1, len(analysis['dataset_info']) * 1000)
        else:
            avg_sample_size_mb = 0.001  # 1KB default
        
        # Account for model overhead (gradients, optimizer states)
        memory_per_sample_with_overhead = avg_sample_size_mb * 3.0  # 3x overhead
        
        max_batch_size = max(1, int(available_memory_mb / memory_per_sample_with_overhead))
        recommended_batch_size = min(initial_batch_size, max_batch_size, 32)  # Cap at 32 for MacBook
        
        return {
            'initial_batch_size': initial_batch_size,
            'recommended_batch_size': recommended_batch_size,
            'max_safe_batch_size': max_batch_size,
            'memory_per_sample_mb': memory_per_sample_with_overhead,
            'available_memory_mb': available_memory_mb,
            'optimization_ratio': recommended_batch_size / initial_batch_size,
            'memory_utilization_percent': (recommended_batch_size * memory_per_sample_with_overhead / available_memory_mb) * 100
        }


def create_memory_efficient_dataloader(dataset_paths: List[str],
                                     batch_size: int,
                                     split: str = "train",
                                     memory_manager: Optional[MemoryManager] = None,
                                     **kwargs) -> Tuple[Union[PuzzleDataset, StreamingPuzzleDataset], Dict[str, Any]]:
    """
    Convenience function to create memory-efficient dataloader.
    
    Args:
        dataset_paths: List of dataset paths
        batch_size: Batch size
        split: Dataset split
        memory_manager: Memory manager instance
        **kwargs: Additional arguments for PuzzleDatasetConfig
        
    Returns:
        Tuple of (dataset, creation_info)
    """
    # Setup default configuration
    config_defaults = {
        'seed': 42,
        'global_batch_size': batch_size,
        'test_set_mode': (split == "test"),
        'epochs_per_iter': 1,
        'rank': 0,
        'num_replicas': 1
    }
    config_defaults.update(kwargs)
    
    config = PuzzleDatasetConfig(
        dataset_paths=dataset_paths,
        **config_defaults
    )
    
    # Create dataset manager
    dataset_manager = DatasetManager(memory_manager=memory_manager)
    
    # Create dataset with fallback
    return dataset_manager.create_dataloader_with_fallback(dataset_paths, config, split)


# Utility functions for dataset preprocessing
def preprocess_for_memory_efficiency(dataset: PuzzleDataset, 
                                    params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess dataset for memory efficiency.
    
    Args:
        dataset: Input dataset
        params: Preprocessing parameters
        
    Returns:
        Preprocessed dataset information
    """
    # Extract useful information without loading full data
    metadata = dataset.metadata
    
    # Calculate memory-efficient parameters
    efficient_params = {
        'seq_len': min(metadata.seq_len, params.get('max_seq_len', 512)),
        'vocab_size': metadata.vocab_size,
        'recommended_batch_size': params.get('batch_size', 8),
        'memory_mapping_recommended': True
    }
    
    return {
        'original_metadata': metadata,
        'efficient_params': efficient_params,
        'preprocessing_applied': True,
        'memory_optimizations': [
            'sequence_length_capping',
            'memory_mapping_enabled',
            'batch_size_optimization'
        ]
    }


def estimate_dataset_memory_usage(dataset_paths: List[str], split: str = "train") -> Dict[str, float]:
    """
    Estimate memory usage for dataset loading.
    
    Args:
        dataset_paths: List of dataset paths
        split: Dataset split
        
    Returns:
        Memory usage estimates in MB
    """
    total_size_mb = 0.0
    file_sizes = {}
    
    for dataset_path in dataset_paths:
        dataset_size = 0.0
        
        # Check common file patterns
        for set_name in ["all", "train", "test"]:
            inputs_path = os.path.join(dataset_path, split, f"{set_name}__inputs.npy")
            labels_path = os.path.join(dataset_path, split, f"{set_name}__labels.npy")
            
            if os.path.exists(inputs_path):
                inputs_size = os.path.getsize(inputs_path) / (1024 * 1024)
                labels_size = os.path.getsize(labels_path) / (1024 * 1024) if os.path.exists(labels_path) else 0
                set_size = inputs_size + labels_size
                dataset_size += set_size
                
                file_sizes[f"{dataset_path}/{set_name}"] = {
                    'inputs_mb': inputs_size,
                    'labels_mb': labels_size,
                    'total_mb': set_size
                }
        
        total_size_mb += dataset_size
    
    return {
        'total_size_mb': total_size_mb,
        'estimated_memory_usage_mb': total_size_mb * 1.2,  # 20% overhead
        'streaming_threshold_mb': 400.0,
        'requires_streaming': total_size_mb > 400.0,
        'file_breakdown': file_sizes
    }