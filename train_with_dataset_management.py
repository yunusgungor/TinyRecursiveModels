#!/usr/bin/env python3
"""
Training script demonstrating dataset management integration.

This script shows how to integrate the dataset management module with
existing TRM training pipelines for memory-efficient training on MacBook.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

# Import TRM training components
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from pretrain import PretrainConfig, create_dataloader

# Import MacBook optimization modules
from macbook_optimization.memory_management import MemoryManager
from macbook_optimization.dataset_management import (
    DatasetManager, DatasetManagementConfig, create_memory_efficient_dataloader,
    estimate_dataset_memory_usage
)


def analyze_datasets(dataset_paths: List[str], splits: List[str] = ["train", "test"]) -> Dict[str, Any]:
    """
    Analyze datasets for memory requirements and loading strategies.
    
    Args:
        dataset_paths: List of dataset paths to analyze
        splits: Dataset splits to analyze
        
    Returns:
        Analysis results
    """
    print("Analyzing datasets for memory requirements...")
    
    analysis_results = {}
    
    for split in splits:
        print(f"\nAnalyzing {split} split:")
        
        # Estimate memory usage
        memory_estimate = estimate_dataset_memory_usage(dataset_paths, split)
        
        print(f"  Total size: {memory_estimate['total_size_mb']:.1f}MB")
        print(f"  Estimated memory usage: {memory_estimate['estimated_memory_usage_mb']:.1f}MB")
        print(f"  Requires streaming: {memory_estimate['requires_streaming']}")
        
        # Detailed file breakdown
        if memory_estimate['file_breakdown']:
            print("  File breakdown:")
            for file_path, sizes in memory_estimate['file_breakdown'].items():
                print(f"    {file_path}: {sizes['total_mb']:.1f}MB")
        
        analysis_results[split] = memory_estimate
    
    return analysis_results


def create_optimized_dataloaders(dataset_paths: List[str], 
                               batch_size: int,
                               memory_manager: MemoryManager) -> Dict[str, Any]:
    """
    Create memory-optimized dataloaders for training and evaluation.
    
    Args:
        dataset_paths: List of dataset paths
        batch_size: Batch size for training
        memory_manager: Memory manager instance
        
    Returns:
        Dictionary containing dataloaders and metadata
    """
    print("Creating memory-optimized dataloaders...")
    
    results = {}
    
    # Create dataset manager
    dataset_config = DatasetManagementConfig(
        max_dataset_memory_mb=800.0,  # Conservative for MacBook
        streaming_threshold_mb=400.0,
        cache_threshold_mb=200.0,
        chunk_size_mb=50.0,
        enable_caching=True,
        auto_fallback_streaming=True
    )
    dataset_manager = DatasetManager(dataset_config, memory_manager)
    
    # Create dataloaders for each split
    for split in ["train", "test"]:
        try:
            print(f"\nCreating {split} dataloader...")
            
            # Get loading recommendations
            recommendations = dataset_manager.get_loading_recommendations(dataset_paths[0])
            print(f"  Recommended batch size: {recommendations['recommendations']['batch_size']}")
            print(f"  Loading strategy: {recommendations['dataset_info']['loading_strategy']}")
            
            # Optimize batch size for this dataset
            batch_optimization = dataset_manager.optimize_batch_size_for_dataset(
                dataset_paths, batch_size, split
            )
            
            optimized_batch_size = batch_optimization['recommended_batch_size']
            if optimized_batch_size != batch_size:
                print(f"  Optimized batch size: {batch_size} -> {optimized_batch_size}")
            
            # Create memory-efficient dataloader
            dataset, creation_info = create_memory_efficient_dataloader(
                dataset_paths=dataset_paths,
                batch_size=optimized_batch_size,
                split=split,
                memory_manager=memory_manager,
                seed=42,
                test_set_mode=(split == "test"),
                epochs_per_iter=1,
                rank=0,
                num_replicas=1
            )
            
            print(f"  Created using {creation_info['final_strategy']} strategy")
            if creation_info['fallback_used']:
                print(f"  Fallback to streaming was used")
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=None,  # Batch size handled by PuzzleDataset
                num_workers=0,    # Conservative for MacBook
                pin_memory=False, # CPU training
                prefetch_factor=2
            )
            
            results[split] = {
                'dataloader': dataloader,
                'dataset': dataset,
                'metadata': dataset.metadata,
                'creation_info': creation_info,
                'batch_optimization': batch_optimization,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"  Failed to create {split} dataloader: {e}")
            results[split] = None
    
    return results


def demonstrate_memory_monitoring(dataloaders: Dict[str, Any], 
                                memory_manager: MemoryManager,
                                num_batches: int = 5):
    """
    Demonstrate memory monitoring during data loading.
    
    Args:
        dataloaders: Dictionary of dataloaders
        memory_manager: Memory manager instance
        num_batches: Number of batches to process for demonstration
    """
    print(f"\nDemonstrating memory monitoring with {num_batches} batches...")
    
    for split_name, split_data in dataloaders.items():
        if split_data is None:
            continue
            
        print(f"\nProcessing {split_name} split:")
        dataloader = split_data['dataloader']
        
        # Monitor memory before processing
        initial_memory = memory_manager.monitor_memory_usage()
        print(f"  Initial memory: {initial_memory.used_mb:.0f}MB ({initial_memory.percent_used:.1f}%)")
        
        batch_count = 0
        for set_name, batch, global_batch_size in dataloader:
            batch_count += 1
            
            # Monitor memory during processing
            current_memory = memory_manager.monitor_memory_usage()
            print(f"  Batch {batch_count}: {current_memory.used_mb:.0f}MB "
                  f"({current_memory.percent_used:.1f}%), "
                  f"batch_size={global_batch_size}")
            
            # Process batch (simulate training step)
            if isinstance(batch, dict):
                for key, tensor in batch.items():
                    if hasattr(tensor, 'shape'):
                        print(f"    {key}: {tensor.shape}")
            
            if batch_count >= num_batches:
                break
        
        # Monitor memory after processing
        final_memory = memory_manager.monitor_memory_usage()
        memory_increase = final_memory.used_mb - initial_memory.used_mb
        print(f"  Final memory: {final_memory.used_mb:.0f}MB "
              f"(+{memory_increase:.0f}MB increase)")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Dataset Management Integration Demo")
    parser.add_argument("--dataset-paths", nargs="+", required=True,
                       help="Paths to datasets")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--demo-batches", type=int, default=3,
                       help="Number of batches to process in demo")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip dataset analysis")
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip memory monitoring demo")
    
    args = parser.parse_args()
    
    print("Dataset Management Integration Demonstration")
    print("=" * 50)
    print(f"Dataset paths: {args.dataset_paths}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Step 1: Analyze datasets
    if not args.skip_analysis:
        analysis_results = analyze_datasets(args.dataset_paths)
        
        # Print summary
        print(f"\nDataset Analysis Summary:")
        total_memory = sum(result['total_size_mb'] for result in analysis_results.values())
        print(f"  Total dataset size: {total_memory:.1f}MB")
        
        requires_streaming = any(result['requires_streaming'] for result in analysis_results.values())
        print(f"  Requires streaming: {requires_streaming}")
    
    # Step 2: Create optimized dataloaders
    print(f"\nCreating optimized dataloaders...")
    dataloaders = create_optimized_dataloaders(
        args.dataset_paths, args.batch_size, memory_manager
    )
    
    # Print dataloader summary
    print(f"\nDataloader Creation Summary:")
    for split_name, split_data in dataloaders.items():
        if split_data:
            strategy = split_data['creation_info']['final_strategy']
            fallback = split_data['creation_info']['fallback_used']
            print(f"  {split_name}: {strategy}" + (" (fallback)" if fallback else ""))
        else:
            print(f"  {split_name}: Failed to create")
    
    # Step 3: Demonstrate memory monitoring
    if not args.skip_demo:
        demonstrate_memory_monitoring(dataloaders, memory_manager, args.demo_batches)
    
    # Step 4: Print final memory summary
    final_summary = memory_manager.get_memory_summary()
    print(f"\nFinal Memory Summary:")
    print(f"  Current usage: {final_summary['current']['used_mb']:.0f}MB "
          f"({final_summary['current']['percent_used']:.1f}%)")
    print(f"  Peak usage: {final_summary['tracking']['peak_mb']:.0f}MB")
    print(f"  Training overhead: {final_summary['tracking']['training_overhead_mb']:.0f}MB")
    
    print(f"\nDemonstration completed successfully!")
    print("The dataset management module is ready for integration with training scripts.")


if __name__ == "__main__":
    main()