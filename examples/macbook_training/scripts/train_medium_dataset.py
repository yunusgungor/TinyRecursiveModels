#!/usr/bin/env python3
"""
MacBook TRM Training Script - Medium Dataset
Optimized training script for medium datasets (100MB - 1GB) on MacBook hardware.
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pretrain_macbook import MacBookTRMTrainer
from pretrain import PretrainConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def detect_macbook_model() -> str:
    """Detect MacBook model based on available RAM."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if total_ram_gb <= 10:
            return "macbook_8gb"
        elif total_ram_gb <= 20:
            return "macbook_16gb"
        else:
            return "macbook_32gb"
    except ImportError:
        print("Warning: psutil not available, defaulting to macbook_8gb")
        return "macbook_8gb"


def estimate_dataset_size(data_paths: list) -> tuple:
    """Estimate dataset size and training time."""
    total_size_mb = 0
    total_files = 0
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            for dirpath, dirnames, filenames in os.walk(data_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size_mb += os.path.getsize(filepath) / (1024 * 1024)
                        total_files += 1
                    except:
                        pass
    
    # Rough estimates for medium datasets
    estimated_samples = int(total_size_mb * 1000)  # 1MB ≈ 1000 samples
    estimated_training_time_hours = total_size_mb / 100  # Rough estimate
    
    return total_size_mb, total_files, estimated_samples, estimated_training_time_hours


def check_system_resources():
    """Check system resources and provide recommendations."""
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Disk space check
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        print("\\n" + "="*50)
        print("System Resource Check")
        print("="*50)
        print(f"Total RAM: {memory_gb:.1f}GB")
        print(f"Available RAM: {memory_available_gb:.1f}GB")
        print(f"CPU cores: {cpu_count}")
        if cpu_freq:
            print(f"CPU frequency: {cpu_freq.current:.0f}MHz")
        print(f"Free disk space: {disk_free_gb:.1f}GB")
        
        # Recommendations
        print("\\nRecommendations:")
        if memory_available_gb < 4:
            print("⚠️  Low available memory - consider closing other applications")
        if disk_free_gb < 5:
            print("⚠️  Low disk space - ensure sufficient space for checkpoints")
        if memory_gb <= 8:
            print("💡 Consider using streaming mode for large datasets")
        
        return {
            'memory_gb': memory_gb,
            'memory_available_gb': memory_available_gb,
            'cpu_count': cpu_count,
            'disk_free_gb': disk_free_gb
        }
        
    except ImportError:
        print("Warning: psutil not available, skipping resource check")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train TRM model on MacBook with medium dataset")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--auto-detect", action="store_true", 
                       help="Auto-detect MacBook model and use appropriate config")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print configuration and exit without training")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--force-streaming", action="store_true",
                       help="Force streaming mode for dataset loading")
    parser.add_argument("--memory-limit", type=int, help="Memory limit in MB")
    
    args = parser.parse_args()
    
    # Check system resources
    system_resources = check_system_resources()
    
    # Auto-detect MacBook model if requested
    if args.auto_detect:
        macbook_model = detect_macbook_model()
        config_path = f"examples/macbook_training/configs/{macbook_model}/medium_dataset.yaml"
        print(f"\\nAuto-detected MacBook model: {macbook_model}")
        print(f"Using configuration: {config_path}")
    elif args.config:
        config_path = args.config
    else:
        # Default based on available memory
        if system_resources.get('memory_gb', 8) > 12:
            config_path = "examples/macbook_training/configs/macbook_16gb/medium_dataset.yaml"
        else:
            config_path = "examples/macbook_training/configs/macbook_8gb/medium_dataset.yaml"
        print(f"\\nUsing default configuration: {config_path}")
    
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config_dict = load_config(config_path)
    
    # Override with command line arguments
    if args.data_path:
        config_dict['data_paths'] = [args.data_path]
    
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    
    if args.resume:
        config_dict['load_checkpoint'] = True
        config_dict['checkpoint_path'] = args.resume
    
    if args.force_streaming:
        if 'macbook_optimizations' not in config_dict:
            config_dict['macbook_optimizations'] = {}
        config_dict['macbook_optimizations']['force_streaming_mode'] = True
    
    if args.memory_limit:
        if 'training' not in config_dict:
            config_dict['training'] = {}
        config_dict['training']['memory_limit_mb'] = args.memory_limit
    
    # Estimate dataset size
    data_paths = config_dict.get('data_paths', [])
    size_mb, num_files, estimated_samples, estimated_hours = estimate_dataset_size(data_paths)
    
    # Print configuration summary
    print("\\n" + "="*60)
    print("MacBook TRM Training - Medium Dataset")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Data paths: {data_paths}")
    print(f"Dataset size: {size_mb:.1f}MB ({num_files:,} files)")
    print(f"Estimated samples: {estimated_samples:,}")
    print(f"Estimated training time: {estimated_hours:.1f} hours")
    print(f"Output directory: {config_dict.get('output_dir', 'outputs/')}")
    print(f"Global batch size: {config_dict.get('global_batch_size', 32)}")
    print(f"Memory limit: {config_dict.get('training', {}).get('memory_limit_mb', 4000)}MB")
    
    # Check if streaming mode is recommended
    memory_gb = system_resources.get('memory_gb', 8)
    if size_mb > (memory_gb * 200):  # If dataset > 25% of RAM
        print("\\n💡 Recommendation: Dataset is large relative to available RAM")
        print("   Streaming mode will be automatically enabled")
    
    if args.dry_run:
        print("\\nDry run mode - configuration loaded successfully")
        return
    
    # Validate data paths
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Error: Data path does not exist: {data_path}")
            sys.exit(1)
    
    # Create PretrainConfig
    try:
        config = PretrainConfig(**config_dict)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Create trainer
    trainer = MacBookTRMTrainer(config)
    
    print("\\nStarting training...")
    print("Press Ctrl+C to stop training gracefully")
    print(f"Training will take approximately {estimated_hours:.1f} hours")
    
    start_time = time.time()
    
    try:
        # Run training
        final_state = trainer.train(estimated_samples)
        
        training_time = time.time() - start_time
        print("\\nTraining completed successfully!")
        print(f"Actual training time: {training_time/3600:.1f} hours")
        
        # Print final summary
        if final_state.training_metrics['loss']:
            final_loss = final_state.training_metrics['loss'][-1]
            print(f"Final loss: {final_loss:.4f}")
        
        print(f"Total samples processed: {final_state.samples_processed:,}")
        print(f"Average speed: {final_state.average_samples_per_second:.1f} samples/second")
        
        # Performance analysis
        if estimated_hours > 0:
            speedup = estimated_hours / (training_time / 3600)
            print(f"Performance: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than estimated")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\nTraining interrupted by user after {training_time/3600:.1f} hours")
    except Exception as e:
        print(f"\\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()