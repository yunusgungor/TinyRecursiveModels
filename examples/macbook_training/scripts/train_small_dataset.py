#!/usr/bin/env python3
"""
MacBook TRM Training Script - Small Dataset
Optimized training script for small datasets on MacBook hardware.
"""

import os
import sys
import argparse
import yaml
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


def main():
    parser = argparse.ArgumentParser(description="Train TRM model on MacBook with small dataset")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--auto-detect", action="store_true", 
                       help="Auto-detect MacBook model and use appropriate config")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print configuration and exit without training")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Auto-detect MacBook model if requested
    if args.auto_detect:
        macbook_model = detect_macbook_model()
        config_path = f"examples/macbook_training/configs/{macbook_model}/small_dataset.yaml"
        print(f"Auto-detected MacBook model: {macbook_model}")
        print(f"Using configuration: {config_path}")
    elif args.config:
        config_path = args.config
    else:
        # Default to 8GB configuration
        config_path = "examples/macbook_training/configs/macbook_8gb/small_dataset.yaml"
        print(f"Using default configuration: {config_path}")
    
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
    
    # Print configuration summary
    print("\\n" + "="*60)
    print("MacBook TRM Training - Small Dataset")
    print("="*60)
    print(f"Configuration: {config_path}")
    print(f"Data paths: {config_dict.get('data_paths', [])}")
    print(f"Output directory: {config_dict.get('output_dir', 'outputs/')}")
    print(f"Global batch size: {config_dict.get('global_batch_size', 32)}")
    print(f"Memory limit: {config_dict.get('training', {}).get('memory_limit_mb', 4000)}MB")
    print(f"CPU threads: {config_dict.get('training', {}).get('torch_threads', 4)}")
    
    if args.dry_run:
        print("\\nDry run mode - configuration loaded successfully")
        return
    
    # Validate data paths
    data_paths = config_dict.get('data_paths', [])
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: Data path does not exist: {data_path}")
    
    # Create PretrainConfig
    try:
        config = PretrainConfig(**config_dict)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Estimate dataset size (simple heuristic)
    dataset_size = 1000  # Default for small datasets
    for data_path in data_paths:
        if os.path.exists(data_path):
            # Simple size estimation based on directory size
            try:
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(data_path)
                    for filename in filenames
                )
                # Rough estimate: 1MB ≈ 1000 samples
                dataset_size = max(dataset_size, total_size // (1024 * 1024) * 1000)
            except:
                pass
    
    print(f"Estimated dataset size: {dataset_size:,} samples")
    
    # Create trainer
    trainer = MacBookTRMTrainer(config)
    
    print("\\nStarting training...")
    print("Press Ctrl+C to stop training gracefully")
    
    try:
        # Run training
        final_state = trainer.train(dataset_size)
        print("\\nTraining completed successfully!")
        
        # Print final summary
        if final_state.training_metrics['loss']:
            final_loss = final_state.training_metrics['loss'][-1]
            print(f"Final loss: {final_loss:.4f}")
        
        print(f"Total samples processed: {final_state.samples_processed:,}")
        print(f"Average speed: {final_state.average_samples_per_second:.1f} samples/second")
        
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
    except Exception as e:
        print(f"\\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()