#!/usr/bin/env python3
"""
MacBook TRM Email Classification Training Script
Specialized script for training email classification models on MacBook hardware.
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


def create_email_config(base_config_path: str, email_data_path: str, macbook_model: str) -> dict:
    """Create email classification configuration based on MacBook model."""
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Email-specific modifications
    config['data_paths'] = [email_data_path]
    config['data_paths_test'] = []  # Will be split from training data
    
    # Use email-specific architecture
    config['defaults'] = ['arch: trm_email']
    
    # Email classification specific settings
    email_config = {
        'num_email_categories': 10,  # Adjust based on your dataset
        'use_category_embedding': True,
        'classification_dropout': 0.1,
        'use_email_structure': True,
        'email_structure_dim': 64,
        'subject_attention_weight': 2.0,
        'sender_attention_weight': 1.5,
        'pooling_strategy': 'weighted'
    }
    
    # Update architecture with email-specific settings
    if 'arch' not in config:
        config['arch'] = {}
    config['arch'].update(email_config)
    
    # Adjust training parameters for email classification
    if macbook_model == "macbook_8gb":
        config['global_batch_size'] = 16
        config['training']['memory_limit_mb'] = 3000
        config['arch']['seq_len'] = 256
        config['arch']['hidden_size'] = 256
    elif macbook_model == "macbook_16gb":
        config['global_batch_size'] = 32
        config['training']['memory_limit_mb'] = 6000
        config['arch']['seq_len'] = 512
        config['arch']['hidden_size'] = 384
    
    # Email-specific training settings
    config['epochs'] = 1000  # Fewer epochs for classification
    config['eval_interval'] = 100
    config['lr'] = 2e-4  # Higher learning rate for classification
    config['lr_warmup_steps'] = 200
    
    # Enable wandb for email classification tracking
    config['use_wandb'] = True
    config['wandb_project'] = 'macbook-email-classification'
    config['experiment_name'] = f'email_trm_{macbook_model}'
    
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


def validate_email_dataset(data_path: str) -> dict:
    """Validate email dataset structure and provide statistics."""
    if not os.path.exists(data_path):
        raise ValueError(f"Email dataset path does not exist: {data_path}")
    
    stats = {
        'total_files': 0,
        'total_size_mb': 0,
        'file_types': {},
        'estimated_emails': 0
    }
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                file_size = os.path.getsize(filepath)
                stats['total_files'] += 1
                stats['total_size_mb'] += file_size / (1024 * 1024)
                
                # Track file types
                ext = os.path.splitext(file)[1].lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                
                # Estimate number of emails (rough heuristic)
                if ext in ['.txt', '.eml', '.msg', '.json']:
                    # Assume average email is 2KB
                    stats['estimated_emails'] += max(1, file_size // 2048)
                    
            except:
                pass
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train TRM email classifier on MacBook")
    parser.add_argument("--email-data", type=str, required=True,
                       help="Path to email classification dataset")
    parser.add_argument("--config", type=str, 
                       help="Base configuration file (will be adapted for email classification)")
    parser.add_argument("--output-dir", type=str, default="outputs/email_classification",
                       help="Output directory for checkpoints")
    parser.add_argument("--auto-detect", action="store_true", default=True,
                       help="Auto-detect MacBook model and use appropriate config")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configuration and exit without training")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--num-categories", type=int, default=10,
                       help="Number of email categories to classify")
    parser.add_argument("--max-sequence-length", type=int,
                       help="Maximum sequence length for emails")
    
    args = parser.parse_args()
    
    # Validate email dataset
    print("Validating email dataset...")
    try:
        dataset_stats = validate_email_dataset(args.email_data)
        print(f"Dataset statistics:")
        print(f"  Total files: {dataset_stats['total_files']:,}")
        print(f"  Total size: {dataset_stats['total_size_mb']:.1f}MB")
        print(f"  Estimated emails: {dataset_stats['estimated_emails']:,}")
        print(f"  File types: {dataset_stats['file_types']}")
    except Exception as e:
        print(f"Error validating dataset: {e}")
        sys.exit(1)
    
    # Auto-detect MacBook model
    if args.auto_detect:
        macbook_model = detect_macbook_model()
        print(f"\\nAuto-detected MacBook model: {macbook_model}")
    else:
        macbook_model = "macbook_8gb"  # Default
    
    # Determine base configuration
    if args.config:
        base_config_path = args.config
    else:
        base_config_path = f"examples/macbook_training/configs/{macbook_model}/small_dataset.yaml"
    
    print(f"Using base configuration: {base_config_path}")
    
    # Create email-specific configuration
    try:
        config_dict = create_email_config(base_config_path, args.email_data, macbook_model)
    except Exception as e:
        print(f"Error creating email configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    
    if args.resume:
        config_dict['load_checkpoint'] = True
        config_dict['checkpoint_path'] = args.resume
    
    if args.num_categories:
        config_dict['arch']['num_email_categories'] = args.num_categories
    
    if args.max_sequence_length:
        config_dict['arch']['seq_len'] = args.max_sequence_length
        config_dict['training']['max_sequence_length'] = args.max_sequence_length
    
    # Print configuration summary
    print("\\n" + "="*60)
    print("MacBook TRM Email Classification Training")
    print("="*60)
    print(f"Email dataset: {args.email_data}")
    print(f"Estimated emails: {dataset_stats['estimated_emails']:,}")
    print(f"MacBook model: {macbook_model}")
    print(f"Output directory: {config_dict.get('output_dir')}")
    print(f"Number of categories: {config_dict['arch']['num_email_categories']}")
    print(f"Sequence length: {config_dict['arch']['seq_len']}")
    print(f"Batch size: {config_dict.get('global_batch_size')}")
    print(f"Memory limit: {config_dict['training']['memory_limit_mb']}MB")
    
    if args.dry_run:
        print("\\nDry run mode - configuration created successfully")
        print("\\nEmail-specific architecture settings:")
        for key, value in config_dict['arch'].items():
            if 'email' in key.lower() or key in ['num_email_categories', 'pooling_strategy']:
                print(f"  {key}: {value}")
        return
    
    # Create PretrainConfig
    try:
        config = PretrainConfig(**config_dict)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Create trainer
    trainer = MacBookTRMTrainer(config)
    
    print("\\nStarting email classification training...")
    print("Press Ctrl+C to stop training gracefully")
    
    # Estimate training time based on dataset size
    estimated_time_hours = dataset_stats['total_size_mb'] / 50  # Rough estimate
    print(f"Estimated training time: {estimated_time_hours:.1f} hours")
    
    start_time = time.time()
    
    try:
        # Run training
        final_state = trainer.train(dataset_stats['estimated_emails'])
        
        training_time = time.time() - start_time
        print("\\nEmail classification training completed successfully!")
        print(f"Training time: {training_time/3600:.1f} hours")
        
        # Print final summary
        if final_state.training_metrics['loss']:
            final_loss = final_state.training_metrics['loss'][-1]
            print(f"Final classification loss: {final_loss:.4f}")
        
        print(f"Total emails processed: {final_state.samples_processed:,}")
        print(f"Average speed: {final_state.average_samples_per_second:.1f} emails/second")
        
        # Save final model info
        model_info = {
            'model_type': 'email_classification',
            'num_categories': config_dict['arch']['num_email_categories'],
            'sequence_length': config_dict['arch']['seq_len'],
            'training_emails': final_state.samples_processed,
            'final_loss': final_loss if final_state.training_metrics['loss'] else None,
            'training_time_hours': training_time / 3600,
            'macbook_model': macbook_model
        }
        
        model_info_path = os.path.join(config_dict.get('output_dir', 'outputs'), 'model_info.yaml')
        os.makedirs(os.path.dirname(model_info_path), exist_ok=True)
        with open(model_info_path, 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        print(f"\\nModel information saved to: {model_info_path}")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\\nTraining interrupted by user after {training_time/3600:.1f} hours")
    except Exception as e:
        print(f"\\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()