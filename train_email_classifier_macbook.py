#!/usr/bin/env python3
"""
MacBook Email Classification Training Script

This script provides a complete email classification training pipeline optimized for MacBook hardware.
It integrates all email training components into a single executable script with command-line interface.

Features:
- Automatic MacBook hardware detection and optimization
- Email dataset management with memory constraints
- Multi-phase training strategies
- Hyperparameter optimization
- Progress monitoring and checkpointing
- Production-ready model export

Requirements: 1.5, 2.1
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from macbook_optimization.email_training_orchestrator import EmailTrainingOrchestrator, HyperparameterOptimizer
    from macbook_optimization.email_training_config import EmailTrainingConfig
    from macbook_optimization.hardware_detection import HardwareDetector
    from macbook_optimization.config_validation import validate_email_training_config
    from models.email_tokenizer import EmailTokenizer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_dataset_path(dataset_path: str) -> bool:
    """Validate email dataset path and structure."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    # Check for required files
    required_files = ["train/dataset.json", "vocab.json", "categories.json"]
    missing_files = []
    
    for file_path in required_files:
        if not (dataset_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Error: Missing required dataset files: {missing_files}")
        print("Please ensure your dataset has the correct structure:")
        print("  dataset/")
        print("    train/dataset.json")
        print("    test/dataset.json (optional)")
        print("    vocab.json")
        print("    categories.json")
        return False
    
    return True


def create_sample_dataset(output_path: str) -> bool:
    """Create a sample email dataset for testing."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample email categories
    categories = {
        "Newsletter": 0,
        "Work": 1,
        "Personal": 2,
        "Spam": 3,
        "Promotional": 4,
        "Social": 5,
        "Finance": 6,
        "Travel": 7,
        "Shopping": 8,
        "Other": 9
    }
    
    # Sample emails
    sample_emails = [
        {
            "id": "email_001",
            "subject": "Weekly Newsletter - Tech Updates",
            "body": "Here are this week's top technology news and updates from our team.",
            "sender": "newsletter@techcompany.com",
            "recipient": "user@example.com",
            "category": "Newsletter",
            "language": "en"
        },
        {
            "id": "email_002",
            "subject": "Meeting Tomorrow at 2 PM",
            "body": "Don't forget about our project meeting tomorrow at 2 PM in conference room A.",
            "sender": "manager@company.com",
            "recipient": "user@example.com",
            "category": "Work",
            "language": "en"
        },
        {
            "id": "email_003",
            "subject": "Happy Birthday!",
            "body": "Wishing you a very happy birthday! Hope you have a wonderful day.",
            "sender": "friend@personal.com",
            "recipient": "user@example.com",
            "category": "Personal",
            "language": "en"
        },
        {
            "id": "email_004",
            "subject": "Congratulations! You've Won!",
            "body": "Click here to claim your prize! Limited time offer!",
            "sender": "noreply@suspicious.com",
            "recipient": "user@example.com",
            "category": "Spam",
            "language": "en"
        },
        {
            "id": "email_005",
            "subject": "50% Off Sale - Limited Time",
            "body": "Don't miss our biggest sale of the year! 50% off all items.",
            "sender": "sales@retailstore.com",
            "recipient": "user@example.com",
            "category": "Promotional",
            "language": "en"
        }
    ]
    
    # Create train dataset
    train_dir = output_path / "train"
    train_dir.mkdir(exist_ok=True)
    
    with open(train_dir / "dataset.json", "w") as f:
        json.dump(sample_emails, f, indent=2)
    
    # Create test dataset (same as train for demo)
    test_dir = output_path / "test"
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / "dataset.json", "w") as f:
        json.dump(sample_emails, f, indent=2)
    
    # Create categories file
    with open(output_path / "categories.json", "w") as f:
        json.dump(categories, f, indent=2)
    
    # Create simple vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    word_id = 4
    
    for email in sample_emails:
        text = f"{email['subject']} {email['body']}"
        words = text.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = word_id
                word_id += 1
    
    with open(output_path / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Sample dataset created at: {output_path}")
    print(f"  - {len(sample_emails)} sample emails")
    print(f"  - {len(categories)} categories")
    print(f"  - {len(vocab)} vocabulary tokens")
    
    return True


def detect_hardware_and_recommend_config() -> Dict[str, Any]:
    """Detect MacBook hardware and recommend training configuration."""
    if not DEPENDENCIES_AVAILABLE:
        return {
            "error": "Hardware detection dependencies not available",
            "recommendation": "Use default configuration"
        }
    
    try:
        detector = HardwareDetector()
        specs = detector.get_hardware_specs()
        
        # Memory-based recommendations
        memory_gb = specs.memory.total_memory / (1024**3)
        
        if memory_gb <= 8:
            config_name = "macbook_8gb"
            batch_size = 2
            gradient_accumulation = 16
            max_steps = 5000
        elif memory_gb <= 16:
            config_name = "macbook_16gb"
            batch_size = 4
            gradient_accumulation = 8
            max_steps = 8000
        else:
            config_name = "macbook_32gb"
            batch_size = 8
            gradient_accumulation = 4
            max_steps = 10000
        
        recommendation = {
            "config_name": config_name,
            "hardware_specs": {
                "cpu_cores": specs.cpu.cores,
                "memory_gb": memory_gb,
                "available_memory_gb": specs.memory.available_memory / (1024**3),
                "platform": specs.platform.macos_version
            },
            "recommended_config": {
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation,
                "max_steps": max_steps,
                "learning_rate": 1e-4,
                "memory_limit_mb": int(memory_gb * 1024 * 0.7)  # Use 70% of available memory
            }
        }
        
        return recommendation
        
    except Exception as e:
        return {
            "error": f"Hardware detection failed: {e}",
            "recommendation": "Use default configuration for 8GB MacBook"
        }


def create_training_config(args: argparse.Namespace, 
                         hardware_recommendation: Dict[str, Any]) -> EmailTrainingConfig:
    """Create training configuration from arguments and hardware recommendation."""
    
    # Start with hardware recommendations
    if "recommended_config" in hardware_recommendation:
        hw_config = hardware_recommendation["recommended_config"]
        batch_size = hw_config.get("batch_size", 4)
        gradient_accumulation = hw_config.get("gradient_accumulation_steps", 8)
        max_steps = hw_config.get("max_steps", 5000)
        learning_rate = hw_config.get("learning_rate", 1e-4)
        memory_limit = hw_config.get("memory_limit_mb", 6000)
    else:
        # Default values for 8GB MacBook
        batch_size = 4
        gradient_accumulation = 8
        max_steps = 5000
        learning_rate = 1e-4
        memory_limit = 6000
    
    # Override with command line arguments
    if args.batch_size:
        batch_size = args.batch_size
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.max_steps:
        max_steps = args.max_steps
    if args.gradient_accumulation_steps:
        gradient_accumulation = args.gradient_accumulation_steps
    
    config = EmailTrainingConfig(
        # Model parameters
        model_name="EmailTRM",
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_email_categories=10,  # Fixed for email classification
        
        # Training parameters
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        
        # Email-specific parameters
        max_sequence_length=args.max_sequence_length,
        use_email_structure=True,
        subject_attention_weight=args.subject_attention_weight,
        pooling_strategy=args.pooling_strategy,
        
        # MacBook optimization parameters
        memory_limit_mb=memory_limit,
        enable_memory_monitoring=True,
        dynamic_batch_sizing=True,
        use_cpu_optimization=True,
        num_workers=min(4, os.cpu_count() or 2),
        
        # Performance targets
        target_accuracy=args.target_accuracy,
        min_category_accuracy=0.90,
        early_stopping_patience=args.early_stopping_patience,
        
        # Email-specific training parameters
        enable_subject_prioritization=True,
        use_hierarchical_attention=True,
        email_augmentation_prob=0.3
    )
    
    return config


def run_training(args: argparse.Namespace) -> bool:
    """Run email classification training."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not available for training")
        return False
    
    print("="*60)
    print("MacBook Email Classification Training")
    print("="*60)
    
    # Validate dataset
    if not validate_dataset_path(args.dataset_path):
        return False
    
    # Detect hardware and get recommendations
    print("\nDetecting MacBook hardware...")
    hardware_recommendation = detect_hardware_and_recommend_config()
    
    if "error" in hardware_recommendation:
        print(f"Warning: {hardware_recommendation['error']}")
    else:
        hw_specs = hardware_recommendation["hardware_specs"]
        print(f"Detected: {hw_specs['cpu_cores']} CPU cores, {hw_specs['memory_gb']:.1f}GB memory")
        print(f"Recommended config: {hardware_recommendation['config_name']}")
    
    # Create training configuration
    print("\nCreating training configuration...")
    config = create_training_config(args, hardware_recommendation)
    
    # Validate configuration
    try:
        validation_result = validate_email_training_config(config)
        if not validation_result.is_valid:
            print("Error: Configuration validation failed:")
            for error in validation_result.errors:
                print(f"  - {error}")
            return False
        
        if validation_result.warnings:
            print("Configuration warnings:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
    
    except Exception as e:
        print(f"Warning: Configuration validation failed: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Configuration saved to: {config_file}")
    
    # Initialize orchestrator
    print("\nInitializing training orchestrator...")
    orchestrator = EmailTrainingOrchestrator(
        output_dir=str(output_dir),
        enable_monitoring=True,
        enable_checkpointing=True
    )
    
    # Execute training pipeline
    print(f"\nStarting training with strategy: {args.strategy}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    try:
        result = orchestrator.execute_training_pipeline(
            dataset_path=args.dataset_path,
            config=config,
            strategy=args.strategy,
            total_steps=config.max_steps
        )
        
        # Print results
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        if result.success:
            print(f"✓ Training successful!")
            print(f"✓ Final accuracy: {result.final_accuracy:.4f}")
            print(f"✓ Best accuracy: {result.best_accuracy:.4f}")
            print(f"✓ Training time: {result.total_training_time/60:.1f} minutes")
            print(f"✓ Phases completed: {', '.join(result.phases_completed)}")
            
            if result.model_path:
                print(f"✓ Model saved: {result.model_path}")
            
            # Check if target accuracy reached
            if result.best_accuracy >= config.target_accuracy:
                print(f"🎉 Target accuracy {config.target_accuracy} achieved!")
            else:
                print(f"⚠️  Target accuracy {config.target_accuracy} not reached")
            
            # Category performance
            if result.category_accuracies:
                print("\nPer-category performance:")
                for category, accuracy in result.category_accuracies.items():
                    print(f"  {category}: {accuracy:.4f}")
        
        else:
            print("✗ Training failed!")
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        return result.success
        
    except Exception as e:
        print(f"\nError: Training failed with exception: {e}")
        return False


def run_hyperparameter_optimization(args: argparse.Namespace) -> bool:
    """Run hyperparameter optimization."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Required dependencies not available for hyperparameter optimization")
        return False
    
    print("="*60)
    print("MacBook Email Classification - Hyperparameter Optimization")
    print("="*60)
    
    # Validate dataset
    if not validate_dataset_path(args.dataset_path):
        return False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = EmailTrainingOrchestrator(
        output_dir=str(output_dir / "optimization"),
        enable_monitoring=True,
        enable_checkpointing=False  # Disable checkpointing for optimization trials
    )
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        orchestrator=orchestrator,
        optimization_strategy=args.optimization_strategy
    )
    
    print(f"\nStarting hyperparameter optimization:")
    print(f"Strategy: {args.optimization_strategy}")
    print(f"Trials: {args.num_trials}")
    print(f"Steps per trial: {args.max_steps_per_trial}")
    
    try:
        results = optimizer.optimize_hyperparameters(
            dataset_path=args.dataset_path,
            num_trials=args.num_trials,
            max_steps_per_trial=args.max_steps_per_trial,
            target_metric="accuracy"
        )
        
        # Print results
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("="*60)
        
        if results["success"]:
            print(f"✓ Optimization successful!")
            print(f"✓ Completed trials: {results['completed_trials']}/{results['num_trials']}")
            print(f"✓ Best performance: {results['best_performance']:.4f}")
            print(f"✓ Total time: {results['total_time']/60:.1f} minutes")
            
            if results["best_config"]:
                print("\nBest configuration:")
                best_config = results["best_config"]
                print(f"  Learning rate: {best_config['learning_rate']}")
                print(f"  Batch size: {best_config['batch_size']}")
                print(f"  Hidden size: {best_config['hidden_size']}")
                print(f"  Weight decay: {best_config['weight_decay']}")
                
                # Save best configuration
                best_config_file = output_dir / "best_config.json"
                with open(best_config_file, "w") as f:
                    json.dump(best_config, f, indent=2)
                print(f"✓ Best config saved: {best_config_file}")
        
        else:
            print("✗ Optimization failed!")
            if results["errors"]:
                print("Errors:")
                for error in results["errors"]:
                    print(f"  - {error}")
        
        # Save full results
        results_file = output_dir / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Full results saved: {results_file}")
        
        return results["success"]
        
    except Exception as e:
        print(f"\nError: Hyperparameter optimization failed: {e}")
        return False


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="MacBook Email Classification Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with sample dataset
  python train_email_classifier_macbook.py --create-sample-dataset --dataset-path ./sample_emails

  # Training with custom dataset
  python train_email_classifier_macbook.py --dataset-path ./my_email_dataset --output-dir ./training_output

  # Multi-phase training strategy
  python train_email_classifier_macbook.py --dataset-path ./emails --strategy multi_phase --max-steps 10000

  # Hyperparameter optimization
  python train_email_classifier_macbook.py --optimize --dataset-path ./emails --num-trials 20

  # Hardware detection only
  python train_email_classifier_macbook.py --detect-hardware
        """
    )
    
    # Main action
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--train", action="store_true", 
                            help="Run email classification training")
    action_group.add_argument("--optimize", action="store_true",
                            help="Run hyperparameter optimization")
    action_group.add_argument("--detect-hardware", action="store_true",
                            help="Detect hardware and show recommendations")
    action_group.add_argument("--create-sample-dataset", action="store_true",
                            help="Create sample dataset for testing")
    
    # Dataset and output
    parser.add_argument("--dataset-path", type=str, default="./email_dataset",
                       help="Path to email dataset directory")
    parser.add_argument("--output-dir", type=str, default="./email_training_output",
                       help="Output directory for training results")
    
    # Model parameters
    parser.add_argument("--vocab-size", type=int, default=5000,
                       help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=512,
                       help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of layers")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int,
                       help="Batch size (auto-detected if not specified)")
    parser.add_argument("--learning-rate", type=float,
                       help="Learning rate (auto-detected if not specified)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=10,
                       help="Maximum epochs")
    parser.add_argument("--max-steps", type=int,
                       help="Maximum steps (auto-detected if not specified)")
    parser.add_argument("--gradient-accumulation-steps", type=int,
                       help="Gradient accumulation steps (auto-detected if not specified)")
    
    # Email-specific parameters
    parser.add_argument("--max-sequence-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--subject-attention-weight", type=float, default=2.0,
                       help="Subject attention weight")
    parser.add_argument("--pooling-strategy", type=str, default="weighted",
                       choices=["weighted", "attention", "mean"],
                       help="Pooling strategy")
    
    # Training strategy
    parser.add_argument("--strategy", type=str, default="multi_phase",
                       choices=["single", "multi_phase", "progressive", "curriculum"],
                       help="Training strategy")
    
    # Performance targets
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                       help="Target accuracy")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                       help="Early stopping patience")
    
    # Hyperparameter optimization
    parser.add_argument("--optimization-strategy", type=str, default="bayesian",
                       choices=["random", "grid", "bayesian"],
                       help="Hyperparameter optimization strategy")
    parser.add_argument("--num-trials", type=int, default=10,
                       help="Number of optimization trials")
    parser.add_argument("--max-steps-per-trial", type=int, default=2000,
                       help="Maximum steps per optimization trial")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Execute requested action
    if args.detect_hardware:
        print("Detecting MacBook hardware...")
        recommendation = detect_hardware_and_recommend_config()
        
        if "error" in recommendation:
            print(f"Error: {recommendation['error']}")
            return 1
        
        hw_specs = recommendation["hardware_specs"]
        print(f"\nHardware detected:")
        print(f"  CPU cores: {hw_specs['cpu_cores']}")
        print(f"  Total memory: {hw_specs['memory_gb']:.1f}GB")
        print(f"  Available memory: {hw_specs['available_memory_gb']:.1f}GB")
        print(f"  Platform: {hw_specs['platform']}")
        
        print(f"\nRecommended configuration: {recommendation['config_name']}")
        rec_config = recommendation["recommended_config"]
        print(f"  Batch size: {rec_config['batch_size']}")
        print(f"  Gradient accumulation: {rec_config['gradient_accumulation_steps']}")
        print(f"  Max steps: {rec_config['max_steps']}")
        print(f"  Learning rate: {rec_config['learning_rate']}")
        print(f"  Memory limit: {rec_config['memory_limit_mb']}MB")
        
        return 0
    
    elif args.create_sample_dataset:
        print("Creating sample email dataset...")
        success = create_sample_dataset(args.dataset_path)
        return 0 if success else 1
    
    elif args.train:
        success = run_training(args)
        return 0 if success else 1
    
    elif args.optimize:
        success = run_hyperparameter_optimization(args)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())