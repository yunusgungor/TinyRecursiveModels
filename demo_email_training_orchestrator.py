#!/usr/bin/env python3
"""
Demo script for Email Training Orchestrator

This script demonstrates the complete email classification training pipeline
with multi-phase training strategies and hyperparameter optimization.
"""

import os
import json
import tempfile
from pathlib import Path

from macbook_optimization.email_training_orchestrator import (
    EmailTrainingOrchestrator, HyperparameterOptimizer, HyperparameterSearchSpace
)
from macbook_optimization.email_training_config import EmailTrainingConfig


def create_demo_email_dataset(num_emails: int = 50) -> str:
    """Create a demo email dataset for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create train directory
    train_dir = os.path.join(temp_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Email categories
    categories = ["newsletter", "work", "personal", "promotional", "spam", 
                 "social", "finance", "travel", "shopping", "other"]
    
    # Generate sample emails
    emails = []
    for i in range(num_emails):
        category = categories[i % len(categories)]
        
        email = {
            "id": f"demo_email_{i:03d}",
            "subject": f"Demo {category.title()} Email {i}",
            "body": f"This is a demo {category} email with content for testing purposes. " * 3,
            "sender": f"sender{i}@{category}.com",
            "recipient": "user@example.com",
            "category": category,
            "language": "en"
        }
        emails.append(email)
    
    # Write to JSONL file
    train_file = os.path.join(train_dir, "emails.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for email in emails:
            f.write(json.dumps(email) + '\n')
    
    print(f"Created demo dataset with {num_emails} emails at: {temp_dir}")
    return temp_dir


def demo_training_environment_setup():
    """Demonstrate training environment setup."""
    print("\n" + "="*60)
    print("DEMO: Training Environment Setup")
    print("="*60)
    
    # Create demo dataset
    dataset_path = create_demo_email_dataset(30)
    
    # Initialize orchestrator
    orchestrator = EmailTrainingOrchestrator(
        output_dir="demo_email_training_output",
        enable_monitoring=True,
        enable_checkpointing=True
    )
    
    # Setup training environment
    print("\nSetting up training environment...")
    env_result = orchestrator.setup_training_environment(dataset_path)
    
    if env_result["success"]:
        print("✅ Environment setup successful!")
        print(f"Hardware: {env_result['hardware_specs']['cpu_cores']} cores, "
              f"{env_result['hardware_specs']['memory_gb']:.1f}GB memory")
        print(f"Dataset: {env_result['dataset_info']['total_emails']} emails, "
              f"{env_result['dataset_info']['total_size_mb']:.2f}MB")
        
        if env_result["warnings"]:
            print(f"⚠️  Warnings: {len(env_result['warnings'])}")
            for warning in env_result["warnings"][:3]:  # Show first 3
                print(f"   - {warning}")
    else:
        print("❌ Environment setup failed!")
        for error in env_result["errors"]:
            print(f"   Error: {error}")
    
    # Cleanup
    import shutil
    shutil.rmtree(dataset_path)
    
    return env_result["success"]


def demo_multi_phase_training():
    """Demonstrate multi-phase training strategies."""
    print("\n" + "="*60)
    print("DEMO: Multi-Phase Training Strategies")
    print("="*60)
    
    orchestrator = EmailTrainingOrchestrator()
    config = EmailTrainingConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        batch_size=4,
        learning_rate=1e-4
    )
    
    strategies = ["single", "multi_phase", "progressive", "curriculum"]
    
    for strategy in strategies:
        print(f"\n📋 Strategy: {strategy}")
        phases = orchestrator.create_training_phases(
            strategy=strategy,
            total_steps=1000,
            base_config=config
        )
        
        print(f"   Phases: {len(phases)}")
        for i, phase in enumerate(phases, 1):
            print(f"   {i}. {phase.name}: {phase.steps} steps, "
                  f"LR={phase.learning_rate:.2e}, BS={phase.batch_size}")
            if phase.model_config_overrides:
                print(f"      Model overrides: {phase.model_config_overrides}")


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization."""
    print("\n" + "="*60)
    print("DEMO: Hyperparameter Optimization")
    print("="*60)
    
    # Create mock orchestrator
    class MockOrchestrator:
        def execute_training_pipeline(self, **kwargs):
            # Simulate training results
            import random
            from macbook_optimization.email_training_orchestrator import TrainingResult
            from datetime import datetime
            
            config = kwargs.get('config', EmailTrainingConfig())
            
            # Simulate performance based on learning rate and batch size
            base_performance = 0.75
            lr_factor = min(0.15, config.learning_rate * 1000)
            batch_factor = min(0.05, config.batch_size / 100)
            noise = random.uniform(-0.05, 0.05)
            
            accuracy = base_performance + lr_factor + batch_factor + noise
            
            return TrainingResult(
                success=True,
                training_id=f"mock_trial_{random.randint(1000, 9999)}",
                start_time=datetime.now(),
                end_time=datetime.now(),
                config=config,
                phases_completed=["main_training"],
                final_accuracy=accuracy,
                best_accuracy=accuracy + 0.01,
                final_loss=1.0 - accuracy,
                best_loss=1.0 - accuracy - 0.01,
                category_accuracies={f"cat_{i}": accuracy + random.uniform(-0.02, 0.02) for i in range(5)},
                total_training_time=120.0,
                total_steps=1000,
                samples_processed=5000,
                peak_memory_usage_mb=3000.0,
                average_cpu_usage=60.0,
                model_path=None,
                checkpoint_path=None,
                errors=[],
                warnings=[]
            )
    
    mock_orchestrator = MockOrchestrator()
    
    # Create hyperparameter optimizer
    search_space = HyperparameterSearchSpace(
        hidden_size=[128, 256, 384],
        learning_rate=[5e-5, 1e-4, 2e-4],
        batch_size=[2, 4, 8]
    )
    
    optimizer = HyperparameterOptimizer(
        orchestrator=mock_orchestrator,
        search_space=search_space,
        optimization_strategy="random"
    )
    
    print("🔍 Starting hyperparameter optimization...")
    print(f"Search space: {len(search_space.hidden_size)} × {len(search_space.learning_rate)} × {len(search_space.batch_size)} = "
          f"{len(search_space.hidden_size) * len(search_space.learning_rate) * len(search_space.batch_size)} combinations")
    
    # Create a temporary dataset for the demo
    dataset_path = create_demo_email_dataset(20)
    
    # Run optimization (this will use the mock orchestrator internally)
    result = optimizer.optimize_hyperparameters(
        dataset_path=dataset_path,
        num_trials=5,
        target_metric="accuracy"
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(dataset_path)
    
    if result["success"]:
        print("✅ Optimization completed!")
        print(f"   Trials completed: {result['completed_trials']}/{result['num_trials']}")
        print(f"   Best performance: {result['best_performance']:.4f}")
        print(f"   Success rate: {result['completed_trials']/result['num_trials']:.1%}")
        
        if result["best_config"]:
            best_config = result["best_config"]
            print(f"   Best config: LR={best_config['learning_rate']:.2e}, "
                  f"BS={best_config['batch_size']}, "
                  f"HS={best_config['hidden_size']}")
    else:
        print("❌ Optimization failed!")
        for error in result["errors"]:
            print(f"   Error: {error}")


def demo_training_orchestrator_summary():
    """Demonstrate training orchestrator summary features."""
    print("\n" + "="*60)
    print("DEMO: Training Orchestrator Summary")
    print("="*60)
    
    orchestrator = EmailTrainingOrchestrator()
    
    # Initially empty
    summary = orchestrator.get_training_summary()
    print("📊 Initial training summary:")
    print(f"   {summary}")
    
    # Simulate some training history
    from macbook_optimization.email_training_orchestrator import TrainingResult
    from datetime import datetime
    
    # Add mock training results
    for i in range(3):
        result = TrainingResult(
            success=i < 2,  # First 2 successful, last one failed
            training_id=f"demo_training_{i+1:03d}",
            start_time=datetime.now(),
            end_time=datetime.now(),
            config=EmailTrainingConfig(),
            phases_completed=["warmup", "main_training"] if i < 2 else [],
            final_accuracy=0.85 + i * 0.05 if i < 2 else None,
            best_accuracy=0.87 + i * 0.04 if i < 2 else None,
            final_loss=0.3 - i * 0.05 if i < 2 else None,
            best_loss=0.25 - i * 0.04 if i < 2 else None,
            category_accuracies={f"cat_{j}": 0.8 + i * 0.05 for j in range(5)} if i < 2 else {},
            total_training_time=300.0 + i * 100,
            total_steps=1000 + i * 500,
            samples_processed=5000 + i * 2000,
            peak_memory_usage_mb=4000.0 + i * 500,
            average_cpu_usage=60.0 + i * 10,
            model_path=f"/tmp/model_{i+1}.pt" if i < 2 else None,
            checkpoint_path=f"/tmp/checkpoint_{i+1}.pt" if i < 2 else None,
            errors=[] if i < 2 else ["Training failed due to memory issues"],
            warnings=[]
        )
        orchestrator.training_history.append(result)
    
    # Get updated summary
    summary = orchestrator.get_training_summary()
    print("\n📊 Updated training summary:")
    print(f"   Total runs: {summary['total_runs']}")
    print(f"   Successful runs: {summary['successful_runs']}")
    print(f"   Failed runs: {summary['failed_runs']}")
    print(f"   Best accuracy: {summary['best_accuracy']:.4f}")
    print(f"   Average training time: {summary['average_training_time']/60:.1f} minutes")
    
    print("\n📋 Recent runs:")
    for run in summary['recent_runs']:
        status = "✅" if run['success'] else "❌"
        acc_str = f"{run['final_accuracy']:.3f}" if run['final_accuracy'] is not None else "N/A"
        print(f"   {status} {run['training_id']}: "
              f"acc={acc_str}, "
              f"time={run['training_time_minutes']:.1f}min, "
              f"phases={len(run['phases_completed'])}")


def main():
    """Run all demos."""
    print("🚀 Email Training Orchestrator Demo")
    print("This demo showcases the complete email classification training pipeline")
    
    try:
        # Demo 1: Environment setup
        setup_success = demo_training_environment_setup()
        
        # Demo 2: Multi-phase training strategies
        demo_multi_phase_training()
        
        # Demo 3: Hyperparameter optimization
        demo_hyperparameter_optimization()
        
        # Demo 4: Training summary
        demo_training_orchestrator_summary()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
        print("\n📚 Key Features Demonstrated:")
        print("   • Automatic hardware detection and configuration adaptation")
        print("   • Multi-phase training strategies (single, multi-phase, progressive, curriculum)")
        print("   • Bayesian hyperparameter optimization with email-specific search spaces")
        print("   • Comprehensive training monitoring and progress tracking")
        print("   • MacBook-optimized memory management and resource utilization")
        print("   • Email-specific tokenization and dataset management")
        
        print("\n🎯 Next Steps:")
        print("   • Use EmailTrainingOrchestrator.execute_training_pipeline() for full training")
        print("   • Customize training phases with create_training_phases()")
        print("   • Optimize hyperparameters with HyperparameterOptimizer")
        print("   • Monitor training progress with built-in progress tracking")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()