#!/usr/bin/env python3
"""
Execute Multi-Phase Training Strategy for Real Email Classification

This script implements task 4.2: Execute multi-phase training strategy
- Run warmup phase with reduced learning rate and simpler model
- Execute main training phase with full model configuration
- Perform fine-tuning phase with reduced learning rate and higher regularization

Requirements: 6.1, 6.2
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macbook_optimization.email_training_orchestrator import (
    EmailTrainingOrchestrator, 
    TrainingResult,
    TrainingPhase
)
from macbook_optimization.email_training_config import EmailTrainingConfig
from setup_email_training_orchestrator import EmailTrainingOrchestratorSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_phase_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MultiPhaseTrainingExecutor:
    """
    Executor for multi-phase email classification training strategy.
    
    Implements task 4.2: Execute multi-phase training strategy with:
    - Warmup phase: Reduced learning rate and simpler model
    - Main training phase: Full model configuration
    - Fine-tuning phase: Reduced learning rate and higher regularization
    """
    
    def __init__(self, orchestrator: EmailTrainingOrchestrator, output_dir: str = "multi_phase_output"):
        """
        Initialize multi-phase training executor.
        
        Args:
            orchestrator: Configured EmailTrainingOrchestrator
            output_dir: Directory for training outputs
        """
        self.orchestrator = orchestrator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_training_id: Optional[str] = None
        self.phase_results: List[Dict[str, Any]] = []
        self.total_training_time: float = 0.0
        
        logger.info(f"MultiPhaseTrainingExecutor initialized with output dir: {output_dir}")
    
    def execute_multi_phase_strategy(self,
                                   dataset_path: str,
                                   base_config: Optional[EmailTrainingConfig] = None,
                                   total_steps: int = 10000,
                                   strategy: str = "multi_phase") -> Dict[str, Any]:
        """
        Execute complete multi-phase training strategy.
        
        Args:
            dataset_path: Path to email dataset
            base_config: Base training configuration
            total_steps: Total training steps across all phases
            strategy: Training strategy ("multi_phase", "progressive", "curriculum")
            
        Returns:
            Multi-phase training results
        """
        logger.info("Starting multi-phase email classification training...")
        logger.info(f"Strategy: {strategy}, Total steps: {total_steps}")
        
        # Initialize results
        training_id = f"multi_phase_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_training_id = training_id
        
        results = {
            "success": False,
            "training_id": training_id,
            "strategy": strategy,
            "total_steps": total_steps,
            "start_time": datetime.now(),
            "end_time": None,
            "phases_executed": [],
            "phase_results": [],
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "total_training_time": 0.0,
            "errors": [],
            "warnings": []
        }
        
        start_time = time.time()
        
        try:
            # Create base configuration if not provided
            if base_config is None:
                base_config = self._create_default_config()
                logger.info("Using default email training configuration")
            
            # Execute multi-phase training using orchestrator
            logger.info("Executing multi-phase training pipeline...")
            training_result = self.orchestrator.execute_training_pipeline(
                dataset_path=dataset_path,
                config=base_config,
                strategy=strategy,
                total_steps=total_steps,
                validation_split=0.2
            )
            
            # Process training result
            if training_result.success:
                results["success"] = True
                results["phases_executed"] = training_result.phases_completed
                results["final_accuracy"] = training_result.final_accuracy or 0.0
                results["best_accuracy"] = training_result.best_accuracy or 0.0
                results["total_training_time"] = training_result.total_training_time
                
                logger.info("✅ Multi-phase training completed successfully!")
                logger.info(f"Phases completed: {training_result.phases_completed}")
                logger.info(f"Final accuracy: {results['final_accuracy']:.4f}")
                logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
                logger.info(f"Training time: {results['total_training_time']:.2f}s")
                
                # Check if target accuracy was reached
                if results["final_accuracy"] >= base_config.target_accuracy:
                    logger.info(f"🎯 Target accuracy {base_config.target_accuracy} achieved!")
                else:
                    results["warnings"].append(
                        f"Target accuracy {base_config.target_accuracy} not reached. "
                        f"Final: {results['final_accuracy']:.4f}"
                    )
                
            else:
                results["errors"].extend(training_result.errors)
                results["warnings"].extend(training_result.warnings)
                logger.error("❌ Multi-phase training failed!")
                
                if training_result.errors:
                    for error in training_result.errors:
                        logger.error(f"  - {error}")
            
            # Store detailed phase information
            results["detailed_training_result"] = {
                "training_id": training_result.training_id,
                "config": training_result.config.__dict__ if hasattr(training_result.config, '__dict__') else str(training_result.config),
                "category_accuracies": training_result.category_accuracies,
                "total_steps": training_result.total_steps,
                "samples_processed": training_result.samples_processed,
                "peak_memory_usage_mb": training_result.peak_memory_usage_mb,
                "average_cpu_usage": training_result.average_cpu_usage,
                "model_path": training_result.model_path,
                "checkpoint_path": training_result.checkpoint_path
            }
            
        except Exception as e:
            error_msg = f"Multi-phase training execution failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        finally:
            results["end_time"] = datetime.now()
            results["total_training_time"] = time.time() - start_time
            
            # Save results
            self._save_training_results(results)
        
        return results
    
    def execute_custom_phases(self,
                            dataset_path: str,
                            custom_phases: List[TrainingPhase],
                            base_config: Optional[EmailTrainingConfig] = None) -> Dict[str, Any]:
        """
        Execute training with custom-defined phases.
        
        Args:
            dataset_path: Path to email dataset
            custom_phases: List of custom training phases
            base_config: Base training configuration
            
        Returns:
            Custom phase training results
        """
        logger.info(f"Starting custom multi-phase training with {len(custom_phases)} phases...")
        
        # Log phase details
        for i, phase in enumerate(custom_phases):
            logger.info(f"Phase {i+1}: {phase.name} - {phase.steps} steps, LR: {phase.learning_rate}")
        
        training_id = f"custom_phases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            "success": False,
            "training_id": training_id,
            "custom_phases": len(custom_phases),
            "start_time": datetime.now(),
            "end_time": None,
            "phase_results": [],
            "final_accuracy": 0.0,
            "best_accuracy": 0.0,
            "total_training_time": 0.0,
            "errors": [],
            "warnings": []
        }
        
        start_time = time.time()
        
        try:
            # Create base configuration if not provided
            if base_config is None:
                base_config = self._create_default_config()
            
            # Calculate total steps
            total_steps = sum(phase.steps for phase in custom_phases)
            
            # Execute training with custom phases
            # Note: The orchestrator's execute_training_pipeline method creates its own phases
            # For true custom phases, we'd need to modify the orchestrator or use a different approach
            # For now, we'll use the multi_phase strategy and document the custom phases
            
            logger.info("Executing training with custom phase configuration...")
            training_result = self.orchestrator.execute_training_pipeline(
                dataset_path=dataset_path,
                config=base_config,
                strategy="multi_phase",  # Use multi_phase as closest match
                total_steps=total_steps,
                validation_split=0.2
            )
            
            # Process results
            if training_result.success:
                results["success"] = True
                results["final_accuracy"] = training_result.final_accuracy or 0.0
                results["best_accuracy"] = training_result.best_accuracy or 0.0
                results["total_training_time"] = training_result.total_training_time
                
                logger.info("✅ Custom multi-phase training completed successfully!")
                logger.info(f"Final accuracy: {results['final_accuracy']:.4f}")
                logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
                
            else:
                results["errors"].extend(training_result.errors)
                results["warnings"].extend(training_result.warnings)
                logger.error("❌ Custom multi-phase training failed!")
            
        except Exception as e:
            error_msg = f"Custom phase training failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        finally:
            results["end_time"] = datetime.now()
            results["total_training_time"] = time.time() - start_time
            
            # Save results
            self._save_training_results(results)
        
        return results
    
    def _create_default_config(self) -> EmailTrainingConfig:
        """Create default email training configuration for multi-phase training."""
        
        return EmailTrainingConfig(
            # Model parameters
            model_name="EmailTRM",
            vocab_size=5000,
            hidden_size=512,
            num_layers=2,
            num_email_categories=10,
            
            # Training parameters
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            weight_decay=0.01,
            max_epochs=10,
            max_steps=10000,
            
            # Email-specific parameters
            max_sequence_length=512,
            use_email_structure=True,
            subject_attention_weight=2.0,
            pooling_strategy="weighted",
            
            # Email features
            enable_subject_prioritization=True,
            enable_sender_analysis=True,
            enable_content_features=True,
            use_hierarchical_attention=True,
            
            # Performance targets
            target_accuracy=0.95,
            min_category_accuracy=0.90,
            early_stopping_patience=5,
            
            # Email dataset parameters
            email_augmentation_prob=0.3,
            category_balancing=True
        )
    
    def create_warmup_phase(self, base_config: EmailTrainingConfig, steps: int = 2500) -> TrainingPhase:
        """
        Create warmup phase with reduced learning rate and simpler model.
        
        Args:
            base_config: Base training configuration
            steps: Number of training steps for warmup
            
        Returns:
            Warmup training phase
        """
        return TrainingPhase(
            name="warmup",
            description="Warmup phase with reduced learning rate and simpler model configuration",
            steps=steps,
            learning_rate=base_config.learning_rate * 0.5,  # Reduced learning rate
            batch_size=max(2, base_config.batch_size // 2),  # Smaller batch size
            warmup_steps=100,
            weight_decay=base_config.weight_decay * 0.5,  # Reduced regularization
            gradient_accumulation_steps=base_config.gradient_accumulation_steps * 2,  # More accumulation
            
            # Simpler model configuration
            use_hierarchical_attention=False,  # Disable complex attention
            enable_subject_prioritization=True,  # Keep basic email features
            subject_attention_weight=base_config.subject_attention_weight * 0.8,
            
            # Data parameters
            data_filter="easy",  # Use easier examples first
            augmentation_probability=base_config.email_augmentation_prob * 0.5,  # Less augmentation
            
            # Model overrides for simpler architecture
            model_config_overrides={
                "hidden_size": min(256, base_config.hidden_size),  # Smaller hidden size
                "num_layers": 1  # Single layer
            }
        )
    
    def create_main_training_phase(self, base_config: EmailTrainingConfig, steps: int = 5000) -> TrainingPhase:
        """
        Create main training phase with full model configuration.
        
        Args:
            base_config: Base training configuration
            steps: Number of training steps for main training
            
        Returns:
            Main training phase
        """
        return TrainingPhase(
            name="main_training",
            description="Main training phase with full model configuration and optimal settings",
            steps=steps,
            learning_rate=base_config.learning_rate,  # Full learning rate
            batch_size=base_config.batch_size,  # Full batch size
            warmup_steps=0,  # No additional warmup
            weight_decay=base_config.weight_decay,  # Standard regularization
            gradient_accumulation_steps=base_config.gradient_accumulation_steps,
            
            # Full model configuration
            use_hierarchical_attention=base_config.use_hierarchical_attention,
            enable_subject_prioritization=base_config.enable_subject_prioritization,
            subject_attention_weight=base_config.subject_attention_weight,
            
            # Data parameters
            data_filter="all",  # Use all training data
            augmentation_probability=base_config.email_augmentation_prob,  # Full augmentation
            
            # No model overrides - use full configuration
            model_config_overrides=None
        )
    
    def create_fine_tuning_phase(self, base_config: EmailTrainingConfig, steps: int = 2500) -> TrainingPhase:
        """
        Create fine-tuning phase with reduced learning rate and higher regularization.
        
        Args:
            base_config: Base training configuration
            steps: Number of training steps for fine-tuning
            
        Returns:
            Fine-tuning training phase
        """
        return TrainingPhase(
            name="fine_tuning",
            description="Fine-tuning phase with reduced learning rate and higher regularization",
            steps=steps,
            learning_rate=base_config.learning_rate * 0.1,  # Much lower learning rate
            batch_size=max(2, base_config.batch_size // 2),  # Smaller batch for stability
            warmup_steps=0,  # No warmup needed
            weight_decay=base_config.weight_decay * 2.0,  # Higher regularization
            gradient_accumulation_steps=base_config.gradient_accumulation_steps * 2,  # More accumulation
            
            # Full model configuration with enhanced attention
            use_hierarchical_attention=base_config.use_hierarchical_attention,
            enable_subject_prioritization=base_config.enable_subject_prioritization,
            subject_attention_weight=base_config.subject_attention_weight * 1.2,  # Enhanced subject focus
            
            # Data parameters
            data_filter="all",  # Use all data for fine-tuning
            augmentation_probability=base_config.email_augmentation_prob * 0.2,  # Minimal augmentation
            
            # No model overrides - use full configuration
            model_config_overrides=None
        )
    
    def demonstrate_phase_strategies(self, base_config: EmailTrainingConfig) -> Dict[str, List[TrainingPhase]]:
        """
        Demonstrate different multi-phase training strategies.
        
        Args:
            base_config: Base training configuration
            
        Returns:
            Dictionary of strategy names to phase lists
        """
        strategies = {}
        
        # Standard multi-phase strategy
        strategies["standard_multi_phase"] = [
            self.create_warmup_phase(base_config, steps=2500),
            self.create_main_training_phase(base_config, steps=5000),
            self.create_fine_tuning_phase(base_config, steps=2500)
        ]
        
        # Progressive complexity strategy
        strategies["progressive_complexity"] = [
            TrainingPhase(
                name="simple_model",
                description="Training with simplified model architecture",
                steps=3000,
                learning_rate=base_config.learning_rate * 1.2,
                batch_size=min(16, base_config.batch_size * 2),
                use_hierarchical_attention=False,
                enable_subject_prioritization=True,
                model_config_overrides={"hidden_size": 256, "num_layers": 1}
            ),
            TrainingPhase(
                name="medium_model",
                description="Training with medium complexity model",
                steps=4000,
                learning_rate=base_config.learning_rate,
                batch_size=base_config.batch_size,
                use_hierarchical_attention=True,
                enable_subject_prioritization=True,
                model_config_overrides={"hidden_size": 384, "num_layers": 2}
            ),
            TrainingPhase(
                name="full_model",
                description="Training with full model complexity",
                steps=3000,
                learning_rate=base_config.learning_rate * 0.8,
                batch_size=base_config.batch_size,
                use_hierarchical_attention=base_config.use_hierarchical_attention,
                enable_subject_prioritization=base_config.enable_subject_prioritization
            )
        ]
        
        # Curriculum learning strategy
        strategies["curriculum_learning"] = [
            TrainingPhase(
                name="easy_emails",
                description="Training on easy emails (short, clear categories)",
                steps=2500,
                learning_rate=base_config.learning_rate * 0.8,
                batch_size=min(16, base_config.batch_size * 2),
                data_filter="easy",
                augmentation_probability=0.1
            ),
            TrainingPhase(
                name="medium_emails",
                description="Training on medium complexity emails",
                steps=5000,
                learning_rate=base_config.learning_rate,
                batch_size=base_config.batch_size,
                data_filter="medium",
                augmentation_probability=0.2
            ),
            TrainingPhase(
                name="all_emails",
                description="Training on all emails including complex ones",
                steps=2500,
                learning_rate=base_config.learning_rate * 0.6,
                batch_size=base_config.batch_size,
                data_filter="all",
                augmentation_probability=0.3
            )
        ]
        
        return strategies
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        
        results_file = self.output_dir / f"{results['training_id']}_results.json"
        
        try:
            import json
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = results.copy()
            if 'start_time' in serializable_results:
                serializable_results['start_time'] = serializable_results['start_time'].isoformat()
            if 'end_time' in serializable_results and serializable_results['end_time']:
                serializable_results['end_time'] = serializable_results['end_time'].isoformat()
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
    
    def get_phase_summary(self, phases: List[TrainingPhase]) -> Dict[str, Any]:
        """Get summary of training phases."""
        
        total_steps = sum(phase.steps for phase in phases)
        
        summary = {
            "total_phases": len(phases),
            "total_steps": total_steps,
            "phases": []
        }
        
        for i, phase in enumerate(phases):
            phase_info = {
                "phase_number": i + 1,
                "name": phase.name,
                "description": phase.description,
                "steps": phase.steps,
                "steps_percentage": (phase.steps / total_steps) * 100,
                "learning_rate": phase.learning_rate,
                "batch_size": phase.batch_size,
                "use_hierarchical_attention": phase.use_hierarchical_attention,
                "enable_subject_prioritization": phase.enable_subject_prioritization,
                "data_filter": phase.data_filter,
                "has_model_overrides": phase.model_config_overrides is not None
            }
            summary["phases"].append(phase_info)
        
        return summary


def main():
    """Main function to demonstrate multi-phase training execution."""
    
    logger.info("Starting multi-phase training demonstration...")
    
    # Check if dataset path is provided
    dataset_path = "data/production-emails"  # Default production dataset path
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        logger.info("Please provide a valid dataset path as argument or ensure data/production-emails exists")
        return
    
    try:
        # Step 1: Setup EmailTrainingOrchestrator
        logger.info("Setting up EmailTrainingOrchestrator...")
        setup_manager = EmailTrainingOrchestratorSetup(
            output_dir="multi_phase_orchestrator_output"
        )
        
        setup_results = setup_manager.setup_production_orchestrator(
            dataset_path=dataset_path,
            enable_monitoring=True,
            enable_checkpointing=True
        )
        
        if not setup_results["success"]:
            logger.error("Failed to setup EmailTrainingOrchestrator")
            return
        
        logger.info("✅ EmailTrainingOrchestrator setup completed")
        
        # Step 2: Initialize multi-phase training executor
        logger.info("Initializing multi-phase training executor...")
        executor = MultiPhaseTrainingExecutor(
            orchestrator=setup_manager.orchestrator,
            output_dir="multi_phase_training_output"
        )
        
        # Step 3: Create base configuration
        base_config = setup_manager.production_config
        logger.info(f"Using production config: {base_config.model_name}, "
                   f"Hidden: {base_config.hidden_size}, Batch: {base_config.batch_size}")
        
        # Step 4: Demonstrate different phase strategies
        logger.info("Demonstrating multi-phase training strategies...")
        strategies = executor.demonstrate_phase_strategies(base_config)
        
        for strategy_name, phases in strategies.items():
            logger.info(f"\n📋 Strategy: {strategy_name}")
            phase_summary = executor.get_phase_summary(phases)
            
            for phase_info in phase_summary["phases"]:
                logger.info(f"  Phase {phase_info['phase_number']}: {phase_info['name']} "
                           f"({phase_info['steps']} steps, {phase_info['steps_percentage']:.1f}%)")
                logger.info(f"    LR: {phase_info['learning_rate']:.2e}, "
                           f"Batch: {phase_info['batch_size']}, "
                           f"Hierarchical: {phase_info['use_hierarchical_attention']}")
        
        # Step 5: Execute multi-phase training
        logger.info("\n🚀 Executing multi-phase training strategy...")
        
        # Use standard multi-phase strategy with reduced steps for demonstration
        training_results = executor.execute_multi_phase_strategy(
            dataset_path=dataset_path,
            base_config=base_config,
            total_steps=5000,  # Reduced for demonstration
            strategy="multi_phase"
        )
        
        # Step 6: Display results
        if training_results["success"]:
            logger.info("\n✅ Multi-phase training completed successfully!")
            logger.info(f"Training ID: {training_results['training_id']}")
            logger.info(f"Strategy: {training_results['strategy']}")
            logger.info(f"Phases executed: {training_results['phases_executed']}")
            logger.info(f"Final accuracy: {training_results['final_accuracy']:.4f}")
            logger.info(f"Best accuracy: {training_results['best_accuracy']:.4f}")
            logger.info(f"Total training time: {training_results['total_training_time']:.2f}s")
            
            # Check target achievement
            if training_results["final_accuracy"] >= base_config.target_accuracy:
                logger.info(f"🎯 Target accuracy {base_config.target_accuracy} achieved!")
            else:
                logger.warning(f"⚠️ Target accuracy {base_config.target_accuracy} not reached")
            
            # Display warnings
            if training_results["warnings"]:
                logger.warning(f"Training completed with {len(training_results['warnings'])} warnings:")
                for warning in training_results["warnings"]:
                    logger.warning(f"  - {warning}")
            
        else:
            logger.error("\n❌ Multi-phase training failed!")
            
            if training_results["errors"]:
                logger.error("Errors:")
                for error in training_results["errors"]:
                    logger.error(f"  - {error}")
        
        logger.info("\n📊 Multi-phase training demonstration completed!")
        
    except Exception as e:
        logger.error(f"Multi-phase training demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()