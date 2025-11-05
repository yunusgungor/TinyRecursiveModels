#!/usr/bin/env python3
"""
Implement Checkpoint Management for Real Email Classification Training

This script implements task 4.4: Implement checkpoint management
- Save training checkpoints at regular intervals
- Enable automatic resumption after training interruptions
- Manage checkpoint storage and cleanup

Requirements: 7.1
"""

import os
import sys
import time
import json
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macbook_optimization.checkpoint_management import (
    CheckpointManager, 
    CheckpointConfig,
    CheckpointMetadata,
    CheckpointSaveResult,
    CheckpointLoadResult
)
from macbook_optimization.email_training_orchestrator import EmailTrainingOrchestrator
from macbook_optimization.email_training_config import EmailTrainingConfig
from macbook_optimization.memory_management import MemoryManager
from setup_email_training_orchestrator import EmailTrainingOrchestratorSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('checkpoint_management.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EmailTrainingCheckpointManager:
    """
    Enhanced checkpoint management specifically for email classification training.
    
    Implements task 4.4: Implement checkpoint management with:
    - Regular checkpoint saving during training
    - Automatic resumption after interruptions
    - Intelligent storage and cleanup management
    """
    
    def __init__(self, 
                 checkpoint_dir: str = "email_training_checkpoints",
                 orchestrator: Optional[EmailTrainingOrchestrator] = None):
        """
        Initialize email training checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            orchestrator: EmailTrainingOrchestrator instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.orchestrator = orchestrator
        
        # Create checkpoint configuration optimized for email training
        self.checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=5,  # Keep more checkpoints for email training
            min_disk_space_mb=2000.0,  # 2GB minimum free space
            save_interval_steps=500,  # Save every 500 steps
            save_interval_minutes=15.0,  # Also save every 15 minutes
            memory_aware_intervals=True,
            memory_threshold_for_saving=75.0,  # Don't save if memory > 75%
            wait_for_memory_cooldown=True,
            max_wait_time_seconds=180.0,  # Max 3 minutes wait
            auto_cleanup=True,
            cleanup_on_low_disk=True,
            disk_space_threshold_mb=3000.0,  # Cleanup if less than 3GB free
            validate_on_load=True,
            strict_config_validation=False,  # Allow some config flexibility
            compress_checkpoints=True,
            compression_level=6
        )
        
        # Initialize checkpoint manager
        self.memory_manager = MemoryManager()
        self.checkpoint_manager = CheckpointManager(
            config=self.checkpoint_config,
            memory_manager=self.memory_manager,
            logger=logger
        )
        
        # Training state tracking
        self.current_training_id: Optional[str] = None
        self.training_interrupted = False
        self.interruption_checkpoint_id: Optional[str] = None
        
        # Setup signal handlers for graceful interruption handling
        self._setup_interruption_handlers()
        
        logger.info(f"EmailTrainingCheckpointManager initialized with checkpoint dir: {checkpoint_dir}")
    
    def _setup_interruption_handlers(self):
        """Setup signal handlers for graceful training interruption."""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} - preparing for graceful shutdown...")
            self.training_interrupted = True
            
            # Save emergency checkpoint if training is active
            if self.current_training_id:
                logger.info("Saving emergency checkpoint before shutdown...")
                self._save_emergency_checkpoint()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        if hasattr(signal, 'SIGHUP'):  # Unix systems only
            signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal
    
    def start_checkpoint_management(self, training_id: str, config: EmailTrainingConfig) -> Dict[str, Any]:
        """
        Start checkpoint management for a training session.
        
        Args:
            training_id: Unique training identifier
            config: Email training configuration
            
        Returns:
            Checkpoint management setup results
        """
        logger.info(f"Starting checkpoint management for training: {training_id}")
        
        self.current_training_id = training_id
        self.training_interrupted = False
        
        setup_results = {
            "success": False,
            "training_id": training_id,
            "checkpoint_dir": str(self.checkpoint_dir),
            "resuming_from_checkpoint": False,
            "resumed_checkpoint_id": None,
            "checkpoint_config": {
                "max_checkpoints": self.checkpoint_config.max_checkpoints,
                "save_interval_steps": self.checkpoint_config.save_interval_steps,
                "save_interval_minutes": self.checkpoint_config.save_interval_minutes,
                "auto_cleanup": self.checkpoint_config.auto_cleanup
            },
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check for existing checkpoints to resume from
            existing_checkpoints = self.checkpoint_manager.list_checkpoints(limit=5)
            
            if existing_checkpoints:
                logger.info(f"Found {len(existing_checkpoints)} existing checkpoints")
                
                # Check if we should resume from the latest checkpoint
                latest_checkpoint = existing_checkpoints[0]
                resume_decision = self._should_resume_from_checkpoint(latest_checkpoint, config)
                
                if resume_decision["should_resume"]:
                    setup_results["resuming_from_checkpoint"] = True
                    setup_results["resumed_checkpoint_id"] = latest_checkpoint.checkpoint_id
                    setup_results["warnings"].append(f"Resuming from checkpoint: {latest_checkpoint.checkpoint_id}")
                    logger.info(f"Will resume training from checkpoint: {latest_checkpoint.checkpoint_id}")
                else:
                    setup_results["warnings"].append(f"Not resuming: {resume_decision['reason']}")
                    logger.info(f"Not resuming from checkpoint: {resume_decision['reason']}")
            
            # Validate checkpoint directory and disk space
            free_space = self._get_free_disk_space()
            if free_space < self.checkpoint_config.min_disk_space_mb:
                setup_results["warnings"].append(f"Low disk space: {free_space:.1f}MB")
                
                # Attempt cleanup
                cleanup_result = self.checkpoint_manager.cleanup_old_checkpoints()
                if cleanup_result["removed_count"] > 0:
                    setup_results["warnings"].append(f"Cleaned up {cleanup_result['removed_count']} old checkpoints")
            
            setup_results["success"] = True
            logger.info("Checkpoint management setup completed successfully")
            
        except Exception as e:
            error_msg = f"Failed to setup checkpoint management: {e}"
            setup_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return setup_results
    
    def save_training_checkpoint(self,
                               model_state: Dict[str, Any],
                               optimizer_state: Dict[str, Any],
                               step: int,
                               epoch: int,
                               loss: float,
                               accuracy: float,
                               learning_rate: float,
                               config: EmailTrainingConfig,
                               training_time: float = 0.0,
                               force: bool = False) -> CheckpointSaveResult:
        """
        Save training checkpoint with email-specific metadata.
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            accuracy: Current accuracy
            learning_rate: Current learning rate
            config: Email training configuration
            training_time: Total training time in seconds
            force: Force saving regardless of conditions
            
        Returns:
            CheckpointSaveResult
        """
        logger.info(f"Saving checkpoint at step {step}, epoch {epoch} (loss: {loss:.4f}, acc: {accuracy:.4f})")
        
        # Add email-specific metadata to model state
        enhanced_model_state = model_state.copy()
        enhanced_model_state['email_training_metadata'] = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate,
            'training_id': self.current_training_id,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'EmailTRM',
            'num_email_categories': config.num_email_categories,
            'vocab_size': config.vocab_size,
            'target_accuracy': config.target_accuracy
        }
        
        # Convert EmailTrainingConfig to MacBookTrainingConfig format for compatibility
        from macbook_optimization.config_management import MacBookTrainingConfig
        
        macbook_config = MacBookTrainingConfig(
            # Model parameters
            model_name=config.model_name,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs,
            
            # Hardware optimization
            memory_limit_mb=config.memory_limit_mb,
            enable_memory_monitoring=config.enable_memory_monitoring,
            dynamic_batch_sizing=config.dynamic_batch_sizing,
            use_cpu_optimization=config.use_cpu_optimization,
            num_workers=config.num_workers,
            
            # Hardware summary (would be populated by hardware detector)
            hardware_summary={
                'model_type': 'EmailTRM',
                'training_type': 'email_classification',
                'vocab_size': config.vocab_size,
                'num_categories': config.num_email_categories
            }
        )
        
        # Save checkpoint using the underlying checkpoint manager
        save_result = self.checkpoint_manager.save_checkpoint(
            model_state=enhanced_model_state,
            optimizer_state=optimizer_state,
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            training_config=macbook_config,
            training_time_seconds=training_time,
            force=force
        )
        
        if save_result.success:
            logger.info(f"✅ Checkpoint saved successfully: {save_result.checkpoint_id}")
            logger.info(f"   Size: {save_result.disk_usage_mb:.1f}MB, "
                       f"Time: {save_result.save_time_seconds:.2f}s, "
                       f"Memory: {save_result.memory_usage_mb:.1f}MB")
        else:
            logger.error(f"❌ Checkpoint save failed: {save_result.errors}")
        
        return save_result
    
    def load_training_checkpoint(self, 
                               checkpoint_id: Optional[str] = None,
                               config: Optional[EmailTrainingConfig] = None) -> CheckpointLoadResult:
        """
        Load training checkpoint for resumption.
        
        Args:
            checkpoint_id: Specific checkpoint ID to load (latest if None)
            config: Current training configuration for validation
            
        Returns:
            CheckpointLoadResult
        """
        logger.info(f"Loading checkpoint: {checkpoint_id or 'latest'}")
        
        # Convert config for compatibility if provided
        macbook_config = None
        if config:
            from macbook_optimization.config_management import MacBookTrainingConfig
            macbook_config = MacBookTrainingConfig(
                model_name=config.model_name,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                max_epochs=config.max_epochs,
                memory_limit_mb=config.memory_limit_mb,
                enable_memory_monitoring=config.enable_memory_monitoring,
                dynamic_batch_sizing=config.dynamic_batch_sizing,
                use_cpu_optimization=config.use_cpu_optimization,
                num_workers=config.num_workers,
                hardware_summary={
                    'model_type': 'EmailTRM',
                    'training_type': 'email_classification'
                }
            )
        
        # Load checkpoint using the underlying checkpoint manager
        load_result = self.checkpoint_manager.load_checkpoint(
            checkpoint_id=checkpoint_id,
            validate_config=True,
            current_config=macbook_config
        )
        
        if load_result.success:
            logger.info(f"✅ Checkpoint loaded successfully: {load_result.checkpoint_id}")
            
            # Extract email-specific metadata if available
            if (load_result.model_state and 
                'email_training_metadata' in load_result.model_state):
                
                email_metadata = load_result.model_state['email_training_metadata']
                logger.info(f"   Step: {email_metadata.get('step', 'unknown')}, "
                           f"Epoch: {email_metadata.get('epoch', 'unknown')}, "
                           f"Accuracy: {email_metadata.get('accuracy', 'unknown')}")
        else:
            logger.error(f"❌ Checkpoint load failed: {load_result.errors}")
        
        return load_result
    
    def resume_training_from_checkpoint(self, 
                                      checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to resume from (latest if None)
            
        Returns:
            Resume operation results
        """
        logger.info("Attempting to resume training from checkpoint...")
        
        resume_results = {
            "success": False,
            "checkpoint_id": None,
            "resumed_step": 0,
            "resumed_epoch": 0,
            "resumed_accuracy": 0.0,
            "resumed_loss": 0.0,
            "model_state": None,
            "optimizer_state": None,
            "training_metadata": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Load the checkpoint
            load_result = self.load_training_checkpoint(checkpoint_id)
            
            if not load_result.success:
                resume_results["errors"].extend(load_result.errors)
                resume_results["warnings"].extend(load_result.warnings)
                return resume_results
            
            # Extract training state
            resume_results["checkpoint_id"] = load_result.checkpoint_id
            resume_results["model_state"] = load_result.model_state
            resume_results["optimizer_state"] = load_result.optimizer_state
            
            # Extract email-specific metadata
            if (load_result.model_state and 
                'email_training_metadata' in load_result.model_state):
                
                email_metadata = load_result.model_state['email_training_metadata']
                resume_results["training_metadata"] = email_metadata
                resume_results["resumed_step"] = email_metadata.get('step', 0)
                resume_results["resumed_epoch"] = email_metadata.get('epoch', 0)
                resume_results["resumed_accuracy"] = email_metadata.get('accuracy', 0.0)
                resume_results["resumed_loss"] = email_metadata.get('loss', 0.0)
            
            # Configuration compatibility warnings
            if load_result.warnings:
                resume_results["warnings"].extend(load_result.warnings)
            
            if not load_result.config_compatible:
                resume_results["warnings"].append("Configuration mismatch detected - proceeding with compatibility mode")
            
            resume_results["success"] = True
            
            logger.info(f"✅ Training resumption prepared from checkpoint: {load_result.checkpoint_id}")
            logger.info(f"   Resuming from step {resume_results['resumed_step']}, "
                       f"epoch {resume_results['resumed_epoch']}")
            logger.info(f"   Previous accuracy: {resume_results['resumed_accuracy']:.4f}, "
                       f"loss: {resume_results['resumed_loss']:.4f}")
            
        except Exception as e:
            error_msg = f"Failed to resume training from checkpoint: {e}"
            resume_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return resume_results
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint during interruption."""
        try:
            if not self.current_training_id:
                return
            
            logger.info("Saving emergency checkpoint...")
            
            # This would need to be called with actual model and optimizer states
            # In a real implementation, this would be integrated with the training loop
            emergency_checkpoint_id = f"emergency_{self.current_training_id}_{datetime.now().strftime('%H%M%S')}"
            
            # Create emergency checkpoint marker
            emergency_file = self.checkpoint_dir / f"{emergency_checkpoint_id}_emergency.json"
            emergency_data = {
                "training_id": self.current_training_id,
                "timestamp": datetime.now().isoformat(),
                "interruption_reason": "signal_received",
                "message": "Training was interrupted - emergency checkpoint created"
            }
            
            with open(emergency_file, 'w') as f:
                json.dump(emergency_data, f, indent=2)
            
            self.interruption_checkpoint_id = emergency_checkpoint_id
            logger.info(f"Emergency checkpoint marker saved: {emergency_checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def _should_resume_from_checkpoint(self, 
                                     checkpoint_metadata: CheckpointMetadata,
                                     current_config: EmailTrainingConfig) -> Dict[str, Any]:
        """
        Determine if training should resume from a checkpoint.
        
        Args:
            checkpoint_metadata: Metadata of the checkpoint to consider
            current_config: Current training configuration
            
        Returns:
            Decision dictionary with should_resume flag and reason
        """
        decision = {
            "should_resume": False,
            "reason": "",
            "compatibility_score": 0.0
        }
        
        try:
            # Check if checkpoint is recent (within last 24 hours)
            checkpoint_age = datetime.now() - checkpoint_metadata.timestamp
            if checkpoint_age > timedelta(hours=24):
                decision["reason"] = f"Checkpoint too old: {checkpoint_age}"
                return decision
            
            # Check basic compatibility
            saved_config = checkpoint_metadata.training_config
            
            # Compare key configuration parameters
            compatibility_checks = {
                "model_name": saved_config.get("model_name") == current_config.model_name,
                "vocab_size": abs(saved_config.get("vocab_size", 0) - current_config.vocab_size) < 1000,
                "num_categories": saved_config.get("num_email_categories", 10) == current_config.num_email_categories,
                "target_accuracy": abs(saved_config.get("target_accuracy", 0.95) - current_config.target_accuracy) < 0.05
            }
            
            compatible_count = sum(compatibility_checks.values())
            compatibility_score = compatible_count / len(compatibility_checks)
            decision["compatibility_score"] = compatibility_score
            
            if compatibility_score < 0.7:  # Require 70% compatibility
                decision["reason"] = f"Low compatibility score: {compatibility_score:.2f}"
                return decision
            
            # Check if checkpoint represents meaningful progress
            if checkpoint_metadata.step < 100:  # Less than 100 steps
                decision["reason"] = "Checkpoint has insufficient training progress"
                return decision
            
            # All checks passed
            decision["should_resume"] = True
            decision["reason"] = f"Compatible checkpoint with {compatibility_score:.2f} compatibility score"
            
        except Exception as e:
            decision["reason"] = f"Error evaluating checkpoint: {e}"
        
        return decision
    
    def _get_free_disk_space(self) -> float:
        """Get free disk space in MB."""
        import shutil
        free_bytes = shutil.disk_usage(self.checkpoint_dir).free
        return free_bytes / (1024**2)
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint management status."""
        
        # Get checkpoint manager summary
        manager_summary = self.checkpoint_manager.get_checkpoint_summary()
        
        # Get recent checkpoints
        recent_checkpoints = self.checkpoint_manager.list_checkpoints(limit=5)
        
        status = {
            "checkpoint_management_active": self.current_training_id is not None,
            "current_training_id": self.current_training_id,
            "training_interrupted": self.training_interrupted,
            "interruption_checkpoint_id": self.interruption_checkpoint_id,
            "checkpoint_directory": str(self.checkpoint_dir),
            "manager_summary": manager_summary,
            "recent_checkpoints": [
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "timestamp": cp.timestamp.isoformat(),
                    "step": cp.step,
                    "epoch": cp.epoch,
                    "loss": cp.loss,
                    "size_mb": cp.disk_usage_mb
                }
                for cp in recent_checkpoints
            ],
            "disk_usage": {
                "total_checkpoint_size_mb": manager_summary["total_size_mb"],
                "free_disk_space_mb": manager_summary["free_disk_space_mb"],
                "checkpoint_count": manager_summary["checkpoint_count"]
            },
            "configuration": {
                "max_checkpoints": self.checkpoint_config.max_checkpoints,
                "save_interval_steps": self.checkpoint_config.save_interval_steps,
                "save_interval_minutes": self.checkpoint_config.save_interval_minutes,
                "auto_cleanup": self.checkpoint_config.auto_cleanup,
                "memory_aware_saving": self.checkpoint_config.memory_aware_intervals
            }
        }
        
        return status
    
    def cleanup_checkpoints(self, keep_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Manually trigger checkpoint cleanup.
        
        Args:
            keep_count: Number of checkpoints to keep (uses config default if None)
            
        Returns:
            Cleanup results
        """
        logger.info("Manually triggering checkpoint cleanup...")
        
        if keep_count is not None:
            # Temporarily override max_checkpoints
            original_max = self.checkpoint_config.max_checkpoints
            self.checkpoint_config.max_checkpoints = keep_count
        
        try:
            cleanup_result = self.checkpoint_manager.cleanup_old_checkpoints()
            
            logger.info(f"Cleanup completed: removed {cleanup_result['removed_count']} checkpoints, "
                       f"freed {cleanup_result['freed_space_mb']:.1f}MB")
            
            return cleanup_result
            
        finally:
            if keep_count is not None:
                # Restore original max_checkpoints
                self.checkpoint_config.max_checkpoints = original_max
    
    def stop_checkpoint_management(self) -> Dict[str, Any]:
        """Stop checkpoint management and return final summary."""
        
        logger.info("Stopping checkpoint management...")
        
        final_summary = {
            "training_id": self.current_training_id,
            "training_interrupted": self.training_interrupted,
            "final_status": self.get_checkpoint_status(),
            "cleanup_performed": False
        }
        
        # Perform final cleanup if configured
        if self.checkpoint_config.auto_cleanup:
            cleanup_result = self.checkpoint_manager.cleanup_old_checkpoints()
            final_summary["cleanup_performed"] = True
            final_summary["cleanup_result"] = cleanup_result
        
        # Reset state
        self.current_training_id = None
        self.training_interrupted = False
        self.interruption_checkpoint_id = None
        
        logger.info("Checkpoint management stopped")
        
        return final_summary


def main():
    """Main function to demonstrate checkpoint management implementation."""
    
    logger.info("Starting checkpoint management demonstration...")
    
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
            output_dir="checkpoint_demo_orchestrator_output"
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
        
        # Step 2: Initialize checkpoint manager
        logger.info("Initializing checkpoint management...")
        checkpoint_manager = EmailTrainingCheckpointManager(
            checkpoint_dir="email_training_checkpoints_demo",
            orchestrator=setup_manager.orchestrator
        )
        
        # Step 3: Start checkpoint management
        training_id = f"demo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = setup_manager.production_config
        
        start_results = checkpoint_manager.start_checkpoint_management(training_id, config)
        
        if start_results["success"]:
            logger.info("✅ Checkpoint management started successfully")
            
            if start_results["resuming_from_checkpoint"]:
                logger.info(f"🔄 Will resume from checkpoint: {start_results['resumed_checkpoint_id']}")
            
            if start_results["warnings"]:
                for warning in start_results["warnings"]:
                    logger.warning(f"  - {warning}")
        else:
            logger.error("❌ Failed to start checkpoint management")
            return
        
        # Step 4: Demonstrate checkpoint saving
        logger.info("Demonstrating checkpoint saving...")
        
        # Simulate training state
        mock_model_state = {
            "layer1.weight": "mock_tensor_data",
            "layer1.bias": "mock_tensor_data",
            "layer2.weight": "mock_tensor_data",
            "layer2.bias": "mock_tensor_data"
        }
        
        mock_optimizer_state = {
            "state": {"param_groups": []},
            "param_groups": [{"lr": 1e-4, "weight_decay": 0.01}]
        }
        
        # Save multiple checkpoints to demonstrate interval saving
        for step in [500, 1000, 1500, 2000]:
            epoch = step // 500
            loss = 2.0 - (step / 2000) * 1.5  # Decreasing loss
            accuracy = 0.5 + (step / 2000) * 0.45  # Increasing accuracy
            
            save_result = checkpoint_manager.save_training_checkpoint(
                model_state=mock_model_state,
                optimizer_state=mock_optimizer_state,
                step=step,
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                learning_rate=1e-4 * (0.95 ** epoch),
                config=config,
                training_time=step * 0.1,  # Mock training time
                force=False
            )
            
            if save_result.success:
                logger.info(f"✅ Checkpoint saved at step {step}: {save_result.checkpoint_id}")
            else:
                logger.error(f"❌ Failed to save checkpoint at step {step}")
            
            # Simulate training delay
            time.sleep(1)
        
        # Step 5: Demonstrate checkpoint loading and resumption
        logger.info("Demonstrating checkpoint loading and resumption...")
        
        # Load the latest checkpoint
        load_result = checkpoint_manager.load_training_checkpoint()
        
        if load_result.success:
            logger.info(f"✅ Latest checkpoint loaded: {load_result.checkpoint_id}")
        else:
            logger.error("❌ Failed to load latest checkpoint")
        
        # Demonstrate resumption
        resume_result = checkpoint_manager.resume_training_from_checkpoint()
        
        if resume_result["success"]:
            logger.info("✅ Training resumption prepared successfully")
            logger.info(f"   Resume from step: {resume_result['resumed_step']}")
            logger.info(f"   Resume from epoch: {resume_result['resumed_epoch']}")
            logger.info(f"   Previous accuracy: {resume_result['resumed_accuracy']:.4f}")
        else:
            logger.error("❌ Failed to prepare training resumption")
        
        # Step 6: Demonstrate checkpoint status and cleanup
        logger.info("Demonstrating checkpoint status and cleanup...")
        
        status = checkpoint_manager.get_checkpoint_status()
        logger.info("📊 Checkpoint Status:")
        logger.info(f"  Active training: {status['checkpoint_management_active']}")
        logger.info(f"  Checkpoint count: {status['disk_usage']['checkpoint_count']}")
        logger.info(f"  Total size: {status['disk_usage']['total_checkpoint_size_mb']:.1f}MB")
        logger.info(f"  Free disk space: {status['disk_usage']['free_disk_space_mb']:.1f}MB")
        
        # Demonstrate cleanup
        cleanup_result = checkpoint_manager.cleanup_checkpoints(keep_count=3)
        logger.info(f"🧹 Cleanup: removed {cleanup_result['removed_count']} checkpoints, "
                   f"freed {cleanup_result['freed_space_mb']:.1f}MB")
        
        # Step 7: Stop checkpoint management
        final_summary = checkpoint_manager.stop_checkpoint_management()
        
        logger.info("✅ Checkpoint management demonstration completed!")
        logger.info(f"Final training ID: {final_summary['training_id']}")
        logger.info(f"Training interrupted: {final_summary['training_interrupted']}")
        logger.info(f"Cleanup performed: {final_summary['cleanup_performed']}")
        
        logger.info("🎯 Checkpoint management is ready for production email training!")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Checkpoint management demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()