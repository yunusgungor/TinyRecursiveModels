#!/usr/bin/env python3
"""
Setup EmailTrainingOrchestrator for Real Email Classification Training

This script implements task 4.1: Setup EmailTrainingOrchestrator
- Initialize real EmailTrainingOrchestrator with production configuration
- Configure training environment with real dataset and model
- Setup progress monitoring and checkpoint management

Requirements: 4.1, 4.2
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from macbook_optimization.email_training_orchestrator import (
    EmailTrainingOrchestrator, 
    TrainingResult,
    HyperparameterOptimizer,
    HyperparameterSearchSpace
)
from macbook_optimization.email_training_config import EmailTrainingConfig
from macbook_optimization.hardware_detection import HardwareDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_training_orchestrator_setup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class EmailTrainingOrchestratorSetup:
    """
    Setup and configuration manager for EmailTrainingOrchestrator.
    
    Implements task 4.1: Setup EmailTrainingOrchestrator with production configuration.
    """
    
    def __init__(self, output_dir: str = "email_training_output"):
        """
        Initialize orchestrator setup.
        
        Args:
            output_dir: Directory for training outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.orchestrator: Optional[EmailTrainingOrchestrator] = None
        self.hardware_detector = HardwareDetector()
        self.production_config: Optional[EmailTrainingConfig] = None
        
        logger.info(f"EmailTrainingOrchestratorSetup initialized with output dir: {output_dir}")
    
    def setup_production_orchestrator(self, 
                                    dataset_path: str,
                                    enable_monitoring: bool = True,
                                    enable_checkpointing: bool = True) -> Dict[str, Any]:
        """
        Setup EmailTrainingOrchestrator with production configuration.
        
        Args:
            dataset_path: Path to production email dataset
            enable_monitoring: Enable resource monitoring
            enable_checkpointing: Enable checkpoint management
            
        Returns:
            Setup results dictionary
        """
        logger.info("Setting up EmailTrainingOrchestrator for production training...")
        
        setup_results = {
            "success": False,
            "orchestrator_initialized": False,
            "environment_setup": None,
            "production_config": None,
            "hardware_specs": None,
            "dataset_validation": None,
            "monitoring_enabled": enable_monitoring,
            "checkpointing_enabled": enable_checkpointing,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Step 1: Initialize EmailTrainingOrchestrator
            logger.info("Initializing EmailTrainingOrchestrator...")
            self.orchestrator = EmailTrainingOrchestrator(
                output_dir=str(self.output_dir),
                enable_monitoring=enable_monitoring,
                enable_checkpointing=enable_checkpointing
            )
            setup_results["orchestrator_initialized"] = True
            logger.info("EmailTrainingOrchestrator initialized successfully")
            
            # Step 2: Create production configuration
            logger.info("Creating production email training configuration...")
            self.production_config = self._create_production_config()
            setup_results["production_config"] = {
                "model_name": self.production_config.model_name,
                "vocab_size": self.production_config.vocab_size,
                "hidden_size": self.production_config.hidden_size,
                "num_layers": self.production_config.num_layers,
                "batch_size": self.production_config.batch_size,
                "learning_rate": self.production_config.learning_rate,
                "target_accuracy": self.production_config.target_accuracy,
                "use_hierarchical_attention": self.production_config.use_hierarchical_attention,
                "enable_subject_prioritization": self.production_config.enable_subject_prioritization
            }
            logger.info(f"Production config created: {self.production_config.model_name} model")
            
            # Step 3: Setup training environment
            logger.info("Setting up training environment...")
            env_setup = self.orchestrator.setup_training_environment(
                dataset_path=dataset_path,
                base_config=self.production_config
            )
            setup_results["environment_setup"] = env_setup
            
            if not env_setup["success"]:
                setup_results["errors"].extend(env_setup["errors"])
                setup_results["warnings"].extend(env_setup["warnings"])
                logger.error("Training environment setup failed")
                return setup_results
            
            setup_results["hardware_specs"] = env_setup["hardware_specs"]
            setup_results["dataset_validation"] = env_setup["dataset_info"]
            setup_results["warnings"].extend(env_setup["warnings"])
            
            logger.info("Training environment setup completed successfully")
            logger.info(f"Hardware: {env_setup['hardware_specs']['cpu_cores']} cores, "
                       f"{env_setup['hardware_specs']['memory_gb']:.1f}GB memory")
            logger.info(f"Dataset: {env_setup['dataset_info']['total_emails']} emails, "
                       f"{env_setup['dataset_info']['total_size_mb']:.1f}MB")
            
            # Step 4: Validate orchestrator components
            logger.info("Validating orchestrator components...")
            component_validation = self._validate_orchestrator_components()
            setup_results.update(component_validation)
            
            if component_validation["warnings"]:
                setup_results["warnings"].extend(component_validation["warnings"])
            
            setup_results["success"] = True
            logger.info("EmailTrainingOrchestrator setup completed successfully!")
            
        except Exception as e:
            error_msg = f"EmailTrainingOrchestrator setup failed: {e}"
            setup_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return setup_results
    
    def _create_production_config(self) -> EmailTrainingConfig:
        """Create production-ready email training configuration."""
        
        # Detect hardware capabilities
        hardware_summary = self.hardware_detector.get_hardware_summary()
        memory_gb = hardware_summary["memory"]["total_gb"]
        cpu_cores = hardware_summary["cpu"]["cores"]
        
        logger.info(f"Detected hardware: {cpu_cores} cores, {memory_gb:.1f}GB memory")
        
        # Create configuration optimized for detected hardware
        if memory_gb >= 24:  # High-end MacBook
            config = EmailTrainingConfig(
                # Model parameters
                model_name="EmailTRM",
                vocab_size=8000,
                hidden_size=512,
                num_layers=3,
                num_email_categories=10,
                
                # Training parameters
                batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                weight_decay=0.01,
                max_epochs=10,
                max_steps=10000,
                
                # Email-specific parameters
                max_sequence_length=512,
                use_email_structure=True,
                subject_attention_weight=2.5,
                pooling_strategy="weighted",
                
                # Email features
                enable_subject_prioritization=True,
                enable_sender_analysis=True,
                enable_content_features=True,
                use_hierarchical_attention=True,
                
                # MacBook optimization
                memory_limit_mb=int(memory_gb * 1024 * 0.7),  # 70% of total memory
                enable_memory_monitoring=True,
                dynamic_batch_sizing=True,
                use_cpu_optimization=True,
                num_workers=min(cpu_cores, 4),
                
                # Performance targets
                target_accuracy=0.95,
                min_category_accuracy=0.90,
                early_stopping_patience=5,
                
                # Email dataset parameters
                email_augmentation_prob=0.3,
                category_balancing=True
            )
        elif memory_gb >= 12:  # Mid-range MacBook
            config = EmailTrainingConfig(
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
                
                # MacBook optimization
                memory_limit_mb=int(memory_gb * 1024 * 0.7),
                enable_memory_monitoring=True,
                dynamic_batch_sizing=True,
                use_cpu_optimization=True,
                num_workers=min(cpu_cores, 3),
                
                # Performance targets
                target_accuracy=0.95,
                min_category_accuracy=0.90,
                early_stopping_patience=5,
                
                # Email dataset parameters
                email_augmentation_prob=0.3,
                category_balancing=True
            )
        else:  # Low-memory MacBook
            config = EmailTrainingConfig(
                # Model parameters
                model_name="EmailTRM",
                vocab_size=3000,
                hidden_size=256,
                num_layers=2,
                num_email_categories=10,
                
                # Training parameters
                batch_size=2,
                gradient_accumulation_steps=16,
                learning_rate=1e-4,
                weight_decay=0.01,
                max_epochs=10,
                max_steps=10000,
                
                # Email-specific parameters
                max_sequence_length=256,
                use_email_structure=True,
                subject_attention_weight=1.5,
                pooling_strategy="weighted",
                
                # Email features (reduced for memory)
                enable_subject_prioritization=True,
                enable_sender_analysis=False,
                enable_content_features=False,
                use_hierarchical_attention=False,
                
                # MacBook optimization
                memory_limit_mb=int(memory_gb * 1024 * 0.6),  # 60% for low memory
                enable_memory_monitoring=True,
                dynamic_batch_sizing=True,
                use_cpu_optimization=True,
                num_workers=min(cpu_cores, 2),
                
                # Performance targets
                target_accuracy=0.95,
                min_category_accuracy=0.90,
                early_stopping_patience=7,  # More patience for smaller batches
                
                # Email dataset parameters
                email_augmentation_prob=0.2,
                category_balancing=True
            )
        
        logger.info(f"Created production config for {memory_gb:.1f}GB MacBook")
        logger.info(f"Config: batch_size={config.batch_size}, hidden_size={config.hidden_size}, "
                   f"vocab_size={config.vocab_size}")
        
        return config
    
    def _validate_orchestrator_components(self) -> Dict[str, Any]:
        """Validate that all orchestrator components are properly initialized."""
        
        validation_results = {
            "hardware_detector_available": False,
            "memory_manager_available": False,
            "resource_monitor_available": False,
            "dataset_manager_available": False,
            "progress_monitor_available": False,
            "checkpoint_manager_available": False,
            "config_adapter_available": False,
            "warnings": []
        }
        
        if not self.orchestrator:
            validation_results["warnings"].append("Orchestrator not initialized")
            return validation_results
        
        # Check hardware detector
        if hasattr(self.orchestrator, 'hardware_detector') and self.orchestrator.hardware_detector:
            validation_results["hardware_detector_available"] = True
            logger.info("Hardware detector: Available")
        else:
            validation_results["warnings"].append("Hardware detector not available")
        
        # Check memory manager
        if hasattr(self.orchestrator, 'memory_manager') and self.orchestrator.memory_manager:
            validation_results["memory_manager_available"] = True
            logger.info("Memory manager: Available")
        else:
            validation_results["warnings"].append("Memory manager not available")
        
        # Check resource monitor
        if hasattr(self.orchestrator, 'resource_monitor') and self.orchestrator.resource_monitor:
            validation_results["resource_monitor_available"] = True
            logger.info("Resource monitor: Available")
        else:
            validation_results["warnings"].append("Resource monitor not available")
        
        # Check dataset manager
        if hasattr(self.orchestrator, 'dataset_manager') and self.orchestrator.dataset_manager:
            validation_results["dataset_manager_available"] = True
            logger.info("Dataset manager: Available")
        else:
            validation_results["warnings"].append("Dataset manager not available")
        
        # Check progress monitor
        if hasattr(self.orchestrator, 'progress_monitor') and self.orchestrator.progress_monitor:
            validation_results["progress_monitor_available"] = True
            logger.info("Progress monitor: Available")
        else:
            validation_results["warnings"].append("Progress monitor not available")
        
        # Check checkpoint manager
        if hasattr(self.orchestrator, 'checkpoint_manager') and self.orchestrator.checkpoint_manager:
            validation_results["checkpoint_manager_available"] = True
            logger.info("Checkpoint manager: Available")
        else:
            validation_results["warnings"].append("Checkpoint manager not available")
        
        # Check config adapter
        if hasattr(self.orchestrator, 'config_adapter') and self.orchestrator.config_adapter:
            validation_results["config_adapter_available"] = True
            logger.info("Config adapter: Available")
        else:
            validation_results["warnings"].append("Config adapter not available")
        
        return validation_results
    
    def setup_hyperparameter_optimization(self, 
                                        search_space: Optional[HyperparameterSearchSpace] = None,
                                        optimization_strategy: str = "bayesian") -> Dict[str, Any]:
        """
        Setup hyperparameter optimization for the orchestrator.
        
        Args:
            search_space: Hyperparameter search space
            optimization_strategy: Optimization strategy
            
        Returns:
            Setup results
        """
        logger.info("Setting up hyperparameter optimization...")
        
        if not self.orchestrator:
            return {
                "success": False,
                "error": "Orchestrator not initialized"
            }
        
        # Create default search space if not provided
        if search_space is None:
            search_space = HyperparameterSearchSpace(
                hidden_size=[256, 384, 512],
                num_layers=[2, 3],
                learning_rate=[5e-5, 1e-4, 2e-4],
                batch_size=[2, 4, 8],
                weight_decay=[0.01, 0.05, 0.1],
                gradient_accumulation_steps=[4, 8, 16],
                subject_attention_weight=[1.5, 2.0, 2.5],
                pooling_strategy=["weighted", "attention"],
                warmup_steps=[100, 200, 500]
            )
        
        # Initialize hyperparameter optimizer
        hyperparameter_optimizer = HyperparameterOptimizer(
            orchestrator=self.orchestrator,
            search_space=search_space,
            optimization_strategy=optimization_strategy
        )
        
        logger.info(f"Hyperparameter optimizer initialized with {optimization_strategy} strategy")
        
        return {
            "success": True,
            "optimizer": hyperparameter_optimizer,
            "search_space": search_space,
            "strategy": optimization_strategy
        }
    
    def get_orchestrator_summary(self) -> Dict[str, Any]:
        """Get summary of orchestrator setup and configuration."""
        
        if not self.orchestrator:
            return {"error": "Orchestrator not initialized"}
        
        # Get hardware specs
        hardware_specs = self.hardware_detector.get_hardware_summary()
        
        summary = {
            "orchestrator_initialized": True,
            "output_directory": str(self.output_dir),
            "hardware_specs": {
                "cpu_brand": hardware_specs["cpu"]["brand"],
                "cpu_cores": hardware_specs["cpu"]["cores"],
                "memory_gb": hardware_specs["memory"]["total_gb"],
                "available_memory_gb": hardware_specs["memory"]["available_gb"],
                "platform": hardware_specs["platform"]["os_version"]
            },
            "production_config": {
                "model_name": self.production_config.model_name if self.production_config else None,
                "vocab_size": self.production_config.vocab_size if self.production_config else None,
                "hidden_size": self.production_config.hidden_size if self.production_config else None,
                "batch_size": self.production_config.batch_size if self.production_config else None,
                "target_accuracy": self.production_config.target_accuracy if self.production_config else None
            },
            "components": {
                "monitoring_enabled": hasattr(self.orchestrator, 'progress_monitor') and self.orchestrator.progress_monitor is not None,
                "checkpointing_enabled": hasattr(self.orchestrator, 'checkpoint_manager') and self.orchestrator.checkpoint_manager is not None,
                "memory_management": hasattr(self.orchestrator, 'memory_manager') and self.orchestrator.memory_manager is not None,
                "resource_monitoring": hasattr(self.orchestrator, 'resource_monitor') and self.orchestrator.resource_monitor is not None
            },
            "training_history": len(self.orchestrator.training_history) if hasattr(self.orchestrator, 'training_history') else 0
        }
        
        return summary


def main():
    """Main function to demonstrate EmailTrainingOrchestrator setup."""
    
    logger.info("Starting EmailTrainingOrchestrator setup demonstration...")
    
    # Check if dataset path is provided
    dataset_path = "data/production-emails"  # Default production dataset path
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        logger.info("Please provide a valid dataset path as argument or ensure data/production-emails exists")
        return
    
    # Initialize setup manager
    setup_manager = EmailTrainingOrchestratorSetup(
        output_dir="email_training_orchestrator_output"
    )
    
    # Setup production orchestrator
    logger.info(f"Setting up orchestrator with dataset: {dataset_path}")
    setup_results = setup_manager.setup_production_orchestrator(
        dataset_path=dataset_path,
        enable_monitoring=True,
        enable_checkpointing=True
    )
    
    # Display results
    if setup_results["success"]:
        logger.info("✅ EmailTrainingOrchestrator setup completed successfully!")
        
        # Display hardware specs
        if setup_results["hardware_specs"]:
            hw = setup_results["hardware_specs"]
            logger.info(f"Hardware: {hw['cpu_cores']} cores, {hw['memory_gb']:.1f}GB memory")
        
        # Display dataset info
        if setup_results["dataset_validation"]:
            ds = setup_results["dataset_validation"]
            logger.info(f"Dataset: {ds['total_emails']} emails, {ds['total_size_mb']:.1f}MB")
        
        # Display production config
        if setup_results["production_config"]:
            cfg = setup_results["production_config"]
            logger.info(f"Model: {cfg['model_name']}, Hidden: {cfg['hidden_size']}, "
                       f"Batch: {cfg['batch_size']}, Target: {cfg['target_accuracy']}")
        
        # Display warnings
        if setup_results["warnings"]:
            logger.warning(f"Setup completed with {len(setup_results['warnings'])} warnings:")
            for warning in setup_results["warnings"]:
                logger.warning(f"  - {warning}")
        
        # Setup hyperparameter optimization
        logger.info("Setting up hyperparameter optimization...")
        hp_setup = setup_manager.setup_hyperparameter_optimization(
            optimization_strategy="bayesian"
        )
        
        if hp_setup["success"]:
            logger.info("✅ Hyperparameter optimization setup completed!")
        else:
            logger.error(f"❌ Hyperparameter optimization setup failed: {hp_setup.get('error')}")
        
        # Get orchestrator summary
        summary = setup_manager.get_orchestrator_summary()
        logger.info("📊 Orchestrator Summary:")
        logger.info(f"  Output Directory: {summary['output_directory']}")
        logger.info(f"  Monitoring Enabled: {summary['components']['monitoring_enabled']}")
        logger.info(f"  Checkpointing Enabled: {summary['components']['checkpointing_enabled']}")
        logger.info(f"  Memory Management: {summary['components']['memory_management']}")
        
        logger.info("🎯 EmailTrainingOrchestrator is ready for multi-phase training!")
        
    else:
        logger.error("❌ EmailTrainingOrchestrator setup failed!")
        
        if setup_results["errors"]:
            logger.error("Errors:")
            for error in setup_results["errors"]:
                logger.error(f"  - {error}")
        
        if setup_results["warnings"]:
            logger.warning("Warnings:")
            for warning in setup_results["warnings"]:
                logger.warning(f"  - {warning}")


if __name__ == "__main__":
    main()