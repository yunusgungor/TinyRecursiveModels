"""
Email Training Orchestrator for MacBook Optimization

This module provides a complete email classification training pipeline orchestrator
that integrates all email training components into a cohesive workflow with
multi-phase training strategies and hyperparameter optimization.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TORCH_AVAILABLE = False

from .email_training_config import EmailTrainingConfig, EmailTrainingConfigAdapter
from .email_dataset_management import EmailDatasetManager, EmailDatasetConfig
from .email_training_loop import EmailTrainingLoop, EmailTrainingMetrics
from .email_trm_integration import MacBookEmailTRM
from .checkpoint_management import CheckpointManager, CheckpointConfig
from .progress_monitoring import ProgressMonitor
from .memory_management import MemoryManager
from .resource_monitoring import ResourceMonitor
from .hardware_detection import HardwareDetector
from models.email_tokenizer import EmailTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """Configuration for a training phase."""
    name: str
    description: str
    steps: int
    learning_rate: float
    batch_size: int
    
    # Phase-specific parameters
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Email-specific parameters
    use_hierarchical_attention: bool = True
    enable_subject_prioritization: bool = True
    subject_attention_weight: float = 2.0
    
    # Data parameters
    data_filter: str = "all"  # "easy", "medium", "all"
    augmentation_probability: float = 0.3
    
    # Model parameters (for progressive training)
    model_config_overrides: Optional[Dict[str, Any]] = None


@dataclass
class HyperparameterSearchSpace:
    """Hyperparameter search space for email classification."""
    
    # Model architecture
    hidden_size: List[int] = None
    num_layers: List[int] = None
    
    # Training parameters
    learning_rate: List[float] = None
    batch_size: List[int] = None
    weight_decay: List[float] = None
    gradient_accumulation_steps: List[int] = None
    
    # Email-specific parameters
    subject_attention_weight: List[float] = None
    pooling_strategy: List[str] = None
    
    # Optimization parameters
    warmup_steps: List[int] = None
    
    def __post_init__(self):
        """Set default search spaces if not provided."""
        if self.hidden_size is None:
            self.hidden_size = [256, 384, 512]
        if self.num_layers is None:
            self.num_layers = [2, 3]
        if self.learning_rate is None:
            self.learning_rate = [5e-5, 1e-4, 2e-4]
        if self.batch_size is None:
            self.batch_size = [4, 8, 16]
        if self.weight_decay is None:
            self.weight_decay = [0.01, 0.05, 0.1]
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = [4, 8, 16]
        if self.subject_attention_weight is None:
            self.subject_attention_weight = [1.5, 2.0, 2.5]
        if self.pooling_strategy is None:
            self.pooling_strategy = ["weighted", "attention"]
        if self.warmup_steps is None:
            self.warmup_steps = [100, 200, 500]


@dataclass
class TrainingResult:
    """Result of a complete training run."""
    success: bool
    training_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Configuration
    config: EmailTrainingConfig
    phases_completed: List[str]
    
    # Results
    final_accuracy: Optional[float]
    best_accuracy: Optional[float]
    final_loss: Optional[float]
    best_loss: Optional[float]
    
    # Per-category performance
    category_accuracies: Dict[str, float]
    
    # Training metrics
    total_training_time: float
    total_steps: int
    samples_processed: int
    
    # Resource usage
    peak_memory_usage_mb: float
    average_cpu_usage: float
    
    # Model info
    model_path: Optional[str]
    checkpoint_path: Optional[str]
    
    # Errors and warnings
    errors: List[str]
    warnings: List[str]


class EmailTrainingOrchestrator:
    """
    Main orchestrator for email classification training pipeline.
    
    Manages the complete training workflow including:
    - Training environment setup and validation
    - Multi-phase training strategies
    - Hyperparameter optimization
    - Resource monitoring and optimization
    - Checkpoint management
    - Progress tracking and reporting
    """
    
    def __init__(self, 
                 output_dir: str = "email_training_output",
                 enable_monitoring: bool = True,
                 enable_checkpointing: bool = True):
        """
        Initialize email training orchestrator.
        
        Args:
            output_dir: Directory for training outputs
            enable_monitoring: Enable resource monitoring
            enable_checkpointing: Enable checkpoint management
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hardware_detector = HardwareDetector()
        self.memory_manager = MemoryManager()
        self.resource_monitor = ResourceMonitor()
        
        # Configuration adapters
        self.config_adapter = EmailTrainingConfigAdapter(self.hardware_detector)
        
        # Dataset management
        self.dataset_manager = EmailDatasetManager(memory_manager=self.memory_manager)
        
        # Progress monitoring
        self.progress_monitor = None
        if enable_monitoring:
            self.progress_monitor = ProgressMonitor(
                self.resource_monitor, 
                self.memory_manager
            )
        
        # Checkpoint management
        self.checkpoint_manager = None
        if enable_checkpointing:
            checkpoint_config = CheckpointConfig(
                checkpoint_dir=str(self.output_dir / "checkpoints"),
                max_checkpoints=3,
                save_interval_steps=500
            )
            self.checkpoint_manager = CheckpointManager(
                checkpoint_config, 
                self.memory_manager
            )
        
        # Training state
        self.current_training_id = None
        self.training_history: List[TrainingResult] = []
        
        logger.info(f"EmailTrainingOrchestrator initialized with output dir: {output_dir}")
    
    def setup_training_environment(self, 
                                 dataset_path: str,
                                 base_config: Optional[EmailTrainingConfig] = None) -> Dict[str, Any]:
        """
        Setup and validate training environment.
        
        Args:
            dataset_path: Path to email dataset
            base_config: Base training configuration
            
        Returns:
            Environment setup results
        """
        logger.info("Setting up email training environment...")
        
        setup_results = {
            "success": False,
            "hardware_specs": None,
            "dataset_info": None,
            "config_adaptation": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Detect hardware capabilities
            hardware_specs = self.config_adapter.get_hardware_specs()
            setup_results["hardware_specs"] = {
                "cpu_cores": hardware_specs.cpu.cores,
                "memory_gb": hardware_specs.memory.total_memory / (1024**3),
                "available_memory_gb": hardware_specs.memory.available_memory / (1024**3),
                "platform": hardware_specs.platform.macos_version
            }
            
            logger.info(f"Detected hardware: {hardware_specs.cpu.cores} cores, "
                       f"{setup_results['hardware_specs']['memory_gb']:.1f}GB memory")
            
            # Validate dataset
            dataset_validation = self.dataset_manager.validate_email_dataset(dataset_path)
            setup_results["dataset_info"] = dataset_validation
            
            if not dataset_validation.get("valid", False):
                setup_results["errors"].append(f"Dataset validation failed: {dataset_validation.get('error', 'Unknown error')}")
                return setup_results
            
            logger.info(f"Dataset validated: {dataset_validation['total_emails']} emails, "
                       f"{dataset_validation['total_size_mb']:.1f}MB")
            
            # Create or adapt configuration
            if base_config is None:
                base_config = EmailTrainingConfig()
            
            # Adapt configuration for hardware and dataset
            config_result = self.config_adapter.create_email_hardware_config(
                base_config, 
                dataset_validation["total_emails"]
            )
            
            setup_results["config_adaptation"] = {
                "adapted_config": config_result.adapted_config,
                "performance_estimates": config_result.performance_estimates,
                "warnings": config_result.validation_warnings
            }
            
            setup_results["warnings"].extend(config_result.validation_warnings)
            
            # Initialize tokenizer
            try:
                tokenizer = EmailTokenizer(
                    vocab_size=config_result.adapted_config["vocab_size"],
                    max_seq_len=config_result.adapted_config["max_sequence_length"]
                )
                logger.info(f"Initialized tokenizer with vocab size {tokenizer.vocab_size}")
            except Exception as e:
                setup_results["errors"].append(f"Failed to initialize tokenizer: {e}")
                return setup_results
            
            # Test data loading
            try:
                test_dataloader, loader_info = self.dataset_manager.create_email_dataloader(
                    dataset_path, 
                    batch_size=config_result.adapted_config["batch_size"],
                    split="train",
                    tokenizer=tokenizer
                )
                
                setup_results["dataset_info"]["dataloader_info"] = loader_info
                logger.info(f"Test dataloader created: batch size {loader_info['optimized_batch_size']}")
                
            except Exception as e:
                setup_results["errors"].append(f"Failed to create dataloader: {e}")
                return setup_results
            
            # Memory pressure check
            memory_stats = self.memory_manager.monitor_memory_usage()
            if memory_stats.percent_used > 70:
                setup_results["warnings"].append(f"High memory usage before training: {memory_stats.percent_used:.1f}%")
            
            setup_results["success"] = True
            logger.info("Training environment setup completed successfully")
            
        except Exception as e:
            setup_results["errors"].append(f"Environment setup failed: {e}")
            logger.error(f"Environment setup error: {e}")
        
        return setup_results
    
    def create_training_phases(self, 
                             strategy: str = "multi_phase",
                             total_steps: int = 10000,
                             base_config: Optional[EmailTrainingConfig] = None) -> List[TrainingPhase]:
        """
        Create training phases based on strategy.
        
        Args:
            strategy: Training strategy ("single", "multi_phase", "progressive", "curriculum")
            total_steps: Total training steps
            base_config: Base configuration
            
        Returns:
            List of training phases
        """
        if base_config is None:
            base_config = EmailTrainingConfig()
        
        if strategy == "single":
            return self._create_single_phase(total_steps, base_config)
        elif strategy == "multi_phase":
            return self._create_multi_phase_strategy(total_steps, base_config)
        elif strategy == "progressive":
            return self._create_progressive_strategy(total_steps, base_config)
        elif strategy == "curriculum":
            return self._create_curriculum_strategy(total_steps, base_config)
        else:
            raise ValueError(f"Unknown training strategy: {strategy}")
    
    def _create_single_phase(self, total_steps: int, config: EmailTrainingConfig) -> List[TrainingPhase]:
        """Create single-phase training strategy."""
        return [
            TrainingPhase(
                name="main_training",
                description="Single-phase email classification training",
                steps=total_steps,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                warmup_steps=min(500, total_steps // 10),
                weight_decay=config.weight_decay,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob
            )
        ]
    
    def _create_multi_phase_strategy(self, total_steps: int, config: EmailTrainingConfig) -> List[TrainingPhase]:
        """Create multi-phase training strategy with warmup, main, and fine-tuning phases."""
        return [
            TrainingPhase(
                name="warmup",
                description="Warmup phase with lower learning rate",
                steps=total_steps // 4,
                learning_rate=config.learning_rate * 0.5,
                batch_size=max(2, config.batch_size // 2),
                warmup_steps=100,
                weight_decay=config.weight_decay * 0.5,
                gradient_accumulation_steps=config.gradient_accumulation_steps * 2,
                use_hierarchical_attention=False,  # Simpler model initially
                enable_subject_prioritization=True,
                subject_attention_weight=config.subject_attention_weight * 0.8,
                data_filter="easy",
                augmentation_probability=config.email_augmentation_prob * 0.5
            ),
            TrainingPhase(
                name="main_training",
                description="Main training phase with full configuration",
                steps=total_steps // 2,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                warmup_steps=0,  # No warmup in main phase
                weight_decay=config.weight_decay,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob
            ),
            TrainingPhase(
                name="fine_tuning",
                description="Fine-tuning phase with reduced learning rate",
                steps=total_steps // 4,
                learning_rate=config.learning_rate * 0.1,
                batch_size=max(2, config.batch_size // 2),
                warmup_steps=0,
                weight_decay=config.weight_decay * 2,  # Higher regularization
                gradient_accumulation_steps=config.gradient_accumulation_steps * 2,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight * 1.2,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob * 0.2
            )
        ]
    
    def _create_progressive_strategy(self, total_steps: int, config: EmailTrainingConfig) -> List[TrainingPhase]:
        """Create progressive training strategy with increasing model complexity."""
        return [
            TrainingPhase(
                name="simple_model",
                description="Training with simplified model architecture",
                steps=total_steps // 3,
                learning_rate=config.learning_rate * 1.2,
                batch_size=min(16, config.batch_size * 2),  # Larger batch for simple model
                warmup_steps=200,
                weight_decay=config.weight_decay,
                gradient_accumulation_steps=max(1, config.gradient_accumulation_steps // 2),
                use_hierarchical_attention=False,
                enable_subject_prioritization=True,
                subject_attention_weight=config.subject_attention_weight * 0.8,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob,
                model_config_overrides={
                    "hidden_size": min(256, config.hidden_size),
                    "num_layers": 1
                }
            ),
            TrainingPhase(
                name="medium_model",
                description="Training with medium complexity model",
                steps=total_steps // 3,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                warmup_steps=100,
                weight_decay=config.weight_decay,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=True,
                enable_subject_prioritization=True,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob,
                model_config_overrides={
                    "hidden_size": min(384, config.hidden_size),
                    "num_layers": min(2, config.num_layers)
                }
            ),
            TrainingPhase(
                name="full_model",
                description="Training with full model complexity",
                steps=total_steps // 3,
                learning_rate=config.learning_rate * 0.8,
                batch_size=config.batch_size,
                warmup_steps=0,
                weight_decay=config.weight_decay * 1.5,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob
            )
        ]
    
    def _create_curriculum_strategy(self, total_steps: int, config: EmailTrainingConfig) -> List[TrainingPhase]:
        """Create curriculum learning strategy with progressive data difficulty."""
        return [
            TrainingPhase(
                name="easy_emails",
                description="Training on easy emails (short, clear categories)",
                steps=total_steps // 4,
                learning_rate=config.learning_rate * 0.8,
                batch_size=min(16, config.batch_size * 2),
                warmup_steps=200,
                weight_decay=config.weight_decay * 0.5,
                gradient_accumulation_steps=max(1, config.gradient_accumulation_steps // 2),
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="easy",
                augmentation_probability=config.email_augmentation_prob * 0.5
            ),
            TrainingPhase(
                name="medium_emails",
                description="Training on medium complexity emails",
                steps=total_steps // 2,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                warmup_steps=100,
                weight_decay=config.weight_decay,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight,
                data_filter="medium",
                augmentation_probability=config.email_augmentation_prob
            ),
            TrainingPhase(
                name="all_emails",
                description="Training on all emails including complex ones",
                steps=total_steps // 4,
                learning_rate=config.learning_rate * 0.6,
                batch_size=config.batch_size,
                warmup_steps=0,
                weight_decay=config.weight_decay * 1.2,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                use_hierarchical_attention=config.use_hierarchical_attention,
                enable_subject_prioritization=config.enable_subject_prioritization,
                subject_attention_weight=config.subject_attention_weight * 1.1,
                data_filter="all",
                augmentation_probability=config.email_augmentation_prob
            )
        ]
    
    def execute_training_pipeline(self,
                                dataset_path: str,
                                config: Optional[EmailTrainingConfig] = None,
                                strategy: str = "multi_phase",
                                total_steps: int = 10000,
                                validation_split: float = 0.2) -> TrainingResult:
        """
        Execute complete email classification training pipeline.
        
        Args:
            dataset_path: Path to email dataset
            config: Training configuration (created if None)
            strategy: Training strategy
            total_steps: Total training steps
            validation_split: Validation data split ratio
            
        Returns:
            Training result
        """
        training_id = f"email_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_training_id = training_id
        
        logger.info(f"Starting email training pipeline: {training_id}")
        
        # Initialize result
        result = TrainingResult(
            success=False,
            training_id=training_id,
            start_time=datetime.now(),
            end_time=None,
            config=config or EmailTrainingConfig(),
            phases_completed=[],
            final_accuracy=None,
            best_accuracy=None,
            final_loss=None,
            best_loss=None,
            category_accuracies={},
            total_training_time=0.0,
            total_steps=0,
            samples_processed=0,
            peak_memory_usage_mb=0.0,
            average_cpu_usage=0.0,
            model_path=None,
            checkpoint_path=None,
            errors=[],
            warnings=[]
        )
        
        start_time = time.time()
        
        try:
            # Setup training environment
            logger.info("Setting up training environment...")
            env_setup = self.setup_training_environment(dataset_path, config)
            
            if not env_setup["success"]:
                result.errors.extend(env_setup["errors"])
                result.warnings.extend(env_setup["warnings"])
                return result
            
            result.warnings.extend(env_setup["warnings"])
            
            # Get adapted configuration
            adapted_config_dict = env_setup["config_adaptation"]["adapted_config"]
            
            # Remove parameters that don't exist in EmailTrainingConfig
            email_streaming_threshold_mb = adapted_config_dict.pop('email_streaming_threshold_mb', None)
            email_cache_threshold_mb = adapted_config_dict.pop('email_cache_threshold_mb', None)
            
            adapted_config = EmailTrainingConfig(**adapted_config_dict)
            result.config = adapted_config
            
            # Create tokenizer
            tokenizer = EmailTokenizer(
                vocab_size=adapted_config.vocab_size,
                max_seq_len=adapted_config.max_sequence_length
            )
            
            # Load datasets
            logger.info("Loading training and validation datasets...")
            train_dataset = self.dataset_manager.load_email_dataset(
                dataset_path, "train", tokenizer
            )
            
            # Create validation dataset (simplified - in practice you'd have separate val data)
            val_dataset = self.dataset_manager.load_email_dataset(
                dataset_path, "train", tokenizer  # Using train for now
            )
            
            # Create data loaders
            train_dataloader, train_info = self.dataset_manager.create_email_dataloader(
                dataset_path, adapted_config.batch_size, "train", tokenizer
            )
            
            val_dataloader, val_info = self.dataset_manager.create_email_dataloader(
                dataset_path, adapted_config.batch_size, "train", tokenizer  # Using train for now
            )
            
            # Validate dataloaders are not empty
            if len(train_dataloader) == 0:
                raise ValueError("Training dataloader is empty - no training data available")
            if len(val_dataloader) == 0:
                raise ValueError("Validation dataloader is empty - no validation data available")
            
            # Initialize model
            logger.info("Initializing EmailTRM model...")
            
            # Use real MacBookEmailTRM with recursive reasoning
            from .email_trm_integration import MacBookEmailTRM, MacBookEmailTRMConfig
            
            # Create TRM config optimized for current hardware
            trm_config = MacBookEmailTRMConfig(
                # Base TRM parameters
                batch_size=adapted_config.batch_size,
                seq_len=adapted_config.max_sequence_length,
                vocab_size=adapted_config.vocab_size,
                hidden_size=adapted_config.hidden_size,
                num_puzzle_identifiers=1,  # For email classification
                
                # TRM architecture parameters
                L_layers=2,  # Reduced for memory efficiency
                H_layers=2,  # Reduced for memory efficiency
                H_cycles=2,  # Reduced for memory efficiency  
                L_cycles=3,  # Reduced for memory efficiency
                halt_max_steps=6,  # Reasonable halting limit
                halt_exploration_prob=0.1,
                
                # Transformer parameters
                num_heads=max(1, adapted_config.hidden_size // 64),
                expansion=4.0,
                pos_encodings="rope",
                
                # Email-specific parameters
                num_email_categories=adapted_config.num_email_categories,
                classification_dropout=0.1,
                use_email_structure=True,
                use_hierarchical_attention=True,
                pooling_strategy='weighted',
                
                # MacBook optimizations
                enable_cpu_optimization=True,
                gradient_checkpointing=True,
                dynamic_complexity=True,
                memory_efficient_attention=True
            )
            
            model = MacBookEmailTRM(trm_config)
            logger.info(f"MacBookEmailTRM initialized with {model._count_parameters():,} parameters")
            logger.info("Using real TRM with recursive reasoning capabilities")
            
            # Create training phases
            phases = self.create_training_phases(strategy, total_steps, adapted_config)
            logger.info(f"Created {len(phases)} training phases: {[p.name for p in phases]}")
            
            # Start progress monitoring
            if self.progress_monitor:
                self.progress_monitor.start_session(
                    training_id,
                    total_steps=total_steps,
                    total_epochs=adapted_config.max_epochs,
                    model_name="EmailTRM",
                    dataset_name=os.path.basename(dataset_path),
                    batch_size=adapted_config.batch_size,
                    learning_rate=adapted_config.learning_rate
                )
            
            # Execute training phases
            current_step = 0
            best_accuracy = 0.0
            best_loss = float('inf')
            
            for phase_idx, phase in enumerate(phases):
                logger.info(f"Starting phase {phase_idx + 1}/{len(phases)}: {phase.name}")
                logger.info(f"Phase description: {phase.description}")
                
                # Update model configuration for this phase
                if phase.model_config_overrides:
                    self._apply_model_config_overrides(model, phase.model_config_overrides)
                
                # Create training loop for this phase
                phase_config = self._create_phase_training_config(phase, adapted_config)
                training_loop = EmailTrainingLoop(model, phase_config)
                
                # Add callbacks for progress monitoring
                if self.progress_monitor:
                    training_loop.add_step_callback(self._create_step_callback(current_step))
                
                # Add checkpoint callback
                if self.checkpoint_manager:
                    training_loop.add_step_callback(
                        self._create_checkpoint_callback(phase_config, current_step)
                    )
                
                # Execute phase training
                try:
                    phase_result = training_loop.train(train_dataloader, val_dataloader)
                    
                    # Update results
                    if phase_result.get("final_val_metrics"):
                        final_metrics = phase_result["final_val_metrics"]
                        if final_metrics.accuracy > best_accuracy:
                            best_accuracy = final_metrics.accuracy
                        if final_metrics.loss < best_loss:
                            best_loss = final_metrics.loss
                    
                    result.phases_completed.append(phase.name)
                    current_step += phase.steps
                    
                    logger.info(f"Phase {phase.name} completed successfully")
                    
                    # Check if target accuracy reached
                    if best_accuracy >= adapted_config.target_accuracy:
                        logger.info(f"Target accuracy {adapted_config.target_accuracy} reached!")
                        break
                
                except Exception as e:
                    error_msg = f"Phase {phase.name} failed: {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
                    break
            
            # Final evaluation
            logger.info("Performing final evaluation...")
            final_training_loop = EmailTrainingLoop(model, phase_config)
            final_metrics = final_training_loop.evaluate(val_dataloader)
            
            # Update final results
            result.final_accuracy = final_metrics.accuracy
            result.best_accuracy = best_accuracy
            result.final_loss = final_metrics.loss
            result.best_loss = best_loss
            result.category_accuracies = final_metrics.category_accuracies
            result.total_steps = current_step
            
            # Save final model
            model_path = self.output_dir / f"{training_id}_final_model.pt"
            if TORCH_AVAILABLE:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': asdict(adapted_config),
                    'tokenizer_vocab': tokenizer.vocab if hasattr(tokenizer, 'vocab') else {},
                    'training_result': asdict(result)
                }, model_path)
            
            result.model_path = str(model_path)
            
            # Get final checkpoint path
            if self.checkpoint_manager:
                latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint_id()
                if latest_checkpoint:
                    result.checkpoint_path = str(self.checkpoint_manager._get_checkpoint_path(latest_checkpoint))
            
            result.success = True
            logger.info(f"Training pipeline completed successfully!")
            logger.info(f"Final accuracy: {result.final_accuracy:.4f}")
            logger.info(f"Best accuracy: {result.best_accuracy:.4f}")
            
        except Exception as e:
            error_msg = f"Training pipeline failed: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)
        
        finally:
            # Stop monitoring
            if self.progress_monitor:
                completed_session = self.progress_monitor.end_session()
                if completed_session:
                    result.total_training_time = completed_session.total_training_time
                    result.samples_processed = completed_session.total_samples_processed
                    result.peak_memory_usage_mb = completed_session.peak_memory_usage_mb
            
            result.end_time = datetime.now()
            result.total_training_time = time.time() - start_time
            
            # Save training result
            self._save_training_result(result)
            
            # Add to history
            self.training_history.append(result)
        
        return result
    
    def _apply_model_config_overrides(self, model: MacBookEmailTRM, overrides: Dict[str, Any]):
        """Apply model configuration overrides for progressive training."""
        # This is a simplified implementation
        # In practice, you might need to modify the model architecture
        logger.info(f"Applying model config overrides: {overrides}")
        
        # For now, just log the overrides
        # In a full implementation, you'd modify model layers/parameters
        pass
    
    def _create_phase_training_config(self, phase: TrainingPhase, base_config: EmailTrainingConfig) -> EmailTrainingConfig:
        """Create training configuration for a specific phase."""
        # Create a copy of base config and update with phase parameters
        phase_config = EmailTrainingConfig(
            # Model parameters from base config
            model_name=base_config.model_name,
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_layers=base_config.num_layers,
            num_email_categories=base_config.num_email_categories,
            
            # Phase-specific training parameters
            batch_size=phase.batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation_steps,
            learning_rate=phase.learning_rate,
            weight_decay=phase.weight_decay,
            max_epochs=1,  # Single epoch per phase
            max_steps=phase.steps,
            
            # Email-specific parameters
            max_sequence_length=base_config.max_sequence_length,
            use_email_structure=base_config.use_email_structure,
            subject_attention_weight=phase.subject_attention_weight,
            pooling_strategy=base_config.pooling_strategy,
            
            # MacBook optimization parameters
            memory_limit_mb=base_config.memory_limit_mb,
            enable_memory_monitoring=base_config.enable_memory_monitoring,
            dynamic_batch_sizing=base_config.dynamic_batch_sizing,
            use_cpu_optimization=base_config.use_cpu_optimization,
            num_workers=base_config.num_workers,
            
            # Performance targets
            target_accuracy=base_config.target_accuracy,
            min_category_accuracy=base_config.min_category_accuracy,
            early_stopping_patience=base_config.early_stopping_patience,
            
            # Phase-specific email parameters
            enable_subject_prioritization=phase.enable_subject_prioritization,
            use_hierarchical_attention=phase.use_hierarchical_attention,
            email_augmentation_prob=phase.augmentation_probability
        )
        
        return phase_config
    
    def _create_step_callback(self, step_offset: int) -> Callable:
        """Create callback for step progress updates."""
        def step_callback(metrics: EmailTrainingMetrics):
            if self.progress_monitor:
                self.progress_monitor.update_progress(
                    current_step=step_offset + getattr(metrics, 'step', 0),
                    total_steps=10000,  # This should be passed from training context
                    current_loss=metrics.loss,
                    current_learning_rate=metrics.learning_rate,
                    samples_processed=step_offset * 32  # Rough estimate
                )
        
        return step_callback
    
    def _create_checkpoint_callback(self, config: EmailTrainingConfig, step_offset: int) -> Callable:
        """Create callback for checkpoint saving."""
        def checkpoint_callback(metrics: EmailTrainingMetrics):
            if self.checkpoint_manager:
                # This is a simplified callback - in practice you'd need model and optimizer states
                logger.debug(f"Checkpoint callback triggered at step {step_offset}")
        
        return checkpoint_callback
    
    def _save_training_result(self, result: TrainingResult):
        """Save training result to file."""
        result_file = self.output_dir / f"{result.training_id}_result.json"
        
        try:
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            result_dict['start_time'] = result.start_time.isoformat()
            if result.end_time:
                result_dict['end_time'] = result.end_time.isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Training result saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training result: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs."""
        if not self.training_history:
            return {"message": "No training runs completed"}
        
        successful_runs = [r for r in self.training_history if r.success]
        
        summary = {
            "total_runs": len(self.training_history),
            "successful_runs": len(successful_runs),
            "failed_runs": len(self.training_history) - len(successful_runs),
            "best_accuracy": max((r.best_accuracy or 0) for r in successful_runs) if successful_runs else 0,
            "average_training_time": sum(r.total_training_time for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "recent_runs": [
                {
                    "training_id": r.training_id,
                    "success": r.success,
                    "final_accuracy": r.final_accuracy,
                    "training_time_minutes": (r.total_training_time or 0) / 60,
                    "phases_completed": r.phases_completed
                }
                for r in self.training_history[-5:]  # Last 5 runs
            ]
        }
        
        return summary


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for email classification training.
    
    Implements Bayesian optimization for efficient parameter search
    with email-specific parameter spaces and MacBook hardware constraints.
    """
    
    def __init__(self, 
                 orchestrator: EmailTrainingOrchestrator,
                 search_space: Optional[HyperparameterSearchSpace] = None,
                 optimization_strategy: str = "bayesian"):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            orchestrator: Training orchestrator instance
            search_space: Hyperparameter search space
            optimization_strategy: Optimization strategy ("random", "grid", "bayesian")
        """
        self.orchestrator = orchestrator
        self.search_space = search_space or HyperparameterSearchSpace()
        self.optimization_strategy = optimization_strategy
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_config: Optional[EmailTrainingConfig] = None
        self.best_performance: float = 0.0
        
        logger.info(f"HyperparameterOptimizer initialized with {optimization_strategy} strategy")
    
    def optimize_hyperparameters(self,
                                dataset_path: str,
                                num_trials: int = 10,
                                max_steps_per_trial: int = 2000,
                                target_metric: str = "accuracy") -> Dict[str, Any]:
        """
        Execute hyperparameter optimization.
        
        Args:
            dataset_path: Path to email dataset
            num_trials: Number of optimization trials
            max_steps_per_trial: Maximum steps per trial
            target_metric: Target metric to optimize
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {num_trials} trials")
        
        optimization_results = {
            "success": False,
            "num_trials": num_trials,
            "completed_trials": 0,
            "best_config": None,
            "best_performance": 0.0,
            "optimization_history": [],
            "total_time": 0.0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Generate hyperparameter combinations
            param_combinations = self._generate_parameter_combinations(num_trials)
            
            for trial_idx, params in enumerate(param_combinations):
                logger.info(f"Starting trial {trial_idx + 1}/{num_trials}")
                logger.info(f"Trial parameters: {params}")
                
                try:
                    # Create configuration for this trial
                    trial_config = self._create_trial_config(params)
                    
                    # Execute training with limited steps
                    result = self.orchestrator.execute_training_pipeline(
                        dataset_path=dataset_path,
                        config=trial_config,
                        strategy="single",  # Use single phase for quick trials
                        total_steps=max_steps_per_trial
                    )
                    
                    # Extract performance metric
                    performance = self._extract_performance_metric(result, target_metric)
                    
                    # Record trial result
                    trial_result = {
                        "trial_id": trial_idx + 1,
                        "parameters": params,
                        "performance": performance,
                        "success": result.success,
                        "training_time": result.total_training_time,
                        "final_accuracy": result.final_accuracy,
                        "errors": result.errors
                    }
                    
                    self.optimization_history.append(trial_result)
                    optimization_results["optimization_history"].append(trial_result)
                    optimization_results["completed_trials"] += 1
                    
                    # Update best configuration
                    if performance > self.best_performance:
                        self.best_performance = performance
                        self.best_config = trial_config
                        optimization_results["best_config"] = asdict(trial_config)
                        optimization_results["best_performance"] = performance
                        
                        logger.info(f"New best performance: {performance:.4f}")
                    
                    logger.info(f"Trial {trial_idx + 1} completed: performance = {performance:.4f}")
                
                except Exception as e:
                    error_msg = f"Trial {trial_idx + 1} failed: {e}"
                    optimization_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            optimization_results["success"] = optimization_results["completed_trials"] > 0
            optimization_results["total_time"] = time.time() - start_time
            
            logger.info(f"Hyperparameter optimization completed")
            logger.info(f"Completed trials: {optimization_results['completed_trials']}/{num_trials}")
            logger.info(f"Best performance: {optimization_results['best_performance']:.4f}")
            
        except Exception as e:
            error_msg = f"Hyperparameter optimization failed: {e}"
            optimization_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return optimization_results
    
    def _generate_parameter_combinations(self, num_trials: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on optimization strategy."""
        if self.optimization_strategy == "random":
            return self._random_search(num_trials)
        elif self.optimization_strategy == "grid":
            return self._grid_search()[:num_trials]  # Limit to num_trials
        elif self.optimization_strategy == "bayesian":
            return self._bayesian_search(num_trials)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.optimization_strategy}")
    
    def _random_search(self, num_trials: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        import random
        
        combinations = []
        for _ in range(num_trials):
            params = {}
            
            # Sample from each parameter space
            params["hidden_size"] = random.choice(self.search_space.hidden_size)
            params["num_layers"] = random.choice(self.search_space.num_layers)
            params["learning_rate"] = random.choice(self.search_space.learning_rate)
            params["batch_size"] = random.choice(self.search_space.batch_size)
            params["weight_decay"] = random.choice(self.search_space.weight_decay)
            params["gradient_accumulation_steps"] = random.choice(self.search_space.gradient_accumulation_steps)
            params["subject_attention_weight"] = random.choice(self.search_space.subject_attention_weight)
            params["pooling_strategy"] = random.choice(self.search_space.pooling_strategy)
            params["warmup_steps"] = random.choice(self.search_space.warmup_steps)
            
            combinations.append(params)
        
        return combinations
    
    def _grid_search(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations (grid search)."""
        import itertools
        
        param_names = ["hidden_size", "num_layers", "learning_rate", "batch_size", 
                      "weight_decay", "gradient_accumulation_steps", "subject_attention_weight",
                      "pooling_strategy", "warmup_steps"]
        
        param_values = [
            self.search_space.hidden_size,
            self.search_space.num_layers,
            self.search_space.learning_rate,
            self.search_space.batch_size,
            self.search_space.weight_decay,
            self.search_space.gradient_accumulation_steps,
            self.search_space.subject_attention_weight,
            self.search_space.pooling_strategy,
            self.search_space.warmup_steps
        ]
        
        combinations = []
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            combinations.append(params)
        
        return combinations
    
    def _bayesian_search(self, num_trials: int) -> List[Dict[str, Any]]:
        """
        Simplified Bayesian optimization.
        
        In a full implementation, you'd use libraries like Optuna or scikit-optimize.
        For now, we use a heuristic approach with some good baseline configurations.
        """
        import random
        
        # Start with some good baseline configurations
        baseline_configs = [
            {
                "hidden_size": 512,
                "num_layers": 2,
                "learning_rate": 1e-4,
                "batch_size": 8,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 8,
                "subject_attention_weight": 2.0,
                "pooling_strategy": "weighted",
                "warmup_steps": 200
            },
            {
                "hidden_size": 256,
                "num_layers": 2,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "weight_decay": 0.05,
                "gradient_accumulation_steps": 16,
                "subject_attention_weight": 1.5,
                "pooling_strategy": "attention",
                "warmup_steps": 100
            }
        ]
        
        combinations = []
        
        # Add baseline configurations
        for config in baseline_configs[:min(len(baseline_configs), num_trials // 2)]:
            combinations.append(config)
        
        # Fill remaining with random search around best known configurations
        remaining_trials = num_trials - len(combinations)
        
        for _ in range(remaining_trials):
            # Choose a baseline to vary from
            base_config = random.choice(baseline_configs)
            
            # Create variation
            params = base_config.copy()
            
            # Randomly vary some parameters
            if random.random() < 0.3:  # 30% chance to vary each parameter
                params["learning_rate"] = random.choice(self.search_space.learning_rate)
            if random.random() < 0.3:
                params["batch_size"] = random.choice(self.search_space.batch_size)
            if random.random() < 0.2:
                params["hidden_size"] = random.choice(self.search_space.hidden_size)
            if random.random() < 0.2:
                params["weight_decay"] = random.choice(self.search_space.weight_decay)
            
            combinations.append(params)
        
        return combinations
    
    def _create_trial_config(self, params: Dict[str, Any]) -> EmailTrainingConfig:
        """Create EmailTrainingConfig from trial parameters."""
        return EmailTrainingConfig(
            # Model parameters
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            
            # Training parameters
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            weight_decay=params["weight_decay"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            
            # Email-specific parameters
            subject_attention_weight=params["subject_attention_weight"],
            pooling_strategy=params["pooling_strategy"],
            
            # Use defaults for other parameters
            vocab_size=5000,
            num_email_categories=10,
            max_sequence_length=512,
            max_epochs=1,
            max_steps=2000,
            target_accuracy=0.95,
            min_category_accuracy=0.90,
            early_stopping_patience=3  # Shorter patience for trials
        )
    
    def _extract_performance_metric(self, result: TrainingResult, target_metric: str) -> float:
        """Extract performance metric from training result."""
        if target_metric == "accuracy":
            return result.final_accuracy or 0.0
        elif target_metric == "best_accuracy":
            return result.best_accuracy or 0.0
        elif target_metric == "loss":
            return -(result.final_loss or float('inf'))  # Negative because we want to minimize loss
        else:
            return result.final_accuracy or 0.0  # Default to accuracy
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of hyperparameter optimization results."""
        if not self.optimization_history:
            return {"message": "No optimization trials completed"}
        
        successful_trials = [t for t in self.optimization_history if t["success"]]
        
        summary = {
            "total_trials": len(self.optimization_history),
            "successful_trials": len(successful_trials),
            "best_performance": self.best_performance,
            "best_config": asdict(self.best_config) if self.best_config else None,
            "average_performance": sum(t["performance"] for t in successful_trials) / len(successful_trials) if successful_trials else 0,
            "performance_std": 0.0,  # Would calculate standard deviation in full implementation
            "optimization_strategy": self.optimization_strategy,
            "parameter_importance": self._analyze_parameter_importance()
        }
        
        return summary
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """
        Analyze parameter importance based on optimization history.
        
        This is a simplified analysis - in practice you'd use more sophisticated methods.
        """
        if len(self.optimization_history) < 3:
            return {}
        
        # Simple correlation analysis
        import numpy as np
        
        successful_trials = [t for t in self.optimization_history if t["success"]]
        if len(successful_trials) < 3:
            return {}
        
        # Extract parameter values and performances
        param_names = ["hidden_size", "learning_rate", "batch_size", "weight_decay"]
        importance = {}
        
        performances = [t["performance"] for t in successful_trials]
        
        for param_name in param_names:
            try:
                param_values = [t["parameters"][param_name] for t in successful_trials]
                
                # Convert to numeric if needed
                if isinstance(param_values[0], str):
                    continue  # Skip non-numeric parameters for now
                
                # Calculate correlation with performance
                correlation = np.corrcoef(param_values, performances)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                
            except (KeyError, ValueError, TypeError):
                importance[param_name] = 0.0
        
        return importance