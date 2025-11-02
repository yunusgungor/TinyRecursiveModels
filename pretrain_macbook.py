"""
MacBook-optimized TRM training script.

This script provides a MacBook-specific version of the TRM training pipeline
with automatic hardware detection, configuration adaptation, and resource
monitoring optimized for Intel-based MacBook hardware.
"""

import os
import sys
import time
import math
import yaml
import shutil
import copy
import signal
from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

# Try to import AdamATan2, fallback to AdamW if not available
try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Warning: adam_atan2 not available, using torch.optim.AdamW as fallback")
    AdamATan2 = torch.optim.AdamW

# Import original training components
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# Import MacBook optimization modules
from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager, MemoryConfig
from macbook_optimization.cpu_optimization import CPUOptimizer, TensorOperationOptimizer
from macbook_optimization.resource_monitoring import ResourceMonitor
from macbook_optimization.training_config_adapter import TrainingConfigAdapter, ConfigurationResult
from macbook_optimization.config_validation import ConfigurationValidator
from macbook_optimization.dataset_management import (
    DatasetManager, DatasetManagementConfig, create_memory_efficient_dataloader
)
from macbook_optimization.checkpoint_management import (
    CheckpointManager, CheckpointConfig, CheckpointMetadata
)

# Import original config classes
from pretrain import (
    LossConfig, ArchConfig, EvaluatorConfig, PretrainConfig, TrainState,
    cosine_schedule_with_warmup_lr_lambda, save_train_state, load_checkpoint,
    compute_lr, create_evaluators, evaluate, save_code_and_config
)


@dataclass
class MacBookTrainingState:
    """Extended training state with MacBook-specific monitoring."""
    # Original training state
    train_state: TrainState
    
    # MacBook-specific components
    hardware_detector: HardwareDetector
    memory_manager: MemoryManager
    cpu_optimizer: CPUOptimizer
    resource_monitor: ResourceMonitor
    checkpoint_manager: CheckpointManager
    
    # Configuration and validation
    config_adapter: TrainingConfigAdapter
    validator: ConfigurationValidator
    original_config: Dict[str, Any]
    adapted_config: Dict[str, Any]
    
    # Monitoring state
    start_time: float
    last_checkpoint_time: float
    training_metrics: Dict[str, List[float]]
    resource_history: List[Dict[str, Any]]
    
    # Performance tracking
    samples_processed: int
    total_training_time: float
    average_samples_per_second: float


class MacBookTRMTrainer:
    """MacBook-optimized TRM trainer with automatic configuration and monitoring."""
    
    def __init__(self, config: PretrainConfig):
        """
        Initialize MacBook TRM trainer.
        
        Args:
            config: Base training configuration
        """
        self.base_config = config
        self.training_state: Optional[MacBookTrainingState] = None
        
        # Initialize MacBook optimization components
        self.hardware_detector = HardwareDetector()
        self.memory_manager = MemoryManager()
        self.cpu_optimizer = CPUOptimizer(self.hardware_detector)
        self.tensor_optimizer = TensorOperationOptimizer(self.cpu_optimizer)
        self.resource_monitor = ResourceMonitor()
        self.config_adapter = TrainingConfigAdapter(self.hardware_detector)
        self.validator = ConfigurationValidator(self.hardware_detector)
        
        # Initialize dataset management
        dataset_config = DatasetManagementConfig(
            max_dataset_memory_mb=800.0,  # Conservative for MacBook
            streaming_threshold_mb=400.0,
            cache_threshold_mb=200.0,
            chunk_size_mb=50.0,
            enable_caching=True,
            auto_fallback_streaming=True
        )
        self.dataset_manager = DatasetManager(dataset_config, self.memory_manager)
        
        # Initialize checkpoint management
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=config.checkpoint_path or "checkpoints",
            max_checkpoints=3,
            save_interval_steps=500,
            save_interval_minutes=30.0,
            memory_aware_intervals=True,
            memory_threshold_for_saving=70.0,
            auto_cleanup=True,
            validate_on_load=True
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config, self.memory_manager
        )
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._shutdown_requested = False
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def setup_hardware_optimization(self) -> Dict[str, Any]:
        """
        Set up hardware-specific optimizations.
        
        Returns:
            Hardware optimization summary
        """
        print("Setting up MacBook hardware optimizations...")
        
        # Get hardware summary
        hardware_summary = self.hardware_detector.get_hardware_summary()
        print(f"Detected hardware: {hardware_summary['cpu']['brand']}")
        print(f"Available memory: {hardware_summary['memory']['available_gb']:.1f}GB")
        print(f"CPU cores: {hardware_summary['cpu']['cores']}")
        
        # Configure CPU optimizations
        cpu_config = self.cpu_optimizer.configure_all()
        print(f"Configured CPU optimization: {cpu_config.torch_threads} threads")
        
        # Configure tensor operations
        self.tensor_optimizer.optimize_for_training()
        print("Configured tensor operations for training")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(interval=2.0)
        print("Started resource monitoring")
        
        return {
            "hardware_summary": hardware_summary,
            "cpu_config": cpu_config,
            "optimization_status": "configured"
        }
    
    def adapt_configuration(self, dataset_size: int) -> ConfigurationResult:
        """
        Adapt training configuration for MacBook hardware.
        
        Args:
            dataset_size: Size of training dataset
            
        Returns:
            Configuration adaptation result
        """
        print("Adapting configuration for MacBook hardware...")
        
        # Convert PretrainConfig to dictionary
        base_config_dict = self.base_config.model_dump()
        
        # Create hardware-appropriate configuration
        config_result = self.config_adapter.create_hardware_appropriate_config(
            base_config_dict, dataset_size
        )
        
        # Validate the adapted configuration
        validation_result = self.validator.validate_configuration(
            config_result.adapted_config, auto_correct=True
        )
        
        if not validation_result.is_valid:
            print("Configuration validation failed. Using auto-corrected version.")
            if validation_result.corrected_config:
                config_result.adapted_config = validation_result.corrected_config
        
        # Print configuration summary
        print(f"Adapted batch size: {config_result.training_params.batch_size}")
        print(f"Gradient accumulation: {config_result.training_params.gradient_accumulation_steps}")
        print(f"Effective batch size: {config_result.training_params.effective_batch_size}")
        print(f"Learning rate: {config_result.training_params.learning_rate:.2e}")
        print(f"Memory limit: {config_result.training_params.memory_limit_mb}MB")
        
        # Print warnings if any
        if config_result.validation_warnings:
            print("\nConfiguration warnings:")
            for warning in config_result.validation_warnings:
                print(f"  ⚠️  {warning}")
        
        return config_result
    
    def create_macbook_dataloader(self, config: PretrainConfig, split: str, 
                                 training_params, **kwargs) -> tuple:
        """
        Create MacBook-optimized dataloader with memory-efficient dataset management.
        
        Args:
            config: Training configuration
            split: Dataset split ('train' or 'test')
            training_params: MacBook training parameters
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (dataloader, metadata)
        """
        # Determine dataset paths
        dataset_paths = (config.data_paths_test 
                        if len(config.data_paths_test) > 0 and split == "test" 
                        else config.data_paths)
        
        # Validate dataset memory constraints
        validation = self.dataset_manager.validate_dataset_memory_constraints(
            dataset_paths, training_params.batch_size, split
        )
        
        if not validation['memory_constraints_met']:
            print(f"Dataset memory constraints not met for {split} split:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        # Create PuzzleDatasetConfig
        puzzle_config = PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=dataset_paths,
            rank=0,  # Single process for MacBook
            num_replicas=1,
            **kwargs
        )
        
        # Create memory-efficient dataset
        dataset, creation_info = self.dataset_manager.create_dataloader_with_fallback(
            dataset_paths, puzzle_config, split
        )
        
        print(f"Created {split} dataset using {creation_info['final_strategy']} strategy")
        if creation_info['fallback_used']:
            print(f"  Fallback to streaming was used due to memory constraints")
        
        # Use MacBook-optimized DataLoader settings
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=training_params.num_workers,
            prefetch_factor=2,  # Reduced for memory constraints
            pin_memory=training_params.pin_memory,
            persistent_workers=training_params.num_workers > 0
        )
        
        return dataloader, dataset.metadata
    
    def create_macbook_model(self, config: PretrainConfig, train_metadata: PuzzleDatasetMetadata,
                           training_params) -> tuple:
        """
        Create MacBook-optimized model.
        
        Args:
            config: Training configuration
            train_metadata: Training dataset metadata
            training_params: MacBook training parameters
            
        Returns:
            Tuple of (model, optimizers, optimizer_lrs)
        """
        model_cfg = dict(
            **config.arch.__pydantic_extra__,  # type: ignore
            batch_size=training_params.batch_size,
            vocab_size=train_metadata.vocab_size,
            seq_len=min(train_metadata.seq_len, training_params.max_sequence_length),
            num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
            causal=False  # Non-autoregressive
        )
        
        # Instantiate model with loss head
        model_cls = load_model_class(config.arch.name)
        loss_head_cls = load_model_class(config.arch.loss.name)
        
        # Create model on CPU
        model: nn.Module = model_cls(model_cfg)
        print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        
        # Load checkpoint if specified
        if config.load_checkpoint:
            load_checkpoint(model, config)
        
        # Model compilation (optional, may not help on CPU)
        if training_params.enable_cpu_optimization and hasattr(torch, 'compile'):
            try:
                if "DISABLE_COMPILE" not in os.environ:
                    print("Compiling model for CPU optimization...")
                    model = torch.compile(model, mode='default')  # type: ignore
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Create optimizers
        if config.arch.puzzle_emb_ndim == 0:
            optimizers = [
                AdamATan2(
                    model.parameters(),
                    lr=0,  # Set by scheduler
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]
            optimizer_lrs = [training_params.learning_rate]
        elif config.freeze_weights:
            optimizers = [
                CastedSparseEmbeddingSignSGD_Distributed(
                    model.model.puzzle_emb.buffers(),  # type: ignore
                    lr=0,  # Set by scheduler
                    weight_decay=config.puzzle_emb_weight_decay,
                    world_size=1  # Single process
                )
            ]
            optimizer_lrs = [config.puzzle_emb_lr]
        else:
            optimizers = [
                CastedSparseEmbeddingSignSGD_Distributed(
                    model.model.puzzle_emb.buffers(),  # type: ignore
                    lr=0,  # Set by scheduler
                    weight_decay=config.puzzle_emb_weight_decay,
                    world_size=1  # Single process
                ),
                AdamATan2(
                    model.parameters(),
                    lr=0,  # Set by scheduler
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]
            optimizer_lrs = [config.puzzle_emb_lr, training_params.learning_rate]
        
        return model, optimizers, optimizer_lrs
    
    def initialize_training_state(self, config_result: ConfigurationResult,
                                train_metadata: PuzzleDatasetMetadata) -> MacBookTrainingState:
        """
        Initialize MacBook training state.
        
        Args:
            config_result: Configuration adaptation result
            train_metadata: Training dataset metadata
            
        Returns:
            Initialized MacBook training state
        """
        # Create adapted PretrainConfig
        adapted_config_obj = PretrainConfig(**config_result.adapted_config)
        
        # Calculate total training steps
        total_steps = int(
            adapted_config_obj.epochs * 
            train_metadata.total_groups * 
            train_metadata.mean_puzzle_examples / 
            config_result.training_params.effective_batch_size
        )
        
        # Create model and optimizers
        model, optimizers, optimizer_lrs = self.create_macbook_model(
            adapted_config_obj, train_metadata, config_result.training_params
        )
        
        # Create original training state
        train_state = TrainState(
            step=0,
            total_steps=total_steps,
            model=model,
            optimizers=optimizers,
            optimizer_lrs=optimizer_lrs,
            carry=None
        )
        
        # Set memory baseline
        self.memory_manager.set_baseline_memory()
        
        # Track model memory
        model_memory_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # Assume float32
        self.memory_manager.track_model_memory(model_memory_mb)
        
        # Create MacBook training state
        macbook_state = MacBookTrainingState(
            train_state=train_state,
            hardware_detector=self.hardware_detector,
            memory_manager=self.memory_manager,
            cpu_optimizer=self.cpu_optimizer,
            resource_monitor=self.resource_monitor,
            checkpoint_manager=self.checkpoint_manager,
            config_adapter=self.config_adapter,
            validator=self.validator,
            original_config=self.base_config.model_dump(),
            adapted_config=config_result.adapted_config,
            start_time=time.time(),
            last_checkpoint_time=time.time(),
            training_metrics={
                'loss': [],
                'learning_rate': [],
                'samples_per_second': [],
                'memory_usage_mb': [],
                'cpu_usage_percent': []
            },
            resource_history=[],
            samples_processed=0,
            total_training_time=0.0,
            average_samples_per_second=0.0
        )
        
        return macbook_state
    
    def train_batch_macbook(self, macbook_state: MacBookTrainingState, batch: Any,
                          training_params) -> Optional[Dict[str, Any]]:
        """
        Train single batch with MacBook optimizations.
        
        Args:
            macbook_state: MacBook training state
            batch: Training batch
            training_params: Training parameters
            
        Returns:
            Training metrics or None
        """
        train_state = macbook_state.train_state
        train_state.step += 1
        
        if train_state.step > train_state.total_steps:
            return None
        
        batch_start_time = time.time()
        
        # Monitor memory before training step
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        # Check for memory pressure and adjust if needed
        if memory_stats.percent_used > 85:
            print(f"High memory usage ({memory_stats.percent_used:.1f}%), triggering cleanup...")
            self.memory_manager.force_garbage_collection()
        
        # Move batch to device (CPU)
        batch = {k: v.cpu() for k, v in batch.items()}
        
        # Initialize carry if needed
        if train_state.carry is None:
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore
        
        # Forward pass
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )
        
        # Backward pass with gradient accumulation
        effective_batch_size = training_params.effective_batch_size
        scaled_loss = loss / training_params.gradient_accumulation_steps
        scaled_loss.backward()
        
        # Update optimizers every gradient_accumulation_steps
        lr_this_step = None
        if train_state.step % training_params.gradient_accumulation_steps == 0:
            for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                lr_this_step = compute_lr(base_lr, self.base_config, train_state)
                
                for param_group in optim.param_groups:
                    param_group['lr'] = lr_this_step
                
                optim.step()
                optim.zero_grad()
        
        # Calculate training speed
        batch_time = time.time() - batch_start_time
        batch_size = training_params.batch_size
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0
        
        # Update tracking
        macbook_state.samples_processed += batch_size
        macbook_state.total_training_time += batch_time
        macbook_state.average_samples_per_second = (
            macbook_state.samples_processed / macbook_state.total_training_time
            if macbook_state.total_training_time > 0 else 0
        )
        
        # Update memory peak
        self.memory_manager.update_peak_memory()
        
        # Collect metrics
        if len(metrics) and lr_this_step is not None:
            metric_keys = list(sorted(metrics.keys()))
            reduced_metrics = {}
            
            for k in metric_keys:
                value = metrics[k].item() if hasattr(metrics[k], 'item') else metrics[k]
                if k.endswith("loss"):
                    reduced_metrics[f"train/{k}"] = value / effective_batch_size
                else:
                    count = max(metrics.get("count", 1), 1)
                    reduced_metrics[f"train/{k}"] = value / count
            
            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics["train/samples_per_second"] = samples_per_second
            reduced_metrics["train/memory_usage_mb"] = memory_stats.used_mb
            reduced_metrics["train/memory_usage_percent"] = memory_stats.percent_used
            
            # Store metrics in history
            macbook_state.training_metrics['loss'].append(reduced_metrics.get("train/loss", 0))
            macbook_state.training_metrics['learning_rate'].append(lr_this_step)
            macbook_state.training_metrics['samples_per_second'].append(samples_per_second)
            macbook_state.training_metrics['memory_usage_mb'].append(memory_stats.used_mb)
            
            return reduced_metrics
        
        return None
    
    def save_checkpoint_macbook(self, macbook_state: MacBookTrainingState, 
                               training_params, force: bool = False) -> bool:
        """
        Save checkpoint using MacBook checkpoint management system.
        
        Args:
            macbook_state: MacBook training state
            training_params: Training parameters
            force: Force saving regardless of conditions
            
        Returns:
            True if checkpoint was saved successfully
        """
        train_state = macbook_state.train_state
        
        # Get current learning rate
        current_lr = 0.0
        if train_state.optimizers and len(train_state.optimizers[0].param_groups) > 0:
            current_lr = train_state.optimizers[0].param_groups[0]['lr']
        
        # Get current loss
        current_loss = (macbook_state.training_metrics['loss'][-1] 
                       if macbook_state.training_metrics['loss'] else 0.0)
        
        # Calculate current epoch
        current_epoch = int(train_state.step * training_params.effective_batch_size / 
                           macbook_state.samples_processed) if macbook_state.samples_processed > 0 else 0
        
        # Prepare model and optimizer states
        model_state = train_state.model.state_dict()
        optimizer_states = [opt.state_dict() for opt in train_state.optimizers]
        
        # Create training configuration for checkpoint
        from macbook_optimization.config_management import MacBookTrainingConfig
        training_config = MacBookTrainingConfig(
            batch_size=training_params.batch_size,
            gradient_accumulation_steps=training_params.gradient_accumulation_steps,
            num_workers=training_params.num_workers,
            pin_memory=training_params.pin_memory,
            memory_limit_mb=training_params.memory_limit_mb,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=training_params.learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=train_state.total_steps,
            checkpoint_interval=training_params.checkpoint_interval,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary=macbook_state.hardware_detector.get_hardware_summary()
        )
        
        # Save checkpoint
        result = macbook_state.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            optimizer_state={'optimizers': optimizer_states, 'carry': train_state.carry},
            step=train_state.step,
            epoch=current_epoch,
            loss=current_loss,
            learning_rate=current_lr,
            training_config=training_config,
            training_time_seconds=macbook_state.total_training_time,
            force=force
        )
        
        if result.success:
            macbook_state.last_checkpoint_time = time.time()
            print(f"Checkpoint saved: {result.checkpoint_id} "
                  f"(size: {result.disk_usage_mb:.1f}MB, "
                  f"time: {result.save_time_seconds:.2f}s)")
            
            if result.warnings:
                for warning in result.warnings:
                    print(f"  Warning: {warning}")
        else:
            print(f"Checkpoint save failed: {', '.join(result.errors)}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"  Warning: {warning}")
        
        return result.success
    
    def load_checkpoint_macbook(self, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint using MacBook checkpoint management system.
        
        Args:
            checkpoint_id: Specific checkpoint ID to load (latest if None)
            
        Returns:
            Loaded checkpoint data or None if failed
        """
        # Create current config for validation
        from macbook_optimization.config_management import MacBookTrainingConfig
        current_config = MacBookTrainingConfig(
            batch_size=8,  # Default values for validation
            gradient_accumulation_steps=8,
            num_workers=2,
            pin_memory=False,
            memory_limit_mb=4000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            memory_pressure_threshold=75.0,
            use_mkl=True,
            torch_threads=4,
            enable_cpu_optimization=True,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_steps=10000,
            checkpoint_interval=500,
            max_checkpoints_to_keep=3,
            monitoring_interval=2.0,
            enable_thermal_monitoring=True,
            hardware_summary={}
        )
        
        result = self.checkpoint_manager.load_checkpoint(
            checkpoint_id=checkpoint_id,
            validate_config=True,
            current_config=current_config
        )
        
        if result.success:
            print(f"Checkpoint loaded: {result.checkpoint_id}")
            if result.metadata:
                print(f"  Step: {result.metadata.step}, Loss: {result.metadata.loss:.4f}")
                print(f"  Training time: {result.metadata.training_time_seconds/60:.1f} minutes")
            
            if result.warnings:
                for warning in result.warnings:
                    print(f"  Warning: {warning}")
            
            if not result.config_compatible:
                print("  Configuration mismatch detected - proceeding with compatibility mode")
            
            return {
                'model_state_dict': result.model_state,
                'optimizer_state': result.optimizer_state,
                'metadata': result.metadata
            }
        else:
            print(f"Checkpoint load failed: {', '.join(result.errors)}")
            return None
    
    def create_progress_display(self, macbook_state: MacBookTrainingState) -> str:
        """
        Create progress display with MacBook-specific metrics.
        
        Args:
            macbook_state: MacBook training state
            
        Returns:
            Formatted progress string
        """
        train_state = macbook_state.train_state
        memory_stats = self.memory_manager.monitor_memory_usage()
        
        # Calculate progress
        progress_percent = (train_state.step / train_state.total_steps) * 100
        
        # Calculate ETA
        elapsed_time = time.time() - macbook_state.start_time
        if train_state.step > 0:
            time_per_step = elapsed_time / train_state.step
            remaining_steps = train_state.total_steps - train_state.step
            eta_seconds = remaining_steps * time_per_step
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0
        
        # Get recent metrics
        recent_loss = macbook_state.training_metrics['loss'][-1] if macbook_state.training_metrics['loss'] else 0
        recent_sps = macbook_state.training_metrics['samples_per_second'][-1] if macbook_state.training_metrics['samples_per_second'] else 0
        
        return (
            f"Step {train_state.step}/{train_state.total_steps} ({progress_percent:.1f}%) | "
            f"Loss: {recent_loss:.4f} | "
            f"Speed: {recent_sps:.1f} samples/s | "
            f"Memory: {memory_stats.percent_used:.1f}% ({memory_stats.used_mb:.0f}MB) | "
            f"ETA: {eta_minutes:.1f}min"
        )
    
    def train(self, dataset_size: int) -> MacBookTrainingState:
        """
        Main training loop with MacBook optimizations.
        
        Args:
            dataset_size: Size of training dataset
            
        Returns:
            Final MacBook training state
        """
        print("Starting MacBook-optimized TRM training...")
        
        # Setup hardware optimizations
        hardware_summary = self.setup_hardware_optimization()
        
        # Adapt configuration
        config_result = self.adapt_configuration(dataset_size)
        
        # Analyze dataset requirements before creating dataloaders
        train_analysis = self.dataset_manager.analyze_dataset_requirements(
            self.base_config.data_paths, "train"
        )
        print(f"Training dataset analysis:")
        print(f"  Total size: {train_analysis['total_size_mb']:.1f}MB")
        print(f"  Recommended strategy: {train_analysis['recommended_strategy']}")
        print(f"  Memory utilization: {train_analysis['memory_utilization_percent']:.1f}%")
        
        # Create dataloaders
        train_epochs_per_iter = (
            self.base_config.eval_interval 
            if self.base_config.eval_interval is not None 
            else self.base_config.epochs
        )
        total_iters = self.base_config.epochs // train_epochs_per_iter
        
        train_loader, train_metadata = self.create_macbook_dataloader(
            self.base_config, "train", config_result.training_params,
            test_set_mode=False, epochs_per_iter=train_epochs_per_iter,
            global_batch_size=config_result.training_params.effective_batch_size
        )
        
        try:
            if self.base_config.data_paths_test:
                test_analysis = self.dataset_manager.analyze_dataset_requirements(
                    self.base_config.data_paths_test, "test"
                )
                print(f"Test dataset analysis:")
                print(f"  Total size: {test_analysis['total_size_mb']:.1f}MB")
                print(f"  Recommended strategy: {test_analysis['recommended_strategy']}")
            
            eval_loader, eval_metadata = self.create_macbook_dataloader(
                self.base_config, "test", config_result.training_params,
                test_set_mode=True, epochs_per_iter=1,
                global_batch_size=config_result.training_params.effective_batch_size
            )
        except:
            print("No evaluation data found")
            eval_loader = eval_metadata = None
        
        # Create evaluators
        try:
            evaluators = create_evaluators(self.base_config, eval_metadata) if eval_metadata else []
        except:
            print("No evaluators found")
            evaluators = []
        
        # Initialize training state
        macbook_state = self.initialize_training_state(config_result, train_metadata)
        
        # Check for checkpoint resumption
        if self.base_config.load_checkpoint:
            print("Checking for existing checkpoints to resume...")
            checkpoint_data = self.load_checkpoint_macbook()
            if checkpoint_data and checkpoint_data['model_state_dict']:
                print("Resuming from checkpoint...")
                
                # Load model state
                macbook_state.train_state.model.load_state_dict(checkpoint_data['model_state_dict'])
                
                # Load optimizer states if available
                if checkpoint_data['optimizer_state'] and 'optimizers' in checkpoint_data['optimizer_state']:
                    optimizer_states = checkpoint_data['optimizer_state']['optimizers']
                    for i, (optimizer, state) in enumerate(zip(macbook_state.train_state.optimizers, optimizer_states)):
                        if i < len(optimizer_states):
                            optimizer.load_state_dict(state)
                
                # Restore carry state if available
                if checkpoint_data['optimizer_state'] and 'carry' in checkpoint_data['optimizer_state']:
                    macbook_state.train_state.carry = checkpoint_data['optimizer_state']['carry']
                
                # Update step counter if metadata is available
                if checkpoint_data['metadata']:
                    macbook_state.train_state.step = checkpoint_data['metadata'].step
                    print(f"Resumed from step {macbook_state.train_state.step}")
            else:
                print("No valid checkpoint found for resumption")
        
        # Setup progress tracking
        progress_bar = tqdm.tqdm(total=macbook_state.train_state.total_steps)
        
        # Setup wandb logging
        if self.base_config.project_name:
            wandb.init(
                project=self.base_config.project_name,
                name=f"{self.base_config.run_name}_macbook",
                config={
                    **config_result.adapted_config,
                    "hardware_summary": hardware_summary,
                    "macbook_optimization": True
                },
                settings=wandb.Settings(_disable_stats=True)
            )
            wandb.log({
                "num_params": sum(x.numel() for x in macbook_state.train_state.model.parameters()),
                "hardware_summary": hardware_summary
            }, step=0)
        
        # EMA helper if enabled
        ema_helper = None
        if self.base_config.ema:
            print('Setup EMA')
            ema_helper = EMAHelper(mu=self.base_config.ema_rate)
            ema_helper.register(macbook_state.train_state.model)
        
        print(f"Starting training for {total_iters} iterations...")
        
        # Main training loop
        try:
            for iter_id in range(total_iters):
                if self._shutdown_requested:
                    print("Shutdown requested, stopping training...")
                    break
                
                print(f"\nEpoch {iter_id * train_epochs_per_iter}")
                
                # Training phase
                macbook_state.train_state.model.train()
                batch_count = 0
                for set_name, batch, global_batch_size in train_loader:
                    if self._shutdown_requested:
                        break
                    
                    batch_count += 1
                    
                    # Monitor dataset memory usage periodically
                    if batch_count % 50 == 0:  # Every 50 batches
                        dataset_memory_stats = self.dataset_manager.monitor_dataset_memory_usage(
                            f"train_epoch_{iter_id}_batch_{batch_count}"
                        )
                        
                        # Log dataset memory metrics
                        if self.base_config.project_name:
                            wandb.log({
                                "dataset_memory_mb": dataset_memory_stats['memory_usage']['used_mb'],
                                "dataset_memory_percent": dataset_memory_stats['memory_usage']['percent_used']
                            }, step=macbook_state.train_state.step)
                    
                    metrics = self.train_batch_macbook(
                        macbook_state, batch, config_result.training_params
                    )
                    
                    if metrics is not None:
                        if self.base_config.project_name:
                            wandb.log(metrics, step=macbook_state.train_state.step)
                        
                        # Update progress bar
                        progress_bar.set_description(self.create_progress_display(macbook_state))
                        progress_bar.update(1)
                    
                    # Update EMA
                    if ema_helper:
                        ema_helper.update(macbook_state.train_state.model)
                    
                    # Checkpoint if needed using MacBook checkpoint management
                    should_save, reason = macbook_state.checkpoint_manager.should_save_checkpoint(
                        macbook_state.train_state.step
                    )
                    if should_save:
                        print(f"\nSaving checkpoint (reason: {reason})...")
                        self.save_checkpoint_macbook(macbook_state, config_result.training_params)
                
                # Evaluation phase
                if iter_id >= self.base_config.min_eval_interval and eval_loader:
                    print("\nRunning evaluation...")
                    
                    # Use EMA model if available
                    if ema_helper:
                        eval_state = copy.deepcopy(macbook_state.train_state)
                        eval_state.model = ema_helper.ema_copy(eval_state.model)
                    else:
                        eval_state = macbook_state.train_state
                    
                    eval_state.model.eval()
                    eval_metrics = evaluate(
                        self.base_config, eval_state, eval_loader, eval_metadata,
                        evaluators, rank=0, world_size=1, cpu_group=None
                    )
                    
                    if eval_metrics and self.base_config.project_name:
                        wandb.log(eval_metrics, step=macbook_state.train_state.step)
                    
                    # Final checkpoint using MacBook checkpoint management
                    if iter_id == total_iters - 1 or self.base_config.checkpoint_every_eval:
                        print("Saving final checkpoint...")
                        self.save_checkpoint_macbook(macbook_state, config_result.training_params, force=True)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        finally:
            # Cleanup
            progress_bar.close()
            self.resource_monitor.stop_monitoring()
            self.cpu_optimizer.restore_environment()
            
            if self.base_config.project_name:
                wandb.finish()
        
        # Final summary
        self.print_training_summary(macbook_state)
        
        return macbook_state
    
    def print_training_summary(self, macbook_state: MacBookTrainingState):
        """Print comprehensive training summary."""
        total_time = time.time() - macbook_state.start_time
        memory_summary = self.memory_manager.get_memory_summary()
        dataset_metrics = self.dataset_manager.get_loading_metrics()
        
        print("\n" + "="*60)
        print("MacBook TRM Training Summary")
        print("="*60)
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Steps completed: {macbook_state.train_state.step}/{macbook_state.train_state.total_steps}")
        print(f"Samples processed: {macbook_state.samples_processed:,}")
        print(f"Average speed: {macbook_state.average_samples_per_second:.1f} samples/second")
        
        print(f"\nMemory Usage:")
        print(f"  Peak memory: {memory_summary['tracking']['peak_mb']:.0f}MB")
        print(f"  Model memory: {memory_summary['tracking']['model_mb']:.0f}MB")
        print(f"  Training overhead: {memory_summary['tracking']['training_overhead_mb']:.0f}MB")
        
        print(f"\nDataset Management:")
        if dataset_metrics:
            for i, metrics in enumerate(dataset_metrics):
                print(f"  Dataset {i+1}: {metrics.total_size_mb:.1f}MB, "
                      f"strategy: {metrics.loading_strategy}, "
                      f"memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        print(f"\nCheckpoint Management:")
        checkpoint_summary = macbook_state.checkpoint_manager.get_checkpoint_summary()
        print(f"  Checkpoints saved: {checkpoint_summary['checkpoint_count']}")
        print(f"  Total checkpoint size: {checkpoint_summary['total_size_mb']:.1f}MB")
        print(f"  Free disk space: {checkpoint_summary['free_disk_space_mb']:.1f}MB")
        if checkpoint_summary['latest_checkpoint']:
            print(f"  Latest checkpoint: {checkpoint_summary['latest_checkpoint']}")
        
        if macbook_state.training_metrics['loss']:
            final_loss = macbook_state.training_metrics['loss'][-1]
            print(f"\nFinal training loss: {final_loss:.4f}")
        
        print("\nHardware utilization was optimized for MacBook CPU training.")
        print("Dataset loading was optimized for memory constraints.")
        print("="*60)


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch_macbook_training(hydra_config: DictConfig):
    """
    Launch MacBook-optimized TRM training.
    
    Args:
        hydra_config: Hydra configuration
    """
    # Convert to PretrainConfig
    config = PretrainConfig(**hydra_config)  # type: ignore
    
    # Set naming defaults
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-MacBook"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]}-macbook-{coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
    
    # Estimate dataset size (rough approximation)
    dataset_size = 10000  # Default estimate, could be improved with actual dataset inspection
    
    # Create and run trainer
    trainer = MacBookTRMTrainer(config)
    
    try:
        final_state = trainer.train(dataset_size)
        print("Training completed successfully!")
        return final_state
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    launch_macbook_training()