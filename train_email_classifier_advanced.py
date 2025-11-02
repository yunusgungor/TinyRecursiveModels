"""
Advanced Training Script for Email Classification using TRM

This script provides advanced training capabilities including:
- Hyperparameter optimization
- Multi-phase training strategies
- Advanced monitoring and early stopping
- Model ensembling
- Performance analysis
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Add training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.email import evaluate_email_model, EmailClassificationEvaluator
from models.ema import EMA
from training.hyperparameter_tuning import (
    HyperparameterSearcher, HyperparameterConfig, 
    AdvancedLRScheduler, TrainingStrategy, PerformanceTracker
)


class AdvancedEmailClassificationTrainer:
    """Advanced trainer with hyperparameter optimization and multi-phase training"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_distributed()
        self.setup_device()
        self.setup_logging()
        
        # Load dataset metadata
        self.load_dataset_info()
        
        # Setup performance tracking
        self.setup_performance_tracking()
        
        # Setup training strategy
        self.setup_training_strategy()
        
        # Initialize hyperparameter search if enabled
        if config.get('hyperparameter_search', {}).get('enabled', False):
            self.setup_hyperparameter_search()
        else:
            self.hyperparameter_searcher = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = 0.0
        self.current_phase = 0
        
    def setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
    
    def setup_device(self):
        """Setup device"""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        print(f"Rank {self.rank}: Using device {self.device}")
    
    def setup_logging(self):
        """Setup logging and wandb"""
        if self.rank == 0:
            # Initialize wandb
            if self.config.get('use_wandb', True):
                wandb.init(
                    project=self.config.get('wandb_project', 'email-classification-trm-advanced'),
                    name=self.config.get('experiment_name', 'advanced_email_trm'),
                    config=OmegaConf.to_container(self.config, resolve=True),
                    tags=self.config.get('wandb_tags', ['email-classification', 'trm', 'advanced'])
                )
    
    def load_dataset_info(self):
        """Load dataset information"""
        dataset_path = self.config.data_paths[0]
        
        # Try to load enhanced tokenizer first
        tokenizer_path = os.path.join(dataset_path, "tokenizer.pkl")
        if os.path.exists(tokenizer_path):
            from models.email_tokenizer import EmailTokenizer
            self.tokenizer = EmailTokenizer.load(tokenizer_path)
            self.vocab = self.tokenizer.vocab
            self.vocab_size = len(self.vocab)
            
            # Load tokenizer stats
            stats_path = os.path.join(dataset_path, "tokenizer_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.tokenizer_stats = json.load(f)
            else:
                self.tokenizer_stats = {}
        else:
            # Fallback to legacy vocabulary
            vocab_path = os.path.join(dataset_path, "vocab.json")
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)
            self.tokenizer = None
            self.tokenizer_stats = {}
        
        # Load categories
        categories_path = os.path.join(dataset_path, "categories.json")
        with open(categories_path, 'r') as f:
            self.categories = json.load(f)
        
        self.num_categories = len(self.categories)
        
        print(f"Loaded vocabulary: {self.vocab_size} tokens")
        print(f"Email categories: {list(self.categories.keys())}")
        if self.tokenizer:
            print(f"Using enhanced EmailTokenizer")
    
    def setup_performance_tracking(self):
        """Setup performance tracking"""
        output_dir = self.config.get('output_dir', 'outputs/email_classification_advanced')
        self.performance_tracker = PerformanceTracker(output_dir)
        
    def setup_training_strategy(self):
        """Setup training strategy"""
        self.training_strategy = TrainingStrategy(self.config)
        self.training_phases = self.training_strategy.get_training_phases()
        
        print(f"Training strategy: {self.config.get('training_strategy', 'standard')}")
        print(f"Training phases: {len(self.training_phases)}")
        for i, phase in enumerate(self.training_phases):
            print(f"  Phase {i+1}: {phase['name']} - {phase['steps']} steps")
    
    def setup_hyperparameter_search(self):
        """Setup hyperparameter search"""
        hp_config_dict = self.config.hyperparameter_search
        
        # Create hyperparameter config
        hp_config = HyperparameterConfig()
        
        # Override with config values
        for key, value in hp_config_dict.get('parameters', {}).items():
            if hasattr(hp_config, key):
                setattr(hp_config, key, value)
        
        # Create searcher
        search_strategy = hp_config_dict.get('strategy', 'random')
        self.hyperparameter_searcher = HyperparameterSearcher(hp_config, search_strategy)
        
        print(f"Hyperparameter search enabled: {search_strategy} strategy")
    
    def create_model_with_config(self, model_config_override: Optional[Dict] = None) -> EmailTRM:
        """Create model with optional config override"""
        
        # Start with base config
        model_config_dict = OmegaConf.to_container(self.config.arch, resolve=True)
        
        # Apply overrides
        if model_config_override:
            model_config_dict.update(model_config_override)
        
        # Ensure required parameters
        model_config_dict.update({
            'vocab_size': self.vocab_size,
            'num_email_categories': self.num_categories,
            'seq_len': self.config.get('max_seq_len', 512),
            'batch_size': self.config.training.batch_size
        })
        
        # Create model config
        model_config = EmailTRMConfig(**model_config_dict)
        
        # Create model
        model = EmailTRM(model_config)
        model.to(self.device)
        
        return model
    
    def setup_model_and_optimizer(self, hyperparams: Optional[Dict] = None, 
                                  model_config_override: Optional[Dict] = None):
        """Setup model and optimizer with optional hyperparameters"""
        
        # Create model
        self.model = self.create_model_with_config(model_config_override)
        
        # Setup EMA if enabled
        if self.config.get('use_ema', True):
            ema_decay = hyperparams.get('ema_decay', self.config.get('ema_decay', 0.999)) if hyperparams else self.config.get('ema_decay', 0.999)
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Setup optimizer with hyperparameters
        lr = hyperparams.get('learning_rate', self.config.optimizer.lr) if hyperparams else self.config.optimizer.lr
        weight_decay = hyperparams.get('weight_decay', self.config.optimizer.weight_decay) if hyperparams else self.config.optimizer.weight_decay
        
        # Separate parameters for different learning rates
        model_params = []
        puzzle_emb_params = []
        
        for name, param in self.model.named_parameters():
            if 'puzzle_emb' in name or 'puzzle_identifiers' in name:
                puzzle_emb_params.append(param)
            else:
                model_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {
                'params': model_params,
                'lr': lr,
                'weight_decay': weight_decay
            }
        ]
        
        if puzzle_emb_params:
            param_groups.append({
                'params': puzzle_emb_params,
                'lr': self.config.optimizer.get('puzzle_emb_lr', 1e-2),
                'weight_decay': self.config.optimizer.get('puzzle_emb_weight_decay', 0.1)
            })
        
        # Create optimizer
        if self.config.optimizer.name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
                eps=self.config.optimizer.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def setup_scheduler(self, total_steps: int, warmup_steps: Optional[int] = None):
        """Setup learning rate scheduler"""
        
        if warmup_steps is None:
            warmup_steps = self.config.scheduler.get('warmup_steps', 1000)
        
        scheduler_type = self.config.scheduler.get('name', 'cosine_with_restarts')
        
        self.scheduler = AdvancedLRScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.config.scheduler.get('min_lr_ratio', 0.01),
            scheduler_type=scheduler_type
        )
    
    def setup_data_loaders(self, batch_size: Optional[int] = None):
        """Setup data loaders"""
        
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        # Training dataset
        train_config = PuzzleDatasetConfig(
            seed=self.config.training.seed,
            dataset_paths=self.config.data_paths,
            global_batch_size=batch_size,
            test_set_mode=False,
            epochs_per_iter=self.config.training.get('epochs_per_iter', 1),
            rank=self.rank,
            num_replicas=self.world_size
        )
        
        self.train_dataset = PuzzleDataset(train_config, split='train')
        
        # Validation dataset
        val_config = PuzzleDatasetConfig(
            seed=self.config.training.seed,
            dataset_paths=self.config.data_paths,
            global_batch_size=self.config.training.get('eval_batch_size', batch_size),
            test_set_mode=True,
            epochs_per_iter=1,
            rank=self.rank,
            num_replicas=self.world_size
        )
        
        self.val_dataset = PuzzleDataset(val_config, split='test')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        inputs = batch['inputs'].to(self.device)
        labels = batch['labels'].to(self.device)
        puzzle_ids = batch.get('puzzle_identifiers')
        if puzzle_ids is not None:
            puzzle_ids = puzzle_ids.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs, labels=labels, puzzle_identifiers=puzzle_ids)
        loss = outputs['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.training.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.grad_clip
            )
        
        self.optimizer.step()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
        
        # Update scheduler
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs['logits'], dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'num_cycles': outputs.get('num_cycles', 0),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on validation set"""
        if self.ema is not None:
            model_to_eval = self.ema.model
        else:
            model_to_eval = self.model
        
        # Use the evaluation function
        metrics = evaluate_email_model(
            model=model_to_eval,
            dataloader=self.val_dataset,
            device=self.device,
            category_names=list(self.categories.keys()),
            output_dir=None  # Don't save during training
        )
        
        return metrics
    
    def train_single_phase(self, phase_config: Dict[str, Any]) -> Dict[str, float]:
        """Train a single phase"""
        
        print(f"\nStarting phase: {phase_config['name']}")
        print(f"Description: {phase_config['description']}")
        print(f"Steps: {phase_config['steps']}")
        
        # Setup for this phase
        phase_batch_size = phase_config.get('batch_size', self.config.training.batch_size)
        self.setup_data_loaders(phase_batch_size)
        
        # Setup scheduler for this phase
        warmup_steps = phase_config.get('warmup_steps', self.config.scheduler.get('warmup_steps', 500))
        self.setup_scheduler(phase_config['steps'], warmup_steps)
        
        # Adjust learning rate if specified
        if 'learning_rate' in phase_config:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = phase_config['learning_rate']
        
        # Training loop for this phase
        phase_start_step = self.global_step
        target_steps = phase_start_step + phase_config['steps']
        
        log_interval = self.config.training.get('log_interval', 100)
        eval_interval = self.config.training.get('eval_interval', 1000)
        
        running_loss = 0.0
        running_accuracy = 0.0
        log_counter = 0
        
        for set_name, batch, batch_size in self.train_dataset:
            
            # Check if phase is complete
            if self.global_step >= target_steps:
                break
            
            # Training step
            metrics = self.train_step(batch)
            
            running_loss += metrics['loss']
            running_accuracy += metrics['accuracy']
            log_counter += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / log_counter
                avg_accuracy = running_accuracy / log_counter
                
                print(f"Phase {phase_config['name']} - Step {self.global_step}: "
                      f"loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, "
                      f"lr={metrics['learning_rate']:.2e}")
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        f"phase_{phase_config['name']}/loss": avg_loss,
                        f"phase_{phase_config['name']}/accuracy": avg_accuracy,
                        f"phase_{phase_config['name']}/learning_rate": metrics['learning_rate'],
                        "global_step": self.global_step,
                        "phase": phase_config['name']
                    }, step=self.global_step)
                
                running_loss = 0.0
                running_accuracy = 0.0
                log_counter = 0
            
            # Evaluation
            if self.global_step % eval_interval == 0:
                eval_metrics = self.evaluate()
                
                if self.rank == 0:
                    accuracy = eval_metrics.get('accuracy', 0.0)
                    f1_score = eval_metrics.get('macro_f1', 0.0)
                    
                    print(f"Phase {phase_config['name']} - Validation at step {self.global_step}: "
                          f"Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
                    
                    # Update performance tracker
                    self.performance_tracker.update(self.global_step, eval_metrics)
                    
                    if self.config.get('use_wandb', True):
                        wandb.log({
                            f"phase_{phase_config['name']}/val_accuracy": accuracy,
                            f"phase_{phase_config['name']}/val_f1": f1_score,
                            "global_step": self.global_step
                        }, step=self.global_step)
                    
                    # Update best accuracy
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
        
        # Phase completion
        phase_steps = self.global_step - phase_start_step
        print(f"Phase {phase_config['name']} completed: {phase_steps} steps")
        
        # Final evaluation for this phase
        final_metrics = self.evaluate()
        return final_metrics
    
    def train_with_hyperparameters(self, hyperparams: Dict[str, Any]) -> float:
        """Train model with specific hyperparameters"""
        
        print(f"\nTraining with hyperparameters: {hyperparams}")
        
        # Setup model and optimizer with hyperparameters
        model_config_override = {}
        for key in ['hidden_size', 'num_heads', 'L_layers', 'H_cycles', 'L_cycles', 
                   'pooling_strategy', 'classification_dropout', 'subject_attention_weight']:
            if key in hyperparams:
                model_config_override[key] = hyperparams[key]
        
        self.setup_model_and_optimizer(hyperparams, model_config_override)
        
        # Setup data loaders with hyperparameter batch size
        batch_size = hyperparams.get('batch_size', self.config.training.batch_size)
        self.setup_data_loaders(batch_size)
        
        # Train through all phases
        final_accuracy = 0.0
        
        for phase_config in self.training_phases:
            # Apply hyperparameter overrides to phase config
            phase_config = phase_config.copy()
            if 'learning_rate' in hyperparams:
                phase_config['learning_rate'] = hyperparams['learning_rate']
            if 'batch_size' in hyperparams:
                phase_config['batch_size'] = hyperparams['batch_size']
            if 'warmup_steps' in hyperparams:
                phase_config['warmup_steps'] = hyperparams['warmup_steps']
            
            phase_metrics = self.train_single_phase(phase_config)
            final_accuracy = phase_metrics.get('accuracy', 0.0)
        
        return final_accuracy
    
    def hyperparameter_search(self) -> Dict[str, Any]:
        """Perform hyperparameter search"""
        
        if not self.hyperparameter_searcher:
            raise ValueError("Hyperparameter search not enabled")
        
        num_trials = self.config.hyperparameter_search.get('num_trials', 10)
        
        print(f"\nStarting hyperparameter search with {num_trials} trials")
        
        # Generate hyperparameter combinations
        hyperparameter_combinations = self.hyperparameter_searcher.generate_hyperparameters(num_trials)
        
        best_hyperparams = None
        best_performance = 0.0
        
        for trial_idx, hyperparams in enumerate(hyperparameter_combinations):
            print(f"\n{'='*60}")
            print(f"HYPERPARAMETER SEARCH - TRIAL {trial_idx + 1}/{num_trials}")
            print(f"{'='*60}")
            
            # Reset global step for each trial
            self.global_step = 0
            
            try:
                # Train with these hyperparameters
                performance = self.train_with_hyperparameters(hyperparams)
                
                # Update search history
                self.hyperparameter_searcher.update_history(hyperparams, performance)
                
                print(f"Trial {trial_idx + 1} completed - Performance: {performance:.4f}")
                
                # Track best performance
                if performance > best_performance:
                    best_performance = performance
                    best_hyperparams = hyperparams.copy()
                    
                    print(f"New best performance: {best_performance:.4f}")
                
                # Log to wandb
                if self.config.get('use_wandb', True) and self.rank == 0:
                    wandb.log({
                        "hp_search/trial": trial_idx + 1,
                        "hp_search/performance": performance,
                        "hp_search/best_performance": best_performance,
                        **{f"hp_search/param_{k}": v for k, v in hyperparams.items()}
                    })
                
            except Exception as e:
                print(f"Trial {trial_idx + 1} failed: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SEARCH COMPLETED")
        print(f"{'='*60}")
        print(f"Best performance: {best_performance:.4f}")
        print(f"Best hyperparameters: {best_hyperparams}")
        
        return {
            "best_hyperparams": best_hyperparams,
            "best_performance": best_performance,
            "search_history": self.hyperparameter_searcher.search_history
        }
    
    def train(self):
        """Main training function"""
        
        if self.hyperparameter_searcher:
            # Perform hyperparameter search
            search_results = self.hyperparameter_search()
            
            # Save search results
            if self.rank == 0:
                output_dir = Path(self.config.get('output_dir', 'outputs/email_classification_advanced'))
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_dir / 'hyperparameter_search_results.json', 'w') as f:
                    json.dump(search_results, f, indent=2)
                
                print(f"Hyperparameter search results saved to {output_dir / 'hyperparameter_search_results.json'}")
        
        else:
            # Standard training
            print("Starting standard training...")
            
            # Setup model and optimizer
            self.setup_model_and_optimizer()
            
            # Train through all phases
            for phase_idx, phase_config in enumerate(self.training_phases):
                self.current_phase = phase_idx
                phase_metrics = self.train_single_phase(phase_config)
                
                print(f"Phase {phase_idx + 1} completed with accuracy: {phase_metrics.get('accuracy', 0.0):.4f}")
        
        # Generate final report
        if self.rank == 0:
            report = self.performance_tracker.generate_report()
            
            output_dir = Path(self.config.get('output_dir', 'outputs/email_classification_advanced'))
            with open(output_dir / 'training_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nTraining completed!")
            print(f"Final accuracy: {report.get('final_accuracy', 0.0):.4f}")
            print(f"Best accuracy: {report.get('best_accuracy', 0.0):.4f}")
            print(f"Training report saved to {output_dir / 'training_report.json'}")


@hydra.main(version_base=None, config_path="config", config_name="cfg_email_train_advanced")
def main(config: DictConfig):
    """Main training function"""
    
    # Create trainer and start training
    trainer = AdvancedEmailClassificationTrainer(config)
    trainer.train()
    
    # Cleanup
    if trainer.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()