"""
Training script for Email Classification using TRM

This script trains the Tiny Recursive Reasoning Model (TRM) for email classification tasks.
"""

import os
import json
import math
import time
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from evaluators.email import evaluate_email_model, EmailClassificationEvaluator
from models.ema import EMA
from models.losses import IGNORE_LABEL_ID


class EmailClassificationTrainer:
    """Trainer for email classification using TRM"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_distributed()
        self.setup_device()
        self.setup_logging()
        
        # Load dataset metadata
        self.load_dataset_info()
        
        # Initialize model
        self.setup_model()
        
        # Setup training components
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_data_loaders()
        
        # Setup evaluation
        self.setup_evaluator()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = 0.0
        
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
                    project=self.config.get('wandb_project', 'email-classification-trm'),
                    name=self.config.get('experiment_name', 'email_trm_experiment'),
                    config=OmegaConf.to_container(self.config, resolve=True)
                )
    
    def load_dataset_info(self):
        """Load dataset information"""
        dataset_path = self.config.data_paths[0]
        
        # Load vocabulary
        vocab_path = os.path.join(dataset_path, "vocab.json")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # Load categories
        categories_path = os.path.join(dataset_path, "categories.json")
        with open(categories_path, 'r') as f:
            self.categories = json.load(f)
        
        self.vocab_size = len(self.vocab)
        self.num_categories = len(self.categories)
        
        print(f"Loaded vocabulary: {self.vocab_size} tokens")
        print(f"Email categories: {list(self.categories.keys())}")
    
    def setup_model(self):
        """Setup model"""
        # Create model config
        model_config = EmailTRMConfig(
            vocab_size=self.vocab_size,
            num_email_categories=self.num_categories,
            **self.config.arch
        )
        
        # Create model
        self.model = EmailTRM(model_config)
        self.model.to(self.device)
        
        # Setup EMA if enabled
        if self.config.get('use_ema', True):
            self.ema = EMA(self.model, decay=self.config.get('ema_decay', 0.999))
        else:
            self.ema = None
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def setup_optimizer(self):
        """Setup optimizer"""
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
                'lr': self.config.optimizer.lr,
                'weight_decay': self.config.optimizer.weight_decay
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
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.scheduler.name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.max_steps,
                eta_min=self.config.scheduler.get('min_lr', 1e-6)
            )
        elif self.config.scheduler.name == 'linear_warmup_cosine':
            # Custom scheduler with warmup
            warmup_steps = self.config.scheduler.get('warmup_steps', 1000)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (self.config.training.max_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        # Training dataset
        train_config = PuzzleDatasetConfig(
            seed=self.config.training.seed,
            dataset_paths=self.config.data_paths,
            global_batch_size=self.config.training.batch_size,
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
            global_batch_size=self.config.training.get('eval_batch_size', self.config.training.batch_size),
            test_set_mode=True,
            epochs_per_iter=1,
            rank=self.rank,
            num_replicas=self.world_size
        )
        
        self.val_dataset = PuzzleDataset(val_config, split='test')
    
    def setup_evaluator(self):
        """Setup evaluator"""
        category_names = list(self.categories.keys())
        output_dir = self.config.get('output_dir', 'outputs/email_classification')
        
        self.evaluator = EmailClassificationEvaluator(
            category_names=category_names,
            output_dir=output_dir
        )
    
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
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs['logits'], dim=-1)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'num_cycles': outputs.get('num_cycles', 0)
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
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        output_dir = Path(self.config.get('output_dir', 'outputs/email_classification'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_accuracy:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.training.max_steps} steps...")
        
        start_time = time.time()
        log_interval = self.config.training.get('log_interval', 100)
        eval_interval = self.config.training.get('eval_interval', 1000)
        save_interval = self.config.training.get('save_interval', 5000)
        
        running_loss = 0.0
        running_accuracy = 0.0
        
        for set_name, batch, batch_size in self.train_dataset:
            
            # Training step
            metrics = self.train_step(batch)
            
            running_loss += metrics['loss']
            running_accuracy += metrics['accuracy']
            self.global_step += 1
            
            # Logging
            if self.global_step % log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / log_interval
                avg_accuracy = running_accuracy / log_interval
                
                elapsed_time = time.time() - start_time
                steps_per_sec = self.global_step / elapsed_time
                
                print(f"Step {self.global_step}: loss={avg_loss:.4f}, "
                      f"acc={avg_accuracy:.4f}, steps/sec={steps_per_sec:.2f}")
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/accuracy': avg_accuracy,
                        'train/steps_per_sec': steps_per_sec,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=self.global_step)
                
                running_loss = 0.0
                running_accuracy = 0.0
            
            # Evaluation
            if self.global_step % eval_interval == 0:
                print(f"Evaluating at step {self.global_step}...")
                eval_metrics = self.evaluate()
                
                if self.rank == 0:
                    accuracy = eval_metrics.get('accuracy', 0.0)
                    f1_score = eval_metrics.get('macro_f1', 0.0)
                    
                    print(f"Validation - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
                    
                    if self.config.get('use_wandb', True):
                        wandb.log({
                            'val/accuracy': accuracy,
                            'val/macro_f1': f1_score,
                            'val/micro_f1': eval_metrics.get('micro_f1', 0.0),
                            'val/weighted_f1': eval_metrics.get('weighted_f1', 0.0)
                        }, step=self.global_step)
                    
                    # Save best model
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if self.global_step % save_interval == 0:
                self.save_checkpoint()
            
            # Check if training is complete
            if self.global_step >= self.config.training.max_steps:
                break
        
        # Final evaluation and save
        if self.rank == 0:
            print("Training completed! Running final evaluation...")
            final_metrics = self.evaluate()
            
            # Save final model
            self.save_checkpoint()
            
            # Save final metrics
            output_dir = Path(self.config.get('output_dir', 'outputs/email_classification'))
            with open(output_dir / 'final_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            print(f"Final validation accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
            print(f"Best validation accuracy: {self.best_accuracy:.4f}")


@hydra.main(version_base=None, config_path="config", config_name="cfg_email_train")
def main(config: DictConfig):
    """Main training function"""
    
    # Create trainer and start training
    trainer = EmailClassificationTrainer(config)
    trainer.train()
    
    # Cleanup
    if trainer.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()