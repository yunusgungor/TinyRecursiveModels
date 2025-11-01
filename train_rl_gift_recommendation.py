#!/usr/bin/env python3
"""
Training script for RL-enhanced Gift Recommendation TRM
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.rl.rl_trm import RLEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, create_sample_gift_catalog
from models.rl.trainer import RLTrainer, TrainingConfig


def setup_data_files():
    """Setup required data files if they don't exist"""
    os.makedirs("data", exist_ok=True)
    
    # Create sample gift catalog if it doesn't exist
    catalog_path = "data/gift_catalog.json"
    if not os.path.exists(catalog_path):
        print("Creating sample gift catalog...")
        create_sample_gift_catalog(catalog_path)
        print(f"Sample gift catalog created at {catalog_path}")
    
    return catalog_path


def create_model_config(config_dict):
    """Create model configuration from config file"""
    arch_config = config_dict["arch"]
    
    model_config = {
        # Base TRM parameters
        "batch_size": config_dict["global_batch_size"],
        "seq_len": 50,  # Fixed for gift recommendation
        "vocab_size": 1000,  # Fixed vocabulary size
        "num_puzzle_identifiers": 1,
        "hidden_size": arch_config["hidden_size"],
        "L_layers": arch_config["L_layers"],
        "H_cycles": arch_config["H_cycles"],
        "L_cycles": arch_config["L_cycles"],
        "num_heads": arch_config["num_heads"],
        "expansion": arch_config["expansion"],
        "pos_encodings": arch_config["pos_encodings"],
        "rms_norm_eps": arch_config["rms_norm_eps"],
        "rope_theta": arch_config["rope_theta"],
        
        # ACT parameters
        "halt_max_steps": arch_config["halt_max_steps"],
        "halt_exploration_prob": arch_config["halt_exploration_prob"],
        "no_ACT_continue": arch_config["no_ACT_continue"],
        
        # RL parameters
        "action_space_size": arch_config["action_space_size"],
        "max_recommendations": arch_config["max_recommendations"],
        "value_head_hidden": arch_config["value_head_hidden"],
        "policy_head_hidden": arch_config["policy_head_hidden"],
        "reward_prediction": arch_config["reward_prediction"],
        "reward_head_hidden": arch_config["reward_head_hidden"],
        "action_selection_method": arch_config["action_selection_method"],
        "epsilon": arch_config["epsilon"],
        "temperature": arch_config["temperature"],
        "ppo_clip_ratio": arch_config["ppo_clip_ratio"],
        "value_loss_coef": arch_config["value_loss_coef"],
        "entropy_coef": arch_config["entropy_coef"],
        "max_grad_norm": arch_config["max_grad_norm"],
        
        # Device settings
        "forward_dtype": config_dict.get("forward_dtype", "float32")
    }
    
    return model_config


def create_training_config(config_dict):
    """Create training configuration from config file"""
    rl_config = config_dict["rl_training"]
    
    training_config = TrainingConfig(
        num_episodes=rl_config["num_episodes"],
        max_steps_per_episode=rl_config["max_steps_per_episode"],
        batch_size=rl_config["batch_size"],
        learning_rate=config_dict["lr"],
        gamma=rl_config["gamma"],
        ppo_epochs=rl_config["ppo_epochs"],
        clip_ratio=config_dict["arch"]["ppo_clip_ratio"],
        value_loss_coef=config_dict["arch"]["value_loss_coef"],
        entropy_coef=config_dict["arch"]["entropy_coef"],
        max_grad_norm=config_dict["arch"]["max_grad_norm"],
        experience_buffer_size=rl_config["experience_buffer_size"],
        min_experiences_for_update=rl_config["min_experiences_for_update"],
        eval_frequency=rl_config["eval_frequency"],
        eval_episodes=rl_config["eval_episodes"],
        log_frequency=config_dict["log_frequency"],
        save_frequency=config_dict["save_frequency"],
        checkpoint_dir=config_dict["checkpoint_path"],
        enable_tools=False  # RL-only training
    )
    
    return training_config


def main():
    parser = argparse.ArgumentParser(description="Train RL-enhanced Gift Recommendation TRM")
    parser.add_argument("--config", type=str, default="config/rl_gift_recommendation.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (smaller model, fewer episodes)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Debug mode adjustments
    if args.debug:
        print("Debug mode enabled - reducing model size and training duration")
        config["arch"]["hidden_size"] = 128
        config["arch"]["L_layers"] = 1
        config["arch"]["H_cycles"] = 2
        config["arch"]["L_cycles"] = 2
        config["rl_training"]["num_episodes"] = 100
        config["rl_training"]["batch_size"] = 8
        config["rl_training"]["eval_frequency"] = 20
        config["global_batch_size"] = 4
    
    # Setup data files
    catalog_path = setup_data_files()
    config["environment"]["gift_catalog_path"] = catalog_path
    
    # Create model configuration
    model_config = create_model_config(config)
    
    # Create training configuration
    training_config = create_training_config(config)
    
    # Initialize Weights & Biases
    if args.wandb:
        wandb.init(
            project=config.get("project_name", "gift-recommendation-rl"),
            name=config.get("run_name", f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config={
                "model_config": model_config,
                "training_config": training_config.__dict__,
                "file_config": config
            }
        )
    
    # Create model
    print("Creating RL-enhanced TRM model...")
    model = RLEnhancedTRM(model_config)
    
    if device == "cuda":
        model = model.cuda()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create environment
    print("Creating gift recommendation environment...")
    env = GiftRecommendationEnvironment(
        gift_catalog_path=config["environment"]["gift_catalog_path"],
        user_feedback_data_path=config["environment"].get("user_feedback_path")
    )
    
    # Create trainer
    print("Creating RL trainer...")
    trainer = RLTrainer(model, env, training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting RL training...")
    print(f"Training configuration:")
    print(f"  Episodes: {training_config.num_episodes}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps per episode: {training_config.max_steps_per_episode}")
    print(f"  Evaluation frequency: {training_config.eval_frequency}")
    
    try:
        training_summary = trainer.train()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Total episodes: {training_summary['total_episodes']}")
        print(f"Training time: {training_summary['total_training_time']:.2f} seconds")
        print(f"Final evaluation reward: {training_summary['final_eval_reward']:.3f}")
        print(f"Best reward achieved: {training_summary['best_reward']:.3f}")
        
        if training_summary['best_model_path']:
            print(f"Best model saved at: {training_summary['best_model_path']}")
        
        # Log final results to wandb
        if args.wandb:
            wandb.log({
                "final/total_episodes": training_summary['total_episodes'],
                "final/training_time": training_summary['total_training_time'],
                "final/eval_reward": training_summary['final_eval_reward'],
                "final/best_reward": training_summary['best_reward']
            })
            
            # Save model artifact
            if training_summary['best_model_path']:
                artifact = wandb.Artifact("best_model", type="model")
                artifact.add_file(training_summary['best_model_path'])
                wandb.log_artifact(artifact)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        checkpoint_path = trainer.save_checkpoint(
            os.path.join(training_config.checkpoint_dir, "interrupted_checkpoint.pt")
        )
        print(f"Checkpoint saved to {checkpoint_path}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save current state for debugging
        try:
            checkpoint_path = trainer.save_checkpoint(
                os.path.join(training_config.checkpoint_dir, "error_checkpoint.pt")
            )
            print(f"Error checkpoint saved to {checkpoint_path}")
        except:
            print("Could not save error checkpoint")
    
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()