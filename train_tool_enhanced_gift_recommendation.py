#!/usr/bin/env python3
"""
Training script for Tool-Enhanced Gift Recommendation TRM
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

from models.tools.tool_enhanced_trm import ToolEnhancedTRM
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
        "seq_len": 50,
        "vocab_size": 1000,
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
        
        # Tool parameters
        "max_tool_calls_per_step": arch_config["max_tool_calls_per_step"],
        "tool_call_threshold": arch_config["tool_call_threshold"],
        "tool_result_encoding_dim": arch_config["tool_result_encoding_dim"],
        "tool_selection_method": arch_config["tool_selection_method"],
        "tool_fusion_method": arch_config["tool_fusion_method"],
        "tool_attention_heads": arch_config["tool_attention_heads"],
        "tool_usage_reward_weight": arch_config["tool_usage_reward_weight"],
        "tool_efficiency_penalty": arch_config["tool_efficiency_penalty"],
        
        # Device settings
        "forward_dtype": config_dict.get("forward_dtype", "float32")
    }
    
    return model_config


def create_training_config(config_dict, phase="full"):
    """Create training configuration from config file"""
    rl_config = config_dict["rl_training"]
    
    # Adjust parameters based on training phase
    if phase in config_dict.get("training_phases", {}):
        phase_config = config_dict["training_phases"][phase]
        enable_tools = phase_config.get("enable_tools", True)
        enable_rl = phase_config.get("enable_rl", True)
        epochs = phase_config.get("epochs", rl_config["num_episodes"])
    else:
        enable_tools = rl_config.get("enable_tools", True)
        enable_rl = True
        epochs = rl_config["num_episodes"]
    
    training_config = TrainingConfig(
        num_episodes=epochs,
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
        enable_tools=enable_tools,
        tool_usage_reward_weight=rl_config.get("tool_usage_reward_weight", 0.1)
    )
    
    return training_config


def run_training_phase(model, env, config_dict, phase_name, previous_checkpoint=None):
    """Run a specific training phase"""
    print(f"\n{'='*20} {phase_name.upper()} {'='*20}")
    
    # Create training config for this phase
    training_config = create_training_config(config_dict, phase_name)
    
    # Update checkpoint directory for this phase
    training_config.checkpoint_dir = os.path.join(
        config_dict["checkpoint_path"], 
        phase_name
    )
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # Create trainer
    trainer = RLTrainer(model, env, training_config)
    
    # Load previous checkpoint if provided
    if previous_checkpoint:
        print(f"Loading checkpoint from previous phase: {previous_checkpoint}")
        trainer.load_checkpoint(previous_checkpoint)
    
    # Run training
    phase_summary = trainer.train()
    
    # Return path to best model from this phase
    return phase_summary.get("best_model_path")


def main():
    parser = argparse.ArgumentParser(description="Train Tool-Enhanced Gift Recommendation TRM")
    parser.add_argument("--config", type=str, default="config/tool_enhanced_gift_recommendation.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--phase", type=str, default="all",
                       choices=["all", "phase1", "phase2", "phase3", "supervised", "tools", "rl"],
                       help="Training phase to run")
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
        
        # Reduce phase durations
        if "training_phases" in config:
            for phase_config in config["training_phases"].values():
                phase_config["epochs"] = min(50, phase_config.get("epochs", 100))
    
    # Setup data files
    catalog_path = setup_data_files()
    config["environment"]["gift_catalog_path"] = catalog_path
    
    # Create model configuration
    model_config = create_model_config(config)
    
    # Initialize Weights & Biases
    if args.wandb:
        wandb.init(
            project=config.get("wandb_project", "gift-recommendation-tools"),
            name=config.get("run_name", f"tool_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config={
                "model_config": model_config,
                "file_config": config,
                "training_phase": args.phase
            }
        )
    
    # Create model
    print("Creating Tool-Enhanced TRM model...")
    model = ToolEnhancedTRM(model_config)
    
    if device == "cuda":
        model = model.cuda()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Available tools: {model.tool_registry.list_tools()}")
    
    # Create environment
    print("Creating gift recommendation environment...")
    env = GiftRecommendationEnvironment(
        gift_catalog_path=config["environment"]["gift_catalog_path"],
        user_feedback_data_path=config["environment"].get("user_feedback_path")
    )
    
    try:
        if args.phase == "all" and "training_phases" in config:
            # Multi-phase training
            print("Starting multi-phase training...")
            
            previous_checkpoint = args.resume
            phase_results = {}
            
            for phase_name, phase_config in config["training_phases"].items():
                checkpoint_path = run_training_phase(
                    model, env, config, phase_name, previous_checkpoint
                )
                phase_results[phase_name] = checkpoint_path
                previous_checkpoint = checkpoint_path
                
                print(f"Phase {phase_name} completed. Best model: {checkpoint_path}")
            
            print("\n" + "="*50)
            print("MULTI-PHASE TRAINING COMPLETED!")
            print("="*50)
            for phase, path in phase_results.items():
                print(f"{phase}: {path}")
        
        else:
            # Single phase training
            phase_name = args.phase if args.phase != "all" else "single_phase"
            
            training_config = create_training_config(config, phase_name)
            
            print(f"Starting single-phase training: {phase_name}")
            print(f"Training configuration:")
            print(f"  Episodes: {training_config.num_episodes}")
            print(f"  Batch size: {training_config.batch_size}")
            print(f"  Learning rate: {training_config.learning_rate}")
            print(f"  Tools enabled: {training_config.enable_tools}")
            print(f"  Max tool calls per step: {model_config.get('max_tool_calls_per_step', 0)}")
            
            # Create trainer
            trainer = RLTrainer(model, env, training_config)
            
            # Resume from checkpoint if specified
            if args.resume:
                print(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            
            # Start training
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
            
            # Show tool usage statistics
            if training_config.enable_tools:
                tool_stats = model.get_tool_usage_stats()
                print(f"\nTool Usage Statistics:")
                print(f"  Total tool calls: {tool_stats.get('total_calls', 0)}")
                print(f"  Most used tool: {tool_stats.get('most_used_tool', 'None')}")
                print(f"  Average execution time: {tool_stats.get('average_execution_time', 0):.3f}s")
            
            # Log final results to wandb
            if args.wandb:
                wandb.log({
                    "final/total_episodes": training_summary['total_episodes'],
                    "final/training_time": training_summary['total_training_time'],
                    "final/eval_reward": training_summary['final_eval_reward'],
                    "final/best_reward": training_summary['best_reward']
                })
                
                if training_config.enable_tools:
                    wandb.log({
                        "final/tool_calls": tool_stats.get('total_calls', 0),
                        "final/tool_avg_time": tool_stats.get('average_execution_time', 0)
                    })
                
                # Save model artifact
                if training_summary['best_model_path']:
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(training_summary['best_model_path'])
                    wandb.log_artifact(artifact)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current state
        checkpoint_path = os.path.join(
            config["checkpoint_path"], 
            "interrupted_checkpoint.pt"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "timestamp": datetime.now().isoformat()
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save current state for debugging
        try:
            error_checkpoint_path = os.path.join(
                config["checkpoint_path"], 
                "error_checkpoint.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, error_checkpoint_path)
            print(f"Error checkpoint saved to {error_checkpoint_path}")
        except:
            print("Could not save error checkpoint")
    
    finally:
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()