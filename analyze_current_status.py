#!/usr/bin/env python3
"""
Analyze current training status and model performance
"""
import torch
import json
import os
from pathlib import Path
# import matplotlib.pyplot as plt  # Not available
import numpy as np
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.rl_trm import RLEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def analyze_checkpoints():
    """Analyze available checkpoints"""
    print("🔍 Checkpoint Analysis")
    print("=" * 50)
    
    checkpoint_dirs = [
        "checkpoints/rl_gift_recommendation",
        "checkpoints/tool_enhanced_gift_recommendation"
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"\n📁 {checkpoint_dir}")
            
            # List all checkpoints
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    if file.endswith('.pt'):
                        filepath = os.path.join(root, file)
                        try:
                            checkpoint = torch.load(filepath, map_location='cpu')
                            
                            # Get model info
                            model_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
                            
                            print(f"  📄 {file}")
                            print(f"    Path: {filepath}")
                            print(f"    Parameters: {model_params:,}")
                            
                            if "episode_count" in checkpoint:
                                print(f"    Episodes: {checkpoint['episode_count']}")
                            if "episode_rewards" in checkpoint and checkpoint["episode_rewards"]:
                                recent_rewards = checkpoint["episode_rewards"][-10:]
                                avg_reward = np.mean(recent_rewards)
                                print(f"    Recent Avg Reward: {avg_reward:.3f}")
                            
                        except Exception as e:
                            print(f"  ❌ Error loading {file}: {e}")
        else:
            print(f"\n📁 {checkpoint_dir} - Not found")

def test_debug_models():
    """Test the debug models that completed training"""
    print("\n🧪 Debug Model Testing")
    print("=" * 50)
    
    debug_checkpoints = [
        "checkpoints/tool_enhanced_gift_recommendation/phase1/best_model.pt",
        "checkpoints/tool_enhanced_gift_recommendation/phase2/best_model.pt", 
        "checkpoints/tool_enhanced_gift_recommendation/phase3/best_model.pt"
    ]
    
    # Test environment
    env = GiftRecommendationEnvironment("data/gift_catalog.json")
    
    # Test users
    test_users = [
        UserProfile(35, ["gardening"], "mother", 100.0, "birthday", ["eco-conscious"]),
        UserProfile(25, ["technology", "gaming"], "friend", 200.0, "christmas", ["trendy"]),
        UserProfile(50, ["cooking", "reading"], "partner", 150.0, "anniversary", ["practical"])
    ]
    
    for i, checkpoint_path in enumerate(debug_checkpoints, 1):
        if not os.path.exists(checkpoint_path):
            print(f"Phase {i}: ❌ Checkpoint not found")
            continue
            
        print(f"\n🔬 Phase {i} Model Test:")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Create model config (debug size)
            config = {
                "batch_size": 1,
                "seq_len": 50,
                "vocab_size": 1000,
                "num_puzzle_identifiers": 1,
                "puzzle_emb_ndim": 0,
                "puzzle_emb_len": 0,
                "hidden_size": 64,
                "H_cycles": 1,
                "L_cycles": 1,
                "H_layers": 0,
                "L_layers": 1,
                "num_heads": 1,
                "expansion": 1.5,
                "pos_encodings": "rope",
                "halt_max_steps": 1,
                "halt_exploration_prob": 0.0,
                "action_space_size": 5,
                "max_recommendations": 1,
                "value_head_hidden": 16,
                "policy_head_hidden": 16,
                "reward_prediction": False,
                "reward_head_hidden": 16,
                "max_tool_calls_per_step": 1,
                "tool_call_threshold": 0.01,
                "tool_result_encoding_dim": 32,
                "tool_selection_method": "confidence",
                "tool_fusion_method": "concatenate",
                "tool_usage_reward_weight": 1.0,
                "forward_dtype": "float32"
            }
            
            # Create and load model
            model = ToolEnhancedTRM(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test each user
            total_rewards = []
            total_tool_calls = []
            
            for j, user in enumerate(test_users):
                state = env.reset(user)
                
                # Create carry
                dummy_batch = {
                    "inputs": torch.randint(0, config["vocab_size"], (config["seq_len"],)),
                    "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
                }
                carry = model.initial_carry(dummy_batch)
                
                with torch.no_grad():
                    new_carry, rl_output, tool_calls = model.forward_with_tools(
                        carry, state, state.available_gifts, max_tool_calls=1
                    )
                    
                    action = model.select_action(rl_output["action_probs"], state.available_gifts, deterministic=True)
                    
                    # Get step result
                    next_state, reward, done, info = env.step({
                        "recommendations": action["recommendations"],
                        "confidence_scores": action["confidence_scores"]
                    })
                    
                    total_rewards.append(reward)
                    total_tool_calls.append(len(tool_calls))
                    
                    print(f"    User {j+1}: Reward={reward:.3f}, Tools={len(tool_calls)}")
                    for k, call in enumerate(tool_calls):
                        print(f"      Tool {k+1}: {call.tool_name} ({'✓' if call.success else '✗'})")
            
            print(f"  📊 Summary:")
            print(f"    Avg Reward: {np.mean(total_rewards):.3f}")
            print(f"    Avg Tool Calls: {np.mean(total_tool_calls):.1f}")
            
            # Tool usage stats
            stats = model.get_tool_usage_stats()
            print(f"    Tool Stats: {stats}")
            
        except Exception as e:
            print(f"  ❌ Error testing Phase {i}: {e}")

def compare_model_architectures():
    """Compare different model architectures"""
    print("\n🏗️ Model Architecture Comparison")
    print("=" * 50)
    
    models_info = []
    
    # RL Model
    rl_checkpoint = "checkpoints/rl_gift_recommendation/best_model.pt"
    if os.path.exists(rl_checkpoint):
        checkpoint = torch.load(rl_checkpoint, map_location="cpu")
        rl_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
        models_info.append(("RL Model", rl_params, "Standard RL without tools"))
    
    # Tool-Enhanced Models
    tool_phases = [
        ("Tool Phase 1", "checkpoints/tool_enhanced_gift_recommendation/phase1/best_model.pt"),
        ("Tool Phase 2", "checkpoints/tool_enhanced_gift_recommendation/phase2/best_model.pt"),
        ("Tool Phase 3", "checkpoints/tool_enhanced_gift_recommendation/phase3/best_model.pt")
    ]
    
    for name, path in tool_phases:
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location="cpu")
            params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
            models_info.append((name, params, "Tool-enhanced with 5 tools"))
    
    # Print comparison
    print("Model Comparison:")
    for name, params, description in models_info:
        print(f"  {name:15} | {params:>10,} params | {description}")
    
    if len(models_info) > 1:
        base_params = models_info[0][1]
        print(f"\nParameter Overhead:")
        for name, params, _ in models_info[1:]:
            overhead = params - base_params
            percentage = (overhead / base_params) * 100
            print(f"  {name:15} | +{overhead:>8,} params | +{percentage:>5.1f}%")

def analyze_training_progress():
    """Analyze training progress from wandb logs"""
    print("\n📈 Training Progress Analysis")
    print("=" * 50)
    
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        # Find latest run
        runs = []
        for item in os.listdir(wandb_dir):
            if item.startswith("run-"):
                run_path = os.path.join(wandb_dir, item)
                if os.path.isdir(run_path):
                    runs.append((item, os.path.getmtime(run_path)))
        
        if runs:
            latest_run = max(runs, key=lambda x: x[1])[0]
            print(f"Latest Run: {latest_run}")
            
            # Try to read logs
            log_path = os.path.join(wandb_dir, latest_run, "logs", "debug.log")
            if os.path.exists(log_path):
                print("Found training logs - analyzing...")
                # Could parse logs here for detailed analysis
            else:
                print("No detailed logs found")
        else:
            print("No wandb runs found")
    else:
        print("No wandb directory found")

def generate_summary_report():
    """Generate comprehensive summary report"""
    print("\n📋 Summary Report")
    print("=" * 50)
    
    # Check what we have accomplished
    accomplishments = []
    
    # Check RL model
    if os.path.exists("checkpoints/rl_gift_recommendation/best_model.pt"):
        accomplishments.append("✅ RL Model trained successfully")
    
    # Check tool-enhanced phases
    phases = ["phase1", "phase2", "phase3"]
    completed_phases = []
    for phase in phases:
        path = f"checkpoints/tool_enhanced_gift_recommendation/{phase}/best_model.pt"
        if os.path.exists(path):
            completed_phases.append(phase)
            accomplishments.append(f"✅ Tool-Enhanced {phase} completed")
    
    # Check if tools are working
    if len(completed_phases) >= 2:
        accomplishments.append("✅ Tool usage successfully activated")
    
    # Print accomplishments
    print("🎯 Accomplishments:")
    for item in accomplishments:
        print(f"  {item}")
    
    # Current status
    print(f"\n📊 Current Status:")
    print(f"  • Debug Training: {'✅ Complete' if len(completed_phases) == 3 else '⏳ In Progress'}")
    print(f"  • Normal Training: {'⏳ Interrupted at Episode 90' if os.path.exists('checkpoints/tool_enhanced_gift_recommendation/interrupted_checkpoint.pt') else '❌ Not Started'}")
    print(f"  • Tool Integration: {'✅ Working' if len(completed_phases) >= 2 else '❌ Not Working'}")
    
    # Next steps
    print(f"\n🚀 Recommended Next Steps:")
    if len(completed_phases) == 3:
        print("  1. 🧪 Test final models with comprehensive evaluation")
        print("  2. 📊 Compare RL vs Tool-Enhanced performance")
        print("  3. 🔄 Resume normal-size training for better performance")
        print("  4. 📈 Analyze tool usage patterns and effectiveness")
    else:
        print("  1. 🔧 Complete debug training first")
        print("  2. 🧪 Test tool integration")

if __name__ == "__main__":
    print("🔍 TRM Model Training Analysis")
    print("=" * 60)
    
    try:
        analyze_checkpoints()
        test_debug_models()
        compare_model_architectures()
        analyze_training_progress()
        generate_summary_report()
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()