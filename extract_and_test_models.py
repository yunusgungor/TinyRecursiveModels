#!/usr/bin/env python3
"""
Extract correct configs from checkpoints and test models
"""
import torch
import numpy as np
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.rl_trm import RLEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def extract_config_from_checkpoint(checkpoint_path, model_type="rl"):
    """Extract model configuration from checkpoint state dict"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    config = {}
    
    # Basic parameters
    config["batch_size"] = 1
    config["seq_len"] = 50
    config["vocab_size"] = state_dict["inner.embed_tokens.embedding_weight"].shape[0]
    config["num_puzzle_identifiers"] = 1
    config["puzzle_emb_ndim"] = 0
    config["puzzle_emb_len"] = 0
    config["hidden_size"] = state_dict["inner.embed_tokens.embedding_weight"].shape[1]
    config["forward_dtype"] = "float32"
    config["pos_encodings"] = "rope"
    
    # Count layers
    l_layers = 0
    while f"inner.L_level.layers.{l_layers}.self_attn.qkv_proj.weight" in state_dict:
        l_layers += 1
    config["L_layers"] = l_layers
    
    h_layers = 0
    while f"inner.H_level.layers.{h_layers}.self_attn.qkv_proj.weight" in state_dict:
        h_layers += 1
    config["H_layers"] = h_layers
    
    # Infer other parameters
    if l_layers > 0:
        qkv_dim = state_dict["inner.L_level.layers.0.self_attn.qkv_proj.weight"].shape[0]
        config["num_heads"] = qkv_dim // (3 * config["hidden_size"])
        
        mlp_dim = state_dict["inner.L_level.layers.0.mlp.gate_up_proj.weight"].shape[0] // 2
        config["expansion"] = mlp_dim / config["hidden_size"]
    else:
        config["num_heads"] = 1
        config["expansion"] = 2.0
    
    # ACT parameters
    config["halt_max_steps"] = 2
    config["halt_exploration_prob"] = 0.1
    
    # RL parameters
    if "policy_head.0.weight" in state_dict:
        config["policy_head_hidden"] = state_dict["policy_head.0.weight"].shape[0]
        config["action_space_size"] = state_dict["policy_head.3.weight"].shape[0]
    else:
        config["policy_head_hidden"] = 32
        config["action_space_size"] = 10
    
    if "value_head.0.weight" in state_dict:
        config["value_head_hidden"] = state_dict["value_head.0.weight"].shape[0]
    else:
        config["value_head_hidden"] = 32
    
    config["max_recommendations"] = min(5, config["action_space_size"])
    
    # Reward prediction
    config["reward_prediction"] = "reward_head.0.weight" in state_dict
    if config["reward_prediction"]:
        config["reward_head_hidden"] = state_dict["reward_head.0.weight"].shape[0]
    else:
        config["reward_head_hidden"] = 32
    
    # Tool-specific parameters
    if model_type == "tool":
        config["max_tool_calls_per_step"] = 1
        config["tool_call_threshold"] = 0.01
        
        if "tool_result_encoder.0.weight" in state_dict:
            config["tool_result_encoding_dim"] = state_dict["tool_result_encoder.0.weight"].shape[1]
        else:
            config["tool_result_encoding_dim"] = 64
            
        config["tool_selection_method"] = "confidence"
        config["tool_fusion_method"] = "concatenate"
        config["tool_usage_reward_weight"] = 1.0
    
    # Cycles (estimate)
    config["H_cycles"] = 2 if h_layers > 0 else 1
    config["L_cycles"] = 2 if l_layers > 0 else 1
    
    return config

def test_model_with_extracted_config(checkpoint_path, model_type="rl", model_name="Model"):
    """Test model with configuration extracted from checkpoint"""
    print(f"\n🧪 Testing {model_name}")
    print("-" * 40)
    
    try:
        # Extract config
        config = extract_config_from_checkpoint(checkpoint_path, model_type)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        print(f"Extracted Config:")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Layers: H={config['H_layers']}, L={config['L_layers']}")
        print(f"  Action Space: {config['action_space_size']}")
        print(f"  Reward Prediction: {config['reward_prediction']}")
        
        # Create model
        if model_type == "tool":
            model = ToolEnhancedTRM(config)
        else:
            model = RLEnhancedTRM(config)
        
        # Load state dict with error handling
        model_state = model.state_dict()
        checkpoint_state = checkpoint["model_state_dict"]
        
        compatible_state = {}
        incompatible_keys = []
        
        for key, value in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    incompatible_keys.append(f"{key}: {value.shape} vs {model_state[key].shape}")
            else:
                incompatible_keys.append(f"{key}: not in model")
        
        if incompatible_keys:
            print(f"  Incompatible keys: {len(incompatible_keys)}")
            for key in incompatible_keys[:3]:  # Show first 3
                print(f"    {key}")
            if len(incompatible_keys) > 3:
                print(f"    ... and {len(incompatible_keys) - 3} more")
        
        model.load_state_dict(compatible_state, strict=False)
        model.eval()
        
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Episodes: {checkpoint.get('episode_count', 'Unknown')}")
        
        # Test performance
        return test_model_performance(model, model_name, has_tools=(model_type=="tool"))
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def test_model_performance(model, model_name, has_tools=False):
    """Test model performance"""
    env = GiftRecommendationEnvironment("data/gift_catalog.json")
    
    test_users = [
        UserProfile(35, ["gardening"], "mother", 100.0, "birthday", ["eco-conscious"]),
        UserProfile(25, ["technology"], "friend", 200.0, "christmas", ["trendy"]),
        UserProfile(50, ["cooking"], "partner", 150.0, "anniversary", ["practical"])
    ]
    
    results = {"rewards": [], "tool_calls": []}
    
    for i, user in enumerate(test_users):
        try:
            state = env.reset(user)
            
            # Create carry
            dummy_batch = {
                "inputs": torch.randint(0, 1000, (50,)),
                "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
            }
            carry = model.initial_carry(dummy_batch)
            
            with torch.no_grad():
                if has_tools and hasattr(model, 'forward_with_tools'):
                    try:
                        new_carry, rl_output, tool_calls = model.forward_with_tools(
                            carry, state, state.available_gifts, max_tool_calls=1
                        )
                        results["tool_calls"].append(len(tool_calls))
                    except:
                        # Fallback to regular RL if tools fail
                        rl_output = model.forward_rl(carry, state, state.available_gifts)
                        tool_calls = []
                        results["tool_calls"].append(0)
                else:
                    rl_output = model.forward_rl(carry, state, state.available_gifts)
                    results["tool_calls"].append(0)
                
                action = model.select_action(rl_output["action_probs"], state.available_gifts, deterministic=True)
                
                next_state, reward, done, info = env.step({
                    "recommendations": action["recommendations"],
                    "confidence_scores": action["confidence_scores"]
                })
                
                results["rewards"].append(reward)
                print(f"    User {i+1}: Reward={reward:.3f}, Tools={results['tool_calls'][-1]}")
                
        except Exception as e:
            print(f"    User {i+1}: ❌ Error - {e}")
            results["rewards"].append(0.0)
            results["tool_calls"].append(0)
    
    # Calculate metrics
    if results["rewards"]:
        avg_reward = np.mean(results["rewards"])
        avg_tools = np.mean(results["tool_calls"])
        
        print(f"  📊 Summary: Reward={avg_reward:.3f}, Tools={avg_tools:.1f}")
        
        return {
            "avg_reward": avg_reward,
            "avg_tools": avg_tools,
            "rewards": results["rewards"],
            "tool_calls": results["tool_calls"]
        }
    
    return None

def main():
    print("🔍 Model Analysis with Extracted Configurations")
    print("=" * 60)
    
    results = {}
    
    # Test RL model
    rl_checkpoint = "checkpoints/rl_gift_recommendation/best_model.pt"
    if torch.load(rl_checkpoint, map_location="cpu"):
        results["rl"] = test_model_with_extracted_config(rl_checkpoint, "rl", "RL-Only Model")
    
    # Test tool-enhanced models
    phases = ["phase1", "phase2", "phase3"]
    for phase in phases:
        checkpoint_path = f"checkpoints/tool_enhanced_gift_recommendation/{phase}/best_model.pt"
        try:
            results[phase] = test_model_with_extracted_config(
                checkpoint_path, "tool", f"Tool-Enhanced {phase.upper()}"
            )
        except:
            print(f"❌ Could not test {phase}")
    
    # Summary comparison
    print("\n📊 Final Comparison")
    print("=" * 50)
    print(f"{'Model':<20} {'Avg Reward':<12} {'Tool Usage':<12}")
    print("-" * 45)
    
    for name, result in results.items():
        if result:
            print(f"{name:<20} {result['avg_reward']:<12.3f} {result['avg_tools']:<12.1f}")
    
    # Key insights
    print("\n🎯 Key Insights:")
    
    if results.get("rl") and results.get("phase3"):
        improvement = results["phase3"]["avg_reward"] - results["rl"]["avg_reward"]
        print(f"  • Tool enhancement improvement: {improvement:+.3f}")
    
    if results.get("phase2") and results.get("phase2")["avg_tools"] > 0:
        print(f"  • Tool usage successfully activated in Phase 2")
    
    if results.get("phase3") and results.get("phase3")["avg_tools"] > 0:
        print(f"  • Tool usage maintained in Phase 3 (RL fine-tuning)")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()