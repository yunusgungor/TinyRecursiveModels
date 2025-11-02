#!/usr/bin/env python3
"""
Test final trained models with correct configurations
"""
import torch
import numpy as np
from models.tools.tool_enhanced_trm import ToolEnhancedTRM
from models.rl.rl_trm import RLEnhancedTRM
from models.rl.environment import GiftRecommendationEnvironment, UserProfile

def test_rl_model():
    """Test the RL-only model"""
    print("🤖 Testing RL-Only Model")
    print("-" * 40)
    
    checkpoint_path = "checkpoints/rl_gift_recommendation/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # RL model config (from previous training)
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 128,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 1,
        "L_layers": 1,
        "num_heads": 2,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.1,
        "action_space_size": 10,
        "max_recommendations": 2,
        "value_head_hidden": 32,
        "policy_head_hidden": 32,
        "reward_prediction": False,
        "forward_dtype": "float32"
    }
    
    model = RLEnhancedTRM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Episodes Trained: {checkpoint.get('episode_count', 'Unknown')}")
    
    return test_model_performance(model, "RL-Only", has_tools=False)

def test_tool_enhanced_models():
    """Test tool-enhanced models"""
    print("\n🛠️ Testing Tool-Enhanced Models")
    print("-" * 40)
    
    phases = ["phase1", "phase2", "phase3"]
    results = {}
    
    for phase in phases:
        checkpoint_path = f"checkpoints/tool_enhanced_gift_recommendation/{phase}/best_model.pt"
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Debug model config (smaller size)
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
                "reward_prediction": False,  # Phase 1 doesn't have reward head
                "reward_head_hidden": 16,
                "max_tool_calls_per_step": 1,
                "tool_call_threshold": 0.01,
                "tool_result_encoding_dim": 32,
                "tool_selection_method": "confidence",
                "tool_fusion_method": "concatenate",
                "tool_usage_reward_weight": 1.0,
                "forward_dtype": "float32"
            }
            
            model = ToolEnhancedTRM(config)
            
            # Load only compatible state dict keys
            model_state = model.state_dict()
            checkpoint_state = checkpoint["model_state_dict"]
            
            # Filter out incompatible keys
            compatible_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    print(f"  Skipping incompatible key: {key}")
            
            model.load_state_dict(compatible_state, strict=False)
            model.eval()
            
            print(f"\n{phase.upper()} Model:")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  Episodes: {checkpoint.get('episode_count', 'Unknown')}")
            print(f"  Compatible keys loaded: {len(compatible_state)}/{len(checkpoint_state)}")
            
            # Test performance
            has_tools = phase in ["phase2", "phase3"]
            results[phase] = test_model_performance(model, f"Tool-{phase}", has_tools=has_tools)
            
        except Exception as e:
            print(f"❌ Error testing {phase}: {e}")
            results[phase] = None
    
    return results

def test_model_performance(model, model_name, has_tools=False):
    """Test model performance on various scenarios"""
    env = GiftRecommendationEnvironment("data/gift_catalog.json")
    
    # Diverse test scenarios
    test_scenarios = [
        UserProfile(35, ["gardening", "plants"], "mother", 100.0, "birthday", ["eco-conscious"]),
        UserProfile(25, ["technology", "gaming"], "friend", 200.0, "christmas", ["trendy"]),
        UserProfile(50, ["cooking", "reading"], "partner", 150.0, "anniversary", ["practical"]),
        UserProfile(28, ["fitness", "sports"], "sibling", 80.0, "graduation", ["active"]),
        UserProfile(45, ["music", "art"], "colleague", 120.0, "promotion", ["creative"])
    ]
    
    results = {
        "rewards": [],
        "tool_calls": [],
        "recommendations": [],
        "success_rate": 0
    }
    
    print(f"  Testing {model_name} on {len(test_scenarios)} scenarios...")
    
    for i, user in enumerate(test_scenarios):
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
                    new_carry, rl_output, tool_calls = model.forward_with_tools(
                        carry, state, state.available_gifts, max_tool_calls=1
                    )
                    results["tool_calls"].append(len(tool_calls))
                else:
                    rl_output = model.forward_rl(carry, state, state.available_gifts)
                    tool_calls = []
                    results["tool_calls"].append(0)
                
                action = model.select_action(rl_output["action_probs"], state.available_gifts, deterministic=True)
                
                # Get step result
                next_state, reward, done, info = env.step({
                    "recommendations": action["recommendations"],
                    "confidence_scores": action["confidence_scores"]
                })
                
                results["rewards"].append(reward)
                results["recommendations"].append(action["recommendations"])
                
                print(f"    Scenario {i+1}: Reward={reward:.3f}, Tools={len(tool_calls)}")
                
        except Exception as e:
            print(f"    Scenario {i+1}: ❌ Error - {e}")
            results["rewards"].append(0.0)
            results["tool_calls"].append(0)
    
    # Calculate metrics
    if results["rewards"]:
        avg_reward = np.mean(results["rewards"])
        std_reward = np.std(results["rewards"])
        avg_tools = np.mean(results["tool_calls"])
        success_rate = sum(1 for r in results["rewards"] if r > 0.5) / len(results["rewards"])
        
        print(f"  📊 Results:")
        print(f"    Avg Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"    Avg Tool Calls: {avg_tools:.1f}")
        print(f"    Success Rate: {success_rate:.1%}")
        
        results.update({
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_tools": avg_tools,
            "success_rate": success_rate
        })
    
    return results

def compare_models(rl_results, tool_results):
    """Compare RL vs Tool-Enhanced models"""
    print("\n📊 Model Comparison")
    print("=" * 50)
    
    print("Performance Summary:")
    print(f"{'Model':<15} {'Avg Reward':<12} {'Tool Calls':<12} {'Success Rate':<12}")
    print("-" * 55)
    
    if rl_results:
        print(f"{'RL-Only':<15} {rl_results['avg_reward']:<12.3f} {rl_results['avg_tools']:<12.1f} {rl_results['success_rate']:<12.1%}")
    
    for phase, results in tool_results.items():
        if results:
            print(f"{'Tool-' + phase:<15} {results['avg_reward']:<12.3f} {results['avg_tools']:<12.1f} {results['success_rate']:<12.1%}")
    
    # Analysis
    print("\n🔍 Analysis:")
    
    if rl_results and tool_results.get("phase3"):
        phase3_results = tool_results["phase3"]
        reward_improvement = phase3_results["avg_reward"] - rl_results["avg_reward"]
        print(f"  • Tool enhancement reward improvement: {reward_improvement:+.3f}")
        print(f"  • Tool usage effectiveness: {phase3_results['avg_tools']:.1f} calls/episode")
    
    # Tool progression
    tool_phases = ["phase1", "phase2", "phase3"]
    phase_rewards = []
    for phase in tool_phases:
        if tool_results.get(phase):
            phase_rewards.append(tool_results[phase]["avg_reward"])
    
    if len(phase_rewards) >= 2:
        print(f"  • Tool learning progression:")
        for i, (phase, reward) in enumerate(zip(tool_phases[:len(phase_rewards)], phase_rewards)):
            if i > 0:
                improvement = reward - phase_rewards[i-1]
                print(f"    {phase}: {reward:.3f} ({improvement:+.3f})")
            else:
                print(f"    {phase}: {reward:.3f}")

def main():
    print("🧪 Final Model Testing & Analysis")
    print("=" * 60)
    
    # Test RL model
    rl_results = test_rl_model()
    
    # Test tool-enhanced models
    tool_results = test_tool_enhanced_models()
    
    # Compare results
    compare_models(rl_results, tool_results)
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()