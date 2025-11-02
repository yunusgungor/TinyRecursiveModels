#!/usr/bin/env python3
"""
Extract exact model config from checkpoint
"""
import torch

def extract_config():
    checkpoint = torch.load("checkpoints/tool_enhanced_gift_recommendation/phase3/best_model.pt", map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Infer config from state dict shapes
    config = {}
    
    # Hidden size from embedding
    config["hidden_size"] = state_dict["inner.embed_tokens.embedding_weight"].shape[1]
    
    # Vocab size
    config["vocab_size"] = state_dict["inner.embed_tokens.embedding_weight"].shape[0]
    
    # Number of heads from attention projection
    qkv_dim = state_dict["inner.L_level.layers.0.self_attn.qkv_proj.weight"].shape[0]
    config["num_heads"] = qkv_dim // (3 * config["hidden_size"])
    
    # Policy head hidden
    config["policy_head_hidden"] = state_dict["policy_head.0.weight"].shape[0]
    
    # Value head hidden  
    config["value_head_hidden"] = state_dict["value_head.0.weight"].shape[0]
    
    # Action space size
    config["action_space_size"] = state_dict["policy_head.3.weight"].shape[0]
    
    # Reward head hidden
    config["reward_head_hidden"] = state_dict["reward_head.0.weight"].shape[0]
    
    # Tool result encoding dim
    config["tool_result_encoding_dim"] = state_dict["tool_result_encoder.0.weight"].shape[1]
    
    # Count layers
    l_layers = 0
    while f"inner.L_level.layers.{l_layers}.self_attn.qkv_proj.weight" in state_dict:
        l_layers += 1
    config["L_layers"] = l_layers
    
    h_layers = 0
    while f"inner.H_level.layers.{h_layers}.self_attn.qkv_proj.weight" in state_dict:
        h_layers += 1
    config["H_layers"] = h_layers
    
    print("Extracted config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create complete config
    complete_config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": config["vocab_size"],
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": config["hidden_size"],
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": config["H_layers"],
        "L_layers": config["L_layers"],
        "num_heads": config["num_heads"],
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 2,
        "halt_exploration_prob": 0.1,
        "action_space_size": config["action_space_size"],
        "max_recommendations": 2,
        "value_head_hidden": config["value_head_hidden"],
        "policy_head_hidden": config["policy_head_hidden"],
        "reward_prediction": True,
        "reward_head_hidden": config["reward_head_hidden"],
        "max_tool_calls_per_step": 1,
        "tool_call_threshold": 0.1,
        "tool_result_encoding_dim": config["tool_result_encoding_dim"],
        "tool_selection_method": "confidence",
        "tool_fusion_method": "concatenate",
        "tool_usage_reward_weight": 0.1,
        "forward_dtype": "float32"
    }
    
    print("\nComplete config:")
    for key, value in complete_config.items():
        print(f"        \"{key}\": {value},")

if __name__ == "__main__":
    extract_config()