#!/usr/bin/env python3
"""
Inspect checkpoint to get correct config
"""
import torch

def inspect_checkpoint():
    checkpoint = torch.load("checkpoints/tool_enhanced_gift_recommendation/phase3/best_model.pt", map_location="cpu")
    
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"  {key}")
    
    if "config" in checkpoint:
        print("\nModel config:")
        config = checkpoint["config"]
        print(f"  Config type: {type(config)}")
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {config}")
    
    print("\nModel state dict shapes (first few):")
    state_dict = checkpoint["model_state_dict"]
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i < 10:
            print(f"  {key}: {tensor.shape}")
        else:
            break

if __name__ == "__main__":
    inspect_checkpoint()