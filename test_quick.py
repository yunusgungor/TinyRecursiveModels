#!/usr/bin/env python3
"""
Quick Test Script - Fast sanity checks
Run this for quick verification
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_integrated_enhanced_model import IntegratedEnhancedTrainer
from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import UserProfile


def quick_test():
    """Quick sanity check"""
    print("🚀 Quick Test - Sanity Checks")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device: {device}")
    
    # Test 1: Model creation
    print("\n1️⃣  Testing model creation...")
    config = create_integrated_enhanced_config()
    config['batch_size'] = 1
    model = IntegratedEnhancedTRM(config).to(device)
    print("   ✅ Model created")
    
    # Test 2: Trainer creation
    print("\n2️⃣  Testing trainer creation...")
    trainer = IntegratedEnhancedTrainer(config)
    print("   ✅ Trainer created")
    
    # Test 3: Forward pass
    print("\n3️⃣  Testing forward pass...")
    user = UserProfile(30, ['technology'], 'friend', 150.0, 'birthday', ['trendy'])
    env_state = trainer.env.reset(user)
    carry = model.initial_carry({
        "inputs": torch.zeros(1, 10, device=device),
        "puzzle_identifiers": torch.zeros(1, 1, device=device)
    })
    
    _, output, tools = model.forward_with_enhancements(carry, env_state, trainer.env.gift_catalog)
    print(f"   ✅ Forward pass completed")
    print(f"   📊 Selected tools: {tools}")
    
    # Test 4: Tool parameters
    print("\n4️⃣  Testing tool parameters...")
    assert 'tool_params' in output, "❌ tool_params missing"
    print(f"   ✅ Tool params generated: {list(output['tool_params'].keys())}")
    
    # Test 5: forward_with_tools
    print("\n5️⃣  Testing forward_with_tools...")
    carry, output, tool_calls = model.forward_with_tools(
        carry, env_state, trainer.env.gift_catalog, max_tool_calls=2
    )
    print(f"   ✅ Executed {len(tool_calls)} tools")
    
    # Test 6: Tool statistics
    print("\n6️⃣  Testing tool statistics...")
    stats = model.get_tool_usage_stats()
    print(f"   ✅ Total tool calls: {stats['total_calls']}")
    
    # Test 7: Checkpoint save/load
    print("\n7️⃣  Testing checkpoint save/load...")
    trainer.save_model("quick_test.pt", epoch=0, metrics={'test': 1.0})
    trainer2 = IntegratedEnhancedTrainer(config)
    trainer2.load_model("checkpoints/integrated_enhanced/quick_test.pt")
    print("   ✅ Checkpoint save/load works")
    os.remove("checkpoints/integrated_enhanced/quick_test.pt")
    
    # Test 8: Training batch
    print("\n8️⃣  Testing training batch...")
    users, gifts, targets = trainer.generate_training_batch(batch_size=2)
    print(f"   ✅ Generated batch of {len(users)} users")
    
    print("\n" + "="*60)
    print("🎉 ALL QUICK TESTS PASSED!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
