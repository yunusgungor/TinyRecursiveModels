#!/bin/bash

# Resume training from best checkpoint (Epoch 30)
# This script stops current training and resumes from the best model

echo "🛑 Stopping current training..."
echo "📂 Looking for best checkpoint..."

# Find the best checkpoint
BEST_CHECKPOINT="checkpoints/integrated_enhanced/integrated_enhanced_best.pt"

if [ -f "$BEST_CHECKPOINT" ]; then
    echo "✅ Found best checkpoint: $BEST_CHECKPOINT"
    echo "🔄 Resuming training with optimized hyperparameters..."
    
    # Resume training from best checkpoint
    python scripts/train.py \
        --resume "$BEST_CHECKPOINT" \
        --epochs 150 \
        --batch_size 16
    
    echo "🎉 Training resumed successfully!"
else
    echo "❌ Best checkpoint not found at $BEST_CHECKPOINT"
    echo "💡 Starting fresh training with optimized hyperparameters..."
    
    # Start fresh training
    python scripts/train.py \
        --epochs 150 \
        --batch_size 16
fi
