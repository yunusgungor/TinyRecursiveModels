#!/bin/bash

# Integrated Enhanced TRM Training Script
# Tüm iyileştirmeler uygulandı - v4.1

echo "🚀 Starting Integrated Enhanced TRM Training..."
echo "📊 Configuration:"
echo "  - Epochs: 150"
echo "  - Batch Size: 16"
echo "  - Learning Rates: Optimized (2-4x increased)"
echo "  - Loss Weights: Balanced"
echo "  - Curriculum: Accelerated (10/25/45)"
echo "  - Gradient Clipping: 2.0"
echo ""

python train_integrated_enhanced_model.py \
    --epochs 150 \
    --batch_size 16

echo ""
echo "✅ Training completed!"
