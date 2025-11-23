#!/bin/bash

# Sentetik Veri ile Eğitim Scripti
# Bu script, temel veri ile eğitilmiş modeli sentetik veri ile eğitmeye devam eder

echo "🤖 Sentetik Veri ile Eğitim Başlatılıyor..."
echo "================================================"

# Checkpoint'ten devam et (temel veri ile eğitilmiş model)
CHECKPOINT="checkpoints/integrated_enhanced/integrated_enhanced_best.pt"

# Sentetik veri oranı (0.0-1.0)
# 0.5 = %50 sentetik, %50 gerçek veri
# 1.0 = %100 sentetik veri
SYNTHETIC_RATIO=0.4

# Eğitim parametreleri
# Not: --epochs ek olarak eğitilecek epoch sayısıdır (checkpoint'ten sonra)
EPOCHS=100  # Checkpoint'ten sonra 100 epoch daha eğit
BATCH_SIZE=16

# Eğitimi başlat
python scripts/train.py \
    --resume "$CHECKPOINT" \
    --use_synthetic_data \
    --synthetic_ratio $SYNTHETIC_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo ""
echo "✅ Eğitim tamamlandı!"
