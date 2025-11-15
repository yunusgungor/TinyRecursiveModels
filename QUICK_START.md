# Quick Start Guide - Tool-Enhanced Training

## 🚀 Hızlı Başlangıç

### 1. Test Et

Önce tüm özelliklerin çalıştığından emin ol:

```bash
python test_tool_integration.py
```

Beklenen çıktı:
```
🎉 ALL TESTS PASSED! 🎉
5/5 tests passed
```

### 2. Training Başlat

```bash
# Sıfırdan training
python train_integrated_enhanced_model.py

# Custom ayarlarla
python train_integrated_enhanced_model.py --epochs 200 --batch_size 32
```

### 3. Checkpoint'ten Devam Et

```bash
# En iyi modelden devam et
python train_integrated_enhanced_model.py --resume checkpoints/integrated_enhanced/integrated_enhanced_best.pt

# Belirli bir epoch'tan devam et
python train_integrated_enhanced_model.py --resume checkpoints/integrated_enhanced/integrated_enhanced_epoch_50.pt
```

---

## 📊 Training Çıktısı

```
🚀 INTEGRATED ENHANCED TRM TRAINING
============================================================
🚀 Integrated Enhanced Trainer initialized
📱 Device: cuda
🧠 Model parameters: 2,345,678
📊 Training scenarios: 80
📊 Validation scenarios: 20

📚 Epoch 1/150 - Curriculum Stage 0 - Tools: ['price_comparison']
Training - Total Loss: 0.4523, Category Loss: 0.1234, Tool Loss: 0.0876, 
          Tool Exec Loss: 0.0543, Tool Reward: 0.156

📚 Epoch 5/150 - Curriculum Stage 0 - Tools: ['price_comparison']
🔍 Evaluating model...
Evaluation - Category Match: 65.0%, Tool Match: 55.0%, 
            Tool Exec Success: 0.350, Avg Reward: 0.550, Quality: 0.517
💾 New best model saved! Score: 0.517
```

---

## 🎯 Yeni Özellikler

### ✅ Tool Feedback
Model artık önceki tool execution sonuçlarını kullanıyor:
- Sequential reasoning
- Iterative improvement
- Context awareness

### ✅ Tool Parameters
Model her tool için özel parametreler üretiyor:
- `price_comparison`: budget (0-500)
- `review_analysis`: min_rating (0-5)
- `inventory_check`: threshold (0-1)
- `trend_analyzer`: window_days (0-30)

### ✅ Resume Training
Training kesintiye uğrarsa devam edebilirsin:
- Curriculum stage korunuyor
- Best score korunuyor
- Optimizer state korunuyor

### ✅ Curriculum Learning
4 aşamalı tool öğrenme:
- Stage 0 (Epoch 0-20): Sadece price_comparison
- Stage 1 (Epoch 20-50): + review_analysis
- Stage 2 (Epoch 50-80): + inventory_check
- Stage 3 (Epoch 80+): Tüm tool'lar

---

## 📈 Beklenen Sonuçlar

| Metric | Target | Açıklama |
|--------|--------|----------|
| Category Match Rate | >70% | Doğru kategori seçimi |
| Tool Match Rate | >60% | Doğru tool seçimi |
| Tool Exec Success | >0.50 | Başarılı tool execution |
| Recommendation Quality | >0.65 | Genel kalite skoru |

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Batch size'ı küçült
python train_integrated_enhanced_model.py --batch_size 8
```

### Training Çok Yavaş
```bash
# Epoch sayısını azalt
python train_integrated_enhanced_model.py --epochs 100
```

### Checkpoint Bulunamadı
```bash
# Checkpoint klasörünü kontrol et
ls -la checkpoints/integrated_enhanced/
```

---

## 📝 Notlar

- **Device**: Otomatik olarak CUDA varsa GPU, yoksa CPU kullanılır
- **Checkpoints**: Her 25 epoch'ta ve her iyileştirmede kaydedilir
- **Early Stopping**: 25 evaluation (5 epoch * 5) boyunca iyileşme yoksa durur
- **Curriculum**: Epoch sayısına göre otomatik ilerler

---

## 🎉 Başarı!

Eğer test'ler geçtiyse ve training başladıysa, her şey hazır! 

Model artık:
- ✅ Tool'ları kullanabiliyor
- ✅ Tool parametreleri üretebiliyor
- ✅ Önceki sonuçlardan öğrenebiliyor
- ✅ Checkpoint'ten devam edebiliyor

**Happy Training! 🚀**
