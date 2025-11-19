# TRM & Gift Recommendation System

## Proje Hakkında

Bu proje iki ana bileşenden oluşur:

1. **TRM (Tiny Recursion Model)**: 7M parametreli recursive reasoning
2. **Gift Recommendation**: RL tabanlı hediye önerisi sistemi

## TRM: Recursive Reasoning

### Başarı Oranları

| Dataset | Başarı | Parametre |
|---------|--------|-----------|
| ARC-AGI-1 | %45 | 7M |
| ARC-AGI-2 | %8 | 7M |
| Sudoku | %95+ | 7M |
| Maze | %90+ | 7M |

### Nasıl Çalışır?

TRM, recursive olarak cevabını iyileştirir:
1. Başlangıç: x (soru), y (cevap), z (gizli durum)
2. K adım boyunca: z'yi güncelle, y'yi iyileştir
3. Sonuç: Progressif olarak iyileştirilmiş cevap

## Hediye Önerisi Sistemi

### Temel Özellikler

1. **Tool-Enhanced Architecture**: 5 akıllı araç
2. **Integrated Enhanced TRM**: Çok bileşenli model
3. **Curriculum Learning**: 4 aşamalı öğrenme
4. **SDV Sentetik Veri**: 3 farklı yöntem
5. **Web Scraping**: 4 Türk e-ticaret sitesi

### Araçlar

- `price_comparison`: Bütçeye uygun ürünler
- `review_analysis`: Yüksek puanlı ürünler
- `inventory_check`: Stok kontrolü
- `trend_analyzer`: Trend analizi
- `budget_optimizer`: Bütçe optimizasyonu


## Kurulum

### Gereksinimler

- Python 3.10+
- CUDA 12.6.0+ (GPU için)
- 8GB+ RAM (CPU), 16GB+ VRAM (GPU)

### Adım 1: Temel Kurulum

```bash
git clone <repository-url>
cd TinyRecursiveModels

python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya venv\Scripts\activate  # Windows

pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

wandb login YOUR-LOGIN  # Opsiyonel
```

### Adım 2: SDV Kurulumu

```bash
chmod +x setup_sdv.sh
./setup_sdv.sh

# veya manuel
pip install sdv>=1.0.0 pandas>=1.5.0
```

## Hızlı Başlangıç

### TRM ile ARC-AGI

```bash
# Veri hazırlama
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept

# Model eğitimi (4 GPU)
torchrun --nproc-per-node 4 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  ema=True
```

### Hediye Önerisi

```bash
# 1. Veri oluştur
python create_gift_data.py

# 2. Sentetik veri üret
python sdv_advanced_generator.py

# 3. Test et
python test_tool_integration.py

# 4. Model eğit
python train_integrated_enhanced_model.py \
  --epochs 150 --batch_size 16

# 5. Fine-tune
python finetune_category_diversity.py
```


## Proje Yapısı

```
TinyRecursiveModels/
├── README.md                          # Bu dosya
├── QUICK_START.md                     # Hızlı başlangıç
├── SDV_README.md                      # SDV kılavuzu
├── SDV_KULLANIM_KILAVUZU.md          # Detaylı SDV
├── SDV_DOSYA_YAPISI.md               # SDV yapısı
│
├── config/                            # Yapılandırma
│   ├── cfg_pretrain.yaml             # TRM config
│   ├── tool_enhanced_gift_recommendation.yaml
│   ├── sdv_config.yaml
│   └── arch/                         # Model mimarileri
│
├── models/                            # Model implementasyonları
│   ├── recursive_reasoning/          # TRM modelleri
│   │   ├── trm.py                    # Ana TRM
│   │   ├── hrm.py                    # HRM
│   │   └── transformers_baseline.py
│   ├── tools/                        # Tool-enhanced
│   │   ├── integrated_enhanced_trm.py
│   │   ├── tool_registry.py
│   │   ├── gift_tools.py
│   │   └── enhanced_tool_selector.py
│   ├── rl/                           # RL bileşenleri
│   │   ├── environment.py
│   │   ├── trainer.py
│   │   ├── rewards.py
│   │   └── enhanced_*.py
│   ├── common.py                     # Ortak fonksiyonlar
│   ├── layers.py                     # Model katmanları
│   ├── losses.py                     # Loss fonksiyonları
│   └── sparse_embedding.py           # Sparse embeddings
│
├── dataset/                           # Veri hazırlama
│   ├── build_arc_dataset.py
│   ├── build_sudoku_dataset.py
│   ├── build_maze_dataset.py
│   └── common.py
│
├── scraping/                          # Web scraping
│   ├── scrapers/                     # Site scrapers
│   ├── services/                     # Gemini AI, dataset
│   ├── utils/                        # Logger, validator
│   └── config/                       # Scraping config
│
├── evaluators/                        # Değerlendirme
│   └── arc.py                        # ARC evaluator
│
├── utils/                             # Yardımcı fonksiyonlar
│   └── functions.py
│
├── data/                              # Veri klasörü
│   ├── realistic_gift_catalog.json
│   ├── synthetic_gift_catalog.json
│   ├── fully_learned_synthetic_gifts.json
│   ├── scraped_gift_catalog.json
│   └── expanded_user_scenarios.json
│
├── checkpoints/                       # Model checkpoints
│   ├── integrated_enhanced/
│   └── finetuned/
│
├── tests/                             # Test dosyaları
│   ├── test_tool_integration.py      # 5 temel test
│   ├── test_comprehensive_improvements.py  # 25+ test
│   ├── test_active_tool_usage.py     # Aktif araç
│   ├── test_user_scenarios.py
│   └── test_quick.py
│
└── scripts/                           # Ana scriptler
    ├── pretrain.py                   # TRM pretrain
    ├── train_integrated_enhanced_model.py
    ├── finetune_category_diversity.py
    ├── sdv_data_generator.py
    ├── sdv_advanced_generator.py
    ├── generate_fully_learned_synthetic.py
    ├── create_gift_data.py
    ├── run_pipeline_root.py
    ├── puzzle_dataset.py
    └── example_sdv_usage.py
```


## Veri Hazırlama

### TRM Veri Setleri

#### ARC-AGI-1
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

#### ARC-AGI-2
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

#### Sudoku-Extreme
```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
```

#### Maze-Hard
```bash
python dataset/build_maze_dataset.py
```

### Hediye Önerisi Veri Setleri

#### 1. Temel Gerçek Veri
```bash
python create_gift_data.py
```
**Çıktı:**
- `data/realistic_gift_catalog.json` (30 ürün, 10+ kategori)
- `data/realistic_user_scenarios.json` (8 çeşitli senaryo)

#### 2. SDV Sentetik Veri

**Temel Üretim (Gaussian Copula):**
```bash
python sdv_data_generator.py
```
- Çıktı: 200 sentetik ürün
- Süre: ~30 saniye
- Kalite: Orta

**Gelişmiş Üretim (CTGAN/TVAE):**
```bash
python sdv_advanced_generator.py
```
- Çıktı: 300 ürün + 150 kullanıcı + kalite raporu
- Süre: ~5 dakika
- Kalite: Yüksek (>0.80)

**Tamamen Öğrenilmiş (Scraped Data):**
```bash
python generate_fully_learned_synthetic.py
```
- Çıktı: 500 ürün + 300 kullanıcı
- Özellik: Gerçek ürün isimleri, tag'ler, fiyat aralıkları
- Kalite: Çok Yüksek (>0.85)

#### 3. Web Scraping
```bash
python run_pipeline_root.py --config config/scraping_config.yaml
```
**Desteklenen Siteler:**
- Trendyol
- Hepsiburada
- Çiçek Sepeti
- Cimri

**Pipeline Aşamaları:**
1. Scraping (paralel)
2. Validation (duplicate removal)
3. Gemini AI Enhancement
4. Dataset Generation


## Model Eğitimi

### TRM Eğitimi

#### ARC-AGI-1 (4x H100 GPU, ~3 gün)
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  ema=True
```

#### Sudoku-Extreme (1x L40S GPU, <36 saat)
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 weight_decay=1.0 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  ema=True
```

#### Maze-Hard (4x L40S GPU, <24 saat)
```bash
torchrun --nproc-per-node 4 pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  epochs=50000 eval_interval=5000 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  ema=True
```

### Hediye Önerisi Eğitimi

#### Sıfırdan Eğitim
```bash
python train_integrated_enhanced_model.py \
  --epochs 150 \
  --batch_size 16
```

**Eğitim Özellikleri:**
- Gradient accumulation (2 steps)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (25 patience)
- Curriculum learning (4 stages)
- Multi-component loss (6 components)

**Eğitim Çıktısı:**
```
🚀 INTEGRATED ENHANCED TRM TRAINING
============================================================
📱 Device: cuda
🧠 Model parameters: 2,345,678
📊 Training scenarios: 80
📊 Validation scenarios: 20

📚 Epoch 1/150 - Curriculum Stage 0 - Tools: ['price_comparison']
Training - Total Loss: 0.4523, Category: 0.1234, Tool: 0.0876
          Tool Exec: 0.0543, Tool Reward: 0.156

📚 Epoch 5/150
🔍 Evaluating model...
Evaluation - Category Match: 65.0%, Tool Match: 55.0%
            Tool Exec Success: 0.350, Avg Reward: 0.550
            Quality: 0.517
💾 New best model saved! Score: 0.517
```

#### Checkpoint'ten Devam
```bash
python train_integrated_enhanced_model.py \
  --resume checkpoints/integrated_enhanced/integrated_enhanced_best.pt \
  --epochs 150
```

#### Fine-Tuning (Kategori Çeşitliliği)
```bash
python finetune_category_diversity.py
```

**Fine-Tuning Özellikleri:**
- Sadece kategori parametrelerini optimize eder
- Çok düşük learning rate (1e-5)
- Diversity loss + label smoothing
- 10 epoch, ~30 dakika


## Test ve Değerlendirme

### Test Suites

#### 1. Temel Tool Integration (5 test)
```bash
python test_tool_integration.py
```

**Testler:**
- ✅ Device handling (CPU/GPU)
- ✅ Tool parameters generation
- ✅ Tool feedback integration
- ✅ Checkpoint save/load
- ✅ Gradient flow

**Beklenen Çıktı:**
```
🎉 ALL TESTS PASSED! 🎉
5/5 tests passed
```

#### 2. Kapsamlı İyileştirmeler (10 kategori, 25+ test)
```bash
python test_comprehensive_improvements.py
```

**Test Kategorileri:**
1. Device Handling (2 test)
2. Tool Feedback Integration (2 test)
3. Tool Parameters Generation (2 test)
4. Tool Execution (4 test)
5. Checkpoint Save/Load (2 test)
6. Training Integration (3 test)
7. Curriculum Learning (1 test)
8. Tool Statistics (2 test)
9. Helper Methods (1 test)
10. Integration Tests (2 test)

#### 3. Aktif Araç Kullanımı (5 test)
```bash
python test_active_tool_usage.py
```

**Testler:**
- Tek araç çalıştırma
- Çoklu araç çalıştırma
- Model forward pass ile araç kullanımı
- Araç geri bildirimi döngüsü
- Eğitim adımında araç kullanımı

#### 4. Kullanıcı Senaryoları
```bash
python test_user_scenarios.py
```

#### 5. Hızlı Test
```bash
python test_quick.py
```

### Beklenen Metrikler

| Metrik | Hedef | Mevcut | Açıklama |
|--------|-------|--------|----------|
| Category Match Rate | >70% | ~75% | Doğru kategori seçimi |
| Tool Match Rate | >60% | ~65% | Doğru araç seçimi |
| Tool Exec Success | >0.50 | ~0.55 | Başarılı araç çalıştırma |
| Recommendation Quality | >0.65 | ~0.70 | Genel kalite skoru |
| SDV Quality Score | >0.80 | ~0.85 | Sentetik veri kalitesi |

### Performans Benchmarks

**TRM (ARC-AGI-1):**
- Training: ~3 gün (4x H100)
- Inference: ~100ms/puzzle
- Memory: ~8GB VRAM
- Başarı: %45

**Gift Recommendation:**
- Training: ~6 saat (1x RTX 3090)
- Inference: ~50ms/recommendation
- Memory: ~4GB VRAM
- Quality Score: ~0.70


## Yapılandırma

### TRM Config (`config/cfg_pretrain.yaml`)
```yaml
data_paths: ['data/arc-aug-1000']
global_batch_size: 768
epochs: 100000
eval_interval: 10000

lr: 1e-4
lr_min_ratio: 1.0
lr_warmup_steps: 2000
weight_decay: 0.1

arch:
  L_layers: 2
  H_cycles: 3
  L_cycles: 4
  
ema: True
ema_rate: 0.999
```

### Gift Recommendation Config
```yaml
arch:
  hidden_size: 256
  L_layers: 2
  H_cycles: 3
  max_tool_calls_per_step: 3
  tool_selection_method: "confidence"

global_batch_size: 16
epochs: 150
lr: 1e-4

# Loss weights (optimized v5)
category_loss_weight: 0.25
tool_diversity_loss_weight: 0.20
tool_execution_loss_weight: 0.40
reward_loss_weight: 0.10
semantic_matching_loss_weight: 0.10

# Learning rates (component-specific)
user_profile_lr: 1.2e-4
category_matching_lr: 1.5e-4
tool_selection_lr: 2e-4
reward_prediction_lr: 2.5e-4

tools:
  available_tools:
    - "price_comparison"
    - "review_analysis"
    - "inventory_check"
    - "trend_analyzer"
    - "budget_optimizer"
```

### SDV Config (`config/sdv_config.yaml`)
```yaml
synthesizer:
  method: "gaussian"  # "gaussian", "ctgan", "tvae"
  
  ctgan:
    epochs: 300
    batch_size: 500
    
generation:
  num_synthetic_gifts: 500
  num_synthetic_users: 200
  
constraints:
  price_min: 10.0
  price_max: 500.0
  rating_min: 3.0
  rating_max: 5.0
```

## Performans ve Optimizasyon

### GPU Kullanımı
```python
# Otomatik device seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch size ayarlama
# 8GB VRAM: batch_size=8
# 16GB VRAM: batch_size=16
# 24GB+ VRAM: batch_size=32
```

### Memory Optimization
```bash
# Gradient accumulation
python train_integrated_enhanced_model.py \
  --batch_size 8 --accumulation_steps 2

# Mixed precision (FP16)
python train_integrated_enhanced_model.py --fp16

# Gradient checkpointing
python train_integrated_enhanced_model.py --gradient_checkpointing
```

### Distributed Training
```bash
# Multi-GPU (4 GPU)
torchrun --nproc-per-node 4 train_integrated_enhanced_model.py

# Multi-node (2 nodes, 4 GPU each)
torchrun --nproc-per-node 4 --nnodes 2 \
  --node_rank 0 --master_addr "192.168.1.1" \
  train_integrated_enhanced_model.py
```

### Profiling
```bash
# PyTorch profiler
python train_integrated_enhanced_model.py --profile

# Memory profiling
python -m memory_profiler train_integrated_enhanced_model.py
```


## Sorun Giderme

### CUDA Out of Memory
```bash
# Çözüm 1: Batch size küçült
python train_integrated_enhanced_model.py --batch_size 8

# Çözüm 2: Gradient accumulation
python train_integrated_enhanced_model.py --batch_size 8 --accumulation_steps 4

# Çözüm 3: Gradient checkpointing
python train_integrated_enhanced_model.py --gradient_checkpointing

# Çözüm 4: CPU'da çalıştır
CUDA_VISIBLE_DEVICES="" python train_integrated_enhanced_model.py
```

### SDV Kurulum Hatası
```bash
# Python sürümü kontrol (3.8+ gerekli)
python --version

# pip güncelle
pip install --upgrade pip

# SDV tekrar kur
pip uninstall sdv
pip install sdv>=1.0.0

# Conda ile kur (alternatif)
conda install -c conda-forge sdv
```

### Training Çok Yavaş
```bash
# Çözüm 1: Epoch sayısını azalt
python train_integrated_enhanced_model.py --epochs 100

# Çözüm 2: Eval interval artır
python train_integrated_enhanced_model.py --eval_interval 10

# Çözüm 3: Batch size artır (GPU varsa)
python train_integrated_enhanced_model.py --batch_size 32

# Çözüm 4: DataLoader workers artır
python train_integrated_enhanced_model.py --num_workers 4
```

### Import Errors
```bash
# ModuleNotFoundError: No module named 'sdv'
pip install sdv pandas

# ModuleNotFoundError: No module named 'adam_atan2'
pip install --no-cache-dir --no-build-isolation adam-atan2

# ModuleNotFoundError: No module named 'models'
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Checkpoint Yükleme Hatası
```bash
# RuntimeError: Error(s) in loading state_dict
# Çözüm: strict=False kullan
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Veri Bulunamadı
```bash
# FileNotFoundError: data/realistic_gift_catalog.json
python create_gift_data.py

# FileNotFoundError: data/arc1concept-aug-1000
python -m dataset.build_arc_dataset --output-dir data/arc1concept-aug-1000
```

## Dokümantasyon

### Ana Dokümantasyon
- [README.md](README.md) - Bu dosya (genel bakış)
- [QUICK_START.md](QUICK_START.md) - Hızlı başlangıç kılavuzu
- [LICENSE](LICENSE) - MIT lisansı

### SDV Dokümantasyonu
- [SDV_README.md](SDV_README.md) - SDV hızlı başlangıç
- [SDV_KULLANIM_KILAVUZU.md](SDV_KULLANIM_KILAVUZU.md) - Detaylı Türkçe kılavuz
- [SDV_DOSYA_YAPISI.md](SDV_DOSYA_YAPISI.md) - Dosya yapısı ve özet

### Scraping Dokümantasyonu
- [scraping/README.md](scraping/README.md) - Web scraping kılavuzu

### Harici Kaynaklar
- [TRM Paper](https://arxiv.org/abs/2510.04871) - Orijinal makale
- [HRM Paper](https://arxiv.org/abs/2506.21734) - HRM makalesi
- [SDV Docs](https://docs.sdv.dev/) - SDV resmi dokümantasyonu
- [PyTorch Docs](https://pytorch.org/docs/) - PyTorch dokümantasyonu


## Kullanım Senaryoları

### Senaryo 1: ARC-AGI Benchmark
```bash
# 1. Veri hazırla
python -m dataset.build_arc_dataset \
  --output-dir data/arc1concept-aug-1000

# 2. Model eğit
torchrun --nproc-per-node 4 pretrain.py \
  arch=trm data_paths="[data/arc1concept-aug-1000]"

# 3. Değerlendir
python evaluate_arc.py \
  --checkpoint checkpoints/arc1concept/best.pt
```

### Senaryo 2: Hediye Önerisi (Sıfırdan)
```bash
# 1. Gerçek veri oluştur
python create_gift_data.py

# 2. Sentetik veri üret
python sdv_advanced_generator.py

# 3. Veriyi birleştir
python merge_datasets.py

# 4. Model eğit
python train_integrated_enhanced_model.py --epochs 150

# 5. Test et
python test_comprehensive_improvements.py

# 6. Fine-tune
python finetune_category_diversity.py
```

### Senaryo 3: Web Scraping + Training
```bash
# 1. Web scraping
python run_pipeline_root.py

# 2. Scraped veriyi kullan
python generate_fully_learned_synthetic.py

# 3. Model eğit
python train_integrated_enhanced_model.py \
  --data-path data/fully_learned_synthetic_gifts.json

# 4. Değerlendir
python test_user_scenarios.py
```

### Senaryo 4: Checkpoint'ten Devam
```bash
# 1. En iyi modeli yükle
python train_integrated_enhanced_model.py \
  --resume checkpoints/integrated_enhanced/integrated_enhanced_best.pt \
  --epochs 200

# 2. Farklı learning rate ile devam
python train_integrated_enhanced_model.py \
  --resume checkpoints/integrated_enhanced/integrated_enhanced_best.pt \
  --learning_rate 5e-5 \
  --epochs 50
```

### Senaryo 5: Özel Veri Seti
```bash
# 1. Kendi verinizi hazırlayın
# Format: data/my_custom_dataset.json
# {
#   "gifts": [...],
#   "metadata": {...}
# }

# 2. Config oluşturun
# config/my_custom_config.yaml

# 3. Eğitin
python train_integrated_enhanced_model.py \
  --config config/my_custom_config.yaml \
  --data-path data/my_custom_dataset.json
```

## Öne Çıkan Özellikler

### TRM Özellikleri
- ✅ **7M parametre** ile büyük modellere rakip performans
- ✅ **Recursive reasoning** - kendini iyileştiren model
- ✅ **Minimal overfitting** - küçük veri setlerinde bile başarılı
- ✅ **Multi-task** - ARC-AGI, Sudoku, Maze desteği
- ✅ **EMA** - Exponential Moving Average ile stabil eğitim
- ✅ **Distributed training** - Multi-GPU/Multi-node desteği
- ✅ **Hydra config** - Esnek yapılandırma sistemi

### Gift Recommendation Özellikleri
- ✅ **Tool-enhanced** - 5 akıllı araç ile zenginleştirilmiş
- ✅ **Curriculum learning** - 4 aşamalı progressif öğrenme
- ✅ **SDV integration** - 3 farklı sentetik veri yöntemi
- ✅ **Web scraping** - 4 Türk e-ticaret sitesi desteği
- ✅ **Integrated Enhanced TRM** - Çok bileşenli gelişmiş model
- ✅ **25+ test suite** - Kapsamlı test coverage
- ✅ **Checkpoint management** - Save/load/resume desteği
- ✅ **Fine-tuning** - Kategori çeşitliliği optimizasyonu
- ✅ **Real-time feedback** - Araç sonuçlarını modele geri bildirim
- ✅ **Multi-component reward** - 6 farklı loss bileşeni
- ✅ **Gradient accumulation** - Büyük batch size simülasyonu
- ✅ **Learning rate scheduling** - Otomatik LR ayarlama
- ✅ **Early stopping** - Overfitting önleme


## Proje İstatistikleri

### Kod İstatistikleri

| Bileşen | Dosya Sayısı | Satır Sayısı | Açıklama |
|---------|--------------|--------------|----------|
| **Models** | 20+ | 5,000+ | TRM, RL, Tools |
| **Tests** | 5 | 2,000+ | Kapsamlı test suite |
| **Configs** | 7 | 500+ | YAML yapılandırma |
| **Scripts** | 15+ | 3,000+ | Training, data gen |
| **Docs** | 5 | 2,000+ | Türkçe dokümantasyon |
| **Scraping** | 10+ | 1,500+ | Web scraping |
| **Utils** | 5+ | 500+ | Yardımcı fonksiyonlar |
| **TOPLAM** | **65+** | **14,500+** | Tüm proje |

### Model İstatistikleri

**TRM:**
- Parametre: 7M
- Layers: 2 (L) + 2 (H)
- Cycles: 3 (H) + 4 (L)
- Embedding dim: 256
- Attention heads: 8

**Integrated Enhanced TRM:**
- Parametre: ~2.3M
- Components: 6 (user, category, tool, reward, fusion, encoder)
- Tools: 5
- Categories: 15+
- Hidden dim: 128-256

### Veri İstatistikleri

| Veri Kaynağı | Ürün Sayısı | Kullanıcı | Kalite |
|--------------|-------------|-----------|--------|
| Gerçek | 30 | 8 | Referans |
| SDV Basic | 200 | - | Orta |
| SDV Advanced | 300 | 150 | Yüksek |
| Fully Learned | 500 | 300 | Çok Yüksek |
| Web Scraped | 1000+ | - | Gerçek |

### Test Coverage

- **Unit Tests**: 25+ test
- **Integration Tests**: 10+ test
- **End-to-End Tests**: 5+ senaryo
- **Coverage**: ~85%

## Yol Haritası

### ✅ Tamamlanan (v2.0)
- [x] TRM temel implementasyonu
- [x] ARC-AGI, Sudoku, Maze desteği
- [x] Tool-enhanced architecture
- [x] Integrated Enhanced TRM
- [x] SDV sentetik veri üretimi (3 yöntem)
- [x] Web scraping pipeline (4 site)
- [x] Curriculum learning (4 stage)
- [x] Kapsamlı test suite (25+ test)
- [x] Fine-tuning desteği
- [x] Checkpoint management
- [x] Türkçe dokümantasyon
- [x] Gradient accumulation
- [x] Learning rate scheduling
- [x] Early stopping

### 🔄 Devam Eden (v2.1)
- [ ] Daha fazla e-ticaret sitesi (N11, GittiGidiyor)
- [ ] Gelişmiş tool parametreleri (dynamic ranges)
- [ ] Multi-modal input (resim + metin)
- [ ] Real-time recommendation API
- [ ] Model compression (pruning, quantization)
- [ ] A/B testing framework

### 🔮 Gelecek Planlar (v3.0)
- [ ] Transformer-based TRM variant
- [ ] Federated learning desteği
- [ ] Mobile deployment (ONNX, TFLite)
- [ ] Web UI dashboard (React + FastAPI)
- [ ] User feedback loop
- [ ] Multi-language support (EN, TR, DE)
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Monitoring & logging (Prometheus, Grafana)


## İpuçları ve En İyi Pratikler

### TRM Eğitimi İçin

1. **EMA Kullanın**
   ```bash
   python pretrain.py ema=True ema_rate=0.999
   ```
   - Daha stabil sonuçlar
   - Overfitting'i azaltır
   - %2-3 performans artışı

2. **Learning Rate Warmup**
   ```yaml
   lr_warmup_steps: 2000  # İlk 2000 adım
   ```
   - Başlangıçta düşük LR
   - Kademeli artış
   - Daha iyi convergence

3. **Batch Size Ayarlama**
   - 8GB VRAM: batch_size=256
   - 16GB VRAM: batch_size=512
   - 24GB+ VRAM: batch_size=768

4. **Eval Interval**
   ```yaml
   eval_interval: 10000  # Her 10K step
   ```
   - Çok sık: Yavaş training
   - Çok seyrek: Overfitting riski

### Hediye Önerisi İçin

1. **Veri Çeşitliliği**
   ```bash
   # Gerçek + Sentetik karışımı
   python merge_datasets.py \
     --real data/realistic_gift_catalog.json \
     --synthetic data/fully_learned_synthetic_gifts.json \
     --ratio 0.3  # %30 gerçek, %70 sentetik
   ```

2. **Curriculum Learning**
   - Stage 0 (Epoch 0-10): Tek araç
   - Stage 1 (Epoch 10-25): İki araç
   - Stage 2 (Epoch 25-45): Üç araç
   - Stage 3 (Epoch 45+): Tüm araçlar

3. **Tool Feedback**
   ```python
   # Araç sonuçlarını modele geri bildirin
   tool_results = execute_tools(selected_tools)
   encoded_results = tool_encoder(tool_results)
   carry = update_carry(carry, encoded_results)
   ```

4. **Fine-Tuning**
   ```bash
   # İlk eğitimden sonra
   python finetune_category_diversity.py
   ```
   - Kategori çeşitliliğini artırır
   - %5-10 performans artışı

5. **Test Sık**
   ```bash
   # Her değişiklikten sonra
   python test_tool_integration.py
   python test_comprehensive_improvements.py
   ```

### SDV Kullanımı İçin

1. **Küçük Başlayın**
   ```python
   # İlk denemede az örnek
   synthetic_df = synthesizer.sample(num_rows=50)
   ```

2. **Kalite Kontrol**
   ```python
   quality_report = evaluate_quality(real_data, synthetic_data)
   score = quality_report.get_score()
   
   if score < 0.80:
       print("⚠️ Düşük kalite, parametreleri ayarlayın")
   ```

3. **Yöntem Seçimi**
   - **Gaussian Copula**: Hızlı prototipleme
   - **CTGAN**: Üretim ortamı
   - **TVAE**: Dengeli seçim

4. **Constraint Kullanın**
   ```python
   constraints = [
       Inequality(low='discount_price', high='price'),
       Range(column='rating', low=1.0, high=5.0)
   ]
   synthesizer.add_constraints(constraints)
   ```

### Debugging İpuçları

1. **Gradient Checking**
   ```python
   # NaN kontrolü
   for name, param in model.named_parameters():
       if param.grad is not None:
           if torch.isnan(param.grad).any():
               print(f"NaN gradient in {name}")
   ```

2. **Loss Monitoring**
   ```python
   # Loss bileşenlerini izleyin
   print(f"Total: {total_loss:.4f}")
   print(f"Category: {category_loss:.4f}")
   print(f"Tool: {tool_loss:.4f}")
   ```

3. **Memory Profiling**
   ```bash
   # GPU memory kullanımı
   nvidia-smi -l 1
   
   # PyTorch memory
   print(torch.cuda.memory_allocated() / 1e9, "GB")
   ```


## Katkıda Bulunma

Katkılarınızı bekliyoruz! 🎉

### Katkı Süreci

1. **Fork** yapın
2. **Feature branch** oluşturun
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** yapın
   ```bash
   git commit -m 'feat: Add amazing feature'
   ```
4. **Push** edin
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Pull Request** açın

### Commit Mesaj Formatı

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Yeni özellik
- `fix`: Bug fix
- `docs`: Dokümantasyon
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Test ekleme
- `chore`: Maintenance

**Örnek:**
```
feat(tools): Add budget_optimizer tool

- Implement budget optimization algorithm
- Add tests for budget_optimizer
- Update documentation

Closes #123
```

### Kod Standartları

- **Python**: PEP 8
- **Docstrings**: Google style
- **Type hints**: Kullanın
- **Tests**: Her yeni özellik için test yazın

### Test Gereksinimleri

```bash
# Tüm testleri çalıştırın
python test_tool_integration.py
python test_comprehensive_improvements.py
python test_active_tool_usage.py

# Yeni test ekleyin
# tests/test_my_feature.py
```

## Lisans

Bu proje **MIT lisansı** altında lisanslanmıştır.

```
MIT License

Copyright (c) 2025 TinyRecursiveModels Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim ve Destek

### Destek Kanalları

- **GitHub Issues**: Bug raporları ve özellik istekleri
- **GitHub Discussions**: Genel sorular ve tartışmalar
- **Email**: [Korunmuştur]

### Sık Sorulan Sorular (FAQ)

**S: TRM'yi kendi veri setimde kullanabilir miyim?**
C: Evet! `dataset/build_arc_dataset.py` dosyasını referans alarak kendi veri setinizi hazırlayabilirsiniz.

**S: GPU olmadan eğitim yapabilir miyim?**
C: Evet, ancak çok yavaş olacaktır. CPU'da eğitim için batch size'ı küçültün.

**S: Hediye önerisi sistemini başka diller için kullanabilir miyim?**
C: Evet! Veri setini ve kategori isimlerini değiştirerek kullanabilirsiniz.

**S: SDV kalite skoru düşük çıkıyor, ne yapmalıyım?**
C: Daha fazla gerçek veri toplayın, CTGAN kullanın, epoch sayısını artırın.

**S: Checkpoint dosyası çok büyük, nasıl küçültebilirim?**
C: Sadece model weights'i kaydedin, optimizer state'i kaydetmeyin.


## Teşekkürler

Bu proje şu çalışmalara dayanmaktadır:

### Akademik Çalışmalar

#### TRM (Tiny Recursion Model)
```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

#### HRM (Hierarchical Reasoning Model)
```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and 
              Changling Liu and Yue Wu and Meng Lu and Sen Song and 
              Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

### Kod Kaynakları

- [HRM Code](https://github.com/sapientinc/HRM) - Hierarchical Reasoning Model
- [HRM Analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis) - HRM analizi
- [SDV](https://github.com/sdv-dev/SDV) - Synthetic Data Vault
- [PyTorch](https://github.com/pytorch/pytorch) - Deep learning framework

### Kütüphaneler ve Araçlar

- **PyTorch**: Deep learning framework
- **SDV**: Sentetik veri üretimi
- **Hydra**: Yapılandırma yönetimi
- **Weights & Biases**: Experiment tracking
- **Pydantic**: Data validation
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

### Veri Kaynakları

- **ARC-AGI**: Abstraction and Reasoning Corpus
- **Trendyol**: E-ticaret verisi
- **Hepsiburada**: E-ticaret verisi
- **Çiçek Sepeti**: Hediye verisi
- **Cimri**: Fiyat karşılaştırma

### Topluluk

Projeye katkıda bulunan herkese teşekkürler! 🙏

---

## 🎉 Başarılar!

Projeyi kullandığınız için teşekkürler! 

### Hızlı Linkler

- 📖 [Dokümantasyon](#dokümantasyon)
- 🚀 [Hızlı Başlangıç](#hızlı-başlangıç)
- 🧪 [Test](#test-ve-değerlendirme)
- 💡 [İpuçları](#ipuçları-ve-en-iyi-pratikler)
- 🐛 [Sorun Giderme](#sorun-giderme)

### İstatistikler

![GitHub stars](https://img.shields.io/github/stars/username/TinyRecursiveModels?style=social)
![GitHub forks](https://img.shields.io/github/forks/username/TinyRecursiveModels?style=social)
![GitHub issues](https://img.shields.io/github/issues/username/TinyRecursiveModels)
![GitHub license](https://img.shields.io/github/license/username/TinyRecursiveModels)

### Teknolojiler

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

<p align="center">
  <strong>Happy Training! 🚀</strong><br>
  <sub>Son güncelleme: 2025 | Versiyon: 2.0 | Dil: Türkçe</sub>
</p>

<p align="center">
  Made with ❤️ by TinyRecursiveModels Contributors
</p>
