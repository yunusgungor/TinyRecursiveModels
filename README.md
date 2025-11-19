# Tiny Recursion Model (TRM) & Tool-Enhanced Gift Recommendation System

Bu proje iki ana bileşenden oluşmaktadır:
1. **TRM (Tiny Recursion Model)**: Sadece 7M parametreli küçük bir sinir ağı ile recursive reasoning
2. **Tool-Enhanced Gift Recommendation System**: RL tabanlı, araç destekli hediye önerisi sistemi

---

## 📋 İçindekiler

- [TRM: Recursive Reasoning](#trm-recursive-reasoning)
- [Hediye Önerisi Sistemi](#hediye-önerisi-sistemi)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Proje Yapısı](#proje-yapısı)
- [Veri Hazırlama](#veri-hazırlama)
- [Model Eğitimi](#model-eğitimi)
- [Test ve Değerlendirme](#test-ve-değerlendirme)
- [Referanslar](#referanslar)

---

## 🧠 TRM: Recursive Reasoning

### Motivasyon

**"Less is More"** - Tiny Recursion Model (TRM), sadece 7M parametreli küçük bir sinir ağı ile ARC-AGI-1'de %45, ARC-AGI-2'de %8 başarı oranına ulaşır. Bu, büyük dil modellerine (LLM) ihtiyaç duymadan zor problemleri çözebileceğinizi gösterir.

Mevcut yaklaşımlar, milyonlarca dolar maliyetli büyük modellere odaklanırken, TRM farklı bir yol izler: **recursive reasoning** ile küçük bir model, kendini tekrar tekrar çalıştırarak cevabını iyileştirir.

### TRM Nasıl Çalışır?

<p align="center">
  <img src="assets/TRM_fig.png" alt="TRM Architecture" width="400">
</p>

TRM, tahmin ettiği cevabı (y) küçük bir ağ ile recursive olarak iyileştirir:

1. **Başlangıç**: Gömülü soru (x), başlangıç cevabı (y) ve gizli durum (z)
2. **K adım boyunca iyileştirme**:
   - **i)** Gizli durumu (z) recursive olarak güncelle (n kez)
   - **ii)** Cevabı (y) mevcut z'ye göre güncelle
3. **Sonuç**: Progressif olarak iyileştirilmiş cevap

Bu recursive süreç, modelin önceki hatalarını düzeltmesine ve minimal parametre ile overfitting'i azaltmasına olanak tanır.

### Başarı Oranları

| Dataset | Başarı Oranı | Parametre Sayısı |
|---------|--------------|------------------|
| ARC-AGI-1 | %45 | 7M |
| ARC-AGI-2 | %8 | 7M |
| Sudoku-Extreme | %95+ | 7M |
| Maze-Hard | %90+ | 7M |

---

## 🎁 Hediye Önerisi Sistemi

### Genel Bakış

Tool-Enhanced Gift Recommendation System, kullanıcı profiline göre kişiselleştirilmiş hediye önerileri sunan gelişmiş bir RL (Reinforcement Learning) sistemidir.

### Temel Özellikler

#### 1. Tool-Enhanced Architecture
Model, hediye önerisi sürecinde 5 farklı araç kullanabilir:
- `price_comparison`: Bütçeye uygun ürünleri filtreler
- `review_analysis`: Yüksek puanlı ürünleri analiz eder
- `inventory_check`: Stok durumunu kontrol eder
- `trend_analyzer`: Trend olan ürünleri belirler
- `budget_optimizer`: Bütçeyi optimize eder

#### 2. Integrated Enhanced TRM
```
IntegratedEnhancedTRM
├── User Profile Encoder (hobi, yaş, ilişki, bütçe)
├── Enhanced Category Matching (semantic attention)
├── Context-Aware Tool Selector (dinamik araç seçimi)
├── Tool Parameter Generator (her araç için özel parametreler)
├── Tool Result Encoder (araç sonuçlarını encode eder)
├── Cross-Modal Fusion (çoklu bilgi kaynağı)
└── Reward Prediction (çok bileşenli ödül tahmini)
```

#### 3. Curriculum Learning
Model, 4 aşamalı bir öğrenme sürecinden geçer:
- **Stage 0 (Epoch 0-20)**: Sadece `price_comparison`
- **Stage 1 (Epoch 20-50)**: + `review_analysis`
- **Stage 2 (Epoch 50-80)**: + `inventory_check`
- **Stage 3 (Epoch 80+)**: Tüm araçlar

#### 4. SDV Sentetik Veri Üretimi
```python
# Gerçek veriden öğrenerek sentetik veri üret
python sdv_data_generator.py          # Temel üretim
python sdv_advanced_generator.py      # Gelişmiş + kalite kontrolü
python generate_fully_learned_synthetic.py  # Tamamen öğrenilmiş
```

#### 5. Web Scraping Pipeline
```python
# Türk e-ticaret sitelerinden veri toplama
python run_pipeline_root.py
```
Desteklenen siteler:
- Trendyol
- Hepsiburada
- Çiçek Sepeti
- Cimri

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- CUDA 12.6.0+ (GPU kullanımı için)
- 8GB+ RAM (CPU), 16GB+ VRAM (GPU)

### Adım 1: Temel Kurulum

```bash
# Repository'yi klonlayın
git clone <repository-url>
cd TinyRecursiveModels

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükleyin
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

# Weights & Biases (opsiyonel)
wandb login YOUR-LOGIN
```

### Adım 2: SDV Kurulumu (Hediye Önerisi için)

```bash
# Otomatik kurulum
chmod +x setup_sdv.sh
./setup_sdv.sh

# Manuel kurulum
pip install sdv>=1.0.0 pandas>=1.5.0
```

---

## ⚡ Hızlı Başlangıç

### TRM ile ARC-AGI Eğitimi

```bash
# ARC-AGI-1 veri hazırlama
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# Model eğitimi (4 GPU)
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
```

### Hediye Önerisi Sistemi

```bash
# 1. Temel veri oluştur
python create_gift_data.py

# 2. Sentetik veri üret
python sdv_advanced_generator.py

# 3. Test et
python test_tool_integration.py

# 4. Model eğit
python train_integrated_enhanced_model.py \
  --config config/tool_enhanced_gift_recommendation.yaml \
  --epochs 150 \
  --batch_size 16

# 5. Fine-tune (kategori çeşitliliği için)
python finetune_category_diversity.py
```

---

## 📁 Proje Yapısı

```
TinyRecursiveModels/
├── README.md                          # Bu dosya
├── QUICK_START.md                     # Hızlı başlangıç kılavuzu
├── SDV_README.md                      # SDV kullanım kılavuzu
├── SDV_KULLANIM_KILAVUZU.md          # Detaylı Türkçe SDV kılavuzu
├── SDV_DOSYA_YAPISI.md               # SDV dosya yapısı
│
├── config/                            # Yapılandırma dosyaları
│   ├── cfg_pretrain.yaml             # TRM pretrain config
│   ├── tool_enhanced_gift_recommendation.yaml
│   ├── sdv_config.yaml               # SDV config
│   └── arch/                         # Model mimarileri
│
├── models/                            # Model implementasyonları
│   ├── recursive_reasoning/          # TRM modelleri
│   ├── tools/                        # Tool-enhanced modeller
│   │   ├── integrated_enhanced_trm.py
│   │   ├── tool_registry.py
│   │   ├── gift_tools.py
│   │   └── enhanced_tool_selector.py
│   └── rl/                           # RL bileşenleri
│       ├── environment.py
│       ├── trainer.py
│       ├── rewards.py
│       └── enhanced_*.py
│
├── dataset/                           # Veri hazırlama
│   ├── build_arc_dataset.py
│   ├── build_sudoku_dataset.py
│   └── build_maze_dataset.py
│
├── scraping/                          # Web scraping
│   ├── scrapers/                     # Site-specific scrapers
│   ├── services/                     # Gemini AI, dataset gen
│   └── utils/                        # Logger, validator
│
├── data/                              # Veri klasörü
│   ├── realistic_gift_catalog.json
│   ├── synthetic_gift_catalog.json
│   ├── fully_learned_synthetic_gifts.json
│   └── scraped_gift_catalog.json
│
├── checkpoints/                       # Model checkpoints
│   ├── integrated_enhanced/
│   └── finetuned/
│
└── tests/                             # Test dosyaları
    ├── test_tool_integration.py      # 5 temel test
    ├── test_comprehensive_improvements.py  # 10 kategori, 25+ test
    ├── test_active_tool_usage.py     # Aktif araç kullanımı
    └── test_user_scenarios.py
```

---

## 📊 Veri Hazırlama

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
  --subsample-size 1000 \
  --num-aug 1000
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
Çıktı:
- `data/realistic_gift_catalog.json` (30 ürün)
- `data/realistic_user_scenarios.json` (8 senaryo)

#### 2. SDV Sentetik Veri

**Temel Üretim:**
```bash
python sdv_data_generator.py
```
Çıktı: `data/synthetic_gift_catalog.json` (200 ürün)

**Gelişmiş Üretim:**
```bash
python sdv_advanced_generator.py
```
Çıktı:
- `data/synthetic_gift_catalog.json` (300 ürün)
- `data/synthetic_user_scenarios.json` (150 kullanıcı)
- `data/sdv_quality_report.json` (kalite raporu)

**Tamamen Öğrenilmiş:**
```bash
python generate_fully_learned_synthetic.py
```
Çıktı:
- `data/fully_learned_synthetic_gifts.json` (500 ürün)
- `data/fully_learned_synthetic_users.json` (300 kullanıcı)

#### 3. Web Scraping
```bash
python run_pipeline_root.py --config config/scraping_config.yaml
```
Çıktı: `data/scraped_gift_catalog.json`

---

## 🎓 Model Eğitimi

### TRM Eğitimi

#### ARC-AGI-1 (4x H100 GPU, ~3 gün)
```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
```

#### Sudoku-Extreme (1x L40S GPU, <36 saat)
```bash
run_name="pretrain_att_sudoku"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True
```

### Hediye Önerisi Eğitimi

#### Sıfırdan Eğitim
```bash
python train_integrated_enhanced_model.py \
  --config config/tool_enhanced_gift_recommendation.yaml \
  --epochs 150 \
  --batch_size 16 \
  --learning_rate 1e-4
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

### Eğitim Çıktısı

```
🚀 INTEGRATED ENHANCED TRM TRAINING
============================================================
📱 Device: cuda
🧠 Model parameters: 2,345,678
📊 Training scenarios: 80
📊 Validation scenarios: 20

📚 Epoch 1/150 - Curriculum Stage 0 - Tools: ['price_comparison']
Training - Total Loss: 0.4523, Category Loss: 0.1234, Tool Loss: 0.0876

📚 Epoch 5/150
🔍 Evaluating model...
Evaluation - Category Match: 65.0%, Tool Match: 55.0%, 
            Tool Exec Success: 0.350, Avg Reward: 0.550
💾 New best model saved! Score: 0.517
```

---

## 🧪 Test ve Değerlendirme

### Test Suites

#### 1. Temel Tool Integration (5 test)
```bash
python test_tool_integration.py
```
Testler:
- Device handling
- Tool parameters generation
- Tool feedback integration
- Checkpoint save/load
- Gradient flow

#### 2. Kapsamlı İyileştirmeler (10 kategori, 25+ test)
```bash
python test_comprehensive_improvements.py
```
Kategoriler:
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

#### 4. Kullanıcı Senaryoları
```bash
python test_user_scenarios.py
```

### Beklenen Metrikler

| Metrik | Hedef | Açıklama |
|--------|-------|----------|
| Category Match Rate | >70% | Doğru kategori seçimi |
| Tool Match Rate | >60% | Doğru araç seçimi |
| Tool Exec Success | >0.50 | Başarılı araç çalıştırma |
| Recommendation Quality | >0.65 | Genel kalite skoru |
| SDV Quality Score | >0.80 | Sentetik veri kalitesi |

---

## 🔧 Yapılandırma

### TRM Config (`config/cfg_pretrain.yaml`)
```yaml
data_paths: ['data/arc-aug-1000']
global_batch_size: 768
epochs: 100000
lr: 1e-4
arch:
  L_layers: 2
  H_cycles: 3
  L_cycles: 4
```

### Hediye Önerisi Config (`config/tool_enhanced_gift_recommendation.yaml`)
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
  method: "gaussian"  # veya "ctgan", "tvae"

generation:
  num_synthetic_gifts: 500
  num_synthetic_users: 200

constraints:
  price_min: 10.0
  price_max: 500.0
  rating_min: 3.0
  rating_max: 5.0
```

---

## 📈 Performans ve Optimizasyon

### GPU Kullanımı
```python
# Otomatik device seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch size ayarlama (GPU memory'ye göre)
# 8GB VRAM: batch_size=8
# 16GB VRAM: batch_size=16
# 24GB+ VRAM: batch_size=32
```

### Memory Optimization
```bash
# Gradient accumulation
python train_integrated_enhanced_model.py --batch_size 8 --accumulation_steps 2

# Mixed precision training
python train_integrated_enhanced_model.py --fp16
```

### Distributed Training
```bash
# Multi-GPU training
torchrun --nproc-per-node 4 train_integrated_enhanced_model.py
```

---

## 🐛 Sorun Giderme

### CUDA Out of Memory
```bash
# Batch size'ı küçült
python train_integrated_enhanced_model.py --batch_size 8

# Gradient checkpointing kullan
python train_integrated_enhanced_model.py --gradient_checkpointing
```

### SDV Kurulum Hatası
```bash
# Python sürümünü kontrol et (3.8+ gerekli)
python --version

# pip'i güncelle
pip install --upgrade pip

# Tekrar dene
pip install sdv
```

### Training Çok Yavaş
```bash
# Epoch sayısını azalt
python train_integrated_enhanced_model.py --epochs 100

# Eval interval'i artır
python train_integrated_enhanced_model.py --eval_interval 10
```

---

## 📚 Dokümantasyon

- [QUICK_START.md](QUICK_START.md) - Hızlı başlangıç kılavuzu
- [SDV_README.md](SDV_README.md) - SDV hızlı başlangıç
- [SDV_KULLANIM_KILAVUZU.md](SDV_KULLANIM_KILAVUZU.md) - Detaylı SDV kılavuzu
- [SDV_DOSYA_YAPISI.md](SDV_DOSYA_YAPISI.md) - SDV dosya yapısı
- [scraping/README.md](scraping/README.md) - Web scraping kılavuzu

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: ARC-AGI Benchmark
```bash
# Veri hazırla
python -m dataset.build_arc_dataset --output-dir data/arc1concept-aug-1000

# Model eğit
torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]"

# Değerlendir
python evaluate_arc.py --checkpoint checkpoints/arc1concept/best.pt
```

### Senaryo 2: Hediye Önerisi Sistemi
```bash
# Veri topla
python run_pipeline_root.py  # Web scraping
python sdv_advanced_generator.py  # Sentetik veri

# Model eğit
python train_integrated_enhanced_model.py

# Test et
python test_comprehensive_improvements.py

# Fine-tune
python finetune_category_diversity.py
```

### Senaryo 3: Özel Veri Seti
```bash
# Kendi verinizi hazırlayın
# data/my_custom_dataset.json

# Config oluşturun
# config/my_custom_config.yaml

# Eğitin
python train_integrated_enhanced_model.py --config config/my_custom_config.yaml
```

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📞 İletişim ve Destek

- **Issues**: GitHub Issues kullanın
- **Discussions**: GitHub Discussions
- **Email**: [email korunmuştur]

---

## 🙏 Teşekkürler

Bu proje şu çalışmalara dayanmaktadır:

### TRM (Tiny Recursion Model)
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

### HRM (Hierarchical Reasoning Model)
```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

### Kod Kaynakları
- [HRM Code](https://github.com/sapientinc/HRM)
- [HRM Analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)

---

## 🌟 Öne Çıkan Özellikler

### TRM Özellikleri
- ✅ Sadece 7M parametre ile %45 ARC-AGI-1 başarısı
- ✅ Recursive reasoning yaklaşımı
- ✅ Minimal overfitting
- ✅ Sudoku, Maze, ARC-AGI desteği
- ✅ EMA (Exponential Moving Average) desteği
- ✅ Multi-GPU distributed training

### Hediye Önerisi Özellikleri
- ✅ Tool-enhanced architecture (5 araç)
- ✅ Curriculum learning (4 aşama)
- ✅ SDV sentetik veri üretimi (3 yöntem)
- ✅ Web scraping (4 Türk e-ticaret sitesi)
- ✅ Integrated Enhanced TRM modeli
- ✅ 25+ kapsamlı test suite
- ✅ Checkpoint save/load/resume
- ✅ Fine-tuning desteği
- ✅ Real-time tool execution feedback
- ✅ Multi-component reward prediction

---

## 📊 Proje İstatistikleri

| Bileşen | Dosya Sayısı | Satır Sayısı (tahmini) |
|---------|--------------|------------------------|
| Models | 20+ | 5,000+ |
| Tests | 5 | 2,000+ |
| Configs | 7 | 500+ |
| Scripts | 15+ | 3,000+ |
| Docs | 5 | 2,000+ |
| **Toplam** | **50+** | **12,500+** |

---

## 🚦 Durum ve Yol Haritası

### Tamamlanan Özellikler ✅
- [x] TRM temel implementasyonu
- [x] ARC-AGI, Sudoku, Maze desteği
- [x] Tool-enhanced architecture
- [x] Integrated Enhanced TRM
- [x] SDV sentetik veri üretimi
- [x] Web scraping pipeline
- [x] Curriculum learning
- [x] Kapsamlı test suite
- [x] Fine-tuning desteği
- [x] Checkpoint management

### Devam Eden Çalışmalar 🔄
- [ ] Daha fazla e-ticaret sitesi desteği
- [ ] Gelişmiş tool parametreleri
- [ ] Multi-modal input desteği
- [ ] Real-time recommendation API

### Gelecek Planlar 🔮
- [ ] Transformer-based TRM variant
- [ ] Federated learning desteği
- [ ] Mobile deployment
- [ ] Web UI dashboard

---

## 💡 İpuçları ve En İyi Pratikler

### TRM Eğitimi İçin
1. **EMA kullanın**: Daha stabil sonuçlar için `ema=True`
2. **Warmup**: Learning rate warmup kullanın
3. **Batch size**: GPU memory'ye göre ayarlayın
4. **Eval interval**: Düzenli değerlendirme yapın

### Hediye Önerisi İçin
1. **Veri çeşitliliği**: Gerçek + sentetik veri karıştırın
2. **Curriculum learning**: Aşamalı öğrenme kullanın
3. **Tool feedback**: Araç sonuçlarını modele geri bildirin
4. **Fine-tuning**: Kategori çeşitliliği için fine-tune edin
5. **Test sık**: Her değişiklikten sonra test edin

### SDV Kullanımı İçin
1. **Küçük başlayın**: İlk denemede az örnek üretin
2. **Kalite kontrol**: Quality score'u kontrol edin (>0.80 hedef)
3. **Yöntem seçimi**: Gaussian (hızlı), CTGAN (kaliteli)
4. **Constraint kullanın**: Geçerli veri için kısıtlamalar ekleyin

---

## 🎉 Başarılar!

Projeyi kullandığınız için teşekkürler! Sorularınız için GitHub Issues'ı kullanabilirsiniz.

**Happy Training! 🚀**

---

*Son güncelleme: 2025*
*Versiyon: 2.0*
*Dil: Türkçe*
