# 🎁 TRM-Based AI Gift Recommendation System

**Tiny Recursive Model (TRM) tabanlı, Tool-Augmented ve Reinforcement Learning ile güçlendirilmiş akıllı hediye öneri sistemi**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Temel Özellikler](#-temel-özellikler)
- [Mimari](#-mimari)
- [Kurulum](#-kurulum)
- [Veri Pipeline](#-veri-pipeline)
- [Model Eğitimi](#-model-eğitimi)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

---

## 🎯 Genel Bakış

Bu proje, **Tiny Recursive Model (TRM)** mimarisini temel alarak geliştirilmiş, **Reinforcement Learning (RL)** ve **Tool-Augmented Reasoning** ile güçlendirilmiş bir hediye öneri sistemidir. Sistem, gerçek e-ticaret sitelerinden toplanan verilerle eğitilir ve kullanıcı profiline göre kişiselleştirilmiş hediye önerileri sunar.

### 🌟 Neden Bu Proje?

- **🧠 Akıllı Reasoning**: TRM'nin recursive reasoning yetenekleri ile derin analiz
- **🔧 Tool Integration**: 5 farklı tool ile zenginleştirilmiş karar verme
- **🎮 RL Training**: PPO-style reinforcement learning ile optimize edilmiş öneriler
- **📊 Gerçek Veri**: Türkiye'nin önde gelen e-ticaret sitelerinden toplanan gerçek ürün verileri
- **🤖 AI Enhancement**: Gemini API ile zenginleştirilmiş ürün metadata'sı
- **🎲 Synthetic Data**: SDV ile oluşturulan gerçekçi kullanıcı senaryoları

---

## ✨ Temel Özellikler

### 🔍 1. Web Scraping Pipeline

Gerçek e-ticaret sitelerinden otomatik veri toplama:

- **Desteklenen Siteler**: Çiçek Sepeti, Hepsiburada, Trendyol
- **Anti-Bot Protection**: Rate limiting, user agent rotation, CAPTCHA detection
- **Veri Validasyonu**: Pydantic ile güçlü veri doğrulama
- **AI Enhancement**: Gemini API ile ürün verilerini zenginleştirme

```bash
# Scraping pipeline'ı çalıştır
python scraping/scripts/scraping.py --website trendyol --max-products 500
```

### 🎲 2. Synthetic Data Generation

SDV (Synthetic Data Vault) kullanarak gerçekçi kullanıcı senaryoları oluşturma:

- **Dinamik Kategori Çıkarımı**: Gerçek veriden otomatik kategori tespiti
- **Gerçekçi Profiller**: Yaş, hobi, bütçe, ilişki, özel gün kombinasyonları
- **Çeşitlilik**: 100+ farklı kullanıcı senaryosu

### 🧠 3. Integrated Enhanced TRM Model

Tüm geliştirmeler model mimarisine entegre edilmiş:

#### a) Enhanced User Profiling
- **Hobby Embeddings**: Kullanıcı hobilerinin semantik temsili
- **Preference Encoding**: Kişilik özelliklerinin vektör temsili
- **Occasion Awareness**: Özel günlere göre uyarlama
- **Age & Budget Encoding**: Yaş ve bütçe bilgisinin sürekli kodlaması

#### b) Enhanced Category Matching
- **Semantic Matching**: Çok katmanlı semantik eşleştirme ağı
- **Category Attention**: Multi-head attention ile kategori skorlama
- **Dynamic Categories**: Veri setinden dinamik kategori yükleme

#### c) Enhanced Tool Selection
- **Context-Aware Selection**: Kullanıcı bağlamına göre tool seçimi
- **Tool Diversity**: Çeşitli tool kullanımını teşvik eden mekanizma
- **Parameter Generation**: Her tool için otomatik parametre üretimi

#### d) Enhanced Reward Prediction
- **Multi-Component Rewards**: 7 farklı reward bileşeni
  - Category match
  - Budget compatibility
  - Hobby alignment
  - Occasion appropriateness
  - Age appropriateness
  - Quality score
  - Diversity bonus
- **Reward Fusion**: Çok katmanlı fusion network

### 🔧 4. Tool System

5 farklı tool ile zenginleştirilmiş karar verme:

| Tool | Açıklama | Kullanım Senaryosu |
|------|----------|-------------------|
| **Price Comparison** | Fiyat karşılaştırma ve bütçe filtreleme | Bütçeye uygun hediye bulma |
| **Inventory Check** | Stok durumu kontrolü | Mevcut ürünleri belirleme |
| **Review Analysis** | Ürün yorumlarını analiz etme | Kaliteli ürünleri seçme |
| **Trend Analysis** | Trend ve popülerlik analizi | Popüler hediyeleri bulma |
| **Budget Optimizer** | Bütçe optimizasyonu | Bütçeyi en iyi şekilde kullanma |

### 🎮 5. Reinforcement Learning

PPO-style training ile optimize edilmiş öneriler:

- **Experience Replay**: Geçmiş deneyimlerden öğrenme
- **Value Estimation**: Durum değeri tahmini
- **Policy Optimization**: PPO clip ratio ile policy güncelleme
- **Entropy Regularization**: Keşif-sömürü dengesi

---

## 🏗️ Mimari

### Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRM Gift Recommendation System               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Web Scraping (Çiçek Sepeti, Hepsiburada, Trendyol)         │
│  2. AI Enhancement (Gemini API)                                  │
│  3. Synthetic Data Generation (SDV)                              │
│  4. Dataset Creation (Gift Catalog + User Scenarios)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Integrated Enhanced TRM Model                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ User Profiling  │  │ Category Match  │  │ Tool Selection  │ │
│  │                 │  │                 │  │                 │ │
│  │ • Hobby Embed   │  │ • Semantic      │  │ • Context-Aware │ │
│  │ • Preference    │  │ • Attention     │  │ • Diversity     │ │
│  │ • Occasion      │  │ • Scoring       │  │ • Parameters    │ │
│  │ • Age/Budget    │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Cross-Modal Fusion & RL Components             │  │
│  │                                                           │  │
│  │  • Multi-head Attention Layers                           │  │
│  │  • Policy Head (Action Probabilities)                    │  │
│  │  • Value Head (State Value Estimation)                   │  │
│  │  • Reward Predictor (Multi-Component)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Tool System                              │
├─────────────────────────────────────────────────────────────────┤
│  Price Comparison │ Inventory Check │ Review Analysis           │
│  Trend Analysis   │ Budget Optimizer                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gift Recommendations                          │
└─────────────────────────────────────────────────────────────────┘
```

### Model Detayları

**Integrated Enhanced TRM** modeli şu bileşenlerden oluşur:

1. **Base TRM**: Recursive reasoning için temel mimari
2. **RL Heads**: Policy, value ve reward prediction heads
3. **Enhanced Components**: User profiling, category matching, tool selection
4. **Cross-Modal Fusion**: User-gift-tool etkileşimlerini birleştiren attention layers

**Model Parametreleri**:
- Hidden Size: 512
- Attention Heads: 8
- L Layers: 3
- H Layers: 3
- H Cycles: 2
- L Cycles: 3
- Action Space: 50 (max gifts)
- Max Recommendations: 3

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU eğitimi için)
- 16GB+ RAM
- 10GB+ Disk alanı

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/trm-gift-recommendation.git
cd trm-gift-recommendation
```

### Adım 2: Sanal Ortam Oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
# Ana bağımlılıklar
pip install -r requirements.txt

# Scraping için ek bağımlılıklar
pip install -r scraping/requirements.txt

# Playwright browser kurulumu
playwright install chromium
```

### Adım 4: Environment Variables

Scraping için Gemini API key'inizi ayarlayın:

```bash
cd scraping
cp .env.example .env
# .env dosyasını düzenleyip GEMINI_API_KEY ekleyin
```

---

## 📊 Veri Pipeline

### 1. Web Scraping

Gerçek e-ticaret sitelerinden veri toplama:

```bash
# Tek siteden scraping
python scraping/scripts/scraping.py --website trendyol --max-products 500

# Tüm sitelerden scraping
python scraping/scripts/scraping.py --max-products 1000

# Test modu (hızlı test)
python scraping/scripts/scraping.py --test
```

**Çıktı**: `data/scraped_gift_catalog.json`

### 2. Synthetic Data Generation

Kullanıcı senaryoları oluşturma:

```bash
# Scraping ile birlikte otomatik oluşturulur
# Veya manuel test:
python scraping/scripts/test_scenario_generator.py
```

**Çıktı**: `data/user_scenarios.json`

### 3. Dataset Yapısı

#### Gift Catalog Format

```json
{
  "gifts": [
    {
      "id": "trendyol_0000",
      "name": "Ürün Adı",
      "category": "technology",
      "price": 299.90,
      "rating": 4.5,
      "tags": ["smart", "portable", "practical"],
      "age_range": [18, 65],
      "occasions": ["birthday", "christmas", "graduation"]
    }
  ],
  "metadata": {
    "total_gifts": 500,
    "categories": ["technology", "home", "beauty", "health", "kitchen"],
    "price_range": {"min": 10, "max": 50000, "avg": 1250}
  }
}
```

#### User Scenarios Format

```json
{
  "scenarios": [
    {
      "id": "scenario_0000",
      "profile": {
        "age": 35,
        "hobbies": ["technology", "gaming", "reading"],
        "relationship": "friend",
        "budget": 500,
        "occasion": "birthday",
        "preferences": ["practical", "modern", "innovative"]
      },
      "expected_categories": ["technology", "gaming"],
      "expected_tools": ["price_comparison", "review_analysis"]
    }
  ]
}
```

---

## 🎓 Model Eğitimi

### Pretrain (Temel Eğitim)

TRM modelini temel görevler üzerinde eğitme:

```bash
# ARC dataset ile pretrain
python pretrain.py \
  --data_paths data/arc-aug-1000 \
  --global_batch_size 768 \
  --epochs 100000 \
  --lr 1e-4 \
  --eval_interval 10000
```

### Fine-tune (Hediye Önerisi için)

Pretrain edilmiş modeli hediye önerisi için fine-tune etme:

```bash
# Gift recommendation için fine-tune
python pretrain.py \
  --data_paths data/gift_recommendation \
  --load_checkpoint checkpoints/pretrained_model \
  --global_batch_size 256 \
  --epochs 50000 \
  --lr 5e-5
```

### Eğitim Parametreleri

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `global_batch_size` | Global batch size | 768 |
| `lr` | Learning rate | 1e-4 |
| `lr_min_ratio` | Minimum LR ratio | 1.0 |
| `lr_warmup_steps` | Warmup steps | 2000 |
| `weight_decay` | Weight decay | 0.1 |
| `beta1` | Adam beta1 | 0.9 |
| `beta2` | Adam beta2 | 0.95 |
| `ema` | Use EMA | False |
| `ema_rate` | EMA rate | 0.999 |

### Distributed Training

Çoklu GPU ile eğitim:

```bash
# 4 GPU ile eğitim
torchrun --nproc_per_node=4 pretrain.py \
  --data_paths data/gift_recommendation \
  --global_batch_size 1024
```

---

## 💻 Kullanım

### 1. Model Yükleme

```python
from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config

# Config oluştur
config = create_integrated_enhanced_config()

# Model yükle
model = IntegratedEnhancedTRM(config)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
model.eval()
```

### 2. Hediye Önerisi Alma

```python
from models.rl.environment import UserProfile, EnvironmentState, GiftItem

# Kullanıcı profili oluştur
user = UserProfile(
    age=35,
    hobbies=["technology", "gaming"],
    relationship="friend",
    budget=500.0,
    occasion="birthday",
    personality_traits=["practical", "modern"]
)

# Mevcut hediyeler
gifts = [
    GiftItem("1", "Wireless Headphones", "technology", 450.0, 4.5, 
             ["wireless", "portable"], "Headphones", (16, 65), ["birthday"]),
    GiftItem("2", "Smart Watch", "technology", 800.0, 4.7,
             ["smart", "fitness"], "Watch", (18, 60), ["birthday"])
]

# Environment state oluştur
env_state = EnvironmentState(user, gifts, [], [], 0)

# Model ile öneri al
with torch.no_grad():
    carry = model.initial_carry({"inputs": torch.randn(50), 
                                 "puzzle_identifiers": torch.zeros(1, dtype=torch.long)})
    carry, rl_output, selected_tools = model.forward_with_enhancements(
        carry, env_state, gifts
    )
    
    # Action seç
    action = model.select_action(rl_output["action_probs"], gifts, deterministic=True)
    
    print("Önerilen Hediyeler:")
    for gift in action["selected_gifts"]:
        print(f"  - {gift.name} ({gift.price} TL)")
```

### 3. Tool Kullanımı

```python
from models.tools.tool_registry import ToolRegistry
from models.tools.gift_tools import GiftRecommendationTools

# Tool registry oluştur
registry = ToolRegistry()
gift_tools = GiftRecommendationTools()

# Toolları kaydet
for tool in gift_tools.get_all_tools():
    registry.register_tool(tool)

# Price comparison tool kullan
result = registry.call_tool_by_name(
    "price_comparison",
    gifts=gifts,
    budget=500.0
)

print(f"Bütçeye uygun: {len(result.result['in_budget'])} ürün")
print(f"Bütçe dışı: {len(result.result['over_budget'])} ürün")
```

### 4. RL Training Loop

```python
# Experience toplama
experiences = []

for episode in range(num_episodes):
    env_state = create_random_environment()
    carry = model.initial_carry(batch)
    
    # Forward pass
    carry, rl_output, tool_calls = model.forward_with_tools(
        carry, env_state, available_gifts, max_tool_calls=2
    )
    
    # Action seç
    action = model.select_action(rl_output["action_probs"], available_gifts)
    
    # Reward hesapla
    reward = calculate_reward(action, env_state)
    
    # Experience kaydet
    experience = {
        "state": env_state,
        "action": action,
        "reward": reward,
        "carry": carry,
        "env_state": env_state,
        "available_gifts": available_gifts,
        "log_prob": action["log_probs"],
        "value": rl_output["state_value"],
        "done": False
    }
    experiences.append(experience)

# RL loss hesapla ve optimize et
loss_dict = model.compute_rl_loss(experiences, gamma=0.99)
loss_dict["total_loss"].backward()
optimizer.step()
```

---

## 📁 Proje Yapısı

```
trm-gift-recommendation/
├── README.md                          # Bu dosya
├── requirements.txt                   # Ana bağımlılıklar
├── pretrain.py                        # Eğitim scripti
├── puzzle_dataset.py                  # Dataset loader
│
├── config/                            # Konfigürasyon dosyaları
│   ├── cfg_pretrain.yaml             # Pretrain config
│   └── arch/                         # Model mimarisi configs
│
├── data/                              # Veri dosyaları
│   ├── gift_catalog.json             # Hediye kataloğu
│   ├── user_scenarios.json           # Kullanıcı senaryoları
│   ├── fully_learned_synthetic_gifts.json
│   └── fully_learned_synthetic_users.json
│
├── dataset/                           # Dataset oluşturma
│   ├── build_arc_dataset.py         # ARC dataset builder
│   ├── build_maze_dataset.py        # Maze dataset builder
│   ├── build_sudoku_dataset.py      # Sudoku dataset builder
│   └── common.py                     # Ortak fonksiyonlar
│
├── models/                            # Model mimarileri
│   ├── common.py                     # Ortak model bileşenleri
│   ├── layers.py                     # Custom layers
│   ├── losses.py                     # Loss fonksiyonları
│   ├── ema.py                        # Exponential Moving Average
│   │
│   ├── recursive_reasoning/          # TRM mimarisi
│   │   └── trm.py                    # Tiny Recursive Model
│   │
│   ├── rl/                           # RL bileşenleri
│   │   ├── rl_trm.py                # RL-enhanced TRM
│   │   └── environment.py           # RL environment
│   │
│   └── tools/                        # Tool system
│       ├── tool_registry.py         # Tool registry
│       ├── gift_tools.py            # Gift-specific tools
│       └── integrated_enhanced_trm.py  # Ana model
│
├── scraping/                          # Web scraping pipeline
│   ├── README.md                     # Scraping dokümantasyonu
│   ├── requirements.txt              # Scraping bağımlılıkları
│   ├── .env                          # Environment variables
│   │
│   ├── config/                       # Scraping configs
│   │   └── scraping_config.yaml
│   │
│   ├── scrapers/                     # Web scrapers
│   │   ├── base_scraper.py
│   │   ├── ciceksepeti_scraper.py
│   │   ├── hepsiburada_scraper.py
│   │   ├── trendyol_scraper.py
│   │   └── orchestrator.py
│   │
│   ├── services/                     # Servisler
│   │   ├── gemini_service.py        # AI enhancement
│   │   └── dataset_generator.py     # Dataset oluşturma
│   │
│   ├── utils/                        # Yardımcı araçlar
│   │   ├── models.py                # Pydantic models
│   │   ├── validator.py             # Veri validasyonu
│   │   ├── rate_limiter.py          # Rate limiting
│   │   ├── anti_bot.py              # Anti-bot protection
│   │   └── logger.py                # Logging
│   │
│   └── scripts/                      # Scraping scriptleri
│       ├── scraping.py              # Ana scraping script
│       └── test_scenario_generator.py
│
├── evaluators/                        # Model değerlendirme
│   └── arc.py                        # ARC evaluator
│
├── utils/                             # Genel yardımcı araçlar
│   └── functions.py                  # Yardımcı fonksiyonlar
│
├── logs/                              # Log dosyaları
│   ├── scraping.log
│   └── scraping_errors.log
│
└── checkpoints/                       # Model checkpoints
    └── [model_checkpoints]
```

---

## 🔧 Konfigürasyon

### Scraping Konfigürasyonu

`scraping/config/scraping_config.yaml`:

```yaml
scraping:
  websites:
    - name: "trendyol"
      enabled: true
      max_products: 500
      categories: ["teknoloji", "ev", "guzellik"]
  
rate_limit:
  requests_per_minute: 20
  delay_between_requests: [2, 5]
  max_concurrent_requests: 10

gemini:
  model: "gemini-1.5-flash"
  max_requests_per_day: 1000
  retry_attempts: 3

output:
  final_dataset_path: "data/gift_catalog.json"
  user_scenarios_path: "data/user_scenarios.json"
  num_user_scenarios: 100
```

### Model Konfigürasyonu

`config/cfg_pretrain.yaml`:

```yaml
# Data paths
data_paths: ['data/gift_recommendation']
data_paths_test: []

# Training hyperparameters
global_batch_size: 768
epochs: 100000
eval_interval: 10000

lr: 1e-4
lr_min_ratio: 1.0
lr_warmup_steps: 2000

weight_decay: 0.1
beta1: 0.9
beta2: 0.95

# Model architecture
arch:
  name: integrated_enhanced_trm
  hidden_size: 512
  num_heads: 8
  L_layers: 3
  H_layers: 3
```

---

## 📈 Değerlendirme

### Metrikler

Model performansı şu metriklerle değerlendirilir:

1. **Recommendation Accuracy**: Önerilen hediyelerin doğruluğu
2. **Category Match Score**: Kategori eşleşme skoru
3. **Budget Compliance**: Bütçeye uygunluk oranı
4. **User Satisfaction**: Kullanıcı memnuniyeti (simüle edilmiş)
5. **Tool Usage Efficiency**: Tool kullanım verimliliği
6. **Diversity Score**: Öneri çeşitliliği

### Değerlendirme Scripti

```python
from evaluators.arc import ARCEvaluator

# Evaluator oluştur
evaluator = ARCEvaluator(
    data_path="data/gift_recommendation",
    eval_metadata=eval_metadata
)

# Değerlendirme yap
metrics = evaluator.result(save_path="results/")

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Category Match: {metrics['category_match']:.2%}")
print(f"Budget Compliance: {metrics['budget_compliance']:.2%}")
```

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları takip edin:

1. **Fork** edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### Geliştirme Kuralları

- Code style: PEP 8
- Docstring: Google style
- Type hints kullanın
- Unit testler ekleyin
- README'yi güncelleyin

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 🙏 Teşekkürler

- **TRM Architecture**: Tiny Recursive Model mimarisi için
- **PyTorch**: Derin öğrenme framework'ü için
- **Gemini API**: AI enhancement için
- **SDV**: Synthetic data generation için
- **Playwright**: Web scraping için

---

## 📧 İletişim

Sorularınız veya önerileriniz için:

- **Email**: your.email@example.com
- **GitHub Issues**: [Issues](https://github.com/yourusername/trm-gift-recommendation/issues)

---

## 🔮 Gelecek Planları

- [ ] Multi-modal input support (resim, ses)
- [ ] Real-time recommendation API
- [ ] Web interface
- [ ] Mobile app
- [ ] Daha fazla e-ticaret sitesi desteği
- [ ] Collaborative filtering entegrasyonu
- [ ] A/B testing framework
- [ ] Production deployment guide

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!**
