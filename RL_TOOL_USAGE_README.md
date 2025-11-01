# RL ve Tool Kullanımı ile TRM Eğitimi

Bu dokümantasyon, TRM (Tiny Recursive Model) modelini Reinforcement Learning (RL) ve Tool kullanımı ile nasıl eğiteceğinizi açıklar.

## 🎯 Özellikler

### Reinforcement Learning (RL) Entegrasyonu
- **PPO (Proximal Policy Optimization)** algoritması
- **Experience replay** ve **advantage estimation**
- **Multi-step episodes** ile gerçekçi eğitim
- **Real-time reward feedback** sistemi
- **Evaluation metrics** ve **performance tracking**

### Tool Kullanımı
- **5 farklı tool** entegrasyonu:
  - `price_comparison`: Fiyat karşılaştırma
  - `inventory_check`: Stok kontrolü
  - `review_analysis`: Yorum analizi
  - `trend_analysis`: Trend analizi
  - `budget_optimizer`: Bütçe optimizasyonu
- **Adaptive tool selection** - model hangi tool'u ne zaman kullanacağını öğrenir
- **Tool result encoding** - tool sonuçları model state'ine entegre edilir
- **Caching system** - tool çağrıları cache'lenir
- **Performance monitoring** - tool kullanım istatistikleri

## 📁 Kod Yapısı

```
models/
├── rl/                          # RL Components
│   ├── __init__.py
│   ├── environment.py           # Gift recommendation environment
│   ├── rl_trm.py               # RL-enhanced TRM model
│   └── trainer.py              # RL training infrastructure
├── tools/                       # Tool Components
│   ├── __init__.py
│   ├── tool_registry.py        # Tool management system
│   ├── gift_tools.py           # Gift-specific tools
│   └── tool_enhanced_trm.py    # Tool-enhanced TRM model
config/
├── rl_gift_recommendation.yaml      # RL training config
└── tool_enhanced_gift_recommendation.yaml  # Tool training config
utils/
└── data_generator.py           # Synthetic data generation
train_rl_gift_recommendation.py     # RL training script
train_tool_enhanced_gift_recommendation.py  # Tool training script
test_rl_tool_integration.py    # Test script
```

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

```bash
# Mevcut TRM requirements'ları + ek paketler
pip install torch torchvision torchaudio
pip install wandb  # Opsiyonel: logging için
pip install requests  # Tool'lar için
pip install pandas numpy scikit-learn
```

### 2. Test Etme

Önce her şeyin çalıştığından emin olun:

```bash
# Tüm testleri çalıştır
python test_rl_tool_integration.py --test all

# Sadece RL testi
python test_rl_tool_integration.py --test rl

# Sadece Tool testi
python test_rl_tool_integration.py --test tools
```

### 3. Veri Hazırlama

```bash
# Synthetic veri oluştur
python utils/data_generator.py
```

Bu komut şunları oluşturur:
- `data/gift_catalog.json` - Hediye kataloğu
- `data/gift_recommendation_train.json` - Eğitim verisi
- `data/gift_recommendation_test.json` - Test verisi

## 🎓 Eğitim Senaryoları

### Senaryo 1: Sadece RL Eğitimi

```bash
# Temel RL eğitimi
python train_rl_gift_recommendation.py \
    --config config/rl_gift_recommendation.yaml \
    --wandb

# Debug mode (küçük model, hızlı test)
python train_rl_gift_recommendation.py \
    --config config/rl_gift_recommendation.yaml \
    --debug
```

**Beklenen Sonuçlar:**
- Eğitim süresi: 2-4 saat (Mac'te)
- Model boyutu: ~10MB
- Final reward: 0.6-0.8

### Senaryo 2: Tool-Enhanced Eğitim

```bash
# Tam tool-enhanced eğitim (3 aşamalı)
python train_tool_enhanced_gift_recommendation.py \
    --config config/tool_enhanced_gift_recommendation.yaml \
    --phase all \
    --wandb

# Sadece tool öğrenme aşaması
python train_tool_enhanced_gift_recommendation.py \
    --config config/tool_enhanced_gift_recommendation.yaml \
    --phase phase2
```

**3 Aşamalı Eğitim:**
1. **Phase 1**: Supervised pre-training (500 epoch)
2. **Phase 2**: Tool usage learning (300 epoch)
3. **Phase 3**: RL fine-tuning with tools (700 epoch)

**Beklenen Sonuçlar:**
- Eğitim süresi: 6-8 saat (Mac'te)
- Model boyutu: ~15MB
- Final reward: 0.7-0.9 (tool'lar sayesinde daha yüksek)
- Tool kullanım oranı: %60-80

### Senaryo 3: Özelleştirilmiş Eğitim

Kendi config'inizi oluşturun:

```yaml
# custom_config.yaml
arch:
  hidden_size: 128  # Mac için küçük
  L_layers: 1
  H_cycles: 2
  L_cycles: 2
  max_tool_calls_per_step: 1  # Az tool kullanımı

rl_training:
  num_episodes: 500  # Kısa eğitim
  batch_size: 8
  eval_frequency: 25
```

```bash
python train_tool_enhanced_gift_recommendation.py \
    --config custom_config.yaml \
    --debug
```

## 🔧 Konfigürasyon Seçenekleri

### Model Parametreleri

```yaml
arch:
  # TRM temel parametreleri
  hidden_size: 256        # Model boyutu
  L_layers: 2            # Layer sayısı
  H_cycles: 3            # Recursive cycles
  L_cycles: 4
  
  # RL parametreleri
  action_space_size: 100  # Maksimum hediye sayısı
  max_recommendations: 5  # Önerilen hediye sayısı
  
  # Tool parametreleri
  max_tool_calls_per_step: 3    # Adım başına max tool
  tool_call_threshold: 0.5      # Tool kullanım eşiği
  tool_fusion_method: "concatenate"  # Tool entegrasyon yöntemi
```

### Eğitim Parametreleri

```yaml
rl_training:
  num_episodes: 2000      # Toplam episode
  batch_size: 32          # Batch boyutu
  learning_rate: 1e-4     # Öğrenme oranı
  gamma: 0.99            # Discount factor
  eval_frequency: 50      # Değerlendirme sıklığı
```

## 📊 Monitoring ve Değerlendirme

### Weights & Biases Integration

```bash
# WandB ile eğitim
python train_tool_enhanced_gift_recommendation.py \
    --config config/tool_enhanced_gift_recommendation.yaml \
    --wandb
```

**Takip Edilen Metrikler:**
- `avg_reward`: Ortalama episode reward'u
- `eval_reward_mean`: Değerlendirme reward'u
- `policy_loss`: Policy loss
- `value_loss`: Value loss
- `tool_calls`: Tool kullanım sayısı
- `tool_avg_time`: Ortalama tool execution süresi

### Manuel Monitoring

```python
# Eğitim sırasında model istatistikleri
model = ToolEnhancedTRM(config)
# ... eğitim ...

# Tool kullanım istatistikleri
stats = model.get_tool_usage_stats()
print(f"Total tool calls: {stats['total_calls']}")
print(f"Most used tool: {stats['most_used_tool']}")
print(f"Success rates: {stats['success_rates']}")
```

## 🎯 Performans Optimizasyonu

### Mac Bilgisayar İçin Optimizasyonlar

```yaml
# Mac-optimized config
arch:
  hidden_size: 128      # 256 yerine
  L_layers: 1          # 2 yerine
  H_cycles: 2          # 3 yerine
  max_tool_calls_per_step: 1  # 3 yerine

global_batch_size: 8   # 16 yerine
rl_training:
  batch_size: 16       # 32 yerine
  experience_buffer_size: 5000  # 10000 yerine
```

### Memory Management

```python
# Gradient accumulation için
if batch_idx % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()

# Tool cache'i temizle
model.tool_registry.clear_cache()
```

## 🐛 Troubleshooting

### Yaygın Problemler

**1. Memory Error (Mac'te)**
```bash
# Batch size'ı küçült
--debug flag'i kullan
# veya config'te batch_size: 4
```

**2. Tool Timeout**
```yaml
tools:
  timeout: 60  # 30'dan 60'a çıkar
  max_concurrent_calls: 2  # 5'ten 2'ye düşür
```

**3. Düşük Reward**
```yaml
# Daha fazla exploration
arch:
  halt_exploration_prob: 0.2  # 0.1'den artır
  epsilon: 0.2  # epsilon-greedy için
```

**4. Tool Kullanımı Az**
```yaml
arch:
  tool_call_threshold: 0.3  # 0.5'ten düşür
  tool_usage_reward_weight: 0.2  # 0.1'den artır
```

### Debug Mode

```bash
# Detaylı logging
export PYTHONPATH=.
python -u train_tool_enhanced_gift_recommendation.py \
    --config config/tool_enhanced_gift_recommendation.yaml \
    --debug \
    --wandb 2>&1 | tee training.log
```

## 📈 Sonuçları Değerlendirme

### Başarı Kriterleri

**RL-Only Model:**
- Final reward > 0.6
- Evaluation reward artış trendi
- Episode length stabilizasyonu

**Tool-Enhanced Model:**
- Final reward > 0.7
- Tool kullanım oranı > 50%
- Tool success rate > 80%
- Çeşitli tool'ların kullanılması

### Model Karşılaştırması

```python
# İki modeli karşılaştır
def compare_models(rl_model_path, tool_model_path):
    # Load models
    rl_model = torch.load(rl_model_path)
    tool_model = torch.load(tool_model_path)
    
    # Test on same scenarios
    test_scenarios = generate_test_scenarios(100)
    
    rl_rewards = evaluate_model(rl_model, test_scenarios)
    tool_rewards = evaluate_model(tool_model, test_scenarios)
    
    print(f"RL-only average reward: {np.mean(rl_rewards):.3f}")
    print(f"Tool-enhanced average reward: {np.mean(tool_rewards):.3f}")
    print(f"Improvement: {(np.mean(tool_rewards) - np.mean(rl_rewards)):.3f}")
```

## 🚀 Production Deployment

### Model Export

```python
# Optimized model için
model = ToolEnhancedTRM(config)
model.load_state_dict(torch.load("best_model.pt"))

# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# TorchScript
scripted_model = torch.jit.script(quantized_model)
torch.jit.save(scripted_model, "production_model.pt")
```

### API Servisi

```python
# FastAPI ile deployment
from fastapi import FastAPI
app = FastAPI()

@app.post("/recommend")
async def recommend_gifts(user_profile: dict):
    # Load model
    model = torch.jit.load("production_model.pt")
    
    # Generate recommendations
    recommendations = model.recommend(user_profile)
    
    return {"recommendations": recommendations}
```

## 📚 İleri Seviye Kullanım

### Custom Tool Ekleme

```python
# Yeni tool oluştur
class CustomTool(BaseTool):
    def __init__(self):
        super().__init__("custom_tool", "My custom tool")
    
    def execute(self, **kwargs):
        # Tool logic
        return {"result": "custom_result"}
    
    def _get_parameter_schema(self):
        return {"type": "object", "properties": {...}}

# Model'e ekle
model.tool_registry.register_tool(CustomTool())
```

### Custom Reward Function

```python
# Özel reward hesaplama
def custom_reward_function(recommendations, user_feedback):
    base_reward = calculate_base_reward(recommendations, user_feedback)
    
    # Özel kriterler
    diversity_bonus = calculate_diversity(recommendations)
    price_penalty = calculate_price_penalty(recommendations, user_feedback)
    
    return base_reward + diversity_bonus - price_penalty
```

### Multi-Agent Training

```python
# Birden fazla agent ile eğitim
agents = [
    ToolEnhancedTRM(config) for _ in range(4)
]

# Population-based training
for generation in range(100):
    # Her agent'ı eğit
    for agent in agents:
        train_agent(agent, episodes=50)
    
    # En iyi agent'ları seç ve mutate et
    best_agents = select_best(agents, top_k=2)
    agents = mutate_and_reproduce(best_agents)
```

Bu dokümantasyon ile TRM modelinizi RL ve tool kullanımı ile başarıyla eğitebilirsiniz. Sorularınız için GitHub issues kullanabilirsiniz.

## 🎉 Sonuç

Bu entegrasyon ile TRM modeliniz:
- **Gerçek zamanlı feedback** ile öğrenir
- **External tool'ları** akıllıca kullanır
- **Sürekli iyileşen** öneriler sunar
- **Production-ready** deployment'a hazır hale gelir

Başarılar! 🚀