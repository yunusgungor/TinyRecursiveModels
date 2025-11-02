# Email Classification with TRM (Tiny Recursive Reasoning Model)

Bu proje, PRD'de belirtilen Akıllı E-posta Düzenleyici için TRM (Tiny Recursive Reasoning Model) kullanarak e-posta sınıflandırma sistemi geliştirmektedir.

## 🎯 Proje Hedefi

PRD'de belirtilen %95+ doğruluk hedefine ulaşmak için TRM modelini e-posta sınıflandırma görevine adapte etmek ve eğitmek.

## 📋 E-posta Kategorileri

Sistem aşağıdaki 10 kategoriyi desteklemektedir:

1. **Newsletter** - Bültenler ve haber mektupları
2. **Work** - İş ile ilgili e-postalar
3. **Personal** - Kişisel e-postalar
4. **Spam** - İstenmeyen e-postalar
5. **Promotional** - Promosyon ve reklam e-postaları
6. **Social** - Sosyal medya bildirimleri
7. **Finance** - Finansal bildirimler
8. **Travel** - Seyahat ile ilgili e-postalar
9. **Shopping** - Alışveriş bildirimleri
10. **Other** - Diğer kategoriler

## 🏗️ Mimari Özellikleri

### TRM Model Adaptasyonu
- **Recursive Reasoning**: E-posta içeriğini iteratif olarak analiz eder
- **Adaptive Computation Time (ACT)**: Dinamik durma mekanizması
- **Parameter Efficiency**: Sadece 7M parametre ile yüksek performans
- **Classification Head**: E-posta kategorileri için özel sınıflandırma katmanı

### Teknik Özellikler
- **Vocabulary Size**: 5000 token (dinamik olarak ayarlanır)
- **Sequence Length**: 512 token (e-posta uzunluğuna göre)
- **Hidden Size**: 256-512 (konfigürasyona göre)
- **Reasoning Cycles**: H_cycles=2, L_cycles=3-4
- **Position Encoding**: RoPE (Rotary Position Embedding)

## 🚀 Kurulum ve Kullanım

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

Temel gereksinimler:
- PyTorch >= 1.12
- transformers
- scikit-learn
- numpy
- pandas
- wandb (opsiyonel, eğitim takibi için)

### 2. Veri Hazırlama

#### Örnek Veri ile Test
```bash
python run_email_training.py --sample-data --max-steps 1000
```

#### Kendi Veriniz ile
E-posta verilerinizi JSON formatında hazırlayın:

```json
[
  {
    "id": "email_001",
    "subject": "Weekly Newsletter - Tech Updates",
    "body": "Here are the latest tech updates...",
    "sender": "newsletter@techblog.com",
    "recipient": "user@example.com",
    "category": "newsletter"
  }
]
```

Veri setini oluşturun:
```bash
python dataset/build_email_dataset.py \
    --input_file data/emails.json \
    --output_dir data/email-classification \
    --num_aug 100 \
    --max_seq_len 512
```

### 3. Model Eğitimi

#### Tek GPU ile Eğitim
```bash
python train_email_classifier.py \
    data_paths=[data/email-classification] \
    training.max_steps=10000 \
    training.batch_size=32
```

#### Çoklu GPU ile Eğitim
```bash
torchrun --nproc-per-node 4 train_email_classifier.py \
    data_paths=[data/email-classification] \
    training.max_steps=10000 \
    training.batch_size=128
```

#### Tam Pipeline
```bash
python run_email_training.py \
    --num-gpus 4 \
    --batch-size 128 \
    --max-steps 10000
```

### 4. Model Değerlendirme

```python
from models.recursive_reasoning.trm_email import EmailTRM
from evaluators.email import evaluate_email_model
import torch

# Model yükleme
checkpoint = torch.load('outputs/email_classification/best_model.pt')
model = EmailTRM(checkpoint['config']['arch'])
model.load_state_dict(checkpoint['model_state_dict'])

# Değerlendirme
metrics = evaluate_email_model(model, test_dataloader, device)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['macro_f1']:.4f}")
```

## 📊 Performans Metrikleri

Model aşağıdaki metriklerle değerlendirilir:

- **Accuracy**: Genel doğruluk oranı
- **Precision/Recall/F1**: Kategori bazlı performans
- **Macro/Micro/Weighted F1**: Farklı ortalama türleri
- **Confusion Matrix**: Kategori karışıklık matrisi
- **Confidence Metrics**: Model güven skorları

## 🔧 Konfigürasyon

### Model Parametreleri (`config/arch/trm_email.yaml`)
```yaml
# Model boyutu
hidden_size: 512
num_heads: 8
L_layers: 2

# Recursive reasoning
H_cycles: 2
L_cycles: 4
halt_max_steps: 8

# E-posta özgü
num_email_categories: 10
classification_dropout: 0.1
```

### Eğitim Parametreleri (`config/cfg_email_train.yaml`)
```yaml
training:
  max_steps: 10000
  batch_size: 32
  lr: 1e-4

optimizer:
  name: "adamw"
  weight_decay: 0.1

scheduler:
  name: "linear_warmup_cosine"
  warmup_steps: 500
```

## 📈 Eğitim Takibi

### Weights & Biases (WandB)
```bash
# WandB ile eğitim takibi
export WANDB_PROJECT="email-classification-trm"
python train_email_classifier.py use_wandb=true
```

### Yerel Loglar
Eğitim logları ve metrikler `outputs/email_classification/` dizininde saklanır:
- `best_model.pt`: En iyi model
- `final_metrics.json`: Son değerlendirme metrikleri
- `confusion_matrix.png`: Karışıklık matrisi grafiği

## 🎛️ Hiperparametre Optimizasyonu

### Önerilen Başlangıç Değerleri
```yaml
# Hızlı test için
hidden_size: 256
H_cycles: 2
L_cycles: 3
batch_size: 32
max_steps: 5000

# Yüksek performans için
hidden_size: 512
H_cycles: 3
L_cycles: 6
batch_size: 64
max_steps: 20000
```

### Grid Search Örneği
```bash
# Farklı learning rate'ler test etme
for lr in 1e-4 5e-5 1e-5; do
    python train_email_classifier.py \
        optimizer.lr=$lr \
        experiment_name="lr_${lr}"
done
```

## 🚀 Production Deployment

### Model Inference
```python
import torch
from models.recursive_reasoning.trm_email import EmailTRM

class EmailClassifier:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = EmailTRM(checkpoint['config']['arch'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Vocabulary ve kategoriler
        self.vocab = checkpoint['vocab']
        self.categories = checkpoint['categories']
    
    def predict(self, email_text):
        # Tokenize email
        tokens = self.tokenize(email_text)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(tokens)
            prediction = torch.argmax(outputs['logits'], dim=-1)
        
        return self.categories[prediction.item()]
```

### API Servisi
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
classifier = EmailClassifier('best_model.pt')

class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str

@app.post("/classify")
async def classify_email(email: EmailRequest):
    category = classifier.predict(email.dict())
    return {"category": category}
```

## 📝 Sonuçlar ve Analiz

### Beklenen Performans
- **Hedef Accuracy**: %95+ (PRD gereksinimi)
- **Eğitim Süresi**: 2-4 saat (4 GPU ile)
- **Model Boyutu**: ~7M parametre
- **Inference Hızı**: <100ms per email

### Performans İyileştirme İpuçları
1. **Veri Artırma**: Daha fazla augmentasyon kullanın
2. **Sequence Length**: E-posta uzunluğuna göre optimize edin
3. **Reasoning Cycles**: Karmaşık e-postalar için artırın
4. **Ensemble**: Birden fazla model kombinasyonu

## 🐛 Sorun Giderme

### Yaygın Sorunlar
1. **CUDA Out of Memory**: Batch size'ı azaltın
2. **Düşük Accuracy**: Daha fazla eğitim verisi ekleyin
3. **Overfitting**: Dropout ve weight decay artırın
4. **Slow Training**: Mixed precision kullanın

### Debug Komutları
```bash
# Model parametrelerini kontrol et
python -c "from models.recursive_reasoning.trm_email import EmailTRM; print(EmailTRM.from_config().num_parameters())"

# Veri setini kontrol et
python -c "from puzzle_dataset import PuzzleDataset; ds = PuzzleDataset(...); print(next(iter(ds)))"
```

## 📚 Referanslar

- [TRM Paper: "Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)
- [Original TRM Repository](https://github.com/AlexiaJM/TinyRecursiveReasoningModel)
- [Akıllı E-posta Düzenleyici PRD](PRD.md)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/email-classification`)
3. Commit yapın (`git commit -am 'Add email classification'`)
4. Push yapın (`git push origin feature/email-classification`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.