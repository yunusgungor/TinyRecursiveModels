# BuildKit Kurulum ve Yapılandırma Rehberi

Bu doküman, Docker BuildKit'in kurulumu, yapılandırması ve kullanımı hakkında detaylı bilgi sağlar.

## İçindekiler

1. [BuildKit Nedir?](#buildkit-nedir)
2. [Kurulum](#kurulum)
3. [Yapılandırma](#yapılandırma)
4. [Kullanım](#kullanım)
5. [Optimizasyon İpuçları](#optimizasyon-ipuçları)
6. [Sorun Giderme](#sorun-giderme)

## BuildKit Nedir?

BuildKit, Docker'ın yeni nesil build motorudur ve şu avantajları sağlar:

- **Paralel Build**: Bağımsız build adımları paralel olarak çalışır
- **Gelişmiş Cache**: Daha akıllı ve verimli cache mekanizması
- **Build Secrets**: Güvenli secret yönetimi
- **SSH Forwarding**: Build sırasında SSH key kullanımı
- **Multi-platform Builds**: Farklı platformlar için build desteği
- **Daha Hızlı Builds**: Genel olarak %2-10x daha hızlı build süreleri

## Kurulum

### Otomatik Kurulum

En kolay yöntem, hazır kurulum scriptini kullanmaktır:

```bash
# Kurulum scriptini çalıştır
make setup-buildkit

# veya doğrudan:
bash scripts/setup-buildkit.sh
```

Bu script şunları yapar:
- Docker'ın BuildKit desteğini kontrol eder
- BuildKit'i etkinleştirir
- Environment değişkenlerini yapılandırır
- Shell profilinize gerekli ayarları ekler
- Kurulumu test eder

### Manuel Kurulum

#### 1. Docker Versiyonunu Kontrol Edin

BuildKit, Docker 18.09 ve üzeri versiyonlarda mevcuttur:

```bash
docker version
```

#### 2. BuildKit'i Etkinleştirin

**Geçici olarak (mevcut terminal için):**

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

**Kalıcı olarak (tüm terminaller için):**

Shell profilinize ekleyin (`~/.bashrc`, `~/.zshrc`, veya `~/.profile`):

```bash
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
source ~/.bashrc
```

#### 3. Docker Config Dosyasını Güncelleyin

`~/.docker/config.json` dosyasını oluşturun veya güncelleyin:

```json
{
  "experimental": "enabled",
  "features": {
    "buildkit": true
  }
}
```

## Yapılandırma

### Environment Değişkenleri

Projeye özel environment değişkenlerini yüklemek için:

```bash
source .buildkit.env
```

Bu dosya şu değişkenleri ayarlar:

- `DOCKER_BUILDKIT=1`: BuildKit'i etkinleştirir
- `COMPOSE_DOCKER_CLI_BUILD=1`: Docker Compose için BuildKit'i etkinleştirir
- `BUILDKIT_PROGRESS=auto`: Build progress çıktı formatı
- `BUILDKIT_COLORS=1`: Renkli çıktı
- `BUILDKIT_INLINE_CACHE=1`: Inline cache desteği

### BuildKit Yapılandırma Dosyası

`.buildkitconfig.toml` dosyası BuildKit'in davranışını kontrol eder:

```toml
[worker.oci]
  enabled = true
  max-parallelism = 4

[cache]
  inline = true
  max-size = 10240  # MB

[gc]
  enabled = true
  policy = [
    {keepDuration = "168h", keepBytes = "10GB"}
  ]
```

### .dockerignore Dosyaları

Her servis için optimize edilmiş `.dockerignore` dosyaları oluşturulmuştur:

- `backend/.dockerignore`: Python backend için
- `frontend/.dockerignore`: Node.js frontend için

Bu dosyalar build context'ini minimize ederek build hızını artırır.

## Kullanım

### Temel Build Komutları

**Docker Build:**

```bash
# BuildKit ile build
docker build -t myimage:latest .

# Progress çıktısı ile
docker build --progress=plain -t myimage:latest .
```

**Docker Compose:**

```bash
# BuildKit ile compose build
docker-compose build

# Optimized build
make build-optimized
```

### BuildKit Özellikleri

#### 1. Cache Mount

Dockerfile'da cache mount kullanımı:

```dockerfile
# syntax=docker/dockerfile:1.4

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

#### 2. Build Secrets

Güvenli secret kullanımı:

```bash
# Secret dosyası ile build
docker build --secret id=mysecret,src=./secret.txt .
```

Dockerfile'da:

```dockerfile
RUN --mount=type=secret,id=mysecret \
    cat /run/secrets/mysecret
```

#### 3. SSH Forwarding

SSH key kullanımı:

```bash
docker build --ssh default .
```

Dockerfile'da:

```dockerfile
RUN --mount=type=ssh \
    git clone git@github.com:user/repo.git
```

#### 4. Multi-stage Build

Optimize edilmiş multi-stage build:

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim as base
# Base dependencies

FROM base as dependencies
# Install dependencies with cache

FROM base as production
# Copy only what's needed
```

### Cache Stratejileri

#### Local Cache

```bash
# Local cache kullanımı
docker build --cache-from myimage:latest .
```

#### Registry Cache

```bash
# Registry'den cache çek
docker build \
  --cache-from registry.example.com/myimage:cache \
  --cache-to type=registry,ref=registry.example.com/myimage:cache,mode=max \
  .
```

#### Inline Cache

```bash
# Inline cache ile build
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t myimage:latest \
  .
```

## Optimizasyon İpuçları

### 1. Layer Sıralaması

En az değişen layer'ları en üste koyun:

```dockerfile
# ✅ İyi
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# ❌ Kötü
FROM python:3.10-slim
COPY . .
RUN pip install -r requirements.txt
```

### 2. Cache Mount Kullanımı

Package manager cache'lerini mount edin:

```dockerfile
# Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Node.js
RUN --mount=type=cache,target=/root/.npm \
    npm ci
```

### 3. .dockerignore Optimizasyonu

Gereksiz dosyaları build context'inden çıkarın:

```
# Büyük dosyalar
*.log
*.tmp
node_modules/
__pycache__/

# Geliştirme dosyaları
.git/
.vscode/
*.md
```

### 4. Multi-stage Build

Development ve production için ayrı stage'ler:

```dockerfile
FROM base as development
# Development dependencies

FROM base as production
# Only production dependencies
```

### 5. Paralel Build

Bağımsız adımları paralel çalıştırın:

```dockerfile
# BuildKit otomatik olarak paralel çalıştırır
RUN task1 &
RUN task2 &
RUN task3
```

## Sorun Giderme

### BuildKit Çalışmıyor

**Kontrol:**

```bash
# BuildKit versiyonunu kontrol et
docker buildx version

# Environment değişkenlerini kontrol et
echo $DOCKER_BUILDKIT
echo $COMPOSE_DOCKER_CLI_BUILD
```

**Çözüm:**

```bash
# Environment değişkenlerini ayarla
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Docker'ı yeniden başlat
sudo systemctl restart docker  # Linux
# veya Docker Desktop'ı yeniden başlat (Mac/Windows)
```

### Cache Çalışmıyor

**Kontrol:**

```bash
# Cache kullanımını görmek için
docker build --progress=plain .
```

**Çözüm:**

```bash
# Cache'i temizle
docker builder prune

# Belirli cache'i temizle
docker builder prune --filter type=exec.cachemount
```

### Build Yavaş

**Kontrol:**

```bash
# Build süresini ölç
time docker build .
```

**Çözüm:**

1. `.dockerignore` dosyasını optimize edin
2. Layer sıralamasını kontrol edin
3. Cache mount kullanın
4. Multi-stage build kullanın
5. Parallelism ayarlarını artırın

### "syntax=docker/dockerfile:1.4" Hatası

**Çözüm:**

```bash
# Docker versiyonunu güncelleyin
docker version

# BuildKit'in etkin olduğundan emin olun
export DOCKER_BUILDKIT=1
```

## Performans Metrikleri

BuildKit ile beklenen performans iyileştirmeleri:

| Senaryo | Geleneksel Build | BuildKit | İyileştirme |
|---------|------------------|----------|-------------|
| Cold Build | 10 dakika | 8 dakika | %20 |
| Warm Build (cache) | 5 dakika | 1 dakika | %80 |
| Incremental Build | 2 dakika | 20 saniye | %83 |
| Hot Reload | 30 saniye | 2 saniye | %93 |

## Ek Kaynaklar

- [Docker BuildKit Dokümantasyonu](https://docs.docker.com/build/buildkit/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [BuildKit GitHub Repository](https://github.com/moby/buildkit)

## Destek

Sorun yaşıyorsanız:

1. Bu dokümandaki sorun giderme bölümünü kontrol edin
2. `make buildkit-verify` komutunu çalıştırın
3. Docker loglarını kontrol edin: `docker logs`
4. Proje issue tracker'ında sorun açın
