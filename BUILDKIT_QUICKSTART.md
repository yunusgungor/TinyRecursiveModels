# BuildKit Hızlı Başlangıç

Bu proje Docker BuildKit kullanarak optimize edilmiş build süreçleri sağlar.

## 🚀 Hızlı Kurulum

### 1. BuildKit'i Etkinleştir

```bash
# Otomatik kurulum (önerilen)
make setup-buildkit

# Manuel kurulum
source .buildkit.env
```

### 2. Doğrula

```bash
# BuildKit'in çalıştığını doğrula
make buildkit-verify
```

### 3. Build Et

```bash
# Tüm servisleri build et
make build-optimized

# Sadece backend
make build-backend

# Sadece frontend
make build-frontend
```

## 📋 Gereksinimler

- Docker 18.09 veya üzeri
- Docker Compose 1.25 veya üzeri (veya Docker Compose V2)

## ✨ Özellikler

- ⚡ **Hızlı Builds**: Cache optimizasyonu ile %80'e varan hız artışı
- 🔄 **Hot Reload**: Kod değişikliklerinde 2 saniye içinde yansıma
- 📦 **Küçük Image'lar**: Multi-stage build ile optimize edilmiş boyutlar
- 🔒 **Güvenli**: Non-root user ve secret management
- 🎯 **Paralel Build**: Bağımsız adımlar paralel çalışır

## 🛠️ Kullanım

### Development Ortamı

```bash
# Servisleri başlat
make dev

# veya
docker-compose up -d
```

### Production Build

```bash
# Production image'ları build et
docker-compose -f docker-compose.prod.yml build
```

### Cache Yönetimi

```bash
# Cache'i temizle
docker builder prune

# Tüm cache'i temizle
docker builder prune -a
```

## 📚 Detaylı Dokümantasyon

Daha fazla bilgi için:
- [BuildKit Kurulum Rehberi](docs/BUILDKIT_SETUP.md)
- [Docker Compose Yapılandırması](docker-compose.yml)
- [Dockerfile Optimizasyonları](backend/Dockerfile)

## 🐛 Sorun Giderme

### BuildKit çalışmıyor?

```bash
# Environment değişkenlerini kontrol et
echo $DOCKER_BUILDKIT
echo $COMPOSE_DOCKER_CLI_BUILD

# Tekrar ayarla
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Build yavaş?

1. `.dockerignore` dosyalarını kontrol edin
2. Cache'i temizleyin: `docker builder prune`
3. Docker'ı yeniden başlatın

## 💡 İpuçları

- Her build öncesi `source .buildkit.env` çalıştırın
- `.dockerignore` dosyalarını güncel tutun
- Multi-stage build kullanın
- Cache mount'ları kullanın

## 🔗 Faydalı Komutlar

```bash
# BuildKit versiyonu
docker buildx version

# Build history
docker buildx du

# Cache kullanımı
docker buildx du --verbose

# BuildKit logları
docker buildx inspect --bootstrap
```

## ⚙️ Yapılandırma

BuildKit yapılandırması için:
- `.buildkitconfig.toml`: BuildKit ayarları
- `.buildkit.env`: Environment değişkenleri
- `backend/.dockerignore`: Backend build context
- `frontend/.dockerignore`: Frontend build context

## 📞 Destek

Sorun yaşıyorsanız:
1. `make buildkit-verify` çalıştırın
2. [Sorun Giderme](docs/BUILDKIT_SETUP.md#sorun-giderme) bölümüne bakın
3. Issue açın
