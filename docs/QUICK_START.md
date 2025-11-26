# Hızlı Başlangıç Kılavuzu

Bu kılavuz, projeyi ilk kez kuran geliştiriciler için adım adım talimatlar içerir.

## Gereksinimler

Başlamadan önce sisteminizde aşağıdakilerin kurulu olduğundan emin olun:

- **Docker**: 20.10 veya üzeri (BuildKit desteği için)
- **Docker Compose**: 2.0 veya üzeri
- **Git**: Kod deposunu klonlamak için
- **En az 8GB RAM**: Tüm servisleri çalıştırmak için
- **En az 20GB disk alanı**: Image'lar ve volume'lar için

## Kurulum Adımları

### 1. Depoyu Klonlayın

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. BuildKit'i Etkinleştirin

BuildKit, Docker'ın gelişmiş build motoru olup cache optimizasyonu ve paralel build özellikleri sağlar.

```bash
# Bash/Zsh için
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Bu ayarları kalıcı yapmak için ~/.bashrc veya ~/.zshrc dosyanıza ekleyin
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
echo 'export COMPOSE_DOCKER_CLI_BUILD=1' >> ~/.bashrc
source ~/.bashrc
```

### 3. Environment Değişkenlerini Yapılandırın (Opsiyonel)

Proje varsayılan development değerleriyle çalışır. Özel yapılandırma için:

```bash
# Backend için
cp backend/.env.example backend/.env

# Frontend için
cp frontend/.env.example frontend/.env

# Gerekirse değerleri düzenleyin
nano backend/.env
```

**Not**: Environment dosyaları sağlanmazsa sistem otomatik olarak varsayılan değerleri kullanır.

### 4. Servisleri Başlatın

İlk kez çalıştırıldığında tüm image'lar build edilecek ve bağımlılıklar indirilecektir. Bu işlem 5-10 dakika sürebilir.

```bash
docker-compose up
```

Arka planda çalıştırmak için:

```bash
docker-compose up -d
```

### 5. Kurulumu Doğrulayın

Tüm servisler başladıktan sonra health check'leri kontrol edin:

```bash
# Tüm servislerin durumunu görüntüle
docker-compose ps

# Health check'leri kontrol et
docker-compose ps | grep "healthy"
```

Beklenen çıktı:
```
backend     Up (healthy)
frontend    Up (healthy)
postgres    Up (healthy)
redis       Up (healthy)
```

### 6. Uygulamaya Erişin

Servisler başarıyla başladıktan sonra:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Dokümantasyonu**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Yaygın Komutlar

### Servisleri Yönetme

```bash
# Tüm servisleri başlat
docker-compose up

# Tüm servisleri durdur
docker-compose down

# Servisleri durdur ve volume'ları temizle
docker-compose down -v

# Belirli bir servisi yeniden başlat
docker-compose restart backend

# Logları görüntüle
docker-compose logs -f

# Belirli bir servisin loglarını görüntüle
docker-compose logs -f backend
```

### Build İşlemleri

```bash
# Tüm servisleri yeniden build et
docker-compose build

# Cache kullanmadan build et
docker-compose build --no-cache

# Belirli bir servisi build et
docker-compose build backend

# Build et ve başlat
docker-compose up --build
```

### Container'lara Erişim

```bash
# Backend container'ına shell ile gir
docker-compose exec backend bash

# Frontend container'ına shell ile gir
docker-compose exec frontend sh

# PostgreSQL'e bağlan
docker-compose exec postgres psql -U postgres

# Redis CLI'ye bağlan
docker-compose exec redis redis-cli
```

## İlk Kurulum Süreleri

Beklenen süreler (internet hızına bağlı):

- **İlk build (cold start)**: 5-10 dakika
- **Sonraki build'ler (warm start)**: 1-2 dakika
- **Servis başlatma**: 30-60 saniye
- **Health check'lerin geçmesi**: 10-20 saniye

## Sorun Giderme

### BuildKit Etkin Değil

**Hata**: `DEPRECATED: The legacy builder is deprecated`

**Çözüm**:
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Port Zaten Kullanımda

**Hata**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Çözüm**:
```bash
# Portu kullanan işlemi bul
lsof -i :8000

# İşlemi sonlandır veya docker-compose.yml'de portu değiştir
```

### Yetersiz Disk Alanı

**Hata**: `no space left on device`

**Çözüm**:
```bash
# Kullanılmayan image'ları temizle
docker image prune -a

# Kullanılmayan volume'ları temizle
docker volume prune

# Tüm kullanılmayan kaynakları temizle
docker system prune -a --volumes
```

### Health Check Başarısız

**Hata**: Servis "unhealthy" durumunda

**Çözüm**:
```bash
# Servis loglarını kontrol et
docker-compose logs backend

# Container'ı yeniden başlat
docker-compose restart backend

# Gerekirse yeniden build et
docker-compose up --build backend
```

### Bağımlılık Hataları

**Hata**: `ModuleNotFoundError` veya `Cannot find module`

**Çözüm**:
```bash
# Backend için
docker-compose build --no-cache backend

# Frontend için
docker-compose build --no-cache frontend
```

## Sonraki Adımlar

Kurulum tamamlandıktan sonra:

1. [Geliştirme İş Akışı](./DEVELOPMENT_WORKFLOW.md) dokümanını okuyun
2. Kod değişikliği yapın ve hot reload'u test edin
3. API dokümantasyonunu inceleyin: http://localhost:8000/docs
4. Test suite'i çalıştırın (detaylar için geliştirme dokümanına bakın)

## Yardım ve Destek

Sorun yaşıyorsanız:

1. [Sorun Giderme Kılavuzu](./TROUBLESHOOTING.md)'na bakın
2. Proje issue tracker'ında arama yapın
3. Yeni bir issue açın ve aşağıdaki bilgileri ekleyin:
   - Docker versiyonu: `docker --version`
   - Docker Compose versiyonu: `docker-compose --version`
   - İşletim sistemi
   - Hata mesajları ve loglar

## Ek Kaynaklar

- [Geliştirme İş Akışı](./DEVELOPMENT_WORKFLOW.md)
- [Production Deployment](./PRODUCTION_DEPLOYMENT.md)
- [Sorun Giderme](./TROUBLESHOOTING.md)
- [API Dokümantasyonu](./API_DOCUMENTATION.md)
