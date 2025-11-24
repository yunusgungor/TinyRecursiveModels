# CI/CD Pipeline Kurulum Rehberi

Bu doküman, Trendyol Gift Recommendation Web uygulaması için GitHub Actions tabanlı CI/CD pipeline'ının kurulum ve yapılandırma adımlarını içerir.

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Gereksinimler](#gereksinimler)
3. [GitHub Secrets Yapılandırması](#github-secrets-yapılandırması)
4. [Workflow'lar](#workflowlar)
5. [Branch Stratejisi](#branch-stratejisi)
6. [Deployment Süreci](#deployment-süreci)
7. [Monitoring ve Alerting](#monitoring-ve-alerting)
8. [Troubleshooting](#troubleshooting)

## Genel Bakış

CI/CD pipeline'ımız aşağıdaki aşamalardan oluşur:

```
Code Push → Linting → Testing → Build → Deploy → Monitor
```

### Pipeline Özellikleri

- ✅ Otomatik linting (Python, TypeScript)
- ✅ Kapsamlı test suite (unit, property, integration)
- ✅ Güvenlik taramaları (dependencies, secrets, SAST, containers)
- ✅ Performans testleri (load testing, Lighthouse)
- ✅ Otomatik Docker image build ve publish
- ✅ Staging ve production deployment
- ✅ Smoke testler
- ✅ Slack bildirimleri
- ✅ Otomatik release notes

## Gereksinimler

### GitHub Repository Ayarları

1. **Branch Protection Rules** (Settings → Branches)
   - `main` branch için:
     - Require pull request reviews (1 approval)
     - Require status checks to pass
     - Require branches to be up to date
     - Include administrators
   - `develop` branch için:
     - Require status checks to pass

2. **Environments** (Settings → Environments)
   - `staging` environment oluşturun
   - `production` environment oluşturun
     - Protection rules ekleyin (opsiyonel: manual approval)

3. **Actions Permissions** (Settings → Actions → General)
   - Allow all actions and reusable workflows
   - Workflow permissions: Read and write permissions

### External Services

1. **Container Registry**: GitHub Container Registry (ghcr.io)
2. **Kubernetes Cluster**: Staging ve production için
3. **Slack**: Bildirimler için webhook
4. **Codecov**: Test coverage raporları için (opsiyonel)
5. **SonarCloud**: Code quality analizi için (opsiyonel)
6. **Snyk**: Dependency scanning için (opsiyonel)

## GitHub Secrets Yapılandırması

### Repository Secrets

Settings → Secrets and variables → Actions → New repository secret

#### Zorunlu Secrets

```bash
# Kubernetes Configuration
KUBECONFIG_STAGING=<base64 encoded kubeconfig for staging>
KUBECONFIG_PRODUCTION=<base64 encoded kubeconfig for production>

# Slack Notifications
SLACK_WEBHOOK=<slack webhook URL>
```

#### Opsiyonel Secrets

```bash
# Code Coverage
CODECOV_TOKEN=<codecov token>

# Code Quality
SONAR_TOKEN=<sonarcloud token>

# Security Scanning
SNYK_TOKEN=<snyk API token>
```

### Environment Secrets

Her environment için ayrı secrets tanımlayabilirsiniz:

**Staging Environment:**
```bash
DATABASE_URL=<staging database URL>
REDIS_URL=<staging redis URL>
TRENDYOL_API_KEY=<staging API key>
```

**Production Environment:**
```bash
DATABASE_URL=<production database URL>
REDIS_URL=<production redis URL>
TRENDYOL_API_KEY=<production API key>
```

### Kubeconfig Oluşturma

```bash
# Staging kubeconfig
kubectl config view --minify --flatten --context=staging-context | base64 -w 0

# Production kubeconfig
kubectl config view --minify --flatten --context=production-context | base64 -w 0
```

## Workflow'lar

### 1. CI/CD Pipeline (`ci.yml`)

**Tetikleyiciler:**
- Push to `main` veya `develop`
- Pull request to `main` veya `develop`
- Manuel tetikleme

**Aşamalar:**
1. Linting (backend + frontend)
2. Testing (unit + property + integration)
3. Build (Docker images)
4. Deploy (staging/production)

**Kullanım:**
```bash
# Otomatik: Branch'e push yapın
git push origin develop

# Manuel: GitHub Actions UI'dan
# Actions → CI/CD Pipeline → Run workflow
```

### 2. Security Scan (`security-scan.yml`)

**Tetikleyiciler:**
- Push to `main` veya `develop`
- Pull request
- Günlük schedule (02:00 UTC)

**Taramalar:**
- Dependency vulnerabilities
- Secret leaks
- SAST (Static Application Security Testing)
- Container image vulnerabilities

### 3. Performance Testing (`performance-test.yml`)

**Tetikleyiciler:**
- Push to `main` veya `develop`
- Pull request
- Manuel tetikleme

**Testler:**
- Load testing (Locust)
- Frontend performance (Lighthouse)

### 4. Code Quality (`code-quality.yml`)

**Tetikleyiciler:**
- Pull request

**Kontroller:**
- SonarCloud analysis
- Automated code review
- Complexity check
- Dependency review
- PR size check
- Commit message check

### 5. Release Management (`release.yml`)

**Tetikleyiciler:**
- Tag push (`v*.*.*`)
- Manuel tetikleme

**İşlemler:**
- Otomatik changelog oluşturma
- GitHub release oluşturma
- Slack bildirimi

### 6. Docker Publishing (`docker-publish.yml`)

**Tetikleyiciler:**
- Release oluşturulduğunda
- Manuel tetikleme

**İşlemler:**
- Docker image build ve publish
- SBOM (Software Bill of Materials) oluşturma

## Branch Stratejisi

### Git Flow

```
main (production)
  ↑
  merge
  ↑
develop (staging)
  ↑
  merge
  ↑
feature/* (development)
```

### Branch'ler

1. **main**: Production branch
   - Her commit otomatik olarak production'a deploy edilir
   - Protected branch
   - Sadece `develop`'dan merge alır

2. **develop**: Staging branch
   - Her commit otomatik olarak staging'e deploy edilir
   - Protected branch
   - Feature branch'lerden merge alır

3. **feature/***: Feature development
   - Yeni özellikler için
   - `develop`'dan branch alınır
   - PR ile `develop`'a merge edilir

4. **hotfix/***: Acil düzeltmeler
   - `main`'den branch alınır
   - Hem `main` hem `develop`'a merge edilir

### Workflow Örneği

```bash
# Yeni feature başlat
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# Değişiklikleri commit et
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# PR oluştur (GitHub UI'dan)
# develop ← feature/new-feature

# PR merge edildikten sonra
# Otomatik olarak staging'e deploy edilir

# Staging'de test et
# Hazır olduğunda develop'ı main'e merge et
git checkout main
git pull origin main
git merge develop
git push origin main

# Otomatik olarak production'a deploy edilir
```

## Deployment Süreci

### Staging Deployment

1. `develop` branch'e push yapın
2. CI/CD pipeline otomatik başlar
3. Testler geçerse build job çalışır
4. Docker image'lar build edilir
5. Kubernetes deployment güncellenir
6. Smoke testler çalıştırılır
7. Slack bildirimi gönderilir

**Deployment URL**: https://staging.trendyol-gift.example.com

### Production Deployment

1. `main` branch'e push yapın (veya `develop`'ı merge edin)
2. CI/CD pipeline otomatik başlar
3. Testler geçerse build job çalışır
4. Docker image'lar build edilir
5. Kubernetes deployment güncellenir (rolling update)
6. Smoke testler çalıştırılır
7. GitHub release oluşturulur
8. Slack bildirimi gönderilir

**Deployment URL**: https://trendyol-gift.example.com

### Manuel Deployment

```bash
# GitHub Actions UI'dan
Actions → CI/CD Pipeline → Run workflow
# Environment seçin: staging veya production
```

### Rollback

```bash
# Kubernetes rollback
kubectl rollout undo deployment/backend-deployment -n production
kubectl rollout undo deployment/frontend-deployment -n production

# Veya önceki revision'a
kubectl rollout undo deployment/backend-deployment -n production --to-revision=2
```

## Monitoring ve Alerting

### Metrics

- **Test Coverage**: Codecov dashboard
- **Code Quality**: SonarCloud dashboard
- **Security**: GitHub Security tab
- **Performance**: Lighthouse reports (artifacts)
- **Deployment**: GitHub Actions logs

### Alerts

**Slack Notifications:**
- Deployment başarılı/başarısız
- Security scan sonuçları
- Performance regression

**GitHub Notifications:**
- PR status checks
- Deployment status
- Security alerts

### Health Checks

```bash
# Staging
curl https://staging.trendyol-gift.example.com/api/health

# Production
curl https://trendyol-gift.example.com/api/health
```

## Troubleshooting

### Test Failures

**Problem**: Testler başarısız oluyor

**Çözüm**:
1. GitHub Actions logs'ları kontrol edin
2. Local'de testleri çalıştırın:
   ```bash
   # Backend
   cd backend
   pytest -v
   
   # Frontend
   cd frontend
   npm test
   ```
3. Coverage raporlarını inceleyin
4. Failing test'i düzeltin ve yeniden push edin

### Build Failures

**Problem**: Docker build başarısız oluyor

**Çözüm**:
1. Build logs'ları kontrol edin
2. Local'de build deneyin:
   ```bash
   docker build -t test ./backend
   docker build -t test ./frontend
   ```
3. Dependency versiyonlarını kontrol edin
4. Dockerfile'ı düzeltin ve yeniden push edin

### Deployment Failures

**Problem**: Deployment başarısız oluyor

**Çözüm**:
1. Kubernetes logs'ları kontrol edin:
   ```bash
   kubectl logs -f deployment/backend-deployment -n staging
   kubectl describe pod <pod-name> -n staging
   ```
2. Rollout status'u kontrol edin:
   ```bash
   kubectl rollout status deployment/backend-deployment -n staging
   ```
3. ConfigMap ve Secret'ları kontrol edin
4. Gerekirse rollback yapın

### Secret Issues

**Problem**: Secrets bulunamıyor veya geçersiz

**Çözüm**:
1. GitHub Settings → Secrets'ı kontrol edin
2. Secret'ların doğru environment'a eklendiğinden emin olun
3. Base64 encoding'i kontrol edin:
   ```bash
   echo "secret-value" | base64
   ```
4. Secret'ı güncelleyin ve workflow'u yeniden çalıştırın

### Performance Issues

**Problem**: Pipeline çok yavaş çalışıyor

**Çözüm**:
1. Cache kullanımını kontrol edin
2. Paralel job'ları optimize edin
3. Test suite'i optimize edin
4. Docker layer caching'i kontrol edin

## Best Practices

### Commit Messages

Conventional Commits formatını kullanın:

```
feat: add new feature
fix: fix bug in component
docs: update documentation
style: format code
refactor: refactor service
perf: improve performance
test: add tests
chore: update dependencies
ci: update workflow
```

### Pull Requests

- Küçük, focused PR'lar oluşturun
- PR template'i doldurun
- Self-review yapın
- Test coverage'ı kontrol edin
- CI checks'lerin geçmesini bekleyin

### Testing

- Her yeni feature için test yazın
- Property-based testler ekleyin
- Integration testler yazın
- Test coverage'ı %80'in üzerinde tutun

### Security

- Secrets'ı asla commit etmeyin
- Dependency'leri güncel tutun
- Security scan'leri düzenli çalıştırın
- Vulnerability'leri hemen düzeltin

### Documentation

- README'yi güncel tutun
- API dokümantasyonunu güncelleyin
- Deployment notlarını ekleyin
- Changelog'u güncelleyin

## Ek Kaynaklar

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## Destek

Sorularınız için:
- GitHub Issues oluşturun
- Slack #devops kanalına yazın
- DevOps team'e ulaşın
