# CI/CD Pipeline Documentation

Bu doküman, Trendyol Gift Recommendation Web uygulaması için GitHub Actions tabanlı CI/CD pipeline'ını açıklar.

## Workflow'lar

### 1. CI/CD Pipeline (`ci.yml`)

Ana CI/CD pipeline'ı. Her push ve pull request'te otomatik olarak çalışır.

**Tetikleyiciler:**
- Push to `main` veya `develop` branch'leri
- Pull request to `main` veya `develop` branch'leri
- Manuel tetikleme (workflow_dispatch)

**Job'lar:**

#### Linting Jobs
- **lint-backend**: Python kodu için Black, Ruff ve mypy kontrolü
- **lint-frontend**: TypeScript/React kodu için ESLint, Prettier ve tsc kontrolü

#### Test Jobs
- **test-backend-unit**: Backend unit testleri (pytest)
- **test-backend-property**: Property-based testler (Hypothesis)
- **test-backend-integration**: Backend integration testleri
- **test-frontend**: Frontend testleri (Vitest)

#### Build Job
- **build**: Docker image'larını build eder ve GitHub Container Registry'ye push eder
- Backend ve frontend için ayrı image'lar oluşturur
- Image metadata ve tag'leri otomatik oluşturur

#### Deployment Jobs
- **deploy-staging**: Staging ortamına deployment (develop branch)
- **deploy-production**: Production ortamına deployment (main branch)
- Kubernetes kullanarak deployment yapar
- Smoke testler çalıştırır
- Slack bildirimleri gönderir

**Gerekli Secrets:**
- `KUBECONFIG_STAGING`: Staging Kubernetes config (base64 encoded)
- `KUBECONFIG_PRODUCTION`: Production Kubernetes config (base64 encoded)
- `SLACK_WEBHOOK`: Slack webhook URL'i

### 2. Security Scan (`security-scan.yml`)

Güvenlik taramaları için workflow.

**Tetikleyiciler:**
- Push to `main` veya `develop`
- Pull request
- Günlük schedule (02:00 UTC)

**Job'lar:**
- **dependency-scan-backend**: Python dependency güvenlik taraması (safety, pip-audit)
- **dependency-scan-frontend**: NPM dependency güvenlik taraması (npm audit, Snyk)
- **secret-scan**: Git history'de secret taraması (Gitleaks)
- **sast-scan**: Static Application Security Testing (Semgrep)
- **container-scan**: Docker image güvenlik taraması (Trivy)

**Gerekli Secrets:**
- `SNYK_TOKEN`: Snyk API token (opsiyonel)

### 3. Performance Testing (`performance-test.yml`)

Performans testleri için workflow.

**Tetikleyiciler:**
- Push to `main` veya `develop`
- Pull request
- Manuel tetikleme

**Job'lar:**
- **load-test**: Locust ile load testing
  - 10 concurrent user simülasyonu
  - 2 dakika test süresi
  - HTML rapor oluşturur
- **lighthouse**: Frontend performance testing
  - Lighthouse CI ile performance metrikleri
  - Accessibility, SEO, best practices kontrolü

### 4. Docker Publishing (`docker-publish.yml`)

Release için Docker image publishing.

**Tetikleyiciler:**
- GitHub release oluşturulduğunda
- Manuel tetikleme

**Job'lar:**
- **publish**: Docker image'larını version tag'i ile publish eder
  - SBOM (Software Bill of Materials) oluşturur
  - Latest tag'ini günceller

## Environment'lar

### Staging
- **URL**: https://staging.trendyol-gift.example.com
- **Branch**: develop
- **Auto-deploy**: Evet
- **Approval**: Gerekli değil

### Production
- **URL**: https://trendyol-gift.example.com
- **Branch**: main
- **Auto-deploy**: Evet
- **Approval**: GitHub environment protection rules ile yapılandırılabilir

## Deployment Stratejisi

### Staging Deployment
1. `develop` branch'e push yapıldığında otomatik tetiklenir
2. Tüm testler geçerse build job çalışır
3. Docker image'lar build edilir ve registry'ye push edilir
4. Kubernetes deployment güncellenir
5. Rollout status kontrol edilir
6. Smoke testler çalıştırılır
7. Slack bildirimi gönderilir

### Production Deployment
1. `main` branch'e push yapıldığında otomatik tetiklenir
2. Tüm testler geçerse build job çalışır
3. Docker image'lar build edilir ve registry'ye push edilir
4. Kubernetes deployment güncellenir (rolling update)
5. Rollout status kontrol edilir (10 dakika timeout)
6. Smoke testler çalıştırılır
7. Slack bildirimi gönderilir
8. GitHub release oluşturulur

## Manuel Deployment

Manuel deployment için:

```bash
# GitHub Actions UI'dan workflow_dispatch kullanarak
# Environment seçimi: staging veya production
```

## Monitoring ve Alerting

### Test Coverage
- Backend ve frontend için ayrı coverage raporları
- Codecov entegrasyonu
- Minimum %80 coverage hedefi

### Performance Metrics
- API response time
- Frontend load time
- Lighthouse scores
- Load test sonuçları

### Security Alerts
- Dependency vulnerabilities
- Secret leaks
- SAST findings
- Container vulnerabilities

## Troubleshooting

### Test Failures
1. GitHub Actions logs'ları kontrol edin
2. Local'de testleri çalıştırın: `pytest` veya `npm test`
3. Coverage raporlarını inceleyin

### Build Failures
1. Docker build logs'ları kontrol edin
2. Local'de build deneyin: `docker build -t test .`
3. Dependency versiyonlarını kontrol edin

### Deployment Failures
1. Kubernetes logs'ları kontrol edin: `kubectl logs`
2. Rollout status'u kontrol edin: `kubectl rollout status`
3. Pod status'u kontrol edin: `kubectl get pods`
4. Smoke test sonuçlarını inceleyin

## Best Practices

1. **Branch Protection**: `main` ve `develop` branch'leri için protection rules aktif olmalı
2. **Required Checks**: Tüm test job'ları merge için gerekli olmalı
3. **Code Review**: En az 1 approval gerekli
4. **Secrets Management**: Tüm secrets GitHub Secrets'ta saklanmalı
5. **Environment Protection**: Production için manual approval eklenebilir
6. **Rollback Strategy**: Kubernetes rollback kullanın: `kubectl rollout undo`

## Gelecek İyileştirmeler

- [ ] Blue-green deployment stratejisi
- [ ] Canary deployment
- [ ] Automated rollback on failure
- [ ] Performance regression detection
- [ ] E2E test automation (Playwright)
- [ ] Infrastructure as Code (Terraform)
- [ ] GitOps with ArgoCD
- [ ] Multi-region deployment
