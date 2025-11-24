# Monitoring ve Logging Altyapısı

Bu dizin, Trendyol Gift Recommendation uygulaması için monitoring ve logging altyapısını içerir.

## Bileşenler

### 1. Prometheus (Metrics Collection)
- **Port**: 9090
- **Amaç**: Uygulama metriklerini toplar ve saklar
- **Erişim**: http://localhost:9090

**Toplanan Metrikler**:
- API response time
- Model inference time
- Tool execution times
- Tool success rates
- CPU, Memory, GPU kullanımı
- Request counts

### 2. Grafana (Metrics Visualization)
- **Port**: 3001
- **Amaç**: Metrikleri görselleştirir ve dashboard'lar sunar
- **Erişim**: http://localhost:3001
- **Varsayılan Kullanıcı**: admin / admin

**Hazır Dashboard'lar**:
- API Performance Dashboard
- System Resources Dashboard
- Tool Analytics Dashboard

### 3. Elasticsearch (Log Storage)
- **Port**: 9200, 9300
- **Amaç**: Log verilerini saklar ve indeksler
- **Erişim**: http://localhost:9200

### 4. Logstash (Log Processing)
- **Port**: 5044, 9600
- **Amaç**: Log verilerini işler ve Elasticsearch'e gönderir

### 5. Kibana (Log Visualization)
- **Port**: 5601
- **Amaç**: Log verilerini görselleştirir ve analiz eder
- **Erişim**: http://localhost:5601

### 6. Jaeger (Distributed Tracing)
- **Port**: 16686 (UI), 6831 (Agent)
- **Amaç**: Distributed tracing ve request flow analizi
- **Erişim**: http://localhost:16686

### 7. Alertmanager (Alert Management)
- **Port**: 9093
- **Amaç**: Alert'leri yönetir ve bildirim gönderir
- **Erişim**: http://localhost:9093

## Kurulum

### 1. Environment Variables Ayarlama

`.env` dosyasına aşağıdaki değişkenleri ekleyin:

```bash
# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=your-secure-password

# Email Alerts
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_DEFAULT=alerts@example.com
ALERT_EMAIL_CRITICAL=critical@example.com
ALERT_EMAIL_WARNING=warning@example.com
ALERT_EMAIL_INFO=info@example.com
```

### 2. Monitoring Stack'i Başlatma

```bash
# Ana uygulamayı başlat
docker-compose up -d

# Monitoring stack'i başlat
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Servislerin Durumunu Kontrol Etme

```bash
# Tüm servisleri kontrol et
docker-compose -f docker-compose.monitoring.yml ps

# Logları görüntüle
docker-compose -f docker-compose.monitoring.yml logs -f
```

## Kullanım

### Prometheus

1. http://localhost:9090 adresine gidin
2. Query alanına metrik adı yazın (örn: `api_response_time_seconds`)
3. "Execute" butonuna tıklayın
4. Graph veya Console sekmesinden sonuçları görüntüleyin

**Örnek Queries**:
```promql
# Ortalama API response time (son 5 dakika)
rate(api_response_time_seconds[5m])

# Tool success rate
tool_success_rate{tool="price_comparison"}

# CPU kullanımı
cpu_usage_percent

# Memory kullanımı
memory_usage_percent
```

### Grafana

1. http://localhost:3001 adresine gidin
2. admin/admin ile giriş yapın (ilk girişte şifre değiştirin)
3. Sol menüden "Dashboards" seçin
4. Hazır dashboard'lardan birini seçin

**Dashboard'ları Özelleştirme**:
- Dashboard üzerinde "Edit" butonuna tıklayın
- Panel'leri düzenleyin veya yeni panel ekleyin
- "Save" ile kaydedin

### Kibana

1. http://localhost:5601 adresine gidin
2. İlk açılışta index pattern oluşturun: `logstash-*`
3. Time field olarak `@timestamp` seçin
4. "Discover" sekmesinden logları görüntüleyin

**Log Filtreleme**:
```
# Error logları
level: "ERROR"

# Belirli bir request ID
request_id: "abc-123"

# Tool execution logları
message: "tool execution"
```

### Jaeger

1. http://localhost:16686 adresine gidin
2. Service dropdown'dan "Trendyol Gift Recommendation API" seçin
3. "Find Traces" butonuna tıklayın
4. Trace'leri görüntüleyin ve analiz edin

## Alert Konfigürasyonu

### Email Alert'leri Aktifleştirme

1. Gmail için App Password oluşturun:
   - Google Account Settings > Security > 2-Step Verification
   - App Passwords > Select app: Mail > Generate

2. `.env` dosyasını güncelleyin:
```bash
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=generated-app-password
ALERT_EMAIL_CRITICAL=your-email@gmail.com
```

3. Alertmanager'ı yeniden başlatın:
```bash
docker-compose -f docker-compose.monitoring.yml restart alertmanager
```

### Slack Alert'leri Ekleme

1. Slack Webhook URL oluşturun
2. `monitoring/alertmanager/alertmanager.yml` dosyasını düzenleyin
3. Slack webhook konfigürasyonunu uncomment edin
4. Alertmanager'ı yeniden başlatın

## Troubleshooting

### Prometheus Metrics Görünmüyor

```bash
# Backend'in metrics endpoint'ini kontrol et
curl http://localhost:8000/api/v1/metrics

# Prometheus target'larını kontrol et
# http://localhost:9090/targets
```

### Elasticsearch Başlamıyor

```bash
# Elasticsearch loglarını kontrol et
docker-compose -f docker-compose.monitoring.yml logs elasticsearch

# Disk alanını kontrol et
df -h

# Memory limitini artır (docker-compose.monitoring.yml)
ES_JAVA_OPTS: "-Xms1g -Xmx1g"
```

### Jaeger Trace'leri Görünmüyor

```bash
# Backend'de tracing'in aktif olduğunu kontrol et
# .env dosyasında:
ENABLE_TRACING=true

# Jaeger agent'ın çalıştığını kontrol et
docker-compose -f docker-compose.monitoring.yml ps jaeger
```

## Performans İpuçları

### Elasticsearch

- Index lifecycle management kullanın
- Eski index'leri silin veya arşivleyin
- Shard sayısını optimize edin

### Prometheus

- Retention period'u ayarlayın (varsayılan: 15 gün)
- Gereksiz metrikleri scrape etmeyin
- Recording rules kullanarak ağır query'leri önceden hesaplayın

### Grafana

- Dashboard'larda time range'i sınırlayın
- Çok fazla panel kullanmayın
- Query'leri optimize edin

## Güvenlik

### Production Ortamı İçin

1. **Şifreleri değiştirin**:
   - Grafana admin şifresi
   - Elasticsearch şifresi (eğer aktifse)

2. **Network izolasyonu**:
   - Monitoring servislerini internal network'e alın
   - Sadece gerekli port'ları expose edin

3. **TLS/SSL aktifleştirin**:
   - Nginx reverse proxy kullanın
   - Let's Encrypt sertifikası ekleyin

4. **Authentication ekleyin**:
   - Prometheus için basic auth
   - Kibana için authentication

## Bakım

### Log Rotation

Logstash otomatik olarak günlük index'ler oluşturur. Eski index'leri silmek için:

```bash
# 30 günden eski index'leri sil
curl -X DELETE "localhost:9200/logstash-$(date -d '30 days ago' +%Y.%m.%d)"
```

### Backup

```bash
# Prometheus data backup
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data

# Grafana dashboards backup
docker run --rm -v grafana_data:/data -v $(pwd):/backup alpine tar czf /backup/grafana-backup.tar.gz /data
```

## Kaynaklar

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
