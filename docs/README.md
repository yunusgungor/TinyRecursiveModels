# Trendyol Gift Recommendation - Documentation

Bu klasör, Trendyol Gift Recommendation projesinin tüm dokümantasyonunu içerir.

## 📚 Dokümantasyon İçeriği

### 1. [API Documentation](./API_DOCUMENTATION.md)
REST API endpoint'leri, request/response formatları ve kullanım örnekleri.

**İçerik:**
- Endpoint referansları
- Request/Response şemaları
- Hata kodları
- cURL, Python ve JavaScript örnekleri
- Rate limiting bilgileri

**Hedef Kitle:** Backend geliştiriciler, API tüketicileri

### 2. [Deployment Guide](./DEPLOYMENT_GUIDE.md)
Uygulamanın farklı ortamlara nasıl deploy edileceği.

**İçerik:**
- Development, Staging, Production setup
- Docker ve Kubernetes deployment
- Environment variables
- Monitoring ve logging setup
- Backup ve recovery prosedürleri
- Troubleshooting

**Hedef Kitle:** DevOps mühendisleri, sistem yöneticileri

### 3. [User Guide](./USER_GUIDE.md)
Son kullanıcılar için detaylı kullanım kılavuzu.

**İçerik:**
- Başlangıç rehberi
- Profil oluşturma
- Öneri alma ve inceleme
- Favoriler ve geçmiş
- Ayarlar
- Sık sorulan sorular

**Hedef Kitle:** Son kullanıcılar

### 4. [Developer Guide](./DEVELOPER_GUIDE.md)
Geliştiriciler için kapsamlı teknik rehber.

**İçerik:**
- Proje yapısı
- Development setup
- Kod standartları
- Testing stratejileri
- API ve Frontend development
- Model entegrasyonu
- Debugging
- Contributing guidelines

**Hedef Kitle:** Yazılım geliştiriciler, katkıda bulunanlar

### 5. [Component Documentation](./COMPONENT_DOCUMENTATION.md)
Frontend bileşenlerinin detaylı dokümantasyonu.

**İçerik:**
- Storybook kullanımı
- Bileşen referansları
- Props ve kullanım örnekleri
- Styling ve responsive design
- Testing
- Best practices

**Hedef Kitle:** Frontend geliştiriciler

### 6. [OpenAPI Specification](./OPENAPI_SPEC.yaml)
API'nin OpenAPI 3.0 formatında spesifikasyonu.

**İçerik:**
- Tüm endpoint'lerin detaylı tanımları
- Schema definitions
- Request/Response examples
- Error responses

**Kullanım:**
- Swagger UI ile görüntüleme
- API client code generation
- API testing

**Hedef Kitle:** API geliştiriciler, test mühendisleri

## 🚀 Hızlı Başlangıç

### Kullanıcılar İçin
1. [User Guide](./USER_GUIDE.md) okuyun
2. Uygulamaya erişin
3. Profil oluşturun
4. Öneriler alın

### Geliştiriciler İçin
1. [Developer Guide](./DEVELOPER_GUIDE.md) okuyun
2. Development environment'ı kurun
3. [API Documentation](./API_DOCUMENTATION.md) inceleyin
4. Kod yazmaya başlayın

### DevOps İçin
1. [Deployment Guide](./DEPLOYMENT_GUIDE.md) okuyun
2. Environment'ları hazırlayın
3. Deploy edin
4. Monitoring setup yapın

## 📖 Dokümantasyon Formatları

### Markdown (.md)
Tüm ana dokümantasyon dosyaları Markdown formatındadır ve GitHub'da doğrudan okunabilir.

### YAML (.yaml)
OpenAPI spesifikasyonu YAML formatındadır ve Swagger UI ile görüntülenebilir.

### Storybook
Frontend bileşen dokümantasyonu Storybook ile interaktif olarak görüntülenebilir:

```bash
cd frontend
npm run storybook
```

## 🔍 Dokümantasyon Arama

### GitHub'da Arama
Repository içinde arama yaparak ilgili dokümantasyonu bulabilirsiniz.

### Lokal Arama
```bash
# Tüm dokümantasyonda arama
grep -r "aranacak_kelime" docs/

# Belirli bir dosyada arama
grep "aranacak_kelime" docs/API_DOCUMENTATION.md
```

## 📝 Dokümantasyon Güncellemeleri

### Versiyon Geçmişi

**v1.0.0 (Ocak 2024)**
- İlk dokümantasyon seti oluşturuldu
- API, Deployment, User, Developer ve Component guide'ları eklendi
- OpenAPI spesifikasyonu eklendi
- Storybook konfigürasyonu eklendi

### Güncelleme Prosedürü

Dokümantasyon güncellemeleri için:

1. İlgili `.md` dosyasını düzenleyin
2. Değişiklikleri commit edin
3. Pull request oluşturun
4. Review sonrası merge edin

## 🛠️ Dokümantasyon Araçları

### Swagger UI
API dokümantasyonunu interaktif olarak görüntüleyin:

**Development:**
```
http://localhost:8000/api/v1/docs
```

**Production:**
```
https://api.example.com/api/v1/docs
```

### ReDoc
Alternatif API dokümantasyon görünümü:

**Development:**
```
http://localhost:8000/api/v1/redoc
```

### Storybook
Frontend bileşen dokümantasyonu:

**Development:**
```bash
cd frontend
npm run storybook
# http://localhost:6006
```

**Build:**
```bash
npm run build-storybook
# Output: storybook-static/
```

## 📊 Dokümantasyon Metrikleri

### Kapsam
- ✅ API Endpoints: 100%
- ✅ Frontend Components: 100%
- ✅ Deployment Procedures: 100%
- ✅ User Workflows: 100%
- ✅ Developer Setup: 100%

### Güncellik
- Son güncelleme: Ocak 2024
- Güncelleme sıklığı: Her major release
- Review sıklığı: Aylık

## 🤝 Katkıda Bulunma

Dokümantasyona katkıda bulunmak için:

1. Eksik veya hatalı bilgi bulun
2. Issue açın veya doğrudan PR gönderin
3. Değişikliklerinizi açıklayın
4. Review sürecini takip edin

### Dokümantasyon Standartları

**Markdown:**
- Başlıklar için `#` kullanın
- Kod blokları için ` ``` ` kullanın
- Linkler için `[text](url)` formatı kullanın
- Listeler için `-` veya `1.` kullanın

**Kod Örnekleri:**
- Çalışan kod örnekleri verin
- Açıklayıcı yorumlar ekleyin
- Hata durumlarını gösterin

**Dil:**
- Türkçe: Kullanıcı dokümantasyonu
- İngilizce: Teknik dokümantasyon (kod, API)
- Tutarlı terminoloji kullanın

## 📞 Destek

Dokümantasyon ile ilgili sorularınız için:

- **GitHub Issues:** Hata bildirimi ve öneriler
- **Email:** docs@example.com
- **Slack:** #documentation kanalı

## 📄 Lisans

Bu dokümantasyon MIT lisansı altında lisanslanmıştır.

## 🔗 Bağlantılar

### İç Bağlantılar
- [API Documentation](./API_DOCUMENTATION.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [User Guide](./USER_GUIDE.md)
- [Developer Guide](./DEVELOPER_GUIDE.md)
- [Component Documentation](./COMPONENT_DOCUMENTATION.md)
- [OpenAPI Spec](./OPENAPI_SPEC.yaml)

### Dış Bağlantılar
- [GitHub Repository](https://github.com/your-org/trendyol-gift-recommendation)
- [Live Demo](https://demo.example.com)
- [API Endpoint](https://api.example.com)
- [Status Page](https://status.example.com)

### Referanslar
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Trendyol API](https://developers.trendyol.com/)
- [OpenAPI Specification](https://swagger.io/specification/)

---

**Son Güncelleme:** Ocak 2024  
**Versiyon:** 1.0.0  
**Maintainers:** Documentation Team
