# Trendyol Gift Recommendation API Documentation

## Overview

Trendyol Gift Recommendation API, kullanıcı profillerine göre kişiselleştirilmiş hediye önerileri sunan bir REST API'dir. API, eğitilmiş TinyRecursiveModels (TRM) modelini kullanarak gerçek zamanlı öneriler üretir ve Trendyol'dan ürün verilerini çeker.

**Base URL:** `http://localhost:8000` (Development)  
**API Version:** v1  
**API Prefix:** `/api/v1`

## Authentication

Şu anda API public erişime açıktır. Production ortamında JWT token tabanlı authentication kullanılacaktır.

## Rate Limiting

- **Limit:** 10 requests per minute per IP
- **Headers:** 
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Time when limit resets

## Endpoints

### 1. Get Recommendations

Kullanıcı profiline göre hediye önerileri alır.

**Endpoint:** `POST /api/v1/recommendations`

**Request Body:**
```json
{
  "user_profile": {
    "age": 35,
    "hobbies": ["gardening", "cooking"],
    "relationship": "mother",
    "budget": 500.0,
    "occasion": "birthday",
    "personality_traits": ["practical", "eco-friendly"]
  },
  "max_recommendations": 5,
  "use_cache": true
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| user_profile | object | Yes | Kullanıcı profil bilgileri |
| user_profile.age | integer | Yes | Yaş (18-100 arası) |
| user_profile.hobbies | array[string] | Yes | Hobiler (1-10 arası) |
| user_profile.relationship | string | Yes | İlişki durumu |
| user_profile.budget | number | Yes | Bütçe (TL, pozitif değer) |
| user_profile.occasion | string | Yes | Özel gün |
| user_profile.personality_traits | array[string] | No | Kişilik özellikleri (max 5) |
| max_recommendations | integer | No | Maksimum öneri sayısı (1-20, default: 5) |
| use_cache | boolean | No | Cache kullanımı (default: true) |

**Response (200 OK):**
```json
{
  "recommendations": [
    {
      "gift": {
        "id": "12345",
        "name": "Premium Coffee Set",
        "category": "Kitchen & Dining",
        "price": 299.99,
        "rating": 4.5,
        "image_url": "https://cdn.trendyol.com/example.jpg",
        "trendyol_url": "https://www.trendyol.com/product/12345",
        "description": "High-quality coffee set",
        "tags": ["coffee", "kitchen", "gift"],
        "age_suitability": [25, 65],
        "occasion_fit": ["birthday", "anniversary"],
        "in_stock": true
      },
      "confidence_score": 0.92,
      "reasoning": [
        "Matches user's cooking hobby",
        "Within budget range",
        "High rating and positive reviews"
      ],
      "tool_insights": {
        "price_comparison": {
          "best_price": 299.99,
          "average_price": 350.0,
          "savings_percentage": 14.3
        },
        "review_analysis": {
          "average_rating": 4.5,
          "sentiment_score": 0.85,
          "key_positives": ["quality", "design", "value"]
        }
      },
      "rank": 1
    }
  ],
  "tool_results": {},
  "inference_time": 1.23,
  "cache_hit": false
}
```

**Error Responses:**

- **422 Unprocessable Entity:** Validation error
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Geçersiz değer: age",
  "details": {
    "field": "age",
    "validation_errors": [...]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc-123"
}
```

- **503 Service Unavailable:** Model or Trendyol API unavailable
```json
{
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Model şu anda kullanılamıyor",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc-123"
}
```

### 2. Health Check

Sistem sağlık durumunu kontrol eder.

**Endpoint:** `GET /api/v1/health`

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "trendyol_api_status": "operational",
  "cache_status": "connected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 3. Get Metrics

Prometheus formatında metrikleri döndürür.

**Endpoint:** `GET /api/v1/metrics`

**Response (200 OK):**
```
# HELP recommendation_requests_total Total recommendation requests
# TYPE recommendation_requests_total counter
recommendation_requests_total 1234

# HELP inference_duration_seconds Model inference duration
# TYPE inference_duration_seconds histogram
inference_duration_seconds_bucket{le="0.5"} 100
inference_duration_seconds_bucket{le="1.0"} 250
...
```

### 4. Get Resources

Sistem kaynak kullanımını döndürür.

**Endpoint:** `GET /api/v1/resources`

**Response (200 OK):**
```json
{
  "cpu_percent": 45.2,
  "memory_percent": 62.8,
  "memory_used_mb": 2048,
  "memory_total_mb": 8192,
  "gpu_available": true,
  "gpu_memory_used_mb": 1024,
  "gpu_memory_total_mb": 4096
}
```

### 5. Get Tool Statistics

Tool kullanım istatistiklerini döndürür.

**Endpoint:** `GET /api/v1/tools/stats`

**Response (200 OK):**
```json
{
  "tool_usage": {
    "price_comparison": 1234,
    "review_analysis": 1150,
    "trend_analysis": 980,
    "budget_optimizer": 890,
    "inventory_check": 1200,
    "gift_recommendation": 1234
  },
  "success_rates": {
    "price_comparison": 0.95,
    "review_analysis": 0.92,
    "trend_analysis": 0.88,
    "budget_optimizer": 0.90,
    "inventory_check": 0.98,
    "gift_recommendation": 0.94
  },
  "average_execution_times": {
    "price_comparison": 0.45,
    "review_analysis": 0.82,
    "trend_analysis": 0.65,
    "budget_optimizer": 0.38,
    "inventory_check": 0.25,
    "gift_recommendation": 1.20
  }
}
```

## Data Models

### UserProfile

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| age | integer | 18-100 | Hediye alınacak kişinin yaşı |
| hobbies | array[string] | 1-10 items | Hobi listesi |
| relationship | string | required | İlişki durumu (örn: "mother", "friend") |
| budget | number | > 0 | Bütçe (Türk Lirası) |
| occasion | string | required | Özel gün (örn: "birthday", "anniversary") |
| personality_traits | array[string] | max 5 items | Kişilik özellikleri |

### GiftItem

| Field | Type | Description |
|-------|------|-------------|
| id | string | Ürün ID |
| name | string | Ürün adı |
| category | string | Kategori |
| price | number | Fiyat (TL) |
| rating | number | Değerlendirme (0-5) |
| image_url | string | Görsel URL |
| trendyol_url | string | Trendyol ürün sayfası URL |
| description | string | Ürün açıklaması |
| tags | array[string] | Etiketler |
| age_suitability | [int, int] | Yaş uygunluğu aralığı |
| occasion_fit | array[string] | Uygun özel günler |
| in_stock | boolean | Stok durumu |

### GiftRecommendation

| Field | Type | Description |
|-------|------|-------------|
| gift | GiftItem | Ürün bilgileri |
| confidence_score | number | Güven skoru (0-1) |
| reasoning | array[string] | Öneri gerekçeleri |
| tool_insights | object | Tool analiz sonuçları |
| rank | integer | Sıralama |

## Error Codes

| Code | Description |
|------|-------------|
| VALIDATION_ERROR | Form validasyon hatası |
| MODEL_INFERENCE_ERROR | Model çalıştırma hatası |
| TRENDYOL_API_ERROR | Trendyol API hatası |
| CACHE_ERROR | Cache işlem hatası |
| RATE_LIMIT_EXCEEDED | Rate limit aşıldı |
| SERVICE_UNAVAILABLE | Servis kullanılamıyor |
| INTERNAL_ERROR | İç hata |

## Examples

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "age": 35,
      "hobbies": ["gardening", "cooking"],
      "relationship": "mother",
      "budget": 500.0,
      "occasion": "birthday",
      "personality_traits": ["practical"]
    },
    "max_recommendations": 5,
    "use_cache": true
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8000/api/v1/recommendations"
payload = {
    "user_profile": {
        "age": 35,
        "hobbies": ["gardening", "cooking"],
        "relationship": "mother",
        "budget": 500.0,
        "occasion": "birthday",
        "personality_traits": ["practical"]
    },
    "max_recommendations": 5,
    "use_cache": True
}

response = requests.post(url, json=payload)
recommendations = response.json()
print(f"Found {len(recommendations['recommendations'])} recommendations")
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/api/v1/recommendations', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    user_profile: {
      age: 35,
      hobbies: ['gardening', 'cooking'],
      relationship: 'mother',
      budget: 500.0,
      occasion: 'birthday',
      personality_traits: ['practical']
    },
    max_recommendations: 5,
    use_cache: true
  })
});

const data = await response.json();
console.log(`Found ${data.recommendations.length} recommendations`);
```

## Interactive Documentation

API'yi interaktif olarak test etmek için:

- **Swagger UI:** http://localhost:8000/api/v1/docs
- **ReDoc:** http://localhost:8000/api/v1/redoc

## Performance

- **Cache Hit Response Time:** < 100ms
- **Cache Miss Response Time:** < 3 seconds
- **Model Inference Time:** < 2 seconds
- **Tool Execution Time:** < 3 seconds per tool

## Support

Sorularınız için:
- GitHub Issues: [repository-url]/issues
- Email: support@example.com
