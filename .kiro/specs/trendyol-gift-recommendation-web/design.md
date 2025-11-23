# Design Document

## Overview

Trendyol Gift Recommendation Web uygulaması, eğitilmiş TinyRecursiveModels (TRM) modelini kullanarak kullanıcılara kişiselleştirilmiş hediye önerileri sunan full-stack bir web uygulamasıdır. Sistem üç ana katmandan oluşur:

1. **Frontend (React + TypeScript)**: Kullanıcı arayüzü ve etkileşim katmanı
2. **Backend API (Python + FastAPI)**: Model inference, tool orchestration ve API gateway
3. **Model Layer**: Eğitilmiş TRM modeli ve tool registry sistemi

Uygulama, gerçek zamanlı olarak Trendyol'dan ürün verisi çeker, altı farklı analiz aracı kullanarak kapsamlı değerlendirme yapar ve kullanıcıya en uygun önerileri sunar.

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           React Frontend (TypeScript)                 │  │
│  │  - User Profile Form                                  │  │
│  │  - Recommendation Display                             │  │
│  │  - Tool Results Visualization                         │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS/REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API Server                       │
│                   (FastAPI + Python)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   API        │  │   Cache      │  │   Queue      │     │
│  │   Gateway    │  │   Manager    │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Model Inference Service                     │  │
│  │  - Profile Encoding                                  │  │
│  │  - TRM Forward Pass                                  │  │
│  │  - Tool Orchestration                                │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Tool Registry & Execution                   │  │
│  │  - Price Comparison                                  │  │
│  │  - Inventory Check                                   │  │
│  │  - Review Analysis                                   │  │
│  │  - Trend Analysis                                    │  │
│  │  - Budget Optimizer                                  │  │
│  │  - Gift Recommendation                               │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Trendyol    │  │   Redis      │  │  PostgreSQL  │     │
│  │    API       │  │   Cache      │  │   Database   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- React 18+ (UI framework)
- TypeScript 5+ (Type safety)
- Vite (Build tool)
- TanStack Query (Data fetching & caching)
- Zustand (State management)
- Tailwind CSS (Styling)
- Recharts (Data visualization)

**Backend:**
- Python 3.10+
- FastAPI (Web framework)
- PyTorch 2.0+ (Model inference)
- Redis (Caching)
- PostgreSQL (Data persistence)
- Celery (Task queue)
- Pydantic (Data validation)

**Infrastructure:**
- Docker & Docker Compose
- Nginx (Reverse proxy)
- Prometheus & Grafana (Monitoring)

## Components and Interfaces

### Frontend Components

#### 1. UserProfileForm Component
```typescript
interface UserProfileFormProps {
  onSubmit: (profile: UserProfile) => void;
  isLoading: boolean;
}

interface UserProfile {
  age: number;
  hobbies: string[];
  relationship: string;
  budget: number;
  occasion: string;
  personalityTraits: string[];
}
```

**Responsibilities:**
- Kullanıcı profil verilerini toplama
- Form validasyonu
- Hata mesajları gösterme

#### 2. RecommendationCard Component
```typescript
interface RecommendationCardProps {
  gift: GiftItem;
  toolResults: ToolResults;
  onDetailsClick: () => void;
  onTrendyolClick: () => void;
}

interface GiftItem {
  id: string;
  name: string;
  category: string;
  price: number;
  rating: number;
  imageUrl: string;
  trendyolUrl: string;
  confidence: number;
}

interface ToolResults {
  priceComparison?: PriceComparisonResult;
  inventoryCheck?: InventoryCheckResult;
  reviewAnalysis?: ReviewAnalysisResult;
  trendAnalysis?: TrendAnalysisResult;
  budgetOptimizer?: BudgetOptimizerResult;
}
```

**Responsibilities:**
- Ürün bilgilerini görselleştirme
- Tool sonuçlarını gösterme
- Trendyol'a yönlendirme

#### 3. ToolResultsModal Component
```typescript
interface ToolResultsModalProps {
  gift: GiftItem;
  toolResults: ToolResults;
  isOpen: boolean;
  onClose: () => void;
}
```

**Responsibilities:**
- Detaylı tool analiz sonuçlarını gösterme
- Grafik ve chart'lar ile görselleştirme
- Karşılaştırmalı analiz sunma

### Backend API Endpoints

#### 1. POST /api/recommendations
```python
class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    max_recommendations: int = 5
    use_cache: bool = True

class RecommendationResponse(BaseModel):
    recommendations: List[GiftRecommendation]
    tool_results: Dict[str, Any]
    inference_time: float
    cache_hit: bool
```

#### 2. GET /api/health
```python
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trendyol_api_status: str
    cache_status: str
    timestamp: datetime
```

#### 3. GET /api/tools/stats
```python
class ToolStatsResponse(BaseModel):
    tool_usage: Dict[str, int]
    success_rates: Dict[str, float]
    average_execution_times: Dict[str, float]
```

### Backend Services

#### 1. ModelInferenceService
```python
class ModelInferenceService:
    def __init__(self, checkpoint_path: str):
        self.model = self._load_model(checkpoint_path)
        self.device = self._get_device()
    
    async def generate_recommendations(
        self, 
        user_profile: UserProfile,
        available_gifts: List[GiftItem]
    ) -> Tuple[List[GiftRecommendation], Dict[str, Any]]:
        """Generate recommendations using TRM model"""
        pass
    
    def _encode_user_profile(self, profile: UserProfile) -> torch.Tensor:
        """Encode user profile to model input format"""
        pass
    
    def _decode_model_output(
        self, 
        output: torch.Tensor,
        gifts: List[GiftItem]
    ) -> List[GiftRecommendation]:
        """Decode model output to recommendations"""
        pass
```

#### 2. ToolOrchestrationService
```python
class ToolOrchestrationService:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.executor = ThreadPoolExecutor(max_workers=6)
    
    async def execute_tools(
        self,
        selected_tools: List[str],
        tool_params: Dict[str, Dict[str, Any]],
        gifts: List[GiftItem]
    ) -> Dict[str, Any]:
        """Execute selected tools in parallel"""
        pass
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        timeout: float = 3.0
    ) -> Optional[Any]:
        """Execute a single tool with timeout"""
        pass
```

#### 3. TrendyolAPIService
```python
class TrendyolAPIService:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = RateLimiter(max_requests=100, window=60)
    
    async def search_products(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 50
    ) -> List[TrendyolProduct]:
        """Search products on Trendyol"""
        pass
    
    async def get_product_details(
        self,
        product_id: str
    ) -> TrendyolProduct:
        """Get detailed product information"""
        pass
    
    def _convert_to_gift_item(
        self,
        product: TrendyolProduct
    ) -> GiftItem:
        """Convert Trendyol product to GiftItem format"""
        pass
```

#### 4. CacheService
```python
class CacheService:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get_recommendations(
        self,
        profile_hash: str
    ) -> Optional[List[GiftRecommendation]]:
        """Get cached recommendations"""
        pass
    
    async def set_recommendations(
        self,
        profile_hash: str,
        recommendations: List[GiftRecommendation],
        ttl: int = None
    ) -> None:
        """Cache recommendations"""
        pass
    
    def _generate_profile_hash(
        self,
        profile: UserProfile
    ) -> str:
        """Generate unique hash for user profile"""
        pass
```

## Data Models

### User Profile
```python
class UserProfile(BaseModel):
    age: int = Field(ge=18, le=100)
    hobbies: List[str] = Field(min_items=1, max_items=10)
    relationship: str
    budget: float = Field(gt=0)
    occasion: str
    personality_traits: List[str] = Field(max_items=5)
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "hobbies": ["gardening", "cooking"],
                "relationship": "mother",
                "budget": 500.0,
                "occasion": "birthday",
                "personality_traits": ["practical", "eco-friendly"]
            }
        }
```

### Gift Item
```python
class GiftItem(BaseModel):
    id: str
    name: str
    category: str
    price: float
    rating: float = Field(ge=0, le=5)
    image_url: HttpUrl
    trendyol_url: HttpUrl
    description: str
    tags: List[str]
    age_suitability: Tuple[int, int]
    occasion_fit: List[str]
    in_stock: bool = True
```

### Gift Recommendation
```python
class GiftRecommendation(BaseModel):
    gift: GiftItem
    confidence_score: float = Field(ge=0, le=1)
    reasoning: List[str]
    tool_insights: Dict[str, Any]
    rank: int
```

### Tool Results
```python
class PriceComparisonResult(BaseModel):
    best_price: float
    average_price: float
    price_range: str
    savings_percentage: float
    checked_platforms: List[str]

class ReviewAnalysisResult(BaseModel):
    average_rating: float
    total_reviews: int
    sentiment_score: float
    key_positives: List[str]
    key_negatives: List[str]
    recommendation_confidence: float

class TrendAnalysisResult(BaseModel):
    trend_direction: str
    popularity_score: float
    growth_rate: float
    trending_items: List[str]

class BudgetOptimizerResult(BaseModel):
    recommended_allocation: Dict[str, float]
    value_score: float
    savings_opportunities: List[str]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Age Input Validation
*For any* age input value, the system should accept values between 18 and 100 (inclusive) and reject all other values
**Validates: Requirements 1.2**

### Property 2: Multi-Select Hobby Persistence
*For any* set of selected hobbies, all selected items should be persisted in the application state
**Validates: Requirements 1.3**

### Property 3: Budget Format and Validation
*For any* budget input, the system should accept only positive numeric values and display them in Turkish Lira format
**Validates: Requirements 1.4**

### Property 4: Form Validation Completeness
*For any* form submission attempt, the system should prevent submission if any required field is empty
**Validates: Requirements 1.5**

### Property 5: Profile JSON Serialization
*For any* user profile, saving and then loading the profile should produce an equivalent profile object (round-trip property)
**Validates: Requirements 1.6**

### Property 6: Profile to Model Input Transformation
*For any* valid user profile, the transformation to model input format should produce a valid tensor with expected dimensions
**Validates: Requirements 2.3**

### Property 7: Device Selection Based on Availability
*For any* inference request, the system should use GPU if available, otherwise CPU
**Validates: Requirements 2.4**

### Property 8: Model Output to Recommendations Transformation
*For any* model output tensor, the transformation should produce a list of recommendations with valid confidence scores (0-1 range)
**Validates: Requirements 2.5**

### Property 9: Tool Execution Order Preservation
*For any* list of selected tools, the system should execute them in the order they appear in the list
**Validates: Requirements 3.1**

### Property 10: Trendyol API Request Parameters
*For any* product search request, the system should include both category and keywords in the API call
**Validates: Requirements 4.1**

### Property 11: Trendyol Product to Gift Item Transformation
*For any* Trendyol product response, the transformation should produce a valid GiftItem with all required fields
**Validates: Requirements 4.2**

### Property 12: URL Validation and Filtering
*For any* list of product image URLs, the system should filter out invalid URLs and keep only valid HTTP/HTTPS URLs
**Validates: Requirements 4.5**

### Property 13: Price Normalization to TL Format
*For any* price value, the system should normalize it to Turkish Lira format with 2 decimal places
**Validates: Requirements 4.6**

### Property 14: Recommendation Card Rendering
*For any* list of recommendations, the UI should render exactly one card component per recommendation
**Validates: Requirements 5.1**

### Property 15: Product Card Content Completeness
*For any* rendered product card, it should display all required fields: image, name, price, rating, and category
**Validates: Requirements 5.2**

### Property 16: Low Confidence Warning Display
*For any* recommendation with confidence score below 0.5, the UI should display a warning message
**Validates: Requirements 5.6**

### Property 17: Error Logging Format
*For any* error that occurs, the system should log it with timestamp, error type, and stack trace
**Validates: Requirements 7.1**

### Property 18: Validation Error Field Identification
*For any* validation error, the error message should identify which specific field caused the error
**Validates: Requirements 7.4**

### Property 19: Responsive Layout Preservation
*For any* screen width (mobile, tablet, desktop), the UI should maintain proper layout without horizontal scrolling
**Validates: Requirements 8.1**

### Property 20: Real-time Validation Feedback
*For any* form field input, the system should provide validation feedback within 100ms of the input change
**Validates: Requirements 8.2**

### Property 21: Local Storage Favorites Persistence
*For any* product added to favorites, it should be retrievable from local storage after page reload
**Validates: Requirements 8.5**

### Property 22: Theme Consistency Across Components
*For any* theme selection (light/dark), all UI components should reflect the selected theme
**Validates: Requirements 8.6**

### Property 23: Personal Data Encryption
*For any* user data stored in the database, personal information fields should be encrypted
**Validates: Requirements 9.3**

### Property 24: Input Sanitization Against XSS
*For any* user input containing HTML/JavaScript, the system should sanitize it before rendering
**Validates: Requirements 9.5**



## Error Handling

### Frontend Error Handling

**Network Errors:**
- Retry mechanism with exponential backoff (3 attempts)
- Fallback to cached data if available
- User-friendly error messages
- Offline mode detection

**Validation Errors:**
- Real-time field-level validation
- Clear error messages next to fields
- Form submission prevention until valid
- Visual indicators (red borders, icons)

**Runtime Errors:**
- Error boundary components to catch React errors
- Graceful degradation of features
- Error reporting to monitoring service
- Fallback UI components

### Backend Error Handling

**Model Errors:**
```python
class ModelInferenceError(Exception):
    """Raised when model inference fails"""
    pass

class ModelLoadError(Exception):
    """Raised when model loading fails"""
    pass
```

**API Errors:**
```python
class TrendyolAPIError(Exception):
    """Raised when Trendyol API fails"""
    pass

class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""
    pass
```

**Tool Errors:**
```python
class ToolExecutionError(Exception):
    """Raised when tool execution fails"""
    pass

class ToolTimeoutError(Exception):
    """Raised when tool execution times out"""
    pass
```

**Error Response Format:**
```python
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]
    timestamp: datetime
    request_id: str
```

### Error Recovery Strategies

1. **Graceful Degradation**: If tools fail, continue with model-only recommendations
2. **Fallback Data**: Use cached data when external APIs fail
3. **Retry Logic**: Automatic retry for transient failures
4. **Circuit Breaker**: Prevent cascading failures by temporarily disabling failing services
5. **Timeout Management**: Set appropriate timeouts for all external calls

## Testing Strategy

### Unit Testing

**Frontend Unit Tests (Vitest + React Testing Library):**
- Component rendering tests
- User interaction tests
- State management tests
- Utility function tests
- Form validation tests

**Backend Unit Tests (pytest):**
- API endpoint tests
- Service layer tests
- Data transformation tests
- Validation tests
- Error handling tests

**Test Coverage Goals:**
- Minimum 80% code coverage
- 100% coverage for critical paths (authentication, payment, data validation)

### Property-Based Testing

**Testing Framework:** Hypothesis (Python) for backend, fast-check (TypeScript) for frontend

**Property Test Configuration:**
- Minimum 100 iterations per property test
- Shrinking enabled for failure case minimization
- Deterministic seed for reproducibility

**Property Test Examples:**

```python
# Backend Property Test Example
from hypothesis import given, strategies as st

@given(
    age=st.integers(min_value=18, max_value=100),
    hobbies=st.lists(st.text(min_size=1), min_size=1, max_size=10),
    budget=st.floats(min_value=0.01, max_value=100000)
)
def test_profile_encoding_preserves_data(age, hobbies, budget):
    """
    Feature: trendyol-gift-recommendation-web, Property 5: Profile JSON Serialization
    """
    profile = UserProfile(
        age=age,
        hobbies=hobbies,
        relationship="friend",
        budget=budget,
        occasion="birthday",
        personality_traits=["practical"]
    )
    
    # Serialize and deserialize
    json_str = profile.json()
    restored_profile = UserProfile.parse_raw(json_str)
    
    # Verify round-trip
    assert restored_profile.age == profile.age
    assert restored_profile.hobbies == profile.hobbies
    assert restored_profile.budget == profile.budget
```

```typescript
// Frontend Property Test Example
import fc from 'fast-check';

test('Property 3: Budget Format and Validation', () => {
  /**
   * Feature: trendyol-gift-recommendation-web, Property 3: Budget Format and Validation
   */
  fc.assert(
    fc.property(
      fc.float({ min: 0.01, max: 1000000 }),
      (budget) => {
        const formatted = formatBudget(budget);
        
        // Should be in TL format
        expect(formatted).toMatch(/^\d+,\d{2} ₺$/);
        
        // Should be parseable back to number
        const parsed = parseBudget(formatted);
        expect(Math.abs(parsed - budget)).toBeLessThan(0.01);
      }
    ),
    { numRuns: 100 }
  );
});
```

### Integration Testing

**API Integration Tests:**
- End-to-end API flow tests
- Database integration tests
- External API mock tests
- Cache integration tests

**Frontend Integration Tests:**
- User flow tests (Playwright)
- API integration tests
- State management integration tests

### Performance Testing

**Load Testing (Locust):**
- Concurrent user simulation
- Response time measurement
- Throughput testing
- Resource utilization monitoring

**Performance Benchmarks:**
- Model inference: < 2 seconds
- API response time: < 500ms (cached), < 3 seconds (uncached)
- Frontend initial load: < 2 seconds
- Tool execution: < 3 seconds per tool

### Security Testing

**Automated Security Scans:**
- OWASP ZAP for vulnerability scanning
- Dependency vulnerability scanning (npm audit, safety)
- SQL injection testing
- XSS testing
- CSRF protection testing

**Manual Security Review:**
- Code review for security issues
- Authentication/authorization testing
- Data encryption verification
- API security testing

## Deployment Architecture

### Development Environment
```
Docker Compose Setup:
- Frontend (Vite dev server)
- Backend (FastAPI with hot reload)
- Redis (cache)
- PostgreSQL (database)
- Nginx (reverse proxy)
```

### Production Environment
```
Kubernetes Cluster:
- Frontend: 3 replicas (Nginx serving static files)
- Backend: 5 replicas (FastAPI with Gunicorn)
- Redis: 1 master + 2 replicas
- PostgreSQL: 1 primary + 1 standby
- Model Server: 2 replicas (GPU-enabled)
- Load Balancer: Nginx Ingress
```

### CI/CD Pipeline

**GitHub Actions Workflow:**
1. Code checkout
2. Dependency installation
3. Linting (ESLint, Black, mypy)
4. Unit tests
5. Property-based tests
6. Integration tests
7. Build Docker images
8. Push to container registry
9. Deploy to staging
10. Run smoke tests
11. Deploy to production (manual approval)

### Monitoring and Observability

**Metrics (Prometheus + Grafana):**
- Request rate, latency, error rate
- Model inference time
- Tool execution time
- Cache hit rate
- Resource utilization (CPU, memory, GPU)

**Logging (ELK Stack):**
- Application logs
- Access logs
- Error logs
- Audit logs

**Tracing (Jaeger):**
- Distributed tracing for request flows
- Tool execution tracing
- Database query tracing

**Alerting (Alertmanager):**
- High error rate alerts
- Slow response time alerts
- Resource exhaustion alerts
- Model failure alerts

## Performance Optimization

### Frontend Optimization
- Code splitting and lazy loading
- Image optimization and lazy loading
- Memoization of expensive computations
- Virtual scrolling for long lists
- Service worker for offline support
- CDN for static assets

### Backend Optimization
- Connection pooling for database
- Redis caching for frequent queries
- Async I/O for external API calls
- Request batching for model inference
- GPU utilization optimization
- Query optimization and indexing

### Model Optimization
- Model quantization (FP16)
- Batch inference when possible
- Model caching in memory
- Warm-up requests on startup
- GPU memory management

## Security Considerations

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management for external services
- Session management with secure cookies

### Data Protection
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII data masking in logs
- Secure key management (AWS KMS / HashiCorp Vault)

### API Security
- Rate limiting per user/IP
- CORS configuration
- Input validation and sanitization
- SQL injection prevention (parameterized queries)
- XSS prevention (Content Security Policy)
- CSRF protection

### Infrastructure Security
- Network segmentation
- Firewall rules
- DDoS protection
- Regular security updates
- Vulnerability scanning
- Penetration testing

## Scalability Considerations

### Horizontal Scaling
- Stateless backend services
- Load balancing across replicas
- Database read replicas
- Cache clustering

### Vertical Scaling
- GPU-enabled instances for model inference
- Memory optimization for large models
- CPU optimization for tool execution

### Data Scaling
- Database partitioning
- Cache eviction policies
- Log rotation and archival
- Backup and disaster recovery

## Future Enhancements

1. **Multi-language Support**: Turkish and English UI
2. **Personalization**: User preference learning over time
3. **Social Features**: Share recommendations, collaborative filtering
4. **Mobile App**: Native iOS and Android apps
5. **Voice Interface**: Voice-based product search
6. **AR Preview**: Augmented reality product preview
7. **Price Alerts**: Notify users of price drops
8. **Wishlist**: Save products for later
9. **Comparison Tool**: Side-by-side product comparison
10. **Gift Registry**: Create and share gift registries
