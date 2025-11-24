# Developer Guide

Bu doküman, Trendyol Gift Recommendation projesine katkıda bulunmak isteyen geliştiriciler için kapsamlı bir rehberdir.

## İçindekiler

1. [Proje Yapısı](#proje-yapısı)
2. [Development Setup](#development-setup)
3. [Kod Standartları](#kod-standartları)
4. [Testing](#testing)
5. [API Development](#api-development)
6. [Frontend Development](#frontend-development)
7. [Model Integration](#model-integration)
8. [Database](#database)
9. [Debugging](#debugging)
10. [Contributing](#contributing)

## Proje Yapısı

```
trendyol-gift-recommendation/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   │   └── v1/
│   │   │       ├── health.py
│   │   │       ├── recommendations.py
│   │   │       ├── tools.py
│   │   │       └── metrics.py
│   │   ├── core/              # Core functionality
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── exceptions.py
│   │   │   ├── logging.py
│   │   │   ├── security.py
│   │   │   └── tracing.py
│   │   ├── middleware/        # Custom middleware
│   │   │   ├── error_handler.py
│   │   │   ├── rate_limiter.py
│   │   │   ├── session.py
│   │   │   └── https_redirect.py
│   │   ├── models/            # Data models
│   │   │   └── schemas.py
│   │   ├── services/          # Business logic
│   │   │   ├── model_inference.py
│   │   │   ├── tool_orchestration.py
│   │   │   ├── trendyol_api.py
│   │   │   ├── cache_service.py
│   │   │   └── monitoring_service.py
│   │   └── main.py            # Application entry point
│   ├── tests/                 # Test suite
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── property/
│   │   └── performance/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/                   # React TypeScript frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── UserProfileForm.tsx
│   │   │   ├── RecommendationCard.tsx
│   │   │   ├── ToolResultsModal.tsx
│   │   │   └── __tests__/
│   │   ├── contexts/          # React contexts
│   │   │   └── ThemeContext.tsx
│   │   ├── hooks/             # Custom hooks
│   │   │   ├── useRecommendations.ts
│   │   │   └── useHealth.ts
│   │   ├── lib/               # Utilities
│   │   │   ├── api/
│   │   │   └── utils/
│   │   ├── pages/             # Page components
│   │   │   └── HomePage.tsx
│   │   ├── store/             # State management
│   │   │   └── useAppStore.ts
│   │   └── main.tsx
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.ts
│
├── models/                     # ML models
│   ├── rl/                    # Reinforcement learning
│   │   ├── enhanced_recommendation_engine.py
│   │   ├── enhanced_user_profiler.py
│   │   └── rl_trm.py
│   └── tools/                 # Tool implementations
│       ├── gift_tools.py
│       ├── integrated_enhanced_trm.py
│       └── tool_registry.py
│
├── checkpoints/               # Model checkpoints
├── data/                      # Data files
├── docs/                      # Documentation
├── k8s/                       # Kubernetes manifests
├── monitoring/                # Monitoring configs
├── nginx/                     # Nginx configs
└── docker-compose.yml
```

## Development Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Node.js 18+
node --version

# Docker & Docker Compose
docker --version
docker-compose --version

# Git
git --version
```

### Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/trendyol-gift-recommendation.git
cd trendyol-gift-recommendation

# 2. Setup pre-commit hooks
pip install pre-commit
pre-commit install

# 3. Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 4. Start services
docker-compose up -d

# 5. Run migrations
docker-compose exec backend alembic upgrade head

# 6. Verify setup
curl http://localhost:8000/api/v1/health
```

### Backend Development

```bash
# Enter backend container
docker-compose exec backend bash

# Install dependencies
pip install -r requirements-dev.txt

# Run development server (with hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run specific test
pytest tests/unit/test_model_inference_service.py -v

# Run with coverage
pytest --cov=app --cov-report=html

# Type checking
mypy app/

# Linting
black app/
flake8 app/
```

### Frontend Development

```bash
# Enter frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm test

# Run tests in watch mode
npm test -- --watch

# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build
```

## Kod Standartları

### Python Code Style

**PEP 8 Compliance:**
```python
# Good
def calculate_confidence_score(
    user_profile: UserProfile,
    gift_item: GiftItem,
    tool_results: Dict[str, Any]
) -> float:
    """
    Calculate confidence score for a gift recommendation.
    
    Args:
        user_profile: User profile data
        gift_item: Gift item to evaluate
        tool_results: Results from analysis tools
        
    Returns:
        Confidence score between 0 and 1
    """
    score = 0.0
    
    # Budget match
    if gift_item.price <= user_profile.budget:
        score += 0.3
    
    # Hobby match
    matching_hobbies = set(gift_item.tags) & set(user_profile.hobbies)
    score += len(matching_hobbies) * 0.1
    
    return min(score, 1.0)


# Bad
def calc_score(profile,item,results):
    s=0.0
    if item.price<=profile.budget:s+=0.3
    s+=len(set(item.tags)&set(profile.hobbies))*0.1
    return min(s,1.0)
```

**Type Hints:**
```python
# Always use type hints
from typing import List, Dict, Optional, Tuple

def process_recommendations(
    recommendations: List[GiftRecommendation],
    max_count: Optional[int] = None
) -> Tuple[List[GiftRecommendation], Dict[str, Any]]:
    """Process and filter recommendations"""
    pass
```

**Docstrings:**
```python
def complex_function(param1: str, param2: int) -> bool:
    """
    One-line summary of function.
    
    Detailed description of what the function does,
    including any important notes or caveats.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
        
    Example:
        >>> complex_function("test", 42)
        True
    """
    pass
```

### TypeScript Code Style

**Naming Conventions:**
```typescript
// Interfaces: PascalCase
interface UserProfile {
  age: number;
  hobbies: string[];
}

// Types: PascalCase
type RecommendationStatus = 'loading' | 'success' | 'error';

// Components: PascalCase
const UserProfileForm: React.FC<Props> = ({ onSubmit }) => {
  // ...
};

// Functions: camelCase
const calculateBudget = (profile: UserProfile): number => {
  // ...
};

// Constants: UPPER_SNAKE_CASE
const MAX_RECOMMENDATIONS = 5;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
```

**Component Structure:**
```typescript
// Good structure
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

interface Props {
  userId: string;
  onComplete: (data: RecommendationData) => void;
}

export const RecommendationCard: React.FC<Props> = ({ 
  userId, 
  onComplete 
}) => {
  // 1. Hooks
  const [isExpanded, setIsExpanded] = useState(false);
  const { data, isLoading } = useQuery(['recommendations', userId], 
    fetchRecommendations
  );
  
  // 2. Effects
  useEffect(() => {
    if (data) {
      onComplete(data);
    }
  }, [data, onComplete]);
  
  // 3. Event handlers
  const handleExpand = () => {
    setIsExpanded(!isExpanded);
  };
  
  // 4. Render helpers
  const renderContent = () => {
    if (isLoading) return <Spinner />;
    return <Content data={data} />;
  };
  
  // 5. Main render
  return (
    <div className="recommendation-card">
      {renderContent()}
    </div>
  );
};
```

### Git Commit Messages

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```bash
# Good
git commit -m "feat(api): add recommendation caching"
git commit -m "fix(frontend): resolve mobile layout issue"
git commit -m "docs(readme): update installation instructions"

# Bad
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "asdfasdf"
```

## Testing

### Unit Tests

**Backend Unit Test Example:**
```python
# tests/unit/test_model_inference_service.py
import pytest
from app.services.model_inference import ModelInferenceService
from app.models.schemas import UserProfile, GiftItem

@pytest.fixture
def model_service():
    """Fixture for model service"""
    return ModelInferenceService(
        checkpoint_path="checkpoints/test_model.pt"
    )

@pytest.fixture
def sample_profile():
    """Fixture for sample user profile"""
    return UserProfile(
        age=35,
        hobbies=["cooking", "gardening"],
        relationship="mother",
        budget=500.0,
        occasion="birthday",
        personality_traits=["practical"]
    )

def test_encode_user_profile(model_service, sample_profile):
    """Test user profile encoding"""
    tensor = model_service._encode_user_profile(sample_profile)
    
    assert tensor is not None
    assert tensor.shape[0] > 0
    assert not torch.isnan(tensor).any()

def test_generate_recommendations_timeout(model_service, sample_profile):
    """Test recommendation generation with timeout"""
    with pytest.raises(TimeoutError):
        model_service.generate_recommendations(
            user_profile=sample_profile,
            available_gifts=[],
            timeout=0.001  # Very short timeout
        )
```

**Frontend Unit Test Example:**
```typescript
// src/components/__tests__/UserProfileForm.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { UserProfileForm } from '../UserProfileForm';

describe('UserProfileForm', () => {
  const mockOnSubmit = jest.fn();
  
  beforeEach(() => {
    mockOnSubmit.mockClear();
  });
  
  it('renders all form fields', () => {
    render(<UserProfileForm onSubmit={mockOnSubmit} isLoading={false} />);
    
    expect(screen.getByLabelText(/yaş/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/hobiler/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/bütçe/i)).toBeInTheDocument();
  });
  
  it('validates age input', async () => {
    render(<UserProfileForm onSubmit={mockOnSubmit} isLoading={false} />);
    
    const ageInput = screen.getByLabelText(/yaş/i);
    fireEvent.change(ageInput, { target: { value: '150' } });
    
    await waitFor(() => {
      expect(screen.getByText(/geçersiz yaş/i)).toBeInTheDocument();
    });
  });
  
  it('submits form with valid data', async () => {
    render(<UserProfileForm onSubmit={mockOnSubmit} isLoading={false} />);
    
    // Fill form
    fireEvent.change(screen.getByLabelText(/yaş/i), { 
      target: { value: '35' } 
    });
    // ... fill other fields
    
    // Submit
    fireEvent.click(screen.getByRole('button', { name: /öneri al/i }));
    
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          age: 35,
          // ... other fields
        })
      );
    });
  });
});
```

### Property-Based Tests

**Backend Property Test:**
```python
# tests/property/test_model_inference_properties.py
from hypothesis import given, strategies as st
from app.models.schemas import UserProfile

@given(
    age=st.integers(min_value=18, max_value=100),
    budget=st.floats(min_value=0.01, max_value=100000),
    hobbies=st.lists(st.text(min_size=1), min_size=1, max_size=10)
)
def test_profile_encoding_round_trip(age, budget, hobbies):
    """
    Feature: trendyol-gift-recommendation-web, Property 5: Profile JSON Serialization
    
    For any valid user profile, serializing and deserializing should
    produce an equivalent profile.
    """
    profile = UserProfile(
        age=age,
        hobbies=hobbies,
        relationship="friend",
        budget=budget,
        occasion="birthday",
        personality_traits=[]
    )
    
    # Serialize
    json_str = profile.json()
    
    # Deserialize
    restored = UserProfile.parse_raw(json_str)
    
    # Verify
    assert restored.age == profile.age
    assert restored.hobbies == profile.hobbies
    assert abs(restored.budget - profile.budget) < 0.01
```

**Frontend Property Test:**
```typescript
// src/lib/utils/__tests__/budget.property.test.ts
import fc from 'fast-check';
import { formatBudget, parseBudget } from '../budget';

test('Property 3: Budget Format and Validation', () => {
  /**
   * Feature: trendyol-gift-recommendation-web, Property 3
   * 
   * For any positive number, formatting and parsing should
   * preserve the value (within floating point precision).
   */
  fc.assert(
    fc.property(
      fc.float({ min: 0.01, max: 1000000 }),
      (budget) => {
        const formatted = formatBudget(budget);
        const parsed = parseBudget(formatted);
        
        // Should preserve value within 0.01 TL
        expect(Math.abs(parsed - budget)).toBeLessThan(0.01);
      }
    ),
    { numRuns: 100 }
  );
});
```

### Integration Tests

```python
# tests/integration/test_recommendation_endpoint.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommendation_endpoint_e2e():
    """Test complete recommendation flow"""
    # Prepare request
    request_data = {
        "user_profile": {
            "age": 35,
            "hobbies": ["cooking"],
            "relationship": "mother",
            "budget": 500.0,
            "occasion": "birthday",
            "personality_traits": ["practical"]
        },
        "max_recommendations": 5,
        "use_cache": False
    }
    
    # Make request
    response = client.post("/api/v1/recommendations", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5
    assert "inference_time" in data
    assert data["inference_time"] < 5.0  # Should be fast
    
    # Verify recommendation structure
    for rec in data["recommendations"]:
        assert "gift" in rec
        assert "confidence_score" in rec
        assert 0 <= rec["confidence_score"] <= 1
        assert "reasoning" in rec
        assert len(rec["reasoning"]) > 0
```

## API Development

### Adding New Endpoint

1. **Define Schema:**
```python
# app/models/schemas.py
class NewFeatureRequest(BaseModel):
    param1: str
    param2: int = Field(ge=0, le=100)

class NewFeatureResponse(BaseModel):
    result: str
    metadata: Dict[str, Any]
```

2. **Create Endpoint:**
```python
# app/api/v1/new_feature.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import NewFeatureRequest, NewFeatureResponse

router = APIRouter()

@router.post("/new-feature", response_model=NewFeatureResponse)
async def new_feature(request: NewFeatureRequest) -> NewFeatureResponse:
    """
    New feature endpoint
    
    Args:
        request: Feature request data
        
    Returns:
        Feature response
    """
    try:
        # Implementation
        result = process_feature(request)
        
        return NewFeatureResponse(
            result=result,
            metadata={"processed": True}
        )
    except Exception as e:
        logger.error(f"Feature error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Feature processing failed"
        )
```

3. **Register Router:**
```python
# app/main.py
from app.api.v1 import new_feature

app.include_router(
    new_feature.router,
    prefix=settings.API_V1_PREFIX,
    tags=["new-feature"]
)
```

4. **Add Tests:**
```python
# tests/unit/test_new_feature.py
def test_new_feature_endpoint():
    response = client.post("/api/v1/new-feature", json={
        "param1": "test",
        "param2": 50
    })
    assert response.status_code == 200
```

### Error Handling

```python
# app/core/exceptions.py
class CustomAPIException(BaseAPIException):
    """Custom exception for specific error"""
    
    def __init__(self, detail: str = None):
        super().__init__(
            error_code="CUSTOM_ERROR",
            message=detail or "Custom error occurred",
            details={}
        )

# Usage in endpoint
from app.core.exceptions import CustomAPIException

@router.post("/endpoint")
async def endpoint():
    if error_condition:
        raise CustomAPIException("Specific error message")
```

## Frontend Development

### Creating New Component

```typescript
// src/components/NewComponent.tsx
import React from 'react';

interface NewComponentProps {
  title: string;
  onAction: () => void;
  optional?: string;
}

/**
 * NewComponent - Brief description
 * 
 * @param title - Component title
 * @param onAction - Action callback
 * @param optional - Optional parameter
 */
export const NewComponent: React.FC<NewComponentProps> = ({
  title,
  onAction,
  optional
}) => {
  return (
    <div className="new-component">
      <h2>{title}</h2>
      {optional && <p>{optional}</p>}
      <button onClick={onAction}>Action</button>
    </div>
  );
};
```

### Custom Hook

```typescript
// src/hooks/useNewFeature.ts
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';

interface UseNewFeatureOptions {
  enabled?: boolean;
  refetchInterval?: number;
}

export const useNewFeature = (
  param: string,
  options: UseNewFeatureOptions = {}
) => {
  const [state, setState] = useState<string>('');
  
  const { data, isLoading, error } = useQuery(
    ['newFeature', param],
    () => fetchNewFeature(param),
    {
      enabled: options.enabled ?? true,
      refetchInterval: options.refetchInterval
    }
  );
  
  useEffect(() => {
    if (data) {
      setState(data.result);
    }
  }, [data]);
  
  return {
    state,
    isLoading,
    error
  };
};
```

### State Management

```typescript
// src/store/useAppStore.ts
import create from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // State
  theme: 'light' | 'dark';
  favorites: string[];
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void;
  addFavorite: (id: string) => void;
  removeFavorite: (id: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'light',
      favorites: [],
      
      // Actions
      setTheme: (theme) => set({ theme }),
      
      addFavorite: (id) => set((state) => ({
        favorites: [...state.favorites, id]
      })),
      
      removeFavorite: (id) => set((state) => ({
        favorites: state.favorites.filter(fav => fav !== id)
      }))
    }),
    {
      name: 'app-storage'
    }
  )
);
```

## Model Integration

### Loading Model

```python
# app/services/model_inference.py
import torch
from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM

class ModelInferenceService:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> torch.device:
        """Determine device (GPU/CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _load_model(self):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
            
            self.model = IntegratedEnhancedTRM(
                # Model configuration
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")
```

### Model Inference

```python
async def generate_recommendations(
    self,
    user_profile: UserProfile,
    available_gifts: List[GiftItem],
    max_recommendations: int = 5
) -> Tuple[List[GiftRecommendation], Dict[str, Any]]:
    """Generate recommendations using model"""
    
    # Encode inputs
    profile_tensor = self._encode_user_profile(user_profile)
    gifts_tensor = self._encode_gifts(available_gifts)
    
    # Run inference
    with torch.no_grad():
        output = self.model(
            profile=profile_tensor,
            gifts=gifts_tensor
        )
    
    # Decode output
    recommendations = self._decode_output(
        output,
        available_gifts,
        max_recommendations
    )
    
    return recommendations, {}
```

## Database

### Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "add user table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# Show current version
alembic current

# Show history
alembic history
```

### Database Models

```python
# app/models/db_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
```

## Debugging

### Backend Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()

# VS Code launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload"
      ],
      "jinja": true
    }
  ]
}
```

### Frontend Debugging

```typescript
// Console debugging
console.log('Debug:', variable);
console.table(arrayData);
console.trace();

// React DevTools
// Install browser extension

// VS Code launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "chrome",
      "request": "launch",
      "name": "Launch Chrome",
      "url": "http://localhost:3000",
      "webRoot": "${workspaceFolder}/frontend/src"
    }
  ]
}
```

### Logging

```python
# Backend logging
from app.core.logging import logger

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
logger.critical("Critical message")

# Structured logging
logger.info(
    "Recommendation generated",
    extra={
        "user_id": user_id,
        "inference_time": 1.23,
        "num_recommendations": 5
    }
)
```

## Contributing

### Pull Request Process

1. **Create Branch:**
```bash
git checkout -b feature/new-feature
# or
git checkout -b fix/bug-fix
```

2. **Make Changes:**
- Write code
- Add tests
- Update documentation

3. **Run Tests:**
```bash
# Backend
pytest
mypy app/
black app/

# Frontend
npm test
npm run lint
npm run type-check
```

4. **Commit:**
```bash
git add .
git commit -m "feat(scope): description"
```

5. **Push:**
```bash
git push origin feature/new-feature
```

6. **Create PR:**
- Go to GitHub
- Create Pull Request
- Fill in template
- Request review

### Code Review Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No console.log/print statements
- [ ] Error handling implemented
- [ ] Performance considered
- [ ] Security reviewed
- [ ] Backwards compatible

### Release Process

1. Update version in `package.json` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to staging
7. Run smoke tests
8. Deploy to production

## Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [PyTorch Docs](https://pytorch.org/docs/)

### Tools
- [VS Code](https://code.visualstudio.com/)
- [Postman](https://www.postman.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [pgAdmin](https://www.pgadmin.org/)

### Community
- GitHub Discussions
- Slack Channel
- Weekly Meetings

---

**Last Updated:** January 2024  
**Maintainers:** Development Team
