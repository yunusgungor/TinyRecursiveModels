"""Pytest configuration and fixtures"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import create_application
from app.services.model_inference import get_model_service
from app.core.config import settings


@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Load model once for all tests"""
    # Check if model checkpoint exists
    checkpoint_path = Path(settings.MODEL_CHECKPOINT_PATH)
    if checkpoint_path.exists():
        try:
            model_service = get_model_service()
            model_service.load_model()
            
            # Notify monitoring service that model is loaded
            from app.services.monitoring_service import monitoring_service
            monitoring_service.set_model_loaded(True)
            
            print(f"\n✓ Model loaded successfully from {settings.MODEL_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"\n✗ Failed to load model: {e}")
            # Don't fail tests if model can't be loaded
            pass
    else:
        print(f"\n⚠ Model checkpoint not found at {settings.MODEL_CHECKPOINT_PATH}")


@pytest.fixture(scope="function")
def app(monkeypatch):
    """Create FastAPI application for testing"""
    # Disable rate limiting for tests by patching settings
    monkeypatch.setattr("app.core.config.settings.RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr("app.core.config.settings.DEBUG", True)
    
    return create_application()


@pytest.fixture(scope="function")
def client(app):
    """Create test client"""
    # Clear rate limiter state before each test
    from app.middleware.rate_limiter import rate_limiter
    rate_limiter.requests.clear()
    
    # Create a new client for each test with HTTPS base URL to avoid redirects
    return TestClient(app, base_url="https://testserver", raise_server_exceptions=False)


@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing"""
    return {
        "age": 35,
        "hobbies": ["gardening", "cooking"],
        "relationship": "mother",
        "budget": 500.0,
        "occasion": "birthday",
        "personality_traits": ["practical", "eco-friendly"]
    }


@pytest.fixture
def sample_gift_item():
    """Sample gift item for testing"""
    return {
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
        "in_stock": True
    }


@pytest.fixture
def mock_model_service():
    """Mock model inference service for testing"""
    from unittest.mock import MagicMock, AsyncMock, patch
    from app.models.schemas import GiftRecommendation, GiftItem
    
    with patch('app.api.v1.recommendations.get_model_service') as mock:
        service_instance = MagicMock()
        service_instance.model_loaded = True
        service_instance.is_loaded = MagicMock(return_value=True)
        service_instance.generate_recommendations = AsyncMock(return_value=(
            [
                GiftRecommendation(
                    gift=GiftItem(
                        id="12345",
                        name="Premium Coffee Set",
                        category="Kitchen & Dining",
                        price=299.99,
                        rating=4.5,
                        image_url="https://cdn.trendyol.com/example.jpg",
                        trendyol_url="https://www.trendyol.com/product/12345",
                        description="High-quality coffee set",
                        tags=["coffee", "kitchen", "gift"],
                        age_suitability=(25, 65),
                        occasion_fit=["birthday", "anniversary"],
                        in_stock=True
                    ),
                    confidence_score=0.85,
                    reasoning=["Matches hobbies", "Within budget"],
                    tool_insights={},
                    rank=1
                )
            ],
            {}  # tool_results
        ))
        mock.return_value = service_instance
        yield service_instance
