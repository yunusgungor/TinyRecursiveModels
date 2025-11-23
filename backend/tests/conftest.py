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


@pytest.fixture
def app():
    """Create FastAPI application for testing"""
    return create_application()


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


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
