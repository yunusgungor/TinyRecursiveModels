"""Unit tests for Pydantic data models"""

import pytest
from pydantic import ValidationError

from app.models.schemas import UserProfile, GiftItem, RecommendationRequest


class TestUserProfile:
    """Test UserProfile model validation"""
    
    def test_valid_user_profile(self, sample_user_profile):
        """Test creating a valid user profile"""
        profile = UserProfile(**sample_user_profile)
        assert profile.age == 35
        assert profile.budget == 500.0
        assert len(profile.hobbies) == 2
    
    def test_age_validation_minimum(self):
        """Test age validation rejects values below 18"""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                age=17,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday"
            )
        errors = exc_info.value.errors()
        assert any("age" in str(error["loc"]) for error in errors)
    
    def test_age_validation_maximum(self):
        """Test age validation rejects values above 100"""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                age=101,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday"
            )
        errors = exc_info.value.errors()
        assert any("age" in str(error["loc"]) for error in errors)
    
    def test_budget_validation_positive(self):
        """Test budget must be positive"""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                age=25,
                hobbies=["reading"],
                relationship="friend",
                budget=-10.0,
                occasion="birthday"
            )
        errors = exc_info.value.errors()
        assert any("budget" in str(error["loc"]) for error in errors)
    
    def test_hobbies_validation_empty_list(self):
        """Test hobbies list cannot be empty"""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                age=25,
                hobbies=[],
                relationship="friend",
                budget=100.0,
                occasion="birthday"
            )
        errors = exc_info.value.errors()
        assert any("hobbies" in str(error["loc"]) for error in errors)
    
    def test_hobbies_validation_empty_strings(self):
        """Test hobbies cannot contain empty strings"""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                age=25,
                hobbies=["reading", ""],
                relationship="friend",
                budget=100.0,
                occasion="birthday"
            )
        errors = exc_info.value.errors()
        assert any("hobbies" in str(error["loc"]) for error in errors)
    
    def test_json_serialization_round_trip(self, sample_user_profile):
        """Test profile can be serialized and deserialized"""
        profile = UserProfile(**sample_user_profile)
        json_str = profile.model_dump_json()
        restored_profile = UserProfile.model_validate_json(json_str)
        
        assert restored_profile.age == profile.age
        assert restored_profile.hobbies == profile.hobbies
        assert restored_profile.budget == profile.budget
        assert restored_profile.relationship == profile.relationship


class TestGiftItem:
    """Test GiftItem model validation"""
    
    def test_valid_gift_item(self, sample_gift_item):
        """Test creating a valid gift item"""
        gift = GiftItem(**sample_gift_item)
        assert gift.id == "12345"
        assert gift.price == 299.99
        assert gift.rating == 4.5
    
    def test_rating_validation_minimum(self, sample_gift_item):
        """Test rating cannot be negative"""
        sample_gift_item["rating"] = -1.0
        with pytest.raises(ValidationError) as exc_info:
            GiftItem(**sample_gift_item)
        errors = exc_info.value.errors()
        assert any("rating" in str(error["loc"]) for error in errors)
    
    def test_rating_validation_maximum(self, sample_gift_item):
        """Test rating cannot exceed 5"""
        sample_gift_item["rating"] = 6.0
        with pytest.raises(ValidationError) as exc_info:
            GiftItem(**sample_gift_item)
        errors = exc_info.value.errors()
        assert any("rating" in str(error["loc"]) for error in errors)
    
    def test_price_validation_negative(self, sample_gift_item):
        """Test price cannot be negative"""
        sample_gift_item["price"] = -10.0
        with pytest.raises(ValidationError) as exc_info:
            GiftItem(**sample_gift_item)
        errors = exc_info.value.errors()
        assert any("price" in str(error["loc"]) for error in errors)
    
    def test_url_validation_invalid_image_url(self, sample_gift_item):
        """Test image URL must be valid"""
        sample_gift_item["image_url"] = "not-a-valid-url"
        with pytest.raises(ValidationError) as exc_info:
            GiftItem(**sample_gift_item)
        errors = exc_info.value.errors()
        assert any("image_url" in str(error["loc"]) for error in errors)


class TestRecommendationRequest:
    """Test RecommendationRequest model validation"""
    
    def test_valid_recommendation_request(self, sample_user_profile):
        """Test creating a valid recommendation request"""
        request = RecommendationRequest(
            user_profile=UserProfile(**sample_user_profile),
            max_recommendations=5,
            use_cache=True
        )
        assert request.max_recommendations == 5
        assert request.use_cache is True
    
    def test_max_recommendations_default(self, sample_user_profile):
        """Test max_recommendations has default value"""
        request = RecommendationRequest(
            user_profile=UserProfile(**sample_user_profile)
        )
        assert request.max_recommendations == 5
    
    def test_max_recommendations_validation_minimum(self, sample_user_profile):
        """Test max_recommendations must be at least 1"""
        with pytest.raises(ValidationError) as exc_info:
            RecommendationRequest(
                user_profile=UserProfile(**sample_user_profile),
                max_recommendations=0
            )
        errors = exc_info.value.errors()
        assert any("max_recommendations" in str(error["loc"]) for error in errors)
    
    def test_max_recommendations_validation_maximum(self, sample_user_profile):
        """Test max_recommendations cannot exceed 20"""
        with pytest.raises(ValidationError) as exc_info:
            RecommendationRequest(
                user_profile=UserProfile(**sample_user_profile),
                max_recommendations=21
            )
        errors = exc_info.value.errors()
        assert any("max_recommendations" in str(error["loc"]) for error in errors)
