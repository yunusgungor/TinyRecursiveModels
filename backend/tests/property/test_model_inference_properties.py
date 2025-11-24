"""
Property-based tests for model inference service

Feature: trendyol-gift-recommendation-web
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import assume
import json

from app.models.schemas import UserProfile


# Strategy for generating valid user profiles
@st.composite
def user_profile_strategy(draw):
    """Generate valid user profile for testing"""
    age = draw(st.integers(min_value=18, max_value=100))
    
    # Generate hobbies (1-10 non-empty strings)
    num_hobbies = draw(st.integers(min_value=1, max_value=10))
    hobbies = [
        draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            min_codepoint=65, max_codepoint=122
        ))) for _ in range(num_hobbies)
    ]
    
    relationship = draw(st.sampled_from([
        "mother", "father", "friend", "partner", "sibling", "colleague"
    ]))
    
    budget = draw(st.floats(min_value=0.01, max_value=100000, allow_nan=False, allow_infinity=False))
    
    occasion = draw(st.sampled_from([
        "birthday", "christmas", "anniversary", "graduation", "wedding"
    ]))
    
    # Generate personality traits (0-5 strings)
    num_traits = draw(st.integers(min_value=0, max_value=5))
    personality_traits = [
        draw(st.sampled_from([
            "practical", "eco-friendly", "creative", "tech-savvy", "traditional"
        ])) for _ in range(num_traits)
    ]
    
    return UserProfile(
        age=age,
        hobbies=hobbies,
        relationship=relationship,
        budget=budget,
        occasion=occasion,
        personality_traits=personality_traits
    )


class TestProfileEncodingRoundTrip:
    """
    Property 5: Profile JSON Serialization
    Validates: Requirements 1.6
    """
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None)
    def test_profile_json_serialization_round_trip(self, profile: UserProfile):
        """
        Feature: trendyol-gift-recommendation-web, Property 5: Profile JSON Serialization
        
        For any user profile, saving and then loading the profile should produce 
        an equivalent profile object (round-trip property)
        """
        # Serialize to JSON
        json_str = profile.model_dump_json()
        
        # Deserialize from JSON
        restored_profile = UserProfile.model_validate_json(json_str)
        
        # Verify round-trip: all fields should match
        assert restored_profile.age == profile.age, \
            f"Age mismatch: {restored_profile.age} != {profile.age}"
        
        assert restored_profile.hobbies == profile.hobbies, \
            f"Hobbies mismatch: {restored_profile.hobbies} != {profile.hobbies}"
        
        assert restored_profile.relationship == profile.relationship, \
            f"Relationship mismatch: {restored_profile.relationship} != {profile.relationship}"
        
        # Budget comparison with small tolerance for floating point
        assert abs(restored_profile.budget - profile.budget) < 0.01, \
            f"Budget mismatch: {restored_profile.budget} != {profile.budget}"
        
        assert restored_profile.occasion == profile.occasion, \
            f"Occasion mismatch: {restored_profile.occasion} != {profile.occasion}"
        
        assert restored_profile.personality_traits == profile.personality_traits, \
            f"Personality traits mismatch: {restored_profile.personality_traits} != {profile.personality_traits}"
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None)
    def test_profile_dict_serialization_round_trip(self, profile: UserProfile):
        """
        Test round-trip through dictionary format
        
        For any user profile, converting to dict and back should preserve all data
        """
        # Convert to dict
        profile_dict = profile.model_dump()
        
        # Convert back from dict
        restored_profile = UserProfile(**profile_dict)
        
        # Verify all fields match
        assert restored_profile.age == profile.age
        assert restored_profile.hobbies == profile.hobbies
        assert restored_profile.relationship == profile.relationship
        assert abs(restored_profile.budget - profile.budget) < 0.01
        assert restored_profile.occasion == profile.occasion
        assert restored_profile.personality_traits == profile.personality_traits
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None)
    def test_profile_validation_preserves_data(self, profile: UserProfile):
        """
        Test that validation doesn't modify data
        
        For any valid profile, re-validating should produce identical data
        """
        # Get original data
        original_dict = profile.model_dump()
        
        # Re-validate
        revalidated = UserProfile(**original_dict)
        revalidated_dict = revalidated.model_dump()
        
        # Should be identical
        assert original_dict == revalidated_dict
    
    def test_profile_with_empty_personality_traits(self):
        """Test edge case: empty personality traits list"""
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert restored.personality_traits == []
    
    def test_profile_with_minimum_age(self):
        """Test edge case: minimum age boundary"""
        profile = UserProfile(
            age=18,
            hobbies=["gaming"],
            relationship="friend",
            budget=50.0,
            occasion="birthday",
            personality_traits=["tech-savvy"]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert restored.age == 18
    
    def test_profile_with_maximum_age(self):
        """Test edge case: maximum age boundary"""
        profile = UserProfile(
            age=100,
            hobbies=["gardening"],
            relationship="grandparent",
            budget=200.0,
            occasion="birthday",
            personality_traits=["traditional"]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert restored.age == 100
    
    def test_profile_with_maximum_hobbies(self):
        """Test edge case: maximum number of hobbies"""
        hobbies = [f"hobby{i}" for i in range(10)]
        
        profile = UserProfile(
            age=30,
            hobbies=hobbies,
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=["creative"]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert len(restored.hobbies) == 10
        assert restored.hobbies == hobbies
    
    def test_profile_with_maximum_personality_traits(self):
        """Test edge case: maximum number of personality traits"""
        traits = ["practical", "creative", "tech-savvy", "traditional", "eco-friendly"]
        
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=traits
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert len(restored.personality_traits) == 5
        assert restored.personality_traits == traits
    
    def test_profile_with_very_large_budget(self):
        """Test edge case: very large budget value"""
        profile = UserProfile(
            age=50,
            hobbies=["luxury"],
            relationship="partner",
            budget=99999.99,
            occasion="anniversary",
            personality_traits=["luxury"]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert abs(restored.budget - 99999.99) < 0.01
    
    def test_profile_with_very_small_budget(self):
        """Test edge case: very small budget value"""
        profile = UserProfile(
            age=20,
            hobbies=["student"],
            relationship="friend",
            budget=0.01,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        json_str = profile.model_dump_json()
        restored = UserProfile.model_validate_json(json_str)
        
        assert abs(restored.budget - 0.01) < 0.001



class TestDeviceSelection:
    """
    Property 7: Device Selection Based on Availability
    Validates: Requirements 2.4
    """
    
    @pytest.fixture
    def mock_cuda_available(self, monkeypatch):
        """Mock torch.cuda.is_available to return True"""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        return True
    
    @pytest.fixture
    def mock_cuda_unavailable(self, monkeypatch):
        """Mock torch.cuda.is_available to return False"""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        return False
    
    def test_device_selection_with_cuda_available(self, mock_cuda_available, monkeypatch):
        """
        Feature: trendyol-gift-recommendation-web, Property 7: Device Selection Based on Availability
        
        For any inference request, the system should use GPU if available
        """
        import torch
        from app.services.model_inference import ModelInferenceService
        from app.core.config import settings
        
        # Set config to prefer CUDA
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cuda")
        
        # Create service
        service = ModelInferenceService()
        
        # Verify device selection
        assert service.device.type == "cuda", \
            f"Expected CUDA device when available, got {service.device.type}"
    
    def test_device_selection_with_cuda_unavailable(self, mock_cuda_unavailable, monkeypatch):
        """
        Feature: trendyol-gift-recommendation-web, Property 7: Device Selection Based on Availability
        
        For any inference request, the system should use CPU if GPU is not available
        """
        import torch
        from app.services.model_inference import ModelInferenceService
        from app.core.config import settings
        
        # Set config to prefer CUDA
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cuda")
        
        # Create service
        service = ModelInferenceService()
        
        # Verify device selection falls back to CPU
        assert service.device.type == "cpu", \
            f"Expected CPU device when CUDA unavailable, got {service.device.type}"
    
    def test_device_selection_with_cpu_configured(self, monkeypatch):
        """
        Test that CPU is used when explicitly configured, even if CUDA is available
        """
        import torch
        from app.services.model_inference import ModelInferenceService
        from app.core.config import settings
        
        # Set config to use CPU
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cpu")
        
        # Create service
        service = ModelInferenceService()
        
        # Verify CPU is used
        assert service.device.type == "cpu", \
            f"Expected CPU device when configured, got {service.device.type}"
    
    @given(
        device_config=st.sampled_from(["cuda", "cpu"]),
        cuda_available=st.booleans()
    )
    @settings(
        max_examples=100, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_device_selection_property(self, device_config, cuda_available, monkeypatch):
        """
        Property test: Device selection logic
        
        For any configuration and CUDA availability, device selection should follow rules:
        - If config is "cuda" and CUDA available -> use CUDA
        - If config is "cuda" and CUDA unavailable -> use CPU
        - If config is "cpu" -> always use CPU
        """
        import torch
        from unittest.mock import patch
        from app.services.model_inference import ModelInferenceService
        
        # Patch settings before creating service
        monkeypatch.setattr("app.core.config.settings.MODEL_DEVICE", device_config)
        
        # Mock CUDA availability
        with patch('torch.cuda.is_available', return_value=cuda_available):
            # Create service
            service = ModelInferenceService()
            
            # Verify device selection logic
            if device_config == "cuda" and cuda_available:
                expected_device = "cuda"
            else:
                expected_device = "cpu"
            
            assert service.device.type == expected_device, \
                f"Device selection failed: config={device_config}, cuda_available={cuda_available}, " \
                f"expected={expected_device}, got={service.device.type}"
    
    def test_get_device_info_with_cuda(self, mock_cuda_available, monkeypatch):
        """Test device info when CUDA is available"""
        import torch
        from app.services.model_inference import ModelInferenceService
        from app.core.config import settings
        
        # Mock CUDA device info
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda x: "Tesla V100")
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda x: 1024 * 1024 * 100)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda x: 1024 * 1024 * 200)
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cuda")
        
        service = ModelInferenceService()
        info = service.get_device_info()
        
        assert info["cuda_available"] is True
        assert "cuda_device_count" in info
        assert "cuda_device_name" in info
        assert "cuda_memory_allocated" in info
        assert "cuda_memory_reserved" in info
    
    def test_get_device_info_without_cuda(self, mock_cuda_unavailable, monkeypatch):
        """Test device info when CUDA is not available"""
        from app.services.model_inference import ModelInferenceService
        from app.core.config import settings
        
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cpu")
        
        service = ModelInferenceService()
        info = service.get_device_info()
        
        assert info["cuda_available"] is False
        assert info["device"] == "cpu"
        assert "cuda_device_count" not in info



class MockGiftItem:
    """Mock gift item for testing"""
    def __init__(self, id, name, category, price, rating, tags, description, age_suitability, occasion_fit):
        self.id = id
        self.name = name
        self.category = category
        self.price = price
        self.rating = rating
        self.tags = tags
        self.description = description
        self.age_suitability = age_suitability
        self.occasion_fit = occasion_fit


class TestModelOutputTransformation:
    """
    Property 8: Model Output to Recommendations Transformation
    Validates: Requirements 2.5
    """
    
    @st.composite
    def model_output_strategy(draw):
        """Generate valid model output for testing"""
        # Generate random number of gifts (1-10)
        num_gifts = draw(st.integers(min_value=1, max_value=10))
        
        # Generate gift items
        selected_gifts = []
        confidence_scores = []
        
        for i in range(num_gifts):
            gift = MockGiftItem(
                id=f"gift_{i}",
                name=draw(st.text(min_size=5, max_size=50, alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'),
                    min_codepoint=32, max_codepoint=126
                ))),
                category=draw(st.sampled_from(["technology", "home", "beauty", "sports"])),
                price=draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
                rating=draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
                tags=[draw(st.text(min_size=3, max_size=20, alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll'),
                    min_codepoint=65, max_codepoint=122
                ))) for _ in range(draw(st.integers(min_value=1, max_value=5)))],
                description=draw(st.text(min_size=10, max_size=100, alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'),
                    min_codepoint=32, max_codepoint=126
                ))),
                age_suitability=(
                    draw(st.integers(min_value=18, max_value=50)),
                    draw(st.integers(min_value=51, max_value=100))
                ),
                occasion_fit=[draw(st.sampled_from(["birthday", "christmas", "anniversary"]))]
            )
            selected_gifts.append(gift)
            
            # Confidence scores should be between 0 and 1
            confidence_scores.append(draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)))
        
        # Create model output
        model_output = {
            "action": {
                "selected_gifts": selected_gifts,
                "confidence_scores": confidence_scores,
                "recommendations": [g.id for g in selected_gifts],
                "action_indices": list(range(num_gifts))
            },
            "tool_results": {
                "price_comparison": {"best_price": 100.0},
                "review_analysis": {"sentiment": 0.8}
            },
            "confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        }
        
        return model_output, selected_gifts
    
    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_model_output_transformation_property(self, data):
        """
        Feature: trendyol-gift-recommendation-web, Property 8: Model Output to Recommendations Transformation
        
        For any model output tensor, the transformation should produce a list of 
        recommendations with valid confidence scores (0-1 range)
        """
        from app.services.model_inference import ModelInferenceService
        from app.models.schemas import GiftItem
        
        # Generate model output
        model_output, model_gifts = data.draw(TestModelOutputTransformation.model_output_strategy())
        
        # Create API gift items
        api_gifts = []
        for gift in model_gifts:
            api_gift = GiftItem(
                id=gift.id,
                name=gift.name,
                category=gift.category,
                price=gift.price,
                rating=gift.rating,
                image_url=f"https://cdn.trendyol.com/{gift.id}.jpg",
                trendyol_url=f"https://www.trendyol.com/product/{gift.id}",
                description=gift.description,
                tags=gift.tags,
                age_suitability=gift.age_suitability,
                occasion_fit=gift.occasion_fit,
                in_stock=True
            )
            api_gifts.append(api_gift)
        
        # Create service and decode output
        service = ModelInferenceService()
        recommendations = service._decode_model_output(model_output, api_gifts)
        
        # Verify properties
        assert len(recommendations) == len(model_gifts), \
            f"Number of recommendations should match number of gifts"
        
        for i, rec in enumerate(recommendations):
            # Confidence score should be in valid range [0, 1]
            assert 0.0 <= rec.confidence_score <= 1.0, \
                f"Confidence score {rec.confidence_score} not in range [0, 1]"
            
            # Rank should be sequential starting from 1
            assert rec.rank == i + 1, \
                f"Rank should be {i + 1}, got {rec.rank}"
            
            # Gift should have all required fields
            assert rec.gift.id is not None
            assert rec.gift.name is not None
            assert rec.gift.category is not None
            assert rec.gift.price >= 0
            assert 0 <= rec.gift.rating <= 5
            
            # Reasoning should be non-empty
            assert len(rec.reasoning) > 0, \
                "Reasoning list should not be empty"
    
    def test_empty_model_output(self):
        """Test edge case: empty model output"""
        from app.services.model_inference import ModelInferenceService
        
        model_output = {
            "action": {
                "selected_gifts": [],
                "confidence_scores": [],
                "recommendations": [],
                "action_indices": []
            },
            "tool_results": {},
            "confidence": 0.5
        }
        
        service = ModelInferenceService()
        recommendations = service._decode_model_output(model_output, [])
        
        assert len(recommendations) == 0
    
    def test_single_recommendation(self):
        """Test edge case: single recommendation"""
        from app.services.model_inference import ModelInferenceService
        from app.models.schemas import GiftItem
        
        model_gift = MockGiftItem(
            id="test_1",
            name="Test Gift",
            category="technology",
            price=100.0,
            rating=4.5,
            tags=["tech"],
            description="Test description",
            age_suitability=(18, 65),
            occasion_fit=["birthday"]
        )
        
        model_output = {
            "action": {
                "selected_gifts": [model_gift],
                "confidence_scores": [0.95],
                "recommendations": ["test_1"],
                "action_indices": [0]
            },
            "tool_results": {},
            "confidence": 0.95
        }
        
        api_gift = GiftItem(
            id="test_1",
            name="Test Gift",
            category="technology",
            price=100.0,
            rating=4.5,
            image_url="https://cdn.trendyol.com/test_1.jpg",
            trendyol_url="https://www.trendyol.com/product/test_1",
            description="Test description",
            tags=["tech"],
            age_suitability=(18, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        service = ModelInferenceService()
        recommendations = service._decode_model_output(model_output, [api_gift])
        
        assert len(recommendations) == 1
        assert recommendations[0].confidence_score == 0.95
        assert recommendations[0].rank == 1
    
    def test_confidence_score_boundaries(self):
        """Test edge cases: confidence scores at boundaries"""
        from app.services.model_inference import ModelInferenceService
        from app.models.schemas import GiftItem
        
        # Test with confidence score = 0.0
        model_gift_low = MockGiftItem(
            id="low_conf",
            name="Low Confidence Gift",
            category="home",
            price=50.0,
            rating=3.0,
            tags=["basic"],
            description="Low confidence",
            age_suitability=(18, 100),
            occasion_fit=["any"]
        )
        
        # Test with confidence score = 1.0
        model_gift_high = MockGiftItem(
            id="high_conf",
            name="High Confidence Gift",
            category="technology",
            price=200.0,
            rating=5.0,
            tags=["premium"],
            description="High confidence",
            age_suitability=(25, 50),
            occasion_fit=["birthday"]
        )
        
        model_output = {
            "action": {
                "selected_gifts": [model_gift_low, model_gift_high],
                "confidence_scores": [0.0, 1.0],
                "recommendations": ["low_conf", "high_conf"],
                "action_indices": [0, 1]
            },
            "tool_results": {},
            "confidence": 0.5
        }
        
        api_gifts = [
            GiftItem(
                id="low_conf",
                name="Low Confidence Gift",
                category="home",
                price=50.0,
                rating=3.0,
                image_url="https://cdn.trendyol.com/low_conf.jpg",
                trendyol_url="https://www.trendyol.com/product/low_conf",
                description="Low confidence",
                tags=["basic"],
                age_suitability=(18, 100),
                occasion_fit=["any"],
                in_stock=True
            ),
            GiftItem(
                id="high_conf",
                name="High Confidence Gift",
                category="technology",
                price=200.0,
                rating=5.0,
                image_url="https://cdn.trendyol.com/high_conf.jpg",
                trendyol_url="https://www.trendyol.com/product/high_conf",
                description="High confidence",
                tags=["premium"],
                age_suitability=(25, 50),
                occasion_fit=["birthday"],
                in_stock=True
            )
        ]
        
        service = ModelInferenceService()
        recommendations = service._decode_model_output(model_output, api_gifts)
        
        assert len(recommendations) == 2
        assert recommendations[0].confidence_score == 0.0
        assert recommendations[1].confidence_score == 1.0
        assert all(0.0 <= rec.confidence_score <= 1.0 for rec in recommendations)
