"""
Unit tests for model inference service

Tests model loading, timeout handling, and error handling
Requirements: 2.1, 2.2, 2.6
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import asyncio

from app.services.model_inference import ModelInferenceService, get_model_service
from app.core.exceptions import ModelLoadError, ModelInferenceError
from app.models.schemas import UserProfile, GiftItem


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_model_service_initialization(self):
        """Test that service initializes correctly"""
        service = ModelInferenceService()
        
        assert service.model is None
        assert service.model_loaded is False
        assert service.device is not None
    
    def test_model_load_file_not_found(self):
        """Test error handling when checkpoint file doesn't exist"""
        service = ModelInferenceService(checkpoint_path="nonexistent/path.pt")
        
        with pytest.raises(ModelLoadError) as exc_info:
            service.load_model()
        
        assert "not found" in str(exc_info.value).lower()
    
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    def test_model_load_missing_state_dict(self, mock_exists, mock_torch_load):
        """Test error handling when checkpoint is missing state dict"""
        mock_exists.return_value = True
        mock_torch_load.return_value = {
            "config": {"batch_size": 1}
            # Missing model_state_dict
        }
        
        service = ModelInferenceService()
        
        # Mock the models module to avoid import error
        with patch.dict('sys.modules', {'models': MagicMock(), 'models.tools': MagicMock(), 'models.tools.integrated_enhanced_trm': MagicMock()}):
            import sys
            mock_model_class = MagicMock()
            sys.modules['models.tools.integrated_enhanced_trm'].IntegratedEnhancedTRM = mock_model_class
            
            with pytest.raises(ModelLoadError) as exc_info:
                service.load_model()
            
            assert "state dict not found" in str(exc_info.value).lower()
    
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    def test_model_load_with_config(self, mock_exists, mock_torch_load):
        """Test model loading with config in checkpoint"""
        mock_exists.return_value = True
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        
        mock_torch_load.return_value = {
            "config": {
                "batch_size": 1,
                "seq_len": 50,
                "vocab_size": 1000,
                "num_puzzle_identifiers": 1,
                "hidden_size": 256,
                "H_cycles": 2,
                "L_cycles": 3,
                "L_layers": 2,
                "num_heads": 8,
                "expansion": 2.0,
                "pos_encodings": "rope",
                "halt_max_steps": 5,
                "halt_exploration_prob": 0.1,
                "action_space_size": 100,
                "max_recommendations": 5,
            },
            "model_state_dict": {}
        }
        
        # Mock the models module
        with patch.dict('sys.modules', {'models': MagicMock(), 'models.tools': MagicMock(), 'models.tools.integrated_enhanced_trm': MagicMock()}):
            import sys
            mock_model_class = MagicMock(return_value=mock_model)
            sys.modules['models.tools.integrated_enhanced_trm'].IntegratedEnhancedTRM = mock_model_class
            
            service = ModelInferenceService()
            service.load_model()
        
        assert service.model_loaded is True
        assert service.model is not None
    
    def test_is_loaded_returns_false_initially(self):
        """Test that is_loaded returns False before loading"""
        service = ModelInferenceService()
        assert service.is_loaded() is False
    
    def test_get_device_info(self):
        """Test device info retrieval"""
        service = ModelInferenceService()
        info = service.get_device_info()
        
        assert "device" in info
        assert "cuda_available" in info
        assert isinstance(info["cuda_available"], bool)


class TestTimeoutHandling:
    """Test timeout handling in inference"""
    
    @pytest.mark.asyncio
    async def test_inference_timeout(self):
        """Test that inference times out correctly"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock a slow inference that takes longer than timeout
        async def slow_inference(*args, **kwargs):
            await asyncio.sleep(10)  # Sleep longer than timeout
            return [], {}
        
        with patch.object(service, '_run_inference', side_effect=slow_inference):
            profile = UserProfile(
                age=30,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            with pytest.raises(ModelInferenceError) as exc_info:
                await service.generate_recommendations(
                    profile,
                    [],
                    timeout=0.1  # Very short timeout
                )
            
            assert "timed out" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_inference_uses_default_timeout(self):
        """Test that inference uses default timeout from settings"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock successful inference
        async def quick_inference(*args, **kwargs):
            return [], {}
        
        with patch.object(service, '_run_inference', side_effect=quick_inference):
            profile = UserProfile(
                age=30,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Should not raise timeout error
            recommendations, tool_results = await service.generate_recommendations(
                profile,
                []
            )
            
            assert recommendations == []
            assert tool_results == {}


class TestErrorHandling:
    """Test error handling in inference service"""
    
    def test_encode_profile_without_loaded_model(self):
        """Test that encoding fails when model is not loaded"""
        service = ModelInferenceService()
        
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        with pytest.raises(ModelInferenceError) as exc_info:
            service._encode_user_profile(profile)
        
        assert "not loaded" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_without_loaded_model(self):
        """Test that generation fails when model is not loaded"""
        service = ModelInferenceService()
        
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        with pytest.raises(ModelInferenceError) as exc_info:
            await service.generate_recommendations(profile, [])
        
        assert "not loaded" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_inference_error_handling(self):
        """Test that inference errors are properly caught and wrapped"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock inference that raises an error
        async def failing_inference(*args, **kwargs):
            raise RuntimeError("Model inference failed")
        
        with patch.object(service, '_run_inference', side_effect=failing_inference):
            profile = UserProfile(
                age=30,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            with pytest.raises(ModelInferenceError) as exc_info:
                await service.generate_recommendations(profile, [])
            
            assert "failed" in str(exc_info.value).lower()
    
    def test_decode_output_with_empty_action(self):
        """Test decoding with empty action result"""
        service = ModelInferenceService()
        
        model_output = {
            "action": {
                "selected_gifts": [],
                "confidence_scores": []
            },
            "tool_results": {}
        }
        
        recommendations = service._decode_model_output(model_output, [])
        assert len(recommendations) == 0
    
    def test_decode_output_error_handling(self):
        """Test error handling in output decoding"""
        service = ModelInferenceService()
        
        # Invalid model output (missing required fields)
        model_output = {
            "action": None  # Invalid action
        }
        
        with pytest.raises((ModelInferenceError, AttributeError, KeyError)):
            service._decode_model_output(model_output, [])


class TestSingletonPattern:
    """Test singleton pattern for model service"""
    
    def test_get_model_service_returns_singleton(self):
        """Test that get_model_service returns the same instance"""
        service1 = get_model_service()
        service2 = get_model_service()
        
        assert service1 is service2
    
    def test_singleton_maintains_state(self):
        """Test that singleton maintains state across calls"""
        service1 = get_model_service()
        service1.test_attribute = "test_value"
        
        service2 = get_model_service()
        assert hasattr(service2, 'test_attribute')
        assert service2.test_attribute == "test_value"
        
        # Cleanup
        delattr(service2, 'test_attribute')


class TestDeviceSelection:
    """Test device selection logic"""
    
    def test_device_selection_respects_config(self, monkeypatch):
        """Test that device selection respects configuration"""
        from app.core.config import settings
        
        # Test CPU configuration
        monkeypatch.setattr(settings, "MODEL_DEVICE", "cpu")
        service = ModelInferenceService()
        assert service.device.type == "cpu"
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_fallback_to_cpu(self, mock_cuda):
        """Test that device falls back to CPU when CUDA unavailable"""
        from app.core.config import settings
        
        with patch.object(settings, 'MODEL_DEVICE', 'cuda'):
            service = ModelInferenceService()
            assert service.device.type == "cpu"


class TestUserProfileEncoding:
    """Test user profile encoding"""
    
    @patch('app.services.model_inference.UserProfile')
    def test_encode_profile_with_all_fields(self, mock_model_profile_class):
        """Test encoding profile with all fields populated"""
        service = ModelInferenceService()
        service.model_loaded = True
        
        # Mock model with encode_user_profile method
        mock_model = MagicMock()
        mock_model.encode_user_profile.return_value = torch.randn(1, 256)
        service.model = mock_model
        
        # Mock the model UserProfile class
        mock_model_profile = MagicMock()
        mock_model_profile_class.return_value = mock_model_profile
        
        profile = UserProfile(
            age=35,
            hobbies=["gardening", "cooking"],
            relationship="mother",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical", "eco-friendly"]
        )
        
        with patch('app.services.model_inference.UserProfile', mock_model_profile_class):
            # Mock the import
            with patch.dict('sys.modules', {'models': MagicMock(), 'models.rl': MagicMock(), 'models.rl.environment': MagicMock()}):
                import sys
                sys.modules['models.rl.environment'].UserProfile = mock_model_profile_class
                
                encoding = service._encode_user_profile(profile)
        
        assert encoding is not None
        assert isinstance(encoding, torch.Tensor)
        mock_model.encode_user_profile.assert_called_once()
    
    @patch('app.services.model_inference.UserProfile')
    def test_encode_profile_with_minimal_fields(self, mock_model_profile_class):
        """Test encoding profile with minimal required fields"""
        service = ModelInferenceService()
        service.model_loaded = True
        
        # Mock model
        mock_model = MagicMock()
        mock_model.encode_user_profile.return_value = torch.randn(1, 256)
        service.model = mock_model
        
        # Mock the model UserProfile class
        mock_model_profile = MagicMock()
        mock_model_profile_class.return_value = mock_model_profile
        
        profile = UserProfile(
            age=25,
            hobbies=["reading"],
            relationship="friend",
            budget=100.0,
            occasion="birthday",
            personality_traits=[]  # Empty list
        )
        
        with patch('app.services.model_inference.UserProfile', mock_model_profile_class):
            # Mock the import
            with patch.dict('sys.modules', {'models': MagicMock(), 'models.rl': MagicMock(), 'models.rl.environment': MagicMock()}):
                import sys
                sys.modules['models.rl.environment'].UserProfile = mock_model_profile_class
                
                encoding = service._encode_user_profile(profile)
        
        assert encoding is not None
        assert isinstance(encoding, torch.Tensor)


class TestReasoningServiceIntegration:
    """
    Test reasoning service integration in ModelInferenceService
    Requirements: 3.1, 10.1, 10.5
    """
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_with_reasoning_enabled(self):
        """Test that reasoning is generated when include_reasoning=True"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Create mock gift
        mock_gift = MagicMock()
        mock_gift.id = "123"
        mock_gift.name = "Test Gift"
        mock_gift.category = "Kitchen"
        mock_gift.price = 100.0
        mock_gift.rating = 4.5
        mock_gift.tags = ["cooking"]
        mock_gift.description = "Test description"
        mock_gift.age_suitability = (18, 100)
        mock_gift.occasion_fit = ["birthday"]
        
        # Mock recommendation
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123",
            tags=["cooking"],
            description="Test description",
            age_suitability=(18, 100),
            occasion_fit=["birthday"]
        )
        mock_recommendation.confidence_score = 0.85
        mock_recommendation.reasoning = []
        
        # Mock _run_inference to return recommendations with reasoning
        async def mock_run_inference(*args, **kwargs):
            return [mock_recommendation], {}, {"tool_selection": [], "category_matching": []}
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=True,
                reasoning_level="detailed"
            )
            
            assert recommendations is not None
            assert reasoning_trace is not None
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_with_reasoning_disabled(self):
        """Test that reasoning is not generated when include_reasoning=False"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock _run_inference to return recommendations without reasoning
        async def mock_run_inference(*args, **kwargs):
            return [], {}, None
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=False
            )
            
            assert reasoning_trace is None
    
    @pytest.mark.asyncio
    async def test_reasoning_level_basic(self):
        """Test reasoning generation with basic level"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock recommendation
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123",
            tags=["cooking"],
            description="Test description",
            age_suitability=(18, 100),
            occasion_fit=["birthday"]
        )
        mock_recommendation.confidence_score = 0.85
        mock_recommendation.reasoning = []
        
        # Mock _run_inference
        async def mock_run_inference(*args, **kwargs):
            reasoning_level = kwargs.get('reasoning_level', 'detailed')
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [],
                "confidence_explanation": None
            }
            if reasoning_level == "basic":
                # Basic level should have minimal reasoning
                pass
            return [mock_recommendation], {}, reasoning_trace
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=True,
                reasoning_level="basic"
            )
            
            assert recommendations is not None
            assert reasoning_trace is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_level_detailed(self):
        """Test reasoning generation with detailed level"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock recommendation
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123",
            tags=["cooking"],
            description="Test description",
            age_suitability=(18, 100),
            occasion_fit=["birthday"]
        )
        mock_recommendation.confidence_score = 0.85
        mock_recommendation.reasoning = []
        
        # Mock _run_inference
        async def mock_run_inference(*args, **kwargs):
            reasoning_level = kwargs.get('reasoning_level', 'detailed')
            reasoning_trace = {
                "tool_selection": [{"name": "price_comparison", "selected": True}],
                "category_matching": [{"category_name": "Kitchen", "score": 0.9}],
                "confidence_explanation": None
            }
            return [mock_recommendation], {}, reasoning_trace
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=True,
                reasoning_level="detailed"
            )
            
            assert recommendations is not None
            assert reasoning_trace is not None
            assert "tool_selection" in reasoning_trace
            assert "category_matching" in reasoning_trace
    
    @pytest.mark.asyncio
    async def test_reasoning_level_full(self):
        """Test reasoning generation with full level"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock recommendation
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123",
            tags=["cooking"],
            description="Test description",
            age_suitability=(18, 100),
            occasion_fit=["birthday"]
        )
        mock_recommendation.confidence_score = 0.85
        mock_recommendation.reasoning = []
        
        # Mock _run_inference
        async def mock_run_inference(*args, **kwargs):
            reasoning_level = kwargs.get('reasoning_level', 'detailed')
            reasoning_trace = {
                "tool_selection": [{"name": "price_comparison", "selected": True}],
                "category_matching": [{"category_name": "Kitchen", "score": 0.9}],
                "attention_weights": {
                    "user_features": {"hobbies": 0.5, "budget": 0.3, "age": 0.1, "occasion": 0.1},
                    "gift_features": {"category": 0.4, "price": 0.35, "rating": 0.25}
                },
                "thinking_steps": [
                    {"step": 1, "action": "Encode user", "result": "Done", "insight": "Strong cooking interest"}
                ],
                "confidence_explanation": None
            }
            return [mock_recommendation], {}, reasoning_trace
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=True,
                reasoning_level="full"
            )
            
            assert recommendations is not None
            assert reasoning_trace is not None
            assert "attention_weights" in reasoning_trace
            assert "thinking_steps" in reasoning_trace
    
    @pytest.mark.asyncio
    async def test_reasoning_generation_error_handling(self):
        """Test that reasoning generation errors are handled gracefully"""
        service = ModelInferenceService()
        service.model_loaded = True
        service.model = MagicMock()
        
        # Mock recommendation
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123",
            tags=["cooking"],
            description="Test description",
            age_suitability=(18, 100),
            occasion_fit=["birthday"]
        )
        mock_recommendation.confidence_score = 0.85
        mock_recommendation.reasoning = []
        
        # Mock _run_inference to simulate error in reasoning extraction
        async def mock_run_inference(*args, **kwargs):
            # Return recommendations but reasoning extraction will fail
            return [mock_recommendation], {}, None
        
        with patch.object(service, '_run_inference', side_effect=mock_run_inference):
            profile = UserProfile(
                age=30,
                hobbies=["cooking"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Should not raise error, should fall back gracefully
            recommendations, tool_results, reasoning_trace = await service.generate_recommendations(
                profile,
                [],
                include_reasoning=True
            )
            
            assert recommendations is not None
            # Reasoning trace may be None due to error, which is acceptable
    
    def test_extract_reasoning_trace_with_valid_data(self):
        """Test reasoning trace extraction with valid model output"""
        service = ModelInferenceService()
        
        model_output = {
            "reasoning_trace": {
                "tool_selection": {
                    "price_comparison": {
                        "selected": True,
                        "score": 0.85,
                        "confidence": 0.85,
                        "priority": 1,
                        "factors": {}
                    }
                },
                "category_matching": {
                    "Kitchen": {
                        "score": 0.9,
                        "reasons": ["User hobby: cooking"],
                        "feature_contributions": {"hobby_match": 0.9}
                    }
                }
            },
            "outputs": {}
        }
        
        profile = UserProfile(
            age=30,
            hobbies=["cooking"],
            relationship="friend",
            budget=200.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        mock_recommendation = MagicMock()
        mock_recommendation.gift = GiftItem(
            id="123",
            name="Test Gift",
            category="Kitchen",
            price=100.0,
            rating=4.5,
            image_url="https://example.com/image.jpg",
            trendyol_url="https://example.com/product/123"
        )
        mock_recommendation.confidence_score = 0.85
        
        reasoning_trace = service._extract_reasoning_trace(
            model_output,
            profile,
            [mock_recommendation],
            {},
            "detailed"
        )
        
        assert reasoning_trace is not None
        assert "tool_selection" in reasoning_trace
        assert "category_matching" in reasoning_trace
    
    def test_extract_reasoning_trace_with_empty_data(self):
        """Test reasoning trace extraction with empty model output"""
        service = ModelInferenceService()
        
        model_output = {
            "reasoning_trace": {},
            "outputs": {}
        }
        
        profile = UserProfile(
            age=30,
            hobbies=["cooking"],
            relationship="friend",
            budget=200.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        reasoning_trace = service._extract_reasoning_trace(
            model_output,
            profile,
            [],
            {},
            "detailed"
        )
        
        assert reasoning_trace is not None
        assert "tool_selection" in reasoning_trace
        assert "category_matching" in reasoning_trace
    
    def test_extract_reasoning_trace_error_handling(self):
        """Test reasoning trace extraction handles errors gracefully"""
        service = ModelInferenceService()
        
        # Invalid model output that will cause errors
        model_output = {
            "reasoning_trace": {
                "tool_selection": "invalid_data"  # Should be dict
            }
        }
        
        profile = UserProfile(
            age=30,
            hobbies=["cooking"],
            relationship="friend",
            budget=200.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        # Should not raise error, should return minimal trace
        reasoning_trace = service._extract_reasoning_trace(
            model_output,
            profile,
            [],
            {},
            "detailed"
        )
        
        assert reasoning_trace is not None
        assert "tool_selection" in reasoning_trace
