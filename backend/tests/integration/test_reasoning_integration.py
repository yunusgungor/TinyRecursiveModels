"""
Integration tests for reasoning functionality

Tests the complete flow from API request to response with reasoning:
- Full flow with various reasoning levels
- Various user profiles and gifts
- Error scenarios (model failures, missing data)
- Caching behavior with reasoning
- Backward compatibility with old clients
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import status

from app.models.schemas import (
    UserProfile,
    GiftItem,
    GiftRecommendation,
    EnhancedGiftRecommendation,
    RecommendationRequest
)


class TestReasoningFullFlow:
    """Integration tests for full reasoning flow from API to response"""
    
    @pytest.mark.asyncio
    async def test_full_flow_with_reasoning_basic(self, client, sample_user_profile):
        """
        Test full flow from API request to response with basic reasoning
        
        Validates: Requirements 10.1, 10.4, 10.5
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="gift123",
                name="Cooking Set",
                category="Kitchen",
                price=350.0,
                rating=4.7,
                image_url="https://cdn.trendyol.com/cooking.jpg",
                trendyol_url="https://www.trendyol.com/product/gift123",
                description="Professional cooking set",
                tags=["cooking", "kitchen", "gift"],
                age_suitability=(25, 65),
                occasion_fit=["birthday", "anniversary"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.88,
                reasoning=["Perfect match for cooking hobby", "Within budget"],
                tool_insights={"price_comparison": {"best_price": True}},
                rank=1,
                reasoning_trace=None
            )
            
            # Basic reasoning trace
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.88,
                    "level": "high",
                    "factors": {
                        "positive": ["Strong hobby match", "Good price point"],
                        "negative": []
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {"price_comparison": {"best_price": True}}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            # Setup cache service (cache miss)
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            # Setup Trendyol service
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "gift123"
            mock_product.name = "Cooking Set"
            mock_product.category = "Kitchen"
            mock_product.price = 350.0
            mock_product.rating = 4.7
            mock_product.image_url = "https://cdn.trendyol.com/cooking.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/gift123"
            mock_product.description = "Professional cooking set"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            # Make request with basic reasoning level
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=basic",
                json=request_data
            )
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "recommendations" in data
            assert "reasoning_trace" in data
            assert "tool_results" in data
            assert "inference_time" in data
            assert "cache_hit" in data
            
            # Verify reasoning trace structure for basic level
            trace = data["reasoning_trace"]
            assert trace is not None
            assert "confidence_explanation" in trace
            assert trace["confidence_explanation"]["score"] == 0.88
            assert trace["confidence_explanation"]["level"] == "high"
            
            # Basic level should not include detailed components
            assert len(trace.get("tool_selection", [])) == 0
            assert len(trace.get("category_matching", [])) == 0
            assert trace.get("attention_weights") is None
            assert len(trace.get("thinking_steps", [])) == 0
            
            # Verify recommendation
            assert len(data["recommendations"]) == 1
            rec = data["recommendations"][0]
            assert rec["gift"]["id"] == "gift123"
            assert rec["confidence_score"] == 0.88
            assert len(rec["reasoning"]) > 0

    
    @pytest.mark.asyncio
    async def test_full_flow_with_reasoning_detailed(self, client, sample_user_profile):
        """
        Test full flow with detailed reasoning level
        
        Validates: Requirements 10.1, 10.5
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="gift456",
                name="Garden Tools Set",
                category="Garden",
                price=450.0,
                rating=4.5,
                image_url="https://cdn.trendyol.com/garden.jpg",
                trendyol_url="https://www.trendyol.com/product/gift456",
                description="Complete garden tools",
                tags=["gardening", "outdoor", "gift"],
                age_suitability=(30, 70),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.82,
                reasoning=["Matches gardening hobby", "Age appropriate"],
                tool_insights={"review_analysis": {"average_rating": 4.5}},
                rank=1,
                reasoning_trace=None
            )
            
            # Detailed reasoning trace
            reasoning_trace = {
                "tool_selection": [
                    {
                        "name": "review_analysis",
                        "selected": True,
                        "score": 0.78,
                        "reason": "User values quality",
                        "confidence": 0.78,
                        "priority": 1,
                        "factors": {"quality_preference": 0.8}
                    },
                    {
                        "name": "price_comparison",
                        "selected": True,
                        "score": 0.85,
                        "reason": "Budget constraint active",
                        "confidence": 0.85,
                        "priority": 2,
                        "factors": {"budget_constraint": 0.9}
                    }
                ],
                "category_matching": [
                    {
                        "category_name": "Garden",
                        "score": 0.82,
                        "reasons": ["Hobby match: gardening (0.9)", "Age appropriate (0.8)"],
                        "feature_contributions": {"hobby_match": 0.5, "age_appropriateness": 0.3}
                    },
                    {
                        "category_name": "Kitchen",
                        "score": 0.75,
                        "reasons": ["Hobby match: cooking (0.8)"],
                        "feature_contributions": {"hobby_match": 0.6}
                    }
                ],
                "attention_weights": None,  # Not in detailed
                "thinking_steps": [],  # Not in detailed
                "confidence_explanation": {
                    "score": 0.82,
                    "level": "high",
                    "factors": {
                        "positive": ["Strong hobby match", "Good reviews"],
                        "negative": ["Slightly above typical budget"]
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {"review_analysis": {"average_rating": 4.5}}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            # Setup cache and Trendyol services
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "gift456"
            mock_product.name = "Garden Tools Set"
            mock_product.category = "Garden"
            mock_product.price = 450.0
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/garden.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/gift456"
            mock_product.description = "Complete garden tools"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            # Make request with detailed reasoning level
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            trace = data["reasoning_trace"]
            assert trace is not None
            
            # Verify detailed reasoning components
            assert len(trace["tool_selection"]) == 2
            assert trace["tool_selection"][0]["name"] == "review_analysis"
            assert trace["tool_selection"][0]["selected"] is True
            
            assert len(trace["category_matching"]) == 2
            assert trace["category_matching"][0]["category_name"] == "Garden"
            assert trace["category_matching"][0]["score"] == 0.82
            
            assert trace["confidence_explanation"] is not None
            
            # Detailed level should not include these
            assert trace.get("attention_weights") is None
            assert len(trace.get("thinking_steps", [])) == 0

    
    @pytest.mark.asyncio
    async def test_full_flow_with_reasoning_full(self, client, sample_user_profile):
        """
        Test full flow with full reasoning level (all components)
        
        Validates: Requirements 10.1, 10.3, 10.5
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            # Setup model service
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="gift789",
                name="Premium Coffee Maker",
                category="Kitchen",
                price=480.0,
                rating=4.8,
                image_url="https://cdn.trendyol.com/coffee.jpg",
                trendyol_url="https://www.trendyol.com/product/gift789",
                description="High-end coffee maker",
                tags=["coffee", "kitchen", "appliance"],
                age_suitability=(25, 65),
                occasion_fit=["birthday", "anniversary"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.91,
                reasoning=["Perfect for coffee lovers", "Premium quality", "Within budget"],
                tool_insights={
                    "review_analysis": {"average_rating": 4.8, "review_count": 250},
                    "trend_analysis": {"trending": True}
                },
                rank=1,
                reasoning_trace=None
            )
            
            # Full reasoning trace with all components
            reasoning_trace = {
                "tool_selection": [
                    {
                        "name": "review_analysis",
                        "selected": True,
                        "score": 0.88,
                        "reason": "User values quality and reviews",
                        "confidence": 0.88,
                        "priority": 1,
                        "factors": {"quality_preference": 0.85, "review_importance": 0.9}
                    },
                    {
                        "name": "price_comparison",
                        "selected": True,
                        "score": 0.82,
                        "reason": "Budget optimization needed",
                        "confidence": 0.82,
                        "priority": 2,
                        "factors": {"budget_constraint": 0.85}
                    },
                    {
                        "name": "trend_analysis",
                        "selected": True,
                        "score": 0.75,
                        "reason": "User interested in popular items",
                        "confidence": 0.75,
                        "priority": 3,
                        "factors": {"trend_preference": 0.7}
                    }
                ],
                "category_matching": [
                    {
                        "category_name": "Kitchen",
                        "score": 0.91,
                        "reasons": [
                            "Hobby match: cooking (0.95)",
                            "Occasion fit: birthday (0.9)",
                            "Age appropriate: 35 years (0.88)"
                        ],
                        "feature_contributions": {
                            "hobby_match": 0.50,
                            "occasion_fit": 0.30,
                            "age_appropriateness": 0.20
                        }
                    },
                    {
                        "category_name": "Garden",
                        "score": 0.78,
                        "reasons": ["Hobby match: gardening (0.85)"],
                        "feature_contributions": {"hobby_match": 0.60}
                    }
                ],
                "attention_weights": {
                    "user_features": {
                        "hobbies": 0.45,
                        "budget": 0.30,
                        "age": 0.15,
                        "occasion": 0.10
                    },
                    "gift_features": {
                        "category": 0.40,
                        "price": 0.35,
                        "rating": 0.25
                    }
                },
                "thinking_steps": [
                    {
                        "step": 1,
                        "action": "Encode user profile",
                        "result": "User encoding completed",
                        "insight": "Strong cooking interest detected with budget awareness"
                    },
                    {
                        "step": 2,
                        "action": "Match categories",
                        "result": "Top categories: Kitchen, Garden",
                        "insight": "Kitchen category shows strongest match (0.91)"
                    },
                    {
                        "step": 3,
                        "action": "Select tools",
                        "result": "Selected tools: review_analysis, price_comparison, trend_analysis",
                        "insight": "Chose 3 tools for comprehensive analysis"
                    },
                    {
                        "step": 4,
                        "action": "Execute tools",
                        "result": "Executed 3 tools successfully",
                        "insight": "Found highly rated items within budget"
                    },
                    {
                        "step": 5,
                        "action": "Rank gifts",
                        "result": "Ranked top 5 gifts",
                        "insight": "Selected gifts based on multi-criteria scoring"
                    }
                ],
                "confidence_explanation": {
                    "score": 0.91,
                    "level": "high",
                    "factors": {
                        "positive": [
                            "Excellent hobby match (0.95)",
                            "Outstanding reviews (4.8/5.0)",
                            "Currently trending",
                            "Within budget range"
                        ],
                        "negative": []
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], 
                             {"review_analysis": {"average_rating": 4.8}, "trend_analysis": {"trending": True}}, 
                             reasoning_trace)
            )
            mock_model.return_value = model_service
            
            # Setup cache and Trendyol services
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "gift789"
            mock_product.name = "Premium Coffee Maker"
            mock_product.category = "Kitchen"
            mock_product.price = 480.0
            mock_product.rating = 4.8
            mock_product.image_url = "https://cdn.trendyol.com/coffee.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/gift789"
            mock_product.description = "High-end coffee maker"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            # Make request with full reasoning level
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?include_reasoning=true&reasoning_level=full",
                json=request_data
            )
            
            # Assertions
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            trace = data["reasoning_trace"]
            assert trace is not None
            
            # Verify all reasoning components are present
            assert len(trace["tool_selection"]) == 3
            assert len(trace["category_matching"]) == 2
            assert trace["attention_weights"] is not None
            assert len(trace["thinking_steps"]) == 5
            assert trace["confidence_explanation"] is not None
            
            # Verify attention weights structure
            attn = trace["attention_weights"]
            assert "user_features" in attn
            assert "gift_features" in attn
            assert abs(sum(attn["user_features"].values()) - 1.0) < 0.01
            assert abs(sum(attn["gift_features"].values()) - 1.0) < 0.01
            
            # Verify thinking steps are chronological
            for i, step in enumerate(trace["thinking_steps"]):
                assert step["step"] == i + 1
                assert "action" in step
                assert "result" in step
                assert "insight" in step



class TestReasoningWithVariousProfiles:
    """Test reasoning with various user profiles and gifts"""
    
    @pytest.mark.asyncio
    async def test_reasoning_with_young_user(self, client):
        """
        Test reasoning with young user profile
        
        Validates: Requirements 3.5 (age appropriateness)
        """
        young_user_profile = {
            "age": 22,
            "hobbies": ["gaming", "music"],
            "relationship": "friend",
            "budget": 300.0,
            "occasion": "birthday",
            "personality_traits": ["tech-savvy", "energetic"]
        }
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="tech123",
                name="Gaming Headset",
                category="Electronics",
                price=250.0,
                rating=4.6,
                image_url="https://cdn.trendyol.com/headset.jpg",
                trendyol_url="https://www.trendyol.com/product/tech123",
                description="Professional gaming headset",
                tags=["gaming", "electronics", "audio"],
                age_suitability=(18, 40),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.89,
                reasoning=["Perfect for gaming hobby", "Age-appropriate for 22 years old"],
                tool_insights={},
                rank=1,
                reasoning_trace=None
            )
            
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [
                    {
                        "category_name": "Electronics",
                        "score": 0.89,
                        "reasons": ["Hobby match: gaming (0.95)", "Age appropriate: 22 years (0.9)"],
                        "feature_contributions": {"hobby_match": 0.6, "age_appropriateness": 0.3}
                    }
                ],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.89,
                    "level": "high",
                    "factors": {
                        "positive": ["Strong hobby match", "Age-appropriate"],
                        "negative": []
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "tech123"
            mock_product.name = "Gaming Headset"
            mock_product.category = "Electronics"
            mock_product.price = 250.0
            mock_product.rating = 4.6
            mock_product.image_url = "https://cdn.trendyol.com/headset.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/tech123"
            mock_product.description = "Professional gaming headset"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": young_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify age-appropriate reasoning
            rec = data["recommendations"][0]
            assert any("22 years" in reason or "age" in reason.lower() for reason in rec["reasoning"])
    
    @pytest.mark.asyncio
    async def test_reasoning_with_senior_user(self, client):
        """
        Test reasoning with senior user profile
        
        Validates: Requirements 3.5 (age appropriateness)
        """
        senior_user_profile = {
            "age": 68,
            "hobbies": ["reading", "gardening"],
            "relationship": "parent",
            "budget": 400.0,
            "occasion": "anniversary",
            "personality_traits": ["traditional", "practical"]
        }
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="book123",
                name="Classic Book Collection",
                category="Books",
                price=350.0,
                rating=4.9,
                image_url="https://cdn.trendyol.com/books.jpg",
                trendyol_url="https://www.trendyol.com/product/book123",
                description="Premium classic literature collection",
                tags=["books", "reading", "classic"],
                age_suitability=(40, 100),
                occasion_fit=["anniversary", "birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.92,
                reasoning=["Perfect for reading hobby", "Age-appropriate for 68 years old", "Premium quality"],
                tool_insights={},
                rank=1,
                reasoning_trace=None
            )
            
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [
                    {
                        "category_name": "Books",
                        "score": 0.92,
                        "reasons": ["Hobby match: reading (0.98)", "Age appropriate: 68 years (0.95)"],
                        "feature_contributions": {"hobby_match": 0.65, "age_appropriateness": 0.25}
                    }
                ],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.92,
                    "level": "high",
                    "factors": {
                        "positive": ["Excellent hobby match", "Age-appropriate", "High quality"],
                        "negative": []
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "book123"
            mock_product.name = "Classic Book Collection"
            mock_product.category = "Books"
            mock_product.price = 350.0
            mock_product.rating = 4.9
            mock_product.image_url = "https://cdn.trendyol.com/books.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/book123"
            mock_product.description = "Premium classic literature collection"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": senior_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify age-appropriate reasoning
            rec = data["recommendations"][0]
            assert any("68 years" in reason or "age" in reason.lower() for reason in rec["reasoning"])

    
    @pytest.mark.asyncio
    async def test_reasoning_with_tight_budget(self, client):
        """
        Test reasoning with tight budget constraint
        
        Validates: Requirements 3.3 (budget optimization)
        """
        budget_user_profile = {
            "age": 30,
            "hobbies": ["cooking"],
            "relationship": "friend",
            "budget": 150.0,  # Tight budget
            "occasion": "birthday",
            "personality_traits": ["practical"]
        }
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="budget123",
                name="Basic Cooking Utensils",
                category="Kitchen",
                price=99.0,  # Well within budget
                rating=4.3,
                image_url="https://cdn.trendyol.com/utensils.jpg",
                trendyol_url="https://www.trendyol.com/product/budget123",
                description="Essential cooking utensils",
                tags=["cooking", "kitchen", "basic"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Great value: Only 66% of your budget", "Matches cooking hobby"],
                tool_insights={"price_comparison": {"best_price": True}},
                rank=1,
                reasoning_trace=None
            )
            
            reasoning_trace = {
                "tool_selection": [
                    {
                        "name": "price_comparison",
                        "selected": True,
                        "score": 0.95,
                        "reason": "Tight budget constraint requires price optimization",
                        "confidence": 0.95,
                        "priority": 1,
                        "factors": {"budget_constraint": 0.98}
                    }
                ],
                "category_matching": [],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.85,
                    "level": "high",
                    "factors": {
                        "positive": ["Excellent value", "Within budget"],
                        "negative": []
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {"price_comparison": {"best_price": True}}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "budget123"
            mock_product.name = "Basic Cooking Utensils"
            mock_product.category = "Kitchen"
            mock_product.price = 99.0
            mock_product.rating = 4.3
            mock_product.image_url = "https://cdn.trendyol.com/utensils.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/budget123"
            mock_product.description = "Essential cooking utensils"
            mock_product.brand = "Test Brand"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": budget_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify budget reasoning
            rec = data["recommendations"][0]
            assert any("budget" in reason.lower() or "value" in reason.lower() for reason in rec["reasoning"])
            
            # Verify price_comparison tool was selected
            trace = data["reasoning_trace"]
            assert any(tool["name"] == "price_comparison" and tool["selected"] for tool in trace["tool_selection"])


class TestReasoningErrorScenarios:
    """Test reasoning with error scenarios"""
    
    @pytest.mark.asyncio
    async def test_reasoning_with_model_failure(self, client, sample_user_profile):
        """
        Test error handling when model fails during reasoning generation
        
        Validates: Requirements 7.1 (error handling)
        """
        from app.core.exceptions import ModelInferenceError
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            model_service.generate_recommendations = AsyncMock(
                side_effect=ModelInferenceError("Model inference failed")
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "test123"
            mock_product.name = "Test Gift"
            mock_product.category = "Test"
            mock_product.price = 100.0
            mock_product.rating = 4.0
            mock_product.image_url = "https://cdn.trendyol.com/test.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/test123"
            mock_product.description = "Test"
            mock_product.brand = "Test"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=MagicMock())
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=full",
                json=request_data
            )
            
            # Should return error
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "kullanılamıyor" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_reasoning_with_missing_data(self, client):
        """
        Test reasoning when some data is missing
        
        Validates: Requirements 7.1 (error handling)
        """
        incomplete_user_profile = {
            "age": 30,
            "hobbies": ["cooking"],
            "relationship": "friend",
            "budget": 200.0,
            "occasion": "birthday",
            "personality_traits": []  # Empty traits
        }
        
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="gift999",
                name="Test Gift",
                category="Kitchen",
                price=180.0,
                rating=4.5,
                image_url="https://cdn.trendyol.com/test.jpg",
                trendyol_url="https://www.trendyol.com/product/gift999",
                description="Test gift",
                tags=["cooking"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.75,
                reasoning=["Matches cooking hobby"],
                tool_insights={},
                rank=1,
                reasoning_trace=None
            )
            
            # Reasoning trace with limited data
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [
                    {
                        "category_name": "Kitchen",
                        "score": 0.75,
                        "reasons": ["Hobby match: cooking (0.8)"],
                        "feature_contributions": {"hobby_match": 0.8}
                    }
                ],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.75,
                    "level": "medium",
                    "factors": {
                        "positive": ["Hobby match"],
                        "negative": ["Limited profile information"]
                    }
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "gift999"
            mock_product.name = "Test Gift"
            mock_product.category = "Kitchen"
            mock_product.price = 180.0
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/test.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/gift999"
            mock_product.description = "Test gift"
            mock_product.brand = "Test"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": incomplete_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            # Should still succeed but with lower confidence
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            rec = data["recommendations"][0]
            assert rec["confidence_score"] <= 0.8  # Lower confidence due to missing data



class TestReasoningCachingBehavior:
    """Test caching behavior with reasoning"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_no_reasoning_trace(self, client, sample_user_profile):
        """
        Test that cached results don't include reasoning trace
        
        Validates: Requirements 6.1 (caching)
        """
        with patch('app.api.v1.recommendations.get_cache_service') as mock_cache:
            
            # Setup cache service with cached recommendations
            cache_service = AsyncMock()
            
            mock_gift = GiftItem(
                id="cached456",
                name="Cached Gift",
                category="Kitchen",
                price=300.0,
                rating=4.5,
                image_url="https://cdn.trendyol.com/cached.jpg",
                trendyol_url="https://www.trendyol.com/product/cached456",
                description="Cached gift",
                tags=["cooking"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            cached_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Cached reasoning"],
                tool_insights={},
                rank=1
            )
            
            cache_service.get_recommendations = AsyncMock(
                return_value=[cached_recommendation]
            )
            mock_cache.return_value = cache_service
            
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=full",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify cache hit
            assert data["cache_hit"] is True
            
            # Verify no reasoning trace for cached results
            assert data["reasoning_trace"] is None
            
            # But recommendations should still have basic reasoning
            assert len(data["recommendations"]) > 0
            assert len(data["recommendations"][0]["reasoning"]) > 0
    
    @pytest.mark.asyncio
    async def test_cache_miss_includes_reasoning(self, client, sample_user_profile):
        """
        Test that cache miss generates fresh reasoning
        
        Validates: Requirements 6.1 (caching), 10.1 (reasoning generation)
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="fresh789",
                name="Fresh Gift",
                category="Kitchen",
                price=350.0,
                rating=4.7,
                image_url="https://cdn.trendyol.com/fresh.jpg",
                trendyol_url="https://www.trendyol.com/product/fresh789",
                description="Fresh gift",
                tags=["cooking"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = EnhancedGiftRecommendation(
                gift=mock_gift,
                confidence_score=0.88,
                reasoning=["Fresh reasoning"],
                tool_insights={},
                rank=1,
                reasoning_trace=None
            )
            
            reasoning_trace = {
                "tool_selection": [
                    {
                        "name": "price_comparison",
                        "selected": True,
                        "score": 0.85,
                        "reason": "Budget optimization",
                        "confidence": 0.85,
                        "priority": 1,
                        "factors": {}
                    }
                ],
                "category_matching": [],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.88,
                    "level": "high",
                    "factors": {"positive": ["Good match"], "negative": []}
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            # Cache miss
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "fresh789"
            mock_product.name = "Fresh Gift"
            mock_product.category = "Kitchen"
            mock_product.price = 350.0
            mock_product.rating = 4.7
            mock_product.image_url = "https://cdn.trendyol.com/fresh.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/fresh789"
            mock_product.description = "Fresh gift"
            mock_product.brand = "Test"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": True
            }
            
            response = client.post(
                "/api/recommendations?reasoning_level=detailed",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify cache miss
            assert data["cache_hit"] is False
            
            # Verify reasoning trace is present
            assert data["reasoning_trace"] is not None
            assert len(data["reasoning_trace"]["tool_selection"]) > 0
            
            # Verify cache was updated
            cache_service.set_recommendations.assert_called_once()


class TestBackwardCompatibility:
    """Test backward compatibility with old clients"""
    
    @pytest.mark.asyncio
    async def test_old_client_without_reasoning_params(self, client, sample_user_profile):
        """
        Test that old clients without reasoning parameters still work
        
        Validates: Requirements 10.4 (backward compatibility)
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="compat123",
                name="Compatible Gift",
                category="Kitchen",
                price=300.0,
                rating=4.5,
                image_url="https://cdn.trendyol.com/compat.jpg",
                trendyol_url="https://www.trendyol.com/product/compat123",
                description="Compatible gift",
                tags=["cooking"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            # Old-style recommendation (without reasoning_trace)
            mock_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Basic reasoning"],
                tool_insights={},
                rank=1
            )
            
            # Default basic reasoning trace
            reasoning_trace = {
                "tool_selection": [],
                "category_matching": [],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": {
                    "score": 0.85,
                    "level": "high",
                    "factors": {"positive": ["Good match"], "negative": []}
                }
            }
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, reasoning_trace)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "compat123"
            mock_product.name = "Compatible Gift"
            mock_product.category = "Kitchen"
            mock_product.price = 300.0
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/compat.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/compat123"
            mock_product.description = "Compatible gift"
            mock_product.brand = "Test"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            # Old client request - no reasoning parameters
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post("/api/recommendations", json=request_data)
            
            # Should work with default behavior
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "recommendations" in data
            assert "tool_results" in data
            assert "inference_time" in data
            assert "cache_hit" in data
            
            # Should include basic reasoning by default
            assert "reasoning_trace" in data
            assert data["reasoning_trace"] is not None
            
            # Verify model was called with default reasoning level
            model_service.generate_recommendations.assert_called_once()
            call_kwargs = model_service.generate_recommendations.call_args[1]
            assert call_kwargs["include_reasoning"] is True
            assert call_kwargs["reasoning_level"] == "basic"  # Default from settings
    
    @pytest.mark.asyncio
    async def test_response_schema_compatibility(self, client, sample_user_profile):
        """
        Test that response schema is backward compatible
        
        Validates: Requirements 8.1 (schema compatibility)
        """
        with patch('app.api.v1.recommendations.get_model_service') as mock_model, \
             patch('app.api.v1.recommendations.get_cache_service') as mock_cache, \
             patch('app.api.v1.recommendations.get_trendyol_service') as mock_trendyol:
            
            model_service = MagicMock()
            model_service.is_loaded.return_value = True
            
            mock_gift = GiftItem(
                id="schema123",
                name="Schema Test Gift",
                category="Kitchen",
                price=300.0,
                rating=4.5,
                image_url="https://cdn.trendyol.com/schema.jpg",
                trendyol_url="https://www.trendyol.com/product/schema123",
                description="Schema test",
                tags=["test"],
                age_suitability=(18, 65),
                occasion_fit=["birthday"],
                in_stock=True
            )
            
            mock_recommendation = GiftRecommendation(
                gift=mock_gift,
                confidence_score=0.85,
                reasoning=["Test reasoning"],
                tool_insights={},
                rank=1
            )
            
            model_service.generate_recommendations = AsyncMock(
                return_value=([mock_recommendation], {}, None)
            )
            mock_model.return_value = model_service
            
            cache_service = AsyncMock()
            cache_service.get_recommendations = AsyncMock(return_value=None)
            cache_service.set_recommendations = AsyncMock()
            mock_cache.return_value = cache_service
            
            trendyol_service = MagicMock()
            mock_product = MagicMock()
            mock_product.id = "schema123"
            mock_product.name = "Schema Test Gift"
            mock_product.category = "Kitchen"
            mock_product.price = 300.0
            mock_product.rating = 4.5
            mock_product.image_url = "https://cdn.trendyol.com/schema.jpg"
            mock_product.product_url = "https://www.trendyol.com/product/schema123"
            mock_product.description = "Schema test"
            mock_product.brand = "Test"
            mock_product.in_stock = True
            
            trendyol_service.search_products = AsyncMock(return_value=[mock_product])
            trendyol_service.convert_to_gift_item = MagicMock(return_value=mock_gift)
            mock_trendyol.return_value = trendyol_service
            
            request_data = {
                "user_profile": sample_user_profile,
                "max_recommendations": 5,
                "use_cache": False
            }
            
            response = client.post(
                "/api/recommendations?include_reasoning=false",
                json=request_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            
            # Verify all required fields are present (backward compatible)
            assert "recommendations" in data
            assert "tool_results" in data
            assert "inference_time" in data
            assert "cache_hit" in data
            
            # New field is optional
            assert "reasoning_trace" in data
            
            # Verify recommendation structure
            rec = data["recommendations"][0]
            assert "gift" in rec
            assert "confidence_score" in rec
            assert "reasoning" in rec
            assert "tool_insights" in rec
            assert "rank" in rec
