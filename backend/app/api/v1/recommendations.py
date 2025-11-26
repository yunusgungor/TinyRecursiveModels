"""Recommendation endpoints"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    EnhancedRecommendationResponse,
    UserProfile,
    GiftItem,
    GiftRecommendation,
    EnhancedGiftRecommendation
)
from app.services.model_inference import get_model_service
from app.services.cache_service import get_cache_service
from app.services.trendyol_api import get_trendyol_service
from app.services.tool_orchestration import ToolOrchestrationService
from app.core.config import settings
from app.core.exceptions import (
    ModelInferenceError,
    TrendyolAPIError,
    CacheError
)
from models.tools import ToolRegistry


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommendations", response_model=EnhancedRecommendationResponse, response_model_by_alias=True)
async def get_recommendations(
    request: RecommendationRequest,
    include_reasoning: Optional[bool] = Query(
        default=None,
        description="Include reasoning trace in response. If not specified, defaults to configured default level."
    ),
    reasoning_level: Optional[str] = Query(
        default=None,
        description="Level of reasoning detail: 'basic', 'detailed', or 'full'",
        pattern="^(basic|detailed|full)$"
    )
) -> EnhancedRecommendationResponse:
    """
    Get gift recommendations based on user profile
    
    This endpoint:
    1. Validates user profile (done by Pydantic)
    2. Checks cache for existing recommendations
    3. Fetches available gifts from Trendyol
    4. Runs model inference if needed
    5. Executes relevant tools
    6. Returns ranked recommendations with optional reasoning
    
    Args:
        request: Recommendation request with user profile
        include_reasoning: Whether to include reasoning trace (default: basic reasoning)
        reasoning_level: Level of reasoning detail ('basic', 'detailed', 'full')
        
    Returns:
        EnhancedRecommendationResponse with recommendations and optional reasoning
        
    Raises:
        HTTPException: If recommendation generation fails or invalid parameters
    """
    start_time = time.time()
    cache_hit = False
    
    # Use default reasoning level from settings if not specified
    if reasoning_level is None:
        reasoning_level = settings.REASONING_DEFAULT_LEVEL
    
    # Validate reasoning_level parameter
    valid_levels = ["basic", "detailed", "full"]
    if reasoning_level not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid reasoning_level. Must be one of: {', '.join(valid_levels)}"
        )
    
    # Determine reasoning behavior based on feature flag
    # If include_reasoning is None (not specified), use configured default level
    # If include_reasoning is False, no reasoning
    # If include_reasoning is True, use specified reasoning_level
    # If REASONING_ENABLED is False, disable reasoning regardless of parameters
    if not settings.REASONING_ENABLED:
        # Feature flag disabled - no reasoning
        actual_include_reasoning = False
        actual_reasoning_level = "basic"  # Won't be used
    elif include_reasoning is None:
        # Default behavior: use specified reasoning_level (defaults to configured level)
        actual_include_reasoning = True
        actual_reasoning_level = reasoning_level
    elif include_reasoning is False:
        # Explicitly disabled
        actual_include_reasoning = False
        actual_reasoning_level = "basic"  # Won't be used
    else:
        # Explicitly enabled with specified level
        actual_include_reasoning = True
        actual_reasoning_level = reasoning_level
    
    logger.info(
        f"Recommendation request received: age={request.user_profile.age}, "
        f"budget={request.user_profile.budget}, occasion={request.user_profile.occasion}"
    )
    
    try:
        # Step 1: Check cache if enabled
        if request.use_cache:
            try:
                cache_service = await get_cache_service()
                cached_recommendations = await cache_service.get_recommendations(
                    request.user_profile
                )
                
                if cached_recommendations:
                    inference_time = time.time() - start_time
                    logger.info(
                        f"Cache hit! Returning {len(cached_recommendations)} "
                        f"cached recommendations in {inference_time:.2f}s"
                    )
                    
                    # Convert to EnhancedGiftRecommendation if needed
                    enhanced_recommendations = []
                    for rec in cached_recommendations[:request.max_recommendations]:
                        if isinstance(rec, EnhancedGiftRecommendation):
                            enhanced_recommendations.append(rec)
                        else:
                            # Convert GiftRecommendation to EnhancedGiftRecommendation
                            enhanced_rec = EnhancedGiftRecommendation(
                                gift=rec.gift,
                                confidence_score=rec.confidence_score,
                                reasoning=rec.reasoning,
                                tool_insights=rec.tool_insights,
                                rank=rec.rank,
                                reasoning_trace=None  # No reasoning trace for cached results
                            )
                            enhanced_recommendations.append(enhanced_rec)
                    
                    return EnhancedRecommendationResponse(
                        recommendations=enhanced_recommendations,
                        tool_results={},
                        reasoning_trace=None,  # No reasoning trace for cached results
                        inference_time=inference_time,
                        cache_hit=True
                    )
            except CacheError as e:
                logger.warning(f"Cache error (continuing without cache): {e}")
        
        # Step 2: Get model service
        model_service = get_model_service()
        
        if not model_service.is_loaded():
            logger.warning("Model not loaded, returning mock recommendations")
            # Return mock recommendations for demo purposes
            mock_recommendations = _generate_mock_recommendations(request.user_profile)
            inference_time = time.time() - start_time
            
            return EnhancedRecommendationResponse(
                recommendations=mock_recommendations[:request.max_recommendations],
                tool_results={"demo_mode": True, "message": "Model yüklenmedi, demo veriler gösteriliyor"},
                reasoning_trace=None,
                inference_time=inference_time,
                cache_hit=False
            )
        
        # Step 3: Fetch available gifts from Trendyol
        try:
            trendyol_service = get_trendyol_service()
            
            # Determine search parameters based on user profile
            category = _determine_category(request.user_profile)
            keywords = _generate_keywords(request.user_profile)
            
            logger.info(f"Searching Trendyol: category={category}, keywords={keywords}")
            
            # Search products
            trendyol_products = await trendyol_service.search_products(
                category=category,
                keywords=keywords,
                max_results=50,
                min_price=max(0, request.user_profile.budget * 0.3),
                max_price=request.user_profile.budget * 1.2
            )
            
            # Convert to GiftItem objects
            available_gifts: List[GiftItem] = []
            for product in trendyol_products:
                gift_item = trendyol_service.convert_to_gift_item(product)
                if gift_item:
                    available_gifts.append(gift_item)
            
            logger.info(f"Found {len(available_gifts)} available gifts")
            
            if not available_gifts:
                logger.warning("No gifts found from Trendyol")
                return EnhancedRecommendationResponse(
                    recommendations=[],
                    tool_results={"error": "Uygun ürün bulunamadı"},
                    reasoning_trace=None,
                    inference_time=time.time() - start_time,
                    cache_hit=False
                )
        
        except TrendyolAPIError as e:
            logger.warning(f"Trendyol API error (using mock data): {e}")
            # Use mock gifts when Trendyol API is unavailable
            mock_recommendations = _generate_mock_recommendations(request.user_profile)
            inference_time = time.time() - start_time
            
            return EnhancedRecommendationResponse(
                recommendations=mock_recommendations[:request.max_recommendations],
                tool_results={"trendyol_api_error": True, "message": "Trendyol API kullanılamıyor, demo veriler gösteriliyor"},
                reasoning_trace=None,
                inference_time=inference_time,
                cache_hit=False
            )
        
        # Step 4: Run model inference with reasoning parameters
        try:
            logger.info(
                f"Running model inference with include_reasoning={actual_include_reasoning}, "
                f"reasoning_level={actual_reasoning_level}"
            )
            
            recommendations, tool_results, reasoning_trace = await model_service.generate_recommendations(
                user_profile=request.user_profile,
                available_gifts=available_gifts,
                max_recommendations=request.max_recommendations,
                include_reasoning=actual_include_reasoning,
                reasoning_level=actual_reasoning_level
            )
            
            logger.info(
                f"Model generated {len(recommendations)} recommendations "
                f"with reasoning_trace={'present' if reasoning_trace else 'absent'}"
            )
        
        except ModelInferenceError as e:
            logger.error(f"Model inference error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
            )
        
        # Step 5: Execute additional tools if needed
        # (Tools are already executed during model inference)
        # We can optionally run additional tools here
        
        # Step 6: Cache results if enabled
        if request.use_cache and recommendations:
            try:
                cache_service = await get_cache_service()
                await cache_service.set_recommendations(
                    request.user_profile,
                    recommendations
                )
                logger.info("Recommendations cached successfully")
            except CacheError as e:
                logger.warning(f"Failed to cache recommendations: {e}")
        
        # Convert recommendations to EnhancedGiftRecommendation
        enhanced_recommendations = []
        for rec in recommendations:
            if isinstance(rec, EnhancedGiftRecommendation):
                enhanced_recommendations.append(rec)
            else:
                # Convert GiftRecommendation to EnhancedGiftRecommendation
                enhanced_rec = EnhancedGiftRecommendation(
                    gift=rec.gift,
                    confidence_score=rec.confidence_score,
                    reasoning=rec.reasoning,
                    tool_insights=rec.tool_insights,
                    rank=rec.rank,
                    reasoning_trace=None  # Individual reasoning traces not included
                )
                enhanced_recommendations.append(enhanced_rec)
        
        # Calculate total inference time
        inference_time = time.time() - start_time
        
        logger.info(
            f"Recommendation request completed: {len(enhanced_recommendations)} recommendations "
            f"in {inference_time:.2f}s with reasoning_level={actual_reasoning_level}"
        )
        
        return EnhancedRecommendationResponse(
            recommendations=enhanced_recommendations,
            tool_results=tool_results,
            reasoning_trace=reasoning_trace,
            inference_time=inference_time,
            cache_hit=cache_hit
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in get_recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bir hata oluştu. Lütfen daha sonra tekrar deneyin."
        )


def _determine_category(user_profile) -> str:
    """
    Determine product category based on user profile
    
    Args:
        user_profile: User profile data
        
    Returns:
        Category string
    """
    # Map occasions to categories
    occasion_category_map = {
        "birthday": "Hediye",
        "anniversary": "Hediye",
        "wedding": "Ev & Yaşam",
        "graduation": "Elektronik",
        "christmas": "Hediye",
        "valentine": "Hediye",
        "mother_day": "Hediye",
        "father_day": "Hediye",
    }
    
    # Map hobbies to categories
    hobby_category_map = {
        "cooking": "Ev & Yaşam",
        "gardening": "Bahçe & Yapı Market",
        "reading": "Kitap",
        "sports": "Spor & Outdoor",
        "gaming": "Elektronik",
        "music": "Elektronik",
        "art": "Hobi",
        "travel": "Spor & Outdoor",
        "photography": "Elektronik",
        "fashion": "Moda",
    }
    
    # Try to determine from occasion
    occasion = user_profile.occasion.lower()
    if occasion in occasion_category_map:
        return occasion_category_map[occasion]
    
    # Try to determine from hobbies
    if user_profile.hobbies:
        for hobby in user_profile.hobbies:
            hobby_lower = hobby.lower()
            if hobby_lower in hobby_category_map:
                return hobby_category_map[hobby_lower]
    
    # Default category
    return "Hediye"


def _generate_mock_recommendations(user_profile: UserProfile) -> List[EnhancedGiftRecommendation]:
    """
    Generate mock recommendations for demo purposes
    
    Args:
        user_profile: User profile data
        
    Returns:
        List of mock recommendations
    """
    mock_gifts = [
        GiftItem(
            id="mock-1",
            name="Premium Kahve Seti",
            category="Ev & Yaşam",
            price=299.99,
            rating=4.5,
            image_url="https://cdn.dummyjson.com/products/images/groceries/Coffee%20Beans/1.png",
            trendyol_url="https://www.trendyol.com/demo",
            description="Özel kahve seti",
            tags=["kahve", "hediye"],
            age_suitability=(18, 100),
            occasion_fit=["birthday", "anniversary"],
            in_stock=True
        ),
        GiftItem(
            id="mock-2",
            name="Spor Ekipmanı Seti",
            category="Spor & Outdoor",
            price=450.00,
            rating=4.7,
            image_url="https://cdn.dummyjson.com/products/images/sports-accessories/Baseball%20Glove/1.png",
            trendyol_url="https://www.trendyol.com/demo",
            description="Kaliteli spor ekipmanları",
            tags=["spor", "fitness"],
            age_suitability=(18, 100),
            occasion_fit=["birthday"],
            in_stock=True
        ),
        GiftItem(
            id="mock-3",
            name="Kitap Seti",
            category="Kitap",
            price=199.99,
            rating=4.8,
            image_url="https://cdn.dummyjson.com/products/images/furniture/Annibale%20Colombo%20Bed/1.png",
            trendyol_url="https://www.trendyol.com/demo",
            description="Bestseller kitap koleksiyonu",
            tags=["kitap", "okuma"],
            age_suitability=(18, 100),
            occasion_fit=["birthday", "graduation"],
            in_stock=True
        ),
    ]
    
    recommendations = []
    for i, gift in enumerate(mock_gifts):
        rec = EnhancedGiftRecommendation(
            gift=gift,
            confidence_score=0.85 - (i * 0.1),
            reasoning=[
                f"Bütçenize uygun: {gift.price} TL",
                f"Yüksek değerlendirme: {gift.rating}/5.0",
                "Demo mod - gerçek öneriler için model yüklenmelidir"
            ],
            tool_insights={},
            rank=i + 1,
            reasoning_trace=None
        )
        recommendations.append(rec)
    
    return recommendations


def _generate_keywords(user_profile) -> List[str]:
    """
    Generate search keywords based on user profile
    
    Args:
        user_profile: User profile data
        
    Returns:
        List of keywords
    """
    keywords = []
    
    # Add occasion
    keywords.append(user_profile.occasion)
    
    # Add relationship
    keywords.append(user_profile.relationship)
    
    # Add hobbies (limit to 2)
    keywords.extend(user_profile.hobbies[:2])
    
    # Add personality traits (limit to 1)
    if user_profile.personality_traits:
        keywords.append(user_profile.personality_traits[0])
    
    # Add generic gift keyword
    keywords.append("hediye")
    
    return keywords
