"""Recommendation endpoints"""

import time
import logging
from typing import List
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    GiftItem,
    GiftRecommendation
)
from app.services.model_inference import get_model_service
from app.services.cache_service import get_cache_service
from app.services.trendyol_api import get_trendyol_service
from app.services.tool_orchestration import ToolOrchestrationService
from app.core.exceptions import (
    ModelInferenceError,
    TrendyolAPIError,
    CacheError
)
from models.tools import ToolRegistry


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get gift recommendations based on user profile
    
    This endpoint:
    1. Validates user profile (done by Pydantic)
    2. Checks cache for existing recommendations
    3. Fetches available gifts from Trendyol
    4. Runs model inference if needed
    5. Executes relevant tools
    6. Returns ranked recommendations
    
    Args:
        request: Recommendation request with user profile
        
    Returns:
        RecommendationResponse with recommendations and metadata
        
    Raises:
        HTTPException: If recommendation generation fails
    """
    start_time = time.time()
    cache_hit = False
    
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
                    
                    return RecommendationResponse(
                        recommendations=cached_recommendations[:request.max_recommendations],
                        tool_results={},
                        inference_time=inference_time,
                        cache_hit=True
                    )
            except CacheError as e:
                logger.warning(f"Cache error (continuing without cache): {e}")
        
        # Step 2: Get model service
        model_service = get_model_service()
        
        if not model_service.is_loaded():
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
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
                return RecommendationResponse(
                    recommendations=[],
                    tool_results={"error": "Uygun ürün bulunamadı"},
                    inference_time=time.time() - start_time,
                    cache_hit=False
                )
        
        except TrendyolAPIError as e:
            logger.error(f"Trendyol API error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ürün verileri şu anda alınamıyor. Lütfen daha sonra tekrar deneyin."
            )
        
        # Step 4: Run model inference
        try:
            logger.info("Running model inference...")
            
            recommendations, tool_results = await model_service.generate_recommendations(
                user_profile=request.user_profile,
                available_gifts=available_gifts,
                max_recommendations=request.max_recommendations
            )
            
            logger.info(f"Model generated {len(recommendations)} recommendations")
        
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
        
        # Calculate total inference time
        inference_time = time.time() - start_time
        
        logger.info(
            f"Recommendation request completed: {len(recommendations)} recommendations "
            f"in {inference_time:.2f}s"
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            tool_results=tool_results,
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
