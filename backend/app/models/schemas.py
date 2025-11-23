"""Pydantic data models for API requests and responses"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, HttpUrl, field_validator


class UserProfile(BaseModel):
    """User profile data model"""
    
    age: int = Field(ge=18, le=100, description="Age of the gift recipient")
    hobbies: List[str] = Field(
        min_length=1,
        max_length=10,
        description="List of hobbies"
    )
    relationship: str = Field(description="Relationship to the recipient")
    budget: float = Field(gt=0, description="Budget in Turkish Lira")
    occasion: str = Field(description="Special occasion")
    personality_traits: List[str] = Field(
        max_length=5,
        default_factory=list,
        description="Personality traits"
    )
    
    @field_validator("hobbies")
    @classmethod
    def validate_hobbies(cls, v: List[str]) -> List[str]:
        """Validate hobbies are non-empty strings"""
        if not all(hobby.strip() for hobby in v):
            raise ValueError("Hobbies cannot be empty strings")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "hobbies": ["gardening", "cooking"],
                "relationship": "mother",
                "budget": 500.0,
                "occasion": "birthday",
                "personality_traits": ["practical", "eco-friendly"]
            }
        }


class GiftItem(BaseModel):
    """Gift item data model"""
    
    id: str = Field(description="Unique product identifier")
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price: float = Field(ge=0, description="Product price in TL")
    rating: float = Field(ge=0, le=5, description="Product rating")
    image_url: HttpUrl = Field(description="Product image URL")
    trendyol_url: HttpUrl = Field(description="Trendyol product page URL")
    description: str = Field(default="", description="Product description")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    age_suitability: Tuple[int, int] = Field(
        default=(18, 100),
        description="Age range suitability"
    )
    occasion_fit: List[str] = Field(
        default_factory=list,
        description="Suitable occasions"
    )
    in_stock: bool = Field(default=True, description="Stock availability")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "12345",
                "name": "Premium Coffee Set",
                "category": "Kitchen & Dining",
                "price": 299.99,
                "rating": 4.5,
                "image_url": "https://cdn.trendyol.com/example.jpg",
                "trendyol_url": "https://www.trendyol.com/product/12345",
                "description": "High-quality coffee set",
                "tags": ["coffee", "kitchen", "gift"],
                "age_suitability": (25, 65),
                "occasion_fit": ["birthday", "anniversary"],
                "in_stock": True
            }
        }


class PriceComparisonResult(BaseModel):
    """Price comparison tool result"""
    
    best_price: float
    average_price: float
    price_range: str
    savings_percentage: float
    checked_platforms: List[str]


class ReviewAnalysisResult(BaseModel):
    """Review analysis tool result"""
    
    average_rating: float
    total_reviews: int
    sentiment_score: float
    key_positives: List[str]
    key_negatives: List[str]
    recommendation_confidence: float


class TrendAnalysisResult(BaseModel):
    """Trend analysis tool result"""
    
    trend_direction: str
    popularity_score: float
    growth_rate: float
    trending_items: List[str]


class BudgetOptimizerResult(BaseModel):
    """Budget optimizer tool result"""
    
    recommended_allocation: Dict[str, float]
    value_score: float
    savings_opportunities: List[str]


class ToolResults(BaseModel):
    """Aggregated tool results"""
    
    price_comparison: Optional[PriceComparisonResult] = None
    inventory_check: Optional[Dict[str, Any]] = None
    review_analysis: Optional[ReviewAnalysisResult] = None
    trend_analysis: Optional[TrendAnalysisResult] = None
    budget_optimizer: Optional[BudgetOptimizerResult] = None


class GiftRecommendation(BaseModel):
    """Gift recommendation with confidence score"""
    
    gift: GiftItem
    confidence_score: float = Field(ge=0, le=1, description="Confidence score")
    reasoning: List[str] = Field(description="Reasoning for recommendation")
    tool_insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Insights from tools"
    )
    rank: int = Field(ge=1, description="Recommendation rank")


class RecommendationRequest(BaseModel):
    """Request for gift recommendations"""
    
    user_profile: UserProfile
    max_recommendations: int = Field(default=5, ge=1, le=20)
    use_cache: bool = Field(default=True)


class RecommendationResponse(BaseModel):
    """Response with gift recommendations"""
    
    recommendations: List[GiftRecommendation]
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    inference_time: float = Field(description="Inference time in seconds")
    cache_hit: bool = Field(description="Whether result was from cache")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str
    model_loaded: bool
    trendyol_api_status: str
    cache_status: str
    timestamp: datetime


class ToolStatsResponse(BaseModel):
    """Tool usage statistics response"""
    
    tool_usage: Dict[str, int]
    success_rates: Dict[str, float]
    average_execution_times: Dict[str, float]


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: str
