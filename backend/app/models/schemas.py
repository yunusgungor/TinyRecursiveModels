"""Pydantic data models for API requests and responses"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase"""
    components = string.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class CamelCaseModel(BaseModel):
    """Base model with camelCase alias generation"""
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )


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


class GiftItem(CamelCaseModel):
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


class GiftRecommendation(CamelCaseModel):
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
    
    user_profile: UserProfile = Field(alias="userProfile")
    max_recommendations: int = Field(default=5, ge=1, le=20, alias="maxRecommendations")
    use_cache: bool = Field(default=True, alias="useCache")
    
    class Config:
        populate_by_name = True  # Accept both snake_case and camelCase


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


# Enhanced Reasoning Schemas

class ToolSelectionReasoning(BaseModel):
    """Tool selection reasoning for a single tool"""
    
    name: str = Field(description="Tool name")
    selected: bool = Field(description="Whether tool was selected")
    score: float = Field(ge=0, le=1, description="Tool selection score")
    reason: str = Field(description="Reason for selection/non-selection")
    confidence: float = Field(ge=0, le=1, description="Confidence in selection")
    priority: int = Field(ge=1, description="Priority order")
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Contributing factors with weights"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "price_comparison",
                "selected": True,
                "score": 0.85,
                "reason": "User has strict budget constraint (500 TL)",
                "confidence": 0.85,
                "priority": 1,
                "factors": {
                    "budget_constraint": 0.9,
                    "price_sensitivity": 0.8
                }
            }
        }


class CategoryMatchingReasoning(BaseModel):
    """Category matching reasoning"""
    
    category_name: str = Field(description="Category name")
    score: float = Field(ge=0, le=1, description="Category match score")
    reasons: List[str] = Field(description="List of matching reasons")
    feature_contributions: Dict[str, float] = Field(
        description="Feature contribution weights"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "category_name": "Kitchen",
                "score": 0.85,
                "reasons": [
                    "User hobby: cooking (0.9 match)",
                    "Occasion: birthday (0.7 match)",
                    "Age appropriate: 35 years (0.8 match)"
                ],
                "feature_contributions": {
                    "hobby_match": 0.45,
                    "occasion_fit": 0.30,
                    "age_appropriateness": 0.25
                }
            }
        }


class AttentionWeights(BaseModel):
    """Attention weights for features"""
    
    user_features: Dict[str, float] = Field(
        description="Attention weights for user features"
    )
    gift_features: Dict[str, float] = Field(
        description="Attention weights for gift features"
    )
    
    @field_validator("user_features", "gift_features")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate weights are non-negative and sum to approximately 1.0"""
        if not v:
            return v
        
        # Check non-negative
        if any(weight < 0 for weight in v.values()):
            raise ValueError("Attention weights cannot be negative")
        
        # Check sum is approximately 1.0 (with tolerance)
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Attention weights must sum to 1.0, got {total}")
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class ThinkingStep(BaseModel):
    """Single step in thinking process"""
    
    step: int = Field(ge=1, description="Step number")
    action: str = Field(description="Action performed in this step")
    result: str = Field(description="Result of the action")
    insight: str = Field(description="Insight gained from this step")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "action": "Encode user profile",
                "result": "User encoding: [0.23, 0.45, ...]",
                "insight": "Strong cooking interest detected"
            }
        }


class ConfidenceExplanation(BaseModel):
    """Confidence score explanation"""
    
    score: float = Field(ge=0, le=1, description="Confidence score")
    level: str = Field(description="Confidence level: high, medium, or low")
    factors: Dict[str, List[str]] = Field(
        description="Positive and negative factors"
    )
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate confidence level"""
        valid_levels = ["high", "medium", "low"]
        if v not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.85,
                "level": "high",
                "factors": {
                    "positive": [
                        "Strong category match (0.9)",
                        "Excellent reviews (4.5/5.0)"
                    ],
                    "negative": [
                        "Slightly above typical budget"
                    ]
                }
            }
        }


class ReasoningTrace(BaseModel):
    """Complete reasoning trace"""
    
    tool_selection: List[ToolSelectionReasoning] = Field(
        default_factory=list,
        description="Tool selection reasoning"
    )
    category_matching: List[CategoryMatchingReasoning] = Field(
        default_factory=list,
        description="Category matching reasoning"
    )
    attention_weights: Optional[AttentionWeights] = Field(
        default=None,
        description="Attention weights"
    )
    thinking_steps: List[ThinkingStep] = Field(
        default_factory=list,
        description="Step-by-step thinking process"
    )
    confidence_explanation: Optional[ConfidenceExplanation] = Field(
        default=None,
        description="Confidence score explanation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_selection": [
                    {
                        "name": "price_comparison",
                        "selected": True,
                        "score": 0.85,
                        "reason": "User has strict budget constraint",
                        "confidence": 0.85,
                        "priority": 1,
                        "factors": {"budget_constraint": 0.9}
                    }
                ],
                "category_matching": [
                    {
                        "category_name": "Kitchen",
                        "score": 0.85,
                        "reasons": ["User hobby: cooking (0.9 match)"],
                        "feature_contributions": {"hobby_match": 0.45}
                    }
                ],
                "attention_weights": {
                    "user_features": {"hobbies": 0.45, "budget": 0.30, "age": 0.15, "occasion": 0.10},
                    "gift_features": {"category": 0.40, "price": 0.35, "rating": 0.25}
                },
                "thinking_steps": [
                    {
                        "step": 1,
                        "action": "Encode user profile",
                        "result": "User encoding completed",
                        "insight": "Strong cooking interest detected"
                    }
                ],
                "confidence_explanation": {
                    "score": 0.85,
                    "level": "high",
                    "factors": {
                        "positive": ["Strong category match"],
                        "negative": []
                    }
                }
            }
        }


class EnhancedGiftRecommendation(GiftRecommendation):
    """Gift recommendation with enhanced reasoning"""
    
    reasoning_trace: Optional[ReasoningTrace] = Field(
        default=None,
        description="Detailed reasoning trace"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "gift": {
                    "id": "12345",
                    "name": "Premium Coffee Set",
                    "category": "Kitchen & Dining",
                    "price": 299.99,
                    "rating": 4.5,
                    "image_url": "https://cdn.trendyol.com/example.jpg",
                    "trendyol_url": "https://www.trendyol.com/product/12345"
                },
                "confidence_score": 0.85,
                "reasoning": [
                    "Perfect match for your hobbies: cooking",
                    "Great value: Only 60% of your budget"
                ],
                "tool_insights": {},
                "rank": 1,
                "reasoning_trace": {
                    "tool_selection": [],
                    "category_matching": [],
                    "thinking_steps": []
                }
            }
        }


class EnhancedRecommendationResponse(CamelCaseModel):
    """Enhanced recommendation response with reasoning"""
    
    recommendations: List[EnhancedGiftRecommendation] = Field(
        description="List of gift recommendations"
    )
    tool_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool execution results"
    )
    reasoning_trace: Optional[ReasoningTrace] = Field(
        default=None,
        description="Overall reasoning trace"
    )
    inference_time: float = Field(
        description="Inference time in seconds"
    )
    cache_hit: bool = Field(
        description="Whether result was from cache"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": [
                    {
                        "gift": {
                            "id": "12345",
                            "name": "Premium Coffee Set",
                            "category": "Kitchen & Dining",
                            "price": 299.99,
                            "rating": 4.5,
                            "image_url": "https://cdn.trendyol.com/example.jpg",
                            "trendyol_url": "https://www.trendyol.com/product/12345"
                        },
                        "confidence_score": 0.85,
                        "reasoning": ["Perfect match for cooking hobby"],
                        "tool_insights": {},
                        "rank": 1
                    }
                ],
                "tool_results": {},
                "reasoning_trace": None,
                "inference_time": 0.5,
                "cache_hit": False
            }
        }
