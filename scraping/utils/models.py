"""
Data Models for Web Scraping Pipeline
Pydantic models for data validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class RawProductData(BaseModel):
    """Raw product data model from scrapers"""
    
    source: str = Field(..., description="Source website name")
    url: str = Field(..., description="Product URL")
    name: str = Field(..., description="Product name")
    price: float = Field(gt=0, description="Product price (must be positive)")
    description: str = Field(..., description="Product description")
    image_url: Optional[str] = Field(None, description="Product image URL")
    rating: Optional[float] = Field(default=0.0, ge=0, le=5, description="Product rating (0-5)")
    in_stock: bool = Field(default=True, description="Stock availability")
    raw_category: str = Field(..., description="Original category from website")
    
    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate product name is not empty and has minimum length"""
        if not v or len(v.strip()) < 3:
            raise ValueError('Product name must be at least 3 characters')
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        """Validate description is not empty and has minimum length"""
        if not v or len(v.strip()) < 10:
            raise ValueError('Description must be at least 10 characters')
        return v.strip()
    
    @field_validator('url')
    @classmethod
    def url_valid(cls, v: str) -> str:
        """Validate URL format"""
        if not v.startswith('http'):
            raise ValueError('URL must start with http or https')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "ciceksepeti",
                "url": "https://www.ciceksepeti.com/product/123",
                "name": "Özel Hediye Paketi",
                "price": 150.00,
                "description": "Sevdikleriniz için özel hazırlanmış hediye paketi",
                "image_url": "https://example.com/image.jpg",
                "rating": 4.5,
                "in_stock": True,
                "raw_category": "Hediye"
            }
        }



class EnhancedProductData(BaseModel):
    """Enhanced product data with AI-generated metadata"""
    
    id: str = Field(..., description="Unique product ID")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Main category")
    price: float = Field(gt=0, description="Product price")
    rating: float = Field(ge=0, le=5, description="Product rating")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    description: str = Field(..., description="Product description")
    age_range: List[int] = Field(..., description="Suitable age range [min, max]")
    occasions: List[str] = Field(default_factory=list, description="Suitable occasions")
    source: str = Field(..., description="Source website")
    source_url: str = Field(..., description="Original product URL")
    image_url: Optional[str] = Field(None, description="Product image URL")
    
    @field_validator('age_range')
    @classmethod
    def validate_age_range(cls, v: List[int]) -> List[int]:
        """Validate age range has exactly 2 elements and min <= max"""
        if len(v) != 2:
            raise ValueError('Age range must have exactly 2 elements [min, max]')
        if v[0] > v[1]:
            raise ValueError('Minimum age cannot be greater than maximum age')
        if v[0] < 0 or v[1] > 120:
            raise ValueError('Age values must be between 0 and 120')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "ciceksepeti_0001",
                "name": "Özel Hediye Paketi",
                "category": "wellness",
                "price": 150.00,
                "rating": 4.5,
                "tags": ["relaxing", "luxury", "adults"],
                "description": "Sevdikleriniz için özel hazırlanmış hediye paketi",
                "age_range": [25, 60],
                "occasions": ["mothers_day", "birthday"],
                "source": "ciceksepeti",
                "source_url": "https://www.ciceksepeti.com/product/123",
                "image_url": "https://example.com/image.jpg"
            }
        }



class GeminiEnhancement(BaseModel):
    """AI enhancement data from Gemini API"""
    
    category: str = Field(..., description="Main product category")
    target_audience: List[str] = Field(default_factory=list, description="Target demographics")
    gift_occasions: List[str] = Field(default_factory=list, description="Suitable gift occasions")
    emotional_tags: List[str] = Field(default_factory=list, description="Emotional attributes")
    age_range: List[int] = Field(..., description="Suitable age range [min, max]")
    
    @field_validator('age_range')
    @classmethod
    def validate_age_range(cls, v: List[int]) -> List[int]:
        """Validate age range"""
        if len(v) != 2:
            raise ValueError('Age range must have exactly 2 elements')
        if v[0] > v[1]:
            raise ValueError('Minimum age cannot be greater than maximum age')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "wellness",
                "target_audience": ["adults", "women"],
                "gift_occasions": ["mothers_day", "birthday"],
                "emotional_tags": ["relaxing", "luxury"],
                "age_range": [25, 60]
            }
        }
