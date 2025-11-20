"""
Gift Recommendation Specific Tools
"""

import requests
import json
import time
import random
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

from .tool_registry import BaseTool


class PriceComparisonTool(BaseTool):
    """Tool for comparing prices across different e-commerce platforms"""
    
    def __init__(self):
        super().__init__(
            "price_comparison",
            "Compare prices for a product across multiple e-commerce platforms"
        )
        # Mock API endpoints - replace with real ones
        self.platforms = {
            "amazon": "https://api.amazon.com/products/search",
            "trendyol": "https://api.trendyol.com/products/search", 
            "hepsiburada": "https://api.hepsiburada.com/products/search",
            "n11": "https://api.n11.com/products/search",
            "gittigidiyor": "https://api.gittigidiyor.com/products/search"
        }
    
    def execute(self, product_name: Optional[str] = None, max_sites: int = 5, 
                category: Optional[str] = None, gifts: Optional[List] = None, 
                budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare prices for a product OR filter gifts by budget
        
        Args:
            product_name: Name of the product to search (for web scraping mode)
            max_sites: Maximum number of sites to check (for web scraping mode)
            category: Product category for better search results
            gifts: List of gift items to filter (for gift catalog mode)
            budget: Budget limit to filter gifts (for gift catalog mode)
            
        Returns:
            Dictionary with price comparison results
        """
        # Gift catalog mode: filter gifts by budget
        if gifts is not None and budget is not None:
            in_budget = []
            over_budget = []
            total_price = 0.0
            
            for gift in gifts:
                price = gift.price if hasattr(gift, 'price') else gift.get('price', 0)
                if price <= budget:
                    in_budget.append(gift)
                else:
                    over_budget.append(gift)
                total_price += price
            
            avg_price = total_price / len(gifts) if gifts else 0
            
            return {
                "in_budget": in_budget,
                "over_budget": over_budget,
                "budget": budget,
                "average_price": avg_price,
                "in_budget_count": len(in_budget),
                "over_budget_count": len(over_budget),
                "timestamp": datetime.now().isoformat()
            }
        
        # Web scraping mode: original functionality
        if product_name is None:
            return {
                "error": "Either product_name or (gifts + budget) must be provided",
                "in_budget": [],
                "over_budget": []
            }
        
        prices = {}
        product_details = {}
        
        # Limit to available platforms
        platforms_to_check = list(self.platforms.keys())[:max_sites]
        
        for platform in platforms_to_check:
            try:
                # Mock API call - replace with real implementation
                price_data = self._mock_price_api(platform, product_name, category)
                if price_data:
                    prices[platform] = price_data["price"]
                    product_details[platform] = {
                        "url": price_data["url"],
                        "rating": price_data.get("rating", 0),
                        "reviews": price_data.get("reviews", 0),
                        "availability": price_data.get("availability", "unknown")
                    }
            except Exception as e:
                print(f"Error fetching price from {platform}: {e}")
                continue
        
        if not prices:
            return {
                "product": product_name,
                "error": "No prices found",
                "prices": {},
                "best_price": None,
                "best_site": None
            }
        
        best_price = min(prices.values())
        best_site = min(prices.keys(), key=lambda x: prices[x])
        
        # Calculate savings
        max_price = max(prices.values())
        savings = max_price - best_price
        savings_percentage = (savings / max_price) * 100 if max_price > 0 else 0
        
        return {
            "product": product_name,
            "category": category,
            "prices": prices,
            "product_details": product_details,
            "best_price": best_price,
            "best_site": best_site,
            "max_price": max_price,
            "savings": savings,
            "savings_percentage": round(savings_percentage, 2),
            "price_range": f"{best_price:.2f} - {max_price:.2f} TL",
            "checked_sites": len(prices),
            "timestamp": datetime.now().isoformat()
        }
    
    def _mock_price_api(self, platform: str, product_name: str, 
                       category: Optional[str] = None) -> Optional[Dict]:
        """Mock API call - replace with real implementation"""
        # Simulate API delay
        time.sleep(random.uniform(0.1, 0.5))
        
        # Mock price generation based on platform
        base_price = random.uniform(50, 500)
        
        # Platform-specific price adjustments
        platform_multipliers = {
            "amazon": 1.1,
            "trendyol": 0.95,
            "hepsiburada": 1.0,
            "n11": 0.9,
            "gittigidiyor": 0.85
        }
        
        price = base_price * platform_multipliers.get(platform, 1.0)
        
        # Sometimes return None to simulate product not found
        if random.random() < 0.1:
            return None
        
        return {
            "price": round(price, 2),
            "url": f"https://{platform}.com/product/{product_name.replace(' ', '-')}",
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews": random.randint(10, 1000),
            "availability": random.choice(["in_stock", "limited", "out_of_stock"])
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to search for (web scraping mode)"
                },
                "max_sites": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum number of sites to check"
                },
                "category": {
                    "type": "string",
                    "description": "Product category for better search results"
                },
                "gifts": {
                    "type": "array",
                    "description": "List of gift items to filter (gift catalog mode)"
                },
                "budget": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Budget limit to filter gifts (gift catalog mode)"
                }
            },
            "required": []  # No required params - either product_name OR (gifts + budget)
        }


class InventoryCheckTool(BaseTool):
    """Tool for checking product inventory and availability"""
    
    def __init__(self):
        super().__init__(
            "inventory_check",
            "Check product inventory and availability across stores"
        )
    
    def execute(self, product_id: Optional[str] = None, location: str = "TR", 
                store_chain: Optional[str] = None, gifts: Optional[List] = None) -> Dict[str, Any]:
        """
        Check inventory for a product OR check gift availability
        
        Args:
            product_id: Product identifier (for web scraping mode)
            location: Location code (TR, US, etc.)
            store_chain: Specific store chain to check
            gifts: List of gift items to check (for gift catalog mode)
            
        Returns:
            Dictionary with inventory information
        """
        # Gift catalog mode: assume all gifts are available
        if gifts is not None:
            available = gifts  # In catalog mode, all gifts are available
            unavailable = []
            
            return {
                "available": available,
                "unavailable": unavailable,
                "available_count": len(available),
                "unavailable_count": len(unavailable),
                "last_updated": datetime.now().isoformat()
            }
        
        # Web scraping mode: original functionality
        if product_id is None:
            return {
                "error": "Either product_id or gifts must be provided",
                "available": [],
                "unavailable": []
            }
        
        # Mock inventory check
        inventory_data = self._mock_inventory_check(product_id, location, store_chain)
        
        return {
            "product_id": product_id,
            "location": location,
            "store_chain": store_chain,
            "in_stock": inventory_data["in_stock"],
            "quantity": inventory_data["quantity"],
            "estimated_delivery": inventory_data["delivery"],
            "last_updated": datetime.now().isoformat(),
            "stores_checked": inventory_data["stores"],
            "alternative_locations": inventory_data.get("alternatives", [])
        }
    
    def _mock_inventory_check(self, product_id: str, location: str, 
                             store_chain: Optional[str]) -> Dict[str, Any]:
        """Mock inventory check"""
        in_stock = random.random() > 0.2  # 80% chance in stock
        quantity = random.randint(0, 50) if in_stock else 0
        
        delivery_options = ["1-2 days", "2-3 days", "3-5 days", "1 week", "2 weeks"]
        delivery = random.choice(delivery_options) if in_stock else "out of stock"
        
        stores = ["Store A", "Store B", "Store C"] if not store_chain else [store_chain]
        
        alternatives = []
        if not in_stock:
            alternatives = [
                {"location": "Istanbul", "quantity": random.randint(1, 10)},
                {"location": "Ankara", "quantity": random.randint(1, 5)}
            ]
        
        return {
            "in_stock": in_stock,
            "quantity": quantity,
            "delivery": delivery,
            "stores": stores,
            "alternatives": alternatives
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "Product identifier to check (web scraping mode)"
                },
                "location": {
                    "type": "string",
                    "default": "TR",
                    "description": "Location code (TR, US, etc.)"
                },
                "store_chain": {
                    "type": "string",
                    "description": "Specific store chain to check"
                },
                "gifts": {
                    "type": "array",
                    "description": "List of gift items to check (gift catalog mode)"
                }
            },
            "required": []  # No required params
        }


class ReviewAnalysisTool(BaseTool):
    """Tool for analyzing product reviews and ratings"""
    
    def __init__(self):
        super().__init__(
            "review_analysis",
            "Analyze product reviews and extract insights"
        )
    
    def execute(self, product_id: Optional[str] = None, max_reviews: int = 100,
                language: str = "tr", gifts: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze product reviews OR analyze gift ratings
        
        Args:
            product_id: Product identifier (for web scraping mode)
            max_reviews: Maximum number of reviews to analyze
            language: Language for analysis (tr, en, etc.)
            gifts: List of gift items to analyze (for gift catalog mode)
            
        Returns:
            Dictionary with review analysis results
        """
        # Gift catalog mode: analyze gift ratings
        if gifts is not None:
            total_rating = 0.0
            rated_gifts = []
            top_rated = []
            
            for gift in gifts:
                rating = gift.rating if hasattr(gift, 'rating') else gift.get('rating', 0)
                if rating > 0:
                    rated_gifts.append(gift)
                    total_rating += rating
                    if rating >= 4.0:
                        top_rated.append(gift)
            
            avg_rating = total_rating / len(rated_gifts) if rated_gifts else 0.0
            
            return {
                "average_rating": avg_rating,
                "top_rated": top_rated,
                "rated_count": len(rated_gifts),
                "total_count": len(gifts),
                "analysis_date": datetime.now().isoformat()
            }
        
        # Web scraping mode: original functionality
        if product_id is None:
            return {
                "error": "Either product_id or gifts must be provided",
                "average_rating": 0.0,
                "top_rated": []
            }
        
        # Mock review analysis
        analysis = self._mock_review_analysis(product_id, max_reviews, language)
        
        return {
            "product_id": product_id,
            "total_reviews": analysis["total_reviews"],
            "avg_rating": analysis["avg_rating"],
            "rating_distribution": analysis["rating_distribution"],
            "sentiment_score": analysis["sentiment_score"],
            "sentiment_label": analysis["sentiment_label"],
            "key_positives": analysis["positives"],
            "key_negatives": analysis["negatives"],
            "common_themes": analysis["themes"],
            "recommendation_confidence": analysis["confidence"],
            "language": language,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _mock_review_analysis(self, product_id: str, max_reviews: int, 
                             language: str) -> Dict[str, Any]:
        """Mock review analysis"""
        total_reviews = random.randint(10, 500)
        avg_rating = round(random.uniform(3.0, 5.0), 1)
        
        # Rating distribution
        rating_dist = {
            "5": random.randint(20, 60),
            "4": random.randint(15, 30), 
            "3": random.randint(5, 20),
            "2": random.randint(2, 10),
            "1": random.randint(1, 8)
        }
        
        # Sentiment analysis
        sentiment_score = (avg_rating - 1) / 4  # Normalize to 0-1
        if sentiment_score > 0.7:
            sentiment_label = "positive"
        elif sentiment_score > 0.4:
            sentiment_label = "neutral"
        else:
            sentiment_label = "negative"
        
        # Key insights
        positive_keywords = [
            "kaliteli", "hızlı kargo", "güzel tasarım", "dayanıklı", 
            "kullanışlı", "değerli", "memnun", "tavsiye ederim"
        ]
        negative_keywords = [
            "pahalı", "geç teslimat", "kırık geldi", "beklediğim gibi değil",
            "kalitesiz", "para etmez", "sorunlu", "iade ettim"
        ]
        
        positives = random.sample(positive_keywords, random.randint(2, 4))
        negatives = random.sample(negative_keywords, random.randint(1, 3))
        
        themes = ["kalite", "fiyat", "teslimat", "tasarım", "kullanım"]
        
        confidence = min(0.95, sentiment_score + random.uniform(0.1, 0.2))
        
        return {
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "rating_distribution": rating_dist,
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": sentiment_label,
            "positives": positives,
            "negatives": negatives,
            "themes": themes,
            "confidence": round(confidence, 2)
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "Product identifier to analyze (web scraping mode)"
                },
                "max_reviews": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Maximum number of reviews to analyze"
                },
                "language": {
                    "type": "string",
                    "default": "tr",
                    "description": "Language for analysis"
                },
                "gifts": {
                    "type": "array",
                    "description": "List of gift items to analyze (gift catalog mode)"
                }
            },
            "required": []  # No required params
        }


class TrendAnalysisTool(BaseTool):
    """Tool for analyzing product trends and popularity"""
    
    def __init__(self):
        super().__init__(
            "trend_analysis",
            "Analyze product trends and market popularity"
        )
    
    def execute(self, category: Optional[str] = None, time_period: str = "30d",
                region: str = "TR", gifts: Optional[List] = None, user_age: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze trends for a product category OR find trending gifts
        
        Args:
            category: Product category to analyze (for web scraping mode)
            time_period: Time period (7d, 30d, 90d, 1y)
            region: Region for analysis
            gifts: List of gift items to analyze (for gift catalog mode)
            user_age: User age for age-appropriate trending items
            
        Returns:
            Dictionary with trend analysis results
        """
        # Gift catalog mode: find trending gifts (high ratings, popular tags)
        if gifts is not None:
            trending = []
            avg_popularity = 0.0
            
            for gift in gifts:
                rating = gift.rating if hasattr(gift, 'rating') else gift.get('rating', 0)
                tags = gift.tags if hasattr(gift, 'tags') else gift.get('tags', [])
                
                # Consider trending if rating > 4.0 or has "trendy" tag
                is_trending = rating >= 4.0 or 'trendy' in tags or 'popular' in tags
                
                if is_trending:
                    trending.append(gift)
                
                # Calculate popularity score based on rating
                avg_popularity += rating / 5.0  # Normalize to 0-1
            
            avg_popularity = avg_popularity / len(gifts) if gifts else 0.0
            
            return {
                "trending": trending,
                "trending_count": len(trending),
                "average_popularity": avg_popularity,
                "analysis_date": datetime.now().isoformat()
            }
        
        # Web scraping mode: original functionality
        if category is None:
            return {
                "error": "Either category or gifts must be provided",
                "trending": [],
                "average_popularity": 0.0
            }
        
        trend_data = self._mock_trend_analysis(category, time_period, region)
        
        return {
            "category": category,
            "time_period": time_period,
            "region": region,
            "trend_direction": trend_data["direction"],
            "popularity_score": trend_data["popularity"],
            "growth_rate": trend_data["growth_rate"],
            "seasonal_factor": trend_data["seasonal"],
            "trending_items": trend_data["trending_items"],
            "declining_items": trend_data["declining_items"],
            "market_insights": trend_data["insights"],
            "analysis_date": datetime.now().isoformat()
        }
    
    def _mock_trend_analysis(self, category: str, time_period: str, 
                            region: str) -> Dict[str, Any]:
        """Mock trend analysis"""
        # Trend direction
        directions = ["increasing", "decreasing", "stable", "volatile"]
        direction = random.choice(directions)
        
        # Popularity score (0-1)
        popularity = round(random.uniform(0.3, 0.95), 2)
        
        # Growth rate
        if direction == "increasing":
            growth_rate = round(random.uniform(5, 50), 1)
        elif direction == "decreasing":
            growth_rate = round(random.uniform(-30, -5), 1)
        else:
            growth_rate = round(random.uniform(-5, 5), 1)
        
        # Seasonal factor
        current_month = datetime.now().month
        if current_month in [11, 12, 1]:  # Holiday season
            seasonal = round(random.uniform(1.2, 1.8), 2)
        elif current_month in [6, 7, 8]:  # Summer
            seasonal = round(random.uniform(0.8, 1.2), 2)
        else:
            seasonal = round(random.uniform(0.9, 1.1), 2)
        
        # Category-specific trending items
        category_items = {
            "gardening": ["organic seeds", "plant pots", "garden tools", "fertilizer"],
            "cooking": ["air fryer", "instant pot", "knife set", "cutting board"],
            "reading": ["bestseller novels", "e-readers", "bookmarks", "reading lights"],
            "technology": ["smart watches", "wireless earbuds", "phone cases", "chargers"],
            "sports": ["yoga mats", "dumbbells", "running shoes", "fitness trackers"]
        }
        
        trending_items = category_items.get(category, ["item1", "item2", "item3"])
        declining_items = ["old item1", "outdated item2"]
        
        insights = [
            f"{category} category showing {direction} trend",
            f"Peak demand expected in {random.choice(['morning', 'evening', 'weekend'])}",
            f"Price sensitivity is {'high' if popularity < 0.6 else 'moderate'}"
        ]
        
        return {
            "direction": direction,
            "popularity": popularity,
            "growth_rate": growth_rate,
            "seasonal": seasonal,
            "trending_items": trending_items,
            "declining_items": declining_items,
            "insights": insights
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Product category to analyze (web scraping mode)"
                },
                "time_period": {
                    "type": "string",
                    "enum": ["7d", "30d", "90d", "1y"],
                    "default": "30d",
                    "description": "Time period for analysis"
                },
                "region": {
                    "type": "string",
                    "default": "TR",
                    "description": "Region for analysis"
                },
                "gifts": {
                    "type": "array",
                    "description": "List of gift items to analyze (gift catalog mode)"
                },
                "user_age": {
                    "type": "integer",
                    "description": "User age for age-appropriate trending items"
                }
            },
            "required": []  # No required params
        }


class BudgetOptimizerTool(BaseTool):
    """Tool for optimizing budget allocation for gift purchases"""
    
    def __init__(self):
        super().__init__(
            "budget_optimizer",
            "Optimize budget allocation for gift recommendations"
        )
    
    def execute(self, budget: float, preferences: List[str],
                occasion: str = "birthday") -> Dict[str, Any]:
        """
        Optimize budget allocation
        
        Args:
            budget: Total budget amount
            preferences: List of preference categories
            occasion: Occasion type
            
        Returns:
            Dictionary with budget optimization results
        """
        optimization = self._optimize_budget(budget, preferences, occasion)
        
        return {
            "total_budget": budget,
            "preferences": preferences,
            "occasion": occasion,
            "recommended_allocation": optimization["allocation"],
            "value_score": optimization["value_score"],
            "savings_opportunities": optimization["savings"],
            "alternative_strategies": optimization["alternatives"],
            "budget_breakdown": optimization["breakdown"],
            "optimization_date": datetime.now().isoformat()
        }
    
    def _optimize_budget(self, budget: float, preferences: List[str], 
                        occasion: str) -> Dict[str, Any]:
        """Mock budget optimization"""
        # Basic allocation strategy
        if budget < 50:
            allocation = {
                "main_gift": budget * 0.9,
                "wrapping": budget * 0.1
            }
        elif budget < 200:
            allocation = {
                "main_gift": budget * 0.75,
                "accessories": budget * 0.15,
                "wrapping": budget * 0.1
            }
        else:
            allocation = {
                "main_gift": budget * 0.6,
                "accessories": budget * 0.25,
                "wrapping": budget * 0.1,
                "backup_gift": budget * 0.05
            }
        
        # Value score based on budget efficiency
        value_score = min(0.95, 0.5 + (budget / 1000) * 0.4)
        
        # Savings opportunities
        savings = [
            "Bundle deals available",
            "Seasonal discounts active",
            "Free shipping over 100 TL"
        ]
        
        # Alternative strategies
        alternatives = [
            "Experience gifts for better value",
            "DIY options to save 30%",
            "Group gift for higher budget items"
        ]
        
        # Detailed breakdown
        breakdown = {
            category: {
                "amount": amount,
                "percentage": round((amount / budget) * 100, 1),
                "suggested_items": self._get_category_suggestions(category, amount)
            }
            for category, amount in allocation.items()
        }
        
        return {
            "allocation": allocation,
            "value_score": round(value_score, 2),
            "savings": savings,
            "alternatives": alternatives,
            "breakdown": breakdown
        }
    
    def _get_category_suggestions(self, category: str, amount: float) -> List[str]:
        """Get suggestions for budget category"""
        suggestions = {
            "main_gift": ["Premium item", "Quality brand", "Personalized option"],
            "accessories": ["Complementary items", "Add-ons", "Enhancement"],
            "wrapping": ["Gift box", "Ribbon", "Card"],
            "backup_gift": ["Small item", "Emergency option"]
        }
        return suggestions.get(category, ["Generic suggestion"])
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "number",
                    "minimum": 10,
                    "description": "Total budget amount"
                },
                "preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of preference categories"
                },
                "occasion": {
                    "type": "string",
                    "default": "birthday",
                    "description": "Occasion type"
                }
            },
            "required": ["budget", "preferences"]
        }


class GiftRecommendationTools:
    """Collection of all gift recommendation tools"""
    
    def __init__(self):
        self.tools = [
            PriceComparisonTool(),
            InventoryCheckTool(),
            ReviewAnalysisTool(),
            TrendAnalysisTool(),
            BudgetOptimizerTool()
        ]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all tools"""
        return [tool.name for tool in self.tools]


if __name__ == "__main__":
    # Test gift recommendation tools
    from .tool_registry import ToolRegistry
    
    registry = ToolRegistry()
    gift_tools = GiftRecommendationTools()
    
    # Register all tools
    for tool in gift_tools.get_all_tools():
        registry.register_tool(tool)
    
    print("Registered tools:", registry.list_tools())
    
    # Test price comparison
    price_result = registry.call_tool_by_name(
        "price_comparison",
        product_name="Organic Seed Set",
        max_sites=3,
        category="gardening"
    )
    print("Price comparison result:", json.dumps(price_result.result, indent=2))
    
    # Test trend analysis
    trend_result = registry.call_tool_by_name(
        "trend_analysis",
        category="gardening",
        time_period="30d"
    )
    print("Trend analysis result:", json.dumps(trend_result.result, indent=2))
    
    # Get tool statistics
    stats = registry.get_tool_stats()
    print("Tool statistics:", json.dumps(stats, indent=2))