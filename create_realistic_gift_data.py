#!/usr/bin/env python3
"""
Create realistic gift data for real-world testing
"""
import json
import random
from datetime import datetime

def create_realistic_gift_catalog():
    """Create a realistic gift catalog with real products"""
    
    gifts = [
        # Technology & Electronics
        {"id": "tech_001", "name": "Wireless Bluetooth Headphones", "category": "technology", "price": 89.99, "rating": 4.5, "tags": ["audio", "wireless", "portable"], "age_range": [16, 65], "occasions": ["birthday", "christmas", "graduation"]},
        {"id": "tech_002", "name": "Smart Fitness Watch", "category": "fitness", "price": 199.99, "rating": 4.3, "tags": ["fitness", "health", "smart"], "age_range": [18, 55], "occasions": ["birthday", "new_year", "graduation"]},
        {"id": "tech_003", "name": "Portable Phone Charger", "category": "technology", "price": 29.99, "rating": 4.7, "tags": ["practical", "portable", "tech"], "age_range": [16, 70], "occasions": ["any"]},
        {"id": "tech_004", "name": "Gaming Mechanical Keyboard", "category": "gaming", "price": 129.99, "rating": 4.6, "tags": ["gaming", "rgb", "mechanical"], "age_range": [16, 35], "occasions": ["birthday", "christmas"]},
        {"id": "tech_005", "name": "Smart Home Speaker", "category": "technology", "price": 79.99, "rating": 4.4, "tags": ["smart", "home", "voice"], "age_range": [25, 65], "occasions": ["housewarming", "christmas"]},
        
        # Home & Kitchen
        {"id": "home_001", "name": "Premium Coffee Maker", "category": "kitchen", "price": 149.99, "rating": 4.5, "tags": ["coffee", "morning", "appliance"], "age_range": [25, 65], "occasions": ["housewarming", "wedding", "christmas"]},
        {"id": "home_002", "name": "Aromatherapy Essential Oil Diffuser", "category": "wellness", "price": 39.99, "rating": 4.3, "tags": ["relaxation", "aromatherapy", "wellness"], "age_range": [20, 60], "occasions": ["birthday", "mother_day", "valentine"]},
        {"id": "home_003", "name": "Luxury Throw Blanket", "category": "home", "price": 69.99, "rating": 4.6, "tags": ["cozy", "soft", "comfort"], "age_range": [18, 70], "occasions": ["any"]},
        {"id": "home_004", "name": "Indoor Plant Care Kit", "category": "gardening", "price": 34.99, "rating": 4.4, "tags": ["plants", "gardening", "green"], "age_range": [20, 65], "occasions": ["housewarming", "spring", "birthday"]},
        {"id": "home_005", "name": "Artisan Candle Set", "category": "home", "price": 45.99, "rating": 4.5, "tags": ["candles", "scented", "ambiance"], "age_range": [18, 65], "occasions": ["any"]},
        
        # Fashion & Accessories
        {"id": "fashion_001", "name": "Leather Crossbody Bag", "category": "fashion", "price": 89.99, "rating": 4.4, "tags": ["leather", "stylish", "practical"], "age_range": [18, 50], "occasions": ["birthday", "graduation", "promotion"]},
        {"id": "fashion_002", "name": "Silk Scarf Collection", "category": "fashion", "price": 59.99, "rating": 4.3, "tags": ["silk", "elegant", "versatile"], "age_range": [25, 65], "occasions": ["mother_day", "valentine", "anniversary"]},
        {"id": "fashion_003", "name": "Minimalist Watch", "category": "fashion", "price": 119.99, "rating": 4.5, "tags": ["watch", "minimalist", "elegant"], "age_range": [20, 60], "occasions": ["graduation", "promotion", "anniversary"]},
        {"id": "fashion_004", "name": "Cozy Wool Socks Set", "category": "fashion", "price": 24.99, "rating": 4.6, "tags": ["cozy", "warm", "comfort"], "age_range": [16, 70], "occasions": ["christmas", "winter", "stocking_stuffer"]},
        
        # Books & Learning
        {"id": "book_001", "name": "Bestseller Novel Collection", "category": "books", "price": 49.99, "rating": 4.7, "tags": ["fiction", "bestseller", "reading"], "age_range": [16, 70], "occasions": ["birthday", "any"]},
        {"id": "book_002", "name": "Mindfulness & Meditation Guide", "category": "wellness", "price": 19.99, "rating": 4.4, "tags": ["mindfulness", "self-help", "wellness"], "age_range": [20, 65], "occasions": ["new_year", "birthday"]},
        {"id": "book_003", "name": "Cooking Masterclass Cookbook", "category": "cooking", "price": 39.99, "rating": 4.5, "tags": ["cooking", "recipes", "culinary"], "age_range": [18, 65], "occasions": ["housewarming", "wedding", "birthday"]},
        
        # Sports & Fitness
        {"id": "sport_001", "name": "Yoga Mat & Accessories Set", "category": "fitness", "price": 79.99, "rating": 4.5, "tags": ["yoga", "fitness", "wellness"], "age_range": [18, 60], "occasions": ["new_year", "birthday", "wellness"]},
        {"id": "sport_002", "name": "Resistance Bands Training Kit", "category": "fitness", "price": 29.99, "rating": 4.4, "tags": ["fitness", "home_gym", "portable"], "age_range": [16, 65], "occasions": ["new_year", "birthday"]},
        {"id": "sport_003", "name": "Hiking Backpack", "category": "outdoor", "price": 89.99, "rating": 4.6, "tags": ["hiking", "outdoor", "adventure"], "age_range": [18, 55], "occasions": ["birthday", "graduation", "adventure"]},
        
        # Art & Creativity
        {"id": "art_001", "name": "Professional Art Supply Set", "category": "art", "price": 69.99, "rating": 4.5, "tags": ["art", "creative", "drawing"], "age_range": [12, 65], "occasions": ["birthday", "graduation", "christmas"]},
        {"id": "art_002", "name": "Adult Coloring Book Collection", "category": "art", "price": 24.99, "rating": 4.3, "tags": ["coloring", "relaxation", "mindful"], "age_range": [18, 70], "occasions": ["stress_relief", "birthday"]},
        {"id": "art_003", "name": "DIY Craft Kit", "category": "craft", "price": 34.99, "rating": 4.4, "tags": ["diy", "craft", "creative"], "age_range": [16, 60], "occasions": ["birthday", "hobby"]},
        
        # Food & Gourmet
        {"id": "food_001", "name": "Gourmet Tea Sampler", "category": "food", "price": 39.99, "rating": 4.6, "tags": ["tea", "gourmet", "relaxation"], "age_range": [20, 70], "occasions": ["any"]},
        {"id": "food_002", "name": "Artisan Chocolate Box", "category": "food", "price": 49.99, "rating": 4.7, "tags": ["chocolate", "luxury", "sweet"], "age_range": [16, 70], "occasions": ["valentine", "anniversary", "birthday"]},
        {"id": "food_003", "name": "Specialty Coffee Bean Set", "category": "food", "price": 44.99, "rating": 4.5, "tags": ["coffee", "gourmet", "morning"], "age_range": [20, 65], "occasions": ["coffee_lover", "birthday"]},
        
        # Experience & Subscription
        {"id": "exp_001", "name": "Online Learning Course Subscription", "category": "education", "price": 99.99, "rating": 4.4, "tags": ["learning", "online", "skill"], "age_range": [16, 65], "occasions": ["graduation", "new_year", "career"]},
        {"id": "exp_002", "name": "Streaming Service Gift Card", "category": "entertainment", "price": 50.00, "rating": 4.5, "tags": ["streaming", "entertainment", "movies"], "age_range": [16, 70], "occasions": ["any"]},
        {"id": "exp_003", "name": "Spa Day Experience Voucher", "category": "wellness", "price": 129.99, "rating": 4.6, "tags": ["spa", "relaxation", "pampering"], "age_range": [18, 65], "occasions": ["mother_day", "birthday", "anniversary"]},
    ]
    
    return gifts

def create_realistic_user_scenarios():
    """Create realistic user scenarios for testing"""
    
    scenarios = [
        # Young Professional
        {
            "name": "Sarah - Young Professional",
            "profile": {
                "age": 28,
                "hobbies": ["technology", "fitness", "coffee"],
                "relationship": "friend",
                "budget": 150.0,
                "occasion": "birthday",
                "preferences": ["trendy", "practical", "tech-savvy"]
            },
            "expected_categories": ["technology", "fitness", "food"],
            "expected_tools": ["price_comparison", "review_analysis", "trend_analysis"]
        },
        
        # New Mother
        {
            "name": "Emily - New Mother",
            "profile": {
                "age": 32,
                "hobbies": ["wellness", "reading", "home_decor"],
                "relationship": "sister",
                "budget": 80.0,
                "occasion": "mother_day",
                "preferences": ["relaxing", "practical", "self-care"]
            },
            "expected_categories": ["wellness", "home", "books"],
            "expected_tools": ["review_analysis", "budget_optimizer"]
        },
        
        # College Student
        {
            "name": "Jake - College Student",
            "profile": {
                "age": 20,
                "hobbies": ["gaming", "music", "studying"],
                "relationship": "brother",
                "budget": 60.0,
                "occasion": "graduation",
                "preferences": ["trendy", "affordable", "tech"]
            },
            "expected_categories": ["technology", "gaming", "education"],
            "expected_tools": ["budget_optimizer", "price_comparison"]
        },
        
        # Retiree
        {
            "name": "Robert - Retiree",
            "profile": {
                "age": 68,
                "hobbies": ["gardening", "reading", "cooking"],
                "relationship": "father",
                "budget": 120.0,
                "occasion": "father_day",
                "preferences": ["traditional", "quality", "practical"]
            },
            "expected_categories": ["gardening", "books", "cooking"],
            "expected_tools": ["review_analysis", "inventory_check"]
        },
        
        # Fitness Enthusiast
        {
            "name": "Maria - Fitness Coach",
            "profile": {
                "age": 35,
                "hobbies": ["fitness", "wellness", "outdoor"],
                "relationship": "colleague",
                "budget": 100.0,
                "occasion": "promotion",
                "preferences": ["active", "healthy", "motivational"]
            },
            "expected_categories": ["fitness", "wellness", "outdoor"],
            "expected_tools": ["trend_analysis", "review_analysis"]
        },
        
        # Creative Artist
        {
            "name": "Alex - Graphic Designer",
            "profile": {
                "age": 29,
                "hobbies": ["art", "design", "photography"],
                "relationship": "friend",
                "budget": 90.0,
                "occasion": "birthday",
                "preferences": ["creative", "unique", "artistic"]
            },
            "expected_categories": ["art", "craft", "technology"],
            "expected_tools": ["trend_analysis", "review_analysis"]
        },
        
        # Busy Executive
        {
            "name": "David - Executive",
            "profile": {
                "age": 45,
                "hobbies": ["business", "travel", "wine"],
                "relationship": "boss",
                "budget": 200.0,
                "occasion": "appreciation",
                "preferences": ["luxury", "professional", "sophisticated"]
            },
            "expected_categories": ["fashion", "food", "experience"],
            "expected_tools": ["price_comparison", "review_analysis"]
        },
        
        # Eco-Conscious Millennial
        {
            "name": "Lisa - Environmental Scientist",
            "profile": {
                "age": 31,
                "hobbies": ["environment", "gardening", "sustainability"],
                "relationship": "partner",
                "budget": 110.0,
                "occasion": "anniversary",
                "preferences": ["eco-friendly", "sustainable", "natural"]
            },
            "expected_categories": ["gardening", "wellness", "home"],
            "expected_tools": ["trend_analysis", "review_analysis"]
        }
    ]
    
    return scenarios

def save_realistic_data():
    """Save realistic data to files"""
    
    # Create realistic gift catalog
    gifts = create_realistic_gift_catalog()
    
    with open("data/realistic_gift_catalog.json", "w") as f:
        json.dump({
            "gifts": gifts,
            "metadata": {
                "total_gifts": len(gifts),
                "categories": list(set(gift["category"] for gift in gifts)),
                "price_range": {
                    "min": min(gift["price"] for gift in gifts),
                    "max": max(gift["price"] for gift in gifts),
                    "avg": sum(gift["price"] for gift in gifts) / len(gifts)
                },
                "created": datetime.now().isoformat()
            }
        }, f, indent=2)
    
    # Create user scenarios
    scenarios = create_realistic_user_scenarios()
    
    with open("data/realistic_user_scenarios.json", "w") as f:
        json.dump({
            "scenarios": scenarios,
            "metadata": {
                "total_scenarios": len(scenarios),
                "age_range": {
                    "min": min(s["profile"]["age"] for s in scenarios),
                    "max": max(s["profile"]["age"] for s in scenarios)
                },
                "budget_range": {
                    "min": min(s["profile"]["budget"] for s in scenarios),
                    "max": max(s["profile"]["budget"] for s in scenarios)
                },
                "created": datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print("✅ Realistic gift catalog created:")
    print(f"  📦 {len(gifts)} products across {len(set(gift['category'] for gift in gifts))} categories")
    print(f"  💰 Price range: ${min(gift['price'] for gift in gifts):.2f} - ${max(gift['price'] for gift in gifts):.2f}")
    print(f"  📁 Saved to: data/realistic_gift_catalog.json")
    
    print(f"\n✅ User scenarios created:")
    print(f"  👥 {len(scenarios)} diverse user profiles")
    print(f"  🎂 Age range: {min(s['profile']['age'] for s in scenarios)} - {max(s['profile']['age'] for s in scenarios)} years")
    print(f"  📁 Saved to: data/realistic_user_scenarios.json")
    
    return gifts, scenarios

if __name__ == "__main__":
    print("🎁 Creating Realistic Gift Data for Real-World Testing")
    print("=" * 60)
    
    gifts, scenarios = save_realistic_data()
    
    print(f"\n🎯 Ready for real-world testing!")
    print(f"Use these files for comprehensive model evaluation.")