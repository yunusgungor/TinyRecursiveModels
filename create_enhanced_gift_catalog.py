#!/usr/bin/env python3
"""
Create enhanced gift catalog with better category coverage
"""

import json
from datetime import datetime


def create_enhanced_gift_catalog():
    """Create comprehensive gift catalog covering all expected categories"""
    
    enhanced_gifts = [
        # Technology (reduced from overuse, but still present)
        {
            "id": "tech_001",
            "name": "Wireless Bluetooth Headphones",
            "category": "technology",
            "price": 89.99,
            "rating": 4.5,
            "tags": ["audio", "wireless", "portable"],
            "age_range": [16, 65],
            "occasions": ["birthday", "christmas", "graduation"]
        },
        {
            "id": "tech_002",
            "name": "Portable Phone Charger",
            "category": "technology",
            "price": 29.99,
            "rating": 4.7,
            "tags": ["practical", "portable", "tech"],
            "age_range": [16, 70],
            "occasions": ["any"]
        },
        {
            "id": "tech_003",
            "name": "Smart Home Speaker",
            "category": "technology",
            "price": 79.99,
            "rating": 4.4,
            "tags": ["smart", "home", "voice"],
            "age_range": [25, 65],
            "occasions": ["housewarming", "christmas"]
        },
        
        # Gardening (expanded for better coverage)
        {
            "id": "garden_001",
            "name": "Organic Seed Starter Kit",
            "category": "gardening",
            "price": 34.99,
            "rating": 4.8,
            "tags": ["organic", "seeds", "sustainable", "educational"],
            "age_range": [25, 70],
            "occasions": ["mothers_day", "fathers_day", "spring", "birthday"]
        },
        {
            "id": "garden_002",
            "name": "Indoor Plant Care Kit",
            "category": "gardening",
            "price": 45.99,
            "rating": 4.6,
            "tags": ["plants", "indoor", "care", "green"],
            "age_range": [20, 65],
            "occasions": ["housewarming", "birthday", "any"]
        },
        {
            "id": "garden_003",
            "name": "Premium Garden Tool Set",
            "category": "gardening",
            "price": 89.99,
            "rating": 4.7,
            "tags": ["tools", "durable", "professional", "outdoor"],
            "age_range": [30, 70],
            "occasions": ["fathers_day", "mothers_day", "birthday"]
        },
        {
            "id": "garden_004",
            "name": "Herb Garden Growing Kit",
            "category": "gardening",
            "price": 39.99,
            "rating": 4.5,
            "tags": ["herbs", "cooking", "fresh", "indoor"],
            "age_range": [25, 65],
            "occasions": ["housewarming", "cooking", "birthday"]
        },
        
        # Books (expanded significantly)
        {
            "id": "book_001",
            "name": "Bestseller Novel Collection",
            "category": "books",
            "price": 49.99,
            "rating": 4.7,
            "tags": ["fiction", "bestseller", "reading"],
            "age_range": [16, 70],
            "occasions": ["birthday", "christmas", "any"]
        },
        {
            "id": "book_002",
            "name": "Gardening Encyclopedia",
            "category": "books",
            "price": 39.99,
            "rating": 4.6,
            "tags": ["gardening", "reference", "educational"],
            "age_range": [25, 70],
            "occasions": ["mothers_day", "fathers_day", "birthday"]
        },
        {
            "id": "book_003",
            "name": "Cooking Masterclass Cookbook",
            "category": "books",
            "price": 44.99,
            "rating": 4.5,
            "tags": ["cooking", "recipes", "culinary", "educational"],
            "age_range": [18, 65],
            "occasions": ["housewarming", "wedding", "birthday"]
        },
        {
            "id": "book_004",
            "name": "Art Techniques Guide",
            "category": "books",
            "price": 34.99,
            "rating": 4.4,
            "tags": ["art", "techniques", "creative", "educational"],
            "age_range": [16, 60],
            "occasions": ["birthday", "graduation", "art"]
        },
        {
            "id": "book_005",
            "name": "Business Strategy Collection",
            "category": "books",
            "price": 59.99,
            "rating": 4.3,
            "tags": ["business", "strategy", "professional", "career"],
            "age_range": [25, 65],
            "occasions": ["promotion", "graduation", "appreciation"]
        },
        
        # Cooking (new comprehensive category)
        {
            "id": "cook_001",
            "name": "Professional Chef Knife Set",
            "category": "cooking",
            "price": 129.99,
            "rating": 4.8,
            "tags": ["knives", "professional", "sharp", "kitchen"],
            "age_range": [25, 65],
            "occasions": ["wedding", "housewarming", "cooking"]
        },
        {
            "id": "cook_002",
            "name": "Cast Iron Cookware Set",
            "category": "cooking",
            "price": 89.99,
            "rating": 4.6,
            "tags": ["cast iron", "durable", "versatile", "traditional"],
            "age_range": [25, 70],
            "occasions": ["wedding", "housewarming", "fathers_day"]
        },
        {
            "id": "cook_003",
            "name": "Spice Rack with Premium Spices",
            "category": "cooking",
            "price": 54.99,
            "rating": 4.5,
            "tags": ["spices", "flavor", "organized", "gourmet"],
            "age_range": [20, 65],
            "occasions": ["housewarming", "cooking", "birthday"]
        },
        {
            "id": "cook_004",
            "name": "Fermentation Starter Kit",
            "category": "cooking",
            "price": 39.99,
            "rating": 4.4,
            "tags": ["fermentation", "healthy", "traditional", "diy"],
            "age_range": [25, 60],
            "occasions": ["health", "birthday", "cooking"]
        },
        
        # Art (expanded)
        {
            "id": "art_001",
            "name": "Professional Art Supply Set",
            "category": "art",
            "price": 79.99,
            "rating": 4.6,
            "tags": ["art", "creative", "drawing", "professional"],
            "age_range": [12, 65],
            "occasions": ["birthday", "graduation", "christmas"]
        },
        {
            "id": "art_002",
            "name": "Watercolor Painting Kit",
            "category": "art",
            "price": 49.99,
            "rating": 4.5,
            "tags": ["watercolor", "painting", "creative", "relaxing"],
            "age_range": [16, 70],
            "occasions": ["birthday", "art", "relaxation"]
        },
        {
            "id": "art_003",
            "name": "Adult Coloring Book Collection",
            "category": "art",
            "price": 24.99,
            "rating": 4.3,
            "tags": ["coloring", "relaxation", "mindful", "stress-relief"],
            "age_range": [18, 70],
            "occasions": ["stress_relief", "birthday", "wellness"]
        },
        {
            "id": "art_004",
            "name": "Pottery Making Kit",
            "category": "art",
            "price": 69.99,
            "rating": 4.4,
            "tags": ["pottery", "clay", "hands-on", "creative"],
            "age_range": [16, 60],
            "occasions": ["birthday", "hobby", "creative"]
        },
        
        # Wellness (expanded significantly)
        {
            "id": "wellness_001",
            "name": "Aromatherapy Essential Oil Set",
            "category": "wellness",
            "price": 59.99,
            "rating": 4.7,
            "tags": ["aromatherapy", "essential oils", "relaxation", "natural"],
            "age_range": [20, 65],
            "occasions": ["mothers_day", "birthday", "wellness", "relaxation"]
        },
        {
            "id": "wellness_002",
            "name": "Meditation Cushion Set",
            "category": "wellness",
            "price": 45.99,
            "rating": 4.5,
            "tags": ["meditation", "mindfulness", "comfort", "zen"],
            "age_range": [18, 70],
            "occasions": ["new_year", "wellness", "birthday"]
        },
        {
            "id": "wellness_003",
            "name": "Spa Day Experience Voucher",
            "category": "wellness",
            "price": 129.99,
            "rating": 4.6,
            "tags": ["spa", "relaxation", "pampering", "experience"],
            "age_range": [18, 65],
            "occasions": ["mothers_day", "birthday", "anniversary"]
        },
        {
            "id": "wellness_004",
            "name": "Herbal Tea Collection",
            "category": "wellness",
            "price": 34.99,
            "rating": 4.4,
            "tags": ["tea", "herbal", "calming", "natural"],
            "age_range": [20, 70],
            "occasions": ["wellness", "birthday", "any"]
        },
        {
            "id": "wellness_005",
            "name": "Yoga Starter Kit",
            "category": "wellness",
            "price": 79.99,
            "rating": 4.5,
            "tags": ["yoga", "fitness", "flexibility", "mindful"],
            "age_range": [18, 60],
            "occasions": ["new_year", "wellness", "birthday"]
        },
        
        # Fitness (expanded)
        {
            "id": "fitness_001",
            "name": "Smart Fitness Watch",
            "category": "fitness",
            "price": 199.99,
            "rating": 4.3,
            "tags": ["fitness", "health", "smart", "tracking"],
            "age_range": [18, 55],
            "occasions": ["birthday", "new_year", "graduation"]
        },
        {
            "id": "fitness_002",
            "name": "Resistance Bands Training Kit",
            "category": "fitness",
            "price": 29.99,
            "rating": 4.4,
            "tags": ["fitness", "home_gym", "portable", "strength"],
            "age_range": [16, 65],
            "occasions": ["new_year", "birthday", "fitness"]
        },
        {
            "id": "fitness_003",
            "name": "Foam Roller Recovery Set",
            "category": "fitness",
            "price": 39.99,
            "rating": 4.6,
            "tags": ["recovery", "massage", "muscle", "therapy"],
            "age_range": [18, 60],
            "occasions": ["fitness", "birthday", "recovery"]
        },
        
        # Outdoor (new category)
        {
            "id": "outdoor_001",
            "name": "Hiking Backpack",
            "category": "outdoor",
            "price": 89.99,
            "rating": 4.6,
            "tags": ["hiking", "outdoor", "adventure", "durable"],
            "age_range": [18, 55],
            "occasions": ["birthday", "graduation", "adventure"]
        },
        {
            "id": "outdoor_002",
            "name": "Camping Gear Starter Kit",
            "category": "outdoor",
            "price": 149.99,
            "rating": 4.5,
            "tags": ["camping", "outdoor", "adventure", "survival"],
            "age_range": [16, 50],
            "occasions": ["birthday", "graduation", "outdoor"]
        },
        {
            "id": "outdoor_003",
            "name": "Bird Watching Kit",
            "category": "outdoor",
            "price": 69.99,
            "rating": 4.4,
            "tags": ["birds", "nature", "observation", "peaceful"],
            "age_range": [25, 70],
            "occasions": ["birthday", "nature", "retirement"]
        },
        
        # Home (expanded)
        {
            "id": "home_001",
            "name": "Luxury Throw Blanket",
            "category": "home",
            "price": 69.99,
            "rating": 4.6,
            "tags": ["cozy", "soft", "comfort", "luxury"],
            "age_range": [18, 70],
            "occasions": ["any", "comfort", "winter"]
        },
        {
            "id": "home_002",
            "name": "Artisan Candle Set",
            "category": "home",
            "price": 45.99,
            "rating": 4.5,
            "tags": ["candles", "scented", "ambiance", "relaxing"],
            "age_range": [18, 65],
            "occasions": ["any", "relaxation", "ambiance"]
        },
        {
            "id": "home_003",
            "name": "Smart Plant Monitoring System",
            "category": "home",
            "price": 79.99,
            "rating": 4.3,
            "tags": ["plants", "smart", "monitoring", "tech"],
            "age_range": [25, 55],
            "occasions": ["housewarming", "tech", "gardening"]
        },
        
        # Food (expanded)
        {
            "id": "food_001",
            "name": "Gourmet Tea Sampler",
            "category": "food",
            "price": 39.99,
            "rating": 4.6,
            "tags": ["tea", "gourmet", "variety", "relaxation"],
            "age_range": [20, 70],
            "occasions": ["any", "tea", "relaxation"]
        },
        {
            "id": "food_002",
            "name": "Artisan Chocolate Box",
            "category": "food",
            "price": 49.99,
            "rating": 4.7,
            "tags": ["chocolate", "luxury", "sweet", "indulgent"],
            "age_range": [16, 70],
            "occasions": ["valentine", "anniversary", "birthday"]
        },
        {
            "id": "food_003",
            "name": "Specialty Coffee Bean Set",
            "category": "food",
            "price": 44.99,
            "rating": 4.5,
            "tags": ["coffee", "gourmet", "morning", "energy"],
            "age_range": [20, 65],
            "occasions": ["coffee_lover", "birthday", "morning"]
        },
        {
            "id": "food_004",
            "name": "Wine Tasting Experience",
            "category": "food",
            "price": 89.99,
            "rating": 4.4,
            "tags": ["wine", "tasting", "experience", "sophisticated"],
            "age_range": [25, 70],
            "occasions": ["anniversary", "appreciation", "sophisticated"]
        },
        
        # Experience (new category)
        {
            "id": "exp_001",
            "name": "Online Learning Course Subscription",
            "category": "experience",
            "price": 99.99,
            "rating": 4.4,
            "tags": ["learning", "online", "skill", "education"],
            "age_range": [16, 65],
            "occasions": ["graduation", "new_year", "career"]
        },
        {
            "id": "exp_002",
            "name": "Photography Workshop Voucher",
            "category": "experience",
            "price": 149.99,
            "rating": 4.5,
            "tags": ["photography", "workshop", "creative", "skill"],
            "age_range": [18, 60],
            "occasions": ["birthday", "creative", "hobby"]
        },
        {
            "id": "exp_003",
            "name": "Cooking Class Experience",
            "category": "experience",
            "price": 119.99,
            "rating": 4.6,
            "tags": ["cooking", "class", "hands-on", "culinary"],
            "age_range": [20, 65],
            "occasions": ["birthday", "cooking", "date"]
        },
        
        # Gaming (for younger demographics)
        {
            "id": "game_001",
            "name": "Gaming Mechanical Keyboard",
            "category": "gaming",
            "price": 129.99,
            "rating": 4.6,
            "tags": ["gaming", "rgb", "mechanical", "responsive"],
            "age_range": [16, 35],
            "occasions": ["birthday", "christmas", "gaming"]
        },
        {
            "id": "game_002",
            "name": "Board Game Collection",
            "category": "gaming",
            "price": 59.99,
            "rating": 4.5,
            "tags": ["board games", "family", "social", "fun"],
            "age_range": [12, 60],
            "occasions": ["birthday", "family", "christmas"]
        },
        
        # Fashion (for style-conscious users)
        {
            "id": "fashion_001",
            "name": "Leather Crossbody Bag",
            "category": "fashion",
            "price": 89.99,
            "rating": 4.4,
            "tags": ["leather", "stylish", "practical", "accessory"],
            "age_range": [18, 50],
            "occasions": ["birthday", "graduation", "promotion"]
        },
        {
            "id": "fashion_002",
            "name": "Minimalist Watch",
            "category": "fashion",
            "price": 119.99,
            "rating": 4.5,
            "tags": ["watch", "minimalist", "elegant", "timeless"],
            "age_range": [20, 60],
            "occasions": ["graduation", "promotion", "anniversary"]
        }
    ]
    
    # Calculate metadata
    categories = list(set(gift["category"] for gift in enhanced_gifts))
    prices = [gift["price"] for gift in enhanced_gifts]
    
    metadata = {
        "total_gifts": len(enhanced_gifts),
        "categories": sorted(categories),
        "category_counts": {cat: sum(1 for gift in enhanced_gifts if gift["category"] == cat) 
                          for cat in categories},
        "price_range": {
            "min": min(prices),
            "max": max(prices),
            "avg": round(sum(prices) / len(prices), 2)
        },
        "created": datetime.now().isoformat(),
        "improvements": [
            "Added comprehensive gardening category (4 items)",
            "Expanded books category (5 items) with diverse topics",
            "Added dedicated cooking category (4 items)",
            "Expanded art category (4 items) with various mediums",
            "Significantly expanded wellness category (5 items)",
            "Added outdoor category (3 items) for adventure lovers",
            "Added experience category (3 items) for memorable gifts",
            "Reduced technology dominance while keeping essential items",
            "Better age range coverage across all categories",
            "More occasion-specific tagging for better matching"
        ]
    }
    
    catalog = {
        "gifts": enhanced_gifts,
        "metadata": metadata
    }
    
    return catalog


def save_enhanced_catalog():
    """Save the enhanced catalog to file"""
    catalog = create_enhanced_gift_catalog()
    
    # Save main catalog
    with open("data/enhanced_realistic_gift_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    
    # Also update the original file for compatibility
    with open("data/realistic_gift_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    
    print("✅ Enhanced gift catalog created successfully!")
    print(f"📦 Total gifts: {catalog['metadata']['total_gifts']}")
    print(f"🏷️ Categories: {len(catalog['metadata']['categories'])}")
    print(f"💰 Price range: ${catalog['metadata']['price_range']['min']:.2f} - ${catalog['metadata']['price_range']['max']:.2f}")
    
    print(f"\n📊 Category distribution:")
    for category, count in sorted(catalog['metadata']['category_counts'].items()):
        percentage = (count / catalog['metadata']['total_gifts']) * 100
        print(f"  • {category}: {count} items ({percentage:.1f}%)")
    
    print(f"\n🎯 Key improvements:")
    for improvement in catalog['metadata']['improvements']:
        print(f"  • {improvement}")
    
    return catalog


if __name__ == "__main__":
    save_enhanced_catalog()