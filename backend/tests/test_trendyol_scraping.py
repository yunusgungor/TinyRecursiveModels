#!/usr/bin/env python3
"""
Test script for Trendyol Scraping Service
Tests the integration between backend and scraping modules
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.trendyol_scraping_service import (
    TrendyolScrapingService,
    get_trendyol_scraping_service
)


async def test_scraping_service():
    """Test the scraping service"""
    print("=" * 80)
    print("Trendyol Scraping Service Test")
    print("=" * 80)
    
    service = None
    try:
        # Initialize service
        print("\n1. Initializing service...")
        service = get_trendyol_scraping_service()
        print("✓ Service initialized")
        
        # Get rate limit info
        print("\n2. Rate limit info:")
        rate_info = service.get_rate_limit_info()
        for key, value in rate_info.items():
            print(f"   - {key}: {value}")
        
        # Test product search
        print("\n3. Testing product search...")
        print("   Scraping 5 products from 'elektronik' category...")
        print("   (This may take 15-30 seconds...)")
        
        products = await service.search_products(
            category="elektronik",
            keywords=["kulaklık"],
            max_results=5
        )
        
        print(f"\n✓ Scraped {len(products)} products")
        
        # Display products
        print("\n4. Product details:")
        for i, product in enumerate(products, 1):
            print(f"\n   Product {i}:")
            print(f"   - Name: {product.name[:60]}...")
            print(f"   - Price: {product.price} TL")
            print(f"   - Rating: {product.rating}/5.0")
            print(f"   - Category: {product.category}")
            print(f"   - In Stock: {product.in_stock}")
            print(f"   - URL: {product.product_url[:80]}...")
        
        # Test conversion to GiftItem
        if products:
            print("\n5. Testing GiftItem conversion...")
            first_product = products[0]
            gift_item = service.convert_to_gift_item(first_product)
            
            if gift_item:
                print("✓ Successfully converted to GiftItem:")
                print(f"   - ID: {gift_item.id}")
                print(f"   - Name: {gift_item.name[:60]}...")
                print(f"   - Price: {gift_item.price} TL")
                print(f"   - Tags: {gift_item.tags[:3]}")
            else:
                print("✗ Failed to convert to GiftItem")
        
        # Test cache
        print("\n6. Testing cache...")
        print("   Requesting same products again (should be cached)...")
        
        import time
        start = time.time()
        cached_products = await service.search_products(
            category="elektronik",
            keywords=["kulaklık"],
            max_results=5
        )
        duration = time.time() - start
        
        print(f"✓ Got {len(cached_products)} products in {duration:.2f}s")
        print(f"   (Cached response should be < 0.1s)")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if service:
            print("\n7. Cleaning up...")
            await service.close()
            print("✓ Service closed")
    
    return True


async def test_singleton():
    """Test that singleton pattern works"""
    print("\n" + "=" * 80)
    print("Testing Singleton Pattern")
    print("=" * 80)
    
    service1 = get_trendyol_scraping_service()
    service2 = get_trendyol_scraping_service()
    
    if service1 is service2:
        print("✓ Singleton pattern working correctly")
        await service1.close()
        return True
    else:
        print("✗ Singleton pattern not working")
        await service1.close()
        return False


async def main():
    """Run all tests"""
    print("\n🧪 Starting Trendyol Scraping Service Tests\n")
    
    # Test 1: Basic functionality
    success1 = await test_scraping_service()
    
    # Test 2: Singleton pattern
    success2 = await test_singleton()
    
    if success1 and success2:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
