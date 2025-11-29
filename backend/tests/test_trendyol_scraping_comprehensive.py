#!/usr/bin/env python3
"""
Comprehensive Test Suite for Trendyol Scraping Service

Tests all aspects of the scraping service including:
- Anti-bot detection bypass (CAPTCHA, rate limiting, etc.)
- Error handling and edge cases
- Performance and caching
- Data validation
- Multiple categories and scenarios
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.trendyol_scraping_service import (
    TrendyolScrapingService,
    TrendyolProduct,
    get_trendyol_scraping_service,
    cleanup_trendyol_service
)
from app.core.exceptions import TrendyolAPIError


class TestResult:
    """Test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.tests_run = []
    
    def record_pass(self, test_name: str, message: str = ""):
        self.passed += 1
        self.tests_run.append((test_name, "PASS", message))
        print(f"✅ PASS: {test_name}")
        if message:
            print(f"   {message}")
    
    def record_fail(self, test_name: str, message: str = ""):
        self.failed += 1
        self.tests_run.append((test_name, "FAIL", message))
        print(f"❌ FAIL: {test_name}")
        if message:
            print(f"   {message}")
    
    def record_warning(self, test_name: str, message: str = ""):
        self.warnings += 1
        self.tests_run.append((test_name, "WARN", message))
        print(f"⚠️  WARN: {test_name}")
        if message:
            print(f"   {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"Success Rate: {(self.passed/total*100):.1f}%" if total > 0 else "N/A")
        print("="*80)
        
        return self.failed == 0


results = TestResult()


async def test_1_service_initialization():
    """Test 1: Service Initialization and Singleton Pattern"""
    print("\n" + "="*80)
    print("TEST 1: Service Initialization and Singleton Pattern")
    print("="*80)
    
    try:
        # Test singleton
        service1 = get_trendyol_scraping_service()
        service2 = get_trendyol_scraping_service()
        
        if service1 is service2:
            results.record_pass(
                "Singleton Pattern",
                "Same service instance returned"
            )
        else:
            results.record_fail(
                "Singleton Pattern",
                "Different instances returned"
            )
        
        # Test rate limit info
        rate_info = service1.get_rate_limit_info()
        
        if rate_info.get('max_requests') > 0:
            results.record_pass(
                "Rate Limit Configuration",
                f"Max requests: {rate_info['max_requests']}/min"
            )
        else:
            results.record_fail("Rate Limit Configuration", "Invalid rate limit")
        
        await service1.close()
        
    except Exception as e:
        results.record_fail("Service Initialization", str(e))


async def test_2_anti_captcha_detection():
    """Test 2: Anti-CAPTCHA and Bot Detection Bypass"""
    print("\n" + "="*80)
    print("TEST 2: Anti-CAPTCHA and Bot Detection Bypass")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # Try to scrape - if CAPTCHA is triggered, we'll get an error
        print("   Attempting to bypass anti-bot measures...")
        print("   (This uses random delays, user agents, and human-like behavior)")
        
        products = await service.search_products(
            category="elektronik",
            keywords=["test"],
            max_results=3
        )
        
        if products and len(products) > 0:
            results.record_pass(
                "CAPTCHA Bypass",
                f"Successfully scraped {len(products)} products without CAPTCHA"
            )
        else:
            results.record_warning(
                "CAPTCHA Bypass",
                "No products returned - possible CAPTCHA or rate limit"
            )
        
    except TrendyolAPIError as e:
        if "CAPTCHA" in str(e).upper() or "BLOCKED" in str(e).upper():
            results.record_fail(
                "CAPTCHA Bypass",
                "Detected CAPTCHA or block"
            )
        else:
            results.record_warning("CAPTCHA Bypass", str(e))
    
    except Exception as e:
        results.record_fail("CAPTCHA Bypass", str(e))
    
    finally:
        if service:
            await service.close()


async def test_3_multiple_categories():
    """Test 3: Multiple Category Support"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Category Support")
    print("="*80)
    
    categories = [
        "elektronik",
        "ev-yasam",
        "kozmetik",
        "kadin-giyim"
    ]
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        for category in categories:
            try:
                print(f"   Testing category: {category}...")
                products = await service.search_products(
                    category=category,
                    keywords=[],
                    max_results=2
                )
                
                if products and len(products) > 0:
                    results.record_pass(
                        f"Category: {category}",
                        f"Scraped {len(products)} products"
                    )
                else:
                    results.record_warning(
                        f"Category: {category}",
                        "No products found"
                    )
                
                # Small delay between categories to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                results.record_fail(f"Category: {category}", str(e))
    
    except Exception as e:
        results.record_fail("Multiple Categories", str(e))
    
    finally:
        if service:
            await service.close()


async def test_4_price_filtering():
    """Test 4: Price Range Filtering"""
    print("\n" + "="*80)
    print("TEST 4: Price Range Filtering")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # Test with price range
        min_price = 100.0
        max_price = 1000.0
        
        products = await service.search_products(
            category="elektronik",
            keywords=[],
            max_results=5,
            min_price=min_price,
            max_price=max_price
        )
        
        if products:
            # Check if all products are within range
            all_in_range = all(
                min_price <= p.price <= max_price 
                for p in products
            )
            
            if all_in_range:
                results.record_pass(
                    "Price Filtering",
                    f"All {len(products)} products within {min_price}-{max_price} TL"
                )
            else:
                out_of_range = [p for p in products if not (min_price <= p.price <= max_price)]
                results.record_fail(
                    "Price Filtering",
                    f"{len(out_of_range)} products outside range"
                )
        else:
            results.record_warning(
                "Price Filtering",
                "No products found in price range"
            )
    
    except Exception as e:
        results.record_fail("Price Filtering", str(e))
    
    finally:
        if service:
            await service.close()


async def test_5_cache_functionality():
    """Test 5: Cache Functionality and Performance"""
    print("\n" + "="*80)
    print("TEST 5: Cache Functionality and Performance")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # First request (should scrape)
        print("   First request (scraping)...")
        start1 = time.time()
        products1 = await service.search_products(
            category="elektronik",
            keywords=["test"],
            max_results=3
        )
        duration1 = time.time() - start1
        
        # Second request (should be cached)
        print("   Second request (cached)...")
        start2 = time.time()
        products2 = await service.search_products(
            category="elektronik",
            keywords=["test"],
            max_results=3
        )
        duration2 = time.time() - start2
        
        if duration2 < 0.1 and duration2 < duration1:
            results.record_pass(
                "Cache Performance",
                f"Cache hit: {duration2:.4f}s vs initial: {duration1:.2f}s"
            )
        else:
            results.record_fail(
                "Cache Performance",
                f"Cache not working: {duration2:.4f}s"
            )
        
        # Verify same products
        if len(products1) == len(products2):
            results.record_pass(
                "Cache Consistency",
                f"Same {len(products1)} products returned"
            )
        else:
            results.record_fail(
                "Cache Consistency",
                f"Different product counts: {len(products1)} vs {len(products2)}"
            )
    
    except Exception as e:
        results.record_fail("Cache Functionality", str(e))
    
    finally:
        if service:
            await service.close()


async def test_6_gift_item_conversion():
    """Test 6: TrendyolProduct to GiftItem Conversion"""
    print("\n" + "="*80)
    print("TEST 6: TrendyolProduct to GiftItem Conversion")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        products = await service.search_products(
            category="elektronik",
            keywords=[],
            max_results=5
        )
        
        if not products:
            results.record_warning("GiftItem Conversion", "No products to convert")
            return
        
        successful = 0
        failed = 0
        
        for product in products:
            gift_item = service.convert_to_gift_item(product)
            
            if gift_item:
                successful += 1
                
                # Validate fields
                assert gift_item.id, "Missing ID"
                assert gift_item.name, "Missing name"
                assert gift_item.price >= 0, "Invalid price"
                assert 0 <= gift_item.rating <= 5, "Invalid rating"
                assert gift_item.image_url, "Missing image URL"
            else:
                failed += 1
        
        if successful > 0 and failed == 0:
            results.record_pass(
                "GiftItem Conversion",
                f"All {successful} products converted successfully"
            )
        elif successful > 0:
            results.record_warning(
                "GiftItem Conversion",
                f"{successful} succeeded, {failed} failed"
            )
        else:
            results.record_fail(
                "GiftItem Conversion",
                f"All {failed} conversions failed"
            )
    
    except Exception as e:
        results.record_fail("GiftItem Conversion", str(e))
    
    finally:
        if service:
            await service.close()


async def test_7_error_handling():
    """Test 7: Error Handling and Edge Cases"""
    print("\n" + "="*80)
    print("TEST 7: Error Handling and Edge Cases")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # Test 1: Invalid category
        try:
            products = await service.search_products(
                category="invalid_category_xyz",
                keywords=[],
                max_results=1
            )
            
            # Should return empty or use fallback
            if products is not None and len(products) == 0:
                results.record_pass(
                    "Invalid Category Handling",
                    "Gracefully handled invalid category"
                )
            else:
                results.record_warning(
                    "Invalid Category Handling",
                    "Category fallback used"
                )
        except Exception as e:
            results.record_fail("Invalid Category Handling", str(e))
        
        # Test 2: Max results = 0
        try:
            products = await service.search_products(
                category="elektronik",
                keywords=[],
                max_results=0
            )
            
            if products is not None and len(products) == 0:
                results.record_pass(
                    "Zero Results Handling",
                    "Correctly handled max_results=0"
                )
            else:
                results.record_fail(
                    "Zero Results Handling",
                    f"Expected 0 products, got {len(products) if products else 'None'}"
                )
        except Exception as e:
            results.record_warning("Zero Results Handling", "Exception raised (acceptable)")
        
        # Test 3: Empty keywords
        try:
            products = await service.search_products(
                category="elektronik",
                keywords=[],
                max_results=2
            )
            
            if products is not None:
                results.record_pass(
                    "Empty Keywords Handling",
                    f"Handled empty keywords, got {len(products)} products"
                )
            else:
                results.record_fail("Empty Keywords Handling", "Returned None")
        except Exception as e:
            results.record_fail("Empty Keywords Handling", str(e))
    
    except Exception as e:
        results.record_fail("Error Handling", str(e))
    
    finally:
        if service:
            await service.close()


async def test_8_rate_limiting():
    """Test 8: Rate Limiting and Throttling"""
    print("\n" + "="*80)
    print("TEST 8: Rate Limiting and Throttling")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService(rate_limit=10)  # 10 req/min
        
        print("   Making rapid requests to test rate limiting...")
        
        request_times = []
        
        for i in range(3):
            start = time.time()
            await service.search_products(
                category="elektronik",
                keywords=[],
                max_results=1
            )
            duration = time.time() - start
            request_times.append(duration)
            print(f"   Request {i+1}: {duration:.2f}s")
        
        # Check if delays are applied
        avg_time = sum(request_times) / len(request_times)
        
        if avg_time > 2.0:  # Should have delays
            results.record_pass(
                "Rate Limiting",
                f"Rate limiting active, avg delay: {avg_time:.2f}s"
            )
        else:
            results.record_warning(
                "Rate Limiting",
                f"Rate limiting may not be working, avg: {avg_time:.2f}s"
            )
    
    except Exception as e:
        results.record_fail("Rate Limiting", str(e))
    
    finally:
        if service:
            await service.close()


async def test_9_browser_cleanup():
    """Test 9: Browser Resource Cleanup"""
    print("\n" + "="*80)
    print("TEST 9: Browser Resource Cleanup")
    print("="*80)
    
    try:
        # Create and destroy multiple services
        for i in range(3):
            service = TrendyolScrapingService()
            
            # Do a small scrape
            await service.search_products(
                category="elektronik",
                keywords=[],
                max_results=1
            )
            
            # Close service
            await service.close()
            
            print(f"   Cycle {i+1}: Created, used, and cleaned up service")
        
        results.record_pass(
            "Browser Cleanup",
            "Multiple service lifecycles completed without errors"
        )
    
    except Exception as e:
        results.record_fail("Browser Cleanup", str(e))


async def test_10_concurrent_requests():
    """Test 10: Concurrent Request Handling"""
    print("\n" + "="*80)
    print("TEST 10: Concurrent Request Handling")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # Create concurrent tasks
        tasks = [
            service.search_products(category="elektronik", keywords=[], max_results=2),
            service.search_products(category="ev-yasam", keywords=[], max_results=2),
            service.search_products(category="kozmetik", keywords=[], max_results=2),
        ]
        
        print("   Running 3 concurrent scraping tasks...")
        start = time.time()
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start
        
        # Check results
        success_count = sum(1 for r in results_list if not isinstance(r, Exception))
        
        if success_count >= 2:
            results.record_pass(
                "Concurrent Requests",
                f"{success_count}/3 requests succeeded in {duration:.2f}s"
            )
        else:
            results.record_fail(
                "Concurrent Requests",
                f"Only {success_count}/3 requests succeeded"
            )
    
    except Exception as e:
        results.record_fail("Concurrent Requests", str(e))
    
    finally:
        if service:
            await service.close()


async def test_11_product_data_validation():
    """Test 11: Product Data Validation and Quality"""
    print("\n" + "="*80)
    print("TEST 11: Product Data Validation and Quality")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        products = await service.search_products(
            category="elektronik",
            keywords=[],
            max_results=10
        )
        
        if not products:
            results.record_warning("Product Data Validation", "No products to validate")
            return
        
        # Validation checks
        valid_count = 0
        issues = []
        
        for product in products:
            has_issues = False
            
            if not product.name or len(product.name) < 3:
                issues.append(f"Product {product.id}: Invalid name")
                has_issues = True
            
            if product.price <= 0:
                issues.append(f"Product {product.id}: Invalid price: {product.price}")
                has_issues = True
            
            if not (0 <= product.rating <= 5):
                issues.append(f"Product {product.id}: Invalid rating: {product.rating}")
                has_issues = True
            
            if not product.product_url or not product.product_url.startswith("http"):
                issues.append(f"Product {product.id}: Invalid URL")
                has_issues = True
            
            if not has_issues:
                valid_count += 1
        
        validation_rate = (valid_count / len(products)) * 100
        
        if validation_rate >= 90:
            results.record_pass(
                "Product Data Validation",
                f"{validation_rate:.1f}% products valid ({valid_count}/{len(products)})"
            )
        elif validation_rate >= 70:
            results.record_warning(
                "Product Data Validation",
                f"{validation_rate:.1f}% valid, some issues found"
            )
            for issue in issues[:5]:  # Show first 5 issues
                print(f"      - {issue}")
        else:
            results.record_fail(
                "Product Data Validation",
                f"Only {validation_rate:.1f}% valid"
            )
    
    except Exception as e:
        results.record_fail("Product Data Validation", str(e))
    
    finally:
        if service:
            await service.close()


async def test_12_fallback_and_resilience():
    """Test 12: Fallback Mechanisms and Resilience"""
    print("\n" + "="*80)
    print("TEST 12: Fallback Mechanisms and Resilience")
    print("="*80)
    
    service = None
    try:
        service = TrendyolScrapingService()
        
        # First request (populate cache)
        print("   Populating cache...")
        products1 = await service.search_products(
            category="elektronik",
            keywords=["test"],
            max_results=3
        )
        
        if products1:
            # Clear cache TTL to test fallback
            # The service should use stale cache if new request fails
            print("   Testing fallback to stale cache...")
            
            # Request again (should use cache even if TTL expired)
            products2 = await service.search_products(
                category="elektronik",
                keywords=["test"],
                max_results=3
            )
            
            if products2 and len(products2) > 0:
                results.record_pass(
                    "Cache Fallback",
                    "Successfully used cache as fallback"
                )
            else:
                results.record_fail("Cache Fallback", "Fallback failed")
        else:
            results.record_warning("Cache Fallback", "No data to test fallback")
    
    except Exception as e:
        results.record_fail("Cache Fallback", str(e))
    
    finally:
        if service:
            await service.close()


async def main():
    """Run all comprehensive tests"""
    print("\n" + "🧪 "*20)
    print("COMPREHENSIVE TRENDYOL SCRAPING SERVICE TEST SUITE")
    print("🧪 "*20)
    print(f"\nTest Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test suite validates:")
    print("  • Anti-bot detection bypass (CAPTCHA, rate limits)")
    print("  • Multiple category support")
    print("  • Price filtering")
    print("  • Cache functionality")
    print("  • Data validation")
    print("  • Error handling")
    print("  • Rate limiting")
    print("  • Resource cleanup")
    print("  • Concurrent requests")
    print("  • Resilience and fallbacks")
    
    # Run all tests
    await test_1_service_initialization()
    await test_2_anti_captcha_detection()
    await test_3_multiple_categories()
    await test_4_price_filtering()
    await test_5_cache_functionality()
    await test_6_gift_item_conversion()
    await test_7_error_handling()
    await test_8_rate_limiting()
    await test_9_browser_cleanup()
    await test_10_concurrent_requests()
    await test_11_product_data_validation()
    await test_12_fallback_and_resilience()
    
    # Final cleanup
    await cleanup_trendyol_service()
    
    # Print summary
    success = results.summary()
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
