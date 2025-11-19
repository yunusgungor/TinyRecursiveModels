"""
Test script for User Scenario Generator
Tests scenario generation with real scraped data
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scraping.services.user_scenario_generator import UserScenarioGenerator


async def test_scenario_generator():
    """Test user scenario generator with real data"""
    
    print("="*60)
    print("Testing User Scenario Generator")
    print("="*60)
    
    # Configuration
    config = {
        'api_key_env': 'GEMINI_API_KEY',
        'model': 'gemini-1.5-flash',
        'max_requests_per_day': 1000,
        'retry_attempts': 3,
        'retry_delay': 2
    }
    
    # Paths
    gift_catalog_path = "data/scraped_gift_catalog.json"
    output_path = "data/test_user_scenarios.json"
    
    # Check if gift catalog exists
    if not Path(gift_catalog_path).exists():
        print(f"\n❌ Gift catalog not found: {gift_catalog_path}")
        print("Please run the scraping pipeline first:")
        print("  python scripts/scraping.py")
        return
    
    # Load and display catalog info
    with open(gift_catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    print(f"\n📦 Gift Catalog Info:")
    print(f"  Total gifts: {catalog['metadata']['total_gifts']}")
    print(f"  Categories: {len(catalog['metadata']['categories'])}")
    print(f"  Price range: {catalog['metadata']['price_range']['min']:.2f} - {catalog['metadata']['price_range']['max']:.2f} TL")
    
    # Initialize generator
    print(f"\n🔧 Initializing generator...")
    generator = UserScenarioGenerator(config, gift_catalog_path)
    
    # Generate scenarios
    num_scenarios = 10
    print(f"\n🎯 Generating {num_scenarios} test scenarios...")
    scenarios = await generator.generate_scenarios(num_scenarios)
    
    # Display sample scenarios
    print(f"\n✅ Generated {len(scenarios)} scenarios")
    print("\n📋 Sample Scenarios:")
    for i, scenario in enumerate(scenarios[:3], 1):
        profile = scenario['profile']
        print(f"\n  Scenario {i}:")
        print(f"    Age: {profile['age']}")
        print(f"    Hobbies: {', '.join(profile['hobbies'])}")
        print(f"    Relationship: {profile['relationship']}")
        print(f"    Budget: {profile['budget']:.2f} TL")
        print(f"    Occasion: {profile['occasion']}")
        print(f"    Preferences: {', '.join(profile['preferences'])}")
        print(f"    Expected categories: {', '.join(scenario['expected_categories'])}")
    
    # Save scenarios
    print(f"\n💾 Saving scenarios to {output_path}...")
    await generator.save_scenarios(scenarios, output_path)
    
    # Display statistics
    stats = generator.get_stats()
    print(f"\n📊 Statistics:")
    print(f"  Generation method: {'AI (Gemini)' if stats['enabled'] else 'Fallback (Rule-based)'}")
    print(f"  Requests made: {stats['requests_made']}")
    print(f"  Requests remaining: {stats['requests_remaining']}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_scenario_generator())
