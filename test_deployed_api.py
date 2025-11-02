#!/usr/bin/env python3
"""
Simple test script for deployed API
"""

import requests
import json
import time

def test_api():
    """Test the deployed API"""
    print("🧪 TESTING DEPLOYED GIFT RECOMMENDATION API")
    print("=" * 60)
    
    api_url = "http://localhost:8000"
    
    # Test data
    test_request = {
        "user_profile": {
            "age": 28,
            "hobbies": ["technology", "fitness", "coffee"],
            "relationship": "friend",
            "budget": 150.0,
            "occasion": "birthday",
            "personality_traits": ["trendy", "practical", "tech-savvy"]
        },
        "max_recommendations": 3,
        "include_explanations": True,
        "include_alternatives": True
    }
    
    print("🎁 Test User Profile:")
    print(f"  Age: {test_request['user_profile']['age']}")
    print(f"  Hobbies: {test_request['user_profile']['hobbies']}")
    print(f"  Budget: ${test_request['user_profile']['budget']}")
    print(f"  Occasion: {test_request['user_profile']['occasion']}")
    
    print(f"\n🚀 Sending recommendation request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{api_url}/recommend",
            json=test_request,
            timeout=30
        )
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Request successful!")
            print(f"⏱️ Response time: {response_time:.0f}ms")
            print(f"🔧 Server processing: {result['processing_time_ms']:.0f}ms")
            
            print(f"\n🎁 Recommendations:")
            for i, gift in enumerate(result['recommendations'], 1):
                print(f"  {i}. {gift['name']}")
                print(f"     Category: {gift['category']}")
                print(f"     Price: ${gift['price']:.2f}")
                print(f"     Confidence: {gift['confidence']:.3f}")
                print(f"     Rating: {gift['rating']}/5.0")
                if gift.get('explanation'):
                    print(f"     Reason: {gift['explanation']}")
                print()
            
            print(f"🛠️ Tools Used: {', '.join(result['selected_tools'])}")
            
            print(f"\n🏷️ Top Categories:")
            sorted_categories = sorted(result['category_scores'].items(), 
                                     key=lambda x: x[1], reverse=True)
            for category, score in sorted_categories[:5]:
                print(f"  • {category}: {score:.3f}")
            
            if result.get('alternatives'):
                print(f"\n🔄 Alternative Suggestions:")
                for alt in result['alternatives']:
                    print(f"  • {alt['name']} ({alt['category']}) - ${alt['price']:.2f}")
            
            return True
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

if __name__ == "__main__":
    print("⚠️ Make sure the API is running first:")
    print("   python3 production/gift_recommendation_api.py")
    print("\nPress Enter when API is ready...")
    input()
    
    success = test_api()
    
    if success:
        print(f"\n🎉 API TEST SUCCESSFUL!")
        print("✅ Deployed model is working perfectly!")
    else:
        print(f"\n❌ API test failed")
        print("🔧 Check API logs for issues")