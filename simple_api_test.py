#!/usr/bin/env python3
"""
Simple API test
"""

import requests
import json

def test_api():
    api_url = "http://localhost:8000"
    
    # Health check
    print("🔍 Health check...")
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return False
    except:
        print("❌ Cannot connect to API")
        return False
    
    # Test recommendation
    print("\n🎁 Testing recommendation...")
    test_data = {
        "user_profile": {
            "age": 28,
            "hobbies": ["technology", "fitness"],
            "relationship": "friend", 
            "budget": 150.0,
            "occasion": "birthday",
            "personality_traits": ["trendy", "practical"]
        },
        "max_recommendations": 3
    }
    
    try:
        response = requests.post(f"{api_url}/recommend", json=test_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Recommendation successful!")
            print(f"⏱️ Processing time: {result['processing_time_ms']:.0f}ms")
            print(f"🎁 Got {len(result['recommendations'])} recommendations")
            
            for i, gift in enumerate(result['recommendations'], 1):
                print(f"  {i}. {gift['name']} ({gift['category']}) - ${gift['price']:.2f}")
            
            print(f"🛠️ Tools used: {', '.join(result['selected_tools'])}")
            return True
        else:
            print(f"❌ Recommendation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    print(f"\n{'🎉 SUCCESS!' if success else '❌ FAILED'}")