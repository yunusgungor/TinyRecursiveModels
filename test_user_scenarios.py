#!/usr/bin/env python3
"""
User Scenarios Validation Test
"""

import json

def test_user_scenarios():
    """Test user scenarios file"""
    
    print("🧪 User Scenarios Doğrulama Testi")
    print("=" * 60)
    
    # Load data
    with open('data/user_scenarios.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    metadata = data['metadata']
    
    # Test 1: Scenario count
    assert len(scenarios) == 100, "Senaryo sayısı 100 olmalı"
    print("✅ Test 1: Senaryo sayısı doğru (100)")
    
    # Test 2: Required fields
    required_fields = ['id', 'profile', 'expected_categories', 'expected_tools']
    for scenario in scenarios[:5]:  # İlk 5'i kontrol et
        for field in required_fields:
            assert field in scenario, f"'{field}' alanı eksik"
    print("✅ Test 2: Gerekli alanlar mevcut")
    
    # Test 3: Profile fields
    profile_fields = ['age', 'hobbies', 'relationship', 'budget', 'occasion', 'preferences']
    for scenario in scenarios[:5]:
        profile = scenario['profile']
        for field in profile_fields:
            assert field in profile, f"Profile '{field}' alanı eksik"
    print("✅ Test 3: Profil alanları doğru")
    
    # Test 4: Age range
    ages = [s['profile']['age'] for s in scenarios]
    assert all(16 <= age <= 70 for age in ages), "Yaş aralığı 16-70 olmalı"
    print(f"✅ Test 4: Yaş aralığı doğru ({min(ages)}-{max(ages)})")
    
    # Test 5: Budget range
    budgets = [s['profile']['budget'] for s in scenarios]
    assert all(30 <= budget <= 300 for budget in budgets), "Bütçe aralığı 30-300 olmalı"
    print(f"✅ Test 5: Bütçe aralığı doğru ({min(budgets):.2f}-{max(budgets):.2f} TL)")
    
    # Test 6: Metadata (matching expanded_user_scenarios.json format)
    assert metadata['total_scenarios'] == 100, "Metadata senaryo sayısı yanlış"
    assert 'generation_method' in metadata, "Generation method metadata eksik"
    assert 'coverage' in metadata, "Coverage metadata eksik"
    assert metadata['version'] == "2.0", "Version yanlış"
    print("✅ Test 6: Metadata doğru (v2.0 format)")
    
    # Test 7: Unique IDs (format: scenario_001, scenario_002, etc.)
    ids = [s['id'] for s in scenarios]
    assert len(ids) == len(set(ids)), "Duplicate ID'ler var"
    assert all(id.startswith('scenario_') for id in ids), "ID formatı yanlış"
    print("✅ Test 7: Tüm ID'ler benzersiz ve doğru formatta")
    
    # Test 8: Expected tools
    for scenario in scenarios[:10]:
        tools = scenario['expected_tools']
        assert len(tools) > 0, "En az 1 tool olmalı"
        assert 'review_analysis' in tools or 'price_comparison' in tools, "Temel tool'lar eksik"
    print("✅ Test 8: Expected tools doğru")
    
    print("\n" + "=" * 60)
    print("🎉 Tüm testler başarılı!")
    print("=" * 60)
    
    # İstatistikler
    print(f"\n📊 İstatistikler:")
    print(f"  • Toplam Senaryo: {len(scenarios)}")
    print(f"  • Yaş Ortalaması: {sum(ages)/len(ages):.1f}")
    print(f"  • Bütçe Ortalaması: {sum(budgets)/len(budgets):.2f} TL")
    print(f"  • Benzersiz İlişkiler: {len(set(s['profile']['relationship'] for s in scenarios))}")
    print(f"  • Benzersiz Özel Günler: {len(set(s['profile']['occasion'] for s in scenarios))}")
    
    # Örnek senaryo
    print(f"\n📝 Örnek Senaryo:")
    example = scenarios[0]
    print(f"  ID: {example['id']}")
    print(f"  Yaş: {example['profile']['age']}")
    print(f"  Hobiler: {', '.join(example['profile']['hobbies'][:3])}")
    print(f"  İlişki: {example['profile']['relationship']}")
    print(f"  Bütçe: {example['profile']['budget']:.2f} TL")
    print(f"  Özel Gün: {example['profile']['occasion']}")
    print(f"  Tercihler: {', '.join(example['profile']['preferences'][:3])}")

if __name__ == "__main__":
    test_user_scenarios()
