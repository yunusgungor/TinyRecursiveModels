#!/usr/bin/env python3
"""
SDV Kullanım Örnekleri
Bu dosya SDV'nin farklı kullanım senaryolarını gösterir
"""
import pandas as pd
import json
from pathlib import Path

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.constraints import Range, Inequality
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("⚠️  SDV yüklü değil. Kurulum: pip install sdv")
    exit(1)


def example_1_basic_synthesis():
    """Örnek 1: Temel sentetik veri üretimi"""
    print("\n" + "="*60)
    print("📚 Örnek 1: Temel Sentetik Veri Üretimi")
    print("="*60)
    
    # Basit bir veri seti oluştur
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [30000, 45000, 55000, 65000, 75000, 85000],
        'category': ['A', 'B', 'A', 'C', 'B', 'C']
    })
    
    print("\n📊 Orijinal veri:")
    print(data)
    
    # Metadata oluştur
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    # Synthesizer oluştur ve eğit
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    
    # Sentetik veri üret
    synthetic = synthesizer.sample(num_rows=10)
    
    print("\n🎲 Sentetik veri:")
    print(synthetic)
    
    return synthetic


def example_2_with_constraints():
    """Örnek 2: Kısıtlamalarla veri üretimi"""
    print("\n" + "="*60)
    print("📚 Örnek 2: Kısıtlamalarla Veri Üretimi")
    print("="*60)
    
    # Veri oluştur
    data = pd.DataFrame({
        'price': [10.0, 25.0, 50.0, 75.0, 100.0],
        'discount_price': [8.0, 20.0, 40.0, 60.0, 80.0],
        'rating': [4.0, 4.5, 3.5, 4.8, 4.2]
    })
    
    print("\n📊 Orijinal veri:")
    print(data)
    
    # Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    # Kısıtlamalar ekle
    constraints = [
        # İndirimli fiyat normal fiyattan düşük olmalı
        Inequality(
            low_column_name='discount_price',
            high_column_name='price'
        ),
        # Rating 1-5 arasında olmalı
        Range(
            column_name='rating',
            low_value=1.0,
            high_value=5.0,
            strict_boundaries=True
        )
    ]
    
    # Synthesizer oluştur
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True
    )
    
    # Kısıtlamaları ekle
    synthesizer.add_constraints(constraints)
    
    # Eğit ve üret
    synthesizer.fit(data)
    synthetic = synthesizer.sample(num_rows=8)
    
    print("\n🎲 Sentetik veri (kısıtlamalarla):")
    print(synthetic)
    
    # Kısıtlamaları kontrol et
    print("\n✅ Kısıtlama kontrolü:")
    print(f"  Tüm discount_price < price: {(synthetic['discount_price'] < synthetic['price']).all()}")
    print(f"  Tüm rating 1-5 arası: {synthetic['rating'].between(1, 5).all()}")
    
    return synthetic


def example_3_quality_evaluation():
    """Örnek 3: Kalite değerlendirmesi"""
    print("\n" + "="*60)
    print("📚 Örnek 3: Kalite Değerlendirmesi")
    print("="*60)
    
    # Gerçek veri yükle
    with open("data/realistic_gift_catalog.json", 'r') as f:
        gift_data = json.load(f)
    
    # DataFrame'e çevir
    gifts = gift_data['gifts']
    real_data = pd.DataFrame([{
        'price': g['price'],
        'rating': g['rating'],
        'age_min': g['age_range'][0],
        'age_max': g['age_range'][1]
    } for g in gifts])
    
    print(f"\n📊 Gerçek veri: {len(real_data)} örnek")
    print(real_data.describe())
    
    # Synthesizer eğit
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    
    # Sentetik veri üret
    synthetic_data = synthesizer.sample(num_rows=len(real_data))
    
    print(f"\n🎲 Sentetik veri: {len(synthetic_data)} örnek")
    print(synthetic_data.describe())
    
    # Kalite değerlendirmesi
    print("\n📊 Kalite değerlendirmesi yapılıyor...")
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    print(f"\n🎯 Genel Kalite Skoru: {quality_report.get_score():.2%}")
    
    # Detaylı rapor
    details = quality_report.get_details()
    print("\n📈 Detaylı Metrikler:")
    print(details)
    
    return quality_report


def example_4_conditional_sampling():
    """Örnek 4: Koşullu örnekleme"""
    print("\n" + "="*60)
    print("📚 Örnek 4: Koşullu Örnekleme")
    print("="*60)
    
    # Veri oluştur
    data = pd.DataFrame({
        'category': ['tech', 'home', 'fashion', 'tech', 'home', 'fashion'] * 3,
        'price': [100, 50, 75, 120, 45, 80, 110, 55, 70, 95, 60, 85, 105, 48, 78, 115, 52, 82],
        'rating': [4.5, 4.0, 4.2, 4.6, 3.9, 4.3, 4.4, 4.1, 4.0, 4.7, 3.8, 4.2, 4.5, 4.0, 4.1, 4.6, 3.9, 4.3]
    })
    
    print("\n📊 Orijinal veri dağılımı:")
    print(data['category'].value_counts())
    
    # Synthesizer eğit
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata.update_column('category', sdtype='categorical')
    
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    
    # Sadece 'tech' kategorisi için veri üret
    print("\n🎯 Sadece 'tech' kategorisi için 10 örnek üretiliyor...")
    conditions = pd.DataFrame({
        'category': ['tech'] * 10
    })
    
    synthetic_tech = synthesizer.sample_from_conditions(conditions)
    
    print("\n🎲 Üretilen tech ürünleri:")
    print(synthetic_tech)
    print(f"\nOrtalama fiyat: ${synthetic_tech['price'].mean():.2f}")
    print(f"Ortalama rating: {synthetic_tech['rating'].mean():.2f}")
    
    return synthetic_tech


def example_5_compare_methods():
    """Örnek 5: Farklı yöntemleri karşılaştırma"""
    print("\n" + "="*60)
    print("📚 Örnek 5: Farklı Synthesizer Yöntemlerini Karşılaştırma")
    print("="*60)
    
    # Basit veri
    data = pd.DataFrame({
        'value1': [10, 20, 30, 40, 50, 60, 70, 80],
        'value2': [15, 25, 35, 45, 55, 65, 75, 85],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    print("\n📊 Orijinal veri:")
    print(data)
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    results = {}
    
    # 1. Gaussian Copula
    print("\n🔧 Gaussian Copula eğitiliyor...")
    gaussian = GaussianCopulaSynthesizer(metadata)
    gaussian.fit(data)
    results['Gaussian'] = gaussian.sample(num_rows=10)
    
    # 2. CTGAN
    print("🔧 CTGAN eğitiliyor (bu biraz zaman alabilir)...")
    ctgan = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    ctgan.fit(data)
    results['CTGAN'] = ctgan.sample(num_rows=10)
    
    # Sonuçları karşılaştır
    print("\n📊 Sonuçlar:")
    for method, synthetic in results.items():
        print(f"\n{method}:")
        print(synthetic.describe())
        
        # Kalite skoru
        quality = evaluate_quality(data, synthetic, metadata)
        print(f"Kalite Skoru: {quality.get_score():.2%}")
    
    return results


def main():
    """Tüm örnekleri çalıştır"""
    print("🎁 SDV Kullanım Örnekleri")
    print("=" * 60)
    
    if not SDV_AVAILABLE:
        return
    
    # Veri klasörünü oluştur
    Path("data").mkdir(exist_ok=True)
    
    # Örnekleri çalıştır
    try:
        example_1_basic_synthesis()
        example_2_with_constraints()
        
        # Bu örnek için gerçek veri gerekli
        if Path("data/realistic_gift_catalog.json").exists():
            example_3_quality_evaluation()
        else:
            print("\n⚠️  Örnek 3 için önce: python create_gift_data.py")
        
        example_4_conditional_sampling()
        example_5_compare_methods()
        
        print("\n" + "="*60)
        print("✅ Tüm örnekler tamamlandı!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
