#!/usr/bin/env python3
"""
Tamamen Öğrenilmiş Sentetik Veri Üretimi
Tüm bilgiler (isimler, tag'ler, fiyatlar vb.) scraped veriden öğrenilir
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.cag import Inequality
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("❌ SDV kurulu değil. Kurulum: conda activate sdv_env && pip install sdv")
    exit(1)


class FullyLearnedSyntheticGenerator:
    """Scraped veriden tamamen öğrenen sentetik veri üretici"""
    
    def __init__(
        self,
        gift_path: str = "data/gift_catalog.json",
        user_path: str = "data/user_scenarios.json"
    ):
        self.gift_path = Path(gift_path)
        self.user_path = Path(user_path)
        self.synthesizers = {}
        
        # Scraped veriden öğrenilecek bilgiler
        self.learned_data = {
            'product_names': {},      # Kategori -> [isimler]
            'tags': {},               # Kategori -> [tag'ler]
            'occasions': set(),       # Tüm occasion'lar
            'price_ranges': {},       # Kategori -> (min, max)
            'hobbies': set(),         # Tüm hobiler
            'preferences': set(),     # Tüm tercihler
            'relationships': set(),   # Tüm ilişkiler
            'occasions_user': set()   # Kullanıcı occasion'ları
        }
    
    @staticmethod
    def normalize_occasion(occasion: str) -> str:
        """Occasion'ı normalize et: küçük harf, alt çizgi, parantez temizle"""
        # Parantez içindeki açıklamaları kaldır
        if '(' in occasion:
            occasion = occasion.split('(')[0].strip()
        # Küçük harfe çevir ve boşlukları alt çizgiye çevir
        return occasion.lower().replace(' ', '_')
    
    def learn_from_scraped_data(self):
        """Scraped veriden tüm bilgileri öğren"""
        print("📚 Scraped veriden öğreniliyor...")
        
        # Hediye verisinden öğren
        with open(self.gift_path, 'r', encoding='utf-8') as f:
            gift_data = json.load(f)
        
        for gift in gift_data['gifts']:
            category = gift['category']
            
            # Unknown'ları atla
            if category == 'unknown':
                continue
            
            # İsimleri öğren
            if category not in self.learned_data['product_names']:
                self.learned_data['product_names'][category] = []
            self.learned_data['product_names'][category].append(gift['name'])
            
            # Tag'leri öğren
            if category not in self.learned_data['tags']:
                self.learned_data['tags'][category] = set()
            for tag in gift.get('tags', []):
                self.learned_data['tags'][category].add(tag)
            
            # Occasion'ları öğren (normalize edilmiş)
            for occasion in gift.get('occasions', []):
                normalized = self.normalize_occasion(occasion)
                self.learned_data['occasions'].add(normalized)
            
            # Fiyat aralıklarını öğren
            price = float(gift['price'])
            if price > 0 and price < 100000:  # Geçerli fiyatlar
                if category not in self.learned_data['price_ranges']:
                    self.learned_data['price_ranges'][category] = [price, price]
                else:
                    self.learned_data['price_ranges'][category][0] = min(
                        self.learned_data['price_ranges'][category][0], price
                    )
                    self.learned_data['price_ranges'][category][1] = max(
                        self.learned_data['price_ranges'][category][1], price
                    )
        
        # Kullanıcı verisinden öğren
        with open(self.user_path, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        self.learned_data['expected_categories'] = set()
        self.learned_data['expected_tools'] = set()
        
        for scenario in user_data['scenarios']:
            profile = scenario['profile']
            
            # Hobileri öğren
            for hobby in profile.get('hobbies', []):
                self.learned_data['hobbies'].add(hobby)
            
            # Tercihleri öğren
            for pref in profile.get('preferences', []):
                self.learned_data['preferences'].add(pref)
            
            # İlişkileri öğren
            self.learned_data['relationships'].add(profile['relationship'])
            
            # Occasion'ları öğren (normalize edilmiş)
            normalized = self.normalize_occasion(profile['occasion'])
            self.learned_data['occasions_user'].add(normalized)
            
            # Expected categories öğren
            for cat in scenario.get('expected_categories', []):
                self.learned_data['expected_categories'].add(cat)
            
            # Expected tools öğren
            for tool in scenario.get('expected_tools', []):
                self.learned_data['expected_tools'].add(tool)
        
        # Set'leri listeye çevir
        self.learned_data['occasions'] = list(self.learned_data['occasions'])
        self.learned_data['hobbies'] = list(self.learned_data['hobbies'])
        self.learned_data['preferences'] = list(self.learned_data['preferences'])
        self.learned_data['relationships'] = list(self.learned_data['relationships'])
        self.learned_data['occasions_user'] = list(self.learned_data['occasions_user'])
        self.learned_data['expected_categories'] = list(self.learned_data['expected_categories'])
        self.learned_data['expected_tools'] = list(self.learned_data['expected_tools'])
        
        # Tag'leri de listeye çevir
        for category in self.learned_data['tags']:
            self.learned_data['tags'][category] = list(self.learned_data['tags'][category])
        
        print(f"  ✅ Öğrenilen bilgiler:")
        print(f"    • {len(self.learned_data['product_names'])} kategoride ürün isimleri")
        print(f"    • {len(self.learned_data['occasions'])} occasion")
        print(f"    • {len(self.learned_data['hobbies'])} hobi")
        print(f"    • {len(self.learned_data['preferences'])} tercih")
        print(f"    • {len(self.learned_data['relationships'])} ilişki tipi")
        print(f"    • {len(self.learned_data['expected_categories'])} beklenen kategori")
        print(f"    • {len(self.learned_data['expected_tools'])} beklenen tool")
    
    def load_and_clean_gifts(self) -> pd.DataFrame:
        """Hediye verisini yükle ve temizle"""
        with open(self.gift_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gifts = data['gifts']
        df_data = []
        removed = {'unknown': 0, 'invalid_price': 0, 'invalid_age': 0}
        
        for gift in gifts:
            # Unknown kategorisini atla
            if gift['category'] == 'unknown':
                removed['unknown'] += 1
                continue
            
            # Geçersiz fiyatları atla
            price = float(gift['price'])
            if price <= 0 or price > 100000:
                removed['invalid_price'] += 1
                continue
            
            # Geçersiz yaş aralıklarını atla
            age_min = int(gift['age_range'][0])
            age_max = int(gift['age_range'][1])
            if age_min >= age_max or age_min < 0 or age_max > 100:
                removed['invalid_age'] += 1
                continue
            
            row = {
                'price': price,
                'rating': float(gift.get('rating', 0.0)),
                'age_min': age_min,
                'age_max': age_max,
                'category': gift['category'],
                'num_tags': len(gift.get('tags', [])),
                'num_occasions': len(gift.get('occasions', [])),
                'age_range_width': age_max - age_min,
                'price_category': self._categorize_price(price),
                'log_price': np.log1p(price)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        print(f"  📦 {len(df)} temiz hediye yüklendi")
        print(f"  🗑️  Temizlenen: {sum(removed.values())} ({removed})")
        print(f"  📂 Kategoriler: {df['category'].unique().tolist()}")
        print(f"  💰 Fiyat: {df['price'].min():.0f} - {df['price'].max():.0f} TL")
        
        return df
    
    def load_and_clean_users(self) -> pd.DataFrame:
        """Kullanıcı verisini yükle ve temizle"""
        with open(self.user_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenarios = data['scenarios']
        df_data = []
        
        for scenario in scenarios:
            profile = scenario['profile']
            
            age = int(profile['age'])
            budget = float(profile['budget'])
            
            if age < 10 or age > 100 or budget <= 0 or budget > 10000:
                continue
            
            row = {
                'age': age,
                'budget': budget,
                'relationship': profile['relationship'],
                'occasion': profile['occasion'],
                'num_hobbies': len(profile.get('hobbies', [])),
                'num_preferences': len(profile.get('preferences', [])),
                'budget_category': self._categorize_price(budget),
                'age_group': self._categorize_age(age),
                'log_budget': np.log1p(budget)
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        print(f"  👥 {len(df)} temiz kullanıcı yüklendi")
        print(f"  🎂 Yaş: {df['age'].min()} - {df['age'].max()}")
        print(f"  💵 Budget: {df['budget'].min():.0f} - {df['budget'].max():.0f} TL")
        
        return df
    
    @staticmethod
    def _categorize_price(price: float) -> str:
        """Fiyat kategorisi"""
        if price < 100:
            return 'budget'
        elif price < 500:
            return 'moderate'
        elif price < 2000:
            return 'premium'
        else:
            return 'luxury'
    
    @staticmethod
    def _categorize_age(age: int) -> str:
        """Yaş kategorisi"""
        if age < 18:
            return 'teen'
        elif age < 30:
            return 'young_adult'
        elif age < 50:
            return 'adult'
        elif age < 65:
            return 'mature'
        else:
            return 'senior'
    
    def create_metadata(self, df: pd.DataFrame, df_type: str) -> SingleTableMetadata:
        """Metadata oluştur"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        if df_type == 'gifts':
            metadata.update_column('category', sdtype='categorical')
            metadata.update_column('price_category', sdtype='categorical')
            metadata.update_column('age_min', sdtype='numerical')
            metadata.update_column('age_max', sdtype='numerical')
            metadata.update_column('price', sdtype='numerical')
            metadata.update_column('rating', sdtype='numerical')
            metadata.update_column('log_price', sdtype='numerical')
        elif df_type == 'users':
            metadata.update_column('relationship', sdtype='categorical')
            metadata.update_column('occasion', sdtype='categorical')
            metadata.update_column('budget_category', sdtype='categorical')
            metadata.update_column('age_group', sdtype='categorical')
            metadata.update_column('age', sdtype='numerical')
            metadata.update_column('budget', sdtype='numerical')
            metadata.update_column('log_budget', sdtype='numerical')
        
        return metadata
    
    def train_synthesizer(self, df: pd.DataFrame, df_type: str):
        """Synthesizer eğit"""
        metadata = self.create_metadata(df, df_type)
        
        # Constraint ekle
        constraints = []
        if df_type == 'gifts':
            constraints.append(
                Inequality(
                    low_column_name='age_min',
                    high_column_name='age_max',
                    strict_boundaries=False
                )
            )
        
        # Küçük veri için Gaussian, büyük veri için CTGAN
        if len(df) < 50:
            print(f"  🔧 Gaussian Copula eğitiliyor...")
            num_distributions = {}
            if df_type == 'gifts':
                num_distributions = {'price': 'gamma', 'rating': 'beta'}
            elif df_type == 'users':
                num_distributions = {'budget': 'gamma'}
            
            synthesizer = GaussianCopulaSynthesizer(
                metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                numerical_distributions=num_distributions
            )
        else:
            print(f"  🔧 CTGAN eğitiliyor...")
            batch_size = max(10, (len(df) // 10) * 10)
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=300,
                batch_size=batch_size,
                verbose=False,
                cuda=False
            )
        
        if constraints:
            synthesizer.add_constraints(constraints)
        
        synthesizer.fit(df)
        self.synthesizers[df_type] = synthesizer
        print(f"  ✅ Synthesizer eğitildi")
    
    def generate_synthetic_data(self, df: pd.DataFrame, df_type: str, num_samples: int) -> pd.DataFrame:
        """Sentetik veri üret ve valide et"""
        synthesizer = self.synthesizers[df_type]
        
        # %20 fazla üret
        num_to_generate = int(num_samples * 1.2)
        print(f"  🎲 {num_to_generate} örnek üretiliyor...")
        
        synthetic_df = synthesizer.sample(num_rows=num_to_generate)
        
        # Validasyon
        print(f"  ✅ Validasyon yapılıyor...")
        if df_type == 'gifts':
            synthetic_df = self._validate_gifts(synthetic_df)
        elif df_type == 'users':
            synthetic_df = self._validate_users(synthetic_df)
        
        # Hedef sayıya indir
        if len(synthetic_df) > num_samples:
            synthetic_df = synthetic_df.sample(n=num_samples, random_state=42)
        
        # Kalite değerlendirmesi
        print(f"  📊 Kalite değerlendiriliyor...")
        quality_report = evaluate_quality(
            real_data=df,
            synthetic_data=synthetic_df,
            metadata=synthesizer.get_metadata()
        )
        
        score = quality_report.get_score()
        print(f"  📈 Kalite Skoru: {score:.2%}")
        
        return synthetic_df
    
    def _validate_gifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hediye validasyonu"""
        original_len = len(df)
        
        df = df[(df['price'] > 0) & (df['price'] < 100000)]
        df['rating'] = df['rating'].clip(0, 5)
        df = df[(df['age_min'] >= 0) & (df['age_max'] <= 100)]
        df = df[df['age_min'] < df['age_max']]
        
        # Kategori bazlı fiyat validasyonu
        valid_rows = []
        for idx, row in df.iterrows():
            category = row['category']
            price = row['price']
            if category in self.learned_data['price_ranges']:
                min_price, max_price = self.learned_data['price_ranges'][category]
                # %20 tolerans
                if min_price * 0.8 <= price <= max_price * 1.2:
                    valid_rows.append(idx)
            else:
                valid_rows.append(idx)
        
        df = df.loc[valid_rows]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"    🧹 {removed} geçersiz örnek temizlendi")
        
        return df
    
    def _validate_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kullanıcı validasyonu"""
        original_len = len(df)
        
        df = df[(df['age'] >= 10) & (df['age'] <= 100)]
        df = df[(df['budget'] > 0) & (df['budget'] < 10000)]
        df = df[~((df['age'] < 18) & (df['budget'] > 1000))]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"    🧹 {removed} geçersiz örnek temizlendi")
        
        return df
    
    def save_synthetic_gifts(self, df: pd.DataFrame, output_path: str):
        """Öğrenilmiş bilgilerle sentetik hediye kaydet"""
        import random
        gifts = []
        
        for idx, row in df.iterrows():
            category = row['category']
            price = float(row['price'])
            num_tags = min(int(row['num_tags']), 3)
            num_occasions = min(int(row['num_occasions']), 3)
            
            # Her zaman gerçek ürün ismini kullan
            if category in self.learned_data['product_names'] and self.learned_data['product_names'][category]:
                # Önce kendi kategorisinden seç
                product_name = random.choice(self.learned_data['product_names'][category])
            else:
                # Eğer o kategoride ürün yoksa, tüm kategorilerden seç
                all_names = []
                for cat_names in self.learned_data['product_names'].values():
                    all_names.extend(cat_names)
                product_name = random.choice(all_names) if all_names else f"Ürün {idx}"
            
            # Öğrenilmiş tag'leri kullan
            if category in self.learned_data['tags'] and self.learned_data['tags'][category]:
                available_tags = self.learned_data['tags'][category]
                selected_tags = random.sample(
                    available_tags, 
                    min(num_tags, len(available_tags))
                ) if num_tags > 0 else []
            else:
                selected_tags = []
            
            # Öğrenilmiş occasion'ları kullan
            if self.learned_data['occasions']:
                selected_occasions = random.sample(
                    self.learned_data['occasions'],
                    min(num_occasions, len(self.learned_data['occasions']))
                ) if num_occasions > 0 else []
            else:
                selected_occasions = []
            
            gift = {
                "id": f"learned_synth_{idx:04d}",
                "name": product_name,
                "category": category,
                "price": round(price, 2),
                "rating": max(3.0, min(5.0, round(float(row['rating']), 1))),
                "age_range": [int(row['age_min']), int(row['age_max'])],
                "tags": selected_tags,
                "occasions": selected_occasions
            }
            gifts.append(gift)
        
        output_data = {
            "gifts": gifts,
            "metadata": {
                "total_gifts": len(gifts),
                "source": "Fully Learned SDV Synthetic from Scraped Data",
                "quality_level": "learned_and_validated",
                "learned_from": str(self.gift_path)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Kaydedildi: {output_path}")
    
    def save_synthetic_users(self, df: pd.DataFrame, output_path: str):
        """Öğrenilmiş bilgilerle sentetik kullanıcı kaydet"""
        import random
        scenarios = []
        
        for idx, row in df.iterrows():
            num_hobbies = min(int(row['num_hobbies']), 4)
            num_preferences = min(int(row['num_preferences']), 3)
            
            # Öğrenilmiş hobileri kullan
            selected_hobbies = random.sample(
                self.learned_data['hobbies'],
                min(num_hobbies, len(self.learned_data['hobbies']))
            ) if num_hobbies > 0 else []
            
            # Öğrenilmiş tercihleri kullan
            selected_preferences = random.sample(
                self.learned_data['preferences'],
                min(num_preferences, len(self.learned_data['preferences']))
            ) if num_preferences > 0 else []
            
            # Öğrenilmiş expected_categories kullan (1-3 arası)
            num_categories = random.randint(1, 3)
            selected_categories = random.sample(
                self.learned_data['expected_categories'],
                min(num_categories, len(self.learned_data['expected_categories']))
            ) if self.learned_data['expected_categories'] else []
            
            # Öğrenilmiş expected_tools kullan (1-4 arası)
            num_tools = random.randint(1, 4)
            selected_tools = random.sample(
                self.learned_data['expected_tools'],
                min(num_tools, len(self.learned_data['expected_tools']))
            ) if self.learned_data['expected_tools'] else []
            
            scenario = {
                "id": f"learned_scenario_{idx:04d}",
                "profile": {
                    "age": int(row['age']),
                    "budget": round(float(row['budget']), 2),
                    "relationship": row['relationship'],
                    "occasion": row['occasion'],
                    "hobbies": selected_hobbies,
                    "preferences": selected_preferences
                },
                "expected_categories": selected_categories,
                "expected_tools": selected_tools
            }
            scenarios.append(scenario)
        
        output_data = {
            "scenarios": scenarios,
            "metadata": {
                "total_scenarios": len(scenarios),
                "source": "Fully Learned SDV Synthetic from User Data",
                "quality_level": "learned_and_validated",
                "learned_from": str(self.user_path)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Kaydedildi: {output_path}")


def main():
    """Ana fonksiyon"""
    if not SDV_AVAILABLE:
        return
    
    print("🎁 Tamamen Öğrenilmiş Sentetik Veri Üretimi")
    print("=" * 60)
    print("✨ Tüm bilgiler scraped veriden öğrenilir:")
    print("  • Ürün isimleri")
    print("  • Tag'ler")
    print("  • Occasion'lar")
    print("  • Fiyat aralıkları")
    print("  • Hobiler ve tercihler")
    print("=" * 60)
    
    generator = FullyLearnedSyntheticGenerator()
    
    # Önce öğren
    generator.learn_from_scraped_data()
    
    # Veri yükle
    print("\n📦 Hediye verisi yükleniyor...")
    gifts_df = generator.load_and_clean_gifts()
    
    print("\n👥 Kullanıcı verisi yükleniyor...")
    users_df = generator.load_and_clean_users()
    
    # Eğit
    print("\n🎓 Hediye synthesizer eğitiliyor...")
    generator.train_synthesizer(gifts_df, 'gifts')
    
    print("\n🎓 Kullanıcı synthesizer eğitiliyor...")
    generator.train_synthesizer(users_df, 'users')
    
    # Üret
    print("\n🎲 Sentetik hediye verisi üretiliyor...")
    synthetic_gifts = generator.generate_synthetic_data(gifts_df, 'gifts', num_samples=500)
    
    print("\n🎲 Sentetik kullanıcı verisi üretiliyor...")
    synthetic_users = generator.generate_synthetic_data(users_df, 'users', num_samples=300)
    
    # Kaydet
    print("\n💾 Öğrenilmiş sentetik veriler kaydediliyor...")
    generator.save_synthetic_gifts(synthetic_gifts, "data/fully_learned_synthetic_gifts.json")
    generator.save_synthetic_users(synthetic_users, "data/fully_learned_synthetic_users.json")
    
    print("\n" + "=" * 60)
    print("✅ Tamamen öğrenilmiş sentetik veri üretimi tamamlandı!")
    print("\n📁 Üretilen dosyalar:")
    print("  • data/fully_learned_synthetic_gifts.json")
    print("  • data/fully_learned_synthetic_users.json")
    print("\n🎯 Tüm bilgiler scraped veriden öğrenildi!")


if __name__ == "__main__":
    main()
