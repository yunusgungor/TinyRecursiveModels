#!/usr/bin/env python3
"""
SDV-based Synthetic Data Generator for Gift Recommendation Training
Uses SDV to generate realistic training data from existing patterns
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("⚠️  SDV not installed. Install with: pip install sdv")


class GiftDataSynthesizer:
    """Generate synthetic gift recommendation training data using SDV"""
    
    def __init__(self, base_data_path: str = "data/realistic_gift_catalog.json"):
        self.base_data_path = Path(base_data_path)
        self.synthesizer = None
        self.metadata = None
        
    def load_base_data(self) -> pd.DataFrame:
        """Load existing gift data as base for synthesis"""
        with open(self.base_data_path, 'r') as f:
            data = json.load(f)
        
        gifts = data['gifts']
        
        # Convert to DataFrame
        df_data = []
        for gift in gifts:
            row = {
                'id': gift['id'],
                'name': gift['name'],
                'category': gift['category'],
                'price': gift['price'],
                'rating': gift['rating'],
                'age_min': gift['age_range'][0],
                'age_max': gift['age_range'][1],
                'num_tags': len(gift['tags']),
                'num_occasions': len(gift['occasions'])
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def setup_metadata(self, df: pd.DataFrame) -> SingleTableMetadata:
        """Configure metadata for SDV"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Customize metadata
        metadata.update_column('id', sdtype='id')
        metadata.update_column('category', sdtype='categorical')
        metadata.update_column('price', sdtype='numerical')
        metadata.update_column('rating', sdtype='numerical')
        metadata.update_column('age_min', sdtype='numerical')
        metadata.update_column('age_max', sdtype='numerical')
        
        # Set primary key
        metadata.set_primary_key('id')
        
        return metadata
    
    def train_synthesizer(self, df: pd.DataFrame, method: str = "gaussian"):
        """Train SDV synthesizer on base data"""
        self.metadata = self.setup_metadata(df)
        
        if method == "gaussian":
            self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
        elif method == "ctgan":
            self.synthesizer = CTGANSynthesizer(
                self.metadata,
                epochs=100,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"🔧 Training {method} synthesizer...")
        self.synthesizer.fit(df)
        print("✅ Synthesizer trained!")
    
    def generate_synthetic_data(self, num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic gift data"""
        if self.synthesizer is None:
            raise ValueError("Synthesizer not trained. Call train_synthesizer first.")
        
        print(f"🎲 Generating {num_samples} synthetic samples...")
        synthetic_data = self.synthesizer.sample(num_samples)
        
        return synthetic_data
    
    def save_synthetic_catalog(self, synthetic_df: pd.DataFrame, output_path: str):
        """Convert synthetic DataFrame back to gift catalog format"""
        gifts = []
        
        for idx, row in synthetic_df.iterrows():
            gift = {
                "id": f"synth_{idx:04d}",
                "name": f"Synthetic Gift {idx}",
                "category": row['category'],
                "price": round(float(row['price']), 2),
                "rating": round(float(row['rating']), 1),
                "age_range": [int(row['age_min']), int(row['age_max'])],
                "tags": [f"tag_{i}" for i in range(int(row['num_tags']))],
                "occasions": [f"occasion_{i}" for i in range(int(row['num_occasions']))]
            }
            gifts.append(gift)
        
        output_data = {
            "gifts": gifts,
            "metadata": {
                "total_gifts": len(gifts),
                "source": "SDV Synthetic Generation",
                "method": "GaussianCopula"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Saved synthetic catalog to: {output_path}")


def main():
    """Main execution"""
    if not SDV_AVAILABLE:
        print("❌ SDV is not installed. Please install it first:")
        print("   pip install sdv")
        return
    
    print("🎁 SDV-Based Gift Data Generator")
    print("=" * 60)
    
    # Initialize synthesizer
    generator = GiftDataSynthesizer()
    
    # Load base data
    print("\n📊 Loading base data...")
    base_df = generator.load_base_data()
    print(f"   Loaded {len(base_df)} base samples")
    print(f"   Categories: {base_df['category'].unique().tolist()}")
    
    # Train synthesizer
    print("\n🎓 Training synthesizer...")
    generator.train_synthesizer(base_df, method="gaussian")
    
    # Generate synthetic data
    print("\n🎲 Generating synthetic data...")
    synthetic_df = generator.generate_synthetic_data(num_samples=200)
    
    # Save results
    output_path = "data/synthetic_gift_catalog.json"
    Path("data").mkdir(exist_ok=True)
    generator.save_synthetic_catalog(synthetic_df, output_path)
    
    print("\n✅ Synthetic data generation complete!")
    print(f"   Generated: {len(synthetic_df)} samples")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
