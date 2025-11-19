#!/usr/bin/env python3
"""
Advanced SDV Integration for Gift Recommendation Training
Includes multi-table synthesis, constraints, and quality evaluation
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
    from sdv.cag import Inequality
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


class AdvancedGiftDataSynthesizer:
    """Advanced synthetic data generation with quality control"""
    
    def __init__(self, config_path: str = "config/sdv_config.yaml"):
        self.config_path = Path(config_path)
        self.synthesizers = {}
        self.quality_reports = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare both gift and user data"""
        # Load gift catalog
        with open("data/realistic_gift_catalog.json", 'r') as f:
            gift_data = json.load(f)
        
        gifts_df = self._prepare_gift_dataframe(gift_data['gifts'])
        
        # Load user scenarios
        with open("data/realistic_user_scenarios.json", 'r') as f:
            user_data = json.load(f)
        
        users_df = self._prepare_user_dataframe(user_data['scenarios'])
        
        return gifts_df, users_df
    
    def _prepare_gift_dataframe(self, gifts: List[Dict]) -> pd.DataFrame:
        """Convert gift data to DataFrame with proper types"""
        data = []
        for gift in gifts:
            row = {
                'id': gift['id'],
                'category': gift['category'],
                'price': float(gift['price']),
                'rating': float(gift['rating']),
                'age_min': int(gift['age_range'][0]),
                'age_max': int(gift['age_range'][1]),
                'num_tags': len(gift['tags']),
                'num_occasions': len(gift['occasions']),
                # Derived features
                'price_category': self._categorize_price(gift['price']),
                'age_range_width': gift['age_range'][1] - gift['age_range'][0],
                'popularity_score': gift['rating'] * len(gift['tags'])
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_user_dataframe(self, scenarios: List[Dict]) -> pd.DataFrame:
        """Convert user scenarios to DataFrame"""
        data = []
        for scenario in scenarios:
            profile = scenario['profile']
            row = {
                'age': int(profile['age']),
                'budget': float(profile['budget']),
                'relationship': profile['relationship'],
                'occasion': profile['occasion'],
                'num_hobbies': len(profile['hobbies']),
                'num_preferences': len(profile['preferences']),
                # Derived features
                'budget_category': self._categorize_price(profile['budget']),
                'age_group': self._categorize_age(profile['age'])
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _categorize_price(price: float) -> str:
        """Categorize price into ranges"""
        if price < 30:
            return 'budget'
        elif price < 80:
            return 'moderate'
        elif price < 150:
            return 'premium'
        else:
            return 'luxury'
    
    @staticmethod
    def _categorize_age(age: int) -> str:
        """Categorize age into groups"""
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
    
    def create_constraints(self, df_type: str) -> List:
        """Create data constraints for realistic generation"""
        constraints = []
        
        if df_type == 'gifts':
            # Price kısıtlaması - dictionary formatında
            constraints.append({
                'constraint_class': 'ScalarRange',
                'constraint_parameters': {
                    'column_name': 'price',
                    'low_value': 5.0,
                    'high_value': 500.0,
                    'strict_boundaries': False
                }
            })
            
            # Rating kısıtlaması
            constraints.append({
                'constraint_class': 'ScalarRange',
                'constraint_parameters': {
                    'column_name': 'rating',
                    'low_value': 3.0,
                    'high_value': 5.0,
                    'strict_boundaries': False
                }
            })
            
            # Age min < age max kısıtlaması
            constraints.append(
                Inequality(
                    low_column_name='age_min',
                    high_column_name='age_max',
                    strict_boundaries=False
                )
            )
            
        elif df_type == 'users':
            # Budget pozitif olmalı
            constraints.append({
                'constraint_class': 'Positive',
                'constraint_parameters': {
                    'column_name': 'budget',
                    'strict_boundaries': False
                }
            })
            
            # Age kısıtlaması
            constraints.append({
                'constraint_class': 'ScalarRange',
                'constraint_parameters': {
                    'column_name': 'age',
                    'low_value': 10,
                    'high_value': 90,
                    'strict_boundaries': False
                }
            })
        
        return constraints
    
    def train_multiple_synthesizers(self, df: pd.DataFrame, df_type: str):
        """Train multiple synthesizers for comparison"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Set appropriate column types
        if df_type == 'gifts':
            metadata.update_column('id', sdtype='id')
            metadata.update_column('category', sdtype='categorical')
            metadata.update_column('price_category', sdtype='categorical')
            metadata.set_primary_key('id')
        elif df_type == 'users':
            metadata.update_column('relationship', sdtype='categorical')
            metadata.update_column('occasion', sdtype='categorical')
            metadata.update_column('budget_category', sdtype='categorical')
            metadata.update_column('age_group', sdtype='categorical')
        
        constraints = self.create_constraints(df_type)
        
        # Train Gaussian Copula (fast)
        print(f"  🔧 Training Gaussian Copula for {df_type}...")
        gaussian = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True
        )
        gaussian.fit(df)
        self.synthesizers[f'{df_type}_gaussian'] = gaussian
        
        # Train CTGAN (high quality, slower)
        print(f"  🔧 Training CTGAN for {df_type}...")
        ctgan = CTGANSynthesizer(
            metadata,
            epochs=100,
            verbose=False
        )
        ctgan.fit(df)
        self.synthesizers[f'{df_type}_ctgan'] = ctgan
        
        print(f"  ✅ Trained 2 synthesizers for {df_type}")
    
    def generate_and_evaluate(
        self, 
        df: pd.DataFrame, 
        df_type: str, 
        num_samples: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data and evaluate quality"""
        results = {}
        
        for synth_name, synthesizer in self.synthesizers.items():
            if df_type not in synth_name:
                continue
            
            print(f"\n  🎲 Generating {num_samples} samples with {synth_name}...")
            synthetic_df = synthesizer.sample(num_samples)
            
            # Evaluate quality
            print(f"  📊 Evaluating quality...")
            quality_report = evaluate_quality(
                real_data=df,
                synthetic_data=synthetic_df,
                metadata=synthesizer.get_metadata()
            )
            
            self.quality_reports[synth_name] = quality_report
            results[synth_name] = synthetic_df
            
            print(f"  📈 Quality Score: {quality_report.get_score():.2%}")
        
        return results
    
    def save_best_synthetic_data(
        self, 
        results: Dict[str, pd.DataFrame], 
        df_type: str,
        output_path: str
    ):
        """Save the best quality synthetic data"""
        # Find best synthesizer by quality score
        best_name = max(
            self.quality_reports.keys(),
            key=lambda k: self.quality_reports[k].get_score() if df_type in k else 0
        )
        
        best_df = results[best_name]
        
        print(f"\n  🏆 Best synthesizer: {best_name}")
        print(f"  💾 Saving to: {output_path}")
        
        if df_type == 'gifts':
            self._save_gift_catalog(best_df, output_path)
        elif df_type == 'users':
            self._save_user_scenarios(best_df, output_path)
    
    def _save_gift_catalog(self, df: pd.DataFrame, output_path: str):
        """Convert DataFrame back to gift catalog format"""
        gifts = []
        for idx, row in df.iterrows():
            gift = {
                "id": f"synth_{idx:04d}",
                "name": f"Synthetic {row['category'].title()} Gift {idx}",
                "category": row['category'],
                "price": round(float(row['price']), 2),
                "rating": round(float(row['rating']), 1),
                "age_range": [int(row['age_min']), int(row['age_max'])],
                "tags": [f"tag_{i}" for i in range(int(row['num_tags']))],
                "occasions": [f"occasion_{i}" for i in range(int(row['num_occasions']))]
            }
            gifts.append(gift)
        
        with open(output_path, 'w') as f:
            json.dump({"gifts": gifts, "metadata": {"source": "SDV"}}, f, indent=2)
    
    def _save_user_scenarios(self, df: pd.DataFrame, output_path: str):
        """Convert DataFrame back to user scenarios format"""
        scenarios = []
        for idx, row in df.iterrows():
            scenario = {
                "name": f"Synthetic User {idx}",
                "profile": {
                    "age": int(row['age']),
                    "budget": float(row['budget']),
                    "relationship": row['relationship'],
                    "occasion": row['occasion'],
                    "hobbies": [f"hobby_{i}" for i in range(int(row['num_hobbies']))],
                    "preferences": [f"pref_{i}" for i in range(int(row['num_preferences']))]
                }
            }
            scenarios.append(scenario)
        
        with open(output_path, 'w') as f:
            json.dump({"scenarios": scenarios, "metadata": {"source": "SDV"}}, f, indent=2)
    
    def generate_quality_report(self, output_path: str = "data/sdv_quality_report.json"):
        """Generate comprehensive quality report"""
        report = {
            "synthesizers": {},
            "summary": {
                "total_synthesizers": len(self.quality_reports),
                "best_overall": None,
                "best_score": 0.0
            }
        }
        
        for name, quality in self.quality_reports.items():
            score = quality.get_score()
            
            # Sadece skoru kaydet, detayları DataFrame olduğu için atlıyoruz
            report["synthesizers"][name] = {
                "overall_score": float(score)
            }
            
            if score > report["summary"]["best_score"]:
                report["summary"]["best_score"] = float(score)
                report["summary"]["best_overall"] = name
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Quality report saved to: {output_path}")


def main():
    """Main execution"""
    if not SDV_AVAILABLE:
        print("❌ SDV is not installed. Install with: pip install sdv")
        return
    
    print("🎁 Advanced SDV-Based Data Generator")
    print("=" * 60)
    
    # Initialize
    generator = AdvancedGiftDataSynthesizer()
    
    # Load data
    print("\n📊 Loading base data...")
    gifts_df, users_df = generator.load_and_prepare_data()
    print(f"  ✅ Loaded {len(gifts_df)} gifts and {len(users_df)} users")
    
    # Train synthesizers for gifts
    print("\n🎓 Training synthesizers for gifts...")
    generator.train_multiple_synthesizers(gifts_df, 'gifts')
    
    # Train synthesizers for users
    print("\n🎓 Training synthesizers for users...")
    generator.train_multiple_synthesizers(users_df, 'users')
    
    # Generate and evaluate gifts
    print("\n🎲 Generating synthetic gifts...")
    gift_results = generator.generate_and_evaluate(gifts_df, 'gifts', num_samples=300)
    
    # Generate and evaluate users
    print("\n🎲 Generating synthetic users...")
    user_results = generator.generate_and_evaluate(users_df, 'users', num_samples=150)
    
    # Save best results
    Path("data").mkdir(exist_ok=True)
    generator.save_best_synthetic_data(gift_results, 'gifts', "data/synthetic_gift_catalog.json")
    generator.save_best_synthetic_data(user_results, 'users', "data/synthetic_user_scenarios.json")
    
    # Generate quality report
    generator.generate_quality_report()
    
    print("\n✅ Synthetic data generation complete!")
    print("🎯 Next steps:")
    print("   1. Review quality report: data/sdv_quality_report.json")
    print("   2. Use synthetic data for training")
    print("   3. Compare model performance with/without synthetic data")


if __name__ == "__main__":
    main()
