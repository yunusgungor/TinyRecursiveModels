#!/usr/bin/env python3
"""
Generate User Scenarios Script
Creates realistic user scenarios from gift catalog data
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scraping.services.user_scenario_generator import UserScenarioGenerator
from scraping.config.config_manager import ConfigurationManager


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/user_scenario_generation.log')
        ]
    )


async def main():
    """Main execution function"""
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("🎯 User Scenario Generator")
    print("=" * 70)
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config_manager = ConfigurationManager()
    gemini_config = config_manager.get_gemini_config()
    
    # Paths
    gift_catalog_path = "data/scraped_gift_catalog.json"
    output_path = "data/user_scenarios.json"
    
    # Check if gift catalog exists
    if not os.path.exists(gift_catalog_path):
        logger.error(f"Gift catalog not found: {gift_catalog_path}")
        print(f"\n❌ Hata: Gift catalog bulunamadı: {gift_catalog_path}")
        print("Önce gift catalog'u oluşturun: python scraping/scripts/run_scraping.py")
        return
    
    # Initialize generator
    logger.info("Initializing user scenario generator...")
    generator = UserScenarioGenerator(gemini_config, gift_catalog_path)
    
    # Generate scenarios
    num_scenarios = 100  # Varsayılan olarak 100 senaryo
    
    if len(sys.argv) > 1:
        try:
            num_scenarios = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid number: {sys.argv[1]}, using default: 100")
    
    print(f"\n📊 {num_scenarios} kullanıcı senaryosu oluşturuluyor...")
    print(f"📁 Kaynak: {gift_catalog_path}")
    print(f"💾 Hedef: {output_path}")
    print()
    
    scenarios = await generator.generate_scenarios(num_scenarios)
    
    # Save scenarios
    await generator.save_scenarios(scenarios, output_path)
    
    # Print statistics
    stats = generator.get_stats()
    
    print("\n" + "=" * 70)
    print("✅ Senaryo Oluşturma Tamamlandı!")
    print("=" * 70)
    print(f"📊 Toplam Senaryo: {len(scenarios)}")
    print(f"🤖 AI İstekleri: {stats['requests_made']}")
    print(f"📁 Dosya: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
