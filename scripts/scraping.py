"""
Main Scraping Pipeline Script
Orchestrates the entire web scraping and data enhancement pipeline
"""

import asyncio
import sys
import os
print("Script starting...")
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from scraping/.env file
load_dotenv(project_root / "scraping" / ".env")

from scraping.config.config_manager import ConfigurationManager
from scraping.utils.logger import setup_logger
from scraping.utils.validator import DataValidator
from scraping.scrapers.orchestrator import ScrapingOrchestrator
from scraping.services.gemini_service import GeminiEnhancementService
from scraping.services.dataset_generator import DatasetGenerator
from scraping.services.user_scenario_generator import UserScenarioGenerator


async def run_pipeline(config_path: str = "scraping/config/scraping_config.yaml"):
    """
    Run the complete scraping pipeline
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    print("Loading configuration...")
    config_manager = ConfigurationManager(config_path)
    
    # Setup logging
    logging_config = config_manager.get_logging_config()
    logger_instance = setup_logger(logging_config)
    logger = logger_instance.get_logger(__name__)
    
    logger.info("="*60)
    logger.info("Starting Web Scraping Pipeline")
    logger.info("="*60)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Validator
        validator = DataValidator()
        
        # Scraping orchestrator
        orchestrator = ScrapingOrchestrator(config_manager)
        orchestrator.initialize_scrapers()
        
        # AI Enhancement Service (Gemini or Ollama)
        gemini_config = config_manager.get_gemini_config()
        ollama_config = config_manager.get_all_config().get('ollama', {})
        
        if ollama_config.get('enabled', False):
            from scraping.services.ollama_service import OllamaEnhancementService
            logger.info(f"Using Ollama for enhancement (Model: {ollama_config.get('model', 'default')})")
            enhancement_service = OllamaEnhancementService(ollama_config)
        else:
            from scraping.services.gemini_service import GeminiEnhancementService
            logger.info("Using Gemini for enhancement")
            enhancement_service = GeminiEnhancementService(gemini_config)
            
        # Compatibility alias
        gemini_service = enhancement_service
        
        # Dataset generator
        output_config = config_manager.get_output_config()
        final_dataset_path = output_config.get('final_dataset_path', 'data/gift_catalog.json')
        dataset_generator = DatasetGenerator(final_dataset_path)

        # Phase 1: Scrape products
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: Scraping Products")
        logger.info("="*60)
        
        raw_products = await orchestrator.scrape_all_websites()
        logger.info(f"Scraped {len(raw_products)} products")
        
        # Phase 2: Validate products
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: Validating Products")
        logger.info("="*60)
        
        validated_products = validator.validate_batch(raw_products)
        logger.info(f"Validated {len(validated_products)} products")
        
        # Remove duplicates
        unique_products = validator.remove_duplicates(validated_products)
        logger.info(f"After deduplication: {len(unique_products)} unique products")
        
        # Phase 3: Enhance with Gemini
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: Enhancing with Gemini AI")
        logger.info("="*60)
        
        enhancements = await gemini_service.enhance_batch(unique_products)
        logger.info(f"Enhanced {len(enhancements)} products")
        
        # Phase 4: Generate dataset
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: Generating Final Dataset")
        logger.info("="*60)
        
        dataset = dataset_generator.generate_dataset(unique_products, enhancements)
        
        # Phase 5: Generate User Scenarios
        logger.info("\n" + "="*60)
        logger.info("PHASE 5: Generating User Scenarios")
        logger.info("="*60)
        
        # Initialize user scenario generator
        user_scenario_generator = UserScenarioGenerator(
            config=gemini_config,
            gift_catalog_path=final_dataset_path,
            ollama_config=ollama_config
        )
        
        # Generate scenarios
        num_scenarios = output_config.get('num_user_scenarios', 100)
        scenarios = await user_scenario_generator.generate_scenarios(num_scenarios)
        
        # Save scenarios
        scenarios_output_path = output_config.get('user_scenarios_path', 'data/user_scenarios.json')
        await user_scenario_generator.save_scenarios(scenarios, scenarios_output_path)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total products in dataset: {dataset['metadata']['total_gifts']}")
        logger.info(f"Categories: {', '.join(dataset['metadata']['categories'])}")
        logger.info(f"Price range: {dataset['metadata']['price_range']['min']:.2f} - {dataset['metadata']['price_range']['max']:.2f} TL")
        logger.info(f"Dataset saved to: {final_dataset_path}")
        logger.info(f"User scenarios: {len(scenarios)} scenarios")
        logger.info(f"Scenarios saved to: {scenarios_output_path}")
        
        # Print statistics
        orchestrator_stats = orchestrator.get_stats()
        validator_stats = validator.get_validation_stats()
        gemini_stats = gemini_service.get_stats()
        scenario_stats = user_scenario_generator.get_stats()
        
        logger.info("\n" + "="*60)
        logger.info("STATISTICS")
        logger.info("="*60)
        logger.info(f"Scraping duration: {orchestrator_stats.get('duration_formatted', 'N/A')}")
        logger.info(f"Products by source: {orchestrator_stats.get('products_by_source', {})}")
        logger.info(f"Validation: {validator_stats['valid']}/{validator_stats['total']} valid")
        logger.info(f"Duplicates removed: {validator_stats['duplicates_removed']}")
        logger.info(f"Gemini requests (enhancement): {gemini_stats['requests_made']}/{gemini_stats['max_requests_per_day']}")
        logger.info(f"Gemini requests (scenarios): {scenario_stats['requests_made']}/{scenario_stats['max_requests_per_day']}")
        logger.info(f"Scenario generation method: {scenario_stats['enabled'] and 'AI' or 'Fallback'}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point with CLI argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Web Scraping Pipeline for Gift Recommendation Dataset"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/scraping_config.yaml',
        help='Path to configuration file (default: config/scraping_config.yaml)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (limited products)'
    )
    
    parser.add_argument(
        '--website',
        type=str,
        choices=['cimri', 'ciceksepeti', 'hepsiburada', 'trendyol'],
        help='Scrape only specific website'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load config and apply CLI overrides
    config_manager = ConfigurationManager(args.config)
    
    if args.test:
        config_manager.update_config('scraping.test_mode', True)
        print("Running in TEST MODE")
    
    if args.verbose:
        config_manager.update_config('logging.verbose', True)
        print("Verbose logging enabled")
    
    if args.website:
        # Disable all websites except the specified one
        for website in config_manager.get_all_config()['scraping']['websites']:
            if website['name'] == args.website:
                website['enabled'] = True
            else:
                website['enabled'] = False
        print(f"Scraping only from: {args.website}")
    
    # Run pipeline
    asyncio.run(run_pipeline(args.config))


if __name__ == "__main__":
    main()
