"""
Dataset Generator
Generates final training dataset from scraped and enhanced data
"""

import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter

from ..utils.models import RawProductData, EnhancedProductData


class DatasetGenerator:
    """Generates final training dataset"""
    
    def __init__(self, output_path: str):
        """
        Initialize dataset generator
        
        Args:
            output_path: Path to save final dataset
        """
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.logger.info(f"DatasetGenerator initialized with output: {output_path}")

    def generate_dataset(self, 
                        validated_products: List[RawProductData],
                        enhancements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate final dataset from validated products and enhancements
        
        Args:
            validated_products: List of validated raw products
            enhancements: List of AI enhancements
            
        Returns:
            Complete dataset dictionary
        """
        self.logger.info("Generating final dataset...")
        
        # Merge data
        merged_products = self._merge_data(validated_products, enhancements)
        
        # Convert to gift catalog format
        gift_items = self._convert_to_gift_format(merged_products)
        
        # Generate metadata
        metadata = self._generate_metadata(gift_items)
        
        # Create final dataset
        dataset = {
            "gifts": gift_items,
            "metadata": metadata
        }
        
        # Save dataset
        self._save_dataset(dataset)
        
        self.logger.info(f"Dataset generated with {len(gift_items)} items")
        
        return dataset
    
    def _merge_data(self, products: List[RawProductData], 
                   enhancements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge raw product data with AI enhancements
        
        Args:
            products: List of raw products
            enhancements: List of enhancements
            
        Returns:
            List of merged product dictionaries
        """
        merged = []
        
        # Ensure we have matching counts
        min_count = min(len(products), len(enhancements))
        
        if len(products) != len(enhancements):
            self.logger.warning(
                f"Product count ({len(products)}) != enhancement count ({len(enhancements)}). "
                f"Using {min_count} items."
            )
        
        for i in range(min_count):
            product = products[i]
            enhancement = enhancements[i]
            
            # Merge dictionaries
            merged_item = {
                **product.model_dump(),
                **enhancement
            }
            
            merged.append(merged_item)
        
        return merged

    def _convert_to_gift_format(self, merged_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert merged data to gift catalog format
        
        Args:
            merged_products: List of merged product dictionaries
            
        Returns:
            List of gift items in catalog format
        """
        gifts = []
        
        for idx, product in enumerate(merged_products):
            # Generate unique ID
            source = product.get('source', 'unknown')
            gift_id = f"{source}_{idx:04d}"
            
            # Combine tags
            emotional_tags = product.get('emotional_tags', [])
            target_audience = product.get('target_audience', [])
            tags = emotional_tags + target_audience
            
            # Build gift item (matching enhanced_realistic_gift_catalog.json structure)
            gift = {
                "id": gift_id,
                "name": product.get('name', 'Unknown Product'),
                "category": product.get('category', 'unknown'),
                "price": product.get('price', 0.0),
                "rating": product.get('rating', 0.0),
                "tags": tags,
                "age_range": product.get('age_range', [18, 65]),
                "occasions": product.get('gift_occasions', ['any'])
            }
            
            gifts.append(gift)
        
        return gifts

    def _generate_metadata(self, gifts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate dataset metadata
        
        Args:
            gifts: List of gift items
            
        Returns:
            Metadata dictionary
        """
        # Category distribution
        categories = [g['category'] for g in gifts]
        category_counts = dict(Counter(categories))
        
        # Price statistics
        prices = [g['price'] for g in gifts if g['price'] > 0]
        
        metadata = {
            "total_gifts": len(gifts),
            "categories": list(set(categories)),
            "category_counts": category_counts,
            "price_range": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0
            },
            "created": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        return metadata

    def _save_dataset(self, dataset: Dict[str, Any]) -> None:
        """
        Save dataset to JSON file
        
        Args:
            dataset: Complete dataset dictionary
        """
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Dataset saved to {self.output_path}")
            
            # Log file size
            file_size = os.path.getsize(self.output_path)
            file_size_mb = file_size / (1024 * 1024)
            self.logger.info(f"Dataset file size: {file_size_mb:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            raise
