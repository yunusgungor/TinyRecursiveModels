"""
Data Validator for Web Scraping Pipeline
Validates and cleans scraped data
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from .models import RawProductData, EnhancedProductData


class DataValidator:
    """Validates and cleans scraped data"""
    
    def __init__(self):
        """Initialize data validator"""
        self.logger = logging.getLogger(__name__)
        self.validation_stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates_removed': 0
        }
    
    def validate_product(self, raw_data: Dict[str, Any]) -> Optional[RawProductData]:
        """
        Validate a single product
        
        Args:
            raw_data: Raw product data dictionary
            
        Returns:
            Validated RawProductData object or None if validation fails
        """
        self.validation_stats['total'] += 1
        
        try:
            validated = RawProductData(**raw_data)
            self.validation_stats['valid'] += 1
            return validated
        except ValidationError as e:
            self.validation_stats['invalid'] += 1
            self.logger.error(f"Validation failed for product: {e}")
            self.logger.debug(f"Invalid data: {raw_data}")
            return None
        except Exception as e:
            self.validation_stats['invalid'] += 1
            self.logger.error(f"Unexpected error during validation: {e}")
            return None

    def validate_batch(self, raw_data_list: List[Dict[str, Any]]) -> List[RawProductData]:
        """
        Validate a batch of products
        
        Args:
            raw_data_list: List of raw product data dictionaries
            
        Returns:
            List of validated RawProductData objects
        """
        validated = []
        
        for data in raw_data_list:
            product = self.validate_product(data)
            if product:
                validated.append(product)
        
        self.logger.info(
            f"Batch validation: {len(validated)}/{len(raw_data_list)} products valid"
        )
        
        return validated
    
    def remove_duplicates(self, products: List[RawProductData]) -> List[RawProductData]:
        """
        Remove duplicate products based on name and price
        
        Args:
            products: List of validated products
            
        Returns:
            List of unique products
        """
        seen = set()
        unique = []
        duplicates = 0
        
        for product in products:
            # Create unique key from name and price
            key = (product.name.lower().strip(), round(product.price, 2))
            
            if key not in seen:
                seen.add(key)
                unique.append(product)
            else:
                duplicates += 1
                self.logger.debug(f"Duplicate found: {product.name} - {product.price} TL")
        
        self.validation_stats['duplicates_removed'] += duplicates
        
        if duplicates > 0:
            self.logger.info(f"Removed {duplicates} duplicate products")
        
        return unique

    def filter_by_price_range(self, products: List[RawProductData], 
                              min_price: float = 0, 
                              max_price: float = float('inf')) -> List[RawProductData]:
        """
        Filter products by price range
        
        Args:
            products: List of products
            min_price: Minimum price (inclusive)
            max_price: Maximum price (inclusive)
            
        Returns:
            Filtered list of products
        """
        filtered = [p for p in products if min_price <= p.price <= max_price]
        
        removed = len(products) - len(filtered)
        if removed > 0:
            self.logger.info(
                f"Filtered {removed} products outside price range "
                f"[{min_price}, {max_price}]"
            )
        
        return filtered
    
    def get_validation_stats(self) -> Dict[str, int]:
        """
        Get validation statistics
        
        Returns:
            Dictionary with validation stats
        """
        return self.validation_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset validation statistics"""
        self.validation_stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates_removed': 0
        }
        self.logger.info("Validation statistics reset")
