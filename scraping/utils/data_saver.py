"""
Data Saver Utility
Saves raw and processed data with backup functionality
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class DataSaver:
    """Utility for saving scraped and processed data"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize data saver
        
        Args:
            raw_data_path: Directory for raw scraped data
            processed_data_path: Directory for processed data
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(processed_data_path, exist_ok=True)
    
    def save_raw_data(self, data: List[Dict[str, Any]], source: str) -> str:
        """
        Save raw scraped data
        
        Args:
            data: List of raw product dictionaries
            source: Source website name
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_raw_{timestamp}.json"
        filepath = os.path.join(self.raw_data_path, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved raw data to {filepath} ({len(data)} items)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save raw data: {e}")
            raise
    
    def save_processed_data(self, data: List[Dict[str, Any]], stage: str) -> str:
        """
        Save processed data at various pipeline stages
        
        Args:
            data: List of processed product dictionaries
            stage: Processing stage name (e.g., 'validated', 'enhanced')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage}_{timestamp}.json"
        filepath = os.path.join(self.processed_data_path, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved processed data to {filepath} ({len(data)} items)")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise
    
    def create_backup(self, source_file: str) -> str:
        """
        Create backup of a file
        
        Args:
            source_file: Path to file to backup
            
        Returns:
            Path to backup file
        """
        if not os.path.exists(source_file):
            self.logger.warning(f"Source file not found for backup: {source_file}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(source_file), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        filename = os.path.basename(source_file)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        try:
            import shutil
            shutil.copy2(source_file, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
