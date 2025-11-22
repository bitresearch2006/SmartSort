#!/usr/bin/env python3
"""
Configuration and Categories Loader for SmartSort

This module provides centralized configuration management and category loading
for the SmartSort biomedical waste classification system. It eliminates dependencies
between modules by using JSON configuration files.

Usage:
    from config_loader import load_config, load_categories, get_model_config
    
    config = load_config()
    categories = load_categories()
    model_params = get_model_config()
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Centralized configuration loader for SmartSort."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            base_dir (str): Base directory containing config files. 
                          Defaults to the directory of this script.
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.base_dir, 'config.json')
        self.categories_file = os.path.join(self.base_dir, 'categories.json')
        
        self._config_cache = None
        self._categories_cache = None
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load the main configuration from config.json.
        
        Args:
            force_reload (bool): Force reload even if cached
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache
        
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"Config file not found: {self.config_file}")
                return self._get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Cache the config
            self._config_cache = config
            logger.info(f"Configuration loaded from {self.config_file}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def load_categories(self, force_reload: bool = False) -> List[str]:
        """
        Load categories from categories.json.
        
        Args:
            force_reload (bool): Force reload even if cached
            
        Returns:
            List[str]: List of category names
        """
        if self._categories_cache is not None and not force_reload:
            return self._categories_cache
        
        try:
            if not os.path.exists(self.categories_file):
                logger.warning(f"Categories file not found: {self.categories_file}")
                return self._get_default_categories()
            
            with open(self.categories_file, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
            
            categories = categories_data.get('categories', [])
            
            if not categories:
                logger.warning("No categories found in JSON file")
                return self._get_default_categories()
            
            # Cache the categories
            self._categories_cache = categories
            logger.info(f"Loaded {len(categories)} categories from {self.categories_file}")
            
            return categories
            
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return self._get_default_categories()
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        config = self.load_config()
        return config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Training configuration
        """
        config = self.load_config()
        return config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data-specific configuration parameters.
        
        Returns:
            Dict[str, Any]: Data configuration
        """
        config = self.load_config()
        return config.get('data', {})
    
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the configured image size.
        
        Returns:
            Tuple[int, int]: Image size as (height, width)
        """
        model_config = self.get_model_config()
        size = model_config.get('image_size', [224, 224])
        return tuple(size)
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes.
        
        Returns:
            int: Number of classification categories
        """
        categories = self.load_categories()
        return len(categories)
    
    def update_categories(self, categories: List[str], detection_method: str = "manual") -> bool:
        """
        Update the categories JSON file with new categories.
        
        Args:
            categories (List[str]): List of category names
            detection_method (str): How categories were detected
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            categories_data = {
                "categories": categories,
                "metadata": {
                    "total_categories": len(categories),
                    "created_date": datetime.now().strftime("%Y-%m-%d"),
                    "last_updated": datetime.now().isoformat(),
                    "data_source": "biomedical_dataset",
                    "detection_method": detection_method
                },
                "category_descriptions": self._get_category_descriptions(categories)
            }
            
            with open(self.categories_file, 'w', encoding='utf-8') as f:
                json.dump(categories_data, f, indent=2, ensure_ascii=False)
            
            # Clear cache to force reload
            self._categories_cache = None
            
            logger.info(f"Updated categories file with {len(categories)} categories")
            return True
            
        except Exception as e:
            logger.error(f"Error updating categories: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config.json is not available."""
        return {
            "project": {
                "name": "SmartSort",
                "description": "Biomedical Waste Classification System",
                "version": "1.0.0"
            },
            "model": {
                "name": "biomedical_waste_classifier.keras",
                "architecture": "MobileNetV2",
                "image_size": [224, 224],
                "input_channels": 3,
                "learning_rate": 0.0001,
                "epochs": 10,
                "batch_size": 32,
                "validation_split": 0.2,
                "seed": 42
            },
            "data": {
                "raw_data_dir": "../data/biomedical_dataset",
                "processed_data_dir": "../data/biomedical_dataset_FLAT",
                "categories_file": "categories.json"
            },
            "training": {
                "early_stopping_patience": 3,
                "data_augmentation": {
                    "random_flip": "horizontal",
                    "random_rotation": 0.2,
                    "random_zoom": 0.2
                }
            },
            "deployment": {
                "log_level": "INFO",
                "metrics_dir": "./metrics"
            }
        }
    
    def _get_default_categories(self) -> List[str]:
        """Get default categories if categories.json is not available."""
        return [
            '(BT) Body Tissue or Organ',
            '(GE) Glass equipment-packaging 551', 
            '(ME) Metal equipment -packaging',
            '(OW) Organic wastes',
            '(PE) Plastic equipment-packaging',
            '(PP) Paper equipment-packaging',
            'Gauze',
            'Gloves', 
            'Mask',
            'Syringe',
            'Tweezers'
        ]
    
    def _get_category_descriptions(self, categories: List[str]) -> Dict[str, str]:
        """Generate category descriptions."""
        descriptions = {}
        
        # Default descriptions for known categories
        default_descriptions = {
            "(BT) Body Tissue or Organ": "Biological tissues and organ waste",
            "(GE) Glass equipment-packaging 551": "Glass laboratory equipment and packaging",
            "(ME) Metal equipment -packaging": "Metal medical equipment and packaging",
            "(OW) Organic wastes": "Organic biological waste materials",
            "(PE) Plastic equipment-packaging": "Plastic medical equipment and packaging",
            "(PP) Paper equipment-packaging": "Paper-based medical packaging",
            "Gauze": "Medical gauze and similar fabric materials",
            "Gloves": "Medical gloves (latex, nitrile, etc.)",
            "Mask": "Medical masks and face coverings",
            "Syringe": "Medical syringes and needles",
            "Tweezers": "Medical tweezers and similar instruments"
        }
        
        for category in categories:
            descriptions[category] = default_descriptions.get(
                category, 
                f"Biomedical waste category: {category}"
            )
        
        return descriptions

# Global instance for easy access
_config_loader = ConfigLoader()

# Convenience functions for backward compatibility and ease of use
def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load the main configuration."""
    return _config_loader.load_config(force_reload)

def load_categories(force_reload: bool = False) -> List[str]:
    """Load categories list."""
    return _config_loader.load_categories(force_reload)

def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return _config_loader.get_model_config()

def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return _config_loader.get_training_config()

def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return _config_loader.get_data_config()

def get_image_size() -> Tuple[int, int]:
    """Get configured image size."""
    return _config_loader.get_image_size()

def get_num_classes() -> int:
    """Get number of classes."""
    return _config_loader.get_num_classes()

def update_categories(categories: List[str], detection_method: str = "manual") -> bool:
    """Update categories file."""
    return _config_loader.update_categories(categories, detection_method)

# Test function
def test_config_loader():
    """Test the configuration loader."""
    print("🧪 Testing Configuration Loader")
    print("=" * 50)
    
    try:
        # Test config loading
        config = load_config()
        print(f"✅ Config loaded: {config['project']['name']}")
        
        # Test categories loading  
        categories = load_categories()
        print(f"✅ Categories loaded: {len(categories)} categories")
        
        # Test model config
        model_config = get_model_config()
        print(f"✅ Model config: {model_config['architecture']}")
        
        # Test image size
        image_size = get_image_size()
        print(f"✅ Image size: {image_size}")
        
        # Test number of classes
        num_classes = get_num_classes()
        print(f"✅ Number of classes: {num_classes}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_config_loader()
