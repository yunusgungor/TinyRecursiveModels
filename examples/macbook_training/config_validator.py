#!/usr/bin/env python3
"""
Configuration Validation and Recommendation System for MacBook Email Training

This module provides validation and recommendation functionality for email classification
training configurations on different MacBook models.

Requirements: 2.1, 2.2
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MacBookModel(Enum):
    """MacBook model categories based on memory."""
    MACBOOK_8GB = "8gb"
    MACBOOK_16GB = "16gb"
    MACBOOK_32GB = "32gb"

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    estimated_memory_usage: Optional[float] = None
    estimated_training_time: Optional[float] = None

@dataclass
class HardwareSpecs:
    """Hardware specifications."""
    memory_gb: float
    cpu_cores: int
    model_category: MacBookModel

class ConfigValidator:
    """Configuration validator for MacBook email training."""
    
    def __init__(self):
        """Initialize validator with hardware-specific limits."""
        self.memory_limits = {
            MacBookModel.MACBOOK_8GB: {
                "max_batch_size": 4,
                "max_vocab_size": 5000,
                "max_hidden_size": 384,
                "max_sequence_length": 512,
                "recommended_memory_limit": 5500
            },
            MacBookModel.MACBOOK_16GB: {
                "max_batch_size": 8,
                "max_vocab_size": 8000,
                "max_hidden_size": 512,
                "max_sequence_length": 768,
                "recommended_memory_limit": 12000
            },
            MacBookModel.MACBOOK_32GB: {
                "max_batch_size": 16,
                "max_vocab_size": 12000,
                "max_hidden_size": 768,
                "max_sequence_length": 1024,
                "recommended_memory_limit": 24000
            }
        }
    
    def detect_macbook_model(self, memory_gb: float) -> MacBookModel:
        """Detect MacBook model category based on memory."""
        if memory_gb <= 8:
            return MacBookModel.MACBOOK_8GB
        elif memory_gb <= 16:
            return MacBookModel.MACBOOK_16GB
        else:
            return MacBookModel.MACBOOK_32GB
    
    def estimate_memory_usage(self, config: Dict[str, Any]) -> float:
        """Estimate memory usage in MB based on configuration."""
        # Base memory for Python and PyTorch
        base_memory = 1000
        
        # Model memory estimation
        vocab_size = config.get("model", {}).get("vocab_size", 5000)
        hidden_size = config.get("model", {}).get("hidden_size", 512)
        num_layers = config.get("model", {}).get("num_layers", 2)
        sequence_length = config.get("model", {}).get("max_sequence_length", 512)
        
        # Embedding layer: vocab_size * hidden_size * 4 bytes
        embedding_memory = vocab_size * hidden_size * 4 / (1024 * 1024)
        
        # Transformer layers: approximate calculation
        layer_memory = num_layers * hidden_size * hidden_size * 4 * 4 / (1024 * 1024)  # 4 matrices per layer
        
        # Batch memory
        batch_size = config.get("training", {}).get("batch_size", 4)
        batch_memory = batch_size * sequence_length * hidden_size * 4 / (1024 * 1024)
        
        # Gradient memory (approximately same as model)
        gradient_memory = embedding_memory + layer_memory
        
        # Optimizer state (Adam: 2x model parameters)
        optimizer_memory = 2 * (embedding_memory + layer_memory)
        
        total_memory = (base_memory + embedding_memory + layer_memory + 
                       batch_memory + gradient_memory + optimizer_memory)
        
        return total_memory
    
    def estimate_training_time(self, config: Dict[str, Any], dataset_size: int = 1000) -> float:
        """Estimate training time in minutes."""
        batch_size = config.get("training", {}).get("batch_size", 4)
        max_steps = config.get("training", {}).get("max_steps", 5000)
        
        # Rough estimation: 0.5 seconds per step on MacBook
        steps_per_minute = 120 / batch_size  # Slower with larger batches
        training_time_minutes = max_steps / steps_per_minute
        
        return training_time_minutes
    
    def validate_config(self, config: Dict[str, Any], 
                       hardware_specs: Optional[HardwareSpecs] = None) -> ValidationResult:
        """Validate training configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        # Detect hardware if not provided
        if hardware_specs is None:
            # Default to 8GB MacBook for conservative validation
            hardware_specs = HardwareSpecs(
                memory_gb=8.0,
                cpu_cores=4,
                model_category=MacBookModel.MACBOOK_8GB
            )
        
        limits = self.memory_limits[hardware_specs.model_category]
        
        # Validate model configuration
        model_config = config.get("model", {})
        
        # Vocabulary size
        vocab_size = model_config.get("vocab_size", 5000)
        if vocab_size > limits["max_vocab_size"]:
            errors.append(f"Vocabulary size {vocab_size} exceeds limit {limits['max_vocab_size']} for {hardware_specs.model_category.value}")
        
        # Hidden size
        hidden_size = model_config.get("hidden_size", 512)
        if hidden_size > limits["max_hidden_size"]:
            errors.append(f"Hidden size {hidden_size} exceeds limit {limits['max_hidden_size']} for {hardware_specs.model_category.value}")
        
        # Sequence length
        seq_length = model_config.get("max_sequence_length", 512)
        if seq_length > limits["max_sequence_length"]:
            warnings.append(f"Sequence length {seq_length} may cause memory issues on {hardware_specs.model_category.value}")
        
        # Validate training configuration
        training_config = config.get("training", {})
        
        # Batch size
        batch_size = training_config.get("batch_size", 4)
        if batch_size > limits["max_batch_size"]:
            errors.append(f"Batch size {batch_size} exceeds limit {limits['max_batch_size']} for {hardware_specs.model_category.value}")
        
        # Gradient accumulation
        grad_accum = training_config.get("gradient_accumulation_steps", 1)
        effective_batch = batch_size * grad_accum
        if effective_batch < 16:
            warnings.append(f"Effective batch size {effective_batch} may be too small for stable training")
        
        # Memory limit
        memory_limit = config.get("hardware", {}).get("memory_limit_mb", 6000)
        if memory_limit > limits["recommended_memory_limit"]:
            warnings.append(f"Memory limit {memory_limit}MB exceeds recommended {limits['recommended_memory_limit']}MB")
        
        # Estimate memory usage
        estimated_memory = self.estimate_memory_usage(config)
        if estimated_memory > memory_limit:
            errors.append(f"Estimated memory usage {estimated_memory:.0f}MB exceeds limit {memory_limit}MB")
        
        # Check for required fields
        required_fields = [
            ("model", "vocab_size"),
            ("model", "hidden_size"),
            ("model", "num_layers"),
            ("training", "batch_size"),
            ("training", "learning_rate"),
            ("training", "max_steps")
        ]
        
        for section, field in required_fields:
            if section not in config or field not in config[section]:
                errors.append(f"Missing required field: {section}.{field}")
        
        # Generate recommendations
        if hardware_specs.model_category == MacBookModel.MACBOOK_8GB:
            recommendations.extend([
                "Consider using gradient accumulation to achieve larger effective batch sizes",
                "Enable memory monitoring and dynamic batch sizing",
                "Use streaming data loading for large datasets",
                "Consider reducing sequence length to 256 for memory efficiency"
            ])
        elif hardware_specs.model_category == MacBookModel.MACBOOK_16GB:
            recommendations.extend([
                "You can use moderate batch sizes and model complexity",
                "Consider enabling hierarchical attention for better performance",
                "Cache preprocessed data for faster training"
            ])
        else:  # 32GB+
            recommendations.extend([
                "You can use larger models and batch sizes",
                "Consider ensemble training for better accuracy",
                "Enable advanced features like curriculum learning"
            ])
        
        # Estimate training time
        estimated_time = self.estimate_training_time(config)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            estimated_memory_usage=estimated_memory,
            estimated_training_time=estimated_time
        )
    
    def recommend_config(self, hardware_specs: HardwareSpecs, 
                        dataset_size: int = 1000,
                        target_accuracy: float = 0.95) -> Dict[str, Any]:
        """Recommend optimal configuration for given hardware."""
        limits = self.memory_limits[hardware_specs.model_category]
        
        # Base configuration
        config = {
            "model": {
                "name": "EmailTRM",
                "vocab_size": min(5000, limits["max_vocab_size"]),
                "hidden_size": min(384, limits["max_hidden_size"]),
                "num_layers": 2,
                "num_email_categories": 10,
                "max_sequence_length": min(512, limits["max_sequence_length"])
            },
            "training": {
                "batch_size": min(4, limits["max_batch_size"]),
                "gradient_accumulation_steps": 8,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_epochs": 10,
                "max_steps": 5000
            },
            "email": {
                "use_email_structure": True,
                "subject_attention_weight": 2.0,
                "pooling_strategy": "weighted",
                "enable_subject_prioritization": True,
                "use_hierarchical_attention": hardware_specs.model_category != MacBookModel.MACBOOK_8GB,
                "email_augmentation_prob": 0.3
            },
            "hardware": {
                "memory_limit_mb": limits["recommended_memory_limit"],
                "enable_memory_monitoring": True,
                "dynamic_batch_sizing": hardware_specs.model_category == MacBookModel.MACBOOK_8GB,
                "use_cpu_optimization": True,
                "num_workers": min(2, hardware_specs.cpu_cores // 2)
            },
            "targets": {
                "target_accuracy": target_accuracy,
                "min_category_accuracy": max(0.85, target_accuracy - 0.1),
                "early_stopping_patience": 5
            }
        }
        
        # Adjust for hardware category
        if hardware_specs.model_category == MacBookModel.MACBOOK_16GB:
            config["model"]["hidden_size"] = 384
            config["training"]["batch_size"] = 4
            config["training"]["max_steps"] = 8000
            config["hardware"]["num_workers"] = 2
        
        elif hardware_specs.model_category == MacBookModel.MACBOOK_32GB:
            config["model"]["vocab_size"] = 8000
            config["model"]["hidden_size"] = 512
            config["model"]["num_layers"] = 3
            config["training"]["batch_size"] = 8
            config["training"]["gradient_accumulation_steps"] = 4
            config["training"]["max_steps"] = 10000
            config["hardware"]["memory_limit_mb"] = 24000
            config["hardware"]["num_workers"] = 4
            config["email"]["pooling_strategy"] = "attention"
        
        # Adjust for dataset size
        if dataset_size > 10000:
            config["training"]["max_steps"] = min(config["training"]["max_steps"] * 2, 20000)
            config["email"]["email_augmentation_prob"] = 0.2  # Less augmentation for large datasets
        elif dataset_size < 500:
            config["training"]["max_steps"] = max(config["training"]["max_steps"] // 2, 2000)
            config["email"]["email_augmentation_prob"] = 0.5  # More augmentation for small datasets
        
        return config
    
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def save_config_to_file(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


def main():
    """Command-line interface for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MacBook Email Training Configuration Validator")
    parser.add_argument("--config", type=str, help="Configuration file to validate")
    parser.add_argument("--memory-gb", type=float, default=8.0, help="MacBook memory in GB")
    parser.add_argument("--cpu-cores", type=int, default=4, help="Number of CPU cores")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Dataset size for recommendations")
    parser.add_argument("--recommend", action="store_true", help="Generate recommended configuration")
    parser.add_argument("--output", type=str, help="Output file for recommended configuration")
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    # Create hardware specs
    model_category = validator.detect_macbook_model(args.memory_gb)
    hardware_specs = HardwareSpecs(
        memory_gb=args.memory_gb,
        cpu_cores=args.cpu_cores,
        model_category=model_category
    )
    
    print(f"Detected MacBook category: {model_category.value}")
    print(f"Hardware: {args.memory_gb}GB memory, {args.cpu_cores} CPU cores")
    
    if args.recommend:
        # Generate recommended configuration
        print("\nGenerating recommended configuration...")
        config = validator.recommend_config(hardware_specs, args.dataset_size)
        
        # Validate the recommended configuration
        result = validator.validate_config(config, hardware_specs)
        
        print(f"Estimated memory usage: {result.estimated_memory_usage:.0f}MB")
        print(f"Estimated training time: {result.estimated_training_time:.1f} minutes")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        # Save configuration if output specified
        if args.output:
            validator.save_config_to_file(config, args.output)
            print(f"\nConfiguration saved to: {args.output}")
        else:
            print("\nRecommended configuration:")
            print(yaml.dump(config, default_flow_style=False, indent=2))
    
    elif args.config:
        # Validate existing configuration
        print(f"\nValidating configuration: {args.config}")
        config = validator.load_config_from_file(args.config)
        result = validator.validate_config(config, hardware_specs)
        
        print(f"Validation result: {'VALID' if result.is_valid else 'INVALID'}")
        print(f"Estimated memory usage: {result.estimated_memory_usage:.0f}MB")
        print(f"Estimated training time: {result.estimated_training_time:.1f} minutes")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()