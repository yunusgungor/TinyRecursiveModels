"""
EmailTRM Production Setup - Complete Integration

This module provides a unified interface for setting up EmailTRM models for production
training, integrating all the configuration, optimization, and enhancement components.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from macbook_optimization.email_trm_config import (
    EmailTRMConfigManager, ProductionEmailTRMConfig, create_production_email_trm
)
from macbook_optimization.recursive_reasoning_optimizer import (
    RecursiveReasoningOptimizer, get_optimal_reasoning_config
)
from macbook_optimization.email_feature_enhancer import (
    EmailFeatureEnhancer, EmailFeatureConfig, enhance_email_trm_for_production
)
from macbook_optimization.email_trm_hardware_optimizer import (
    MacBookHardwareOptimizer, MacBookOptimizationConfig, optimize_email_trm_for_macbook
)
from macbook_optimization.hardware_detection import HardwareDetector


logger = logging.getLogger(__name__)


@dataclass
class ProductionSetupResult:
    """Result of production EmailTRM setup"""
    
    model: EmailTRM
    production_config: ProductionEmailTRMConfig
    feature_config: EmailFeatureConfig
    optimization_config: MacBookOptimizationConfig
    hardware_summary: Dict[str, Any]
    performance_estimates: Dict[str, float]
    setup_summary: Dict[str, Any]


class EmailTRMProductionSetup:
    """Complete production setup for EmailTRM models"""
    
    def __init__(self):
        self.hardware_detector = HardwareDetector()
        self.config_manager = EmailTRMConfigManager(self.hardware_detector)
        self.reasoning_optimizer = RecursiveReasoningOptimizer(self.hardware_detector)
        self.feature_enhancer = EmailFeatureEnhancer()
        self.hardware_optimizer = MacBookHardwareOptimizer(self.hardware_detector)
        
        # Get hardware summary once
        self.hardware_summary = self.hardware_detector.get_hardware_summary()
        
        logger.info("EmailTRM Production Setup initialized")
    
    def setup_production_model(self, 
                             vocab_size: int,
                             target_accuracy: float = 0.95,
                             memory_conservative: bool = False,
                             enable_all_features: bool = True) -> ProductionSetupResult:
        """
        Complete setup of EmailTRM model for production training
        
        Args:
            vocab_size: Vocabulary size from dataset
            target_accuracy: Target accuracy requirement
            memory_conservative: Whether to use memory-conservative settings
            enable_all_features: Whether to enable all email-specific features
            
        Returns:
            ProductionSetupResult with configured model and settings
        """
        
        logger.info(f"Setting up production EmailTRM model (vocab_size={vocab_size}, "
                   f"target_accuracy={target_accuracy})")
        
        # Step 1: Create optimized base configuration
        logger.info("Step 1: Creating optimized base configuration")
        production_config = self.config_manager.create_optimized_config(vocab_size)
        
        # Step 2: Optimize recursive reasoning parameters
        logger.info("Step 2: Optimizing recursive reasoning parameters")
        reasoning_config = get_optimal_reasoning_config(
            vocab_size, target_accuracy, self.hardware_detector
        )
        
        # Apply reasoning optimizations to production config
        reasoning_profile = reasoning_config['profile']
        production_config.H_cycles = reasoning_profile.H_cycles
        production_config.L_cycles = reasoning_profile.L_cycles
        production_config.halt_max_steps = reasoning_profile.halt_max_steps
        production_config.halt_exploration_prob = reasoning_profile.halt_exploration_prob
        
        # Step 3: Initialize base model
        logger.info("Step 3: Initializing base EmailTRM model")
        model = self.config_manager.initialize_email_trm_model(production_config)
        
        # Step 4: Configure email-specific features
        logger.info("Step 4: Configuring email-specific features")
        if enable_all_features:
            feature_config = EmailFeatureConfig(
                enable_structure_embeddings=True,
                enable_hierarchical_attention=True,
                enable_confidence_calibration=True,
                enable_category_embeddings=True,
                enable_prediction_explanations=True,
                enable_uncertainty_estimation=True,
                pooling_strategy="hierarchical"
            )
        else:
            feature_config = EmailFeatureConfig(
                enable_structure_embeddings=True,
                enable_hierarchical_attention=True,
                enable_confidence_calibration=True,
                enable_category_embeddings=False,
                enable_prediction_explanations=False,
                enable_uncertainty_estimation=False,
                pooling_strategy="weighted"
            )
        
        # Apply feature enhancements
        model = enhance_email_trm_for_production(model, feature_config)
        
        # Step 5: Apply MacBook hardware optimizations
        logger.info("Step 5: Applying MacBook hardware optimizations")
        optimization_config = MacBookOptimizationConfig(
            enable_gradient_checkpointing=True,
            enable_memory_efficient_attention=True,
            dynamic_batch_sizing=True,
            memory_threshold_mb=1500.0 if memory_conservative else 1000.0,
            gradient_accumulation_steps=16 if memory_conservative else 8,
            max_batch_size=8 if memory_conservative else 16,
            enable_performance_monitoring=True,
            enable_thermal_monitoring=True
        )
        
        model = optimize_email_trm_for_macbook(model, optimization_config)
        
        # Step 6: Validate and estimate performance
        logger.info("Step 6: Validating configuration and estimating performance")
        validation_results = self.config_manager.validate_model_configuration(
            model, production_config
        )
        
        # Create performance estimates
        performance_estimates = {
            'estimated_memory_usage_gb': validation_results['resource_estimates']['estimated_memory_gb'],
            'estimated_inference_time_ms': reasoning_config['recommendations']['performance_predictions']['expected_inference_time_ms'],
            'expected_accuracy_range': reasoning_config['recommendations']['performance_predictions']['expected_accuracy_range'],
            'optimal_batch_size': production_config.max_batch_size,
            'optimal_sequence_length': production_config.max_sequence_length
        }
        
        # Create setup summary
        setup_summary = {
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024**2,
            'hardware_tier': self._get_hardware_tier(),
            'optimizations_applied': {
                'recursive_reasoning': True,
                'email_features': enable_all_features,
                'hardware_optimization': True,
                'memory_efficiency': True,
                'thermal_monitoring': True
            },
            'configuration_summary': {
                'H_cycles': production_config.H_cycles,
                'L_cycles': production_config.L_cycles,
                'hidden_size': production_config.hidden_size,
                'email_categories': production_config.num_email_categories,
                'structure_awareness': production_config.use_email_structure,
                'hierarchical_attention': production_config.use_hierarchical_attention
            },
            'warnings': validation_results.get('warnings', [])
        }
        
        logger.info("Production EmailTRM setup completed successfully")
        
        return ProductionSetupResult(
            model=model,
            production_config=production_config,
            feature_config=feature_config,
            optimization_config=optimization_config,
            hardware_summary=self.hardware_summary,
            performance_estimates=performance_estimates,
            setup_summary=setup_summary
        )
    
    def _get_hardware_tier(self) -> str:
        """Determine hardware tier based on specifications"""
        
        memory_gb = self.hardware_summary['memory']['total_gb']
        cpu_cores = self.hardware_summary['cpu']['cores']
        
        if memory_gb >= 16 and cpu_cores >= 8:
            return "high"
        elif memory_gb >= 8 and cpu_cores >= 4:
            return "medium"
        else:
            return "low"
    
    def create_training_config(self, 
                             setup_result: ProductionSetupResult,
                             dataset_path: str) -> Dict[str, Any]:
        """
        Create complete training configuration based on setup result
        
        Args:
            setup_result: Result from setup_production_model
            dataset_path: Path to training dataset
            
        Returns:
            Complete training configuration
        """
        
        base_config = {
            'dataset_path': dataset_path,
            'model_config': setup_result.production_config,
            'max_sequence_length': setup_result.production_config.max_sequence_length,
            'num_email_categories': setup_result.production_config.num_email_categories,
            'target_accuracy': setup_result.production_config.target_accuracy,
            'min_category_accuracy': setup_result.production_config.min_category_accuracy,
        }
        
        # Apply hardware optimizations
        training_config = self.hardware_optimizer.create_optimized_training_config(base_config)
        
        # Add production-specific settings
        training_config.update({
            'enable_recursive_reasoning': True,
            'H_cycles': setup_result.production_config.H_cycles,
            'L_cycles': setup_result.production_config.L_cycles,
            'halt_max_steps': setup_result.production_config.halt_max_steps,
            'halt_exploration_prob': setup_result.production_config.halt_exploration_prob,
            
            'enable_email_features': True,
            'use_email_structure': setup_result.production_config.use_email_structure,
            'use_hierarchical_attention': setup_result.production_config.use_hierarchical_attention,
            'pooling_strategy': setup_result.production_config.pooling_strategy,
            
            'enable_confidence_calibration': setup_result.feature_config.enable_confidence_calibration,
            'enable_category_embeddings': setup_result.feature_config.enable_category_embeddings,
            
            'hardware_optimization': True,
            'gradient_checkpointing': setup_result.optimization_config.enable_gradient_checkpointing,
            'memory_efficient_attention': setup_result.optimization_config.enable_memory_efficient_attention,
            'performance_monitoring': setup_result.optimization_config.enable_performance_monitoring,
        })
        
        return training_config
    
    def save_production_setup(self, 
                            setup_result: ProductionSetupResult, 
                            output_dir: Path) -> None:
        """
        Save complete production setup to directory
        
        Args:
            setup_result: Setup result to save
            output_dir: Output directory
        """
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configurations
        self.config_manager.save_config(
            setup_result.production_config, 
            output_dir / "production_config.json"
        )
        
        # Save model state dict
        torch.save(
            setup_result.model.state_dict(), 
            output_dir / "model_state_dict.pt"
        )
        
        # Save setup summary
        import json
        with open(output_dir / "setup_summary.json", 'w') as f:
            json.dump(setup_result.setup_summary, f, indent=2)
        
        # Save hardware summary
        with open(output_dir / "hardware_summary.json", 'w') as f:
            json.dump(setup_result.hardware_summary, f, indent=2)
        
        # Save performance estimates
        with open(output_dir / "performance_estimates.json", 'w') as f:
            json.dump(setup_result.performance_estimates, f, indent=2)
        
        logger.info(f"Production setup saved to {output_dir}")
    
    def load_production_setup(self, 
                            setup_dir: Path, 
                            vocab_size: int) -> ProductionSetupResult:
        """
        Load production setup from directory
        
        Args:
            setup_dir: Directory containing saved setup
            vocab_size: Vocabulary size for model creation
            
        Returns:
            Loaded ProductionSetupResult
        """
        
        # Load configurations
        production_config = self.config_manager.load_config(
            setup_dir / "production_config.json"
        )
        
        # Create model
        model = self.config_manager.initialize_email_trm_model(production_config)
        
        # Load model state dict
        state_dict = torch.load(setup_dir / "model_state_dict.pt", map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Load other components
        import json
        
        with open(setup_dir / "setup_summary.json", 'r') as f:
            setup_summary = json.load(f)
        
        with open(setup_dir / "hardware_summary.json", 'r') as f:
            hardware_summary = json.load(f)
        
        with open(setup_dir / "performance_estimates.json", 'r') as f:
            performance_estimates = json.load(f)
        
        # Create dummy configs (would need to be saved/loaded properly in practice)
        feature_config = EmailFeatureConfig()
        optimization_config = MacBookOptimizationConfig()
        
        logger.info(f"Production setup loaded from {setup_dir}")
        
        return ProductionSetupResult(
            model=model,
            production_config=production_config,
            feature_config=feature_config,
            optimization_config=optimization_config,
            hardware_summary=hardware_summary,
            performance_estimates=performance_estimates,
            setup_summary=setup_summary
        )
    
    def get_setup_recommendations(self, vocab_size: int) -> Dict[str, Any]:
        """
        Get setup recommendations based on hardware
        
        Args:
            vocab_size: Vocabulary size
            
        Returns:
            Setup recommendations
        """
        
        recommendations = {
            'hardware_analysis': {},
            'recommended_configuration': {},
            'performance_expectations': {},
            'optimization_suggestions': []
        }
        
        # Hardware analysis
        memory_gb = self.hardware_summary['memory']['total_gb']
        cpu_cores = self.hardware_summary['cpu']['cores']
        
        recommendations['hardware_analysis'] = {
            'memory_tier': self._get_hardware_tier(),
            'suitable_for_training': memory_gb >= 4,
            'expected_training_speed': 'fast' if cpu_cores >= 8 else 'medium' if cpu_cores >= 4 else 'slow',
            'memory_constraint_level': 'low' if memory_gb >= 12 else 'medium' if memory_gb >= 8 else 'high'
        }
        
        # Configuration recommendations
        if memory_gb >= 12:
            config_profile = "high_performance"
            batch_size = 16
            hidden_size = 768
        elif memory_gb >= 8:
            config_profile = "balanced"
            batch_size = 8
            hidden_size = 512
        else:
            config_profile = "efficient"
            batch_size = 4
            hidden_size = 256
        
        recommendations['recommended_configuration'] = {
            'profile': config_profile,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
            'enable_all_features': memory_gb >= 8,
            'memory_conservative': memory_gb < 8,
            'gradient_accumulation_steps': max(1, 32 // batch_size)
        }
        
        # Performance expectations
        reasoning_config = get_optimal_reasoning_config(vocab_size, 0.95, self.hardware_detector)
        
        recommendations['performance_expectations'] = {
            'expected_accuracy': '93-97%',
            'training_time_estimate': self._estimate_training_time(memory_gb, cpu_cores),
            'inference_speed_ms': reasoning_config['recommendations']['performance_predictions']['expected_inference_time_ms'],
            'memory_usage_gb': reasoning_config['recommendations']['performance_predictions']['expected_inference_time_ms'] / 100  # Rough estimate
        }
        
        # Optimization suggestions
        if memory_gb < 8:
            recommendations['optimization_suggestions'].extend([
                "Consider using memory-conservative settings",
                "Enable gradient checkpointing",
                "Use smaller batch sizes with gradient accumulation"
            ])
        
        if cpu_cores < 4:
            recommendations['optimization_suggestions'].append(
                "Training may be slow due to limited CPU cores"
            )
        
        if not self.hardware_summary['platform']['supports_avx2']:
            recommendations['optimization_suggestions'].append(
                "CPU lacks AVX2 support, consider upgrading for better performance"
            )
        
        return recommendations
    
    def _estimate_training_time(self, memory_gb: float, cpu_cores: int) -> str:
        """Estimate training time based on hardware"""
        
        # Very rough estimates based on typical email classification training
        if memory_gb >= 12 and cpu_cores >= 8:
            return "2-4 hours for full training"
        elif memory_gb >= 8 and cpu_cores >= 4:
            return "4-8 hours for full training"
        else:
            return "8-16 hours for full training"


# Convenience functions
def setup_production_email_trm(vocab_size: int, 
                             target_accuracy: float = 0.95,
                             memory_conservative: bool = False) -> ProductionSetupResult:
    """
    Convenience function to set up production EmailTRM model
    
    Args:
        vocab_size: Vocabulary size from dataset
        target_accuracy: Target accuracy requirement
        memory_conservative: Whether to use memory-conservative settings
        
    Returns:
        Complete production setup result
    """
    
    setup = EmailTRMProductionSetup()
    return setup.setup_production_model(
        vocab_size=vocab_size,
        target_accuracy=target_accuracy,
        memory_conservative=memory_conservative,
        enable_all_features=not memory_conservative
    )


def get_production_recommendations(vocab_size: int) -> Dict[str, Any]:
    """
    Get production setup recommendations
    
    Args:
        vocab_size: Vocabulary size
        
    Returns:
        Setup recommendations
    """
    
    setup = EmailTRMProductionSetup()
    return setup.get_setup_recommendations(vocab_size)


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test complete production setup
    vocab_size = 5000
    
    print("Setting up production EmailTRM model...")
    setup_result = setup_production_email_trm(
        vocab_size=vocab_size,
        target_accuracy=0.95,
        memory_conservative=False
    )
    
    print(f"\nProduction setup completed!")
    print(f"Model parameters: {setup_result.setup_summary['model_parameters']:,}")
    print(f"Model size: {setup_result.setup_summary['model_size_mb']:.1f}MB")
    print(f"Hardware tier: {setup_result.setup_summary['hardware_tier']}")
    print(f"Expected accuracy: {setup_result.performance_estimates['expected_accuracy_range']}")
    print(f"Optimal batch size: {setup_result.performance_estimates['optimal_batch_size']}")
    
    # Test recommendations
    print(f"\nGetting setup recommendations...")
    recommendations = get_production_recommendations(vocab_size)
    print(f"Recommended profile: {recommendations['recommended_configuration']['profile']}")
    print(f"Training time estimate: {recommendations['performance_expectations']['training_time_estimate']}")
    
    if recommendations['optimization_suggestions']:
        print(f"Optimization suggestions:")
        for suggestion in recommendations['optimization_suggestions']:
            print(f"  - {suggestion}")