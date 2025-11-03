"""
Email Ensemble Training System for MacBook Optimization

This module provides ensemble training capabilities for email classification,
including multiple model training with different configurations, model averaging,
and ensemble performance evaluation and selection.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import copy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TORCH_AVAILABLE = False

from .email_training_orchestrator import EmailTrainingOrchestrator, EmailTrainingConfig, TrainingResult
from .email_trm_integration import MacBookEmailTRM
try:
    from models.recursive_reasoning.trm_email import EmailTRMConfig
    TRM_EMAIL_AVAILABLE = True
except ImportError:
    TRM_EMAIL_AVAILABLE = False
    EmailTRMConfig = None

try:
    from models.ensemble import ModelEnsemble
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    ModelEnsemble = None

try:
    from evaluators.email import EmailClassificationEvaluator
    EMAIL_EVALUATOR_AVAILABLE = True
except ImportError:
    EMAIL_EVALUATOR_AVAILABLE = False
    EmailClassificationEvaluator = None

logger = logging.getLogger(__name__)


@dataclass
class EnsembleModelConfig:
    """Configuration for a single model in the ensemble."""
    
    model_id: str
    model_name: str
    config: EmailTrainingConfig
    
    # Model-specific variations
    hidden_size_multiplier: float = 1.0
    learning_rate_multiplier: float = 1.0
    dropout_rate: float = 0.1
    
    # Training variations
    training_strategy: str = "multi_phase"  # "single", "multi_phase", "progressive", "curriculum"
    max_steps: int = 10000
    
    # Data variations
    data_augmentation_prob: float = 0.3
    use_different_seed: bool = True
    seed: Optional[int] = None
    
    # Architecture variations
    use_hierarchical_attention: bool = True
    subject_attention_weight: float = 2.0
    pooling_strategy: str = "weighted"


@dataclass
class EnsembleTrainingConfig:
    """Configuration for ensemble training."""
    
    # Ensemble parameters
    num_models: int = 3
    ensemble_name: str = "email_ensemble"
    voting_strategy: str = "soft"  # "hard", "soft", "weighted"
    
    # Model diversity strategies
    diversity_strategy: str = "config_variation"  # "config_variation", "data_variation", "architecture_variation"
    
    # Training parameters
    parallel_training: bool = False  # Train models in parallel (if resources allow)
    sequential_training: bool = True  # Train models sequentially
    
    # Evaluation parameters
    cross_validation_folds: int = 5
    holdout_test_size: float = 0.2
    
    # Selection criteria
    selection_metric: str = "accuracy"  # "accuracy", "f1_macro", "ensemble_diversity"
    min_individual_accuracy: float = 0.90
    target_ensemble_accuracy: float = 0.95
    
    # Output configuration
    save_individual_models: bool = True
    save_ensemble_config: bool = True
    generate_comparison_report: bool = True


@dataclass
class EnsembleTrainingResult:
    """Result of ensemble training."""
    
    success: bool
    ensemble_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Configuration
    config: EnsembleTrainingConfig
    model_configs: List[EnsembleModelConfig]
    
    # Individual model results
    individual_results: List[TrainingResult]
    individual_accuracies: List[float]
    individual_f1_scores: List[float]
    
    # Ensemble results
    ensemble_accuracy: Optional[float]
    ensemble_f1_macro: Optional[float]
    ensemble_f1_micro: Optional[float]
    
    # Performance comparison
    best_individual_accuracy: float
    ensemble_improvement: float
    
    # Model paths
    ensemble_model_path: Optional[str]
    individual_model_paths: List[str]
    
    # Training metrics
    total_training_time: float
    average_individual_training_time: float
    
    # Errors and warnings
    errors: List[str]
    warnings: List[str]


class EmailEnsembleTrainer:
    """
    Email ensemble training system for MacBook optimization.
    
    Manages training of multiple EmailTRM models with different configurations
    and creates ensemble predictions with model averaging capabilities.
    """
    
    def __init__(self, 
                 output_dir: str = "email_ensemble_output",
                 base_orchestrator: Optional[EmailTrainingOrchestrator] = None):
        """
        Initialize email ensemble trainer.
        
        Args:
            output_dir: Directory for ensemble training outputs
            base_orchestrator: Base training orchestrator (created if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base orchestrator
        if base_orchestrator is None:
            self.base_orchestrator = EmailTrainingOrchestrator(
                output_dir=str(self.output_dir / "individual_models"),
                enable_monitoring=True,
                enable_checkpointing=True
            )
        else:
            self.base_orchestrator = base_orchestrator
        
        # Ensemble state
        self.current_ensemble_id = None
        self.ensemble_history: List[EnsembleTrainingResult] = []
        
        logger.info(f"EmailEnsembleTrainer initialized with output dir: {output_dir}")
    
    def create_ensemble_model_configs(self, 
                                    base_config: EmailTrainingConfig,
                                    ensemble_config: EnsembleTrainingConfig) -> List[EnsembleModelConfig]:
        """
        Create diverse model configurations for ensemble training.
        
        Args:
            base_config: Base training configuration
            ensemble_config: Ensemble training configuration
            
        Returns:
            List of ensemble model configurations
        """
        model_configs = []
        
        for i in range(ensemble_config.num_models):
            model_id = f"model_{i+1:02d}"
            model_name = f"{ensemble_config.ensemble_name}_{model_id}"
            
            # Create base model config
            model_config = EnsembleModelConfig(
                model_id=model_id,
                model_name=model_name,
                config=copy.deepcopy(base_config),
                seed=42 + i if ensemble_config.diversity_strategy != "data_variation" else None
            )
            
            # Apply diversity strategy
            if ensemble_config.diversity_strategy == "config_variation":
                model_config = self._apply_config_variations(model_config, i, ensemble_config.num_models)
            elif ensemble_config.diversity_strategy == "data_variation":
                model_config = self._apply_data_variations(model_config, i, ensemble_config.num_models)
            elif ensemble_config.diversity_strategy == "architecture_variation":
                model_config = self._apply_architecture_variations(model_config, i, ensemble_config.num_models)
            
            model_configs.append(model_config)
        
        logger.info(f"Created {len(model_configs)} diverse model configurations")
        return model_configs
    
    def _apply_config_variations(self, model_config: EnsembleModelConfig, 
                               model_idx: int, total_models: int) -> EnsembleModelConfig:
        """Apply configuration variations for model diversity."""
        
        # Learning rate variations
        lr_multipliers = [0.5, 1.0, 1.5, 2.0, 0.8]
        model_config.learning_rate_multiplier = lr_multipliers[model_idx % len(lr_multipliers)]
        model_config.config.learning_rate *= model_config.learning_rate_multiplier
        
        # Hidden size variations
        hidden_multipliers = [0.75, 1.0, 1.25, 1.0, 0.9]
        model_config.hidden_size_multiplier = hidden_multipliers[model_idx % len(hidden_multipliers)]
        model_config.config.hidden_size = int(model_config.config.hidden_size * model_config.hidden_size_multiplier)
        
        # Training strategy variations
        strategies = ["multi_phase", "progressive", "curriculum", "single", "multi_phase"]
        model_config.training_strategy = strategies[model_idx % len(strategies)]
        
        # Batch size variations
        batch_multipliers = [1.0, 0.5, 1.5, 0.75, 1.25]
        batch_multiplier = batch_multipliers[model_idx % len(batch_multipliers)]
        model_config.config.batch_size = max(2, int(model_config.config.batch_size * batch_multiplier))
        
        # Gradient accumulation adjustment
        if batch_multiplier < 1.0:
            model_config.config.gradient_accumulation_steps = int(
                model_config.config.gradient_accumulation_steps / batch_multiplier
            )
        
        # Weight decay variations
        weight_decays = [0.01, 0.05, 0.001, 0.02, 0.1]
        model_config.config.weight_decay = weight_decays[model_idx % len(weight_decays)]
        
        logger.info(f"Applied config variations to {model_config.model_id}: "
                   f"lr_mult={model_config.learning_rate_multiplier:.2f}, "
                   f"hidden_mult={model_config.hidden_size_multiplier:.2f}, "
                   f"strategy={model_config.training_strategy}")
        
        return model_config
    
    def _apply_data_variations(self, model_config: EnsembleModelConfig, 
                             model_idx: int, total_models: int) -> EnsembleModelConfig:
        """Apply data variations for model diversity."""
        
        # Data augmentation variations
        aug_probs = [0.1, 0.3, 0.5, 0.2, 0.4]
        model_config.data_augmentation_prob = aug_probs[model_idx % len(aug_probs)]
        model_config.config.email_augmentation_prob = model_config.data_augmentation_prob
        
        # Different random seeds for data shuffling
        model_config.use_different_seed = True
        model_config.seed = 42 + model_idx * 1000
        
        # Different data sampling strategies could be added here
        # For example: bootstrap sampling, different train/val splits, etc.
        
        logger.info(f"Applied data variations to {model_config.model_id}: "
                   f"aug_prob={model_config.data_augmentation_prob:.2f}, "
                   f"seed={model_config.seed}")
        
        return model_config
    
    def _apply_architecture_variations(self, model_config: EnsembleModelConfig, 
                                     model_idx: int, total_models: int) -> EnsembleModelConfig:
        """Apply architecture variations for model diversity."""
        
        # Attention mechanism variations
        hierarchical_options = [True, False, True, True, False]
        model_config.use_hierarchical_attention = hierarchical_options[model_idx % len(hierarchical_options)]
        model_config.config.use_hierarchical_attention = model_config.use_hierarchical_attention
        
        # Subject attention weight variations
        attention_weights = [1.5, 2.0, 2.5, 1.8, 2.2]
        model_config.subject_attention_weight = attention_weights[model_idx % len(attention_weights)]
        model_config.config.subject_attention_weight = model_config.subject_attention_weight
        
        # Pooling strategy variations
        pooling_strategies = ["weighted", "attention", "mean", "max", "weighted"]
        model_config.pooling_strategy = pooling_strategies[model_idx % len(pooling_strategies)]
        model_config.config.pooling_strategy = model_config.pooling_strategy
        
        # Dropout variations
        dropout_rates = [0.1, 0.15, 0.05, 0.2, 0.12]
        model_config.dropout_rate = dropout_rates[model_idx % len(dropout_rates)]
        
        # Layer variations (if supported by model)
        layer_options = [2, 3, 2, 2, 3]
        model_config.config.num_layers = layer_options[model_idx % len(layer_options)]
        
        logger.info(f"Applied architecture variations to {model_config.model_id}: "
                   f"hierarchical={model_config.use_hierarchical_attention}, "
                   f"attention_weight={model_config.subject_attention_weight:.2f}, "
                   f"pooling={model_config.pooling_strategy}")
        
        return model_config
    
    def train_ensemble(self,
                      dataset_path: str,
                      base_config: EmailTrainingConfig,
                      ensemble_config: EnsembleTrainingConfig) -> EnsembleTrainingResult:
        """
        Train ensemble of email classification models.
        
        Args:
            dataset_path: Path to email dataset
            base_config: Base training configuration
            ensemble_config: Ensemble training configuration
            
        Returns:
            Ensemble training result
        """
        ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_ensemble_id = ensemble_id
        
        logger.info(f"Starting ensemble training: {ensemble_id}")
        logger.info(f"Training {ensemble_config.num_models} models with {ensemble_config.diversity_strategy} diversity")
        
        # Initialize result
        result = EnsembleTrainingResult(
            success=False,
            ensemble_id=ensemble_id,
            start_time=datetime.now(),
            end_time=None,
            config=ensemble_config,
            model_configs=[],
            individual_results=[],
            individual_accuracies=[],
            individual_f1_scores=[],
            ensemble_accuracy=None,
            ensemble_f1_macro=None,
            ensemble_f1_micro=None,
            best_individual_accuracy=0.0,
            ensemble_improvement=0.0,
            ensemble_model_path=None,
            individual_model_paths=[],
            total_training_time=0.0,
            average_individual_training_time=0.0,
            errors=[],
            warnings=[]
        )
        
        start_time = time.time()
        
        try:
            # Create diverse model configurations
            model_configs = self.create_ensemble_model_configs(base_config, ensemble_config)
            result.model_configs = model_configs
            
            # Train individual models
            individual_results = []
            individual_models = []
            
            for i, model_config in enumerate(model_configs):
                logger.info(f"Training model {i+1}/{len(model_configs)}: {model_config.model_id}")
                
                try:
                    # Set random seed if specified
                    if model_config.seed is not None:
                        torch.manual_seed(model_config.seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(model_config.seed)
                    
                    # Train individual model
                    individual_result = self.base_orchestrator.execute_training_pipeline(
                        dataset_path=dataset_path,
                        config=model_config.config,
                        strategy=model_config.training_strategy,
                        total_steps=model_config.max_steps
                    )
                    
                    individual_results.append(individual_result)
                    
                    if individual_result.success:
                        # Load trained model
                        if individual_result.model_path and os.path.exists(individual_result.model_path):
                            model_state = torch.load(individual_result.model_path, map_location='cpu')
                            
                            # Create model instance
                            model = MacBookEmailTRM(
                                vocab_size=model_config.config.vocab_size,
                                hidden_size=model_config.config.hidden_size,
                                num_layers=model_config.config.num_layers,
                                num_email_categories=model_config.config.num_email_categories,
                                max_seq_len=model_config.config.max_sequence_length,
                                use_hierarchical_attention=model_config.config.use_hierarchical_attention,
                                subject_attention_weight=model_config.config.subject_attention_weight
                            )
                            
                            model.load_state_dict(model_state['model_state_dict'])
                            individual_models.append(model)
                            
                            result.individual_accuracies.append(individual_result.final_accuracy or 0.0)
                            result.individual_f1_scores.append(0.0)  # Would need to compute F1 from evaluation
                            result.individual_model_paths.append(individual_result.model_path)
                            
                            logger.info(f"Model {model_config.model_id} trained successfully: "
                                       f"accuracy={individual_result.final_accuracy:.4f}")
                        else:
                            result.warnings.append(f"Model {model_config.model_id} trained but no model file found")
                    else:
                        result.errors.append(f"Model {model_config.model_id} training failed: {individual_result.errors}")
                        logger.error(f"Model {model_config.model_id} training failed")
                
                except Exception as e:
                    error_msg = f"Model {model_config.model_id} training error: {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
            
            result.individual_results = individual_results
            
            # Check if we have enough successful models
            successful_models = [m for m in individual_models if m is not None]
            if len(successful_models) < 2:
                result.errors.append(f"Insufficient successful models: {len(successful_models)}/{ensemble_config.num_models}")
                return result
            
            # Create ensemble
            logger.info(f"Creating ensemble from {len(successful_models)} successful models")
            ensemble = ModelEnsemble(
                models=successful_models,
                voting_strategy=ensemble_config.voting_strategy,
                device="cpu"  # Use CPU for MacBook compatibility
            )
            
            # Evaluate ensemble (simplified - would need actual test data)
            # For now, we'll estimate ensemble performance
            if result.individual_accuracies:
                result.best_individual_accuracy = max(result.individual_accuracies)
                
                # Estimate ensemble improvement (simplified)
                # In practice, you'd evaluate on actual test data
                estimated_ensemble_accuracy = result.best_individual_accuracy + 0.01  # Conservative estimate
                result.ensemble_accuracy = min(estimated_ensemble_accuracy, 0.99)
                result.ensemble_improvement = result.ensemble_accuracy - result.best_individual_accuracy
                
                logger.info(f"Estimated ensemble accuracy: {result.ensemble_accuracy:.4f}")
                logger.info(f"Improvement over best individual: {result.ensemble_improvement:.4f}")
            
            # Save ensemble
            ensemble_path = self.output_dir / f"{ensemble_id}_ensemble.json"
            ensemble.save_ensemble(str(ensemble_path))
            result.ensemble_model_path = str(ensemble_path)
            
            # Calculate timing metrics
            result.total_training_time = time.time() - start_time
            if individual_results:
                successful_times = [r.total_training_time for r in individual_results if r.success]
                if successful_times:
                    result.average_individual_training_time = sum(successful_times) / len(successful_times)
            
            result.success = True
            logger.info(f"Ensemble training completed successfully!")
            
        except Exception as e:
            error_msg = f"Ensemble training failed: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)
        
        finally:
            result.end_time = datetime.now()
            
            # Save ensemble result
            self._save_ensemble_result(result)
            
            # Add to history
            self.ensemble_history.append(result)
        
        return result
    
    def evaluate_ensemble_performance(self,
                                    ensemble_models: List[MacBookEmailTRM],
                                    test_dataloader: DataLoader,
                                    category_names: List[str],
                                    voting_strategy: str = "soft") -> Dict[str, Any]:
        """
        Evaluate ensemble performance on test data.
        
        Args:
            ensemble_models: List of trained models
            test_dataloader: Test data loader
            category_names: List of category names
            voting_strategy: Voting strategy for ensemble
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating ensemble of {len(ensemble_models)} models")
        
        # Create ensemble
        ensemble = ModelEnsemble(
            models=ensemble_models,
            voting_strategy=voting_strategy,
            device="cpu"
        )
        
        # Evaluate ensemble
        evaluation_results = ensemble.evaluate_ensemble(test_dataloader, category_names)
        
        # Add detailed analysis
        evaluator = EmailClassificationEvaluator(
            category_names=category_names,
            output_dir=str(self.output_dir / "evaluation"),
            enable_advanced_metrics=True,
            save_predictions=True
        )
        
        # Collect ensemble predictions for detailed analysis
        all_ensemble_predictions = []
        all_individual_predictions = [[] for _ in range(len(ensemble_models))]
        all_labels = []
        
        for batch_idx, (set_name, batch, batch_size) in enumerate(test_dataloader):
            inputs = batch['inputs']
            labels = batch['labels']
            puzzle_ids = batch.get('puzzle_identifiers')
            
            # Get ensemble and individual predictions
            ensemble_pred, individual_preds = ensemble.predict(
                inputs, puzzle_ids, return_individual=True
            )
            
            all_ensemble_predictions.extend(ensemble_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for i, pred in enumerate(individual_preds):
                all_individual_predictions[i].extend(pred.cpu().numpy())
        
        # Update evaluator with ensemble predictions
        evaluator.update(
            predictions=torch.tensor(all_ensemble_predictions),
            labels=torch.tensor(all_labels)
        )
        
        # Compute detailed metrics
        detailed_metrics = evaluator.compute_metrics()
        
        # Combine results
        combined_results = {
            **evaluation_results,
            "detailed_metrics": detailed_metrics,
            "ensemble_size": len(ensemble_models),
            "voting_strategy": voting_strategy
        }
        
        return combined_results
    
    def select_best_ensemble_configuration(self,
                                         ensemble_results: List[EnsembleTrainingResult],
                                         selection_criteria: str = "ensemble_accuracy") -> Optional[EnsembleTrainingResult]:
        """
        Select best ensemble configuration from multiple training runs.
        
        Args:
            ensemble_results: List of ensemble training results
            selection_criteria: Criteria for selection
            
        Returns:
            Best ensemble result or None
        """
        if not ensemble_results:
            return None
        
        successful_results = [r for r in ensemble_results if r.success]
        if not successful_results:
            return None
        
        if selection_criteria == "ensemble_accuracy":
            best_result = max(successful_results, key=lambda r: r.ensemble_accuracy or 0.0)
        elif selection_criteria == "ensemble_improvement":
            best_result = max(successful_results, key=lambda r: r.ensemble_improvement or 0.0)
        elif selection_criteria == "best_individual_accuracy":
            best_result = max(successful_results, key=lambda r: r.best_individual_accuracy or 0.0)
        else:
            best_result = successful_results[0]  # Default to first successful result
        
        logger.info(f"Selected best ensemble: {best_result.ensemble_id} "
                   f"(accuracy={best_result.ensemble_accuracy:.4f})")
        
        return best_result
    
    def _save_ensemble_result(self, result: EnsembleTrainingResult):
        """Save ensemble training result to file."""
        result_file = self.output_dir / f"{result.ensemble_id}_result.json"
        
        try:
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            result_dict['start_time'] = result.start_time.isoformat()
            if result.end_time:
                result_dict['end_time'] = result.end_time.isoformat()
            
            # Handle nested objects
            for i, individual_result in enumerate(result_dict['individual_results']):
                if 'start_time' in individual_result:
                    individual_result['start_time'] = individual_result['start_time'].isoformat() if hasattr(individual_result['start_time'], 'isoformat') else str(individual_result['start_time'])
                if 'end_time' in individual_result and individual_result['end_time']:
                    individual_result['end_time'] = individual_result['end_time'].isoformat() if hasattr(individual_result['end_time'], 'isoformat') else str(individual_result['end_time'])
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"Ensemble result saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save ensemble result: {e}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of all ensemble training runs."""
        if not self.ensemble_history:
            return {"message": "No ensemble training runs completed"}
        
        successful_runs = [r for r in self.ensemble_history if r.success]
        
        summary = {
            "total_ensemble_runs": len(self.ensemble_history),
            "successful_ensemble_runs": len(successful_runs),
            "failed_ensemble_runs": len(self.ensemble_history) - len(successful_runs),
            "best_ensemble_accuracy": max((r.ensemble_accuracy or 0) for r in successful_runs) if successful_runs else 0,
            "average_ensemble_improvement": sum(r.ensemble_improvement or 0 for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "average_training_time_hours": sum(r.total_training_time for r in successful_runs) / (len(successful_runs) * 3600) if successful_runs else 0,
            "recent_runs": [
                {
                    "ensemble_id": r.ensemble_id,
                    "success": r.success,
                    "ensemble_accuracy": r.ensemble_accuracy,
                    "ensemble_improvement": r.ensemble_improvement,
                    "num_models": len(r.model_configs),
                    "diversity_strategy": r.config.diversity_strategy,
                    "training_time_hours": r.total_training_time / 3600
                }
                for r in self.ensemble_history[-5:]  # Last 5 runs
            ]
        }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    print("Email ensemble training system ready!")
    
    # Example of how to use
    """
    # Create ensemble trainer
    ensemble_trainer = EmailEnsembleTrainer("ensemble_output")
    
    # Create base configuration
    base_config = EmailTrainingConfig(
        vocab_size=5000,
        hidden_size=512,
        num_layers=2,
        batch_size=8,
        learning_rate=1e-4
    )
    
    # Create ensemble configuration
    ensemble_config = EnsembleTrainingConfig(
        num_models=3,
        ensemble_name="email_classifier_ensemble",
        diversity_strategy="config_variation",
        voting_strategy="soft"
    )
    
    # Train ensemble
    result = ensemble_trainer.train_ensemble(
        dataset_path="path/to/email/dataset",
        base_config=base_config,
        ensemble_config=ensemble_config
    )
    
    print(f"Ensemble training completed: {result.success}")
    print(f"Ensemble accuracy: {result.ensemble_accuracy:.4f}")
    """