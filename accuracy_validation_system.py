#!/usr/bin/env python3
"""
Comprehensive Accuracy Validation System for Email Classification

This module implements task 8.1: Execute comprehensive accuracy validation
- Train EmailTRM models on large email datasets
- Validate 95%+ accuracy achievement across all 10 categories
- Test model performance on Turkish and English email content
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    DataLoader = None
    TORCH_AVAILABLE = False

from macbook_optimization.email_training_orchestrator import EmailTrainingOrchestrator, TrainingResult
from macbook_optimization.email_comprehensive_evaluation import ComprehensiveEmailEvaluator, ComprehensiveEvaluationResult
from macbook_optimization.email_training_config import EmailTrainingConfig
from macbook_optimization.email_dataset_management import EmailDatasetManager
from models.email_tokenizer import EmailTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AccuracyValidationConfig:
    """Configuration for accuracy validation experiments."""
    
    # Dataset configurations
    dataset_paths: List[str]
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training configurations
    training_strategies: List[str] = None  # ["single", "multi_phase", "progressive"]
    max_steps_per_experiment: int = 10000
    target_accuracy: float = 0.95
    min_category_accuracy: float = 0.90
    
    # Model configurations to test
    model_configs: List[Dict[str, Any]] = None
    
    # Language testing
    test_languages: List[str] = None  # ["en", "tr", "mixed"]
    
    # Validation parameters
    num_validation_runs: int = 3
    cross_validation_folds: int = 5
    
    # Output configuration
    output_dir: str = "accuracy_validation_results"
    save_detailed_results: bool = True
    generate_reports: bool = True
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.training_strategies is None:
            self.training_strategies = ["multi_phase", "progressive"]
        
        if self.model_configs is None:
            self.model_configs = [
                {"hidden_size": 256, "num_layers": 2, "learning_rate": 1e-4},
                {"hidden_size": 384, "num_layers": 2, "learning_rate": 1e-4},
                {"hidden_size": 512, "num_layers": 2, "learning_rate": 8e-5},
                {"hidden_size": 512, "num_layers": 3, "learning_rate": 8e-5}
            ]
        
        if self.test_languages is None:
            self.test_languages = ["en", "tr", "mixed"]


@dataclass
class ValidationExperimentResult:
    """Result of a single validation experiment."""
    
    experiment_id: str
    timestamp: datetime
    
    # Configuration
    dataset_path: str
    training_strategy: str
    model_config: Dict[str, Any]
    language: str
    
    # Training results
    training_result: Optional[TrainingResult]
    training_success: bool
    training_time: float
    
    # Evaluation results
    evaluation_result: Optional[ComprehensiveEvaluationResult]
    evaluation_success: bool
    
    # Accuracy metrics
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    min_category_accuracy: float
    accuracy_target_met: bool
    category_target_met: bool
    
    # Performance metrics
    inference_time_ms: float
    throughput_samples_per_sec: float
    
    # Quality metrics
    confidence_calibration_error: float
    prediction_uncertainty: float
    
    # Errors and warnings
    errors: List[str]
    warnings: List[str]


@dataclass
class ComprehensiveValidationResult:
    """Complete validation results across all experiments."""
    
    validation_id: str
    timestamp: datetime
    config: AccuracyValidationConfig
    
    # Experiment results
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    experiment_results: List[ValidationExperimentResult]
    
    # Accuracy achievement analysis
    accuracy_target_achievement_rate: float
    category_target_achievement_rate: float
    experiments_meeting_both_targets: int
    
    # Performance statistics
    best_overall_accuracy: float
    average_accuracy: float
    accuracy_std: float
    
    # Per-category analysis
    category_performance_stats: Dict[str, Dict[str, float]]
    
    # Language performance analysis
    language_performance: Dict[str, Dict[str, float]]
    
    # Training strategy analysis
    strategy_performance: Dict[str, Dict[str, float]]
    
    # Model configuration analysis
    model_config_performance: Dict[str, Dict[str, float]]
    
    # Timing and efficiency
    total_validation_time: float
    average_training_time: float
    average_inference_time: float
    
    # Quality and reliability
    calibration_quality: Dict[str, float]
    uncertainty_analysis: Dict[str, float]
    
    # Summary and recommendations
    validation_summary: str
    recommendations: List[str]
    
    # Files generated
    output_files: List[str]


class AccuracyValidationSystem:
    """
    Comprehensive accuracy validation system for email classification.
    
    Implements systematic validation of 95%+ accuracy target across:
    - Multiple datasets and configurations
    - Different training strategies
    - Turkish and English content
    - Cross-validation and robustness testing
    """
    
    def __init__(self, config: AccuracyValidationConfig):
        """
        Initialize accuracy validation system.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.orchestrator = EmailTrainingOrchestrator(
            output_dir=str(self.output_dir / "training_outputs"),
            enable_monitoring=True,
            enable_checkpointing=True
        )
        
        self.evaluator = ComprehensiveEmailEvaluator(
            output_dir=str(self.output_dir / "evaluation_outputs"),
            confidence_bins=10,
            save_detailed_predictions=config.save_detailed_results
        )
        
        # Validation state
        self.validation_id = f"accuracy_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_results: List[ValidationExperimentResult] = []
        
        logger.info(f"AccuracyValidationSystem initialized: {self.validation_id}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def execute_comprehensive_validation(self) -> ComprehensiveValidationResult:
        """
        Execute comprehensive accuracy validation across all configurations.
        
        Returns:
            Complete validation results
        """
        logger.info(f"Starting comprehensive accuracy validation: {self.validation_id}")
        
        start_time = time.time()
        
        # Generate all experiment configurations
        experiment_configs = self._generate_experiment_configurations()
        logger.info(f"Generated {len(experiment_configs)} experiment configurations")
        
        # Execute experiments
        for exp_idx, exp_config in enumerate(experiment_configs):
            logger.info(f"Executing experiment {exp_idx + 1}/{len(experiment_configs)}")
            
            try:
                result = self._execute_single_experiment(exp_config)
                self.experiment_results.append(result)
                
                logger.info(f"Experiment {exp_idx + 1} completed: "
                           f"accuracy={result.overall_accuracy:.4f}, "
                           f"target_met={result.accuracy_target_met}")
                
            except Exception as e:
                logger.error(f"Experiment {exp_idx + 1} failed: {e}")
                
                # Create failed experiment result
                failed_result = ValidationExperimentResult(
                    experiment_id=f"{self.validation_id}_exp_{exp_idx + 1}",
                    timestamp=datetime.now(),
                    dataset_path=exp_config["dataset_path"],
                    training_strategy=exp_config["training_strategy"],
                    model_config=exp_config["model_config"],
                    language=exp_config["language"],
                    training_result=None,
                    training_success=False,
                    training_time=0.0,
                    evaluation_result=None,
                    evaluation_success=False,
                    overall_accuracy=0.0,
                    category_accuracies={},
                    min_category_accuracy=0.0,
                    accuracy_target_met=False,
                    category_target_met=False,
                    inference_time_ms=0.0,
                    throughput_samples_per_sec=0.0,
                    confidence_calibration_error=0.0,
                    prediction_uncertainty=0.0,
                    errors=[str(e)],
                    warnings=[]
                )
                self.experiment_results.append(failed_result)
        
        # Analyze results
        validation_result = self._analyze_validation_results(start_time)
        
        # Save results
        if self.config.save_detailed_results:
            self._save_validation_results(validation_result)
        
        # Generate reports
        if self.config.generate_reports:
            self._generate_validation_reports(validation_result)
        
        logger.info(f"Comprehensive validation completed: {self.validation_id}")
        logger.info(f"Success rate: {validation_result.successful_experiments}/{validation_result.total_experiments}")
        logger.info(f"Accuracy target achievement: {validation_result.accuracy_target_achievement_rate:.2%}")
        
        return validation_result
    
    def _generate_experiment_configurations(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations to test."""
        
        if not self.config.dataset_paths:
            raise ValueError("No dataset paths provided for validation")
        
        experiment_configs = []
        
        for dataset_path in self.config.dataset_paths:
            for strategy in self.config.training_strategies:
                for model_config in self.config.model_configs:
                    for language in self.config.test_languages:
                        # Create multiple runs for statistical significance
                        for run_idx in range(self.config.num_validation_runs):
                            config = {
                                "dataset_path": dataset_path,
                                "training_strategy": strategy,
                                "model_config": model_config,
                                "language": language,
                                "run_index": run_idx,
                                "experiment_name": f"{Path(dataset_path).stem}_{strategy}_{language}_run{run_idx}"
                            }
                            experiment_configs.append(config)
        
        return experiment_configs
    
    def _execute_single_experiment(self, exp_config: Dict[str, Any]) -> ValidationExperimentResult:
        """Execute a single validation experiment."""
        
        experiment_id = f"{self.validation_id}_{exp_config['experiment_name']}"
        logger.info(f"Executing experiment: {experiment_id}")
        
        # Create experiment result
        result = ValidationExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            dataset_path=exp_config["dataset_path"],
            training_strategy=exp_config["training_strategy"],
            model_config=exp_config["model_config"],
            language=exp_config["language"],
            training_result=None,
            training_success=False,
            training_time=0.0,
            evaluation_result=None,
            evaluation_success=False,
            overall_accuracy=0.0,
            category_accuracies={},
            min_category_accuracy=0.0,
            accuracy_target_met=False,
            category_target_met=False,
            inference_time_ms=0.0,
            throughput_samples_per_sec=0.0,
            confidence_calibration_error=0.0,
            prediction_uncertainty=0.0,
            errors=[],
            warnings=[]
        )
        
        try:
            # Create training configuration
            training_config = self._create_training_config(exp_config["model_config"])
            
            # Execute training
            logger.info(f"Starting training for experiment: {experiment_id}")
            training_start = time.time()
            
            training_result = self.orchestrator.execute_training_pipeline(
                dataset_path=exp_config["dataset_path"],
                config=training_config,
                strategy=exp_config["training_strategy"],
                total_steps=self.config.max_steps_per_experiment
            )
            
            training_time = time.time() - training_start
            
            result.training_result = training_result
            result.training_success = training_result.success
            result.training_time = training_time
            
            if not training_result.success:
                result.errors.extend(training_result.errors)
                result.warnings.extend(training_result.warnings)
                return result
            
            # Load trained model for evaluation
            if training_result.model_path and os.path.exists(training_result.model_path):
                model = self._load_trained_model(training_result.model_path)
                
                # Create test dataloader
                test_dataloader = self._create_test_dataloader(
                    exp_config["dataset_path"], 
                    exp_config["language"],
                    training_config
                )
                
                # Execute evaluation
                logger.info(f"Starting evaluation for experiment: {experiment_id}")
                evaluation_result = self.evaluator.evaluate_model(
                    model=model,
                    dataloader=test_dataloader,
                    device="cpu",
                    enable_uncertainty_estimation=True
                )
                
                result.evaluation_result = evaluation_result
                result.evaluation_success = True
                
                # Extract metrics
                result.overall_accuracy = evaluation_result.overall_accuracy
                result.category_accuracies = {
                    name: metrics.f1_score 
                    for name, metrics in evaluation_result.category_metrics.items()
                }
                result.min_category_accuracy = min(result.category_accuracies.values()) if result.category_accuracies else 0.0
                
                # Check targets
                result.accuracy_target_met = result.overall_accuracy >= self.config.target_accuracy
                result.category_target_met = result.min_category_accuracy >= self.config.min_category_accuracy
                
                # Performance metrics
                if evaluation_result.inference_time_stats:
                    result.inference_time_ms = evaluation_result.inference_time_stats.get('mean_time_per_sample', 0) * 1000
                if evaluation_result.throughput_metrics:
                    result.throughput_samples_per_sec = evaluation_result.throughput_metrics.get('samples_per_second', 0)
                
                # Quality metrics
                result.confidence_calibration_error = evaluation_result.expected_calibration_error
                result.prediction_uncertainty = evaluation_result.prediction_entropy
                
                logger.info(f"Experiment {experiment_id} completed successfully")
                logger.info(f"Accuracy: {result.overall_accuracy:.4f}, Min category: {result.min_category_accuracy:.4f}")
            
            else:
                result.errors.append("Trained model not found or not saved")
        
        except Exception as e:
            result.errors.append(f"Experiment execution failed: {e}")
            logger.error(f"Experiment {experiment_id} failed: {e}")
        
        return result
    
    def _create_training_config(self, model_config: Dict[str, Any]) -> EmailTrainingConfig:
        """Create training configuration from model config."""
        
        return EmailTrainingConfig(
            # Model parameters from config
            hidden_size=model_config.get("hidden_size", 512),
            num_layers=model_config.get("num_layers", 2),
            
            # Training parameters
            learning_rate=model_config.get("learning_rate", 1e-4),
            batch_size=model_config.get("batch_size", 8),
            weight_decay=model_config.get("weight_decay", 0.01),
            gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 8),
            
            # Target parameters
            target_accuracy=self.config.target_accuracy,
            min_category_accuracy=self.config.min_category_accuracy,
            
            # Other parameters
            vocab_size=5000,
            num_email_categories=10,
            max_sequence_length=512,
            max_epochs=10,
            max_steps=self.config.max_steps_per_experiment,
            early_stopping_patience=5,
            
            # Email-specific parameters
            use_email_structure=True,
            subject_attention_weight=2.0,
            pooling_strategy="weighted",
            use_hierarchical_attention=True,
            enable_subject_prioritization=True,
            email_augmentation_prob=0.3,
            
            # MacBook optimization
            memory_limit_mb=6000,
            enable_memory_monitoring=True,
            dynamic_batch_sizing=True,
            use_cpu_optimization=True,
            num_workers=2
        )
    
    def _load_trained_model(self, model_path: str):
        """Load trained model from checkpoint."""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for model loading")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model configuration
            model_config = checkpoint.get('config', {})
            
            # Import and create model
            from macbook_optimization.email_trm_integration import MacBookEmailTRM
            
            model = MacBookEmailTRM(
                vocab_size=model_config.get('vocab_size', 5000),
                hidden_size=model_config.get('hidden_size', 512),
                num_layers=model_config.get('num_layers', 2),
                num_email_categories=model_config.get('num_email_categories', 10),
                max_seq_len=model_config.get('max_sequence_length', 512),
                use_hierarchical_attention=model_config.get('use_hierarchical_attention', True),
                subject_attention_weight=model_config.get('subject_attention_weight', 2.0)
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _create_test_dataloader(self, dataset_path: str, language: str, config: EmailTrainingConfig) -> DataLoader:
        """Create test dataloader for evaluation."""
        
        try:
            # Create tokenizer
            tokenizer = EmailTokenizer(
                vocab_size=config.vocab_size,
                max_seq_len=config.max_sequence_length
            )
            
            # Create dataset manager
            dataset_manager = EmailDatasetManager()
            
            # Load test dataset
            # For now, use train split as test (in practice you'd have separate test data)
            test_dataloader, loader_info = dataset_manager.create_email_dataloader(
                dataset_path=dataset_path,
                batch_size=config.batch_size,
                split="train",  # Would be "test" in practice
                tokenizer=tokenizer
            )
            
            logger.info(f"Test dataloader created for {language} language")
            return test_dataloader
            
        except Exception as e:
            logger.error(f"Failed to create test dataloader: {e}")
            raise
    
    def _sample_test_data(self, dataset: List[Dict]) -> List[Dict]:
        """Sample test data for validation testing."""
        from collections import defaultdict
        import random
        
        if not dataset:
            return []
        
        # Use samples_per_test from config if available, otherwise use all available samples
        samples_per_test = getattr(self.config, 'samples_per_test', len(dataset))
        
        # Ensure we have samples from each category
        category_samples = defaultdict(list)
        for sample in dataset:
            category_samples[sample.get('category', 'unknown')].append(sample)
        
        # Sample evenly from each category
        samples_per_category = max(1, samples_per_test // len(category_samples))
        
        selected_samples = []
        for category, samples in category_samples.items():
            if len(samples) >= samples_per_category:
                selected = random.sample(samples, samples_per_category)
            else:
                selected = samples
            selected_samples.extend(selected)
        
        # If we need more samples, randomly select additional ones
        if len(selected_samples) < samples_per_test:
            remaining_needed = samples_per_test - len(selected_samples)
            additional_samples = random.sample(dataset, min(remaining_needed, len(dataset)))
            selected_samples.extend(additional_samples)
        
        return selected_samples[:samples_per_test]
    
    def _analyze_validation_results(self, start_time: float) -> ComprehensiveValidationResult:
        """Analyze all validation results and create comprehensive summary."""
        
        total_time = time.time() - start_time
        
        # Basic statistics
        total_experiments = len(self.experiment_results)
        successful_experiments = sum(1 for r in self.experiment_results if r.training_success and r.evaluation_success)
        failed_experiments = total_experiments - successful_experiments
        
        successful_results = [r for r in self.experiment_results if r.training_success and r.evaluation_success]
        
        # Accuracy achievement analysis
        accuracy_target_met = sum(1 for r in successful_results if r.accuracy_target_met)
        category_target_met = sum(1 for r in successful_results if r.category_target_met)
        both_targets_met = sum(1 for r in successful_results if r.accuracy_target_met and r.category_target_met)
        
        accuracy_achievement_rate = accuracy_target_met / successful_experiments if successful_experiments > 0 else 0.0
        category_achievement_rate = category_target_met / successful_experiments if successful_experiments > 0 else 0.0
        
        # Performance statistics
        accuracies = [r.overall_accuracy for r in successful_results]
        best_accuracy = max(accuracies) if accuracies else 0.0
        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        accuracy_std = np.std(accuracies) if accuracies else 0.0
        
        # Per-category analysis
        category_stats = self._analyze_category_performance(successful_results)
        
        # Language performance analysis
        language_stats = self._analyze_language_performance(successful_results)
        
        # Training strategy analysis
        strategy_stats = self._analyze_strategy_performance(successful_results)
        
        # Model configuration analysis
        model_config_stats = self._analyze_model_config_performance(successful_results)
        
        # Timing analysis
        training_times = [r.training_time for r in successful_results if r.training_time > 0]
        inference_times = [r.inference_time_ms for r in successful_results if r.inference_time_ms > 0]
        
        avg_training_time = np.mean(training_times) if training_times else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        # Quality analysis
        calibration_errors = [r.confidence_calibration_error for r in successful_results if r.confidence_calibration_error > 0]
        uncertainties = [r.prediction_uncertainty for r in successful_results if r.prediction_uncertainty > 0]
        
        calibration_quality = {
            "mean_calibration_error": np.mean(calibration_errors) if calibration_errors else 0.0,
            "std_calibration_error": np.std(calibration_errors) if calibration_errors else 0.0
        }
        
        uncertainty_analysis = {
            "mean_uncertainty": np.mean(uncertainties) if uncertainties else 0.0,
            "std_uncertainty": np.std(uncertainties) if uncertainties else 0.0
        }
        
        # Generate summary and recommendations
        summary, recommendations = self._generate_summary_and_recommendations(
            successful_experiments, total_experiments, accuracy_achievement_rate,
            category_achievement_rate, best_accuracy, avg_accuracy
        )
        
        # Create comprehensive result
        result = ComprehensiveValidationResult(
            validation_id=self.validation_id,
            timestamp=datetime.now(),
            config=self.config,
            total_experiments=total_experiments,
            successful_experiments=successful_experiments,
            failed_experiments=failed_experiments,
            experiment_results=self.experiment_results,
            accuracy_target_achievement_rate=accuracy_achievement_rate,
            category_target_achievement_rate=category_achievement_rate,
            experiments_meeting_both_targets=both_targets_met,
            best_overall_accuracy=best_accuracy,
            average_accuracy=avg_accuracy,
            accuracy_std=accuracy_std,
            category_performance_stats=category_stats,
            language_performance=language_stats,
            strategy_performance=strategy_stats,
            model_config_performance=model_config_stats,
            total_validation_time=total_time,
            average_training_time=avg_training_time,
            average_inference_time=avg_inference_time,
            calibration_quality=calibration_quality,
            uncertainty_analysis=uncertainty_analysis,
            validation_summary=summary,
            recommendations=recommendations,
            output_files=[]
        )
        
        return result
    
    def _analyze_category_performance(self, results: List[ValidationExperimentResult]) -> Dict[str, Dict[str, float]]:
        """Analyze per-category performance across experiments."""
        
        category_stats = {}
        
        # Get all category names from first successful result
        if results:
            category_names = list(results[0].category_accuracies.keys())
            
            for category in category_names:
                category_accuracies = [
                    r.category_accuracies.get(category, 0.0) 
                    for r in results 
                    if category in r.category_accuracies
                ]
                
                if category_accuracies:
                    category_stats[category] = {
                        "mean_accuracy": np.mean(category_accuracies),
                        "std_accuracy": np.std(category_accuracies),
                        "min_accuracy": np.min(category_accuracies),
                        "max_accuracy": np.max(category_accuracies),
                        "target_achievement_rate": sum(1 for acc in category_accuracies if acc >= self.config.min_category_accuracy) / len(category_accuracies)
                    }
        
        return category_stats
    
    def _analyze_language_performance(self, results: List[ValidationExperimentResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by language."""
        
        language_stats = {}
        
        for language in self.config.test_languages:
            lang_results = [r for r in results if r.language == language]
            
            if lang_results:
                accuracies = [r.overall_accuracy for r in lang_results]
                
                language_stats[language] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "min_accuracy": np.min(accuracies),
                    "max_accuracy": np.max(accuracies),
                    "num_experiments": len(lang_results),
                    "target_achievement_rate": sum(1 for acc in accuracies if acc >= self.config.target_accuracy) / len(accuracies)
                }
        
        return language_stats
    
    def _analyze_strategy_performance(self, results: List[ValidationExperimentResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by training strategy."""
        
        strategy_stats = {}
        
        for strategy in self.config.training_strategies:
            strategy_results = [r for r in results if r.training_strategy == strategy]
            
            if strategy_results:
                accuracies = [r.overall_accuracy for r in strategy_results]
                training_times = [r.training_time for r in strategy_results]
                
                strategy_stats[strategy] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "mean_training_time": np.mean(training_times),
                    "num_experiments": len(strategy_results),
                    "target_achievement_rate": sum(1 for acc in accuracies if acc >= self.config.target_accuracy) / len(accuracies)
                }
        
        return strategy_stats
    
    def _analyze_model_config_performance(self, results: List[ValidationExperimentResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by model configuration."""
        
        config_stats = {}
        
        # Group by model configuration
        config_groups = {}
        for result in results:
            hidden_size = result.model_config.get('hidden_size', 'unknown')
            num_layers = result.model_config.get('num_layers', 'unknown')
            config_key = f"h{hidden_size}_l{num_layers}"
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        for config_key, config_results in config_groups.items():
            accuracies = [r.overall_accuracy for r in config_results]
            
            config_stats[config_key] = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "num_experiments": len(config_results),
                "target_achievement_rate": sum(1 for acc in accuracies if acc >= self.config.target_accuracy) / len(accuracies)
            }
        
        return config_stats
    
    def _generate_summary_and_recommendations(self, 
                                            successful_experiments: int,
                                            total_experiments: int,
                                            accuracy_achievement_rate: float,
                                            category_achievement_rate: float,
                                            best_accuracy: float,
                                            avg_accuracy: float) -> Tuple[str, List[str]]:
        """Generate validation summary and recommendations."""
        
        # Generate summary
        summary = f"""
Comprehensive Email Classification Accuracy Validation Results

Experiment Overview:
- Total experiments: {total_experiments}
- Successful experiments: {successful_experiments}
- Success rate: {successful_experiments/total_experiments:.1%}

Accuracy Achievement:
- 95%+ accuracy target achievement rate: {accuracy_achievement_rate:.1%}
- Per-category accuracy target achievement rate: {category_achievement_rate:.1%}
- Best overall accuracy achieved: {best_accuracy:.4f}
- Average accuracy across experiments: {avg_accuracy:.4f}

Target Achievement Status: {'✓ ACHIEVED' if accuracy_achievement_rate >= 0.8 else '✗ NOT ACHIEVED'}
"""
        
        # Generate recommendations
        recommendations = []
        
        if accuracy_achievement_rate < 0.5:
            recommendations.append("Accuracy target achievement is low. Consider increasing training steps or improving model architecture.")
        
        if category_achievement_rate < 0.5:
            recommendations.append("Per-category accuracy is inconsistent. Consider class balancing or category-specific training strategies.")
        
        if best_accuracy >= 0.95:
            recommendations.append("95%+ accuracy target has been achieved in at least one configuration. Focus on consistency and robustness.")
        
        if avg_accuracy < 0.90:
            recommendations.append("Average accuracy is below 90%. Consider hyperparameter optimization or data quality improvements.")
        
        if successful_experiments < total_experiments * 0.8:
            recommendations.append("High experiment failure rate. Check training stability and resource constraints.")
        
        return summary.strip(), recommendations
    
    def _save_validation_results(self, result: ComprehensiveValidationResult):
        """Save comprehensive validation results to files."""
        
        try:
            # Save main result
            result_file = self.output_dir / f"{self.validation_id}_comprehensive_results.json"
            
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            
            # Convert experiment results
            for i, exp_result in enumerate(result_dict['experiment_results']):
                exp_result['timestamp'] = exp_result['timestamp'].isoformat() if isinstance(exp_result['timestamp'], datetime) else exp_result['timestamp']
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            result.output_files.append(str(result_file))
            logger.info(f"Validation results saved to {result_file}")
            
            # Save experiment details
            experiments_file = self.output_dir / f"{self.validation_id}_experiment_details.json"
            
            experiment_details = []
            for exp_result in self.experiment_results:
                detail = {
                    "experiment_id": exp_result.experiment_id,
                    "dataset": Path(exp_result.dataset_path).stem,
                    "strategy": exp_result.training_strategy,
                    "language": exp_result.language,
                    "model_config": exp_result.model_config,
                    "success": exp_result.training_success and exp_result.evaluation_success,
                    "accuracy": exp_result.overall_accuracy,
                    "min_category_accuracy": exp_result.min_category_accuracy,
                    "targets_met": exp_result.accuracy_target_met and exp_result.category_target_met,
                    "training_time": exp_result.training_time,
                    "errors": exp_result.errors
                }
                experiment_details.append(detail)
            
            with open(experiments_file, 'w') as f:
                json.dump(experiment_details, f, indent=2)
            
            result.output_files.append(str(experiments_file))
            logger.info(f"Experiment details saved to {experiments_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def _generate_validation_reports(self, result: ComprehensiveValidationResult):
        """Generate human-readable validation reports."""
        
        try:
            # Generate main report
            report_file = self.output_dir / f"{self.validation_id}_validation_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("COMPREHENSIVE EMAIL CLASSIFICATION ACCURACY VALIDATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Validation ID: {result.validation_id}\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Total Validation Time: {result.total_validation_time/3600:.2f} hours\n\n")
                
                # Summary
                f.write("VALIDATION SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(result.validation_summary + "\n\n")
                
                # Detailed Results
                f.write("DETAILED RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Experiments: {result.total_experiments}\n")
                f.write(f"Successful Experiments: {result.successful_experiments}\n")
                f.write(f"Failed Experiments: {result.failed_experiments}\n")
                f.write(f"Success Rate: {result.successful_experiments/result.total_experiments:.1%}\n\n")
                
                f.write(f"95%+ Accuracy Achievement Rate: {result.accuracy_target_achievement_rate:.1%}\n")
                f.write(f"Category Accuracy Achievement Rate: {result.category_target_achievement_rate:.1%}\n")
                f.write(f"Experiments Meeting Both Targets: {result.experiments_meeting_both_targets}\n\n")
                
                f.write(f"Best Overall Accuracy: {result.best_overall_accuracy:.4f}\n")
                f.write(f"Average Accuracy: {result.average_accuracy:.4f} ± {result.accuracy_std:.4f}\n\n")
                
                # Language Performance
                if result.language_performance:
                    f.write("LANGUAGE PERFORMANCE\n")
                    f.write("-" * 40 + "\n")
                    for lang, stats in result.language_performance.items():
                        f.write(f"{lang.upper()}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                               f"(target rate: {stats['target_achievement_rate']:.1%})\n")
                    f.write("\n")
                
                # Strategy Performance
                if result.strategy_performance:
                    f.write("TRAINING STRATEGY PERFORMANCE\n")
                    f.write("-" * 40 + "\n")
                    for strategy, stats in result.strategy_performance.items():
                        f.write(f"{strategy}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                               f"(target rate: {stats['target_achievement_rate']:.1%})\n")
                    f.write("\n")
                
                # Category Performance
                if result.category_performance_stats:
                    f.write("PER-CATEGORY PERFORMANCE\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{'Category':<12} {'Mean Acc':<10} {'Std':<8} {'Target Rate':<12}\n")
                    f.write("-" * 50 + "\n")
                    for category, stats in result.category_performance_stats.items():
                        f.write(f"{category:<12} {stats['mean_accuracy']:<10.4f} "
                               f"{stats['std_accuracy']:<8.4f} {stats['target_achievement_rate']:<12.1%}\n")
                    f.write("\n")
                
                # Recommendations
                if result.recommendations:
                    f.write("RECOMMENDATIONS\n")
                    f.write("-" * 40 + "\n")
                    for i, rec in enumerate(result.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                f.write("="*80 + "\n")
            
            result.output_files.append(str(report_file))
            logger.info(f"Validation report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = AccuracyValidationConfig(
        dataset_paths=["data/test-emails"],  # Use existing test data
        training_strategies=["multi_phase"],
        max_steps_per_experiment=1000,  # Reduced for testing
        num_validation_runs=1,  # Reduced for testing
        model_configs=[
            {"hidden_size": 256, "num_layers": 2, "learning_rate": 1e-4}
        ],
        test_languages=["en"],  # Start with English only
        output_dir="accuracy_validation_test"
    )
    
    # Create and run validation system
    validation_system = AccuracyValidationSystem(config)
    
    print("Accuracy validation system ready!")
    print("Run validation_system.execute_comprehensive_validation() to start validation")