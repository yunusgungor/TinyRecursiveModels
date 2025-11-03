"""
Email Classification Hyperparameter Optimization

This module implements Bayesian optimization and other advanced hyperparameter
search strategies specifically designed for email classification with MacBook
hardware constraints and automated model selection.
"""

import os
import json
import time
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .email_training_config import EmailTrainingConfig
from .hardware_detection import HardwareDetector

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the search space for email classification hyperparameters."""
    
    # Model architecture parameters
    hidden_size: Dict[str, Any] = None
    num_layers: Dict[str, Any] = None
    
    # Training parameters
    learning_rate: Dict[str, Any] = None
    batch_size: Dict[str, Any] = None
    weight_decay: Dict[str, Any] = None
    gradient_accumulation_steps: Dict[str, Any] = None
    
    # Email-specific parameters
    subject_attention_weight: Dict[str, Any] = None
    pooling_strategy: Dict[str, Any] = None
    max_sequence_length: Dict[str, Any] = None
    
    # Optimization parameters
    warmup_steps: Dict[str, Any] = None
    early_stopping_patience: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default parameter spaces if not provided."""
        if self.hidden_size is None:
            self.hidden_size = {"type": "choice", "values": [256, 384, 512]}
        
        if self.num_layers is None:
            self.num_layers = {"type": "choice", "values": [1, 2, 3]}
        
        if self.learning_rate is None:
            self.learning_rate = {"type": "log_uniform", "low": 1e-5, "high": 5e-4}
        
        if self.batch_size is None:
            self.batch_size = {"type": "choice", "values": [2, 4, 8, 16]}
        
        if self.weight_decay is None:
            self.weight_decay = {"type": "log_uniform", "low": 1e-3, "high": 1e-1}
        
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = {"type": "choice", "values": [4, 8, 16, 32]}
        
        if self.subject_attention_weight is None:
            self.subject_attention_weight = {"type": "uniform", "low": 1.0, "high": 3.0}
        
        if self.pooling_strategy is None:
            self.pooling_strategy = {"type": "choice", "values": ["mean", "weighted", "attention"]}
        
        if self.max_sequence_length is None:
            self.max_sequence_length = {"type": "choice", "values": [256, 384, 512]}
        
        if self.warmup_steps is None:
            self.warmup_steps = {"type": "choice", "values": [100, 200, 500]}
        
        if self.early_stopping_patience is None:
            self.early_stopping_patience = {"type": "choice", "values": [3, 5, 7]}


@dataclass
class OptimizationTrial:
    """Represents a single hyperparameter optimization trial."""
    trial_id: int
    parameters: Dict[str, Any]
    
    # Results
    objective_value: Optional[float] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    training_time: Optional[float] = None
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Additional metrics
    category_accuracies: Dict[str, float] = None
    memory_usage: Optional[float] = None
    convergence_step: Optional[int] = None
    
    # Error information
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.category_accuracies is None:
            self.category_accuracies = {}


@dataclass
class OptimizationResult:
    """Results of hyperparameter optimization."""
    optimization_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Configuration
    search_space: HyperparameterSpace
    optimization_strategy: str
    num_trials: int
    
    # Results
    best_trial: Optional[OptimizationTrial]
    all_trials: List[OptimizationTrial]
    
    # Performance analysis
    convergence_history: List[float]
    parameter_importance: Dict[str, float]
    
    # Summary statistics
    success_rate: float
    average_objective: float
    objective_std: float
    total_optimization_time: float
    
    # Hardware utilization
    average_memory_usage: float
    peak_memory_usage: float
    
    # Recommendations
    recommended_config: Optional[EmailTrainingConfig]
    optimization_insights: List[str]


class BayesianOptimizer:
    """
    Bayesian optimization implementation for hyperparameter search.
    
    Uses Gaussian Process regression to model the objective function
    and acquisition functions to select promising hyperparameter combinations.
    """
    
    def __init__(self, 
                 search_space: HyperparameterSpace,
                 acquisition_function: str = "expected_improvement",
                 n_initial_points: int = 5,
                 exploration_factor: float = 0.1):
        """
        Initialize Bayesian optimizer.
        
        Args:
            search_space: Hyperparameter search space
            acquisition_function: Acquisition function type
            n_initial_points: Number of random initial points
            exploration_factor: Exploration vs exploitation balance
        """
        self.search_space = search_space
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.exploration_factor = exploration_factor
        
        # Optimization state
        self.trials: List[OptimizationTrial] = []
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.categorical_mappings: Dict[str, Dict[str, int]] = {}
        
        # Initialize parameter space
        self._initialize_parameter_space()
        
        logger.info(f"BayesianOptimizer initialized with {acquisition_function} acquisition function")
    
    def _initialize_parameter_space(self):
        """Initialize parameter bounds and categorical mappings."""
        space_dict = asdict(self.search_space)
        
        for param_name, param_config in space_dict.items():
            if param_config["type"] == "uniform":
                self.parameter_bounds[param_name] = (param_config["low"], param_config["high"])
            
            elif param_config["type"] == "log_uniform":
                self.parameter_bounds[param_name] = (
                    math.log(param_config["low"]), 
                    math.log(param_config["high"])
                )
            
            elif param_config["type"] == "choice":
                # Map categorical values to integers
                values = param_config["values"]
                self.categorical_mappings[param_name] = {str(v): i for i, v in enumerate(values)}
                self.parameter_bounds[param_name] = (0, len(values) - 1)
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """
        Suggest next set of hyperparameters to try.
        
        Returns:
            Dictionary of suggested hyperparameters
        """
        if len(self.trials) < self.n_initial_points:
            # Random exploration for initial points
            return self._random_sample()
        else:
            # Bayesian optimization
            return self._bayesian_sample()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Generate random hyperparameter sample."""
        parameters = {}
        space_dict = asdict(self.search_space)
        
        for param_name, param_config in space_dict.items():
            if param_config["type"] == "uniform":
                value = random.uniform(param_config["low"], param_config["high"])
                parameters[param_name] = value
            
            elif param_config["type"] == "log_uniform":
                log_value = random.uniform(
                    math.log(param_config["low"]), 
                    math.log(param_config["high"])
                )
                parameters[param_name] = math.exp(log_value)
            
            elif param_config["type"] == "choice":
                parameters[param_name] = random.choice(param_config["values"])
        
        return parameters
    
    def _bayesian_sample(self) -> Dict[str, Any]:
        """
        Generate hyperparameter sample using Bayesian optimization.
        
        This is a simplified implementation. In practice, you'd use libraries
        like scikit-optimize, Optuna, or GPyOpt for full Gaussian Process modeling.
        """
        # For now, use a heuristic approach based on trial history
        successful_trials = [t for t in self.trials if t.status == "completed" and t.objective_value is not None]
        
        if not successful_trials:
            return self._random_sample()
        
        # Find best performing trials
        best_trials = sorted(successful_trials, key=lambda t: t.objective_value, reverse=True)[:3]
        
        # Generate sample based on best trials with some exploration
        if random.random() < self.exploration_factor:
            # Exploration: random sample
            return self._random_sample()
        else:
            # Exploitation: sample around best trials
            base_trial = random.choice(best_trials)
            return self._perturb_parameters(base_trial.parameters)
    
    def _perturb_parameters(self, base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb parameters around a base configuration."""
        parameters = base_parameters.copy()
        space_dict = asdict(self.search_space)
        
        # Randomly perturb some parameters
        for param_name, param_config in space_dict.items():
            if random.random() < 0.3:  # 30% chance to perturb each parameter
                
                if param_config["type"] == "uniform":
                    current_value = parameters[param_name]
                    range_size = param_config["high"] - param_config["low"]
                    perturbation = random.gauss(0, range_size * 0.1)  # 10% std
                    new_value = max(param_config["low"], 
                                  min(param_config["high"], 
                                      current_value + perturbation))
                    parameters[param_name] = new_value
                
                elif param_config["type"] == "log_uniform":
                    current_value = parameters[param_name]
                    log_current = math.log(current_value)
                    log_range = math.log(param_config["high"]) - math.log(param_config["low"])
                    perturbation = random.gauss(0, log_range * 0.1)
                    new_log_value = max(math.log(param_config["low"]), 
                                       min(math.log(param_config["high"]), 
                                           log_current + perturbation))
                    parameters[param_name] = math.exp(new_log_value)
                
                elif param_config["type"] == "choice":
                    # For categorical, sometimes choose randomly
                    if random.random() < 0.5:
                        parameters[param_name] = random.choice(param_config["values"])
        
        return parameters
    
    def update_trial(self, trial: OptimizationTrial):
        """Update trial with results."""
        # Find and update the trial
        for i, existing_trial in enumerate(self.trials):
            if existing_trial.trial_id == trial.trial_id:
                self.trials[i] = trial
                break
        else:
            # Add new trial
            self.trials.append(trial)
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        successful_trials = [t for t in self.trials if t.status == "completed" and t.objective_value is not None]
        
        if not successful_trials:
            return None
        
        best_trial = max(successful_trials, key=lambda t: t.objective_value)
        return best_trial.parameters


class EmailHyperparameterOptimizer:
    """
    Main hyperparameter optimizer for email classification.
    
    Coordinates different optimization strategies and provides automated
    model selection based on validation performance with MacBook constraints.
    """
    
    def __init__(self, 
                 output_dir: str = "hyperopt_results",
                 hardware_detector: Optional[HardwareDetector] = None):
        """
        Initialize email hyperparameter optimizer.
        
        Args:
            output_dir: Directory for optimization results
            hardware_detector: Hardware detector for constraint adaptation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hardware_detector = hardware_detector or HardwareDetector()
        
        # Optimization state
        self.current_optimization: Optional[OptimizationResult] = None
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info(f"EmailHyperparameterOptimizer initialized with output dir: {output_dir}")
    
    def create_hardware_adapted_search_space(self, 
                                           dataset_size: int,
                                           base_space: Optional[HyperparameterSpace] = None) -> HyperparameterSpace:
        """
        Create search space adapted for current hardware constraints.
        
        Args:
            dataset_size: Size of training dataset
            base_space: Base search space (default created if None)
            
        Returns:
            Hardware-adapted search space
        """
        # Create a temporary config adapter to get hardware specs
        from .training_config_adapter import TrainingConfigAdapter
        config_adapter = TrainingConfigAdapter(self.hardware_detector)
        hardware_specs = config_adapter.get_hardware_specs()
        memory_gb = hardware_specs.memory.available_memory / (1024**3)
        
        if base_space is None:
            base_space = HyperparameterSpace()
        
        # Adapt search space based on hardware constraints
        adapted_space = HyperparameterSpace()
        
        # Adapt model size based on memory
        if memory_gb < 8:
            adapted_space.hidden_size = {"type": "choice", "values": [128, 256]}
            adapted_space.num_layers = {"type": "choice", "values": [1, 2]}
            adapted_space.batch_size = {"type": "choice", "values": [2, 4]}
            adapted_space.max_sequence_length = {"type": "choice", "values": [256, 384]}
        elif memory_gb < 16:
            adapted_space.hidden_size = {"type": "choice", "values": [256, 384, 512]}
            adapted_space.num_layers = {"type": "choice", "values": [1, 2, 3]}
            adapted_space.batch_size = {"type": "choice", "values": [4, 8, 16]}
            adapted_space.max_sequence_length = {"type": "choice", "values": [384, 512]}
        else:
            adapted_space.hidden_size = {"type": "choice", "values": [384, 512, 768]}
            adapted_space.num_layers = {"type": "choice", "values": [2, 3, 4]}
            adapted_space.batch_size = {"type": "choice", "values": [8, 16, 32]}
            adapted_space.max_sequence_length = {"type": "choice", "values": [512]}
        
        # Adapt gradient accumulation based on batch size constraints
        max_batch = max(adapted_space.batch_size["values"])
        target_effective_batch = min(64, dataset_size // 100)
        max_accumulation = max(1, target_effective_batch // max_batch)
        
        adapted_space.gradient_accumulation_steps = {
            "type": "choice", 
            "values": [i for i in [4, 8, 16, 32] if i <= max_accumulation]
        }
        
        # Keep other parameters from base space or defaults
        adapted_space.learning_rate = base_space.learning_rate
        adapted_space.weight_decay = base_space.weight_decay
        adapted_space.subject_attention_weight = base_space.subject_attention_weight
        adapted_space.pooling_strategy = base_space.pooling_strategy
        adapted_space.warmup_steps = base_space.warmup_steps
        adapted_space.early_stopping_patience = base_space.early_stopping_patience
        
        logger.info(f"Created hardware-adapted search space for {memory_gb:.1f}GB memory")
        logger.info(f"Hidden size range: {adapted_space.hidden_size['values']}")
        logger.info(f"Batch size range: {adapted_space.batch_size['values']}")
        
        return adapted_space
    
    def optimize_hyperparameters(self,
                                training_function: Callable[[EmailTrainingConfig], Dict[str, Any]],
                                search_space: Optional[HyperparameterSpace] = None,
                                num_trials: int = 20,
                                optimization_strategy: str = "bayesian",
                                objective_metric: str = "accuracy",
                                timeout_minutes: Optional[float] = None) -> OptimizationResult:
        """
        Execute hyperparameter optimization.
        
        Args:
            training_function: Function that trains model and returns metrics
            search_space: Hyperparameter search space
            num_trials: Number of optimization trials
            optimization_strategy: Optimization strategy ("random", "bayesian", "grid")
            objective_metric: Metric to optimize
            timeout_minutes: Maximum optimization time
            
        Returns:
            Optimization results
        """
        optimization_id = f"email_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting hyperparameter optimization: {optimization_id}")
        logger.info(f"Strategy: {optimization_strategy}, Trials: {num_trials}")
        
        # Initialize optimization result
        result = OptimizationResult(
            optimization_id=optimization_id,
            start_time=datetime.now(),
            end_time=None,
            search_space=search_space or HyperparameterSpace(),
            optimization_strategy=optimization_strategy,
            num_trials=num_trials,
            best_trial=None,
            all_trials=[],
            convergence_history=[],
            parameter_importance={},
            success_rate=0.0,
            average_objective=0.0,
            objective_std=0.0,
            total_optimization_time=0.0,
            average_memory_usage=0.0,
            peak_memory_usage=0.0,
            recommended_config=None,
            optimization_insights=[]
        )
        
        self.current_optimization = result
        start_time = time.time()
        
        try:
            # Initialize optimizer
            if optimization_strategy == "bayesian":
                optimizer = BayesianOptimizer(result.search_space)
            else:
                optimizer = None  # Use simple strategies
            
            # Execute trials
            for trial_idx in range(num_trials):
                # Check timeout
                if timeout_minutes and (time.time() - start_time) / 60 > timeout_minutes:
                    logger.info(f"Optimization timeout reached after {trial_idx} trials")
                    break
                
                logger.info(f"Starting trial {trial_idx + 1}/{num_trials}")
                
                # Generate parameters
                if optimizer:
                    parameters = optimizer.suggest_parameters()
                else:
                    parameters = self._generate_parameters_simple(result.search_space, optimization_strategy)
                
                # Create trial
                trial = OptimizationTrial(
                    trial_id=trial_idx + 1,
                    parameters=parameters,
                    status="running",
                    start_time=datetime.now()
                )
                
                try:
                    # Create configuration
                    config = self._parameters_to_config(parameters)
                    
                    # Execute training
                    trial_start = time.time()
                    training_result = training_function(config)
                    trial_time = time.time() - trial_start
                    
                    # Extract results
                    trial.objective_value = training_result.get(objective_metric, 0.0)
                    trial.accuracy = training_result.get("accuracy", 0.0)
                    trial.loss = training_result.get("loss", float('inf'))
                    trial.training_time = trial_time
                    trial.memory_usage = training_result.get("memory_usage", 0.0)
                    trial.convergence_step = training_result.get("convergence_step", 0)
                    trial.category_accuracies = training_result.get("category_accuracies", {})
                    trial.status = "completed"
                    trial.end_time = datetime.now()
                    
                    logger.info(f"Trial {trial_idx + 1} completed: {objective_metric} = {trial.objective_value:.4f}")
                    
                except Exception as e:
                    trial.status = "failed"
                    trial.error_message = str(e)
                    trial.end_time = datetime.now()
                    logger.error(f"Trial {trial_idx + 1} failed: {e}")
                
                # Update results
                result.all_trials.append(trial)
                
                if trial.status == "completed" and trial.objective_value is not None:
                    result.convergence_history.append(trial.objective_value)
                    
                    # Update best trial
                    if result.best_trial is None or trial.objective_value > result.best_trial.objective_value:
                        result.best_trial = trial
                        logger.info(f"New best trial: {objective_metric} = {trial.objective_value:.4f}")
                
                # Update optimizer
                if optimizer:
                    optimizer.update_trial(trial)
                
                # Save intermediate results
                self._save_optimization_result(result)
            
            # Finalize results
            result.end_time = datetime.now()
            result.total_optimization_time = time.time() - start_time
            
            # Calculate summary statistics
            successful_trials = [t for t in result.all_trials if t.status == "completed"]
            result.success_rate = len(successful_trials) / len(result.all_trials) if result.all_trials else 0
            
            if successful_trials:
                objectives = [t.objective_value for t in successful_trials if t.objective_value is not None]
                result.average_objective = sum(objectives) / len(objectives) if objectives else 0
                result.objective_std = np.std(objectives) if len(objectives) > 1 else 0
                
                # Memory usage statistics
                memory_usages = [t.memory_usage for t in successful_trials if t.memory_usage is not None]
                result.average_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
                result.peak_memory_usage = max(memory_usages) if memory_usages else 0
            
            # Analyze parameter importance
            result.parameter_importance = self._analyze_parameter_importance(successful_trials)
            
            # Generate recommendations
            result.recommended_config = self._generate_recommended_config(result)
            result.optimization_insights = self._generate_optimization_insights(result)
            
            logger.info(f"Hyperparameter optimization completed")
            logger.info(f"Best {objective_metric}: {result.best_trial.objective_value:.4f}")
            logger.info(f"Success rate: {result.success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            result.optimization_insights.append(f"Optimization failed: {e}")
        
        finally:
            # Save final results
            self._save_optimization_result(result)
            self.optimization_history.append(result)
        
        return result
    
    def _generate_parameters_simple(self, 
                                  search_space: HyperparameterSpace, 
                                  strategy: str) -> Dict[str, Any]:
        """Generate parameters using simple strategies."""
        if strategy == "random":
            return self._random_sample_from_space(search_space)
        elif strategy == "grid":
            # Simplified grid search - just return random for now
            return self._random_sample_from_space(search_space)
        else:
            return self._random_sample_from_space(search_space)
    
    def _random_sample_from_space(self, search_space: HyperparameterSpace) -> Dict[str, Any]:
        """Generate random sample from search space."""
        parameters = {}
        space_dict = asdict(search_space)
        
        for param_name, param_config in space_dict.items():
            if param_config["type"] == "uniform":
                value = random.uniform(param_config["low"], param_config["high"])
                parameters[param_name] = value
            
            elif param_config["type"] == "log_uniform":
                log_value = random.uniform(
                    math.log(param_config["low"]), 
                    math.log(param_config["high"])
                )
                parameters[param_name] = math.exp(log_value)
            
            elif param_config["type"] == "choice":
                parameters[param_name] = random.choice(param_config["values"])
        
        return parameters
    
    def _parameters_to_config(self, parameters: Dict[str, Any]) -> EmailTrainingConfig:
        """Convert parameters dictionary to EmailTrainingConfig."""
        return EmailTrainingConfig(
            # Model parameters
            hidden_size=int(parameters.get("hidden_size", 512)),
            num_layers=int(parameters.get("num_layers", 2)),
            
            # Training parameters
            learning_rate=float(parameters.get("learning_rate", 1e-4)),
            batch_size=int(parameters.get("batch_size", 8)),
            weight_decay=float(parameters.get("weight_decay", 0.01)),
            gradient_accumulation_steps=int(parameters.get("gradient_accumulation_steps", 8)),
            
            # Email-specific parameters
            subject_attention_weight=float(parameters.get("subject_attention_weight", 2.0)),
            pooling_strategy=str(parameters.get("pooling_strategy", "weighted")),
            max_sequence_length=int(parameters.get("max_sequence_length", 512)),
            
            # Other parameters with defaults
            vocab_size=5000,
            num_email_categories=10,
            max_epochs=1,
            max_steps=2000,  # Short trials
            target_accuracy=0.95,
            min_category_accuracy=0.90,
            early_stopping_patience=int(parameters.get("early_stopping_patience", 5))
        )
    
    def _analyze_parameter_importance(self, trials: List[OptimizationTrial]) -> Dict[str, float]:
        """Analyze parameter importance based on trial results."""
        if len(trials) < 3:
            return {}
        
        importance = {}
        param_names = list(trials[0].parameters.keys())
        
        # Simple correlation analysis
        objectives = [t.objective_value for t in trials if t.objective_value is not None]
        
        for param_name in param_names:
            try:
                param_values = [t.parameters[param_name] for t in trials]
                
                # Skip non-numeric parameters for now
                if not all(isinstance(v, (int, float)) for v in param_values):
                    continue
                
                # Calculate correlation with objective
                if len(objectives) == len(param_values) and len(objectives) > 1:
                    correlation = np.corrcoef(param_values, objectives)[0, 1]
                    importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                
            except (ValueError, TypeError):
                importance[param_name] = 0.0
        
        return importance
    
    def _generate_recommended_config(self, result: OptimizationResult) -> Optional[EmailTrainingConfig]:
        """Generate recommended configuration based on optimization results."""
        if result.best_trial is None:
            return None
        
        return self._parameters_to_config(result.best_trial.parameters)
    
    def _generate_optimization_insights(self, result: OptimizationResult) -> List[str]:
        """Generate optimization insights and recommendations."""
        insights = []
        
        if result.success_rate < 0.5:
            insights.append(f"Low success rate ({result.success_rate:.1%}) - consider adjusting search space or training parameters")
        
        if result.best_trial and result.best_trial.objective_value < 0.8:
            insights.append("Best performance is below 80% - consider expanding search space or increasing training time")
        
        # Parameter importance insights
        if result.parameter_importance:
            most_important = max(result.parameter_importance.items(), key=lambda x: x[1])
            if most_important[1] > 0.3:
                insights.append(f"Parameter '{most_important[0]}' shows high importance (correlation: {most_important[1]:.3f})")
        
        # Memory usage insights
        if result.peak_memory_usage > 0:
            if result.peak_memory_usage > 6000:  # 6GB
                insights.append("High memory usage detected - consider reducing model size or batch size")
            elif result.peak_memory_usage < 2000:  # 2GB
                insights.append("Low memory usage - could potentially increase model size for better performance")
        
        # Convergence insights
        if len(result.convergence_history) > 5:
            recent_improvement = result.convergence_history[-1] - result.convergence_history[-5]
            if recent_improvement > 0.05:
                insights.append("Strong convergence trend - optimization is finding better configurations")
            elif recent_improvement < 0.01:
                insights.append("Slow convergence - consider different search strategy or expanded search space")
        
        return insights
    
    def _save_optimization_result(self, result: OptimizationResult):
        """Save optimization result to file."""
        result_file = self.output_dir / f"{result.optimization_id}_result.json"
        
        try:
            # Convert to JSON-serializable format
            result_dict = asdict(result)
            result_dict['start_time'] = result.start_time.isoformat()
            if result.end_time:
                result_dict['end_time'] = result.end_time.isoformat()
            
            # Convert trial timestamps
            for trial_dict in result_dict['all_trials']:
                if trial_dict['start_time']:
                    trial_dict['start_time'] = trial_dict['start_time'].isoformat()
                if trial_dict['end_time']:
                    trial_dict['end_time'] = trial_dict['end_time'].isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.debug(f"Optimization result saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization result: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs."""
        if not self.optimization_history:
            return {"message": "No optimization runs completed"}
        
        successful_optimizations = [opt for opt in self.optimization_history if opt.best_trial is not None]
        
        summary = {
            "total_optimizations": len(self.optimization_history),
            "successful_optimizations": len(successful_optimizations),
            "best_overall_performance": max(
                (opt.best_trial.objective_value for opt in successful_optimizations), 
                default=0.0
            ),
            "average_success_rate": sum(opt.success_rate for opt in successful_optimizations) / len(successful_optimizations) if successful_optimizations else 0,
            "total_trials": sum(len(opt.all_trials) for opt in self.optimization_history),
            "recent_optimizations": [
                {
                    "optimization_id": opt.optimization_id,
                    "strategy": opt.optimization_strategy,
                    "num_trials": len(opt.all_trials),
                    "best_performance": opt.best_trial.objective_value if opt.best_trial else None,
                    "success_rate": opt.success_rate,
                    "optimization_time_minutes": opt.total_optimization_time / 60
                }
                for opt in self.optimization_history[-5:]  # Last 5 optimizations
            ]
        }
        
        return summary