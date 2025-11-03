"""
Comprehensive Email Classification Evaluation System

This module provides advanced evaluation capabilities including detailed metrics
for all 10 email categories, confusion matrix analysis, confidence calibration,
and prediction uncertainty estimation for MacBook-optimized email classification.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

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

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
        classification_report, roc_auc_score, average_precision_score,
        cohen_kappa_score, matthews_corrcoef, brier_score_loss
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from evaluators.email import EmailClassificationEvaluator
    EMAIL_EVALUATOR_AVAILABLE = True
except ImportError:
    EMAIL_EVALUATOR_AVAILABLE = False
    EmailClassificationEvaluator = None

try:
    from .email_trm_integration import MacBookEmailTRM
    MACBOOK_EMAIL_TRM_AVAILABLE = True
except ImportError:
    MACBOOK_EMAIL_TRM_AVAILABLE = False
    MacBookEmailTRM = None

logger = logging.getLogger(__name__)


@dataclass
class CategoryPerformanceMetrics:
    """Detailed performance metrics for a single email category."""
    
    category_id: int
    category_name: str
    
    # Basic metrics
    precision: float
    recall: float
    f1_score: float
    support: int
    
    # Advanced metrics
    specificity: float
    npv: float  # Negative Predictive Value
    balanced_accuracy: float
    
    # Confidence metrics
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    confidence_gap: float
    
    # Calibration metrics
    brier_score: float
    calibration_error: float
    
    # ROC metrics
    auc_score: float
    average_precision: float
    
    # Error analysis
    most_confused_with: List[Tuple[str, int]]  # (category_name, count)
    error_rate: float


@dataclass
class ComprehensiveEvaluationResult:
    """Comprehensive evaluation result with all metrics and analysis."""
    
    evaluation_id: str
    timestamp: datetime
    
    # Dataset info
    total_samples: int
    num_categories: int
    category_names: List[str]
    
    # Overall metrics
    overall_accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    cohen_kappa: float
    matthews_corrcoef: float
    
    # Per-category metrics
    category_metrics: Dict[str, CategoryPerformanceMetrics]
    
    # Confusion matrix analysis
    confusion_matrix: List[List[int]]
    normalized_confusion_matrix: List[List[float]]
    
    # Confidence and calibration analysis
    overall_confidence: float
    expected_calibration_error: float
    reliability_diagram_data: List[Dict[str, float]]
    confidence_histogram_data: Dict[str, List[float]]
    
    # Uncertainty estimation
    prediction_entropy: float
    mutual_information: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    
    # Error analysis
    error_patterns: Dict[str, Any]
    difficult_samples: List[Dict[str, Any]]
    
    # Performance by confidence bins
    performance_by_confidence: List[Dict[str, float]]
    
    # Timing and efficiency
    inference_time_stats: Dict[str, float]
    throughput_metrics: Dict[str, float]


class ComprehensiveEmailEvaluator:
    """
    Comprehensive evaluation system for email classification models.
    
    Provides detailed analysis including per-category performance, confusion matrix
    analysis, confidence calibration, and uncertainty estimation.
    """
    
    def __init__(self,
                 category_names: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 confidence_bins: int = 10,
                 save_detailed_predictions: bool = True):
        """
        Initialize comprehensive evaluator.
        
        Args:
            category_names: List of email category names
            output_dir: Directory to save evaluation results
            confidence_bins: Number of bins for confidence analysis
            save_detailed_predictions: Whether to save individual predictions
        """
        self.category_names = category_names or [
            "Newsletter", "Work", "Personal", "Spam", "Promotional",
            "Social", "Finance", "Travel", "Shopping", "Other"
        ]
        self.num_categories = len(self.category_names)
        self.output_dir = Path(output_dir) if output_dir else None
        self.confidence_bins = confidence_bins
        self.save_detailed_predictions = save_detailed_predictions
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base evaluator
        if EMAIL_EVALUATOR_AVAILABLE and EmailClassificationEvaluator is not None:
            self.base_evaluator = EmailClassificationEvaluator(
                category_names=self.category_names,
                output_dir=str(self.output_dir) if self.output_dir else None,
                enable_advanced_metrics=True,
                save_predictions=save_detailed_predictions
            )
        else:
            self.base_evaluator = None
        
        # Additional data storage for comprehensive analysis
        self.detailed_predictions = []
        self.inference_times = []
        self.model_outputs = []
        
        logger.info(f"ComprehensiveEmailEvaluator initialized for {self.num_categories} categories")
    
    def evaluate_model(self,
                      model: MacBookEmailTRM,
                      dataloader: DataLoader,
                      device: str = "cpu",
                      enable_uncertainty_estimation: bool = True) -> ComprehensiveEvaluationResult:
        """
        Perform comprehensive evaluation of email classification model.
        
        Args:
            model: MacBook-optimized EmailTRM model
            dataloader: Test data loader
            device: Device to run evaluation on
            enable_uncertainty_estimation: Whether to estimate prediction uncertainty
            
        Returns:
            Comprehensive evaluation result
        """
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting comprehensive evaluation: {evaluation_id}")
        
        model.eval()
        model.to(device)
        
        # Reset evaluators
        self.base_evaluator.reset_metrics()
        self.detailed_predictions = []
        self.inference_times = []
        self.model_outputs = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (set_name, batch, batch_size) in enumerate(dataloader):
                start_time = time.time()
                
                # Move to device
                inputs = batch['inputs'].to(device)
                labels = batch['labels'].to(device)
                puzzle_ids = batch.get('puzzle_identifiers')
                if puzzle_ids is not None:
                    puzzle_ids = puzzle_ids.to(device)
                
                # Forward pass with detailed outputs
                outputs = model(inputs, labels=labels, puzzle_identifiers=puzzle_ids, return_all_cycles=True)
                
                # Get predictions and probabilities
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time / batch_size)  # Per sample
                
                # Store detailed outputs for uncertainty estimation
                if enable_uncertainty_estimation:
                    self.model_outputs.append({
                        'logits': logits.cpu(),
                        'probabilities': probabilities.cpu(),
                        'predictions': predictions.cpu(),
                        'labels': labels.cpu(),
                        'all_logits': outputs.get('all_logits', None),
                        'num_cycles': outputs.get('num_cycles', 1)
                    })
                
                # Update base evaluator
                self.base_evaluator.update(
                    predictions=predictions,
                    labels=labels,
                    logits=logits,
                    probabilities=probabilities,
                    inference_time=inference_time
                )
                
                # Store detailed predictions
                if self.save_detailed_predictions:
                    for i in range(batch_size):
                        self.detailed_predictions.append({
                            'sample_id': total_samples + i,
                            'true_label': int(labels[i].cpu()),
                            'predicted_label': int(predictions[i].cpu()),
                            'confidence': float(probabilities[i].max().cpu()),
                            'probabilities': probabilities[i].cpu().numpy().tolist(),
                            'correct': bool(predictions[i] == labels[i]),
                            'inference_time': inference_time / batch_size,
                            'num_reasoning_cycles': outputs.get('num_cycles', 1)
                        })
                
                total_samples += batch_size
                
                if batch_idx % 20 == 0:
                    logger.info(f"Evaluated {total_samples} samples...")
        
        logger.info(f"Evaluation completed. Processing {total_samples} samples...")
        
        # Compute comprehensive metrics
        result = self._compute_comprehensive_metrics(evaluation_id, enable_uncertainty_estimation)
        
        # Save results
        if self.output_dir:
            self._save_evaluation_result(result)
            self._generate_evaluation_report(result)
            if PLOTTING_AVAILABLE:
                self._create_evaluation_plots(result)
        
        logger.info(f"Comprehensive evaluation completed: {evaluation_id}")
        return result
    
    def _compute_comprehensive_metrics(self, 
                                     evaluation_id: str,
                                     enable_uncertainty_estimation: bool) -> ComprehensiveEvaluationResult:
        """Compute all comprehensive evaluation metrics."""
        
        # Get base metrics
        base_metrics = self.base_evaluator.compute_metrics()
        
        # Extract basic data
        predictions = np.array(self.base_evaluator.all_predictions)
        labels = np.array(self.base_evaluator.all_labels)
        probabilities = np.array(self.base_evaluator.all_probabilities)
        
        # Compute per-category metrics
        category_metrics = self._compute_category_metrics(predictions, labels, probabilities)
        
        # Confusion matrix analysis
        cm = confusion_matrix(labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Confidence and calibration analysis
        confidence_analysis = self._analyze_confidence_calibration(predictions, labels, probabilities)
        
        # Uncertainty estimation
        uncertainty_metrics = {}
        if enable_uncertainty_estimation and self.model_outputs:
            uncertainty_metrics = self._estimate_prediction_uncertainty()
        
        # Error analysis
        error_analysis = self._analyze_prediction_errors(predictions, labels, probabilities)
        
        # Performance by confidence bins
        confidence_performance = self._analyze_performance_by_confidence(predictions, labels, probabilities)
        
        # Timing analysis
        timing_stats = self._analyze_timing_performance()
        
        # Create comprehensive result
        result = ComprehensiveEvaluationResult(
            evaluation_id=evaluation_id,
            timestamp=datetime.now(),
            total_samples=len(predictions),
            num_categories=self.num_categories,
            category_names=self.category_names,
            overall_accuracy=base_metrics['accuracy'],
            macro_f1=base_metrics['macro_f1'],
            micro_f1=base_metrics['micro_f1'],
            weighted_f1=base_metrics['weighted_f1'],
            cohen_kappa=base_metrics.get('cohen_kappa', 0.0),
            matthews_corrcoef=base_metrics.get('matthews_corrcoef', 0.0),
            category_metrics=category_metrics,
            confusion_matrix=cm.tolist(),
            normalized_confusion_matrix=cm_normalized.tolist(),
            overall_confidence=confidence_analysis['overall_confidence'],
            expected_calibration_error=confidence_analysis['expected_calibration_error'],
            reliability_diagram_data=confidence_analysis['reliability_data'],
            confidence_histogram_data=confidence_analysis['histogram_data'],
            prediction_entropy=uncertainty_metrics.get('prediction_entropy', 0.0),
            mutual_information=uncertainty_metrics.get('mutual_information', 0.0),
            epistemic_uncertainty=uncertainty_metrics.get('epistemic_uncertainty', 0.0),
            aleatoric_uncertainty=uncertainty_metrics.get('aleatoric_uncertainty', 0.0),
            error_patterns=error_analysis['error_patterns'],
            difficult_samples=error_analysis['difficult_samples'],
            performance_by_confidence=confidence_performance,
            inference_time_stats=timing_stats['inference_time_stats'],
            throughput_metrics=timing_stats['throughput_metrics']
        )
        
        return result
    
    def _compute_category_metrics(self, 
                                predictions: np.ndarray, 
                                labels: np.ndarray, 
                                probabilities: np.ndarray) -> Dict[str, CategoryPerformanceMetrics]:
        """Compute detailed metrics for each category."""
        
        category_metrics = {}
        
        # Compute confusion matrix for detailed analysis
        cm = confusion_matrix(labels, predictions, labels=range(self.num_categories))
        
        for i, category_name in enumerate(self.category_names):
            # Basic metrics
            category_mask = (labels == i)
            if not category_mask.any():
                continue
            
            # True/False Positives/Negatives
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Basic metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            balanced_acc = (recall + specificity) / 2
            
            # Confidence analysis for this category
            category_predictions = predictions[category_mask]
            category_confidences = probabilities[category_mask, i]
            correct_mask = category_predictions == i
            
            avg_conf_correct = np.mean(category_confidences[correct_mask]) if correct_mask.any() else 0.0
            avg_conf_incorrect = np.mean(category_confidences[~correct_mask]) if (~correct_mask).any() else 0.0
            confidence_gap = avg_conf_correct - avg_conf_incorrect
            
            # Calibration metrics
            if SKLEARN_AVAILABLE:
                try:
                    binary_labels = (labels == i).astype(int)
                    brier_score = brier_score_loss(binary_labels, probabilities[:, i])
                    
                    # Simple calibration error
                    prob_true, prob_pred = calibration_curve(binary_labels, probabilities[:, i], n_bins=5)
                    calibration_error = np.mean(np.abs(prob_true - prob_pred))
                except:
                    brier_score = 0.0
                    calibration_error = 0.0
            else:
                brier_score = 0.0
                calibration_error = 0.0
            
            # ROC metrics
            if SKLEARN_AVAILABLE:
                try:
                    binary_labels = (labels == i).astype(int)
                    if len(np.unique(binary_labels)) > 1:
                        auc_score = roc_auc_score(binary_labels, probabilities[:, i])
                        avg_precision = average_precision_score(binary_labels, probabilities[:, i])
                    else:
                        auc_score = 0.0
                        avg_precision = 0.0
                except:
                    auc_score = 0.0
                    avg_precision = 0.0
            else:
                auc_score = 0.0
                avg_precision = 0.0
            
            # Error analysis - most confused categories
            confused_with = []
            for j in range(self.num_categories):
                if i != j and cm[i, j] > 0:
                    confused_with.append((self.category_names[j], int(cm[i, j])))
            confused_with.sort(key=lambda x: x[1], reverse=True)
            
            error_rate = (fn + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
            
            category_metrics[category_name] = CategoryPerformanceMetrics(
                category_id=i,
                category_name=category_name,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=int(category_mask.sum()),
                specificity=specificity,
                npv=npv,
                balanced_accuracy=balanced_acc,
                avg_confidence_correct=avg_conf_correct,
                avg_confidence_incorrect=avg_conf_incorrect,
                confidence_gap=confidence_gap,
                brier_score=brier_score,
                calibration_error=calibration_error,
                auc_score=auc_score,
                average_precision=avg_precision,
                most_confused_with=confused_with[:3],  # Top 3
                error_rate=error_rate
            )
        
        return category_metrics
    
    def _analyze_confidence_calibration(self, 
                                      predictions: np.ndarray, 
                                      labels: np.ndarray, 
                                      probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze confidence calibration and reliability."""
        
        confidences = np.max(probabilities, axis=1)
        correct = (predictions == labels)
        
        # Overall confidence
        overall_confidence = float(np.mean(confidences))
        
        # Reliability diagram data
        n_bins = self.confidence_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = correct[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                bin_count = int(in_bin.sum())
                
                reliability_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(bin_accuracy),
                    'confidence': float(bin_confidence),
                    'count': bin_count,
                    'proportion': float(bin_count / len(predictions))
                })
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        for data in reliability_data:
            ece += data['proportion'] * abs(data['accuracy'] - data['confidence'])
        
        # Confidence histogram data
        histogram_data = {
            'correct_confidences': confidences[correct].tolist(),
            'incorrect_confidences': confidences[~correct].tolist(),
            'all_confidences': confidences.tolist()
        }
        
        return {
            'overall_confidence': overall_confidence,
            'expected_calibration_error': float(ece),
            'reliability_data': reliability_data,
            'histogram_data': histogram_data
        }
    
    def _estimate_prediction_uncertainty(self) -> Dict[str, float]:
        """Estimate prediction uncertainty using model outputs."""
        
        if not self.model_outputs:
            return {}
        
        all_probabilities = []
        all_entropies = []
        
        for output in self.model_outputs:
            probs = output['probabilities']
            
            # Prediction entropy (aleatoric uncertainty)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            all_entropies.extend(entropy.numpy())
            all_probabilities.append(probs)
        
        # Average prediction entropy
        prediction_entropy = float(np.mean(all_entropies))
        
        # If we have multiple reasoning cycles, estimate epistemic uncertainty
        epistemic_uncertainty = 0.0
        mutual_information = 0.0
        
        # Check if we have multiple cycles for uncertainty estimation
        multi_cycle_outputs = [o for o in self.model_outputs if o.get('all_logits') is not None]
        
        if multi_cycle_outputs:
            # Estimate epistemic uncertainty from cycle variations
            cycle_variations = []
            
            for output in multi_cycle_outputs:
                all_logits = output['all_logits']  # [num_cycles, batch_size, num_classes]
                if all_logits is not None and len(all_logits.shape) == 3:
                    # Convert to probabilities
                    all_probs = F.softmax(all_logits, dim=-1)
                    
                    # Compute variation across cycles
                    prob_std = torch.std(all_probs, dim=0)  # [batch_size, num_classes]
                    cycle_variations.extend(torch.mean(prob_std, dim=-1).numpy())
            
            if cycle_variations:
                epistemic_uncertainty = float(np.mean(cycle_variations))
                
                # Mutual information approximation
                mutual_information = prediction_entropy - epistemic_uncertainty
        
        # Aleatoric uncertainty (inherent data uncertainty)
        aleatoric_uncertainty = prediction_entropy - epistemic_uncertainty
        
        return {
            'prediction_entropy': prediction_entropy,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': max(0.0, aleatoric_uncertainty),
            'mutual_information': mutual_information
        }
    
    def _analyze_prediction_errors(self, 
                                 predictions: np.ndarray, 
                                 labels: np.ndarray, 
                                 probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors and identify difficult samples."""
        
        # Error patterns
        error_patterns = defaultdict(list)
        difficult_samples = []
        
        confidences = np.max(probabilities, axis=1)
        correct = (predictions == labels)
        
        for i, (pred, true, conf, probs) in enumerate(zip(predictions, labels, confidences, probabilities)):
            if pred != true:
                error_info = {
                    'sample_id': i,
                    'predicted': int(pred),
                    'true': int(true),
                    'confidence': float(conf),
                    'true_class_prob': float(probs[true]),
                    'predicted_class_prob': float(probs[pred]),
                    'entropy': float(-np.sum(probs * np.log(probs + 1e-8)))
                }
                
                error_key = f"{self.category_names[true]}_to_{self.category_names[pred]}"
                error_patterns[error_key].append(error_info)
                
                # Identify difficult samples (high confidence but wrong)
                if conf > 0.8:  # High confidence threshold
                    difficult_samples.append(error_info)
        
        # Sort difficult samples by confidence (descending)
        difficult_samples.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Summarize error patterns
        error_summary = {}
        for error_type, errors in error_patterns.items():
            error_summary[error_type] = {
                'count': len(errors),
                'avg_confidence': float(np.mean([e['confidence'] for e in errors])),
                'avg_entropy': float(np.mean([e['entropy'] for e in errors])),
                'examples': errors[:3]  # First 3 examples
            }
        
        return {
            'error_patterns': error_summary,
            'difficult_samples': difficult_samples[:20]  # Top 20 difficult samples
        }
    
    def _analyze_performance_by_confidence(self, 
                                         predictions: np.ndarray, 
                                         labels: np.ndarray, 
                                         probabilities: np.ndarray) -> List[Dict[str, float]]:
        """Analyze performance across different confidence bins."""
        
        confidences = np.max(probabilities, axis=1)
        correct = (predictions == labels)
        
        # Create confidence bins
        n_bins = self.confidence_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        performance_by_confidence = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = correct[in_bin].mean()
                bin_count = int(in_bin.sum())
                bin_avg_confidence = confidences[in_bin].mean()
                
                performance_by_confidence.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(bin_accuracy),
                    'count': bin_count,
                    'avg_confidence': float(bin_avg_confidence),
                    'proportion': float(bin_count / len(predictions))
                })
        
        return performance_by_confidence
    
    def _analyze_timing_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze timing and throughput performance."""
        
        if not self.inference_times:
            return {'inference_time_stats': {}, 'throughput_metrics': {}}
        
        times = np.array(self.inference_times)
        
        inference_time_stats = {
            'mean_time_per_sample': float(np.mean(times)),
            'std_time_per_sample': float(np.std(times)),
            'min_time_per_sample': float(np.min(times)),
            'max_time_per_sample': float(np.max(times)),
            'median_time_per_sample': float(np.median(times)),
            'p95_time_per_sample': float(np.percentile(times, 95)),
            'p99_time_per_sample': float(np.percentile(times, 99))
        }
        
        throughput_metrics = {
            'samples_per_second': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
            'samples_per_minute': float(60.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
            'total_samples': len(self.inference_times),
            'total_inference_time': float(np.sum(times))
        }
        
        return {
            'inference_time_stats': inference_time_stats,
            'throughput_metrics': throughput_metrics
        }
    
    def _save_evaluation_result(self, result: ComprehensiveEvaluationResult):
        """Save comprehensive evaluation result to file."""
        
        if not self.output_dir:
            return
        
        # Save main result
        result_file = self.output_dir / f"{result.evaluation_id}_comprehensive_result.json"
        
        try:
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"Comprehensive evaluation result saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {e}")
        
        # Save detailed predictions if available
        if self.detailed_predictions:
            predictions_file = self.output_dir / f"{result.evaluation_id}_detailed_predictions.json"
            
            try:
                with open(predictions_file, 'w') as f:
                    json.dump(self.detailed_predictions, f, indent=2)
                
                logger.info(f"Detailed predictions saved to {predictions_file}")
                
            except Exception as e:
                logger.error(f"Failed to save detailed predictions: {e}")
    
    def _generate_evaluation_report(self, result: ComprehensiveEvaluationResult):
        """Generate human-readable evaluation report."""
        
        if not self.output_dir:
            return
        
        report_file = self.output_dir / f"{result.evaluation_id}_report.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("COMPREHENSIVE EMAIL CLASSIFICATION EVALUATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Evaluation ID: {result.evaluation_id}\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Total Samples: {result.total_samples:,}\n")
                f.write(f"Number of Categories: {result.num_categories}\n\n")
                
                # Overall Performance
                f.write("OVERALL PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Accuracy: {result.overall_accuracy:.4f}\n")
                f.write(f"Macro F1 Score: {result.macro_f1:.4f}\n")
                f.write(f"Micro F1 Score: {result.micro_f1:.4f}\n")
                f.write(f"Weighted F1 Score: {result.weighted_f1:.4f}\n")
                f.write(f"Cohen's Kappa: {result.cohen_kappa:.4f}\n")
                f.write(f"Matthews Correlation: {result.matthews_corrcoef:.4f}\n\n")
                
                # Per-Category Performance
                f.write("PER-CATEGORY PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Category':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
                f.write("-" * 60 + "\n")
                
                for category_name, metrics in result.category_metrics.items():
                    f.write(f"{category_name:<12} {metrics.precision:<10.4f} "
                           f"{metrics.recall:<10.4f} {metrics.f1_score:<10.4f} "
                           f"{metrics.support:<10}\n")
                
                f.write("\n")
                
                # Confidence and Calibration
                f.write("CONFIDENCE AND CALIBRATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Confidence: {result.overall_confidence:.4f}\n")
                f.write(f"Expected Calibration Error: {result.expected_calibration_error:.4f}\n\n")
                
                # Uncertainty Estimation
                if result.prediction_entropy > 0:
                    f.write("UNCERTAINTY ESTIMATION\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Prediction Entropy: {result.prediction_entropy:.4f}\n")
                    f.write(f"Epistemic Uncertainty: {result.epistemic_uncertainty:.4f}\n")
                    f.write(f"Aleatoric Uncertainty: {result.aleatoric_uncertainty:.4f}\n\n")
                
                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n")
                if result.inference_time_stats:
                    f.write(f"Mean Inference Time: {result.inference_time_stats['mean_time_per_sample']*1000:.2f} ms/sample\n")
                    f.write(f"Throughput: {result.throughput_metrics['samples_per_second']:.1f} samples/sec\n")
                
                f.write("\n" + "="*80 + "\n")
            
            logger.info(f"Evaluation report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
    
    def _create_evaluation_plots(self, result: ComprehensiveEvaluationResult):
        """Create comprehensive evaluation plots."""
        
        if not PLOTTING_AVAILABLE or not self.output_dir:
            return
        
        try:
            # Confusion Matrix
            plt.figure(figsize=(12, 10))
            cm = np.array(result.confusion_matrix)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=result.category_names,
                       yticklabels=result.category_names)
            plt.title('Email Classification Confusion Matrix')
            plt.xlabel('Predicted Category')
            plt.ylabel('True Category')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{result.evaluation_id}_confusion_matrix.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Reliability Diagram
            if result.reliability_diagram_data:
                plt.figure(figsize=(10, 8))
                
                bin_centers = [(d['bin_lower'] + d['bin_upper']) / 2 for d in result.reliability_diagram_data]
                accuracies = [d['accuracy'] for d in result.reliability_diagram_data]
                counts = [d['count'] for d in result.reliability_diagram_data]
                
                plt.subplot(2, 2, 1)
                plt.plot(bin_centers, accuracies, 'o-', label='Model', markersize=8)
                plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration', color='gray')
                plt.xlabel('Confidence')
                plt.ylabel('Accuracy')
                plt.title('Reliability Diagram')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Confidence histogram
                plt.subplot(2, 2, 2)
                if result.confidence_histogram_data:
                    correct_conf = result.confidence_histogram_data['correct_confidences']
                    incorrect_conf = result.confidence_histogram_data['incorrect_confidences']
                    
                    plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', density=True)
                    plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
                    plt.xlabel('Confidence')
                    plt.ylabel('Density')
                    plt.title('Confidence Distribution')
                    plt.legend()
                
                # Per-category F1 scores
                plt.subplot(2, 2, 3)
                categories = list(result.category_metrics.keys())
                f1_scores = [result.category_metrics[cat].f1_score for cat in categories]
                
                plt.bar(range(len(categories)), f1_scores, color='skyblue')
                plt.xlabel('Category')
                plt.ylabel('F1 Score')
                plt.title('Per-Category F1 Scores')
                plt.xticks(range(len(categories)), categories, rotation=45)
                
                # Performance by confidence
                plt.subplot(2, 2, 4)
                if result.performance_by_confidence:
                    conf_centers = [(d['bin_lower'] + d['bin_upper']) / 2 for d in result.performance_by_confidence]
                    conf_accuracies = [d['accuracy'] for d in result.performance_by_confidence]
                    
                    plt.plot(conf_centers, conf_accuracies, 'o-', color='purple', markersize=6)
                    plt.xlabel('Confidence Bin')
                    plt.ylabel('Accuracy')
                    plt.title('Accuracy by Confidence Bin')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f"{result.evaluation_id}_analysis_plots.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Evaluation plots saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create evaluation plots: {e}")


# Example usage and testing
if __name__ == "__main__":
    print("Comprehensive email evaluation system ready!")
    
    # Example of how to use
    """
    # Create comprehensive evaluator
    evaluator = ComprehensiveEmailEvaluator(
        output_dir="comprehensive_evaluation_output",
        confidence_bins=10,
        save_detailed_predictions=True
    )
    
    # Evaluate model
    result = evaluator.evaluate_model(
        model=trained_model,
        dataloader=test_dataloader,
        device="cpu",
        enable_uncertainty_estimation=True
    )
    
    print(f"Evaluation completed: {result.evaluation_id}")
    print(f"Overall accuracy: {result.overall_accuracy:.4f}")
    print(f"Expected calibration error: {result.expected_calibration_error:.4f}")
    """