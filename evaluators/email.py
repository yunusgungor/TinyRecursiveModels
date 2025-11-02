"""
Enhanced Email Classification Evaluator

Provides comprehensive evaluation metrics, analysis tools, and utilities 
for email classification tasks with advanced reporting capabilities.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, cohen_kappa_score, matthews_corrcoef
)
from collections import defaultdict, Counter
import pandas as pd

from models.recursive_reasoning.trm_email import EmailTRM


class EmailClassificationEvaluator:
    """Enhanced evaluator for email classification tasks with comprehensive analysis"""
    
    def __init__(
        self, 
        category_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        enable_advanced_metrics: bool = True,
        save_predictions: bool = False
    ):
        """
        Initialize enhanced evaluator
        
        Args:
            category_names: List of category names in order
            output_dir: Directory to save evaluation results
            enable_advanced_metrics: Whether to compute advanced metrics (ROC-AUC, etc.)
            save_predictions: Whether to save individual predictions for analysis
        """
        
        self.category_names = category_names or [
            "newsletter", "work", "personal", "spam", "promotional",
            "social", "finance", "travel", "shopping", "other"
        ]
        self.output_dir = Path(output_dir) if output_dir else None
        self.enable_advanced_metrics = enable_advanced_metrics
        self.save_predictions = save_predictions
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.reset_metrics()
        
        # Advanced analysis storage
        self.prediction_details = []
        self.error_analysis = defaultdict(list)
        self.confidence_analysis = defaultdict(list)
    
    def reset_metrics(self):
        """Reset accumulated metrics"""
        self.all_predictions = []
        self.all_labels = []
        self.all_logits = []
        self.all_probabilities = []
        self.batch_metrics = []
        self.prediction_details = []
        self.error_analysis = defaultdict(list)
        self.confidence_analysis = defaultdict(list)
        self.timing_info = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probabilities: Optional[torch.Tensor] = None,
        inference_time: Optional[float] = None,
        email_metadata: Optional[List[Dict]] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: [batch_size] - predicted categories
            labels: [batch_size] - true categories  
            logits: [batch_size, num_categories] - prediction logits (optional)
            probabilities: [batch_size, num_categories] - prediction probabilities (optional)
            inference_time: Time taken for inference (optional)
            email_metadata: Metadata for each email in batch (optional)
        """
        
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Store for global metrics
        self.all_predictions.extend(pred_np.tolist())
        self.all_labels.extend(labels_np.tolist())
        
        if logits is not None:
            logits_np = logits.detach().cpu().numpy()
            self.all_logits.extend(logits_np.tolist())
            
            # Convert logits to probabilities if not provided
            if probabilities is None:
                probabilities = torch.softmax(logits, dim=-1)
        
        if probabilities is not None:
            probs_np = probabilities.detach().cpu().numpy()
            self.all_probabilities.extend(probs_np.tolist())
        
        # Store detailed predictions for analysis
        if self.save_predictions:
            batch_size = len(pred_np)
            for i in range(batch_size):
                detail = {
                    'prediction': int(pred_np[i]),
                    'true_label': int(labels_np[i]),
                    'correct': pred_np[i] == labels_np[i],
                    'confidence': float(probs_np[i, pred_np[i]]) if probabilities is not None else None,
                    'probabilities': probs_np[i].tolist() if probabilities is not None else None,
                    'metadata': email_metadata[i] if email_metadata and i < len(email_metadata) else None
                }
                self.prediction_details.append(detail)
        
        # Error analysis
        for i in range(len(pred_np)):
            if pred_np[i] != labels_np[i]:
                error_info = {
                    'predicted': int(pred_np[i]),
                    'actual': int(labels_np[i]),
                    'confidence': float(probs_np[i, pred_np[i]]) if probabilities is not None else None,
                    'metadata': email_metadata[i] if email_metadata and i < len(email_metadata) else None
                }
                self.error_analysis[f"{labels_np[i]}_to_{pred_np[i]}"].append(error_info)
        
        # Confidence analysis
        if probabilities is not None:
            for i in range(len(pred_np)):
                confidence = float(probs_np[i, pred_np[i]])
                is_correct = pred_np[i] == labels_np[i]
                self.confidence_analysis[int(labels_np[i])].append({
                    'confidence': confidence,
                    'correct': is_correct,
                    'predicted': int(pred_np[i])
                })
        
        # Timing info
        if inference_time is not None:
            self.timing_info.append({
                'batch_size': len(pred_np),
                'inference_time': inference_time,
                'time_per_sample': inference_time / len(pred_np)
            })
        
        # Compute batch metrics
        batch_acc = accuracy_score(labels_np, pred_np)
        self.batch_metrics.append({
            'accuracy': batch_acc,
            'batch_size': len(pred_np)
        })
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics
        
        Returns:
            Dictionary of metrics
        """
        
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro/micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Advanced metrics
        advanced_metrics = {}
        if self.enable_advanced_metrics:
            # Cohen's Kappa
            kappa = cohen_kappa_score(labels, predictions)
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(labels, predictions)
            
            advanced_metrics.update({
                'cohen_kappa': float(kappa),
                'matthews_corrcoef': float(mcc)
            })
            
            # ROC-AUC and Average Precision (if probabilities available)
            if self.all_probabilities:
                probs = np.array(self.all_probabilities)
                
                try:
                    # Multi-class ROC-AUC (one-vs-rest)
                    roc_auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
                    roc_auc_ovo = roc_auc_score(labels, probs, multi_class='ovo', average='macro')
                    
                    advanced_metrics.update({
                        'roc_auc_ovr': float(roc_auc_ovr),
                        'roc_auc_ovo': float(roc_auc_ovo)
                    })
                except ValueError:
                    # Handle case where not all classes are present
                    pass
                
                # Per-class ROC-AUC
                per_class_auc = {}
                for i, category in enumerate(self.category_names):
                    if i < probs.shape[1]:
                        try:
                            binary_labels = (labels == i).astype(int)
                            if len(np.unique(binary_labels)) > 1:  # Need both classes
                                auc = roc_auc_score(binary_labels, probs[:, i])
                                per_class_auc[category] = float(auc)
                        except ValueError:
                            per_class_auc[category] = 0.0
                
                advanced_metrics['per_class_auc'] = per_class_auc
        
        # Per-category metrics
        category_metrics = {}
        for i, category in enumerate(self.category_names):
            if i < len(precision):
                category_metrics[category] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
        
        # Category distribution
        label_dist = Counter(labels)
        pred_dist = Counter(predictions)
        
        # Confidence analysis
        confidence_metrics = {}
        if self.all_probabilities:
            probs = np.array(self.all_probabilities)
            
            # Overall confidence metrics
            max_probs = np.max(probs, axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            
            # Confidence vs accuracy correlation
            correct_predictions = (predictions == labels)
            confidence_correct = max_probs[correct_predictions]
            confidence_incorrect = max_probs[~correct_predictions]
            
            confidence_metrics = {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'mean_entropy': float(np.mean(entropy)),
                'std_entropy': float(np.std(entropy)),
                'confidence_correct_mean': float(np.mean(confidence_correct)) if len(confidence_correct) > 0 else 0.0,
                'confidence_incorrect_mean': float(np.mean(confidence_incorrect)) if len(confidence_incorrect) > 0 else 0.0,
                'confidence_gap': float(np.mean(confidence_correct) - np.mean(confidence_incorrect)) if len(confidence_correct) > 0 and len(confidence_incorrect) > 0 else 0.0
            }
        
        # Timing metrics
        timing_metrics = {}
        if self.timing_info:
            total_samples = sum(info['batch_size'] for info in self.timing_info)
            total_time = sum(info['inference_time'] for info in self.timing_info)
            
            timing_metrics = {
                'total_inference_time': total_time,
                'average_time_per_sample': total_time / total_samples if total_samples > 0 else 0.0,
                'throughput_samples_per_second': total_samples / total_time if total_time > 0 else 0.0
            }
        
        # Compile all metrics
        metrics = {
            # Overall metrics
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            
            # Per-category metrics
            'category_metrics': category_metrics,
            
            # Confusion matrix
            'confusion_matrix': cm.tolist(),
            
            # Distributions
            'label_distribution': dict(label_dist),
            'prediction_distribution': dict(pred_dist),
            
            # Sample counts
            'total_samples': len(predictions),
            'num_categories': len(self.category_names),
            
            # Advanced metrics
            **advanced_metrics,
            
            # Confidence metrics
            **confidence_metrics,
            
            # Timing metrics
            **timing_metrics
        }
        
        return metrics
    
    def print_metrics(self, metrics: Optional[Dict[str, Any]] = None):
        """Print formatted metrics"""
        
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("EMAIL CLASSIFICATION EVALUATION RESULTS")
        print("="*60)
        
        # Overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # Per-category metrics
        print(f"\nPer-Category Metrics:")
        print(f"{'Category':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        
        for category, cat_metrics in metrics['category_metrics'].items():
            print(f"{category:<12} {cat_metrics['precision']:<10.4f} "
                  f"{cat_metrics['recall']:<10.4f} {cat_metrics['f1']:<10.4f} "
                  f"{cat_metrics['support']:<10}")
        
        # Confidence metrics
        if 'mean_confidence' in metrics:
            print(f"\nConfidence Metrics:")
            print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
            print(f"  Mean Entropy: {metrics['mean_entropy']:.4f}")
        
        # Sample distribution
        print(f"\nSample Distribution:")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Categories: {metrics['num_categories']}")
        
        print("="*60)
    
    def save_metrics(self, metrics: Optional[Dict[str, Any]] = None, filename: str = "email_eval_results.json"):
        """Save metrics to file"""
        
        if metrics is None:
            metrics = self.compute_metrics()
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Metrics saved to {filepath}")
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors in detail"""
        
        if not self.error_analysis:
            return {"message": "No errors to analyze"}
        
        error_summary = {}
        
        # Most common error types
        error_counts = {error_type: len(errors) for error_type, errors in self.error_analysis.items()}
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        error_summary['most_common_errors'] = sorted_errors[:10]
        
        # Error patterns by category
        error_patterns = defaultdict(dict)
        for error_type, errors in self.error_analysis.items():
            actual, predicted = error_type.split('_to_')
            actual_name = self.category_names[int(actual)] if int(actual) < len(self.category_names) else f"class_{actual}"
            predicted_name = self.category_names[int(predicted)] if int(predicted) < len(self.category_names) else f"class_{predicted}"
            
            error_patterns[actual_name][predicted_name] = {
                'count': len(errors),
                'avg_confidence': np.mean([e['confidence'] for e in errors if e['confidence'] is not None]) if any(e['confidence'] is not None for e in errors) else None,
                'examples': errors[:3]  # First 3 examples
            }
        
        error_summary['error_patterns'] = dict(error_patterns)
        
        return error_summary
    
    def analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze confidence calibration"""
        
        if not self.all_probabilities:
            return {"message": "No probability data available for calibration analysis"}
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)
        
        # Get confidence scores (max probability)
        confidences = np.max(probs, axis=1)
        correct = (predictions == labels)
        
        # Reliability diagram data
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                calibration_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'proportion': float(prop_in_bin),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'count': int(in_bin.sum())
                })
        
        # Expected Calibration Error (ECE)
        ece = 0
        for data in calibration_data:
            ece += data['proportion'] * abs(data['accuracy'] - data['confidence'])
        
        return {
            'expected_calibration_error': float(ece),
            'calibration_data': calibration_data,
            'overall_confidence': float(np.mean(confidences)),
            'overall_accuracy': float(correct.mean())
        }
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': time.time(),
            'basic_metrics': self.compute_metrics(),
            'error_analysis': self.analyze_errors(),
            'confidence_calibration': self.analyze_confidence_calibration()
        }
        
        # Add category-specific analysis
        if self.confidence_analysis:
            category_analysis = {}
            for category_idx, analyses in self.confidence_analysis.items():
                category_name = self.category_names[category_idx] if category_idx < len(self.category_names) else f"class_{category_idx}"
                
                confidences = [a['confidence'] for a in analyses]
                correct_flags = [a['correct'] for a in analyses]
                
                if confidences:
                    category_analysis[category_name] = {
                        'sample_count': len(analyses),
                        'accuracy': np.mean(correct_flags),
                        'avg_confidence': np.mean(confidences),
                        'confidence_std': np.std(confidences),
                        'confidence_range': [float(np.min(confidences)), float(np.max(confidences))]
                    }
            
            report['category_analysis'] = category_analysis
        
        return report
    
    def plot_confusion_matrix(self, metrics: Optional[Dict[str, Any]] = None, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        if metrics is None:
            metrics = self.compute_metrics()
        
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.category_names[:cm.shape[1]],
            yticklabels=self.category_names[:cm.shape[0]]
        )
        
        plt.title('Email Classification Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confidence_distribution(self, save_path: Optional[str] = None):
        """Plot confidence distribution by correctness"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        if not self.all_probabilities:
            print("No probability data available for confidence plotting")
            return
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probabilities)
        
        confidences = np.max(probs, axis=1)
        correct = (predictions == labels)
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(confidences[correct], bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct], bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Correctness')
        plt.legend()
        
        # Plot 2: Calibration plot
        plt.subplot(1, 2, 2)
        calibration_data = self.analyze_confidence_calibration()
        
        if 'calibration_data' in calibration_data:
            bin_centers = [(d['bin_lower'] + d['bin_upper']) / 2 for d in calibration_data['calibration_data']]
            accuracies = [d['accuracy'] for d in calibration_data['calibration_data']]
            
            plt.plot(bin_centers, accuracies, 'o-', label='Model')
            plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration', color='gray')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title('Reliability Diagram')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence plots saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_predictions_to_csv(self, filepath: str):
        """Export detailed predictions to CSV for further analysis"""
        
        if not self.prediction_details:
            print("No prediction details available. Enable save_predictions=True")
            return
        
        df = pd.DataFrame(self.prediction_details)
        
        # Add category names
        df['predicted_category'] = df['prediction'].apply(
            lambda x: self.category_names[x] if x < len(self.category_names) else f"class_{x}"
        )
        df['true_category'] = df['true_label'].apply(
            lambda x: self.category_names[x] if x < len(self.category_names) else f"class_{x}"
        )
        
        df.to_csv(filepath, index=False)
        print(f"Predictions exported to {filepath}")


def evaluate_email_model(
    model: EmailTRM,
    dataloader,
    device: torch.device,
    category_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate email classification model
    
    Args:
        model: EmailTRM model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        category_names: List of category names
        output_dir: Directory to save results
        
    Returns:
        Evaluation metrics dictionary
    """
    
    model.eval()
    evaluator = EmailClassificationEvaluator(category_names, output_dir)
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (set_name, batch, batch_size) in enumerate(dataloader):
            
            # Move to device
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            puzzle_ids = batch.get('puzzle_identifiers', None)
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(device)
            
            # Forward pass
            outputs = model(inputs, labels=labels, puzzle_identifiers=puzzle_ids)
            
            # Get predictions
            predictions = torch.argmax(outputs['logits'], dim=-1)
            
            # Update evaluator
            evaluator.update(predictions, labels, outputs['logits'])
            
            total_samples += batch_size
            
            if batch_idx % 10 == 0:
                print(f"Evaluated {total_samples} samples...")
    
    # Compute final metrics
    metrics = evaluator.compute_metrics()
    
    # Print and save results
    evaluator.print_metrics(metrics)
    
    if output_dir:
        evaluator.save_metrics(metrics)
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        evaluator.plot_confusion_matrix(metrics, cm_path)
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Test evaluator
    evaluator = EmailClassificationEvaluator()
    
    # Simulate some predictions
    np.random.seed(42)
    n_samples = 100
    n_categories = 10
    
    # Generate random predictions and labels
    predictions = torch.randint(0, n_categories, (n_samples,))
    labels = torch.randint(0, n_categories, (n_samples,))
    logits = torch.randn(n_samples, n_categories)
    
    # Update evaluator
    evaluator.update(predictions, labels, logits)
    
    # Compute and print metrics
    metrics = evaluator.compute_metrics()
    evaluator.print_metrics(metrics)
    
    print("Email evaluator test completed successfully!")