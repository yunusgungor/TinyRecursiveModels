"""
Email Classification Evaluator

Provides evaluation metrics and utilities for email classification tasks.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from collections import defaultdict, Counter

from models.recursive_reasoning.trm_email import EmailTRM


class EmailClassificationEvaluator:
    """Evaluator for email classification tasks"""
    
    def __init__(
        self, 
        category_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            category_names: List of category names in order
            output_dir: Directory to save evaluation results
        """
        
        self.category_names = category_names or [
            "newsletter", "work", "personal", "spam", "promotional",
            "social", "finance", "travel", "shopping", "other"
        ]
        self.output_dir = output_dir
        
        # Metrics storage
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset accumulated metrics"""
        self.all_predictions = []
        self.all_labels = []
        self.all_logits = []
        self.batch_metrics = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with batch results
        
        Args:
            predictions: [batch_size] - predicted categories
            labels: [batch_size] - true categories  
            logits: [batch_size, num_categories] - prediction logits (optional)
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
            'num_categories': len(self.category_names)
        }
        
        # Add confidence metrics if logits available
        if self.all_logits:
            logits = np.array(self.all_logits)
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            
            # Confidence metrics
            max_probs = np.max(probs, axis=1)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            
            metrics.update({
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'mean_entropy': float(np.mean(entropy)),
                'std_entropy': float(np.std(entropy))
            })
        
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
        
        plt.figure(figsize=(10, 8))
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