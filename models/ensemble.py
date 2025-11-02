"""
Model Ensemble and A/B Testing for Email Classification

This module provides ensemble methods and A/B testing capabilities
for comparing different email classification models.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from models.recursive_reasoning.trm_email import EmailTRM


class ModelEnsemble:
    """Ensemble of email classification models"""
    
    def __init__(self, models: List[EmailTRM], weights: Optional[List[float]] = None,
                 voting_strategy: str = "soft", device: str = "cuda"):
        """
        Initialize model ensemble
        
        Args:
            models: List of trained EmailTRM models
            weights: Optional weights for each model (for weighted voting)
            voting_strategy: "hard", "soft", or "weighted"
            device: Device to run models on
        """
        self.models = models
        self.num_models = len(models)
        self.voting_strategy = voting_strategy
        self.device = device
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(weights) == self.num_models, "Number of weights must match number of models"
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Move models to device and set to eval mode
        for model in self.models:
            model.to(device)
            model.eval()
        
        # Performance tracking
        self.individual_performances = {}
        self.ensemble_performance = {}
    
    def predict(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None,
                return_individual: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Make ensemble predictions
        
        Args:
            inputs: Input tensor [batch_size, seq_len]
            puzzle_identifiers: Optional puzzle identifiers
            return_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions, optionally with individual predictions
        """
        
        individual_predictions = []
        individual_logits = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(inputs, puzzle_identifiers=puzzle_identifiers)
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                
                individual_predictions.append(predictions)
                individual_logits.append(logits)
        
        # Ensemble prediction based on voting strategy
        if self.voting_strategy == "hard":
            # Hard voting: majority vote
            ensemble_pred = self._hard_voting(individual_predictions)
        
        elif self.voting_strategy == "soft":
            # Soft voting: average probabilities
            ensemble_pred = self._soft_voting(individual_logits)
        
        elif self.voting_strategy == "weighted":
            # Weighted voting: weighted average of probabilities
            ensemble_pred = self._weighted_voting(individual_logits)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        if return_individual:
            return ensemble_pred, individual_predictions
        else:
            return ensemble_pred
    
    def _hard_voting(self, individual_predictions: List[torch.Tensor]) -> torch.Tensor:
        """Hard voting: majority vote"""
        
        batch_size = individual_predictions[0].shape[0]
        num_classes = max(pred.max().item() for pred in individual_predictions) + 1
        
        # Count votes for each class
        vote_counts = torch.zeros(batch_size, num_classes, device=self.device)
        
        for i, predictions in enumerate(individual_predictions):
            weight = self.weights[i]
            for batch_idx in range(batch_size):
                pred_class = predictions[batch_idx].item()
                vote_counts[batch_idx, pred_class] += weight
        
        # Return class with most votes
        ensemble_predictions = torch.argmax(vote_counts, dim=1)
        return ensemble_predictions
    
    def _soft_voting(self, individual_logits: List[torch.Tensor]) -> torch.Tensor:
        """Soft voting: average probabilities"""
        
        # Convert logits to probabilities and average
        avg_probs = torch.zeros_like(individual_logits[0])
        
        for logits in individual_logits:
            probs = F.softmax(logits, dim=-1)
            avg_probs += probs
        
        avg_probs /= len(individual_logits)
        
        # Return class with highest average probability
        ensemble_predictions = torch.argmax(avg_probs, dim=-1)
        return ensemble_predictions
    
    def _weighted_voting(self, individual_logits: List[torch.Tensor]) -> torch.Tensor:
        """Weighted voting: weighted average of probabilities"""
        
        # Weighted average of probabilities
        weighted_probs = torch.zeros_like(individual_logits[0])
        
        for i, logits in enumerate(individual_logits):
            probs = F.softmax(logits, dim=-1)
            weighted_probs += self.weights[i] * probs
        
        # Return class with highest weighted probability
        ensemble_predictions = torch.argmax(weighted_probs, dim=-1)
        return ensemble_predictions
    
    def evaluate_ensemble(self, dataloader, category_names: List[str]) -> Dict[str, Any]:
        """Evaluate ensemble performance"""
        
        all_predictions = []
        all_individual_predictions = [[] for _ in range(self.num_models)]
        all_labels = []
        
        for batch_idx, (set_name, batch, batch_size) in enumerate(dataloader):
            inputs = batch['inputs'].to(self.device)
            labels = batch['labels'].to(self.device)
            puzzle_ids = batch.get('puzzle_identifiers')
            if puzzle_ids is not None:
                puzzle_ids = puzzle_ids.to(self.device)
            
            # Get ensemble and individual predictions
            ensemble_pred, individual_preds = self.predict(
                inputs, puzzle_ids, return_individual=True
            )
            
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for i, pred in enumerate(individual_preds):
                all_individual_predictions[i].extend(pred.cpu().numpy())
        
        # Calculate metrics
        ensemble_accuracy = accuracy_score(all_labels, all_predictions)
        ensemble_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Individual model performances
        individual_accuracies = []
        individual_f1s = []
        
        for i in range(self.num_models):
            acc = accuracy_score(all_labels, all_individual_predictions[i])
            f1 = f1_score(all_labels, all_individual_predictions[i], average='macro')
            individual_accuracies.append(acc)
            individual_f1s.append(f1)
        
        results = {
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_f1": ensemble_f1,
            "individual_accuracies": individual_accuracies,
            "individual_f1s": individual_f1s,
            "improvement_over_best": ensemble_accuracy - max(individual_accuracies),
            "voting_strategy": self.voting_strategy,
            "model_weights": self.weights
        }
        
        return results
    
    def save_ensemble(self, filepath: str):
        """Save ensemble configuration"""
        
        ensemble_config = {
            "num_models": self.num_models,
            "weights": self.weights,
            "voting_strategy": self.voting_strategy,
            "individual_performances": self.individual_performances,
            "ensemble_performance": self.ensemble_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        print(f"Ensemble configuration saved to {filepath}")


class ABTestManager:
    """A/B Testing manager for comparing email classification models"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.results = {}
        
    def create_experiment(self, experiment_name: str, model_a: EmailTRM, model_b: EmailTRM,
                         model_a_name: str = "Model A", model_b_name: str = "Model B") -> str:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_name: Name of the experiment
            model_a: First model to compare
            model_b: Second model to compare
            model_a_name: Display name for model A
            model_b_name: Display name for model B
            
        Returns:
            Experiment ID
        """
        
        experiment_id = f"exp_{len(self.experiments):03d}_{experiment_name}"
        
        self.experiments[experiment_id] = {
            "name": experiment_name,
            "model_a": model_a,
            "model_b": model_b,
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "created_at": torch.cuda.Event(enable_timing=True),
            "status": "created"
        }
        
        print(f"Created A/B test experiment: {experiment_id}")
        return experiment_id
    
    def run_experiment(self, experiment_id: str, test_dataloader, 
                      category_names: List[str], device: str = "cuda") -> Dict[str, Any]:
        """
        Run A/B test experiment
        
        Args:
            experiment_id: ID of the experiment to run
            test_dataloader: Test data loader
            category_names: List of category names
            device: Device to run on
            
        Returns:
            Experiment results
        """
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        experiment["status"] = "running"
        
        print(f"Running A/B test: {experiment_id}")
        print(f"Comparing {experiment['model_a_name']} vs {experiment['model_b_name']}")
        
        # Evaluate both models
        model_a = experiment["model_a"].to(device)
        model_b = experiment["model_b"].to(device)
        
        model_a.eval()
        model_b.eval()
        
        # Collect predictions and metrics
        results_a = self._evaluate_model(model_a, test_dataloader, device)
        results_b = self._evaluate_model(model_b, test_dataloader, device)
        
        # Statistical significance testing
        significance_results = self._statistical_significance_test(
            results_a["predictions"], results_b["predictions"], results_a["labels"]
        )
        
        # Compile results
        experiment_results = {
            "experiment_id": experiment_id,
            "experiment_name": experiment["name"],
            "model_a_name": experiment["model_a_name"],
            "model_b_name": experiment["model_b_name"],
            "model_a_results": results_a["metrics"],
            "model_b_results": results_b["metrics"],
            "winner": self._determine_winner(results_a["metrics"], results_b["metrics"]),
            "significance": significance_results,
            "category_breakdown": self._category_breakdown(
                results_a["predictions"], results_b["predictions"], 
                results_a["labels"], category_names
            )
        }
        
        # Save results
        self.results[experiment_id] = experiment_results
        experiment["status"] = "completed"
        
        # Save to file
        results_file = self.output_dir / f"{experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"A/B test completed. Results saved to {results_file}")
        return experiment_results
    
    def _evaluate_model(self, model: EmailTRM, dataloader, device: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch_idx, (set_name, batch, batch_size) in enumerate(dataloader):
                inputs = batch['inputs'].to(device)
                labels = batch['labels'].to(device)
                puzzle_ids = batch.get('puzzle_identifiers')
                if puzzle_ids is not None:
                    puzzle_ids = puzzle_ids.to(device)
                
                outputs = model(inputs, puzzle_identifiers=puzzle_ids)
                predictions = torch.argmax(outputs["logits"], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(outputs["logits"].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "num_samples": len(all_labels)
        }
        
        return {
            "predictions": all_predictions,
            "labels": all_labels,
            "logits": all_logits,
            "metrics": metrics
        }
    
    def _statistical_significance_test(self, predictions_a: List[int], 
                                     predictions_b: List[int], 
                                     labels: List[int]) -> Dict[str, Any]:
        """Perform statistical significance test"""
        
        # McNemar's test for comparing two classifiers
        correct_a = np.array(predictions_a) == np.array(labels)
        correct_b = np.array(predictions_b) == np.array(labels)
        
        # Contingency table
        both_correct = np.sum(correct_a & correct_b)
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_correct = np.sum(~correct_a & correct_b)
        both_wrong = np.sum(~correct_a & ~correct_b)
        
        # McNemar's statistic
        if a_correct_b_wrong + a_wrong_b_correct > 0:
            mcnemar_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (a_correct_b_wrong + a_wrong_b_correct)
            # Chi-square critical value for p=0.05 is 3.84
            is_significant = mcnemar_stat > 3.84
            p_value = 1 - torch.distributions.chi2.Chi2(1).cdf(torch.tensor(mcnemar_stat)).item()
        else:
            mcnemar_stat = 0
            is_significant = False
            p_value = 1.0
        
        return {
            "mcnemar_statistic": mcnemar_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "contingency_table": {
                "both_correct": int(both_correct),
                "a_correct_b_wrong": int(a_correct_b_wrong),
                "a_wrong_b_correct": int(a_wrong_b_correct),
                "both_wrong": int(both_wrong)
            }
        }
    
    def _determine_winner(self, metrics_a: Dict[str, float], 
                         metrics_b: Dict[str, float]) -> Dict[str, Any]:
        """Determine the winner based on metrics"""
        
        # Primary metric: accuracy
        if metrics_a["accuracy"] > metrics_b["accuracy"]:
            winner = "model_a"
            margin = metrics_a["accuracy"] - metrics_b["accuracy"]
        elif metrics_b["accuracy"] > metrics_a["accuracy"]:
            winner = "model_b"
            margin = metrics_b["accuracy"] - metrics_a["accuracy"]
        else:
            winner = "tie"
            margin = 0.0
        
        # Secondary metric: F1 macro
        f1_winner = "model_a" if metrics_a["f1_macro"] > metrics_b["f1_macro"] else "model_b"
        f1_margin = abs(metrics_a["f1_macro"] - metrics_b["f1_macro"])
        
        return {
            "primary_winner": winner,
            "primary_margin": margin,
            "secondary_winner": f1_winner,
            "secondary_margin": f1_margin,
            "metrics_comparison": {
                "accuracy_diff": metrics_a["accuracy"] - metrics_b["accuracy"],
                "f1_macro_diff": metrics_a["f1_macro"] - metrics_b["f1_macro"],
                "f1_micro_diff": metrics_a["f1_micro"] - metrics_b["f1_micro"]
            }
        }
    
    def _category_breakdown(self, predictions_a: List[int], predictions_b: List[int],
                           labels: List[int], category_names: List[str]) -> Dict[str, Any]:
        """Analyze performance breakdown by category"""
        
        breakdown = {}
        
        for category_idx, category_name in enumerate(category_names):
            # Find samples for this category
            category_mask = np.array(labels) == category_idx
            
            if not category_mask.any():
                continue
            
            category_labels = np.array(labels)[category_mask]
            category_pred_a = np.array(predictions_a)[category_mask]
            category_pred_b = np.array(predictions_b)[category_mask]
            
            # Calculate category-specific metrics
            acc_a = accuracy_score(category_labels, category_pred_a)
            acc_b = accuracy_score(category_labels, category_pred_b)
            
            breakdown[category_name] = {
                "num_samples": int(category_mask.sum()),
                "accuracy_a": acc_a,
                "accuracy_b": acc_b,
                "accuracy_diff": acc_a - acc_b,
                "winner": "model_a" if acc_a > acc_b else "model_b" if acc_b > acc_a else "tie"
            }
        
        return breakdown
    
    def generate_report(self, experiment_id: str) -> str:
        """Generate a human-readable report for an experiment"""
        
        if experiment_id not in self.results:
            return f"No results found for experiment {experiment_id}"
        
        results = self.results[experiment_id]
        
        report = f"""
A/B Test Report: {results['experiment_name']}
{'='*60}

Models Compared:
- {results['model_a_name']}: Accuracy = {results['model_a_results']['accuracy']:.4f}, F1 = {results['model_a_results']['f1_macro']:.4f}
- {results['model_b_name']}: Accuracy = {results['model_b_results']['accuracy']:.4f}, F1 = {results['model_b_results']['f1_macro']:.4f}

Winner: {results['winner']['primary_winner'].replace('_', ' ').title()}
Margin: {results['winner']['primary_margin']:.4f} accuracy points

Statistical Significance:
- McNemar's Test: {'Significant' if results['significance']['is_significant'] else 'Not Significant'}
- p-value: {results['significance']['p_value']:.4f}

Category Breakdown:
"""
        
        for category, stats in results['category_breakdown'].items():
            report += f"- {category}: {stats['winner'].replace('_', ' ').title()} wins "
            report += f"({stats['accuracy_diff']:+.4f} accuracy difference)\n"
        
        return report
    
    def save_all_results(self):
        """Save all experiment results"""
        
        summary_file = self.output_dir / "ab_test_summary.json"
        
        summary = {
            "total_experiments": len(self.experiments),
            "completed_experiments": len(self.results),
            "experiments": {exp_id: exp for exp_id, exp in self.experiments.items()},
            "results": self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"A/B test summary saved to {summary_file}")


# Example usage and testing
if __name__ == "__main__":
    print("Model ensemble and A/B testing module ready!")
    
    # Example of how to use (would need actual trained models)
    """
    # Create ensemble
    models = [model1, model2, model3]  # Trained EmailTRM models
    ensemble = ModelEnsemble(models, voting_strategy="soft")
    
    # Evaluate ensemble
    results = ensemble.evaluate_ensemble(test_dataloader, category_names)
    print(f"Ensemble accuracy: {results['ensemble_accuracy']:.4f}")
    
    # A/B testing
    ab_manager = ABTestManager("ab_test_results")
    exp_id = ab_manager.create_experiment("baseline_vs_enhanced", model1, model2)
    results = ab_manager.run_experiment(exp_id, test_dataloader, category_names)
    
    # Generate report
    report = ab_manager.generate_report(exp_id)
    print(report)
    """