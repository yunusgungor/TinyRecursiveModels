"""
Email Classification Benchmarking and Comparison Suite

This module provides comprehensive benchmarking capabilities for comparing
different email classification models and approaches.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from models.recursive_reasoning.trm_email import EmailTRM
from evaluators.email import EmailClassificationEvaluator


class EmailClassificationBenchmark:
    """Comprehensive benchmarking suite for email classification models"""
    
    def __init__(self, output_dir: str, category_names: List[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.category_names = category_names
        self.num_categories = len(category_names)
        
        # Benchmark results storage
        self.benchmark_results = {}
        self.comparison_results = {}
        
        # Baseline models
        self.baseline_models = {}
        
    def add_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Add traditional ML baseline models"""
        
        print("Training baseline models...")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.baseline_models['random_forest'] = rf_model
        
        # Multinomial Naive Bayes
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        self.baseline_models['naive_bayes'] = nb_model
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        self.baseline_models['logistic_regression'] = lr_model
        
        # SVM (with smaller dataset due to computational cost)
        if len(X_train) < 10000:  # Only for smaller datasets
            svm_model = SVC(probability=True, random_state=42)
            svm_model.fit(X_train, y_train)
            self.baseline_models['svm'] = svm_model
        
        print(f"Trained {len(self.baseline_models)} baseline models")
    
    def benchmark_model(self, model_name: str, model: Union[EmailTRM, Any], 
                       test_dataloader, device: str = "cuda",
                       is_neural_model: bool = True) -> Dict[str, Any]:
        """
        Benchmark a single model
        
        Args:
            model_name: Name identifier for the model
            model: Model to benchmark (EmailTRM or sklearn model)
            test_dataloader: Test data loader
            device: Device for neural models
            is_neural_model: Whether this is a neural network model
            
        Returns:
            Benchmark results dictionary
        """
        
        print(f"Benchmarking model: {model_name}")
        
        start_time = time.time()
        
        # Initialize evaluator
        evaluator = EmailClassificationEvaluator(
            category_names=self.category_names,
            output_dir=str(self.output_dir / f"{model_name}_eval"),
            enable_advanced_metrics=True,
            save_predictions=True
        )
        
        if is_neural_model:
            # Neural model evaluation
            model.to(device)
            model.eval()
            
            total_samples = 0
            inference_times = []
            
            with torch.no_grad():
                for batch_idx, (set_name, batch, batch_size) in enumerate(test_dataloader):
                    
                    batch_start_time = time.time()
                    
                    inputs = batch['inputs'].to(device)
                    labels = batch['labels'].to(device)
                    puzzle_ids = batch.get('puzzle_identifiers')
                    if puzzle_ids is not None:
                        puzzle_ids = puzzle_ids.to(device)
                    
                    # Forward pass
                    outputs = model(inputs, puzzle_identifiers=puzzle_ids)
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    probabilities = torch.softmax(outputs['logits'], dim=-1)
                    
                    batch_inference_time = time.time() - batch_start_time
                    inference_times.append(batch_inference_time)
                    
                    # Update evaluator
                    evaluator.update(
                        predictions=predictions,
                        labels=labels,
                        logits=outputs['logits'],
                        probabilities=probabilities,
                        inference_time=batch_inference_time
                    )
                    
                    total_samples += batch_size
                    
                    if batch_idx % 10 == 0:
                        print(f"  Processed {total_samples} samples...")
        
        else:
            # Sklearn model evaluation
            all_inputs = []
            all_labels = []
            
            # Collect all data first
            for batch_idx, (set_name, batch, batch_size) in enumerate(test_dataloader):
                inputs = batch['inputs'].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                # Flatten inputs for sklearn models
                inputs_flat = inputs.reshape(inputs.shape[0], -1)
                
                all_inputs.append(inputs_flat)
                all_labels.append(labels)
            
            all_inputs = np.vstack(all_inputs)
            all_labels = np.concatenate(all_labels)
            
            # Predict
            batch_start_time = time.time()
            predictions = model.predict(all_inputs)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(all_inputs)
                probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
            else:
                probabilities_tensor = None
            
            batch_inference_time = time.time() - batch_start_time
            
            # Update evaluator
            evaluator.update(
                predictions=torch.tensor(predictions),
                labels=torch.tensor(all_labels),
                probabilities=probabilities_tensor,
                inference_time=batch_inference_time
            )
            
            total_samples = len(all_labels)
            inference_times = [batch_inference_time]
        
        # Compute metrics
        metrics = evaluator.compute_metrics()
        
        # Add timing information
        total_time = time.time() - start_time
        avg_inference_time = np.mean(inference_times)
        
        benchmark_result = {
            'model_name': model_name,
            'metrics': metrics,
            'timing': {
                'total_evaluation_time': total_time,
                'average_inference_time': avg_inference_time,
                'total_samples': total_samples,
                'samples_per_second': total_samples / total_time,
                'is_neural_model': is_neural_model
            },
            'evaluator': evaluator
        }
        
        # Save individual results
        result_file = self.output_dir / f"{model_name}_benchmark.json"
        with open(result_file, 'w') as f:
            # Create serializable version
            serializable_result = benchmark_result.copy()
            del serializable_result['evaluator']  # Remove non-serializable evaluator
            json.dump(serializable_result, f, indent=2)
        
        self.benchmark_results[model_name] = benchmark_result
        
        print(f"  Completed: Accuracy = {metrics['accuracy']:.4f}, "
              f"F1 = {metrics['macro_f1']:.4f}, "
              f"Time = {total_time:.2f}s")
        
        return benchmark_result
    
    def benchmark_baseline_models(self, test_dataloader) -> Dict[str, Any]:
        """Benchmark all baseline models"""
        
        baseline_results = {}
        
        for model_name, model in self.baseline_models.items():
            result = self.benchmark_model(
                model_name=f"baseline_{model_name}",
                model=model,
                test_dataloader=test_dataloader,
                is_neural_model=False
            )
            baseline_results[model_name] = result
        
        return baseline_results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison report
        
        Args:
            model_results: Dictionary of model benchmark results
            
        Returns:
            Comparison analysis
        """
        
        print("Generating model comparison...")
        
        # Extract key metrics for comparison
        comparison_data = []
        
        for model_name, result in model_results.items():
            metrics = result['metrics']
            timing = result['timing']
            
            comparison_data.append({
                'model_name': model_name,
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'micro_f1': metrics['micro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'samples_per_second': timing['samples_per_second'],
                'total_samples': timing['total_samples'],
                'is_neural': timing['is_neural_model']
            })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        
        # Rankings
        rankings = {
            'accuracy': df.nlargest(len(df), 'accuracy')[['model_name', 'accuracy']].to_dict('records'),
            'macro_f1': df.nlargest(len(df), 'macro_f1')[['model_name', 'macro_f1']].to_dict('records'),
            'speed': df.nlargest(len(df), 'samples_per_second')[['model_name', 'samples_per_second']].to_dict('records')
        }
        
        # Best model overall (weighted score)
        df['overall_score'] = (
            0.4 * df['accuracy'] + 
            0.4 * df['macro_f1'] + 
            0.2 * (df['samples_per_second'] / df['samples_per_second'].max())
        )
        
        best_overall = df.nlargest(1, 'overall_score').iloc[0]
        
        # Performance gaps
        best_accuracy = df['accuracy'].max()
        best_f1 = df['macro_f1'].max()
        
        performance_gaps = []
        for _, row in df.iterrows():
            gaps = {
                'model_name': row['model_name'],
                'accuracy_gap': best_accuracy - row['accuracy'],
                'f1_gap': best_f1 - row['macro_f1']
            }
            performance_gaps.append(gaps)
        
        # Category-wise comparison (if available)
        category_comparison = {}
        for model_name, result in model_results.items():
            if 'category_metrics' in result['metrics']:
                category_comparison[model_name] = result['metrics']['category_metrics']
        
        comparison_result = {
            'summary_statistics': {
                'num_models_compared': len(model_results),
                'best_accuracy': float(best_accuracy),
                'best_f1': float(best_f1),
                'accuracy_range': [float(df['accuracy'].min()), float(df['accuracy'].max())],
                'f1_range': [float(df['macro_f1'].min()), float(df['macro_f1'].max())]
            },
            'rankings': rankings,
            'best_overall': {
                'model_name': best_overall['model_name'],
                'overall_score': float(best_overall['overall_score']),
                'accuracy': float(best_overall['accuracy']),
                'macro_f1': float(best_overall['macro_f1']),
                'samples_per_second': float(best_overall['samples_per_second'])
            },
            'performance_gaps': performance_gaps,
            'category_comparison': category_comparison,
            'detailed_comparison': df.to_dict('records')
        }
        
        self.comparison_results = comparison_result
        
        # Save comparison results
        comparison_file = self.output_dir / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        print(f"Model comparison saved to {comparison_file}")
        
        return comparison_result
    
    def generate_benchmark_report(self) -> str:
        """Generate human-readable benchmark report"""
        
        if not self.comparison_results:
            return "No comparison results available. Run compare_models() first."
        
        results = self.comparison_results
        
        report = f"""
Email Classification Model Benchmark Report
{'='*60}

Summary:
- Models Compared: {results['summary_statistics']['num_models_compared']}
- Best Accuracy: {results['summary_statistics']['best_accuracy']:.4f}
- Best F1 Score: {results['summary_statistics']['best_f1']:.4f}
- Accuracy Range: {results['summary_statistics']['accuracy_range'][0]:.4f} - {results['summary_statistics']['accuracy_range'][1]:.4f}

Best Overall Model:
- Name: {results['best_overall']['model_name']}
- Overall Score: {results['best_overall']['overall_score']:.4f}
- Accuracy: {results['best_overall']['accuracy']:.4f}
- F1 Score: {results['best_overall']['macro_f1']:.4f}
- Speed: {results['best_overall']['samples_per_second']:.1f} samples/sec

Top 3 Models by Accuracy:
"""
        
        for i, model in enumerate(results['rankings']['accuracy'][:3]):
            report += f"{i+1}. {model['model_name']}: {model['accuracy']:.4f}\n"
        
        report += "\nTop 3 Models by F1 Score:\n"
        for i, model in enumerate(results['rankings']['macro_f1'][:3]):
            report += f"{i+1}. {model['model_name']}: {model['macro_f1']:.4f}\n"
        
        report += "\nTop 3 Models by Speed:\n"
        for i, model in enumerate(results['rankings']['speed'][:3]):
            report += f"{i+1}. {model['model_name']}: {model['samples_per_second']:.1f} samples/sec\n"
        
        # Performance gaps
        report += "\nPerformance Gaps (vs Best Model):\n"
        for gap in results['performance_gaps']:
            if gap['accuracy_gap'] > 0 or gap['f1_gap'] > 0:
                report += f"- {gap['model_name']}: "
                report += f"Accuracy: -{gap['accuracy_gap']:.4f}, "
                report += f"F1: -{gap['f1_gap']:.4f}\n"
        
        return report
    
    def plot_comparison_charts(self, save_dir: Optional[str] = None):
        """Generate comparison visualization charts"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        if not self.comparison_results:
            print("No comparison results available")
            return
        
        df = pd.DataFrame(self.comparison_results['detailed_comparison'])
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.output_dir
        
        # Plot 1: Accuracy vs F1 Score
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        colors = ['red' if neural else 'blue' for neural in df['is_neural']]
        plt.scatter(df['accuracy'], df['macro_f1'], c=colors, alpha=0.7, s=100)
        
        for i, row in df.iterrows():
            plt.annotate(row['model_name'], (row['accuracy'], row['macro_f1']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Accuracy')
        plt.ylabel('Macro F1 Score')
        plt.title('Accuracy vs F1 Score')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.scatter([], [], c='red', label='Neural Models')
        plt.scatter([], [], c='blue', label='Traditional ML')
        plt.legend()
        
        # Plot 2: Speed vs Accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(df['samples_per_second'], df['accuracy'], c=colors, alpha=0.7, s=100)
        
        for i, row in df.iterrows():
            plt.annotate(row['model_name'], (row['samples_per_second'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Samples per Second')
        plt.ylabel('Accuracy')
        plt.title('Speed vs Accuracy Trade-off')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 3: Model Performance Bar Chart
        plt.subplot(2, 2, 3)
        model_names = df['model_name'].tolist()
        x_pos = np.arange(len(model_names))
        
        plt.bar(x_pos, df['accuracy'], alpha=0.7, label='Accuracy')
        plt.bar(x_pos, df['macro_f1'], alpha=0.7, label='Macro F1')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Overall Score Ranking
        plt.subplot(2, 2, 4)
        df_sorted = df.sort_values('overall_score', ascending=True)
        
        plt.barh(range(len(df_sorted)), df_sorted['overall_score'])
        plt.yticks(range(len(df_sorted)), df_sorted['model_name'])
        plt.xlabel('Overall Score')
        plt.title('Overall Model Ranking')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = save_path / "model_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Comparison charts saved to {chart_path}")
        
        plt.close()
    
    def export_results_to_excel(self, filepath: str):
        """Export all benchmark results to Excel file"""
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # Summary sheet
            if self.comparison_results:
                summary_df = pd.DataFrame(self.comparison_results['detailed_comparison'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Rankings sheet
                rankings_data = []
                for metric, ranking in self.comparison_results['rankings'].items():
                    for i, model in enumerate(ranking):
                        rankings_data.append({
                            'Metric': metric,
                            'Rank': i + 1,
                            'Model': model['model_name'],
                            'Score': list(model.values())[1]  # Get the score value
                        })
                
                rankings_df = pd.DataFrame(rankings_data)
                rankings_df.to_excel(writer, sheet_name='Rankings', index=False)
            
            # Individual model results
            for model_name, result in self.benchmark_results.items():
                if 'category_metrics' in result['metrics']:
                    category_df = pd.DataFrame(result['metrics']['category_metrics']).T
                    category_df.to_excel(writer, sheet_name=f'{model_name}_categories')
        
        print(f"Benchmark results exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    print("Email classification benchmark module ready!")
    
    # Example of how to use
    """
    # Create benchmark suite
    benchmark = EmailClassificationBenchmark("benchmark_results", category_names)
    
    # Add baseline models (requires training data)
    # benchmark.add_baseline_models(X_train, y_train)
    
    # Benchmark models
    trm_result = benchmark.benchmark_model("trm_email", trm_model, test_dataloader)
    baseline_results = benchmark.benchmark_baseline_models(test_dataloader)
    
    # Compare all models
    all_results = {**{"trm_email": trm_result}, **baseline_results}
    comparison = benchmark.compare_models(all_results)
    
    # Generate report
    report = benchmark.generate_benchmark_report()
    print(report)
    
    # Create visualizations
    benchmark.plot_comparison_charts()
    
    # Export to Excel
    benchmark.export_results_to_excel("benchmark_results.xlsx")
    """