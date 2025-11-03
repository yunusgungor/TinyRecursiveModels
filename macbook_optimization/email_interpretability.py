"""
Email Classification Model Interpretability System

This module provides interpretability features for email classification models,
including attention visualization, feature importance analysis, and reasoning
cycle analysis for TRM decision processes.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from collections import defaultdict

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .email_trm_integration import MacBookEmailTRM
from models.email_tokenizer import EmailTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AttentionVisualization:
    """Attention visualization data for a single email."""
    
    sample_id: str
    email_text: str
    tokens: List[str]
    predicted_category: str
    true_category: str
    confidence: float
    
    # Attention weights
    attention_weights: List[float]  # Per token
    structure_attention: Dict[str, float]  # Per email structure (subject, body, etc.)
    
    # Hierarchical attention (if available)
    layer_attentions: Optional[List[List[float]]]  # [layer][token]
    
    # Reasoning cycle attention progression
    cycle_attentions: Optional[List[List[float]]]  # [cycle][token]


@dataclass
class FeatureImportance:
    """Feature importance analysis for email classification."""
    
    # Token-level importance
    token_importance: Dict[str, float]  # token -> importance score
    
    # N-gram importance
    ngram_importance: Dict[str, float]  # ngram -> importance score
    
    # Email structure importance
    structure_importance: Dict[str, float]  # structure_type -> importance
    
    # Category-specific importance
    category_token_importance: Dict[str, Dict[str, float]]  # category -> token -> importance
    
    # Global feature statistics
    most_important_tokens: List[Tuple[str, float]]
    least_important_tokens: List[Tuple[str, float]]
    
    # Discriminative features between categories
    discriminative_features: Dict[str, List[Tuple[str, float]]]  # category_pair -> features


@dataclass
class ReasoningCycleAnalysis:
    """Analysis of TRM reasoning cycles for interpretability."""
    
    sample_id: str
    num_cycles: int
    
    # Prediction progression
    cycle_predictions: List[int]  # Prediction at each cycle
    cycle_confidences: List[float]  # Confidence at each cycle
    cycle_entropies: List[float]  # Prediction entropy at each cycle
    
    # Attention evolution
    attention_evolution: List[List[float]]  # [cycle][token] attention weights
    
    # Feature focus changes
    feature_focus_changes: List[Dict[str, float]]  # [cycle] -> feature -> focus_change
    
    # Reasoning stability
    prediction_stability: float  # How stable predictions are across cycles
    attention_stability: float  # How stable attention is across cycles
    
    # Decision confidence progression
    confidence_progression: List[float]
    final_decision_cycle: int  # Cycle where final decision was made
    
    # Reasoning quality metrics
    reasoning_efficiency: float  # How quickly model converges
    reasoning_consistency: float  # How consistent reasoning is


class EmailInterpretabilityAnalyzer:
    """
    Comprehensive interpretability analyzer for email classification models.
    
    Provides attention visualization, feature importance analysis, and reasoning
    cycle analysis for understanding model decisions.
    """
    
    def __init__(self,
                 tokenizer: EmailTokenizer,
                 category_names: Optional[List[str]] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize interpretability analyzer.
        
        Args:
            tokenizer: Email tokenizer for text processing
            category_names: List of email category names
            output_dir: Directory to save analysis results
        """
        self.tokenizer = tokenizer
        self.category_names = category_names or [
            "Newsletter", "Work", "Personal", "Spam", "Promotional",
            "Social", "Finance", "Travel", "Shopping", "Other"
        ]
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis storage
        self.attention_visualizations = []
        self.feature_importance_cache = {}
        self.reasoning_analyses = []
        
        logger.info("EmailInterpretabilityAnalyzer initialized")
    
    def analyze_sample_attention(self,
                               model: MacBookEmailTRM,
                               inputs: torch.Tensor,
                               labels: Optional[torch.Tensor] = None,
                               email_text: Optional[str] = None,
                               sample_id: Optional[str] = None) -> AttentionVisualization:
        """
        Analyze attention patterns for a single email sample.
        
        Args:
            model: EmailTRM model
            inputs: Input tensor [1, seq_len] (single sample)
            labels: True labels [1] (optional)
            email_text: Original email text (optional)
            sample_id: Sample identifier (optional)
            
        Returns:
            Attention visualization data
        """
        model.eval()
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
        
        sample_id = sample_id or f"sample_{torch.randint(0, 10000, (1,)).item()}"
        
        with torch.no_grad():
            # Forward pass with attention extraction
            outputs = model(inputs, return_all_cycles=True)
            
            # Get predictions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)
            
            predicted_category = self.category_names[predictions[0].item()]
            true_category = self.category_names[labels[0].item()] if labels is not None else "Unknown"
            confidence = probabilities[0].max().item()
            
            # Extract tokens
            tokens = self._extract_tokens(inputs[0])
            
            # Extract attention weights (simplified - would need model modifications for full attention)
            attention_weights = self._extract_attention_weights(model, inputs, outputs)
            
            # Structure attention analysis
            structure_attention = self._analyze_structure_attention(inputs[0], attention_weights)
            
            # Cycle attention progression (if available)
            cycle_attentions = None
            if 'all_logits' in outputs and outputs['all_logits'] is not None:
                cycle_attentions = self._analyze_cycle_attention_progression(
                    model, inputs, outputs['all_logits']
                )
            
            visualization = AttentionVisualization(
                sample_id=sample_id,
                email_text=email_text or self._reconstruct_text(tokens),
                tokens=tokens,
                predicted_category=predicted_category,
                true_category=true_category,
                confidence=confidence,
                attention_weights=attention_weights,
                structure_attention=structure_attention,
                layer_attentions=None,  # Would need model modifications
                cycle_attentions=cycle_attentions
            )
            
            self.attention_visualizations.append(visualization)
            return visualization
    
    def analyze_feature_importance(self,
                                 model: MacBookEmailTRM,
                                 dataloader: DataLoader,
                                 num_samples: int = 1000,
                                 method: str = "gradient") -> FeatureImportance:
        """
        Analyze feature importance across the dataset.
        
        Args:
            model: EmailTRM model
            dataloader: Data loader for analysis
            num_samples: Number of samples to analyze
            method: Importance analysis method ("gradient", "permutation", "integrated_gradients")
            
        Returns:
            Feature importance analysis
        """
        logger.info(f"Analyzing feature importance using {method} method...")
        
        model.eval()
        
        # Token importance accumulation
        token_importance = defaultdict(float)
        token_counts = defaultdict(int)
        
        # Category-specific importance
        category_token_importance = {cat: defaultdict(float) for cat in self.category_names}
        category_token_counts = {cat: defaultdict(int) for cat in self.category_names}
        
        # Structure importance
        structure_importance = defaultdict(float)
        structure_counts = defaultdict(int)
        
        samples_processed = 0
        
        for batch_idx, (set_name, batch, batch_size) in enumerate(dataloader):
            if samples_processed >= num_samples:
                break
            
            inputs = batch['inputs']
            labels = batch['labels']
            
            for i in range(min(batch_size, num_samples - samples_processed)):
                sample_input = inputs[i:i+1]  # Single sample
                sample_label = labels[i:i+1]
                
                # Compute importance scores
                if method == "gradient":
                    importance_scores = self._compute_gradient_importance(model, sample_input, sample_label)
                elif method == "permutation":
                    importance_scores = self._compute_permutation_importance(model, sample_input, sample_label)
                elif method == "integrated_gradients":
                    importance_scores = self._compute_integrated_gradients(model, sample_input, sample_label)
                else:
                    raise ValueError(f"Unknown importance method: {method}")
                
                # Extract tokens and analyze
                tokens = self._extract_tokens(sample_input[0])
                category = self.category_names[sample_label[0].item()]
                
                # Accumulate token importance
                for token, importance in zip(tokens, importance_scores):
                    if token and token.strip():  # Skip empty tokens
                        token_importance[token] += importance
                        token_counts[token] += 1
                        
                        category_token_importance[category][token] += importance
                        category_token_counts[category][token] += 1
                
                # Analyze structure importance
                structure_scores = self._analyze_structure_importance(sample_input[0], importance_scores)
                for structure, score in structure_scores.items():
                    structure_importance[structure] += score
                    structure_counts[structure] += 1
                
                samples_processed += 1
        
        # Normalize importance scores
        normalized_token_importance = {
            token: importance / token_counts[token] 
            for token, importance in token_importance.items()
        }
        
        normalized_category_importance = {}
        for category in self.category_names:
            normalized_category_importance[category] = {
                token: importance / category_token_counts[category][token]
                for token, importance in category_token_importance[category].items()
                if category_token_counts[category][token] > 0
            }
        
        normalized_structure_importance = {
            structure: importance / structure_counts[structure]
            for structure, importance in structure_importance.items()
            if structure_counts[structure] > 0
        }
        
        # Find most/least important tokens
        sorted_tokens = sorted(normalized_token_importance.items(), key=lambda x: x[1], reverse=True)
        most_important = sorted_tokens[:50]  # Top 50
        least_important = sorted_tokens[-50:]  # Bottom 50
        
        # Find discriminative features between categories
        discriminative_features = self._find_discriminative_features(normalized_category_importance)
        
        # Create n-gram importance (simplified)
        ngram_importance = self._compute_ngram_importance(normalized_token_importance)
        
        feature_importance = FeatureImportance(
            token_importance=normalized_token_importance,
            ngram_importance=ngram_importance,
            structure_importance=normalized_structure_importance,
            category_token_importance=normalized_category_importance,
            most_important_tokens=most_important,
            least_important_tokens=least_important,
            discriminative_features=discriminative_features
        )
        
        # Cache for future use
        self.feature_importance_cache[method] = feature_importance
        
        logger.info(f"Feature importance analysis completed for {samples_processed} samples")
        return feature_importance
    
    def analyze_reasoning_cycles(self,
                               model: MacBookEmailTRM,
                               inputs: torch.Tensor,
                               labels: Optional[torch.Tensor] = None,
                               sample_id: Optional[str] = None) -> ReasoningCycleAnalysis:
        """
        Analyze TRM reasoning cycles for interpretability.
        
        Args:
            model: EmailTRM model
            inputs: Input tensor [1, seq_len] (single sample)
            labels: True labels [1] (optional)
            sample_id: Sample identifier (optional)
            
        Returns:
            Reasoning cycle analysis
        """
        model.eval()
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        
        sample_id = sample_id or f"reasoning_{torch.randint(0, 10000, (1,)).item()}"
        
        with torch.no_grad():
            # Forward pass with all cycles
            outputs = model(inputs, return_all_cycles=True)
            
            if 'all_logits' not in outputs or outputs['all_logits'] is None:
                logger.warning("Model does not return cycle-wise outputs")
                return self._create_empty_reasoning_analysis(sample_id)
            
            all_logits = outputs['all_logits']  # [num_cycles, batch_size, num_classes]
            num_cycles = all_logits.shape[0]
            
            # Analyze prediction progression
            cycle_predictions = []
            cycle_confidences = []
            cycle_entropies = []
            
            for cycle in range(num_cycles):
                cycle_logits = all_logits[cycle, 0]  # Single sample
                cycle_probs = F.softmax(cycle_logits, dim=-1)
                
                prediction = torch.argmax(cycle_logits).item()
                confidence = cycle_probs.max().item()
                entropy = -torch.sum(cycle_probs * torch.log(cycle_probs + 1e-8)).item()
                
                cycle_predictions.append(prediction)
                cycle_confidences.append(confidence)
                cycle_entropies.append(entropy)
            
            # Analyze attention evolution (simplified)
            attention_evolution = []
            for cycle in range(num_cycles):
                # Simplified attention extraction
                cycle_attention = self._extract_cycle_attention(model, inputs, cycle)
                attention_evolution.append(cycle_attention)
            
            # Compute stability metrics
            prediction_stability = self._compute_prediction_stability(cycle_predictions)
            attention_stability = self._compute_attention_stability(attention_evolution)
            
            # Analyze feature focus changes
            feature_focus_changes = self._analyze_feature_focus_changes(
                inputs[0], attention_evolution
            )
            
            # Find final decision cycle
            final_decision_cycle = self._find_final_decision_cycle(
                cycle_predictions, cycle_confidences
            )
            
            # Compute reasoning quality metrics
            reasoning_efficiency = self._compute_reasoning_efficiency(
                cycle_confidences, cycle_entropies
            )
            reasoning_consistency = self._compute_reasoning_consistency(
                cycle_predictions, cycle_confidences
            )
            
            analysis = ReasoningCycleAnalysis(
                sample_id=sample_id,
                num_cycles=num_cycles,
                cycle_predictions=cycle_predictions,
                cycle_confidences=cycle_confidences,
                cycle_entropies=cycle_entropies,
                attention_evolution=attention_evolution,
                feature_focus_changes=feature_focus_changes,
                prediction_stability=prediction_stability,
                attention_stability=attention_stability,
                confidence_progression=cycle_confidences,
                final_decision_cycle=final_decision_cycle,
                reasoning_efficiency=reasoning_efficiency,
                reasoning_consistency=reasoning_consistency
            )
            
            self.reasoning_analyses.append(analysis)
            return analysis
    
    def visualize_attention(self,
                          visualization: AttentionVisualization,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> Optional[str]:
        """
        Create attention visualization plot.
        
        Args:
            visualization: Attention visualization data
            save_path: Path to save plot (optional)
            show_plot: Whether to display plot
            
        Returns:
            Path to saved plot or None
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Token attention heatmap
        ax1 = axes[0, 0]
        tokens = visualization.tokens[:50]  # Limit for readability
        attention = visualization.attention_weights[:50]
        
        # Create color map
        colors = ['white', 'lightblue', 'blue', 'darkblue']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
        
        # Plot attention as heatmap
        attention_matrix = np.array(attention).reshape(1, -1)
        im1 = ax1.imshow(attention_matrix, cmap=cmap, aspect='auto')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right')
        ax1.set_yticks([])
        ax1.set_title(f'Token Attention Weights\nPredicted: {visualization.predicted_category} '
                     f'(Confidence: {visualization.confidence:.3f})')
        plt.colorbar(im1, ax=ax1)
        
        # Structure attention bar plot
        ax2 = axes[0, 1]
        if visualization.structure_attention:
            structures = list(visualization.structure_attention.keys())
            struct_weights = list(visualization.structure_attention.values())
            
            bars = ax2.bar(structures, struct_weights, color='skyblue')
            ax2.set_title('Email Structure Attention')
            ax2.set_ylabel('Attention Weight')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, weight in zip(bars, struct_weights):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')
        
        # Cycle attention progression (if available)
        ax3 = axes[1, 0]
        if visualization.cycle_attentions:
            cycle_data = np.array(visualization.cycle_attentions)
            im3 = ax3.imshow(cycle_data, cmap='viridis', aspect='auto')
            ax3.set_xlabel('Token Position')
            ax3.set_ylabel('Reasoning Cycle')
            ax3.set_title('Attention Evolution Across Cycles')
            plt.colorbar(im3, ax=ax3)
        else:
            ax3.text(0.5, 0.5, 'Cycle attention data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Attention Evolution Across Cycles')
        
        # Top attended tokens
        ax4 = axes[1, 1]
        if len(tokens) > 0 and len(attention) > 0:
            # Get top 10 tokens by attention
            token_attention_pairs = list(zip(tokens, attention))
            token_attention_pairs.sort(key=lambda x: x[1], reverse=True)
            top_tokens = token_attention_pairs[:10]
            
            if top_tokens:
                top_token_names = [pair[0] for pair in top_tokens]
                top_token_weights = [pair[1] for pair in top_tokens]
                
                bars = ax4.barh(range(len(top_token_names)), top_token_weights, color='lightcoral')
                ax4.set_yticks(range(len(top_token_names)))
                ax4.set_yticklabels(top_token_names)
                ax4.set_xlabel('Attention Weight')
                ax4.set_title('Top 10 Attended Tokens')
                ax4.invert_yaxis()
                
                # Add value labels
                for i, (bar, weight) in enumerate(zip(bars, top_token_weights)):
                    width = bar.get_width()
                    ax4.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                            f'{weight:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.output_dir:
            if save_path is None:
                save_path = self.output_dir / f"{visualization.sample_id}_attention_viz.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(save_path) if save_path else None
    
    def visualize_reasoning_cycles(self,
                                 analysis: ReasoningCycleAnalysis,
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True) -> Optional[str]:
        """
        Create reasoning cycle visualization.
        
        Args:
            analysis: Reasoning cycle analysis data
            save_path: Path to save plot (optional)
            show_plot: Whether to display plot
            
        Returns:
            Path to saved plot or None
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        cycles = range(analysis.num_cycles)
        
        # Prediction confidence progression
        ax1 = axes[0, 0]
        ax1.plot(cycles, analysis.cycle_confidences, 'o-', color='blue', linewidth=2, markersize=6)
        ax1.set_xlabel('Reasoning Cycle')
        ax1.set_ylabel('Prediction Confidence')
        ax1.set_title('Confidence Progression')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Mark final decision cycle
        if analysis.final_decision_cycle < len(analysis.cycle_confidences):
            ax1.axvline(x=analysis.final_decision_cycle, color='red', linestyle='--', 
                       label=f'Final Decision (Cycle {analysis.final_decision_cycle})')
            ax1.legend()
        
        # Prediction entropy progression
        ax2 = axes[0, 1]
        ax2.plot(cycles, analysis.cycle_entropies, 'o-', color='green', linewidth=2, markersize=6)
        ax2.set_xlabel('Reasoning Cycle')
        ax2.set_ylabel('Prediction Entropy')
        ax2.set_title('Uncertainty Progression')
        ax2.grid(True, alpha=0.3)
        
        # Prediction changes
        ax3 = axes[1, 0]
        prediction_changes = [1 if i == 0 or analysis.cycle_predictions[i] != analysis.cycle_predictions[i-1] 
                             else 0 for i in range(analysis.num_cycles)]
        ax3.bar(cycles, prediction_changes, color='orange', alpha=0.7)
        ax3.set_xlabel('Reasoning Cycle')
        ax3.set_ylabel('Prediction Changed')
        ax3.set_title('Prediction Stability')
        ax3.set_ylim(0, 1.2)
        
        # Add prediction labels
        for i, pred in enumerate(analysis.cycle_predictions):
            category = self.category_names[pred] if pred < len(self.category_names) else f"Class {pred}"
            ax3.text(i, 0.1, category[:3], ha='center', va='bottom', rotation=45, fontsize=8)
        
        # Attention evolution heatmap
        ax4 = axes[1, 1]
        if analysis.attention_evolution and len(analysis.attention_evolution) > 0:
            attention_matrix = np.array(analysis.attention_evolution)
            if attention_matrix.size > 0:
                # Limit tokens for visualization
                max_tokens = 30
                if attention_matrix.shape[1] > max_tokens:
                    attention_matrix = attention_matrix[:, :max_tokens]
                
                im4 = ax4.imshow(attention_matrix, cmap='viridis', aspect='auto')
                ax4.set_xlabel('Token Position')
                ax4.set_ylabel('Reasoning Cycle')
                ax4.set_title('Attention Evolution')
                plt.colorbar(im4, ax=ax4)
        else:
            ax4.text(0.5, 0.5, 'Attention evolution\ndata not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Attention Evolution')
        
        # Add overall statistics as text
        stats_text = f"""Reasoning Quality Metrics:
Prediction Stability: {analysis.prediction_stability:.3f}
Attention Stability: {analysis.attention_stability:.3f}
Reasoning Efficiency: {analysis.reasoning_efficiency:.3f}
Reasoning Consistency: {analysis.reasoning_consistency:.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path or self.output_dir:
            if save_path is None:
                save_path = self.output_dir / f"{analysis.sample_id}_reasoning_cycles.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reasoning cycle visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(save_path) if save_path else None
    
    def generate_interpretability_report(self,
                                       model: MacBookEmailTRM,
                                       sample_inputs: List[torch.Tensor],
                                       sample_labels: Optional[List[torch.Tensor]] = None,
                                       sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report for multiple samples.
        
        Args:
            model: EmailTRM model
            sample_inputs: List of input tensors
            sample_labels: List of label tensors (optional)
            sample_texts: List of original email texts (optional)
            
        Returns:
            Comprehensive interpretability report
        """
        logger.info(f"Generating interpretability report for {len(sample_inputs)} samples...")
        
        report = {
            'timestamp': torch.cuda.Event(enable_timing=True),
            'num_samples': len(sample_inputs),
            'attention_analyses': [],
            'reasoning_analyses': [],
            'feature_importance': None,
            'summary_statistics': {}
        }
        
        # Analyze individual samples
        for i, inputs in enumerate(sample_inputs):
            sample_id = f"sample_{i:03d}"
            labels = sample_labels[i] if sample_labels else None
            text = sample_texts[i] if sample_texts else None
            
            # Attention analysis
            attention_viz = self.analyze_sample_attention(
                model, inputs, labels, text, sample_id
            )
            report['attention_analyses'].append(asdict(attention_viz))
            
            # Reasoning analysis
            reasoning_analysis = self.analyze_reasoning_cycles(
                model, inputs, labels, sample_id
            )
            report['reasoning_analyses'].append(asdict(reasoning_analysis))
        
        # Compute summary statistics
        report['summary_statistics'] = self._compute_interpretability_summary_stats()
        
        # Save report
        if self.output_dir:
            report_file = self.output_dir / "interpretability_report.json"
            
            try:
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"Interpretability report saved to {report_file}")
                
            except Exception as e:
                logger.error(f"Failed to save interpretability report: {e}")
        
        return report
    
    # Helper methods
    
    def _extract_tokens(self, inputs: torch.Tensor) -> List[str]:
        """Extract tokens from input tensor."""
        # This is a simplified implementation
        # In practice, you'd use the tokenizer's decode functionality
        token_ids = inputs.cpu().numpy()
        tokens = []
        
        for token_id in token_ids:
            if hasattr(self.tokenizer, 'id_to_token'):
                token = self.tokenizer.id_to_token.get(token_id, f"<unk_{token_id}>")
            else:
                token = f"token_{token_id}"
            tokens.append(token)
        
        return tokens
    
    def _extract_attention_weights(self, 
                                 model: MacBookEmailTRM, 
                                 inputs: torch.Tensor, 
                                 outputs: Dict[str, Any]) -> List[float]:
        """Extract attention weights (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd need to modify the model to return attention weights
        
        seq_len = inputs.shape[1]
        
        # Use gradient-based attention approximation
        if 'logits' in outputs:
            logits = outputs['logits']
            predicted_class = torch.argmax(logits, dim=-1)
            
            # Compute gradients w.r.t. inputs
            inputs.requires_grad_(True)
            model.zero_grad()
            
            class_logit = logits[0, predicted_class[0]]
            class_logit.backward(retain_graph=True)
            
            if inputs.grad is not None:
                attention_weights = torch.abs(inputs.grad[0]).cpu().numpy()
                # Normalize
                attention_weights = attention_weights / (attention_weights.sum() + 1e-8)
                return attention_weights.tolist()
        
        # Fallback: uniform attention
        uniform_weight = 1.0 / seq_len
        return [uniform_weight] * seq_len
    
    def _analyze_structure_attention(self, 
                                   inputs: torch.Tensor, 
                                   attention_weights: List[float]) -> Dict[str, float]:
        """Analyze attention by email structure."""
        # Simplified structure detection based on special tokens
        structure_attention = {
            'subject': 0.0,
            'body': 0.0,
            'sender': 0.0,
            'recipient': 0.0,
            'other': 0.0
        }
        
        # This would need to be implemented based on your tokenizer's special tokens
        # For now, return uniform distribution
        return {k: 0.2 for k in structure_attention.keys()}
    
    def _compute_gradient_importance(self, 
                                   model: MacBookEmailTRM, 
                                   inputs: torch.Tensor, 
                                   labels: torch.Tensor) -> List[float]:
        """Compute gradient-based feature importance."""
        model.eval()
        inputs.requires_grad_(True)
        
        outputs = model(inputs, labels=labels)
        loss = outputs.get('loss', 0)
        
        if loss != 0:
            model.zero_grad()
            loss.backward()
            
            if inputs.grad is not None:
                importance = torch.abs(inputs.grad[0]).cpu().numpy()
                return importance.tolist()
        
        # Fallback
        return [0.0] * inputs.shape[1]
    
    def _compute_permutation_importance(self, 
                                      model: MacBookEmailTRM, 
                                      inputs: torch.Tensor, 
                                      labels: torch.Tensor) -> List[float]:
        """Compute permutation-based feature importance."""
        # Simplified implementation
        model.eval()
        
        with torch.no_grad():
            # Baseline prediction
            baseline_outputs = model(inputs, labels=labels)
            baseline_logits = baseline_outputs['logits']
            baseline_pred = torch.argmax(baseline_logits, dim=-1)
            
            importance_scores = []
            
            for i in range(inputs.shape[1]):
                # Permute token at position i
                permuted_inputs = inputs.clone()
                permuted_inputs[0, i] = torch.randint(0, 1000, (1,)).item()  # Random token
                
                # Get new prediction
                permuted_outputs = model(permuted_inputs)
                permuted_logits = permuted_outputs['logits']
                permuted_pred = torch.argmax(permuted_logits, dim=-1)
                
                # Importance is change in prediction confidence
                baseline_conf = F.softmax(baseline_logits, dim=-1)[0, baseline_pred[0]]
                permuted_conf = F.softmax(permuted_logits, dim=-1)[0, baseline_pred[0]]
                
                importance = abs(baseline_conf - permuted_conf).item()
                importance_scores.append(importance)
            
            return importance_scores
    
    def _compute_integrated_gradients(self, 
                                    model: MacBookEmailTRM, 
                                    inputs: torch.Tensor, 
                                    labels: torch.Tensor,
                                    steps: int = 20) -> List[float]:
        """Compute integrated gradients importance."""
        # Simplified implementation
        model.eval()
        
        # Baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)
        integrated_grads = torch.zeros_like(inputs[0], dtype=torch.float)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            outputs = model(interpolated, labels=labels)
            loss = outputs.get('loss', 0)
            
            if loss != 0:
                model.zero_grad()
                loss.backward()
                
                if interpolated.grad is not None:
                    integrated_grads += interpolated.grad[0]
        
        # Average and multiply by input difference
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (inputs[0] - baseline[0])
        
        return torch.abs(integrated_grads).cpu().numpy().tolist()
    
    def _analyze_structure_importance(self, 
                                    inputs: torch.Tensor, 
                                    importance_scores: List[float]) -> Dict[str, float]:
        """Analyze importance by email structure."""
        # Simplified implementation
        return {
            'subject': sum(importance_scores[:10]) if len(importance_scores) > 10 else 0.0,
            'body': sum(importance_scores[10:]) if len(importance_scores) > 10 else sum(importance_scores),
            'sender': 0.0,
            'recipient': 0.0
        }
    
    def _find_discriminative_features(self, 
                                    category_importance: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Find discriminative features between categories."""
        discriminative = {}
        
        categories = list(category_importance.keys())
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                # Find tokens that are important for cat1 but not cat2
                cat1_tokens = category_importance[cat1]
                cat2_tokens = category_importance[cat2]
                
                discriminative_tokens = []
                
                for token in cat1_tokens:
                    importance_diff = cat1_tokens[token] - cat2_tokens.get(token, 0.0)
                    if importance_diff > 0.01:  # Threshold
                        discriminative_tokens.append((token, importance_diff))
                
                discriminative_tokens.sort(key=lambda x: x[1], reverse=True)
                discriminative[f"{cat1}_vs_{cat2}"] = discriminative_tokens[:10]
        
        return discriminative
    
    def _compute_ngram_importance(self, 
                                token_importance: Dict[str, float]) -> Dict[str, float]:
        """Compute n-gram importance from token importance."""
        # Simplified implementation - just return token importance
        return token_importance.copy()
    
    def _create_empty_reasoning_analysis(self, sample_id: str) -> ReasoningCycleAnalysis:
        """Create empty reasoning analysis for models without cycle support."""
        return ReasoningCycleAnalysis(
            sample_id=sample_id,
            num_cycles=1,
            cycle_predictions=[0],
            cycle_confidences=[0.0],
            cycle_entropies=[0.0],
            attention_evolution=[],
            feature_focus_changes=[],
            prediction_stability=1.0,
            attention_stability=1.0,
            confidence_progression=[0.0],
            final_decision_cycle=0,
            reasoning_efficiency=1.0,
            reasoning_consistency=1.0
        )
    
    def _extract_cycle_attention(self, 
                               model: MacBookEmailTRM, 
                               inputs: torch.Tensor, 
                               cycle: int) -> List[float]:
        """Extract attention for a specific cycle."""
        # Simplified implementation
        seq_len = inputs.shape[1]
        return [1.0 / seq_len] * seq_len
    
    def _compute_prediction_stability(self, cycle_predictions: List[int]) -> float:
        """Compute prediction stability across cycles."""
        if len(cycle_predictions) <= 1:
            return 1.0
        
        changes = sum(1 for i in range(1, len(cycle_predictions)) 
                     if cycle_predictions[i] != cycle_predictions[i-1])
        
        return 1.0 - (changes / (len(cycle_predictions) - 1))
    
    def _compute_attention_stability(self, attention_evolution: List[List[float]]) -> float:
        """Compute attention stability across cycles."""
        if len(attention_evolution) <= 1:
            return 1.0
        
        # Compute average cosine similarity between consecutive cycles
        similarities = []
        
        for i in range(1, len(attention_evolution)):
            att1 = np.array(attention_evolution[i-1])
            att2 = np.array(attention_evolution[i])
            
            if len(att1) == len(att2) and len(att1) > 0:
                # Cosine similarity
                dot_product = np.dot(att1, att2)
                norm1 = np.linalg.norm(att1)
                norm2 = np.linalg.norm(att2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _analyze_feature_focus_changes(self, 
                                     inputs: torch.Tensor, 
                                     attention_evolution: List[List[float]]) -> List[Dict[str, float]]:
        """Analyze how feature focus changes across cycles."""
        # Simplified implementation
        return [{'focus_change': 0.0} for _ in attention_evolution]
    
    def _find_final_decision_cycle(self, 
                                 cycle_predictions: List[int], 
                                 cycle_confidences: List[float]) -> int:
        """Find the cycle where the final decision was made."""
        if not cycle_predictions:
            return 0
        
        final_prediction = cycle_predictions[-1]
        
        # Find first cycle where prediction matches final and confidence is high
        for i, (pred, conf) in enumerate(zip(cycle_predictions, cycle_confidences)):
            if pred == final_prediction and conf > 0.7:
                return i
        
        return len(cycle_predictions) - 1
    
    def _compute_reasoning_efficiency(self, 
                                    cycle_confidences: List[float], 
                                    cycle_entropies: List[float]) -> float:
        """Compute reasoning efficiency."""
        if not cycle_confidences:
            return 1.0
        
        # Efficiency is how quickly confidence increases and entropy decreases
        confidence_increase = cycle_confidences[-1] - cycle_confidences[0] if len(cycle_confidences) > 1 else 0
        entropy_decrease = cycle_entropies[0] - cycle_entropies[-1] if len(cycle_entropies) > 1 else 0
        
        efficiency = (confidence_increase + entropy_decrease) / 2
        return max(0.0, min(1.0, efficiency))
    
    def _compute_reasoning_consistency(self, 
                                     cycle_predictions: List[int], 
                                     cycle_confidences: List[float]) -> float:
        """Compute reasoning consistency."""
        if len(cycle_predictions) <= 1:
            return 1.0
        
        # Consistency is stability + confidence progression
        stability = self._compute_prediction_stability(cycle_predictions)
        
        # Confidence should generally increase
        confidence_trend = 0.0
        if len(cycle_confidences) > 1:
            increases = sum(1 for i in range(1, len(cycle_confidences)) 
                          if cycle_confidences[i] >= cycle_confidences[i-1])
            confidence_trend = increases / (len(cycle_confidences) - 1)
        
        return (stability + confidence_trend) / 2
    
    def _compute_interpretability_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for interpretability analyses."""
        stats = {
            'num_attention_analyses': len(self.attention_visualizations),
            'num_reasoning_analyses': len(self.reasoning_analyses),
            'avg_reasoning_cycles': 0.0,
            'avg_prediction_stability': 0.0,
            'avg_attention_stability': 0.0,
            'avg_reasoning_efficiency': 0.0
        }
        
        if self.reasoning_analyses:
            stats['avg_reasoning_cycles'] = np.mean([a.num_cycles for a in self.reasoning_analyses])
            stats['avg_prediction_stability'] = np.mean([a.prediction_stability for a in self.reasoning_analyses])
            stats['avg_attention_stability'] = np.mean([a.attention_stability for a in self.reasoning_analyses])
            stats['avg_reasoning_efficiency'] = np.mean([a.reasoning_efficiency for a in self.reasoning_analyses])
        
        return stats
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        return " ".join(tokens)


# Example usage and testing
if __name__ == "__main__":
    print("Email interpretability system ready!")
    
    # Example of how to use
    """
    # Create interpretability analyzer
    from models.email_tokenizer import EmailTokenizer
    
    tokenizer = EmailTokenizer(vocab_size=5000)
    analyzer = EmailInterpretabilityAnalyzer(
        tokenizer=tokenizer,
        output_dir="interpretability_output"
    )
    
    # Analyze attention for a sample
    attention_viz = analyzer.analyze_sample_attention(
        model=trained_model,
        inputs=sample_inputs,
        labels=sample_labels,
        email_text=sample_text
    )
    
    # Visualize attention
    analyzer.visualize_attention(attention_viz)
    
    # Analyze reasoning cycles
    reasoning_analysis = analyzer.analyze_reasoning_cycles(
        model=trained_model,
        inputs=sample_inputs,
        labels=sample_labels
    )
    
    # Visualize reasoning
    analyzer.visualize_reasoning_cycles(reasoning_analysis)
    
    print(f"Attention analysis completed for {attention_viz.sample_id}")
    print(f"Reasoning analysis completed: {reasoning_analysis.num_cycles} cycles")
    """