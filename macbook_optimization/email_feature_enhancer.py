"""
Email-Specific Feature Enhancer for EmailTRM

This module provides utilities to configure and enhance email-specific features
in EmailTRM models, including structure embeddings, hierarchical attention,
and confidence calibration for production predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig


logger = logging.getLogger(__name__)


@dataclass
class EmailFeatureConfig:
    """Configuration for email-specific features"""
    
    # Email structure configuration
    enable_structure_embeddings: bool = True
    structure_embedding_dim: int = 64
    structure_dropout: float = 0.1
    
    # Hierarchical attention configuration
    enable_hierarchical_attention: bool = True
    subject_attention_weight: float = 2.0
    sender_attention_weight: float = 1.5
    body_attention_weight: float = 1.0
    metadata_attention_weight: float = 0.8
    
    # Attention pooling configuration
    pooling_strategy: str = "weighted"  # "mean", "max", "weighted", "attention", "hierarchical"
    attention_temperature: float = 1.0
    pooling_dropout: float = 0.1
    
    # Confidence calibration configuration
    enable_confidence_calibration: bool = True
    temperature_init: float = 1.0
    calibration_regularization: float = 0.01
    
    # Category-specific enhancements
    enable_category_embeddings: bool = True
    category_embedding_dim: int = 128
    contrastive_margin: float = 0.1
    contrastive_weight: float = 0.1
    
    # Production-specific features
    enable_prediction_explanations: bool = True
    enable_uncertainty_estimation: bool = True
    enable_attention_visualization: bool = True


class EmailStructureEnhancer:
    """Enhancer for email structure awareness features"""
    
    def __init__(self, config: EmailFeatureConfig):
        self.config = config
        
    def create_enhanced_structure_embeddings(self, 
                                           vocab_size: int, 
                                           hidden_size: int) -> nn.Module:
        """
        Create enhanced email structure embeddings
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            
        Returns:
            Enhanced structure embedding module
        """
        
        class EnhancedEmailStructureEmbedding(nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int, config: EmailFeatureConfig):
                super().__init__()
                self.config = config
                self.hidden_size = hidden_size
                
                if config.enable_structure_embeddings:
                    # Structure type embeddings for different email parts
                    self.structure_types = {
                        'subject': 3, 'body': 4, 'from': 5, 'to': 6,
                        'cc': 7, 'bcc': 8, 'date': 9, 'reply_to': 10
                    }
                    
                    # Structure embeddings
                    self.structure_embeddings = nn.Embedding(
                        len(self.structure_types) + 3,  # +3 for pad, eos, unk
                        config.structure_embedding_dim
                    )
                    
                    # Position embeddings for email structure
                    self.position_embeddings = nn.Embedding(512, config.structure_embedding_dim)
                    
                    # Structure projection to hidden size
                    self.structure_proj = nn.Sequential(
                        nn.Linear(config.structure_embedding_dim, hidden_size),
                        nn.GELU(),
                        nn.Dropout(config.structure_dropout),
                        nn.Linear(hidden_size, hidden_size)
                    )
                    
                    # Layer normalization for structure embeddings
                    self.structure_norm = nn.LayerNorm(hidden_size)
                    
                    # Learnable mixing weights for structure and content
                    self.mixing_weights = nn.Parameter(torch.tensor([0.8, 0.2]))  # [content, structure]
                    
                    self._init_weights()
            
            def _init_weights(self):
                """Initialize embedding weights"""
                if self.config.enable_structure_embeddings:
                    nn.init.normal_(self.structure_embeddings.weight, std=0.02)
                    nn.init.normal_(self.position_embeddings.weight, std=0.02)
                    
                    for module in self.structure_proj:
                        if isinstance(module, nn.Linear):
                            nn.init.xavier_uniform_(module.weight)
                            nn.init.zeros_(module.bias)
            
            def forward(self, input_ids: torch.Tensor, 
                       position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Forward pass for enhanced structure embeddings
                
                Args:
                    input_ids: [batch_size, seq_len]
                    position_ids: [batch_size, seq_len] optional
                    
                Returns:
                    structure_embeddings: [batch_size, seq_len, hidden_size]
                """
                
                if not self.config.enable_structure_embeddings:
                    return torch.zeros(input_ids.shape + (self.hidden_size,), 
                                     device=input_ids.device, dtype=torch.float32)
                
                batch_size, seq_len = input_ids.shape
                
                # Create structure type tensor
                structure_types = torch.zeros_like(input_ids)
                
                # Map special tokens to structure types
                for struct_name, token_id in self.structure_types.items():
                    structure_types = torch.where(input_ids == token_id, token_id, structure_types)
                
                # Get structure embeddings
                structure_emb = self.structure_embeddings(structure_types)
                
                # Add position embeddings
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                
                position_emb = self.position_embeddings(position_ids)
                structure_emb = structure_emb + position_emb
                
                # Project to hidden size
                structure_emb = self.structure_proj(structure_emb)
                structure_emb = self.structure_norm(structure_emb)
                
                return structure_emb
        
        return EnhancedEmailStructureEmbedding(vocab_size, hidden_size, self.config)


class HierarchicalAttentionEnhancer:
    """Enhancer for hierarchical attention mechanisms"""
    
    def __init__(self, config: EmailFeatureConfig):
        self.config = config
        
    def create_hierarchical_attention_pooling(self, hidden_size: int) -> nn.Module:
        """
        Create hierarchical attention pooling module
        
        Args:
            hidden_size: Hidden dimension size
            
        Returns:
            Hierarchical attention pooling module
        """
        
        class HierarchicalAttentionPooling(nn.Module):
            def __init__(self, hidden_size: int, config: EmailFeatureConfig):
                super().__init__()
                self.config = config
                self.hidden_size = hidden_size
                
                if config.enable_hierarchical_attention:
                    # Multi-head attention for different email parts
                    self.subject_attention = nn.MultiheadAttention(
                        hidden_size, num_heads=8, dropout=config.pooling_dropout, batch_first=True
                    )
                    self.body_attention = nn.MultiheadAttention(
                        hidden_size, num_heads=8, dropout=config.pooling_dropout, batch_first=True
                    )
                    self.metadata_attention = nn.MultiheadAttention(
                        hidden_size, num_heads=4, dropout=config.pooling_dropout, batch_first=True
                    )
                    
                    # Attention weights for different parts
                    self.part_weights = nn.Parameter(torch.tensor([
                        config.subject_attention_weight,
                        config.body_attention_weight,
                        config.sender_attention_weight,
                        config.metadata_attention_weight
                    ]))
                    
                    # Final attention pooling
                    self.final_attention = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.GELU(),
                        nn.Dropout(config.pooling_dropout),
                        nn.Linear(hidden_size // 2, 1)
                    )
                    
                    # Layer normalization
                    self.layer_norm = nn.LayerNorm(hidden_size)
                    
                    # Temperature parameter for attention
                    self.temperature = nn.Parameter(torch.tensor(config.attention_temperature))
            
            def _extract_email_parts(self, hidden_states: torch.Tensor, 
                                   input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
                """Extract different parts of email from hidden states"""
                
                parts = {}
                
                # Define token mappings
                token_mappings = {
                    'subject': 3, 'body': 4, 'from': 5, 'to': 6
                }
                
                for part_name, token_id in token_mappings.items():
                    # Find positions of this token type
                    mask = (input_ids == token_id)
                    
                    if mask.any():
                        # Extract hidden states for this part
                        part_states = hidden_states[mask.unsqueeze(-1).expand_as(hidden_states)]
                        part_states = part_states.view(-1, hidden_states.size(-1))
                        
                        if part_states.size(0) > 0:
                            parts[part_name] = part_states.unsqueeze(0)  # Add batch dimension
                    
                    if part_name not in parts:
                        # Create empty tensor if part not found
                        parts[part_name] = torch.zeros(1, 1, hidden_states.size(-1), 
                                                     device=hidden_states.device)
                
                return parts
            
            def forward(self, hidden_states: torch.Tensor, 
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Forward pass for hierarchical attention pooling
                
                Args:
                    hidden_states: [batch_size, seq_len, hidden_size]
                    input_ids: [batch_size, seq_len]
                    attention_mask: [batch_size, seq_len] optional
                    
                Returns:
                    pooled_output: [batch_size, hidden_size]
                """
                
                if not self.config.enable_hierarchical_attention:
                    # Fallback to simple mean pooling
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                        hidden_states = hidden_states * mask_expanded
                        pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    else:
                        pooled = hidden_states.mean(dim=1)
                    return pooled
                
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Apply layer normalization
                hidden_states = self.layer_norm(hidden_states)
                
                # Extract email parts
                email_parts = self._extract_email_parts(hidden_states, input_ids)
                
                # Apply part-specific attention
                part_representations = []
                
                # Subject attention (highest weight)
                if 'subject' in email_parts and email_parts['subject'].size(1) > 0:
                    subject_repr, _ = self.subject_attention(
                        email_parts['subject'], email_parts['subject'], email_parts['subject']
                    )
                    subject_repr = subject_repr.mean(dim=1)  # Pool over sequence
                    part_representations.append(subject_repr * self.part_weights[0])
                
                # Body attention
                if 'body' in email_parts and email_parts['body'].size(1) > 0:
                    body_repr, _ = self.body_attention(
                        email_parts['body'], email_parts['body'], email_parts['body']
                    )
                    body_repr = body_repr.mean(dim=1)
                    part_representations.append(body_repr * self.part_weights[1])
                
                # Metadata attention (from, to)
                metadata_parts = []
                for part_name in ['from', 'to']:
                    if part_name in email_parts and email_parts[part_name].size(1) > 0:
                        metadata_parts.append(email_parts[part_name])
                
                if metadata_parts:
                    metadata_concat = torch.cat(metadata_parts, dim=1)
                    metadata_repr, _ = self.metadata_attention(
                        metadata_concat, metadata_concat, metadata_concat
                    )
                    metadata_repr = metadata_repr.mean(dim=1)
                    part_representations.append(metadata_repr * self.part_weights[3])
                
                # Combine part representations
                if part_representations:
                    # Stack and apply final attention
                    stacked_parts = torch.stack(part_representations, dim=1)  # [batch, num_parts, hidden]
                    
                    # Compute attention weights
                    attention_scores = self.final_attention(stacked_parts).squeeze(-1)  # [batch, num_parts]
                    attention_scores = attention_scores / self.temperature
                    attention_weights = F.softmax(attention_scores, dim=1)
                    
                    # Weighted combination
                    pooled_output = (stacked_parts * attention_weights.unsqueeze(-1)).sum(dim=1)
                else:
                    # Fallback to global attention pooling
                    attention_scores = self.final_attention(hidden_states).squeeze(-1)  # [batch, seq_len]
                    
                    if attention_mask is not None:
                        attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
                    
                    attention_weights = F.softmax(attention_scores / self.temperature, dim=1)
                    pooled_output = (hidden_states * attention_weights.unsqueeze(-1)).sum(dim=1)
                
                return pooled_output
        
        return HierarchicalAttentionPooling(hidden_size, self.config)


class ConfidenceCalibrator:
    """Confidence calibration for production predictions"""
    
    def __init__(self, config: EmailFeatureConfig):
        self.config = config
        
    def create_calibration_module(self, num_classes: int) -> nn.Module:
        """
        Create confidence calibration module
        
        Args:
            num_classes: Number of email categories
            
        Returns:
            Confidence calibration module
        """
        
        class ConfidenceCalibrationModule(nn.Module):
            def __init__(self, num_classes: int, config: EmailFeatureConfig):
                super().__init__()
                self.config = config
                self.num_classes = num_classes
                
                if config.enable_confidence_calibration:
                    # Temperature scaling parameter
                    self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
                    
                    # Platt scaling parameters (optional enhancement)
                    self.platt_a = nn.Parameter(torch.ones(num_classes))
                    self.platt_b = nn.Parameter(torch.zeros(num_classes))
                    
                    # Uncertainty estimation head
                    if config.enable_uncertainty_estimation:
                        self.uncertainty_head = nn.Sequential(
                            nn.Linear(num_classes, num_classes * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(num_classes * 2, num_classes)
                        )
            
            def forward(self, logits: torch.Tensor, 
                       return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
                """
                Apply confidence calibration to logits
                
                Args:
                    logits: [batch_size, num_classes]
                    return_uncertainty: Whether to return uncertainty estimates
                    
                Returns:
                    Dictionary with calibrated outputs
                """
                
                outputs = {'logits': logits}
                
                if not self.config.enable_confidence_calibration:
                    outputs['probabilities'] = F.softmax(logits, dim=-1)
                    outputs['confidence'] = torch.max(outputs['probabilities'], dim=-1)[0]
                    return outputs
                
                # Temperature scaling
                calibrated_logits = logits / self.temperature
                
                # Platt scaling (optional enhancement)
                platt_scaled = calibrated_logits * self.platt_a.unsqueeze(0) + self.platt_b.unsqueeze(0)
                
                # Compute calibrated probabilities
                calibrated_probs = F.softmax(platt_scaled, dim=-1)
                
                # Confidence scores
                confidence_scores = torch.max(calibrated_probs, dim=-1)[0]
                
                outputs.update({
                    'calibrated_logits': calibrated_logits,
                    'platt_scaled_logits': platt_scaled,
                    'probabilities': calibrated_probs,
                    'confidence': confidence_scores,
                    'temperature': self.temperature.item()
                })
                
                # Uncertainty estimation
                if return_uncertainty and self.config.enable_uncertainty_estimation:
                    uncertainty_logits = self.uncertainty_head(logits)
                    uncertainty_scores = torch.sigmoid(uncertainty_logits)
                    
                    # Epistemic uncertainty (model uncertainty)
                    epistemic_uncertainty = torch.mean(uncertainty_scores, dim=-1)
                    
                    # Aleatoric uncertainty (data uncertainty)
                    entropy = -torch.sum(calibrated_probs * torch.log(calibrated_probs + 1e-8), dim=-1)
                    aleatoric_uncertainty = entropy / np.log(self.num_classes)  # Normalized entropy
                    
                    outputs.update({
                        'epistemic_uncertainty': epistemic_uncertainty,
                        'aleatoric_uncertainty': aleatoric_uncertainty,
                        'total_uncertainty': epistemic_uncertainty + aleatoric_uncertainty
                    })
                
                return outputs
            
            def calibration_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                """Compute calibration loss for training"""
                
                if not self.config.enable_confidence_calibration:
                    return torch.tensor(0.0, device=logits.device)
                
                # Temperature scaling loss
                calibrated_logits = logits / self.temperature
                temp_loss = F.cross_entropy(calibrated_logits, targets)
                
                # Regularization to prevent extreme temperatures
                temp_reg = self.config.calibration_regularization * torch.abs(self.temperature - 1.0)
                
                return temp_loss + temp_reg
        
        return ConfidenceCalibrationModule(num_classes, self.config)


class EmailFeatureEnhancer:
    """Main class for enhancing EmailTRM with email-specific features"""
    
    def __init__(self, config: Optional[EmailFeatureConfig] = None):
        self.config = config or EmailFeatureConfig()
        
        self.structure_enhancer = EmailStructureEnhancer(self.config)
        self.attention_enhancer = HierarchicalAttentionEnhancer(self.config)
        self.confidence_calibrator = ConfidenceCalibrator(self.config)
        
    def enhance_email_trm(self, model: EmailTRM) -> EmailTRM:
        """
        Enhance EmailTRM model with additional email-specific features
        
        Args:
            model: Base EmailTRM model
            
        Returns:
            Enhanced EmailTRM model
        """
        
        logger.info("Enhancing EmailTRM with email-specific features")
        
        # Get model configuration
        model_config = model.config
        
        # Add enhanced structure embeddings
        if self.config.enable_structure_embeddings:
            enhanced_structure_emb = self.structure_enhancer.create_enhanced_structure_embeddings(
                model_config.vocab_size, model_config.hidden_size
            )
            
            # Replace or enhance existing structure embedding
            if hasattr(model.model.lm_head, 'structure_embedding'):
                model.model.lm_head.enhanced_structure_embedding = enhanced_structure_emb
                logger.info("Added enhanced structure embeddings")
        
        # Add hierarchical attention pooling
        if self.config.enable_hierarchical_attention:
            hierarchical_pooling = self.attention_enhancer.create_hierarchical_attention_pooling(
                model_config.hidden_size
            )
            
            # Replace existing pooling in classification head
            if hasattr(model.model.lm_head, 'pooling'):
                model.model.lm_head.hierarchical_pooling = hierarchical_pooling
                logger.info("Added hierarchical attention pooling")
        
        # Add confidence calibration
        if self.config.enable_confidence_calibration:
            calibration_module = self.confidence_calibrator.create_calibration_module(
                model_config.num_email_categories
            )
            
            model.confidence_calibrator = calibration_module
            logger.info("Added confidence calibration module")
        
        # Enhance forward pass
        original_forward = model.forward
        
        def enhanced_forward(self, inputs, labels=None, return_enhanced_outputs=False, **kwargs):
            """Enhanced forward pass with additional features"""
            
            # Call original forward pass
            outputs = original_forward(inputs, labels=labels, **kwargs)
            
            # Apply confidence calibration if available
            if hasattr(self, 'confidence_calibrator') and 'logits' in outputs:
                calibrated_outputs = self.confidence_calibrator(
                    outputs['logits'], 
                    return_uncertainty=return_enhanced_outputs
                )
                outputs.update(calibrated_outputs)
            
            # Add enhanced outputs if requested
            if return_enhanced_outputs:
                enhanced_outputs = self._compute_enhanced_outputs(inputs, outputs)
                outputs.update(enhanced_outputs)
            
            return outputs
        
        # Bind enhanced forward method
        model.forward = enhanced_forward.__get__(model, EmailTRM)
        
        # Add enhanced output computation method
        def _compute_enhanced_outputs(self, inputs, base_outputs):
            """Compute enhanced outputs for interpretability"""
            
            enhanced = {}
            
            # Attention visualization
            if self.config.enable_attention_visualization:
                # This would require access to attention weights from the model
                # Implementation depends on model architecture details
                enhanced['attention_weights'] = {}
            
            # Prediction explanations
            if self.config.enable_prediction_explanations:
                # Compute feature importance scores
                if 'probabilities' in base_outputs:
                    probs = base_outputs['probabilities']
                    predictions = torch.argmax(probs, dim=-1)
                    
                    enhanced['predictions'] = predictions
                    enhanced['prediction_explanations'] = self._generate_explanations(
                        inputs, probs, predictions
                    )
            
            return enhanced
        
        model._compute_enhanced_outputs = _compute_enhanced_outputs.__get__(model, EmailTRM)
        
        logger.info("EmailTRM enhancement completed")
        return model
    
    def _generate_explanations(self, inputs: torch.Tensor, 
                             probabilities: torch.Tensor, 
                             predictions: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate prediction explanations"""
        
        explanations = []
        
        for i in range(inputs.size(0)):
            input_seq = inputs[i]
            probs = probabilities[i]
            pred = predictions[i]
            
            explanation = {
                'predicted_category': pred.item(),
                'confidence': torch.max(probs).item(),
                'probability_distribution': probs.tolist(),
                'key_features': self._extract_key_features(input_seq, probs),
                'reasoning_summary': self._generate_reasoning_summary(input_seq, pred, probs)
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _extract_key_features(self, input_seq: torch.Tensor, 
                            probabilities: torch.Tensor) -> List[str]:
        """Extract key features that influenced the prediction"""
        
        # This is a simplified implementation
        # In practice, you'd use gradient-based or attention-based methods
        
        key_features = []
        
        # Check for structure tokens
        structure_tokens = {3: 'subject', 4: 'body', 5: 'from', 6: 'to'}
        
        for token_id, feature_name in structure_tokens.items():
            if token_id in input_seq:
                key_features.append(f"{feature_name}_present")
        
        return key_features
    
    def _generate_reasoning_summary(self, input_seq: torch.Tensor, 
                                  prediction: torch.Tensor, 
                                  probabilities: torch.Tensor) -> str:
        """Generate human-readable reasoning summary"""
        
        pred_idx = prediction.item()
        confidence = torch.max(probabilities).item()
        
        category_names = [
            'Newsletter', 'Work', 'Personal', 'Spam', 'Promotional',
            'Social', 'Finance', 'Travel', 'Shopping', 'Other'
        ]
        
        pred_category = category_names[pred_idx] if pred_idx < len(category_names) else 'Unknown'
        
        summary = f"Classified as '{pred_category}' with {confidence:.1%} confidence. "
        
        # Add reasoning based on structure
        if 3 in input_seq:  # subject token
            summary += "Subject content was a key factor. "
        if 5 in input_seq:  # from token
            summary += "Sender information influenced the decision. "
        
        return summary
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of enabled features"""
        
        return {
            'structure_embeddings': self.config.enable_structure_embeddings,
            'hierarchical_attention': self.config.enable_hierarchical_attention,
            'confidence_calibration': self.config.enable_confidence_calibration,
            'category_embeddings': self.config.enable_category_embeddings,
            'prediction_explanations': self.config.enable_prediction_explanations,
            'uncertainty_estimation': self.config.enable_uncertainty_estimation,
            'attention_visualization': self.config.enable_attention_visualization,
            'pooling_strategy': self.config.pooling_strategy,
            'attention_weights': {
                'subject': self.config.subject_attention_weight,
                'sender': self.config.sender_attention_weight,
                'body': self.config.body_attention_weight,
                'metadata': self.config.metadata_attention_weight
            }
        }


# Convenience functions
def enhance_email_trm_for_production(model: EmailTRM, 
                                   feature_config: Optional[EmailFeatureConfig] = None) -> EmailTRM:
    """
    Enhance EmailTRM model for production use with email-specific features
    
    Args:
        model: Base EmailTRM model
        feature_config: Feature configuration (uses defaults if None)
        
    Returns:
        Enhanced EmailTRM model
    """
    
    enhancer = EmailFeatureEnhancer(feature_config)
    enhanced_model = enhancer.enhance_email_trm(model)
    
    logger.info("EmailTRM enhanced for production use")
    return enhanced_model


def create_production_feature_config(enable_all: bool = True) -> EmailFeatureConfig:
    """
    Create production-ready feature configuration
    
    Args:
        enable_all: Whether to enable all features
        
    Returns:
        EmailFeatureConfig for production
    """
    
    if enable_all:
        return EmailFeatureConfig(
            enable_structure_embeddings=True,
            enable_hierarchical_attention=True,
            enable_confidence_calibration=True,
            enable_category_embeddings=True,
            enable_prediction_explanations=True,
            enable_uncertainty_estimation=True,
            enable_attention_visualization=True,
            pooling_strategy="hierarchical",
            subject_attention_weight=2.5,
            sender_attention_weight=1.8,
            body_attention_weight=1.0,
            metadata_attention_weight=0.9
        )
    else:
        # Minimal configuration for resource-constrained environments
        return EmailFeatureConfig(
            enable_structure_embeddings=True,
            enable_hierarchical_attention=True,
            enable_confidence_calibration=True,
            enable_category_embeddings=False,
            enable_prediction_explanations=False,
            enable_uncertainty_estimation=False,
            enable_attention_visualization=False,
            pooling_strategy="weighted"
        )


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test feature enhancement
    from models.recursive_reasoning.trm_email import EmailTRMConfig, EmailTRM
    
    # Create base model
    config = EmailTRMConfig(vocab_size=5000, num_email_categories=10)
    model = EmailTRM(config)
    
    # Create feature config
    feature_config = create_production_feature_config(enable_all=True)
    
    # Enhance model
    enhanced_model = enhance_email_trm_for_production(model, feature_config)
    
    print("Model enhanced successfully!")
    
    # Test enhanced forward pass
    batch_size = 2
    seq_len = 128
    
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.num_email_categories, (batch_size,))
    
    with torch.no_grad():
        outputs = enhanced_model(inputs, labels=labels, return_enhanced_outputs=True)
    
    print(f"Enhanced outputs available: {list(outputs.keys())}")
    if 'confidence' in outputs:
        print(f"Confidence scores: {outputs['confidence']}")
    if 'temperature' in outputs:
        print(f"Calibration temperature: {outputs['temperature']:.3f}")