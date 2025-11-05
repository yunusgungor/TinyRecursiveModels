"""
TRM (Tiny Recursive Reasoning Model) adapted for Email Classification

This module adapts the TRM architecture for email classification tasks,
modifying the output layer and training objectives for multi-class classification.
Enhanced with email-specific features and optimizations.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1InnerCarry
)
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


class EmailTRMConfig(TinyRecursiveReasoningModel_ACTV1Config):
    """Configuration for Email Classification TRM"""
    
    # Email-specific parameters
    num_email_categories: int = 10  # Number of email categories
    use_category_embedding: bool = True  # Whether to use category embeddings
    classification_dropout: float = 0.1  # Dropout for classification head
    
    # Email structure awareness
    use_email_structure: bool = True  # Use email structure tokens (subject, body, etc.)
    email_structure_dim: int = 64  # Dimension for email structure embeddings
    
    # Attention mechanisms for emails
    use_hierarchical_attention: bool = True  # Use hierarchical attention for email parts
    subject_attention_weight: float = 2.0  # Weight for subject attention
    sender_attention_weight: float = 1.5  # Weight for sender attention
    
    # Email-specific optimizations
    use_email_pooling: bool = True  # Use email-aware pooling strategy
    pooling_strategy: str = "weighted"  # "mean", "max", "weighted", "attention"
    
    # Override some defaults for email classification
    H_cycles: int = 2  # Fewer cycles for faster inference
    L_cycles: int = 4  # Reduced cycles for email classification
    halt_max_steps: int = 8  # Fewer steps needed for classification
    
    # Email token IDs (should match dataset preprocessing)
    pad_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    subject_id: int = 3
    body_id: int = 4
    from_id: int = 5
    to_id: int = 6


class EmailStructureEmbedding(nn.Module):
    """Email structure-aware embeddings"""
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        if config.use_email_structure:
            # Structure type embeddings (subject, body, from, to)
            self.structure_embeddings = nn.Embedding(7, config.email_structure_dim)  # 7 special tokens
            
            # Project structure embeddings to hidden size
            self.structure_proj = nn.Linear(config.email_structure_dim, config.hidden_size)
            
            # Initialize
            nn.init.normal_(self.structure_embeddings.weight, std=0.02)
            nn.init.xavier_uniform_(self.structure_proj.weight)
    
    def forward(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Generate structure embeddings based on special tokens
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            structure_emb: [batch_size, seq_len, hidden_size] or None
        """
        if not self.config.use_email_structure:
            return None
        
        # Create structure type tensor based on special tokens
        structure_types = torch.zeros_like(input_ids)
        
        # Map special tokens to structure types
        structure_types = torch.where(input_ids == self.config.subject_id, 3, structure_types)
        structure_types = torch.where(input_ids == self.config.body_id, 4, structure_types)
        structure_types = torch.where(input_ids == self.config.from_id, 5, structure_types)
        structure_types = torch.where(input_ids == self.config.to_id, 6, structure_types)
        
        # Get embeddings and project
        structure_emb = self.structure_embeddings(structure_types)
        structure_emb = self.structure_proj(structure_emb)
        
        return structure_emb


class EmailAttentionPooling(nn.Module):
    """Email-aware attention pooling"""
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        if config.use_hierarchical_attention:
            # Attention weights for different email parts
            self.attention_weights = nn.Parameter(torch.ones(7))  # For each special token type
            
            # Learnable attention mechanism
            self.attention_proj = nn.Linear(config.hidden_size, 1)
            
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Email-aware pooling with hierarchical attention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled: [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.config.pooling_strategy == "mean":
            # Simple mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states * mask_expanded
                pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = hidden_states.mean(dim=1)
                
        elif self.config.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states.masked_fill(~mask_expanded.bool(), float('-inf'))
            pooled = hidden_states.max(dim=1)[0]
            
        elif self.config.pooling_strategy == "weighted" and self.config.use_hierarchical_attention:
            # Weighted pooling based on email structure
            weights = torch.ones_like(input_ids, dtype=torch.float)
            
            # Apply structure-specific weights
            weights = torch.where(input_ids == self.config.subject_id, 
                                self.config.subject_attention_weight, weights)
            weights = torch.where(input_ids == self.config.from_id, 
                                self.config.sender_attention_weight, weights)
            
            if attention_mask is not None:
                weights = weights * attention_mask.float()
            
            # Normalize weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Apply weights
            weights_expanded = weights.unsqueeze(-1).expand_as(hidden_states)
            pooled = (hidden_states * weights_expanded).sum(dim=1)
            
        elif self.config.pooling_strategy == "attention":
            # Learnable attention pooling
            attention_scores = self.attention_proj(hidden_states).squeeze(-1)  # [batch_size, seq_len]
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
            
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = (hidden_states * attention_weights.unsqueeze(-1)).sum(dim=1)
            
        else:
            # Fallback to mean pooling
            pooled = hidden_states.mean(dim=1)
        
        return pooled


class EmailClassificationHead(nn.Module):
    """Enhanced classification head for email categorization"""
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        # Email structure embeddings
        self.structure_embedding = EmailStructureEmbedding(config)
        
        # Email-aware pooling
        self.pooling = EmailAttentionPooling(config)
        
        # Classification layers with residual connections
        self.dropout = nn.Dropout(config.classification_dropout)
        
        # Multi-layer classification head
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_email_categories)
        
        # Optional category embeddings for contrastive learning
        if config.use_category_embedding:
            self.category_embeddings = nn.Embedding(
                config.num_email_categories, 
                config.hidden_size
            )
            self.category_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights"""
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.zeros_(self.pre_classifier.bias)
        
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        if self.config.use_category_embedding:
            nn.init.normal_(self.category_embeddings.weight, std=0.02)
            nn.init.xavier_uniform_(self.category_proj.weight)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] optional
            
        Returns:
            Dictionary with logits and additional outputs
        """
        
        # Add structure embeddings if enabled
        if self.config.use_email_structure:
            structure_emb = self.structure_embedding(input_ids)
            if structure_emb is not None:
                hidden_states = hidden_states + structure_emb
        
        # Email-aware pooling
        pooled = self.pooling(hidden_states, input_ids, attention_mask)
        
        # Layer normalization
        pooled = self.layer_norm(pooled)
        
        # Pre-classification layer with residual connection
        pre_logits = self.pre_classifier(pooled)
        pre_logits = F.gelu(pre_logits)
        pre_logits = self.dropout(pre_logits)
        
        # Add residual connection
        pooled = pooled + pre_logits
        
        # Final classification
        logits = self.classifier(pooled)
        
        outputs = {"logits": logits, "pooled_output": pooled}
        
        # Category embeddings for contrastive learning
        if self.config.use_category_embedding:
            category_embs = self.category_embeddings.weight  # [num_categories, hidden_size]
            category_embs = self.category_proj(category_embs)
            
            # Compute similarity scores
            pooled_norm = F.normalize(pooled, p=2, dim=1)
            category_norm = F.normalize(category_embs, p=2, dim=1)
            similarity_scores = torch.matmul(pooled_norm, category_norm.t())
            
            outputs["similarity_scores"] = similarity_scores
            outputs["category_embeddings"] = category_embs
        
        return outputs


class EmailTRM_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """Inner TRM model adapted for email classification with enhanced features"""
    
    def __init__(self, config: EmailTRMConfig):
        # Initialize parent with modified config
        super().__init__(config)
        self.config = config
        
        # Replace language modeling head with enhanced classification head
        self.lm_head = EmailClassificationHead(config)
        
        # Enhanced Q-learning heads for better halting decisions
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        
        # Email-specific enhancements
        if config.use_email_structure:
            # Additional processing for email structure
            self.email_structure_processor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        
        # Adaptive reasoning controller
        self.reasoning_controller = nn.Linear(config.hidden_size, 1)
        
        # Initialize new components
        self._init_email_components()
    
    def _init_email_components(self):
        """Initialize email-specific components"""
        # Q head special init for email classification
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-3)  # Less aggressive halting for emails
        
        if hasattr(self, 'email_structure_processor'):
            for module in self.email_structure_processor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.reasoning_controller.weight)
        nn.init.zeros_(self.reasoning_controller.bias)
    
    def _enhanced_input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """Enhanced input embeddings with email-specific processing"""
        
        # Get base embeddings
        embeddings = self._input_embeddings(inputs, puzzle_identifiers)
        
        # Email structure processing
        if self.config.use_email_structure and hasattr(self, 'email_structure_processor'):
            # Identify email structure tokens
            structure_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for token_id in [self.config.subject_id, self.config.body_id, 
                           self.config.from_id, self.config.to_id]:
                structure_mask |= (inputs == token_id)
            
            # Apply structure processing to relevant tokens
            if structure_mask.any():
                structure_embeddings = self.email_structure_processor(embeddings)
                embeddings = torch.where(structure_mask.unsqueeze(-1), 
                                       structure_embeddings, embeddings)
        
        return embeddings
    
    def _adaptive_reasoning_cycles(self, hidden_states: torch.Tensor, cycle: int) -> torch.Tensor:
        """Determine if more reasoning cycles are needed based on content complexity"""
        
        # Compute complexity score based on hidden states
        complexity_score = self.reasoning_controller(hidden_states.mean(dim=1))  # [batch_size, 1]
        complexity_score = torch.sigmoid(complexity_score).squeeze(-1)  # [batch_size]
        
        # Adaptive cycle decision
        # More complex emails (higher score) get more cycles
        continue_reasoning = complexity_score > (0.5 - 0.1 * cycle)  # Decreasing threshold
        
        return continue_reasoning
    
    def forward(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        cycle: int = 0,
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, TinyRecursiveReasoningModel_ACTV1Carry]:
        """
        Enhanced forward pass for email classification
        
        Args:
            inputs: [batch_size, seq_len] - tokenized email sequences
            puzzle_identifiers: [batch_size] - email identifiers
            carry: Model state carry
            cycle: Current reasoning cycle
            
        Returns:
            classification_outputs: Dictionary with logits and additional outputs
            q_halt_logits: [batch_size] - halting decisions
            new_carry: Updated model state
        """
        
        batch_size, seq_len = inputs.shape
        
        # Enhanced input embeddings
        x = self._enhanced_input_embeddings(inputs, puzzle_identifiers)
        
        # Apply reasoning layers with carry state
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # L-level reasoning
        for _ in range(self.config.L_cycles):
            x = self.L_level(x, input_injection=torch.zeros_like(x), **seq_info)
        
        # Create attention mask (non-padding tokens)
        attention_mask = (inputs != self.config.pad_id).float()
        
        # Enhanced classification head
        classification_outputs = self.lm_head(x, inputs, attention_mask)
        
        # Adaptive Q-learning for halting
        pooled_repr = classification_outputs["pooled_output"]
        q_halt_logits = self.q_head(pooled_repr)
        
        # Add adaptive reasoning signal
        if cycle > 0:
            continue_reasoning = self._adaptive_reasoning_cycles(x, cycle)
            # Modify halt logits based on reasoning needs
            reasoning_bonus = continue_reasoning.float().unsqueeze(-1) * 2.0  # Encourage continuation
            q_halt_logits = q_halt_logits + torch.cat([-reasoning_bonus, reasoning_bonus], dim=-1)
        
        return classification_outputs, q_halt_logits, carry


class EmailTRM(nn.Module):
    """
    Enhanced Email Classification TRM Model
    
    Adapts the Tiny Recursive Reasoning Model for email classification tasks.
    Uses recursive reasoning to progressively refine email category predictions.
    Enhanced with email-specific features and optimizations.
    """
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        # Core TRM model
        self.model = EmailTRM_Inner(config)
        
        # Enhanced loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Contrastive loss for category embeddings
        if config.use_category_embedding:
            self.contrastive_criterion = nn.CosineEmbeddingLoss(margin=0.1)
        
        # Confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Email-specific metrics tracking
        self.register_buffer('category_counts', torch.zeros(config.num_email_categories))
        self.register_buffer('correct_predictions', torch.zeros(config.num_email_categories))
    
    def empty_carry(self, batch_size: int, device: torch.device) -> TinyRecursiveReasoningModel_ACTV1Carry:
        """Create empty carry state"""
        inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self.model.H_init.unsqueeze(0).expand(batch_size, -1, -1).to(device),
            z_L=self.model.L_init.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        )
        
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device),
            current_data={}
        )
    
    def _compute_enhanced_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute enhanced loss with multiple components"""
        
        logits = outputs["logits"]
        
        # Main classification loss
        classification_loss = self.criterion(logits, labels)
        
        # Halt regularization loss
        halt_loss = 0.0
        if "halt_logits" in outputs:
            halt_probs = torch.sigmoid(outputs["halt_logits"])
            # Encourage efficient halting (not too early, not too late)
            halt_loss = torch.mean(torch.abs(halt_probs - 0.6))  # Target ~60% halt probability
        
        # Contrastive loss for category embeddings
        contrastive_loss = 0.0
        if self.config.use_category_embedding and "similarity_scores" in outputs:
            # Use similarity scores for contrastive learning
            similarity_scores = outputs["similarity_scores"]
            target_similarities = torch.zeros_like(similarity_scores)
            target_similarities.scatter_(1, labels.unsqueeze(1), 1.0)
            
            # Contrastive loss encourages high similarity with correct category
            contrastive_loss = F.mse_loss(similarity_scores, target_similarities)
        
        # Confidence calibration loss
        calibration_loss = 0.0
        if self.training:
            # Temperature scaling for better calibration
            scaled_logits = logits / self.temperature
            calibration_loss = F.cross_entropy(scaled_logits, labels) - classification_loss
            calibration_loss = torch.abs(calibration_loss)
        
        # Combine losses
        total_loss = (classification_loss + 
                     0.01 * halt_loss + 
                     0.1 * contrastive_loss + 
                     0.05 * calibration_loss)
        
        return total_loss
    
    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        return_all_cycles: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with recursive reasoning for email classification
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            labels: [batch_size] - email category labels (optional)
            puzzle_identifiers: [batch_size] - email identifiers
            return_all_cycles: Whether to return outputs from all reasoning cycles
            
        Returns:
            Dictionary with logits, loss, and other outputs
        """
        
        batch_size, seq_len = inputs.shape
        device = inputs.device
        
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Initialize carry state
        carry = self.empty_carry(batch_size, device)
        
        # Recursive reasoning cycles with adaptive halting
        all_outputs = []
        all_halt_logits = []
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for cycle in range(self.config.H_cycles):
            # Forward pass through model
            cycle_outputs, halt_logits, carry = self.model(
                inputs=inputs,
                puzzle_identifiers=puzzle_identifiers,
                carry=carry,
                cycle=cycle
            )
            
            all_outputs.append(cycle_outputs)
            all_halt_logits.append(halt_logits)
            
            # Adaptive halting mechanism
            if cycle < self.config.H_cycles - 1:
                halt_probs = torch.sigmoid(halt_logits[:, 1])  # Use second output for halt decision
                
                # Dynamic halting threshold based on confidence
                confidence = torch.max(F.softmax(cycle_outputs["logits"], dim=-1), dim=-1)[0]
                adaptive_threshold = 0.5 + 0.3 * confidence  # Higher confidence -> easier to halt
                
                should_halt = (halt_probs > adaptive_threshold) | (~active_mask)
                active_mask = active_mask & (~should_halt)
                
                # If all samples have halted, break early
                if not active_mask.any():
                    break
        
        # Use final cycle predictions (or last available for each sample)
        final_outputs = all_outputs[-1]
        final_halt_logits = all_halt_logits[-1]
        
        # Prepare return dictionary
        result = {
            "logits": final_outputs["logits"],
            "halt_logits": final_halt_logits,
            "num_cycles": len(all_outputs),
            "pooled_output": final_outputs.get("pooled_output"),
        }
        
        # Add similarity scores if available
        if "similarity_scores" in final_outputs:
            result["similarity_scores"] = final_outputs["similarity_scores"]
        
        # Temperature-scaled logits for better calibration
        result["calibrated_logits"] = final_outputs["logits"] / self.temperature
        
        # Compute loss if labels provided
        if labels is not None:
            result["loss"] = self._compute_enhanced_loss(final_outputs, labels)
            
            # Update category statistics for monitoring
            if self.training:
                predictions = torch.argmax(final_outputs["logits"], dim=-1)
                for i in range(self.config.num_email_categories):
                    category_mask = (labels == i)
                    if category_mask.any():
                        self.category_counts[i] += category_mask.sum()
                        self.correct_predictions[i] += ((predictions == labels) & category_mask).sum()
        
        # Return all cycles if requested
        if return_all_cycles:
            result["all_logits"] = torch.stack([out["logits"] for out in all_outputs])
            result["all_halt_logits"] = torch.stack(all_halt_logits)
        
        return result
    
    def predict(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None, 
                return_confidence: bool = False) -> torch.Tensor:
        """
        Predict email categories with optional confidence scores
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            puzzle_identifiers: [batch_size] - email identifiers
            return_confidence: Whether to return confidence scores
            
        Returns:
            predictions: [batch_size] - predicted category indices
            confidences: [batch_size] - confidence scores (if return_confidence=True)
        """
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, puzzle_identifiers=puzzle_identifiers)
            
            # Use calibrated logits for better confidence estimation
            probs = F.softmax(outputs["calibrated_logits"], dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            if return_confidence:
                confidences = torch.max(probs, dim=-1)[0]
                return predictions, confidences
            else:
                return predictions
    
    def get_category_performance(self) -> Dict[str, float]:
        """Get per-category performance statistics"""
        
        accuracies = {}
        for i in range(self.config.num_email_categories):
            if self.category_counts[i] > 0:
                accuracy = (self.correct_predictions[i] / self.category_counts[i]).item()
                accuracies[f"category_{i}_accuracy"] = accuracy
            else:
                accuracies[f"category_{i}_accuracy"] = 0.0
        
        return accuracies
    
    def reset_statistics(self):
        """Reset category performance statistics"""
        self.category_counts.zero_()
        self.correct_predictions.zero_()
    
    def get_reasoning_analysis(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Analyze reasoning process for interpretability
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            puzzle_identifiers: [batch_size] - email identifiers
            
        Returns:
            Dictionary with reasoning analysis
        """
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, puzzle_identifiers=puzzle_identifiers, return_all_cycles=True)
            
            analysis = {
                "num_cycles_used": outputs["num_cycles"],
                "final_confidence": torch.max(F.softmax(outputs["logits"], dim=-1), dim=-1)[0],
                "reasoning_progression": [],
            }
            
            # Analyze how predictions change across cycles
            if "all_logits" in outputs:
                all_probs = F.softmax(outputs["all_logits"], dim=-1)
                for cycle in range(outputs["num_cycles"]):
                    cycle_preds = torch.argmax(all_probs[cycle], dim=-1)
                    cycle_conf = torch.max(all_probs[cycle], dim=-1)[0]
                    
                    analysis["reasoning_progression"].append({
                        "cycle": cycle,
                        "predictions": cycle_preds,
                        "confidence": cycle_conf,
                        "prediction_stability": (cycle_preds == analysis.get("final_predictions", cycle_preds)).float().mean() if cycle > 0 else 1.0
                    })
                
                analysis["final_predictions"] = torch.argmax(outputs["logits"], dim=-1)
            
            return analysis


def create_email_trm_model(
    vocab_size: int,
    num_categories: int = 10,
    hidden_size: int = 512,
    num_layers: int = 2,
    **kwargs
) -> EmailTRM:
    """
    Create EmailTRM model with specified parameters
    
    Args:
        vocab_size: Size of vocabulary
        num_categories: Number of email categories
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        **kwargs: Additional config parameters
        
    Returns:
        EmailTRM model instance
    """
    
    config = EmailTRMConfig(
        vocab_size=vocab_size,
        num_email_categories=num_categories,
        hidden_size=hidden_size,
        L_layers=num_layers,
        **kwargs
    )
    
    return EmailTRM(config)


# Example usage and testing
if __name__ == "__main__":
    # Test model creation and forward pass
    config = EmailTRMConfig(
        vocab_size=5000,
        num_email_categories=10,
        hidden_size=256,
        L_layers=2,
        H_cycles=2,
        L_cycles=3
    )
    
    model = EmailTRM(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    inputs = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.num_email_categories, (batch_size,))
    
    outputs = model(inputs, labels=labels)
    
    print(f"Model created successfully!")
    print(f"Input shape: {inputs.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of reasoning cycles: {outputs['num_cycles']}")

def create_email_trm_model(vocab_size: int, num_categories: int = 10, **kwargs) -> EmailTRM:
    """
    Create EmailTRM model with specified configuration
    
    Args:
        vocab_size: Vocabulary size
        num_categories: Number of email categories
        **kwargs: Additional configuration parameters
        
    Returns:
        EmailTRM model instance
    """
    config = EmailTRMConfig(
        vocab_size=vocab_size,
        num_email_categories=num_categories,
        **kwargs
    )
    
    return EmailTRM(config)