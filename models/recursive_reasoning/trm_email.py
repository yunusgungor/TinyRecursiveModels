"""
TRM (Tiny Recursive Reasoning Model) adapted for Email Classification

This module adapts the TRM architecture for email classification tasks,
modifying the output layer and training objectives for multi-class classification.
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule
)
from models.common import CastedEmbedding, CastedSparseEmbedding
from models.layers import RMSNorm


@dataclass
class EmailTRMConfig(TinyRecursiveReasoningModel_ACTV1Config):
    """Configuration for Email Classification TRM"""
    
    # Email-specific parameters
    num_email_categories: int = 10  # Number of email categories
    use_category_embedding: bool = True  # Whether to use category embeddings
    classification_dropout: float = 0.1  # Dropout for classification head
    
    # Override some defaults for email classification
    H_cycles: int = 2  # Fewer cycles for faster inference
    L_cycles: int = 4  # Reduced cycles for email classification
    halt_max_steps: int = 8  # Fewer steps needed for classification


class EmailClassificationHead(nn.Module):
    """Classification head for email categorization"""
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        # Classification layers
        self.dropout = nn.Dropout(config.classification_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_email_categories)
        
        # Optional category embeddings for better representation
        if config.use_category_embedding:
            self.category_embeddings = nn.Embedding(
                config.num_email_categories, 
                config.hidden_size
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        if self.config.use_category_embedding:
            nn.init.normal_(self.category_embeddings.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] optional
            
        Returns:
            logits: [batch_size, num_categories]
        """
        
        # Pool sequence representations (mean pooling with attention mask)
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask_expanded
            pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Apply dropout and classification
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


class EmailTRM_Inner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """Inner TRM model adapted for email classification"""
    
    def __init__(self, config: EmailTRMConfig):
        # Initialize parent with modified config
        super().__init__(config)
        self.config = config
        
        # Replace language modeling head with classification head
        self.lm_head = EmailClassificationHead(config)
        
        # Keep Q-learning heads for ACT mechanism
        # self.q_head is inherited from parent
    
    def forward(
        self,
        inputs: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TinyRecursiveReasoningModel_ACTV1Carry]:
        """
        Forward pass adapted for email classification
        
        Args:
            inputs: [batch_size, seq_len] - tokenized email sequences
            puzzle_identifiers: [batch_size] - email identifiers
            carry: Model state carry
            
        Returns:
            classification_logits: [batch_size, num_categories]
            q_halt_logits: [batch_size] - halting decisions
            new_carry: Updated model state
        """
        
        batch_size, seq_len = inputs.shape
        
        # Input embeddings (inherited from parent)
        x = self._input_embeddings(inputs, puzzle_identifiers)
        
        # Apply reasoning layers
        for layer in self.layers:
            x = layer(x, carry.inner_carry)
        
        # Create attention mask (non-padding tokens)
        attention_mask = (inputs != self.config.pad_id).float()
        
        # Classification head
        classification_logits = self.lm_head(x, attention_mask)
        
        # Q-learning for halting (inherited)
        q_halt_logits = self.q_head(x.mean(dim=1))  # Pool for halt decision
        
        return classification_logits, q_halt_logits, carry


class EmailTRM(nn.Module):
    """
    Email Classification TRM Model
    
    Adapts the Tiny Recursive Reasoning Model for email classification tasks.
    Uses recursive reasoning to progressively refine email category predictions.
    """
    
    def __init__(self, config: EmailTRMConfig):
        super().__init__()
        self.config = config
        
        # Core TRM model
        self.model = EmailTRM_Inner(config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def empty_carry(self, batch_size: int, device: torch.device) -> TinyRecursiveReasoningModel_ACTV1Carry:
        """Create empty carry state"""
        return self.model.empty_carry(batch_size, device)
    
    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with recursive reasoning for email classification
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            labels: [batch_size] - email category labels (optional)
            puzzle_identifiers: [batch_size] - email identifiers
            
        Returns:
            Dictionary with logits, loss, and other outputs
        """
        
        batch_size, seq_len = inputs.shape
        device = inputs.device
        
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Initialize carry state
        carry = self.empty_carry(batch_size, device)
        
        # Recursive reasoning cycles
        all_logits = []
        all_halt_logits = []
        
        for cycle in range(self.config.H_cycles):
            # Forward pass through model
            logits, halt_logits, carry = self.model(
                inputs=inputs,
                puzzle_identifiers=puzzle_identifiers,
                carry=carry
            )
            
            all_logits.append(logits)
            all_halt_logits.append(halt_logits)
            
            # ACT halting mechanism (simplified for classification)
            if cycle < self.config.H_cycles - 1:
                halt_probs = torch.sigmoid(halt_logits)
                # Continue reasoning if halt probability is low
                continue_mask = halt_probs < 0.5
                if not continue_mask.any():
                    break
        
        # Use final cycle predictions
        final_logits = all_logits[-1]
        final_halt_logits = all_halt_logits[-1]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Classification loss
            classification_loss = self.criterion(final_logits, labels)
            
            # Optional: Add halt regularization loss
            halt_loss = torch.mean(torch.sigmoid(final_halt_logits))  # Encourage halting
            
            loss = classification_loss + 0.01 * halt_loss
        
        return {
            "logits": final_logits,
            "loss": loss,
            "halt_logits": final_halt_logits,
            "all_logits": torch.stack(all_logits),
            "num_cycles": len(all_logits)
        }
    
    def predict(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict email categories
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            puzzle_identifiers: [batch_size] - email identifiers
            
        Returns:
            predictions: [batch_size] - predicted category indices
        """
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, puzzle_identifiers=puzzle_identifiers)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        
        return predictions
    
    def get_attention_weights(self, inputs: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for interpretability
        
        Args:
            inputs: [batch_size, seq_len] - tokenized emails
            puzzle_identifiers: [batch_size] - email identifiers
            
        Returns:
            attention_weights: Attention weights from the model
        """
        
        # This would require modifications to extract attention weights
        # from the transformer layers - placeholder for now
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs, puzzle_identifiers=puzzle_identifiers)
        
        # Return dummy attention weights for now
        batch_size, seq_len = inputs.shape
        return torch.ones(batch_size, seq_len, device=inputs.device)


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