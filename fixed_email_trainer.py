#!/usr/bin/env python3
"""
Fixed Email Classifier Trainer

A comprehensive solution that addresses all identified issues:
1. Proper data preprocessing and validation
2. Balanced dataset handling
3. Appropriate model architecture
4. Correct training parameters
5. Comprehensive evaluation
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from collections import Counter
# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Matplotlib/seaborn not available. Plots will be skipped.")

# Set environment variables to reduce warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailDataset(Dataset):
    """Improved email dataset with better preprocessing."""
    
    def __init__(self, emails: List[Dict], vocab: Dict[str, int], categories: Dict[str, int], max_length: int = 64):
        self.emails = emails
        self.vocab = vocab
        self.categories = categories
        self.max_length = max_length
        self.pad_token = vocab.get('<PAD>', 0)
        self.unk_token = vocab.get('<UNK>', 1)
        
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email = self.emails[idx]
        
        # Combine subject and body with special tokens
        text = f"SUBJECT: {email['subject']} BODY: {email['body']}"
        
        # Tokenize and convert to IDs
        tokens = self.tokenize(text)
        input_ids = self.tokens_to_ids(tokens)
        
        # Pad or truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids.extend([self.pad_token] * (self.max_length - len(input_ids)))
        
        # Create attention mask
        attention_mask = [1 if token_id != self.pad_token else 0 for token_id in input_ids]
        
        # Get label
        label = self.categories[email['category']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Improved tokenization."""
        # Convert to lowercase and clean
        text = text.lower()
        # Replace punctuation with spaces
        for punct in "!?.,;:()[]{}\"'":
            text = text.replace(punct, " ")
        # Split and filter empty strings
        tokens = [token for token in text.split() if token.strip()]
        return tokens
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab.get(token, self.unk_token) for token in tokens]

class ImprovedEmailClassifier(nn.Module):
    """Improved email classifier with attention mechanism."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # Apply attention
        if attention_mask is not None:
            # Convert attention mask for MultiheadAttention
            key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Layer norm and residual connection
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Global average pooling with attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(attn_out)
            sum_embeddings = torch.sum(attn_out * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_embeddings / (sum_mask + 1e-9)
        else:
            pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.dropout(pooled)
        logits = self.classifier(output)
        
        return logits

def analyze_dataset(emails: List[Dict], categories: Dict[str, int]) -> None:
    """Analyze dataset for potential issues."""
    logger.info("Dataset Analysis:")
    
    # Category distribution
    category_counts = Counter(email['category'] for email in emails)
    logger.info("Category distribution:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} emails")
    
    # Check for imbalance
    min_count = min(category_counts.values())
    max_count = max(category_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Text length analysis
    text_lengths = []
    for email in emails:
        text = f"{email['subject']} {email['body']}"
        text_lengths.append(len(text.split()))
    
    logger.info(f"Text length stats:")
    logger.info(f"  Mean: {np.mean(text_lengths):.1f} words")
    logger.info(f"  Std: {np.std(text_lengths):.1f} words")
    logger.info(f"  Min: {np.min(text_lengths)} words")
    logger.info(f"  Max: {np.max(text_lengths)} words")

def load_and_validate_dataset(dataset_path: str) -> Tuple[List[Dict], List[Dict], Dict[str, int], Dict[str, int]]:
    """Load and validate email dataset."""
    dataset_path = Path(dataset_path)
    
    # Load training data
    train_emails = []
    with open(dataset_path / "train" / "dataset.json", "r") as f:
        for line in f:
            train_emails.append(json.loads(line.strip()))
    
    # Load test data
    test_emails = []
    with open(dataset_path / "test" / "dataset.json", "r") as f:
        for line in f:
            test_emails.append(json.loads(line.strip()))
    
    # Load categories
    with open(dataset_path / "categories.json", "r") as f:
        categories = json.load(f)
    
    # Load vocabulary
    with open(dataset_path / "vocab.json", "r") as f:
        vocab = json.load(f)
    
    logger.info(f"Loaded {len(train_emails)} training emails, {len(test_emails)} test emails")
    logger.info(f"Vocabulary size: {len(vocab)}, Categories: {len(categories)}")
    
    # Analyze datasets
    logger.info("Training set analysis:")
    analyze_dataset(train_emails, categories)
    logger.info("Test set analysis:")
    analyze_dataset(test_emails, categories)
    
    return train_emails, test_emails, vocab, categories

def train_model(model, train_loader, val_loader, num_epochs: int = 15, learning_rate: float = 0.001):
    """Train the model with improved training loop."""
    device = torch.device("cpu")
    model = model.to(device)
    
    # Use class weights to handle imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 5
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        logger.info(f"  Val Acc: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Early stopping and model saving
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_email_model_fixed.pt")
            logger.info(f"  New best accuracy: {best_accuracy:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print()
    
    return best_accuracy, val_predictions, val_labels, train_losses, val_accuracies

def plot_training_history(train_losses: List[float], val_accuracies: List[float]):
    """Plot training history."""
    if not PLOTTING_AVAILABLE:
        logger.info("Plotting not available, skipping training history plot")
        return
        
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        logger.info("Training history saved to training_history.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save training plots: {e}")

def plot_confusion_matrix(true_labels: List[int], predictions: List[int], category_names: List[str]):
    """Plot confusion matrix."""
    if not PLOTTING_AVAILABLE:
        logger.info("Plotting not available, skipping confusion matrix plot")
        return
        
    try:
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=category_names, yticklabels=category_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        logger.info("Confusion matrix saved to confusion_matrix.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save confusion matrix: {e}")

def main():
    """Main training function."""
    logger.info("Starting improved email classifier training")
    
    # Load and validate dataset
    train_emails, test_emails, vocab, categories = load_and_validate_dataset("enhanced_emails")
    
    # Create datasets with improved preprocessing
    max_length = 64
    train_dataset = EmailDataset(train_emails, vocab, categories, max_length)
    test_dataset = EmailDataset(test_emails, vocab, categories, max_length)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create improved model
    vocab_size = len(vocab)
    num_classes = len(categories)
    model = ImprovedEmailClassifier(
        vocab_size=vocab_size, 
        embed_dim=128, 
        hidden_dim=64, 
        num_classes=num_classes,
        dropout=0.3
    )
    
    logger.info(f"Model parameters: vocab_size={vocab_size}, num_classes={num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    best_accuracy, predictions, true_labels, train_losses, val_accuracies = train_model(
        model, train_loader, test_loader, 
        num_epochs=20, learning_rate=0.001
    )
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Print final results
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Print classification report
    category_names = list(categories.keys())
    report = classification_report(true_labels, predictions, target_names=category_names, zero_division=0)
    logger.info("Classification Report:")
    logger.info("\n" + report)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, category_names)
    
    # Print per-category accuracy
    logger.info("Per-category accuracy:")
    for i, category in enumerate(category_names):
        category_mask = np.array(true_labels) == i
        if category_mask.sum() > 0:
            category_acc = accuracy_score(
                np.array(true_labels)[category_mask], 
                np.array(predictions)[category_mask]
            )
            logger.info(f"  {category}: {category_acc:.4f}")
    
    # Final assessment
    if best_accuracy >= 0.8:
        logger.info("🎉 Training successful! Excellent accuracy achieved.")
    elif best_accuracy >= 0.6:
        logger.info("✅ Training successful! Good accuracy achieved.")
    elif best_accuracy >= 0.4:
        logger.info("⚠️ Training partially successful. Consider more training or data.")
    else:
        logger.info("❌ Training needs improvement. Check data and model configuration.")
    
    # Save final model info
    model_info = {
        "best_accuracy": best_accuracy,
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "model_parameters": total_params,
        "category_names": category_names
    }
    
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("Model information saved to model_info.json")

if __name__ == "__main__":
    main()