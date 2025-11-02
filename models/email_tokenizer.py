"""
Enhanced Email Tokenizer and Preprocessing

Specialized tokenizer for email content with structure awareness,
domain-specific preprocessing, and optimized vocabulary.
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F


class EmailTokenizer:
    """Enhanced tokenizer specifically designed for email content"""
    
    # Special tokens
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<EOS>": 1, 
        "<UNK>": 2,
        "<SUBJECT>": 3,
        "<BODY>": 4,
        "<FROM>": 5,
        "<TO>": 6,
        "<EMAIL>": 7,
        "<PHONE>": 8,
        "<URL>": 9,
        "<NUMBER>": 10,
        "<DATE>": 11,
        "<TIME>": 12,
        "<CURRENCY>": 13
    }
    
    # Email-specific patterns
    EMAIL_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?|euros?)',
        'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
        'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    }
    
    # Common email domains for domain-specific processing
    COMMON_DOMAINS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
        'icloud.com', 'protonmail.com', 'company.com', 'business.com'
    }
    
    # Newsletter/marketing indicators
    NEWSLETTER_INDICATORS = {
        'unsubscribe', 'newsletter', 'marketing', 'promotional', 'offer',
        'deal', 'sale', 'discount', 'limited time', 'exclusive', 'subscribe'
    }
    
    # Work-related indicators
    WORK_INDICATORS = {
        'meeting', 'project', 'deadline', 'report', 'presentation', 'team',
        'client', 'proposal', 'budget', 'schedule', 'conference', 'urgent'
    }
    
    def __init__(self, vocab_size: int = 5000, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Initialize vocabulary with special tokens
        self.vocab = self.SPECIAL_TOKENS.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Compiled regex patterns for efficiency
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.EMAIL_PATTERNS.items()
        }
        
        # Category-specific vocabularies
        self.category_vocabs = defaultdict(set)
        
        # Statistics
        self.token_frequencies = Counter()
        self.category_token_frequencies = defaultdict(Counter)
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text with email-specific preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace email-specific patterns with special tokens
        for pattern_name, pattern in self.compiled_patterns.items():
            if pattern_name == 'email':
                text = pattern.sub('<EMAIL>', text)
            elif pattern_name == 'phone':
                text = pattern.sub('<PHONE>', text)
            elif pattern_name == 'url':
                text = pattern.sub('<URL>', text)
            elif pattern_name == 'currency':
                text = pattern.sub('<CURRENCY>', text)
            elif pattern_name == 'date':
                text = pattern.sub('<DATE>', text)
            elif pattern_name == 'time':
                text = pattern.sub('<TIME>', text)
            elif pattern_name == 'number':
                # Only replace standalone numbers, not those in words
                text = pattern.sub(lambda m: '<NUMBER>' if m.group().replace(',', '').replace('.', '').isdigit() else m.group(), text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-<>]', '', text)
        
        return text.strip()
    
    def _extract_domain_features(self, sender: str) -> List[str]:
        """Extract domain-specific features from sender"""
        features = []
        
        if '@' in sender:
            domain = sender.split('@')[-1].lower()
            
            # Common domain indicators
            if domain in self.COMMON_DOMAINS:
                features.append(f"domain_{domain.replace('.', '_')}")
            else:
                features.append("domain_other")
            
            # Business vs personal domain heuristics
            if any(biz in domain for biz in ['company', 'corp', 'inc', 'ltd', 'org']):
                features.append("business_domain")
            elif domain in {'gmail.com', 'yahoo.com', 'hotmail.com'}:
                features.append("personal_domain")
        
        return features
    
    def _extract_content_features(self, text: str, category: str = None) -> List[str]:
        """Extract content-based features"""
        features = []
        
        # Newsletter indicators
        newsletter_count = sum(1 for indicator in self.NEWSLETTER_INDICATORS if indicator in text.lower())
        if newsletter_count > 0:
            features.append(f"newsletter_indicators_{min(newsletter_count, 5)}")
        
        # Work indicators
        work_count = sum(1 for indicator in self.WORK_INDICATORS if indicator in text.lower())
        if work_count > 0:
            features.append(f"work_indicators_{min(work_count, 5)}")
        
        # Length-based features
        word_count = len(text.split())
        if word_count < 10:
            features.append("short_content")
        elif word_count > 100:
            features.append("long_content")
        else:
            features.append("medium_content")
        
        # Urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'deadline', 'rush']
        if any(word in text.lower() for word in urgency_words):
            features.append("urgent_content")
        
        return features
    
    def build_vocabulary(self, email_data: List[Dict], min_freq: int = 2) -> None:
        """Build vocabulary from email data with category awareness"""
        
        print("Building email-aware vocabulary...")
        
        # Collect all tokens with category information
        all_tokens = Counter()
        category_tokens = defaultdict(Counter)
        
        for email in email_data:
            category = email.get('category', 'other').lower()
            
            # Process different email parts
            parts = []
            
            # Subject
            if email.get('subject'):
                subject_text = self._normalize_text(email['subject'])
                parts.append(('subject', subject_text))
            
            # Body
            if email.get('body'):
                body_text = self._normalize_text(email['body'])
                parts.append(('body', body_text))
            
            # Sender features
            if email.get('sender'):
                sender_features = self._extract_domain_features(email['sender'])
                parts.append(('sender', ' '.join(sender_features)))
            
            # Process all parts
            for part_type, text in parts:
                if not text:
                    continue
                
                # Extract content features
                content_features = self._extract_content_features(text, category)
                
                # Tokenize
                tokens = text.split()
                tokens.extend(content_features)  # Add extracted features as tokens
                
                # Update counters
                for token in tokens:
                    if token and len(token) > 0:
                        all_tokens[token] += 1
                        category_tokens[category][token] += 1
                        self.category_vocabs[category].add(token)
        
        # Build final vocabulary
        # Start with special tokens
        current_vocab_size = len(self.SPECIAL_TOKENS)
        
        # Add high-frequency tokens
        for token, freq in all_tokens.most_common():
            if current_vocab_size >= self.vocab_size:
                break
            
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = current_vocab_size
                self.id_to_token[current_vocab_size] = token
                current_vocab_size += 1
        
        # Store statistics
        self.token_frequencies = all_tokens
        self.category_token_frequencies = dict(category_tokens)
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        print(f"Category vocabularies: {[(cat, len(vocab)) for cat, vocab in self.category_vocabs.items()]}")
    
    def encode_email(self, email: Dict) -> Tuple[List[int], Dict[str, any]]:
        """
        Encode email into token sequence with metadata
        
        Args:
            email: Dictionary with email data
            
        Returns:
            token_ids: List of token IDs
            metadata: Dictionary with encoding metadata
        """
        
        sequence = []
        metadata = {
            'subject_length': 0,
            'body_length': 0,
            'sender_features': [],
            'content_features': [],
            'structure_positions': {}
        }
        
        # Add subject
        if email.get('subject'):
            sequence.append(self.SPECIAL_TOKENS["<SUBJECT>"])
            metadata['structure_positions']['subject_start'] = len(sequence) - 1
            
            subject_text = self._normalize_text(email['subject'])
            subject_tokens = subject_text.split()
            
            for token in subject_tokens:
                token_id = self.vocab.get(token, self.SPECIAL_TOKENS["<UNK>"])
                sequence.append(token_id)
            
            metadata['subject_length'] = len(subject_tokens)
            metadata['structure_positions']['subject_end'] = len(sequence) - 1
        
        # Add sender info
        if email.get('sender'):
            sequence.append(self.SPECIAL_TOKENS["<FROM>"])
            metadata['structure_positions']['sender_start'] = len(sequence) - 1
            
            sender_features = self._extract_domain_features(email['sender'])
            metadata['sender_features'] = sender_features
            
            for feature in sender_features:
                feature_id = self.vocab.get(feature, self.SPECIAL_TOKENS["<UNK>"])
                sequence.append(feature_id)
            
            metadata['structure_positions']['sender_end'] = len(sequence) - 1
        
        # Add body
        if email.get('body'):
            sequence.append(self.SPECIAL_TOKENS["<BODY>"])
            metadata['structure_positions']['body_start'] = len(sequence) - 1
            
            body_text = self._normalize_text(email['body'])
            body_tokens = body_text.split()
            
            # Extract and add content features
            content_features = self._extract_content_features(body_text, email.get('category'))
            metadata['content_features'] = content_features
            
            # Add content feature tokens
            for feature in content_features:
                feature_id = self.vocab.get(feature, self.SPECIAL_TOKENS["<UNK>"])
                sequence.append(feature_id)
            
            # Add body tokens (truncate if too long)
            remaining_space = self.max_seq_len - len(sequence) - 2  # Reserve space for EOS
            body_tokens = body_tokens[:remaining_space]
            
            for token in body_tokens:
                token_id = self.vocab.get(token, self.SPECIAL_TOKENS["<UNK>"])
                sequence.append(token_id)
            
            metadata['body_length'] = len(body_tokens)
            metadata['structure_positions']['body_end'] = len(sequence) - 1
        
        # Add EOS token
        sequence.append(self.SPECIAL_TOKENS["<EOS>"])
        
        # Truncate if too long
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len-1] + [self.SPECIAL_TOKENS["<EOS>"]]
        
        return sequence, metadata
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "<UNK>")
            if token not in ["<PAD>", "<EOS>"]:
                tokens.append(token)
        
        return " ".join(tokens)
    
    def get_attention_mask(self, token_ids: List[int]) -> List[int]:
        """Generate attention mask (1 for real tokens, 0 for padding)"""
        return [1 if token_id != self.SPECIAL_TOKENS["<PAD>"] else 0 for token_id in token_ids]
    
    def pad_sequence(self, token_ids: List[int], max_length: Optional[int] = None) -> List[int]:
        """Pad sequence to specified length"""
        if max_length is None:
            max_length = self.max_seq_len
        
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            padding = [self.SPECIAL_TOKENS["<PAD>"]] * (max_length - len(token_ids))
            return token_ids + padding
    
    def batch_encode(self, emails: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple emails
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Dictionary with batched tensors
        """
        
        batch_token_ids = []
        batch_attention_masks = []
        batch_metadata = []
        
        for email in emails:
            token_ids, metadata = self.encode_email(email)
            padded_ids = self.pad_sequence(token_ids)
            attention_mask = self.get_attention_mask(padded_ids)
            
            batch_token_ids.append(padded_ids)
            batch_attention_masks.append(attention_mask)
            batch_metadata.append(metadata)
        
        return {
            'input_ids': torch.tensor(batch_token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_masks, dtype=torch.long),
            'metadata': batch_metadata
        }
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'token_frequencies': dict(self.token_frequencies),
            'category_token_frequencies': dict(self.category_token_frequencies),
            'category_vocabs': {k: list(v) for k, v in self.category_vocabs.items()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EmailTokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            max_seq_len=tokenizer_data['max_seq_len']
        )
        
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.token_frequencies = Counter(tokenizer_data['token_frequencies'])
        tokenizer.category_token_frequencies = defaultdict(Counter)
        
        for cat, freqs in tokenizer_data['category_token_frequencies'].items():
            tokenizer.category_token_frequencies[cat] = Counter(freqs)
        
        for cat, vocab_list in tokenizer_data['category_vocabs'].items():
            tokenizer.category_vocabs[cat] = set(vocab_list)
        
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    
    def get_vocabulary_stats(self) -> Dict[str, any]:
        """Get vocabulary statistics"""
        stats = {
            'total_tokens': len(self.vocab),
            'special_tokens': len(self.SPECIAL_TOKENS),
            'regular_tokens': len(self.vocab) - len(self.SPECIAL_TOKENS),
            'most_frequent_tokens': self.token_frequencies.most_common(20),
            'category_vocab_sizes': {cat: len(vocab) for cat, vocab in self.category_vocabs.items()},
            'avg_token_frequency': np.mean(list(self.token_frequencies.values())) if self.token_frequencies else 0
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test tokenizer
    sample_emails = [
        {
            "id": "email_001",
            "subject": "Weekly Newsletter - Tech Updates",
            "body": "Here are the latest tech updates from this week. Visit https://example.com for more info. Contact us at support@company.com or call 555-123-4567.",
            "sender": "newsletter@techblog.com",
            "category": "newsletter"
        },
        {
            "id": "email_002",
            "subject": "Meeting Tomorrow at 2:30 PM",
            "body": "Hi team, urgent reminder about our project meeting tomorrow at 2:30 PM in conference room A. Please bring your reports and budget proposals ($50,000 allocated).",
            "sender": "manager@company.com",
            "category": "work"
        }
    ]
    
    # Create and train tokenizer
    tokenizer = EmailTokenizer(vocab_size=1000, max_seq_len=256)
    tokenizer.build_vocabulary(sample_emails)
    
    # Test encoding
    for email in sample_emails:
        token_ids, metadata = tokenizer.encode_email(email)
        print(f"\nEmail: {email['subject']}")
        print(f"Token IDs: {token_ids[:20]}...")  # First 20 tokens
        print(f"Metadata: {metadata}")
        print(f"Decoded: {tokenizer.decode_tokens(token_ids)[:100]}...")  # First 100 chars
    
    # Test batch encoding
    batch = tokenizer.batch_encode(sample_emails)
    print(f"\nBatch shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    
    # Print statistics
    stats = tokenizer.get_vocabulary_stats()
    print(f"\nVocabulary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")