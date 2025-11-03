"""
Enhanced Email Preprocessing and Data Augmentation

This module provides advanced email preprocessing techniques and data augmentation
methods specifically designed for email classification training diversity and
improved model robustness.
"""

import re
import random
import string
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from .email_tokenizer import EmailTokenizer


class EmailDataAugmentation:
    """Data augmentation techniques for email classification training."""
    
    # Synonym mappings for common email terms
    SYNONYMS = {
        'urgent': ['important', 'critical', 'priority', 'asap'],
        'meeting': ['conference', 'call', 'session', 'discussion'],
        'project': ['task', 'assignment', 'work', 'initiative'],
        'deadline': ['due date', 'timeline', 'cutoff', 'target date'],
        'report': ['document', 'summary', 'analysis', 'findings'],
        'team': ['group', 'staff', 'colleagues', 'department'],
        'client': ['customer', 'user', 'account', 'contact'],
        'update': ['news', 'information', 'status', 'progress'],
        'schedule': ['calendar', 'timeline', 'agenda', 'plan'],
        'budget': ['cost', 'expense', 'funding', 'allocation']
    }
    
    # Common email domain variations
    DOMAIN_VARIATIONS = {
        'gmail.com': ['googlemail.com'],
        'yahoo.com': ['yahoo.co.uk', 'ymail.com'],
        'hotmail.com': ['outlook.com', 'live.com'],
        'company.com': ['corp.com', 'inc.com', 'ltd.com']
    }
    
    # Paraphrasing patterns for common email phrases
    PARAPHRASE_PATTERNS = {
        r'please find attached': ['attached you will find', 'i have attached', 'please see attached'],
        r'thank you for': ['thanks for', 'i appreciate', 'grateful for'],
        r'let me know': ['please inform me', 'keep me posted', 'update me'],
        r'as soon as possible': ['asap', 'urgently', 'at your earliest convenience'],
        r'looking forward to': ['anticipating', 'excited about', 'awaiting'],
        r'best regards': ['kind regards', 'sincerely', 'best wishes']
    }
    
    def __init__(self, augmentation_probability: float = 0.3):
        """
        Initialize email data augmentation.
        
        Args:
            augmentation_probability: Probability of applying augmentation
        """
        self.augmentation_probability = augmentation_probability
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            pattern: re.compile(pattern, re.IGNORECASE)
            for pattern in self.PARAPHRASE_PATTERNS.keys()
        }
    
    def augment_email(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """
        Apply data augmentation to an email.
        
        Args:
            email: Email dictionary with subject, body, sender, etc.
            category: Email category for category-specific augmentation
            
        Returns:
            Augmented email dictionary
        """
        if random.random() > self.augmentation_probability:
            return email  # No augmentation
        
        augmented_email = email.copy()
        
        # Apply different augmentation techniques
        augmentation_methods = [
            self._synonym_replacement,
            self._paraphrase_replacement,
            self._domain_variation,
            self._punctuation_variation,
            self._case_variation,
            self._whitespace_variation
        ]
        
        # Randomly select 1-3 augmentation methods
        num_methods = random.randint(1, 3)
        selected_methods = random.sample(augmentation_methods, num_methods)
        
        for method in selected_methods:
            augmented_email = method(augmented_email, category)
        
        return augmented_email
    
    def _synonym_replacement(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Replace words with synonyms."""
        augmented = email.copy()
        
        # Apply to subject
        if 'subject' in augmented:
            augmented['subject'] = self._replace_synonyms(augmented['subject'])
        
        # Apply to body
        if 'body' in augmented:
            augmented['body'] = self._replace_synonyms(augmented['body'])
        
        return augmented
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms in text."""
        words = text.lower().split()
        
        for i, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = word.strip(string.punctuation)
            
            if clean_word in self.SYNONYMS and random.random() < 0.3:  # 30% chance
                synonym = random.choice(self.SYNONYMS[clean_word])
                # Preserve original case and punctuation
                if word.isupper():
                    synonym = synonym.upper()
                elif word.istitle():
                    synonym = synonym.title()
                
                # Preserve punctuation
                punctuation = ''.join(c for c in word if c in string.punctuation)
                words[i] = synonym + punctuation
        
        return ' '.join(words)
    
    def _paraphrase_replacement(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Replace common phrases with paraphrases."""
        augmented = email.copy()
        
        for field in ['subject', 'body']:
            if field in augmented:
                text = augmented[field]
                
                for pattern, replacements in self.PARAPHRASE_PATTERNS.items():
                    compiled_pattern = self.compiled_patterns[pattern]
                    if compiled_pattern.search(text) and random.random() < 0.4:  # 40% chance
                        replacement = random.choice(replacements)
                        text = compiled_pattern.sub(replacement, text, count=1)
                
                augmented[field] = text
        
        return augmented
    
    def _domain_variation(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Vary email domains in sender field."""
        augmented = email.copy()
        
        if 'sender' in augmented and '@' in augmented['sender']:
            sender_parts = augmented['sender'].split('@')
            domain = sender_parts[1].lower()
            
            if domain in self.DOMAIN_VARIATIONS and random.random() < 0.2:  # 20% chance
                new_domain = random.choice(self.DOMAIN_VARIATIONS[domain])
                augmented['sender'] = sender_parts[0] + '@' + new_domain
        
        return augmented
    
    def _punctuation_variation(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Vary punctuation patterns."""
        augmented = email.copy()
        
        for field in ['subject', 'body']:
            if field in augmented:
                text = augmented[field]
                
                # Add/remove exclamation marks (especially for promotional emails)
                if category in ['promotional', 'newsletter'] and random.random() < 0.3:
                    if '!' not in text and random.random() < 0.5:
                        text = text.rstrip('.') + '!'
                    elif text.count('!') == 1 and random.random() < 0.5:
                        text = text.replace('!', '.')
                
                # Vary ellipsis usage
                if random.random() < 0.2:
                    text = re.sub(r'\.{2,}', '...', text)
                
                augmented[field] = text
        
        return augmented
    
    def _case_variation(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Vary case patterns in subject lines."""
        augmented = email.copy()
        
        if 'subject' in augmented and random.random() < 0.2:  # 20% chance
            subject = augmented['subject']
            
            # Convert to title case or sentence case
            if random.random() < 0.5:
                augmented['subject'] = subject.title()
            else:
                augmented['subject'] = subject.capitalize()
        
        return augmented
    
    def _whitespace_variation(self, email: Dict[str, str], category: str) -> Dict[str, str]:
        """Vary whitespace patterns."""
        augmented = email.copy()
        
        for field in ['subject', 'body']:
            if field in augmented:
                text = augmented[field]
                
                # Normalize multiple spaces
                if random.random() < 0.3:
                    text = re.sub(r'\s+', ' ', text)
                
                # Add/remove trailing spaces
                if random.random() < 0.2:
                    text = text.strip()
                
                augmented[field] = text
        
        return augmented
    
    def generate_category_specific_variations(self, email: Dict[str, str], category: str) -> List[Dict[str, str]]:
        """
        Generate multiple category-specific variations of an email.
        
        Args:
            email: Original email
            category: Email category
            
        Returns:
            List of augmented email variations
        """
        variations = []
        
        # Generate 2-5 variations based on category
        if category in ['promotional', 'newsletter']:
            num_variations = random.randint(3, 5)  # More variations for marketing emails
        elif category in ['work', 'personal']:
            num_variations = random.randint(2, 4)
        else:
            num_variations = random.randint(1, 3)
        
        for _ in range(num_variations):
            variation = self.augment_email(email, category)
            variations.append(variation)
        
        return variations


class EmailPreprocessor:
    """Advanced email preprocessing with structure awareness and cleaning."""
    
    # Email signature patterns
    SIGNATURE_PATTERNS = [
        r'--\s*\n.*',  # Standard signature delimiter
        r'best regards.*',
        r'sincerely.*',
        r'thank you.*\n.*',
        r'sent from my.*',
        r'this email was sent.*'
    ]
    
    # Confidentiality notice patterns
    CONFIDENTIALITY_PATTERNS = [
        r'this email is confidential.*',
        r'confidentiality notice.*',
        r'privileged and confidential.*',
        r'if you are not the intended recipient.*'
    ]
    
    # Auto-reply patterns
    AUTO_REPLY_PATTERNS = [
        r'out of office.*',
        r'automatic reply.*',
        r'i am currently out.*',
        r'this is an automated.*'
    ]
    
    def __init__(self, remove_signatures: bool = True,
                 remove_confidentiality: bool = True,
                 normalize_urls: bool = True,
                 normalize_emails: bool = True):
        """
        Initialize email preprocessor.
        
        Args:
            remove_signatures: Remove email signatures
            remove_confidentiality: Remove confidentiality notices
            normalize_urls: Normalize URLs to tokens
            normalize_emails: Normalize email addresses to tokens
        """
        self.remove_signatures = remove_signatures
        self.remove_confidentiality = remove_confidentiality
        self.normalize_urls = normalize_urls
        self.normalize_emails = normalize_emails
        
        # Compile patterns for efficiency
        self.signature_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.SIGNATURE_PATTERNS]
        self.confidentiality_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL)
                                       for pattern in self.CONFIDENTIALITY_PATTERNS]
        self.auto_reply_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL)
                                  for pattern in self.AUTO_REPLY_PATTERNS]
    
    def preprocess_email(self, email: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess email with advanced cleaning and normalization.
        
        Args:
            email: Email dictionary
            
        Returns:
            Preprocessed email dictionary
        """
        processed = email.copy()
        
        # Preprocess body
        if 'body' in processed:
            processed['body'] = self._preprocess_body(processed['body'])
        
        # Preprocess subject
        if 'subject' in processed:
            processed['subject'] = self._preprocess_subject(processed['subject'])
        
        # Preprocess sender
        if 'sender' in processed:
            processed['sender'] = self._preprocess_sender(processed['sender'])
        
        return processed
    
    def _preprocess_body(self, body: str) -> str:
        """Preprocess email body."""
        text = body
        
        # Remove signatures
        if self.remove_signatures:
            for pattern in self.signature_patterns:
                text = pattern.sub('', text)
        
        # Remove confidentiality notices
        if self.remove_confidentiality:
            for pattern in self.confidentiality_patterns:
                text = pattern.sub('', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize URLs
        if self.normalize_urls:
            text = re.sub(r'http[s]?://\S+', '<URL>', text)
            text = re.sub(r'www\.\S+', '<URL>', text)
        
        # Normalize email addresses
        if self.normalize_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
        
        # Normalize phone numbers
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '<PHONE>', text)
        
        # Normalize dates
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '<DATE>', text)
        text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', '<DATE>', text, flags=re.IGNORECASE)
        
        # Normalize times
        text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', '<TIME>', text)
        
        # Normalize currency
        text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '<CURRENCY>', text)
        
        # Normalize multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _preprocess_subject(self, subject: str) -> str:
        """Preprocess email subject."""
        text = subject
        
        # Remove common prefixes
        text = re.sub(r'^(re:|fwd?:|fw:)\s*', '', text, flags=re.IGNORECASE)
        
        # Normalize URLs and emails (less aggressive than body)
        if self.normalize_urls:
            text = re.sub(r'http[s]?://\S+', '<URL>', text)
        
        if self.normalize_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _preprocess_sender(self, sender: str) -> str:
        """Preprocess sender field."""
        # Extract just the email address if name is included
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', sender)
        if email_match:
            return email_match.group().lower()
        
        return sender.lower().strip()
    
    def detect_email_type(self, email: Dict[str, str]) -> Dict[str, float]:
        """
        Detect email type characteristics for better preprocessing.
        
        Args:
            email: Email dictionary
            
        Returns:
            Dictionary with type probabilities
        """
        scores = {
            'auto_reply': 0.0,
            'newsletter': 0.0,
            'promotional': 0.0,
            'personal': 0.0,
            'work': 0.0,
            'spam': 0.0
        }
        
        body = email.get('body', '').lower()
        subject = email.get('subject', '').lower()
        sender = email.get('sender', '').lower()
        
        # Auto-reply detection
        for pattern in self.auto_reply_patterns:
            if pattern.search(body) or pattern.search(subject):
                scores['auto_reply'] += 0.3
        
        # Newsletter indicators
        newsletter_terms = ['unsubscribe', 'newsletter', 'mailing list', 'update preferences']
        scores['newsletter'] = sum(0.2 for term in newsletter_terms if term in body) / len(newsletter_terms)
        
        # Promotional indicators
        promo_terms = ['sale', 'discount', 'offer', 'deal', 'limited time', 'buy now']
        scores['promotional'] = sum(0.15 for term in promo_terms if term in body or term in subject) / len(promo_terms)
        
        # Work indicators
        work_terms = ['meeting', 'project', 'deadline', 'report', 'team', 'client']
        scores['work'] = sum(0.15 for term in work_terms if term in body or term in subject) / len(work_terms)
        
        # Personal indicators (harder to detect, use domain heuristics)
        personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if any(domain in sender for domain in personal_domains):
            scores['personal'] += 0.2
        
        # Spam indicators
        spam_terms = ['viagra', 'lottery', 'winner', 'congratulations', 'click here', 'free money']
        scores['spam'] = sum(0.2 for term in spam_terms if term in body or term in subject) / len(spam_terms)
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores


class EnhancedEmailTokenizer(EmailTokenizer):
    """Enhanced email tokenizer with advanced preprocessing and augmentation."""
    
    def __init__(self, vocab_size: int = 5000, max_seq_len: int = 512,
                 enable_augmentation: bool = True, augmentation_prob: float = 0.3):
        """
        Initialize enhanced email tokenizer.
        
        Args:
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            enable_augmentation: Enable data augmentation
            augmentation_prob: Probability of applying augmentation
        """
        super().__init__(vocab_size, max_seq_len)
        
        self.enable_augmentation = enable_augmentation
        self.preprocessor = EmailPreprocessor()
        self.augmenter = EmailDataAugmentation(augmentation_prob) if enable_augmentation else None
        
        # Enhanced special tokens for better email understanding
        self.ENHANCED_SPECIAL_TOKENS = {
            "<SIGNATURE>": len(self.SPECIAL_TOKENS),
            "<QUOTE>": len(self.SPECIAL_TOKENS) + 1,
            "<FORWARD>": len(self.SPECIAL_TOKENS) + 2,
            "<REPLY>": len(self.SPECIAL_TOKENS) + 3,
            "<ATTACHMENT>": len(self.SPECIAL_TOKENS) + 4,
            "<URGENT>": len(self.SPECIAL_TOKENS) + 5,
            "<CONFIDENTIAL>": len(self.SPECIAL_TOKENS) + 6
        }
        
        # Update vocabulary with enhanced tokens
        self.vocab.update(self.ENHANCED_SPECIAL_TOKENS)
        self.id_to_token.update({v: k for k, v in self.ENHANCED_SPECIAL_TOKENS.items()})
    
    def build_vocabulary_with_augmentation(self, email_data: List[Dict], 
                                         augmentation_factor: int = 2,
                                         min_freq: int = 2) -> None:
        """
        Build vocabulary with data augmentation for better coverage.
        
        Args:
            email_data: List of email dictionaries
            augmentation_factor: Number of augmented samples per original
            min_freq: Minimum frequency for token inclusion
        """
        print(f"Building vocabulary with augmentation (factor: {augmentation_factor})...")
        
        # Original data
        all_emails = email_data.copy()
        
        # Generate augmented data if enabled
        if self.enable_augmentation and self.augmenter:
            augmented_emails = []
            for email in email_data:
                category = email.get('category', 'other')
                variations = self.augmenter.generate_category_specific_variations(email, category)
                augmented_emails.extend(variations[:augmentation_factor])
            
            all_emails.extend(augmented_emails)
            print(f"Generated {len(augmented_emails)} augmented samples")
        
        # Build vocabulary with all data
        self.build_vocabulary(all_emails, min_freq)
    
    def encode_email_with_preprocessing(self, email: Dict) -> Tuple[List[int], Dict[str, any]]:
        """
        Encode email with advanced preprocessing and optional augmentation.
        
        Args:
            email: Email dictionary
            
        Returns:
            Tuple of (token_ids, metadata)
        """
        # Preprocess email
        processed_email = self.preprocessor.preprocess_email(email)
        
        # Apply augmentation during training if enabled
        if self.enable_augmentation and self.augmenter and email.get('training', False):
            category = email.get('category', 'other')
            processed_email = self.augmenter.augment_email(processed_email, category)
        
        # Detect email type for enhanced processing
        email_type_scores = self.preprocessor.detect_email_type(processed_email)
        
        # Encode with enhanced features
        token_ids, metadata = self.encode_email(processed_email)
        
        # Add enhanced metadata
        metadata['email_type_scores'] = email_type_scores
        metadata['preprocessing_applied'] = True
        metadata['augmentation_applied'] = self.enable_augmentation and email.get('training', False)
        
        return token_ids, metadata
    
    def batch_encode_with_preprocessing(self, emails: List[Dict]) -> Dict[str, any]:
        """
        Batch encode emails with preprocessing and augmentation.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Dictionary with batched tensors and metadata
        """
        batch_token_ids = []
        batch_attention_masks = []
        batch_metadata = []
        
        for email in emails:
            token_ids, metadata = self.encode_email_with_preprocessing(email)
            padded_ids = self.pad_sequence(token_ids)
            attention_mask = self.get_attention_mask(padded_ids)
            
            batch_token_ids.append(padded_ids)
            batch_attention_masks.append(attention_mask)
            batch_metadata.append(metadata)
        
        result = {
            'input_ids': torch.tensor(batch_token_ids, dtype=torch.long) if TORCH_AVAILABLE else np.array(batch_token_ids),
            'attention_mask': torch.tensor(batch_attention_masks, dtype=torch.long) if TORCH_AVAILABLE else np.array(batch_attention_masks),
            'metadata': batch_metadata
        }
        
        return result
    
    def get_augmentation_stats(self) -> Dict[str, any]:
        """Get statistics about data augmentation."""
        if not self.enable_augmentation or not self.augmenter:
            return {'augmentation_enabled': False}
        
        return {
            'augmentation_enabled': True,
            'augmentation_probability': self.augmenter.augmentation_probability,
            'synonym_mappings': len(self.augmenter.SYNONYMS),
            'paraphrase_patterns': len(self.augmenter.PARAPHRASE_PATTERNS),
            'domain_variations': len(self.augmenter.DOMAIN_VARIATIONS)
        }


# Utility functions
def create_enhanced_email_tokenizer(vocab_size: int = 5000, max_seq_len: int = 512,
                                  enable_augmentation: bool = True) -> EnhancedEmailTokenizer:
    """Create enhanced email tokenizer with preprocessing and augmentation."""
    return EnhancedEmailTokenizer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        enable_augmentation=enable_augmentation
    )


def preprocess_email_dataset(emails: List[Dict], remove_auto_replies: bool = True) -> List[Dict]:
    """
    Preprocess email dataset with filtering and cleaning.
    
    Args:
        emails: List of email dictionaries
        remove_auto_replies: Remove auto-reply emails
        
    Returns:
        Preprocessed email list
    """
    preprocessor = EmailPreprocessor()
    processed_emails = []
    
    for email in emails:
        # Detect email type
        email_types = preprocessor.detect_email_type(email)
        
        # Skip auto-replies if requested
        if remove_auto_replies and email_types.get('auto_reply', 0) > 0.5:
            continue
        
        # Preprocess email
        processed_email = preprocessor.preprocess_email(email)
        processed_email['email_type_scores'] = email_types
        
        processed_emails.append(processed_email)
    
    return processed_emails