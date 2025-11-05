#!/usr/bin/env python3
"""
Robustness and Generalization Testing System for Email Classification

This module implements task 8.2: Perform robustness and generalization testing
- Test model performance on diverse email formats and styles
- Validate cross-domain generalization capabilities
- Conduct adversarial testing with challenging email examples
"""

import os
import json
import time
import logging
import re
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False

from macbook_optimization.email_comprehensive_evaluation import ComprehensiveEmailEvaluator
from models.email_tokenizer import EmailTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RobustnessTestConfig:
    """Configuration for robustness testing."""
    
    # Test categories
    test_email_formats: bool = True
    test_cross_domain: bool = True
    test_adversarial: bool = True
    test_noise_robustness: bool = True
    test_length_variations: bool = True
    
    # Email format variations
    format_variations: List[str] = None  # ["html", "plain_text", "mixed", "forwarded", "replied"]
    
    # Cross-domain testing
    domain_variations: List[str] = None  # ["corporate", "personal", "academic", "marketing", "support"]
    
    # Adversarial testing
    adversarial_techniques: List[str] = None  # ["typos", "synonyms", "structure_changes", "spam_evasion"]
    
    # Noise testing
    noise_levels: List[float] = None  # [0.1, 0.2, 0.3]
    noise_types: List[str] = None  # ["character", "word", "sentence"]
    
    # Length variations
    length_variations: List[str] = None  # ["very_short", "short", "medium", "long", "very_long"]
    
    # Testing parameters
    samples_per_test: int = 1000
    confidence_threshold: float = 0.8
    performance_degradation_threshold: float = 0.1  # 10% max degradation
    
    # Output configuration
    output_dir: str = "robustness_testing_results"
    save_failed_cases: bool = True
    generate_detailed_reports: bool = True
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.format_variations is None:
            self.format_variations = ["html", "plain_text", "mixed", "forwarded", "replied"]
        
        if self.domain_variations is None:
            self.domain_variations = ["corporate", "personal", "academic", "marketing", "support"]
        
        if self.adversarial_techniques is None:
            self.adversarial_techniques = ["typos", "synonyms", "structure_changes", "spam_evasion"]
        
        if self.noise_levels is None:
            self.noise_levels = [0.05, 0.1, 0.2]
        
        if self.noise_types is None:
            self.noise_types = ["character", "word", "sentence"]
        
        if self.length_variations is None:
            self.length_variations = ["very_short", "short", "medium", "long", "very_long"]


@dataclass
class RobustnessTestResult:
    """Result of a single robustness test."""
    
    test_id: str
    test_type: str
    test_variation: str
    timestamp: datetime
    
    # Test configuration
    original_samples: int
    modified_samples: int
    
    # Performance metrics
    original_accuracy: float
    modified_accuracy: float
    accuracy_degradation: float
    
    # Detailed metrics
    original_f1_macro: float
    modified_f1_macro: float
    f1_degradation: float
    
    # Per-category performance
    category_degradation: Dict[str, float]
    most_affected_categories: List[Tuple[str, float]]
    
    # Confidence analysis
    original_avg_confidence: float
    modified_avg_confidence: float
    confidence_degradation: float
    
    # Error analysis
    new_errors: int
    error_patterns: Dict[str, int]
    failed_samples: List[Dict[str, Any]]
    
    # Robustness metrics
    robustness_score: float  # 0-1, higher is better
    passes_threshold: bool
    
    # Timing
    test_duration: float


@dataclass
class ComprehensiveRobustnessResult:
    """Complete robustness testing results."""
    
    robustness_id: str
    timestamp: datetime
    config: RobustnessTestConfig
    
    # Test results
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[RobustnessTestResult]
    
    # Overall robustness metrics
    overall_robustness_score: float
    worst_case_degradation: float
    average_degradation: float
    
    # Test type analysis
    format_robustness: Dict[str, float]
    domain_robustness: Dict[str, float]
    adversarial_robustness: Dict[str, float]
    noise_robustness: Dict[str, float]
    length_robustness: Dict[str, float]
    
    # Category vulnerability analysis
    most_vulnerable_categories: List[Tuple[str, float]]
    most_robust_categories: List[Tuple[str, float]]
    
    # Recommendations
    robustness_summary: str
    improvement_recommendations: List[str]
    
    # Output files
    output_files: List[str]


class EmailVariationGenerator:
    """Generate various email format and content variations for robustness testing."""
    
    def __init__(self):
        """Initialize variation generator."""
        self.typo_patterns = [
            (r'the', ['teh', 'th', 'te']),
            (r'and', ['adn', 'an', 'nd']),
            (r'you', ['yu', 'u', 'yuo']),
            (r'ing', ['in', 'ng', 'ig']),
            (r'tion', ['ton', 'tio', 'sion'])
        ]
        
        self.synonym_replacements = {
            'important': ['crucial', 'vital', 'significant', 'critical'],
            'meeting': ['conference', 'session', 'gathering', 'appointment'],
            'urgent': ['pressing', 'immediate', 'critical', 'priority'],
            'please': ['kindly', 'would you', 'could you'],
            'thanks': ['thank you', 'appreciate', 'grateful']
        }
        
        self.spam_evasion_techniques = [
            lambda text: text.replace('o', '0'),  # Character substitution
            lambda text: text.replace(' ', '_'),  # Space replacement
            lambda text: re.sub(r'([a-z])', r'\1.', text),  # Add dots
            lambda text: text.upper(),  # All caps
            lambda text: ''.join([c + ' ' if i % 2 == 0 else c for i, c in enumerate(text)])  # Add spaces
        ]
    
    def generate_format_variations(self, email_text: str, subject: str) -> Dict[str, Dict[str, str]]:
        """Generate different email format variations."""
        
        variations = {}
        
        # HTML format
        variations['html'] = {
            'subject': subject,
            'body': f"<html><body><p>{email_text.replace(chr(10), '</p><p>')}</p></body></html>"
        }
        
        # Plain text (original)
        variations['plain_text'] = {
            'subject': subject,
            'body': email_text
        }
        
        # Mixed format
        variations['mixed'] = {
            'subject': f"Re: {subject}",
            'body': f"<html><body>{email_text}<br><br>--<br>Sent from my iPhone</body></html>"
        }
        
        # Forwarded email
        variations['forwarded'] = {
            'subject': f"Fwd: {subject}",
            'body': f"---------- Forwarded message ---------\nFrom: sender@example.com\nSubject: {subject}\n\n{email_text}"
        }
        
        # Replied email
        variations['replied'] = {
            'subject': f"Re: {subject}",
            'body': f"Thanks for your email.\n\n> {email_text.replace(chr(10), chr(10) + '> ')}"
        }
        
        return variations
    
    def generate_domain_variations(self, email_text: str, subject: str, target_domain: str) -> Dict[str, str]:
        """Generate domain-specific variations."""
        
        domain_templates = {
            'corporate': {
                'prefix': "Dear Team,\n\n",
                'suffix': "\n\nBest regards,\nJohn Smith\nSenior Manager",
                'subject_prefix': "[INTERNAL] "
            },
            'personal': {
                'prefix': "Hi!\n\n",
                'suffix': "\n\nTalk soon!\nSarah",
                'subject_prefix': ""
            },
            'academic': {
                'prefix': "Dear Students,\n\n",
                'suffix': "\n\nSincerely,\nDr. Johnson\nProfessor of Computer Science",
                'subject_prefix': "[CS101] "
            },
            'marketing': {
                'prefix': "🎉 Special Offer! 🎉\n\n",
                'suffix': "\n\nClick here to claim your discount!\nUnsubscribe | Privacy Policy",
                'subject_prefix': "🔥 "
            },
            'support': {
                'prefix': "Thank you for contacting support.\n\n",
                'suffix': "\n\nBest regards,\nSupport Team\nTicket #12345",
                'subject_prefix': "[Support] "
            }
        }
        
        template = domain_templates.get(target_domain, domain_templates['personal'])
        
        return {
            'subject': template['subject_prefix'] + subject,
            'body': template['prefix'] + email_text + template['suffix']
        }
    
    def generate_adversarial_variations(self, email_text: str, technique: str) -> str:
        """Generate adversarial variations using specified technique."""
        
        if technique == "typos":
            return self._add_typos(email_text)
        elif technique == "synonyms":
            return self._replace_synonyms(email_text)
        elif technique == "structure_changes":
            return self._change_structure(email_text)
        elif technique == "spam_evasion":
            return self._apply_spam_evasion(email_text)
        else:
            return email_text
    
    def _add_typos(self, text: str) -> str:
        """Add realistic typos to text."""
        
        words = text.split()
        modified_words = []
        
        for word in words:
            if random.random() < 0.1:  # 10% chance of typo
                # Apply typo patterns
                for pattern, replacements in self.typo_patterns:
                    if re.search(pattern, word.lower()):
                        replacement = random.choice(replacements)
                        word = re.sub(pattern, replacement, word.lower())
                        break
                else:
                    # Random character deletion/insertion
                    if len(word) > 3 and random.random() < 0.5:
                        pos = random.randint(1, len(word) - 2)
                        if random.random() < 0.5:
                            # Delete character
                            word = word[:pos] + word[pos+1:]
                        else:
                            # Insert random character
                            char = random.choice('abcdefghijklmnopqrstuvwxyz')
                            word = word[:pos] + char + word[pos:]
            
            modified_words.append(word)
        
        return ' '.join(modified_words)
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms."""
        
        for word, synonyms in self.synonym_replacements.items():
            if word in text.lower():
                synonym = random.choice(synonyms)
                text = re.sub(r'\b' + word + r'\b', synonym, text, flags=re.IGNORECASE)
        
        return text
    
    def _change_structure(self, text: str) -> str:
        """Change sentence structure."""
        
        sentences = text.split('.')
        if len(sentences) > 2:
            # Randomly reorder sentences
            random.shuffle(sentences)
            return '.'.join(sentences)
        
        return text
    
    def _apply_spam_evasion(self, text: str) -> str:
        """Apply spam evasion techniques."""
        
        technique = random.choice(self.spam_evasion_techniques)
        return technique(text)
    
    def generate_noise_variations(self, email_text: str, noise_type: str, noise_level: float) -> str:
        """Generate noisy variations of email text."""
        
        if noise_type == "character":
            return self._add_character_noise(email_text, noise_level)
        elif noise_type == "word":
            return self._add_word_noise(email_text, noise_level)
        elif noise_type == "sentence":
            return self._add_sentence_noise(email_text, noise_level)
        else:
            return email_text
    
    def _add_character_noise(self, text: str, noise_level: float) -> str:
        """Add character-level noise."""
        
        chars = list(text)
        num_changes = int(len(chars) * noise_level)
        
        for _ in range(num_changes):
            pos = random.randint(0, len(chars) - 1)
            if random.random() < 0.5:
                # Replace character
                chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz ')
            else:
                # Delete character
                if pos < len(chars):
                    chars.pop(pos)
        
        return ''.join(chars)
    
    def _add_word_noise(self, text: str, noise_level: float) -> str:
        """Add word-level noise."""
        
        words = text.split()
        num_changes = int(len(words) * noise_level)
        
        for _ in range(num_changes):
            pos = random.randint(0, len(words) - 1)
            if random.random() < 0.3:
                # Delete word
                words.pop(pos)
            elif random.random() < 0.6:
                # Replace word
                words[pos] = random.choice(['the', 'and', 'or', 'but', 'with', 'for'])
            else:
                # Insert random word
                words.insert(pos, random.choice(['very', 'really', 'quite', 'somewhat']))
        
        return ' '.join(words)
    
    def _add_sentence_noise(self, text: str, noise_level: float) -> str:
        """Add sentence-level noise."""
        
        sentences = text.split('.')
        num_changes = int(len(sentences) * noise_level)
        
        for _ in range(num_changes):
            if sentences:
                pos = random.randint(0, len(sentences) - 1)
                if random.random() < 0.5:
                    # Delete sentence
                    if len(sentences) > 1:
                        sentences.pop(pos)
                else:
                    # Duplicate sentence
                    sentences.insert(pos, sentences[pos])
        
        return '.'.join(sentences)
    
    def generate_length_variations(self, email_text: str, target_length: str) -> str:
        """Generate length variations."""
        
        words = email_text.split()
        
        if target_length == "very_short":
            # Keep only first 10 words
            return ' '.join(words[:10])
        elif target_length == "short":
            # Keep first 25 words
            return ' '.join(words[:25])
        elif target_length == "medium":
            # Keep original or truncate to 100 words
            return ' '.join(words[:100])
        elif target_length == "long":
            # Extend by repeating content
            extended = words + words[:len(words)//2]
            return ' '.join(extended)
        elif target_length == "very_long":
            # Significantly extend
            extended = words * 3
            return ' '.join(extended)
        else:
            return email_text


class RobustnessTestDataset(Dataset):
    """Dataset for robustness testing with variations."""
    
    def __init__(self, original_samples: List[Dict], variations: List[Dict], tokenizer: EmailTokenizer):
        """
        Initialize robustness test dataset.
        
        Args:
            original_samples: Original email samples
            variations: Modified email samples
            tokenizer: Email tokenizer
        """
        self.original_samples = original_samples
        self.variations = variations
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.variations)
    
    def __getitem__(self, idx):
        """Get tokenized sample."""
        
        variation = self.variations[idx]
        
        # Tokenize email
        tokens = self.tokenizer.tokenize_email(
            subject=variation['subject'],
            body=variation['body'],
            sender=variation.get('sender', ''),
            recipient=variation.get('recipient', '')
        )
        
        return {
            'inputs': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'labels': torch.tensor(variation['category_id'], dtype=torch.long),
            'original_idx': variation['original_idx'],
            'variation_type': variation['variation_type']
        }


class RobustnessTestingSystem:
    """
    Comprehensive robustness and generalization testing system.
    
    Tests model performance across various email format variations,
    domain changes, adversarial examples, and noise conditions.
    """
    
    def __init__(self, config: RobustnessTestConfig):
        """
        Initialize robustness testing system.
        
        Args:
            config: Robustness testing configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.variation_generator = EmailVariationGenerator()
        self.evaluator = ComprehensiveEmailEvaluator(
            output_dir=str(self.output_dir / "evaluation_outputs"),
            save_detailed_predictions=config.save_failed_cases
        )
        
        # Testing state
        self.robustness_id = f"robustness_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_results: List[RobustnessTestResult] = []
        
        logger.info(f"RobustnessTestingSystem initialized: {self.robustness_id}")
    
    def execute_comprehensive_robustness_testing(self, 
                                               model,
                                               original_dataset: List[Dict],
                                               tokenizer: EmailTokenizer) -> ComprehensiveRobustnessResult:
        """
        Execute comprehensive robustness testing.
        
        Args:
            model: Trained email classification model
            original_dataset: Original email dataset samples
            tokenizer: Email tokenizer
            
        Returns:
            Comprehensive robustness results
        """
        logger.info(f"Starting comprehensive robustness testing: {self.robustness_id}")
        
        start_time = time.time()
        
        # Sample test data
        test_samples = self._sample_test_data(original_dataset)
        logger.info(f"Selected {len(test_samples)} samples for robustness testing")
        
        # Execute different test categories
        if self.config.test_email_formats:
            self._test_email_format_robustness(model, test_samples, tokenizer)
        
        if self.config.test_cross_domain:
            self._test_cross_domain_robustness(model, test_samples, tokenizer)
        
        if self.config.test_adversarial:
            self._test_adversarial_robustness(model, test_samples, tokenizer)
        
        if self.config.test_noise_robustness:
            self._test_noise_robustness(model, test_samples, tokenizer)
        
        if self.config.test_length_variations:
            self._test_length_variation_robustness(model, test_samples, tokenizer)
        
        # Analyze results
        comprehensive_result = self._analyze_robustness_results(start_time)
        
        # Save results
        if self.config.save_failed_cases:
            self._save_robustness_results(comprehensive_result)
        
        # Generate reports
        if self.config.generate_detailed_reports:
            self._generate_robustness_reports(comprehensive_result)
        
        logger.info(f"Comprehensive robustness testing completed: {self.robustness_id}")
        logger.info(f"Overall robustness score: {comprehensive_result.overall_robustness_score:.3f}")
        
        return comprehensive_result
    
    def _sample_test_data(self, dataset: List[Dict]) -> List[Dict]:
        """Sample test data for robustness testing."""
        
        if not dataset:
            return []
        
        # Ensure we have samples from each category
        category_samples = defaultdict(list)
        for sample in dataset:
            category_samples[sample.get('category', 'unknown')].append(sample)
        
        # Sample evenly from each category
        samples_per_category = max(1, self.config.samples_per_test // len(category_samples))
        
        selected_samples = []
        for category, samples in category_samples.items():
            if len(samples) >= samples_per_category:
                selected = random.sample(samples, samples_per_category)
            else:
                selected = samples
            selected_samples.extend(selected)
        
        # If we need more samples, randomly select additional ones
        if len(selected_samples) < self.config.samples_per_test:
            remaining_needed = self.config.samples_per_test - len(selected_samples)
            additional_samples = random.sample(dataset, min(remaining_needed, len(dataset)))
            selected_samples.extend(additional_samples)
        
        return selected_samples[:self.config.samples_per_test]
    
    def _test_email_format_robustness(self, model, test_samples: List[Dict], tokenizer: EmailTokenizer):
        """Test robustness to different email formats."""
        
        logger.info("Testing email format robustness...")
        
        for format_type in self.config.format_variations:
            logger.info(f"Testing format variation: {format_type}")
            
            # Generate format variations
            variations = []
            for i, sample in enumerate(test_samples):
                format_variations = self.variation_generator.generate_format_variations(
                    sample['body'], sample['subject']
                )
                
                if format_type in format_variations:
                    variation = format_variations[format_type].copy()
                    variation.update({
                        'category_id': sample.get('category_id', 0),
                        'category': sample.get('category', 'unknown'),
                        'original_idx': i,
                        'variation_type': f"format_{format_type}"
                    })
                    variations.append(variation)
            
            # Test variations
            if variations:
                result = self._execute_robustness_test(
                    model, test_samples, variations, tokenizer,
                    test_type="email_format", test_variation=format_type
                )
                self.test_results.append(result)
    
    def _test_cross_domain_robustness(self, model, test_samples: List[Dict], tokenizer: EmailTokenizer):
        """Test cross-domain generalization."""
        
        logger.info("Testing cross-domain robustness...")
        
        for domain in self.config.domain_variations:
            logger.info(f"Testing domain variation: {domain}")
            
            # Generate domain variations
            variations = []
            for i, sample in enumerate(test_samples):
                domain_variation = self.variation_generator.generate_domain_variations(
                    sample['body'], sample['subject'], domain
                )
                
                variation = domain_variation.copy()
                variation.update({
                    'category_id': sample.get('category_id', 0),
                    'category': sample.get('category', 'unknown'),
                    'original_idx': i,
                    'variation_type': f"domain_{domain}"
                })
                variations.append(variation)
            
            # Test variations
            result = self._execute_robustness_test(
                model, test_samples, variations, tokenizer,
                test_type="cross_domain", test_variation=domain
            )
            self.test_results.append(result)
    
    def _test_adversarial_robustness(self, model, test_samples: List[Dict], tokenizer: EmailTokenizer):
        """Test adversarial robustness."""
        
        logger.info("Testing adversarial robustness...")
        
        for technique in self.config.adversarial_techniques:
            logger.info(f"Testing adversarial technique: {technique}")
            
            # Generate adversarial variations
            variations = []
            for i, sample in enumerate(test_samples):
                modified_body = self.variation_generator.generate_adversarial_variations(
                    sample['body'], technique
                )
                
                variation = {
                    'subject': sample['subject'],
                    'body': modified_body,
                    'category_id': sample.get('category_id', 0),
                    'category': sample.get('category', 'unknown'),
                    'original_idx': i,
                    'variation_type': f"adversarial_{technique}"
                }
                variations.append(variation)
            
            # Test variations
            result = self._execute_robustness_test(
                model, test_samples, variations, tokenizer,
                test_type="adversarial", test_variation=technique
            )
            self.test_results.append(result)
    
    def _test_noise_robustness(self, model, test_samples: List[Dict], tokenizer: EmailTokenizer):
        """Test noise robustness."""
        
        logger.info("Testing noise robustness...")
        
        for noise_type in self.config.noise_types:
            for noise_level in self.config.noise_levels:
                test_name = f"{noise_type}_{noise_level}"
                logger.info(f"Testing noise variation: {test_name}")
                
                # Generate noisy variations
                variations = []
                for i, sample in enumerate(test_samples):
                    modified_body = self.variation_generator.generate_noise_variations(
                        sample['body'], noise_type, noise_level
                    )
                    
                    variation = {
                        'subject': sample['subject'],
                        'body': modified_body,
                        'category_id': sample.get('category_id', 0),
                        'category': sample.get('category', 'unknown'),
                        'original_idx': i,
                        'variation_type': f"noise_{test_name}"
                    }
                    variations.append(variation)
                
                # Test variations
                result = self._execute_robustness_test(
                    model, test_samples, variations, tokenizer,
                    test_type="noise", test_variation=test_name
                )
                self.test_results.append(result)
    
    def _test_length_variation_robustness(self, model, test_samples: List[Dict], tokenizer: EmailTokenizer):
        """Test length variation robustness."""
        
        logger.info("Testing length variation robustness...")
        
        for length_type in self.config.length_variations:
            logger.info(f"Testing length variation: {length_type}")
            
            # Generate length variations
            variations = []
            for i, sample in enumerate(test_samples):
                modified_body = self.variation_generator.generate_length_variations(
                    sample['body'], length_type
                )
                
                variation = {
                    'subject': sample['subject'],
                    'body': modified_body,
                    'category_id': sample.get('category_id', 0),
                    'category': sample.get('category', 'unknown'),
                    'original_idx': i,
                    'variation_type': f"length_{length_type}"
                }
                variations.append(variation)
            
            # Test variations
            result = self._execute_robustness_test(
                model, test_samples, variations, tokenizer,
                test_type="length_variation", test_variation=length_type
            )
            self.test_results.append(result)
    
    def _execute_robustness_test(self, 
                               model,
                               original_samples: List[Dict],
                               variations: List[Dict],
                               tokenizer: EmailTokenizer,
                               test_type: str,
                               test_variation: str) -> RobustnessTestResult:
        """Execute a single robustness test."""
        
        test_id = f"{self.robustness_id}_{test_type}_{test_variation}"
        start_time = time.time()
        
        try:
            # Create datasets
            original_dataset = RobustnessTestDataset(original_samples, original_samples, tokenizer)
            variation_dataset = RobustnessTestDataset(original_samples, variations, tokenizer)
            
            # Create data loaders
            original_loader = DataLoader(original_dataset, batch_size=16, shuffle=False)
            variation_loader = DataLoader(variation_dataset, batch_size=16, shuffle=False)
            
            # Evaluate original samples
            original_results = self._evaluate_samples(model, original_loader)
            
            # Evaluate variations
            variation_results = self._evaluate_samples(model, variation_loader)
            
            # Calculate metrics
            original_accuracy = original_results['accuracy']
            modified_accuracy = variation_results['accuracy']
            accuracy_degradation = original_accuracy - modified_accuracy
            
            original_f1 = original_results['f1_macro']
            modified_f1 = variation_results['f1_macro']
            f1_degradation = original_f1 - modified_f1
            
            # Analyze per-category degradation
            category_degradation = {}
            for category in original_results['category_accuracies']:
                orig_acc = original_results['category_accuracies'][category]
                mod_acc = variation_results['category_accuracies'].get(category, 0.0)
                category_degradation[category] = orig_acc - mod_acc
            
            # Find most affected categories
            most_affected = sorted(category_degradation.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Confidence analysis
            original_confidence = original_results['avg_confidence']
            modified_confidence = variation_results['avg_confidence']
            confidence_degradation = original_confidence - modified_confidence
            
            # Error analysis
            original_errors = set(original_results['error_indices'])
            modified_errors = set(variation_results['error_indices'])
            new_errors = len(modified_errors - original_errors)
            
            # Calculate robustness score
            robustness_score = max(0.0, 1.0 - (accuracy_degradation / max(original_accuracy, 0.01)))
            passes_threshold = accuracy_degradation <= self.config.performance_degradation_threshold
            
            # Create result
            result = RobustnessTestResult(
                test_id=test_id,
                test_type=test_type,
                test_variation=test_variation,
                timestamp=datetime.now(),
                original_samples=len(original_samples),
                modified_samples=len(variations),
                original_accuracy=original_accuracy,
                modified_accuracy=modified_accuracy,
                accuracy_degradation=accuracy_degradation,
                original_f1_macro=original_f1,
                modified_f1_macro=modified_f1,
                f1_degradation=f1_degradation,
                category_degradation=category_degradation,
                most_affected_categories=most_affected,
                original_avg_confidence=original_confidence,
                modified_avg_confidence=modified_confidence,
                confidence_degradation=confidence_degradation,
                new_errors=new_errors,
                error_patterns={},  # Would be filled with detailed error analysis
                failed_samples=[],  # Would be filled with failed sample details
                robustness_score=robustness_score,
                passes_threshold=passes_threshold,
                test_duration=time.time() - start_time
            )
            
            logger.info(f"Robustness test completed: {test_id}")
            logger.info(f"Accuracy degradation: {accuracy_degradation:.4f}, Robustness score: {robustness_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Robustness test failed: {test_id}: {e}")
            
            # Return failed result
            return RobustnessTestResult(
                test_id=test_id,
                test_type=test_type,
                test_variation=test_variation,
                timestamp=datetime.now(),
                original_samples=len(original_samples),
                modified_samples=len(variations),
                original_accuracy=0.0,
                modified_accuracy=0.0,
                accuracy_degradation=1.0,  # Maximum degradation
                original_f1_macro=0.0,
                modified_f1_macro=0.0,
                f1_degradation=1.0,
                category_degradation={},
                most_affected_categories=[],
                original_avg_confidence=0.0,
                modified_avg_confidence=0.0,
                confidence_degradation=1.0,
                new_errors=len(variations),
                error_patterns={},
                failed_samples=[],
                robustness_score=0.0,
                passes_threshold=False,
                test_duration=time.time() - start_time
            )
    
    def _evaluate_samples(self, model, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on samples and return metrics."""
        
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        error_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = batch['inputs']
                labels = batch['labels']
                
                # Forward pass
                outputs = model(inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                # Track errors
                if hasattr(predictions, 'cpu'):
                    pred_numpy = predictions.cpu().numpy()
                else:
                    pred_numpy = predictions
                
                if hasattr(labels, 'cpu'):
                    label_numpy = labels.cpu().numpy()
                else:
                    label_numpy = labels
                
                errors = (pred_numpy != label_numpy)
                batch_start = batch_idx * dataloader.batch_size
                error_indices.extend([batch_start + i for i, is_error in enumerate(errors) if is_error])
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        accuracy = np.mean(all_predictions == all_labels)
        avg_confidence = np.mean(all_confidences)
        
        # Per-category accuracy
        category_accuracies = {}
        unique_labels = np.unique(all_labels)
        
        for label in unique_labels:
            mask = all_labels == label
            if mask.sum() > 0:
                category_acc = np.mean(all_predictions[mask] == all_labels[mask])
                category_accuracies[f"category_{label}"] = category_acc
        
        # Calculate F1 macro (simplified)
        from sklearn.metrics import f1_score
        try:
            f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        except:
            f1_macro = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'avg_confidence': avg_confidence,
            'category_accuracies': category_accuracies,
            'error_indices': error_indices,
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences
        }
    
    def _analyze_robustness_results(self, start_time: float) -> ComprehensiveRobustnessResult:
        """Analyze all robustness test results."""
        
        total_time = time.time() - start_time
        
        # Basic statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passes_threshold)
        failed_tests = total_tests - passed_tests
        
        # Overall robustness metrics
        robustness_scores = [r.robustness_score for r in self.test_results]
        overall_robustness_score = np.mean(robustness_scores) if robustness_scores else 0.0
        
        degradations = [r.accuracy_degradation for r in self.test_results]
        worst_case_degradation = max(degradations) if degradations else 0.0
        average_degradation = np.mean(degradations) if degradations else 0.0
        
        # Analyze by test type
        format_robustness = self._analyze_test_type_robustness("email_format")
        domain_robustness = self._analyze_test_type_robustness("cross_domain")
        adversarial_robustness = self._analyze_test_type_robustness("adversarial")
        noise_robustness = self._analyze_test_type_robustness("noise")
        length_robustness = self._analyze_test_type_robustness("length_variation")
        
        # Category vulnerability analysis
        category_vulnerabilities = self._analyze_category_vulnerabilities()
        most_vulnerable = sorted(category_vulnerabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        most_robust = sorted(category_vulnerabilities.items(), key=lambda x: x[1])[:3]
        
        # Generate summary and recommendations
        summary, recommendations = self._generate_robustness_summary(
            overall_robustness_score, worst_case_degradation, passed_tests, total_tests
        )
        
        # Create comprehensive result
        result = ComprehensiveRobustnessResult(
            robustness_id=self.robustness_id,
            timestamp=datetime.now(),
            config=self.config,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=self.test_results,
            overall_robustness_score=overall_robustness_score,
            worst_case_degradation=worst_case_degradation,
            average_degradation=average_degradation,
            format_robustness=format_robustness,
            domain_robustness=domain_robustness,
            adversarial_robustness=adversarial_robustness,
            noise_robustness=noise_robustness,
            length_robustness=length_robustness,
            most_vulnerable_categories=most_vulnerable,
            most_robust_categories=most_robust,
            robustness_summary=summary,
            improvement_recommendations=recommendations,
            output_files=[]
        )
        
        return result
    
    def _analyze_test_type_robustness(self, test_type: str) -> Dict[str, float]:
        """Analyze robustness for a specific test type."""
        
        type_results = [r for r in self.test_results if r.test_type == test_type]
        
        if not type_results:
            return {}
        
        robustness_by_variation = {}
        for result in type_results:
            robustness_by_variation[result.test_variation] = result.robustness_score
        
        return robustness_by_variation
    
    def _analyze_category_vulnerabilities(self) -> Dict[str, float]:
        """Analyze category vulnerabilities across all tests."""
        
        category_degradations = defaultdict(list)
        
        for result in self.test_results:
            for category, degradation in result.category_degradation.items():
                category_degradations[category].append(degradation)
        
        category_vulnerabilities = {}
        for category, degradations in category_degradations.items():
            category_vulnerabilities[category] = np.mean(degradations)
        
        return category_vulnerabilities
    
    def _generate_robustness_summary(self, 
                                   overall_score: float,
                                   worst_degradation: float,
                                   passed_tests: int,
                                   total_tests: int) -> Tuple[str, List[str]]:
        """Generate robustness summary and recommendations."""
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        summary = f"""
Robustness Testing Summary:
- Overall robustness score: {overall_score:.3f}/1.000
- Test pass rate: {pass_rate:.1%} ({passed_tests}/{total_tests})
- Worst case performance degradation: {worst_degradation:.1%}
- Robustness level: {'EXCELLENT' if overall_score >= 0.9 else 'GOOD' if overall_score >= 0.7 else 'FAIR' if overall_score >= 0.5 else 'POOR'}
"""
        
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Overall robustness is below acceptable levels. Consider data augmentation and adversarial training.")
        
        if worst_degradation > 0.2:
            recommendations.append("Significant performance degradation detected in some scenarios. Focus on improving worst-case robustness.")
        
        if pass_rate < 0.8:
            recommendations.append("Low test pass rate indicates systematic robustness issues. Review model architecture and training strategy.")
        
        return summary.strip(), recommendations
    
    def _save_robustness_results(self, result: ComprehensiveRobustnessResult):
        """Save robustness testing results."""
        
        try:
            # Save main result
            result_file = self.output_dir / f"{self.robustness_id}_robustness_results.json"
            
            result_dict = asdict(result)
            result_dict['timestamp'] = result.timestamp.isoformat()
            
            # Convert test results
            for i, test_result in enumerate(result_dict['test_results']):
                test_result['timestamp'] = test_result['timestamp'].isoformat() if isinstance(test_result['timestamp'], datetime) else test_result['timestamp']
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            result.output_files.append(str(result_file))
            logger.info(f"Robustness results saved to {result_file}")
            
        except Exception as e:
            logger.error(f"Failed to save robustness results: {e}")
    
    def _generate_robustness_reports(self, result: ComprehensiveRobustnessResult):
        """Generate human-readable robustness reports."""
        
        try:
            # Generate main report
            report_file = self.output_dir / f"{self.robustness_id}_robustness_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("EMAIL CLASSIFICATION ROBUSTNESS TESTING REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Robustness Test ID: {result.robustness_id}\n")
                f.write(f"Timestamp: {result.timestamp}\n\n")
                
                # Summary
                f.write("ROBUSTNESS SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(result.robustness_summary + "\n\n")
                
                # Test Results by Type
                f.write("ROBUSTNESS BY TEST TYPE\n")
                f.write("-" * 40 + "\n")
                
                if result.format_robustness:
                    f.write("Email Format Robustness:\n")
                    for variation, score in result.format_robustness.items():
                        f.write(f"  {variation}: {score:.3f}\n")
                    f.write("\n")
                
                if result.adversarial_robustness:
                    f.write("Adversarial Robustness:\n")
                    for variation, score in result.adversarial_robustness.items():
                        f.write(f"  {variation}: {score:.3f}\n")
                    f.write("\n")
                
                if result.noise_robustness:
                    f.write("Noise Robustness:\n")
                    for variation, score in result.noise_robustness.items():
                        f.write(f"  {variation}: {score:.3f}\n")
                    f.write("\n")
                
                # Category Analysis
                if result.most_vulnerable_categories:
                    f.write("MOST VULNERABLE CATEGORIES\n")
                    f.write("-" * 40 + "\n")
                    for category, vulnerability in result.most_vulnerable_categories:
                        f.write(f"{category}: {vulnerability:.4f} avg degradation\n")
                    f.write("\n")
                
                # Recommendations
                if result.improvement_recommendations:
                    f.write("IMPROVEMENT RECOMMENDATIONS\n")
                    f.write("-" * 40 + "\n")
                    for i, rec in enumerate(result.improvement_recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                f.write("="*80 + "\n")
            
            result.output_files.append(str(report_file))
            logger.info(f"Robustness report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate robustness report: {e}")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = RobustnessTestConfig(
        test_email_formats=True,
        test_adversarial=True,
        test_noise_robustness=True,
        samples_per_test=50,  # Reduced for testing
        output_dir="robustness_testing_test"
    )
    
    # Create robustness testing system
    robustness_system = RobustnessTestingSystem(config)
    
    print("Robustness testing system ready!")
    print("Use robustness_system.execute_comprehensive_robustness_testing() to start testing")