#!/usr/bin/env python3
"""
Production Dataset Validation Script

This script performs comprehensive validation of the production email dataset
to ensure it meets all requirements for training the EmailTRM model.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
import re
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionDatasetValidator:
    """Comprehensive validator for production email dataset."""
    
    def __init__(self, dataset_path: str = "data/production-emails"):
        self.dataset_path = Path(dataset_path)
        self.required_categories = [
            "newsletter", "work", "personal", "spam", "promotional",
            "social", "finance", "travel", "shopping", "other"
        ]
        self.min_emails_per_category = 1000
        self.min_subject_length = 5
        self.max_subject_length = 200
        self.min_body_length = 10
        self.max_body_length = 10000
        self.supported_languages = ["en", "tr"]
        
    def validate_production_dataset(self) -> Dict[str, Any]:
        """
        Run comprehensive production dataset validation.
        
        Returns:
            Complete validation results
        """
        logger.info("Starting comprehensive production dataset validation...")
        
        validation_results = {
            'dataset_path': str(self.dataset_path),
            'validation_timestamp': datetime.now().isoformat(),
            'dataset_structure': {},
            'content_validation': {},
            'category_analysis': {},
            'quality_metrics': {},
            'production_readiness': {},
            'performance_estimates': {},
            'recommendations': [],
            'overall_status': 'unknown'
        }
        
        try:
            # 1. Validate dataset structure
            structure_results = self._validate_dataset_structure()
            validation_results['dataset_structure'] = structure_results
            
            # 2. Validate email content
            content_results = self._validate_email_content()
            validation_results['content_validation'] = content_results
            
            # 3. Analyze categories
            category_results = self._analyze_categories()
            validation_results['category_analysis'] = category_results
            
            # 4. Calculate quality metrics
            quality_results = self._calculate_quality_metrics(content_results, category_results)
            validation_results['quality_metrics'] = quality_results
            
            # 5. Assess production readiness
            readiness_results = self._assess_production_readiness(validation_results)
            validation_results['production_readiness'] = readiness_results
            
            # 6. Estimate performance characteristics
            performance_results = self._estimate_performance_characteristics()
            validation_results['performance_estimates'] = performance_results
            
            # 7. Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
            
            # 8. Determine overall status
            validation_results['overall_status'] = self._determine_overall_status(validation_results)
            
            logger.info(f"Production dataset validation completed: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _validate_dataset_structure(self) -> Dict[str, Any]:
        """Validate the basic structure of the dataset."""
        logger.info("Validating dataset structure...")
        
        structure_results = {
            'dataset_exists': self.dataset_path.exists(),
            'splits_found': [],
            'metadata_files': {},
            'file_counts': {},
            'total_files': 0,
            'structure_score': 0.0
        }
        
        if not structure_results['dataset_exists']:
            return structure_results
        
        # Check for expected splits
        expected_splits = ['train', 'val', 'test']
        for split in expected_splits:
            split_dir = self.dataset_path / split
            if split_dir.exists() and split_dir.is_dir():
                structure_results['splits_found'].append(split)
                json_files = list(split_dir.glob("*.json"))
                structure_results['file_counts'][split] = len(json_files)
                structure_results['total_files'] += len(json_files)
        
        # Check for metadata files
        metadata_file = self.dataset_path / "dataset_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                structure_results['metadata_files']['dataset_metadata'] = {
                    'exists': True,
                    'content': metadata
                }
            except Exception as e:
                structure_results['metadata_files']['dataset_metadata'] = {
                    'exists': True,
                    'error': str(e)
                }
        
        # Calculate structure score
        score = 0
        if len(structure_results['splits_found']) == 3:
            score += 40  # All splits present
        if structure_results['total_files'] >= 10000:
            score += 30  # Sufficient data
        if 'dataset_metadata' in structure_results['metadata_files']:
            score += 30  # Metadata present
        
        structure_results['structure_score'] = score
        
        return structure_results
    
    def _validate_email_content(self) -> Dict[str, Any]:
        """Validate the content of email samples."""
        logger.info("Validating email content...")
        
        content_results = {
            'total_emails_processed': 0,
            'valid_emails': 0,
            'validation_errors': [],
            'content_statistics': {},
            'language_distribution': Counter(),
            'sender_analysis': {},
            'content_quality_score': 0.0
        }
        
        validation_errors = []
        all_emails = []
        
        # Process all splits
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                continue
            
            for email_file in split_dir.glob("*.json"):
                content_results['total_emails_processed'] += 1
                
                try:
                    with open(email_file, 'r', encoding='utf-8') as f:
                        email_data = json.load(f)
                    
                    # Validate required fields
                    required_fields = ['id', 'subject', 'body', 'sender', 'recipient', 'category']
                    missing_fields = [field for field in required_fields if field not in email_data]
                    
                    if missing_fields:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Missing required fields: {missing_fields}"
                        })
                        continue
                    
                    # Validate content constraints
                    subject = email_data.get('subject', '')
                    body = email_data.get('body', '')
                    category = email_data.get('category', '')
                    language = email_data.get('language', 'en')
                    
                    # Subject validation
                    if len(subject) < self.min_subject_length:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Subject too short: {len(subject)} < {self.min_subject_length}"
                        })
                        continue
                    
                    if len(subject) > self.max_subject_length:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Subject too long: {len(subject)} > {self.max_subject_length}"
                        })
                        continue
                    
                    # Body validation
                    if len(body) < self.min_body_length:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Body too short: {len(body)} < {self.min_body_length}"
                        })
                        continue
                    
                    if len(body) > self.max_body_length:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Body too long: {len(body)} > {self.max_body_length}"
                        })
                        continue
                    
                    # Category validation
                    if category not in self.required_categories:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Invalid category: {category}"
                        })
                        continue
                    
                    # Language validation
                    if language not in self.supported_languages:
                        validation_errors.append({
                            'file': str(email_file),
                            'error': f"Unsupported language: {language}"
                        })
                        continue
                    
                    # If we get here, email is valid
                    content_results['valid_emails'] += 1
                    content_results['language_distribution'][language] += 1
                    
                    # Collect for statistics
                    all_emails.append({
                        'subject_length': len(subject),
                        'body_length': len(body),
                        'category': category,
                        'language': language,
                        'sender_domain': email_data.get('sender', '').split('@')[-1] if '@' in email_data.get('sender', '') else 'unknown'
                    })
                    
                except Exception as e:
                    validation_errors.append({
                        'file': str(email_file),
                        'error': f"JSON parsing error: {str(e)}"
                    })
        
        # Store validation errors (limit to first 100 for readability)
        content_results['validation_errors'] = validation_errors[:100]
        
        # Calculate content statistics
        if all_emails:
            subject_lengths = [e['subject_length'] for e in all_emails]
            body_lengths = [e['body_length'] for e in all_emails]
            sender_domains = Counter(e['sender_domain'] for e in all_emails)
            
            content_results['content_statistics'] = {
                'subject_length': {
                    'mean': round(np.mean(subject_lengths), 2),
                    'std': round(np.std(subject_lengths), 2),
                    'min': min(subject_lengths),
                    'max': max(subject_lengths),
                    'median': round(np.median(subject_lengths), 2)
                },
                'body_length': {
                    'mean': round(np.mean(body_lengths), 2),
                    'std': round(np.std(body_lengths), 2),
                    'min': min(body_lengths),
                    'max': max(body_lengths),
                    'median': round(np.median(body_lengths), 2)
                }
            }
            
            content_results['sender_analysis'] = {
                'unique_domains': len(sender_domains),
                'top_domains': dict(sender_domains.most_common(10))
            }
        
        # Calculate content quality score
        valid_rate = (content_results['valid_emails'] / content_results['total_emails_processed']) * 100 if content_results['total_emails_processed'] > 0 else 0
        content_results['content_quality_score'] = round(valid_rate, 2)
        
        # Convert Counter to dict for JSON serialization
        content_results['language_distribution'] = dict(content_results['language_distribution'])
        
        return content_results
    
    def _analyze_categories(self) -> Dict[str, Any]:
        """Analyze category distribution and balance."""
        logger.info("Analyzing category distribution...")
        
        category_results = {
            'category_counts': Counter(),
            'category_balance': {},
            'missing_categories': [],
            'category_quality_metrics': {},
            'balance_score': 0.0
        }
        
        # Count categories across all splits
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                continue
            
            for email_file in split_dir.glob("*.json"):
                try:
                    with open(email_file, 'r', encoding='utf-8') as f:
                        email_data = json.load(f)
                    
                    category = email_data.get('category', 'unknown')
                    category_results['category_counts'][category] += 1
                    
                except Exception:
                    continue
        
        # Check for missing categories
        found_categories = set(category_results['category_counts'].keys())
        required_categories = set(self.required_categories)
        category_results['missing_categories'] = list(required_categories - found_categories)
        
        # Analyze balance
        total_emails = sum(category_results['category_counts'].values())
        expected_per_category = total_emails / len(self.required_categories)
        
        balance_scores = []
        for category in self.required_categories:
            count = category_results['category_counts'].get(category, 0)
            
            # Calculate balance metrics
            deviation = abs(count - expected_per_category)
            balance_score = max(0, 100 - (deviation / expected_per_category * 100)) if expected_per_category > 0 else 0
            
            meets_minimum = count >= self.min_emails_per_category
            
            category_results['category_balance'][category] = {
                'count': count,
                'expected': round(expected_per_category, 2),
                'deviation': round(deviation, 2),
                'balance_score': round(balance_score, 2),
                'meets_minimum': meets_minimum,
                'percentage': round((count / total_emails * 100), 2) if total_emails > 0 else 0
            }
            
            balance_scores.append(balance_score)
        
        # Overall balance score
        category_results['balance_score'] = round(np.mean(balance_scores), 2) if balance_scores else 0
        
        # Convert Counter to dict
        category_results['category_counts'] = dict(category_results['category_counts'])
        
        return category_results
    
    def _calculate_quality_metrics(self, content_results: Dict[str, Any], category_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        logger.info("Calculating quality metrics...")
        
        quality_metrics = {
            'data_completeness': 0.0,
            'content_quality': 0.0,
            'category_balance': 0.0,
            'size_adequacy': 0.0,
            'overall_quality': 0.0
        }
        
        # Data completeness (based on validation errors)
        total_processed = content_results.get('total_emails_processed', 0)
        valid_emails = content_results.get('valid_emails', 0)
        completeness = (valid_emails / total_processed * 100) if total_processed > 0 else 0
        quality_metrics['data_completeness'] = round(completeness, 2)
        
        # Content quality (based on content statistics)
        quality_metrics['content_quality'] = content_results.get('content_quality_score', 0)
        
        # Category balance
        quality_metrics['category_balance'] = category_results.get('balance_score', 0)
        
        # Size adequacy (do we have enough data?)
        total_emails = sum(category_results.get('category_counts', {}).values())
        min_required = len(self.required_categories) * self.min_emails_per_category
        size_score = min(100, (total_emails / min_required * 100)) if min_required > 0 else 0
        quality_metrics['size_adequacy'] = round(size_score, 2)
        
        # Overall quality (weighted average)
        weights = {
            'data_completeness': 0.3,
            'content_quality': 0.3,
            'category_balance': 0.2,
            'size_adequacy': 0.2
        }
        
        overall = sum(quality_metrics[metric] * weight for metric, weight in weights.items())
        quality_metrics['overall_quality'] = round(overall, 2)
        
        return quality_metrics
    
    def _assess_production_readiness(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if dataset is ready for production training."""
        logger.info("Assessing production readiness...")
        
        readiness_results = {
            'meets_size_requirements': False,
            'meets_quality_standards': False,
            'meets_balance_requirements': False,
            'has_all_categories': False,
            'ready_for_training': False,
            'confidence_score': 0.0,
            'blocking_issues': [],
            'warnings': []
        }
        
        quality_metrics = validation_results.get('quality_metrics', {})
        category_analysis = validation_results.get('category_analysis', {})
        content_validation = validation_results.get('content_validation', {})
        
        # Check size requirements
        total_emails = content_validation.get('valid_emails', 0)
        min_required = len(self.required_categories) * self.min_emails_per_category
        readiness_results['meets_size_requirements'] = total_emails >= min_required
        
        if not readiness_results['meets_size_requirements']:
            readiness_results['blocking_issues'].append(
                f"Insufficient data: {total_emails} emails < {min_required} required"
            )
        
        # Check quality standards
        overall_quality = quality_metrics.get('overall_quality', 0)
        readiness_results['meets_quality_standards'] = overall_quality >= 90
        
        if not readiness_results['meets_quality_standards']:
            readiness_results['blocking_issues'].append(
                f"Quality too low: {overall_quality}% < 90% required"
            )
        
        # Check balance requirements
        balance_score = category_analysis.get('balance_score', 0)
        readiness_results['meets_balance_requirements'] = balance_score >= 80
        
        if not readiness_results['meets_balance_requirements']:
            readiness_results['warnings'].append(
                f"Category balance could be improved: {balance_score}% < 80% ideal"
            )
        
        # Check all categories present
        missing_categories = category_analysis.get('missing_categories', [])
        readiness_results['has_all_categories'] = len(missing_categories) == 0
        
        if not readiness_results['has_all_categories']:
            readiness_results['blocking_issues'].append(
                f"Missing categories: {missing_categories}"
            )
        
        # Overall readiness
        critical_checks = [
            readiness_results['meets_size_requirements'],
            readiness_results['meets_quality_standards'],
            readiness_results['has_all_categories']
        ]
        
        readiness_results['ready_for_training'] = all(critical_checks)
        
        # Confidence score
        confidence_factors = [
            quality_metrics.get('data_completeness', 0) / 100,
            quality_metrics.get('content_quality', 0) / 100,
            quality_metrics.get('category_balance', 0) / 100,
            quality_metrics.get('size_adequacy', 0) / 100
        ]
        
        readiness_results['confidence_score'] = round(np.mean(confidence_factors) * 100, 2)
        
        return readiness_results
    
    def _estimate_performance_characteristics(self) -> Dict[str, Any]:
        """Estimate training performance characteristics."""
        logger.info("Estimating performance characteristics...")
        
        # Get dataset size info
        total_files = 0
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split
            if split_dir.exists():
                total_files += len(list(split_dir.glob("*.json")))
        
        performance_estimates = {
            'dataset_size': {
                'total_emails': total_files,
                'estimated_memory_usage_mb': total_files * 0.5,  # ~0.5MB per email
                'estimated_disk_space_mb': total_files * 0.1     # ~0.1MB per email file
            },
            'training_estimates': {
                'estimated_epochs_needed': 10,
                'estimated_training_time_hours': max(1, total_files / 1000),  # Rough estimate
                'recommended_batch_size': min(32, max(8, total_files // 1000)),
                'memory_requirements_gb': max(4, total_files / 2500)  # Rough estimate
            },
            'model_performance_expectations': {
                'expected_accuracy_range': "92-97%",
                'confidence_in_estimate': "medium",
                'factors_affecting_performance': [
                    "Dataset size and quality",
                    "Category balance",
                    "Content diversity",
                    "Hardware capabilities"
                ]
            }
        }
        
        return performance_estimates
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        production_readiness = validation_results.get('production_readiness', {})
        quality_metrics = validation_results.get('quality_metrics', {})
        
        # Critical issues
        blocking_issues = production_readiness.get('blocking_issues', [])
        for issue in blocking_issues:
            recommendations.append(f"CRITICAL: {issue}")
        
        # Quality improvements
        overall_quality = quality_metrics.get('overall_quality', 0)
        if overall_quality < 95:
            recommendations.append(f"Consider improving overall quality from {overall_quality}% to 95%+")
        
        # Balance improvements
        balance_score = quality_metrics.get('category_balance', 0)
        if balance_score < 90:
            recommendations.append(f"Improve category balance from {balance_score}% to 90%+")
        
        # Warnings
        warnings = production_readiness.get('warnings', [])
        for warning in warnings:
            recommendations.append(f"WARNING: {warning}")
        
        # Positive feedback
        if production_readiness.get('ready_for_training', False):
            recommendations.append("✓ Dataset is ready for production training!")
            recommendations.append("✓ All critical requirements are met")
            
            confidence = production_readiness.get('confidence_score', 0)
            if confidence >= 95:
                recommendations.append("✓ High confidence in training success")
            elif confidence >= 85:
                recommendations.append("✓ Good confidence in training success")
        
        return recommendations
    
    def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        production_readiness = validation_results.get('production_readiness', {})
        quality_metrics = validation_results.get('quality_metrics', {})
        
        if validation_results.get('error'):
            return 'error'
        
        if not production_readiness.get('ready_for_training', False):
            return 'not_ready'
        
        overall_quality = quality_metrics.get('overall_quality', 0)
        confidence = production_readiness.get('confidence_score', 0)
        
        if overall_quality >= 95 and confidence >= 95:
            return 'excellent'
        elif overall_quality >= 90 and confidence >= 85:
            return 'good'
        elif overall_quality >= 80 and confidence >= 75:
            return 'acceptable'
        else:
            return 'needs_improvement'
    
    def print_validation_report(self, validation_results: Dict[str, Any]) -> None:
        """Print a comprehensive validation report."""
        print("\n" + "="*80)
        print("PRODUCTION DATASET VALIDATION REPORT")
        print("="*80)
        
        print(f"Dataset: {validation_results['dataset_path']}")
        print(f"Validation Time: {validation_results['validation_timestamp']}")
        print(f"Overall Status: {validation_results['overall_status'].upper()}")
        print()
        
        # Dataset structure
        structure = validation_results.get('dataset_structure', {})
        print("DATASET STRUCTURE:")
        print(f"  Total Files: {structure.get('total_files', 0)}")
        print(f"  Splits Found: {structure.get('splits_found', [])}")
        print(f"  Structure Score: {structure.get('structure_score', 0)}%")
        print()
        
        # Content validation
        content = validation_results.get('content_validation', {})
        print("CONTENT VALIDATION:")
        print(f"  Total Processed: {content.get('total_emails_processed', 0)}")
        print(f"  Valid Emails: {content.get('valid_emails', 0)}")
        print(f"  Validation Errors: {len(content.get('validation_errors', []))}")
        print(f"  Content Quality: {content.get('content_quality_score', 0)}%")
        print()
        
        # Category analysis
        category = validation_results.get('category_analysis', {})
        print("CATEGORY ANALYSIS:")
        print(f"  Categories Found: {len(category.get('category_counts', {}))}")
        print(f"  Missing Categories: {category.get('missing_categories', [])}")
        print(f"  Balance Score: {category.get('balance_score', 0)}%")
        print()
        
        # Quality metrics
        quality = validation_results.get('quality_metrics', {})
        print("QUALITY METRICS:")
        for metric, score in quality.items():
            print(f"  {metric.replace('_', ' ').title()}: {score}%")
        print()
        
        # Production readiness
        readiness = validation_results.get('production_readiness', {})
        print("PRODUCTION READINESS:")
        print(f"  Ready for Training: {readiness.get('ready_for_training', False)}")
        print(f"  Confidence Score: {readiness.get('confidence_score', 0)}%")
        print()
        
        # Recommendations
        recommendations = validation_results.get('recommendations', [])
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("="*80)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate production dataset")
    parser.add_argument(
        '--dataset-path',
        default='data/production-emails',
        help='Path to production dataset'
    )
    parser.add_argument(
        '--output-json',
        help='Save validation results to JSON file'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Show only report, not full JSON output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = ProductionDatasetValidator(args.dataset_path)
        
        # Run validation
        results = validator.validate_production_dataset()
        
        # Print report
        validator.print_validation_report(results)
        
        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation results saved to {args.output_json}")
        
        # Print full JSON if not report-only
        if not args.report_only:
            print("\nFULL VALIDATION RESULTS:")
            print(json.dumps(results, indent=2))
        
        # Return appropriate exit code
        status = results.get('overall_status', 'error')
        if status in ['excellent', 'good']:
            return 0
        elif status == 'acceptable':
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 3

if __name__ == "__main__":
    exit(main())