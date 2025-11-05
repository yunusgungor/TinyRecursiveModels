#!/usr/bin/env python3
"""
Dataset Splits Validation Script

This script validates that the production email dataset splits are properly created
with balanced category representation across train/validation/test splits.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetSplitsValidator:
    """Validates dataset splits for quality and balance."""
    
    def __init__(self, dataset_path: str = "data/production-emails"):
        self.dataset_path = Path(dataset_path)
        self.expected_categories = [
            "newsletter", "work", "personal", "spam", "promotional",
            "social", "finance", "travel", "shopping", "other"
        ]
        self.expected_splits = ["train", "val", "test"]
        self.expected_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        
    def validate_splits(self) -> Dict[str, Any]:
        """
        Validate all dataset splits for proper structure and balance.
        
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting dataset splits validation...")
        
        validation_results = {
            'dataset_path': str(self.dataset_path),
            'splits_found': [],
            'split_statistics': {},
            'category_distribution': {},
            'balance_analysis': {},
            'quality_checks': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        try:
            # 1. Check if all expected splits exist
            splits_found = []
            for split in self.expected_splits:
                split_dir = self.dataset_path / split
                if split_dir.exists() and split_dir.is_dir():
                    splits_found.append(split)
                else:
                    logger.warning(f"Split directory not found: {split}")
            
            validation_results['splits_found'] = splits_found
            
            if not splits_found:
                validation_results['overall_status'] = 'error'
                validation_results['error'] = 'No split directories found'
                return validation_results
            
            # 2. Analyze each split
            total_emails = 0
            all_categories = set()
            
            for split in splits_found:
                split_stats = self._analyze_split(split)
                validation_results['split_statistics'][split] = split_stats
                total_emails += split_stats['total_emails']
                all_categories.update(split_stats['categories'].keys())
            
            # 3. Validate category distribution across splits
            category_analysis = self._analyze_category_distribution(validation_results['split_statistics'])
            validation_results['category_distribution'] = category_analysis
            
            # 4. Check split ratios
            ratio_analysis = self._analyze_split_ratios(validation_results['split_statistics'], total_emails)
            validation_results['balance_analysis'] = ratio_analysis
            
            # 5. Quality checks
            quality_results = self._perform_quality_checks(validation_results['split_statistics'])
            validation_results['quality_checks'] = quality_results
            
            # 6. Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
            
            # 7. Determine overall status
            validation_results['overall_status'] = self._determine_overall_status(validation_results)
            
            logger.info(f"Dataset splits validation completed: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _analyze_split(self, split: str) -> Dict[str, Any]:
        """Analyze a single dataset split."""
        logger.info(f"Analyzing {split} split...")
        
        split_dir = self.dataset_path / split
        email_files = list(split_dir.glob("*.json"))
        
        split_stats = {
            'total_emails': 0,
            'categories': Counter(),
            'languages': Counter(),
            'file_count': len(email_files),
            'validation_errors': 0,
            'sample_emails': []
        }
        
        for email_file in email_files:
            try:
                with open(email_file, 'r', encoding='utf-8') as f:
                    email_data = json.load(f)
                
                # Basic validation
                required_fields = ['id', 'subject', 'body', 'sender', 'recipient', 'category']
                if not all(field in email_data for field in required_fields):
                    split_stats['validation_errors'] += 1
                    continue
                
                # Count statistics
                split_stats['total_emails'] += 1
                split_stats['categories'][email_data.get('category', 'unknown')] += 1
                split_stats['languages'][email_data.get('language', 'unknown')] += 1
                
                # Store sample for quality analysis
                if len(split_stats['sample_emails']) < 10:
                    split_stats['sample_emails'].append({
                        'id': email_data.get('id'),
                        'subject_length': len(email_data.get('subject', '')),
                        'body_length': len(email_data.get('body', '')),
                        'category': email_data.get('category')
                    })
                
            except Exception as e:
                logger.warning(f"Error processing {email_file}: {e}")
                split_stats['validation_errors'] += 1
        
        # Convert counters to dicts for JSON serialization
        split_stats['categories'] = dict(split_stats['categories'])
        split_stats['languages'] = dict(split_stats['languages'])
        
        logger.info(f"{split} split: {split_stats['total_emails']} emails, "
                   f"{len(split_stats['categories'])} categories, "
                   f"{split_stats['validation_errors']} errors")
        
        return split_stats
    
    def _analyze_category_distribution(self, split_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze category distribution across splits."""
        logger.info("Analyzing category distribution across splits...")
        
        # Collect category counts per split
        category_by_split = defaultdict(dict)
        total_by_category = Counter()
        
        for split, stats in split_statistics.items():
            for category, count in stats['categories'].items():
                category_by_split[category][split] = count
                total_by_category[category] += count
        
        # Calculate distribution percentages
        distribution_analysis = {
            'categories_found': list(category_by_split.keys()),
            'expected_categories': self.expected_categories,
            'missing_categories': [],
            'category_balance': {},
            'split_representation': {}
        }
        
        # Check for missing categories
        for expected_cat in self.expected_categories:
            if expected_cat not in category_by_split:
                distribution_analysis['missing_categories'].append(expected_cat)
        
        # Analyze balance for each category
        for category, split_counts in category_by_split.items():
            total_cat = total_by_category[category]
            
            balance_info = {
                'total_count': total_cat,
                'split_counts': split_counts,
                'split_percentages': {},
                'balance_score': 0.0
            }
            
            # Calculate percentages
            for split in self.expected_splits:
                count = split_counts.get(split, 0)
                percentage = (count / total_cat * 100) if total_cat > 0 else 0
                balance_info['split_percentages'][split] = round(percentage, 2)
            
            # Calculate balance score (how close to expected ratios)
            balance_score = 0.0
            for split in self.expected_splits:
                expected_pct = self.expected_ratios[split] * 100
                actual_pct = balance_info['split_percentages'].get(split, 0)
                balance_score += abs(expected_pct - actual_pct)
            
            balance_info['balance_score'] = round(100 - balance_score, 2)  # Higher is better
            distribution_analysis['category_balance'][category] = balance_info
        
        # Overall split representation
        for split in self.expected_splits:
            split_total = split_statistics.get(split, {}).get('total_emails', 0)
            categories_in_split = len(split_statistics.get(split, {}).get('categories', {}))
            
            distribution_analysis['split_representation'][split] = {
                'total_emails': split_total,
                'categories_present': categories_in_split,
                'categories_missing': len(self.expected_categories) - categories_in_split
            }
        
        return distribution_analysis
    
    def _analyze_split_ratios(self, split_statistics: Dict[str, Any], total_emails: int) -> Dict[str, Any]:
        """Analyze if split ratios match expected proportions."""
        logger.info("Analyzing split ratios...")
        
        ratio_analysis = {
            'total_emails': total_emails,
            'expected_ratios': self.expected_ratios,
            'actual_ratios': {},
            'ratio_deviations': {},
            'ratio_quality_score': 0.0
        }
        
        # Calculate actual ratios
        total_deviation = 0.0
        for split in self.expected_splits:
            split_count = split_statistics.get(split, {}).get('total_emails', 0)
            actual_ratio = split_count / total_emails if total_emails > 0 else 0
            expected_ratio = self.expected_ratios[split]
            deviation = abs(actual_ratio - expected_ratio)
            
            ratio_analysis['actual_ratios'][split] = round(actual_ratio, 4)
            ratio_analysis['ratio_deviations'][split] = round(deviation, 4)
            total_deviation += deviation
        
        # Calculate quality score (lower deviation = higher score)
        ratio_analysis['ratio_quality_score'] = round(max(0, 100 - (total_deviation * 1000)), 2)
        
        return ratio_analysis
    
    def _perform_quality_checks(self, split_statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform various quality checks on the dataset splits."""
        logger.info("Performing quality checks...")
        
        quality_results = {
            'validation_errors': {},
            'content_quality': {},
            'consistency_checks': {},
            'overall_quality_score': 0.0
        }
        
        total_errors = 0
        total_emails = 0
        
        # Check validation errors per split
        for split, stats in split_statistics.items():
            errors = stats.get('validation_errors', 0)
            emails = stats.get('total_emails', 0)
            error_rate = (errors / (errors + emails)) * 100 if (errors + emails) > 0 else 0
            
            quality_results['validation_errors'][split] = {
                'error_count': errors,
                'total_processed': errors + emails,
                'error_rate_percent': round(error_rate, 2)
            }
            
            total_errors += errors
            total_emails += emails
        
        # Content quality analysis
        for split, stats in split_statistics.items():
            samples = stats.get('sample_emails', [])
            if samples:
                subject_lengths = [s['subject_length'] for s in samples]
                body_lengths = [s['body_length'] for s in samples]
                
                quality_results['content_quality'][split] = {
                    'avg_subject_length': round(np.mean(subject_lengths), 2),
                    'avg_body_length': round(np.mean(body_lengths), 2),
                    'min_subject_length': min(subject_lengths),
                    'max_subject_length': max(subject_lengths),
                    'min_body_length': min(body_lengths),
                    'max_body_length': max(body_lengths)
                }
        
        # Consistency checks
        all_categories = set()
        all_languages = set()
        for split, stats in split_statistics.items():
            all_categories.update(stats.get('categories', {}).keys())
            all_languages.update(stats.get('languages', {}).keys())
        
        quality_results['consistency_checks'] = {
            'unique_categories_across_splits': len(all_categories),
            'expected_categories': len(self.expected_categories),
            'unique_languages': len(all_languages),
            'categories_list': sorted(list(all_categories)),
            'languages_list': sorted(list(all_languages))
        }
        
        # Overall quality score
        error_rate = (total_errors / (total_errors + total_emails)) * 100 if (total_errors + total_emails) > 0 else 0
        quality_score = max(0, 100 - error_rate)
        quality_results['overall_quality_score'] = round(quality_score, 2)
        
        return quality_results
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check missing categories
        missing_cats = validation_results.get('category_distribution', {}).get('missing_categories', [])
        if missing_cats:
            recommendations.append(f"Missing categories detected: {missing_cats}. Consider adding samples for these categories.")
        
        # Check ratio quality
        ratio_score = validation_results.get('balance_analysis', {}).get('ratio_quality_score', 0)
        if ratio_score < 95:
            recommendations.append(f"Split ratios deviate from expected 70/15/15. Current quality score: {ratio_score}%")
        
        # Check validation errors
        total_errors = sum(
            stats.get('error_count', 0) 
            for stats in validation_results.get('quality_checks', {}).get('validation_errors', {}).values()
        )
        if total_errors > 0:
            recommendations.append(f"Found {total_errors} validation errors. Review and fix malformed email files.")
        
        # Check category balance
        category_balance = validation_results.get('category_distribution', {}).get('category_balance', {})
        low_balance_categories = [
            cat for cat, info in category_balance.items() 
            if info.get('balance_score', 0) < 80
        ]
        if low_balance_categories:
            recommendations.append(f"Categories with poor split balance: {low_balance_categories}")
        
        if not recommendations:
            recommendations.append("Dataset splits are well-balanced and meet all quality requirements.")
        
        return recommendations
    
    def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        
        # Check critical issues
        if not validation_results.get('splits_found'):
            return 'error'
        
        if validation_results.get('error'):
            return 'error'
        
        # Check quality metrics
        ratio_score = validation_results.get('balance_analysis', {}).get('ratio_quality_score', 0)
        quality_score = validation_results.get('quality_checks', {}).get('overall_quality_score', 0)
        missing_cats = validation_results.get('category_distribution', {}).get('missing_categories', [])
        
        # Determine status based on scores
        if ratio_score >= 95 and quality_score >= 95 and not missing_cats:
            return 'excellent'
        elif ratio_score >= 90 and quality_score >= 90 and len(missing_cats) <= 1:
            return 'good'
        elif ratio_score >= 80 and quality_score >= 80:
            return 'acceptable'
        else:
            return 'needs_improvement'
    
    def print_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """Print a human-readable summary of validation results."""
        print("\n" + "="*60)
        print("DATASET SPLITS VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Dataset Path: {validation_results['dataset_path']}")
        print(f"Overall Status: {validation_results['overall_status'].upper()}")
        print()
        
        # Split statistics
        print("SPLIT STATISTICS:")
        for split, stats in validation_results.get('split_statistics', {}).items():
            print(f"  {split.upper()}: {stats['total_emails']} emails, "
                  f"{len(stats['categories'])} categories, "
                  f"{stats['validation_errors']} errors")
        print()
        
        # Balance analysis
        balance = validation_results.get('balance_analysis', {})
        print("SPLIT RATIOS:")
        print(f"  Expected: {balance.get('expected_ratios', {})}")
        print(f"  Actual: {balance.get('actual_ratios', {})}")
        print(f"  Quality Score: {balance.get('ratio_quality_score', 0)}%")
        print()
        
        # Category distribution
        cat_dist = validation_results.get('category_distribution', {})
        print("CATEGORY ANALYSIS:")
        print(f"  Categories Found: {len(cat_dist.get('categories_found', []))}")
        print(f"  Expected Categories: {len(cat_dist.get('expected_categories', []))}")
        missing = cat_dist.get('missing_categories', [])
        if missing:
            print(f"  Missing Categories: {missing}")
        print()
        
        # Quality checks
        quality = validation_results.get('quality_checks', {})
        print("QUALITY METRICS:")
        print(f"  Overall Quality Score: {quality.get('overall_quality_score', 0)}%")
        print()
        
        # Recommendations
        recommendations = validation_results.get('recommendations', [])
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("="*60)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dataset splits")
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
        '--summary-only',
        action='store_true',
        help='Show only summary, not full JSON output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = DatasetSplitsValidator(args.dataset_path)
        
        # Run validation
        results = validator.validate_splits()
        
        # Print summary
        validator.print_validation_summary(results)
        
        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation results saved to {args.output_json}")
        
        # Print full JSON if not summary-only
        if not args.summary_only:
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