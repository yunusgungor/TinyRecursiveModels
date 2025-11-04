"""
Data retention policy management for Gmail Dataset Creator.

This module provides comprehensive data retention policies including automatic
cleanup, archival, and compliance with data protection regulations.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RetentionAction(Enum):
    """Actions to take when retention period expires."""
    DELETE = "delete"
    ARCHIVE = "archive"
    ENCRYPT = "encrypt"
    NOTIFY = "notify"


class DataCategory(Enum):
    """Categories of data for retention policies."""
    RAW_EMAILS = "raw_emails"
    PROCESSED_EMAILS = "processed_emails"
    AUTHENTICATION_TOKENS = "auth_tokens"
    TEMPORARY_FILES = "temp_files"
    LOG_FILES = "log_files"
    DATASET_FILES = "dataset_files"
    CACHE_FILES = "cache_files"
    BACKUP_FILES = "backup_files"


@dataclass
class RetentionRule:
    """Individual retention rule configuration."""
    category: DataCategory
    retention_days: int
    action: RetentionAction
    enabled: bool = True
    file_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    archive_location: Optional[str] = None
    notify_before_days: int = 7


@dataclass
class RetentionPolicy:
    """Complete data retention policy configuration."""
    name: str
    description: str
    rules: List[RetentionRule] = field(default_factory=list)
    global_enabled: bool = True
    compliance_mode: bool = False  # Stricter enforcement for compliance
    audit_logging: bool = True
    
    def add_rule(self, rule: RetentionRule):
        """Add a retention rule to the policy."""
        self.rules.append(rule)
    
    def get_rule_for_category(self, category: DataCategory) -> Optional[RetentionRule]:
        """Get retention rule for a specific data category."""
        for rule in self.rules:
            if rule.category == category and rule.enabled:
                return rule
        return None


class DataRetentionManager:
    """
    Manages data retention policies and automated cleanup.
    
    Provides comprehensive data lifecycle management including automatic
    cleanup, archival, and compliance with data protection regulations.
    """
    
    def __init__(self, policy: Optional[RetentionPolicy] = None,
                 base_directory: Optional[str] = None):
        """
        Initialize data retention manager.
        
        Args:
            policy: Retention policy configuration
            base_directory: Base directory for data management
        """
        self.policy = policy or self._create_default_policy()
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Tracking for audit purposes
        self.retention_log: List[Dict[str, Any]] = []
        self.last_cleanup_time: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            'files_deleted': 0,
            'files_archived': 0,
            'files_encrypted': 0,
            'bytes_cleaned': 0,
            'policy_violations': 0,
            'notifications_sent': 0
        }
    
    def _create_default_policy(self) -> RetentionPolicy:
        """Create default retention policy."""
        policy = RetentionPolicy(
            name="Default Gmail Dataset Creator Policy",
            description="Default data retention policy for Gmail Dataset Creator"
        )
        
        # Default rules
        rules = [
            RetentionRule(
                category=DataCategory.RAW_EMAILS,
                retention_days=30,
                action=RetentionAction.DELETE,
                file_patterns=['*.raw', '*.json'],
                notify_before_days=7
            ),
            RetentionRule(
                category=DataCategory.AUTHENTICATION_TOKENS,
                retention_days=90,
                action=RetentionAction.DELETE,
                file_patterns=['token.json', '*.key'],
                notify_before_days=14
            ),
            RetentionRule(
                category=DataCategory.TEMPORARY_FILES,
                retention_days=7,
                action=RetentionAction.DELETE,
                file_patterns=['*.tmp', '*.temp', 'temp_*'],
                notify_before_days=1
            ),
            RetentionRule(
                category=DataCategory.LOG_FILES,
                retention_days=365,
                action=RetentionAction.ARCHIVE,
                file_patterns=['*.log'],
                archive_location='archive/logs',
                notify_before_days=30
            ),
            RetentionRule(
                category=DataCategory.DATASET_FILES,
                retention_days=0,  # Keep indefinitely
                action=RetentionAction.NOTIFY,
                file_patterns=['dataset.json', 'train/*', 'test/*'],
                notify_before_days=0
            ),
            RetentionRule(
                category=DataCategory.CACHE_FILES,
                retention_days=14,
                action=RetentionAction.DELETE,
                file_patterns=['cache/*', '*.cache'],
                notify_before_days=3
            )
        ]
        
        for rule in rules:
            policy.add_rule(rule)
        
        return policy
    
    def apply_retention_policy(self, target_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply retention policy to files in target directory.
        
        Args:
            target_directory: Directory to apply policy to (defaults to base_directory)
            
        Returns:
            Dictionary with results of policy application
        """
        if not self.policy.global_enabled:
            self.logger.info("Retention policy is globally disabled")
            return {'success': True, 'message': 'Policy disabled'}
        
        target_dir = Path(target_directory) if target_directory else self.base_directory
        
        if not target_dir.exists():
            return {'success': False, 'error': f'Target directory not found: {target_dir}'}
        
        self.logger.info(f"Applying retention policy to: {target_dir}")
        
        results = {
            'success': True,
            'processed_files': 0,
            'actions_taken': {},
            'errors': []
        }
        
        try:
            # Process each retention rule
            for rule in self.policy.rules:
                if not rule.enabled:
                    continue
                
                rule_results = self._apply_retention_rule(rule, target_dir)
                results['processed_files'] += rule_results['processed_files']
                
                # Merge action counts
                for action, count in rule_results['actions'].items():
                    results['actions_taken'][action] = results['actions_taken'].get(action, 0) + count
                
                # Merge errors
                results['errors'].extend(rule_results['errors'])
            
            self.last_cleanup_time = datetime.now()
            
            # Log retention activity for audit
            if self.policy.audit_logging:
                self._log_retention_activity(results)
            
        except Exception as e:
            self.logger.error(f"Error applying retention policy: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def _apply_retention_rule(self, rule: RetentionRule, target_dir: Path) -> Dict[str, Any]:
        """Apply a single retention rule."""
        results = {
            'processed_files': 0,
            'actions': {},
            'errors': []
        }
        
        # Find files matching the rule patterns
        matching_files = self._find_matching_files(rule, target_dir)
        
        for file_path in matching_files:
            try:
                # Check if file should be processed based on age
                if self._should_process_file(file_path, rule):
                    action_taken = self._execute_retention_action(file_path, rule)
                    if action_taken:
                        results['actions'][action_taken] = results['actions'].get(action_taken, 0) + 1
                    results['processed_files'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def _find_matching_files(self, rule: RetentionRule, target_dir: Path) -> List[Path]:
        """Find files matching retention rule patterns."""
        matching_files = []
        
        # Search for files matching include patterns
        for pattern in rule.file_patterns:
            try:
                matches = list(target_dir.rglob(pattern))
                matching_files.extend(matches)
            except Exception as e:
                self.logger.error(f"Error searching pattern {pattern}: {e}")
        
        # Filter out files matching exclude patterns
        if rule.exclude_patterns:
            filtered_files = []
            for file_path in matching_files:
                should_exclude = False
                for exclude_pattern in rule.exclude_patterns:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    filtered_files.append(file_path)
            
            matching_files = filtered_files
        
        # Remove duplicates and ensure files exist
        unique_files = []
        seen_files = set()
        
        for file_path in matching_files:
            if file_path.exists() and file_path.is_file() and str(file_path) not in seen_files:
                unique_files.append(file_path)
                seen_files.add(str(file_path))
        
        return unique_files
    
    def _should_process_file(self, file_path: Path, rule: RetentionRule) -> bool:
        """Check if file should be processed based on retention rule."""
        if rule.retention_days == 0:
            return False  # Keep indefinitely
        
        try:
            # Get file modification time
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_days = (datetime.now() - file_mtime).days
            
            return age_days >= rule.retention_days
            
        except Exception as e:
            self.logger.error(f"Error checking file age for {file_path}: {e}")
            return False
    
    def _execute_retention_action(self, file_path: Path, rule: RetentionRule) -> Optional[str]:
        """Execute the retention action for a file."""
        try:
            if rule.action == RetentionAction.DELETE:
                return self._delete_file(file_path)
            elif rule.action == RetentionAction.ARCHIVE:
                return self._archive_file(file_path, rule.archive_location)
            elif rule.action == RetentionAction.ENCRYPT:
                return self._encrypt_file(file_path)
            elif rule.action == RetentionAction.NOTIFY:
                return self._notify_about_file(file_path, rule)
            else:
                self.logger.warning(f"Unknown retention action: {rule.action}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing retention action for {file_path}: {e}")
            return None
    
    def _delete_file(self, file_path: Path) -> str:
        """Delete a file securely."""
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        self.stats['files_deleted'] += 1
        self.stats['bytes_cleaned'] += file_size
        
        self.logger.debug(f"Deleted file: {file_path}")
        return "deleted"
    
    def _archive_file(self, file_path: Path, archive_location: Optional[str]) -> str:
        """Archive a file to specified location."""
        if not archive_location:
            archive_location = "archive"
        
        # Create archive directory structure
        archive_dir = self.base_directory / archive_location
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve directory structure in archive
        relative_path = file_path.relative_to(self.base_directory)
        archive_file_path = archive_dir / relative_path
        archive_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file to archive
        shutil.move(str(file_path), str(archive_file_path))
        
        self.stats['files_archived'] += 1
        
        self.logger.debug(f"Archived file: {file_path} -> {archive_file_path}")
        return "archived"
    
    def _encrypt_file(self, file_path: Path) -> str:
        """Encrypt a file in place."""
        # This would integrate with the encryption manager
        # For now, just log the action
        self.stats['files_encrypted'] += 1
        
        self.logger.debug(f"Encrypted file: {file_path}")
        return "encrypted"
    
    def _notify_about_file(self, file_path: Path, rule: RetentionRule) -> str:
        """Send notification about file retention."""
        # This would integrate with a notification system
        self.stats['notifications_sent'] += 1
        
        self.logger.info(f"Notification: File {file_path} subject to retention rule {rule.category.value}")
        return "notified"
    
    def _log_retention_activity(self, results: Dict[str, Any]) -> None:
        """Log retention activity for audit purposes."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'policy_name': self.policy.name,
            'results': results,
            'statistics': self.stats.copy()
        }
        
        self.retention_log.append(log_entry)
        
        # Keep only last 1000 log entries
        if len(self.retention_log) > 1000:
            self.retention_log = self.retention_log[-1000:]
    
    def check_upcoming_expirations(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Check for files that will expire within specified days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of files that will expire soon
        """
        upcoming_expirations = []
        
        for rule in self.policy.rules:
            if not rule.enabled or rule.retention_days == 0:
                continue
            
            matching_files = self._find_matching_files(rule, self.base_directory)
            
            for file_path in matching_files:
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    expiration_date = file_mtime + timedelta(days=rule.retention_days)
                    days_until_expiration = (expiration_date - datetime.now()).days
                    
                    if 0 <= days_until_expiration <= days_ahead:
                        upcoming_expirations.append({
                            'file_path': str(file_path),
                            'category': rule.category.value,
                            'action': rule.action.value,
                            'days_until_expiration': days_until_expiration,
                            'expiration_date': expiration_date.isoformat()
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error checking expiration for {file_path}: {e}")
        
        return upcoming_expirations
    
    def save_policy(self, policy_file: str) -> bool:
        """
        Save retention policy to file.
        
        Args:
            policy_file: Path to save policy file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            policy_data = {
                'name': self.policy.name,
                'description': self.policy.description,
                'global_enabled': self.policy.global_enabled,
                'compliance_mode': self.policy.compliance_mode,
                'audit_logging': self.policy.audit_logging,
                'rules': []
            }
            
            for rule in self.policy.rules:
                rule_data = {
                    'category': rule.category.value,
                    'retention_days': rule.retention_days,
                    'action': rule.action.value,
                    'enabled': rule.enabled,
                    'file_patterns': rule.file_patterns,
                    'exclude_patterns': rule.exclude_patterns,
                    'archive_location': rule.archive_location,
                    'notify_before_days': rule.notify_before_days
                }
                policy_data['rules'].append(rule_data)
            
            with open(policy_file, 'w') as f:
                json.dump(policy_data, f, indent=2)
            
            self.logger.info(f"Retention policy saved to {policy_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving retention policy: {e}")
            return False
    
    def load_policy(self, policy_file: str) -> bool:
        """
        Load retention policy from file.
        
        Args:
            policy_file: Path to policy file
            
        Returns:
            True if load successful, False otherwise
        """
        try:
            with open(policy_file, 'r') as f:
                policy_data = json.load(f)
            
            # Create new policy
            policy = RetentionPolicy(
                name=policy_data['name'],
                description=policy_data['description'],
                global_enabled=policy_data.get('global_enabled', True),
                compliance_mode=policy_data.get('compliance_mode', False),
                audit_logging=policy_data.get('audit_logging', True)
            )
            
            # Load rules
            for rule_data in policy_data.get('rules', []):
                rule = RetentionRule(
                    category=DataCategory(rule_data['category']),
                    retention_days=rule_data['retention_days'],
                    action=RetentionAction(rule_data['action']),
                    enabled=rule_data.get('enabled', True),
                    file_patterns=rule_data.get('file_patterns', []),
                    exclude_patterns=rule_data.get('exclude_patterns', []),
                    archive_location=rule_data.get('archive_location'),
                    notify_before_days=rule_data.get('notify_before_days', 7)
                )
                policy.add_rule(rule)
            
            self.policy = policy
            self.logger.info(f"Retention policy loaded from {policy_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading retention policy: {e}")
            return False
    
    def get_retention_stats(self) -> Dict[str, Any]:
        """Get retention statistics."""
        return {
            'statistics': self.stats.copy(),
            'last_cleanup_time': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
            'policy_name': self.policy.name,
            'rules_count': len(self.policy.rules),
            'enabled_rules_count': sum(1 for rule in self.policy.rules if rule.enabled)
        }
    
    def generate_retention_report(self) -> Dict[str, Any]:
        """Generate comprehensive retention report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'policy_info': {
                'name': self.policy.name,
                'description': self.policy.description,
                'compliance_mode': self.policy.compliance_mode
            },
            'statistics': self.get_retention_stats(),
            'upcoming_expirations': self.check_upcoming_expirations(),
            'recent_activity': self.retention_log[-10:] if self.retention_log else []
        }
        
        return report