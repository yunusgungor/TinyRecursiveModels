"""
Security validation utilities for Gmail Dataset Creator.

This module provides comprehensive security validation including configuration
validation, data integrity checks, and security compliance verification.
"""

import os
import json
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class SecurityLevel(Enum):
    """Security levels for validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    WEAK_ENCRYPTION = "weak_encryption"
    INSECURE_STORAGE = "insecure_storage"
    WEAK_PASSWORDS = "weak_passwords"
    EXPOSED_CREDENTIALS = "exposed_credentials"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    DATA_LEAKAGE = "data_leakage"
    INSECURE_CONFIGURATION = "insecure_configuration"
    OUTDATED_DEPENDENCIES = "outdated_dependencies"


@dataclass
class SecurityIssue:
    """Represents a security issue found during validation."""
    issue_type: VulnerabilityType
    severity: SecurityLevel
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cve_references: List[str] = None
    
    def __post_init__(self):
        if self.cve_references is None:
            self.cve_references = []


@dataclass
class SecurityValidationResult:
    """Results of security validation."""
    passed: bool
    issues: List[SecurityIssue]
    score: float  # 0-100 security score
    timestamp: datetime
    validation_duration: float
    
    def get_issues_by_severity(self, severity: SecurityLevel) -> List[SecurityIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical security issues."""
        return any(issue.severity == SecurityLevel.CRITICAL for issue in self.issues)


class SecurityValidator:
    """
    Comprehensive security validator for Gmail Dataset Creator.
    
    Performs security validation including configuration checks, file permissions,
    encryption validation, and compliance verification.
    """
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        Initialize security validator.
        
        Args:
            base_directory: Base directory for validation (defaults to current directory)
        """
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Security validation rules
        self.validation_rules = self._load_validation_rules()
        
        # File patterns to check
        self.sensitive_file_patterns = [
            '*.key', '*.pem', '*.p12', '*.pfx',
            'credentials.json', 'token.json', '*.token',
            '.env', '.env.*', 'config.json', '*.conf'
        ]
        
        # Weak password patterns
        self.weak_password_patterns = [
            'password', '123456', 'admin', 'root', 'guest',
            'test', 'demo', 'default', 'changeme'
        ]
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load security validation rules."""
        return {
            'min_password_length': 12,
            'require_special_chars': True,
            'min_encryption_key_length': 256,
            'allowed_encryption_algorithms': ['AES-256', 'ChaCha20-Poly1305'],
            'max_file_permissions': 0o600,
            'require_https': True,
            'min_tls_version': '1.2',
            'require_certificate_validation': True
        }
    
    def validate_security(self, 
                         config_path: Optional[str] = None,
                         check_files: bool = True,
                         check_permissions: bool = True,
                         check_encryption: bool = True) -> SecurityValidationResult:
        """
        Perform comprehensive security validation.
        
        Args:
            config_path: Path to configuration file to validate
            check_files: Whether to check file security
            check_permissions: Whether to check file permissions
            check_encryption: Whether to validate encryption settings
            
        Returns:
            SecurityValidationResult with validation results
        """
        start_time = datetime.now()
        issues = []
        
        try:
            # Validate configuration
            if config_path:
                config_issues = self._validate_configuration(config_path)
                issues.extend(config_issues)
            
            # Check file security
            if check_files:
                file_issues = self._validate_file_security()
                issues.extend(file_issues)
            
            # Check file permissions
            if check_permissions:
                permission_issues = self._validate_file_permissions()
                issues.extend(permission_issues)
            
            # Validate encryption settings
            if check_encryption:
                encryption_issues = self._validate_encryption_settings()
                issues.extend(encryption_issues)
            
            # Check for exposed credentials
            credential_issues = self._check_exposed_credentials()
            issues.extend(credential_issues)
            
            # Validate dependencies
            dependency_issues = self._validate_dependencies()
            issues.extend(dependency_issues)
            
            # Calculate security score
            security_score = self._calculate_security_score(issues)
            
            # Determine if validation passed
            passed = not any(issue.severity == SecurityLevel.CRITICAL for issue in issues)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return SecurityValidationResult(
                passed=passed,
                issues=issues,
                score=security_score,
                timestamp=end_time,
                validation_duration=duration
            )
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            
            # Return failed result with error
            critical_issue = SecurityIssue(
                issue_type=VulnerabilityType.INSECURE_CONFIGURATION,
                severity=SecurityLevel.CRITICAL,
                description=f"Security validation failed: {str(e)}",
                recommendation="Fix validation errors and retry"
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return SecurityValidationResult(
                passed=False,
                issues=[critical_issue],
                score=0.0,
                timestamp=end_time,
                validation_duration=duration
            )
    
    def _validate_configuration(self, config_path: str) -> List[SecurityIssue]:
        """Validate security configuration."""
        issues = []
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                issues.append(SecurityIssue(
                    issue_type=VulnerabilityType.INSECURE_CONFIGURATION,
                    severity=SecurityLevel.HIGH,
                    description=f"Configuration file not found: {config_path}",
                    file_path=config_path,
                    recommendation="Ensure configuration file exists and is properly configured"
                ))
                return issues
            
            # Load and validate configuration
            with open(config_file, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Assume YAML
                    import yaml
                    config = yaml.safe_load(f)
            
            # Check encryption settings
            if 'privacy' in config:
                privacy_config = config['privacy']
                
                if not privacy_config.get('encrypt_tokens', False):
                    issues.append(SecurityIssue(
                        issue_type=VulnerabilityType.WEAK_ENCRYPTION,
                        severity=SecurityLevel.HIGH,
                        description="Token encryption is disabled",
                        file_path=config_path,
                        recommendation="Enable token encryption in privacy settings"
                    ))
                
                if not privacy_config.get('anonymize_senders', True):
                    issues.append(SecurityIssue(
                        issue_type=VulnerabilityType.DATA_LEAKAGE,
                        severity=SecurityLevel.MEDIUM,
                        description="Sender anonymization is disabled",
                        file_path=config_path,
                        recommendation="Enable sender anonymization to protect privacy"
                    ))
            
            # Check API key configuration
            if 'gemini_api' in config:
                gemini_config = config['gemini_api']
                api_key = gemini_config.get('api_key', '')
                
                if not api_key:
                    issues.append(SecurityIssue(
                        issue_type=VulnerabilityType.EXPOSED_CREDENTIALS,
                        severity=SecurityLevel.HIGH,
                        description="Gemini API key is empty",
                        file_path=config_path,
                        recommendation="Set API key via environment variable"
                    ))
                elif not api_key.startswith('${'):
                    issues.append(SecurityIssue(
                        issue_type=VulnerabilityType.EXPOSED_CREDENTIALS,
                        severity=SecurityLevel.CRITICAL,
                        description="API key appears to be hardcoded in configuration",
                        file_path=config_path,
                        recommendation="Use environment variable for API key"
                    ))
            
        except Exception as e:
            issues.append(SecurityIssue(
                issue_type=VulnerabilityType.INSECURE_CONFIGURATION,
                severity=SecurityLevel.HIGH,
                description=f"Error validating configuration: {str(e)}",
                file_path=config_path,
                recommendation="Fix configuration file format and content"
            ))
        
        return issues
    
    def _validate_file_security(self) -> List[SecurityIssue]:
        """Validate file security."""
        issues = []
        
        # Check for sensitive files
        for pattern in self.sensitive_file_patterns:
            matching_files = list(self.base_directory.rglob(pattern))
            
            for file_path in matching_files:
                if file_path.is_file():
                    # Check if file contains sensitive data
                    if self._contains_sensitive_data(file_path):
                        issues.append(SecurityIssue(
                            issue_type=VulnerabilityType.EXPOSED_CREDENTIALS,
                            severity=SecurityLevel.HIGH,
                            description=f"Sensitive file found: {file_path.name}",
                            file_path=str(file_path),
                            recommendation="Ensure sensitive files are properly encrypted and secured"
                        ))
        
        return issues
    
    def _validate_file_permissions(self) -> List[SecurityIssue]:
        """Validate file permissions."""
        issues = []
        
        # Check permissions on sensitive files
        for pattern in self.sensitive_file_patterns:
            matching_files = list(self.base_directory.rglob(pattern))
            
            for file_path in matching_files:
                if file_path.is_file():
                    try:
                        file_mode = file_path.stat().st_mode & 0o777
                        max_allowed = self.validation_rules['max_file_permissions']
                        
                        if file_mode > max_allowed:
                            issues.append(SecurityIssue(
                                issue_type=VulnerabilityType.INSUFFICIENT_PERMISSIONS,
                                severity=SecurityLevel.MEDIUM,
                                description=f"File has overly permissive permissions: {oct(file_mode)}",
                                file_path=str(file_path),
                                recommendation=f"Set file permissions to {oct(max_allowed)} or more restrictive"
                            ))
                    except Exception as e:
                        self.logger.error(f"Error checking permissions for {file_path}: {e}")
        
        return issues
    
    def _validate_encryption_settings(self) -> List[SecurityIssue]:
        """Validate encryption settings."""
        issues = []
        
        # Check for encryption configuration files
        encryption_configs = list(self.base_directory.rglob('*encryption*'))
        encryption_configs.extend(list(self.base_directory.rglob('*crypto*')))
        
        for config_file in encryption_configs:
            if config_file.is_file() and config_file.suffix in ['.json', '.yaml', '.yml']:
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.json':
                            config = json.load(f)
                        else:
                            import yaml
                            config = yaml.safe_load(f)
                    
                    # Check encryption algorithm
                    algorithm = config.get('algorithm', '').upper()
                    if algorithm and algorithm not in self.validation_rules['allowed_encryption_algorithms']:
                        issues.append(SecurityIssue(
                            issue_type=VulnerabilityType.WEAK_ENCRYPTION,
                            severity=SecurityLevel.HIGH,
                            description=f"Weak encryption algorithm: {algorithm}",
                            file_path=str(config_file),
                            recommendation="Use AES-256 or ChaCha20-Poly1305 encryption"
                        ))
                    
                    # Check key length
                    key_length = config.get('key_length', 0)
                    min_key_length = self.validation_rules['min_encryption_key_length']
                    if key_length > 0 and key_length < min_key_length:
                        issues.append(SecurityIssue(
                            issue_type=VulnerabilityType.WEAK_ENCRYPTION,
                            severity=SecurityLevel.HIGH,
                            description=f"Encryption key length too short: {key_length} bits",
                            file_path=str(config_file),
                            recommendation=f"Use at least {min_key_length}-bit encryption keys"
                        ))
                
                except Exception as e:
                    self.logger.error(f"Error validating encryption config {config_file}: {e}")
        
        return issues
    
    def _check_exposed_credentials(self) -> List[SecurityIssue]:
        """Check for exposed credentials in files."""
        issues = []
        
        # Patterns that might indicate exposed credentials
        credential_patterns = [
            r'password\s*=\s*["\']([^"\']+)["\']',
            r'api_key\s*=\s*["\']([^"\']+)["\']',
            r'secret\s*=\s*["\']([^"\']+)["\']',
            r'token\s*=\s*["\']([^"\']+)["\']',
        ]
        
        # File types to check
        text_files = []
        for ext in ['.py', '.json', '.yaml', '.yml', '.txt', '.conf', '.cfg']:
            text_files.extend(list(self.base_directory.rglob(f'*{ext}')))
        
        import re
        
        for file_path in text_files:
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for pattern in credential_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            credential_value = match.group(1)
                            
                            # Skip if it looks like a placeholder or environment variable
                            if (credential_value.startswith('${') or 
                                credential_value.startswith('$') or
                                credential_value in ['your_key_here', 'placeholder', 'example']):
                                continue
                            
                            # Check if it's a weak credential
                            if any(weak in credential_value.lower() for weak in self.weak_password_patterns):
                                severity = SecurityLevel.CRITICAL
                                description = f"Weak credential found in {file_path.name}"
                            else:
                                severity = SecurityLevel.HIGH
                                description = f"Potential exposed credential in {file_path.name}"
                            
                            issues.append(SecurityIssue(
                                issue_type=VulnerabilityType.EXPOSED_CREDENTIALS,
                                severity=severity,
                                description=description,
                                file_path=str(file_path),
                                line_number=content[:match.start()].count('\n') + 1,
                                recommendation="Use environment variables or secure credential storage"
                            ))
                
                except Exception as e:
                    self.logger.error(f"Error checking credentials in {file_path}: {e}")
        
        return issues
    
    def _validate_dependencies(self) -> List[SecurityIssue]:
        """Validate dependencies for known vulnerabilities."""
        issues = []
        
        # Check requirements.txt files
        requirements_files = list(self.base_directory.rglob('requirements*.txt'))
        
        for req_file in requirements_files:
            if req_file.is_file():
                try:
                    with open(req_file, 'r') as f:
                        requirements = f.read().splitlines()
                    
                    for req in requirements:
                        req = req.strip()
                        if req and not req.startswith('#'):
                            # Basic check for pinned versions
                            if '==' not in req and '>=' not in req:
                                issues.append(SecurityIssue(
                                    issue_type=VulnerabilityType.OUTDATED_DEPENDENCIES,
                                    severity=SecurityLevel.MEDIUM,
                                    description=f"Unpinned dependency: {req}",
                                    file_path=str(req_file),
                                    recommendation="Pin dependency versions for security"
                                ))
                
                except Exception as e:
                    self.logger.error(f"Error checking requirements {req_file}: {e}")
        
        return issues
    
    def _contains_sensitive_data(self, file_path: Path) -> bool:
        """Check if file contains sensitive data."""
        try:
            # Check file size first (avoid reading very large files)
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return False
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
            
            # Look for patterns that indicate sensitive data
            sensitive_indicators = [
                'BEGIN PRIVATE KEY', 'BEGIN RSA PRIVATE KEY',
                'BEGIN CERTIFICATE', 'client_secret',
                'private_key', 'access_token', 'refresh_token'
            ]
            
            content_lower = content.lower()
            return any(indicator.lower() in content_lower for indicator in sensitive_indicators)
            
        except Exception:
            return False
    
    def _calculate_security_score(self, issues: List[SecurityIssue]) -> float:
        """Calculate security score based on issues found."""
        if not issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 3,
            SecurityLevel.HIGH: 7,
            SecurityLevel.CRITICAL: 15
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        
        # Calculate score (100 - penalty)
        max_penalty = 100
        penalty = min(total_weight * 2, max_penalty)  # Cap at 100
        
        return max(0.0, 100.0 - penalty)
    
    def generate_security_report(self, validation_result: SecurityValidationResult) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            'summary': {
                'passed': validation_result.passed,
                'security_score': validation_result.score,
                'total_issues': len(validation_result.issues),
                'critical_issues': len(validation_result.get_issues_by_severity(SecurityLevel.CRITICAL)),
                'high_issues': len(validation_result.get_issues_by_severity(SecurityLevel.HIGH)),
                'medium_issues': len(validation_result.get_issues_by_severity(SecurityLevel.MEDIUM)),
                'low_issues': len(validation_result.get_issues_by_severity(SecurityLevel.LOW)),
                'validation_time': validation_result.validation_duration,
                'timestamp': validation_result.timestamp.isoformat()
            },
            'issues': [],
            'recommendations': []
        }
        
        # Add detailed issue information
        for issue in validation_result.issues:
            issue_data = {
                'type': issue.issue_type.value,
                'severity': issue.severity.value,
                'description': issue.description,
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'recommendation': issue.recommendation,
                'cve_references': issue.cve_references
            }
            report['issues'].append(issue_data)
        
        # Generate general recommendations
        if validation_result.has_critical_issues():
            report['recommendations'].append(
                "Address critical security issues immediately before proceeding"
            )
        
        if validation_result.score < 70:
            report['recommendations'].append(
                "Security score is below acceptable threshold. Review and fix security issues"
            )
        
        return report
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_strong, list_of_issues)
        """
        issues = []
        
        # Check length
        min_length = self.validation_rules['min_password_length']
        if len(password) < min_length:
            issues.append(f"Password must be at least {min_length} characters long")
        
        # Check for different character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        if not has_upper:
            issues.append("Password must contain at least one uppercase letter")
        if not has_lower:
            issues.append("Password must contain at least one lowercase letter")
        if not has_digit:
            issues.append("Password must contain at least one digit")
        if self.validation_rules['require_special_chars'] and not has_special:
            issues.append("Password must contain at least one special character")
        
        # Check for weak patterns
        password_lower = password.lower()
        for weak_pattern in self.weak_password_patterns:
            if weak_pattern in password_lower:
                issues.append(f"Password contains weak pattern: {weak_pattern}")
        
        return len(issues) == 0, issues