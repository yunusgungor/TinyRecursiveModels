# Privacy and Security Features

This document describes the comprehensive privacy and security features implemented in the Gmail Dataset Creator to protect sensitive data and ensure compliance with data protection regulations.

## Overview

The Gmail Dataset Creator includes robust privacy and security measures to:

- **Protect Personal Information**: Automatically detect and anonymize sensitive data
- **Secure Data Storage**: Encrypt authentication tokens and sensitive files
- **Control Data Access**: Filter and exclude personal or sensitive emails
- **Ensure Data Integrity**: Verify data integrity and detect tampering
- **Manage Data Lifecycle**: Implement retention policies and secure cleanup
- **Export Securely**: Create encrypted, integrity-verified dataset exports

## Privacy Controls

### Data Anonymization

The system automatically anonymizes sensitive information in email content:

```python
from gmail_dataset_creator.privacy import PrivacyController, PrivacySettings

# Configure privacy settings
privacy_settings = PrivacySettings(
    exclude_personal=True,
    exclude_sensitive=True,
    anonymize_senders=True,
    anonymize_recipients=True,
    remove_sensitive_content=True,
    exclude_keywords=['confidential', 'private'],
    exclude_domains=['internal.company.com'],
    min_confidence_threshold=0.8
)

# Create privacy controller
privacy_controller = PrivacyController(privacy_settings)

# Apply privacy controls to email
processed_email = privacy_controller.apply_privacy_controls(email_data)
```

### Sensitive Data Detection

The system detects various types of sensitive information:

- **Email addresses**: Anonymized while preserving domain structure
- **Phone numbers**: Replaced with placeholders
- **Credit card numbers**: Completely removed
- **Social Security Numbers**: Completely removed
- **IP addresses**: Anonymized while preserving class
- **URLs**: Optionally anonymized
- **Medical information**: Detected and flagged
- **Financial information**: Detected and flagged

### Email Filtering

Emails can be automatically excluded based on:

- **Personal content**: Detected using pattern matching
- **Sensitive data**: Emails containing PII or confidential information
- **Keywords**: User-defined exclusion keywords
- **Domains**: Emails from specific domains
- **Confidence thresholds**: Low-confidence classifications

## Security Measures

### Encryption

#### Token Encryption

Authentication tokens are encrypted using strong encryption:

```python
from gmail_dataset_creator.security import EncryptionManager, EncryptionConfig

# Configure encryption
encryption_config = EncryptionConfig(
    algorithm=EncryptionAlgorithm.FERNET,
    kdf=KeyDerivationFunction.PBKDF2,
    iterations=150000,
    salt_length=64,
    key_length=32
)

# Create encryption manager
encryption_manager = EncryptionManager(encryption_config)

# Encrypt sensitive data
encrypted_data, salt = encryption_manager.encrypt_data(sensitive_data, password)
```

#### Supported Encryption Algorithms

- **Fernet**: Symmetric encryption with built-in authentication
- **AES-256-GCM**: Advanced Encryption Standard with Galois/Counter Mode
- **AES-256-CBC**: Advanced Encryption Standard with Cipher Block Chaining

#### Key Derivation Functions

- **PBKDF2**: Password-Based Key Derivation Function 2
- **Scrypt**: Memory-hard key derivation function

### Secure Export

Datasets can be exported with multiple security layers:

```python
from gmail_dataset_creator.security import SecureExporter, ExportConfig

# Configure secure export
export_config = ExportConfig(
    encrypt_data=True,
    include_checksums=True,
    password_protect=True,
    compression=CompressionType.GZIP,
    security_level=SecurityLevel.HIGH
)

# Create secure exporter
secure_exporter = SecureExporter(export_config)

# Export dataset securely
export_result = secure_exporter.export_dataset(
    dataset_path="./dataset",
    output_path="./secure_export",
    password="secure_password",
    metadata={"version": "1.0"}
)
```

### Data Integrity

Data integrity is ensured through:

- **Checksums**: SHA-256 hashes for all files
- **Digital signatures**: Optional signing of exports
- **Verification**: Automatic integrity verification
- **Tamper detection**: Detection of unauthorized modifications

## Data Retention and Cleanup

### Retention Policies

Automated data retention policies ensure compliance:

```python
from gmail_dataset_creator.security import DataRetentionManager, RetentionPolicy, RetentionRule

# Create retention policy
policy = RetentionPolicy(
    name="GDPR Compliance Policy",
    description="Data retention policy for GDPR compliance",
    compliance_mode=True
)

# Add retention rules
policy.add_rule(RetentionRule(
    category=DataCategory.RAW_EMAILS,
    retention_days=30,
    action=RetentionAction.DELETE
))

# Apply retention policy
retention_manager = DataRetentionManager(policy)
results = retention_manager.apply_retention_policy()
```

### Secure Cleanup

Secure deletion ensures data cannot be recovered:

```python
from gmail_dataset_creator.privacy import SecureDataCleanup, CleanupPolicy

# Configure cleanup policy
cleanup_policy = CleanupPolicy(
    delete_temp_files=True,
    secure_delete_passes=3,
    retention_days=0
)

# Perform secure cleanup
secure_cleanup = SecureDataCleanup(cleanup_policy)
cleanup_success = secure_cleanup.perform_full_cleanup(
    token_storage_path="./tokens",
    data_directory="./temp_data"
)
```

## Security Validation

### Automated Security Checks

The system performs comprehensive security validation:

```python
from gmail_dataset_creator.security import SecurityValidator

# Create security validator
security_validator = SecurityValidator()

# Perform security validation
validation_result = security_validator.validate_security(
    config_path="config.yaml",
    check_files=True,
    check_permissions=True,
    check_encryption=True
)

# Generate security report
security_report = security_validator.generate_security_report(validation_result)
```

### Validation Checks

The security validator checks for:

- **Configuration security**: Weak encryption settings, exposed credentials
- **File permissions**: Overly permissive file access
- **Encryption strength**: Weak algorithms or key lengths
- **Exposed credentials**: Hardcoded passwords or API keys
- **Dependency vulnerabilities**: Outdated or vulnerable packages
- **Data leakage**: Sensitive information in logs or temporary files

### Password Strength Validation

Password strength is validated against security best practices:

```python
# Validate password strength
is_strong, issues = security_validator.validate_password_strength("MyPassword123!")
```

## Configuration

### Privacy Configuration

```yaml
privacy:
  anonymize_senders: true
  anonymize_recipients: true
  exclude_personal: true
  exclude_sensitive: true
  remove_sensitive_content: true
  exclude_keywords:
    - "confidential"
    - "private"
    - "secret"
  exclude_domains:
    - "internal.company.com"
    - "private.org"
  min_confidence_threshold: 0.8
```

### Security Configuration

```yaml
security:
  encryption_algorithm: "fernet"
  key_derivation_function: "pbkdf2"
  encryption_iterations: 150000
  salt_length: 64
  secure_export: true
  data_retention_days: 30
  secure_cleanup: true
  audit_logging: true
```

## Compliance Features

### GDPR Compliance

The system supports GDPR compliance through:

- **Data minimization**: Only collect necessary data
- **Purpose limitation**: Use data only for specified purposes
- **Storage limitation**: Automatic data retention and deletion
- **Data portability**: Secure export capabilities
- **Right to erasure**: Secure deletion of personal data
- **Privacy by design**: Built-in privacy protections

### Audit Logging

Comprehensive audit logging tracks:

- **Data access**: Who accessed what data when
- **Privacy actions**: Anonymization and filtering operations
- **Security events**: Encryption, decryption, and validation
- **Retention actions**: Data deletion and archival
- **Export operations**: Dataset creation and export

## Best Practices

### Password Security

- Use strong, unique passwords for encryption
- Store passwords securely (environment variables, key management systems)
- Regularly rotate encryption passwords
- Use multi-factor authentication where possible

### Data Handling

- Minimize data collection to what's necessary
- Apply privacy controls before processing
- Regularly validate security configurations
- Monitor for security vulnerabilities
- Implement proper access controls

### Operational Security

- Regularly update dependencies
- Monitor security advisories
- Perform security audits
- Train users on security best practices
- Implement incident response procedures

## Troubleshooting

### Common Issues

1. **Encryption Errors**
   - Check password strength requirements
   - Verify encryption configuration
   - Ensure sufficient entropy for key generation

2. **Privacy Control Issues**
   - Review exclusion criteria
   - Check confidence thresholds
   - Validate anonymization settings

3. **Export Problems**
   - Verify file permissions
   - Check available disk space
   - Validate export configuration

4. **Security Validation Failures**
   - Review security recommendations
   - Fix configuration issues
   - Update vulnerable dependencies

### Getting Help

For additional support:

1. Check the troubleshooting guide
2. Review security documentation
3. Consult privacy compliance guidelines
4. Contact security team for critical issues

## Security Considerations

### Threat Model

The system protects against:

- **Data breaches**: Unauthorized access to sensitive data
- **Data leakage**: Accidental exposure of personal information
- **Insider threats**: Malicious or negligent internal users
- **Supply chain attacks**: Compromised dependencies
- **Configuration errors**: Insecure system configuration

### Limitations

- **Zero-day vulnerabilities**: Unknown security flaws
- **Social engineering**: Human factor vulnerabilities
- **Physical security**: Hardware-level attacks
- **Advanced persistent threats**: Sophisticated, targeted attacks

### Recommendations

- Implement defense in depth
- Regular security assessments
- Continuous monitoring
- Incident response planning
- Security awareness training