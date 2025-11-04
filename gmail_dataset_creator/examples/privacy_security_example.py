"""
Example demonstrating privacy and security features integration.

This example shows how to use the privacy controls and security measures
together to create a secure and privacy-compliant dataset creation process.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

from ..config.manager import ConfigManager, SystemConfig
from ..privacy.privacy_controls import PrivacyController, PrivacySettings
from ..privacy.secure_cleanup import SecureDataCleanup, CleanupPolicy
from ..security.encryption_manager import EncryptionManager, EncryptionConfig
from ..security.secure_export import SecureExporter, ExportConfig
from ..security.data_retention import DataRetentionManager, RetentionPolicy
from ..security.security_validator import SecurityValidator
from ..models import EmailData
from datetime import datetime


def setup_privacy_and_security_example():
    """Set up privacy and security components for demonstration."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Setting up privacy and security example...")
    
    # 1. Load system configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # 2. Set up privacy controls
    privacy_settings = PrivacySettings(
        exclude_personal=True,
        exclude_sensitive=True,
        anonymize_senders=True,
        anonymize_recipients=True,
        remove_sensitive_content=True,
        exclude_keywords=['confidential', 'private', 'secret'],
        exclude_domains=['internal.company.com', 'private.org'],
        min_confidence_threshold=0.8
    )
    
    privacy_controller = PrivacyController(privacy_settings)
    
    # 3. Set up encryption manager
    encryption_config = EncryptionConfig(
        iterations=150000,  # High security
        salt_length=64,
        key_length=32
    )
    
    encryption_manager = EncryptionManager(encryption_config)
    
    # 4. Set up secure export
    export_config = ExportConfig(
        encrypt_data=True,
        include_checksums=True,
        password_protect=True,
        remove_source_after_export=False  # Keep for demo
    )
    
    secure_exporter = SecureExporter(export_config)
    
    # 5. Set up data retention
    retention_policy = RetentionPolicy(
        name="High Security Policy",
        description="High security data retention policy",
        compliance_mode=True,
        audit_logging=True
    )
    
    retention_manager = DataRetentionManager(retention_policy)
    
    # 6. Set up secure cleanup
    cleanup_policy = CleanupPolicy(
        delete_temp_files=True,
        delete_raw_emails=True,
        delete_tokens_on_completion=False,  # Keep for demo
        secure_delete_passes=3,
        retention_days=30
    )
    
    secure_cleanup = SecureDataCleanup(cleanup_policy)
    
    # 7. Set up security validator
    security_validator = SecurityValidator()
    
    return {
        'config': config,
        'privacy_controller': privacy_controller,
        'encryption_manager': encryption_manager,
        'secure_exporter': secure_exporter,
        'retention_manager': retention_manager,
        'secure_cleanup': secure_cleanup,
        'security_validator': security_validator,
        'logger': logger
    }


def demonstrate_privacy_controls(components: Dict[str, Any]):
    """Demonstrate privacy control features."""
    logger = components['logger']
    privacy_controller = components['privacy_controller']
    
    logger.info("Demonstrating privacy controls...")
    
    # Create sample email data
    sample_emails = [
        EmailData(
            id="email_001",
            subject="Meeting with John Doe",
            body="Hi John, let's meet at 3pm. My phone is 555-123-4567.",
            sender="alice@company.com",
            recipient="john@company.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_002",
            subject="Personal: Family dinner",
            body="Dear Mom, I'll be there for dinner. Love you!",
            sender="alice@gmail.com",
            recipient="mom@gmail.com",
            timestamp=datetime.now(),
            raw_content=""
        ),
        EmailData(
            id="email_003",
            subject="Confidential: Project Alpha",
            body="This is confidential information about our secret project.",
            sender="boss@internal.company.com",
            recipient="alice@company.com",
            timestamp=datetime.now(),
            raw_content=""
        )
    ]
    
    # Apply privacy controls
    processed_emails = []
    for email in sample_emails:
        processed_email = privacy_controller.apply_privacy_controls(email)
        if processed_email:
            processed_emails.append(processed_email)
            logger.info(f"Processed email: {processed_email.id}")
        else:
            logger.info(f"Excluded email: {email.id}")
    
    # Get privacy statistics
    stats = privacy_controller.get_privacy_stats()
    logger.info(f"Privacy stats: {stats}")
    
    return processed_emails


def demonstrate_encryption(components: Dict[str, Any]):
    """Demonstrate encryption features."""
    logger = components['logger']
    encryption_manager = components['encryption_manager']
    
    logger.info("Demonstrating encryption features...")
    
    # Sample sensitive data
    sensitive_data = {
        "api_key": "sk-1234567890abcdef",
        "user_credentials": {
            "username": "alice",
            "password": "secure_password_123"
        },
        "email_content": "This is sensitive email content"
    }
    
    # Generate secure password
    password = encryption_manager.generate_secure_password(16)
    logger.info(f"Generated secure password: {password[:4]}...")
    
    # Encrypt data
    encrypted_data, salt = encryption_manager.encrypt_data(sensitive_data, password)
    logger.info(f"Data encrypted successfully. Salt length: {len(salt)} bytes")
    
    # Decrypt data
    decrypted_data = encryption_manager.decrypt_data(encrypted_data, password, salt)
    logger.info("Data decrypted successfully")
    
    # Verify integrity
    data_hash = encryption_manager.hash_data(str(sensitive_data))
    is_valid = encryption_manager.verify_data_integrity(str(sensitive_data), data_hash)
    logger.info(f"Data integrity verified: {is_valid}")
    
    return encrypted_data, salt


def demonstrate_secure_export(components: Dict[str, Any], processed_emails):
    """Demonstrate secure export features."""
    logger = components['logger']
    secure_exporter = components['secure_exporter']
    
    logger.info("Demonstrating secure export...")
    
    # Create temporary dataset directory
    temp_dir = Path("temp_dataset_demo")
    temp_dir.mkdir(exist_ok=True)
    
    # Save processed emails to temporary files
    for i, email in enumerate(processed_emails):
        email_file = temp_dir / f"email_{i:03d}.json"
        import json
        with open(email_file, 'w') as f:
            json.dump({
                'id': email.id,
                'subject': email.subject,
                'body': email.body,
                'sender': email.sender,
                'recipient': email.recipient,
                'timestamp': email.timestamp.isoformat()
            }, f, indent=2)
    
    # Export with security measures
    export_password = "secure_export_password_123"
    export_result = secure_exporter.export_dataset(
        dataset_path=str(temp_dir),
        output_path="secure_export_demo",
        password=export_password,
        metadata={"created_by": "privacy_security_example", "version": "1.0"}
    )
    
    logger.info(f"Export result: {export_result['success']}")
    if export_result['success']:
        logger.info(f"Export statistics: {export_result['statistics']}")
    
    # Verify export integrity
    if export_result['success']:
        verification_result = secure_exporter.verify_export_integrity(
            export_result['export_path']
        )
        logger.info(f"Export integrity verified: {verification_result['success']}")
    
    # Cleanup temporary directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return export_result


def demonstrate_security_validation(components: Dict[str, Any]):
    """Demonstrate security validation."""
    logger = components['logger']
    security_validator = components['security_validator']
    
    logger.info("Demonstrating security validation...")
    
    # Perform security validation
    validation_result = security_validator.validate_security(
        check_files=True,
        check_permissions=True,
        check_encryption=True
    )
    
    logger.info(f"Security validation passed: {validation_result.passed}")
    logger.info(f"Security score: {validation_result.score:.1f}/100")
    logger.info(f"Issues found: {len(validation_result.issues)}")
    
    # Generate security report
    security_report = security_validator.generate_security_report(validation_result)
    logger.info(f"Security report generated with {len(security_report['issues'])} issues")
    
    # Validate password strength
    test_passwords = ["weak", "StrongPassword123!", "VerySecurePassword2024@#$"]
    for password in test_passwords:
        is_strong, issues = security_validator.validate_password_strength(password)
        logger.info(f"Password '{password[:4]}...': Strong={is_strong}, Issues={len(issues)}")
    
    return validation_result


def demonstrate_data_retention(components: Dict[str, Any]):
    """Demonstrate data retention features."""
    logger = components['logger']
    retention_manager = components['retention_manager']
    
    logger.info("Demonstrating data retention...")
    
    # Check upcoming expirations
    upcoming_expirations = retention_manager.check_upcoming_expirations(days_ahead=30)
    logger.info(f"Files expiring in next 30 days: {len(upcoming_expirations)}")
    
    # Generate retention report
    retention_report = retention_manager.generate_retention_report()
    logger.info(f"Retention report generated at {retention_report['report_timestamp']}")
    
    # Get retention statistics
    retention_stats = retention_manager.get_retention_stats()
    logger.info(f"Retention statistics: {retention_stats}")
    
    return retention_report


def demonstrate_secure_cleanup(components: Dict[str, Any]):
    """Demonstrate secure cleanup features."""
    logger = components['logger']
    secure_cleanup = components['secure_cleanup']
    
    logger.info("Demonstrating secure cleanup...")
    
    # Create some temporary files for demonstration
    temp_files = []
    for i in range(3):
        temp_file = secure_cleanup.create_temp_file(suffix=f"_demo_{i}.tmp")
        temp_files.append(temp_file)
        
        # Write some data to the file
        with open(temp_file, 'w') as f:
            f.write(f"Temporary data for demo file {i}")
    
    logger.info(f"Created {len(temp_files)} temporary files")
    
    # Register sensitive files
    for temp_file in temp_files[:2]:  # Mark first 2 as sensitive
        secure_cleanup.register_sensitive_file(temp_file)
    
    # Perform cleanup
    cleanup_success = secure_cleanup.cleanup_temp_files()
    logger.info(f"Temp file cleanup successful: {cleanup_success}")
    
    sensitive_cleanup_success = secure_cleanup.cleanup_sensitive_data()
    logger.info(f"Sensitive data cleanup successful: {sensitive_cleanup_success}")
    
    # Get cleanup statistics
    cleanup_stats = secure_cleanup.get_cleanup_stats()
    logger.info(f"Cleanup statistics: {cleanup_stats}")
    
    return cleanup_stats


def main():
    """Main demonstration function."""
    print("Gmail Dataset Creator - Privacy and Security Features Demo")
    print("=" * 60)
    
    try:
        # Set up components
        components = setup_privacy_and_security_example()
        logger = components['logger']
        
        # Demonstrate privacy controls
        processed_emails = demonstrate_privacy_controls(components)
        
        # Demonstrate encryption
        demonstrate_encryption(components)
        
        # Demonstrate secure export
        if processed_emails:
            demonstrate_secure_export(components, processed_emails)
        
        # Demonstrate security validation
        demonstrate_security_validation(components)
        
        # Demonstrate data retention
        demonstrate_data_retention(components)
        
        # Demonstrate secure cleanup
        demonstrate_secure_cleanup(components)
        
        logger.info("Privacy and security demonstration completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()