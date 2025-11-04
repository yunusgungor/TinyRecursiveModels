# Gmail Dataset Creator - Test Suite

This directory contains a comprehensive test suite for the Gmail Dataset Creator project, covering unit tests, integration tests, and performance tests.

## Test Structure

### Core Components Unit Tests (`test_core_components_unit.py`)

Tests the core logic of individual components without external dependencies:

- **Authentication Configuration Logic**: Tests auth config structure and validation
- **Email Processing Logic**: Tests email anonymization, HTML-to-text conversion, and sensitive data removal
- **Dataset Builder Logic**: Tests train/test splitting, vocabulary building, and category balance checking
- **Gemini Classification Logic**: Tests rule-based classification and confidence scoring

### Integration and Performance Tests (`test_integration_performance.py`)

Tests end-to-end workflows, performance characteristics, and security features:

- **End-to-End Integration Tests**:
  - Complete dataset creation workflow
  - Authentication failure handling
  - Configuration validation
  - Error recovery and resume functionality

- **Performance Tests**:
  - Large dataset processing (1000+ emails)
  - Concurrent processing performance
  - Memory usage patterns
  - Processing rate consistency

- **Security Tests**:
  - Token storage security and encryption
  - Data anonymization and sensitive information removal
  - Secure data cleanup procedures
  - Access control validation

### Existing Component Tests

The following test files were created during development and test specific components:

- `test_authentication.py`: Authentication handler tests (requires Google API dependencies)
- `test_dataset_builder.py`: Dataset builder tests (requires Google API dependencies)
- `test_email_processor.py`: Email processing tests
- `test_gemini_classifier.py`: Gemini API integration tests
- `test_gmail_client.py`: Gmail API client tests

## Running Tests

### Run All Tests

```bash
# Run the comprehensive test suite
python gmail_dataset_creator/tests/run_all_tests.py

# Or use pytest directly
python -m pytest gmail_dataset_creator/tests/test_core_components_unit.py gmail_dataset_creator/tests/test_integration_performance.py -v
```

### Run Specific Test Categories

```bash
# Run only unit tests
python gmail_dataset_creator/tests/run_all_tests.py unit

# Run only integration tests
python gmail_dataset_creator/tests/run_all_tests.py integration

# Run existing component tests (may require dependencies)
python gmail_dataset_creator/tests/run_all_tests.py existing
```

### Run Individual Test Files

```bash
# Run unit tests
python -m pytest gmail_dataset_creator/tests/test_core_components_unit.py -v

# Run integration tests
python -m pytest gmail_dataset_creator/tests/test_integration_performance.py -v

# Run specific test class
python -m pytest gmail_dataset_creator/tests/test_core_components_unit.py::TestEmailProcessingLogic -v

# Run specific test method
python -m pytest gmail_dataset_creator/tests/test_integration_performance.py::TestPerformanceTests::test_large_dataset_processing_performance -v
```

## Test Coverage

The test suite covers the following areas:

### Functional Testing
- ✅ Authentication flow and token management
- ✅ Email content extraction and processing
- ✅ HTML-to-text conversion
- ✅ Sensitive data anonymization
- ✅ Email classification logic
- ✅ Dataset building and train/test splitting
- ✅ Vocabulary generation
- ✅ File export and format validation

### Integration Testing
- ✅ End-to-end dataset creation workflow
- ✅ Configuration loading and validation
- ✅ Error handling and recovery
- ✅ File system operations
- ✅ Multi-component interaction

### Performance Testing
- ✅ Large dataset processing (1000+ emails)
- ✅ Memory usage patterns
- ✅ Processing rate consistency
- ✅ Concurrent processing capabilities

### Security Testing
- ✅ Token encryption and secure storage
- ✅ Data anonymization effectiveness
- ✅ Sensitive information removal
- ✅ Secure file cleanup
- ✅ Access control validation

## Test Design Principles

### Minimal Dependencies
The core test suite (`test_core_components_unit.py` and `test_integration_performance.py`) is designed to run without external API dependencies by:
- Mocking all Google API modules
- Using mock implementations for testing logic
- Focusing on core algorithmic functionality

### Comprehensive Coverage
Tests cover both happy path and error scenarios:
- Valid inputs and expected outputs
- Invalid inputs and error handling
- Edge cases and boundary conditions
- Performance under load

### Realistic Test Data
Tests use realistic email data patterns:
- Various email formats and content types
- Different classification categories
- Realistic sender/recipient patterns
- Appropriate data volumes for performance testing

## Mock Strategy

The test suite uses extensive mocking to avoid external dependencies:

- **Google APIs**: All Google API modules are mocked to prevent import errors
- **File System**: Temporary directories are used for file operations
- **Network Calls**: API calls are mocked with realistic response patterns
- **Time-based Operations**: Processing delays are simulated for performance testing

## Performance Benchmarks

The performance tests establish baseline expectations:

- **Processing Rate**: Minimum 50 emails/second
- **Memory Usage**: Maximum 10KB per email + 100MB base
- **Large Dataset**: 1000 emails processed within 10 seconds
- **Concurrent Processing**: At least 20% improvement over sequential

## Security Validation

Security tests verify:

- **Token Encryption**: Sensitive tokens are properly encrypted
- **Data Anonymization**: Personal information is anonymized or removed
- **Secure Cleanup**: Temporary files are securely deleted
- **Access Control**: File permissions are appropriately restricted

## Continuous Integration

The test suite is designed for CI/CD environments:

- No external API dependencies for core tests
- Fast execution (< 5 seconds for full suite)
- Clear pass/fail indicators
- Detailed error reporting
- Performance regression detection

## Adding New Tests

When adding new functionality, include tests for:

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Error Handling**: Test failure scenarios
4. **Performance**: Test with realistic data volumes
5. **Security**: Test data protection measures

Follow the existing patterns for mocking and test structure to maintain consistency.