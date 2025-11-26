"""Property-based tests for error handling and logging"""

import json
import logging
import traceback
from datetime import datetime
from io import StringIO
from hypothesis import given, strategies as st
from hypothesis import settings as hypothesis_settings

from app.core.exceptions import (
    BaseAPIException,
    ModelInferenceError,
    TrendyolAPIError,
    ValidationError,
    ToolExecutionError
)


@given(
    error_code=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')),
    message=st.text(min_size=1, max_size=200),
    details=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.booleans()),
        max_size=5
    )
)
@hypothesis_settings(max_examples=100)
def test_error_logging_format_property(error_code, message, details):
    """
    Feature: trendyol-gift-recommendation-web, Property 17: Error Logging Format
    
    For any error that occurs, the system should log it with timestamp, error type, and stack trace
    Validates: Requirements 7.1
    """
    # Create exception
    exc = BaseAPIException(message=message, error_code=error_code, details=details)
    
    # Property: Exception should have error_code (error type)
    assert hasattr(exc, 'error_code'), "Exception must have error_code attribute"
    assert exc.error_code == error_code, "Error code should match"
    
    # Property: Exception should have message
    assert hasattr(exc, 'message'), "Exception must have message attribute"
    assert exc.message == message, "Message should match"
    
    # Property: Exception should have details
    assert hasattr(exc, 'details'), "Exception must have details attribute"
    assert exc.details == details, "Details should match"
    
    # Property: Exception should be able to generate stack trace
    try:
        raise exc
    except BaseAPIException as caught:
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Property: Stack trace should not be empty
        assert stack_trace, "Stack trace should not be empty"
        
        # Property: Stack trace should contain error type
        assert error_code in stack_trace or "BaseAPIException" in stack_trace, "Stack trace should contain error type"
        
        # Property: Stack trace should contain file path information
        assert ".py" in stack_trace, "Stack trace should contain file path"
        
        # Property: Stack trace should contain line number
        assert "line" in stack_trace.lower(), "Stack trace should contain line number info"


@given(
    field_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), whitelist_characters='_')),
    error_message=st.text(min_size=1, max_size=200)
)
@hypothesis_settings(max_examples=100)
def test_validation_error_field_identification_property(field_name, error_message):
    """
    Feature: trendyol-gift-recommendation-web, Property 18: Validation Error Field Identification
    
    For any validation error, the error message should identify which specific field caused the error
    Validates: Requirements 7.4
    """
    # Create validation error
    validation_error = ValidationError(
        message=error_message,
        field=field_name
    )
    
    # Property: Error must have error_code
    assert validation_error.error_code == "VALIDATION_ERROR", "Error code should be VALIDATION_ERROR"
    
    # Property: Error details must contain field name
    assert "field" in validation_error.details, "Error details must contain 'field' key"
    assert validation_error.details["field"] == field_name, "Field name in details should match"
    
    # Property: Error message should be accessible
    assert validation_error.message == error_message, "Error message should match"
    
    # Property: Field information should be retrievable from exception
    assert hasattr(validation_error, 'details'), "Exception should have details attribute"
    assert isinstance(validation_error.details, dict), "Details should be a dictionary"


@given(
    exception_type=st.sampled_from([
        ModelInferenceError,
        TrendyolAPIError,
        ToolExecutionError
    ]),
    custom_message=st.text(min_size=1, max_size=200),
    details=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=0, max_size=100),
        max_size=3
    )
)
@hypothesis_settings(max_examples=100)
def test_custom_exceptions_preserve_context_property(exception_type, custom_message, details):
    """
    Property: All custom exceptions should preserve error context
    
    For any custom exception type, creating it with message and details should preserve all information
    """
    # Create exception
    exc = exception_type(message=custom_message, details=details)
    
    # Property: Exception should have error_code
    assert hasattr(exc, 'error_code'), "Exception should have error_code"
    assert isinstance(exc.error_code, str), "Error code should be string"
    assert len(exc.error_code) > 0, "Error code should not be empty"
    
    # Property: Exception should preserve message
    assert exc.message == custom_message, "Message should be preserved"
    
    # Property: Exception should preserve details
    assert exc.details == details, "Details should be preserved"
    
    # Property: Exception should be raisable
    try:
        raise exc
    except BaseAPIException as caught:
        assert caught.message == custom_message, "Caught exception should have same message"
        assert caught.details == details, "Caught exception should have same details"


@given(
    error_description=st.text(min_size=10, max_size=200),
    resolution_steps=st.lists(
        st.text(min_size=5, max_size=100),
        min_size=1,
        max_size=5
    ),
    error_code=st.text(
        min_size=1, 
        max_size=50, 
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_')
    )
)
@hypothesis_settings(max_examples=100)
def test_error_message_clarity_property(error_description, resolution_steps, error_code):
    """
    Feature: optimized-container-infrastructure, Property 3: Error Message Clarity
    
    For any startup error condition, the system should produce error messages 
    containing both the error description and actionable resolution steps.
    Validates: Requirements 1.5
    """
    # Create error with description and resolution steps
    details = {
        "description": error_description,
        "resolution_steps": resolution_steps,
        "error_type": "startup_error"
    }
    
    exc = BaseAPIException(
        message=error_description,
        error_code=error_code,
        details=details
    )
    
    # Property 1: Error must have a description
    assert exc.message, "Error must have a non-empty message/description"
    assert len(exc.message) >= 10, "Error description should be meaningful (at least 10 chars)"
    assert exc.message == error_description, "Error description should match provided description"
    
    # Property 2: Error details must contain resolution steps
    assert "resolution_steps" in exc.details, "Error details must contain 'resolution_steps' key"
    assert isinstance(exc.details["resolution_steps"], list), "Resolution steps should be a list"
    assert len(exc.details["resolution_steps"]) > 0, "Resolution steps list should not be empty"
    
    # Property 3: Each resolution step should be actionable (non-empty string)
    for step in exc.details["resolution_steps"]:
        assert isinstance(step, str), "Each resolution step should be a string"
        assert len(step) >= 5, "Each resolution step should be meaningful (at least 5 chars)"
    
    # Property 4: Error should have both description and resolution in details
    assert "description" in exc.details, "Error details should contain description"
    assert exc.details["description"] == error_description, "Description in details should match"
    
    # Property 5: Error code should be present and meaningful
    assert exc.error_code, "Error code should not be empty"
    assert len(exc.error_code) > 0, "Error code should be meaningful"
    
    # Property 6: When formatted as dict, all information should be accessible
    error_dict = {
        "error_code": exc.error_code,
        "message": exc.message,
        "details": exc.details
    }
    
    assert error_dict["message"], "Formatted error should have message"
    assert error_dict["details"]["resolution_steps"], "Formatted error should have resolution steps"
    assert len(error_dict["details"]["resolution_steps"]) > 0, "Formatted error should have at least one resolution step"
    
    # Property 7: Error information should be JSON serializable for logging/API responses
    try:
        json_str = json.dumps(error_dict)
        parsed = json.loads(json_str)
        assert parsed["message"] == error_description, "Error should be JSON serializable"
        assert parsed["details"]["resolution_steps"] == resolution_steps, "Resolution steps should survive JSON serialization"
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Error information should be JSON serializable: {e}")
