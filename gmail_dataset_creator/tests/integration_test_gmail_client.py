"""Integration tests for Gmail API client (requires Google API dependencies)."""

import logging
import sys
from datetime import datetime, timedelta

# This test requires Google API dependencies to be installed
try:
    from gmail_dataset_creator.auth.authentication import AuthenticationHandler, AuthConfig
    from gmail_dataset_creator.gmail.client import GmailAPIClient, EmailFilter, RateLimitConfig
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Google API dependencies not available: {e}")
    print("Install with: pip install -r requirements_gmail_dataset.txt")
    DEPENDENCIES_AVAILABLE = False


def test_gmail_client_integration():
    """Integration test for Gmail API client."""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping integration test - dependencies not available")
        return False
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Configure authentication (requires credentials.json)
        auth_config = AuthConfig(
            credentials_file="credentials.json",
            token_file="test_token.json",
            scopes=['https://www.googleapis.com/auth/gmail.readonly']
        )
        
        # Initialize authentication handler
        auth_handler = AuthenticationHandler(auth_config)
        
        # Check if we can authenticate (this will fail if no credentials.json)
        if not auth_handler.authenticate():
            logger.warning("Authentication failed - this is expected without credentials.json")
            return False
        
        # Get credentials
        credentials = auth_handler.get_credentials()
        if not credentials:
            logger.warning("Failed to get credentials")
            return False
        
        # Configure conservative rate limiting for testing
        rate_limit_config = RateLimitConfig(
            requests_per_second=2.0,  # Very conservative
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0
        )
        
        # Initialize Gmail client
        gmail_client = GmailAPIClient(
            credentials=credentials,
            rate_limit_config=rate_limit_config
        )
        
        logger.info("Gmail API client initialized successfully")
        
        # Test basic message listing
        result = gmail_client.list_messages(max_results=5)
        messages = result.get('messages', [])
        logger.info(f"Successfully retrieved {len(messages)} messages")
        
        # Test message count estimate
        count = gmail_client.get_message_count_estimate()
        logger.info(f"Estimated total messages: {count}")
        
        # Test date range filtering
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        result = gmail_client.list_messages_by_date_range(
            start_date=start_date,
            end_date=end_date,
            max_results=3
        )
        filtered_messages = result.get('messages', [])
        logger.info(f"Found {len(filtered_messages)} messages in last 7 days")
        
        # Test labels retrieval
        labels = gmail_client.get_labels()
        logger.info(f"Found {len(labels)} labels")
        
        # Test finding a common label
        inbox_id = gmail_client.find_label_id('INBOX')
        logger.info(f"INBOX label ID: {inbox_id}")
        
        # Show client statistics
        stats = gmail_client.get_stats()
        logger.info(f"Integration test stats: {stats}")
        
        logger.info("Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def test_query_builder():
    """Test query builder functionality (no API calls needed)."""
    if not DEPENDENCIES_AVAILABLE:
        print("Skipping query builder test - dependencies not available")
        return False
    
    from gmail_dataset_creator.gmail.client import QueryBuilder, EmailFilter
    
    print("Testing QueryBuilder...")
    
    # Test date query
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    date_query = QueryBuilder.build_date_query(start_date, end_date)
    print(f"Date query: {date_query}")
    assert "after:2024/01/01" in date_query
    assert "before:2024/01/31" in date_query
    
    # Test sender query
    senders = ["test@example.com", "@company.com"]
    sender_query = QueryBuilder.build_sender_query(senders)
    print(f"Sender query: {sender_query}")
    assert "from:test@example.com" in sender_query
    assert "from:@company.com" in sender_query
    
    # Test comprehensive query
    email_filter = EmailFilter(
        date_range=(start_date, end_date),
        sender_filters=["test@example.com"],
        query="is:unread",
        exclude_labels=["SPAM"]
    )
    
    comprehensive_query = QueryBuilder.build_query(email_filter)
    print(f"Comprehensive query: {comprehensive_query}")
    assert "is:unread" in comprehensive_query
    assert "after:2024/01/01" in comprehensive_query
    assert "from:test@example.com" in comprehensive_query
    assert "-label:SPAM" in comprehensive_query
    
    print("QueryBuilder tests passed!")
    return True


if __name__ == "__main__":
    print("Gmail API Client Integration Tests")
    print("=" * 40)
    
    # Test query builder (doesn't require API access)
    query_test_passed = test_query_builder()
    
    # Test full integration (requires credentials and API access)
    integration_test_passed = test_gmail_client_integration()
    
    print("\nTest Results:")
    print(f"Query Builder Test: {'PASSED' if query_test_passed else 'FAILED'}")
    print(f"Integration Test: {'PASSED' if integration_test_passed else 'SKIPPED/FAILED'}")
    
    if not DEPENDENCIES_AVAILABLE:
        print("\nTo run full integration tests:")
        print("1. Install dependencies: pip install -r requirements_gmail_dataset.txt")
        print("2. Set up Gmail API credentials (credentials.json)")
        print("3. Run this script again")
    
    sys.exit(0 if query_test_passed else 1)