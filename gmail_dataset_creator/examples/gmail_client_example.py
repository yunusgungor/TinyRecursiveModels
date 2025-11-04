"""Example usage of the Gmail API client with filtering and rate limiting."""

import logging
from datetime import datetime, timedelta
from gmail_dataset_creator.auth.authentication import AuthenticationHandler, AuthConfig
from gmail_dataset_creator.gmail.client import GmailAPIClient, EmailFilter, RateLimitConfig, BatchConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Gmail API client usage."""
    
    # Configure authentication
    auth_config = AuthConfig(
        credentials_file="credentials.json",
        token_file="token.json",
        scopes=['https://www.googleapis.com/auth/gmail.readonly']
    )
    
    # Initialize authentication handler
    auth_handler = AuthenticationHandler(auth_config)
    
    # Authenticate
    if not auth_handler.authenticate():
        logger.error("Authentication failed")
        return
    
    # Get credentials
    credentials = auth_handler.get_credentials()
    if not credentials:
        logger.error("Failed to get credentials")
        return
    
    # Configure rate limiting (conservative settings)
    rate_limit_config = RateLimitConfig(
        requests_per_second=5.0,  # Conservative rate
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0
    )
    
    # Configure batch processing
    batch_config = BatchConfig(
        batch_size=50,  # Smaller batches for testing
        max_concurrent_batches=3
    )
    
    # Initialize Gmail client
    gmail_client = GmailAPIClient(
        credentials=credentials,
        rate_limit_config=rate_limit_config,
        batch_config=batch_config
    )
    
    logger.info("Gmail API client initialized successfully")
    
    # Example 1: List recent messages
    logger.info("=== Example 1: List recent messages ===")
    try:
        result = gmail_client.list_messages(max_results=10)
        messages = result.get('messages', [])
        logger.info(f"Found {len(messages)} recent messages")
        logger.info(f"Total estimated messages: {result.get('resultSizeEstimate', 0)}")
    except Exception as e:
        logger.error(f"Failed to list messages: {e}")
    
    # Example 2: Filter messages by date range
    logger.info("=== Example 2: Filter by date range ===")
    try:
        # Get messages from last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = gmail_client.list_messages_by_date_range(
            start_date=start_date,
            end_date=end_date,
            max_results=20
        )
        messages = result.get('messages', [])
        logger.info(f"Found {len(messages)} messages in last 30 days")
    except Exception as e:
        logger.error(f"Failed to filter by date: {e}")
    
    # Example 3: Search messages with custom query
    logger.info("=== Example 3: Search with custom query ===")
    try:
        # Search for unread messages in inbox
        result = gmail_client.search_messages(
            search_query="is:unread in:inbox",
            max_results=15
        )
        messages = result.get('messages', [])
        logger.info(f"Found {len(messages)} unread messages in inbox")
    except Exception as e:
        logger.error(f"Failed to search messages: {e}")
    
    # Example 4: Use advanced filtering
    logger.info("=== Example 4: Advanced filtering ===")
    try:
        # Create email filter
        email_filter = EmailFilter(
            date_range=(datetime.now() - timedelta(days=7), datetime.now()),
            label_ids=["INBOX"],
            include_spam_trash=False,
            max_results=25
        )
        
        result = gmail_client.list_messages_filtered(email_filter)
        messages = result.get('messages', [])
        logger.info(f"Found {len(messages)} messages with advanced filter")
    except Exception as e:
        logger.error(f"Failed to use advanced filter: {e}")
    
    # Example 5: Get message count estimate
    logger.info("=== Example 5: Message count estimate ===")
    try:
        count = gmail_client.get_message_count_estimate()
        logger.info(f"Estimated total messages: {count}")
    except Exception as e:
        logger.error(f"Failed to get message count: {e}")
    
    # Example 6: Batch message retrieval (if we have messages)
    if 'messages' in locals() and messages:
        logger.info("=== Example 6: Batch message retrieval ===")
        try:
            # Get first few message IDs
            message_ids = [msg['id'] for msg in messages[:5]]
            
            # Fetch full messages in batch
            full_messages = gmail_client.get_messages_batch(
                message_ids=message_ids,
                format='metadata'  # Get metadata only for faster processing
            )
            
            logger.info(f"Retrieved {len(full_messages)} full messages via batch")
            
            # Show some basic info about the messages
            for msg in full_messages[:3]:  # Show first 3
                headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
                subject = headers.get('Subject', 'No Subject')[:50]
                sender = headers.get('From', 'Unknown Sender')[:30]
                logger.info(f"  - From: {sender}, Subject: {subject}...")
                
        except Exception as e:
            logger.error(f"Failed batch retrieval: {e}")
    
    # Show client statistics
    logger.info("=== Client Statistics ===")
    stats = gmail_client.get_stats()
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Successful requests: {stats['successful_requests']}")
    logger.info(f"Failed requests: {stats['failed_requests']}")
    logger.info(f"Rate limit hits: {stats['rate_limit_hits']}")
    logger.info(f"Success rate: {stats['success_rate']:.2%}")
    
    logger.info("Gmail API client example completed successfully")


if __name__ == "__main__":
    main()