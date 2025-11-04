# Gmail API Client

This module provides a robust Gmail API client with advanced features for email fetching, filtering, and batch processing.

## Features

- **Rate Limiting**: Automatic rate limiting with exponential backoff to respect Gmail API quotas
- **Batch Processing**: Efficient batch requests for retrieving multiple messages
- **Advanced Filtering**: Comprehensive email filtering by date, sender, labels, and custom queries
- **Pagination Support**: Automatic pagination for large email collections
- **Error Handling**: Robust error handling with retry logic
- **Statistics Tracking**: Built-in request statistics and performance monitoring

## Quick Start

```python
from gmail_dataset_creator.auth.authentication import AuthenticationHandler, AuthConfig
from gmail_dataset_creator.gmail.client import GmailAPIClient, EmailFilter
from datetime import datetime, timedelta

# Set up authentication
auth_config = AuthConfig(
    credentials_file="credentials.json",
    token_file="token.json",
    scopes=['https://www.googleapis.com/auth/gmail.readonly']
)

auth_handler = AuthenticationHandler(auth_config)
auth_handler.authenticate()

# Initialize Gmail client
gmail_client = GmailAPIClient(auth_handler.get_credentials())

# List recent messages
messages = gmail_client.list_messages(max_results=10)
print(f"Found {len(messages.get('messages', []))} messages")
```

## Configuration

### Rate Limiting Configuration

```python
from gmail_dataset_creator.gmail.client import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_second=10.0,  # Conservative rate
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    jitter=True  # Add randomness to prevent thundering herd
)
```

### Batch Processing Configuration

```python
from gmail_dataset_creator.gmail.client import BatchConfig

batch_config = BatchConfig(
    batch_size=100,  # Gmail API supports up to 100 requests per batch
    max_concurrent_batches=5,
    timeout_seconds=300
)
```

## Usage Examples

### Basic Message Listing

```python
# List messages with basic parameters
result = gmail_client.list_messages(
    max_results=50,
    label_ids=["INBOX"],
    include_spam_trash=False
)

messages = result.get('messages', [])
next_page_token = result.get('nextPageToken')
```

### Date Range Filtering

```python
from datetime import datetime, timedelta

# Get messages from last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

result = gmail_client.list_messages_by_date_range(
    start_date=start_date,
    end_date=end_date,
    max_results=100
)
```

### Sender Filtering

```python
# Filter by specific senders
result = gmail_client.list_messages_by_sender(
    senders=["important@company.com", "@newsletter.com"],
    max_results=50
)
```

### Advanced Filtering

```python
from gmail_dataset_creator.gmail.client import EmailFilter

# Create comprehensive filter
email_filter = EmailFilter(
    date_range=(start_date, end_date),
    sender_filters=["@company.com"],
    label_ids=["INBOX"],
    exclude_labels=["SPAM", "TRASH"],
    query="is:unread",
    max_results=100
)

result = gmail_client.list_messages_filtered(email_filter)
```

### Batch Message Retrieval

```python
# Get message IDs first
result = gmail_client.list_messages(max_results=50)
message_ids = [msg['id'] for msg in result.get('messages', [])]

# Fetch full messages in batch
full_messages = gmail_client.get_messages_batch(
    message_ids=message_ids,
    format='full'  # or 'metadata', 'minimal', 'raw'
)
```

### Pagination

```python
# Paginate through all messages
for message in gmail_client.paginate_messages(email_filter, max_total_results=1000):
    print(f"Processing message: {message['id']}")
```

### Search with Gmail Query Syntax

```python
# Use Gmail's powerful search syntax
result = gmail_client.search_messages(
    search_query="from:noreply@github.com subject:security",
    max_results=25
)
```

## Query Builder

The `QueryBuilder` class helps construct Gmail search queries:

```python
from gmail_dataset_creator.gmail.client import QueryBuilder

# Build date range query
date_query = QueryBuilder.build_date_query(start_date, end_date)

# Build sender query
sender_query = QueryBuilder.build_sender_query(["user@example.com", "@company.com"])

# Build comprehensive query from filter
query = QueryBuilder.build_query(email_filter)
```

## Error Handling

The client includes comprehensive error handling:

- **Rate Limiting**: Automatic exponential backoff for rate limit errors
- **Network Errors**: Retry logic for transient network issues
- **Authentication Errors**: Clear error messages for auth problems
- **Batch Errors**: Individual error handling for batch requests

## Statistics and Monitoring

Track client performance:

```python
# Get usage statistics
stats = gmail_client.get_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Rate limit hits: {stats['rate_limit_hits']}")

# Reset statistics
gmail_client.reset_stats()
```

## Gmail API Quotas

The Gmail API has the following quotas (as of 2024):

- **Quota units per user per second**: 250
- **Quota units per user per day**: 1,000,000,000
- **Batch requests**: Up to 100 individual requests per batch

Each API call consumes different quota units:
- `messages.list`: 5 units
- `messages.get`: 5 units
- `messages.get` (metadata format): 2 units

## Best Practices

1. **Use Batch Requests**: For multiple message retrievals, always use batch requests
2. **Conservative Rate Limiting**: Start with lower rates and increase as needed
3. **Appropriate Message Format**: Use 'metadata' format when full content isn't needed
4. **Pagination**: Use pagination for large result sets instead of high max_results
5. **Error Handling**: Always handle potential API errors gracefully
6. **Caching**: Consider caching results to reduce API calls

## Dependencies

- `google-auth-oauthlib>=1.0.0`
- `google-auth-httplib2>=0.1.0`
- `google-api-python-client>=2.0.0`

## See Also

- [Gmail API Documentation](https://developers.google.com/gmail/api)
- [Gmail API Python Quickstart](https://developers.google.com/gmail/api/quickstart/python)
- [Gmail Search Operators](https://support.google.com/mail/answer/7190?hl=en)