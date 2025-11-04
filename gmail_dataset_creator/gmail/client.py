"""Gmail API client with rate limiting and batch processing capabilities."""

import time
import logging
import random
from typing import List, Dict, Optional, Any, Callable, Iterator, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import BatchHttpRequest


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0  # Gmail API allows 250 quota units per user per second
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 100  # Gmail API supports up to 100 requests per batch
    max_concurrent_batches: int = 5
    timeout_seconds: int = 300


@dataclass
class EmailFilter:
    """Email filtering configuration."""
    date_range: Optional[tuple[datetime, datetime]] = None
    sender_filters: Optional[List[str]] = None
    label_ids: Optional[List[str]] = None
    exclude_labels: Optional[List[str]] = None
    include_spam_trash: bool = False
    query: Optional[str] = None
    max_results: Optional[int] = None


class QueryBuilder:
    """Helper class for building Gmail search queries."""
    
    @staticmethod
    def build_query(email_filter: EmailFilter) -> str:
        """Build Gmail search query from filter parameters.
        
        Args:
            email_filter: Email filter configuration
            
        Returns:
            Gmail search query string
        """
        query_parts = []
        
        # Add custom query if provided
        if email_filter.query:
            query_parts.append(email_filter.query)
        
        # Add date range filter
        if email_filter.date_range:
            start_date, end_date = email_filter.date_range
            start_str = start_date.strftime('%Y/%m/%d')
            end_str = end_date.strftime('%Y/%m/%d')
            query_parts.append(f'after:{start_str} before:{end_str}')
        
        # Add sender filters
        if email_filter.sender_filters:
            sender_query = ' OR '.join([f'from:{sender}' for sender in email_filter.sender_filters])
            if len(email_filter.sender_filters) > 1:
                sender_query = f'({sender_query})'
            query_parts.append(sender_query)
        
        # Add exclude labels (convert to negative query)
        if email_filter.exclude_labels:
            for label in email_filter.exclude_labels:
                query_parts.append(f'-label:{label}')
        
        return ' '.join(query_parts) if query_parts else None
    
    @staticmethod
    def build_date_query(start_date: datetime, end_date: datetime) -> str:
        """Build date range query.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Date range query string
        """
        start_str = start_date.strftime('%Y/%m/%d')
        end_str = end_date.strftime('%Y/%m/%d')
        return f'after:{start_str} before:{end_str}'
    
    @staticmethod
    def build_sender_query(senders: List[str]) -> str:
        """Build sender filter query.
        
        Args:
            senders: List of sender email addresses or domains
            
        Returns:
            Sender filter query string
        """
        if not senders:
            return ''
        
        sender_parts = []
        for sender in senders:
            # Handle domain filters (e.g., "@example.com")
            if sender.startswith('@'):
                sender_parts.append(f'from:{sender}')
            # Handle email addresses
            elif '@' in sender:
                sender_parts.append(f'from:{sender}')
            # Handle partial matches
            else:
                sender_parts.append(f'from:*{sender}*')
        
        query = ' OR '.join(sender_parts)
        return f'({query})' if len(sender_parts) > 1 else query


class RateLimiter:
    """Rate limiter with exponential backoff and jitter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.last_request_time = 0.0
        self.request_count = 0
        self.window_start = time.time()
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        # Reset window if needed (1-second window)
        if current_time - self.window_start >= 1.0:
            self.window_start = current_time
            self.request_count = 0
        
        # Check if we need to wait
        if self.request_count >= self.config.requests_per_second:
            wait_time = 1.0 - (current_time - self.window_start)
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.window_start = time.time()
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = time.time()
    
    def handle_rate_limit_error(self, attempt: int) -> float:
        """Calculate delay for rate limit error with exponential backoff."""
        if attempt >= self.config.max_retries:
            raise Exception(f"Max retries ({self.config.max_retries}) exceeded")
        
        # Exponential backoff: base_delay * 2^attempt
        delay = min(
            self.config.base_delay * (2 ** attempt),
            self.config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        self.logger.warning(f"Rate limit hit, backing off for {delay:.2f} seconds (attempt {attempt + 1})")
        return delay


class GmailAPIClient:
    """Gmail API client with rate limiting and batch processing."""
    
    def __init__(
        self,
        credentials: Credentials,
        rate_limit_config: Optional[RateLimitConfig] = None,
        batch_config: Optional[BatchConfig] = None
    ):
        """Initialize Gmail API client.
        
        Args:
            credentials: Google OAuth2 credentials
            rate_limit_config: Rate limiting configuration
            batch_config: Batch processing configuration
        """
        self.credentials = credentials
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.batch_config = batch_config or BatchConfig()
        
        self.service = build('gmail', 'v1', credentials=credentials)
        self.rate_limiter = RateLimiter(self.rate_limit_config)
        self.logger = logging.getLogger(__name__)
        
        # Enhanced logging components
        from ..utils.logging import get_api_logger, get_error_logger
        from ..utils.process_state import NetworkInterruptionHandler
        self.api_logger = get_api_logger(__name__)
        self.error_logger = get_error_logger(__name__)
        self.network_handler = NetworkInterruptionHandler(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0
        )
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'batch_requests': 0
        }
    
    def list_messages(
        self,
        query: Optional[str] = None,
        max_results: int = 100,
        label_ids: Optional[List[str]] = None,
        include_spam_trash: bool = False,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List messages with rate limiting.
        
        Args:
            query: Gmail search query string
            max_results: Maximum number of messages to return (1-500)
            label_ids: List of label IDs to filter by
            include_spam_trash: Include SPAM and TRASH messages
            page_token: Token for pagination
            
        Returns:
            Dictionary containing messages list and pagination info
            
        Raises:
            HttpError: If API request fails after retries
        """
        # Validate parameters
        max_results = min(max(1, max_results), 500)
        
        params = {
            'userId': 'me',
            'maxResults': max_results,
            'includeSpamTrash': include_spam_trash
        }
        
        if query:
            params['q'] = query
        if label_ids:
            params['labelIds'] = label_ids
        if page_token:
            params['pageToken'] = page_token
        
        return self._make_request_with_retry(
            lambda: self.service.users().messages().list(**params).execute()
        )
    
    def get_message(self, message_id: str, format: str = 'full') -> Dict[str, Any]:
        """Get a single message with rate limiting.
        
        Args:
            message_id: Gmail message ID
            format: Message format ('minimal', 'full', 'raw', 'metadata')
            
        Returns:
            Message data dictionary
            
        Raises:
            HttpError: If API request fails after retries
        """
        return self._make_request_with_retry(
            lambda: self.service.users().messages().get(
                userId='me',
                id=message_id,
                format=format
            ).execute()
        )
    
    def get_messages_batch(
        self,
        message_ids: List[str],
        format: str = 'full',
        callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Get multiple messages using batch requests.
        
        Args:
            message_ids: List of Gmail message IDs
            format: Message format for all messages
            callback: Optional callback for individual message responses
            
        Returns:
            List of message data dictionaries
            
        Raises:
            HttpError: If batch request fails
        """
        if not message_ids:
            return []
        
        results = []
        errors = []
        
        # Process messages in batches
        for i in range(0, len(message_ids), self.batch_config.batch_size):
            batch_ids = message_ids[i:i + self.batch_config.batch_size]
            batch_results, batch_errors = self._execute_batch_get_messages(
                batch_ids, format, callback
            )
            results.extend(batch_results)
            errors.extend(batch_errors)
        
        if errors:
            self.logger.warning(f"Batch processing had {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                self.logger.warning(f"Batch error: {error}")
        
        self.stats['batch_requests'] += 1
        return results
    
    def _execute_batch_get_messages(
        self,
        message_ids: List[str],
        format: str,
        callback: Optional[Callable]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Execute a single batch request for getting messages.
        
        Args:
            message_ids: List of message IDs for this batch
            format: Message format
            callback: Optional callback function
            
        Returns:
            Tuple of (results, errors)
        """
        results = []
        errors = []
        
        def batch_callback(request_id: str, response: Dict, exception: Exception):
            """Handle individual batch request responses."""
            if exception:
                error_msg = f"Message {request_id}: {str(exception)}"
                errors.append(error_msg)
                self.stats['failed_requests'] += 1
            else:
                results.append(response)
                self.stats['successful_requests'] += 1
                
                # Call user-provided callback if available
                if callback:
                    try:
                        callback(request_id, response, None)
                    except Exception as cb_error:
                        self.logger.error(f"Callback error for {request_id}: {cb_error}")
        
        # Wait for rate limiting before creating batch
        self.rate_limiter.wait_if_needed()
        
        # Create and execute batch request
        batch = self.service.new_batch_http_request(callback=batch_callback)
        
        for msg_id in message_ids:
            batch.add(
                self.service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format=format
                ),
                request_id=msg_id
            )
        
        try:
            batch.execute()
            self.stats['total_requests'] += len(message_ids)
        except HttpError as e:
            self.logger.error(f"Batch request failed: {e}")
            # Add all message IDs to errors
            for msg_id in message_ids:
                errors.append(f"Message {msg_id}: Batch request failed - {str(e)}")
            self.stats['failed_requests'] += len(message_ids)
        
        return results, errors
    
    def _make_request_with_retry(self, request_func: Callable) -> Any:
        """Make API request with exponential backoff retry logic and enhanced logging.
        
        Args:
            request_func: Function that makes the API request
            
        Returns:
            API response
            
        Raises:
            HttpError: If request fails after all retries
        """
        operation_name = getattr(request_func, '__name__', 'unknown_operation')
        
        # Use network interruption handler for robust API calls
        with self.network_handler.handle_network_operation(
            operation_name=operation_name,
            context_data={'max_retries': self.rate_limit_config.max_retries}
        ) as network_attempt:
            
            for attempt in range(self.rate_limit_config.max_retries + 1):
                # Log API call attempt with timing
                with self.api_logger.time_api_call("Gmail", operation_name, "gmail_api", attempt) as request_id:
                    try:
                        # Wait for rate limiting
                        self.rate_limiter.wait_if_needed()
                        
                        # Make the request
                        response = request_func()
                        self.stats['total_requests'] += 1
                        self.stats['successful_requests'] += 1
                        
                        # Log successful API call
                        self.api_logger.log_api_call(
                            api_name="Gmail",
                            method=operation_name,
                            endpoint="gmail_api",
                            status_code=200,
                            retry_attempt=attempt,
                            request_id=request_id
                        )
                        
                        return response
                        
                    except HttpError as e:
                        self.stats['total_requests'] += 1
                        self.stats['failed_requests'] += 1
                        
                        # Check if it's a rate limit error (429 or 403 with rate limit message)
                        is_rate_limit = (e.resp.status == 429 or 
                                       (e.resp.status == 403 and 'rate' in str(e).lower()))
                        
                        if is_rate_limit:
                            self.stats['rate_limit_hits'] += 1
                            
                            # Log rate limit hit
                            self.api_logger.log_api_call(
                                api_name="Gmail",
                                method=operation_name,
                                endpoint="gmail_api",
                                status_code=e.resp.status,
                                error=str(e),
                                retry_attempt=attempt,
                                rate_limited=True
                            )
                            
                            if attempt < self.rate_limit_config.max_retries:
                                delay = self.rate_limiter.handle_rate_limit_error(attempt)
                                time.sleep(delay)
                                continue
                            else:
                                self.error_logger.log_error(
                                    error=e,
                                    component="GmailAPIClient",
                                    operation=operation_name,
                                    context_data={
                                        'attempt': attempt,
                                        'status_code': e.resp.status,
                                        'rate_limit_hits': self.stats['rate_limit_hits']
                                    },
                                    recovery_action="Wait longer before retrying or reduce request rate",
                                    request_id=request_id
                                )
                                raise
                        
                        # For non-rate-limit errors, log and don't retry
                        self.api_logger.log_api_call(
                            api_name="Gmail",
                            method=operation_name,
                            endpoint="gmail_api",
                            status_code=e.resp.status,
                            error=str(e),
                            retry_attempt=attempt
                        )
                        
                        self.error_logger.log_error(
                            error=e,
                            component="GmailAPIClient",
                            operation=operation_name,
                            context_data={
                                'status_code': e.resp.status,
                                'attempt': attempt
                            },
                            recovery_action="Check API credentials and permissions",
                            request_id=request_id
                        )
                        raise
                    
                    except Exception as e:
                        self.stats['total_requests'] += 1
                        self.stats['failed_requests'] += 1
                        
                        # Log unexpected error
                        self.api_logger.log_api_call(
                            api_name="Gmail",
                            method=operation_name,
                            endpoint="gmail_api",
                            error=str(e),
                            retry_attempt=attempt
                        )
                        
                        self.error_logger.log_error(
                            error=e,
                            component="GmailAPIClient",
                            operation=operation_name,
                            context_data={'attempt': attempt},
                            recovery_action="Check network connection and API configuration",
                            request_id=request_id
                        )
                        raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_requests'] / max(1, self.stats['total_requests'])
            ),
            'rate_limit_config': {
                'requests_per_second': self.rate_limit_config.requests_per_second,
                'max_retries': self.rate_limit_config.max_retries,
                'base_delay': self.rate_limit_config.base_delay,
                'max_delay': self.rate_limit_config.max_delay
            },
            'batch_config': {
                'batch_size': self.batch_config.batch_size,
                'max_concurrent_batches': self.batch_config.max_concurrent_batches
            }
        }
    
    def reset_stats(self) -> None:
        """Reset client statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limit_hits': 0,
            'batch_requests': 0
        }
    
    def list_messages_filtered(
        self,
        email_filter: EmailFilter,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List messages with advanced filtering.
        
        Args:
            email_filter: Email filter configuration
            page_token: Token for pagination
            
        Returns:
            Dictionary containing messages list and pagination info
        """
        # Build query from filter
        query = QueryBuilder.build_query(email_filter)
        
        # Use filter parameters
        max_results = email_filter.max_results or 100
        label_ids = email_filter.label_ids
        include_spam_trash = email_filter.include_spam_trash
        
        return self.list_messages(
            query=query,
            max_results=max_results,
            label_ids=label_ids,
            include_spam_trash=include_spam_trash,
            page_token=page_token
        )
    
    def list_messages_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 100,
        label_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """List messages within a specific date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_results: Maximum number of messages to return
            label_ids: List of label IDs to filter by
            
        Returns:
            Dictionary containing messages list and pagination info
        """
        query = QueryBuilder.build_date_query(start_date, end_date)
        
        return self.list_messages(
            query=query,
            max_results=max_results,
            label_ids=label_ids
        )
    
    def list_messages_by_sender(
        self,
        senders: Union[str, List[str]],
        max_results: int = 100,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """List messages from specific senders.
        
        Args:
            senders: Single sender or list of senders (email addresses or domains)
            max_results: Maximum number of messages to return
            date_range: Optional date range filter
            
        Returns:
            Dictionary containing messages list and pagination info
        """
        if isinstance(senders, str):
            senders = [senders]
        
        query_parts = [QueryBuilder.build_sender_query(senders)]
        
        if date_range:
            start_date, end_date = date_range
            query_parts.append(QueryBuilder.build_date_query(start_date, end_date))
        
        query = ' '.join(query_parts)
        
        return self.list_messages(
            query=query,
            max_results=max_results
        )
    
    def paginate_messages(
        self,
        email_filter: Optional[EmailFilter] = None,
        max_total_results: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Paginate through all messages matching the filter.
        
        Args:
            email_filter: Email filter configuration
            max_total_results: Maximum total messages to retrieve
            
        Yields:
            Individual message dictionaries
        """
        page_token = None
        total_retrieved = 0
        
        while True:
            # Get next page of messages
            if email_filter:
                result = self.list_messages_filtered(email_filter, page_token)
            else:
                result = self.list_messages(page_token=page_token)
            
            messages = result.get('messages', [])
            
            if not messages:
                break
            
            # Yield messages one by one
            for message in messages:
                if max_total_results and total_retrieved >= max_total_results:
                    return
                
                yield message
                total_retrieved += 1
            
            # Check for next page
            page_token = result.get('nextPageToken')
            if not page_token:
                break
            
            self.logger.debug(f"Retrieved {total_retrieved} messages so far, continuing pagination")
    
    def get_all_messages_filtered(
        self,
        email_filter: EmailFilter,
        max_total_results: Optional[int] = None,
        fetch_full_messages: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all messages matching the filter with pagination.
        
        Args:
            email_filter: Email filter configuration
            max_total_results: Maximum total messages to retrieve
            fetch_full_messages: Whether to fetch full message content
            
        Returns:
            List of message dictionaries
        """
        messages = []
        message_ids = []
        
        # Collect all message IDs first
        for message in self.paginate_messages(email_filter, max_total_results):
            if fetch_full_messages:
                message_ids.append(message['id'])
            else:
                messages.append(message)
        
        # If fetching full messages, use batch processing
        if fetch_full_messages and message_ids:
            self.logger.info(f"Fetching full content for {len(message_ids)} messages")
            messages = self.get_messages_batch(message_ids)
        
        return messages
    
    def search_messages(
        self,
        search_query: str,
        max_results: int = 100,
        label_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search messages using Gmail search syntax.
        
        Args:
            search_query: Gmail search query (e.g., "from:example@gmail.com subject:important")
            max_results: Maximum number of messages to return
            label_ids: List of label IDs to filter by
            
        Returns:
            Dictionary containing messages list and pagination info
        """
        return self.list_messages(
            query=search_query,
            max_results=max_results,
            label_ids=label_ids
        )
    
    def get_labels(self) -> List[Dict[str, Any]]:
        """Get all labels for the user.
        
        Returns:
            List of label dictionaries
        """
        return self._make_request_with_retry(
            lambda: self.service.users().labels().list(userId='me').execute()
        ).get('labels', [])
    
    def find_label_id(self, label_name: str) -> Optional[str]:
        """Find label ID by name.
        
        Args:
            label_name: Label name to search for
            
        Returns:
            Label ID if found, None otherwise
        """
        labels = self.get_labels()
        for label in labels:
            if label.get('name', '').lower() == label_name.lower():
                return label.get('id')
        return None
    
    def get_message_count_estimate(
        self,
        email_filter: Optional[EmailFilter] = None
    ) -> int:
        """Get estimated count of messages matching the filter.
        
        Args:
            email_filter: Email filter configuration
            
        Returns:
            Estimated message count
        """
        if email_filter:
            result = self.list_messages_filtered(email_filter)
        else:
            result = self.list_messages(max_results=1)
        
        return result.get('resultSizeEstimate', 0)