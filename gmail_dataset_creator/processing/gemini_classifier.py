"""
Gemini API-based email classification module.

This module handles email classification using Google's Gemini API, including
confidence scoring, uncertainty handling, and batch processing capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import pickle
import os

from google import genai
from google.genai import types

from ..models import EmailData, ClassificationResult, CATEGORIES, validate_category
from ..config.manager import GeminiAPIConfig


logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    batch_size: int = 5
    max_concurrent_batches: int = 2
    retry_failed_emails: bool = True
    save_progress: bool = True
    progress_file: str = "classification_progress.pkl"


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    backoff_factor: float = 2.0
    max_backoff_time: float = 300.0  # 5 minutes
    jitter: bool = True


@dataclass
class ClassificationPrompt:
    """Template for email classification prompts."""
    
    SYSTEM_PROMPT = """You are an expert email classifier. Your task is to categorize emails into one of these specific categories:

Categories:
- newsletter: Newsletters, mailing lists, regular updates from organizations
- work: Work-related emails, business communications, professional correspondence
- personal: Personal emails from friends, family, personal communications
- spam: Unwanted emails, suspicious content, obvious spam
- promotional: Marketing emails, sales offers, promotional content from businesses
- social: Social media notifications, social platform updates
- finance: Banking, financial services, investment, payment notifications
- travel: Travel bookings, airline notifications, hotel confirmations, travel services
- shopping: E-commerce, online shopping, order confirmations, shipping notifications
- other: Any email that doesn't clearly fit the above categories

Instructions:
1. Analyze the email subject, body content, and sender information
2. Choose the MOST APPROPRIATE single category
3. Provide a confidence score between 0.0 and 1.0
4. Give a brief reasoning for your classification
5. Flag for manual review if confidence is below 0.7

Respond in this exact JSON format:
{
    "category": "category_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this category was chosen",
    "needs_review": false
}"""

    @staticmethod
    def create_classification_prompt(email_data: EmailData) -> str:
        """
        Create a classification prompt for an email.
        
        Args:
            email_data: Email data to classify
            
        Returns:
            Formatted prompt string
        """
        # Truncate content for API efficiency
        subject = email_data.subject[:200] if email_data.subject else ""
        body_preview = email_data.body[:1000] if email_data.body else ""
        
        # Extract domain from sender for context
        sender_domain = ""
        if email_data.sender and "@" in email_data.sender:
            try:
                sender_domain = email_data.sender.split("@")[1].lower()
            except IndexError:
                sender_domain = email_data.sender
        
        prompt = f"""
Email to classify:

Subject: {subject}
Sender Domain: {sender_domain}
Body Preview: {body_preview}

Please classify this email according to the instructions above.
"""
        return prompt


class GeminiClassifier:
    """
    Email classifier using Google's Gemini API.
    
    Provides email classification with confidence scoring, batch processing,
    and error handling capabilities.
    """
    
    def __init__(self, 
                 config: GeminiAPIConfig, 
                 confidence_threshold: float = 0.7,
                 batch_config: Optional[BatchProcessingConfig] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize the Gemini classifier.
        
        Args:
            config: Gemini API configuration
            confidence_threshold: Minimum confidence for automatic classification
            batch_config: Configuration for batch processing
            rate_limit_config: Configuration for rate limiting
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.batch_config = batch_config or BatchProcessingConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.client = None
        self._initialize_client()
        
        # Enhanced logging components
        from ..utils.logging import get_api_logger, get_error_logger, get_progress_tracker
        self.api_logger = get_api_logger(__name__)
        self.error_logger = get_error_logger(__name__)
        self.progress_tracker = get_progress_tracker(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 60.0 / self.rate_limit_config.requests_per_minute
        self.request_times = []  # Track request times for rate limiting
        self._rate_limit_lock = threading.Lock()
        
        # Batch processing
        self._processing_queue = Queue()
        self._result_queue = Queue()
        self._failed_emails = []
        self._progress_data = {}
        
        # Statistics
        self.total_classifications = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        self.low_confidence_classifications = 0
        self.api_errors = {}  # Track different types of API errors
        self.retry_attempts = 0
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini API client."""
        try:
            if not self.config.api_key:
                raise ValueError("Gemini API key is required")
            
            self.client = genai.Client(api_key=self.config.api_key)
            logger.info("Gemini API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API client: {e}")
            raise
    
    def classify_email(self, email_data: EmailData) -> ClassificationResult:
        """
        Classify a single email using Gemini API.
        
        Args:
            email_data: Email data to classify
            
        Returns:
            Classification result with category, confidence, and metadata
            
        Raises:
            ValueError: If email data is invalid
            RuntimeError: If API call fails after retries
        """
        if not email_data or not email_data.id:
            raise ValueError("Invalid email data provided")
        
        self.total_classifications += 1
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Create classification prompt
            prompt = ClassificationPrompt.create_classification_prompt(email_data)
            
            # Make API call with retry logic
            result = self._classify_with_retry(prompt, email_data.id)
            
            # Validate and process result
            classification = self._process_classification_result(result, email_data.id)
            
            # Update statistics
            if classification.confidence < self.confidence_threshold:
                self.low_confidence_classifications += 1
                classification.needs_review = True
            
            self.successful_classifications += 1
            logger.debug(f"Successfully classified email {email_data.id}: {classification.category} (confidence: {classification.confidence:.2f})")
            
            return classification
            
        except Exception as e:
            self.failed_classifications += 1
            logger.error(f"Failed to classify email {email_data.id}: {e}")
            
            # Try to handle specific API errors
            try:
                return self.handle_api_error(e, email_data.id)
            except Exception:
                # If error handling also fails, use fallback
                return self._create_fallback_classification(email_data, str(e))
    
    def classify_batch(self, 
                      emails: List[EmailData], 
                      batch_size: Optional[int] = None,
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      resume_from_checkpoint: bool = True) -> List[ClassificationResult]:
        """
        Classify multiple emails in batches with advanced error handling and progress tracking.
        
        Args:
            emails: List of email data to classify
            batch_size: Number of emails to process in each batch (uses config default if None)
            progress_callback: Optional callback function for progress updates
            resume_from_checkpoint: Whether to resume from saved progress
            
        Returns:
            List of classification results in the same order as input
        """
        if not emails:
            return []
        
        batch_size = batch_size or self.batch_config.batch_size
        
        # Start progress tracking
        self.progress_tracker.start_stage("email_classification", len(emails))
        logger.info(f"Starting batch classification of {len(emails)} emails (batch size: {batch_size})")
        
        # Load progress if resuming
        results = [None] * len(emails)  # Pre-allocate results list
        processed_indices = set()
        
        if resume_from_checkpoint and self.batch_config.save_progress:
            processed_indices, partial_results = self._load_progress(emails)
            for idx, result in partial_results.items():
                if idx < len(results):
                    results[idx] = result
            
            if processed_indices:
                logger.info(f"Resuming from checkpoint: {len(processed_indices)} emails already processed")
                # Update progress tracker with resumed state
                self.progress_tracker.update_progress(
                    processed_items=len(processed_indices),
                    force_log=True
                )
        
        # Filter out already processed emails
        remaining_emails = [(i, email) for i, email in enumerate(emails) if i not in processed_indices]
        
        if not remaining_emails:
            logger.info("All emails already processed")
            self.progress_tracker.complete_stage()
            return [r for r in results if r is not None]
        
        logger.info(f"Processing {len(remaining_emails)} remaining emails")
        
        # Process in batches with concurrent processing
        failed_emails = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.batch_config.max_concurrent_batches) as executor:
                # Submit batch processing tasks
                future_to_batch = {}
                
                for batch_start in range(0, len(remaining_emails), batch_size):
                    batch_end = min(batch_start + batch_size, len(remaining_emails))
                    batch_emails = remaining_emails[batch_start:batch_end]
                    
                    future = executor.submit(self._process_batch_with_retry, batch_emails)
                    future_to_batch[future] = (batch_start, batch_emails)
                
                # Collect results as they complete
                completed_batches = 0
                total_batches = len(future_to_batch)
                
                for future in as_completed(future_to_batch):
                    batch_start, batch_emails = future_to_batch[future]
                    completed_batches += 1
                    
                    try:
                        batch_results = future.result()
                        
                        # Store results in correct positions
                        for (original_idx, email_data), result in zip(batch_emails, batch_results):
                            results[original_idx] = result
                            processed_indices.add(original_idx)
                        
                        # Update progress tracking
                        failed_count = sum(1 for r in batch_results if r and r.needs_review)
                        self.progress_tracker.update_progress(
                            processed_items=len(processed_indices),
                            failed_items=failed_count,
                            current_item_id=batch_emails[-1][1].id if batch_emails else None
                        )
                        
                        # Save progress
                        if self.batch_config.save_progress:
                            self._save_progress(emails, processed_indices, results)
                        
                        # Call progress callback
                        if progress_callback:
                            progress_callback(len(processed_indices), len(emails))
                        
                        logger.debug(f"Completed batch {completed_batches}/{total_batches}")
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        # Add failed emails to retry list
                        for original_idx, email_data in batch_emails:
                            failed_emails.append((original_idx, email_data))
        
        except KeyboardInterrupt:
            logger.warning("Batch processing interrupted by user")
            # Save current progress before exiting
            if self.batch_config.save_progress:
                self._save_progress(emails, processed_indices, results)
            raise
        
        # Retry failed emails if configured
        if failed_emails and self.batch_config.retry_failed_emails:
            logger.info(f"Retrying {len(failed_emails)} failed emails")
            retry_results = self._retry_failed_emails(failed_emails)
            
            for (original_idx, _), result in zip(failed_emails, retry_results):
                results[original_idx] = result
                processed_indices.add(original_idx)
        
        # Fill any remaining None results with fallback classifications
        for i, result in enumerate(results):
            if result is None:
                logger.warning(f"Creating fallback classification for email {emails[i].id}")
                results[i] = self._create_fallback_classification(emails[i], "No classification result available")
        
        # Clean up progress file if all emails processed successfully
        if self.batch_config.save_progress and len(processed_indices) == len(emails):
            self._cleanup_progress_file()
        
        # Complete progress tracking
        successful_count = sum(1 for r in results if r and not r.needs_review)
        failed_count = len(results) - successful_count
        
        self.progress_tracker.update_progress(
            processed_items=len(results),
            failed_items=failed_count,
            force_log=True
        )
        self.progress_tracker.complete_stage()
        
        logger.info(f"Batch classification completed. Total: {len(results)}, Successful: {successful_count}, Failed/Low confidence: {failed_count}")
        
        return results
    
    def _apply_rate_limiting(self) -> None:
        """Apply advanced rate limiting to API requests."""
        with self._rate_limit_lock:
            current_time = time.time()
            
            # Clean old request times (older than 1 hour)
            self.request_times = [t for t in self.request_times if current_time - t < 3600]
            
            # Check hourly limit
            if len(self.request_times) >= self.rate_limit_config.requests_per_hour:
                oldest_request = min(self.request_times)
                sleep_time = 3600 - (current_time - oldest_request)
                if sleep_time > 0:
                    logger.warning(f"Hourly rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    current_time = time.time()
            
            # Check per-minute limit
            recent_requests = [t for t in self.request_times if current_time - t < 60]
            if len(recent_requests) >= self.rate_limit_config.requests_per_minute:
                oldest_recent = min(recent_requests)
                sleep_time = 60 - (current_time - oldest_recent)
                if sleep_time > 0:
                    logger.debug(f"Per-minute rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    current_time = time.time()
            
            # Apply minimum interval
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                
                # Add jitter to avoid thundering herd
                if self.rate_limit_config.jitter:
                    import random
                    sleep_time += random.uniform(0, 0.5)
                
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            # Record this request
            self.last_request_time = time.time()
            self.request_times.append(self.last_request_time)
    
    def _classify_with_retry(self, prompt: str, email_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make classification API call with advanced retry logic and error handling.
        
        Args:
            prompt: Classification prompt
            email_id: Email ID for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception = None
        backoff_time = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Classification attempt {attempt + 1} for email {email_id}")
                
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=[
                        types.Content(
                            role='user',
                            parts=[types.Part.from_text(ClassificationPrompt.SYSTEM_PROMPT)]
                        ),
                        types.Content(
                            role='user',
                            parts=[types.Part.from_text(prompt)]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=0.1,  # Low temperature for consistent classification
                        top_p=0.8,
                        top_k=40,
                        safety_settings=[
                            types.SafetySetting(
                                category='HARM_CATEGORY_HATE_SPEECH',
                                threshold='BLOCK_ONLY_HIGH'
                            ),
                            types.SafetySetting(
                                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                                threshold='BLOCK_ONLY_HIGH'
                            )
                        ]
                    )
                )
                
                if response and response.text:
                    return {"text": response.text, "usage": getattr(response, 'usage_metadata', None)}
                else:
                    raise RuntimeError("Empty response from Gemini API")
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_msg or "quota" in error_msg:
                    if attempt < max_retries:
                        # Use longer backoff for rate limit errors
                        sleep_time = min(backoff_time * (self.rate_limit_config.backoff_factor ** (attempt + 1)), 
                                       self.rate_limit_config.max_backoff_time)
                        logger.warning(f"Rate limit hit for email {email_id}. Backing off for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                        continue
                
                elif "content policy" in error_msg or "safety" in error_msg:
                    # Don't retry content policy violations
                    logger.warning(f"Content policy violation for email {email_id}: {e}")
                    raise e
                
                elif "timeout" in error_msg or "connection" in error_msg:
                    if attempt < max_retries:
                        # Shorter backoff for network errors
                        sleep_time = min(backoff_time * (1.5 ** attempt), 30.0)
                        logger.warning(f"Network error for email {email_id}. Retrying in {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                        continue
                
                logger.warning(f"Classification attempt {attempt + 1} failed for email {email_id}: {e}")
                
                if attempt < max_retries:
                    # Standard exponential backoff for other errors
                    sleep_time = min(backoff_time * (self.rate_limit_config.backoff_factor ** attempt), 
                                   self.rate_limit_config.max_backoff_time)
                    
                    # Add jitter to avoid thundering herd
                    if self.rate_limit_config.jitter:
                        import random
                        sleep_time += random.uniform(0, sleep_time * 0.1)
                    
                    logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All retry attempts failed for email {email_id}")
        
        raise RuntimeError(f"Failed to classify email {email_id} after {max_retries + 1} attempts: {last_exception}")
    
    def _process_classification_result(self, api_result: Dict[str, Any], email_id: str) -> ClassificationResult:
        """
        Process and validate API classification result.
        
        Args:
            api_result: Raw API response
            email_id: Email ID for logging
            
        Returns:
            Validated ClassificationResult
            
        Raises:
            ValueError: If result cannot be parsed or validated
        """
        try:
            response_text = api_result.get("text", "").strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result_dict = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                result_dict = json.loads(response_text)
            
            # Extract and validate fields
            category = result_dict.get("category", "").lower().strip()
            confidence = float(result_dict.get("confidence", 0.0))
            reasoning = result_dict.get("reasoning", "").strip()
            needs_review = result_dict.get("needs_review", False)
            
            # Validate category
            if not validate_category(category):
                logger.warning(f"Invalid category '{category}' for email {email_id}, defaulting to 'other'")
                category = "other"
                confidence = max(0.0, confidence - 0.3)  # Reduce confidence for invalid category
                reasoning = f"Invalid category detected, defaulting to 'other'. Original: {reasoning}"
                needs_review = True
            
            # Validate confidence
            if confidence < 0.0 or confidence > 1.0:
                logger.warning(f"Invalid confidence {confidence} for email {email_id}, clamping to valid range")
                confidence = max(0.0, min(1.0, confidence))
            
            # Set needs_review based on confidence threshold
            if confidence < self.confidence_threshold:
                needs_review = True
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                reasoning=reasoning or f"Classified as {category}",
                needs_review=needs_review
            )
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse classification result for email {email_id}: {e}")
            logger.debug(f"Raw response: {api_result.get('text', 'No text')}")
            
            # Try to extract category from text using fallback parsing
            return self._fallback_parse_result(api_result.get("text", ""), email_id)
    
    def _fallback_parse_result(self, response_text: str, email_id: str) -> ClassificationResult:
        """
        Fallback parsing when JSON parsing fails.
        
        Args:
            response_text: Raw response text
            email_id: Email ID for logging
            
        Returns:
            Best-effort ClassificationResult
        """
        logger.debug(f"Attempting fallback parsing for email {email_id}")
        
        # Look for category names in the response
        response_lower = response_text.lower()
        found_category = "other"
        confidence = 0.3  # Low confidence for fallback parsing
        
        for category in CATEGORIES.keys():
            if category in response_lower:
                found_category = category
                confidence = 0.5  # Slightly higher confidence if category found
                break
        
        return ClassificationResult(
            category=found_category,
            confidence=confidence,
            reasoning=f"Fallback parsing from response: {response_text[:100]}...",
            needs_review=True
        )
    
    def _create_fallback_classification(self, email_data: EmailData, error_msg: str) -> ClassificationResult:
        """
        Create a fallback classification when API fails.
        
        Args:
            email_data: Original email data
            error_msg: Error message
            
        Returns:
            Fallback ClassificationResult
        """
        # Simple rule-based fallback classification
        category = self._rule_based_classification(email_data)
        
        return ClassificationResult(
            category=category,
            confidence=0.2,  # Very low confidence for fallback
            reasoning=f"API classification failed: {error_msg}. Used rule-based fallback.",
            needs_review=True
        )
    
    def _rule_based_classification(self, email_data: EmailData) -> str:
        """
        Simple rule-based classification as fallback.
        
        Args:
            email_data: Email data to classify
            
        Returns:
            Category name based on simple rules
        """
        subject = (email_data.subject or "").lower()
        body = (email_data.body or "").lower()
        sender = (email_data.sender or "").lower()
        
        # Simple keyword-based rules
        if any(word in subject + body for word in ["unsubscribe", "newsletter", "weekly", "monthly"]):
            return "newsletter"
        elif any(word in subject + body for word in ["work", "meeting", "project", "deadline", "office"]):
            return "work"
        elif any(word in subject + body for word in ["sale", "discount", "offer", "promo", "deal"]):
            return "promotional"
        elif any(word in subject + body for word in ["bank", "payment", "invoice", "transaction", "account"]):
            return "finance"
        elif any(word in subject + body for word in ["flight", "hotel", "booking", "travel", "trip"]):
            return "travel"
        elif any(word in subject + body for word in ["order", "shipping", "delivery", "purchase", "cart"]):
            return "shopping"
        elif any(word in subject + body for word in ["facebook", "twitter", "instagram", "linkedin", "social"]):
            return "social"
        elif "noreply" in sender or "no-reply" in sender:
            return "promotional"
        else:
            return "other"
    
    def get_confidence_score(self, classification: str) -> float:
        """
        Get confidence score for a classification (deprecated - kept for compatibility).
        
        Args:
            classification: Category name
            
        Returns:
            Default confidence score
        """
        logger.warning("get_confidence_score is deprecated. Confidence is now included in ClassificationResult.")
        return 0.8  # Default confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Dictionary with classification statistics
        """
        success_rate = 0.0
        if self.total_classifications > 0:
            success_rate = self.successful_classifications / self.total_classifications
        
        return {
            "total_classifications": self.total_classifications,
            "successful_classifications": self.successful_classifications,
            "failed_classifications": self.failed_classifications,
            "low_confidence_classifications": self.low_confidence_classifications,
            "success_rate": success_rate,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_statistics(self) -> None:
        """Reset classification statistics."""
        self.total_classifications = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        self.low_confidence_classifications = 0
        self.api_errors = {}
        self.retry_attempts = 0
        logger.info("Classification statistics reset")
    
    def _process_batch_with_retry(self, batch_emails: List[Tuple[int, EmailData]]) -> List[ClassificationResult]:
        """
        Process a batch of emails with retry logic.
        
        Args:
            batch_emails: List of (index, EmailData) tuples
            
        Returns:
            List of classification results
        """
        results = []
        
        for original_idx, email_data in batch_emails:
            try:
                result = self.classify_email(email_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify email {email_data.id}: {e}")
                # Track error types
                error_type = type(e).__name__
                self.api_errors[error_type] = self.api_errors.get(error_type, 0) + 1
                
                # Create fallback result
                fallback_result = self._create_fallback_classification(email_data, str(e))
                results.append(fallback_result)
        
        return results
    
    def _retry_failed_emails(self, failed_emails: List[Tuple[int, EmailData]]) -> List[ClassificationResult]:
        """
        Retry classification for failed emails with exponential backoff.
        
        Args:
            failed_emails: List of (index, EmailData) tuples that failed
            
        Returns:
            List of classification results
        """
        results = []
        backoff_time = 1.0
        
        for original_idx, email_data in failed_emails:
            max_retries = 3
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    self.retry_attempts += 1
                    
                    # Apply exponential backoff
                    if attempt > 0:
                        sleep_time = min(backoff_time * (self.rate_limit_config.backoff_factor ** attempt), 
                                       self.rate_limit_config.max_backoff_time)
                        logger.debug(f"Retrying email {email_data.id} after {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                    
                    result = self.classify_email(email_data)
                    results.append(result)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Retry attempt {attempt + 1} failed for email {email_data.id}: {e}")
                    
                    if attempt == max_retries - 1:
                        # Final attempt failed, create fallback
                        logger.error(f"All retry attempts failed for email {email_data.id}")
                        fallback_result = self._create_fallback_classification(email_data, str(last_exception))
                        results.append(fallback_result)
        
        return results
    
    def _save_progress(self, emails: List[EmailData], processed_indices: set, results: List[ClassificationResult]) -> None:
        """
        Save classification progress to file.
        
        Args:
            emails: Original email list
            processed_indices: Set of processed email indices
            results: Current results list
        """
        try:
            progress_data = {
                'processed_indices': processed_indices,
                'results': {i: results[i] for i in processed_indices if i < len(results) and results[i] is not None},
                'email_ids': [email.id for email in emails],
                'timestamp': time.time()
            }
            
            with open(self.batch_config.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            logger.debug(f"Progress saved: {len(processed_indices)} emails processed")
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _load_progress(self, emails: List[EmailData]) -> Tuple[set, Dict[int, ClassificationResult]]:
        """
        Load classification progress from file.
        
        Args:
            emails: Current email list
            
        Returns:
            Tuple of (processed_indices, results_dict)
        """
        try:
            if not os.path.exists(self.batch_config.progress_file):
                return set(), {}
            
            with open(self.batch_config.progress_file, 'rb') as f:
                progress_data = pickle.load(f)
            
            # Validate that the email list matches
            saved_email_ids = progress_data.get('email_ids', [])
            current_email_ids = [email.id for email in emails]
            
            if saved_email_ids != current_email_ids:
                logger.warning("Email list has changed since last progress save. Starting fresh.")
                return set(), {}
            
            processed_indices = progress_data.get('processed_indices', set())
            results_dict = progress_data.get('results', {})
            
            logger.info(f"Loaded progress: {len(processed_indices)} emails already processed")
            return processed_indices, results_dict
            
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return set(), {}
    
    def _cleanup_progress_file(self) -> None:
        """Clean up progress file after successful completion."""
        try:
            if os.path.exists(self.batch_config.progress_file):
                os.remove(self.batch_config.progress_file)
                logger.debug("Progress file cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup progress file: {e}")
    
    def handle_api_error(self, error: Exception, email_id: str) -> ClassificationResult:
        """
        Handle specific API errors with appropriate responses.
        
        Args:
            error: The API error that occurred
            email_id: ID of the email being processed
            
        Returns:
            Appropriate ClassificationResult based on error type
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Track error for statistics
        self.api_errors[error_type] = self.api_errors.get(error_type, 0) + 1
        
        # Handle specific error types
        if "rate limit" in error_msg or "quota" in error_msg:
            logger.warning(f"Rate limit hit for email {email_id}. Will retry with backoff.")
            # This should trigger a retry with longer backoff
            raise error
        
        elif "content policy" in error_msg or "safety" in error_msg:
            logger.warning(f"Content policy violation for email {email_id}. Classifying as spam.")
            return ClassificationResult(
                category="spam",
                confidence=0.8,
                reasoning="Classified as spam due to content policy violation",
                needs_review=True
            )
        
        elif "timeout" in error_msg or "connection" in error_msg:
            logger.warning(f"Network error for email {email_id}. Will retry.")
            # This should trigger a retry
            raise error
        
        elif "invalid" in error_msg or "malformed" in error_msg:
            logger.error(f"Invalid request for email {email_id}. Using fallback classification.")
            return self._create_fallback_classification_from_content(email_id)
        
        else:
            # Unknown error, use generic fallback
            logger.error(f"Unknown API error for email {email_id}: {error}")
            return self._create_fallback_classification_from_content(email_id)
    
    def _create_fallback_classification_from_content(self, email_id: str) -> ClassificationResult:
        """
        Create fallback classification when we don't have email content.
        
        Args:
            email_id: Email ID
            
        Returns:
            Generic fallback classification
        """
        return ClassificationResult(
            category="other",
            confidence=0.1,
            reasoning=f"Fallback classification due to API error for email {email_id}",
            needs_review=True
        )
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get detailed classification statistics including error breakdown.
        
        Returns:
            Dictionary with detailed statistics
        """
        base_stats = self.get_statistics()
        
        detailed_stats = {
            **base_stats,
            "api_errors": self.api_errors.copy(),
            "retry_attempts": self.retry_attempts,
            "rate_limit_config": {
                "requests_per_minute": self.rate_limit_config.requests_per_minute,
                "requests_per_hour": self.rate_limit_config.requests_per_hour,
                "current_request_count": len(self.request_times)
            },
            "batch_config": {
                "batch_size": self.batch_config.batch_size,
                "max_concurrent_batches": self.batch_config.max_concurrent_batches,
                "retry_failed_emails": self.batch_config.retry_failed_emails
            }
        }
        
        return detailed_stats