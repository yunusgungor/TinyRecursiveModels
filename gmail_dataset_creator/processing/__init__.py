"""Email processing and classification modules."""

from .email_processor import EmailProcessor
from .gemini_classifier import GeminiClassifier, BatchProcessingConfig, RateLimitConfig

__all__ = ["EmailProcessor", "GeminiClassifier", "BatchProcessingConfig", "RateLimitConfig"]