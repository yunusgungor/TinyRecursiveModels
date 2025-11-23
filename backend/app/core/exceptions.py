"""Custom exception classes"""

from typing import Any, Dict, Optional


class BaseAPIException(Exception):
    """Base exception for API errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelInferenceError(BaseAPIException):
    """Raised when model inference fails"""
    
    def __init__(self, message: str = "Model şu anda kullanılamıyor", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MODEL_INFERENCE_ERROR",
            details=details
        )


class ModelLoadError(BaseAPIException):
    """Raised when model loading fails"""
    
    def __init__(self, message: str = "Model yüklenemedi", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details=details
        )


class TrendyolAPIError(BaseAPIException):
    """Raised when Trendyol API fails"""
    
    def __init__(self, message: str = "Ürün verileri şu anda alınamıyor", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRENDYOL_API_ERROR",
            details=details
        )


class RateLimitError(BaseAPIException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "İstek limiti aşıldı", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


class ToolExecutionError(BaseAPIException):
    """Raised when tool execution fails"""
    
    def __init__(self, message: str = "Araç çalıştırılamadı", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details=details
        )


class ToolTimeoutError(BaseAPIException):
    """Raised when tool execution times out"""
    
    def __init__(self, message: str = "Araç zaman aşımına uğradı", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TOOL_TIMEOUT_ERROR",
            details=details
        )


class ValidationError(BaseAPIException):
    """Raised when validation fails"""
    
    def __init__(self, message: str, field: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class CacheError(BaseAPIException):
    """Raised when cache operations fail"""
    
    def __init__(self, message: str = "Önbellek hatası", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details
        )
