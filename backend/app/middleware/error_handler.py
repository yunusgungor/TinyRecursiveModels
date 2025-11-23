"""Error handling middleware"""

import uuid
from datetime import datetime
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import BaseAPIException, ValidationError, ModelInferenceError, ModelLoadError
from app.core.logging import logger
from app.models.schemas import ErrorResponse
from app.services.alert_service import alert_service


async def error_handler_middleware(request: Request, call_next: Callable) -> Response:
    """Global error handling middleware"""
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        return response
        
    except BaseAPIException as exc:
        # Log the error with full context
        logger.error(
            f"API Error: {exc.error_code}",
            extra={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True
        )
        
        # Send email alert for critical errors
        if isinstance(exc, (ModelInferenceError, ModelLoadError)):
            await alert_service.send_critical_error_alert(
                error_code=exc.error_code,
                error_message=exc.message,
                details=exc.details,
                request_id=request_id
            )
        
        error_response = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response.model_dump(mode='json')
        )
        
    except PydanticValidationError as exc:
        # Handle Pydantic validation errors
        errors = exc.errors()
        field = errors[0]["loc"][-1] if errors else "unknown"
        message = f"Geçersiz değer: {field}"
        
        logger.error(
            "Validation Error",
            extra={
                "errors": errors,
                "request_id": request_id,
                "path": request.url.path,
            }
        )
        
        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message=message,
            details={"field": field, "validation_errors": errors},
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(mode='json')
        )
        
    except Exception as exc:
        # Handle unexpected errors
        logger.error(
            "Unexpected Error",
            extra={
                "error": str(exc),
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
            exc_info=True
        )
        
        # Send email alert for unexpected errors
        await alert_service.send_critical_error_alert(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="Beklenmeyen bir hata oluştu",
            details={"error": str(exc), "path": request.url.path, "method": request.method},
            request_id=request_id
        )
        
        error_response = ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="Beklenmeyen bir hata oluştu",
            details={"error": str(exc)},
            timestamp=datetime.utcnow(),
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(mode='json')
        )
