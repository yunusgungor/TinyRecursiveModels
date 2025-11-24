"""Session management middleware"""

import time
import uuid
from typing import Callable, Dict, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import ErrorResponse


class SessionManager:
    """In-memory session manager (for production, use Redis)"""
    
    def __init__(self, timeout_minutes: int = 30):
        """
        Initialize session manager
        
        Args:
            timeout_minutes: Session timeout in minutes
        """
        self.timeout_seconds = timeout_minutes * 60
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, client_id: str) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "client_id": client_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "data": {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session by ID"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session expired
        if time.time() - session["last_activity"] > self.timeout_seconds:
            self.delete_session(session_id)
            return None
        
        return session
    
    def update_activity(self, session_id: str) -> bool:
        """Update last activity time for session"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id]["last_activity"] = time.time()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = time.time()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now - session["last_activity"] > self.timeout_seconds
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Global session manager instance
session_manager = SessionManager(timeout_minutes=settings.SESSION_TIMEOUT_MINUTES)


async def session_middleware(request: Request, call_next: Callable) -> Response:
    """Session management middleware"""
    
    # Get session ID from cookie or header
    session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
    
    # Get client identifier
    client_ip = request.client.host if request.client else "unknown"
    
    # If no session or session expired, create new one
    if not session_id or not session_manager.get_session(session_id):
        session_id = session_manager.create_session(client_ip)
        logger.info(f"Created new session: {session_id} for client: {client_ip}")
    else:
        # Update last activity
        session_manager.update_activity(session_id)
    
    # Store session ID in request state
    request.state.session_id = session_id
    
    # Process request
    response = await call_next(request)
    
    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=session_manager.timeout_seconds,
        httponly=True,
        secure=not settings.DEBUG,  # HTTPS only in production
        samesite="lax"
    )
    
    return response


async def require_session(request: Request) -> dict:
    """Dependency to require valid session"""
    session_id = getattr(request.state, "session_id", None)
    
    if not session_id:
        raise ValueError("No session ID found")
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise ValueError("Session expired or invalid")
    
    return session
