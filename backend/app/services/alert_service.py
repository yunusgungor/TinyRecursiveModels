"""Email alert service for critical errors"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, Any

from app.core.config import settings
from app.core.logging import logger


class AlertService:
    """Service for sending email alerts on critical errors"""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.alert_email_to = settings.ALERT_EMAIL_TO
        self.alert_email_from = settings.ALERT_EMAIL_FROM
        self.enabled = settings.ENABLE_EMAIL_ALERTS
    
    async def send_critical_error_alert(
        self,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Send email alert for critical errors
        
        Args:
            error_code: Error code identifier
            error_message: Human-readable error message
            details: Additional error details
            request_id: Request ID for tracking
            
        Returns:
            True if alert sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Email alerts disabled, skipping alert")
            return False
        
        if not all([self.smtp_user, self.smtp_password, self.alert_email_to, self.alert_email_from]):
            logger.warning("Email alert configuration incomplete, skipping alert")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.alert_email_from
            msg['To'] = self.alert_email_to
            msg['Subject'] = f"[CRITICAL] Trendyol Gift API Error: {error_code}"
            
            # Build email body
            body = self._build_email_body(error_code, error_message, details, request_id)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Critical error alert sent for {error_code}", extra={"request_id": request_id})
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}", exc_info=True)
            return False
    
    def _build_email_body(
        self,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]],
        request_id: Optional[str]
    ) -> str:
        """Build HTML email body"""
        timestamp = datetime.utcnow().isoformat()
        
        html = f"""
        <html>
            <body>
                <h2 style="color: #d32f2f;">Critical Error Alert</h2>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Error Code:</strong> {error_code}</p>
                <p><strong>Message:</strong> {error_message}</p>
                <p><strong>Request ID:</strong> {request_id or 'N/A'}</p>
                
                <h3>Details:</h3>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px;">
{self._format_details(details)}
                </pre>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from Trendyol Gift Recommendation API
                </p>
            </body>
        </html>
        """
        return html
    
    def _format_details(self, details: Optional[Dict[str, Any]]) -> str:
        """Format details dictionary for display"""
        if not details:
            return "No additional details"
        
        lines = []
        for key, value in details.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


# Global alert service instance
alert_service = AlertService()
