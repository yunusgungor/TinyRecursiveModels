"""Application configuration settings"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Trendyol Gift Recommendation API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Model Settings
    MODEL_CHECKPOINT_PATH: str = "../checkpoints/integrated_enhanced/integrated_enhanced_best.pt"
    MODEL_DEVICE: str = "cuda"  # cuda or cpu
    MODEL_INFERENCE_TIMEOUT: int = 5  # seconds
    
    # Trendyol API Settings
    TRENDYOL_API_KEY: str = ""
    TRENDYOL_API_BASE_URL: str = "https://api.trendyol.com"
    TRENDYOL_RATE_LIMIT: int = 100  # requests per minute
    
    # Cache Settings
    CACHE_TTL_RECOMMENDATIONS: int = 3600  # 1 hour
    CACHE_TTL_TRENDYOL_DATA: int = 1800  # 30 minutes
    CACHE_MAX_SIZE_MB: int = 500
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "logs/app.log"
    LOG_MAX_SIZE_MB: int = 100
    LOG_BACKUP_COUNT: int = 5
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    RATE_LIMIT_PER_MINUTE: int = 10
    SESSION_TIMEOUT_MINUTES: int = 30
    
    # Email Alert Settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    ALERT_EMAIL_TO: str = ""
    ALERT_EMAIL_FROM: str = ""
    ENABLE_EMAIL_ALERTS: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()
