"""
FastAPI-based Inference Service for Email Classification

This module provides a production-ready REST API for email classification
with batch processing, confidence scoring, and prediction explanations.
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = None
    BackgroundTasks = None
    Depends = None
    CORSMiddleware = None
    JSONResponse = None
    BaseModel = None
    Field = None
    validator = None
    uvicorn = None
    FASTAPI_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from .model_export import ModelExporter, ModelMetadata
from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from models.email_tokenizer import EmailTokenizer


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class EmailInput(BaseModel):
        """Single email input for classification."""
        id: Optional[str] = Field(None, description="Optional email ID")
        subject: str = Field(..., description="Email subject line")
        body: str = Field(..., description="Email body content")
        sender: Optional[str] = Field(None, description="Sender email address")
        recipient: Optional[str] = Field(None, description="Recipient email address")
        
        @validator('subject', 'body')
        def validate_content(cls, v):
            if not v or not v.strip():
                raise ValueError("Subject and body cannot be empty")
            return v.strip()
    
    class BatchEmailInput(BaseModel):
        """Batch email input for classification."""
        emails: List[EmailInput] = Field(..., description="List of emails to classify")
        
        @validator('emails')
        def validate_batch_size(cls, v):
            if len(v) == 0:
                raise ValueError("Batch cannot be empty")
            if len(v) > 100:  # Reasonable batch size limit
                raise ValueError("Batch size cannot exceed 100 emails")
            return v
    
    class ClassificationResult(BaseModel):
        """Single email classification result."""
        email_id: Optional[str]
        predicted_category: str
        confidence: float
        category_probabilities: Dict[str, float]
        processing_time_ms: float
        reasoning_cycles: Optional[int] = None
        
    class BatchClassificationResult(BaseModel):
        """Batch classification result."""
        results: List[ClassificationResult]
        total_processing_time_ms: float
        average_confidence: float
        batch_size: int
        
    class ModelInfo(BaseModel):
        """Model information."""
        model_id: str
        version: str
        categories: List[str]
        accuracy_metrics: Dict[str, float]
        parameter_count: int
        model_size_mb: float
        
    class HealthStatus(BaseModel):
        """API health status."""
        status: str
        model_loaded: bool
        uptime_seconds: float
        total_predictions: int
        average_response_time_ms: float


@dataclass
class InferenceConfig:
    """Configuration for inference API."""
    # Model settings
    model_path: str
    tokenizer_path: str
    device: str = "cpu"
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Performance settings
    batch_size_limit: int = 100
    max_sequence_length: int = 512
    enable_batch_processing: bool = True
    
    # Caching settings
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Monitoring settings
    enable_metrics: bool = True
    log_predictions: bool = True
    
    # Security settings
    enable_cors: bool = True
    allowed_origins: List[str] = None
    api_key_required: bool = False
    api_key: Optional[str] = None


class ModelLoader:
    """Handles model loading and management."""
    
    def __init__(self, config: InferenceConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.model: Optional[EmailTRM] = None
        self.tokenizer: Optional[EmailTokenizer] = None
        self.metadata: Optional[ModelMetadata] = None
        self.categories: List[str] = []
        
        self.model_loaded = False
        self.load_time: Optional[datetime] = None
    
    def load_model(self) -> bool:
        """Load model and tokenizer from files."""
        try:
            self.logger.info(f"Loading model from {self.config.model_path}")
            
            # Load model
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            if TORCH_AVAILABLE:
                model_data = torch.load(self.config.model_path, map_location=self.config.device)
            else:
                import pickle
                with open(self.config.model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Create model from config
            model_config = EmailTRMConfig(**model_data['model_config'])
            self.model = EmailTRM(model_config)
            
            # Load state dict
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()
            
            if TORCH_AVAILABLE and self.config.device != "cpu":
                self.model = self.model.to(self.config.device)
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer from {self.config.tokenizer_path}")
            
            if not os.path.exists(self.config.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found: {self.config.tokenizer_path}")
            
            self.tokenizer = EmailTokenizer.load(self.config.tokenizer_path)
            
            # Load metadata if available
            metadata_path = Path(self.config.model_path).parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    # Convert timestamp string back to datetime
                    if 'export_timestamp' in metadata_dict:
                        metadata_dict['export_timestamp'] = datetime.fromisoformat(metadata_dict['export_timestamp'])
                    self.metadata = ModelMetadata(**metadata_dict)
            
            # Set up categories
            self.categories = [f"category_{i}" for i in range(model_config.num_email_categories)]
            
            self.model_loaded = True
            self.load_time = datetime.now()
            
            self.logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_loaded": True,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "categories": self.categories,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "max_sequence_length": self.tokenizer.max_seq_len if self.tokenizer else 0,
            "device": self.config.device
        }
        
        if self.metadata:
            info.update({
                "model_id": self.metadata.model_id,
                "version": self.metadata.version,
                "parameter_count": self.metadata.parameter_count,
                "model_size_mb": self.metadata.model_size_mb,
                "accuracy_metrics": self.metadata.accuracy_metrics
            })
        
        return info


class PredictionCache:
    """Simple in-memory cache for predictions."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
    
    def _generate_key(self, email_data: Dict[str, str]) -> str:
        """Generate cache key from email data."""
        import hashlib
        content = f"{email_data.get('subject', '')}{email_data.get('body', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, email_data: Dict[str, str]) -> Optional[Any]:
        """Get cached prediction."""
        key = self._generate_key(email_data)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if expired
            if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                return result
            else:
                del self.cache[key]
        
        return None
    
    def set(self, email_data: Dict[str, str], result: Any):
        """Cache prediction result."""
        key = self._generate_key(email_data)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (result, datetime.now())
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class InferenceEngine:
    """Core inference engine for email classification."""
    
    def __init__(self, model_loader: ModelLoader, config: InferenceConfig, logger: logging.Logger):
        self.model_loader = model_loader
        self.config = config
        self.logger = logger
        
        # Caching
        self.cache = PredictionCache(
            max_size=config.cache_size,
            ttl_seconds=config.cache_ttl_seconds
        ) if config.enable_caching else None
        
        # Statistics
        self.total_predictions = 0
        self.total_processing_time = 0.0
        self.start_time = datetime.now()
    
    def predict_single(self, email_data: Dict[str, str]) -> Dict[str, Any]:
        """Predict category for single email."""
        start_time = time.time()
        
        try:
            # Check cache
            if self.cache:
                cached_result = self.cache.get(email_data)
                if cached_result:
                    return cached_result
            
            # Tokenize email
            token_ids, metadata = self.model_loader.tokenizer.encode_email(email_data)
            
            # Convert to tensor
            if TORCH_AVAILABLE:
                input_tensor = torch.tensor([token_ids], dtype=torch.long)
                if self.config.device != "cpu":
                    input_tensor = input_tensor.to(self.config.device)
            else:
                input_tensor = [token_ids]  # Fallback for non-PyTorch
            
            # Run inference
            with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
                outputs = self.model_loader.model(input_tensor)
                
                # Get predictions and confidence
                logits = outputs['logits'][0]  # Remove batch dimension
                probabilities = F.softmax(logits, dim=-1) if TORCH_AVAILABLE else logits
                
                predicted_category_idx = torch.argmax(probabilities).item() if TORCH_AVAILABLE else 0
                confidence = probabilities[predicted_category_idx].item() if TORCH_AVAILABLE else 0.5
                
                # Get all category probabilities
                category_probs = {}
                for i, category in enumerate(self.model_loader.categories):
                    prob = probabilities[i].item() if TORCH_AVAILABLE else (1.0 / len(self.model_loader.categories))
                    category_probs[category] = prob
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = {
                "predicted_category": self.model_loader.categories[predicted_category_idx],
                "confidence": confidence,
                "category_probabilities": category_probs,
                "processing_time_ms": processing_time,
                "reasoning_cycles": outputs.get('num_cycles', 1)
            }
            
            # Cache result
            if self.cache:
                self.cache.set(email_data, result)
            
            # Update statistics
            self.total_predictions += 1
            self.total_processing_time += processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, emails: List[Dict[str, str]]) -> Dict[str, Any]:
        """Predict categories for batch of emails."""
        start_time = time.time()
        
        try:
            results = []
            
            # Process each email
            for i, email_data in enumerate(emails):
                try:
                    result = self.predict_single(email_data)
                    result["email_id"] = email_data.get("id", f"email_{i}")
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process email {i}: {e}")
                    # Add error result
                    results.append({
                        "email_id": email_data.get("id", f"email_{i}"),
                        "predicted_category": "error",
                        "confidence": 0.0,
                        "category_probabilities": {},
                        "processing_time_ms": 0.0,
                        "error": str(e)
                    })
            
            total_processing_time = (time.time() - start_time) * 1000
            
            # Calculate average confidence (excluding errors)
            valid_results = [r for r in results if "error" not in r]
            avg_confidence = sum(r["confidence"] for r in valid_results) / len(valid_results) if valid_results else 0.0
            
            return {
                "results": results,
                "total_processing_time_ms": total_processing_time,
                "average_confidence": avg_confidence,
                "batch_size": len(emails)
            }
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = self.total_processing_time / self.total_predictions if self.total_predictions > 0 else 0.0
        
        stats = {
            "total_predictions": self.total_predictions,
            "uptime_seconds": uptime,
            "average_response_time_ms": avg_response_time,
            "predictions_per_second": self.total_predictions / uptime if uptime > 0 else 0.0
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats


class EmailClassificationAPI:
    """FastAPI application for email classification."""
    
    def __init__(self, config: InferenceConfig):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_loader = ModelLoader(config, self.logger)
        self.inference_engine = None
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Email Classification API",
            description="Production API for email classification using EmailTRM",
            version="1.0.0"
        )
        
        # Add CORS middleware
        if config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=config.allowed_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Setup routes
        self._setup_routes()
        
        # Load model on startup
        self._load_model_on_startup()
    
    def _load_model_on_startup(self):
        """Load model during startup."""
        @self.app.on_event("startup")
        async def startup_event():
            self.logger.info("Loading model on startup...")
            success = self.model_loader.load_model()
            
            if success:
                self.inference_engine = InferenceEngine(self.model_loader, self.config, self.logger)
                self.logger.info("API ready for inference")
            else:
                self.logger.error("Failed to load model - API will not be functional")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthStatus)
        async def health_check():
            """Health check endpoint."""
            stats = self.inference_engine.get_statistics() if self.inference_engine else {}
            
            return HealthStatus(
                status="healthy" if self.model_loader.model_loaded else "unhealthy",
                model_loaded=self.model_loader.model_loaded,
                uptime_seconds=stats.get("uptime_seconds", 0.0),
                total_predictions=stats.get("total_predictions", 0),
                average_response_time_ms=stats.get("average_response_time_ms", 0.0)
            )
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def get_model_info():
            """Get model information."""
            if not self.model_loader.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            info = self.model_loader.get_model_info()
            
            return ModelInfo(
                model_id=info.get("model_id", "unknown"),
                version=info.get("version", "unknown"),
                categories=info.get("categories", []),
                accuracy_metrics=info.get("accuracy_metrics", {}),
                parameter_count=info.get("parameter_count", 0),
                model_size_mb=info.get("model_size_mb", 0.0)
            )
        
        @self.app.post("/predict", response_model=ClassificationResult)
        async def predict_email(email: EmailInput):
            """Classify single email."""
            if not self.inference_engine:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            email_data = {
                "id": email.id,
                "subject": email.subject,
                "body": email.body,
                "sender": email.sender,
                "recipient": email.recipient
            }
            
            result = self.inference_engine.predict_single(email_data)
            
            return ClassificationResult(
                email_id=email.id,
                predicted_category=result["predicted_category"],
                confidence=result["confidence"],
                category_probabilities=result["category_probabilities"],
                processing_time_ms=result["processing_time_ms"],
                reasoning_cycles=result.get("reasoning_cycles")
            )
        
        @self.app.post("/predict/batch", response_model=BatchClassificationResult)
        async def predict_batch(batch: BatchEmailInput):
            """Classify batch of emails."""
            if not self.inference_engine:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            emails_data = []
            for email in batch.emails:
                email_data = {
                    "id": email.id,
                    "subject": email.subject,
                    "body": email.body,
                    "sender": email.sender,
                    "recipient": email.recipient
                }
                emails_data.append(email_data)
            
            result = self.inference_engine.predict_batch(emails_data)
            
            # Convert results to ClassificationResult objects
            classification_results = []
            for r in result["results"]:
                classification_results.append(ClassificationResult(
                    email_id=r.get("email_id"),
                    predicted_category=r["predicted_category"],
                    confidence=r["confidence"],
                    category_probabilities=r["category_probabilities"],
                    processing_time_ms=r["processing_time_ms"],
                    reasoning_cycles=r.get("reasoning_cycles")
                ))
            
            return BatchClassificationResult(
                results=classification_results,
                total_processing_time_ms=result["total_processing_time_ms"],
                average_confidence=result["average_confidence"],
                batch_size=result["batch_size"]
            )
        
        @self.app.get("/stats")
        async def get_statistics():
            """Get API statistics."""
            if not self.inference_engine:
                return {"error": "Model not loaded"}
            
            return self.inference_engine.get_statistics()
        
        @self.app.post("/cache/clear")
        async def clear_cache():
            """Clear prediction cache."""
            if self.inference_engine and self.inference_engine.cache:
                self.inference_engine.cache.clear()
                return {"message": "Cache cleared"}
            else:
                return {"message": "No cache to clear"}
    
    def run(self):
        """Run the API server."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")
        
        self.logger.info(f"Starting API server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level="info"
        )


def create_inference_api(model_path: str, 
                        tokenizer_path: str,
                        host: str = "0.0.0.0",
                        port: int = 8000,
                        **kwargs) -> EmailClassificationAPI:
    """
    Create and configure inference API.
    
    Args:
        model_path: Path to exported model file
        tokenizer_path: Path to tokenizer file
        host: API host
        port: API port
        **kwargs: Additional configuration options
        
    Returns:
        Configured EmailClassificationAPI instance
    """
    config = InferenceConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        host=host,
        port=port,
        **kwargs
    )
    
    return EmailClassificationAPI(config)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Email Classification Inference API")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer file")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Create and run API
    api = create_inference_api(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        host=args.host,
        port=args.port,
        workers=args.workers
    )
    
    api.run()