"""
Email-specific Memory Management for MacBook Training

This module provides specialized memory management for email classification training,
including dynamic batch sizing, email-specific memory estimation, and real-time
monitoring optimized for EmailTRM models.
"""

import gc
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Any
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from .memory_management import MemoryManager, MemoryConfig, BatchSizeRecommendation, MemoryPressureInfo
from .resource_monitoring import ResourceMonitor, MemoryStats

logger = logging.getLogger(__name__)


@dataclass
class EmailMemoryConfig:
    """Email-specific memory management configuration."""
    # Base memory config
    base_config: MemoryConfig
    
    # Email-specific parameters
    email_cache_size_mb: float = 100.0  # Cache for processed emails
    tokenizer_memory_mb: float = 50.0   # Memory for tokenizer
    model_overhead_multiplier: float = 2.5  # Higher overhead for EmailTRM
    
    # Dynamic sizing parameters
    min_sequence_length: int = 128
    max_sequence_length: int = 512
    sequence_length_step: int = 64
    
    # Email structure memory
    structure_embedding_mb: float = 20.0
    hierarchical_attention_mb: float = 30.0
    
    # Monitoring parameters
    email_processing_threshold_ms: float = 100.0  # Alert if email processing is slow
    batch_processing_threshold_ms: float = 5000.0  # Alert if batch processing is slow


@dataclass
class EmailBatchRecommendation:
    """Email-specific batch size recommendation."""
    recommended_batch_size: int
    recommended_sequence_length: int
    estimated_memory_usage_mb: float
    estimated_processing_time_ms: float
    memory_breakdown: Dict[str, float]
    warnings: List[str]
    reasoning: str


@dataclass
class EmailMemoryMetrics:
    """Email training memory metrics."""
    # Current usage
    total_memory_mb: float
    model_memory_mb: float
    batch_memory_mb: float
    cache_memory_mb: float
    overhead_memory_mb: float
    
    # Email-specific metrics
    average_email_size_kb: float
    emails_per_second: float
    cache_hit_rate: float
    
    # Performance metrics
    batch_processing_time_ms: float
    memory_efficiency: float  # Useful memory / total memory
    
    # Pressure indicators
    memory_pressure_level: str
    recommended_action: str


class EmailMemoryManager:
    """
    Specialized memory manager for email classification training.
    
    Extends the base MemoryManager with email-specific optimizations
    including dynamic sequence length adjustment, email caching,
    and EmailTRM-specific memory estimation.
    """
    
    def __init__(self, config: Optional[EmailMemoryConfig] = None):
        """
        Initialize email memory manager.
        
        Args:
            config: Email memory configuration
        """
        self.config = config or EmailMemoryConfig(base_config=MemoryConfig())
        
        # Initialize base memory manager
        self.base_manager = MemoryManager(self.config.base_config)
        
        # Email-specific state
        self.email_cache_size_mb = 0.0
        self.current_sequence_length = 512
        self.email_processing_times: List[float] = []
        self.batch_processing_times: List[float] = []
        
        # Memory tracking
        self.model_memory_baseline_mb = 0.0
        self.email_memory_usage: Dict[str, float] = {}
        
        # Performance metrics
        self.emails_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("EmailMemoryManager initialized")
    
    def estimate_email_model_memory(self, 
                                  model_params: int,
                                  batch_size: int,
                                  sequence_length: int,
                                  num_categories: int = 10) -> Dict[str, float]:
        """
        Estimate memory usage for EmailTRM model.
        
        Args:
            model_params: Number of model parameters
            batch_size: Training batch size
            sequence_length: Input sequence length
            num_categories: Number of email categories
            
        Returns:
            Memory breakdown in MB
        """
        # Base model memory (parameters + gradients + optimizer states)
        param_memory_mb = (model_params * 4) / (1024**2)  # float32
        gradient_memory_mb = param_memory_mb  # Gradients
        optimizer_memory_mb = param_memory_mb * 2  # Adam optimizer states
        
        # Activation memory (batch-dependent)
        # EmailTRM has additional memory for recursive reasoning
        base_activation_mb = (batch_size * sequence_length * 512 * 4) / (1024**2)  # Hidden states
        recursive_reasoning_mb = base_activation_mb * 0.5  # Additional memory for TRM cycles
        attention_memory_mb = (batch_size * sequence_length * sequence_length * 4) / (1024**2)
        
        # Email-specific memory
        email_structure_mb = self.config.structure_embedding_mb
        hierarchical_attention_mb = self.config.hierarchical_attention_mb
        classification_head_mb = (num_categories * 512 * 4) / (1024**2)
        
        # Tokenizer and preprocessing
        tokenizer_memory_mb = self.config.tokenizer_memory_mb
        email_cache_mb = self.config.email_cache_size_mb
        
        # Apply overhead multiplier
        total_model_mb = (param_memory_mb + gradient_memory_mb + optimizer_memory_mb + 
                         base_activation_mb + recursive_reasoning_mb + attention_memory_mb +
                         email_structure_mb + hierarchical_attention_mb + classification_head_mb)
        
        total_with_overhead = total_model_mb * self.config.model_overhead_multiplier
        
        memory_breakdown = {
            "model_parameters": param_memory_mb,
            "gradients": gradient_memory_mb,
            "optimizer_states": optimizer_memory_mb,
            "activations": base_activation_mb,
            "recursive_reasoning": recursive_reasoning_mb,
            "attention": attention_memory_mb,
            "email_structure": email_structure_mb,
            "hierarchical_attention": hierarchical_attention_mb,
            "classification_head": classification_head_mb,
            "tokenizer": tokenizer_memory_mb,
            "email_cache": email_cache_mb,
            "overhead": total_with_overhead - total_model_mb,
            "total": total_with_overhead
        }
        
        return memory_breakdown
    
    def calculate_optimal_email_batch_config(self, 
                                           model_params: int,
                                           available_memory_mb: Optional[float] = None) -> EmailBatchRecommendation:
        """
        Calculate optimal batch size and sequence length for email training.
        
        Args:
            model_params: Number of model parameters
            available_memory_mb: Available memory (auto-detected if None)
            
        Returns:
            Email batch configuration recommendation
        """
        if available_memory_mb is None:
            memory_stats = self.base_manager.monitor_memory_usage()
            available_memory_mb = memory_stats.available_mb - self.config.base_config.safety_margin_mb
        
        best_config = None
        best_score = 0.0
        
        # Try different batch sizes and sequence lengths
        batch_sizes = [1, 2, 4, 8, 16, 32]
        sequence_lengths = list(range(self.config.min_sequence_length, 
                                    self.config.max_sequence_length + 1, 
                                    self.config.sequence_length_step))
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                memory_breakdown = self.estimate_email_model_memory(
                    model_params, batch_size, seq_len
                )
                
                total_memory = memory_breakdown["total"]
                
                if total_memory <= available_memory_mb:
                    # Calculate score based on throughput and memory efficiency
                    throughput_score = batch_size * seq_len  # Higher is better
                    memory_efficiency = total_memory / available_memory_mb  # Higher is better (up to a point)
                    
                    # Prefer configurations that use 60-80% of available memory
                    if memory_efficiency < 0.6:
                        efficiency_penalty = (0.6 - memory_efficiency) * 2
                    elif memory_efficiency > 0.8:
                        efficiency_penalty = (memory_efficiency - 0.8) * 3
                    else:
                        efficiency_penalty = 0
                    
                    score = throughput_score * (1 - efficiency_penalty)
                    
                    if score > best_score:
                        best_score = score
                        
                        # Estimate processing time (simplified)
                        estimated_time_ms = (batch_size * seq_len * 0.1) + 50  # Base processing time
                        
                        warnings = []
                        if memory_efficiency > 0.8:
                            warnings.append("High memory utilization - may cause instability")
                        if seq_len < 256:
                            warnings.append("Short sequence length may limit model performance")
                        if batch_size < 4:
                            warnings.append("Small batch size may slow training")
                        
                        reasoning = (f"Optimal balance of batch size ({batch_size}) and sequence length ({seq_len}) "
                                   f"for {memory_efficiency:.1%} memory utilization")
                        
                        best_config = EmailBatchRecommendation(
                            recommended_batch_size=batch_size,
                            recommended_sequence_length=seq_len,
                            estimated_memory_usage_mb=total_memory,
                            estimated_processing_time_ms=estimated_time_ms,
                            memory_breakdown=memory_breakdown,
                            warnings=warnings,
                            reasoning=reasoning
                        )
        
        if best_config is None:
            # Fallback to minimal configuration
            memory_breakdown = self.estimate_email_model_memory(model_params, 1, 128)
            best_config = EmailBatchRecommendation(
                recommended_batch_size=1,
                recommended_sequence_length=128,
                estimated_memory_usage_mb=memory_breakdown["total"],
                estimated_processing_time_ms=200.0,
                memory_breakdown=memory_breakdown,
                warnings=["Minimal configuration due to memory constraints"],
                reasoning="Fallback to minimal configuration"
            )
        
        return best_config
    
    def monitor_email_processing_performance(self, 
                                           processing_time_ms: float,
                                           batch_size: int,
                                           num_emails: int) -> None:
        """
        Monitor email processing performance.
        
        Args:
            processing_time_ms: Time taken to process batch
            batch_size: Size of processed batch
            num_emails: Number of emails processed
        """
        # Track processing times
        self.batch_processing_times.append(processing_time_ms)
        if len(self.batch_processing_times) > 100:
            self.batch_processing_times.pop(0)
        
        # Calculate per-email processing time
        per_email_time = processing_time_ms / max(1, num_emails)
        self.email_processing_times.append(per_email_time)
        if len(self.email_processing_times) > 100:
            self.email_processing_times.pop(0)
        
        # Update counters
        self.emails_processed += num_emails
        
        # Check for performance issues
        if processing_time_ms > self.config.batch_processing_threshold_ms:
            logger.warning(f"Slow batch processing: {processing_time_ms:.1f}ms for {num_emails} emails")
        
        if per_email_time > self.config.email_processing_threshold_ms:
            logger.warning(f"Slow email processing: {per_email_time:.1f}ms per email")
    
    def update_cache_metrics(self, cache_hits: int, cache_misses: int) -> None:
        """
        Update email cache performance metrics.
        
        Args:
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
        """
        self.cache_hits += cache_hits
        self.cache_misses += cache_misses
    
    def get_email_memory_metrics(self) -> EmailMemoryMetrics:
        """
        Get comprehensive email memory metrics.
        
        Returns:
            Email memory metrics
        """
        # Get current memory stats
        memory_stats = self.base_manager.monitor_memory_usage()
        
        # Calculate memory breakdown
        total_memory_mb = memory_stats.used_mb
        model_memory_mb = self.model_memory_baseline_mb
        cache_memory_mb = self.email_cache_size_mb
        
        # Estimate batch and overhead memory
        batch_memory_mb = max(0, total_memory_mb - model_memory_mb - cache_memory_mb)
        overhead_memory_mb = total_memory_mb * 0.1  # Estimate 10% overhead
        
        # Calculate performance metrics
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(1, total_cache_requests)
        
        # Calculate processing speed
        if self.batch_processing_times:
            avg_batch_time = sum(self.batch_processing_times) / len(self.batch_processing_times)
            emails_per_second = (self.emails_processed / (avg_batch_time / 1000)) if avg_batch_time > 0 else 0
        else:
            avg_batch_time = 0
            emails_per_second = 0
        
        # Calculate average email size (estimate)
        if self.emails_processed > 0:
            avg_email_size_kb = (batch_memory_mb * 1024) / self.emails_processed
        else:
            avg_email_size_kb = 0
        
        # Memory efficiency
        useful_memory = model_memory_mb + batch_memory_mb
        memory_efficiency = useful_memory / max(1, total_memory_mb)
        
        # Get pressure info
        pressure_info = self.base_manager._analyze_memory_pressure(memory_stats)
        
        return EmailMemoryMetrics(
            total_memory_mb=total_memory_mb,
            model_memory_mb=model_memory_mb,
            batch_memory_mb=batch_memory_mb,
            cache_memory_mb=cache_memory_mb,
            overhead_memory_mb=overhead_memory_mb,
            average_email_size_kb=avg_email_size_kb,
            emails_per_second=emails_per_second,
            cache_hit_rate=cache_hit_rate,
            batch_processing_time_ms=avg_batch_time,
            memory_efficiency=memory_efficiency,
            memory_pressure_level=pressure_info.pressure_level,
            recommended_action=pressure_info.recommended_action
        )
    
    def optimize_sequence_length_dynamically(self, 
                                           current_memory_usage: float,
                                           target_memory_usage: float = 0.75) -> int:
        """
        Dynamically optimize sequence length based on memory usage.
        
        Args:
            current_memory_usage: Current memory usage percentage (0-1)
            target_memory_usage: Target memory usage percentage (0-1)
            
        Returns:
            Optimized sequence length
        """
        if current_memory_usage > target_memory_usage + 0.1:
            # Reduce sequence length
            new_length = max(
                self.config.min_sequence_length,
                self.current_sequence_length - self.config.sequence_length_step
            )
            if new_length != self.current_sequence_length:
                logger.info(f"Reducing sequence length: {self.current_sequence_length} -> {new_length}")
                self.current_sequence_length = new_length
        
        elif current_memory_usage < target_memory_usage - 0.1:
            # Increase sequence length
            new_length = min(
                self.config.max_sequence_length,
                self.current_sequence_length + self.config.sequence_length_step
            )
            if new_length != self.current_sequence_length:
                logger.info(f"Increasing sequence length: {self.current_sequence_length} -> {new_length}")
                self.current_sequence_length = new_length
        
        return self.current_sequence_length
    
    def cleanup_email_cache(self, force: bool = False) -> float:
        """
        Clean up email cache to free memory.
        
        Args:
            force: Force aggressive cleanup
            
        Returns:
            Amount of memory freed in MB
        """
        initial_memory = psutil.virtual_memory().used / (1024**2)
        
        # Force garbage collection
        gc.collect()
        
        if TORCH_AVAILABLE:
            # Clear PyTorch cache
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Reset cache size tracking
        if force:
            self.email_cache_size_mb = 0.0
        else:
            self.email_cache_size_mb *= 0.5  # Assume 50% cache cleanup
        
        final_memory = psutil.virtual_memory().used / (1024**2)
        freed_memory = max(0, initial_memory - final_memory)
        
        if freed_memory > 0:
            logger.info(f"Email cache cleanup freed {freed_memory:.1f}MB")
        
        return freed_memory
    
    def get_memory_recommendations(self, model_params: int) -> Dict[str, Any]:
        """
        Get comprehensive memory recommendations for email training.
        
        Args:
            model_params: Number of model parameters
            
        Returns:
            Memory recommendations
        """
        # Get base recommendations
        base_recommendations = self.base_manager.get_memory_recommendations(model_params)
        
        # Get email-specific recommendations
        email_batch_config = self.calculate_optimal_email_batch_config(model_params)
        email_metrics = self.get_email_memory_metrics()
        
        # Combine recommendations
        recommendations = {
            "base_memory": base_recommendations,
            "email_specific": {
                "optimal_batch_size": email_batch_config.recommended_batch_size,
                "optimal_sequence_length": email_batch_config.recommended_sequence_length,
                "estimated_memory_usage_mb": email_batch_config.estimated_memory_usage_mb,
                "memory_breakdown": email_batch_config.memory_breakdown,
                "warnings": email_batch_config.warnings,
                "reasoning": email_batch_config.reasoning
            },
            "current_performance": {
                "emails_per_second": email_metrics.emails_per_second,
                "cache_hit_rate": email_metrics.cache_hit_rate,
                "memory_efficiency": email_metrics.memory_efficiency,
                "pressure_level": email_metrics.memory_pressure_level
            },
            "optimization_suggestions": []
        }
        
        # Add optimization suggestions
        if email_metrics.cache_hit_rate < 0.5:
            recommendations["optimization_suggestions"].append(
                "Consider increasing email cache size for better performance"
            )
        
        if email_metrics.memory_efficiency < 0.6:
            recommendations["optimization_suggestions"].append(
                "Memory usage is low - consider increasing batch size or sequence length"
            )
        
        if email_metrics.emails_per_second < 10:
            recommendations["optimization_suggestions"].append(
                "Email processing speed is low - consider optimizing preprocessing"
            )
        
        if email_metrics.memory_pressure_level in ["high", "critical"]:
            recommendations["optimization_suggestions"].append(
                "High memory pressure - consider reducing batch size or sequence length"
            )
        
        return recommendations
    
    def create_memory_callback(self) -> Callable:
        """
        Create callback for memory monitoring during training.
        
        Returns:
            Memory monitoring callback function
        """
        def memory_callback(step: int, metrics: Dict[str, Any]):
            """Callback to monitor memory during training."""
            # Update memory tracking
            current_memory = psutil.virtual_memory()
            memory_usage_percent = current_memory.percent
            
            # Log memory status periodically
            if step % 100 == 0:
                email_metrics = self.get_email_memory_metrics()
                logger.info(f"Step {step}: Memory {memory_usage_percent:.1f}%, "
                           f"Processing {email_metrics.emails_per_second:.1f} emails/sec, "
                           f"Cache hit rate {email_metrics.cache_hit_rate:.2f}")
            
            # Check for memory pressure
            if memory_usage_percent > 85:
                logger.warning(f"High memory usage at step {step}: {memory_usage_percent:.1f}%")
                self.cleanup_email_cache(force=False)
            
            # Dynamic sequence length optimization
            if step % 500 == 0:  # Check every 500 steps
                self.optimize_sequence_length_dynamically(memory_usage_percent / 100.0)
        
        return memory_callback
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory summary for email training.
        
        Returns:
            Memory summary
        """
        base_summary = self.base_manager.get_memory_summary()
        email_metrics = self.get_email_memory_metrics()
        
        return {
            "base_memory": base_summary,
            "email_specific": {
                "current_sequence_length": self.current_sequence_length,
                "email_cache_size_mb": self.email_cache_size_mb,
                "emails_processed": self.emails_processed,
                "cache_performance": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": email_metrics.cache_hit_rate
                },
                "processing_performance": {
                    "emails_per_second": email_metrics.emails_per_second,
                    "avg_batch_time_ms": email_metrics.batch_processing_time_ms,
                    "memory_efficiency": email_metrics.memory_efficiency
                }
            },
            "recommendations": {
                "memory_pressure_level": email_metrics.memory_pressure_level,
                "recommended_action": email_metrics.recommended_action,
                "optimization_opportunities": self._get_optimization_opportunities(email_metrics)
            }
        }
    
    def _get_optimization_opportunities(self, metrics: EmailMemoryMetrics) -> List[str]:
        """Get optimization opportunities based on current metrics."""
        opportunities = []
        
        if metrics.cache_hit_rate < 0.3:
            opportunities.append("Increase email cache size")
        
        if metrics.memory_efficiency < 0.5:
            opportunities.append("Increase batch size or sequence length")
        
        if metrics.emails_per_second < 5:
            opportunities.append("Optimize email preprocessing pipeline")
        
        if metrics.memory_pressure_level == "low" and metrics.memory_efficiency < 0.7:
            opportunities.append("Consider more aggressive memory utilization")
        
        return opportunities