"""
Recursive Reasoning Parameter Optimizer for EmailTRM

This module provides optimization utilities for configuring recursive reasoning parameters
in EmailTRM models based on hardware capabilities and email classification requirements.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from pathlib import Path

from macbook_optimization.hardware_detection import HardwareDetector
from macbook_optimization.memory_management import MemoryManager


logger = logging.getLogger(__name__)


@dataclass
class RecursiveReasoningProfile:
    """Profile for recursive reasoning parameters"""
    
    # Core reasoning parameters
    H_cycles: int  # Number of high-level reasoning cycles
    L_cycles: int  # Number of low-level cycles per H cycle
    halt_max_steps: int  # Maximum steps before forced halting
    halt_exploration_prob: float  # Probability of exploration in halting
    
    # Adaptive parameters
    adaptive_halting: bool = True  # Enable adaptive halting
    complexity_threshold: float = 0.5  # Threshold for complexity-based halting
    confidence_threshold: float = 0.8  # Threshold for confidence-based halting
    
    # Performance characteristics
    avg_inference_time_ms: float = 0.0  # Average inference time
    memory_usage_mb: float = 0.0  # Memory usage estimate
    accuracy_estimate: float = 0.0  # Estimated accuracy
    
    # Hardware compatibility
    compatible_memory_gb: float = 4.0  # Minimum memory requirement
    optimal_batch_size: int = 8  # Optimal batch size for this profile


class RecursiveReasoningOptimizer:
    """Optimizer for recursive reasoning parameters in EmailTRM"""
    
    def __init__(self, hardware_detector: Optional[HardwareDetector] = None):
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.memory_manager = MemoryManager()
        
        # Predefined profiles for different hardware configurations
        self.profiles = self._create_reasoning_profiles()
        
    def _create_reasoning_profiles(self) -> Dict[str, RecursiveReasoningProfile]:
        """Create predefined reasoning profiles for different scenarios"""
        
        profiles = {
            # High-performance profile for powerful MacBooks
            "high_performance": RecursiveReasoningProfile(
                H_cycles=4,
                L_cycles=6,
                halt_max_steps=12,
                halt_exploration_prob=0.15,
                adaptive_halting=True,
                complexity_threshold=0.4,
                confidence_threshold=0.85,
                compatible_memory_gb=12.0,
                optimal_batch_size=16
            ),
            
            # Balanced profile for mid-range MacBooks
            "balanced": RecursiveReasoningProfile(
                H_cycles=3,
                L_cycles=4,
                halt_max_steps=8,
                halt_exploration_prob=0.1,
                adaptive_halting=True,
                complexity_threshold=0.5,
                confidence_threshold=0.8,
                compatible_memory_gb=8.0,
                optimal_batch_size=8
            ),
            
            # Efficient profile for memory-constrained MacBooks
            "efficient": RecursiveReasoningProfile(
                H_cycles=2,
                L_cycles=3,
                halt_max_steps=6,
                halt_exploration_prob=0.05,
                adaptive_halting=True,
                complexity_threshold=0.6,
                confidence_threshold=0.75,
                compatible_memory_gb=4.0,
                optimal_batch_size=4
            ),
            
            # Fast profile for quick inference
            "fast": RecursiveReasoningProfile(
                H_cycles=1,
                L_cycles=2,
                halt_max_steps=4,
                halt_exploration_prob=0.02,
                adaptive_halting=False,
                complexity_threshold=0.7,
                confidence_threshold=0.7,
                compatible_memory_gb=2.0,
                optimal_batch_size=2
            ),
            
            # Email-optimized profile specifically tuned for email classification
            "email_optimized": RecursiveReasoningProfile(
                H_cycles=3,
                L_cycles=4,
                halt_max_steps=8,
                halt_exploration_prob=0.08,  # Lower exploration for email classification
                adaptive_halting=True,
                complexity_threshold=0.45,  # Emails are generally less complex than puzzles
                confidence_threshold=0.82,  # Higher confidence threshold for production
                compatible_memory_gb=6.0,
                optimal_batch_size=8
            )
        }
        
        return profiles
    
    def select_optimal_profile(self, 
                             target_accuracy: float = 0.95,
                             max_inference_time_ms: Optional[float] = None,
                             memory_constraint_gb: Optional[float] = None) -> RecursiveReasoningProfile:
        """
        Select optimal reasoning profile based on hardware and requirements
        
        Args:
            target_accuracy: Target accuracy requirement
            max_inference_time_ms: Maximum acceptable inference time
            memory_constraint_gb: Memory constraint in GB
            
        Returns:
            Optimal RecursiveReasoningProfile
        """
        
        # Get hardware specifications
        hw_summary = self.hardware_detector.get_hardware_summary()
        available_memory_gb = hw_summary['memory']['available_gb']
        cpu_cores = hw_summary['cpu']['cores']
        
        logger.info(f"Selecting reasoning profile for: {available_memory_gb:.1f}GB RAM, {cpu_cores} cores")
        
        # Apply memory constraint
        if memory_constraint_gb is None:
            memory_constraint_gb = available_memory_gb * 0.7  # Use 70% of available memory
        
        # Filter profiles by memory compatibility
        compatible_profiles = {}
        for name, profile in self.profiles.items():
            if profile.compatible_memory_gb <= memory_constraint_gb:
                compatible_profiles[name] = profile
        
        if not compatible_profiles:
            logger.warning("No compatible profiles found, using 'fast' profile")
            return self.profiles["fast"]
        
        # For email classification, prefer the email-optimized profile if compatible
        if "email_optimized" in compatible_profiles:
            if available_memory_gb >= 6.0:
                logger.info("Selected email-optimized profile")
                return compatible_profiles["email_optimized"]
        
        # Select based on available memory
        if available_memory_gb >= 12.0 and "high_performance" in compatible_profiles:
            selected_profile = compatible_profiles["high_performance"]
            profile_name = "high_performance"
        elif available_memory_gb >= 8.0 and "balanced" in compatible_profiles:
            selected_profile = compatible_profiles["balanced"]
            profile_name = "balanced"
        elif available_memory_gb >= 4.0 and "efficient" in compatible_profiles:
            selected_profile = compatible_profiles["efficient"]
            profile_name = "efficient"
        else:
            selected_profile = compatible_profiles["fast"]
            profile_name = "fast"
        
        logger.info(f"Selected {profile_name} reasoning profile")
        return selected_profile
    
    def optimize_halting_parameters(self, 
                                  base_profile: RecursiveReasoningProfile,
                                  email_complexity_distribution: Optional[List[float]] = None) -> RecursiveReasoningProfile:
        """
        Optimize halting parameters based on email complexity characteristics
        
        Args:
            base_profile: Base reasoning profile to optimize
            email_complexity_distribution: Distribution of email complexity scores
            
        Returns:
            Optimized RecursiveReasoningProfile
        """
        
        optimized_profile = RecursiveReasoningProfile(**base_profile.__dict__)
        
        # If we have email complexity data, optimize based on it
        if email_complexity_distribution:
            complexity_stats = {
                'mean': np.mean(email_complexity_distribution),
                'std': np.std(email_complexity_distribution),
                'p75': np.percentile(email_complexity_distribution, 75),
                'p90': np.percentile(email_complexity_distribution, 90)
            }
            
            logger.info(f"Email complexity stats: mean={complexity_stats['mean']:.3f}, "
                       f"std={complexity_stats['std']:.3f}, p90={complexity_stats['p90']:.3f}")
            
            # Adjust complexity threshold based on email characteristics
            # If emails are generally simple, use higher threshold (halt earlier)
            if complexity_stats['mean'] < 0.3:
                optimized_profile.complexity_threshold = min(0.7, base_profile.complexity_threshold + 0.1)
                optimized_profile.halt_exploration_prob = max(0.02, base_profile.halt_exploration_prob - 0.02)
                logger.info("Adjusted for simple emails: higher complexity threshold, lower exploration")
            
            # If emails are complex, use lower threshold (more reasoning)
            elif complexity_stats['mean'] > 0.7:
                optimized_profile.complexity_threshold = max(0.3, base_profile.complexity_threshold - 0.1)
                optimized_profile.halt_exploration_prob = min(0.2, base_profile.halt_exploration_prob + 0.03)
                logger.info("Adjusted for complex emails: lower complexity threshold, higher exploration")
        
        # Email-specific optimizations
        # Emails typically need fewer reasoning cycles than complex puzzles
        if optimized_profile.H_cycles > 3:
            optimized_profile.H_cycles = 3
            logger.info("Reduced H_cycles to 3 for email classification")
        
        # Adjust halt_max_steps to be proportional to total cycles
        total_cycles = optimized_profile.H_cycles * optimized_profile.L_cycles
        optimized_profile.halt_max_steps = min(optimized_profile.halt_max_steps, total_cycles + 2)
        
        # For production email classification, use higher confidence threshold
        optimized_profile.confidence_threshold = max(0.8, optimized_profile.confidence_threshold)
        
        return optimized_profile
    
    def benchmark_reasoning_profile(self, 
                                  profile: RecursiveReasoningProfile,
                                  model: torch.nn.Module,
                                  sample_inputs: torch.Tensor,
                                  num_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark a reasoning profile to measure performance characteristics
        
        Args:
            profile: Reasoning profile to benchmark
            model: EmailTRM model to test
            sample_inputs: Sample input tensors for testing
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics dictionary
        """
        
        model.eval()
        device = next(model.parameters()).device
        sample_inputs = sample_inputs.to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_inputs)
        
        # Benchmark inference time
        inference_times = []
        memory_usage = []
        
        for _ in range(num_iterations):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(sample_inputs, return_all_cycles=True)
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Estimate memory usage (rough approximation)
            if hasattr(torch.cuda, 'memory_allocated'):
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            else:
                # CPU memory estimation (very rough)
                memory_usage.append(sum(p.numel() * 4 for p in model.parameters()) / 1024**2)
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0.0
        
        # Analyze reasoning cycles used
        num_cycles_used = outputs.get('num_cycles', profile.H_cycles)
        reasoning_efficiency = num_cycles_used / profile.H_cycles
        
        metrics = {
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'avg_memory_usage_mb': avg_memory_usage,
            'reasoning_efficiency': reasoning_efficiency,
            'cycles_used': num_cycles_used,
            'max_cycles': profile.H_cycles
        }
        
        logger.info(f"Benchmark results: {avg_inference_time:.1f}±{std_inference_time:.1f}ms, "
                   f"{avg_memory_usage:.1f}MB, {reasoning_efficiency:.2f} efficiency")
        
        return metrics
    
    def create_adaptive_halting_config(self, 
                                     profile: RecursiveReasoningProfile,
                                     email_categories: List[str]) -> Dict[str, Any]:
        """
        Create adaptive halting configuration for different email categories
        
        Args:
            profile: Base reasoning profile
            email_categories: List of email category names
            
        Returns:
            Adaptive halting configuration
        """
        
        # Different email categories may need different reasoning depths
        category_complexity = {
            'spam': 0.3,      # Usually easy to identify
            'promotional': 0.4,  # Moderate complexity
            'newsletter': 0.4,   # Moderate complexity
            'social': 0.5,       # Medium complexity
            'shopping': 0.5,     # Medium complexity
            'travel': 0.6,       # Higher complexity
            'work': 0.7,         # High complexity (varied content)
            'personal': 0.7,     # High complexity (varied content)
            'finance': 0.8,      # Very high complexity (important to get right)
            'other': 0.6         # Medium complexity (catch-all)
        }
        
        adaptive_config = {
            'base_profile': profile,
            'category_thresholds': {},
            'category_exploration_probs': {},
            'category_max_steps': {}
        }
        
        for category in email_categories:
            category_lower = category.lower()
            complexity = category_complexity.get(category_lower, 0.5)
            
            # Adjust thresholds based on category complexity
            # More complex categories get lower thresholds (more reasoning)
            threshold_adjustment = (0.5 - complexity) * 0.2
            adaptive_config['category_thresholds'][category] = max(
                0.2, profile.complexity_threshold + threshold_adjustment
            )
            
            # Adjust exploration probability
            exploration_adjustment = complexity * 0.05
            adaptive_config['category_exploration_probs'][category] = min(
                0.3, profile.halt_exploration_prob + exploration_adjustment
            )
            
            # Adjust max steps
            steps_adjustment = int(complexity * 4)
            adaptive_config['category_max_steps'][category] = min(
                profile.halt_max_steps * 2, profile.halt_max_steps + steps_adjustment
            )
        
        logger.info(f"Created adaptive halting config for {len(email_categories)} categories")
        
        return adaptive_config
    
    def get_reasoning_recommendations(self, 
                                    hardware_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get reasoning parameter recommendations based on hardware
        
        Args:
            hardware_summary: Hardware summary (auto-detected if None)
            
        Returns:
            Reasoning recommendations
        """
        
        if hardware_summary is None:
            hardware_summary = self.hardware_detector.get_hardware_summary()
        
        recommendations = {
            'optimal_profile': None,
            'alternative_profiles': [],
            'hardware_analysis': {},
            'performance_predictions': {},
            'warnings': []
        }
        
        # Analyze hardware capabilities
        memory_gb = hardware_summary['memory']['available_gb']
        cpu_cores = hardware_summary['cpu']['cores']
        has_avx2 = hardware_summary['platform']['supports_avx2']
        
        recommendations['hardware_analysis'] = {
            'memory_tier': 'high' if memory_gb >= 12 else 'medium' if memory_gb >= 8 else 'low',
            'cpu_tier': 'high' if cpu_cores >= 8 else 'medium' if cpu_cores >= 4 else 'low',
            'optimization_support': 'good' if has_avx2 else 'basic'
        }
        
        # Select optimal profile
        optimal_profile = self.select_optimal_profile()
        recommendations['optimal_profile'] = optimal_profile
        
        # Suggest alternative profiles
        if memory_gb >= 8:
            recommendations['alternative_profiles'].extend(['balanced', 'email_optimized'])
        if memory_gb >= 4:
            recommendations['alternative_profiles'].append('efficient')
        recommendations['alternative_profiles'].append('fast')
        
        # Performance predictions
        recommendations['performance_predictions'] = {
            'expected_inference_time_ms': self._estimate_inference_time(optimal_profile, hardware_summary),
            'expected_memory_usage_mb': self._estimate_memory_usage(optimal_profile, hardware_summary),
            'expected_accuracy_range': (0.93, 0.97)  # Based on EmailTRM capabilities
        }
        
        # Generate warnings
        if memory_gb < 6:
            recommendations['warnings'].append(
                "Limited memory may affect reasoning performance. Consider using 'efficient' or 'fast' profile."
            )
        
        if cpu_cores < 4:
            recommendations['warnings'].append(
                "Limited CPU cores may slow down training. Consider reducing batch size."
            )
        
        return recommendations
    
    def _estimate_inference_time(self, profile: RecursiveReasoningProfile, hardware_summary: Dict[str, Any]) -> float:
        """Estimate inference time based on profile and hardware"""
        
        base_time = 50.0  # Base inference time in ms
        
        # Adjust for reasoning complexity
        complexity_factor = profile.H_cycles * profile.L_cycles / 12.0  # Normalized to balanced profile
        
        # Adjust for hardware
        cpu_factor = 4.0 / hardware_summary['cpu']['cores']  # Normalized to 4 cores
        memory_factor = 8.0 / hardware_summary['memory']['available_gb']  # Normalized to 8GB
        
        estimated_time = base_time * complexity_factor * cpu_factor * memory_factor
        
        return max(10.0, estimated_time)  # Minimum 10ms
    
    def _estimate_memory_usage(self, profile: RecursiveReasoningProfile, hardware_summary: Dict[str, Any]) -> float:
        """Estimate memory usage based on profile and hardware"""
        
        base_memory = 200.0  # Base memory usage in MB
        
        # Adjust for model complexity
        complexity_factor = (profile.H_cycles * profile.L_cycles) / 12.0
        
        # Adjust for batch size
        batch_factor = profile.optimal_batch_size / 8.0
        
        estimated_memory = base_memory * complexity_factor * batch_factor
        
        return estimated_memory


# Convenience functions
def get_optimal_reasoning_config(vocab_size: int, 
                               target_accuracy: float = 0.95,
                               hardware_detector: Optional[HardwareDetector] = None) -> Dict[str, Any]:
    """
    Get optimal reasoning configuration for EmailTRM
    
    Args:
        vocab_size: Vocabulary size
        target_accuracy: Target accuracy requirement
        hardware_detector: Hardware detector instance
        
    Returns:
        Optimal reasoning configuration
    """
    
    optimizer = RecursiveReasoningOptimizer(hardware_detector)
    
    # Get optimal profile
    profile = optimizer.select_optimal_profile(target_accuracy=target_accuracy)
    
    # Get recommendations
    recommendations = optimizer.get_reasoning_recommendations()
    
    return {
        'profile': profile,
        'recommendations': recommendations,
        'config_dict': {
            'H_cycles': profile.H_cycles,
            'L_cycles': profile.L_cycles,
            'halt_max_steps': profile.halt_max_steps,
            'halt_exploration_prob': profile.halt_exploration_prob,
            'adaptive_halting': profile.adaptive_halting,
            'complexity_threshold': profile.complexity_threshold,
            'confidence_threshold': profile.confidence_threshold
        }
    }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test reasoning optimization
    optimizer = RecursiveReasoningOptimizer()
    
    # Get optimal profile
    profile = optimizer.select_optimal_profile(target_accuracy=0.95)
    print(f"Optimal profile: H={profile.H_cycles}, L={profile.L_cycles}, "
          f"halt_steps={profile.halt_max_steps}, exploration={profile.halt_exploration_prob}")
    
    # Get recommendations
    recommendations = optimizer.get_reasoning_recommendations()
    print(f"Hardware tier: {recommendations['hardware_analysis']['memory_tier']}")
    print(f"Expected inference time: {recommendations['performance_predictions']['expected_inference_time_ms']:.1f}ms")
    
    # Test adaptive halting config
    email_categories = ['spam', 'work', 'personal', 'finance', 'other']
    adaptive_config = optimizer.create_adaptive_halting_config(profile, email_categories)
    print(f"Adaptive config created for {len(email_categories)} categories")