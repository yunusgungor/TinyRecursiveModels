"""
Performance reporting system for MacBook TRM training.

This module provides comprehensive performance reporting, training summaries,
resource usage statistics, and optimization suggestions for MacBook training.
"""

import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import statistics
from datetime import datetime, timedelta

from .progress_monitoring import TrainingSession, TrainingProgress, ResourceMetrics, PerformanceMetrics
from .resource_monitoring import ResourceMonitor
from .memory_management import MemoryManager


@dataclass
class TrainingSummary:
    """Comprehensive training session summary."""
    # Session information
    session_id: str
    model_name: str
    dataset_name: str
    start_time: str
    end_time: str
    total_duration_minutes: float
    
    # Training configuration
    batch_size: int
    learning_rate: float
    total_steps: int
    total_epochs: int
    
    # Progress metrics
    steps_completed: int
    samples_processed: int
    completion_percentage: float
    
    # Performance metrics
    average_samples_per_second: float
    peak_samples_per_second: float
    training_efficiency_score: float
    
    # Resource utilization
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    peak_cpu_usage_percent: float
    average_cpu_usage_percent: float
    thermal_throttling_detected: bool
    
    # Training outcomes
    final_loss: Optional[float]
    best_loss: Optional[float]
    loss_improvement: Optional[float]
    convergence_achieved: bool
    
    # Optimization analysis
    primary_bottlenecks: List[str]
    optimization_suggestions: List[str]
    performance_warnings: List[str]
    
    # Hardware utilization efficiency
    memory_efficiency_score: float
    cpu_efficiency_score: float
    overall_hardware_efficiency: float


@dataclass
class ResourceUsageReport:
    """Detailed resource usage analysis."""
    # Memory analysis
    memory_stats: Dict[str, float]
    memory_timeline: List[Tuple[float, float]]  # (timestamp, usage_mb)
    memory_pressure_events: List[Dict[str, Any]]
    
    # CPU analysis
    cpu_stats: Dict[str, float]
    cpu_timeline: List[Tuple[float, float]]  # (timestamp, usage_percent)
    cpu_frequency_timeline: List[Tuple[float, float]]  # (timestamp, frequency_mhz)
    
    # Thermal analysis
    thermal_events: List[Dict[str, Any]]
    thermal_timeline: List[Tuple[float, str]]  # (timestamp, thermal_state)
    
    # Performance bottlenecks
    bottleneck_analysis: Dict[str, Any]
    resource_recommendations: List[str]


@dataclass
class OptimizationReport:
    """Performance optimization recommendations."""
    # Current configuration analysis
    current_config_efficiency: float
    config_bottlenecks: List[str]
    
    # Recommended optimizations
    batch_size_recommendations: Dict[str, Any]
    memory_optimizations: List[str]
    cpu_optimizations: List[str]
    data_loading_optimizations: List[str]
    
    # Expected improvements
    estimated_speed_improvement: float
    estimated_memory_savings: float
    estimated_efficiency_gain: float
    
    # Implementation priority
    high_priority_optimizations: List[str]
    medium_priority_optimizations: List[str]
    low_priority_optimizations: List[str]


class PerformanceReporter:
    """Performance reporting and analysis system."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize performance reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report storage
        self.session_reports: Dict[str, TrainingSummary] = {}
        self.resource_reports: Dict[str, ResourceUsageReport] = {}
        self.optimization_reports: Dict[str, OptimizationReport] = {}
    
    def generate_training_summary(self, session: TrainingSession) -> TrainingSummary:
        """
        Generate comprehensive training summary from session data.
        
        Args:
            session: Completed training session
            
        Returns:
            Training summary report
        """
        # Calculate duration
        duration_minutes = session.total_training_time / 60 if session.total_training_time > 0 else 0
        
        # Analyze progress history
        if session.progress_history:
            final_progress = session.progress_history[-1]
            steps_completed = final_progress.current_step
            samples_processed = final_progress.samples_processed
            completion_percentage = final_progress.progress_percent
            
            # Calculate speed metrics
            speeds = [p.current_samples_per_second for p in session.progress_history 
                     if p.current_samples_per_second > 0]
            peak_sps = max(speeds) if speeds else 0
            avg_sps = session.average_samples_per_second
        else:
            steps_completed = 0
            samples_processed = 0
            completion_percentage = 0
            peak_sps = 0
            avg_sps = 0
        
        # Analyze resource usage
        if session.resource_history:
            memory_usages = [r.memory_used_mb for r in session.resource_history]
            cpu_usages = [r.cpu_usage_percent for r in session.resource_history]
            
            peak_memory = max(memory_usages)
            avg_memory = statistics.mean(memory_usages)
            peak_cpu = max(cpu_usages)
            avg_cpu = statistics.mean(cpu_usages)
            
            # Check for thermal throttling
            thermal_throttling = any(
                r.thermal_state == "hot" for r in session.resource_history
            )
        else:
            peak_memory = session.peak_memory_usage_mb
            avg_memory = peak_memory
            peak_cpu = 0
            avg_cpu = 0
            thermal_throttling = False
        
        # Analyze training outcomes
        losses = [p.current_loss for p in session.progress_history 
                 if p.current_loss is not None]
        
        final_loss = session.final_loss
        best_loss = session.best_loss
        loss_improvement = None
        convergence_achieved = False
        
        if losses and len(losses) > 1:
            loss_improvement = losses[0] - losses[-1]
            # Simple convergence check: loss stable for last 10% of training
            if len(losses) > 10:
                recent_losses = losses[-max(1, len(losses)//10):]
                loss_std = statistics.stdev(recent_losses) if len(recent_losses) > 1 else 0
                convergence_achieved = loss_std < 0.01  # Threshold for convergence
        
        # Analyze performance metrics
        if session.performance_history:
            efficiency_scores = [p.training_efficiency_score for p in session.performance_history]
            training_efficiency = statistics.mean(efficiency_scores)
            
            # Collect bottlenecks and suggestions
            all_bottlenecks = [p.primary_bottleneck for p in session.performance_history]
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            primary_bottlenecks = sorted(bottleneck_counts.items(), 
                                       key=lambda x: x[1], reverse=True)
            primary_bottlenecks = [b[0] for b in primary_bottlenecks[:3]]
            
            # Collect unique suggestions and warnings
            all_suggestions = []
            all_warnings = []
            for perf in session.performance_history:
                all_suggestions.extend(perf.optimization_suggestions)
                all_warnings.extend(perf.performance_warnings)
            
            optimization_suggestions = list(set(all_suggestions))
            performance_warnings = list(set(all_warnings))
        else:
            training_efficiency = 0
            primary_bottlenecks = []
            optimization_suggestions = []
            performance_warnings = []
        
        # Calculate efficiency scores
        memory_efficiency = min(100, avg_memory / peak_memory * 100) if peak_memory > 0 else 0
        cpu_efficiency = avg_cpu
        overall_efficiency = (memory_efficiency + cpu_efficiency + training_efficiency) / 3
        
        # Format timestamps
        start_time_str = datetime.fromtimestamp(session.start_time).isoformat()
        end_time_str = (datetime.fromtimestamp(session.end_time).isoformat() 
                       if session.end_time else "")
        
        summary = TrainingSummary(
            session_id=session.session_id,
            model_name=session.model_name,
            dataset_name=session.dataset_name,
            start_time=start_time_str,
            end_time=end_time_str,
            total_duration_minutes=duration_minutes,
            batch_size=session.batch_size,
            learning_rate=session.learning_rate,
            total_steps=len(session.progress_history),
            total_epochs=1,  # Simplified for now
            steps_completed=steps_completed,
            samples_processed=samples_processed,
            completion_percentage=completion_percentage,
            average_samples_per_second=avg_sps,
            peak_samples_per_second=peak_sps,
            training_efficiency_score=training_efficiency,
            peak_memory_usage_mb=peak_memory,
            average_memory_usage_mb=avg_memory,
            peak_cpu_usage_percent=peak_cpu,
            average_cpu_usage_percent=avg_cpu,
            thermal_throttling_detected=thermal_throttling,
            final_loss=final_loss,
            best_loss=best_loss,
            loss_improvement=loss_improvement,
            convergence_achieved=convergence_achieved,
            primary_bottlenecks=primary_bottlenecks,
            optimization_suggestions=optimization_suggestions,
            performance_warnings=performance_warnings,
            memory_efficiency_score=memory_efficiency,
            cpu_efficiency_score=cpu_efficiency,
            overall_hardware_efficiency=overall_efficiency
        )
        
        self.session_reports[session.session_id] = summary
        return summary
    
    def generate_resource_usage_report(self, session: TrainingSession) -> ResourceUsageReport:
        """
        Generate detailed resource usage analysis.
        
        Args:
            session: Training session with resource history
            
        Returns:
            Resource usage report
        """
        if not session.resource_history:
            return ResourceUsageReport(
                memory_stats={}, memory_timeline=[], memory_pressure_events=[],
                cpu_stats={}, cpu_timeline=[], cpu_frequency_timeline=[],
                thermal_events=[], thermal_timeline=[],
                bottleneck_analysis={}, resource_recommendations=[]
            )
        
        # Memory analysis
        memory_usages = [r.memory_used_mb for r in session.resource_history]
        memory_stats = {
            "peak_mb": max(memory_usages),
            "average_mb": statistics.mean(memory_usages),
            "min_mb": min(memory_usages),
            "std_dev_mb": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
        }
        
        memory_timeline = [
            (session.start_time + i * 2.0, usage)  # Assuming 2s intervals
            for i, usage in enumerate(memory_usages)
        ]
        
        # Identify memory pressure events
        memory_pressure_events = []
        for i, resource in enumerate(session.resource_history):
            if resource.memory_pressure_level in ["high", "critical"]:
                memory_pressure_events.append({
                    "timestamp": session.start_time + i * 2.0,
                    "pressure_level": resource.memory_pressure_level,
                    "memory_usage_mb": resource.memory_used_mb,
                    "memory_usage_percent": resource.memory_usage_percent
                })
        
        # CPU analysis
        cpu_usages = [r.cpu_usage_percent for r in session.resource_history]
        cpu_stats = {
            "peak_percent": max(cpu_usages),
            "average_percent": statistics.mean(cpu_usages),
            "min_percent": min(cpu_usages),
            "std_dev_percent": statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0
        }
        
        cpu_timeline = [
            (session.start_time + i * 2.0, usage)
            for i, usage in enumerate(cpu_usages)
        ]
        
        cpu_frequencies = [r.cpu_frequency_mhz for r in session.resource_history]
        cpu_frequency_timeline = [
            (session.start_time + i * 2.0, freq)
            for i, freq in enumerate(cpu_frequencies)
        ]
        
        # Thermal analysis
        thermal_events = []
        thermal_timeline = []
        
        for i, resource in enumerate(session.resource_history):
            timestamp = session.start_time + i * 2.0
            thermal_timeline.append((timestamp, resource.thermal_state))
            
            if resource.thermal_state in ["warm", "hot"]:
                thermal_events.append({
                    "timestamp": timestamp,
                    "thermal_state": resource.thermal_state,
                    "cpu_usage": resource.cpu_usage_percent,
                    "duration_estimate": 2.0  # Simplified
                })
        
        # Bottleneck analysis
        bottleneck_counts = {}
        if session.performance_history:
            for perf in session.performance_history:
                bottleneck = perf.primary_bottleneck
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        total_measurements = len(session.performance_history)
        bottleneck_analysis = {
            "bottleneck_distribution": {
                k: (v / total_measurements * 100) for k, v in bottleneck_counts.items()
            } if total_measurements > 0 else {},
            "primary_bottleneck": max(bottleneck_counts.items(), key=lambda x: x[1])[0] if bottleneck_counts else "unknown",
            "bottleneck_severity": max(bottleneck_counts.values()) / total_measurements if total_measurements > 0 else 0
        }
        
        # Resource recommendations
        recommendations = []
        
        if memory_stats["peak_mb"] > 6000:  # > 6GB on 8GB system
            recommendations.append("Consider reducing batch size to lower peak memory usage")
        elif memory_stats["average_mb"] < 2000:  # < 2GB average
            recommendations.append("Memory usage is low - consider increasing batch size")
        
        if cpu_stats["average_percent"] < 30:
            recommendations.append("Low CPU utilization - check for data loading bottlenecks")
        elif cpu_stats["peak_percent"] > 95:
            recommendations.append("Very high CPU usage detected - may cause thermal throttling")
        
        if thermal_events:
            recommendations.append("Thermal events detected - consider reducing computational load")
        
        if bottleneck_analysis["primary_bottleneck"] == "memory":
            recommendations.append("Memory is the primary bottleneck - optimize memory usage")
        elif bottleneck_analysis["primary_bottleneck"] == "data_loading":
            recommendations.append("Data loading is bottleneck - increase workers or optimize preprocessing")
        
        report = ResourceUsageReport(
            memory_stats=memory_stats,
            memory_timeline=memory_timeline,
            memory_pressure_events=memory_pressure_events,
            cpu_stats=cpu_stats,
            cpu_timeline=cpu_timeline,
            cpu_frequency_timeline=cpu_frequency_timeline,
            thermal_events=thermal_events,
            thermal_timeline=thermal_timeline,
            bottleneck_analysis=bottleneck_analysis,
            resource_recommendations=recommendations
        )
        
        self.resource_reports[session.session_id] = report
        return report
    
    def generate_optimization_report(self, session: TrainingSession, 
                                   current_config: Dict[str, Any]) -> OptimizationReport:
        """
        Generate optimization recommendations based on training performance.
        
        Args:
            session: Training session data
            current_config: Current training configuration
            
        Returns:
            Optimization recommendations report
        """
        # Analyze current configuration efficiency
        if session.performance_history:
            efficiency_scores = [p.training_efficiency_score for p in session.performance_history]
            current_efficiency = statistics.mean(efficiency_scores)
        else:
            current_efficiency = 0
        
        # Identify configuration bottlenecks
        config_bottlenecks = []
        if session.resource_history:
            avg_memory_usage = statistics.mean([r.memory_usage_percent for r in session.resource_history])
            if avg_memory_usage > 85:
                config_bottlenecks.append("batch_size_too_large")
            elif avg_memory_usage < 40:
                config_bottlenecks.append("batch_size_too_small")
            
            avg_cpu_usage = statistics.mean([r.cpu_usage_percent for r in session.resource_history])
            if avg_cpu_usage < 30:
                config_bottlenecks.append("underutilized_cpu")
        
        # Batch size recommendations
        current_batch_size = session.batch_size
        batch_recommendations = {
            "current_batch_size": current_batch_size,
            "recommended_batch_size": current_batch_size,
            "reasoning": "Current batch size appears optimal"
        }
        
        if "batch_size_too_large" in config_bottlenecks:
            recommended_batch_size = max(1, int(current_batch_size * 0.75))
            batch_recommendations.update({
                "recommended_batch_size": recommended_batch_size,
                "reasoning": "Reduce batch size to lower memory pressure"
            })
        elif "batch_size_too_small" in config_bottlenecks:
            recommended_batch_size = min(64, int(current_batch_size * 1.5))
            batch_recommendations.update({
                "recommended_batch_size": recommended_batch_size,
                "reasoning": "Increase batch size to better utilize available memory"
            })
        
        # Memory optimizations
        memory_optimizations = []
        if session.resource_history:
            peak_memory = max(r.memory_used_mb for r in session.resource_history)
            if peak_memory > 6000:
                memory_optimizations.extend([
                    "Enable gradient accumulation to reduce memory per step",
                    "Use mixed precision training if supported",
                    "Implement gradient checkpointing for large models"
                ])
            
            memory_pressure_events = sum(
                1 for r in session.resource_history 
                if r.memory_pressure_level in ["high", "critical"]
            )
            if memory_pressure_events > len(session.resource_history) * 0.1:
                memory_optimizations.append("Implement dynamic batch size adjustment")
        
        # CPU optimizations
        cpu_optimizations = []
        if session.resource_history:
            avg_cpu = statistics.mean([r.cpu_usage_percent for r in session.resource_history])
            if avg_cpu < 50:
                cpu_optimizations.extend([
                    "Increase number of data loading workers",
                    "Enable CPU-specific optimizations (MKL, OpenMP)",
                    "Optimize data preprocessing pipeline"
                ])
            elif avg_cpu > 90:
                cpu_optimizations.extend([
                    "Reduce computational complexity per batch",
                    "Implement computation batching",
                    "Add cooling breaks during training"
                ])
        
        # Data loading optimizations
        data_loading_optimizations = []
        if session.performance_history:
            data_loading_bottlenecks = sum(
                1 for p in session.performance_history 
                if p.primary_bottleneck == "data_loading"
            )
            if data_loading_bottlenecks > len(session.performance_history) * 0.2:
                data_loading_optimizations.extend([
                    "Increase prefetch factor in DataLoader",
                    "Use memory-mapped datasets for large data",
                    "Implement data caching for frequently accessed samples",
                    "Optimize data preprocessing and augmentation"
                ])
        
        # Estimate improvements
        estimated_speed_improvement = 0.0
        estimated_memory_savings = 0.0
        estimated_efficiency_gain = 0.0
        
        if batch_recommendations["recommended_batch_size"] != current_batch_size:
            # Rough estimates based on batch size changes
            batch_ratio = batch_recommendations["recommended_batch_size"] / current_batch_size
            if batch_ratio < 1:  # Reducing batch size
                estimated_memory_savings = (1 - batch_ratio) * 30  # 30% memory savings estimate
                estimated_speed_improvement = -10  # Slight speed reduction
            else:  # Increasing batch size
                estimated_speed_improvement = (batch_ratio - 1) * 20  # 20% speed improvement estimate
        
        if memory_optimizations:
            estimated_memory_savings += 15  # 15% additional savings from optimizations
            estimated_efficiency_gain += 10
        
        if cpu_optimizations:
            estimated_speed_improvement += 25  # 25% speed improvement from CPU optimizations
            estimated_efficiency_gain += 15
        
        if data_loading_optimizations:
            estimated_speed_improvement += 20  # 20% speed improvement from data loading
            estimated_efficiency_gain += 10
        
        # Prioritize optimizations
        high_priority = []
        medium_priority = []
        low_priority = []
        
        # High priority: critical performance issues
        if "batch_size_too_large" in config_bottlenecks:
            high_priority.append("Reduce batch size to prevent OOM errors")
        if session.resource_history and any(r.thermal_state == "hot" for r in session.resource_history):
            high_priority.append("Address thermal throttling issues")
        
        # Medium priority: significant improvements
        if memory_optimizations:
            medium_priority.extend(memory_optimizations[:2])  # Top 2 memory optimizations
        if cpu_optimizations:
            medium_priority.extend(cpu_optimizations[:2])  # Top 2 CPU optimizations
        
        # Low priority: minor improvements
        if data_loading_optimizations:
            low_priority.extend(data_loading_optimizations)
        if len(memory_optimizations) > 2:
            low_priority.extend(memory_optimizations[2:])
        if len(cpu_optimizations) > 2:
            low_priority.extend(cpu_optimizations[2:])
        
        report = OptimizationReport(
            current_config_efficiency=current_efficiency,
            config_bottlenecks=config_bottlenecks,
            batch_size_recommendations=batch_recommendations,
            memory_optimizations=memory_optimizations,
            cpu_optimizations=cpu_optimizations,
            data_loading_optimizations=data_loading_optimizations,
            estimated_speed_improvement=estimated_speed_improvement,
            estimated_memory_savings=estimated_memory_savings,
            estimated_efficiency_gain=estimated_efficiency_gain,
            high_priority_optimizations=high_priority,
            medium_priority_optimizations=medium_priority,
            low_priority_optimizations=low_priority
        )
        
        self.optimization_reports[session.session_id] = report
        return report
    
    def save_reports(self, session: TrainingSession, current_config: Dict[str, Any] = None):
        """
        Generate and save all reports for a training session.
        
        Args:
            session: Completed training session
            current_config: Current training configuration
        """
        session_dir = self.output_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        training_summary = self.generate_training_summary(session)
        resource_report = self.generate_resource_usage_report(session)
        
        if current_config:
            optimization_report = self.generate_optimization_report(session, current_config)
        else:
            optimization_report = None
        
        # Save training summary
        with open(session_dir / "training_summary.json", "w") as f:
            json.dump(asdict(training_summary), f, indent=2, default=str)
        
        # Save resource usage report
        with open(session_dir / "resource_usage.json", "w") as f:
            json.dump(asdict(resource_report), f, indent=2, default=str)
        
        # Save optimization report
        if optimization_report:
            with open(session_dir / "optimization_recommendations.json", "w") as f:
                json.dump(asdict(optimization_report), f, indent=2, default=str)
        
        # Save raw session data
        session_data = {
            "session_info": {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "model_name": session.model_name,
                "dataset_name": session.dataset_name,
                "batch_size": session.batch_size,
                "learning_rate": session.learning_rate
            },
            "progress_history": [asdict(p) for p in session.progress_history],
            "resource_history": [asdict(r) for r in session.resource_history],
            "performance_history": [asdict(p) for p in session.performance_history]
        }
        
        with open(session_dir / "raw_session_data.json", "w") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"Reports saved to: {session_dir}")
    
    def generate_text_summary(self, session: TrainingSession) -> str:
        """
        Generate human-readable text summary of training session.
        
        Args:
            session: Training session
            
        Returns:
            Formatted text summary
        """
        summary = self.generate_training_summary(session)
        resource_report = self.generate_resource_usage_report(session)
        
        lines = [
            "="*60,
            f"MacBook TRM Training Report - {summary.session_id}",
            "="*60,
            "",
            "SESSION INFORMATION:",
            f"  Model: {summary.model_name}",
            f"  Dataset: {summary.dataset_name}",
            f"  Duration: {summary.total_duration_minutes:.1f} minutes",
            f"  Completion: {summary.completion_percentage:.1f}%",
            "",
            "TRAINING CONFIGURATION:",
            f"  Batch Size: {summary.batch_size}",
            f"  Learning Rate: {summary.learning_rate:.2e}",
            f"  Steps Completed: {summary.steps_completed:,}",
            f"  Samples Processed: {summary.samples_processed:,}",
            "",
            "PERFORMANCE METRICS:",
            f"  Average Speed: {summary.average_samples_per_second:.1f} samples/second",
            f"  Peak Speed: {summary.peak_samples_per_second:.1f} samples/second",
            f"  Training Efficiency: {summary.training_efficiency_score:.1f}%",
            "",
            "RESOURCE UTILIZATION:",
            f"  Peak Memory: {summary.peak_memory_usage_mb:.0f}MB ({summary.peak_memory_usage_mb/1024:.1f}GB)",
            f"  Average Memory: {summary.average_memory_usage_mb:.0f}MB",
            f"  Peak CPU: {summary.peak_cpu_usage_percent:.1f}%",
            f"  Average CPU: {summary.average_cpu_usage_percent:.1f}%",
            f"  Thermal Throttling: {'Yes' if summary.thermal_throttling_detected else 'No'}",
            "",
            "TRAINING OUTCOMES:",
            f"  Final Loss: {summary.final_loss:.4f}" if summary.final_loss else "  Final Loss: N/A",
            f"  Best Loss: {summary.best_loss:.4f}" if summary.best_loss else "  Best Loss: N/A",
            f"  Loss Improvement: {summary.loss_improvement:.4f}" if summary.loss_improvement else "  Loss Improvement: N/A",
            f"  Convergence: {'Yes' if summary.convergence_achieved else 'No'}",
            "",
            "EFFICIENCY ANALYSIS:",
            f"  Memory Efficiency: {summary.memory_efficiency_score:.1f}%",
            f"  CPU Efficiency: {summary.cpu_efficiency_score:.1f}%",
            f"  Overall Hardware Efficiency: {summary.overall_hardware_efficiency:.1f}%",
        ]
        
        # Add bottleneck analysis
        if summary.primary_bottlenecks:
            lines.extend([
                "",
                "PRIMARY BOTTLENECKS:",
            ])
            for i, bottleneck in enumerate(summary.primary_bottlenecks, 1):
                lines.append(f"  {i}. {bottleneck.replace('_', ' ').title()}")
        
        # Add optimization suggestions
        if summary.optimization_suggestions:
            lines.extend([
                "",
                "OPTIMIZATION SUGGESTIONS:",
            ])
            for suggestion in summary.optimization_suggestions:
                lines.append(f"  • {suggestion}")
        
        # Add warnings
        if summary.performance_warnings:
            lines.extend([
                "",
                "⚠️  PERFORMANCE WARNINGS:",
            ])
            for warning in summary.performance_warnings:
                lines.append(f"  • {warning}")
        
        # Add resource recommendations
        if resource_report.resource_recommendations:
            lines.extend([
                "",
                "RESOURCE RECOMMENDATIONS:",
            ])
            for rec in resource_report.resource_recommendations:
                lines.append(f"  • {rec}")
        
        lines.extend([
            "",
            "="*60,
            f"Report generated at: {datetime.now().isoformat()}",
            "="*60
        ])
        
        return "\n".join(lines)
    
    def save_text_summary(self, session: TrainingSession, filename: Optional[str] = None):
        """
        Save human-readable text summary to file.
        
        Args:
            session: Training session
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{session.session_id}_summary.txt"
        
        summary_text = self.generate_text_summary(session)
        
        session_dir = self.output_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        with open(session_dir / filename, "w") as f:
            f.write(summary_text)
        
        print(f"Text summary saved to: {session_dir / filename}")
    
    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple training sessions.
        
        Args:
            session_ids: List of session IDs to compare
            
        Returns:
            Comparison report
        """
        if not session_ids or len(session_ids) < 2:
            return {"error": "Need at least 2 sessions to compare"}
        
        # Get summaries for all sessions
        summaries = []
        for session_id in session_ids:
            if session_id in self.session_reports:
                summaries.append(self.session_reports[session_id])
            else:
                return {"error": f"Session {session_id} not found"}
        
        # Compare key metrics
        comparison = {
            "sessions": session_ids,
            "metrics_comparison": {
                "training_efficiency": [s.training_efficiency_score for s in summaries],
                "average_speed_sps": [s.average_samples_per_second for s in summaries],
                "peak_memory_mb": [s.peak_memory_usage_mb for s in summaries],
                "average_cpu_percent": [s.average_cpu_usage_percent for s in summaries],
                "final_loss": [s.final_loss for s in summaries if s.final_loss is not None],
                "completion_percentage": [s.completion_percentage for s in summaries]
            },
            "best_session": {
                "highest_efficiency": session_ids[max(range(len(summaries)), 
                                                   key=lambda i: summaries[i].training_efficiency_score)],
                "fastest_training": session_ids[max(range(len(summaries)), 
                                                  key=lambda i: summaries[i].average_samples_per_second)],
                "lowest_memory": session_ids[min(range(len(summaries)), 
                                               key=lambda i: summaries[i].peak_memory_usage_mb)],
                "best_loss": session_ids[min(range(len(summaries)), 
                                           key=lambda i: summaries[i].final_loss or float('inf'))]
            }
        }
        
        return comparison