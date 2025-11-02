"""
System resource monitoring module for MacBook optimization.

This module provides real-time monitoring of system resources including
memory usage, CPU utilization, and thermal management for TRM training.
"""

import time
import threading
import psutil
import subprocess
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from collections import deque


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    swap_used_mb: float
    swap_percent: float


@dataclass
class CPUStats:
    """CPU utilization statistics."""
    percent_total: float
    percent_per_core: List[float]
    frequency_current: float  # MHz
    frequency_max: float      # MHz
    load_average: List[float]  # 1, 5, 15 minute averages


@dataclass
class ThermalStats:
    """Thermal monitoring statistics."""
    cpu_temperature: Optional[float]  # Celsius
    fan_speed: Optional[int]          # RPM
    thermal_state: str                # "normal", "warm", "hot"


@dataclass
class ResourceSnapshot:
    """Complete resource usage snapshot."""
    timestamp: float
    memory: MemoryStats
    cpu: CPUStats
    thermal: ThermalStats


class ResourceMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize resource monitor.
        
        Args:
            history_size: Number of historical snapshots to keep
        """
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        self.callbacks: List[Callable[[ResourceSnapshot], None]] = []
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return MemoryStats(
            total_mb=memory.total / (1024**2),
            available_mb=memory.available / (1024**2),
            used_mb=memory.used / (1024**2),
            percent_used=memory.percent,
            swap_used_mb=swap.used / (1024**2),
            swap_percent=swap.percent
        )
    
    def get_cpu_stats(self) -> CPUStats:
        """Get current CPU utilization statistics."""
        # Get CPU percentage (non-blocking)
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
        
        # Get CPU frequency
        freq = psutil.cpu_freq()
        current_freq = freq.current if freq else 0.0
        max_freq = freq.max if freq else 0.0
        
        # Get load average (Unix-like systems)
        load_avg = [0.0, 0.0, 0.0]
        try:
            if hasattr(psutil, 'getloadavg'):
                load_avg = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass
            
        return CPUStats(
            percent_total=cpu_percent,
            percent_per_core=cpu_percent_per_core,
            frequency_current=current_freq,
            frequency_max=max_freq,
            load_average=load_avg
        )
    
    def get_thermal_stats(self) -> ThermalStats:
        """Get thermal monitoring statistics (macOS specific)."""
        cpu_temp = None
        fan_speed = None
        thermal_state = "normal"
        
        if platform.system() == "Darwin":
            # Try to get CPU temperature using powermetrics (requires sudo)
            # For non-sudo access, we'll use alternative methods
            try:
                # Try using system_profiler for thermal info
                result = subprocess.run(
                    ["system_profiler", "SPPowerDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse thermal information if available
                    output = result.stdout.lower()
                    if "thermal" in output:
                        # Basic thermal state detection
                        if "high" in output or "critical" in output:
                            thermal_state = "hot"
                        elif "warm" in output or "elevated" in output:
                            thermal_state = "warm"
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
            
            # Try to get fan information
            try:
                # Use ioreg to get fan information
                result = subprocess.run(
                    ["ioreg", "-n", "AppleSMC"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # This is a simplified approach - actual fan speed detection
                    # would require more complex parsing or third-party tools
                    pass
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass
        
        # Estimate thermal state based on CPU usage
        cpu_stats = self.get_cpu_stats()
        if cpu_stats.percent_total > 80:
            thermal_state = "hot"
        elif cpu_stats.percent_total > 60:
            thermal_state = "warm"
            
        return ThermalStats(
            cpu_temperature=cpu_temp,
            fan_speed=fan_speed,
            thermal_state=thermal_state
        )
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource usage snapshot."""
        return ResourceSnapshot(
            timestamp=time.time(),
            memory=self.get_memory_stats(),
            cpu=self.get_cpu_stats(),
            thermal=self.get_thermal_stats()
        )
    
    def add_callback(self, callback: Callable[[ResourceSnapshot], None]):
        """Add callback to be called on each monitoring update."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[ResourceSnapshot], None]):
        """Remove callback from monitoring updates."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self.get_current_snapshot()
                self.history.append(snapshot)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        print(f"Error in monitoring callback: {e}")
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
            
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def get_history(self, last_n: Optional[int] = None) -> List[ResourceSnapshot]:
        """
        Get monitoring history.
        
        Args:
            last_n: Number of recent snapshots to return (None for all)
            
        Returns:
            List of resource snapshots
        """
        history_list = list(self.history)
        if last_n is not None:
            return history_list[-last_n:]
        return history_list
    
    def get_average_stats(self, last_n: Optional[int] = None) -> Dict:
        """
        Get average statistics over recent history.
        
        Args:
            last_n: Number of recent snapshots to average (None for all)
            
        Returns:
            Dictionary with averaged statistics
        """
        history = self.get_history(last_n)
        if not history:
            return {}
        
        # Calculate averages
        avg_memory_percent = sum(s.memory.percent_used for s in history) / len(history)
        avg_cpu_percent = sum(s.cpu.percent_total for s in history) / len(history)
        avg_memory_used_mb = sum(s.memory.used_mb for s in history) / len(history)
        
        # Count thermal states
        thermal_states = [s.thermal.thermal_state for s in history]
        thermal_counts = {state: thermal_states.count(state) for state in set(thermal_states)}
        
        return {
            "average_memory_percent": round(avg_memory_percent, 2),
            "average_cpu_percent": round(avg_cpu_percent, 2),
            "average_memory_used_mb": round(avg_memory_used_mb, 2),
            "thermal_state_distribution": thermal_counts,
            "sample_count": len(history),
            "time_span_seconds": history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0
        }
    
    def check_memory_pressure(self, threshold: float = 80.0) -> bool:
        """
        Check if system is under memory pressure.
        
        Args:
            threshold: Memory usage percentage threshold
            
        Returns:
            True if memory usage exceeds threshold
        """
        memory_stats = self.get_memory_stats()
        return memory_stats.percent_used > threshold
    
    def check_thermal_throttling(self) -> bool:
        """
        Check if system might be thermal throttling.
        
        Returns:
            True if thermal throttling is likely
        """
        thermal_stats = self.get_thermal_stats()
        return thermal_stats.thermal_state == "hot"
    
    def get_resource_summary(self) -> Dict:
        """Get a comprehensive resource summary."""
        current = self.get_current_snapshot()
        averages = self.get_average_stats(last_n=10)  # Last 10 samples
        
        return {
            "current": {
                "memory_used_percent": current.memory.percent_used,
                "memory_available_gb": round(current.memory.available_mb / 1024, 2),
                "cpu_percent": current.cpu.percent_total,
                "thermal_state": current.thermal.thermal_state,
            },
            "recent_averages": averages,
            "alerts": {
                "memory_pressure": self.check_memory_pressure(),
                "thermal_throttling": self.check_thermal_throttling(),
            }
        }