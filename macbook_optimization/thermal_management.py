"""
Thermal Management for MacBook Email Training

This module provides thermal monitoring and management specifically for email
classification training, including thermal throttling, cooling optimization,
and training intensity adjustment based on thermal conditions.
"""

import os
import time
import threading
import logging
import subprocess
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import deque

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ThermalState:
    """Thermal state information."""
    timestamp: float
    cpu_temperature: Optional[float]  # Celsius
    gpu_temperature: Optional[float]  # Celsius (if available)
    fan_speed: Optional[int]          # RPM
    thermal_pressure: str             # "low", "medium", "high", "critical"
    throttling_active: bool
    power_consumption: Optional[float]  # Watts (if available)


@dataclass
class ThermalThresholds:
    """Thermal management thresholds."""
    # Temperature thresholds (Celsius)
    temp_warning: float = 75.0
    temp_critical: float = 85.0
    temp_emergency: float = 95.0
    
    # CPU usage thresholds for thermal management
    cpu_throttle_threshold: float = 85.0
    cpu_recovery_threshold: float = 70.0
    
    # Fan speed thresholds (RPM)
    fan_speed_high: int = 4000
    fan_speed_critical: int = 6000
    
    # Thermal pressure response times (seconds)
    throttle_response_time: float = 5.0
    recovery_response_time: float = 30.0


@dataclass
class ThermalAction:
    """Thermal management action."""
    timestamp: float
    action_type: str  # "throttle", "recover", "alert", "emergency"
    severity: str     # "info", "warning", "critical"
    description: str
    parameters: Dict[str, Any]
    success: bool


class ThermalMonitor:
    """
    Thermal monitoring system for MacBook email training.
    
    Monitors system thermal state and provides thermal management
    capabilities to prevent overheating during intensive training.
    """
    
    def __init__(self, 
                 thresholds: Optional[ThermalThresholds] = None,
                 monitoring_interval: float = 2.0):
        """
        Initialize thermal monitor.
        
        Args:
            thresholds: Thermal management thresholds
            monitoring_interval: Monitoring interval in seconds
        """
        self.thresholds = thresholds or ThermalThresholds()
        self.monitoring_interval = monitoring_interval
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Thermal history
        self.thermal_history: deque = deque(maxlen=300)  # 10 minutes at 2s interval
        self.thermal_actions: List[ThermalAction] = []
        
        # Callbacks
        self.thermal_callbacks: List[Callable[[ThermalState], None]] = []
        self.action_callbacks: List[Callable[[ThermalAction], None]] = []
        
        # Platform-specific detection
        self.is_macos = platform.system() == "Darwin"
        self.thermal_tools_available = self._check_thermal_tools()
        
        logger.info("ThermalMonitor initialized")
    
    def _check_thermal_tools(self) -> bool:
        """Check if thermal monitoring tools are available."""
        if self.is_macos:
            try:
                # Check if we can run system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPPowerDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                return False
        else:
            # For non-macOS systems, check for lm-sensors or similar
            try:
                result = subprocess.run(
                    ["sensors", "-u"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                return False
    
    def get_thermal_state(self) -> ThermalState:
        """
        Get current thermal state.
        
        Returns:
            Current thermal state
        """
        timestamp = time.time()
        cpu_temp = None
        gpu_temp = None
        fan_speed = None
        power_consumption = None
        
        # Get CPU temperature
        if self.is_macos and self.thermal_tools_available:
            cpu_temp = self._get_macos_cpu_temperature()
            fan_speed = self._get_macos_fan_speed()
            power_consumption = self._get_macos_power_consumption()
        elif not self.is_macos and self.thermal_tools_available:
            cpu_temp = self._get_linux_cpu_temperature()
            fan_speed = self._get_linux_fan_speed()
        
        # Determine thermal pressure based on available metrics
        thermal_pressure = self._calculate_thermal_pressure(cpu_temp, fan_speed)
        
        # Check for throttling
        throttling_active = self._detect_thermal_throttling(cpu_temp, thermal_pressure)
        
        return ThermalState(
            timestamp=timestamp,
            cpu_temperature=cpu_temp,
            gpu_temperature=gpu_temp,
            fan_speed=fan_speed,
            thermal_pressure=thermal_pressure,
            throttling_active=throttling_active,
            power_consumption=power_consumption
        )
    
    def _get_macos_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature on macOS."""
        try:
            # Try using powermetrics (requires sudo, so may not work)
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "smc", "-n", "1", "-i", "1"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse powermetrics output for CPU temperature
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CPU die temperature' in line:
                        try:
                            temp_str = line.split(':')[1].strip().split()[0]
                            return float(temp_str)
                        except (ValueError, IndexError):
                            pass
            
            # Fallback: estimate based on CPU usage and fan speed
            cpu_percent = psutil.cpu_percent(interval=1.0)
            estimated_temp = 40 + (cpu_percent * 0.5)  # Rough estimation
            return estimated_temp
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # Fallback estimation
            cpu_percent = psutil.cpu_percent(interval=1.0)
            return 40 + (cpu_percent * 0.5)
    
    def _get_macos_fan_speed(self) -> Optional[int]:
        """Get fan speed on macOS."""
        try:
            # Try using system_profiler
            result = subprocess.run(
                ["system_profiler", "SPPowerDataType"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse for fan information (simplified)
                # Real implementation would need more sophisticated parsing
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'fan' in line.lower() and 'rpm' in line.lower():
                        try:
                            # Extract RPM value
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if 'rpm' in part.lower() and i > 0:
                                    return int(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            
            # Fallback: estimate based on CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            estimated_rpm = 2000 + int(cpu_percent * 40)  # Rough estimation
            return estimated_rpm
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            return None
    
    def _get_macos_power_consumption(self) -> Optional[float]:
        """Get power consumption on macOS."""
        try:
            # Try using powermetrics
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "1"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse powermetrics output for power consumption
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CPU Power' in line and 'mW' in line:
                        try:
                            power_str = line.split(':')[1].strip().split()[0]
                            return float(power_str) / 1000.0  # Convert mW to W
                        except (ValueError, IndexError):
                            pass
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None
    
    def _get_linux_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature on Linux."""
        try:
            result = subprocess.run(
                ["sensors", "-u"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'temp1_input' in line or 'Core 0' in line:
                        try:
                            temp_str = line.split(':')[1].strip()
                            return float(temp_str)
                        except (ValueError, IndexError):
                            pass
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None
    
    def _get_linux_fan_speed(self) -> Optional[int]:
        """Get fan speed on Linux."""
        try:
            result = subprocess.run(
                ["sensors", "-u"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'fan1_input' in line:
                        try:
                            rpm_str = line.split(':')[1].strip()
                            return int(float(rpm_str))
                        except (ValueError, IndexError):
                            pass
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return None
    
    def _calculate_thermal_pressure(self, 
                                  cpu_temp: Optional[float], 
                                  fan_speed: Optional[int]) -> str:
        """Calculate thermal pressure level."""
        pressure_score = 0
        
        # Temperature contribution
        if cpu_temp is not None:
            if cpu_temp >= self.thresholds.temp_emergency:
                pressure_score += 4
            elif cpu_temp >= self.thresholds.temp_critical:
                pressure_score += 3
            elif cpu_temp >= self.thresholds.temp_warning:
                pressure_score += 2
            elif cpu_temp >= 60:
                pressure_score += 1
        
        # Fan speed contribution
        if fan_speed is not None:
            if fan_speed >= self.thresholds.fan_speed_critical:
                pressure_score += 3
            elif fan_speed >= self.thresholds.fan_speed_high:
                pressure_score += 2
            elif fan_speed >= 3000:
                pressure_score += 1
        
        # CPU usage contribution
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent >= 95:
            pressure_score += 2
        elif cpu_percent >= 85:
            pressure_score += 1
        
        # Determine pressure level
        if pressure_score >= 6:
            return "critical"
        elif pressure_score >= 4:
            return "high"
        elif pressure_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _detect_thermal_throttling(self, 
                                 cpu_temp: Optional[float], 
                                 thermal_pressure: str) -> bool:
        """Detect if thermal throttling is active."""
        # Check CPU frequency throttling
        cpu_freq = psutil.cpu_freq()
        if cpu_freq and cpu_freq.current < cpu_freq.max * 0.8:
            return True
        
        # Check temperature-based throttling
        if cpu_temp and cpu_temp >= self.thresholds.temp_critical:
            return True
        
        # Check pressure-based throttling
        if thermal_pressure == "critical":
            return True
        
        return False
    
    def start_monitoring(self) -> None:
        """Start thermal monitoring."""
        if self.monitoring_active:
            logger.warning("Thermal monitoring already active")
            return
        
        logger.info("Starting thermal monitoring")
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring."""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping thermal monitoring")
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Thermal monitoring loop."""
        while self.monitoring_active:
            try:
                # Get current thermal state
                thermal_state = self.get_thermal_state()
                
                # Add to history
                self.thermal_history.append(thermal_state)
                
                # Check for thermal actions needed
                self._check_thermal_actions(thermal_state)
                
                # Call thermal callbacks
                for callback in self.thermal_callbacks:
                    try:
                        callback(thermal_state)
                    except Exception as e:
                        logger.error(f"Error in thermal callback: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_thermal_actions(self, thermal_state: ThermalState) -> None:
        """Check if thermal actions are needed."""
        current_time = thermal_state.timestamp
        
        # Check for emergency thermal action
        if thermal_state.thermal_pressure == "critical":
            self._execute_thermal_action(
                "emergency", "critical",
                "Critical thermal state detected - emergency throttling",
                {"thermal_pressure": thermal_state.thermal_pressure,
                 "cpu_temperature": thermal_state.cpu_temperature},
                current_time
            )
        
        # Check for thermal throttling
        elif thermal_state.thermal_pressure == "high" or thermal_state.throttling_active:
            self._execute_thermal_action(
                "throttle", "warning",
                "High thermal pressure - applying throttling",
                {"thermal_pressure": thermal_state.thermal_pressure,
                 "throttling_active": thermal_state.throttling_active},
                current_time
            )
        
        # Check for thermal recovery
        elif thermal_state.thermal_pressure == "low":
            # Check if we were previously throttling
            recent_actions = [a for a in self.thermal_actions[-5:] 
                            if a.action_type in ["throttle", "emergency"]]
            if recent_actions:
                self._execute_thermal_action(
                    "recover", "info",
                    "Thermal pressure reduced - recovering performance",
                    {"thermal_pressure": thermal_state.thermal_pressure},
                    current_time
                )
    
    def _execute_thermal_action(self, 
                              action_type: str,
                              severity: str,
                              description: str,
                              parameters: Dict[str, Any],
                              timestamp: float) -> None:
        """Execute thermal management action."""
        # Check if similar action was recently executed
        recent_actions = [a for a in self.thermal_actions[-3:] 
                         if a.action_type == action_type and 
                         timestamp - a.timestamp < 30.0]  # 30 second cooldown
        
        if recent_actions:
            return  # Skip if similar action was recent
        
        success = True
        
        try:
            # Execute action based on type
            if action_type == "emergency":
                success = self._emergency_thermal_response()
            elif action_type == "throttle":
                success = self._apply_thermal_throttling()
            elif action_type == "recover":
                success = self._recover_from_thermal_throttling()
            
        except Exception as e:
            logger.error(f"Error executing thermal action {action_type}: {e}")
            success = False
        
        # Record action
        action = ThermalAction(
            timestamp=timestamp,
            action_type=action_type,
            severity=severity,
            description=description,
            parameters=parameters,
            success=success
        )
        
        self.thermal_actions.append(action)
        if len(self.thermal_actions) > 100:  # Keep only recent actions
            self.thermal_actions.pop(0)
        
        # Log action
        log_level = logging.CRITICAL if severity == "critical" else logging.WARNING
        logger.log(log_level, f"Thermal Action [{action_type.upper()}]: {description}")
        
        # Call action callbacks
        for callback in self.action_callbacks:
            try:
                callback(action)
            except Exception as e:
                logger.error(f"Error in thermal action callback: {e}")
    
    def _emergency_thermal_response(self) -> bool:
        """Execute emergency thermal response."""
        logger.critical("EMERGENCY THERMAL RESPONSE ACTIVATED")
        
        # Drastically reduce CPU usage
        try:
            import torch
            if torch.get_num_threads() > 1:
                torch.set_num_threads(1)
                logger.info("Reduced PyTorch threads to 1 for emergency thermal response")
        except ImportError:
            pass
        
        # Set emergency environment variables
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        return True
    
    def _apply_thermal_throttling(self) -> bool:
        """Apply thermal throttling."""
        logger.warning("Applying thermal throttling")
        
        try:
            import torch
            current_threads = torch.get_num_threads()
            if current_threads > 2:
                new_threads = max(1, current_threads // 2)
                torch.set_num_threads(new_threads)
                logger.info(f"Reduced PyTorch threads from {current_threads} to {new_threads}")
        except ImportError:
            pass
        
        return True
    
    def _recover_from_thermal_throttling(self) -> bool:
        """Recover from thermal throttling."""
        logger.info("Recovering from thermal throttling")
        
        # This would restore normal thread counts
        # Implementation depends on how the original configuration was stored
        
        return True
    
    def add_thermal_callback(self, callback: Callable[[ThermalState], None]) -> None:
        """Add callback for thermal state updates."""
        self.thermal_callbacks.append(callback)
    
    def add_action_callback(self, callback: Callable[[ThermalAction], None]) -> None:
        """Add callback for thermal actions."""
        self.action_callbacks.append(callback)
    
    def get_thermal_summary(self) -> Dict[str, Any]:
        """Get comprehensive thermal summary."""
        if not self.thermal_history:
            return {"status": "no_data", "message": "No thermal data available"}
        
        current_state = self.thermal_history[-1]
        recent_states = list(self.thermal_history)[-30:]  # Last minute
        
        # Calculate averages
        avg_temp = None
        if any(s.cpu_temperature for s in recent_states):
            temps = [s.cpu_temperature for s in recent_states if s.cpu_temperature]
            avg_temp = sum(temps) / len(temps)
        
        avg_fan_speed = None
        if any(s.fan_speed for s in recent_states):
            speeds = [s.fan_speed for s in recent_states if s.fan_speed]
            avg_fan_speed = sum(speeds) / len(speeds)
        
        # Count thermal events
        throttling_events = sum(1 for s in recent_states if s.throttling_active)
        high_pressure_events = sum(1 for s in recent_states if s.thermal_pressure in ["high", "critical"])
        
        return {
            "status": "active",
            "current_state": {
                "cpu_temperature": current_state.cpu_temperature,
                "fan_speed": current_state.fan_speed,
                "thermal_pressure": current_state.thermal_pressure,
                "throttling_active": current_state.throttling_active,
                "power_consumption": current_state.power_consumption
            },
            "recent_averages": {
                "cpu_temperature": avg_temp,
                "fan_speed": avg_fan_speed
            },
            "thermal_events": {
                "throttling_events": throttling_events,
                "high_pressure_events": high_pressure_events,
                "total_actions": len(self.thermal_actions)
            },
            "thermal_actions": [
                {
                    "timestamp": action.timestamp,
                    "action_type": action.action_type,
                    "severity": action.severity,
                    "description": action.description,
                    "success": action.success
                }
                for action in self.thermal_actions[-10:]  # Last 10 actions
            ],
            "recommendations": self._get_thermal_recommendations(current_state, recent_states)
        }
    
    def _get_thermal_recommendations(self, 
                                   current_state: ThermalState,
                                   recent_states: List[ThermalState]) -> List[str]:
        """Get thermal management recommendations."""
        recommendations = []
        
        if current_state.thermal_pressure in ["high", "critical"]:
            recommendations.append("High thermal pressure - consider reducing training intensity")
        
        if current_state.throttling_active:
            recommendations.append("Thermal throttling active - improve cooling or reduce workload")
        
        throttling_frequency = sum(1 for s in recent_states if s.throttling_active) / len(recent_states)
        if throttling_frequency > 0.3:
            recommendations.append("Frequent thermal throttling - consider sustained workload reduction")
        
        if current_state.cpu_temperature and current_state.cpu_temperature > 80:
            recommendations.append("High CPU temperature - ensure adequate ventilation")
        
        if current_state.fan_speed and current_state.fan_speed > 5000:
            recommendations.append("High fan speed - system working hard to cool down")
        
        return recommendations