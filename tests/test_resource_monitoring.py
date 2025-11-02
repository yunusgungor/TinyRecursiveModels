"""
Unit tests for resource monitoring module.

Tests real-time memory usage monitoring, CPU utilization tracking,
and thermal monitoring functionality.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from macbook_optimization.resource_monitoring import (
    MemoryStats,
    CPUStats,
    ThermalStats,
    ResourceSnapshot,
    ResourceMonitor
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""
    
    def test_memory_stats_creation(self):
        """Test MemoryStats dataclass creation."""
        stats = MemoryStats(
            total_mb=8192.0,
            available_mb=4096.0,
            used_mb=4096.0,
            percent_used=50.0,
            swap_used_mb=0.0,
            swap_percent=0.0
        )
        
        assert stats.total_mb == 8192.0
        assert stats.available_mb == 4096.0
        assert stats.percent_used == 50.0


class TestCPUStats:
    """Test CPUStats dataclass."""
    
    def test_cpu_stats_creation(self):
        """Test CPUStats dataclass creation."""
        stats = CPUStats(
            percent_total=25.5,
            percent_per_core=[20.0, 30.0, 25.0, 27.0],
            frequency_current=2400.0,
            frequency_max=3800.0,
            load_average=[1.2, 1.5, 1.8]
        )
        
        assert stats.percent_total == 25.5
        assert len(stats.percent_per_core) == 4
        assert stats.frequency_current == 2400.0


class TestThermalStats:
    """Test ThermalStats dataclass."""
    
    def test_thermal_stats_creation(self):
        """Test ThermalStats dataclass creation."""
        stats = ThermalStats(
            cpu_temperature=65.0,
            fan_speed=2000,
            thermal_state="warm"
        )
        
        assert stats.cpu_temperature == 65.0
        assert stats.fan_speed == 2000
        assert stats.thermal_state == "warm"


class TestResourceMonitor:
    """Test ResourceMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor(history_size=10)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_get_memory_stats(self, mock_swap, mock_virtual):
        """Test memory statistics collection."""
        # Mock psutil return values
        mock_virtual.return_value = Mock(
            total=8589934592,  # 8GB
            available=4294967296,  # 4GB
            used=4294967296,  # 4GB
            percent=50.0
        )
        mock_swap.return_value = Mock(
            used=0,
            percent=0.0
        )
        
        stats = self.monitor.get_memory_stats()
        
        assert stats.total_mb == 8192.0
        assert stats.available_mb == 4096.0
        assert stats.used_mb == 4096.0
        assert stats.percent_used == 50.0
        assert stats.swap_used_mb == 0.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_freq')
    @patch('psutil.getloadavg')
    def test_get_cpu_stats(self, mock_loadavg, mock_freq, mock_cpu_percent):
        """Test CPU statistics collection."""
        # Mock psutil return values
        mock_cpu_percent.side_effect = [25.5, [20.0, 30.0, 25.0, 27.0]]
        mock_freq.return_value = Mock(current=2400.0, max=3800.0)
        mock_loadavg.return_value = [1.2, 1.5, 1.8]
        
        stats = self.monitor.get_cpu_stats()
        
        assert stats.percent_total == 25.5
        assert len(stats.percent_per_core) == 4
        assert stats.frequency_current == 2400.0
        assert stats.frequency_max == 3800.0
        assert stats.load_average == [1.2, 1.5, 1.8]
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_get_thermal_stats_macos(self, mock_subprocess, mock_system):
        """Test thermal statistics collection on macOS."""
        mock_system.return_value = "Darwin"
        
        # Mock subprocess for system_profiler
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "thermal state: normal"
        mock_subprocess.return_value = mock_result
        
        # Mock CPU stats for thermal estimation
        with patch.object(self.monitor, 'get_cpu_stats') as mock_cpu_stats:
            mock_cpu_stats.return_value = CPUStats(
                percent_total=30.0,
                percent_per_core=[25.0, 35.0, 30.0, 30.0],
                frequency_current=2400.0,
                frequency_max=3800.0,
                load_average=[1.0, 1.0, 1.0]
            )
            
            stats = self.monitor.get_thermal_stats()
            
            assert stats.thermal_state in ["normal", "warm", "hot"]
    
    @patch('platform.system')
    def test_get_thermal_stats_non_macos(self, mock_system):
        """Test thermal statistics collection on non-macOS systems."""
        mock_system.return_value = "Linux"
        
        # Mock CPU stats for thermal estimation
        with patch.object(self.monitor, 'get_cpu_stats') as mock_cpu_stats:
            mock_cpu_stats.return_value = CPUStats(
                percent_total=85.0,  # High CPU usage
                percent_per_core=[80.0, 90.0, 85.0, 85.0],
                frequency_current=2400.0,
                frequency_max=3800.0,
                load_average=[2.0, 2.0, 2.0]
            )
            
            stats = self.monitor.get_thermal_stats()
            
            # Should estimate "hot" based on high CPU usage
            assert stats.thermal_state == "hot"
            assert stats.cpu_temperature is None
            assert stats.fan_speed is None
    
    def test_get_current_snapshot(self):
        """Test resource snapshot creation."""
        with patch.object(self.monitor, 'get_memory_stats') as mock_memory, \
             patch.object(self.monitor, 'get_cpu_stats') as mock_cpu, \
             patch.object(self.monitor, 'get_thermal_stats') as mock_thermal:
            
            mock_memory.return_value = MemoryStats(8192, 4096, 4096, 50.0, 0, 0)
            mock_cpu.return_value = CPUStats(25.0, [25.0], 2400.0, 3800.0, [1.0])
            mock_thermal.return_value = ThermalStats(None, None, "normal")
            
            snapshot = self.monitor.get_current_snapshot()
            
            assert isinstance(snapshot, ResourceSnapshot)
            assert snapshot.memory.total_mb == 8192
            assert snapshot.cpu.percent_total == 25.0
            assert snapshot.thermal.thermal_state == "normal"
            assert snapshot.timestamp > 0
    
    def test_callback_system(self):
        """Test callback registration and execution."""
        callback_called = []
        
        def test_callback(snapshot):
            callback_called.append(snapshot)
        
        # Add callback
        self.monitor.add_callback(test_callback)
        
        # Mock snapshot creation
        with patch.object(self.monitor, 'get_current_snapshot') as mock_snapshot:
            mock_snapshot.return_value = ResourceSnapshot(
                timestamp=time.time(),
                memory=MemoryStats(8192, 4096, 4096, 50.0, 0, 0),
                cpu=CPUStats(25.0, [25.0], 2400.0, 3800.0, [1.0]),
                thermal=ThermalStats(None, None, "normal")
            )
            
            # Start monitoring briefly
            self.monitor.start_monitoring(interval=0.1)
            time.sleep(0.2)  # Let it run briefly
            self.monitor.stop_monitoring()
        
        # Check callback was called
        assert len(callback_called) > 0
        
        # Remove callback
        self.monitor.remove_callback(test_callback)
        assert test_callback not in self.monitor.callbacks
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        assert not self.monitor.monitoring
        
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        assert self.monitor.monitoring
        assert self.monitor.monitor_thread is not None
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        assert not self.monitor.monitoring
    
    def test_history_management(self):
        """Test history storage and retrieval."""
        # Create mock snapshots
        snapshots = []
        for i in range(5):
            snapshot = ResourceSnapshot(
                timestamp=time.time() + i,
                memory=MemoryStats(8192, 4096, 4096, 50.0 + i, 0, 0),
                cpu=CPUStats(25.0 + i, [25.0], 2400.0, 3800.0, [1.0]),
                thermal=ThermalStats(None, None, "normal")
            )
            snapshots.append(snapshot)
            self.monitor.history.append(snapshot)
        
        # Test get_history
        all_history = self.monitor.get_history()
        assert len(all_history) == 5
        
        last_3 = self.monitor.get_history(last_n=3)
        assert len(last_3) == 3
        assert last_3[-1].memory.percent_used == 54.0  # Last snapshot
    
    def test_average_stats_calculation(self):
        """Test average statistics calculation."""
        # Add mock snapshots to history
        for i in range(3):
            snapshot = ResourceSnapshot(
                timestamp=time.time() + i,
                memory=MemoryStats(8192, 4096, 4096, 50.0 + i * 10, 0, 0),
                cpu=CPUStats(20.0 + i * 10, [25.0], 2400.0, 3800.0, [1.0]),
                thermal=ThermalStats(None, None, "normal" if i < 2 else "warm")
            )
            self.monitor.history.append(snapshot)
        
        avg_stats = self.monitor.get_average_stats()
        
        assert avg_stats["average_memory_percent"] == 60.0  # (50+60+70)/3
        assert avg_stats["average_cpu_percent"] == 30.0     # (20+30+40)/3
        assert avg_stats["sample_count"] == 3
        assert "thermal_state_distribution" in avg_stats
    
    def test_memory_pressure_check(self):
        """Test memory pressure detection."""
        with patch.object(self.monitor, 'get_memory_stats') as mock_memory:
            # Test normal memory usage
            mock_memory.return_value = MemoryStats(8192, 4096, 4096, 50.0, 0, 0)
            assert not self.monitor.check_memory_pressure(threshold=80.0)
            
            # Test high memory usage
            mock_memory.return_value = MemoryStats(8192, 1024, 7168, 87.5, 0, 0)
            assert self.monitor.check_memory_pressure(threshold=80.0)
    
    def test_thermal_throttling_check(self):
        """Test thermal throttling detection."""
        with patch.object(self.monitor, 'get_thermal_stats') as mock_thermal:
            # Test normal thermal state
            mock_thermal.return_value = ThermalStats(None, None, "normal")
            assert not self.monitor.check_thermal_throttling()
            
            # Test hot thermal state
            mock_thermal.return_value = ThermalStats(85.0, 3000, "hot")
            assert self.monitor.check_thermal_throttling()
    
    def test_resource_summary(self):
        """Test comprehensive resource summary."""
        with patch.object(self.monitor, 'get_current_snapshot') as mock_snapshot, \
             patch.object(self.monitor, 'get_average_stats') as mock_averages, \
             patch.object(self.monitor, 'check_memory_pressure') as mock_memory_pressure, \
             patch.object(self.monitor, 'check_thermal_throttling') as mock_thermal_throttling:
            
            mock_snapshot.return_value = ResourceSnapshot(
                timestamp=time.time(),
                memory=MemoryStats(8192, 4096, 4096, 50.0, 0, 0),
                cpu=CPUStats(25.0, [25.0], 2400.0, 3800.0, [1.0]),
                thermal=ThermalStats(None, None, "normal")
            )
            mock_averages.return_value = {"average_memory_percent": 45.0}
            mock_memory_pressure.return_value = False
            mock_thermal_throttling.return_value = False
            
            summary = self.monitor.get_resource_summary()
            
            assert "current" in summary
            assert "recent_averages" in summary
            assert "alerts" in summary
            assert summary["current"]["memory_used_percent"] == 50.0
            assert summary["alerts"]["memory_pressure"] is False
    
    def test_history_size_limit(self):
        """Test that history respects size limit."""
        monitor = ResourceMonitor(history_size=3)
        
        # Add more snapshots than the limit
        for i in range(5):
            snapshot = ResourceSnapshot(
                timestamp=time.time() + i,
                memory=MemoryStats(8192, 4096, 4096, 50.0, 0, 0),
                cpu=CPUStats(25.0, [25.0], 2400.0, 3800.0, [1.0]),
                thermal=ThermalStats(None, None, "normal")
            )
            monitor.history.append(snapshot)
        
        # Should only keep the last 3
        assert len(monitor.history) == 3