"""
Hardware detection module for MacBook optimization.

This module provides utilities to detect MacBook hardware specifications
including CPU, memory, and platform capabilities for TRM training optimization.
"""

import platform
import psutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional
import cpuinfo


@dataclass
class CPUSpecs:
    """CPU specifications for MacBook hardware."""
    cores: int
    threads: int
    architecture: str
    features: List[str]
    base_frequency: float  # GHz
    max_frequency: float   # GHz
    brand: str
    model: str


@dataclass
class MemorySpecs:
    """Memory specifications for MacBook hardware."""
    total_memory: int      # bytes
    available_memory: int  # bytes
    memory_type: str       # e.g., "LPDDR3", "LPDDR4"
    memory_speed: Optional[int]  # MHz


@dataclass
class PlatformCapabilities:
    """Platform-specific capabilities for optimization."""
    has_mkl: bool
    has_accelerate: bool   # macOS Accelerate framework
    torch_version: str
    python_version: str
    macos_version: str
    optimal_dtype: str     # "float32" or "float16"
    supports_avx: bool
    supports_avx2: bool


class HardwareDetector:
    """Hardware detection utilities for MacBook optimization."""
    
    def __init__(self):
        self._cpu_info = None
        self._memory_info = None
        
    def detect_cpu_specs(self) -> CPUSpecs:
        """Detect CPU specifications."""
        if self._cpu_info is None:
            self._cpu_info = cpuinfo.get_cpu_info()
            
        # Get basic CPU information
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        architecture = platform.machine()
        
        # Get CPU features
        features = []
        if 'flags' in self._cpu_info:
            cpu_flags = self._cpu_info['flags']
            # Check for common Intel features
            intel_features = ['avx', 'avx2', 'sse4_1', 'sse4_2', 'fma']
            features = [f for f in intel_features if f in cpu_flags]
        
        # Get frequency information
        base_freq = 0.0
        max_freq = 0.0
        
        try:
            # Try to get frequency from cpuinfo
            if 'hz_advertised_friendly' in self._cpu_info:
                freq_str = self._cpu_info['hz_advertised_friendly']
                # Parse frequency string like "2.4000 GHz"
                if 'GHz' in freq_str:
                    base_freq = float(freq_str.split()[0])
            
            # Try psutil for current frequency
            freq_info = psutil.cpu_freq()
            if freq_info:
                if base_freq == 0.0:
                    base_freq = freq_info.current / 1000.0  # Convert MHz to GHz
                max_freq = freq_info.max / 1000.0 if freq_info.max else base_freq
        except Exception:
            pass
            
        # Use fallback values if no frequency info found
        if base_freq == 0.0:
            base_freq = 2.4  # Common base frequency
        if max_freq == 0.0:
            max_freq = 3.8   # Common turbo frequency
            
        brand = self._cpu_info.get('brand_raw', 'Unknown')
        model = self._cpu_info.get('brand_raw', 'Unknown')
        
        return CPUSpecs(
            cores=cores,
            threads=threads,
            architecture=architecture,
            features=features,
            base_frequency=base_freq,
            max_frequency=max_freq,
            brand=brand,
            model=model
        )
    
    def detect_memory_specs(self) -> MemorySpecs:
        """Detect memory specifications."""
        memory = psutil.virtual_memory()
        
        # Try to detect memory type (LPDDR3/LPDDR4) on macOS
        memory_type = "Unknown"
        memory_speed = None
        
        if platform.system() == "Darwin":
            try:
                # Use system_profiler to get detailed memory info
                result = subprocess.run(
                    ["system_profiler", "SPMemoryDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    output = result.stdout
                    if "LPDDR4" in output:
                        memory_type = "LPDDR4"
                    elif "LPDDR3" in output:
                        memory_type = "LPDDR3"
                    elif "DDR4" in output:
                        memory_type = "DDR4"
                    elif "DDR3" in output:
                        memory_type = "DDR3"
                        
                    # Try to extract speed
                    lines = output.split('\n')
                    for line in lines:
                        if "Speed:" in line and "MHz" in line:
                            try:
                                speed_str = line.split("Speed:")[1].strip()
                                memory_speed = int(speed_str.split()[0])
                                break
                            except (ValueError, IndexError):
                                pass
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, Exception):
                pass
        
        return MemorySpecs(
            total_memory=memory.total,
            available_memory=memory.available,
            memory_type=memory_type,
            memory_speed=memory_speed
        )
    
    def detect_platform_capabilities(self) -> PlatformCapabilities:
        """Detect platform-specific capabilities."""
        # Check for Intel MKL
        has_mkl = False
        try:
            import torch
            torch_version = torch.__version__
            # Check if PyTorch was built with MKL
            has_mkl = torch.backends.mkl.is_available()
        except ImportError:
            torch_version = "Not installed"
        
        # Check for macOS Accelerate framework
        has_accelerate = platform.system() == "Darwin"
        
        # Get system versions
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        macos_version = platform.mac_ver()[0] if platform.system() == "Darwin" else "N/A"
        
        # Determine optimal dtype (float32 for CPU training)
        optimal_dtype = "float32"
        
        # Check AVX support
        cpu_specs = self.detect_cpu_specs()
        supports_avx = "avx" in cpu_specs.features
        supports_avx2 = "avx2" in cpu_specs.features
        
        return PlatformCapabilities(
            has_mkl=has_mkl,
            has_accelerate=has_accelerate,
            torch_version=torch_version,
            python_version=python_version,
            macos_version=macos_version,
            optimal_dtype=optimal_dtype,
            supports_avx=supports_avx,
            supports_avx2=supports_avx2
        )
    
    def get_optimal_worker_count(self) -> int:
        """Calculate optimal number of workers for data loading."""
        cpu_specs = self.detect_cpu_specs()
        memory_specs = self.detect_memory_specs()
        
        # Base worker count on CPU cores
        base_workers = cpu_specs.cores
        
        # Adjust based on memory constraints
        # Each worker needs roughly 100MB of memory for data loading
        memory_gb = memory_specs.available_memory / (1024**3)
        max_workers_by_memory = max(1, int(memory_gb * 10))  # 10 workers per GB
        
        # Conservative approach: use fewer workers on memory-constrained systems
        optimal_workers = min(base_workers, max_workers_by_memory, 4)
        
        return max(1, optimal_workers)
    
    def get_hardware_summary(self) -> dict:
        """Get a comprehensive hardware summary."""
        cpu_specs = self.detect_cpu_specs()
        memory_specs = self.detect_memory_specs()
        platform_caps = self.detect_platform_capabilities()
        
        return {
            "cpu": {
                "cores": cpu_specs.cores,
                "threads": cpu_specs.threads,
                "brand": cpu_specs.brand,
                "base_frequency_ghz": cpu_specs.base_frequency,
                "max_frequency_ghz": cpu_specs.max_frequency,
                "features": cpu_specs.features,
            },
            "memory": {
                "total_gb": round(memory_specs.total_memory / (1024**3), 2),
                "available_gb": round(memory_specs.available_memory / (1024**3), 2),
                "type": memory_specs.memory_type,
                "speed_mhz": memory_specs.memory_speed,
            },
            "platform": {
                "os": platform.system(),
                "os_version": platform_caps.macos_version,
                "python_version": platform_caps.python_version,
                "torch_version": platform_caps.torch_version,
                "has_mkl": platform_caps.has_mkl,
                "has_accelerate": platform_caps.has_accelerate,
                "supports_avx": platform_caps.supports_avx,
                "supports_avx2": platform_caps.supports_avx2,
            },
            "optimization": {
                "optimal_workers": self.get_optimal_worker_count(),
                "optimal_dtype": platform_caps.optimal_dtype,
            }
        }