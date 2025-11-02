"""
MacBook optimization infrastructure for TRM training.

This package provides hardware detection, resource monitoring, and optimization
utilities specifically designed for training Tiny Recursive Models on MacBook hardware.
"""

from .hardware_detection import (
    CPUSpecs,
    MemorySpecs,
    PlatformCapabilities,
    HardwareDetector,
)
from .resource_monitoring import ResourceMonitor
from .config_management import MacBookConfigManager
from .memory_management import (
    MemoryConfig,
    MemoryManager,
    BatchSizeRecommendation,
    MemoryPressureInfo,
)
from .gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulator,
    AccumulationState,
    TrainingLoopIntegration,
)
from .data_loading import (
    DataLoadingConfig,
    DatasetInfo,
    MemoryMappedDataset,
    StreamingDataset,
    MemoryAwareDataLoader,
    DatasetAnalyzer,
    DataLoadingManager,
)
from .cpu_optimization import (
    CPUOptimizationConfig,
    CPUOptimizer,
    TensorOperationOptimizer,
)

__all__ = [
    "CPUSpecs",
    "MemorySpecs", 
    "PlatformCapabilities",
    "HardwareDetector",
    "ResourceMonitor",
    "MacBookConfigManager",
    "MemoryConfig",
    "MemoryManager",
    "BatchSizeRecommendation",
    "MemoryPressureInfo",
    "GradientAccumulationConfig",
    "GradientAccumulator",
    "AccumulationState",
    "TrainingLoopIntegration",
    "DataLoadingConfig",
    "DatasetInfo",
    "MemoryMappedDataset",
    "StreamingDataset",
    "MemoryAwareDataLoader",
    "DatasetAnalyzer",
    "DataLoadingManager",
    "CPUOptimizationConfig",
    "CPUOptimizer",
    "TensorOperationOptimizer",
]