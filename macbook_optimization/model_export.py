"""
Production Model Export and Serialization for Email Classification

This module provides comprehensive model export functionality for production deployment,
including model compression, optimization, metadata management, and version control.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import zipfile
import pickle

try:
    import torch
    import torch.nn as nn
    from torch.jit import ScriptModule
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    ScriptModule = None
    TORCH_AVAILABLE = False

from .email_training_config import EmailTrainingConfig
from models.recursive_reasoning.trm_email import EmailTRM, EmailTRMConfig
from models.email_tokenizer import EmailTokenizer


@dataclass
class ModelMetadata:
    """Comprehensive metadata for exported models."""
    # Basic info
    model_id: str
    model_name: str
    version: str
    export_timestamp: datetime
    
    # Model architecture
    model_type: str
    architecture_config: Dict[str, Any]
    parameter_count: int
    model_size_mb: float
    
    # Training info
    training_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    
    # Performance
    accuracy_metrics: Dict[str, float]
    inference_speed_ms: float
    memory_requirements_mb: float
    
    # Tokenizer info
    tokenizer_config: Dict[str, Any]
    vocab_size: int
    max_sequence_length: int
    
    # Deployment info
    target_platform: str
    optimization_level: str
    compression_applied: bool
    
    # Validation
    validation_results: Dict[str, Any]
    compatibility_info: Dict[str, Any]
    
    # Files
    model_files: List[str]
    checksum: str


@dataclass
class ExportConfig:
    """Configuration for model export process."""
    # Export settings
    export_format: str = "pytorch"  # "pytorch", "torchscript", "onnx"
    optimization_level: str = "standard"  # "minimal", "standard", "aggressive"
    target_platform: str = "cpu"  # "cpu", "gpu", "mobile"
    
    # Compression settings
    enable_compression: bool = True
    compression_method: str = "zip"  # "zip", "gzip", "lzma"
    quantization: bool = False
    pruning: bool = False
    
    # Validation settings
    validate_export: bool = True
    run_inference_test: bool = True
    performance_benchmark: bool = True
    
    # Output settings
    output_dir: str = "exported_models"
    include_tokenizer: bool = True
    include_config: bool = True
    include_metadata: bool = True
    
    # Version control
    auto_version: bool = True
    version_format: str = "semantic"  # "semantic", "timestamp", "incremental"


@dataclass
class ExportResult:
    """Result of model export operation."""
    success: bool
    model_id: str
    export_path: str
    export_time_seconds: float
    
    # Files created
    model_files: List[str]
    total_size_mb: float
    
    # Validation results
    validation_passed: bool
    inference_test_passed: bool
    performance_metrics: Dict[str, float]
    
    # Metadata
    metadata: Optional[ModelMetadata]
    
    # Issues
    warnings: List[str]
    errors: List[str]


class ModelExporter:
    """
    Production model exporter for email classification models.
    
    Handles model serialization, optimization, compression, and validation
    for production deployment with comprehensive metadata management.
    """
    
    def __init__(self, 
                 config: Optional[ExportConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize model exporter.
        
        Args:
            config: Export configuration
            logger: Logger instance
        """
        self.config = config or ExportConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export history
        self.export_history: List[ExportResult] = []
        
        self.logger.info(f"ModelExporter initialized with output dir: {self.output_dir}")
    
    def export_model(self,
                    model: EmailTRM,
                    tokenizer: EmailTokenizer,
                    training_config: EmailTrainingConfig,
                    training_metrics: Dict[str, float],
                    model_name: str = "EmailTRM",
                    version: Optional[str] = None) -> ExportResult:
        """
        Export trained model for production deployment.
        
        Args:
            model: Trained EmailTRM model
            tokenizer: Email tokenizer
            training_config: Training configuration used
            training_metrics: Final training metrics
            model_name: Name for the exported model
            version: Model version (auto-generated if None)
            
        Returns:
            ExportResult with export details
        """
        start_time = time.time()
        warnings = []
        errors = []
        
        # Generate model ID and version
        model_id = self._generate_model_id(model_name)
        if version is None:
            version = self._generate_version()
        
        self.logger.info(f"Starting model export: {model_id} v{version}")
        
        try:
            # Create export directory
            export_dir = self.output_dir / f"{model_id}_v{version}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare model for export
            model.eval()
            
            # Export model in specified format
            model_files = []
            
            if self.config.export_format == "pytorch":
                model_file = self._export_pytorch_model(model, export_dir, model_id)
                model_files.append(model_file)
            elif self.config.export_format == "torchscript":
                model_file = self._export_torchscript_model(model, export_dir, model_id)
                model_files.append(model_file)
            elif self.config.export_format == "onnx":
                model_file = self._export_onnx_model(model, export_dir, model_id)
                model_files.append(model_file)
            else:
                raise ValueError(f"Unsupported export format: {self.config.export_format}")
            
            # Export tokenizer
            if self.config.include_tokenizer:
                tokenizer_file = self._export_tokenizer(tokenizer, export_dir, model_id)
                model_files.append(tokenizer_file)
            
            # Export configuration
            if self.config.include_config:
                config_file = self._export_config(training_config, export_dir, model_id)
                model_files.append(config_file)
            
            # Apply optimizations
            if self.config.optimization_level != "minimal":
                optimized_files = self._apply_optimizations(model, export_dir, model_id)
                model_files.extend(optimized_files)
            
            # Generate metadata
            metadata = self._generate_metadata(
                model, tokenizer, training_config, training_metrics,
                model_id, version, model_files, export_dir
            )
            
            # Export metadata
            if self.config.include_metadata:
                metadata_file = self._export_metadata(metadata, export_dir, model_id)
                model_files.append(metadata_file)
            
            # Apply compression
            if self.config.enable_compression:
                compressed_file = self._compress_export(export_dir, model_id, version)
                if compressed_file:
                    model_files = [compressed_file]  # Replace with compressed version
            
            # Calculate total size
            total_size_mb = sum(
                os.path.getsize(export_dir / f) / (1024**2) 
                for f in model_files if (export_dir / f).exists()
            )
            
            # Validation
            validation_passed = True
            inference_test_passed = True
            performance_metrics = {}
            
            if self.config.validate_export:
                validation_result = self._validate_export(export_dir, model_files, metadata)
                validation_passed = validation_result["success"]
                if not validation_passed:
                    errors.extend(validation_result["errors"])
                warnings.extend(validation_result["warnings"])
            
            if self.config.run_inference_test:
                inference_result = self._test_inference(export_dir, model_files)
                inference_test_passed = inference_result["success"]
                if not inference_test_passed:
                    errors.extend(inference_result["errors"])
                warnings.extend(inference_result["warnings"])
            
            if self.config.performance_benchmark:
                perf_result = self._benchmark_performance(export_dir, model_files)
                performance_metrics = perf_result.get("metrics", {})
                warnings.extend(perf_result.get("warnings", []))
            
            # Create export result
            export_time = time.time() - start_time
            
            result = ExportResult(
                success=len(errors) == 0,
                model_id=model_id,
                export_path=str(export_dir),
                export_time_seconds=export_time,
                model_files=model_files,
                total_size_mb=total_size_mb,
                validation_passed=validation_passed,
                inference_test_passed=inference_test_passed,
                performance_metrics=performance_metrics,
                metadata=metadata,
                warnings=warnings,
                errors=errors
            )
            
            # Save export result
            self._save_export_result(result, export_dir)
            self.export_history.append(result)
            
            if result.success:
                self.logger.info(f"Model export completed successfully in {export_time:.2f}s")
                self.logger.info(f"Export path: {export_dir}")
                self.logger.info(f"Total size: {total_size_mb:.1f}MB")
            else:
                self.logger.error(f"Model export failed: {errors}")
            
            return result
            
        except Exception as e:
            errors.append(f"Export failed: {str(e)}")
            self.logger.error(f"Model export error: {e}")
            
            return ExportResult(
                success=False,
                model_id=model_id,
                export_path="",
                export_time_seconds=time.time() - start_time,
                model_files=[],
                total_size_mb=0.0,
                validation_passed=False,
                inference_test_passed=False,
                performance_metrics={},
                metadata=None,
                warnings=warnings,
                errors=errors
            )
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def _generate_version(self) -> str:
        """Generate version string based on configuration."""
        if self.config.version_format == "semantic":
            # Simple semantic versioning (would be more sophisticated in practice)
            return "1.0.0"
        elif self.config.version_format == "timestamp":
            return datetime.now().strftime("%Y%m%d.%H%M%S")
        elif self.config.version_format == "incremental":
            # Simple incremental versioning
            return str(len(self.export_history) + 1)
        else:
            return "1.0.0"
    
    def _export_pytorch_model(self, model: EmailTRM, export_dir: Path, model_id: str) -> str:
        """Export model in PyTorch format."""
        model_file = f"{model_id}_model.pt"
        model_path = export_dir / model_file
        
        # Prepare model state for export
        export_data = {
            'model_state_dict': model.state_dict(),
            'model_config': asdict(model.config),
            'model_class': 'EmailTRM',
            'export_timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else 'unknown'
        }
        
        if TORCH_AVAILABLE:
            torch.save(export_data, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(export_data, f)
        
        self.logger.info(f"Exported PyTorch model: {model_file}")
        return model_file
    
    def _export_torchscript_model(self, model: EmailTRM, export_dir: Path, model_id: str) -> str:
        """Export model in TorchScript format for optimized inference."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for TorchScript export")
        
        model_file = f"{model_id}_torchscript.pt"
        model_path = export_dir / model_file
        
        # Create example input for tracing
        batch_size = 1
        seq_len = model.config.max_sequence_length or 512
        example_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(model_path))
            
            self.logger.info(f"Exported TorchScript model: {model_file}")
            return model_file
            
        except Exception as e:
            # Fallback to scripting if tracing fails
            self.logger.warning(f"Tracing failed, trying scripting: {e}")
            try:
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(model_path))
                
                self.logger.info(f"Exported TorchScript model (scripted): {model_file}")
                return model_file
                
            except Exception as e2:
                raise RuntimeError(f"Both tracing and scripting failed: {e2}")
    
    def _export_onnx_model(self, model: EmailTRM, export_dir: Path, model_id: str) -> str:
        """Export model in ONNX format for cross-platform deployment."""
        try:
            import torch.onnx
        except ImportError:
            raise RuntimeError("ONNX export requires torch.onnx")
        
        model_file = f"{model_id}_model.onnx"
        model_path = export_dir / model_file
        
        # Create example input
        batch_size = 1
        seq_len = model.config.max_sequence_length or 512
        example_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            str(model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Exported ONNX model: {model_file}")
        return model_file
    
    def _export_tokenizer(self, tokenizer: EmailTokenizer, export_dir: Path, model_id: str) -> str:
        """Export tokenizer for production use."""
        tokenizer_file = f"{model_id}_tokenizer.pkl"
        tokenizer_path = export_dir / tokenizer_file
        
        tokenizer.save(str(tokenizer_path))
        
        self.logger.info(f"Exported tokenizer: {tokenizer_file}")
        return tokenizer_file
    
    def _export_config(self, config: EmailTrainingConfig, export_dir: Path, model_id: str) -> str:
        """Export training configuration."""
        config_file = f"{model_id}_config.json"
        config_path = export_dir / config_file
        
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"Exported config: {config_file}")
        return config_file
    
    def _apply_optimizations(self, model: EmailTRM, export_dir: Path, model_id: str) -> List[str]:
        """Apply model optimizations based on configuration."""
        optimized_files = []
        
        if self.config.optimization_level == "standard":
            # Standard optimizations
            if self.config.quantization and TORCH_AVAILABLE:
                quantized_file = self._apply_quantization(model, export_dir, model_id)
                if quantized_file:
                    optimized_files.append(quantized_file)
        
        elif self.config.optimization_level == "aggressive":
            # Aggressive optimizations
            if self.config.quantization and TORCH_AVAILABLE:
                quantized_file = self._apply_quantization(model, export_dir, model_id)
                if quantized_file:
                    optimized_files.append(quantized_file)
            
            if self.config.pruning and TORCH_AVAILABLE:
                pruned_file = self._apply_pruning(model, export_dir, model_id)
                if pruned_file:
                    optimized_files.append(pruned_file)
        
        return optimized_files
    
    def _apply_quantization(self, model: EmailTRM, export_dir: Path, model_id: str) -> Optional[str]:
        """Apply quantization to reduce model size."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            
            quantized_file = f"{model_id}_quantized.pt"
            quantized_path = export_dir / quantized_file
            
            torch.save({
                'model_state_dict': quantized_model.state_dict(),
                'model_config': asdict(model.config),
                'optimization': 'quantized'
            }, quantized_path)
            
            self.logger.info(f"Applied quantization: {quantized_file}")
            return quantized_file
            
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return None
    
    def _apply_pruning(self, model: EmailTRM, export_dir: Path, model_id: str) -> Optional[str]:
        """Apply pruning to reduce model parameters."""
        try:
            import torch.nn.utils.prune as prune
            
            # Simple magnitude-based pruning
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
            
            # Remove pruning reparameterization
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.remove(module, 'weight')
            
            pruned_file = f"{model_id}_pruned.pt"
            pruned_path = export_dir / pruned_file
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': asdict(model.config),
                'optimization': 'pruned'
            }, pruned_path)
            
            self.logger.info(f"Applied pruning: {pruned_file}")
            return pruned_file
            
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return None
    
    def _generate_metadata(self,
                          model: EmailTRM,
                          tokenizer: EmailTokenizer,
                          training_config: EmailTrainingConfig,
                          training_metrics: Dict[str, float],
                          model_id: str,
                          version: str,
                          model_files: List[str],
                          export_dir: Path) -> ModelMetadata:
        """Generate comprehensive model metadata."""
        
        # Calculate model parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Calculate model size
        model_size_mb = sum(
            os.path.getsize(export_dir / f) / (1024**2) 
            for f in model_files if (export_dir / f).exists()
        )
        
        # Generate checksum
        checksum = self._calculate_checksum(export_dir, model_files)
        
        return ModelMetadata(
            model_id=model_id,
            model_name="EmailTRM",
            version=version,
            export_timestamp=datetime.now(),
            model_type="EmailTRM",
            architecture_config=asdict(model.config),
            parameter_count=param_count,
            model_size_mb=model_size_mb,
            training_config=asdict(training_config),
            training_metrics=training_metrics,
            dataset_info={
                "vocab_size": tokenizer.vocab_size,
                "max_seq_len": tokenizer.max_seq_len,
                "categories": training_config.num_email_categories
            },
            accuracy_metrics=training_metrics,
            inference_speed_ms=0.0,  # Will be filled by benchmarking
            memory_requirements_mb=0.0,  # Will be filled by benchmarking
            tokenizer_config={
                "vocab_size": tokenizer.vocab_size,
                "max_seq_len": tokenizer.max_seq_len,
                "special_tokens": tokenizer.SPECIAL_TOKENS
            },
            vocab_size=tokenizer.vocab_size,
            max_sequence_length=tokenizer.max_seq_len,
            target_platform=self.config.target_platform,
            optimization_level=self.config.optimization_level,
            compression_applied=self.config.enable_compression,
            validation_results={},  # Will be filled by validation
            compatibility_info={
                "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "unknown",
                "export_format": self.config.export_format,
                "python_version": "3.8+"  # Minimum requirement
            },
            model_files=model_files,
            checksum=checksum
        )
    
    def _calculate_checksum(self, export_dir: Path, model_files: List[str]) -> str:
        """Calculate checksum for model files."""
        hasher = hashlib.sha256()
        
        for file_name in sorted(model_files):
            file_path = export_dir / file_name
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _export_metadata(self, metadata: ModelMetadata, export_dir: Path, model_id: str) -> str:
        """Export model metadata."""
        metadata_file = f"{model_id}_metadata.json"
        metadata_path = export_dir / metadata_file
        
        metadata_dict = asdict(metadata)
        metadata_dict['export_timestamp'] = metadata.export_timestamp.isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        self.logger.info(f"Exported metadata: {metadata_file}")
        return metadata_file
    
    def _compress_export(self, export_dir: Path, model_id: str, version: str) -> Optional[str]:
        """Compress exported model files."""
        try:
            compressed_file = f"{model_id}_v{version}.zip"
            compressed_path = export_dir.parent / compressed_file
            
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(export_dir)
                        zipf.write(file_path, arcname)
            
            self.logger.info(f"Created compressed export: {compressed_file}")
            return compressed_file
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return None
    
    def _validate_export(self, export_dir: Path, model_files: List[str], metadata: ModelMetadata) -> Dict[str, Any]:
        """Validate exported model files."""
        validation_result = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        # Check file existence
        for file_name in model_files:
            file_path = export_dir / file_name
            if not file_path.exists():
                validation_result["errors"].append(f"Missing file: {file_name}")
                validation_result["success"] = False
        
        # Validate model file
        model_files_found = [f for f in model_files if f.endswith(('.pt', '.onnx'))]
        if not model_files_found:
            validation_result["errors"].append("No model file found")
            validation_result["success"] = False
        
        # Validate tokenizer
        tokenizer_files = [f for f in model_files if 'tokenizer' in f]
        if self.config.include_tokenizer and not tokenizer_files:
            validation_result["warnings"].append("Tokenizer not found but was requested")
        
        # Validate metadata
        if metadata.parameter_count <= 0:
            validation_result["warnings"].append("Invalid parameter count in metadata")
        
        return validation_result
    
    def _test_inference(self, export_dir: Path, model_files: List[str]) -> Dict[str, Any]:
        """Test inference on exported model."""
        inference_result = {
            "success": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Find model file
            model_file = None
            for f in model_files:
                if f.endswith('.pt') and 'model' in f:
                    model_file = f
                    break
            
            if not model_file:
                inference_result["errors"].append("No suitable model file for inference test")
                inference_result["success"] = False
                return inference_result
            
            # Load and test model (simplified test)
            model_path = export_dir / model_file
            
            if TORCH_AVAILABLE:
                try:
                    model_data = torch.load(model_path, map_location='cpu')
                    
                    # Basic validation that model data is loadable
                    if 'model_state_dict' not in model_data:
                        inference_result["errors"].append("Invalid model file format")
                        inference_result["success"] = False
                    
                except Exception as e:
                    inference_result["errors"].append(f"Failed to load model: {e}")
                    inference_result["success"] = False
            
        except Exception as e:
            inference_result["errors"].append(f"Inference test failed: {e}")
            inference_result["success"] = False
        
        return inference_result
    
    def _benchmark_performance(self, export_dir: Path, model_files: List[str]) -> Dict[str, Any]:
        """Benchmark model performance."""
        benchmark_result = {
            "metrics": {},
            "warnings": []
        }
        
        try:
            # Simple performance metrics (would be more comprehensive in practice)
            model_file = None
            for f in model_files:
                if f.endswith('.pt') and 'model' in f:
                    model_file = f
                    break
            
            if model_file:
                model_path = export_dir / model_file
                file_size_mb = os.path.getsize(model_path) / (1024**2)
                
                benchmark_result["metrics"] = {
                    "model_size_mb": file_size_mb,
                    "estimated_inference_ms": file_size_mb * 10,  # Rough estimate
                    "estimated_memory_mb": file_size_mb * 2  # Rough estimate
                }
        
        except Exception as e:
            benchmark_result["warnings"].append(f"Performance benchmarking failed: {e}")
        
        return benchmark_result
    
    def _save_export_result(self, result: ExportResult, export_dir: Path):
        """Save export result for tracking."""
        result_file = export_dir / "export_result.json"
        
        try:
            result_dict = asdict(result)
            if result.metadata:
                result_dict['metadata']['export_timestamp'] = result.metadata.export_timestamp.isoformat()
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.warning(f"Failed to save export result: {e}")
    
    def list_exports(self) -> List[Dict[str, Any]]:
        """List all exported models."""
        exports = []
        
        for export_result in self.export_history:
            export_info = {
                "model_id": export_result.model_id,
                "export_path": export_result.export_path,
                "success": export_result.success,
                "size_mb": export_result.total_size_mb,
                "validation_passed": export_result.validation_passed
            }
            
            if export_result.metadata:
                export_info.update({
                    "version": export_result.metadata.version,
                    "export_timestamp": export_result.metadata.export_timestamp.isoformat(),
                    "parameter_count": export_result.metadata.parameter_count
                })
            
            exports.append(export_info)
        
        return exports
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of all exports."""
        successful_exports = [e for e in self.export_history if e.success]
        
        return {
            "total_exports": len(self.export_history),
            "successful_exports": len(successful_exports),
            "failed_exports": len(self.export_history) - len(successful_exports),
            "total_size_mb": sum(e.total_size_mb for e in successful_exports),
            "average_export_time": sum(e.export_time_seconds for e in successful_exports) / len(successful_exports) if successful_exports else 0,
            "export_formats": list(set([self.config.export_format])),  # Would track multiple formats
            "recent_exports": [
                {
                    "model_id": e.model_id,
                    "success": e.success,
                    "size_mb": e.total_size_mb
                }
                for e in self.export_history[-5:]  # Last 5 exports
            ]
        }