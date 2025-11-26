#!/usr/bin/env python3
"""
Build Optimization and Layer-Specific Cache Invalidation

This module provides utilities for detecting which Docker layers need to be rebuilt
based on file changes, ensuring optimal cache utilization.

Usage:
    python scripts/build_optimizer.py analyze <service>
    python scripts/build_optimizer.py detect-changes <service>
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class LayerType:
    """Docker layer types for cache invalidation analysis."""
    BASE = "base"
    SYSTEM_DEPS = "system_dependencies"
    APP_DEPS = "application_dependencies"
    CODE = "application_code"
    CONFIG = "configuration"


class BuildOptimizer:
    """Analyzes Dockerfile and detects which layers need rebuilding."""
    
    def __init__(self, service_dir: Path):
        self.service_dir = service_dir
        self.dockerfile_path = service_dir / "Dockerfile"
        self.cache_file = service_dir / ".build_cache.json"
        
    def parse_dockerfile(self) -> List[Tuple[int, str, LayerType]]:
        """
        Parse Dockerfile and categorize layers.
        
        Returns:
            List of (line_number, instruction, layer_type) tuples
        """
        if not self.dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {self.dockerfile_path}")
        
        with open(self.dockerfile_path, 'r') as f:
            content = f.read()
        
        layers = []
        lines = content.split('\n')
        
        for idx, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Categorize layer type
            layer_type = self._categorize_layer(line, lines, idx)
            
            if layer_type:
                layers.append((idx, line, layer_type))
        
        return layers
    
    def _categorize_layer(self, line: str, all_lines: List[str], idx: int) -> Optional[LayerType]:
        """Categorize a Dockerfile instruction into a layer type."""
        
        # Base image layers
        if line.startswith('FROM'):
            return LayerType.BASE
        
        # System dependencies (apt-get, apk, yum, etc.)
        if 'apt-get' in line or 'apk add' in line or 'yum install' in line:
            return LayerType.SYSTEM_DEPS
        
        # Application dependencies
        if 'COPY requirements' in line or 'COPY package.json' in line:
            return LayerType.APP_DEPS
        
        if 'pip install' in line or 'npm install' in line or 'npm ci' in line:
            return LayerType.APP_DEPS
        
        # Application code
        if 'COPY . .' in line:
            return LayerType.CODE
        
        # Configuration
        if 'ENV' in line or 'ARG' in line:
            return LayerType.CONFIG
        
        # Other instructions (RUN, WORKDIR, etc.)
        if any(line.startswith(cmd) for cmd in ['RUN', 'WORKDIR', 'USER', 'EXPOSE', 'CMD', 'ENTRYPOINT']):
            # Try to infer from context
            if idx > 0:
                prev_line = all_lines[idx - 1].strip()
                if 'requirements' in prev_line or 'package.json' in prev_line:
                    return LayerType.APP_DEPS
            
            return None
        
        return None
    
    def get_dependency_files(self, service_name: str) -> List[Path]:
        """Get list of dependency files for a service."""
        dependency_files = []
        
        if service_name == "backend":
            dependency_files = [
                self.service_dir / "requirements.txt",
                self.service_dir / "requirements-dev.txt",
            ]
        elif service_name == "frontend":
            dependency_files = [
                self.service_dir / "package.json",
                self.service_dir / "package-lock.json",
            ]
        
        return [f for f in dependency_files if f.exists()]
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        if not file_path.exists():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}", file=sys.stderr)
            return ""
    
    def load_cache(self) -> Dict:
        """Load previous build cache."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}", file=sys.stderr)
            return {}
    
    def save_cache(self, cache_data: Dict):
        """Save build cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}", file=sys.stderr)
    
    def detect_changed_layers(self, service_name: str) -> Set[LayerType]:
        """
        Detect which layer types need to be rebuilt based on file changes.
        
        Returns:
            Set of LayerType values that need rebuilding
        """
        changed_layers = set()
        
        # Load previous cache
        cache = self.load_cache()
        previous_hashes = cache.get('file_hashes', {})
        
        # Check dependency files
        dependency_files = self.get_dependency_files(service_name)
        current_dep_hashes = {}
        
        for dep_file in dependency_files:
            file_key = str(dep_file.relative_to(self.service_dir))
            current_hash = self.calculate_file_hash(dep_file)
            current_dep_hashes[file_key] = current_hash
            
            previous_hash = previous_hashes.get(file_key, "")
            
            if current_hash != previous_hash:
                print(f"Dependency file changed: {file_key}")
                changed_layers.add(LayerType.APP_DEPS)
        
        # Check code files (simplified - check if any .py or .ts/.tsx files changed)
        code_patterns = []
        if service_name == "backend":
            code_patterns = ["**/*.py"]
        elif service_name == "frontend":
            code_patterns = ["**/*.ts", "**/*.tsx", "**/*.jsx", "**/*.js", "**/*.css"]
        
        code_changed = False
        for pattern in code_patterns:
            code_files = list(self.service_dir.glob(pattern))
            
            # Sample check (for performance, don't hash all files)
            # In production, use git diff or similar
            for code_file in code_files[:10]:  # Check first 10 files
                if code_file.is_file():
                    file_key = str(code_file.relative_to(self.service_dir))
                    
                    # Skip node_modules, __pycache__, etc.
                    if any(skip in file_key for skip in ['node_modules', '__pycache__', '.pytest_cache', 'dist', 'build']):
                        continue
                    
                    current_hash = self.calculate_file_hash(code_file)
                    previous_hash = previous_hashes.get(file_key, "")
                    
                    if current_hash != previous_hash:
                        code_changed = True
                        break
            
            if code_changed:
                break
        
        if code_changed:
            print("Code files changed")
            changed_layers.add(LayerType.CODE)
        
        # Update cache
        new_cache = {
            'file_hashes': {**previous_hashes, **current_dep_hashes},
            'last_check': str(Path.cwd())
        }
        self.save_cache(new_cache)
        
        return changed_layers
    
    def verify_layer_ordering(self) -> bool:
        """
        Verify that Dockerfile layers are ordered for optimal caching.
        
        Returns:
            True if ordering is correct, False otherwise
        """
        layers = self.parse_dockerfile()
        
        # Expected order: BASE → SYSTEM_DEPS → APP_DEPS → CODE
        layer_order = {
            LayerType.BASE: 0,
            LayerType.SYSTEM_DEPS: 1,
            LayerType.APP_DEPS: 2,
            LayerType.CODE: 3,
        }
        
        prev_order = -1
        
        for line_num, instruction, layer_type in layers:
            if layer_type in layer_order:
                current_order = layer_order[layer_type]
                
                if current_order < prev_order:
                    print(f"❌ Layer ordering violation at line {line_num}: {instruction}")
                    print(f"   {layer_type} should come before previous layer")
                    return False
                
                prev_order = current_order
        
        print("✅ Layer ordering is optimal for caching")
        return True
    
    def analyze_cache_efficiency(self, service_name: str):
        """Analyze and report cache efficiency."""
        print(f"\n=== Build Cache Analysis for {service_name} ===\n")
        
        # Verify layer ordering
        print("1. Layer Ordering Check:")
        self.verify_layer_ordering()
        
        # Detect changed layers
        print("\n2. Changed Layers Detection:")
        changed_layers = self.detect_changed_layers(service_name)
        
        if not changed_layers:
            print("✅ No changes detected - full cache reuse possible")
        else:
            print(f"⚠️  Changed layers: {', '.join(changed_layers)}")
            
            if LayerType.APP_DEPS in changed_layers:
                print("   → Dependency layer will be rebuilt")
                print("   → Code layer will be rebuilt (depends on dependencies)")
            
            if LayerType.CODE in changed_layers and LayerType.APP_DEPS not in changed_layers:
                print("   → Only code layer will be rebuilt")
                print("   → Dependency cache will be reused ✅")
        
        # Parse and display layer structure
        print("\n3. Dockerfile Layer Structure:")
        layers = self.parse_dockerfile()
        
        for line_num, instruction, layer_type in layers:
            if layer_type:
                print(f"   Line {line_num:3d}: [{layer_type:20s}] {instruction[:60]}")
        
        print("\n" + "=" * 50 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_optimizer.py {analyze|detect-changes} <service>")
        print("")
        print("Commands:")
        print("  analyze <service>         Analyze build cache efficiency")
        print("  detect-changes <service>  Detect which layers need rebuilding")
        print("")
        print("Services: backend, frontend")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command in ["analyze", "detect-changes"]:
        if len(sys.argv) < 3:
            print(f"Error: {command} requires a service name")
            print("Usage: python scripts/build_optimizer.py {command} <service>")
            sys.exit(1)
        
        service = sys.argv[2]
        service_dir = Path(service)
        
        if not service_dir.exists():
            print(f"Error: Service directory not found: {service_dir}")
            sys.exit(1)
        
        optimizer = BuildOptimizer(service_dir)
        
        if command == "analyze":
            optimizer.analyze_cache_efficiency(service)
        elif command == "detect-changes":
            changed_layers = optimizer.detect_changed_layers(service)
            
            if not changed_layers:
                print("No changes detected")
            else:
                print(f"Changed layers: {', '.join(changed_layers)}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
