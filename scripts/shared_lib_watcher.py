#!/usr/bin/env python3
"""
Shared Library Change Detection and Rebuild Trigger

This module monitors shared libraries and triggers rebuilds of dependent services
when changes are detected.

Usage:
    python scripts/shared_lib_watcher.py watch
    python scripts/shared_lib_watcher.py detect <file>
    python scripts/shared_lib_watcher.py rebuild <library>
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Optional


class SharedLibraryWatcher:
    """Monitors shared libraries and triggers dependent service rebuilds."""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path.cwd()
        self.cache_file = self.root_dir / ".shared_lib_cache.json"
        self.lib_service_map: Dict[str, Set[str]] = {}
        self.file_hashes: Dict[str, str] = {}
        
    def load_cache(self) -> Dict:
        """Load previous file hashes from cache."""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}", file=sys.stderr)
            return {}
    
    def save_cache(self):
        """Save current file hashes to cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}", file=sys.stderr)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        if not file_path.exists() or not file_path.is_file():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}", file=sys.stderr)
            return ""
    
    def discover_shared_libraries(self) -> Dict[str, Set[str]]:
        """
        Discover shared libraries and which services use them.
        
        Returns:
            Dict mapping library paths to sets of service names
        """
        lib_map = {}
        
        # Import build dependency graph to get service dependencies
        try:
            sys.path.insert(0, str(self.root_dir / "scripts"))
            from build_dependency_graph import BuildDependencyGraph
            
            graph = BuildDependencyGraph(self.root_dir)
            graph.build_graph()
            
            # Get shared library usage from dependency graph
            for lib, services in graph.shared_libs.items():
                lib_map[lib] = services
        
        except Exception as e:
            print(f"Error discovering shared libraries: {e}", file=sys.stderr)
        
        return lib_map
    
    def detect_changed_files(self, directory: Path) -> List[Path]:
        """
        Detect files that have changed since last check.
        
        Args:
            directory: Directory to scan for changes
            
        Returns:
            List of changed file paths
        """
        changed_files = []
        
        # Load previous hashes
        cache = self.load_cache()
        previous_hashes = cache.get('file_hashes', {})
        
        # Scan directory for files
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip certain directories
            skip_dirs = ['.git', '__pycache__', 'node_modules', '.pytest_cache', 
                        '.hypothesis', 'dist', 'build', '.venv', 'venv']
            
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            # Calculate current hash
            file_key = str(file_path.relative_to(self.root_dir))
            current_hash = self.calculate_file_hash(file_path)
            
            if not current_hash:
                continue
            
            # Compare with previous hash
            previous_hash = previous_hashes.get(file_key, "")
            
            if current_hash != previous_hash:
                changed_files.append(file_path)
                self.file_hashes[file_key] = current_hash
            else:
                self.file_hashes[file_key] = current_hash
        
        return changed_files
    
    def identify_affected_services(self, changed_file: Path) -> Set[str]:
        """
        Identify which services are affected by a file change.
        
        Args:
            changed_file: Path to the changed file
            
        Returns:
            Set of affected service names
        """
        affected = set()
        
        # Check if file is in a shared library
        file_str = str(changed_file)
        
        for lib, services in self.lib_service_map.items():
            if lib in file_str:
                print(f"📚 Shared library '{lib}' changed")
                print(f"   Affected services: {', '.join(sorted(services))}")
                affected.update(services)
        
        # Check if file is in a service directory
        try:
            # Import build dependency graph
            sys.path.insert(0, str(self.root_dir / "scripts"))
            from build_dependency_graph import BuildDependencyGraph
            
            graph = BuildDependencyGraph(self.root_dir)
            graph.build_graph()
            
            # Use the graph's method to get affected services
            affected_from_graph = graph.get_affected_services([str(changed_file)])
            affected.update(affected_from_graph)
        
        except Exception as e:
            print(f"Error identifying affected services: {e}", file=sys.stderr)
        
        return affected
    
    def trigger_rebuild(self, services: Set[str], dry_run: bool = False):
        """
        Trigger rebuild for affected services.
        
        Args:
            services: Set of service names to rebuild
            dry_run: If True, only print what would be done
        """
        if not services:
            print("✅ No services need rebuilding")
            return
        
        print(f"\n🔨 Services to rebuild: {', '.join(sorted(services))}")
        
        for service in sorted(services):
            if dry_run:
                print(f"   [DRY RUN] Would rebuild: {service}")
            else:
                print(f"   Rebuilding: {service}")
                
                try:
                    # Rebuild using docker-compose
                    subprocess.run(
                        ["docker-compose", "build", service],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"   ✅ {service} rebuilt successfully")
                
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Error rebuilding {service}: {e}")
                    print(f"      {e.stderr}")
    
    def watch(self, interval: int = 5, dry_run: bool = False):
        """
        Watch for changes and trigger rebuilds.
        
        Args:
            interval: Seconds between checks
            dry_run: If True, only print what would be done
        """
        print("=== Shared Library Change Watcher ===")
        print(f"Watch interval: {interval}s")
        print(f"Dry run: {dry_run}")
        print("Discovering shared libraries...")
        
        # Discover shared libraries
        self.lib_service_map = self.discover_shared_libraries()
        
        if self.lib_service_map:
            print(f"\nMonitoring {len(self.lib_service_map)} shared libraries:")
            for lib, services in sorted(self.lib_service_map.items()):
                print(f"  📚 {lib} → {', '.join(sorted(services))}")
        else:
            print("\n⚠️  No shared libraries detected")
        
        print("\nPress Ctrl+C to stop")
        print("=" * 40)
        
        # Initial scan
        print("\nPerforming initial scan...")
        for lib in self.lib_service_map.keys():
            lib_path = self.root_dir / lib
            if lib_path.exists():
                self.detect_changed_files(lib_path)
        
        self.save_cache()
        print("✅ Initial scan complete\n")
        
        # Watch loop
        try:
            while True:
                time.sleep(interval)
                
                changed_files = []
                
                # Check each shared library directory
                for lib in self.lib_service_map.keys():
                    lib_path = self.root_dir / lib
                    if lib_path.exists():
                        lib_changes = self.detect_changed_files(lib_path)
                        changed_files.extend(lib_changes)
                
                if changed_files:
                    print(f"\n🔍 Detected {len(changed_files)} changed file(s)")
                    
                    # Identify affected services
                    all_affected = set()
                    
                    for changed_file in changed_files:
                        print(f"   📝 {changed_file.relative_to(self.root_dir)}")
                        affected = self.identify_affected_services(changed_file)
                        all_affected.update(affected)
                    
                    # Trigger rebuilds
                    if all_affected:
                        self.trigger_rebuild(all_affected, dry_run=dry_run)
                    
                    # Save updated cache
                    self.save_cache()
                    print()
        
        except KeyboardInterrupt:
            print("\n\nStopping watcher...")
            self.save_cache()
    
    def detect_file_changes(self, file_path: str):
        """
        Detect if a specific file has changed and identify affected services.
        
        Args:
            file_path: Path to the file to check
        """
        file = Path(file_path)
        
        if not file.exists():
            print(f"❌ File not found: {file_path}")
            return
        
        # Discover shared libraries
        self.lib_service_map = self.discover_shared_libraries()
        
        # Identify affected services
        affected = self.identify_affected_services(file)
        
        if affected:
            print(f"\n📦 File: {file_path}")
            print(f"🔨 Affected services: {', '.join(sorted(affected))}")
        else:
            print(f"\n📦 File: {file_path}")
            print("✅ No services affected")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        command = "watch"
    else:
        command = sys.argv[1]
    
    watcher = SharedLibraryWatcher()
    
    if command == "watch":
        interval = int(os.environ.get("WATCH_INTERVAL", "5"))
        dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
        watcher.watch(interval=interval, dry_run=dry_run)
    
    elif command == "detect":
        if len(sys.argv) < 3:
            print("Usage: python scripts/shared_lib_watcher.py detect <file>")
            sys.exit(1)
        
        file_path = sys.argv[2]
        watcher.detect_file_changes(file_path)
    
    elif command == "rebuild":
        if len(sys.argv) < 3:
            print("Usage: python scripts/shared_lib_watcher.py rebuild <library>")
            sys.exit(1)
        
        library = sys.argv[2]
        
        # Discover shared libraries
        watcher.lib_service_map = watcher.discover_shared_libraries()
        
        # Get services using this library
        services = watcher.lib_service_map.get(library, set())
        
        if services:
            print(f"📚 Library: {library}")
            print(f"🔨 Services using it: {', '.join(sorted(services))}")
            
            dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
            watcher.trigger_rebuild(services, dry_run=dry_run)
        else:
            print(f"❌ Library '{library}' not found or not used by any service")
    
    else:
        print(f"Unknown command: {command}")
        print("\nUsage: python scripts/shared_lib_watcher.py {watch|detect|rebuild}")
        print("")
        print("Commands:")
        print("  watch              Watch for shared library changes and trigger rebuilds")
        print("  detect <file>      Detect which services are affected by a file change")
        print("  rebuild <library>  Rebuild all services using a shared library")
        print("")
        print("Environment variables:")
        print("  WATCH_INTERVAL     Seconds between checks (default: 5)")
        print("  DRY_RUN            Only print actions, don't execute (true/false)")
        sys.exit(1)


if __name__ == "__main__":
    main()
