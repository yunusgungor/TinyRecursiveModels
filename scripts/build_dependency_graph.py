#!/usr/bin/env python3
"""
Build Dependency Graph for Microservices

This module analyzes service dependencies and calculates optimal build order
for monorepo microservices architecture.

Usage:
    python scripts/build_dependency_graph.py analyze
    python scripts/build_dependency_graph.py order
    python scripts/build_dependency_graph.py visualize
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque


class ServiceDependency:
    """Represents a service and its dependencies."""
    
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.dependencies: Set[str] = set()
        self.shared_libraries: Set[str] = set()
        self.docker_dependencies: Set[str] = set()
    
    def __repr__(self):
        return f"Service({self.name}, deps={self.dependencies})"


class BuildDependencyGraph:
    """Analyzes and manages build dependencies between services."""
    
    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or Path.cwd()
        self.services: Dict[str, ServiceDependency] = {}
        self.shared_libs: Dict[str, Set[str]] = defaultdict(set)  # lib -> services using it
        
    def discover_services(self) -> List[str]:
        """
        Discover all services in the monorepo.
        
        Returns:
            List of service names
        """
        services = []
        
        # Look for directories with Dockerfile
        for item in self.root_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dockerfile = item / "Dockerfile"
                if dockerfile.exists():
                    services.append(item.name)
        
        return services
    
    def analyze_service(self, service_name: str) -> ServiceDependency:
        """
        Analyze a service to determine its dependencies.
        
        Args:
            service_name: Name of the service to analyze
            
        Returns:
            ServiceDependency object
        """
        service_path = self.root_dir / service_name
        service = ServiceDependency(service_name, service_path)
        
        # Analyze Dockerfile for dependencies
        dockerfile = service_path / "Dockerfile"
        if dockerfile.exists():
            service.docker_dependencies = self._analyze_dockerfile(dockerfile)
        
        # Analyze package files for shared library dependencies
        if service_name == "backend":
            requirements_file = service_path / "requirements.txt"
            if requirements_file.exists():
                service.shared_libraries = self._analyze_python_deps(requirements_file)
        
        elif service_name == "frontend":
            package_file = service_path / "package.json"
            if package_file.exists():
                service.shared_libraries = self._analyze_npm_deps(package_file)
        
        # Check for explicit service dependencies in docker-compose
        compose_file = self.root_dir / "docker-compose.yml"
        if compose_file.exists():
            service.dependencies = self._analyze_compose_deps(compose_file, service_name)
        
        return service
    
    def _analyze_dockerfile(self, dockerfile: Path) -> Set[str]:
        """Analyze Dockerfile for build dependencies."""
        dependencies = set()
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        # Look for COPY --from=<service> patterns
        copy_from_pattern = r'COPY\s+--from=(\w+)'
        matches = re.findall(copy_from_pattern, content)
        
        for match in matches:
            # Filter out stage names (builder, base, etc.)
            if match not in ['builder', 'base', 'dependencies', 'development', 'production']:
                dependencies.add(match)
        
        return dependencies
    
    def _analyze_python_deps(self, requirements_file: Path) -> Set[str]:
        """Analyze Python requirements for shared libraries."""
        shared_libs = set()
        
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Extract package name (before ==, >=, etc.)
                package = re.split(r'[=<>!]', line)[0].strip()
                
                # Check if it's a local/shared package (starts with ./ or ../)
                if package.startswith('.') or package.startswith('..'):
                    shared_libs.add(package)
        
        return shared_libs
    
    def _analyze_npm_deps(self, package_file: Path) -> Set[str]:
        """Analyze npm package.json for shared libraries."""
        shared_libs = set()
        
        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            # Check dependencies for local packages (file: protocol)
            for dep_type in ['dependencies', 'devDependencies']:
                deps = package_data.get(dep_type, {})
                
                for name, version in deps.items():
                    if version.startswith('file:') or version.startswith('link:'):
                        shared_libs.add(name)
        
        except Exception as e:
            print(f"Error analyzing {package_file}: {e}", file=sys.stderr)
        
        return shared_libs
    
    def _analyze_compose_deps(self, compose_file: Path, service_name: str) -> Set[str]:
        """Analyze docker-compose.yml for service dependencies."""
        dependencies = set()
        
        try:
            import yaml
            
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            service_config = services.get(service_name, {})
            
            # Check depends_on
            depends_on = service_config.get('depends_on', [])
            
            if isinstance(depends_on, list):
                dependencies.update(depends_on)
            elif isinstance(depends_on, dict):
                dependencies.update(depends_on.keys())
        
        except Exception as e:
            print(f"Error analyzing {compose_file}: {e}", file=sys.stderr)
        
        return dependencies
    
    def build_graph(self):
        """Build the complete dependency graph."""
        # Discover all services
        service_names = self.discover_services()
        
        print(f"Discovered services: {', '.join(service_names)}")
        
        # Analyze each service
        for service_name in service_names:
            service = self.analyze_service(service_name)
            self.services[service_name] = service
            
            # Track shared library usage
            for lib in service.shared_libraries:
                self.shared_libs[lib].add(service_name)
        
        print(f"\nAnalyzed {len(self.services)} services")
    
    def calculate_build_order(self) -> List[str]:
        """
        Calculate optimal build order using topological sort.
        
        Returns:
            List of service names in build order
            
        Raises:
            ValueError: If circular dependency detected
        """
        # Build adjacency list (service -> services that depend on it)
        in_degree = {service: 0 for service in self.services}
        adj_list = defaultdict(list)
        
        for service_name, service in self.services.items():
            for dep in service.dependencies:
                if dep in self.services:
                    adj_list[dep].append(service_name)
                    in_degree[service_name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([s for s in self.services if in_degree[s] == 0])
        build_order = []
        
        while queue:
            service = queue.popleft()
            build_order.append(service)
            
            for dependent in adj_list[service]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(build_order) != len(self.services):
            remaining = set(self.services.keys()) - set(build_order)
            raise ValueError(f"Circular dependency detected involving: {remaining}")
        
        return build_order
    
    def get_affected_services(self, changed_files: List[str]) -> Set[str]:
        """
        Determine which services are affected by file changes.
        
        Args:
            changed_files: List of changed file paths
            
        Returns:
            Set of service names that need rebuilding
        """
        affected = set()
        
        for file_path in changed_files:
            path = Path(file_path)
            
            # Check if file belongs to a service
            for service_name, service in self.services.items():
                try:
                    # Check if file is within service directory
                    path.relative_to(service.path)
                    affected.add(service_name)
                    
                    # If it's a shared library, add all dependent services
                    if any(lib in str(path) for lib in service.shared_libraries):
                        for lib in service.shared_libraries:
                            if lib in str(path):
                                affected.update(self.shared_libs.get(lib, set()))
                    
                    break
                except ValueError:
                    # File not in this service
                    continue
        
        # Add services that depend on affected services
        expanded = set(affected)
        for service_name in affected:
            expanded.update(self._get_dependent_services(service_name))
        
        return expanded
    
    def _get_dependent_services(self, service_name: str) -> Set[str]:
        """Get all services that depend on the given service."""
        dependents = set()
        
        for name, service in self.services.items():
            if service_name in service.dependencies:
                dependents.add(name)
                # Recursively get dependents
                dependents.update(self._get_dependent_services(name))
        
        return dependents
    
    def visualize(self):
        """Print a visual representation of the dependency graph."""
        print("\n=== Build Dependency Graph ===\n")
        
        for service_name, service in sorted(self.services.items()):
            print(f"📦 {service_name}")
            
            if service.dependencies:
                print(f"   ├─ Service Dependencies: {', '.join(sorted(service.dependencies))}")
            
            if service.docker_dependencies:
                print(f"   ├─ Docker Dependencies: {', '.join(sorted(service.docker_dependencies))}")
            
            if service.shared_libraries:
                print(f"   ├─ Shared Libraries: {', '.join(sorted(service.shared_libraries))}")
            
            # Show which services depend on this one
            dependents = self._get_dependent_services(service_name)
            if dependents:
                print(f"   └─ Depended on by: {', '.join(sorted(dependents))}")
            
            print()
        
        # Show shared libraries
        if self.shared_libs:
            print("=== Shared Libraries ===\n")
            for lib, services in sorted(self.shared_libs.items()):
                print(f"📚 {lib}")
                print(f"   └─ Used by: {', '.join(sorted(services))}")
                print()
        
        # Show build order
        try:
            build_order = self.calculate_build_order()
            print("=== Optimal Build Order ===\n")
            for idx, service in enumerate(build_order, 1):
                print(f"{idx}. {service}")
            print()
        except ValueError as e:
            print(f"❌ Error calculating build order: {e}\n")
    
    def export_json(self, output_file: Path):
        """Export dependency graph to JSON."""
        data = {
            'services': {
                name: {
                    'dependencies': list(service.dependencies),
                    'shared_libraries': list(service.shared_libraries),
                    'docker_dependencies': list(service.docker_dependencies),
                }
                for name, service in self.services.items()
            },
            'shared_libraries': {
                lib: list(services)
                for lib, services in self.shared_libs.items()
            }
        }
        
        try:
            build_order = self.calculate_build_order()
            data['build_order'] = build_order
        except ValueError as e:
            data['build_order_error'] = str(e)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Dependency graph exported to {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        command = "analyze"
    else:
        command = sys.argv[1]
    
    graph = BuildDependencyGraph()
    
    if command == "analyze":
        graph.build_graph()
        print("\n✅ Dependency analysis complete")
        
    elif command == "order":
        graph.build_graph()
        try:
            build_order = graph.calculate_build_order()
            print("\n=== Build Order ===")
            for idx, service in enumerate(build_order, 1):
                print(f"{idx}. {service}")
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    elif command == "visualize":
        graph.build_graph()
        graph.visualize()
    
    elif command == "export":
        output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("build_graph.json")
        graph.build_graph()
        graph.export_json(output_file)
    
    elif command == "affected":
        if len(sys.argv) < 3:
            print("Usage: python scripts/build_dependency_graph.py affected <file1> [file2] ...")
            sys.exit(1)
        
        changed_files = sys.argv[2:]
        graph.build_graph()
        affected = graph.get_affected_services(changed_files)
        
        print(f"\n=== Affected Services ===")
        print(f"Changed files: {', '.join(changed_files)}")
        print(f"Services to rebuild: {', '.join(sorted(affected)) if affected else 'None'}")
    
    else:
        print(f"Unknown command: {command}")
        print("\nUsage: python scripts/build_dependency_graph.py {analyze|order|visualize|export|affected}")
        print("")
        print("Commands:")
        print("  analyze              Analyze service dependencies")
        print("  order                Calculate and display optimal build order")
        print("  visualize            Show visual dependency graph")
        print("  export [file]        Export graph to JSON (default: build_graph.json)")
        print("  affected <files...>  Show which services are affected by file changes")
        sys.exit(1)


if __name__ == "__main__":
    main()
