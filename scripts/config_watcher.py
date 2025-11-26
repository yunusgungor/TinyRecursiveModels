#!/usr/bin/env python3
"""
Configuration Change Detection and Selective Service Restart

This script watches configuration files and triggers selective service restarts
when changes are detected.

Usage:
    python scripts/config_watcher.py watch
    python scripts/config_watcher.py list
    python scripts/config_watcher.py test <config-file>
"""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

# Configuration file to service mapping
CONFIG_SERVICE_MAP = {
    ".env": ["backend", "frontend"],
    ".env.development": ["backend", "frontend"],
    ".env.production": ["backend", "frontend"],
    "backend/.env": ["backend"],
    "backend/requirements.txt": ["backend"],
    "backend/requirements-dev.txt": ["backend"],
    "frontend/.env": ["frontend"],
    "frontend/package.json": ["frontend"],
    "frontend/package-lock.json": ["frontend"],
    "nginx/nginx.conf": ["nginx"],
    "nginx/conf.d/default.conf": ["nginx"],
    "docker-compose.yml": ["all"],
    "docker-compose.prod.yml": ["all"],
}

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_info(message: str):
    """Print info message in green."""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")


def print_warning(message: str):
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def print_error(message: str):
    """Print error message in red."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def print_debug(message: str):
    """Print debug message in blue if DEBUG is enabled."""
    if os.environ.get("DEBUG", "").lower() == "true":
        print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {message}")


def get_file_checksum(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    if not file_path.exists():
        return ""
    
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print_error(f"Error reading {file_path}: {e}")
        return ""


def get_affected_services(config_file: str) -> List[str]:
    """Get list of services affected by a configuration file change."""
    # Check exact match first
    if config_file in CONFIG_SERVICE_MAP:
        return CONFIG_SERVICE_MAP[config_file]
    
    # Check for pattern matches
    for pattern, services in CONFIG_SERVICE_MAP.items():
        if pattern in config_file:
            return services
    
    return []


def get_running_services() -> Set[str]:
    """Get list of currently running services."""
    try:
        result = subprocess.run(
            ["docker-compose", "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.strip().split('\n'))
    except subprocess.CalledProcessError as e:
        print_error(f"Error getting running services: {e}")
        return set()


def restart_service(service: str) -> bool:
    """Restart a single service."""
    try:
        print_info(f"Restarting service: {service}")
        subprocess.run(
            ["docker-compose", "restart", service],
            check=True,
            capture_output=True
        )
        print_info(f"Service {service} restarted successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Error restarting {service}: {e}")
        return False


def restart_affected_services(config_file: str):
    """Restart services affected by a configuration file change."""
    services = get_affected_services(config_file)
    
    if not services:
        print_debug(f"No services affected by {config_file}")
        return
    
    print_info(f"Configuration file changed: {config_file}")
    print_info(f"Affected services: {', '.join(services)}")
    
    if "all" in services:
        print_warning("docker-compose.yml changed - this may require full restart")
        print_info("Run: docker-compose up -d --force-recreate")
        return
    
    # Get currently running services
    running_services = get_running_services()
    
    # Restart each affected service that is running
    for service in services:
        if service in running_services:
            restart_service(service)
        else:
            print_warning(f"Service {service} not running, skipping restart")


def watch_configs(watch_interval: int = 5):
    """Watch configuration files and trigger restarts on changes."""
    print_info("=== Configuration Change Watcher ===")
    print_info(f"Watch interval: {watch_interval}s")
    print_info("Monitoring configuration files...")
    print_info("Press Ctrl+C to stop")
    print_info("====================================")
    
    # Initialize checksums
    file_checksums: Dict[str, str] = {}
    
    for config_file in CONFIG_SERVICE_MAP.keys():
        file_path = Path(config_file)
        if file_path.exists():
            file_checksums[config_file] = get_file_checksum(file_path)
            print_debug(f"Watching: {config_file}")
    
    # Watch loop
    try:
        while True:
            time.sleep(watch_interval)
            
            for config_file in CONFIG_SERVICE_MAP.keys():
                file_path = Path(config_file)
                
                if not file_path.exists():
                    continue
                
                current_checksum = get_file_checksum(file_path)
                previous_checksum = file_checksums.get(config_file, "")
                
                if current_checksum != previous_checksum:
                    print_info(f"Change detected in: {config_file}")
                    
                    # Update checksum
                    file_checksums[config_file] = current_checksum
                    
                    # Restart affected services
                    restart_affected_services(config_file)
    
    except KeyboardInterrupt:
        print_info("\nStopping configuration watcher...")


def list_watched_files():
    """List all watched configuration files."""
    print_info("=== Watched Configuration Files ===")
    
    for config_file, services in CONFIG_SERVICE_MAP.items():
        file_path = Path(config_file)
        exists = "✅" if file_path.exists() else "❌"
        services_str = ", ".join(services)
        
        print(f"{exists} {config_file} → {services_str}")
    
    print_info("====================================")


def test_config_detection(config_file: str):
    """Test which services would be affected by a configuration file change."""
    print_info(f"Testing configuration change detection for: {config_file}")
    
    services = get_affected_services(config_file)
    
    if not services:
        print_warning(f"No services configured for {config_file}")
        print_info("Add mapping to CONFIG_SERVICE_MAP in script")
    else:
        print_info(f"Affected services: {', '.join(services)}")
        
        if "all" in services:
            print_warning("This would trigger a full stack restart")
        else:
            print_info(f"Would restart: {', '.join(services)}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        command = "watch"
    else:
        command = sys.argv[1]
    
    if command == "watch":
        watch_interval = int(os.environ.get("WATCH_INTERVAL", "5"))
        watch_configs(watch_interval)
    
    elif command == "list":
        list_watched_files()
    
    elif command == "test":
        if len(sys.argv) < 3:
            print_error("Usage: python scripts/config_watcher.py test <config-file>")
            sys.exit(1)
        test_config_detection(sys.argv[2])
    
    elif command == "restart":
        if len(sys.argv) < 3:
            print_error("Usage: python scripts/config_watcher.py restart <config-file>")
            sys.exit(1)
        restart_affected_services(sys.argv[2])
    
    else:
        print_error(f"Unknown command: {command}")
        print("Usage: python scripts/config_watcher.py {watch|list|test|restart} [config-file]")
        print("")
        print("Commands:")
        print("  watch              Watch configuration files and auto-restart services")
        print("  list               List all watched configuration files")
        print("  test <file>        Test which services would be affected by a file change")
        print("  restart <file>     Manually trigger restart for services affected by a file")
        print("")
        print("Environment variables:")
        print("  WATCH_INTERVAL     Seconds between checks (default: 5)")
        print("  DEBUG              Enable debug output (true/false)")
        sys.exit(1)


if __name__ == "__main__":
    main()
