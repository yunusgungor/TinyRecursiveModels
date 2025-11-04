#!/usr/bin/env python3
"""
Command-line interface examples for Gmail Dataset Creator.

This script demonstrates various CLI usage patterns and can be used
to test CLI functionality programmatically.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a CLI command and display results."""
    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(cmd)}")
    print("Output:")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error running command: {e}")


def main():
    """Demonstrate CLI usage examples."""
    print("=== Gmail Dataset Creator - CLI Examples ===")
    
    # Check if the CLI is available
    cli_cmd = ["python", "-m", "gmail_dataset_creator.cli"]
    
    # Example 1: Generate sample configuration
    run_command(
        cli_cmd + ["--generate-config", "sample_config.yaml"],
        "Generate Sample Configuration"
    )
    
    # Example 2: Show help
    run_command(
        cli_cmd + ["--help"],
        "Show Help Information"
    )
    
    # Example 3: Interactive mode (would require user input)
    print("\n=== Interactive Mode Example ===")
    print("Command: python -m gmail_dataset_creator.cli --interactive")
    print("Note: This would start interactive configuration mode")
    
    # Example 4: Dry run with configuration
    if os.path.exists("sample_config.yaml"):
        run_command(
            cli_cmd + ["--config", "sample_config.yaml", "--dry-run"],
            "Dry Run with Configuration"
        )
    
    # Example 5: Authentication test
    print("\n=== Authentication Test Example ===")
    print("Command: python -m gmail_dataset_creator.cli --auth-only --verbose")
    print("Note: This would test Gmail API authentication")
    
    # Example 6: Status check
    print("\n=== Status Check Example ===")
    print("Command: python -m gmail_dataset_creator.cli --status")
    print("Note: This would show current system status")
    
    # Example 7: Full dataset creation with parameters
    print("\n=== Full Dataset Creation Example ===")
    example_cmd = [
        "python", "-m", "gmail_dataset_creator.cli",
        "--config", "config.yaml",
        "--max-emails", "500",
        "--date-start", "2023-01-01",
        "--date-end", "2023-12-31",
        "--output", "./my_dataset",
        "--anonymize-senders",
        "--confidence-threshold", "0.8",
        "--verbose"
    ]
    print(f"Command: {' '.join(example_cmd)}")
    print("Note: This would create a dataset with specific parameters")
    
    # Example 8: Resume interrupted process
    print("\n=== Resume Process Example ===")
    resume_cmd = [
        "python", "-m", "gmail_dataset_creator.cli",
        "--config", "config.yaml",
        "--resume",
        "--verbose"
    ]
    print(f"Command: {' '.join(resume_cmd)}")
    print("Note: This would resume an interrupted dataset creation process")
    
    print("\n=== CLI Examples Complete ===")
    print("\nTo use these examples:")
    print("1. Install the package: pip install -e .")
    print("2. Set up your credentials and API keys")
    print("3. Run the commands shown above")


if __name__ == '__main__':
    main()