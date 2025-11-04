#!/usr/bin/env python3
"""
Optimized Training Runner

This script sets up the optimal environment for training and runs the email classifier
with proper configuration to avoid common issues like OMP warnings and memory constraints.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up optimal environment variables for training."""
    
    # Fix OMP warnings by setting proper thread limits
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2" 
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    
    # Prevent fork warnings in multiprocessing
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    # Set PyTorch threading
    os.environ["TORCH_NUM_THREADS"] = "2"
    
    print("Environment variables set:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    print(f"  TORCH_NUM_THREADS: {os.environ.get('TORCH_NUM_THREADS')}")

def run_training_with_optimized_settings():
    """Run training with optimized settings for MacBook."""
    
    # Training command with optimized parameters
    cmd = [
        "python3", "./train_email_classifier_macbook.py",
        "--train",
        "--dataset-path", "./enhanced_emails",
        "--output-dir", "./training_output_optimized",
        "--batch-size", "4",  # Slightly larger batch size for better learning
        "--learning-rate", "5e-5",  # Lower learning rate for better convergence
        "--max-steps", "2000",  # Reasonable number of steps for the dataset size
        "--gradient-accumulation-steps", "4",  # Reduce memory pressure
        "--max-sequence-length", "256",  # Reduce sequence length to save memory
        "--hidden-size", "256",  # Smaller hidden size for memory efficiency
        "--num-layers", "2",  # Keep layers reasonable
        "--vocab-size", "500",  # Match our actual vocab size
        "--target-accuracy", "0.85",  # More realistic target
        "--early-stopping-patience", "10",  # More patience for small dataset
        "--strategy", "multi_phase",  # Use multi-phase training
        "--log-level", "INFO"
    ]
    
    print("Running training with optimized settings:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False

def main():
    """Main function."""
    print("="*60)
    print("Optimized Email Classification Training")
    print("="*60)
    
    # Setup environment
    setup_environment()
    print()
    
    # Check if enhanced dataset exists
    if not Path("enhanced_emails").exists():
        print("Enhanced dataset not found. Creating it first...")
        try:
            subprocess.run(["python3", "create_enhanced_dataset.py"], check=True)
            print("Enhanced dataset created successfully.")
        except subprocess.CalledProcessError:
            print("Failed to create enhanced dataset.")
            return 1
    
    # Run training
    success = run_training_with_optimized_settings()
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the training_output_optimized directory for results.")
    else:
        print("\n" + "="*60)
        print("TRAINING FAILED!")
        print("="*60)
        print("Check the error messages above for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())