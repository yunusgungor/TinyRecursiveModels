#!/usr/bin/env python3
"""
Complete Email Classification Training Pipeline

This script runs the complete pipeline for training an email classification model:
1. Build email dataset
2. Train TRM model
3. Evaluate results
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description} completed!")
        if result.stdout:
            print(f"Output: {result.stdout}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Email Classification Training Pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="outputs/email_classification", help="Output directory")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset creation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--sample-data", action="store_true", help="Use sample data for testing")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Email Classification Training Pipeline")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.num_gpus} GPU(s)")
    
    # Step 1: Build email dataset
    if not args.skip_dataset:
        dataset_output = os.path.join(args.data_dir, "email-classification")
        
        if args.sample_data:
            # Create sample email data
            sample_emails_file = os.path.join(args.data_dir, "emails.json")
            
            dataset_cmd = f"""python dataset/build_email_dataset.py \\
                --input_file {sample_emails_file} \\
                --output_dir {dataset_output} \\
                --num_aug 50 \\
                --max_seq_len 256 \\
                --seed 42"""
        else:
            # Use provided email data
            emails_file = os.path.join(args.data_dir, "emails.json")
            if not os.path.exists(emails_file):
                print(f"ERROR: Email data file not found at {emails_file}")
                print("Please provide emails.json or use --sample-data flag")
                sys.exit(1)
            
            dataset_cmd = f"""python dataset/build_email_dataset.py \\
                --input_file {emails_file} \\
                --output_dir {dataset_output} \\
                --num_aug 100 \\
                --max_seq_len 512 \\
                --seed 42"""
        
        run_command(dataset_cmd, "Building email classification dataset")
        
        # Verify dataset was created
        if not os.path.exists(os.path.join(dataset_output, "train", "dataset.json")):
            print("ERROR: Dataset creation failed - train dataset not found")
            sys.exit(1)
        
        print(f"Dataset created successfully at {dataset_output}")
    
    # Step 2: Train model
    if not args.skip_training:
        
        # Update config with command line arguments
        config_overrides = [
            f"training.batch_size={args.batch_size}",
            f"training.max_steps={args.max_steps}",
            f"data_paths=[{os.path.join(args.data_dir, 'email-classification')}]",
            f"output_dir={args.output_dir}"
        ]
        
        if args.num_gpus > 1:
            # Distributed training
            train_cmd = f"""torchrun --nproc-per-node {args.num_gpus} train_email_classifier.py \\
                {' '.join(config_overrides)}"""
        else:
            # Single GPU training
            train_cmd = f"""python train_email_classifier.py \\
                {' '.join(config_overrides)}"""
        
        run_command(train_cmd, "Training email classification model")
        
        # Verify training completed
        if not os.path.exists(os.path.join(args.output_dir, "best_model.pt")):
            print("WARNING: Training may not have completed successfully - best_model.pt not found")
    
    # Step 3: Evaluate model (if training was completed)
    if not args.skip_training and os.path.exists(os.path.join(args.output_dir, "best_model.pt")):
        
        eval_cmd = f"""python -c "
import torch
import json
from models.recursive_reasoning.trm_email import EmailTRM
from evaluators.email import EmailClassificationEvaluator

# Load model
checkpoint = torch.load('{args.output_dir}/best_model.pt', map_location='cpu')
print('Model loaded successfully!')
print(f'Best accuracy: {{checkpoint.get(\"best_accuracy\", \"N/A\")}}')

# Load final metrics if available
metrics_file = '{args.output_dir}/final_metrics.json'
try:
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    print('\\nFinal Evaluation Metrics:')
    print(f'Accuracy: {{metrics.get(\"accuracy\", \"N/A\"):.4f}}')
    print(f'Macro F1: {{metrics.get(\"macro_f1\", \"N/A\"):.4f}}')
    print(f'Micro F1: {{metrics.get(\"micro_f1\", \"N/A\"):.4f}}')
except FileNotFoundError:
    print('Final metrics not found')
"
"""
        
        run_command(eval_cmd, "Evaluating trained model")
    
    print("\n" + "="*60)
    print("EMAIL CLASSIFICATION TRAINING PIPELINE COMPLETED!")
    print("="*60)
    
    if not args.skip_dataset:
        print(f"✓ Dataset created at: {os.path.join(args.data_dir, 'email-classification')}")
    
    if not args.skip_training:
        print(f"✓ Model trained and saved at: {args.output_dir}")
        print(f"✓ Best model: {os.path.join(args.output_dir, 'best_model.pt')}")
    
    print("\nNext steps:")
    print("1. Check training logs and metrics")
    print("2. Evaluate model on your own email data")
    print("3. Deploy model for production use")
    print("4. Fine-tune hyperparameters if needed")


if __name__ == "__main__":
    main()