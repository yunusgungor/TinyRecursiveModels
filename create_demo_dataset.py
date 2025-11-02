#!/usr/bin/env python3
"""
Create a minimal demo dataset for MacBook training system.
"""

import os
import json
import numpy as np
from dataset.common import PuzzleDatasetMetadata

def create_demo_dataset():
    """Create a minimal ARC-style dataset for demonstration."""
    
    # Create directories
    os.makedirs("data/arc-demo/train", exist_ok=True)
    os.makedirs("data/arc-demo/test", exist_ok=True)
    
    # Demo parameters
    seq_len = 900  # 30x30 grid flattened
    vocab_size = 12  # PAD + EOS + digits 0-9
    num_examples = 100
    num_puzzles = 10
    num_groups = 5
    
    # Create training data
    np.random.seed(42)
    
    # Generate simple pattern data (random for demo)
    train_inputs = np.random.randint(0, vocab_size, (num_examples, seq_len), dtype=np.uint8)
    train_labels = np.random.randint(0, vocab_size, (num_examples, seq_len), dtype=np.uint8)
    train_puzzle_identifiers = np.random.randint(1, 6, num_puzzles, dtype=np.int32)
    train_puzzle_indices = np.linspace(0, num_examples, num_puzzles + 1, dtype=np.int32)
    train_group_indices = np.linspace(0, num_puzzles, num_groups + 1, dtype=np.int32)
    
    # Save training data
    np.save("data/arc-demo/train/all__inputs.npy", train_inputs)
    np.save("data/arc-demo/train/all__labels.npy", train_labels)
    np.save("data/arc-demo/train/all__puzzle_identifiers.npy", train_puzzle_identifiers)
    np.save("data/arc-demo/train/all__puzzle_indices.npy", train_puzzle_indices)
    np.save("data/arc-demo/train/all__group_indices.npy", train_group_indices)
    
    # Create test data (smaller)
    test_examples = 20
    test_puzzles = 5
    test_groups = 2
    
    test_inputs = np.random.randint(0, vocab_size, (test_examples, seq_len), dtype=np.uint8)
    test_labels = np.random.randint(0, vocab_size, (test_examples, seq_len), dtype=np.uint8)
    test_puzzle_identifiers = np.random.randint(1, 6, test_puzzles, dtype=np.int32)
    test_puzzle_indices = np.linspace(0, test_examples, test_puzzles + 1, dtype=np.int32)
    test_group_indices = np.linspace(0, test_puzzles, test_groups + 1, dtype=np.int32)
    
    # Save test data
    np.save("data/arc-demo/test/all__inputs.npy", test_inputs)
    np.save("data/arc-demo/test/all__labels.npy", test_labels)
    np.save("data/arc-demo/test/all__puzzle_identifiers.npy", test_puzzle_identifiers)
    np.save("data/arc-demo/test/all__puzzle_indices.npy", test_puzzle_indices)
    np.save("data/arc-demo/test/all__group_indices.npy", test_group_indices)
    
    # Create metadata for training
    train_metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=6,  # 0 (blank) + 5 puzzle types
        total_groups=num_groups,
        mean_puzzle_examples=num_examples / num_puzzles,
        total_puzzles=num_puzzles,
        sets=["all"]
    )
    
    # Create metadata for test
    test_metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=6,
        total_groups=test_groups,
        mean_puzzle_examples=test_examples / test_puzzles,
        total_puzzles=test_puzzles,
        sets=["all"]
    )
    
    # Save metadata
    with open("data/arc-demo/train/dataset.json", "w") as f:
        json.dump(train_metadata.model_dump(), f, indent=2)
    
    with open("data/arc-demo/test/dataset.json", "w") as f:
        json.dump(test_metadata.model_dump(), f, indent=2)
    
    # Create identifiers mapping
    identifiers = ["<blank>", "pattern_1", "pattern_2", "pattern_3", "pattern_4", "pattern_5"]
    with open("data/arc-demo/identifiers.json", "w") as f:
        json.dump(identifiers, f, indent=2)
    
    print("Demo dataset created successfully!")
    print(f"Training examples: {num_examples}")
    print(f"Test examples: {test_examples}")
    print(f"Sequence length: {seq_len}")
    print(f"Vocabulary size: {vocab_size}")

if __name__ == "__main__":
    create_demo_dataset()