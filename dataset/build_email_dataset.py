from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np
import re
from collections import Counter

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class EmailDataProcessConfig(BaseModel):
    input_file: str = "data/emails.json"  # JSON file with email data
    output_dir: str = "data/email-classification"
    seed: int = 42
    num_aug: int = 100  # Number of augmentations per email
    max_seq_len: int = 512  # Maximum sequence length for emails
    min_emails_per_category: int = 10  # Minimum emails per category
    test_split_ratio: float = 0.2  # Ratio for test split
    puzzle_identifiers_start: int = 1


# Email categories for classification
EMAIL_CATEGORIES = {
    "newsletter": 0,
    "work": 1, 
    "personal": 2,
    "spam": 3,
    "promotional": 4,
    "social": 5,
    "finance": 6,
    "travel": 7,
    "shopping": 8,
    "other": 9
}

# Special tokens
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<EOS>": 1,
    "<UNK>": 2,
    "<SUBJECT>": 3,
    "<BODY>": 4,
    "<FROM>": 5,
    "<TO>": 6
}


@dataclass
class EmailExample:
    id: str
    subject: str
    body: str
    sender: str
    recipient: str
    category: str
    
    
def clean_text(text: str) -> str:
    """Clean and normalize email text"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses (keep structure but anonymize)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE>', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-<>]', '', text)
    
    return text.strip()


def tokenize_email(email: EmailExample, vocab: Dict[str, int], max_seq_len: int) -> Tuple[np.ndarray, int]:
    """Tokenize email into sequence of token IDs"""
    
    # Build email sequence with special tokens
    sequence = []
    
    # Add subject
    if email.subject:
        sequence.append(SPECIAL_TOKENS["<SUBJECT>"])
        subject_tokens = clean_text(email.subject).lower().split()
        for token in subject_tokens:
            sequence.append(vocab.get(token, SPECIAL_TOKENS["<UNK>"]))
    
    # Add sender info
    if email.sender:
        sequence.append(SPECIAL_TOKENS["<FROM>"])
        sender_tokens = clean_text(email.sender).lower().split()
        for token in sender_tokens:
            sequence.append(vocab.get(token, SPECIAL_TOKENS["<UNK>"]))
    
    # Add body
    if email.body:
        sequence.append(SPECIAL_TOKENS["<BODY>"])
        body_tokens = clean_text(email.body).lower().split()
        for token in body_tokens:
            sequence.append(vocab.get(token, SPECIAL_TOKENS["<UNK>"]))
    
    # Add EOS token
    sequence.append(SPECIAL_TOKENS["<EOS>"])
    
    # Truncate or pad to max_seq_len
    if len(sequence) > max_seq_len:
        sequence = sequence[:max_seq_len-1] + [SPECIAL_TOKENS["<EOS>"]]
    
    # Convert to numpy array and pad
    padded_sequence = np.full(max_seq_len, SPECIAL_TOKENS["<PAD>"], dtype=np.int32)
    padded_sequence[:len(sequence)] = sequence
    
    # Return sequence and category label
    category_id = EMAIL_CATEGORIES.get(email.category.lower(), EMAIL_CATEGORIES["other"])
    
    return padded_sequence, category_id


def build_vocabulary(emails: List[EmailExample], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from email texts"""
    
    # Count all tokens
    token_counts = Counter()
    
    for email in emails:
        # Process subject
        if email.subject:
            tokens = clean_text(email.subject).lower().split()
            token_counts.update(tokens)
        
        # Process body
        if email.body:
            tokens = clean_text(email.body).lower().split()
            token_counts.update(tokens)
        
        # Process sender (simplified)
        if email.sender:
            tokens = clean_text(email.sender).lower().split()
            token_counts.update(tokens)
    
    # Build vocabulary starting with special tokens
    vocab = SPECIAL_TOKENS.copy()
    
    # Add frequent tokens
    for token, count in token_counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    
    print(f"Built vocabulary with {len(vocab)} tokens")
    return vocab


def augment_email(email: EmailExample, aug_id: int) -> EmailExample:
    """Create augmented version of email"""
    
    # Simple augmentation strategies
    augmented = EmailExample(
        id=f"{email.id}_aug_{aug_id}",
        subject=email.subject,
        body=email.body,
        sender=email.sender,
        recipient=email.recipient,
        category=email.category
    )
    
    # Apply different augmentation strategies based on aug_id
    if aug_id % 4 == 1:
        # Shuffle words in subject (simple word order variation)
        if augmented.subject:
            words = augmented.subject.split()
            if len(words) > 1:
                np.random.shuffle(words)
                augmented.subject = " ".join(words)
    
    elif aug_id % 4 == 2:
        # Truncate body (simulate partial email reading)
        if augmented.body and len(augmented.body) > 100:
            truncate_point = np.random.randint(50, len(augmented.body) - 50)
            augmented.body = augmented.body[:truncate_point]
    
    elif aug_id % 4 == 3:
        # Add noise to sender (simulate different sender formats)
        if augmented.sender:
            noise_options = ["", " (via system)", " <noreply>", " [automated]"]
            augmented.sender += np.random.choice(noise_options)
    
    return augmented


def load_email_data(config: EmailDataProcessConfig) -> List[EmailExample]:
    """Load email data from JSON file"""
    
    if not os.path.exists(config.input_file):
        # Create sample data if file doesn't exist
        print(f"Input file {config.input_file} not found. Creating sample data...")
        create_sample_email_data(config.input_file)
    
    with open(config.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    emails = []
    for item in data:
        email = EmailExample(
            id=item.get('id', f'email_{len(emails)}'),
            subject=item.get('subject', ''),
            body=item.get('body', ''),
            sender=item.get('sender', ''),
            recipient=item.get('recipient', ''),
            category=item.get('category', 'other')
        )
        emails.append(email)
    
    print(f"Loaded {len(emails)} emails")
    return emails


def create_sample_email_data(output_file: str):
    """Create sample email data for testing"""
    
    sample_emails = [
        {
            "id": "email_001",
            "subject": "Weekly Newsletter - Tech Updates",
            "body": "Here are the latest tech updates from this week. New AI developments, startup news, and industry insights.",
            "sender": "newsletter@techblog.com",
            "recipient": "user@example.com",
            "category": "newsletter"
        },
        {
            "id": "email_002", 
            "subject": "Meeting Tomorrow at 2 PM",
            "body": "Hi team, reminder about our project meeting tomorrow at 2 PM in conference room A. Please bring your reports.",
            "sender": "manager@company.com",
            "recipient": "team@company.com",
            "category": "work"
        },
        {
            "id": "email_003",
            "subject": "Happy Birthday!",
            "body": "Hope you have a wonderful birthday celebration with family and friends. Best wishes!",
            "sender": "friend@personal.com", 
            "recipient": "user@example.com",
            "category": "personal"
        },
        {
            "id": "email_004",
            "subject": "URGENT: Claim your prize now!",
            "body": "Congratulations! You've won $1000000. Click here immediately to claim your prize before it expires.",
            "sender": "noreply@suspicious.com",
            "recipient": "user@example.com", 
            "category": "spam"
        },
        {
            "id": "email_005",
            "subject": "50% Off Sale - Limited Time",
            "body": "Don't miss our biggest sale of the year! 50% off all items. Shop now before stocks run out.",
            "sender": "sales@retailstore.com",
            "recipient": "user@example.com",
            "category": "promotional"
        }
    ]
    
    # Create more samples by duplicating with variations
    extended_samples = []
    for i, email in enumerate(sample_emails * 20):  # 100 samples total
        new_email = email.copy()
        new_email["id"] = f"email_{i:03d}"
        # Add some variation
        if "newsletter" in email["category"]:
            new_email["subject"] = f"Newsletter #{i} - " + email["subject"].split(" - ")[-1]
        elif "work" in email["category"]:
            new_email["subject"] = email["subject"].replace("Tomorrow", f"Day {i%7}")
        extended_samples.append(new_email)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extended_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample email data with {len(extended_samples)} emails at {output_file}")


def convert_email_dataset(config: EmailDataProcessConfig):
    """Convert email data to TRM-compatible format"""
    
    np.random.seed(config.seed)
    
    # Load emails
    emails = load_email_data(config)
    
    # Filter categories with minimum examples
    category_counts = Counter(email.category.lower() for email in emails)
    valid_categories = {cat for cat, count in category_counts.items() if count >= config.min_emails_per_category}
    emails = [email for email in emails if email.category.lower() in valid_categories]
    
    print(f"Filtered to {len(emails)} emails with valid categories: {valid_categories}")
    
    # Build vocabulary
    vocab = build_vocabulary(emails)
    
    # Split into train/test
    np.random.shuffle(emails)
    split_idx = int(len(emails) * (1 - config.test_split_ratio))
    train_emails = emails[:split_idx]
    test_emails = emails[split_idx:]
    
    print(f"Split: {len(train_emails)} train, {len(test_emails)} test")
    
    # Process each split
    for split_name, split_emails in [("train", train_emails), ("test", test_emails)]:
        
        # Create augmented dataset for training
        if split_name == "train":
            augmented_emails = []
            for email in split_emails:
                augmented_emails.append(email)  # Original
                for aug_id in range(config.num_aug):
                    augmented_emails.append(augment_email(email, aug_id))
            split_emails = augmented_emails
        
        print(f"Processing {split_name} split with {len(split_emails)} emails")
        
        # Convert to sequences
        inputs = []
        labels = []
        puzzle_identifiers = []
        
        for i, email in enumerate(split_emails):
            input_seq, label = tokenize_email(email, vocab, config.max_seq_len)
            inputs.append(input_seq)
            labels.append(label)
            puzzle_identifiers.append(config.puzzle_identifiers_start + i)
        
        # Create indices (each email is its own puzzle and group)
        puzzle_indices = np.arange(len(inputs) + 1, dtype=np.int32)
        group_indices = np.arange(len(inputs) + 1, dtype=np.int32)
        
        # Save data
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        np.save(os.path.join(split_dir, "all__inputs.npy"), np.array(inputs))
        np.save(os.path.join(split_dir, "all__labels.npy"), np.array(labels, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), np.array(puzzle_identifiers, dtype=np.int32))
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)
        
        # Create metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.max_seq_len,
            vocab_size=len(vocab),
            pad_id=SPECIAL_TOKENS["<PAD>"],
            ignore_label_id=None,
            blank_identifier_id=0,
            num_puzzle_identifiers=len(inputs) + config.puzzle_identifiers_start,
            total_groups=len(inputs),
            mean_puzzle_examples=1.0,
            total_puzzles=len(inputs),
            sets=["all"]
        )
        
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    # Save vocabulary and category mappings
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    
    with open(os.path.join(config.output_dir, "categories.json"), "w") as f:
        json.dump(EMAIL_CATEGORIES, f, indent=2)
    
    # Save identifiers mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        identifiers = ["<blank>"] + [f"email_{i}" for i in range(len(emails))]
        json.dump(identifiers, f, indent=2)
    
    print(f"Email classification dataset created at {config.output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Categories: {list(EMAIL_CATEGORIES.keys())}")


@cli.command(singleton=True)
def main(config: EmailDataProcessConfig):
    convert_email_dataset(config)


if __name__ == "__main__":
    cli()