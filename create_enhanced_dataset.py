#!/usr/bin/env python3
"""
Enhanced Email Dataset Creator

Creates a larger, more diverse email dataset for training the email classifier.
This addresses the issue of insufficient training data (only 5 emails in original dataset).
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

def create_enhanced_email_dataset(output_path: str, emails_per_category: int = 20) -> bool:
    """Create an enhanced email dataset with more samples per category."""
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Email categories (matching the original structure)
    categories = {
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
    
    # Enhanced email templates for each category
    email_templates = {
        "newsletter": [
            {"subject": "Weekly Newsletter - Tech Updates", "body": "Here are this week's top technology news and updates from our team.", "sender": "newsletter@techcompany.com"},
            {"subject": "Monthly Industry Report", "body": "Our comprehensive analysis of industry trends and market insights for this month.", "sender": "reports@industry.com"},
            {"subject": "Breaking News Alert", "body": "Important breaking news in the technology sector that affects our industry.", "sender": "alerts@newstech.com"},
            {"subject": "Weekly Digest - AI Advances", "body": "Latest developments in artificial intelligence and machine learning research.", "sender": "ai-digest@research.org"},
            {"subject": "Market Analysis Newsletter", "body": "Weekly market analysis and financial insights for technology investors.", "sender": "market@fintech.com"},
        ],
        "work": [
            {"subject": "Meeting Tomorrow at 2 PM", "body": "Don't forget about our project meeting tomorrow at 2 PM in conference room A.", "sender": "manager@company.com"},
            {"subject": "Project Deadline Reminder", "body": "This is a reminder that the project deadline is approaching next Friday.", "sender": "pm@company.com"},
            {"subject": "Team Building Event", "body": "Join us for our quarterly team building event this Thursday at 5 PM.", "sender": "hr@company.com"},
            {"subject": "Budget Review Meeting", "body": "Please attend the budget review meeting scheduled for Monday morning.", "sender": "finance@company.com"},
            {"subject": "Performance Review Schedule", "body": "Your annual performance review has been scheduled for next week.", "sender": "hr@company.com"},
        ],
        "personal": [
            {"subject": "Happy Birthday!", "body": "Wishing you a very happy birthday! Hope you have a wonderful day.", "sender": "friend@personal.com"},
            {"subject": "Dinner Plans Tonight", "body": "Are we still on for dinner tonight at 7 PM? Let me know if you need to reschedule.", "sender": "sarah@gmail.com"},
            {"subject": "Family Reunion Update", "body": "Here are the latest details about our family reunion planned for next month.", "sender": "mom@family.com"},
            {"subject": "Weekend Plans", "body": "What are your plans for the weekend? Want to catch up over coffee?", "sender": "john@personal.com"},
            {"subject": "Congratulations on Your Promotion", "body": "I heard about your promotion! Congratulations, you deserve it.", "sender": "colleague@friend.com"},
        ],
        "spam": [
            {"subject": "Congratulations! You've Won!", "body": "Click here to claim your prize! Limited time offer!", "sender": "noreply@suspicious.com"},
            {"subject": "Urgent: Verify Your Account", "body": "Your account will be suspended unless you verify immediately.", "sender": "security@fake-bank.com"},
            {"subject": "Make Money Fast!", "body": "Earn thousands from home with this simple trick! No experience needed!", "sender": "money@scam.com"},
            {"subject": "You Have Inherited Money", "body": "You have inherited a large sum of money from a distant relative.", "sender": "lawyer@fake.com"},
            {"subject": "Free Gift Card Waiting", "body": "Claim your free $500 gift card now! Limited time offer expires soon!", "sender": "gifts@spam.com"},
        ],
        "promotional": [
            {"subject": "50% Off Sale - Limited Time", "body": "Don't miss our biggest sale of the year! 50% off all items.", "sender": "sales@retailstore.com"},
            {"subject": "New Product Launch", "body": "Be the first to try our revolutionary new product with exclusive early access.", "sender": "marketing@techstore.com"},
            {"subject": "Flash Sale - 24 Hours Only", "body": "Flash sale starts now! Up to 70% off selected items for 24 hours only.", "sender": "deals@fashion.com"},
            {"subject": "Exclusive Member Discount", "body": "As a valued member, enjoy 30% off your next purchase with code MEMBER30.", "sender": "vip@store.com"},
            {"subject": "Black Friday Preview", "body": "Get early access to our Black Friday deals before anyone else!", "sender": "blackfriday@retail.com"},
        ],
        "social": [
            {"subject": "You Have New Messages", "body": "You have 3 new messages waiting for you on our platform.", "sender": "notifications@social.com"},
            {"subject": "Friend Request Received", "body": "John Smith has sent you a friend request. Accept or decline?", "sender": "friends@network.com"},
            {"subject": "Event Invitation", "body": "You're invited to Sarah's birthday party this Saturday at 8 PM.", "sender": "events@social.com"},
            {"subject": "Photo Tagged", "body": "You've been tagged in 5 new photos. Check them out now!", "sender": "photos@social.com"},
            {"subject": "Group Activity Update", "body": "There's been new activity in the 'Tech Enthusiasts' group you follow.", "sender": "groups@social.com"},
        ],
        "finance": [
            {"subject": "Monthly Statement Available", "body": "Your monthly bank statement is now available for download.", "sender": "statements@bank.com"},
            {"subject": "Payment Due Reminder", "body": "Your credit card payment of $250.00 is due in 3 days.", "sender": "billing@creditcard.com"},
            {"subject": "Investment Portfolio Update", "body": "Your investment portfolio has gained 2.5% this month. View details.", "sender": "portfolio@investment.com"},
            {"subject": "Transaction Alert", "body": "A transaction of $150.00 was made on your account today at 2:30 PM.", "sender": "alerts@bank.com"},
            {"subject": "Tax Document Ready", "body": "Your annual tax documents are ready for download from your account.", "sender": "tax@financial.com"},
        ],
        "travel": [
            {"subject": "Flight Confirmation", "body": "Your flight from NYC to LAX on March 15th has been confirmed. Check-in opens 24 hours before.", "sender": "bookings@airline.com"},
            {"subject": "Hotel Reservation Confirmed", "body": "Your reservation at Grand Hotel for March 15-18 has been confirmed.", "sender": "reservations@hotel.com"},
            {"subject": "Travel Insurance Reminder", "body": "Don't forget to purchase travel insurance for your upcoming trip.", "sender": "insurance@travel.com"},
            {"subject": "Boarding Pass Ready", "body": "Your boarding pass is ready for download. Flight departs in 24 hours.", "sender": "checkin@airline.com"},
            {"subject": "Travel Itinerary Update", "body": "Your travel itinerary has been updated with new flight times.", "sender": "updates@travelagency.com"},
        ],
        "shopping": [
            {"subject": "Order Confirmation", "body": "Your order #12345 has been confirmed and will ship within 2 business days.", "sender": "orders@shop.com"},
            {"subject": "Shipping Update", "body": "Your package is out for delivery and should arrive today between 2-6 PM.", "sender": "shipping@logistics.com"},
            {"subject": "Cart Abandonment", "body": "You left items in your cart! Complete your purchase now and get free shipping.", "sender": "cart@ecommerce.com"},
            {"subject": "Product Recommendation", "body": "Based on your recent purchases, we think you'll love these new arrivals.", "sender": "recommendations@store.com"},
            {"subject": "Return Processed", "body": "Your return has been processed and refund will appear in 3-5 business days.", "sender": "returns@shop.com"},
        ],
        "other": [
            {"subject": "System Maintenance Notice", "body": "Scheduled system maintenance will occur tonight from 2-4 AM EST.", "sender": "admin@system.com"},
            {"subject": "Survey Request", "body": "Help us improve our service by taking this 5-minute customer satisfaction survey.", "sender": "survey@feedback.com"},
            {"subject": "Password Reset Request", "body": "A password reset was requested for your account. Click here to reset.", "sender": "security@service.com"},
            {"subject": "Newsletter Subscription Confirmed", "body": "Thank you for subscribing to our newsletter. Welcome aboard!", "sender": "welcome@newsletter.com"},
            {"subject": "Account Verification", "body": "Please verify your email address to complete your account setup.", "sender": "verify@platform.com"},
        ]
    }
    
    # Generate emails for training
    train_emails = []
    test_emails = []
    
    email_id = 1
    
    for category, templates in email_templates.items():
        for i in range(emails_per_category):
            # Select template (cycle through available templates)
            template = templates[i % len(templates)]
            
            # Add some variation to avoid exact duplicates
            variations = [
                "", " - Update", " - Important", " - Reminder", " - Notice"
            ]
            
            email = {
                "id": f"email_{email_id:03d}",
                "subject": template["subject"] + (variations[i % len(variations)] if i > 0 else ""),
                "body": template["body"],
                "sender": template["sender"],
                "recipient": "user@example.com",
                "category": category,
                "language": "en"
            }
            
            # 80% for training, 20% for testing
            if i < int(emails_per_category * 0.8):
                train_emails.append(email)
            else:
                test_emails.append(email)
            
            email_id += 1
    
    # Shuffle the emails
    random.shuffle(train_emails)
    random.shuffle(test_emails)
    
    # Create train dataset (JSONL format)
    train_dir = output_path / "train"
    train_dir.mkdir(exist_ok=True)
    
    with open(train_dir / "dataset.json", "w") as f:
        for email in train_emails:
            json.dump(email, f)
            f.write('\n')
    
    # Create test dataset (JSONL format)
    test_dir = output_path / "test"
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / "dataset.json", "w") as f:
        for email in test_emails:
            json.dump(email, f)
            f.write('\n')
    
    # Create categories file
    with open(output_path / "categories.json", "w") as f:
        json.dump(categories, f, indent=2)
    
    # Create enhanced vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    word_id = 4
    
    all_emails = train_emails + test_emails
    for email in all_emails:
        text = f"{email['subject']} {email['body']}"
        words = text.lower().replace("!", "").replace("?", "").replace(".", "").replace(",", "").split()
        for word in words:
            if word and word not in vocab:
                vocab[word] = word_id
                word_id += 1
    
    with open(output_path / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Enhanced dataset created at: {output_path}")
    print(f"  - {len(train_emails)} training emails")
    print(f"  - {len(test_emails)} test emails")
    print(f"  - {len(categories)} categories")
    print(f"  - {len(vocab)} vocabulary tokens")
    print(f"  - {emails_per_category} emails per category")
    
    # Print category distribution
    print("\nCategory distribution:")
    for category in categories:
        train_count = sum(1 for email in train_emails if email['category'] == category)
        test_count = sum(1 for email in test_emails if email['category'] == category)
        print(f"  {category}: {train_count} train, {test_count} test")
    
    return True

if __name__ == "__main__":
    create_enhanced_email_dataset("./enhanced_emails", emails_per_category=20)