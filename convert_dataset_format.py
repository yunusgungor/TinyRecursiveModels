#!/usr/bin/env python3
"""
Dataset Format Converter
Genişletilmiş dataset'i training script'in beklediği formata çevirir
"""

import json
import os
from collections import defaultdict

def convert_expanded_to_training_format():
    """Genişletilmiş dataset'i training formatına çevir"""
    
    input_dir = "data/expanded_emails"
    output_dir = "data/training_format"
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Kategorileri topla
    categories = {}
    category_counter = 0
    
    # Vocabulary'yi topla
    vocab_to_id = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    vocab_counter = 4
    
    # Her split için veri topla
    for split in ["train", "test", "val"]:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        emails = []
        
        for filename in os.listdir(split_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(split_dir, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    email = json.load(f)
                
                # Kategoriyi kaydet
                category = email.get('category', 'other')
                if category not in categories:
                    categories[category] = category_counter
                    category_counter += 1
                
                # Vocabulary'yi genişlet
                text = f"{email.get('subject', '')} {email.get('body', '')}"
                words = text.lower().split()
                
                for word in words:
                    if word not in vocab_to_id:
                        vocab_to_id[word] = vocab_counter
                        vocab_counter += 1
                
                # Email'i kaydet
                emails.append({
                    "id": email.get('id', filename.replace('.json', '')),
                    "subject": email.get('subject', ''),
                    "body": email.get('body', ''),
                    "sender": email.get('sender', ''),
                    "category": category,
                    "label": categories[category]
                })
        
        # Split dosyasını kaydet
        if split == "val":
            # Val'i test olarak kaydet
            output_split = "test"
        else:
            output_split = split
            
        output_file = os.path.join(output_dir, output_split, "dataset.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for email in emails:
                f.write(json.dumps(email, ensure_ascii=False) + '\n')
        
        print(f"✅ {split} -> {output_split}: {len(emails)} emails")
    
    # Vocabulary dosyasını kaydet
    vocab_data = {
        "vocab_to_id": vocab_to_id,
        "id_to_vocab": {str(v): k for k, v in vocab_to_id.items()},
        "vocab_size": len(vocab_to_id)
    }
    
    with open(os.path.join(output_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    # Kategoriler dosyasını kaydet
    categories_data = {
        "categories": categories,
        "id_to_category": {str(v): k for k, v in categories.items()},
        "num_categories": len(categories)
    }
    
    with open(os.path.join(output_dir, "categories.json"), 'w', encoding='utf-8') as f:
        json.dump(categories_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Dönüştürme Tamamlandı!")
    print(f"📁 Çıktı: {output_dir}")
    print(f"📚 Vocabulary: {len(vocab_to_id)} words")
    print(f"🏷️ Categories: {len(categories)}")
    
    for category, cat_id in categories.items():
        print(f"  {cat_id}: {category}")
    
    return output_dir

if __name__ == "__main__":
    convert_expanded_to_training_format()