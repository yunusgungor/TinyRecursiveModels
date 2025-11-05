#!/usr/bin/env python3
"""
Training Configuration Optimizer
Bellek kısıtlamalarına uygun optimal config oluşturur
"""

import json
import os
import psutil
from typing import Dict

def get_system_memory() -> float:
    """Sistem bellek bilgisini al (GB)"""
    memory = psutil.virtual_memory()
    return memory.total / (1024**3)

def create_optimized_config(available_memory_gb: float = None) -> Dict:
    """Bellek durumuna göre optimize edilmiş config oluştur"""
    
    if available_memory_gb is None:
        available_memory_gb = get_system_memory()
    
    print(f"💾 Sistem belleği: {available_memory_gb:.1f} GB")
    
    # Bellek durumuna göre config ayarla
    if available_memory_gb >= 16:
        # Yüksek bellek - Full model
        config = {
            "model_name": "EmailTRM_Full",
            "vocab_size": 10000,
            "hidden_size": 768,
            "num_layers": 6,
            "num_email_categories": 10,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_sequence_length": 512,
            "use_hierarchical_attention": True,
            "enable_content_features": True,
            "enable_subject_prioritization": True,
            "memory_limit_mb": int(available_memory_gb * 1024 * 0.7)
        }
        print("🚀 Yüksek performans konfigürasyonu")
        
    elif available_memory_gb >= 8:
        # Orta bellek - Balanced model
        config = {
            "model_name": "EmailTRM_Balanced",
            "vocab_size": 5000,
            "hidden_size": 512,
            "num_layers": 4,
            "num_email_categories": 10,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_sequence_length": 384,
            "use_hierarchical_attention": True,
            "enable_content_features": True,
            "enable_subject_prioritization": True,
            "memory_limit_mb": int(available_memory_gb * 1024 * 0.6)
        }
        print("⚖️ Dengeli konfigürasyon")
        
    elif available_memory_gb >= 4:
        # Düşük bellek - Efficient model
        config = {
            "model_name": "EmailTRM_Efficient",
            "vocab_size": 3000,
            "hidden_size": 384,
            "num_layers": 3,
            "num_email_categories": 10,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_sequence_length": 256,
            "use_hierarchical_attention": False,
            "enable_content_features": True,
            "enable_subject_prioritization": False,
            "memory_limit_mb": int(available_memory_gb * 1024 * 0.5)
        }
        print("💡 Verimli konfigürasyon")
        
    else:
        # Çok düşük bellek - Minimal model
        config = {
            "model_name": "EmailTRM_Minimal",
            "vocab_size": 2000,
            "hidden_size": 256,
            "num_layers": 2,
            "num_email_categories": 10,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_sequence_length": 128,
            "use_hierarchical_attention": False,
            "enable_content_features": False,
            "enable_subject_prioritization": False,
            "memory_limit_mb": int(available_memory_gb * 1024 * 0.4)
        }
        print("🔧 Minimal konfigürasyon")
    
    # Ortak ayarlar
    config.update({
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": 20,
        "max_steps": 10000,
        "use_lr_scheduler": True,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 500,
        "use_email_structure": True,
        "subject_attention_weight": 2.0,
        "pooling_strategy": "weighted",
        "min_token_frequency": 2,
        "special_token_ratio": 0.1,
        "domain_feature_weight": 1.5,
        "content_feature_weight": 1.2,
        "email_augmentation_prob": 0.2,
        "category_balancing": True,
        "cross_validation_folds": 5,
        "enable_memory_monitoring": True,
        "dynamic_batch_sizing": True,
        "use_cpu_optimization": True,
        "num_workers": min(4, os.cpu_count()),
        "target_accuracy": 0.85,  # Daha gerçekçi hedef
        "min_category_accuracy": 0.7,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "classification_loss_weight": 1.0,
        "halt_loss_weight": 0.01,
        "contrastive_loss_weight": 0.1,
        "calibration_loss_weight": 0.05,
        "max_grad_norm": 1.0,
        "log_interval": 100,
        "eval_interval": 500,
        "enable_sender_analysis": True
    })
    
    return config

def fix_omp_warnings():
    """OpenMP uyarılarını düzelt"""
    
    # Environment variables to fix OMP warnings
    omp_fixes = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1'
    }
    
    print("🔧 OpenMP uyarıları düzeltiliyor...")
    
    for key, value in omp_fixes.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    
    # Create a shell script for future runs
    with open('fix_omp.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# OpenMP Warning Fix\n")
        for key, value in omp_fixes.items():
            f.write(f"export {key}={value}\n")
        f.write("echo '✅ OpenMP environment variables set'\n")
    
    os.chmod('fix_omp.sh', 0o755)
    print("📝 fix_omp.sh scripti oluşturuldu")

def main():
    print("⚙️ Training Configuration Optimizer")
    print("=" * 50)
    
    # Sistem bilgilerini al
    memory_gb = get_system_memory()
    cpu_count = os.cpu_count()
    
    print(f"🖥️ Sistem Bilgileri:")
    print(f"  💾 RAM: {memory_gb:.1f} GB")
    print(f"  🔧 CPU: {cpu_count} cores")
    
    # Optimize edilmiş config oluştur
    config = create_optimized_config(memory_gb)
    
    # Config'i kaydet
    config_path = "optimized_training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Optimize edilmiş config kaydedildi: {config_path}")
    
    # OpenMP uyarılarını düzelt
    fix_omp_warnings()
    
    # Özet
    print(f"\n📊 Konfigürasyon Özeti:")
    print(f"  🏷️ Model: {config['model_name']}")
    print(f"  📚 Vocab: {config['vocab_size']}")
    print(f"  🧠 Hidden: {config['hidden_size']}")
    print(f"  📏 Layers: {config['num_layers']}")
    print(f"  📦 Batch: {config['batch_size']}")
    print(f"  📐 Max Length: {config['max_sequence_length']}")
    print(f"  💾 Memory Limit: {config['memory_limit_mb']} MB")
    
    print(f"\n🚀 Kullanım:")
    print(f"  1. source fix_omp.sh")
    print(f"  2. python run_optimized_training.py --config {config_path}")

if __name__ == "__main__":
    main()