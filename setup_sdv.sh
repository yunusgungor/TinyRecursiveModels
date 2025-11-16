#!/bin/bash
# SDV Kurulum ve Hızlı Başlangıç Scripti

echo "🎁 SDV Kurulum ve Veri Üretimi"
echo "================================"

# Renk kodları
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. SDV'yi kur
echo -e "\n${BLUE}📦 SDV kütüphanesi kuruluyor...${NC}"
pip install sdv pandas -q

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SDV başarıyla kuruldu${NC}"
else
    echo -e "${YELLOW}⚠️  SDV kurulumunda sorun olabilir${NC}"
fi

# 2. Veri klasörünü oluştur
echo -e "\n${BLUE}📁 Veri klasörü oluşturuluyor...${NC}"
mkdir -p data

# 3. Temel veriyi oluştur
echo -e "\n${BLUE}📊 Temel veri oluşturuluyor...${NC}"
python create_gift_data.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Temel veri oluşturuldu${NC}"
else
    echo -e "${YELLOW}⚠️  Temel veri oluşturulamadı${NC}"
    exit 1
fi

# 4. Sentetik veri üret
echo -e "\n${BLUE}🎲 Sentetik veri üretiliyor...${NC}"
python sdv_data_generator.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Sentetik veri üretildi${NC}"
else
    echo -e "${YELLOW}⚠️  Sentetik veri üretilemedi${NC}"
    exit 1
fi

# 5. Özet bilgi
echo -e "\n${GREEN}🎉 Kurulum tamamlandı!${NC}"
echo -e "\n${BLUE}📊 Oluşturulan dosyalar:${NC}"
ls -lh data/*.json

echo -e "\n${BLUE}🚀 Sonraki adımlar:${NC}"
echo "  1. Gelişmiş üretim için: python sdv_advanced_generator.py"
echo "  2. Kılavuzu okuyun: cat SDV_KULLANIM_KILAVUZU.md"
echo "  3. Model eğitimi için sentetik veriyi kullanın"

echo -e "\n${GREEN}✨ Başarılar!${NC}"
