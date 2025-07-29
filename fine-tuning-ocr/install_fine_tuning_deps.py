#!/usr/bin/env python3
"""
Installation automatique des dépendances pour le fine-tuning OCR
"""

import subprocess
import sys
import logging
from typing import List, Dict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package: str, description: str = "") -> bool:
    """Installe un package via pip"""
    try:
        logger.info(f"Installation de {package}... {description}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade"])
        logger.info(f"✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'installation de {package}: {e}")
        return False

def install_pytorch():
    """Installe PyTorch avec support CUDA si disponible"""
    logger.info("🔥 Installation de PyTorch...")
    
    # Détecter si CUDA est disponible
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA détecté, PyTorch déjà installé avec support GPU")
            return True
    except ImportError:
        pass
    
    # Installation de PyTorch
    pytorch_packages = [
        "torch",
        "torchvision", 
        "torchaudio"
    ]
    
    for package in pytorch_packages:
        if not install_package(package, "Framework PyTorch"):
            return False
    
    return True

def install_transformers_stack():
    """Installe la stack Hugging Face Transformers"""
    logger.info("🤗 Installation de Hugging Face Transformers...")
    
    packages = [
        "transformers[torch]",
        "datasets",
        "tokenizers",
        "accelerate", 
        "evaluate",
        "rouge_score",
        "sacrebleu"
    ]
    
    for package in packages:
        if not install_package(package, "Hugging Face ecosystem"):
            return False
    
    return True

def install_ocr_packages():
    """Installe les packages OCR"""
    logger.info("👁️ Installation des packages OCR...")
    
    packages = [
        "easyocr",
        "paddlepaddle", 
        "pytesseract",
        "opencv-python",
        "opencv-contrib-python"
    ]
    
    for package in packages:
        if not install_package(package, "OCR engines"):
            logger.warning(f"⚠️ {package} n'a pas pu être installé - continuez manuellement si nécessaire")
    
    return True

def install_data_science_stack():
    """Installe les packages data science"""
    logger.info("📊 Installation des packages data science...")
    
    packages = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "Pillow",
        "tqdm",
        "jupyter",
        "ipywidgets"
    ]
    
    for package in packages:
        install_package(package, "Data Science")
    
    return True

def install_optional_packages():
    """Installe les packages optionnels"""
    logger.info("⭐ Installation des packages optionnels...")
    
    optional_packages = [
        ("wandb", "Weights & Biases pour le monitoring"),
        ("tensorboard", "TensorBoard pour la visualisation"),
        ("albumentations", "Augmentation d'images avancée"),
        ("timm", "Modèles de vision pré-entraînés"),
        ("sentencepiece", "Tokenisation avancée")
    ]
    
    for package, description in optional_packages:
        install_package(package, description)
    
    return True

def verify_installation():
    """Vérifie que l'installation s'est bien passée"""
    logger.info("🔍 Vérification de l'installation...")
    
    critical_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers", 
        "datasets": "Hugging Face Datasets",
        "easyocr": "EasyOCR",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "pandas": "Pandas"
    }
    
    success_count = 0
    total_count = len(critical_packages)
    
    for package, name in critical_packages.items():
        try:
            if package == "cv2":
                import cv2
            else:
                __import__(package)
            logger.info(f"✅ {name} - OK")
            success_count += 1
        except ImportError:
            logger.error(f"❌ {name} - ÉCHEC")
    
    logger.info(f"\n📊 Résultat: {success_count}/{total_count} packages critiques installés")
    
    if success_count == total_count:
        logger.info("🎉 Installation complète réussie!")
        return True
    else:
        logger.warning("⚠️ Certains packages n'ont pas pu être installés")
        return False

def create_requirements_file():
    """Crée un fichier requirements.txt"""
    logger.info("📝 Création du fichier requirements.txt...")
    
    requirements = """# FacturAI Fine-tuning Requirements
# Core ML/DL packages
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Hugging Face ecosystem
transformers[torch]>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
accelerate>=0.20.0
evaluate>=0.4.0
rouge_score>=0.1.2
sacrebleu>=2.3.0

# OCR packages
easyocr>=1.7.0
paddlepaddle>=2.5.0
paddleocr>=2.7.0
pytesseract>=0.3.10

# Computer vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
Pillow>=9.5.0

# Data science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
jupyter>=1.0.0
ipywidgets>=8.0.0

# Optional monitoring
wandb>=0.15.0
tensorboard>=2.13.0

# Image augmentation
albumentations>=1.3.0

# Additional models
timm>=0.9.0
sentencepiece>=0.1.99

# Development
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
"""

    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    logger.info("✅ requirements.txt créé")

def main():
    """Installation principale"""
    logger.info("🚀 INSTALLATION DES DÉPENDANCES FINE-TUNING OCR")
    logger.info("=" * 60)
    
    steps = [
        ("PyTorch", install_pytorch),
        ("Transformers Stack", install_transformers_stack), 
        ("OCR Packages", install_ocr_packages),
        ("Data Science Stack", install_data_science_stack),
        ("Optional Packages", install_optional_packages)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        logger.info(f"\n📦 {step_name}")
        logger.info("-" * 30)
        
        try:
            if step_func():
                success_count += 1
                logger.info(f"✅ {step_name} - Terminé")
            else:
                logger.warning(f"⚠️ {step_name} - Problèmes détectés")
        except Exception as e:
            logger.error(f"❌ {step_name} - Erreur: {str(e)}")
    
    # Vérification finale
    logger.info("\n" + "=" * 60)
    verification_success = verify_installation()
    
    # Créer requirements.txt
    create_requirements_file()
    
    # Résumé final
    logger.info("\n🎯 RÉSUMÉ DE L'INSTALLATION")
    logger.info("=" * 60)
    logger.info(f"Étapes réussies: {success_count}/{len(steps)}")
    
    if verification_success:
        logger.info("🎉 Installation terminée avec succès!")
        logger.info("\nPROCHAINES ÉTAPES:")
        logger.info("1. Préparez vos données d'entraînement")
        logger.info("2. Configurez fine_tuning_config.json") 
        logger.info("3. Lancez: python fine_tuning_manager.py")
    else:
        logger.warning("⚠️ Installation incomplète")
        logger.info("Installez manuellement les packages manquants avec:")
        logger.info("pip install -r requirements.txt")
    
    logger.info("\n📚 DOCUMENTATION:")
    logger.info("- Guide complet: README_FINE_TUNING.md")
    logger.info("- Config exemple: fine_tuning_config.json")
    logger.info("- Support: Consultez les logs ci-dessus")

if __name__ == "__main__":
    main()