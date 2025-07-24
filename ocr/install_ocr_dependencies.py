#!/usr/bin/env python3
"""
Script d'installation des dépendances OCR pour FacturAI
Installe automatiquement les moteurs OCR disponibles
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n🔧 {description}")
    print(f"Commande: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Succès: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur: {description}")
        print(f"Message d'erreur: {e.stderr}")
        return False

def install_tesseract():
    """Installe Tesseract OCR"""
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y tesseract-ocr",
            "sudo apt-get install -y tesseract-ocr-fra",  # Français
            "sudo apt-get install -y tesseract-ocr-ara",  # Arabe (pour le Maroc)
            "sudo apt-get install -y libtesseract-dev"
        ]
        
        print("🐧 Installation Tesseract pour Linux...")
        for cmd in commands:
            run_command(cmd, f"Installation: {cmd}")
            
    elif system == "darwin":  # macOS
        print("🍎 Installation Tesseract pour macOS...")
        run_command("brew install tesseract", "Installation Tesseract via Homebrew")
        run_command("brew install tesseract-lang", "Installation langues supplémentaires")
        
    elif system == "windows":
        print("🪟 Pour Windows:")
        print("1. Téléchargez Tesseract depuis: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Installez le fichier .exe")
        print("3. Ajoutez le chemin Tesseract à votre PATH")
        print("   Exemple: C:\\Program Files\\Tesseract-OCR")
    
    # Installer le package Python
    run_command(f"{sys.executable} -m pip install pytesseract", 
                "Installation pytesseract (Python wrapper)")

def install_easyocr():
    """Installe EasyOCR"""
    print("\n📱 Installation EasyOCR...")
    
    # Installer EasyOCR
    success = run_command(f"{sys.executable} -m pip install easyocr", 
                         "Installation EasyOCR")
    
    if success:
        # Installer les dépendances supplémentaires si nécessaire
        run_command(f"{sys.executable} -m pip install opencv-python", 
                   "Installation OpenCV (requis pour EasyOCR)")

def install_paddleocr():
    """Installe PaddleOCR"""
    print("\n🏓 Installation PaddleOCR...")
    
    # Détecter si GPU disponible
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    if gpu_available:
        print("🚀 GPU détecté - Installation version GPU")
        run_command(f"{sys.executable} -m pip install paddlepaddle-gpu", 
                   "Installation PaddlePaddle GPU")
    else:
        print("🖥️ Installation version CPU")
        run_command(f"{sys.executable} -m pip install paddlepaddle", 
                   "Installation PaddlePaddle CPU")
    
    # Installer PaddleOCR
    run_command(f"{sys.executable} -m pip install paddleocr", 
               "Installation PaddleOCR")

def install_common_dependencies():
    """Installe les dépendances communes"""
    print("\n📦 Installation des dépendances communes...")
    
    dependencies = [
        "opencv-python",
        "numpy",
        "pillow",
        "matplotlib",
        "pathlib",
        "typing-extensions"
    ]
    
    for dep in dependencies:
        run_command(f"{sys.executable} -m pip install {dep}", 
                   f"Installation {dep}")

def test_installations():
    """Teste les installations"""
    print("\n🧪 TEST DES INSTALLATIONS")
    print("=" * 40)
    
    # Test Tesseract
    try:
        import pytesseract
        from PIL import Image
        print("✅ Tesseract: Disponible")
        
        # Test de version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   Version: {version}")
        except:
            print("   Version: Non détectable")
            
    except ImportError:
        print("❌ Tesseract: Non disponible")
    
    # Test EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR: Disponible")
    except ImportError:
        print("❌ EasyOCR: Non disponible")
    
    # Test PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR: Disponible")
    except ImportError:
        print("❌ PaddleOCR: Non disponible")
    
    print("=" * 40)

def create_test_structure():
    """Crée la structure de dossiers pour les tests"""
    print("\n📁 Création de la structure de dossiers...")
    
    directories = [
        "Data",
        "Data/processed_images",
        "Data/ocr_results",
        "Data/test_images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Dossier créé: {directory}")

def main():
    """Fonction principale"""
    print("🚀 INSTALLATION AUTOMATIQUE DES DÉPENDANCES OCR")
    print("=" * 60)
    print("Ce script va installer les moteurs OCR suivants:")
    print("- Tesseract OCR (+ langues FR/AR)")
    print("- EasyOCR")
    print("- PaddleOCR")
    print("=" * 60)
    
    response = input("\nContinuer l'installation? (y/N): ").lower().strip()
    
    if response not in ['y', 'yes', 'oui']:
        print("❌ Installation annulée")
        return
    
    print("\n🔧 DÉBUT DE L'INSTALLATION...")
    
    # Mise à jour pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", 
               "Mise à jour de pip")
    
    # Installation des dépendances communes
    install_common_dependencies()
    
    # Installation des moteurs OCR
    print("\n" + "=" * 60)
    print("INSTALLATION DES MOTEURS OCR")
    print("=" * 60)
    
    # Tesseract
    install_tesseract()
    
    # EasyOCR
    install_easyocr()
    
    # PaddleOCR
    install_paddleocr()
    
    # Création de la structure
    create_test_structure()
    
    # Tests finaux
    test_installations()
    
    print("\n🎉 INSTALLATION TERMINÉE!")
    print("=" * 60)
    print("Prochaines étapes:")
    print("1. Placez vos images prétraitées dans: Data/processed_images/")
    print("2. Exécutez: python ocr_starter.py")
    print("3. Vérifiez les résultats dans: Data/ocr_results/")
    print("=" * 60)

if __name__ == "__main__":
    main()