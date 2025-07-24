#!/usr/bin/env python3
"""
Script de validation pour vérifier que tout est prêt pour l'OCR FacturAI
"""

import os
import sys
from pathlib import Path
import json

def check_directory_structure():
    """Vérifie la structure des dossiers"""
    print("📁 VÉRIFICATION DE LA STRUCTURE DES DOSSIERS")
    print("-" * 50)
    
    required_dirs = [
        "Data",
        "Data/processed_images",
        "Data/ocr_results"
    ]
    
    issues = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory} - Existe")
            
            # Vérifier les permissions
            if os.access(directory, os.R_OK | os.W_OK):
                print(f"   📝 Permissions lecture/écriture: OK")
            else:
                print(f"   ❌ Permissions lecture/écriture: MANQUANTES")
                issues.append(f"Permissions manquantes pour {directory}")
        else:
            print(f"❌ {directory} - MANQUANT")
            issues.append(f"Dossier manquant: {directory}")
            
            # Créer le dossier automatiquement
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   ✅ Dossier créé automatiquement")
            except Exception as e:
                print(f"   ❌ Impossible de créer le dossier: {e}")
    
    return issues

def check_processed_images():
    """Vérifie les images prétraitées"""
    print("\n🖼️ VÉRIFICATION DES IMAGES PRÉTRAITÉES")
    print("-" * 50)
    
    processed_dir = "Data/processed_images"
    issues = []
    
    if not os.path.exists(processed_dir):
        issues.append("Dossier processed_images manquant")
        return issues
    
    # Extensions d'images supportées
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(processed_dir).glob(f'*{ext}'))
        image_files.extend(Path(processed_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans {processed_dir}")
        print(f"   Extensions supportées: {', '.join(image_extensions)}")
        issues.append("Aucune image prétraitée trouvée")
    else:
        print(f"✅ {len(image_files)} images trouvées")
        
        # Vérifier quelques échantillons
        sample_size = min(3, len(image_files))
        print(f"📋 Échantillon de {sample_size} images:")
        
        for i, img_file in enumerate(image_files[:sample_size]):
            try:
                file_size = os.path.getsize(img_file)
                print(f"   {i+1}. {img_file.name} ({file_size // 1024} KB)")
                
                # Vérification basique avec PIL
                try:
                    from PIL import Image
                    with Image.open(img_file) as img:
                        width, height = img.size
                        print(f"      Dimensions: {width}x{height}")
                        
                        if width < 800 or height < 600:
                            print(f"      ⚠️ Résolution faible (recommandé: >800x600)")
                            
                except ImportError:
                    print(f"      ⚠️ Impossible de vérifier les dimensions (PIL non installé)")
                except Exception as e:
                    print(f"      ❌ Erreur lecture image: {e}")
                    issues.append(f"Image corrompue: {img_file.name}")
                    
            except Exception as e:
                print(f"   ❌ Erreur accès fichier {img_file.name}: {e}")
                issues.append(f"Impossible d'accéder à {img_file.name}")
    
    return issues

def check_ocr_engines():
    """Vérifie les moteurs OCR disponibles"""
    print("\n🔍 VÉRIFICATION DES MOTEURS OCR")
    print("-" * 50)
    
    engines_status = {}
    issues = []
    
    # Tesseract
    print("🔧 Tesseract OCR:")
    try:
        import pytesseract
        from PIL import Image
        engines_status['tesseract'] = True
        print("   ✅ pytesseract installé")
        
        # Test de version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"   📌 Version: {version}")
        except Exception as e:
            print(f"   ⚠️ Impossible de déterminer la version: {e}")
            
        # Test basique
        try:
            # Créer une image de test simple
            test_img = Image.new('RGB', (200, 50), color='white')
            test_result = pytesseract.image_to_string(test_img)
            print("   ✅ Test fonctionnel: OK")
        except Exception as e:
            print(f"   ❌ Test fonctionnel: ÉCHEC - {e}")
            engines_status['tesseract'] = False
            issues.append("Tesseract non fonctionnel")
            
    except ImportError:
        engines_status['tesseract'] = False
        print("   ❌ pytesseract NON installé")
        issues.append("pytesseract manquant")
    
    # EasyOCR
    print("\n📱 EasyOCR:")
    try:
        import easyocr
        engines_status['easyocr'] = True
        print("   ✅ easyocr installé")
        
        # Test d'initialisation
        try:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("   ✅ Initialisation: OK")
        except Exception as e:
            print(f"   ❌ Initialisation: ÉCHEC - {e}")
            engines_status['easyocr'] = False
            issues.append("EasyOCR non fonctionnel")
            
    except ImportError:
        engines_status['easyocr'] = False
        print("   ❌ easyocr NON installé")
        issues.append("easyocr manquant")
    
    # PaddleOCR
    print("\n🏓 PaddleOCR:")
    try:
        from paddleocr import PaddleOCR
        engines_status['paddleocr'] = True
        print("   ✅ paddleocr installé")
        
        # Test d'initialisation
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("   ✅ Initialisation: OK")
        except Exception as e:
            print(f"   ❌ Initialisation: ÉCHEC - {e}")
            engines_status['paddleocr'] = False
            issues.append("PaddleOCR non fonctionnel")
            
    except ImportError:
        engines_status['paddleocr'] = False
        print("   ❌ paddleocr NON installé")
        issues.append("paddleocr manquant")
    
    # Résumé
    available_engines = [name for name, status in engines_status.items() if status]
    
    if not available_engines:
        issues.append("Aucun moteur OCR fonctionnel")
        print("\n❌ AUCUN moteur OCR disponible!")
    else:
        print(f"\n✅ Moteurs OCR disponibles: {', '.join(available_engines)}")
    
    return issues, engines_status

def check_python_dependencies():
    """Vérifie les dépendances Python"""
    print("\n🐍 VÉRIFICATION DES DÉPENDANCES PYTHON")
    print("-" * 50)
    
    required_packages = [
        'opencv-python',
        'numpy',
        'pillow',
        'pathlib'
    ]
    
    optional_packages = [
        'matplotlib',
        'json',
        'datetime'
    ]
    
    issues = []
    
    print("📦 Dépendances requises:")
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"   ✅ opencv-python (cv2)")
            elif package == 'pillow':
                from PIL import Image
                print(f"   ✅ pillow (PIL)")
            else:
                __import__(package)
                print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MANQUANT")
            issues.append(f"Dépendance manquante: {package}")
    
    print("\n📦 Dépendances optionnelles:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ⚠️ {package} - manquant (optionnel)")
    
    return issues

def generate_config_file(engines_status):
    """Génère un fichier de configuration basé sur l'environnement"""
    print("\n⚙️ GÉNÉRATION DU FICHIER DE CONFIGURATION")
    print("-" * 50)
    
    config = {
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        },
        "ocr_engines": engines_status,
        "directories": {
            "input": "Data/processed_images",
            "output": "Data/ocr_results"
        },
        "default_settings": {
            "preferred_engine": "auto",
            "confidence_threshold": 30,
            "languages": ["fr", "en"],
            "output_format": "json"
        }
    }
    
    # Choisir le moteur préféré
    if engines_status.get('easyocr', False):
        config["default_settings"]["preferred_engine"] = "easyocr"
    elif engines_status.get('paddleocr', False):
        config["default_settings"]["preferred_engine"] = "paddleocr"
    elif engines_status.get('tesseract', False):
        config["default_settings"]["preferred_engine"] = "tesseract"
    
    # Sauvegarder la configuration
    config_path = "facturai_config.json"
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration sauvegardée: {config_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde configuration: {e}")

def main():
    """Validation complète de l'environnement"""
    print("🚀 VALIDATION DE L'ENVIRONNEMENT FACTURAI")
    print("=" * 60)
    print("Vérification que tout est prêt pour l'extraction OCR...")
    print("=" * 60)
    
    all_issues = []
    
    # 1. Structure des dossiers
    issues = check_directory_structure()
    all_issues.extend(issues)
    
    # 2. Images prétraitées
    issues = check_processed_images()
    all_issues.extend(issues)
    
    # 3. Moteurs OCR
    issues, engines_status = check_ocr_engines()
    all_issues.extend(issues)
    
    # 4. Dépendances Python
    issues = check_python_dependencies()
    all_issues.extend(issues)
    
    # 5. Génération de la configuration
    generate_config_file(engines_status)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE LA VALIDATION")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 SUCCÈS: Environnement prêt pour l'OCR!")
        print("\n📋 Prochaines étapes:")
        print("1. Exécutez: python ocr_starter.py")
        print("2. Vérifiez les résultats dans: Data/ocr_results/")
        print("3. Consultez: quick_start_guide.md")
        
        return 0
    else:
        print(f"❌ {len(all_issues)} problème(s) détecté(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n🔧 ACTIONS RECOMMANDÉES:")
        print("1. Exécutez: python install_ocr_dependencies.py")
        print("2. Placez vos images dans: Data/processed_images/")
        print("3. Relancez cette validation")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)