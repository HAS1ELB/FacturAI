#!/usr/bin/env python3
"""
Script d'installation des dépendances pour le module VLM FacturAI
"""

import subprocess
import sys
import os
import importlib

def check_python_version():
    """Vérifie la version de Python"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 ou supérieur requis")
        return False
    print(f"✅ Python {sys.version.split()[0]} détecté")
    return True

def install_package(package, description=""):
    """Installe un package avec pip"""
    try:
        print(f"📦 Installation de {package}... {description}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erreur lors de l'installation de {package}")
        return False

def check_package(package_name, import_name=None):
    """Vérifie si un package est installé"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} est disponible")
        return True
    except ImportError:
        print(f"❌ {package_name} n'est pas installé")
        return False

def install_basic_dependencies():
    """Installe les dépendances de base"""
    print("\n🔧 Installation des dépendances de base...")
    
    basic_packages = [
        ("torch", "Framework PyTorch pour l'IA"),
        ("torchvision", "Utilitaires vision pour PyTorch"),
        ("transformers", "Bibliothèque Hugging Face Transformers"),
        ("pillow", "Traitement d'images Python"),
        ("opencv-python", "Computer Vision OpenCV"),
        ("numpy", "Calculs numériques"),
        ("matplotlib", "Visualisations graphiques"),
        ("pandas", "Manipulation de données"),
    ]
    
    success_count = 0
    for package, description in basic_packages:
        if install_package(package, description):
            success_count += 1
    
    print(f"\n📊 {success_count}/{len(basic_packages)} dépendances de base installées")
    return success_count == len(basic_packages)

def install_vlm_models():
    """Installe les dépendances spécifiques aux modèles VLM"""
    print("\n🤖 Installation des dépendances pour les modèles VLM...")
    
    vlm_packages = [
        ("accelerate", "Accélération d'entraînement Hugging Face"),
        ("sentencepiece", "Tokenisation pour certains modèles"),
        ("protobuf", "Sérialisation de données"),
        ("bitsandbytes", "Optimisation mémoire (optionnel)"),
        ("transformers_stream_generator", "Support pour Qwen-VL (optionnel)"),
    ]
    
    success_count = 0
    for package, description in vlm_packages:
        print(f"\n📦 {package}: {description}")
        if package == "transformers_stream_generator":
            # Package optionnel pour Qwen-VL
            print("  (Optionnel pour Qwen-VL)")
            try:
                if install_package(package):
                    success_count += 1
            except:
                print(f"  ⚠️  {package} non installé (optionnel)")
        elif package == "bitsandbytes":
            # Package optionnel pour l'optimisation
            print("  (Optionnel pour l'optimisation mémoire)")
            try:
                if install_package(package):
                    success_count += 1
            except:
                print(f"  ⚠️  {package} non installé (optionnel)")
        else:
            if install_package(package, description):
                success_count += 1
    
    print(f"\n📊 {success_count}/{len(vlm_packages)} dépendances VLM installées")
    return True  # Certaines sont optionnelles

def verify_installation():
    """Vérifie l'installation en testant les imports"""
    print("\n🔍 Vérification de l'installation...")
    
    # Packages essentiels
    essential_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]
    
    all_good = True
    for package_name, import_name in essential_packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    # Test spécifique pour les modèles VLM
    print("\n🤖 Test des modèles VLM disponibles...")
    
    vlm_models = [
        ("BLIP-2", "transformers", "Blip2Processor"),
        ("LLaVA", "transformers", "LlavaNextProcessor"),
    ]
    
    for model_name, module_name, class_name in vlm_models:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ Support {model_name} disponible")
        except (ImportError, AttributeError):
            print(f"⚠️  Support {model_name} non disponible")
    
    return all_good

def test_vlm_module():
    """Test rapide du module VLM"""
    print("\n🧪 Test du module VLM...")
    
    try:
        # Ajouter le chemin parent pour importer le module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Test d'importation
        from vlm import VLMProcessor
        from vlm.config import vlm_config
        
        print("✅ Module VLM importé avec succès")
        
        # Test d'initialisation
        processor = VLMProcessor()
        available_models = processor.available_models
        
        print(f"✅ Processeur VLM initialisé")
        print(f"📋 Modèles disponibles: {available_models}")
        
        if available_models:
            print("🎉 Module VLM prêt à l'utilisation!")
        else:
            print("⚠️  Aucun modèle VLM disponible (utilisation en mode factice)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test du module VLM: {e}")
        return False

def main():
    """Fonction principale d'installation"""
    print("=" * 60)
    print("🚀 INSTALLATION DU MODULE VLM FACTURAI")
    print("=" * 60)
    
    # Vérification de Python
    if not check_python_version():
        return 1
    
    # Installation des dépendances
    success = True
    
    if not install_basic_dependencies():
        print("⚠️  Certaines dépendances de base ont échoué")
        success = False
    
    if not install_vlm_models():
        print("⚠️  Certaines dépendances VLM ont échoué")
        # Ne pas marquer comme échec car certaines sont optionnelles
    
    # Vérification
    if not verify_installation():
        print("⚠️  Problèmes détectés lors de la vérification")
        success = False
    
    # Test du module
    if not test_vlm_module():
        print("⚠️  Le module VLM n'est pas totalement fonctionnel")
        success = False
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DE L'INSTALLATION")
    print("=" * 60)
    
    if success:
        print("🎉 Installation réussie!")
        print("\nPour utiliser le module VLM:")
        print("1. cd FacturAI/vlm")
        print("2. python quick_test.py")
        print("3. python examples/basic_usage.py")
        
        print("\nDocumentation:")
        print("- README.md : Guide d'utilisation")
        print("- examples/ : Exemples d'utilisation")
        print("- tests/ : Tests unitaires")
        
        return 0
    else:
        print("⚠️  Installation partiellement réussie")
        print("Consultez les erreurs ci-dessus pour résoudre les problèmes.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)