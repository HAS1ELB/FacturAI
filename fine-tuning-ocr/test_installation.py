#!/usr/bin/env python3
"""
🧪 Test d'Installation - FacturAI Fine-Tuning
Vérifie que tous les composants sont correctement installés et fonctionnels
"""

import sys
import os
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_python_version():
    """Test de la version Python"""
    logger.info("🐍 Test version Python...")
    
    if sys.version_info >= (3, 8):
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} OK")
        return True
    else:
        logger.error(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} trop ancien (requis: 3.8+)")
        return False

def test_core_dependencies():
    """Test des dépendances principales"""
    logger.info("📦 Test des dépendances principales...")
    
    dependencies = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn'
    }
    
    failed = []
    
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package} OK")
        except ImportError:
            logger.error(f"❌ {package} manquant")
            failed.append(package)
    
    return len(failed) == 0

def test_ml_dependencies():
    """Test des dépendances ML/Deep Learning"""
    logger.info("🤖 Test des dépendances ML...")
    
    # PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} OK")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 GPU disponible: {gpu_name}")
        else:
            logger.warning("⚠️ Pas de GPU CUDA disponible (CPU seulement)")
    except ImportError:
        logger.error("❌ PyTorch manquant")
        return False
    
    # Transformers
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__} OK")
    except ImportError:
        logger.error("❌ Transformers manquant")
        return False
    
    return True

def test_ocr_dependencies():
    """Test des dépendances OCR"""
    logger.info("👁️ Test des dépendances OCR...")
    
    # EasyOCR
    try:
        import easyocr
        logger.info("✅ EasyOCR OK")
    except ImportError:
        logger.error("❌ EasyOCR manquant")
        return False
    
    # PaddleOCR
    try:
        from paddleocr import PaddleOCR
        logger.info("✅ PaddleOCR OK")
    except ImportError:
        logger.warning("⚠️ PaddleOCR manquant (optionnel)")
    
    # Tesseract
    try:
        import pytesseract
        logger.info("✅ Pytesseract OK")
    except ImportError:
        logger.warning("⚠️ Pytesseract manquant (optionnel)")
    
    return True

def test_data_structure():
    """Test de la structure des données"""
    logger.info("📁 Test de la structure des données...")
    
    # Vérifier les dossiers requis
    required_dirs = [
        "Data/processed_images",
        "Data/ocr_results"
    ]
    
    issues = []
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            issues.append(f"Dossier manquant: {dir_path}")
            logger.warning(f"⚠️ {dir_path} n'existe pas")
        else:
            logger.info(f"✅ {dir_path} OK")
            
            # Compter les fichiers
            if "images" in dir_path:
                images = list(path.glob("*.png")) + list(path.glob("*.jpg"))
                logger.info(f"  📷 {len(images)} images trouvées")
            elif "ocr_results" in dir_path:
                json_files = list(path.glob("*.json"))
                logger.info(f"  📄 {len(json_files)} fichiers JSON trouvés")
    
    return len(issues) == 0

def test_module_imports():
    """Test d'import des modules du projet"""
    logger.info("🔧 Test des modules du projet...")
    
    modules = [
        'fine-tuning-ocr/data_preparation/data_preparation',
        'fine-tuning-ocr/fine_tuning_manager/fine_tuning_manager',
        'fine-tuning-ocr/evaluation/model_evaluation',
        'fine-tuning-ocr/fine_tuning_model/easyocr_finetuning',
        'fine-tuning-ocr/fine_tuning_model/trocr_finetuning'
    ]
    
    failed = []
    
    for module in modules:
        try:
            if Path(f"{module}.py").exists():
                # Test syntaxique seulement
                with open(f"{module}.py", 'r', encoding='utf-8') as f:
                    compile(f.read(), f"{module}.py", 'exec')
                logger.info(f"✅ {module}.py OK")
            else:
                logger.warning(f"⚠️ {module}.py non trouvé")
                failed.append(module)
        except SyntaxError as e:
            logger.error(f"❌ {module}.py erreur syntaxe: {e}")
            failed.append(module)
        except Exception as e:
            logger.error(f"❌ {module}.py erreur: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_quick_functionality():
    """Test rapide de fonctionnalité"""
    logger.info("⚡ Test de fonctionnalité rapide...")
    
    try:
        # Test création d'un dataset factice
        from data_preparation.data_preparation import InvoiceDataPreparator
        
        # Créer un dossier de test temporaire
        test_dir = Path("test_temp")
        test_dir.mkdir(exist_ok=True)
        
        # Test d'initialisation
        preparator = InvoiceDataPreparator(
            images_dir="Data/processed_images",
            ocr_results_dir="Data/ocr_results", 
            output_dir=str(test_dir)
        )
        
        logger.info("✅ Module data_preparation fonctionnel")
        
        # Nettoyer
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test fonctionnel échoué: {e}")
        return False

def test_gpu_functionality():
    """Test spécifique du GPU pour l'entraînement"""
    logger.info("🎮 Test de fonctionnalité GPU...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA non disponible - CPU seulement")
            return True
        
        # Test simple GPU
        device = torch.device('cuda')
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        
        logger.info(f"✅ GPU fonctionnel: {torch.cuda.get_device_name(0)}")
        logger.info(f"📊 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test GPU échoué: {e}")
        return False

def generate_installation_report():
    """Génère un rapport d'installation"""
    logger.info("📊 Génération du rapport d'installation...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "tests": {}
    }
    
    # Exécuter tous les tests
    tests = [
        ("python_version", test_python_version),
        ("core_dependencies", test_core_dependencies),
        ("ml_dependencies", test_ml_dependencies),
        ("ocr_dependencies", test_ocr_dependencies),
        ("data_structure", test_data_structure),
        ("module_imports", test_module_imports),
        ("quick_functionality", test_quick_functionality),
        ("gpu_functionality", test_gpu_functionality)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            report["tests"][test_name] = {"passed": result, "error": None}
            if not result:
                all_passed = False
        except Exception as e:
            report["tests"][test_name] = {"passed": False, "error": str(e)}
            all_passed = False
            logger.error(f"❌ Test {test_name} a échoué: {e}")
    
    # Sauvegarder le rapport
    report_file = "fine-tuning-ocr/installation_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Rapport sauvegardé: {report_file}")
    
    return all_passed, report

def print_summary(all_passed, report):
    """Affiche le résumé des tests"""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS D'INSTALLATION")
    print("="*60)
    
    if all_passed:
        print("🎉 TOUS LES TESTS ONT RÉUSSI!")
        print("✅ Votre installation est prête pour le fine-tuning OCR")
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("❌ Veuillez corriger les problèmes avant de continuer")
    
    print(f"\n📍 Plateforme: {report['platform']}")
    print(f"🐍 Python: {report['python_version']}")
    print(f"⏰ Testé le: {report['timestamp']}")
    
    print("\n📋 DÉTAILS DES TESTS:")
    for test_name, result in report["tests"].items():
        status = "✅" if result["passed"] else "❌"
        print(f"  {status} {test_name}")
        if result["error"]:
            print(f"      Erreur: {result['error']}")
    
    if not all_passed:
        print("\n🔧 ACTIONS RECOMMANDÉES:")
        print("1. Installez les dépendances manquantes:")
        print("   python install_fine_tuning_deps.py")
        print("2. Vérifiez la structure des dossiers:")
        print("   mkdir -p Data/processed_images Data/ocr_results")
        print("3. Consultez le guide d'installation:")
        print("   cat GUIDE_FINE_TUNING_COMPLET.md")
    else:
        print("\n🚀 PROCHAINES ÉTAPES:")
        print("1. Placez vos images dans Data/processed_images/")
        print("2. Placez vos résultats OCR dans Data/ocr_results/")
        print("3. Lancez le fine-tuning:")
        print("   python quick_start.py --full-pipeline")

def main():
    """Fonction principale"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🧪 Test d'Installation FacturAI              ║
    ║                  Fine-Tuning OCR System                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("🚀 Démarrage des tests d'installation...")
    
    # Exécuter tous les tests et générer le rapport
    all_passed, report = generate_installation_report()
    
    # Afficher le résumé
    print_summary(all_passed, report)
    
    # Code de sortie
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()