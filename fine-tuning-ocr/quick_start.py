#!/usr/bin/env python3
"""
🚀 Quick Start - FacturAI Fine-Tuning
Interface simplifiée pour démarrer rapidement le fine-tuning OCR
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Affiche le banner de démarrage"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎯 FacturAI Fine-Tuning                  ║
    ║                      Quick Start Guide                       ║
    ║                                                              ║
    ║  🤖 TrOCR • 👁️ EasyOCR • 🏓 PaddleOCR                      ║
    ║                                                              ║
    ║  Transformez vos 1000+ factures en données précises !       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Vérifie la version Python"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ requis")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True

def check_directory_structure():
    """Vérifie la structure des dossiers"""
    logger.info("🔍 Vérification de la structure des dossiers...")
    
    required_dirs = {
        "Data/processed_images": "Images prétraitées",
        "Data/ocr_results": "Résultats OCR existants"
    }
    
    missing_dirs = []
    for dir_path, description in required_dirs.items():
        if not Path(dir_path).exists():
            missing_dirs.append((dir_path, description))
        else:
            # Vérifier le contenu
            if dir_path == "Data/processed_images":
                images = list(Path(dir_path).glob("*.png")) + list(Path(dir_path).glob("*.jpg"))
                if len(images) == 0:
                    logger.warning(f"⚠️ Aucune image trouvée dans {dir_path}")
                else:
                    logger.info(f"✅ {len(images)} images trouvées dans {dir_path}")
            
            elif dir_path == "Data/ocr_results":
                json_files = list(Path(dir_path).glob("*.json"))
                if len(json_files) == 0:
                    logger.warning(f"⚠️ Aucun fichier JSON trouvé dans {dir_path}")
                else:
                    logger.info(f"✅ {len(json_files)} résultats OCR trouvés dans {dir_path}")
    
    if missing_dirs:
        logger.error("❌ Dossiers manquants:")
        for dir_path, description in missing_dirs:
            logger.error(f"  - {dir_path} ({description})")
        return False
    
    return True

def install_dependencies():
    """Installe les dépendances automatiquement"""
    logger.info("📦 Installation des dépendances...")
    
    try:
        # Lancer le script d'installation
        result = subprocess.run([sys.executable, "install_fine_tuning_deps.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Dépendances installées avec succès")
            return True
        else:
            logger.error(f"❌ Erreur d'installation: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Script d'installation non trouvé")
        return False

def quick_setup():
    """Configuration rapide du système"""
    logger.info("⚙️ Configuration rapide du système...")
    
    # Créer les dossiers nécessaires
    directories = [
        "Data/fine_tuning",
        "fine_tuning_ocr/models/easyocr_finetuned",
        "fine_tuning_ocr/models/trocr_finetuned", 
        "fine_tuning_ocr/models/paddleocr_finetuned",
        "fine_tuning_ocr/evaluation_results",
        "fine_tuning_ocr/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Dossiers créés")
    
    # Créer un fichier de configuration simplifié
    config = {
        "project_name": "FacturAI Fine-Tuning Quick Start",
        "data": {
            "images_dir": "Data/processed_images",
            "ocr_results_dir": "Data/ocr_results"
        },
        "quick_start": True
    }
    
    import json
    with open("fine_tuning_ocr/quick_start_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ Configuration créée")

def show_usage_options():
    """Affiche les options d'utilisation"""
    print("""
    🎯 OPTIONS DE DÉMARRAGE RAPIDE:
    
    1️⃣  INSTALLATION UNIQUEMENT
        python quick_start.py --install-only
        
    2️⃣  VÉRIFICATION SEULEMENT  
        python quick_start.py --check-only
        
    3️⃣  PRÉPARATION DES DONNÉES
        python quick_start.py --prepare-data
        
    4️⃣  EASYOCR SEULEMENT (votre demande)
        python quick_start.py --easyocr-only
        
    5️⃣  TROCR SEULEMENT (recommandé)
        python quick_start.py --trocr-only
        
    6️⃣  PIPELINE COMPLET
        python quick_start.py --full-pipeline
        
    7️⃣  AIDE ET GUIDE
        python quick_start.py --help
    """)

def run_preparation():
    """Lance la préparation des données"""
    logger.info("📊 Lancement de la préparation des données...")
    
    cmd = [sys.executable, "fine-tuning-ocr/data_preparation/data_preparation.py"]
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("✅ Préparation terminée")
        return True
    else:
        logger.error("❌ Erreur lors de la préparation")
        return False

def run_easyocr_only():
    """Lance le fine-tuning EasyOCR uniquement"""
    logger.info("👁️ Fine-tuning EasyOCR uniquement...")
    
    # Préparer les données d'abord
    if not run_preparation():
        return False
    
    cmd = [
        sys.executable, "fine-tuning-ocr/fine_tuning_model/easyocr_finetuning.py",
        "--dataset", "Data/fine_tuning/datasets/easyocr/dataset.json",
        "--epochs", "20",  # Réduire pour le test
        "--batch_size", "4"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("✅ EasyOCR fine-tuning terminé")
        return True
    else:
        logger.error("❌ Erreur EasyOCR fine-tuning")
        return False

def run_trocr_only():
    """Lance le fine-tuning TrOCR uniquement"""
    logger.info("🤖 Fine-tuning TrOCR uniquement...")
    
    # Préparer les données d'abord
    if not run_preparation():
        return False
    
    cmd = [
        sys.executable, "fine-tuning-ocr/fine_tuning_model/trocr_finetuning.py",
        "--dataset", "Data/fine_tuning/datasets/trocr/dataset.json",
        "--epochs", "5",  # Réduire pour le test
        "--batch_size", "2"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("✅ TrOCR fine-tuning terminé")
        return True
    else:
        logger.error("❌ Erreur TrOCR fine-tuning")
        return False

def run_full_pipeline():
    """Lance le pipeline complet"""
    logger.info("🚀 Lancement du pipeline complet...")
    
    cmd = [
        sys.executable, "fine-tuning-ocr/run_fine_tuning.py",
        "--mode", "all",
        "--models", "trocr", "easyocr"  # TrOCR + EasyOCR pour commencer
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("✅ Pipeline complet terminé")
        return True
    else:
        logger.error("❌ Erreur dans le pipeline")
        return False

def show_next_steps():
    """Affiche les prochaines étapes"""
    print("""
    🎉 FÉLICITATIONS! Fine-tuning terminé avec succès!
    
    📋 PROCHAINES ÉTAPES:
    
    1️⃣  Consulter les résultats:
        📁 models/ - Modèles entraînés
        📊 evaluation_results/ - Rapports de performance
        📝 logs/ - Journaux détaillés
        
    2️⃣  Tester sur de nouvelles factures:
        python -c "
        from model_evaluation import OCRModelEvaluator
        evaluator = OCRModelEvaluator()
        # Tester sur vos propres images
        "
        
    3️⃣  Intégrer dans votre pipeline:
        # Utiliser le meilleur modèle identifié
        # Adapter le post-processing
        
    4️⃣  Optimiser les performances:
        # Ajuster les hyperparamètres
        # Ajouter plus de données d'entraînement
        
    💡 CONSEIL: Consultez le GUIDE_FINE_TUNING_COMPLET.md pour plus de détails!
    """)

def main():
    """Fonction principale du quick start"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FacturAI Quick Start")
    parser.add_argument('--install-only', action='store_true', help='Installation seulement')
    parser.add_argument('--check-only', action='store_true', help='Vérification seulement')
    parser.add_argument('--prepare-data', action='store_true', help='Préparation données seulement')
    parser.add_argument('--easyocr-only', action='store_true', help='EasyOCR fine-tuning seulement')
    parser.add_argument('--trocr-only', action='store_true', help='TrOCR fine-tuning seulement')
    parser.add_argument('--full-pipeline', action='store_true', help='Pipeline complet')
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Vérifications de base
    if not check_python_version():
        sys.exit(1)
    
    # Installation des dépendances
    if args.install_only or not any(vars(args).values()):
        if install_dependencies():
            quick_setup()
            print("✅ Installation terminée! Relancez avec d'autres options.")
        else:
            print("❌ Problème d'installation")
        return
    
    # Vérification seulement
    if args.check_only:
        success = True
        success &= check_directory_structure()
        
        try:
            import torch, transformers, easyocr
            logger.info("✅ Dépendances Python OK")
        except ImportError as e:
            logger.error(f"❌ Dépendance manquante: {e}")
            success = False
        
        if success:
            print("✅ Tout est prêt pour le fine-tuning!")
        else:
            print("❌ Problèmes détectés")
        return
    
    # Vérifier la structure avant de continuer
    if not check_directory_structure():
        logger.error("❌ Structure incorrecte. Consultez le guide d'installation.")
        sys.exit(1)
    
    # Configuration rapide
    quick_setup()
    
    # Exécution selon l'option
    success = False
    
    if args.prepare_data:
        success = run_preparation()
    elif args.easyocr_only:
        success = run_easyocr_only()
    elif args.trocr_only:
        success = run_trocr_only()
    elif args.full_pipeline:
        success = run_full_pipeline()
    else:
        show_usage_options()
        return
    
    # Résultat final
    if success:
        show_next_steps()
    else:
        print("""
        ❌ Problème rencontré!
        
        🔧 SOLUTIONS:
        1. Consultez les logs dans le dossier logs/
        2. Vérifiez le GUIDE_FINE_TUNING_COMPLET.md
        3. Relancez avec --check-only pour diagnostiquer
        """)

if __name__ == "__main__":
    main()