#!/usr/bin/env python3
"""
Script principal pour lancer le fine-tuning OCR complet
Interface utilisateur simplifiée pour FacturAI
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time
from datetime import datetime

# Imports des modules du système
from data_preparation.data_preparation import InvoiceDataPreparator
from fine_tuning_manager.fine_tuning_manager import OCRFineTuningManager
from evaluation.model_evaluation import OCRModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FacturAIFineTuningOrchestrator:
    """Orchestrateur principal pour le fine-tuning FacturAI"""
    
    def __init__(self, config_file: str = "fine_tuning_config.json"):
        self.config_file = config_file
        self.config = self.load_or_create_config()
        self.setup_directories()
        
        logger.info("🚀 FacturAI Fine-Tuning Orchestrator initialisé")
    
    def load_or_create_config(self) -> Dict[str, Any]:
        """Charge ou crée la configuration"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration chargée depuis {self.config_file}")
        else:
            config = self.create_default_config()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration par défaut créée: {self.config_file}")
        
        return config
    
    def create_default_config(self) -> Dict[str, Any]:
        """Crée la configuration par défaut"""
        return {
            "project_name": "FacturAI Fine-Tuning",
            "version": "1.0.0",
            "data": {
                "images_dir": "Data/processed_images",
                "ocr_results_dir": "Data/ocr_results",
                "output_dir": "Data/fine_tuning",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "models": {
                "easyocr": {
                    "enabled": True,
                    "epochs": 50,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "output_dir": "fine-tuning-ocr/models/easyocr_finetuned"
                },
                "trocr": {
                    "enabled": True,
                    "base_model": "microsoft/trocr-large-printed",
                    "epochs": 10,
                    "batch_size": 4,
                    "learning_rate": 5e-5,
                    "output_dir": "fine-tuning-ocr/models/trocr_finetuned"
                }
            },
            "evaluation": {
                "output_dir": "evaluation_results",
                "metrics": ["similarity", "confidence", "speed", "accuracy"]
            },
            "hardware": {
                "use_gpu": True,
                "gpu_memory_limit": "8GB"
            }
        }
    
    def setup_directories(self):
        """Crée les dossiers nécessaires"""
        directories = [
            self.config["data"]["output_dir"],
            self.config["models"]["easyocr"]["output_dir"],
            self.config["models"]["trocr"]["output_dir"],
            self.config["evaluation"]["output_dir"],
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def check_requirements(self) -> bool:
        """Vérifie que tout est prêt pour le fine-tuning"""
        logger.info("🔍 Vérification des prérequis...")
        
        issues = []
        
        # Vérifier les dossiers de données
        images_dir = Path(self.config["data"]["images_dir"])
        ocr_dir = Path(self.config["data"]["ocr_results_dir"])
        
        if not images_dir.exists():
            issues.append(f"Dossier images manquant: {images_dir}")
        elif len(list(images_dir.glob("*.png"))) == 0:
            issues.append(f"Aucune image trouvée dans: {images_dir}")
        
        if not ocr_dir.exists():
            issues.append(f"Dossier OCR manquant: {ocr_dir}")
        elif len(list(ocr_dir.glob("*.json"))) == 0:
            issues.append(f"Aucun résultat OCR trouvé dans: {ocr_dir}")
        
        # Vérifier les dépendances Python
        try:
            import torch
            import transformers
            import easyocr
            logger.info("✅ Dépendances Python OK")
        except ImportError as e:
            issues.append(f"Dépendance manquante: {e}")
        
        # Vérifier GPU si activé
        if self.config["hardware"]["use_gpu"]:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"✅ GPU disponible: {gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    issues.append("GPU demandé mais non disponible")
            except:
                issues.append("Impossible de vérifier le GPU")
        
        if issues:
            logger.error("❌ Problèmes détectés:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ Tous les prérequis sont satisfaits")
        return True
    
    def prepare_data(self) -> Dict[str, Any]:
        """Prépare les données pour le fine-tuning"""
        logger.info("📊 ÉTAPE 1: Préparation des données")
        logger.info("=" * 50)
        
        preparator = InvoiceDataPreparator(
            images_dir=self.config["data"]["images_dir"],
            ocr_results_dir=self.config["data"]["ocr_results_dir"],
            output_dir=self.config["data"]["output_dir"]
        )
        
        results = preparator.run_complete_preparation()
        
        if results:
            logger.info("✅ Préparation des données terminée")
            
            # Sauvegarder les statistiques dans la config
            self.config["data"]["statistics"] = results["statistics"]
            self.save_config()
            
            return results
        else:
            logger.error("❌ Échec de la préparation des données")
            return {}
    
    def train_models(self, enabled_models: List[str] = None) -> Dict[str, Any]:
        """Lance l'entraînement des modèles sélectionnés"""
        logger.info("🎯 ÉTAPE 2: Entraînement des modèles")
        logger.info("=" * 50)
        
        if enabled_models is None:
            enabled_models = [model for model in self.config["models"] 
                            if self.config["models"][model].get("enabled", False)]
        
        training_results = {}
        
        # EasyOCR
        if "easyocr" in enabled_models:
            logger.info("\n👁️ Entraînement EasyOCR...")
            try:
                from fine_tuning_model.easyocr_finetuning import EasyOCRFineTuner
                
                tuner = EasyOCRFineTuner(self.config["models"]["easyocr"])
                dataset_file = f"{self.config['data']['output_dir']}/datasets/easyocr/dataset.json"
                
                results = tuner.train(
                    dataset_file=dataset_file,
                    epochs=self.config["models"]["easyocr"]["epochs"],
                    batch_size=self.config["models"]["easyocr"]["batch_size"]
                )
                
                training_results["easyocr"] = results
                logger.info("✅ EasyOCR entraîné avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur EasyOCR: {e}")
                training_results["easyocr"] = {"error": str(e)}
        
        # TrOCR
        if "trocr" in enabled_models:
            logger.info("\n🤖 Entraînement TrOCR...")
            try:
                from fine_tuning_model.trocr_finetuning import TrOCRFineTuner
                
                tuner = TrOCRFineTuner(self.config["models"]["trocr"])
                dataset_file = f"{self.config['data']['output_dir']}/datasets/trocr/dataset.json"
                
                results = tuner.train(
                    dataset_file=dataset_file,
                    epochs=self.config["models"]["trocr"]["epochs"],
                    batch_size=self.config["models"]["trocr"]["batch_size"]
                )
                
                training_results["trocr"] = results
                logger.info("✅ TrOCR entraîné avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur TrOCR: {e}")
                training_results["trocr"] = {"error": str(e)}
        
        return training_results
    
    def evaluate_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue et compare les modèles entraînés"""
        logger.info("📈 ÉTAPE 3: Évaluation des modèles")
        logger.info("=" * 50)
        
        # Préparer les chemins des modèles
        models_config = {}
        
        for model_name, results in training_results.items():
            if "error" not in results:
                if model_name == "easyocr" and "model_path" in results:
                    models_config["easyocr_finetuned"] = results["model_path"]
                elif model_name == "trocr" and "model_path" in results:
                    models_config["trocr_finetuned"] = results["model_path"]
        
        # Lancer l'évaluation
        evaluator = OCRModelEvaluator(self.config["evaluation"]["output_dir"])
        
        test_data_file = f"{self.config['data']['output_dir']}/splits/test.json"
        ground_truth_file = f"{self.config['data']['output_dir']}/annotations/ground_truth.json"
        
        evaluation_results = evaluator.run_complete_evaluation(
            test_data_file=test_data_file,
            ground_truth_file=ground_truth_file,
            models_config=models_config
        )
        
        return evaluation_results
    
    def generate_final_report(self, preparation_results: Dict, training_results: Dict, 
                            evaluation_results: Dict) -> str:
        """Génère le rapport final complet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("logs") / f"facturai_fine_tuning_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🎯 Rapport Final - Fine-Tuning FacturAI\n\n")
            f.write(f"**Date :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Version :** {self.config['version']}\n\n")
            
            # Résumé exécutif
            f.write("## 📊 Résumé Exécutif\n\n")
            
            if preparation_results and "statistics" in preparation_results:
                stats = preparation_results["statistics"]
                f.write(f"- **Images traitées :** {stats['total_images']}\n")
                f.write(f"- **Annotations générées :** {stats['total_annotations']}\n")
                f.write(f"- **Confiance moyenne initiale :** {stats['avg_confidence']:.2f}\n\n")
            
            # Modèles entraînés
            f.write("## 🤖 Modèles Entraînés\n\n")
            
            for model_name, results in training_results.items():
                f.write(f"### {model_name.upper()}\n")
                if "error" in results:
                    f.write(f"❌ **Status :** Échec - {results['error']}\n\n")
                else:
                    f.write("✅ **Status :** Succès\n")
                    if "total_time" in results:
                        f.write(f"⏱️ **Temps d'entraînement :** {results['total_time']/60:.1f} minutes\n")
                    if "best_val_loss" in results:
                        f.write(f"📉 **Meilleure perte :** {results['best_val_loss']:.4f}\n")
                    f.write("\n")
            
            # Résultats d'évaluation
            if evaluation_results and "comparison_df" in evaluation_results:
                f.write("## 🏆 Classement des Modèles\n\n")
                df = evaluation_results["comparison_df"]
                if not df.empty:
                    for i, (_, row) in enumerate(df.iterrows(), 1):
                        f.write(f"{i}. **{row['Model']}**\n")
                        if 'avg_similarity' in row:
                            f.write(f"   - Similarité : {row['avg_similarity']:.3f}\n")
                        if 'avg_confidence' in row:
                            f.write(f"   - Confiance : {row['avg_confidence']:.3f}\n")
                        if 'avg_processing_time' in row:
                            f.write(f"   - Vitesse : {row['avg_processing_time']:.2f}s\n")
                        f.write("\n")
            
            # Recommandations
            f.write("## 💡 Recommandations\n\n")
            f.write("### Pour la Production\n")
            f.write("1. **Modèle recommandé :** Le modèle avec la meilleure similarité\n")
            f.write("2. **Optimisations :**\n")
            f.write("   - Préprocessing adapté aux factures\n")
            f.write("   - Post-processing avec validation métier\n")
            f.write("   - Système de confiance pour la validation manuelle\n\n")
            
            f.write("### Prochaines Étapes\n")
            f.write("1. Tests sur de nouvelles factures\n")
            f.write("2. Intégration dans le pipeline FacturAI\n")
            f.write("3. Monitoring des performances en production\n")
            f.write("4. Amélioration continue avec nouvelles données\n\n")
            
            # Fichiers générés
            f.write("## 📁 Fichiers Générés\n\n")
            f.write(f"- **Configuration :** `{self.config_file}`\n")
            f.write(f"- **Données :** `{self.config['data']['output_dir']}/`\n")
            f.write(f"- **Modèles :** `models/`\n")
            f.write(f"- **Évaluation :** `{self.config['evaluation']['output_dir']}/`\n")
            
            if evaluation_results and "report_file" in evaluation_results:
                f.write(f"- **Rapport détaillé :** `{evaluation_results['report_file']}`\n")
        
        return str(report_file)
    
    def save_config(self):
        """Sauvegarde la configuration mise à jour"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def run_complete_pipeline(self, enabled_models: List[str] = None) -> Dict[str, Any]:
        """Lance le pipeline complet de fine-tuning"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET FACTURAI")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Vérifications
        if not self.check_requirements():
            return {"error": "Prérequis non satisfaits"}
        
        try:
            # 1. Préparation des données
            preparation_results = self.prepare_data()
            if not preparation_results:
                return {"error": "Échec de la préparation des données"}
            
            # 2. Entraînement des modèles
            training_results = self.train_models(enabled_models)
            
            # 3. Évaluation
            evaluation_results = self.evaluate_models(training_results)
            
            # 4. Rapport final
            final_report = self.generate_final_report(
                preparation_results, training_results, evaluation_results
            )
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
            logger.info(f"⏱️ Temps total: {total_time/60:.1f} minutes")
            logger.info(f"📊 Rapport final: {final_report}")
            
            return {
                "success": True,
                "preparation_results": preparation_results,
                "training_results": training_results,
                "evaluation_results": evaluation_results,
                "final_report": final_report,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur dans le pipeline: {e}")
            return {"error": str(e)}

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="FacturAI Fine-Tuning Orchestrator")
    parser.add_argument('--config', default='fine_tuning_config.json', 
                       help='Fichier de configuration')
    parser.add_argument('--mode', choices=['prepare', 'train', 'evaluate', 'all'], 
                       default='all', help='Mode d\'exécution')
    parser.add_argument('--models', nargs='+', 
                       choices=['easyocr', 'trocr', 'paddleocr'],
                       help='Modèles à entraîner')
    parser.add_argument('--check-only', action='store_true',
                       help='Vérifier les prérequis uniquement')
    
    args = parser.parse_args()
    
    # Créer l'orchestrateur
    orchestrator = FacturAIFineTuningOrchestrator(args.config)
    
    # Mode vérification uniquement
    if args.check_only:
        if orchestrator.check_requirements():
            print("✅ Tous les prérequis sont satisfaits")
            sys.exit(0)
        else:
            print("❌ Problèmes détectés")
            sys.exit(1)
    
    # Exécution selon le mode
    if args.mode == 'prepare':
        results = orchestrator.prepare_data()
    elif args.mode == 'train':
        orchestrator.prepare_data()  # S'assurer que les données sont prêtes
        results = orchestrator.train_models(args.models)
    elif args.mode == 'evaluate':
        # Dummy training results pour l'évaluation
        training_results = {}
        results = orchestrator.evaluate_models(training_results)
    else:  # mode 'all'
        results = orchestrator.run_complete_pipeline(args.models)
    
    # Affichage des résultats
    if "error" in results:
        print(f"❌ Erreur: {results['error']}")
        sys.exit(1)
    elif "success" in results:
        print(f"\n🎉 Processus terminé avec succès!")
        if "final_report" in results:
            print(f"📊 Rapport final: {results['final_report']}")
    else:
        print("✅ Étape terminée")

if __name__ == "__main__":
    main()