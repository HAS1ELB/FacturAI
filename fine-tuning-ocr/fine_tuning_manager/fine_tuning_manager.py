#!/usr/bin/env python3
"""
Système complet de fine-tuning OCR pour factures
Supportant multiple approches : EasyOCR, TrOCR, PaddleOCR
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRFineTuningManager:
    """Gestionnaire principal pour le fine-tuning OCR"""
    
    def __init__(self, config_path: str = "fine_tuning_config.json"):
        self.config = self.load_config(config_path)
        self.setup_directories()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration du fine-tuning"""
        default_config = {
            "data": {
                "images_dir": "Data/processed_images",
                "annotations_dir": "Data/annotations",
                "output_dir": "Data/fine_tuning",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "models": {
                "easyocr": {
                    "enabled": True,
                    "custom_model_path": "models/easyocr_finetuned",
                    "epochs": 50,
                    "batch_size": 8,
                    "learning_rate": 0.001
                },
                "trocr": {
                    "enabled": True,
                    "base_model": "microsoft/trocr-large-printed",
                    "output_dir": "models/trocr_finetuned",
                    "epochs": 30,
                    "batch_size": 4,
                    "learning_rate": 5e-5
                },
                "paddleocr": {
                    "enabled": True,
                    "output_dir": "models/paddleocr_finetuned",
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.001
                }
            },
            "training": {
                "use_gpu": True,
                "mixed_precision": True,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 1000,
                "max_length": 512,
                "early_stopping_patience": 5
            },
            "evaluation": {
                "metrics": ["accuracy", "edit_distance", "bleu", "character_error_rate"],
                "save_predictions": True,
                "generate_report": True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # Merge configs
            default_config.update(user_config)
        else:
            # Sauvegarder la config par défaut
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration par défaut créée: {config_path}")
        
        return default_config
    
    def setup_directories(self):
        """Crée les dossiers nécessaires"""
        dirs_to_create = [
            self.config["data"]["annotations_dir"],
            self.config["data"]["output_dir"],
            "models/easyocr_finetuned",
            "models/trocr_finetuned", 
            "models/paddleocr_finetuned",
            "logs/training",
            "results/evaluation"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info("Dossiers de fine-tuning initialisés")
    
    def get_available_approaches(self) -> List[str]:
        """Retourne les approches de fine-tuning disponibles"""
        approaches = []
        for model_name, config in self.config["models"].items():
            if config.get("enabled", False):
                approaches.append(model_name)
        return approaches
    
    def run_full_pipeline(self, approach: str = "all"):
        """Exécute le pipeline complet de fine-tuning"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE DE FINE-TUNING")
        logger.info("=" * 60)
        
        # 1. Vérification des prérequis
        self.check_prerequisites()
        
        # 2. Préparation des données
        self.prepare_dataset()
        
        # 3. Fine-tuning selon l'approche choisie
        if approach == "all":
            approaches = self.get_available_approaches()
        else:
            approaches = [approach] if approach in self.get_available_approaches() else []
        
        if not approaches:
            raise ValueError(f"Aucune approche valide trouvée. Disponibles: {self.get_available_approaches()}")
        
        results = {}
        for approach_name in approaches:
            logger.info(f"\n🎯 FINE-TUNING AVEC {approach_name.upper()}")
            logger.info("-" * 40)
            
            try:
                if approach_name == "easyocr":
                    results[approach_name] = self.finetune_easyocr()
                elif approach_name == "trocr":
                    results[approach_name] = self.finetune_trocr()
                elif approach_name == "paddleocr":
                    results[approach_name] = self.finetune_paddleocr()
                
                logger.info(f"✅ {approach_name} fine-tuning terminé avec succès")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du fine-tuning {approach_name}: {str(e)}")
                results[approach_name] = {"error": str(e)}
        
        # 4. Évaluation comparative
        self.compare_models(results)
        
        # 5. Génération du rapport final
        self.generate_final_report(results)
        
        logger.info("\n🎉 PIPELINE DE FINE-TUNING TERMINÉ")
        logger.info("=" * 60)
        
        return results
    
    def check_prerequisites(self):
        """Vérifie que tous les prérequis sont installés"""
        logger.info("📋 Vérification des prérequis...")
        
        required_packages = {
            "torch": "PyTorch pour l'entraînement",
            "transformers": "Hugging Face Transformers pour TrOCR",
            "datasets": "Datasets pour la gestion des données",
            "easyocr": "EasyOCR pour le fine-tuning",
            "paddlepaddle": "PaddlePaddle pour PaddleOCR",
            "paddleocr": "PaddleOCR",
            "opencv-python": "OpenCV pour le traitement d'images",
            "Pillow": "PIL pour la manipulation d'images",
            "numpy": "NumPy pour les calculs numériques",
            "pandas": "Pandas pour la gestion des données",
            "scikit-learn": "Scikit-learn pour les métriques",
            "matplotlib": "Matplotlib pour la visualisation",
            "seaborn": "Seaborn pour les graphiques",
            "tqdm": "TQDM pour les barres de progression",
            "wandb": "Weights & Biases pour le monitoring (optionnel)"
        }
        
        missing_packages = []
        for package, description in required_packages.items():
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"  ✅ {package}: {description}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"  ❌ {package}: {description} - MANQUANT")
        
        if missing_packages:
            logger.error(f"Packages manquants: {', '.join(missing_packages)}")
            logger.error("Installer avec: pip install " + " ".join(missing_packages))
            raise ImportError("Packages requis manquants")
        
        logger.info("✅ Tous les prérequis sont installés")
    
    def prepare_dataset(self):
        """Prépare le dataset pour l'entraînement"""
        logger.info("📊 Préparation du dataset...")
        
        # Cette méthode sera implémentée dans le prochain fichier
        # car elle est complexe et nécessite plusieurs classes
        from data_preparation import DatasetPreparator
        
        preparator = DatasetPreparator(self.config)
        dataset_info = preparator.prepare_all_datasets()
        
        logger.info(f"✅ Dataset préparé: {dataset_info}")
        return dataset_info
    
    def finetune_easyocr(self) -> Dict[str, Any]:
        """Fine-tuning d'EasyOCR"""
        logger.info("🔧 Fine-tuning EasyOCR...")
        
        from easyocr_finetuning import EasyOCRFineTuner
        
        trainer = EasyOCRFineTuner(self.config["models"]["easyocr"])
        results = trainer.train()
        
        return results
    
    def finetune_trocr(self) -> Dict[str, Any]:
        """Fine-tuning de TrOCR"""
        logger.info("🔧 Fine-tuning TrOCR...")
        
        from trocr_finetuning import TrOCRFineTuner
        
        trainer = TrOCRFineTuner(self.config["models"]["trocr"])
        results = trainer.train()
        
        return results
    
    def finetune_paddleocr(self) -> Dict[str, Any]:
        """Fine-tuning de PaddleOCR"""
        logger.info("🔧 Fine-tuning PaddleOCR...")
        
        from paddleocr_finetuning import PaddleOCRFineTuner
        
        trainer = PaddleOCRFineTuner(self.config["models"]["paddleocr"])
        results = trainer.train()
        
        return results
    
    def compare_models(self, results: Dict[str, Any]):
        """Compare les performances des différents modèles"""
        logger.info("📊 Comparaison des modèles...")
        
        from model_evaluation import ModelComparator
        
        comparator = ModelComparator(self.config)
        comparison_report = comparator.compare_all_models(results)
        
        # Sauvegarder le rapport de comparaison
        report_path = "results/evaluation/model_comparison.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Rapport de comparaison sauvegardé: {report_path}")
        
        return comparison_report
    
    def generate_final_report(self, results: Dict[str, Any]):
        """Génère le rapport final du fine-tuning"""
        logger.info("📄 Génération du rapport final...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": results,
            "summary": {
                "total_models_trained": len([r for r in results.values() if "error" not in r]),
                "failed_trainings": len([r for r in results.values() if "error" in r]),
                "best_model": self.find_best_model(results),
                "recommendations": self.generate_recommendations(results)
            }
        }
        
        # Sauvegarder le rapport
        report_path = f"results/evaluation/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Rapport final sauvegardé: {report_path}")
        
        # Générer aussi un rapport HTML
        self.generate_html_report(report, report_path.replace('.json', '.html'))
        
        return report
    
    def find_best_model(self, results: Dict[str, Any]) -> str:
        """Trouve le meilleur modèle basé sur les métriques"""
        best_model = None
        best_score = 0
        
        for model_name, result in results.items():
            if "error" not in result and "evaluation" in result:
                # Score composite basé sur accuracy et faible taux d'erreur
                score = result["evaluation"].get("accuracy", 0) * 0.6 + \
                       (1 - result["evaluation"].get("character_error_rate", 1)) * 0.4
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model or "aucun"
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        successful_models = [name for name, result in results.items() if "error" not in result]
        
        if not successful_models:
            recommendations.append("Aucun modèle n'a pu être entraîné avec succès. Vérifiez les données et la configuration.")
            return recommendations
        
        best_model = self.find_best_model(results)
        if best_model != "aucun":
            recommendations.append(f"Utilisez le modèle {best_model} pour la production.")
        
        # Autres recommandations basées sur les performances
        for model_name, result in results.items():
            if "error" not in result and "evaluation" in result:
                accuracy = result["evaluation"].get("accuracy", 0)
                if accuracy < 0.8:
                    recommendations.append(f"Le modèle {model_name} nécessite plus de données d'entraînement (accuracy: {accuracy:.2%}).")
                elif accuracy > 0.95:
                    recommendations.append(f"Le modèle {model_name} montre d'excellentes performances (accuracy: {accuracy:.2%}).")
        
        recommendations.append("Considérez l'ensemble des modèles pour créer un système hybride.")
        recommendations.append("Collectez plus de données difficiles pour améliorer la robustesse.")
        
        return recommendations
    
    def generate_html_report(self, report: Dict[str, Any], output_path: str):
        """Génère un rapport HTML lisible"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de Fine-tuning OCR - FacturAI</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #667eea; background: #f8f9fa; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; min-width: 120px; text-align: center; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .recommendation {{ background: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Rapport de Fine-tuning OCR</h1>
                    <p>Projet FacturAI - {report['timestamp']}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Résumé</h2>
                    <div class="metric">
                        <h3>{report['summary']['total_models_trained']}</h3>
                        <p>Modèles entraînés</p>
                    </div>
                    <div class="metric">
                        <h3 class="{'success' if report['summary']['failed_trainings'] == 0 else 'error'}">{report['summary']['failed_trainings']}</h3>
                        <p>Échecs</p>
                    </div>
                    <div class="metric">
                        <h3 class="success">{report['summary']['best_model']}</h3>
                        <p>Meilleur modèle</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Recommandations</h2>
                    {''.join([f'<div class="recommendation">💡 {rec}</div>' for rec in report['summary']['recommendations']])}
                </div>
                
                <div class="section">
                    <h2>📈 Détails des résultats</h2>
                    <pre>{json.dumps(report['results'], indent=2, ensure_ascii=False)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Rapport HTML généré: {output_path}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Fine-tuning OCR pour FacturAI")
    parser.add_argument("--approach", choices=["easyocr", "trocr", "paddleocr", "all"], 
                       default="all", help="Approche de fine-tuning à utiliser")
    parser.add_argument("--config", default="fine_tuning_config.json", 
                       help="Fichier de configuration")
    parser.add_argument("--prepare-only", action="store_true", 
                       help="Préparer seulement les données, sans entraînement")
    
    args = parser.parse_args()
    
    # Initialiser le gestionnaire
    manager = OCRFineTuningManager(args.config)
    
    if args.prepare_only:
        logger.info("Mode préparation uniquement")
        manager.prepare_dataset()
    else:
        # Exécuter le pipeline complet
        results = manager.run_full_pipeline(args.approach)
        
        # Afficher un résumé
        print("\n" + "="*60)
        print("🎉 FINE-TUNING TERMINÉ")
        print("="*60)
        
        for model_name, result in results.items():
            if "error" in result:
                print(f"❌ {model_name}: {result['error']}")
            else:
                accuracy = result.get("evaluation", {}).get("accuracy", 0)
                print(f"✅ {model_name}: {accuracy:.2%} accuracy")
        
        best_model = manager.find_best_model(results)
        print(f"\n🏆 Meilleur modèle: {best_model}")
        print(f"📁 Résultats dans: results/evaluation/")

if __name__ == "__main__":
    main()