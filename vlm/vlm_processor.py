"""
Processeur principal VLM pour l'analyse visuelle et linguistique des factures
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from .config import vlm_config
from .models.model_adapter import ModelAdapter, create_adapter
from .utils.zone_detector import ZoneDetector
from .utils.layout_analyzer import LayoutAnalyzer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMProcessor:
    """
    Processeur principal pour l'analyse VLM des factures
    
    Ce processeur implémente l'étape 3 de l'architecture FacturAI :
    - Compréhension conjointe de l'apparence visuelle et du contenu textuel
    - Analyse de la mise en page et identification des structures
    - Détection des zones clés des factures
    """
    
    def __init__(self, config_path: str = None, output_dir: str = "Data/vlm_results"):
        """
        Initialise le processeur VLM
        
        Args:
            config_path: Chemin vers le fichier de configuration
            output_dir: Répertoire de sortie pour les résultats
        """
        self.config = vlm_config if config_path is None else VLMConfig(config_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialisation des composants
        self.model_adapter = None
        self.zone_detector = ZoneDetector(self.config.get_zone_detection_config())
        self.layout_analyzer = LayoutAnalyzer(self.config.get_layout_analysis_config())
        
        # Modèles disponibles
        self.available_models = self._check_available_models()
        logger.info(f"Modèles VLM disponibles: {self.available_models}")
        
        # Initialisation du modèle par défaut
        self._initialize_default_model()
    
    def _check_available_models(self) -> List[str]:
        """Vérifie quels modèles VLM sont disponibles"""
        available = []
        enabled_models = self.config.get_enabled_models()
        
        for model_name in enabled_models:
            try:
                # Test d'importation basique pour vérifier la disponibilité
                if model_name == "blip2":
                    from transformers import Blip2Processor, Blip2ForConditionalGeneration
                    available.append(model_name)
                elif model_name == "llava":
                    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                    available.append(model_name)
                elif model_name == "qwen_vl":
                    # Qwen-VL nécessite une vérification spéciale
                    try:
                        import transformers
                        available.append(model_name)
                    except:
                        logger.warning(f"Qwen-VL nécessite une installation spéciale")
                        
            except ImportError as e:
                logger.warning(f"Modèle {model_name} non disponible: {e}")
        
        return available
    
    def _initialize_default_model(self):
        # Prioriser LLaVA > Qwen-VL > BLIP-2
        priority_order = ["llava", "qwen_vl", "blip2"]
        for model in priority_order:
            if model in self.available_models:
                logger.info(f"Initialisation du modèle prioritaire: {model}")
                self.load_model(model)
                return
    
    def load_model(self, model_name: str):
        """
        Charge un modèle VLM spécifique
        
        Args:
            model_name: Nom du modèle à charger
        """
        if model_name not in self.available_models:
            raise ValueError(f"Modèle '{model_name}' non disponible. Modèles disponibles: {self.available_models}")
        
        model_config = self.config.get_model_config(model_name)
        self.model_adapter = create_adapter(model_name, model_config)
        logger.info(f"Modèle '{model_name}' chargé avec succès")
    
    def process_invoice(self, image_path: str, ocr_results: Dict = None) -> Dict[str, Any]:
        """
        Traite une facture avec analyse VLM complète
        
        Args:
            image_path: Chemin vers l'image de la facture
            ocr_results: Résultats OCR existants (optionnel)
        
        Returns:
            Dictionnaire contenant les résultats d'analyse VLM
        """
        start_time = time.time()
        
        # Validation des entrées
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        if self.model_adapter is None:
            raise RuntimeError("Aucun modèle VLM chargé")
        
        # Chargement de l'image
        image = self._load_and_preprocess_image(image_path)
        
        # Analyse VLM
        vlm_analysis = self._perform_vlm_analysis(image, ocr_results)
        
        # Détection des zones
        zones = self.zone_detector.detect_zones(image, vlm_analysis, ocr_results)
        
        # Analyse de la mise en page
        layout = self.layout_analyzer.analyze_layout(image, vlm_analysis, zones)
        
        # Compilation des résultats
        results = {
            "image_path": image_path,
            "timestamp": time.time(),
            "processing_time": time.time() - start_time,
            "model_used": self.model_adapter.model_name,
            "vlm_analysis": vlm_analysis,
            "detected_zones": zones,
            "layout_analysis": layout,
            "metadata": {
                "image_size": image.size,
                "model_config": self.model_adapter.config
            }
        }
        
        # Sauvegarde des résultats
        self._save_results(results, image_path)
        
        logger.info(f"Traitement VLM terminé en {results['processing_time']:.2f}s")
        return results
    
    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """Charge et prétraite l'image"""
        image = Image.open(image_path)
        
        # Redimensionnement si nécessaire
        max_size = self.config.get_processing_config().get("max_image_size", [1024, 1024])
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Conversion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def _perform_vlm_analysis(self, image: Image.Image, ocr_results: Dict = None) -> Dict[str, Any]:
        """Effectue l'analyse VLM principale"""
        if not self.model_adapter:
            return {"error": "Aucun modèle VLM disponible"}
        
        try:
            # Analyse de base avec le modèle VLM
            basic_analysis = self.model_adapter.analyze_image(image)
            
            # Questions spécifiques pour les factures
            invoice_questions = [
                "Quelles sont les informations principales visibles sur cette facture?",
                "Où sont situés les montants et totaux?",
                "Y a-t-il des tableaux ou listes d'articles?",
                "Quelles sont les zones d'en-tête et de pied de page?"
            ]
            
            detailed_analysis = {}
            for question in invoice_questions:
                answer = self.model_adapter.answer_question(image, question)
                detailed_analysis[question] = answer
            
            return {
                "basic_description": basic_analysis,
                "detailed_analysis": detailed_analysis,
                "confidence": self.model_adapter.last_confidence,
                "processing_info": self.model_adapter.get_processing_info()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse VLM: {e}")
            return {"error": str(e)}
    
    def _save_results(self, results: Dict[str, Any], image_path: str):
        """Sauvegarde les résultats de l'analyse"""
        # Nom de fichier de sortie basé sur l'image d'entrée
        input_name = Path(image_path).stem
        output_file = os.path.join(self.output_dir, f"vlm_{input_name}.json")
        
        # Sauvegarde JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Résultats sauvegardés: {output_file}")
    
    def batch_process(self, image_paths: List[str], ocr_results_dir: str = None) -> List[Dict[str, Any]]:
        """
        Traite un lot d'images en parallèle
        
        Args:
            image_paths: Liste des chemins d'images
            ocr_results_dir: Répertoire contenant les résultats OCR correspondants
        
        Returns:
            Liste des résultats d'analyse
        """
        results = []
        
        for image_path in image_paths:
            try:
                # Chargement des résultats OCR correspondants si disponibles
                ocr_results = None
                if ocr_results_dir:
                    ocr_file = os.path.join(ocr_results_dir, f"{Path(image_path).stem}_ocr.json")
                    if os.path.exists(ocr_file):
                        with open(ocr_file, 'r', encoding='utf-8') as f:
                            ocr_results = json.load(f)
                
                # Traitement de l'image
                result = self.process_invoice(image_path, ocr_results)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle actuel"""
        if self.model_adapter:
            return {
                "model_name": self.model_adapter.model_name,
                "model_config": self.model_adapter.config,
                "is_loaded": True
            }
        else:
            return {
                "model_name": None,
                "is_loaded": False,
                "available_models": self.available_models
            }