"""
Adaptateur de base pour les modèles VLM
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class ModelAdapter(ABC):
    """
    Classe de base abstraite pour tous les adaptateurs de modèles VLM
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialise l'adaptateur de modèle
        
        Args:
            model_name: Nom du modèle
            config: Configuration du modèle
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.last_confidence = 0.0
        self.processing_time = 0.0
        
        logger.info(f"Initialisation de l'adaptateur {model_name} sur {self.device}")
        self._load_model()
    
    def _get_device(self) -> str:
        """Détermine le device optimal pour le modèle"""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_config
    
    @abstractmethod
    def _load_model(self):
        """Charge le modèle et le processeur spécifiques"""
        pass
    
    @abstractmethod
    def analyze_image(self, image: Image.Image) -> str:
        """
        Analyse une image et retourne une description
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Description textuelle de l'image
        """
        pass
    
    @abstractmethod
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Répond à une question sur une image
        
        Args:
            image: Image PIL à analyser
            question: Question à poser sur l'image
            
        Returns:
            Réponse à la question
        """
        pass
    
    def extract_layout_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extrait des informations sur la mise en page
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Informations sur la mise en page
        """
        questions = [
            "Décris la disposition générale de ce document",
            "Y a-t-il des tableaux dans cette image?",
            "Où sont situés les éléments importants?",
            "Quelles sont les différentes sections visibles?"
        ]
        
        layout_info = {}
        for question in questions:
            answer = self.answer_question(image, question)
            layout_info[question] = answer
        
        return layout_info
    
    def detect_invoice_elements(self, image: Image.Image) -> Dict[str, Any]:
        """
        Détecte les éléments spécifiques aux factures
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Éléments détectés de la facture
        """
        invoice_questions = [
            "Quel est le numéro de cette facture?",
            "Quelle est la date de la facture?",
            "Qui est l'émetteur de cette facture?",
            "Quel est le montant total?",
            "Y a-t-il des informations de TVA?",
            "Quels sont les articles ou services facturés?"
        ]
        
        elements = {}
        for question in invoice_questions:
            answer = self.answer_question(image, question)
            elements[question] = answer
        
        return elements
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Retourne les informations de traitement"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "last_confidence": self.last_confidence,
            "processing_time": self.processing_time,
            "config": self.config
        }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Prétraite l'image si nécessaire"""
        # Conversion en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionnement si configuré
        max_size = self.config.get("max_size")
        if max_size and (image.size[0] > max_size[0] or image.size[1] > max_size[1]):
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _postprocess_text(self, text: str) -> str:
        """Post-traite le texte généré"""
        # Nettoyage basique du texte
        text = text.strip()
        
        # Suppression des marqueurs spéciaux si présents
        if text.startswith("Réponse:"):
            text = text[8:].strip()
        
        return text


class DummyAdapter(ModelAdapter):
    """
    Adaptateur factice pour les tests ou quand aucun modèle n'est disponible
    """
    
    def _load_model(self):
        """Pas de modèle à charger pour l'adaptateur factice"""
        logger.warning("Utilisation de l'adaptateur factice - aucun modèle VLM réel chargé")
    
    def analyze_image(self, image: Image.Image) -> str:
        """Retourne une analyse factice"""
        time.sleep(0.1)  # Simulation du temps de traitement
        return "Cette image semble contenir un document avec du texte et possiblement des tableaux. [Analyse factice]"
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Retourne une réponse factice"""
        time.sleep(0.1)  # Simulation du temps de traitement
        return f"Réponse factice à la question: {question}"


def create_adapter(model_name: str, config: Dict[str, Any]) -> ModelAdapter:
    """
    Factory function pour créer le bon adaptateur selon le modèle
    
    Args:
        model_name: Nom du modèle
        config: Configuration du modèle
        
    Returns:
        Instance de l'adaptateur approprié
    """
    try:
        if model_name == "blip2":
            from .blip2_adapter import BLIP2Adapter
            return BLIP2Adapter(model_name, config)
        elif model_name == "llava":
            from .llava_adapter import LLaVAAdapter
            return LLaVAAdapter(model_name, config)
        elif model_name == "qwen_vl":
            from .qwen_vl_adapter import QwenVLAdapter
            return QwenVLAdapter(model_name, config)
        else:
            logger.warning(f"Modèle '{model_name}' non reconnu, utilisation de l'adaptateur factice")
            return DummyAdapter(model_name, config)
    except ImportError as e:
        logger.error(f"Impossible de charger l'adaptateur pour {model_name}: {e}")
        logger.info("Utilisation de l'adaptateur factice")
        return DummyAdapter(model_name, config)