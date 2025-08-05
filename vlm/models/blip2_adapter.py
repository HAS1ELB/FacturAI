"""
Adaptateur pour le modèle BLIP-2 de Salesforce
"""

import time
import logging
from typing import Dict, Any
from PIL import Image
import torch

from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)

class BLIP2Adapter(ModelAdapter):
    """
    Adaptateur pour le modèle BLIP-2
    
    BLIP-2 est un modèle vision-langage développé par Salesforce
    qui excelle dans la compréhension d'images et la génération de descriptions
    """
    
    def _load_model(self):
        """Charge le modèle BLIP-2 et son processeur"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            model_name = self.config.get("model_name", "Salesforce/blip2-opt-2.7b")
            processor_name = self.config.get("processor_name", model_name)
            
            logger.info(f"Chargement de BLIP-2: {model_name}")
            
            # Chargement du processeur
            self.processor = Blip2Processor.from_pretrained(processor_name)
            
            # Chargement du modèle
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"BLIP-2 chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de BLIP-2: {e}")
            raise
    
    def analyze_image(self, image: Image.Image) -> str:
        """
        Analyse une image avec BLIP-2
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Description générée par BLIP-2
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Préparation des inputs
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.config.get("max_length", 512),
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Décodage
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Post-traitement
            result = self._postprocess_text(generated_text)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.8  # BLIP-2 n'a pas de score de confiance natif
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse BLIP-2: {e}")
            return f"Erreur d'analyse: {str(e)}"
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Répond à une question sur une image avec BLIP-2
        
        Args:
            image: Image PIL à analyser
            question: Question à poser
            
        Returns:
            Réponse générée par BLIP-2
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Préparation des inputs avec la question
            inputs = self.processor(
                image, 
                question, 
                return_tensors="pt"
            ).to(self.device)
            
            # Génération de la réponse
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.config.get("max_length", 512),
                    num_beams=3,
                    temperature=0.5,
                    do_sample=True,
                    early_stopping=True
                )
            
            # Décodage
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Post-traitement
            result = self._postprocess_text(generated_text)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.8
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la réponse BLIP-2: {e}")
            return f"Erreur de réponse: {str(e)}"
    
    def extract_text_blocks(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extrait les blocs de texte détectés dans l'image
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Information sur les blocs de texte
        """
        questions = [
            "Quels sont les textes principaux visibles dans cette image?",
            "Y a-t-il des nombres ou montants importants?",
            "Quelles sont les informations en haut de la page?",
            "Quelles sont les informations en bas de la page?"
        ]
        
        text_blocks = {}
        for question in questions:
            answer = self.answer_question(image, question)
            text_blocks[question] = answer
        
        return text_blocks
    
    def detect_invoice_structure(self, image: Image.Image) -> Dict[str, Any]:
        """
        Détecte la structure spécifique d'une facture
        
        Args:
            image: Image PIL de la facture
            
        Returns:
            Structure détectée de la facture
        """
        structure_questions = [
            "Cette image contient-elle une facture ou un document commercial?",
            "Y a-t-il un en-tête avec logo ou nom d'entreprise?",
            "Y a-t-il un tableau avec des articles ou services?",
            "Y a-t-il des totaux ou montants en bas du document?",
            "Y a-t-il des informations de contact ou d'adresse?"
        ]
        
        structure = {}
        for question in structure_questions:
            answer = self.answer_question(image, question)
            structure[question] = answer
        
        return structure