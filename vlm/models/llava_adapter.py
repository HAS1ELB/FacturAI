"""
Adaptateur pour le modèle LLaVA (Large Language and Vision Assistant)
"""

import time
import logging
from typing import Dict, Any
from PIL import Image
import torch

from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)

class LLaVAAdapter(ModelAdapter):
    """
    Adaptateur pour le modèle LLaVA
    
    LLaVA est un modèle multimodal qui combine un modèle de vision (CLIP)
    avec un modèle de langage (Vicuna) pour une compréhension avancée
    """
    
    def _load_model(self):
        """Charge le modèle LLaVA et son processeur"""
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            model_name = self.config.get("model_name", "llava-hf/llava-1.5-7b-hf")
            processor_name = self.config.get("processor_name", model_name)
            
            logger.info(f"Chargement de LLaVA: {model_name}")
            
            # Chargement du processeur
            self.processor = LlavaNextProcessor.from_pretrained(processor_name)
            
            # Chargement du modèle
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"LLaVA chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de LLaVA: {e}")
            raise
    
    def analyze_image(self, image: Image.Image) -> str:
        """
        Analyse une image avec LLaVA
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Description générée par LLaVA
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Prompt pour la description générale
            prompt = "USER: <image>\nDécris cette image en détail, en te concentrant sur la structure, le contenu et la mise en page.\nASSISTANT:"
            
            # Préparation des inputs
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_length", 512),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Décodage
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extraction de la réponse (après "ASSISTANT:")
            if "ASSISTANT:" in generated_text:
                result = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                result = generated_text
            
            # Post-traitement
            result = self._postprocess_text(result)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.85  # LLaVA est généralement très fiable
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse LLaVA: {e}")
            return f"Erreur d'analyse: {str(e)}"
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Répond à une question sur une image avec LLaVA
        
        Args:
            image: Image PIL à analyser
            question: Question à poser
            
        Returns:
            Réponse générée par LLaVA
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Formatage du prompt LLaVA
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            
            # Préparation des inputs
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)
            
            # Génération de la réponse
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_length", 512),
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Décodage
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extraction de la réponse (après "ASSISTANT:")
            if "ASSISTANT:" in generated_text:
                result = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                result = generated_text
            
            # Post-traitement
            result = self._postprocess_text(result)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.85
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la réponse LLaVA: {e}")
            return f"Erreur de réponse: {str(e)}"
    
    def analyze_document_structure(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyse la structure d'un document avec LLaVA
        
        Args:
            image: Image PIL du document
            
        Returns:
            Analyse détaillée de la structure
        """
        structure_questions = [
            "Quelle est la structure générale de ce document?",
            "Y a-t-il des sections distinctes dans ce document?",
            "Comment l'information est-elle organisée visuellement?",
            "Y a-t-il des éléments graphiques comme des tableaux, listes ou encadrés?"
        ]
        
        structure_analysis = {}
        for question in structure_questions:
            answer = self.answer_question(image, question)
            structure_analysis[question] = answer
        
        return structure_analysis
    
    def extract_invoice_fields(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extrait les champs spécifiques d'une facture avec LLaVA
        
        Args:
            image: Image PIL de la facture
            
        Returns:
            Champs extraits de la facture
        """
        field_questions = [
            "Quel est le numéro de facture visible dans ce document?",
            "Quelle est la date de cette facture?",
            "Qui est l'émetteur (nom de l'entreprise) de cette facture?",
            "Qui est le destinataire de cette facture?",
            "Quel est le montant total à payer?",
            "Y a-t-il un montant de TVA indiqué?",
            "Quels sont les articles ou services principaux facturés?",
            "Y a-t-il des informations de paiement (RIB, IBAN, etc.)?"
        ]
        
        extracted_fields = {}
        for question in field_questions:
            answer = self.answer_question(image, question)
            extracted_fields[question] = answer
        
        return extracted_fields
    
    def detect_visual_elements(self, image: Image.Image) -> Dict[str, Any]:
        """
        Détecte les éléments visuels importants
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Éléments visuels détectés
        """
        visual_questions = [
            "Y a-t-il un logo ou une image dans ce document?",
            "Quelles sont les couleurs dominantes utilisées?",
            "Y a-t-il des encadrés, bordures ou séparateurs visuels?",
            "Comment le texte est-il formaté (tailles, styles)?",
            "Y a-t-il des éléments mis en évidence (gras, surligné, etc.)?"
        ]
        
        visual_elements = {}
        for question in visual_questions:
            answer = self.answer_question(image, question)
            visual_elements[question] = answer
        
        return visual_elements