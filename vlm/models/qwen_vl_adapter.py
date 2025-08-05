"""
Adaptateur pour le modèle Qwen-VL
"""

import time
import logging
from typing import Dict, Any
from PIL import Image
import torch

from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)

class QwenVLAdapter(ModelAdapter):
    """
    Adaptateur pour le modèle Qwen-VL
    
    Qwen-VL est un modèle multimodal développé par Alibaba Cloud
    qui excelle dans la compréhension de documents et d'images complexes
    """
    
    def _load_model(self):
        """Charge le modèle Qwen-VL et son processeur"""
        try:
            # Qwen-VL nécessite une installation spéciale
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation import GenerationConfig
            
            model_name = self.config.get("model_name", "Qwen/Qwen-VL-Chat")
            
            logger.info(f"Chargement de Qwen-VL: {model_name}")
            
            # Note: Qwen-VL peut nécessiter trust_remote_code=True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Configuration de génération
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            logger.info(f"Qwen-VL chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de Qwen-VL: {e}")
            logger.warning("Qwen-VL nécessite 'pip install transformers_stream_generator' et trust_remote_code=True")
            raise
    
    def analyze_image(self, image: Image.Image) -> str:
        """
        Analyse une image avec Qwen-VL
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Description générée par Qwen-VL
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Sauvegarde temporaire de l'image pour Qwen-VL
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                image_path = tmp_file.name
            
            try:
                # Formatage du prompt Qwen-VL avec l'image
                query = f'<img>{image_path}</img>Décris cette image en détail, en te concentrant sur la structure, le contenu et la mise en page du document.'
                
                # Génération avec Qwen-VL
                response, history = self.model.chat(
                    self.tokenizer, 
                    query=query, 
                    history=None
                )
                
                # Post-traitement
                result = self._postprocess_text(response)
                
            finally:
                # Nettoyage du fichier temporaire
                os.unlink(image_path)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.9  # Qwen-VL est très performant sur les documents
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse Qwen-VL: {e}")
            return f"Erreur d'analyse: {str(e)}"
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Répond à une question sur une image avec Qwen-VL
        
        Args:
            image: Image PIL à analyser
            question: Question à poser
            
        Returns:
            Réponse générée par Qwen-VL
        """
        start_time = time.time()
        
        try:
            # Prétraitement de l'image
            image = self._preprocess_image(image)
            
            # Sauvegarde temporaire de l'image
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                image_path = tmp_file.name
            
            try:
                # Formatage du prompt avec la question
                query = f'<img>{image_path}</img>{question}'
                
                # Génération de la réponse
                response, history = self.model.chat(
                    self.tokenizer, 
                    query=query, 
                    history=None
                )
                
                # Post-traitement
                result = self._postprocess_text(response)
                
            finally:
                # Nettoyage du fichier temporaire
                os.unlink(image_path)
            
            # Métadonnées
            self.processing_time = time.time() - start_time
            self.last_confidence = 0.9
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la réponse Qwen-VL: {e}")
            return f"Erreur de réponse: {str(e)}"
    
    def extract_structured_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extrait des données structurées avec Qwen-VL
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Données structurées extraites
        """
        extraction_questions = [
            "Extrait toutes les informations structurées de ce document sous forme de liste organisée.",
            "Quels sont tous les montants numériques visibles dans ce document?",
            "Quelles sont toutes les dates mentionnées dans ce document?",
            "Quels sont tous les noms d'entreprises ou personnes mentionnés?",
            "Y a-t-il des numéros de référence, codes ou identifiants?"
        ]
        
        structured_data = {}
        for question in extraction_questions:
            answer = self.answer_question(image, question)
            structured_data[question] = answer
        
        return structured_data
    
    def analyze_table_content(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyse le contenu des tableaux avec Qwen-VL
        
        Args:
            image: Image PIL contenant des tableaux
            
        Returns:
            Analyse du contenu des tableaux
        """
        table_questions = [
            "Y a-t-il des tableaux dans cette image? Si oui, décris leur structure.",
            "Quels sont les en-têtes des colonnes dans le(s) tableau(x)?",
            "Combien de lignes contient chaque tableau?",
            "Quelles sont les données principales dans chaque colonne?",
            "Y a-t-il des totaux ou sous-totaux dans le(s) tableau(x)?"
        ]
        
        table_analysis = {}
        for question in table_questions:
            answer = self.answer_question(image, question)
            table_analysis[question] = answer
        
        return table_analysis
    
    def detect_document_type(self, image: Image.Image) -> Dict[str, Any]:
        """
        Détecte le type de document avec Qwen-VL
        
        Args:
            image: Image PIL du document
            
        Returns:
            Information sur le type de document
        """
        type_questions = [
            "Quel type de document est-ce (facture, devis, bon de commande, etc.)?",
            "Quels sont les indices visuels qui permettent d'identifier le type de document?",
            "Dans quel contexte commercial ce document est-il utilisé?",
            "Quels sont les éléments obligatoires présents sur ce type de document?"
        ]
        
        document_type = {}
        for question in type_questions:
            answer = self.answer_question(image, question)
            document_type[question] = answer
        
        return document_type
    
    def get_document_quality_assessment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Évalue la qualité du document pour l'extraction de données
        
        Args:
            image: Image PIL du document
            
        Returns:
            Évaluation de la qualité
        """
        quality_questions = [
            "Quelle est la qualité visuelle de ce document (lisibilité, résolution)?",
            "Y a-t-il des parties floues, tachées ou illisibles?",
            "Le document est-il bien orienté et cadré?",
            "Tous les textes sont-ils clairement visibles?",
            "Y a-t-il des problèmes qui pourraient affecter l'extraction automatique de données?"
        ]
        
        quality_assessment = {}
        for question in quality_questions:
            answer = self.answer_question(image, question)
            quality_assessment[question] = answer
        
        return quality_assessment