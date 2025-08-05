"""
FacturAI VLM Module - Visual Language Model pour l'analyse de factures

Ce module implémente l'étape 3 de l'architecture FacturAI :
Analyse visuelle et linguistique par VLM (Visual Language Model)

Fonctionnalités :
- Compréhension conjointe de l'apparence visuelle et du contenu textuel
- Analyse de la mise en page et identification des structures
- Détection des zones clés des factures
- Intégration avec les résultats OCR

Auteur: FacturAI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FacturAI Team"

from .vlm_processor import VLMProcessor
from .models import ModelAdapter
from .utils import ZoneDetector, LayoutAnalyzer

__all__ = [
    "VLMProcessor",
    "ModelAdapter", 
    "ZoneDetector",
    "LayoutAnalyzer"
]