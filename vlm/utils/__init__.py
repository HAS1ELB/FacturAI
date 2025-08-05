"""
Utilitaires pour le module VLM de FacturAI

Ce module contient les utilitaires pour :
- Détection et classification des zones clés
- Analyse de la mise en page
- Traitement des coordonnées et géométrie
- Visualisation des résultats
"""

from .zone_detector import ZoneDetector
from .layout_analyzer import LayoutAnalyzer
from .geometry_utils import GeometryUtils
from .visualization import VLMVisualizer

__all__ = [
    "ZoneDetector",
    "LayoutAnalyzer", 
    "GeometryUtils",
    "VLMVisualizer"
]