"""
Analyseur de mise en page pour les factures
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """
    Analyseur de mise en page pour les factures
    
    Analyse la structure visuelle et spatiale des documents :
    - Organisation générale (grille, colonnes)
    - Relations spatiales entre éléments
    - Hiérarchie visuelle
    - Alignements et espacement
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'analyseur de mise en page
        
        Args:
            config: Configuration d'analyse de mise en page
        """
        self.config = config
        self.grid_detection = config.get("grid_detection", True)
        self.table_detection = config.get("table_detection", True)
        self.text_block_detection = config.get("text_block_detection", True)
        self.logo_detection = config.get("logo_detection", False)
        self.min_text_confidence = config.get("min_text_confidence", 0.1)
    
    def analyze_layout(self, image: Image.Image, vlm_analysis: Dict[str, Any], 
                      zones: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse la mise en page d'une facture
        
        Args:
            image: Image PIL de la facture
            vlm_analysis: Résultats de l'analyse VLM
            zones: Zones détectées
        
        Returns:
            Analyse de la mise en page
        """
        layout_analysis = {
            "document_structure": self._analyze_document_structure(vlm_analysis),
            "spatial_organization": self._analyze_spatial_organization(vlm_analysis, zones),
            "visual_hierarchy": self._analyze_visual_hierarchy(vlm_analysis),
            "text_blocks": self._identify_text_blocks(vlm_analysis),
            "alignment_analysis": self._analyze_alignment(vlm_analysis),
            "layout_quality": self._assess_layout_quality(vlm_analysis, zones),
            "metadata": {
                "image_size": image.size,
                "analysis_confidence": self._calculate_layout_confidence(vlm_analysis, zones)
            }
        }
        
        return layout_analysis
    
    def _analyze_document_structure(self, vlm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la structure générale du document"""
        structure = {
            "type": "unknown",
            "orientation": "portrait",
            "sections": [],
            "complexity": "simple",
            "format": "single_page"
        }
        
        # Analyse du contenu VLM pour déterminer la structure
        basic_desc = vlm_analysis.get("basic_description", "").lower()
        detailed_analysis = vlm_analysis.get("detailed_analysis", {})
        
        # Détection du type de document
        if any(term in basic_desc for term in ["facture", "invoice"]):
            structure["type"] = "invoice"
        elif any(term in basic_desc for term in ["devis", "quote"]):
            structure["type"] = "quote"
        elif any(term in basic_desc for term in ["commande", "order"]):
            structure["type"] = "order"
        
        # Analyse de la complexité
        complexity_indicators = 0
        if any(term in basic_desc for term in ["tableau", "table"]):
            complexity_indicators += 1
        if any(term in basic_desc for term in ["multiple", "plusieurs"]):
            complexity_indicators += 1
        if len(detailed_analysis) > 5:
            complexity_indicators += 1
        
        if complexity_indicators >= 2:
            structure["complexity"] = "complex"
        elif complexity_indicators == 1:
            structure["complexity"] = "medium"
        
        # Identification des sections
        structure["sections"] = self._identify_document_sections(vlm_analysis)
        
        return structure
    
    def _analyze_spatial_organization(self, vlm_analysis: Dict[str, Any], 
                                    zones: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse l'organisation spatiale"""
        spatial_org = {
            "layout_type": "standard",
            "column_count": 1,
            "regions": {},
            "flow_direction": "top_to_bottom",
            "spacing_analysis": {}
        }
        
        # Analyse des zones pour déterminer l'organisation
        zones_count = 0
        # Compter les zones de type dictionnaire détectées
        for key, value in zones.items():
            if isinstance(value, dict) and value.get("detected", False):
                zones_count += 1
        # Compter les zones de type liste
        zones_count += len(zones.get("tables", [])) if isinstance(zones.get("tables", []), list) else 0
        zones_count += len(zones.get("address_blocks", [])) if isinstance(zones.get("address_blocks", []), list) else 0
        zones_count += len(zones.get("amount_zones", [])) if isinstance(zones.get("amount_zones", []), list) else 0
        
        if zones_count > 5:
            spatial_org["layout_type"] = "complex"
        elif zones_count > 3:
            spatial_org["layout_type"] = "structured"
        
        # Analyse du nombre de colonnes
        spatial_org["column_count"] = self._estimate_column_count(vlm_analysis)
        
        # Analyse des régions
        spatial_org["regions"] = self._map_document_regions(vlm_analysis, zones)
        
        return spatial_org
    
    def _analyze_visual_hierarchy(self, vlm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la hiérarchie visuelle"""
        hierarchy = {
            "primary_elements": [],
            "secondary_elements": [],
            "emphasis_techniques": [],
            "readability_score": 0.0
        }
        
        # Extraction des éléments primaires et secondaires depuis l'analyse VLM
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            answer_str = str(answer).lower()
            
            # Éléments primaires (titres, totaux, etc.)
            if any(term in answer_str for term in ["titre", "total", "montant", "principal"]):
                hierarchy["primary_elements"].append({
                    "type": self._classify_element_type(answer_str),
                    "content": answer,
                    "importance": "high"
                })
            
            # Techniques d'emphase
            if any(term in answer_str for term in ["gras", "bold", "grand", "large"]):
                hierarchy["emphasis_techniques"].append("bold_text")
            if any(term in answer_str for term in ["couleur", "color"]):
                hierarchy["emphasis_techniques"].append("color")
            if any(term in answer_str for term in ["encadré", "bordered"]):
                hierarchy["emphasis_techniques"].append("borders")
        
        # Score de lisibilité basique
        hierarchy["readability_score"] = self._calculate_readability_score(vlm_analysis)
        
        return hierarchy
    
    def _identify_text_blocks(self, vlm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les blocs de texte"""
        text_blocks = []
        
        # Analyse des réponses pour identifier les blocs de texte distincts
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            if len(str(answer)) > 20:  # Seulement les réponses substantielles
                block = {
                    "content": answer,
                    "type": self._classify_text_block_type(question, answer),
                    "estimated_position": self._estimate_text_position(question),
                    "importance": self._assess_text_importance(answer)
                }
                text_blocks.append(block)
        
        return text_blocks
    
    def _analyze_alignment(self, vlm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse l'alignement des éléments"""
        alignment = {
            "main_alignment": "left",
            "consistency": "good",
            "alignment_patterns": [],
            "margin_analysis": {}
        }
        
        # Analyse basique de l'alignement à partir des descriptions VLM
        basic_desc = vlm_analysis.get("basic_description", "").lower()
        
        if any(term in basic_desc for term in ["centré", "center"]):
            alignment["main_alignment"] = "center"
        elif any(term in basic_desc for term in ["droite", "right"]):
            alignment["main_alignment"] = "right"
        
        # Évaluation de la consistance
        if any(term in basic_desc for term in ["organisé", "structured", "aligné"]):
            alignment["consistency"] = "excellent"
        elif any(term in basic_desc for term in ["désorganisé", "messy"]):
            alignment["consistency"] = "poor"
        
        return alignment
    
    def _assess_layout_quality(self, vlm_analysis: Dict[str, Any], 
                              zones: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue la qualité de la mise en page"""
        quality = {
            "overall_score": 0.0,
            "clarity": 0.0,
            "organization": 0.0,
            "completeness": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Évaluation de la clarté
        clarity_score = 0.8  # Score de base
        basic_desc = vlm_analysis.get("basic_description", "").lower()
        
        if any(term in basic_desc for term in ["claire", "clear", "lisible"]):
            clarity_score += 0.2
        if any(term in basic_desc for term in ["flou", "illisible", "blur"]):
            clarity_score -= 0.3
            quality["issues"].append("Image quality issues detected")
        
        quality["clarity"] = min(max(clarity_score, 0.0), 1.0)
        
        # Évaluation de l'organisation
        detected_zones = sum([
            1 if isinstance(zones.get("header", {}), dict) and zones.get("header", {}).get("detected", False) else 0,
            1 if isinstance(zones.get("footer", {}), dict) and zones.get("footer", {}).get("detected", False) else 0,
            len(zones.get("tables", [])) if isinstance(zones.get("tables", []), list) else 0,
            len(zones.get("address_blocks", [])) if isinstance(zones.get("address_blocks", []), list) else 0,
            len(zones.get("amount_zones", [])) if isinstance(zones.get("amount_zones", []), list) else 0
        ])
        
        quality["organization"] = min(detected_zones / 5.0, 1.0)
        
        # Évaluation de la complétude
        required_elements = ["header", "amount_zones"]
        present_elements = 0
        for elem in required_elements:
            zone_data = zones.get(elem, {})
            if isinstance(zone_data, dict):
                # Pour les zones de type dictionnaire (header, footer)
                if zone_data.get("detected", False):
                    present_elements += 1
            elif isinstance(zone_data, list):
                # Pour les zones de type liste (tables, address_blocks, amount_zones)
                if len(zone_data) > 0:
                    present_elements += 1
        
        quality["completeness"] = present_elements / len(required_elements)
        
        # Score global
        quality["overall_score"] = (
            quality["clarity"] * 0.4 + 
            quality["organization"] * 0.4 + 
            quality["completeness"] * 0.2
        )
        
        # Recommandations
        if quality["clarity"] < 0.6:
            quality["recommendations"].append("Améliorer la qualité d'image")
        if quality["organization"] < 0.5:
            quality["recommendations"].append("Améliorer la détection des zones")
        if quality["completeness"] < 0.7:
            quality["recommendations"].append("Vérifier les éléments manquants")
        
        return quality
    
    def _identify_document_sections(self, vlm_analysis: Dict[str, Any]) -> List[str]:
        """Identifie les sections du document"""
        sections = []
        
        # Sections typiques d'une facture
        potential_sections = {
            "header": ["en-tête", "header", "titre"],
            "sender_info": ["émetteur", "sender", "entreprise"],
            "recipient_info": ["destinataire", "recipient", "client"],
            "invoice_details": ["détails", "numéro", "date"],
            "items_table": ["tableau", "articles", "services"],
            "totals": ["total", "montant", "somme"],
            "footer": ["pied", "footer", "conditions"]
        }
        
        all_text = str(vlm_analysis.get("basic_description", "")).lower()
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            all_text += " " + str(answer).lower()
        
        for section, keywords in potential_sections.items():
            if any(keyword in all_text for keyword in keywords):
                sections.append(section)
        
        return sections
    
    def _estimate_column_count(self, vlm_analysis: Dict[str, Any]) -> int:
        """Estime le nombre de colonnes"""
        # Analyse basique du nombre de colonnes
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            if "colonne" in str(answer).lower():
                # Recherche de nombres
                numbers = re.findall(r'\b(\d+)\b', str(answer))
                if numbers:
                    col_count = max(int(num) for num in numbers if int(num) <= 10)
                    return col_count
        
        return 1  # Valeur par défaut
    
    def _map_document_regions(self, vlm_analysis: Dict[str, Any], 
                             zones: Dict[str, Any]) -> Dict[str, Any]:
        """Mappe les régions du document"""
        regions = {
            "top": {"elements": [], "density": "low"},
            "middle": {"elements": [], "density": "medium"},
            "bottom": {"elements": [], "density": "low"}
        }
        
        # Répartition basique des zones détectées
        header_zone = zones.get("header", {})
        if isinstance(header_zone, dict) and header_zone.get("detected", False):
            regions["top"]["elements"].append("header")
            regions["top"]["density"] = "medium"
        
        tables_zone = zones.get("tables", [])
        if isinstance(tables_zone, list) and len(tables_zone) > 0:
            regions["middle"]["elements"].extend(["table"] * len(tables_zone))
            regions["middle"]["density"] = "high"
        
        footer_zone = zones.get("footer", {})
        if isinstance(footer_zone, dict) and footer_zone.get("detected", False):
            regions["bottom"]["elements"].append("footer")
            regions["bottom"]["density"] = "medium"
        
        return regions
    
    def _classify_element_type(self, text: str) -> str:
        """Classifie le type d'élément"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["titre", "title"]):
            return "title"
        elif any(term in text_lower for term in ["total", "montant"]):
            return "amount"
        elif any(term in text_lower for term in ["date"]):
            return "date"
        elif any(term in text_lower for term in ["adresse", "address"]):
            return "address"
        else:
            return "text"
    
    def _classify_text_block_type(self, question: str, answer: str) -> str:
        """Classifie le type de bloc de texte"""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["en-tête", "header"]):
            return "header_text"
        elif any(term in question_lower for term in ["tableau", "table"]):
            return "table_text"
        elif any(term in question_lower for term in ["total", "montant"]):
            return "amount_text"
        else:
            return "body_text"
    
    def _estimate_text_position(self, question: str) -> str:
        """Estime la position du texte"""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ["haut", "top", "en-tête"]):
            return "top"
        elif any(term in question_lower for term in ["bas", "bottom", "pied"]):
            return "bottom"
        else:
            return "middle"
    
    def _assess_text_importance(self, text: str) -> str:
        """Évalue l'importance du texte"""
        text_lower = str(text).lower()
        
        if any(term in text_lower for term in ["total", "montant", "facture", "invoice"]):
            return "high"
        elif any(term in text_lower for term in ["date", "numéro", "reference"]):
            return "medium"
        else:
            return "low"
    
    def _calculate_readability_score(self, vlm_analysis: Dict[str, Any]) -> float:
        """Calcule un score de lisibilité"""
        score = 0.7  # Score de base
        
        basic_desc = vlm_analysis.get("basic_description", "").lower()
        
        # Facteurs positifs
        if any(term in basic_desc for term in ["claire", "clear", "lisible"]):
            score += 0.2
        if any(term in basic_desc for term in ["organisé", "structured"]):
            score += 0.1
        
        # Facteurs négatifs
        if any(term in basic_desc for term in ["flou", "blur", "illisible"]):
            score -= 0.3
        if any(term in basic_desc for term in ["désorganisé", "messy"]):
            score -= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_layout_confidence(self, vlm_analysis: Dict[str, Any], 
                                   zones: Dict[str, Any]) -> float:
        """Calcule la confiance de l'analyse de mise en page"""
        # Confiance basée sur le nombre de zones détectées et la qualité de l'analyse
        zones_detected = sum([
            1 if isinstance(zones.get("header", {}), dict) and zones.get("header", {}).get("detected", False) else 0,
            1 if isinstance(zones.get("footer", {}), dict) and zones.get("footer", {}).get("detected", False) else 0,
            len(zones.get("tables", [])) if isinstance(zones.get("tables", []), list) else 0,
            len(zones.get("address_blocks", [])) if isinstance(zones.get("address_blocks", []), list) else 0,
            len(zones.get("amount_zones", [])) if isinstance(zones.get("amount_zones", []), list) else 0
        ])
        
        base_confidence = min(zones_detected / 5.0, 1.0) * 0.7
        
        # Bonus pour la qualité de l'analyse VLM
        if len(vlm_analysis.get("detailed_analysis", {})) > 3:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)