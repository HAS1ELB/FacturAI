"""
Détecteur de zones clés pour les factures
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class ZoneDetector:
    """
    Détecteur de zones clés dans les factures
    
    Identifie et classifie les différentes zones :
    - En-tête (header)
    - Pied de page (footer) 
    - Tableaux (tables)
    - Blocs d'adresse (address_blocks)
    - Zones de montants (amount_zones)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le détecteur de zones
        
        Args:
            config: Configuration de détection des zones
        """
        self.config = config
        self.header_keywords = config.get("header_keywords", [])
        self.footer_keywords = config.get("footer_keywords", [])
        self.table_keywords = config.get("table_keywords", [])
        self.address_keywords = config.get("address_keywords", [])
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
    
    def detect_zones(self, image: Image.Image, vlm_analysis: Dict[str, Any], 
                     ocr_results: Dict = None) -> Dict[str, Any]:
        """
        Détecte les zones clés dans une facture
        
        Args:
            image: Image PIL de la facture
            vlm_analysis: Résultats de l'analyse VLM
            ocr_results: Résultats OCR optionnels
        
        Returns:
            Dictionnaire des zones détectées
        """
        zones = {
            "header": self._detect_header_zone(vlm_analysis, ocr_results),
            "footer": self._detect_footer_zone(vlm_analysis, ocr_results),
            "tables": self._detect_table_zones(vlm_analysis, ocr_results),
            "address_blocks": self._detect_address_zones(vlm_analysis, ocr_results),
            "amount_zones": self._detect_amount_zones(vlm_analysis, ocr_results),
            "metadata": {
                "image_size": image.size,
                "detection_confidence": self._calculate_overall_confidence(),
                "zones_count": 0  # Will be updated
            }
        }
        
        # Compter les zones détectées
        zones["metadata"]["zones_count"] = self._count_detected_zones(zones)
        
        return zones
    
    def _detect_header_zone(self, vlm_analysis: Dict[str, Any], 
                           ocr_results: Dict = None) -> Dict[str, Any]:
        """Détecte la zone d'en-tête"""
        header_info = {
            "detected": False,
            "confidence": 0.0,
            "content": {},
            "coordinates": None,
            "elements": []
        }
        
        # Analyse VLM pour l'en-tête
        vlm_text = str(vlm_analysis.get("basic_description", "")).lower()
        detailed_analysis = vlm_analysis.get("detailed_analysis", {})
        
        # Recherche des indicateurs d'en-tête
        header_indicators = 0
        total_indicators = len(self.header_keywords)
        
        for keyword in self.header_keywords:
            if keyword.lower() in vlm_text:
                header_indicators += 1
        
        # Analyse des réponses détaillées
        for question, answer in detailed_analysis.items():
            if "en-tête" in question.lower() or "header" in question.lower():
                header_info["content"][question] = answer
                if any(kw in str(answer).lower() for kw in self.header_keywords):
                    header_indicators += 2
        
        # Calcul de la confiance
        if total_indicators > 0:
            confidence = min(header_indicators / total_indicators, 1.0)
        else:
            confidence = 0.5 if header_indicators > 0 else 0.0
        
        header_info["confidence"] = confidence
        header_info["detected"] = confidence > self.confidence_threshold
        
        # Analyse OCR pour affiner la détection
        if ocr_results and header_info["detected"]:
            header_info.update(self._analyze_ocr_for_header(ocr_results))
        
        return header_info
    
    def _detect_footer_zone(self, vlm_analysis: Dict[str, Any], 
                           ocr_results: Dict = None) -> Dict[str, Any]:
        """Détecte la zone de pied de page"""
        footer_info = {
            "detected": False,
            "confidence": 0.0,
            "content": {},
            "coordinates": None,
            "elements": [],
            "totals": []
        }
        
        # Analyse VLM pour le pied de page
        vlm_text = str(vlm_analysis.get("basic_description", "")).lower()
        detailed_analysis = vlm_analysis.get("detailed_analysis", {})
        
        # Recherche des indicateurs de pied de page
        footer_indicators = 0
        total_indicators = len(self.footer_keywords)
        
        for keyword in self.footer_keywords:
            if keyword.lower() in vlm_text:
                footer_indicators += 1
        
        # Analyse des réponses détaillées
        for question, answer in detailed_analysis.items():
            if any(term in question.lower() for term in ["total", "montant", "pied"]):
                footer_info["content"][question] = answer
                if any(kw in str(answer).lower() for kw in self.footer_keywords):
                    footer_indicators += 2
        
        # Calcul de la confiance
        if total_indicators > 0:
            confidence = min(footer_indicators / total_indicators, 1.0)
        else:
            confidence = 0.5 if footer_indicators > 0 else 0.0
        
        footer_info["confidence"] = confidence
        footer_info["detected"] = confidence > self.confidence_threshold
        
        # Extraction des montants
        footer_info["totals"] = self._extract_amounts_from_analysis(vlm_analysis)
        
        return footer_info
    
    def _detect_table_zones(self, vlm_analysis: Dict[str, Any], 
                           ocr_results: Dict = None) -> List[Dict[str, Any]]:
        """Détecte les zones de tableaux"""
        tables = []
        
        # Analyse VLM pour les tableaux
        vlm_text = str(vlm_analysis.get("basic_description", "")).lower()
        detailed_analysis = vlm_analysis.get("detailed_analysis", {})
        
        # Recherche d'indicateurs de tableaux
        table_detected = False
        table_content = {}
        
        for keyword in self.table_keywords:
            if keyword.lower() in vlm_text:
                table_detected = True
                break
        
        # Analyse des réponses détaillées
        for question, answer in detailed_analysis.items():
            if "tableau" in question.lower() or "table" in question.lower():
                table_content[question] = answer
                if any(kw in str(answer).lower() for kw in ["oui", "yes", "tableau", "table"]):
                    table_detected = True
        
        if table_detected:
            table_info = {
                "detected": True,
                "confidence": 0.8,  # Base confidence for detected tables
                "content": table_content,
                "coordinates": None,
                "columns": self._extract_table_structure(vlm_analysis),
                "rows_count": self._estimate_rows_count(vlm_analysis)
            }
            tables.append(table_info)
        
        return tables
    
    def _detect_address_zones(self, vlm_analysis: Dict[str, Any], 
                             ocr_results: Dict = None) -> List[Dict[str, Any]]:
        """Détecte les blocs d'adresse"""
        address_blocks = []
        
        # Analyse VLM pour les adresses
        detailed_analysis = vlm_analysis.get("detailed_analysis", {})
        
        for question, answer in detailed_analysis.items():
            if any(term in question.lower() for term in ["adresse", "address", "émetteur", "destinataire"]):
                # Extraction des informations d'adresse
                address_info = self._parse_address_info(answer)
                if address_info:
                    address_block = {
                        "detected": True,
                        "confidence": 0.7,
                        "content": answer,
                        "parsed_info": address_info,
                        "type": self._classify_address_type(question, answer),
                        "coordinates": None
                    }
                    address_blocks.append(address_block)
        
        return address_blocks
    
    def _detect_amount_zones(self, vlm_analysis: Dict[str, Any], 
                            ocr_results: Dict = None) -> List[Dict[str, Any]]:
        """Détecte les zones de montants"""
        amount_zones = []
        
        # Extraction des montants depuis l'analyse VLM
        amounts = self._extract_amounts_from_analysis(vlm_analysis)
        
        for amount in amounts:
            amount_zone = {
                "detected": True,
                "confidence": amount.get("confidence", 0.6),
                "value": amount.get("value"),
                "currency": amount.get("currency", "EUR"),
                "type": amount.get("type", "unknown"),
                "coordinates": None,
                "context": amount.get("context", "")
            }
            amount_zones.append(amount_zone)
        
        return amount_zones
    
    def _extract_amounts_from_analysis(self, vlm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait les montants depuis l'analyse VLM"""
        amounts = []
        
        # Patterns pour détecter les montants
        amount_patterns = [
            r'(\d+[,.]?\d*)\s*€',
            r'(\d+[,.]?\d*)\s*EUR',
            r'€\s*(\d+[,.]?\d*)',
            r'(\d+[,.]?\d*)\s*euros?',
            r'Total[:\s]*(\d+[,.]?\d*)',
            r'Montant[:\s]*(\d+[,.]?\d*)',
        ]
        
        # Recherche dans toutes les réponses VLM
        all_text = str(vlm_analysis.get("basic_description", ""))
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            all_text += " " + str(answer)
        
        for pattern in amount_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1)
                try:
                    value = float(value_str.replace(',', '.'))
                    amount_info = {
                        "value": value,
                        "currency": "EUR",
                        "confidence": 0.7,
                        "context": match.group(0),
                        "type": self._classify_amount_type(match.group(0))
                    }
                    amounts.append(amount_info)
                except ValueError:
                    continue
        
        return amounts
    
    def _classify_amount_type(self, context: str) -> str:
        """Classifie le type de montant basé sur le contexte"""
        context_lower = context.lower()
        
        if any(term in context_lower for term in ["total", "somme"]):
            return "total"
        elif any(term in context_lower for term in ["tva", "tax"]):
            return "tax"
        elif any(term in context_lower for term in ["ht", "hors"]):
            return "subtotal"
        elif any(term in context_lower for term in ["ttc", "toutes"]):
            return "total_with_tax"
        else:
            return "amount"
    
    def _extract_table_structure(self, vlm_analysis: Dict[str, Any]) -> List[str]:
        """Extrait la structure des colonnes de tableau"""
        columns = []
        
        # Recherche d'informations sur les colonnes
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            if "tableau" in question.lower() and "colonne" in answer.lower():
                # Extraction basique des noms de colonnes
                column_patterns = [
                    r'colonnes?[:\s]*([^.]+)',
                    r'en-têtes?[:\s]*([^.]+)',
                ]
                
                for pattern in column_patterns:
                    match = re.search(pattern, answer, re.IGNORECASE)
                    if match:
                        cols_text = match.group(1)
                        # Séparer les colonnes
                        cols = [col.strip() for col in re.split(r'[,;]', cols_text)]
                        columns.extend(cols)
        
        return list(set(columns))  # Supprimer les doublons
    
    def _estimate_rows_count(self, vlm_analysis: Dict[str, Any]) -> int:
        """Estime le nombre de lignes dans les tableaux"""
        # Recherche d'informations sur le nombre de lignes
        for question, answer in vlm_analysis.get("detailed_analysis", {}).items():
            if "ligne" in answer.lower():
                # Recherche de nombres
                numbers = re.findall(r'\b(\d+)\b', answer)
                if numbers:
                    return max(int(num) for num in numbers if int(num) < 100)  # Limite raisonnable
        
        return 0
    
    def _parse_address_info(self, address_text: str) -> Dict[str, Any]:
        """Parse les informations d'adresse"""
        if not address_text or len(address_text.strip()) < 10:
            return None
        
        address_info = {
            "raw_text": address_text,
            "company": None,
            "street": None,
            "city": None,
            "postal_code": None,
            "country": None
        }
        
        # Patterns pour extraire les informations
        postal_code_pattern = r'\b(\d{5})\b'
        postal_match = re.search(postal_code_pattern, address_text)
        if postal_match:
            address_info["postal_code"] = postal_match.group(1)
        
        return address_info
    
    def _classify_address_type(self, question: str, answer: str) -> str:
        """Classifie le type d'adresse"""
        question_lower = question.lower()
        
        if "émetteur" in question_lower or "sender" in question_lower:
            return "sender"
        elif "destinataire" in question_lower or "recipient" in question_lower:
            return "recipient"
        else:
            return "unknown"
    
    def _analyze_ocr_for_header(self, ocr_results: Dict) -> Dict[str, Any]:
        """Analyse les résultats OCR pour affiner la détection d'en-tête"""
        # Cette méthode peut être étendue pour utiliser les coordonnées OCR
        return {
            "ocr_enhanced": True,
            "coordinates": None  # À implémenter avec les coordonnées OCR
        }
    
    def _calculate_overall_confidence(self) -> float:
        """Calcule la confiance globale de détection"""
        # Méthode simple, peut être améliorée
        return 0.7
    
    def _count_detected_zones(self, zones: Dict[str, Any]) -> int:
        """Compte le nombre de zones détectées"""
        count = 0
        
        if zones.get("header", {}).get("detected", False):
            count += 1
        if zones.get("footer", {}).get("detected", False):
            count += 1
        
        count += len(zones.get("tables", []))
        count += len(zones.get("address_blocks", []))
        count += len(zones.get("amount_zones", []))
        
        return count