"""
Module de visualisation pour les résultats VLM
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import json

logger = logging.getLogger(__name__)

class VLMVisualizer:
    """
    Visualiseur pour les résultats d'analyse VLM
    
    Fonctionnalités :
    - Annotation des zones détectées
    - Visualisation des structures
    - Génération de rapports visuels
    - Export des résultats annotés
    """
    
    def __init__(self, output_dir: str = "Data/vlm_visualizations"):
        """
        Initialise le visualiseur
        
        Args:
            output_dir: Répertoire de sortie pour les visualisations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration des couleurs pour les différents types de zones
        self.zone_colors = {
            "header": "#FF6B6B",      # Rouge
            "footer": "#4ECDC4",      # Turquoise
            "table": "#45B7D1",       # Bleu
            "address": "#96CEB4",     # Vert
            "amount": "#FFEAA7",      # Jaune
            "text": "#DDA0DD",        # Violet clair
            "unknown": "#808080"      # Gris
        }
        
        # Configuration des styles
        self.text_size = 12
        self.line_width = 2
        self.alpha = 128  # Transparence pour les zones
    
    def visualize_analysis_results(self, image_path: str, vlm_results: Dict[str, Any], 
                                  save_path: str = None) -> str:
        """
        Visualise les résultats complets d'analyse VLM
        
        Args:
            image_path: Chemin vers l'image originale
            vlm_results: Résultats de l'analyse VLM
            save_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Chemin vers l'image annotée
        """
        try:
            # Chargement de l'image
            image = Image.open(image_path).convert("RGBA")
            
            # Création d'une couche de transparence pour les annotations
            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Annotation des zones détectées
            zones = vlm_results.get("detected_zones", {})
            self._draw_zones(draw, zones, image.size)
            
            # Ajout des informations textuelles
            self._add_text_annotations(draw, vlm_results)
            
            # Fusion de l'image originale avec les annotations
            annotated_image = Image.alpha_composite(image, overlay)
            annotated_image = annotated_image.convert("RGB")
            
            # Sauvegarde
            if save_path is None:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(self.output_dir, f"annotated_{image_name}.jpg")
            
            annotated_image.save(save_path, "JPEG", quality=95)
            logger.info(f"Image annotée sauvegardée: {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation: {e}")
            return None
    
    def _draw_zones(self, draw: ImageDraw.Draw, zones: Dict[str, Any], image_size: Tuple[int, int]):
        """Dessine les zones détectées"""
        # En-tête
        if zones.get("header", {}).get("detected", False):
            self._draw_zone_rectangle(draw, "header", (0, 0, image_size[0], image_size[1] // 4))
        
        # Pied de page
        if zones.get("footer", {}).get("detected", False):
            footer_y = int(image_size[1] * 0.75)
            self._draw_zone_rectangle(draw, "footer", (0, footer_y, image_size[0], image_size[1]))
        
        # Tableaux
        for i, table in enumerate(zones.get("tables", [])):
            if table.get("detected", False):
                # Zone de tableau estimée (centre de l'image)
                table_y1 = int(image_size[1] * 0.3)
                table_y2 = int(image_size[1] * 0.7)
                self._draw_zone_rectangle(draw, "table", (50, table_y1, image_size[0] - 50, table_y2))
                self._add_zone_label(draw, f"Table {i+1}", (60, table_y1 + 10))
        
        # Blocs d'adresse
        for i, address in enumerate(zones.get("address_blocks", [])):
            # Position estimée pour les adresses
            addr_x = 50 + (i * 200)
            addr_y = int(image_size[1] * 0.15)
            self._draw_zone_rectangle(draw, "address", (addr_x, addr_y, addr_x + 180, addr_y + 80))
            self._add_zone_label(draw, f"Address {i+1}", (addr_x + 5, addr_y + 5))
        
        # Zones de montants
        for i, amount in enumerate(zones.get("amount_zones", [])):
            # Position estimée pour les montants (côté droit, bas)
            amount_x = int(image_size[0] * 0.7)
            amount_y = int(image_size[1] * 0.6) + (i * 30)
            self._draw_zone_rectangle(draw, "amount", (amount_x, amount_y, image_size[0] - 20, amount_y + 25))
            
            # Affichage de la valeur
            value = amount.get("value", "N/A")
            currency = amount.get("currency", "")
            self._add_zone_label(draw, f"{value} {currency}", (amount_x + 5, amount_y + 5))
    
    def _draw_zone_rectangle(self, draw: ImageDraw.Draw, zone_type: str, coordinates: Tuple[int, int, int, int]):
        """Dessine un rectangle pour une zone"""
        color = self.zone_colors.get(zone_type, self.zone_colors["unknown"])
        
        # Conversion de la couleur hex en RGB avec alpha
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        color_rgba = color_rgb + (self.alpha,)
        
        # Rectangle semi-transparent
        draw.rectangle(coordinates, fill=color_rgba, outline=color_rgb, width=self.line_width)
    
    def _add_zone_label(self, draw: ImageDraw.Draw, text: str, position: Tuple[int, int]):
        """Ajoute un label pour une zone"""
        try:
            # Tentative d'utilisation d'une police système
            font = ImageFont.truetype("arial.ttf", self.text_size)
        except (OSError, IOError):
            # Police par défaut si arial n'est pas disponible
            font = ImageFont.load_default()
        
        # Fond pour le texte
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill=(255, 255, 255, 200))
        
        # Texte
        draw.text(position, text, fill=(0, 0, 0, 255), font=font)
    
    def _add_text_annotations(self, draw: ImageDraw.Draw, vlm_results: Dict[str, Any]):
        """Ajoute les annotations textuelles"""
        # Informations sur le modèle utilisé
        model_info = f"Model: {vlm_results.get('model_used', 'Unknown')}"
        processing_time = vlm_results.get('processing_time', 0)
        time_info = f"Time: {processing_time:.2f}s"
        
        # Position en haut à gauche
        info_text = f"{model_info} | {time_info}"
        self._add_zone_label(draw, info_text, (10, 10))
    
    def generate_analysis_report(self, vlm_results: Dict[str, Any], save_path: str = None) -> str:
        """
        Génère un rapport textuel de l'analyse
        
        Args:
            vlm_results: Résultats de l'analyse VLM
            save_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Chemin vers le rapport généré
        """
        if save_path is None:
            timestamp = vlm_results.get("timestamp", "unknown")
            save_path = os.path.join(self.output_dir, f"vlm_report_{timestamp}.txt")
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT D'ANALYSE VLM ===\n\n")
                
                # Informations générales
                f.write("INFORMATIONS GÉNÉRALES:\n")
                f.write(f"Image: {vlm_results.get('image_path', 'N/A')}\n")
                f.write(f"Modèle utilisé: {vlm_results.get('model_used', 'N/A')}\n")
                f.write(f"Temps de traitement: {vlm_results.get('processing_time', 0):.2f}s\n")
                f.write(f"Timestamp: {vlm_results.get('timestamp', 'N/A')}\n\n")
                
                # Analyse VLM
                vlm_analysis = vlm_results.get("vlm_analysis", {})
                f.write("ANALYSE VLM:\n")
                f.write(f"Description: {vlm_analysis.get('basic_description', 'N/A')}\n")
                f.write(f"Confiance: {vlm_analysis.get('confidence', 0):.2f}\n\n")
                
                # Analyse détaillée
                detailed = vlm_analysis.get("detailed_analysis", {})
                if detailed:
                    f.write("ANALYSE DÉTAILLÉE:\n")
                    for question, answer in detailed.items():
                        f.write(f"Q: {question}\n")
                        f.write(f"R: {answer}\n\n")
                
                # Zones détectées
                zones = vlm_results.get("detected_zones", {})
                f.write("ZONES DÉTECTÉES:\n")
                
                # En-tête
                header = zones.get("header", {})
                f.write(f"En-tête: {'Détecté' if header.get('detected', False) else 'Non détecté'}")
                f.write(f" (Confiance: {header.get('confidence', 0):.2f})\n")
                
                # Pied de page
                footer = zones.get("footer", {})
                f.write(f"Pied de page: {'Détecté' if footer.get('detected', False) else 'Non détecté'}")
                f.write(f" (Confiance: {footer.get('confidence', 0):.2f})\n")
                
                # Tableaux
                tables = zones.get("tables", [])
                f.write(f"Tableaux: {len(tables)} détecté(s)\n")
                for i, table in enumerate(tables):
                    f.write(f"  Table {i+1}: {table.get('rows_count', 0)} lignes, ")
                    f.write(f"{len(table.get('columns', []))} colonnes\n")
                
                # Blocs d'adresse
                addresses = zones.get("address_blocks", [])
                f.write(f"Adresses: {len(addresses)} détectée(s)\n")
                for i, addr in enumerate(addresses):
                    f.write(f"  Adresse {i+1}: {addr.get('type', 'unknown')}\n")
                
                # Montants
                amounts = zones.get("amount_zones", [])
                f.write(f"Montants: {len(amounts)} détecté(s)\n")
                for i, amount in enumerate(amounts):
                    value = amount.get("value", "N/A")
                    currency = amount.get("currency", "")
                    amount_type = amount.get("type", "unknown")
                    f.write(f"  Montant {i+1}: {value} {currency} ({amount_type})\n")
                
                # Analyse de mise en page
                layout = vlm_results.get("layout_analysis", {})
                if layout:
                    f.write("\nANALYSE DE MISE EN PAGE:\n")
                    structure = layout.get("document_structure", {})
                    f.write(f"Type de document: {structure.get('type', 'unknown')}\n")
                    f.write(f"Complexité: {structure.get('complexity', 'unknown')}\n")
                    
                    quality = layout.get("layout_quality", {})
                    f.write(f"Score de qualité: {quality.get('overall_score', 0):.2f}\n")
                    f.write(f"Clarté: {quality.get('clarity', 0):.2f}\n")
                    f.write(f"Organisation: {quality.get('organization', 0):.2f}\n")
                    f.write(f"Complétude: {quality.get('completeness', 0):.2f}\n")
                
                f.write("\n=== FIN DU RAPPORT ===\n")
            
            logger.info(f"Rapport généré: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            return None
    
    def create_comparison_visualization(self, results_list: List[Dict[str, Any]], 
                                      save_path: str = None) -> str:
        """
        Crée une visualisation comparative de plusieurs analyses
        
        Args:
            results_list: Liste de résultats d'analyse VLM
            save_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Chemin vers la visualisation comparative
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "vlm_comparison.json")
        
        try:
            comparison_data = {
                "timestamp": "now",
                "total_analyses": len(results_list),
                "models_used": [],
                "performance_summary": {},
                "zone_detection_summary": {},
                "detailed_results": results_list
            }
            
            # Extraction des modèles utilisés
            models = set()
            total_time = 0
            zone_counts = {
                "header": 0,
                "footer": 0,
                "tables": 0,
                "addresses": 0,
                "amounts": 0
            }
            
            for result in results_list:
                models.add(result.get("model_used", "unknown"))
                total_time += result.get("processing_time", 0)
                
                # Comptage des zones
                zones = result.get("detected_zones", {})
                if zones.get("header", {}).get("detected", False):
                    zone_counts["header"] += 1
                if zones.get("footer", {}).get("detected", False):
                    zone_counts["footer"] += 1
                zone_counts["tables"] += len(zones.get("tables", []))
                zone_counts["addresses"] += len(zones.get("address_blocks", []))
                zone_counts["amounts"] += len(zones.get("amount_zones", []))
            
            comparison_data["models_used"] = list(models)
            comparison_data["performance_summary"] = {
                "total_processing_time": total_time,
                "average_processing_time": total_time / len(results_list) if results_list else 0,
                "fastest_analysis": min((r.get("processing_time", float('inf')) for r in results_list), default=0),
                "slowest_analysis": max((r.get("processing_time", 0) for r in results_list), default=0)
            }
            comparison_data["zone_detection_summary"] = zone_counts
            
            # Sauvegarde du fichier de comparaison
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Visualisation comparative sauvegardée: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la comparaison: {e}")
            return None
    
    def export_zones_to_json(self, vlm_results: Dict[str, Any], save_path: str = None) -> str:
        """
        Exporte les zones détectées au format JSON
        
        Args:
            vlm_results: Résultats de l'analyse VLM
            save_path: Chemin de sauvegarde (optionnel)
        
        Returns:
            Chemin vers le fichier JSON
        """
        if save_path is None:
            timestamp = vlm_results.get("timestamp", "unknown")
            save_path = os.path.join(self.output_dir, f"zones_{timestamp}.json")
        
        try:
            zones_data = {
                "image_path": vlm_results.get("image_path"),
                "model_used": vlm_results.get("model_used"),
                "timestamp": vlm_results.get("timestamp"),
                "zones": vlm_results.get("detected_zones", {}),
                "layout": vlm_results.get("layout_analysis", {})
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(zones_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Zones exportées: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export des zones: {e}")
            return None