"""
Exemple d'intégration du module VLM avec les résultats OCR existants
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Ajouter le chemin vers le module parent
sys.path.append(str(Path(__file__).parent.parent.parent))

from vlm import VLMProcessor
from vlm.utils import VLMVisualizer, GeometryUtils

class OCRVLMIntegrator:
    """
    Intégrateur pour combiner les résultats OCR et VLM
    """
    
    def __init__(self):
        self.vlm_processor = VLMProcessor()
        self.visualizer = VLMVisualizer()
    
    def integrate_ocr_vlm(self, image_path: str, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intègre les résultats OCR avec l'analyse VLM
        
        Args:
            image_path: Chemin vers l'image
            ocr_results: Résultats OCR existants
        
        Returns:
            Résultats intégrés OCR + VLM
        """
        print(f"🔄 Intégration OCR-VLM pour: {os.path.basename(image_path)}")
        
        # Analyse VLM
        vlm_results = self.vlm_processor.process_invoice(image_path, ocr_results)
        
        # Enrichissement avec les données OCR
        enriched_results = self._enrich_with_ocr_data(vlm_results, ocr_results)
        
        # Validation croisée
        validated_results = self._cross_validate_results(enriched_results)
        
        return validated_results
    
    def _enrich_with_ocr_data(self, vlm_results: Dict[str, Any], 
                             ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les résultats VLM avec les données OCR précises"""
        
        enriched = vlm_results.copy()
        
        # Extraction des textes OCR avec coordonnées
        ocr_texts = self._extract_ocr_texts(ocr_results)
        
        # Enrichissement des zones détectées
        zones = enriched.get('detected_zones', {})
        
        # Enrichissement de l'en-tête
        if zones.get('header', {}).get('detected', False):
            header_texts = self._find_texts_in_region(ocr_texts, 'top')
            zones['header']['ocr_texts'] = header_texts
            zones['header']['coordinates'] = self._estimate_region_bbox(header_texts)
        
        # Enrichissement du pied de page
        if zones.get('footer', {}).get('detected', False):
            footer_texts = self._find_texts_in_region(ocr_texts, 'bottom')
            zones['footer']['ocr_texts'] = footer_texts
            zones['footer']['coordinates'] = self._estimate_region_bbox(footer_texts)
        
        # Enrichissement des zones de montants
        amount_zones = zones.get('amount_zones', [])
        for i, amount_zone in enumerate(amount_zones):
            # Recherche de montants dans l'OCR qui correspondent
            matching_amounts = self._find_matching_amounts(
                amount_zone.get('value'), ocr_texts
            )
            if matching_amounts:
                amount_zones[i]['ocr_matches'] = matching_amounts
                amount_zones[i]['coordinates'] = matching_amounts[0].get('bbox')
        
        # Enrichissement des tableaux
        tables = zones.get('tables', [])
        if tables and ocr_texts:
            table_structure = self._detect_table_structure_from_ocr(ocr_texts)
            for i, table in enumerate(tables):
                tables[i]['ocr_structure'] = table_structure
                tables[i]['coordinates'] = table_structure.get('bbox')
        
        # Ajout des métadonnées d'intégration
        enriched['integration_metadata'] = {
            'ocr_texts_count': len(ocr_texts),
            'ocr_confidence_avg': self._calculate_avg_ocr_confidence(ocr_texts),
            'integration_score': self._calculate_integration_score(zones, ocr_texts)
        }
        
        return enriched
    
    def _extract_ocr_texts(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrait les textes OCR avec leurs coordonnées"""
        texts = []
        
        # Support pour différents formats de résultats OCR
        if 'results' in ocr_results:
            # Format FacturAI OCR
            for result in ocr_results['results']:
                text_info = {
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0),
                    'bbox': result.get('bbox', [0, 0, 0, 0]),
                    'coordinates': result.get('bbox', [0, 0, 0, 0])
                }
                texts.append(text_info)
        
        elif 'text_annotations' in ocr_results:
            # Format Google Vision API
            for annotation in ocr_results['text_annotations']:
                vertices = annotation.get('bounding_poly', {}).get('vertices', [])
                if len(vertices) >= 4:
                    bbox = [
                        vertices[0].get('x', 0),
                        vertices[0].get('y', 0),
                        vertices[2].get('x', 0),
                        vertices[2].get('y', 0)
                    ]
                else:
                    bbox = [0, 0, 0, 0]
                
                text_info = {
                    'text': annotation.get('description', ''),
                    'confidence': annotation.get('confidence', 0.8),
                    'bbox': bbox,
                    'coordinates': bbox
                }
                texts.append(text_info)
        
        return texts
    
    def _find_texts_in_region(self, ocr_texts: List[Dict[str, Any]], 
                             region: str) -> List[Dict[str, Any]]:
        """Trouve les textes OCR dans une région spécifique"""
        region_texts = []
        
        if not ocr_texts:
            return region_texts
        
        # Calcul des dimensions de l'image approximatives
        max_y = max(text['bbox'][3] for text in ocr_texts if text['bbox'])
        
        for text in ocr_texts:
            bbox = text.get('bbox', [0, 0, 0, 0])
            if not bbox or len(bbox) < 4:
                continue
            
            y_center = (bbox[1] + bbox[3]) / 2
            
            if region == 'top' and y_center < max_y * 0.25:
                region_texts.append(text)
            elif region == 'bottom' and y_center > max_y * 0.75:
                region_texts.append(text)
            elif region == 'middle' and max_y * 0.25 <= y_center <= max_y * 0.75:
                region_texts.append(text)
        
        return region_texts
    
    def _estimate_region_bbox(self, texts: List[Dict[str, Any]]) -> List[float]:
        """Estime la bounding box d'une région à partir des textes"""
        if not texts:
            return [0, 0, 0, 0]
        
        all_bboxes = [text['bbox'] for text in texts if text.get('bbox')]
        if not all_bboxes:
            return [0, 0, 0, 0]
        
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)
        
        return [min_x, min_y, max_x, max_y]
    
    def _find_matching_amounts(self, vlm_amount: Any, 
                              ocr_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trouve les montants OCR qui correspondent au montant VLM"""
        matches = []
        
        if vlm_amount is None:
            return matches
        
        vlm_value = str(vlm_amount).replace(',', '.')
        
        for text in ocr_texts:
            text_content = text.get('text', '').replace(',', '.')
            
            # Recherche de correspondances numériques
            import re
            numbers = re.findall(r'\d+\.?\d*', text_content)
            
            for number in numbers:
                try:
                    if abs(float(number) - float(vlm_value)) < 0.01:
                        matches.append(text)
                        break
                except ValueError:
                    continue
        
        return matches
    
    def _detect_table_structure_from_ocr(self, ocr_texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Détecte la structure de tableau à partir des textes OCR"""
        table_structure = {
            'detected': False,
            'rows': 0,
            'columns': 0,
            'bbox': [0, 0, 0, 0],
            'cells': []
        }
        
        if not ocr_texts:
            return table_structure
        
        # Utilisation des utilitaires géométriques pour détecter la structure
        elements = [{'coordinates': text['bbox'], 'text': text['text']} 
                   for text in ocr_texts if text.get('bbox')]
        
        if elements:
            detected_structure = GeometryUtils.detect_table_structure(elements)
            
            table_structure.update({
                'detected': True,
                'rows': detected_structure.get('rows', 0),
                'columns': detected_structure.get('cols', 0),
                'is_regular': detected_structure.get('is_regular', False),
                'bbox': self._estimate_region_bbox(ocr_texts)
            })
            
            # Extraction des cellules
            structure_rows = detected_structure.get('structure', [])
            for row in structure_rows:
                for cell in row:
                    table_structure['cells'].append({
                        'text': cell.get('text', ''),
                        'coordinates': cell.get('coordinates', [0, 0, 0, 0])
                    })
        
        return table_structure
    
    def _calculate_avg_ocr_confidence(self, ocr_texts: List[Dict[str, Any]]) -> float:
        """Calcule la confiance moyenne des textes OCR"""
        if not ocr_texts:
            return 0.0
        
        confidences = [text.get('confidence', 0) for text in ocr_texts]
        return sum(confidences) / len(confidences)
    
    def _calculate_integration_score(self, zones: Dict[str, Any], 
                                   ocr_texts: List[Dict[str, Any]]) -> float:
        """Calcule un score d'intégration OCR-VLM"""
        score = 0.0
        max_score = 5.0
        
        # Points pour les zones avec données OCR
        if zones.get('header', {}).get('ocr_texts'):
            score += 1.0
        if zones.get('footer', {}).get('ocr_texts'):
            score += 1.0
        if any(table.get('ocr_structure') for table in zones.get('tables', [])):
            score += 1.0
        if any(amount.get('ocr_matches') for amount in zones.get('amount_zones', [])):
            score += 1.0
        
        # Point pour la qualité générale des données OCR
        if ocr_texts and self._calculate_avg_ocr_confidence(ocr_texts) > 0.7:
            score += 1.0
        
        return score / max_score
    
    def _cross_validate_results(self, enriched_results: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue une validation croisée entre OCR et VLM"""
        validated = enriched_results.copy()
        
        # Validation des montants
        zones = validated.get('detected_zones', {})
        amount_zones = zones.get('amount_zones', [])
        
        for i, amount_zone in enumerate(amount_zones):
            ocr_matches = amount_zone.get('ocr_matches', [])
            if ocr_matches:
                # Augmenter la confiance si confirmé par OCR
                amount_zones[i]['confidence'] = min(
                    amount_zone.get('confidence', 0) + 0.2, 1.0
                )
                amount_zones[i]['validated_by_ocr'] = True
            else:
                amount_zones[i]['validated_by_ocr'] = False
        
        # Score de validation globale
        integration_meta = validated.get('integration_metadata', {})
        integration_score = integration_meta.get('integration_score', 0)
        
        validated['validation_score'] = {
            'ocr_vlm_agreement': integration_score,
            'overall_confidence': self._calculate_overall_confidence(validated),
            'reliability': 'high' if integration_score > 0.7 else 'medium' if integration_score > 0.4 else 'low'
        }
        
        return validated
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calcule la confiance globale des résultats intégrés"""
        vlm_confidence = results.get('vlm_analysis', {}).get('confidence', 0)
        integration_score = results.get('integration_metadata', {}).get('integration_score', 0)
        
        # Moyenne pondérée
        return (vlm_confidence * 0.6 + integration_score * 0.4)

def main():
    """Exemple d'intégration OCR-VLM"""
    print("=== Intégration OCR-VLM FacturAI ===\n")
    
    # Chemins
    images_dir = "../Data/processed_images"
    ocr_results_dir = "../Data/ocr_results"
    
    # Initialisation
    integrator = OCRVLMIntegrator()
    
    # Recherche d'une paire image-OCR
    if not os.path.exists(images_dir) or not os.path.exists(ocr_results_dir):
        print("⚠️  Répertoires de données non trouvés")
        return
    
    # Trouver des fichiers correspondants
    image_files = list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpg"))
    
    for image_file in image_files:
        image_name = image_file.stem
        # Recherche du fichier OCR correspondant
        ocr_patterns = [
            f"enhanced_{image_name}_ocr.json",
            f"{image_name}_ocr.json",
            f"ocr_{image_name}.json"
        ]
        
        ocr_file = None
        for pattern in ocr_patterns:
            potential_ocr = Path(ocr_results_dir) / pattern
            if potential_ocr.exists():
                ocr_file = potential_ocr
                break
        
        if ocr_file:
            print(f"📄 Traitement de la paire:")
            print(f"  Image: {image_file.name}")
            print(f"  OCR: {ocr_file.name}")
            
            # Chargement des résultats OCR
            try:
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # Intégration
                integrated_results = integrator.integrate_ocr_vlm(str(image_file), ocr_data)
                
                # Affichage des résultats
                print(f"\n📊 Résultats d'intégration:")
                print(f"  Modèle VLM: {integrated_results.get('model_used', 'N/A')}")
                print(f"  Temps de traitement: {integrated_results.get('processing_time', 0):.2f}s")
                
                # Métadonnées d'intégration
                integration_meta = integrated_results.get('integration_metadata', {})
                print(f"  Textes OCR: {integration_meta.get('ocr_texts_count', 0)}")
                print(f"  Confiance OCR moyenne: {integration_meta.get('ocr_confidence_avg', 0):.2f}")
                print(f"  Score d'intégration: {integration_meta.get('integration_score', 0):.2f}")
                
                # Score de validation
                validation = integrated_results.get('validation_score', {})
                print(f"  Accord OCR-VLM: {validation.get('ocr_vlm_agreement', 0):.2f}")
                print(f"  Confiance globale: {validation.get('overall_confidence', 0):.2f}")
                print(f"  Fiabilité: {validation.get('reliability', 'unknown')}")
                
                # Zones enrichies
                zones = integrated_results.get('detected_zones', {})
                print(f"\n🎯 Zones enrichies avec OCR:")
                
                # En-tête
                header = zones.get('header', {})
                if header.get('ocr_texts'):
                    print(f"  En-tête: {len(header['ocr_texts'])} textes OCR associés")
                
                # Montants
                amount_zones = zones.get('amount_zones', [])
                for i, amount in enumerate(amount_zones):
                    validated = "✅" if amount.get('validated_by_ocr', False) else "❓"
                    matches = len(amount.get('ocr_matches', []))
                    print(f"  Montant {i+1}: {validated} {matches} correspondance(s) OCR")
                
                # Tableaux
                tables = zones.get('tables', [])
                for i, table in enumerate(tables):
                    ocr_struct = table.get('ocr_structure', {})
                    if ocr_struct.get('detected', False):
                        rows = ocr_struct.get('rows', 0)
                        cols = ocr_struct.get('columns', 0)
                        cells = len(ocr_struct.get('cells', []))
                        print(f"  Table {i+1}: {rows}x{cols} structure, {cells} cellules OCR")
                
                # Sauvegarde des résultats intégrés
                output_file = f"integrated_{image_name}.json"
                output_path = Path("../Data/vlm_results") / output_file
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(integrated_results, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"✅ Résultats sauvegardés: {output_path}")
                
                # Génération d'une visualisation
                visual_path = integrator.visualizer.visualize_analysis_results(
                    str(image_file), integrated_results
                )
                if visual_path:
                    print(f"✅ Visualisation: {visual_path}")
                
                print("\n" + "="*50)
                break  # Traiter seulement le premier exemple
                
            except Exception as e:
                print(f"❌ Erreur lors de l'intégration: {e}")
                import traceback
                traceback.print_exc()
    
    print("🎉 Exemple d'intégration terminé!")

if __name__ == "__main__":
    main()