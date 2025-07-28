#!/usr/bin/env python3
"""
Préparation des données pour le fine-tuning OCR des factures
Génère automatiquement les annotations et prépare les datasets
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# OCR imports
import easyocr
import pytesseract
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataPreparator:
    """Préparation des données de factures pour le fine-tuning"""
    
    def __init__(self, images_dir: str, ocr_results_dir: str, output_dir: str):
        self.images_dir = Path(images_dir)
        self.ocr_results_dir = Path(ocr_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer les dossiers de sortie
        self.annotations_dir = self.output_dir / "annotations"
        self.datasets_dir = self.output_dir / "datasets"
        self.splits_dir = self.output_dir / "splits"
        
        for dir_path in [self.annotations_dir, self.datasets_dir, self.splits_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_ground_truth_from_ocr(self) -> Dict[str, Any]:
        """Génère la vérité terrain à partir des résultats OCR existants"""
        logger.info("🎯 Génération de la vérité terrain à partir des résultats OCR...")
        
        ground_truth = {}
        processed_files = 0
        
        # Parcourir tous les fichiers OCR
        for ocr_file in self.ocr_results_dir.glob("*.json"):
            try:
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # Extraire le nom de l'image
                image_name = ocr_file.stem.replace('_ocr', '').replace('enhanced_', '')

                # Essai 1: Recherche directe avec le nom extrait
                image_files = list(self.images_dir.glob(f"*{image_name}*"))

                # Essai 2: Si aucune image trouvée, essayez avec le chemin indiqué dans le fichier JSON
                if not image_files and 'image_path' in ocr_data:
                    # Extraire le nom du fichier du chemin stocké dans image_path
                    json_image_path = Path(ocr_data['image_path'])
                    image_name_from_json = json_image_path.name
                    image_files = list(self.images_dir.glob(f"*{image_name_from_json}*"))

                if not image_files:
                    logger.warning(f"Image non trouvée pour {image_name}")
                    continue

                
                image_path = image_files[0]
                
                # Traiter selon le format OCR
                annotations = self._process_ocr_results(ocr_data, str(image_path))
                
                if annotations:
                    ground_truth[str(image_path)] = annotations
                    processed_files += 1
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {ocr_file}: {e}")
        
        logger.info(f"✅ {processed_files} fichiers traités avec succès")
        return ground_truth
    
    def _process_ocr_results(self, ocr_data: Dict, image_path: str) -> List[Dict]:
        """Traite les résultats OCR pour créer des annotations"""
        annotations = []
        
        try:
            # Format avec texts, bboxes, confidences (ancien format)
            if 'texts' in ocr_data:
                for i, (text, bbox, confidence) in enumerate(zip(
                    ocr_data.get('texts', []),
                    ocr_data.get('bboxes', []),
                    ocr_data.get('confidences', [])
                )):
                    if confidence > 0.5 and len(text.strip()) > 1:
                        annotations.append({
                            'text': text.strip(),
                            'bbox': bbox,
                            'confidence': confidence,
                            'type': self._classify_text_type(text)
                        })
            
            # Format avec text_blocks (nouveau format)
            elif 'text_blocks' in ocr_data:
                for block in ocr_data.get('text_blocks', []):
                    if 'text' in block and 'confidence' in block and 'bbox' in block:
                        confidence = block['confidence']
                        text = block['text']
                        bbox = block['bbox']
                        
                        # Convertir le bbox du format {x,y,width,height} en [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        if isinstance(bbox, dict) and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                            x, y = bbox['x'], bbox['y']
                            w, h = bbox['width'], bbox['height']
                            bbox_points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        else:
                            bbox_points = bbox
                        
                        if confidence > 50 and len(text.strip()) > 1:  # Note: confidence peut être sur 100
                            annotations.append({
                                'text': text.strip(),
                                'bbox': bbox_points,
                                'confidence': confidence / 100 if confidence > 1 else confidence,  # Normaliser à [0,1]
                                'type': self._classify_text_type(text)
                            })
            
            # Si c'est un autre format, essayer de l'adapter
            elif isinstance(ocr_data, list):
                for item in ocr_data:
                    if isinstance(item, dict) and 'text' in item:
                        annotations.append(item)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement OCR: {e}")
        
        return annotations

    
    def _classify_text_type(self, text: str) -> str:
        """Classifie le type de texte détecté"""
        text_lower = text.lower().strip()
        
        # Patterns pour identifier le type de contenu
        if any(word in text_lower for word in ['facture', 'invoice', 'bill']):
            return 'header'
        elif any(word in text_lower for word in ['€', '$', 'eur', 'usd', 'ttc', 'ht','mad','dh', 'dhs']):
            return 'amount'
        elif any(word in text_lower for word in ['date', '/', '-']) and len(text) < 15:
            return 'date'
        elif '@' in text_lower or 'email' in text_lower:
            return 'email'
        elif any(word in text_lower for word in ['tel', 'phone', '+33', '01', '02', '03', '04', '05', '06', '07', '+212']):
            return 'phone'
        elif any(word in text_lower for word in ['rue', 'avenue', 'place', 'boulevard']):
            return 'address'
        elif text.replace('.', '').replace('-', '').isdigit():
            return 'number'
        else:
            return 'text'
    
    def enhance_annotations_with_context(self, ground_truth: Dict) -> Dict:
        """Améliore les annotations avec du contexte métier"""
        logger.info("🔧 Amélioration des annotations avec le contexte métier...")
        
        enhanced_gt = {}
        
        for image_path, annotations in ground_truth.items():
            enhanced_annotations = []
            
            # Regrouper les textes par zones
            zones = self._group_by_zones(annotations)
            
            for zone_type, zone_texts in zones.items():
                for annotation in zone_texts:
                    # Ajouter le contexte de zone
                    annotation['zone_type'] = zone_type
                    
                    # Améliorer la classification
                    annotation['enhanced_type'] = self._enhance_text_classification(
                        annotation['text'], zone_type, zone_texts
                    )
                    
                    enhanced_annotations.append(annotation)
            
            enhanced_gt[image_path] = enhanced_annotations
        
        return enhanced_gt
    
    def _group_by_zones(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """Regroupe les annotations par zones géographiques"""
        zones = {
            'header': [],
            'company_info': [],
            'client_info': [],
            'items': [],
            'total': [],
            'footer': []
        }
        
        if not annotations:
            return zones
        
        # Trier par position verticale
        sorted_annotations = sorted(annotations, key=lambda x: x.get('bbox', [[0,0]])[0][1])
        
        # Diviser en zones basées sur la position Y
        total_height = max([max([point[1] for point in ann.get('bbox', [[0,0]])]) 
                           for ann in annotations])
        
        for ann in sorted_annotations:
            if 'bbox' in ann and ann['bbox']:
                y_center = sum([point[1] for point in ann['bbox']]) / len(ann['bbox'])
                relative_pos = y_center / total_height if total_height > 0 else 0
                
                if relative_pos < 0.2:
                    zones['header'].append(ann)
                elif relative_pos < 0.4:
                    zones['company_info'].append(ann)
                elif relative_pos < 0.6:
                    zones['client_info'].append(ann)
                elif relative_pos < 0.8:
                    zones['items'].append(ann)
                elif relative_pos < 0.95:
                    zones['total'].append(ann)
                else:
                    zones['footer'].append(ann)
        
        return zones
    
    def _enhance_text_classification(self, text: str, zone_type: str, zone_texts: List[Dict]) -> str:
        """Classification améliorée basée sur le contexte"""
        text_lower = text.lower().strip()
        
        # Classification contextuelle par zone
        if zone_type == 'header':
            if any(word in text_lower for word in ['facture', 'invoice', 'devis', 'quote']):
                return 'document_type'
            elif any(char.isdigit() for char in text) and len(text) < 20:
                return 'document_number'
        
        elif zone_type == 'total':
            if any(word in text_lower for word in ['total', 'ttc', 'ht']):
                return 'total_label'
            elif '€' in text or any(char.isdigit() for char in text):
                return 'total_amount'
        
        # Classification générale améliorée
        return self._classify_text_type(text)
    
    def create_datasets(self, ground_truth: Dict, train_split: float = 0.8, 
                       val_split: float = 0.1, test_split: float = 0.1) -> Dict[str, List]:
        """Crée les datasets d'entraînement, validation et test"""
        logger.info("📊 Création des datasets...")
        
        # Préparer les données
        images = list(ground_truth.keys())
        
        # Split train/temp
        train_images, temp_images = train_test_split(
            images, test_size=(1-train_split), random_state=42
        )
        
        # Split val/test
        val_size = val_split / (val_split + test_split)
        val_images, test_images = train_test_split(
            temp_images, test_size=(1-val_size), random_state=42
        )
        
        datasets = {
            'train': train_images,
            'validation': val_images,
            'test': test_images
        }
        
        # Sauvegarder les splits
        for split_name, split_images in datasets.items():
            split_data = []
            for image_path in split_images:
                split_data.append({
                    'image_path': image_path,
                    'annotations': ground_truth[image_path]
                })
            
            split_file = self.splits_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Datasets créés: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        return datasets
    
    def create_trocr_dataset(self, ground_truth: Dict) -> None:
        """Crée le dataset au format TrOCR"""
        logger.info("🤖 Création du dataset TrOCR...")
        
        trocr_dir = self.datasets_dir / "trocr"
        trocr_dir.mkdir(exist_ok=True)
        
        trocr_data = []
        
        for image_path, annotations in ground_truth.items():
            # Concatener tous les textes de l'image
            full_text = " ".join([ann['text'] for ann in annotations if ann.get('text')])
            
            if full_text.strip():
                trocr_data.append({
                    'image_path': image_path,
                    'text': full_text.strip()
                })
        
        # Sauvegarder
        with open(trocr_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(trocr_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset TrOCR créé avec {len(trocr_data)} échantillons")
    
    def create_easyocr_dataset(self, ground_truth: Dict) -> None:
        """Crée le dataset au format EasyOCR"""
        logger.info("👁️ Création du dataset EasyOCR...")
        
        easyocr_dir = self.datasets_dir / "easyocr"
        easyocr_dir.mkdir(exist_ok=True)
        
        # Format spécifique pour EasyOCR fine-tuning
        easyocr_data = []
        
        for image_path, annotations in ground_truth.items():
            for ann in annotations:
                if ann.get('text') and ann.get('bbox'):
                    easyocr_data.append({
                        'image_path': image_path,
                        'text': ann['text'],
                        'bbox': ann['bbox'],
                        'confidence': ann.get('confidence', 1.0)
                    })
        
        # Sauvegarder
        with open(easyocr_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(easyocr_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset EasyOCR créé avec {len(easyocr_data)} échantillons")
    
    def create_paddleocr_dataset(self, ground_truth: Dict) -> None:
        """Crée le dataset au format PaddleOCR"""
        logger.info("🏓 Création du dataset PaddleOCR...")
        
        paddle_dir = self.datasets_dir / "paddleocr"
        paddle_dir.mkdir(exist_ok=True)
        
        paddle_data = []
        
        for image_path, annotations in ground_truth.items():
            # Format PaddleOCR : image + lignes de texte
            lines = []
            for ann in annotations:
                if ann.get('text') and ann.get('bbox'):
                    lines.append({
                        'transcription': ann['text'],
                        'points': ann['bbox']
                    })
            
            if lines:
                paddle_data.append({
                    'image_path': image_path,
                    'lines': lines
                })
        
        # Sauvegarder
        with open(paddle_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(paddle_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset PaddleOCR créé avec {len(paddle_data)} échantillons")
    
    def generate_statistics(self, ground_truth: Dict) -> Dict[str, Any]:
        """Génère des statistiques sur le dataset"""
        logger.info("📈 Génération des statistiques...")
        
        stats = {
            'total_images': len(ground_truth),
            'total_annotations': sum(len(anns) for anns in ground_truth.values()),
            'text_types': {},
            'avg_confidence': 0,
            'avg_annotations_per_image': 0
        }
        
        all_confidences = []
        type_counts = {}
        
        for annotations in ground_truth.values():
            for ann in annotations:
                # Compter les types
                text_type = ann.get('type', 'unknown')
                type_counts[text_type] = type_counts.get(text_type, 0) + 1
                
                # Collecter les confidences
                if 'confidence' in ann:
                    all_confidences.append(ann['confidence'])
        
        stats['text_types'] = type_counts
        stats['avg_confidence'] = np.mean(all_confidences) if all_confidences else 0
        stats['avg_annotations_per_image'] = stats['total_annotations'] / stats['total_images']
        
        # Sauvegarder les statistiques
        with open(self.output_dir / "dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Statistiques générées et sauvegardées")
        return stats
    
    def run_complete_preparation(self) -> Dict[str, Any]:
        """Lance la préparation complète des données"""
        logger.info("🚀 DÉMARRAGE DE LA PRÉPARATION COMPLÈTE DES DONNÉES")
        logger.info("=" * 60)
        
        # 1. Générer la vérité terrain
        ground_truth = self.generate_ground_truth_from_ocr()
        
        if not ground_truth:
            logger.error("❌ Aucune donnée trouvée")
            return {}
        
        # 2. Améliorer avec le contexte
        enhanced_gt = self.enhance_annotations_with_context(ground_truth)
        
        # 3. Sauvegarder les annotations
        annotations_file = self.annotations_dir / "ground_truth.json"
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_gt, f, ensure_ascii=False, indent=2)
        
        # 4. Créer les datasets
        datasets = self.create_datasets(enhanced_gt)
        
        # 5. Créer les formats spécifiques
        self.create_trocr_dataset(enhanced_gt)
        self.create_easyocr_dataset(enhanced_gt)
        self.create_paddleocr_dataset(enhanced_gt)
        
        # 6. Générer les statistiques
        stats = self.generate_statistics(enhanced_gt)
        
        logger.info("=" * 60)
        logger.info("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS")
        logger.info(f"📊 {stats['total_images']} images, {stats['total_annotations']} annotations")
        logger.info(f"🎯 Confiance moyenne: {stats['avg_confidence']:.2f}")
        
        return {
            'ground_truth': enhanced_gt,
            'datasets': datasets,
            'statistics': stats,
            'output_dir': str(self.output_dir)
        }

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Préparation des données pour le fine-tuning OCR")
    parser.add_argument('--images_dir', default='Data/processed_images', 
                       help='Dossier des images')
    parser.add_argument('--ocr_results_dir', default='Data/ocr_results', 
                       help='Dossier des résultats OCR')
    parser.add_argument('--output_dir', default='Data/fine_tuning', 
                       help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Créer le préparateur
    preparator = InvoiceDataPreparator(
        images_dir=args.images_dir,
        ocr_results_dir=args.ocr_results_dir,
        output_dir=args.output_dir
    )
    
    # Lancer la préparation
    results = preparator.run_complete_preparation()
    
    if results:
        print("\n🎉 Préparation terminée avec succès!")
        print(f"📁 Résultats dans: {results['output_dir']}")
    else:
        print("❌ Échec de la préparation")

if __name__ == "__main__":
    main()
    
# python fine-tuning-ocr/data_preparation/data_preparation.py --images_dir Data/processed_images --ocr_results_dir Data/ocr_results --output_dir Data/fine_tuning