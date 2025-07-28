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

# OCR imports - commented out as not needed for data preparation
# import easyocr
# import pytesseract
# from paddleocr import PaddleOCR

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
        
        # Sauvegarder les splits (format original pour compatibilité)
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
    
    def create_trocr_dataset(self, ground_truth: Dict, datasets: Dict[str, List]) -> None:
        """Crée le dataset au format TrOCR avec splits appropriés"""
        logger.info("🤖 Création du dataset TrOCR...")
        
        trocr_dir = self.datasets_dir / "trocr"
        trocr_dir.mkdir(exist_ok=True)
        
        # Créer les splits pour TrOCR
        trocr_splits = {}
        
        for split_name, split_images in datasets.items():
            split_data = []
            
            for image_path in split_images:
                annotations = ground_truth[image_path]
                
                # Stratégie améliorée pour combiner les textes
                text_parts = []
                
                # Grouper par zones pour une meilleure structure
                zones = self._group_by_zones(annotations)
                zone_order = ['header', 'company_info', 'client_info', 'items', 'total', 'footer']
                
                for zone in zone_order:
                    if zone in zones and zones[zone]:
                        zone_texts = [ann['text'] for ann in zones[zone] if ann.get('text', '').strip()]
                        if zone_texts:
                            text_parts.extend(zone_texts)
                
                # Si pas de zones détectées, utiliser tous les textes
                if not text_parts:
                    text_parts = [ann['text'] for ann in annotations if ann.get('text', '').strip()]
                
                full_text = " ".join(text_parts).strip()
                
                if full_text:
                    # Convertir en chemin absolu
                    abs_image_path = str(Path(image_path).resolve())
                    split_data.append({
                        'image_path': abs_image_path,
                        'text': full_text
                    })
            
            trocr_splits[split_name] = split_data
            
            # Sauvegarder chaque split séparément pour TrOCR
            split_file = trocr_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Split {split_name}: {len(split_data)} échantillons")
        
        # Créer aussi un dataset complet
        all_data = []
        for split_data in trocr_splits.values():
            all_data.extend(split_data)
        
        with open(trocr_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Créer un fichier de métadonnées
        metadata = {
            'total_samples': len(all_data),
            'splits': {name: len(data) for name, data in trocr_splits.items()},
            'created_at': datetime.now().isoformat(),
            'format': 'trocr',
            'description': 'Dataset préparé pour le fine-tuning TrOCR sur des factures'
        }
        
        with open(trocr_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset TrOCR créé avec {len(all_data)} échantillons au total")
    
    def create_easyocr_dataset(self, ground_truth: Dict, datasets: Dict[str, List]) -> None:
        """Crée le dataset au format EasyOCR avec splits et formatage correct"""
        logger.info("👁️ Création du dataset EasyOCR...")
        
        easyocr_dir = self.datasets_dir / "easyocr"
        easyocr_dir.mkdir(exist_ok=True)
        
        # Créer les splits pour EasyOCR
        easyocr_splits = {}
        all_data = []
        
        for split_name, split_images in datasets.items():
            split_data = []
            
            for image_path in split_images:
                annotations = ground_truth[image_path]
                abs_image_path = str(Path(image_path).resolve())
                
                # Format EasyOCR: chaque annotation de texte séparée
                for ann in annotations:
                    if ann.get('text', '').strip() and ann.get('bbox'):
                        # Normaliser le bbox si nécessaire
                        bbox = ann['bbox']
                        if isinstance(bbox, dict):
                            # Convertir du format {x,y,width,height} vers [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                            x, y = bbox['x'], bbox['y']
                            w, h = bbox['width'], bbox['height']
                            bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        
                        text_sample = {
                            'image_path': abs_image_path,
                            'text': ann['text'].strip(),
                            'bbox': bbox,
                            'confidence': ann.get('confidence', 1.0),
                            'type': ann.get('type', 'text'),
                            'enhanced_type': ann.get('enhanced_type', ann.get('type', 'text'))
                        }
                        
                        split_data.append(text_sample)
                        all_data.append(text_sample)
            
            easyocr_splits[split_name] = split_data
            
            # Sauvegarder chaque split
            split_file = easyocr_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"EasyOCR split {split_name}: {len(split_data)} échantillons de texte")
        
        # Sauvegarder le dataset complet
        with open(easyocr_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Créer les statistiques par type de texte
        type_stats = {}
        for sample in all_data:
            text_type = sample.get('enhanced_type', 'unknown')
            type_stats[text_type] = type_stats.get(text_type, 0) + 1
        
        # Métadonnées
        metadata = {
            'total_samples': len(all_data),
            'splits': {name: len(data) for name, data in easyocr_splits.items()},
            'text_type_distribution': type_stats,
            'created_at': datetime.now().isoformat(),
            'format': 'easyocr',
            'description': 'Dataset préparé pour le fine-tuning EasyOCR - annotations de texte individuelles'
        }
        
        with open(easyocr_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset EasyOCR créé avec {len(all_data)} échantillons de texte")
    
    def create_paddleocr_dataset(self, ground_truth: Dict, datasets: Dict[str, List]) -> None:
        """Crée le dataset au format PaddleOCR avec format correct pour l'entraînement"""
        logger.info("🏓 Création du dataset PaddleOCR...")
        
        paddle_dir = self.datasets_dir / "paddleocr"
        paddle_dir.mkdir(exist_ok=True)
        
        # Créer les splits pour PaddleOCR
        paddle_splits = {}
        all_data = []
        
        for split_name, split_images in datasets.items():
            split_data = []
            
            for image_path in split_images:
                annotations = ground_truth[image_path]
                abs_image_path = str(Path(image_path).resolve())
                
                # Format PaddleOCR : image + lignes de texte avec métadonnées
                lines = []
                for ann in annotations:
                    if ann.get('text', '').strip() and ann.get('bbox'):
                        # S'assurer que les points sont dans le bon format
                        bbox = ann['bbox']
                        if isinstance(bbox, dict):
                            # Convertir du format {x,y,width,height} vers [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                            x, y = bbox['x'], bbox['y']
                            w, h = bbox['width'], bbox['height']
                            points = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        else:
                            points = bbox
                        
                        line_data = {
                            'transcription': ann['text'].strip(),
                            'points': points,
                            'difficult': False,  # Standard PaddleOCR field
                            'confidence': ann.get('confidence', 1.0),
                            'type': ann.get('enhanced_type', ann.get('type', 'text'))
                        }
                        lines.append(line_data)
                
                if lines:
                    image_sample = {
                        'image_path': abs_image_path,
                        'lines': lines,
                        'image_name': Path(image_path).name,
                        'num_lines': len(lines)
                    }
                    split_data.append(image_sample)
                    all_data.append(image_sample)
            
            paddle_splits[split_name] = split_data
            
            # Sauvegarder chaque split
            split_file = paddle_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"PaddleOCR split {split_name}: {len(split_data)} images avec {sum(len(img['lines']) for img in split_data)} lignes de texte")
        
        # Sauvegarder le dataset complet
        with open(paddle_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Créer les fichiers de liste au format PaddleOCR pour l'entraînement
        self._create_paddle_training_lists(paddle_dir, paddle_splits)
        
        # Statistiques
        total_lines = sum(len(img['lines']) for img in all_data)
        total_images = len(all_data)
        
        # Métadonnées
        metadata = {
            'total_images': total_images,
            'total_text_lines': total_lines,
            'avg_lines_per_image': total_lines / total_images if total_images > 0 else 0,
            'splits': {name: {'images': len(data), 'lines': sum(len(img['lines']) for img in data)} 
                      for name, data in paddle_splits.items()},
            'created_at': datetime.now().isoformat(),
            'format': 'paddleocr',
            'description': 'Dataset préparé pour le fine-tuning PaddleOCR - détection et reconnaissance'
        }
        
        with open(paddle_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset PaddleOCR créé: {total_images} images, {total_lines} lignes de texte")
    
    def _create_paddle_training_lists(self, paddle_dir: Path, paddle_splits: Dict) -> None:
        """Crée les fichiers de liste au format attendu par PaddleOCR pour l'entraînement"""
        
        for split_name, split_data in paddle_splits.items():
            # Format pour la détection: image_path\tannotation_json
            det_list = []
            # Format pour la reconnaissance: image_path\ttexte
            rec_list = []
            
            for image_sample in split_data:
                image_path = image_sample['image_path']
                image_name = Path(image_path).name
                
                # Format détection
                det_annotation = {
                    'transcription': [line['transcription'] for line in image_sample['lines']],
                    'points': [line['points'] for line in image_sample['lines']]
                }
                det_list.append(f"{image_name}\t{json.dumps(det_annotation, ensure_ascii=False)}")
                
                # Format reconnaissance (une ligne par annotation de texte)
                for line in image_sample['lines']:
                    rec_list.append(f"{image_name}\t{line['transcription']}")
            
            # Sauvegarder les listes
            with open(paddle_dir / f"det_{split_name}_list.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(det_list))
            
            with open(paddle_dir / f"rec_{split_name}_list.txt", 'w', encoding='utf-8') as f:
                f.write('\n'.join(rec_list))
    
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
        
        # 5. Créer les formats spécifiques avec splits
        self.create_trocr_dataset(enhanced_gt, datasets)
        self.create_easyocr_dataset(enhanced_gt, datasets)
        self.create_paddleocr_dataset(enhanced_gt, datasets)
        
        # 6. Créer les fichiers de configuration pour les fine-tuning
        self._create_training_configs(enhanced_gt, datasets)
        
        # 7. Générer les statistiques
        stats = self.generate_statistics(enhanced_gt)
        
        logger.info("=" * 60)
        logger.info("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS")
        logger.info(f"📊 {stats['total_images']} images, {stats['total_annotations']} annotations")
        logger.info(f"🎯 Confiance moyenne: {stats['avg_confidence']:.2f}")
        logger.info(f"📁 Datasets créés pour 3 modèles OCR")
        logger.info(f"ℹ️ Voir les métadonnées dans chaque dossier de dataset")
        
        return {
            'ground_truth': enhanced_gt,
            'datasets': datasets,
            'statistics': stats,
            'output_dir': str(self.output_dir),
            'trocr_path': str(self.datasets_dir / "trocr"),
            'easyocr_path': str(self.datasets_dir / "easyocr"),
            'paddleocr_path': str(self.datasets_dir / "paddleocr")
        }

    def _create_training_configs(self, ground_truth: Dict, datasets: Dict) -> None:
        """Crée les fichiers de configuration pour les scripts de fine-tuning"""
        logger.info("⚙️ Création des configurations d'entraînement...")
        
        configs_dir = self.output_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        # Configuration pour TrOCR
        trocr_config = {
            "model_type": "trocr",
            "base_model": "microsoft/trocr-large-printed",
            "dataset_path": str(self.datasets_dir / "trocr" / "dataset.json"),
            "train_split_path": str(self.datasets_dir / "trocr" / "train.json"),
            "val_split_path": str(self.datasets_dir / "trocr" / "validation.json"),
            "test_split_path": str(self.datasets_dir / "trocr" / "test.json"),
            "output_dir": "models/trocr_finetuned",
            "epochs": 30,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "max_length": 512
        }
        
        with open(configs_dir / "trocr_config.json", 'w', encoding='utf-8') as f:
            json.dump(trocr_config, f, ensure_ascii=False, indent=2)
        
        # Configuration pour EasyOCR
        easyocr_config = {
            "model_type": "easyocr",
            "dataset_path": str(self.datasets_dir / "easyocr" / "dataset.json"),
            "train_split_path": str(self.datasets_dir / "easyocr" / "train.json"),
            "val_split_path": str(self.datasets_dir / "easyocr" / "validation.json"),
            "test_split_path": str(self.datasets_dir / "easyocr" / "test.json"),
            "output_dir": "models/easyocr_finetuned",
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 0.001,
            "hidden_size": 256,
            "patience": 10
        }
        
        with open(configs_dir / "easyocr_config.json", 'w', encoding='utf-8') as f:
            json.dump(easyocr_config, f, ensure_ascii=False, indent=2)
        
        # Configuration pour PaddleOCR
        paddleocr_config = {
            "model_type": "paddleocr",
            "dataset_path": str(self.datasets_dir / "paddleocr" / "dataset.json"),
            "dataset_dir": str(self.datasets_dir / "paddleocr"),
            "output_dir": "models/paddleocr_finetuned"
        }
        
        with open(configs_dir / "paddleocr_config.json", 'w', encoding='utf-8') as f:
            json.dump(paddleocr_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Configurations créées dans {configs_dir}")

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
        print(f"🤖 TrOCR dataset: {results.get('trocr_path', 'Non créé')}")
        print(f"👁️ EasyOCR dataset: {results.get('easyocr_path', 'Non créé')}")
        print(f"🏓 PaddleOCR dataset: {results.get('paddleocr_path', 'Non créé')}")
        print("\n🚀 Pour lancer les fine-tuning:")
        print("   - TrOCR: python fine-tuning-ocr/fine_tuning_model/trocr_finetuning.py --dataset Data/fine_tuning/datasets/trocr/dataset.json")
        print("   - EasyOCR: python fine-tuning-ocr/fine_tuning_model/easyocr_finetuning.py --dataset Data/fine_tuning/datasets/easyocr/dataset.json")
        print("   - PaddleOCR: python fine-tuning-ocr/fine_tuning_model/paddleocr_finetuning.py --dataset Data/fine_tuning/datasets/paddleocr/dataset.json")
    else:
        print("❌ Échec de la préparation")

if __name__ == "__main__":
    main()
    
# python fine-tuning-ocr/data_preparation/data_preparation.py --images_dir Data/processed_images --ocr_results_dir Data/ocr_results --output_dir Data/fine_tuning