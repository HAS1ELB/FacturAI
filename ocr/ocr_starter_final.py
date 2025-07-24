#!/usr/bin/env python3
"""
Module OCR de démarrage pour FacturAI - VERSION FINALE
Compatible avec les nouvelles versions de PaddleOCR
"""

import os
import cv2
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceOCRProcessor:
    """Processeur OCR pour factures avec support multi-moteurs - VERSION FINALE"""
    
    def __init__(self, output_dir: str = "Data/ocr_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Vérifier les moteurs OCR disponibles
        self.available_engines = self._check_available_engines()
        logger.info(f"Moteurs OCR disponibles: {list(self.available_engines.keys())}")
    
    def _check_available_engines(self) -> Dict[str, bool]:
        """Vérifie quels moteurs OCR sont disponibles"""
        engines = {}
        
        # Tesseract
        try:
            import pytesseract
            engines['tesseract'] = True
            logger.info("✅ Tesseract disponible")
        except ImportError:
            engines['tesseract'] = False
            logger.warning("❌ Tesseract non disponible - pip install pytesseract")
        
        # EasyOCR
        try:
            import easyocr
            engines['easyocr'] = True
            logger.info("✅ EasyOCR disponible")
        except ImportError:
            engines['easyocr'] = False
            logger.warning("❌ EasyOCR non disponible - pip install easyocr")
        
        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            engines['paddleocr'] = True
            logger.info("✅ PaddleOCR disponible")
        except ImportError:
            engines['paddleocr'] = False
            logger.warning("❌ PaddleOCR non disponible - pip install paddlepaddle paddleocr")
        
        return engines
    
    def extract_text_tesseract(self, image_path: str) -> Dict[str, Any]:
        """Extraction OCR avec Tesseract"""
        if not self.available_engines.get('tesseract', False):
            raise ValueError("Tesseract n'est pas disponible")
        
        import pytesseract
        from PIL import Image
        
        # Ouvrir l'image
        image = Image.open(image_path)
        
        # Configuration optimisée pour factures
        custom_config = r'--oem 3 --psm 6 -l fra+eng'
        
        # Extraction avec coordonnées
        data = pytesseract.image_to_data(
            image, 
            config=custom_config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Traitement des résultats
        results = {
            'engine': 'tesseract',
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'text_blocks': [],
            'full_text': '',
            'confidence_scores': []
        }
        
        # Filtrer les résultats valides
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 30:  # Seuil de confiance minimum
                block = {
                    'text': text,
                    'confidence': conf,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    },
                    'level': data['level'][i]
                }
                results['text_blocks'].append(block)
                results['confidence_scores'].append(conf)
        
        # Texte complet
        results['full_text'] = pytesseract.image_to_string(image, config=custom_config)
        results['average_confidence'] = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
        
        return results
    
    def extract_text_easyocr(self, image_path: str) -> Dict[str, Any]:
        """Extraction OCR avec EasyOCR"""
        if not self.available_engines.get('easyocr', False):
            raise ValueError("EasyOCR n'est pas disponible")
        
        import easyocr
        
        # Initialiser le lecteur (français et anglais)
        reader = easyocr.Reader(['fr', 'en'], gpu=False)  # Changez à True si GPU disponible
        
        # Lecture de l'image
        result = reader.readtext(image_path)
        
        # Traitement des résultats
        results = {
            'engine': 'easyocr',
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'text_blocks': [],
            'full_text': '',
            'confidence_scores': []
        }
        
        for (bbox, text, confidence) in result:
            if confidence > 0.3:  # Seuil de confiance minimum
                # Convertir bbox en format standardisé
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                block = {
                    'text': text,
                    'confidence': float(confidence * 100),  # Convertir en pourcentage
                    'bbox': {
                        'x': int(min(x_coords)),
                        'y': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords))
                    }
                }
                results['text_blocks'].append(block)
                results['confidence_scores'].append(confidence * 100)
        
        # Texte complet
        results['full_text'] = ' '.join([block['text'] for block in results['text_blocks']])
        results['average_confidence'] = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
        
        return results
    
    def extract_text_paddleocr(self, image_path: str) -> Dict[str, Any]:
        """Extraction OCR avec PaddleOCR - Compatible nouvelle version"""
        if not self.available_engines.get('paddleocr', False):
            raise ValueError("PaddleOCR n'est pas disponible")
        
        from paddleocr import PaddleOCR
        
        # Initialiser PaddleOCR avec la nouvelle API
        ocr = PaddleOCR(use_textline_orientation=True, lang='fr', show_log=False)
        
        # Lecture de l'image
        result = ocr.ocr(image_path)
        
        # Traitement des résultats
        results = {
            'engine': 'paddleocr',
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'text_blocks': [],
            'full_text': '',
            'confidence_scores': [],
            'debug_info': {
                'result_type': str(type(result)),
                'result_structure': 'nouvelle_version'
            }
        }
        
        logger.info(f"PaddleOCR result type: {type(result)}")
        
        # Gestion de la nouvelle structure PaddleOCR
        if result:
            # Nouvelle version : result peut être un dict avec des clés spécifiques
            if isinstance(result, dict):
                logger.info("Détection de la nouvelle API PaddleOCR (format dict)")
                
                # Extraire les textes et scores de la nouvelle structure
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                rec_polys = result.get('rec_polys', [])
                
                logger.info(f"Textes détectés: {len(rec_texts)}")
                
                for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                    if score > 0.3:  # Seuil de confiance minimum
                        try:
                            # Convertir poly en bbox
                            if len(poly) >= 4:
                                x_coords = [point[0] for point in poly]
                                y_coords = [point[1] for point in poly]
                                
                                block = {
                                    'text': str(text),
                                    'confidence': float(score * 100),
                                    'bbox': {
                                        'x': int(min(x_coords)),
                                        'y': int(min(y_coords)),
                                        'width': int(max(x_coords) - min(x_coords)),
                                        'height': int(max(y_coords) - min(y_coords))
                                    },
                                    'raw_poly': poly.tolist() if hasattr(poly, 'tolist') else poly
                                }
                                results['text_blocks'].append(block)
                                results['confidence_scores'].append(score * 100)
                                
                        except Exception as e:
                            logger.warning(f"Erreur traitement bloc {i}: {e}")
                            continue
            
            # Ancienne version : result est une liste
            elif isinstance(result, list) and result and result[0]:
                logger.info("Détection de l'ancienne API PaddleOCR (format liste)")
                
                for i, line in enumerate(result[0]):
                    try:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox_points = line[0]
                            text_info = line[1]
                            
                            # Gérer différents formats de text_info
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = float(text_info[1])
                            elif isinstance(text_info, str):
                                text = text_info
                                confidence = 0.5
                            else:
                                continue
                            
                            if confidence > 0.3:
                                x_coords = [float(point[0]) for point in bbox_points]
                                y_coords = [float(point[1]) for point in bbox_points]
                                
                                block = {
                                    'text': str(text),
                                    'confidence': float(confidence * 100),
                                    'bbox': {
                                        'x': int(min(x_coords)),
                                        'y': int(min(y_coords)),
                                        'width': int(max(x_coords) - min(x_coords)),
                                        'height': int(max(y_coords) - min(y_coords))
                                    }
                                }
                                results['text_blocks'].append(block)
                                results['confidence_scores'].append(confidence * 100)
                                
                    except Exception as e:
                        logger.warning(f"Erreur ligne {i}: {e}")
                        continue
            
            else:
                logger.warning(f"Structure PaddleOCR non reconnue: {type(result)}")
                if hasattr(result, 'keys') or hasattr(result, '__dict__'):
                    logger.info(f"Clés disponibles: {list(result.keys()) if hasattr(result, 'keys') else 'N/A'}")
        
        # Texte complet
        results['full_text'] = ' '.join([block['text'] for block in results['text_blocks']])
        results['average_confidence'] = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
        
        logger.info(f"PaddleOCR - Blocs détectés: {len(results['text_blocks'])}")
        
        return results
    
    def process_image(self, image_path: str, engine: str = 'auto') -> Dict[str, Any]:
        """Traite une image avec le moteur OCR spécifié"""
        
        if engine == 'auto':
            # Sélectionner automatiquement le meilleur moteur disponible
            if self.available_engines.get('easyocr', False):
                engine = 'easyocr'  # EasyOCR en premier car il fonctionne bien
            elif self.available_engines.get('tesseract', False):
                engine = 'tesseract'
            elif self.available_engines.get('paddleocr', False):
                engine = 'paddleocr'
            else:
                raise ValueError("Aucun moteur OCR disponible!")
        
        logger.info(f"Traitement de {image_path} avec {engine}")
        
        try:
            if engine == 'tesseract':
                return self.extract_text_tesseract(image_path)
            elif engine == 'easyocr':
                return self.extract_text_easyocr(image_path)
            elif engine == 'paddleocr':
                return self.extract_text_paddleocr(image_path)
            else:
                raise ValueError(f"Moteur OCR non supporté: {engine}")
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement OCR: {str(e)}")
            return {
                'engine': engine,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
    
    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """Test un moteur OCR sur une image spécifique avec diagnostic détaillé"""
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image non trouvée: {image_path}")
        
        print(f"🔍 Test OCR sur: {image_path}")
        
        # Tester tous les moteurs disponibles
        results = {}
        
        for engine_name, is_available in self.available_engines.items():
            if is_available:
                print(f"\n📊 Test avec {engine_name}...")
                try:
                    result = self.process_image(image_path, engine_name)
                    results[engine_name] = result
                    
                    if 'error' in result:
                        print(f"  ❌ Erreur: {result['error']}")
                    else:
                        blocks_count = len(result.get('text_blocks', []))
                        avg_conf = result.get('average_confidence', 0)
                        print(f"  ✅ Succès: {blocks_count} blocs, confiance: {avg_conf:.1f}%")
                        
                        # Afficher quelques exemples de texte
                        if blocks_count > 0:
                            print("  📝 Exemples de texte détecté:")
                            for i, block in enumerate(result['text_blocks'][:3]):
                                print(f"    {i+1}. '{block['text']}' (conf: {block['confidence']:.1f}%)")
                            if blocks_count > 3:
                                print(f"    ... et {blocks_count - 3} autres blocs")
                        
                except Exception as e:
                    print(f"  ❌ Erreur inattendue: {str(e)}")
                    results[engine_name] = {'error': str(e)}
            else:
                print(f"\n⏭️  {engine_name} non disponible")
        
        # Sauvegarder les résultats de test
        test_results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'engines_tested': results,
            'summary': {
                'total_engines': len(self.available_engines),
                'available_engines': sum(self.available_engines.values()),
                'successful_engines': len([r for r in results.values() if 'error' not in r])
            }
        }
        
        # Corriger le chemin pour Windows
        output_name = f"test_results_{Path(image_path).stem}.json"
        output_path = os.path.join(self.output_dir, output_name)
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"\n📁 Résultats de test sauvegardés: {output_path}")
        except Exception as e:
            print(f"⚠️ Impossible de sauvegarder les résultats: {e}")
        
        return test_results
    
    def process_directory(self, input_dir: str = "Data/processed_images", 
                         engine: str = 'auto') -> List[Dict[str, Any]]:
        """Traite toutes les images d'un dossier"""
        
        if not os.path.exists(input_dir):
            raise ValueError(f"Le dossier {input_dir} n'existe pas!")
        
        # Trouver toutes les images
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"Aucune image trouvée dans {input_dir}")
            return []
        
        logger.info(f"Traitement de {len(image_files)} images avec {engine}")
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"[{i}/{len(image_files)}] Traitement: {image_file.name}")
            
            result = self.process_image(str(image_file), engine)
            results.append(result)
            
            # Sauvegarder le résultat individuel avec nom de fichier sécurisé
            safe_name = "".join(c for c in image_file.stem if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_name = f"{safe_name}_ocr.json"
            output_path = os.path.join(self.output_dir, output_name)
            
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"Résultat sauvegardé: {output_path}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder {output_path}: {e}")
        
        # Sauvegarder le rapport global
        global_report = {
            'processed_images': len(results),
            'engine_used': engine,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'statistics': self._calculate_statistics(results)
        }
        
        report_path = os.path.join(self.output_dir, "ocr_global_report.json")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(global_report, f, ensure_ascii=False, indent=2)
            logger.info(f"Rapport global sauvegardé: {report_path}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder le rapport global: {e}")
        
        return results
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule des statistiques sur les résultats OCR"""
        
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            return {
                'success_rate': 0,
                'failed_count': len(failed_results),
                'average_confidence': 0,
                'total_text_blocks': 0
            }
        
        confidences = [r.get('average_confidence', 0) for r in successful_results]
        text_blocks_counts = [len(r.get('text_blocks', [])) for r in successful_results]
        
        return {
            'success_rate': len(successful_results) / len(results) * 100,
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'total_text_blocks': sum(text_blocks_counts),
            'average_blocks_per_image': np.mean(text_blocks_counts) if text_blocks_counts else 0
        }

def main():
    """Fonction principale pour tester l'OCR"""
    
    print("🚀 DÉMARRAGE DU MODULE OCR FACTURAI - VERSION FINALE")
    print("=" * 60)
    
    # Initialiser le processeur
    processor = InvoiceOCRProcessor()
    
    # Vérifier les dossiers
    input_dir = "Data/processed_images"
    test_image = "Data/processed_images/enhanced_complex_invoice_0001.png"
    
    if not os.path.exists(input_dir):
        print(f"❌ Le dossier {input_dir} n'existe pas!")
        print("📁 Créer le dossier et y placer vos images prétraitées")
        os.makedirs(input_dir, exist_ok=True)
        print(f"✅ Dossier {input_dir} créé")
        return
    
    # Recommandation moteur
    print(f"\n💡 RECOMMANDATION:")
    if processor.available_engines.get('easyocr', False):
        print("   EasyOCR est disponible et recommandé (meilleurs résultats)")
    elif processor.available_engines.get('tesseract', False):
        print("   Tesseract est disponible et fiable")
    else:
        print("   Installez EasyOCR pour de meilleurs résultats: pip install easyocr")
    
    # Mode de test spécifique si l'image problématique existe
    if os.path.exists(test_image):
        print(f"\n🎯 TEST SPÉCIFIQUE sur l'image problématique")
        print(f"Image: {test_image}")
        try:
            test_results = processor.test_single_image(test_image)
            print("\n✅ Test spécifique terminé")
        except Exception as e:
            print(f"❌ Erreur pendant le test: {str(e)}")
    
    # Traitement avec le meilleur moteur
    best_engine = 'easyocr' if processor.available_engines.get('easyocr', False) else 'auto'
    
    try:
        print(f"\n📂 TRAITEMENT GLOBAL du dossier {input_dir} avec {best_engine}")
        results = processor.process_directory(input_dir, best_engine)
        
        if results:
            # Afficher les résultats
            stats = processor._calculate_statistics(results)
            
            print("\n📊 RÉSULTATS:")
            print(f"Images traitées: {len(results)}")
            print(f"Taux de succès: {stats['success_rate']:.1f}%")
            print(f"Confiance moyenne: {stats['average_confidence']:.1f}%")
            print(f"Blocs de texte détectés: {stats['total_text_blocks']}")
            print(f"📁 Résultats dans: Data/ocr_results/")
            
            # Recommandations
            if stats['success_rate'] < 90:
                print(f"\n💡 CONSEILS D'AMÉLIORATION:")
                print("   - Vérifiez la qualité des images prétraitées")
                print("   - Essayez différents moteurs OCR")
                print("   - Ajustez les seuils de confiance")
                
        else:
            print("❌ Aucune image à traiter")
            
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
    
    print("=" * 60)
    print("✅ TRAITEMENT TERMINÉ")

if __name__ == "__main__":
    main()