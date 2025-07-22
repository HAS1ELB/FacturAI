#!/usr/bin/env python3
"""
Module d'amélioration d'images pour les factures - VERSION OPTIMISÉE
Paramètres ajustés pour préserver la qualité tout en améliorant la lisibilité
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import logging
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import glob
import json
from datetime import datetime
import concurrent.futures
from typing import List
import numpy as np


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceImageEnhancer:
    """Classe principale pour l'amélioration des images de factures - Version optimisée"""
    
    def __init__(self):
        # Paramètres plus conservateurs pour préserver la qualité
        self.enhancement_params = {
            'target_dpi': 300,
            'contrast_factor': 1.1,      # Réduit de 1.2 à 1.1
            'brightness_factor': 1.05,   # Réduit de 1.1 à 1.05
            'sharpness_factor': 1.15,    # Réduit de 1.3 à 1.15
            'noise_reduction': True,
            'auto_rotate': True,
            'normalize_colors': False,   # Désactivé par défaut
            'ocr_optimization': False    # Désactivé par défaut pour préserver la qualité
        }
        
    def enhance_invoice_image(self, image_path: str, output_path: Optional[str] = None, 
                            preserve_quality: bool = True) -> np.ndarray:
        """
        Pipeline d'amélioration d'une image de facture avec préservation de qualité
        
        Args:
            image_path: Chemin vers l'image source
            output_path: Chemin de sauvegarde (optionnel)
            preserve_quality: Si True, utilise des paramètres conservateurs
            
        Returns:
            np.ndarray: Image améliorée
        """
        logger.info(f"Début de l'amélioration de l'image: {image_path}")
        
        # Chargement de l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Analyser la qualité de l'image d'origine
        original_quality = self.analyze_image_quality(image)
        logger.info(f"Qualité originale: {original_quality['quality_score']:.1f}/100")
        
        # Adapter les paramètres selon la qualité originale
        if preserve_quality and original_quality['quality_score'] > 70:
            logger.info("Image de bonne qualité détectée - traitement conservateur")
            self._set_conservative_params()
        
        # Pipeline d'amélioration adaptatif
        enhanced_image = self._adaptive_preprocessing_pipeline(image, original_quality)
        
        # Vérifier si l'amélioration est bénéfique
        enhanced_quality = self.analyze_image_quality(enhanced_image)
        
        if enhanced_quality['quality_score'] < original_quality['quality_score'] - 5:
            logger.warning("L'amélioration dégrade la qualité - retour à l'original avec légères modifications")
            enhanced_image = self._minimal_enhancement(image)
        
        # Sauvegarde si chemin spécifié
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, enhanced_image)
            logger.info(f"Image améliorée sauvegardée: {output_path}")
        
        return enhanced_image
    
    def _set_conservative_params(self):
        """Paramètres très conservateurs pour images de bonne qualité"""
        self.enhancement_params.update({
            'contrast_factor': 1.05,
            'brightness_factor': 1.02,
            'sharpness_factor': 1.1,
            'normalize_colors': False,
            'ocr_optimization': False
        })
    
    def _adaptive_preprocessing_pipeline(self, image: np.ndarray, quality_metrics: Dict) -> np.ndarray:
        """Pipeline adaptatif basé sur la qualité de l'image"""
        
        # 1. Correction d'orientation seulement si nécessaire
        if self.enhancement_params['auto_rotate'] and quality_metrics['sharpness'] < 500:
            try:
                image = self._gentle_auto_rotate(image)
            except Exception as e:
                logger.warning(f"Rotation échouée: {str(e)}")
        
        # 2. Normalisation douce de la taille
        image = self._normalize_image_size(image)
        
        # 3. Amélioration adaptative du contraste
        if quality_metrics['contrast'] < 40:
            image = self._adaptive_contrast_enhancement(image, quality_metrics)
        
        # 4. Réduction de bruit douce uniquement si nécessaire
        if quality_metrics['noise_estimate'] > 8:
            image = self._gentle_noise_reduction(image)
        
        # 5. Amélioration de netteté douce
        if quality_metrics['sharpness'] < 800:
            image = self._gentle_sharpening(image)
        
        # 6. Normalisation des couleurs seulement si problématique
        if self.enhancement_params['normalize_colors'] and quality_metrics['brightness'] < 100 or quality_metrics['brightness'] > 200:
            image = self._gentle_color_normalization(image)
        
        return image
    
    def _minimal_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Amélioration minimale pour préserver la qualité"""
        logger.info("Application d'une amélioration minimale...")
        
        # Conversion en PIL pour des ajustements très doux
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Légère amélioration du contraste
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = contrast_enhancer.enhance(1.03)
        
        # Légère amélioration de la netteté
        sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = sharpness_enhancer.enhance(1.05)
        
        # Reconversion
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _gentle_auto_rotate(self, image: np.ndarray) -> np.ndarray:
        """Rotation douce avec seuils plus élevés"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)  # Seuil plus élevé
            
            if lines is not None and len(lines) > 5:  # Minimum 5 lignes
                angles = []
                
                for line in lines[:10]:  # Moins de lignes
                    if isinstance(line, (list, tuple, np.ndarray)):
                        if len(line) == 1:
                            rho, theta = line[0]
                        elif len(line) == 2:
                            rho, theta = line
                        else:
                            continue
                    
                    angle = np.degrees(theta) - 90
                    if -15 <= angle <= 15:  # Seuil plus strict
                        angles.append(angle)
                
                if len(angles) >= 3:  # Minimum 3 angles cohérents
                    median_angle = np.median(angles)
                    
                    if abs(median_angle) > 1.5:  # Seuil plus élevé
                        logger.info(f"Rotation douce appliquée: {median_angle:.2f}°")
                        
                        rows, cols = image.shape[:2]
                        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), median_angle, 1)
                        
                        image = cv2.warpAffine(
                            image, rotation_matrix, (cols, rows),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255)
                        )
                    else:
                        logger.info("Angle de rotation trop faible - pas de rotation")
        except Exception as e:
            logger.warning(f"Erreur rotation douce: {str(e)}")
        
        return image
    
    def _adaptive_contrast_enhancement(self, image: np.ndarray, quality_metrics: Dict) -> np.ndarray:
        """Amélioration adaptative du contraste basée sur les métriques"""
        try:
            contrast_level = quality_metrics['contrast']
            
            if contrast_level < 20:
                factor = 1.15  # Amélioration plus forte
            elif contrast_level < 35:
                factor = 1.08  # Amélioration modérée  
            else:
                factor = 1.03  # Amélioration légère
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = contrast_enhancer.enhance(factor)
            
            logger.info(f"Contraste amélioré avec facteur: {factor}")
            return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Erreur amélioration contraste: {str(e)}")
            return image
    
    def _gentle_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Réduction douce du bruit qui préserve les détails"""
        try:
            # Filtre bilateral plus doux
            denoised = cv2.bilateralFilter(image, 5, 30, 30)  # Paramètres réduits
            
            # Mélange avec l'original pour préserver les détails
            result = cv2.addWeighted(image, 0.7, denoised, 0.3, 0)
            
            logger.info("Réduction douce du bruit appliquée")
            return result
            
        except Exception as e:
            logger.error(f"Erreur réduction bruit: {str(e)}")
            return image
    
    def _gentle_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Amélioration douce de la netteté"""
        try:
            # Noyau de netteté doux
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Mélange avec l'original
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            logger.info("Amélioration douce de netteté appliquée")
            return result
            
        except Exception as e:
            logger.error(f"Erreur amélioration netteté: {str(e)}")
            return image
    
    def _gentle_color_normalization(self, image: np.ndarray) -> np.ndarray:
        """Normalisation douce des couleurs"""
        try:
            # Conversion LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # CLAHE très doux
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Reconstruction
            lab = cv2.merge([l_channel, a_channel, b_channel])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Mélange avec l'original
            result = cv2.addWeighted(image, 0.8, normalized, 0.2, 0)
            
            logger.info("Normalisation douce des couleurs appliquée")
            return result
            
        except Exception as e:
            logger.error(f"Erreur normalisation couleurs: {str(e)}")
            return image
    
    def _normalize_image_size(self, image: np.ndarray, target_width: int = 1200) -> np.ndarray:
        """Normalisation douce de la taille"""
        height, width = image.shape[:2]
        
        if width != target_width and abs(width - target_width) > 100:  # Seuil plus élevé
            ratio = target_width / width
            new_height = int(height * ratio)
            
            # Utiliser INTER_LANCZOS4 pour une meilleure qualité
            image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Redimensionnement doux: {width}x{height} → {target_width}x{new_height}")
        
        return image
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse améliorée de la qualité d'image"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Netteté avec Laplacian
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contraste
            contrast = gray.std()
            
            # Luminosité
            brightness = gray.mean()
            
            # Estimation du bruit
            noise_estimate = np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))
            
            # Analyse de l'histogramme
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            
            # Score de qualité amélioré
            quality_score = self._calculate_improved_quality_score(
                sharpness, contrast, brightness, noise_estimate, hist_std
            )
            
            return {
                'sharpness': float(sharpness),
                'contrast': float(contrast),
                'brightness': float(brightness),
                'noise_estimate': float(noise_estimate),
                'histogram_std': float(hist_std),
                'resolution': f"{image.shape[1]}x{image.shape[0]}",
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse qualité: {str(e)}")
            return {
                'sharpness': 0.0, 'contrast': 0.0, 'brightness': 0.0,
                'noise_estimate': 0.0, 'histogram_std': 0.0,
                'resolution': f"{image.shape[1]}x{image.shape[0]}",
                'quality_score': 0.0
            }
    
    def _calculate_improved_quality_score(self, sharpness: float, contrast: float, 
                                        brightness: float, noise: float, hist_std: float) -> float:
        """Calcul amélioré du score de qualité"""
        
        # Score de netteté (0-30 points)
        sharpness_score = min(sharpness / 1000, 1.0) * 30
        
        # Score de contraste (0-25 points)
        contrast_score = min(contrast / 80, 1.0) * 25
        
        # Score de luminosité (0-20 points) - optimal autour de 128
        brightness_score = (1 - abs(brightness - 128) / 128) * 20
        
        # Pénalité bruit (0-15 points)
        noise_penalty = max(0, 15 - noise / 8)
        
        # Score distribution couleurs (0-10 points)
        hist_score = min(hist_std / 50000, 1.0) * 10
        
        total_score = sharpness_score + contrast_score + brightness_score + noise_penalty + hist_score
        
        return min(total_score, 100.0)
    
    def compare_before_after(self, original_path: str, enhanced_path: str, 
                           save_comparison: bool = True) -> Dict[str, Any]:
        """Compare les métriques avant et après amélioration"""
        
        try:
            original = cv2.imread(original_path)
            enhanced = cv2.imread(enhanced_path)
            
            if original is None or enhanced is None:
                raise ValueError("Impossible de charger les images")
            
            original_quality = self.analyze_image_quality(original)
            enhanced_quality = self.analyze_image_quality(enhanced)
            
            improvements = {
                'sharpness_improvement': enhanced_quality['sharpness'] - original_quality['sharpness'],
                'contrast_improvement': enhanced_quality['contrast'] - original_quality['contrast'],
                'quality_score_improvement': enhanced_quality['quality_score'] - original_quality['quality_score']
            }
            
            comparison_result = {
                'original': original_quality,
                'enhanced': enhanced_quality,
                'improvements': improvements,
                'improvement_successful': improvements['quality_score_improvement'] > 0
            }
            
            if save_comparison:
                try:
                    self._save_comparison_chart(comparison_result, original_path)
                except Exception as e:
                    logger.warning(f"Erreur sauvegarde graphique: {str(e)}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Erreur comparaison: {str(e)}")
            return {}
    
    def _save_comparison_chart(self, comparison_result: Dict[str, Any], original_path: str):
        """Sauvegarde un graphique de comparaison amélioré"""
        
        try:
            metrics = ['sharpness', 'contrast', 'brightness', 'quality_score']
            original_values = [comparison_result['original'][metric] for metric in metrics]
            enhanced_values = [comparison_result['enhanced'][metric] for metric in metrics]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Graphique en barres
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, original_values, width, label='Original', alpha=0.7, color='skyblue')
            bars2 = ax1.bar(x + width/2, enhanced_values, width, label='Amélioré', alpha=0.7, color='lightgreen')
            
            ax1.set_xlabel('Métriques')
            ax1.set_ylabel('Valeurs')
            ax1.set_title('Comparaison Avant/Après Amélioration')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Graphique des améliorations
            improvements = comparison_result['improvements']
            improvement_names = list(improvements.keys())
            improvement_values = list(improvements.values())
            
            colors = ['green' if val > 0 else 'red' for val in improvement_values]
            bars3 = ax2.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
            
            ax2.set_title('Améliorations (+ = mieux, - = pire)')
            ax2.set_ylabel('Différence')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Ajouter les valeurs
            for bar in bars3:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:+.1f}', ha='center', 
                        va='bottom' if height >= 0 else 'top', fontsize=9)
            
            # Sauvegarde
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            comparison_dir = "Data/quality_comparisons"
            os.makedirs(comparison_dir, exist_ok=True)
            
            comparison_path = os.path.join(comparison_dir, f"{base_name}_comparison.png")
            
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Graphique de comparaison sauvegardé: {comparison_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde graphique: {str(e)}")
            
    def enhance_batch_images(self, input_dir: str, output_dir: str, 
                        file_extensions: List[str] = None, 
                        preserve_quality: bool = True,
                        parallel_processing: bool = True,
                        max_workers: int = 4) -> Dict[str, Any]:
        """
        Traite un ensemble d'images en lot
        
        Args:
            input_dir: Dossier contenant les images source
            output_dir: Dossier de sortie pour les images améliorées
            file_extensions: Extensions autorisées (par défaut: ['.png', '.jpg', '.jpeg'])
            preserve_quality: Utiliser le mode conservateur
            parallel_processing: Traitement parallèle
            max_workers: Nombre de workers pour le parallélisme
            
        Returns:
            Dict: Résultats du traitement batch
        """
        
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Trouver toutes les images
        image_files = []
        for ext in file_extensions:
            pattern = os.path.join(input_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern_upper = os.path.join(input_dir, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern_upper))
        
        if not image_files:
            logger.warning(f"Aucune image trouvée dans {input_dir}")
            return {'total': 0, 'successful': 0, 'failed': 0, 'results': []}
        
        logger.info(f"🖼️  {len(image_files)} images trouvées pour traitement")
        
        # Statistiques
        batch_stats = {
            'total': len(image_files),
            'successful': 0,
            'failed': 0,
            'failed_files': [],
            'results': [],
            'start_time': datetime.now()
        }
        
        if parallel_processing and len(image_files) > 1:
            # Traitement parallèle
            batch_stats = self._process_images_parallel(
                image_files, output_dir, preserve_quality, max_workers, batch_stats
            )
        else:
            # Traitement séquentiel
            batch_stats = self._process_images_sequential(
                image_files, output_dir, preserve_quality, batch_stats
            )
        
        batch_stats['end_time'] = datetime.now()
        batch_stats['duration'] = (batch_stats['end_time'] - batch_stats['start_time']).total_seconds()
        
        # Génération du rapport
        self._generate_batch_report(batch_stats, input_dir, output_dir)
        
        return batch_stats

    def _process_images_parallel(self, image_files: List[str], output_dir: str, 
                            preserve_quality: bool, max_workers: int, 
                            batch_stats: Dict) -> Dict:
        """Traitement parallèle des images"""
        
        import concurrent.futures
        from datetime import datetime
        
        logger.info(f"🚀 Traitement parallèle avec {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre toutes les tâches
            future_to_file = {
                executor.submit(self._process_single_image_safe, img_file, output_dir, preserve_quality): img_file
                for img_file in image_files
            }
            
            # Collecter les résultats
            for future in concurrent.futures.as_completed(future_to_file):
                img_file = future_to_file[future]
                try:
                    result = future.result()
                    if result['success']:
                        batch_stats['successful'] += 1
                        logger.info(f"✅ Traité: {os.path.basename(img_file)}")
                    else:
                        batch_stats['failed'] += 1
                        batch_stats['failed_files'].append(img_file)
                        logger.error(f"❌ Échec: {os.path.basename(img_file)}")
                    
                    batch_stats['results'].append(result)
                    
                except Exception as e:
                    batch_stats['failed'] += 1
                    batch_stats['failed_files'].append(img_file)
                    logger.error(f"❌ Erreur {os.path.basename(img_file)}: {str(e)}")
        
        return batch_stats

    def _process_images_sequential(self, image_files: List[str], output_dir: str, 
                                preserve_quality: bool, batch_stats: Dict) -> Dict:
        """Traitement séquentiel des images"""
        
        logger.info("🔄 Traitement séquentiel")
        
        for i, img_file in enumerate(image_files, 1):
            logger.info(f"📸 [{i}/{len(image_files)}] Traitement: {os.path.basename(img_file)}")
            
            result = self._process_single_image_safe(img_file, output_dir, preserve_quality)
            
            if result['success']:
                batch_stats['successful'] += 1
                logger.info(f"✅ Traité avec succès")
            else:
                batch_stats['failed'] += 1
                batch_stats['failed_files'].append(img_file)
                logger.error(f"❌ Échec: {result.get('error', 'Erreur inconnue')}")
            
            batch_stats['results'].append(result)
        
        return batch_stats

    def _process_single_image_safe(self, img_file: str, output_dir: str, 
                                preserve_quality: bool) -> Dict[str, Any]:
        """Traite une seule image avec gestion d'erreurs"""
        
        try:
            # Générer le chemin de sortie
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            output_path = os.path.join(output_dir, f"enhanced_{base_name}.png")
            
            # Amélioration de l'image
            enhanced = self.enhance_invoice_image(img_file, output_path, preserve_quality)
            
            # Analyse des résultats
            original_quality = self.analyze_image_quality(cv2.imread(img_file))
            enhanced_quality = self.analyze_image_quality(enhanced)
            
            improvement = enhanced_quality['quality_score'] - original_quality['quality_score']
            
            return {
                'success': True,
                'input_file': img_file,
                'output_file': output_path,
                'original_quality': original_quality,
                'enhanced_quality': enhanced_quality,
                'quality_improvement': improvement,
                'improvement_successful': improvement > -2  # Tolérance de -2 points
            }
            
        except Exception as e:
            return {
                'success': False,
                'input_file': img_file,
                'error': str(e)
            }

    def _generate_batch_report(self, batch_stats: Dict, input_dir: str, output_dir: str):
        """Génère un rapport détaillé du traitement batch"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = "Data/batch_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # Rapport JSON
        json_report_path = os.path.join(report_dir, f"batch_report_{timestamp}.json")
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Rapport HTML
        html_report_path = os.path.join(report_dir, f"batch_report_{timestamp}.html")
        self._create_batch_html_report(batch_stats, html_report_path, input_dir, output_dir)
        
        # Statistiques console
        print("\n" + "="*70)
        print("📊 RÉSULTATS DU TRAITEMENT BATCH")
        print("="*70)
        print(f"📁 Dossier source    : {input_dir}")
        print(f"📁 Dossier sortie    : {output_dir}")
        print(f"⏱️  Durée totale      : {batch_stats['duration']:.1f} secondes")
        print(f"📸 Total images      : {batch_stats['total']}")
        print(f"✅ Succès            : {batch_stats['successful']}")
        print(f"❌ Échecs            : {batch_stats['failed']}")
        
        if batch_stats['failed_files']:
            print(f"\n❌ Fichiers échoués:")
            for failed_file in batch_stats['failed_files'][:5]:  # Max 5 pour éviter spam
                print(f"   - {os.path.basename(failed_file)}")
            if len(batch_stats['failed_files']) > 5:
                print(f"   ... et {len(batch_stats['failed_files']) - 5} autres")
        
        # Statistiques de qualité
        successful_results = [r for r in batch_stats['results'] if r.get('success', False)]
        if successful_results:
            improvements = [r['quality_improvement'] for r in successful_results]
            avg_improvement = np.mean(improvements)
            positive_improvements = sum(1 for i in improvements if i > 0)
            
            print(f"\n📈 QUALITÉ:")
            print(f"   Amélioration moyenne : {avg_improvement:+.1f} points")
            print(f"   Images améliorées    : {positive_improvements}/{len(successful_results)}")
            print(f"   Taux de succès       : {positive_improvements/len(successful_results)*100:.1f}%")
        
        print("="*70)
        print(f"📄 Rapports sauvegardés dans: {report_dir}")
        print("="*70)

    def _create_batch_html_report(self, batch_stats: Dict, html_path: str, 
                                input_dir: str, output_dir: str):
        """Crée un rapport HTML détaillé"""
        
        successful_results = [r for r in batch_stats['results'] if r.get('success', False)]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport Batch - Amélioration d'Images</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }}
                .stat-number {{ font-size: 28px; font-weight: bold; color: #667eea; }}
                .results-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .results-table th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
                .results-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .success {{ background-color: #d4edda; }}
                .failure {{ background-color: #f8d7da; }}
                .improvement-positive {{ color: #28a745; font-weight: bold; }}
                .improvement-negative {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Rapport de Traitement Batch</h1>
                    <p>Amélioration automatique d'images de factures</p>
                    <p>Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{batch_stats['total']}</div>
                        <div>Images Totales</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{batch_stats['successful']}</div>
                        <div>Succès</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{batch_stats['failed']}</div>
                        <div>Échecs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{batch_stats['duration']:.1f}s</div>
                        <div>Durée</div>
                    </div>
                </div>
                
                <h2>📁 Informations</h2>
                <p><strong>Dossier source:</strong> {input_dir}</p>
                <p><strong>Dossier sortie:</strong> {output_dir}</p>
                
                <h2>📸 Résultats Détaillés</h2>
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Fichier</th>
                            <th>Statut</th>
                            <th>Qualité Originale</th>
                            <th>Qualité Améliorée</th>
                            <th>Amélioration</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for result in batch_stats['results'][:50]:  # Limiter à 50 pour éviter HTML trop lourd
            if result.get('success', False):
                filename = os.path.basename(result['input_file'])
                original_score = result['original_quality']['quality_score']
                enhanced_score = result['enhanced_quality']['quality_score']
                improvement = result['quality_improvement']
                
                status_class = 'success'
                status_text = '✅ Succès'
                improvement_class = 'improvement-positive' if improvement > 0 else 'improvement-negative'
                
                html_content += f"""
                    <tr class="{status_class}">
                        <td>{filename}</td>
                        <td>{status_text}</td>
                        <td>{original_score:.1f}/100</td>
                        <td>{enhanced_score:.1f}/100</td>
                        <td class="{improvement_class}">{improvement:+.1f}</td>
                    </tr>
                """
            else:
                filename = os.path.basename(result['input_file'])
                error = result.get('error', 'Erreur inconnue')
                
                html_content += f"""
                    <tr class="failure">
                        <td>{filename}</td>
                        <td>❌ Échec</td>
                        <td colspan="3">{error}</td>
                    </tr>
                """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    """Fonction de test avec options multiples"""
    enhancer = InvoiceImageEnhancer()
    
    # Test sur l'image disponible
    test_image = "Data/images_Factures/F1.jpeg"
    
    if os.path.exists(test_image):
        try:
            output_dir = "Data/images_Factures/"
            os.makedirs(output_dir, exist_ok=True)
            
            # Test avec préservation de qualité
            output_path = os.path.join(output_dir, "F1_enhanced.jpeg")
            
            print("🔧 Test avec amélioration DOUCE (preserve_quality=True)")
            enhanced = enhancer.enhance_invoice_image(test_image, output_path, preserve_quality=True)
            
            # Analyse
            quality_metrics = enhancer.analyze_image_quality(enhanced)
            
            print("\n" + "="*60)
            print("🎯 ANALYSE DE QUALITÉ - VERSION OPTIMISÉE")
            print("="*60)
            for key, value in quality_metrics.items():
                print(f"{key.replace('_', ' ').title():<20}: {value}")
            print("="*60)
            
            # Comparaison
            if os.path.exists(output_path):
                comparison = enhancer.compare_before_after(test_image, output_path)
                if comparison:
                    print("\n🔄 RÉSULTATS D'AMÉLIORATION:")
                    improvements = comparison.get('improvements', {})
                    success = comparison.get('improvement_successful', False)
                    
                    status = "✅ SUCCÈS" if success else "❌ DÉGRADATION"
                    print(f"Statut global: {status}")
                    print("-" * 40)
                    
                    for key, value in improvements.items():
                        icon = "📈" if value > 0 else "📉"
                        print(f"{icon} {key.replace('_', ' ').title():<25}: {value:+.2f}")
                    
                    print("-" * 40)
                    print(f"Score final: {quality_metrics['quality_score']:.1f}/100")
            
            # Test avec différents modes
            print(f"\n🧪 Tests supplémentaires...")
            
            # Mode agressif pour comparaison
            enhancer.enhancement_params.update({
                'contrast_factor': 1.2,
                'brightness_factor': 1.1,
                'sharpness_factor': 1.3,
                'ocr_optimization': True
            })
            
            aggressive_path = os.path.join(output_dir, "aggressive_complex_invoice_0001.png")
            print("🔧 Test avec amélioration AGRESSIVE")
            enhancer.enhance_invoice_image(test_image, aggressive_path, preserve_quality=False)
            
            print(f"\n📁 Images générées dans: {output_dir}/")
            print("📊 Graphiques de comparaison dans: Data/quality_comparisons/")
            
        except Exception as e:
            logger.error(f"❌ Erreur test: {str(e)}")
            print(f"Erreur: {str(e)}")
    else:
        print(f"❌ Image de test non trouvée: {test_image}")

if __name__ == "__main__":
    main()
