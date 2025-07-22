#!/usr/bin/env python3
"""
Pipeline de prétraitement complet pour les factures
Intègre PDF → Images → Amélioration → Validation qualité
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
from datetime import datetime
import pandas as pd

# Imports des modules personnalisés
from preprocessing.pdf_to_image import convert_from_path
from image_enhancement import InvoiceImageEnhancer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class InvoicePreprocessingPipeline:
    """Pipeline complet de prétraitement des factures"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise la pipeline de prétraitement
        
        Args:
            config: Configuration personnalisée (optionnel)
        """
        self.config = config or self._get_default_config()
        self.enhancer = InvoiceImageEnhancer()
        
        # Création des dossiers nécessaires
        self._setup_directories()
        
        # Statistiques de traitement
        self.processing_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'failed_files': []
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut de la pipeline"""
        return {
            'input_pdf_dir': 'Data/ids_factures',
            'temp_images_dir': 'Data/temp_images',
            'output_images_dir': 'Data/processed_images',
            'quality_reports_dir': 'Data/quality_reports',
            'backup_dir': 'Data/backup_originals',
            'pdf_conversion': {
                'dpi': 300,
                'format': 'PNG',
                'poppler_path': r"C:/tools/poppler/Library/bin"
            },
            'parallel_processing': True,
            'max_workers': 4,
            'quality_threshold': 60.0,
            'backup_originals': True,
            'generate_reports': True
        }
    
    def _setup_directories(self):
        """Crée tous les dossiers nécessaires"""
        directories = [
            self.config['temp_images_dir'],
            self.config['output_images_dir'],
            self.config['quality_reports_dir'],
            self.config['backup_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Dossier préparé: {directory}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Exécute la pipeline complète de prétraitement
        
        Returns:
            Dict: Résultats du traitement avec statistiques
        """
        logger.info("🚀 Début de la pipeline de prétraitement complète")
        self.processing_stats['start_time'] = datetime.now()
        
        try:
            # Étape 1: Conversion PDF → Images
            pdf_files = self._get_pdf_files()
            if not pdf_files:
                logger.warning("Aucun fichier PDF trouvé dans le dossier d'entrée")
                return self.processing_stats
            
            self.processing_stats['total_files'] = len(pdf_files)
            logger.info(f"📁 {len(pdf_files)} fichiers PDF trouvés")
            
            # Étape 2: Conversion en images
            extracted_images = self._convert_pdfs_to_images(pdf_files)
            
            # Étape 3: Amélioration des images
            enhanced_images = self._enhance_images(extracted_images)
            
            # Étape 4: Validation qualité
            quality_results = self._validate_quality(enhanced_images)
            
            # Étape 5: Génération des rapports
            if self.config['generate_reports']:
                self._generate_processing_report(quality_results)
            
            # Étape 6: Nettoyage des fichiers temporaires
            self._cleanup_temp_files()
            
        except Exception as e:
            logger.error(f"❌ Erreur dans la pipeline: {str(e)}")
            self.processing_stats['failed'] = self.processing_stats['total_files']
        
        finally:
            self.processing_stats['end_time'] = datetime.now()
            self._log_final_statistics()
        
        return self.processing_stats
    
    def _get_pdf_files(self) -> List[str]:
        """Récupère la liste des fichiers PDF à traiter"""
        input_dir = Path(self.config['input_pdf_dir'])
        
        if not input_dir.exists():
            logger.error(f"❌ Dossier d'entrée non trouvé: {input_dir}")
            return []
        
        pdf_files = list(input_dir.glob('*.pdf'))
        return [str(pdf_file) for pdf_file in pdf_files]
    
    def _convert_pdfs_to_images(self, pdf_files: List[str]) -> List[str]:
        """Convertit les PDFs en images"""
        logger.info("📄 Début de la conversion PDF → Images")
        
        extracted_images = []
        
        for pdf_file in pdf_files:
            try:
                # Sauvegarde de l'original si demandé
                if self.config['backup_originals']:
                    self._backup_original_file(pdf_file)
                
                # Conversion PDF → Images
                images = convert_from_path(
                    pdf_file,
                    dpi=self.config['pdf_conversion']['dpi'],
                    poppler_path=self.config['pdf_conversion'].get('poppler_path')
                )
                
                # Sauvegarde des images extraites
                for i, image in enumerate(images):
                    base_name = Path(pdf_file).stem
                    image_filename = f"{base_name}_page{i+1}.png"
                    image_path = os.path.join(self.config['temp_images_dir'], image_filename)
                    
                    image.save(image_path, 'PNG', dpi=(300, 300))
                    extracted_images.append(image_path)
                    
                    logger.info(f"✅ Image extraite: {image_filename}")
                
                self.processing_stats['successful'] += 1
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de la conversion de {pdf_file}: {str(e)}")
                self.processing_stats['failed'] += 1
                self.processing_stats['failed_files'].append(pdf_file)
        
        logger.info(f"📄 Conversion terminée: {len(extracted_images)} images extraites")
        return extracted_images
    
    def _enhance_images(self, image_paths: List[str]) -> List[str]:
        """Améliore la qualité des images extraites"""
        logger.info("🎨 Début de l'amélioration des images")
        
        enhanced_images = []
        
        if self.config['parallel_processing']:
            # Traitement parallèle
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config['max_workers']
            ) as executor:
                
                future_to_path = {
                    executor.submit(self._enhance_single_image, img_path): img_path 
                    for img_path in image_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    img_path = future_to_path[future]
                    try:
                        enhanced_path = future.result()
                        if enhanced_path:
                            enhanced_images.append(enhanced_path)
                    except Exception as e:
                        logger.error(f"❌ Erreur amélioration {img_path}: {str(e)}")
        else:
            # Traitement séquentiel
            for img_path in image_paths:
                enhanced_path = self._enhance_single_image(img_path)
                if enhanced_path:
                    enhanced_images.append(enhanced_path)
        
        logger.info(f"🎨 Amélioration terminée: {len(enhanced_images)} images améliorées")
        return enhanced_images
    
    def _enhance_single_image(self, image_path: str) -> Optional[str]:
        """Améliore une seule image"""
        try:
            # Génération du chemin de sortie
            base_name = Path(image_path).stem
            output_path = os.path.join(
                self.config['output_images_dir'], 
                f"enhanced_{base_name}.png"
            )
            
            # Amélioration de l'image
            self.enhancer.enhance_invoice_image(image_path, output_path)
            
            logger.info(f"✅ Image améliorée: {base_name}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erreur amélioration {image_path}: {str(e)}")
            return None
    
    def _validate_quality(self, enhanced_images: List[str]) -> List[Dict[str, Any]]:
        """Valide la qualité des images améliorées"""
        logger.info("🔍 Début de la validation qualité")
        
        quality_results = []
        
        for image_path in enhanced_images:
            try:
                # Chargement et analyse de l'image
                import cv2
                image = cv2.imread(image_path)
                
                if image is not None:
                    quality_metrics = self.enhancer.analyze_image_quality(image)
                    
                    result = {
                        'image_path': image_path,
                        'filename': Path(image_path).name,
                        'metrics': quality_metrics,
                        'quality_pass': quality_metrics['quality_score'] >= self.config['quality_threshold']
                    }
                    
                    quality_results.append(result)
                    
                    # Log du résultat
                    status = "✅ PASS" if result['quality_pass'] else "⚠️ BELOW_THRESHOLD"
                    score = quality_metrics['quality_score']
                    logger.info(f"{status} - {result['filename']}: Score = {score:.1f}")
                
            except Exception as e:
                logger.error(f"❌ Erreur validation {image_path}: {str(e)}")
        
        # Statistiques de qualité
        passed = sum(1 for r in quality_results if r['quality_pass'])
        total = len(quality_results)
        logger.info(f"🔍 Validation terminée: {passed}/{total} images passent le seuil qualité")
        
        return quality_results
    
    def _generate_processing_report(self, quality_results: List[Dict[str, Any]]):
        """Génère un rapport détaillé du traitement"""
        logger.info("📊 Génération du rapport de traitement")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rapport JSON détaillé
        json_report_path = os.path.join(
            self.config['quality_reports_dir'], 
            f"processing_report_{timestamp}.json"
        )
        
        full_report = {
            'processing_info': {
                'timestamp': timestamp,
                'config': self.config,
                'stats': self.processing_stats
            },
            'quality_results': quality_results
        }
        
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Rapport CSV pour analyse
        csv_report_path = os.path.join(
            self.config['quality_reports_dir'], 
            f"quality_metrics_{timestamp}.csv"
        )
        
        if quality_results:
            df_data = []
            for result in quality_results:
                row = {
                    'filename': result['filename'],
                    'quality_pass': result['quality_pass'],
                    **result['metrics']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_report_path, index=False)
        
        # Rapport HTML visuel
        self._generate_html_report(quality_results, timestamp)
        
        logger.info(f"📊 Rapports générés:")
        logger.info(f"   - JSON: {json_report_path}")
        logger.info(f"   - CSV: {csv_report_path}")
    
    def _generate_html_report(self, quality_results: List[Dict[str, Any]], timestamp: str):
        """Génère un rapport HTML visuel"""
        html_report_path = os.path.join(
            self.config['quality_reports_dir'], 
            f"visual_report_{timestamp}.html"
        )
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de Traitement - Factures</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2e8b57; color: white; padding: 10px; text-align: center; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }}
                .quality-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .quality-table th, .quality-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .quality-table th {{ background-color: #f2f2f2; }}
                .pass {{ background-color: #d4edda; }}
                .fail {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Rapport de Traitement des Factures</h1>
                <p>Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>📄 Total Traités</h3>
                    <p style="font-size: 24px; color: #2e8b57;">{self.processing_stats['total_files']}</p>
                </div>
                <div class="stat-box">
                    <h3>✅ Succès</h3>
                    <p style="font-size: 24px; color: #28a745;">{self.processing_stats['successful']}</p>
                </div>
                <div class="stat-box">
                    <h3>❌ Échecs</h3>
                    <p style="font-size: 24px; color: #dc3545;">{self.processing_stats['failed']}</p>
                </div>
                <div class="stat-box">
                    <h3>🎯 Qualité OK</h3>
                    <p style="font-size: 24px; color: #007bff;">{sum(1 for r in quality_results if r['quality_pass'])}</p>
                </div>
            </div>
            
            <table class="quality-table">
                <thead>
                    <tr>
                        <th>Fichier</th>
                        <th>Score Qualité</th>
                        <th>Netteté</th>
                        <th>Contraste</th>
                        <th>Luminosité</th>
                        <th>Statut</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for result in quality_results:
            metrics = result['metrics']
            status_class = 'pass' if result['quality_pass'] else 'fail'
            status_text = 'PASS ✅' if result['quality_pass'] else 'FAIL ❌'
            
            html_content += f"""
                    <tr class="{status_class}">
                        <td>{result['filename']}</td>
                        <td>{metrics['quality_score']:.1f}</td>
                        <td>{metrics['sharpness']:.1f}</td>
                        <td>{metrics['contrast']:.1f}</td>
                        <td>{metrics['brightness']:.1f}</td>
                        <td>{status_text}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"   - HTML: {html_report_path}")
    
    def _backup_original_file(self, pdf_file: str):
        """Sauvegarde le fichier original"""
        try:
            import shutil
            backup_path = os.path.join(
                self.config['backup_dir'], 
                Path(pdf_file).name
            )
            shutil.copy2(pdf_file, backup_path)
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde {pdf_file}: {str(e)}")
    
    def _cleanup_temp_files(self):
        """Nettoie les fichiers temporaires"""
        try:
            import shutil
            temp_dir = Path(self.config['temp_images_dir'])
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir, exist_ok=True)
                logger.info("🧹 Fichiers temporaires nettoyés")
        except Exception as e:
            logger.warning(f"⚠️ Erreur nettoyage fichiers temporaires: {str(e)}")
    
    def _log_final_statistics(self):
        """Affiche les statistiques finales"""
        duration = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
        
        logger.info("=" * 60)
        logger.info("📊 STATISTIQUES FINALES DE TRAITEMENT")
        logger.info("=" * 60)
        logger.info(f"⏱️  Durée totale: {duration:.1f} secondes")
        logger.info(f"📄 Fichiers traités: {self.processing_stats['total_files']}")
        logger.info(f"✅ Succès: {self.processing_stats['successful']}")
        logger.info(f"❌ Échecs: {self.processing_stats['failed']}")
        
        if self.processing_stats['failed_files']:
            logger.info("❌ Fichiers échoués:")
            for failed_file in self.processing_stats['failed_files']:
                logger.info(f"   - {failed_file}")
        
        logger.info("=" * 60)

def main():
    """Fonction principale avec interface CLI"""
    parser = argparse.ArgumentParser(description='Pipeline de prétraitement de factures')
    
    parser.add_argument('--input-dir', default='Data/ids_factures',
                       help='Dossier des PDFs source')
    parser.add_argument('--output-dir', default='Data/processed_images',
                       help='Dossier des images finales')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Résolution pour la conversion PDF')
    parser.add_argument('--quality-threshold', type=float, default=60.0,
                       help='Seuil de qualité minimum')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Activation du traitement parallèle')
    parser.add_argument('--workers', type=int, default=4,
                       help='Nombre de workers pour le traitement parallèle')
    
    args = parser.parse_args()
    
    # Configuration personnalisée
    custom_config = {
        'input_pdf_dir': args.input_dir,
        'output_images_dir': args.output_dir,
        'pdf_conversion': {'dpi': args.dpi},
        'quality_threshold': args.quality_threshold,
        'parallel_processing': args.parallel,
        'max_workers': args.workers
    }
    
    # Lancement de la pipeline
    pipeline = InvoicePreprocessingPipeline(custom_config)
    results = pipeline.run_full_pipeline()
    
    # Affichage du résumé
    print("\n" + "=" * 50)
    print("🎉 TRAITEMENT TERMINÉ")
    print("=" * 50)
    print(f"✅ {results['successful']} fichiers traités avec succès")
    print(f"❌ {results['failed']} fichiers en échec")
    print(f"📊 Consultez les rapports dans: Data/quality_reports/")
    print("=" * 50)

if __name__ == "__main__":
    main()
