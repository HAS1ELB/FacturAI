"""
Système d'évaluation pour le module VLM de FacturAI
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import pandas as pd

from .vlm_processor import VLMProcessor
from .utils import VLMVisualizer

logger = logging.getLogger(__name__)

class VLMEvaluator:
    """
    Système d'évaluation pour le module VLM
    
    Évalue les performances de détection de zones, qualité d'analyse,
    temps de traitement et comparaison entre modèles
    """
    
    def __init__(self, output_dir: str = "Data/vlm_evaluation"):
        """
        Initialise l'évaluateur
        
        Args:
            output_dir: Répertoire de sortie pour les évaluations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.processor = VLMProcessor()
        self.visualizer = VLMVisualizer()
        
        # Métriques par défaut
        self.metrics = {
            'zone_detection': {
                'header_precision': 0.0,
                'header_recall': 0.0,
                'footer_precision': 0.0,
                'footer_recall': 0.0,
                'table_precision': 0.0,
                'table_recall': 0.0,
                'amount_precision': 0.0,
                'amount_recall': 0.0,
                'overall_f1': 0.0
            },
            'performance': {
                'avg_processing_time': 0.0,
                'min_processing_time': 0.0,
                'max_processing_time': 0.0,
                'throughput': 0.0  # images/second
            },
            'quality': {
                'avg_confidence': 0.0,
                'layout_quality_score': 0.0,
                'integration_score': 0.0
            }
        }
    
    def evaluate_dataset(self, images_dir: str, ground_truth_file: str = None, 
                        ocr_results_dir: str = None) -> Dict[str, Any]:
        """
        Évalue le module VLM sur un dataset complet
        
        Args:
            images_dir: Répertoire contenant les images de test
            ground_truth_file: Fichier JSON avec les vérités terrain (optionnel)
            ocr_results_dir: Répertoire avec les résultats OCR correspondants
        
        Returns:
            Métriques d'évaluation complètes
        """
        print(f"📊 Évaluation du dataset: {images_dir}")
        
        # Collecte des images
        image_files = self._collect_image_files(images_dir)
        if not image_files:
            raise ValueError(f"Aucune image trouvée dans {images_dir}")
        
        print(f"📁 {len(image_files)} images à traiter")
        
        # Chargement de la vérité terrain
        ground_truth = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            print(f"✅ Vérité terrain chargée: {len(ground_truth)} annotations")
        
        # Traitement et évaluation
        results = []
        processing_times = []
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            print(f"🔄 Traitement {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                # Chargement des résultats OCR si disponibles
                ocr_results = self._load_ocr_results(image_path, ocr_results_dir)
                
                # Traitement VLM
                result = self.processor.process_invoice(image_path, ocr_results)
                results.append(result)
                
                processing_times.append(result.get('processing_time', 0))
                
                # Évaluation contre la vérité terrain
                if ground_truth:
                    image_name = os.path.basename(image_path)
                    if image_name in ground_truth:
                        eval_result = self._evaluate_against_ground_truth(
                            result, ground_truth[image_name]
                        )
                        result['evaluation'] = eval_result
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {image_path}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Calcul des métriques globales
        metrics = self._calculate_global_metrics(results, processing_times, total_time)
        
        # Sauvegarde des résultats
        evaluation_report = {
            'timestamp': time.time(),
            'dataset_info': {
                'images_dir': images_dir,
                'total_images': len(image_files),
                'processed_images': len(results),
                'ground_truth_available': bool(ground_truth),
                'ocr_integration': ocr_results_dir is not None
            },
            'metrics': metrics,
            'detailed_results': results,
            'model_info': self.processor.get_model_info()
        }
        
        # Sauvegarde
        report_file = os.path.join(self.output_dir, f"evaluation_report_{int(time.time())}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Évaluation terminée: {report_file}")
        
        # Génération de visualisations
        self._generate_evaluation_visualizations(evaluation_report)
        
        return evaluation_report
    
    def compare_models(self, images_dir: str, models: List[str], 
                      ground_truth_file: str = None) -> Dict[str, Any]:
        """
        Compare les performances de différents modèles VLM
        
        Args:
            images_dir: Répertoire d'images de test
            models: Liste des modèles à comparer
            ground_truth_file: Fichier de vérité terrain
        
        Returns:
            Comparaison détaillée des modèles
        """
        print(f"🔬 Comparaison de {len(models)} modèles VLM")
        
        comparison_results = {
            'timestamp': time.time(),
            'models_compared': models,
            'results_by_model': {},
            'comparison_metrics': {}
        }
        
        image_files = self._collect_image_files(images_dir)
        
        for model_name in models:
            print(f"\n📈 Évaluation du modèle: {model_name}")
            
            try:
                # Chargement du modèle
                self.processor.load_model(model_name)
                
                # Évaluation sur le dataset
                model_results = self.evaluate_dataset(images_dir, ground_truth_file)
                comparison_results['results_by_model'][model_name] = model_results
                
            except Exception as e:
                logger.error(f"Erreur avec le modèle {model_name}: {e}")
                comparison_results['results_by_model'][model_name] = {'error': str(e)}
        
        # Calcul des métriques de comparaison
        comparison_results['comparison_metrics'] = self._calculate_comparison_metrics(
            comparison_results['results_by_model']
        )
        
        # Sauvegarde
        comparison_file = os.path.join(self.output_dir, f"model_comparison_{int(time.time())}.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Génération de graphiques comparatifs
        self._generate_comparison_charts(comparison_results)
        
        print(f"✅ Comparaison terminée: {comparison_file}")
        return comparison_results
    
    def benchmark_performance(self, images_dir: str, iterations: int = 3) -> Dict[str, Any]:
        """
        Effectue un benchmark de performance
        
        Args:
            images_dir: Répertoire d'images de test
            iterations: Nombre d'itérations pour la moyenne
        
        Returns:
            Résultats de benchmark
        """
        print(f"⚡ Benchmark de performance ({iterations} itérations)")
        
        image_files = self._collect_image_files(images_dir)
        if len(image_files) > 10:
            image_files = image_files[:10]  # Limiter pour le benchmark
        
        benchmark_results = {
            'timestamp': time.time(),
            'iterations': iterations,
            'images_count': len(image_files),
            'model_info': self.processor.get_model_info(),
            'performance_data': []
        }
        
        for iteration in range(iterations):
            print(f"🔄 Itération {iteration + 1}/{iterations}")
            
            iteration_times = []
            iteration_start = time.time()
            
            for image_path in image_files:
                start_time = time.time()
                try:
                    result = self.processor.process_invoice(image_path)
                    processing_time = time.time() - start_time
                    iteration_times.append(processing_time)
                except Exception as e:
                    logger.error(f"Erreur: {e}")
                    continue
            
            iteration_total = time.time() - iteration_start
            
            iteration_data = {
                'iteration': iteration + 1,
                'total_time': iteration_total,
                'avg_time_per_image': statistics.mean(iteration_times) if iteration_times else 0,
                'min_time': min(iteration_times) if iteration_times else 0,
                'max_time': max(iteration_times) if iteration_times else 0,
                'throughput': len(iteration_times) / iteration_total if iteration_total > 0 else 0,
                'individual_times': iteration_times
            }
            
            benchmark_results['performance_data'].append(iteration_data)
        
        # Calcul des moyennes
        avg_metrics = self._calculate_benchmark_averages(benchmark_results['performance_data'])
        benchmark_results['average_metrics'] = avg_metrics
        
        # Sauvegarde
        benchmark_file = os.path.join(self.output_dir, f"benchmark_{int(time.time())}.json")
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Benchmark terminé: {benchmark_file}")
        print(f"📊 Débit moyen: {avg_metrics['avg_throughput']:.2f} images/sec")
        print(f"⏱️  Temps moyen: {avg_metrics['avg_time_per_image']:.2f}s/image")
        
        return benchmark_results
    
    def _collect_image_files(self, images_dir: str) -> List[str]:
        """Collecte les fichiers images dans un répertoire"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        return [str(f) for f in sorted(image_files)]
    
    def _load_ocr_results(self, image_path: str, ocr_results_dir: str) -> Optional[Dict]:
        """Charge les résultats OCR correspondant à une image"""
        if not ocr_results_dir:
            return None
        
        image_name = Path(image_path).stem
        ocr_patterns = [
            f"enhanced_{image_name}_ocr.json",
            f"{image_name}_ocr.json",
            f"ocr_{image_name}.json"
        ]
        
        for pattern in ocr_patterns:
            ocr_file = os.path.join(ocr_results_dir, pattern)
            if os.path.exists(ocr_file):
                try:
                    with open(ocr_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement OCR {ocr_file}: {e}")
        
        return None
    
    def _evaluate_against_ground_truth(self, result: Dict[str, Any], 
                                     ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue un résultat contre la vérité terrain"""
        evaluation = {
            'zone_detection': {
                'header': self._evaluate_zone_detection(
                    result.get('detected_zones', {}).get('header', {}),
                    ground_truth.get('header', {})
                ),
                'footer': self._evaluate_zone_detection(
                    result.get('detected_zones', {}).get('footer', {}),
                    ground_truth.get('footer', {})
                ),
                'tables': self._evaluate_tables_detection(
                    result.get('detected_zones', {}).get('tables', []),
                    ground_truth.get('tables', [])
                ),
                'amounts': self._evaluate_amounts_detection(
                    result.get('detected_zones', {}).get('amount_zones', []),
                    ground_truth.get('amounts', [])
                )
            },
            'layout_quality': self._evaluate_layout_quality(result, ground_truth)
        }
        
        return evaluation
    
    def _evaluate_zone_detection(self, detected: Dict[str, Any], 
                                expected: Dict[str, Any]) -> Dict[str, float]:
        """Évalue la détection d'une zone spécifique"""
        detected_present = detected.get('detected', False)
        expected_present = expected.get('present', False)
        
        # Calcul précision/rappel
        if detected_present and expected_present:
            precision = recall = 1.0  # True Positive
        elif detected_present and not expected_present:
            precision = 0.0  # False Positive
            recall = 0.0
        elif not detected_present and expected_present:
            precision = 0.0
            recall = 0.0  # False Negative
        else:
            precision = recall = 1.0  # True Negative
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confidence': detected.get('confidence', 0)
        }
    
    def _evaluate_tables_detection(self, detected: List[Dict], 
                                  expected: List[Dict]) -> Dict[str, float]:
        """Évalue la détection de tableaux"""
        detected_count = len(detected)
        expected_count = len(expected)
        
        if expected_count == 0:
            precision = 1.0 if detected_count == 0 else 0.0
            recall = 1.0
        else:
            # Approximation simple
            true_positives = min(detected_count, expected_count)
            precision = true_positives / detected_count if detected_count > 0 else 0
            recall = true_positives / expected_count
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detected_count': detected_count,
            'expected_count': expected_count
        }
    
    def _evaluate_amounts_detection(self, detected: List[Dict], 
                                   expected: List[Dict]) -> Dict[str, float]:
        """Évalue la détection de montants"""
        # Évaluation basée sur la correspondance des valeurs
        detected_values = [float(str(a.get('value', 0)).replace(',', '.')) 
                          for a in detected if a.get('value')]
        expected_values = [float(str(a.get('value', 0)).replace(',', '.')) 
                          for a in expected if a.get('value')]
        
        matches = 0
        for expected_val in expected_values:
            for detected_val in detected_values:
                if abs(detected_val - expected_val) < 0.01:
                    matches += 1
                    break
        
        precision = matches / len(detected_values) if detected_values else 0
        recall = matches / len(expected_values) if expected_values else 1
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'matches': matches,
            'detected_count': len(detected_values),
            'expected_count': len(expected_values)
        }
    
    def _evaluate_layout_quality(self, result: Dict[str, Any], 
                                ground_truth: Dict[str, Any]) -> float:
        """Évalue la qualité de l'analyse de mise en page"""
        layout_analysis = result.get('layout_analysis', {})
        quality_score = layout_analysis.get('layout_quality', {}).get('overall_score', 0)
        
        # Comparaison avec la vérité terrain si disponible
        expected_quality = ground_truth.get('layout_quality', 0.8)  # Valeur par défaut
        
        # Score basé sur la proximité avec la qualité attendue
        quality_accuracy = 1 - abs(quality_score - expected_quality)
        
        return max(quality_accuracy, 0)
    
    def _calculate_global_metrics(self, results: List[Dict], 
                                 processing_times: List[float], 
                                 total_time: float) -> Dict[str, Any]:
        """Calcule les métriques globales"""
        metrics = {
            'zone_detection': {
                'header_precision': 0.0,
                'header_recall': 0.0,
                'footer_precision': 0.0,
                'footer_recall': 0.0,
                'table_f1': 0.0,
                'amount_f1': 0.0,
                'overall_f1': 0.0
            },
            'performance': {
                'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
                'min_processing_time': min(processing_times) if processing_times else 0,
                'max_processing_time': max(processing_times) if processing_times else 0,
                'total_time': total_time,
                'throughput': len(results) / total_time if total_time > 0 else 0
            },
            'quality': {
                'avg_confidence': 0.0,
                'avg_layout_quality': 0.0,
                'success_rate': len([r for r in results if 'error' not in r]) / len(results) if results else 0
            }
        }
        
        # Calcul des moyennes pour les métriques de qualité
        confidences = []
        layout_qualities = []
        
        for result in results:
            vlm_analysis = result.get('vlm_analysis', {})
            confidences.append(vlm_analysis.get('confidence', 0))
            
            layout_analysis = result.get('layout_analysis', {})
            layout_quality = layout_analysis.get('layout_quality', {}).get('overall_score', 0)
            layout_qualities.append(layout_quality)
        
        if confidences:
            metrics['quality']['avg_confidence'] = statistics.mean(confidences)
        if layout_qualities:
            metrics['quality']['avg_layout_quality'] = statistics.mean(layout_qualities)
        
        return metrics
    
    def _calculate_comparison_metrics(self, results_by_model: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les métriques de comparaison entre modèles"""
        comparison = {
            'best_model': {
                'overall_performance': None,
                'fastest': None,
                'most_accurate': None,
                'best_quality': None
            },
            'rankings': {}
        }
        
        model_scores = {}
        
        for model_name, model_results in results_by_model.items():
            if 'error' in model_results:
                continue
            
            metrics = model_results.get('metrics', {})
            performance = metrics.get('performance', {})
            quality = metrics.get('quality', {})
            
            # Score composite
            composite_score = (
                quality.get('avg_confidence', 0) * 0.3 +
                quality.get('avg_layout_quality', 0) * 0.3 +
                (1 / max(performance.get('avg_processing_time', 1), 0.1)) * 0.2 +
                quality.get('success_rate', 0) * 0.2
            )
            
            model_scores[model_name] = {
                'composite_score': composite_score,
                'avg_time': performance.get('avg_processing_time', float('inf')),
                'confidence': quality.get('avg_confidence', 0),
                'layout_quality': quality.get('avg_layout_quality', 0)
            }
        
        if model_scores:
            # Détermination des meilleurs modèles
            best_overall = max(model_scores.items(), key=lambda x: x[1]['composite_score'])
            fastest = min(model_scores.items(), key=lambda x: x[1]['avg_time'])
            most_accurate = max(model_scores.items(), key=lambda x: x[1]['confidence'])
            best_quality = max(model_scores.items(), key=lambda x: x[1]['layout_quality'])
            
            comparison['best_model'].update({
                'overall_performance': best_overall[0],
                'fastest': fastest[0],
                'most_accurate': most_accurate[0],
                'best_quality': best_quality[0]
            })
            
            # Classements
            comparison['rankings'] = {
                'by_composite_score': sorted(model_scores.items(), 
                                           key=lambda x: x[1]['composite_score'], reverse=True),
                'by_speed': sorted(model_scores.items(), key=lambda x: x[1]['avg_time']),
                'by_accuracy': sorted(model_scores.items(), 
                                    key=lambda x: x[1]['confidence'], reverse=True)
            }
        
        return comparison
    
    def _calculate_benchmark_averages(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Calcule les moyennes de benchmark"""
        if not performance_data:
            return {}
        
        total_times = [d['total_time'] for d in performance_data]
        avg_times = [d['avg_time_per_image'] for d in performance_data]
        throughputs = [d['throughput'] for d in performance_data]
        
        return {
            'avg_total_time': statistics.mean(total_times),
            'avg_time_per_image': statistics.mean(avg_times),
            'avg_throughput': statistics.mean(throughputs),
            'std_time_per_image': statistics.stdev(avg_times) if len(avg_times) > 1 else 0
        }
    
    def _generate_evaluation_visualizations(self, evaluation_report: Dict[str, Any]):
        """Génère les visualisations d'évaluation"""
        try:
            metrics = evaluation_report.get('metrics', {})
            
            # Graphique des performances
            plt.figure(figsize=(12, 8))
            
            # Temps de traitement
            plt.subplot(2, 2, 1)
            performance = metrics.get('performance', {})
            times = [performance.get('min_processing_time', 0),
                    performance.get('avg_processing_time', 0),
                    performance.get('max_processing_time', 0)]
            labels = ['Min', 'Avg', 'Max']
            plt.bar(labels, times)
            plt.title('Temps de traitement (secondes)')
            plt.ylabel('Secondes')
            
            # Qualité
            plt.subplot(2, 2, 2)
            quality = metrics.get('quality', {})
            quality_metrics = [
                quality.get('avg_confidence', 0),
                quality.get('avg_layout_quality', 0),
                quality.get('success_rate', 0)
            ]
            quality_labels = ['Confiance', 'Qualité Layout', 'Taux Succès']
            plt.bar(quality_labels, quality_metrics)
            plt.title('Métriques de qualité')
            plt.ylabel('Score (0-1)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarde
            viz_file = os.path.join(self.output_dir, f"evaluation_viz_{int(time.time())}.png")
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Visualisation générée: {viz_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de visualisations: {e}")
    
    def _generate_comparison_charts(self, comparison_results: Dict[str, Any]):
        """Génère les graphiques de comparaison de modèles"""
        try:
            results_by_model = comparison_results.get('results_by_model', {})
            
            if len(results_by_model) < 2:
                return
            
            # Préparation des données
            models = []
            avg_times = []
            confidences = []
            
            for model_name, model_data in results_by_model.items():
                if 'error' in model_data:
                    continue
                
                models.append(model_name)
                metrics = model_data.get('metrics', {})
                avg_times.append(metrics.get('performance', {}).get('avg_processing_time', 0))
                confidences.append(metrics.get('quality', {}).get('avg_confidence', 0))
            
            if len(models) < 2:
                return
            
            # Graphique de comparaison
            plt.figure(figsize=(14, 6))
            
            # Temps de traitement
            plt.subplot(1, 2, 1)
            plt.bar(models, avg_times)
            plt.title('Temps de traitement moyen par modèle')
            plt.ylabel('Secondes')
            plt.xticks(rotation=45)
            
            # Confiance
            plt.subplot(1, 2, 2)
            plt.bar(models, confidences)
            plt.title('Confiance moyenne par modèle')
            plt.ylabel('Score de confiance')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarde
            comp_file = os.path.join(self.output_dir, f"model_comparison_{int(time.time())}.png")
            plt.savefig(comp_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Comparaison générée: {comp_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de comparaison: {e}")

def main():
    """Exemple d'utilisation du système d'évaluation"""
    evaluator = VLMEvaluator()
    
    # Évaluation simple
    results = evaluator.evaluate_dataset("../Data/processed_images")
    print(f"Résultats d'évaluation: {results['metrics']}")
    
    # Benchmark de performance
    benchmark = evaluator.benchmark_performance("../Data/processed_images", iterations=2)
    print(f"Benchmark: {benchmark['average_metrics']}")

if __name__ == "__main__":
    main()