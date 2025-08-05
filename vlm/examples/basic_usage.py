"""
Exemple d'utilisation de base du module VLM FacturAI
"""

import os
import sys
import json
from pathlib import Path

# Ajouter le chemin vers le module parent
sys.path.append(str(Path(__file__).parent.parent.parent))

from vlm import VLMProcessor
from vlm.utils import VLMVisualizer

def main():
    """Exemple d'utilisation complète du module VLM"""
    
    print("=== Exemple d'utilisation du module VLM FacturAI ===\n")
    
    # Chemins vers les données de test
    images_dir = "../Data/processed_images"
    ocr_results_dir = "../Data/ocr_results"
    
    # Vérification de l'existence des répertoires
    if not os.path.exists(images_dir):
        print(f"⚠️  Répertoire d'images non trouvé: {images_dir}")
        print("Veuillez vous assurer que les images sont présentes dans Data/processed_images")
        return
    
    # Initialisation du processeur VLM
    print("🚀 Initialisation du processeur VLM...")
    try:
        processor = VLMProcessor()
        print(f"✅ Processeur initialisé avec succès")
        print(f"Modèles disponibles: {processor.available_models}")
        
        # Informations sur le modèle actuel
        model_info = processor.get_model_info()
        print(f"Modèle actuel: {model_info['model_name']}")
        print(f"Modèle chargé: {model_info['is_loaded']}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        print("Utilisation du mode démo avec adaptateur factice...")
        processor = VLMProcessor()
    
    print("\n" + "="*50)
    
    # Recherche d'images à traiter
    image_files = []
    if os.path.exists(images_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(images_dir).glob(ext))
    
    if not image_files:
        print("⚠️  Aucune image trouvée pour le traitement")
        return
    
    # Traitement d'une image d'exemple
    test_image = str(image_files[0])
    print(f"📄 Traitement de l'image: {os.path.basename(test_image)}")
    
    # Chargement des résultats OCR correspondants si disponibles
    ocr_results = None
    image_name = Path(test_image).stem
    ocr_file = os.path.join(ocr_results_dir, f"enhanced_{image_name}_ocr.json")
    
    if os.path.exists(ocr_file):
        print(f"📖 Chargement des résultats OCR: {os.path.basename(ocr_file)}")
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_results = json.load(f)
            print(f"✅ Résultats OCR chargés ({len(ocr_results.get('results', []))} éléments)")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement OCR: {e}")
            ocr_results = None
    else:
        print("ℹ️  Aucun résultat OCR correspondant trouvé")
    
    # Traitement VLM
    print("\n🔍 Analyse VLM en cours...")
    try:
        results = processor.process_invoice(test_image, ocr_results)
        
        print(f"✅ Analyse terminée en {results['processing_time']:.2f}s")
        print(f"Modèle utilisé: {results['model_used']}")
        
        # Affichage des résultats
        print("\n📊 Résultats de l'analyse:")
        
        # Analyse VLM de base
        vlm_analysis = results.get('vlm_analysis', {})
        print(f"Description: {vlm_analysis.get('basic_description', 'N/A')[:100]}...")
        print(f"Confiance: {vlm_analysis.get('confidence', 0):.2f}")
        
        # Zones détectées
        zones = results.get('detected_zones', {})
        print(f"\n🎯 Zones détectées:")
        
        # En-tête
        header = zones.get('header', {})
        print(f"  En-tête: {'✅' if header.get('detected', False) else '❌'} "
              f"(confiance: {header.get('confidence', 0):.2f})")
        
        # Pied de page
        footer = zones.get('footer', {})
        print(f"  Pied de page: {'✅' if footer.get('detected', False) else '❌'} "
              f"(confiance: {footer.get('confidence', 0):.2f})")
        
        # Tableaux
        tables = zones.get('tables', [])
        print(f"  Tableaux: {len(tables)} détecté(s)")
        for i, table in enumerate(tables):
            print(f"    Table {i+1}: {table.get('rows_count', 'N/A')} lignes, "
                  f"{len(table.get('columns', []))} colonnes")
        
        # Adresses
        addresses = zones.get('address_blocks', [])
        print(f"  Adresses: {len(addresses)} détectée(s)")
        for i, addr in enumerate(addresses):
            addr_type = addr.get('type', 'unknown')
            print(f"    Adresse {i+1}: {addr_type}")
        
        # Montants
        amounts = zones.get('amount_zones', [])
        print(f"  Montants: {len(amounts)} détecté(s)")
        for i, amount in enumerate(amounts):
            value = amount.get('value', 'N/A')
            currency = amount.get('currency', '')
            amount_type = amount.get('type', 'unknown')
            print(f"    Montant {i+1}: {value} {currency} ({amount_type})")
        
        # Analyse de mise en page
        layout = results.get('layout_analysis', {})
        if layout:
            quality = layout.get('layout_quality', {})
            print(f"\n📐 Qualité de mise en page:")
            print(f"  Score global: {quality.get('overall_score', 0):.2f}/1.0")
            print(f"  Clarté: {quality.get('clarity', 0):.2f}")
            print(f"  Organisation: {quality.get('organization', 0):.2f}")
            print(f"  Complétude: {quality.get('completeness', 0):.2f}")
        
        # Génération de visualisations
        print("\n🎨 Génération des visualisations...")
        visualizer = VLMVisualizer()
        
        # Image annotée
        annotated_path = visualizer.visualize_analysis_results(test_image, results)
        if annotated_path:
            print(f"✅ Image annotée: {annotated_path}")
        
        # Rapport textuel
        report_path = visualizer.generate_analysis_report(results)
        if report_path:
            print(f"✅ Rapport généré: {report_path}")
        
        # Export JSON des zones
        zones_path = visualizer.export_zones_to_json(results)
        if zones_path:
            print(f"✅ Zones exportées: {zones_path}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse VLM: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    
    # Traitement par lots (si plusieurs images)
    if len(image_files) > 1:
        print(f"📁 Traitement par lots de {len(image_files)} images...")
        
        try:
            # Limiter à 3 images pour l'exemple
            batch_images = [str(img) for img in image_files[:3]]
            batch_results = processor.batch_process(batch_images, ocr_results_dir)
            
            print(f"✅ Traitement par lots terminé: {len(batch_results)} résultats")
            
            # Statistiques du batch
            total_time = sum(r.get('processing_time', 0) for r in batch_results)
            avg_time = total_time / len(batch_results)
            
            print(f"📊 Statistiques:")
            print(f"  Temps total: {total_time:.2f}s")
            print(f"  Temps moyen: {avg_time:.2f}s")
            
            # Génération d'une comparaison
            comparison_path = visualizer.create_comparison_visualization(batch_results)
            if comparison_path:
                print(f"✅ Comparaison générée: {comparison_path}")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement par lots: {e}")
    
    print("\n🎉 Exemple terminé avec succès!")
    print("Consultez le dossier Data/vlm_results pour voir tous les fichiers générés.")

def demo_model_switching():
    """Démo du changement de modèles"""
    print("\n=== Démo: Changement de modèles ===")
    
    processor = VLMProcessor()
    
    print(f"Modèles disponibles: {processor.available_models}")
    
    for model_name in processor.available_models:
        print(f"\n📝 Test du modèle: {model_name}")
        try:
            processor.load_model(model_name)
            info = processor.get_model_info()
            print(f"  ✅ Modèle chargé: {info['model_name']}")
            print(f"  Device: {info.get('device', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")

if __name__ == "__main__":
    main()
    
    # Démo optionnelle du changement de modèles
    demo_choice = input("\nVoulez-vous tester le changement de modèles? (y/N): ")
    if demo_choice.lower() == 'y':
        demo_model_switching()