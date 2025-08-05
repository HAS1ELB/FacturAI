#!/usr/bin/env python3
"""
Test rapide du module VLM FacturAI
"""

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image

# Ajouter le chemin vers le module parent
sys.path.append(str(Path(__file__).parent.parent))

def create_test_image(path: str) -> str:
    """Crée une image de test simple"""
    # Création d'une image factice avec du texte simulé
    img = Image.new('RGB', (800, 600), color='white')
    img.save(path)
    return path

def test_vlm_import():
    """Test l'importation du module VLM"""
    print("🔍 Test d'importation du module VLM...")
    
    try:
        from vlm import VLMProcessor
        from vlm.utils import VLMVisualizer, ZoneDetector, LayoutAnalyzer
        from vlm.config import vlm_config
        print("✅ Importations réussies")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        return False

def test_vlm_initialization():
    """Test l'initialisation du processeur VLM"""
    print("\n🚀 Test d'initialisation du processeur VLM...")
    
    try:
        from vlm import VLMProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VLMProcessor(output_dir=temp_dir)
            
            print(f"  Modèles disponibles: {processor.available_models}")
            
            model_info = processor.get_model_info()
            print(f"  Modèle chargé: {model_info['is_loaded']}")
            
            if processor.available_models:
                print(f"  Modèle actuel: {model_info.get('model_name', 'Aucun')}")
            
        print("✅ Initialisation réussie")
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def test_vlm_processing():
    """Test le traitement d'une image factice"""
    print("\n📄 Test de traitement d'image...")
    
    try:
        from vlm import VLMProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Création d'une image de test
            test_image_path = os.path.join(temp_dir, "test_invoice.jpg")
            create_test_image(test_image_path)
            
            # Initialisation du processeur
            processor = VLMProcessor(output_dir=temp_dir)
            
            # Traitement de l'image
            print("  Traitement en cours...")
            result = processor.process_invoice(test_image_path)
            
            # Vérification des résultats
            required_keys = ['image_path', 'vlm_analysis', 'detected_zones', 'layout_analysis']
            
            for key in required_keys:
                if key in result:
                    print(f"  ✅ {key}: présent")
                else:
                    print(f"  ❌ {key}: manquant")
                    return False
            
            # Affichage de quelques métriques
            processing_time = result.get('processing_time', 0)
            model_used = result.get('model_used', 'N/A')
            
            print(f"  ⏱️  Temps de traitement: {processing_time:.2f}s")
            print(f"  🤖 Modèle utilisé: {model_used}")
            
            # Vérification des zones détectées
            zones = result.get('detected_zones', {})
            zones_count = 0
            
            if zones.get('header', {}).get('detected', False):
                zones_count += 1
            if zones.get('footer', {}).get('detected', False):
                zones_count += 1
            zones_count += len(zones.get('tables', []))
            zones_count += len(zones.get('address_blocks', []))
            zones_count += len(zones.get('amount_zones', []))
            
            print(f"  🎯 Zones détectées: {zones_count}")
            
        print("✅ Traitement réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de traitement: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlm_utilities():
    """Test les utilitaires VLM"""
    print("\n🛠️  Test des utilitaires VLM...")
    
    try:
        from vlm.utils import VLMVisualizer, ZoneDetector, LayoutAnalyzer, GeometryUtils
        
        # Test du visualiseur
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = VLMVisualizer(output_dir=temp_dir)
            print("  ✅ VLMVisualizer: initialisé")
        
        # Test du détecteur de zones
        zone_config = {
            "header_keywords": ["facture", "invoice"],
            "footer_keywords": ["total", "tva"],
            "confidence_threshold": 0.3
        }
        detector = ZoneDetector(zone_config)
        print("  ✅ ZoneDetector: initialisé")
        
        # Test de l'analyseur de mise en page
        layout_config = {
            "grid_detection": True,
            "table_detection": True,
            "text_block_detection": True
        }
        analyzer = LayoutAnalyzer(layout_config)
        print("  ✅ LayoutAnalyzer: initialisé")
        
        # Test des utilitaires géométriques
        distance = GeometryUtils.calculate_distance((0, 0), (3, 4))
        expected_distance = 5.0
        if abs(distance - expected_distance) < 0.001:
            print("  ✅ GeometryUtils: fonctionnel")
        else:
            print(f"  ❌ GeometryUtils: erreur de calcul ({distance} != {expected_distance})")
            return False
        
        print("✅ Utilitaires fonctionnels")
        return True
        
    except Exception as e:
        print(f"❌ Erreur des utilitaires: {e}")
        return False

def test_vlm_config():
    """Test de la configuration VLM"""
    print("\n⚙️  Test de la configuration VLM...")
    
    try:
        from vlm.config import vlm_config
        
        # Test des méthodes de configuration
        zone_config = vlm_config.get_zone_detection_config()
        layout_config = vlm_config.get_layout_analysis_config()
        processing_config = vlm_config.get_processing_config()
        
        print("  ✅ Configuration de détection de zones: accessible")
        print("  ✅ Configuration d'analyse de mise en page: accessible")
        print("  ✅ Configuration de traitement: accessible")
        
        # Test des modèles activés
        enabled_models = vlm_config.get_enabled_models()
        print(f"  ✅ Modèles activés: {enabled_models}")
        
        print("✅ Configuration fonctionnelle")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False

def test_vlm_batch_processing():
    """Test du traitement par lots"""
    print("\n📁 Test du traitement par lots...")
    
    try:
        from vlm import VLMProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Création de plusieurs images de test
            image_paths = []
            for i in range(3):
                image_path = os.path.join(temp_dir, f"test_invoice_{i}.jpg")
                create_test_image(image_path)
                image_paths.append(image_path)
            
            # Traitement par lots
            processor = VLMProcessor(output_dir=temp_dir)
            results = processor.batch_process(image_paths)
            
            # Vérifications
            if len(results) == len(image_paths):
                print(f"  ✅ {len(results)} images traitées")
            else:
                print(f"  ❌ Erreur: {len(results)}/{len(image_paths)} images traitées")
                return False
            
            # Calcul des statistiques
            processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                print(f"  ⏱️  Temps moyen: {avg_time:.2f}s")
            
        print("✅ Traitement par lots réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur de traitement par lots: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🧪 TEST RAPIDE DU MODULE VLM FACTURAI")
    print("=" * 60)
    
    tests = [
        ("Importation", test_vlm_import),
        ("Initialisation", test_vlm_initialization),
        ("Configuration", test_vlm_config),
        ("Utilitaires", test_vlm_utilities),
        ("Traitement d'image", test_vlm_processing),
        ("Traitement par lots", test_vlm_batch_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erreur inattendue dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print("\n" + "-" * 60)
    print(f"Tests réussis: {passed}/{total}")
    print(f"Taux de réussite: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés! Le module VLM est prêt à l'utilisation.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)