"""
Tests unitaires pour le processeur VLM principal
"""

import unittest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vlm.vlm_processor import VLMProcessor
from vlm.config import VLMConfig

class TestVLMProcessor(unittest.TestCase):
    """Tests pour la classe VLMProcessor"""
    
    def setUp(self):
        """Configuration des tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = VLMProcessor(output_dir=self.temp_dir)
        
        # Création d'une image de test
        self.test_image = Image.new('RGB', (800, 600), color='white')
        self.test_image_path = os.path.join(self.temp_dir, "test_invoice.jpg")
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Nettoyage après les tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test l'initialisation du processeur"""
        self.assertIsNotNone(self.processor.config)
        self.assertIsNotNone(self.processor.zone_detector)
        self.assertIsNotNone(self.processor.layout_analyzer)
        self.assertEqual(self.processor.output_dir, self.temp_dir)
    
    def test_check_available_models(self):
        """Test la vérification des modèles disponibles"""
        available_models = self.processor._check_available_models()
        self.assertIsInstance(available_models, list)
        # Au minimum, l'adaptateur factice devrait être disponible
        # En production, au moins un vrai modèle devrait être disponible
    
    def test_model_info(self):
        """Test les informations sur le modèle"""
        model_info = self.processor.get_model_info()
        self.assertIn('is_loaded', model_info)
        self.assertIn('available_models', model_info)
    
    @patch('vlm.models.model_adapter.create_adapter')
    def test_load_model(self, mock_create_adapter):
        """Test le chargement d'un modèle"""
        # Mock de l'adaptateur
        mock_adapter = Mock()
        mock_create_adapter.return_value = mock_adapter
        
        # Ajout d'un modèle factice à la configuration
        self.processor.available_models = ['test_model']
        
        # Test du chargement
        self.processor.load_model('test_model')
        
        # Vérifications
        mock_create_adapter.assert_called_once()
        self.assertEqual(self.processor.model_adapter, mock_adapter)
    
    def test_load_invalid_model(self):
        """Test le chargement d'un modèle invalide"""
        with self.assertRaises(ValueError):
            self.processor.load_model('invalid_model')
    
    @patch('vlm.vlm_processor.VLMProcessor._perform_vlm_analysis')
    @patch('vlm.utils.zone_detector.ZoneDetector.detect_zones')
    @patch('vlm.utils.layout_analyzer.LayoutAnalyzer.analyze_layout')
    def test_process_invoice(self, mock_layout, mock_zones, mock_vlm):
        """Test le traitement d'une facture"""
        # Configuration des mocks
        mock_vlm.return_value = {
            'basic_description': 'Test invoice description',
            'confidence': 0.85,
            'detailed_analysis': {}
        }
        
        mock_zones.return_value = {
            'header': {'detected': True, 'confidence': 0.9},
            'footer': {'detected': True, 'confidence': 0.8},
            'tables': [],
            'address_blocks': [],
            'amount_zones': []
        }
        
        mock_layout.return_value = {
            'document_structure': {'type': 'invoice'},
            'layout_quality': {'overall_score': 0.85}
        }
        
        # Mock de l'adaptateur de modèle
        mock_adapter = Mock()
        self.processor.model_adapter = mock_adapter
        
        # Test du traitement
        result = self.processor.process_invoice(self.test_image_path)
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('image_path', result)
        self.assertIn('vlm_analysis', result)
        self.assertIn('detected_zones', result)
        self.assertIn('layout_analysis', result)
        self.assertIn('processing_time', result)
        
        # Vérification des appels de méthodes
        mock_vlm.assert_called_once()
        mock_zones.assert_called_once()
        mock_layout.assert_called_once()
    
    def test_process_nonexistent_image(self):
        """Test le traitement d'une image inexistante"""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_invoice("nonexistent_image.jpg")
    
    def test_process_without_model(self):
        """Test le traitement sans modèle chargé"""
        # Suppression du modèle
        self.processor.model_adapter = None
        
        with self.assertRaises(RuntimeError):
            self.processor.process_invoice(self.test_image_path)
    
    @patch('vlm.vlm_processor.VLMProcessor.process_invoice')
    def test_batch_process(self, mock_process):
        """Test le traitement par lots"""
        # Configuration du mock
        mock_process.return_value = {
            'image_path': 'test.jpg',
            'processing_time': 1.5,
            'vlm_analysis': {}
        }
        
        # Création d'images de test
        image_paths = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f"test_{i}.jpg")
            self.test_image.save(img_path)
            image_paths.append(img_path)
        
        # Test du traitement par lots
        results = self.processor.batch_process(image_paths)
        
        # Vérifications
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_process.call_count, 3)
        
        for result in results:
            self.assertIn('image_path', result)
            self.assertIn('processing_time', result)
    
    def test_load_and_preprocess_image(self):
        """Test le chargement et prétraitement d'image"""
        # Test avec image normale
        processed_image = self.processor._load_and_preprocess_image(self.test_image_path)
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.mode, 'RGB')
        
        # Test avec image trop grande
        large_image = Image.new('RGB', (2000, 2000), color='white')
        large_image_path = os.path.join(self.temp_dir, "large_test.jpg")
        large_image.save(large_image_path)
        
        processed_large = self.processor._load_and_preprocess_image(large_image_path)
        # L'image devrait être redimensionnée
        self.assertTrue(processed_large.size[0] <= 1024 and processed_large.size[1] <= 1024)
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_save_results_error(self, mock_open):
        """Test la gestion d'erreur lors de la sauvegarde"""
        results = {
            'image_path': self.test_image_path,
            'processing_time': 1.0,
            'vlm_analysis': {}
        }
        
        # La méthode ne devrait pas lever d'exception même si la sauvegarde échoue
        try:
            self.processor._save_results(results, self.test_image_path)
        except Exception as e:
            self.fail(f"_save_results a levé une exception: {e}")

class TestVLMConfig(unittest.TestCase):
    """Tests pour la configuration VLM"""
    
    def setUp(self):
        """Configuration des tests"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Création d'un fichier de configuration de test
        self.test_config = {
            "vlm_models": {
                "test_model": {
                    "model_name": "test/model",
                    "enabled": True,
                    "device": "cpu",
                    "confidence_threshold": 0.5
                }
            },
            "zone_detection": {
                "header_keywords": ["test", "header"],
                "confidence_threshold": 0.3
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Nettoyage après les tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config(self):
        """Test le chargement de configuration"""
        config = VLMConfig(self.config_file)
        self.assertEqual(config.config, self.test_config)
    
    def test_get_model_config(self):
        """Test la récupération de configuration de modèle"""
        config = VLMConfig(self.config_file)
        model_config = config.get_model_config("test_model")
        
        self.assertEqual(model_config["model_name"], "test/model")
        self.assertTrue(model_config["enabled"])
        self.assertEqual(model_config["device"], "cpu")
    
    def test_get_nonexistent_model_config(self):
        """Test la récupération d'une configuration de modèle inexistant"""
        config = VLMConfig(self.config_file)
        
        with self.assertRaises(ValueError):
            config.get_model_config("nonexistent_model")
    
    def test_get_zone_detection_config(self):
        """Test la récupération de configuration de détection de zones"""
        config = VLMConfig(self.config_file)
        zone_config = config.get_zone_detection_config()
        
        self.assertIn("header_keywords", zone_config)
        self.assertEqual(zone_config["confidence_threshold"], 0.3)
    
    def test_get_enabled_models(self):
        """Test la récupération des modèles activés"""
        config = VLMConfig(self.config_file)
        enabled_models = config.get_enabled_models()
        
        self.assertIn("test_model", enabled_models)
    
    def test_update_config(self):
        """Test la mise à jour de configuration"""
        config = VLMConfig(self.config_file)
        
        # Mise à jour d'une valeur
        config.update_config("zone_detection.confidence_threshold", 0.4)
        
        # Vérification
        zone_config = config.get_zone_detection_config()
        self.assertEqual(zone_config["confidence_threshold"], 0.4)
    
    def test_load_invalid_config(self):
        """Test le chargement d'une configuration invalide"""
        # Fichier inexistant
        with self.assertRaises(FileNotFoundError):
            VLMConfig("nonexistent_config.json")
        
        # JSON invalide
        invalid_config_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_config_file, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(ValueError):
            VLMConfig(invalid_config_file)

if __name__ == '__main__':
    # Configuration du logging pour les tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Exécution des tests
    unittest.main()