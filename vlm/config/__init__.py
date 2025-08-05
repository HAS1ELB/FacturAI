"""
Configuration du module VLM de FacturAI
"""

import json
import os
from typing import Dict, Any, List
from pathlib import Path

class VLMConfig:
    """Gestionnaire de configuration pour le module VLM"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "vlm_config.json")
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de configuration non trouvé : {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur de format JSON dans {self.config_path}: {e}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Retourne la configuration d'un modèle spécifique"""
        models = self.config.get("vlm_models", {})
        if model_name not in models:
            raise ValueError(f"Modèle '{model_name}' non configuré. Modèles disponibles: {list(models.keys())}")
        return models[model_name]
    
    def get_zone_detection_config(self) -> Dict[str, Any]:
        """Retourne la configuration de détection de zones"""
        return self.config.get("zone_detection", {})
    
    def get_layout_analysis_config(self) -> Dict[str, Any]:
        """Retourne la configuration d'analyse de mise en page"""
        return self.config.get("layout_analysis", {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Retourne la configuration de traitement"""
        return self.config.get("processing", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Retourne la configuration de sortie"""
        return self.config.get("output", {})
    
    def get_enabled_models(self) -> List[str]:
        """Retourne la liste des modèles activés"""
        models = self.config.get("vlm_models", {})
        return [name for name, config in models.items() if config.get("enabled", False)]
    
    def update_config(self, key: str, value: Any):
        """Met à jour une valeur de configuration"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save_config(self):
        """Sauvegarde la configuration modifiée"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

# Instance globale de configuration
vlm_config = VLMConfig()