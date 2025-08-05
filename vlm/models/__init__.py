"""
Adaptateurs de modèles VLM pour FacturAI

Ce module contient les adaptateurs pour différents modèles VLM:
- BLIP-2: Modèle vision-langage de Salesforce
- LLaVA: Large Language and Vision Assistant
- Qwen-VL: Modèle multimodal de Qwen
"""

from .model_adapter import ModelAdapter
from .blip2_adapter import BLIP2Adapter
from .llava_adapter import LLaVAAdapter
from .qwen_vl_adapter import QwenVLAdapter

__all__ = [
    "ModelAdapter",
    "BLIP2Adapter", 
    "LLaVAAdapter",
    "QwenVLAdapter"
]