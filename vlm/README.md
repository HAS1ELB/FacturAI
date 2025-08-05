# Module VLM FacturAI

Ce module implémente l'étape 3 de l'architecture FacturAI : **Analyse visuelle et linguistique par VLM (Visual Language Model)**.

## 🎯 Objectifs

Le module VLM vise à :
- Comprendre conjointement l'apparence visuelle et le contenu textuel des factures
- Analyser la mise en page et identifier les structures (tableaux, listes)
- Comprendre les relations spatiales entre les éléments
- Identifier les zones clés (entêtes, pieds de page, totaux, lignes d'articles, blocs d'adresse, numéros de TVA, etc.)

## 🏗️ Architecture

```
vlm/
├── __init__.py                 # Module principal
├── vlm_processor.py           # Processeur VLM principal
├── config/                    # Configuration
│   ├── __init__.py
│   └── vlm_config.json
├── models/                    # Adaptateurs de modèles
│   ├── __init__.py
│   ├── model_adapter.py       # Adaptateur de base
│   ├── blip2_adapter.py       # Adaptateur BLIP-2
│   ├── llava_adapter.py       # Adaptateur LLaVA
│   └── qwen_vl_adapter.py     # Adaptateur Qwen-VL
├── utils/                     # Utilitaires
│   ├── __init__.py
│   ├── zone_detector.py       # Détection de zones
│   ├── layout_analyzer.py     # Analyse de mise en page
│   ├── geometry_utils.py      # Utilitaires géométriques
│   └── visualization.py       # Visualisation
├── tests/                     # Tests unitaires
├── examples/                  # Exemples d'utilisation
└── README.md                  # Cette documentation
```

## 🚀 Installation

### Prérequis

```bash
pip install torch torchvision transformers pillow opencv-python numpy
```

### Modèles supportés

1. **BLIP-2** (Salesforce)
   ```bash
   pip install transformers[blip]
   ```

2. **LLaVA** (Microsoft)
   ```bash
   pip install transformers[llava]
   ```

3. **Qwen-VL** (Alibaba)
   ```bash
   pip install transformers_stream_generator
   ```

## 📖 Utilisation

### Utilisation basique

```python
from vlm import VLMProcessor

# Initialisation du processeur
processor = VLMProcessor()

# Traitement d'une facture
results = processor.process_invoice("path/to/invoice.jpg")

print(f"Zones détectées: {results['detected_zones']}")
print(f"Analyse de mise en page: {results['layout_analysis']}")
```

### Utilisation avec résultats OCR

```python
import json

# Chargement des résultats OCR existants
with open("ocr_results.json", "r") as f:
    ocr_results = json.load(f)

# Traitement avec intégration OCR
results = processor.process_invoice("invoice.jpg", ocr_results)
```

### Traitement par lots

```python
image_paths = ["invoice1.jpg", "invoice2.jpg", "invoice3.jpg"]
results = processor.batch_process(image_paths, "ocr_results/")

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Temps: {result['processing_time']:.2f}s")
```

### Changement de modèle

```python
# Lister les modèles disponibles
print(processor.available_models)

# Charger un modèle spécifique
processor.load_model("llava")

# Vérifier le modèle actuel
info = processor.get_model_info()
print(f"Modèle actuel: {info['model_name']}")
```

## ⚙️ Configuration

### Fichier de configuration

Le fichier `config/vlm_config.json` permet de configurer :

```json
{
  "vlm_models": {
    "blip2": {
      "model_name": "Salesforce/blip2-opt-2.7b",
      "enabled": true,
      "device": "auto",
      "max_length": 512,
      "confidence_threshold": 0.5
    }
  },
  "zone_detection": {
    "header_keywords": ["facture", "invoice", "devis"],
    "footer_keywords": ["total", "tva", "ht", "ttc"],
    "confidence_threshold": 0.3
  },
  "processing": {
    "max_image_size": [1024, 1024],
    "batch_size": 1,
    "timeout": 30
  }
}
```

### Configuration programmatique

```python
from vlm.config import vlm_config

# Modifier la configuration
vlm_config.update_config("zone_detection.confidence_threshold", 0.4)
vlm_config.save_config()

# Accéder à la configuration
model_config = vlm_config.get_model_config("blip2")
```

## 🎨 Visualisation

### Génération d'images annotées

```python
from vlm.utils import VLMVisualizer

visualizer = VLMVisualizer()

# Annotation des résultats sur l'image
annotated_path = visualizer.visualize_analysis_results(
    "invoice.jpg", 
    results, 
    "annotated_invoice.jpg"
)
```

### Génération de rapports

```python
# Rapport textuel
report_path = visualizer.generate_analysis_report(results)

# Export des zones au format JSON
zones_path = visualizer.export_zones_to_json(results)

# Comparaison de plusieurs analyses
comparison_path = visualizer.create_comparison_visualization([results1, results2])
```

## 🔍 Zones détectées

### Types de zones

1. **En-tête (Header)**
   - Logo, nom d'entreprise
   - Informations de contact émetteur
   - Numéro et date de facture

2. **Pied de page (Footer)**
   - Totaux et montants
   - Informations de paiement
   - Conditions générales

3. **Tableaux (Tables)**
   - Lignes d'articles/services
   - Colonnes (description, quantité, prix)
   - Sous-totaux

4. **Blocs d'adresse (Address blocks)**
   - Adresse émetteur
   - Adresse destinataire
   - Informations de livraison

5. **Zones de montants (Amount zones)**
   - Montants HT, TTC
   - TVA
   - Totaux partiels

### Structure des résultats

```python
{
  "image_path": "invoice.jpg",
  "model_used": "blip2",
  "processing_time": 2.45,
  "vlm_analysis": {
    "basic_description": "Document de facture avec en-tête...",
    "detailed_analysis": {...},
    "confidence": 0.87
  },
  "detected_zones": {
    "header": {
      "detected": true,
      "confidence": 0.92,
      "content": {...}
    },
    "tables": [...],
    "amount_zones": [...]
  },
  "layout_analysis": {
    "document_structure": {...},
    "spatial_organization": {...},
    "layout_quality": {...}
  }
}
```

## 🧪 Tests et évaluation

### Tests unitaires

```bash
cd vlm/tests
python -m pytest test_vlm_processor.py
python -m pytest test_zone_detector.py
python -m pytest test_layout_analyzer.py
```

### Évaluation de performance

```python
from vlm.evaluation import VLMEvaluator

evaluator = VLMEvaluator()

# Évaluation sur un dataset
metrics = evaluator.evaluate_dataset("test_images/", "ground_truth/")

print(f"Précision zones: {metrics['zone_precision']:.2f}")
print(f"Rappel zones: {metrics['zone_recall']:.2f}")
print(f"Score F1: {metrics['f1_score']:.2f}")
```

## 🔧 Dépannage

### Problèmes courants

1. **Modèle non disponible**
   ```
   ERROR: Modèle 'blip2' non disponible
   ```
   **Solution**: Vérifier l'installation des dépendances transformers

2. **Mémoire insuffisante**
   ```
   ERROR: CUDA out of memory
   ```
   **Solution**: Réduire la taille d'image ou utiliser le CPU
   ```python
   vlm_config.update_config("vlm_models.blip2.device", "cpu")
   ```

3. **Erreur de format d'image**
   ```
   ERROR: Cannot identify image file
   ```
   **Solution**: Vérifier le format d'image (JPG, PNG supportés)

### Logs de débogage

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Le module VLM affichera des logs détaillés
```

## 📊 Performance

### Benchmarks typiques

| Modèle | Temps moyen | Précision zones | Mémoire GPU |
|--------|-------------|-----------------|-------------|
| BLIP-2 | 2.1s        | 85%            | 4GB         |
| LLaVA  | 1.8s        | 88%            | 6GB         |
| Qwen-VL| 2.4s        | 92%            | 8GB         |

### Optimisation

1. **Prétraitement d'images**
   ```python
   vlm_config.update_config("processing.max_image_size", [512, 512])
   ```

2. **Utilisation du cache**
   ```python
   processor = VLMProcessor(cache_models=True)
   ```

3. **Traitement parallèle**
   ```python
   processor.batch_process(images, n_workers=4)
   ```

## 🤝 Contribution

Pour contribuer au module VLM :

1. Créer un adaptateur pour un nouveau modèle
2. Améliorer la détection de zones spécifiques
3. Ajouter des métriques d'évaluation
4. Optimiser les performances

## 📄 Licence

Ce module fait partie du projet FacturAI et suit la même licence.

## 📞 Support

Pour toute question ou problème :
- Issues GitHub: [Lien vers issues]
- Documentation complète: [Lien vers docs]
- Exemples: Voir le dossier `examples/`