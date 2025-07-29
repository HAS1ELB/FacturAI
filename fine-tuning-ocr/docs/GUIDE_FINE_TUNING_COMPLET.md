# 🎯 Guide Complet de Fine-Tuning OCR pour FacturAI

## 📋 Vue d'ensemble

Ce guide vous accompagne dans l'utilisation du système complet de fine-tuning OCR développé pour FacturAI. Le système supporte **3 approches principales** :

1. **🤖 TrOCR** - Moderne, basé sur Transformers (Recommandé)
2. **👁️ EasyOCR** - Extension du modèle existant

## 🚀 Installation et Configuration

### 1. Installation des Dépendances

```bash
# Installation automatique de toutes les dépendances
python install_fine_tuning_deps.py

# Ou installation manuelle
pip install torch torchvision transformers
pip install easyocr paddleocr
pip install opencv-python pillow matplotlib seaborn
pip install scikit-learn pandas numpy
pip install python-Levenshtein
```

### 2. Vérification de l'Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers OK')"
python -c "import easyocr; print('EasyOCR OK')"
```

## 📊 Préparation des Données

### 1. Structure des Dossiers Requise

```
FacturAI/
├── Data/
│   ├── processed_images/     # Images prétraitées
│   ├── ocr_results/         # Résultats OCR existants
│   └── fine_tuning/         # Données préparées (créé automatiquement)
├── models/                  # Modèles entraînés
└── evaluation_results/      # Résultats d'évaluation
```

### 2. Préparation Automatique des Données

```bash
# Préparer toutes les données à partir des résultats OCR existants
python fine-tuning-ocr/data_preparation/data_preparation.py \
    --images_dir Data/processed_images \
    --ocr_results_dir Data/ocr_results \
    --output_dir Data/fine_tuning
```

**Ce script va :**

- ✅ Analyser vos résultats OCR existants
- ✅ Générer des annotations de vérité terrain
- ✅ Classifier les types de texte (montants, dates, adresses, etc.)
- ✅ Créer les splits train/validation/test
- ✅ Préparer les formats pour chaque modèle
- ✅ Générer des statistiques détaillées

## 🎯 Fine-Tuning par Modèle

### 1. TrOCR (Recommandé) 🤖

**Pourquoi TrOCR ?**

- Architecture Transformer moderne
- Excellente performance sur documents complexes
- Pré-entraîné sur des millions de documents

```bash
# Fine-tuning TrOCR
python fine-tuning-ocr/fine_tuning_model/trocr_finetuning.py \
    --dataset Data/fine_tuning/datasets/trocr/dataset.json \
    --base_model microsoft/trocr-large-printed \
    --output_dir fine-tuning-ocr/models/trocr_finetuned \
    --epochs 10 \
    --batch_size 4 \
    --learning_rate 5e-5
```

**Configuration recommandée :**

- GPU : Au moins 8GB VRAM
- Epochs : 10-20
- Batch size : 4-8
- Learning rate : 5e-5

### 2. EasyOCR (Votre demande spécifique) 👁️

**Avantages :**

- Continuation de votre pipeline existant
- Modèle CRNN personnalisé
- Optimisé pour les caractères français

```bash
# Fine-tuning EasyOCR
python easyocr_finetuning.py \
    --dataset Data/fine_tuning/datasets/easyocr/dataset.json \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --output_dir models/easyocr_finetuned
```

**Configuration recommandée :**

- GPU : Au moins 4GB VRAM
- Epochs : 50-100
- Batch size : 8-16
- Learning rate : 0.001

### 🎛️ Gestionnaire Principal

### Utilisation du Gestionnaire Unifié

```bash
# Lancer tout le processus automatiquement
python fine_tuning_manager.py \
    --config fine_tuning_config.json \
    --mode train \
    --models trocr,easyocr
```

**Modes disponibles :**

- `prepare` : Préparation des données uniquement
- `train` : Entraînement des modèles
- `evaluate` : Évaluation des modèles
- `all` : Processus complet

## 📈 Évaluation et Comparaison

### Évaluation Complète

```bash
python model_evaluation.py \
    --test_data Data/fine_tuning/splits/test.json \
    --ground_truth Data/fine_tuning/annotations/ground_truth.json \
    --output_dir evaluation_results \
    --easyocr_model models/easyocr_finetuned/final_model.pth \
    --trocr_model models/trocr_finetuned \
```

**Métriques calculées :**

- 🎯 Similarité moyenne (Levenshtein)
- ✅ Accuracy exacte
- 🔍 Confiance moyenne
- ⏱️ Temps de traitement
- 📊 Distance d'édition

## 🏆 Résultats et Recommandations

### Métriques de Performance Attendues

| Modèle              | Similarité | Confiance | Vitesse |
| -------------------- | ----------- | --------- | ------- |
| **TrOCR**      | 85-95%      | 80-90%    | 2-3s    |
| **EasyOCR FT** | 80-90%      | 75-85%    | 1-2s    |

### Recommandations d'Usage

#### 🥇 Pour la Production

**TrOCR fine-tuné** - Meilleure précision globale

#### 🥈 Pour la Vitesse

**EasyOCR fine-tuné** - Bon compromis vitesse/précision

## 🔧 Configuration Avancée

### 1. Augmentation des Données

```python
# Dans data_preparation.py, personnaliser les transformations
transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])
```

### 2. Hyperparamètres TrOCR

```json
{
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "max_target_length": 128,
    "num_beams": 4,
    "early_stopping": true
}
```

### 3. Architecture EasyOCR Personnalisée

```python
# Modifier dans easyocr_finetuning.py
class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=512):  # Augmenter hidden_size
        # Ajouter plus de couches CNN
        # Utiliser attention mechanism
```

## 🎯 Optimisation pour Factures

### 1. Post-Processing Spécialisé

```python
def invoice_postprocess(text):
    """Post-processing spécialisé factures"""
    # Correction montants
    text = re.sub(r'(\d+)[,.](\d{2})\s*€', r'\1,\2€', text)
  
    # Correction dates
    text = re.sub(r'(\d{2})[/.,-](\d{2})[/.,-](\d{4})', r'\1/\2/\3', text)
  
    # Mots-clés factures
    corrections = {
        'FACTIJRE': 'FACTURE',
        'TTC': 'TTC',
        'HT': 'HT'
    }
  
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
  
    return text
```

### 2. Validation Métier

```python
def validate_invoice_data(ocr_results):
    """Validation des données extraites"""
    required_fields = ['numero_facture', 'date', 'montant_ttc']
  
    validation_errors = []
  
    # Vérifier présence champs obligatoires
    for field in required_fields:
        if field not in ocr_results:
            validation_errors.append(f"Champ manquant: {field}")
  
    # Vérifier format montant
    if 'montant_ttc' in ocr_results:
        montant = ocr_results['montant_ttc']
        if not re.match(r'\d+[,.]?\d*\s*€', montant):
            validation_errors.append("Format montant invalide")
  
    return validation_errors
```

## 🚨 Dépannage

### Problèmes Courants

#### 1. Erreur CUDA Out of Memory

```bash
# Réduire batch_size
python trocr_finetuning.py --batch_size 2

# Ou utiliser CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 2. Erreur de Dépendances

```bash
# Réinstaller avec versions compatibles
pip install torch==1.13.0 torchvision==0.14.0
pip install transformers==4.21.0
```

#### 3. Performances Faibles

- ✅ Vérifier qualité des images (résolution, contraste)
- ✅ Augmenter le nombre d'époques
- ✅ Ajuster le learning rate
- ✅ Utiliser plus de données d'entraînement

## 📊 Monitoring de l'Entraînement

### TensorBoard (TrOCR)

```bash
tensorboard --logdir models/trocr_finetuned/logs
```

### Graphiques Matplotlib (EasyOCR)

```python
# Automatiquement généré dans easyocr_finetuning.py
# Fichiers : training_curves.png, training_history.json
```

## 🎉 Étapes Suivantes

### 1. Intégration en Production

```python
# Exemple d'utilisation
from fine_tuning_manager import OCRFineTuningManager

manager = OCRFineTuningManager()
best_model = manager.get_best_model()
results = best_model.predict(image_path)
```

### 2. Amélioration Continue

- 📈 Collecter plus de données de factures variées
- 🔄 Ré-entraîner périodiquement
- 📊 Surveiller les performances en production
- 🎯 Affiner le post-processing

### 3. Déploiement

- 🐳 Containerisation Docker
- ☁️ Déploiement cloud
- 🔌 API REST
- 📱 Interface web

## 📞 Support

Pour toute question ou problème :

1. 📖 Consulter les logs dans `logs/`
2. 🔍 Vérifier les fichiers de configuration
3. 📊 Analyser les métriques d'évaluation
4. 🛠️ Ajuster les hyperparamètres

---

**🎯 Objectif final :** Atteindre **90%+ de précision** sur vos factures avec une vitesse de traitement **< 2 secondes** par document.

**✅ Avec vos 1000+ images de factures, vous avez toutes les chances de réussir !**
