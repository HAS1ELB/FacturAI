# Plan d'Action FacturAI - Prochaines Étapes

## État Actuel ✅
Vous avez terminé avec succès :
- ✅ **Collecte des données** : Images de factures collectées
- ✅ **Prétraitement** : PDFs convertis en images et images améliorées dans `Data/processed_images`

## Prochaines Étapes Prioritaires

### 🔍 **ÉTAPE 2 : OCR Avancée** (Semaine 4 selon votre planning)

#### Objectifs
- Extraire le texte des images prétraitées avec les coordonnées spatiales
- Obtenir un texte brut avec les bounding boxes pour chaque mot/bloc

#### Actions Immédiates

1. **Installation des dépendances OCR**
```bash
pip install pytesseract
pip install easyocr
pip install paddlepaddle-gpu paddleocr  # ou paddlepaddle pour CPU
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-fra  # pour le français
```

2. **Créer le module OCR principal** (`ocr_module.py`)
   - Intégration de Tesseract optimisé pour factures
   - Support pour EasyOCR et PaddleOCR en alternative
   - Extraction texte + coordonnées (bounding boxes)
   - Gestion multilingue (français/arabe pour le Maroc)

3. **Tests et évaluation OCR**
   - Tester sur vos images prétraitées
   - Mesurer le taux de précision
   - Optimiser les paramètres pour les factures

---

### 🧠 **ÉTAPE 3 : Analyse par VLM** (Semaines 5-6)

#### Objectifs
- Comprendre la mise en page des factures
- Identifier les zones clés (en-tête, tableau, totaux)

#### Modèles VLM Recommandés
- **LLaVA** (open source, performant)
- **Qwen-VL** (excellent pour documents)
- **BLIP-2** (plus léger)

---

### 🔤 **ÉTAPE 4 : Compréhension Contextuelle** (Semaine 7)

#### Objectifs
- Corriger les erreurs d'OCR
- Interpréter le contexte sémantique

#### Solutions
- **LLM** : GPT-4, Claude, ou modèles open source (Llama, Mistral)
- **MLM** : BERT ou RoBERTa pour correction de texte

---

## Structure de Code Recommandée

```
FacturAI/
├── data/
│   ├── processed_images/          # Vos images prétraitées ✅
│   ├── ocr_results/              # Résultats OCR
│   └── structured_data/          # Données extraites finales
├── src/
│   ├── preprocessing/            # Vos modules existants ✅
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── tesseract_engine.py
│   │   ├── easyocr_engine.py
│   │   └── ocr_pipeline.py
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── layout_analyzer.py
│   │   └── zone_detector.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── text_corrector.py
│   │   └── semantic_analyzer.py
│   └── utils/
│       ├── __init__.py
│       ├── validation.py
│       └── data_structures.py
├── config/
│   ├── ocr_config.yaml
│   └── model_config.yaml
├── tests/
├── notebooks/                    # Pour expérimentation
└── main.py                      # Pipeline principal
```

## Action Immédiate : Commencer par l'OCR

### Script de Test OCR Initial
Je vais créer un script de test pour démarrer immédiatement l'extraction OCR sur vos images prétraitées.

### Métriques à Mesurer
1. **Précision OCR** : % de mots correctement reconnus
2. **Couverture** : % de texte détecté vs texte réel
3. **Performance** : Temps de traitement par image
4. **Qualité des coordonnées** : Précision des bounding boxes

## Calendrier Adapté

- **Cette semaine** : OCR avancée + tests
- **Semaine suivante** : Intégration VLM
- **Semaine d'après** : LLM/MLM pour correction

## Questions pour Optimiser

1. Avez-vous une préférence pour l'API OCR (Google Vision, Azure) vs solution locale ?
2. Quel est votre budget/contraintes pour les modèles payants ?
3. Les factures sont-elles principalement en français, arabe, ou bilingues ?
4. Avez-vous accès à un GPU pour l'entraînement des modèles ?

Voulez-vous que je commence par créer le module OCR pour traiter vos images prétraitées ?