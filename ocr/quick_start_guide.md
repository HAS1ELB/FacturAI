# 🚀 Guide de Démarrage Rapide - FacturAI OCR

## 📍 Votre Position Actuelle

✅ **TERMINÉ** - Collecte et prétraitement des images  
🔄 **SUIVANT** - Extraction OCR avancée  
⏳ **À VENIR** - Analyse VLM, LLM, structuration  

---

## 🎯 Objectif Immédiat : OCR sur Vos Images Prétraitées

Extraire tout le texte de vos images dans `Data/processed_images` avec les coordonnées spatiales pour les étapes suivantes.

---

## ⚡ Démarrage Express (5 minutes)

### Étape 1 : Installation des Dépendances
```bash
# Option A: Installation automatique
python install_ocr_dependencies.py

# Option B: Installation manuelle
pip install pytesseract easyocr paddlepaddle paddleocr
pip install opencv-python pillow numpy

# Sur Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### Étape 2 : Structure des Dossiers
```
FacturAI/
├── Data/
│   ├── processed_images/     ← VOS IMAGES PRÉTRAITÉES ICI
│   └── ocr_results/         ← RÉSULTATS OCR ICI
├── ocr_starter.py           ← SCRIPT PRINCIPAL
└── install_ocr_dependencies.py
```

### Étape 3 : Lancement
```bash
# Traitement automatique de toutes vos images
python ocr_starter.py
```

---

## 📊 Ce Que Vous Obtiendrez

### Résultats par Image
Fichier JSON avec:
- **Texte extrait** avec coordonnées précises
- **Scores de confiance** pour chaque bloc
- **Bounding boxes** (x, y, largeur, hauteur)
- **Métriques de qualité**

### Exemple de Résultat OCR
```json
{
  "engine": "easyocr",
  "timestamp": "2024-01-20T10:30:00",
  "image_path": "Data/processed_images/facture_001.png",
  "text_blocks": [
    {
      "text": "FACTURE N° 2024-001",
      "confidence": 95.3,
      "bbox": {"x": 100, "y": 50, "width": 200, "height": 30}
    },
    {
      "text": "Date: 15/01/2024",
      "confidence": 88.7,
      "bbox": {"x": 400, "y": 50, "width": 120, "height": 25}
    }
  ],
  "full_text": "FACTURE N° 2024-001 Date: 15/01/2024...",
  "average_confidence": 91.2
}
```

### Rapport Global
```json
{
  "processed_images": 50,
  "engine_used": "easyocr",
  "statistics": {
    "success_rate": 96.0,
    "average_confidence": 87.3,
    "total_text_blocks": 1247
  }
}
```

---

## 🔧 Moteurs OCR Disponibles

### 1. **EasyOCR** (Recommandé)
- ✅ Excellent pour factures
- ✅ Support multilingue (FR/EN/AR)
- ✅ Coordonnées précises
- ✅ Installation simple

### 2. **Tesseract**
- ✅ Très mature et stable
- ✅ Hautement configurable
- ✅ Excellent pour texte imprimé
- ⚠️ Nécessite fine-tuning

### 3. **PaddleOCR**
- ✅ Performance excellente
- ✅ Support GPU
- ✅ Modèles pré-entraînés
- ⚠️ Plus lourd à installer

---

## 🎛️ Configuration Avancée

### Personnaliser les Paramètres OCR

```python
# Dans ocr_starter.py, modifier ces variables:

# Pour Tesseract
custom_config = r'--oem 3 --psm 6 -l fra+eng+ara'

# Pour EasyOCR
languages = ['fr', 'en', 'ar']  # Ajouter l'arabe pour le Maroc

# Seuils de confiance
min_confidence = 30  # Augmenter pour plus de précision
```

### Traitement par Lot Personnalisé

```python
from ocr_starter import InvoiceOCRProcessor

processor = InvoiceOCRProcessor()

# Traiter avec un moteur spécifique
results = processor.process_directory(
    input_dir="Data/processed_images",
    engine="easyocr"  # ou "tesseract", "paddleocr"
)
```

---

## 📈 Métriques de Performance à Surveiller

### 1. **Taux de Succès**
- **Objectif**: > 95%
- **Si < 90%**: Vérifier qualité des images prétraitées

### 2. **Confiance Moyenne**
- **Objectif**: > 80%
- **Si < 70%**: Améliorer le prétraitement ou changer de moteur

### 3. **Couverture Textuelle**
- **Vérifier**: Tous les champs importants sont détectés
- **Manqués**: Ajuster les seuils ou le preprocessing

---

## 🐛 Résolution de Problèmes

### Erreur: "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Télécharger depuis GitHub UB-Mannheim
```

### Erreur: "EasyOCR GPU memory"
```python
# Forcer l'utilisation du CPU
reader = easyocr.Reader(['fr', 'en'], gpu=False)
```

### Images non détectées
```bash
# Vérifier les extensions supportées
ls Data/processed_images/*.{png,jpg,jpeg,tiff}

# Vérifier les permissions
chmod 755 Data/processed_images/
```

---

## 📋 Checklist Avant de Continuer

- [ ] ✅ Images prétraitées dans `Data/processed_images/`
- [ ] ✅ Au moins un moteur OCR installé
- [ ] ✅ Script `ocr_starter.py` exécuté avec succès
- [ ] ✅ Résultats JSON générés dans `Data/ocr_results/`
- [ ] ✅ Taux de succès > 90%
- [ ] ✅ Confiance moyenne > 80%

---

## 🎯 Prochaines Étapes (Semaines 5-6)

Une fois l'OCR validé:

1. **Analyse VLM** - Comprendre la mise en page
2. **Détection de Zones** - Identifier en-têtes, tableaux, totaux
3. **Extraction Structurée** - Organiser les données par champs

---

## 💡 Conseils pour Optimiser l'OCR

### 1. **Qualité des Images**
- Résolution minimale: 300 DPI
- Contraste élevé (noir sur blanc)
- Pas de rotation résiduelle

### 2. **Choix du Moteur**
- **Factures simples**: Tesseract
- **Layouts complexes**: EasyOCR
- **Performance/GPU**: PaddleOCR

### 3. **Post-Processing**
- Filtrer les blocs avec confiance < 30%
- Fusionner les mots coupés
- Corriger l'orientation du texte

---

## 🆘 Support

En cas de problème:
1. Vérifier les logs dans `ocr_starter.py`
2. Tester sur une seule image d'abord
3. Comparer les résultats des différents moteurs
4. Valider la qualité du prétraitement

**Prêt à démarrer l'OCR sur vos factures? Exécutez `python ocr_starter.py` ! 🚀**