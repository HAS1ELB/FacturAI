#!/usr/bin/env python3
"""
Script de conversion des datasets pour TrOCR
Convertit les datasets générés par data_preparation.py au format attendu par trocr_finetuning.py
"""

import json
import os
from pathlib import Path

def convert_datasets(fine_tuning_dir):
    """
    Convertit les datasets au format attendu par trocr_finetuning.py
    
    Args:
        fine_tuning_dir (str): Chemin vers le répertoire contenant les données de fine-tuning
    """
    print("Conversion des datasets pour TrOCR...")
    
    # Chemins
    fine_tuning_dir = Path(fine_tuning_dir)
    splits_dir = fine_tuning_dir / "splits"
    trocr_dir = fine_tuning_dir / "datasets" / "trocr"
    annotations_dir = fine_tuning_dir / "annotations"
    
    # Vérifier les répertoires
    if not trocr_dir.exists() or not (trocr_dir / "dataset.json").exists():
        print(f"❌ Fichier TrOCR dataset.json non trouvé dans {trocr_dir}")
        return False
    
    # S'assurer que le répertoire annotations existe
    annotations_dir.mkdir(exist_ok=True)
    
    # Charger le dataset TrOCR
    with open(trocr_dir / "dataset.json", 'r', encoding='utf-8') as f:
        trocr_data = json.load(f)
    
    print(f"📊 Dataset TrOCR chargé: {len(trocr_data)} échantillons")
    
    # Charger les splits
    splits = {}
    for split_name in ["train", "validation", "test"]:
        split_file = splits_dir / f"{split_name}.json"
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                splits[split_name] = json.load(f)
                print(f"📊 Split {split_name} chargé: {len(splits[split_name])} échantillons")
    
    # Si pas de splits, créer manuellement
    if not splits:
        print("⚠️ Aucun fichier de split trouvé, création manuelle...")
        # On prend 80% train, 10% validation, 10% test
        n = len(trocr_data)
        train_size = int(n * 0.8)
        val_size = int(n * 0.1)
        
        splits["train"] = trocr_data[:train_size]
        splits["validation"] = trocr_data[train_size:train_size+val_size]
        splits["test"] = trocr_data[train_size+val_size:]
    
    # Créer les annotations pour chaque split
    for split_name, split_data in splits.items():
        # Filtrer les données TrOCR pour ce split
        annotations = []
        
        if isinstance(split_data, list):
            for item in split_data:
                image_path = item.get("image_path")
                
                # Trouver cet item dans le dataset TrOCR
                for trocr_item in trocr_data:
                    if trocr_item.get("image_path") == image_path:
                        annotations.append(trocr_item)
                        break
        
        # Si aucune annotation trouvée et nous avons créé les splits manuellement
        if not annotations and not all(os.path.exists(splits_dir / f"{s}.json") for s in ["train", "validation", "test"]):
            annotations = splits[split_name]
        
        # Sauvegarder les annotations
        output_file = annotations_dir / f"{split_name}_annotations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Annotations {split_name} sauvegardées: {len(annotations)} items")
    
    print("✅ Conversion terminée!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convertir les datasets pour TrOCR")
    parser.add_argument('--fine_tuning_dir', default='Data/fine_tuning',
                        help='Répertoire contenant les données de fine-tuning')
    
    args = parser.parse_args()
    convert_datasets(args.fine_tuning_dir)
