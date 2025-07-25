#!/usr/bin/env python3
"""
TrOCR Fine-tuning pour FacturAI
Approche moderne utilisant Transformers de Hugging Face
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import evaluate

logger = logging.getLogger(__name__)

class TrOCRDataset:
    """Dataset personnalisé pour TrOCR"""
    
    def __init__(self, annotations: List[Dict], processor, max_target_length: int = 512):
        self.annotations = annotations
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Charger l'image
        image = Image.open(annotation["image_path"]).convert("RGB")
        
        # Texte de référence
        text = annotation["text"]
        
        # Préprocessing
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Remplacer les tokens de padding par -100 pour ignorer dans la loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }

class TrOCRFineTuner:
    """Fine-tuner pour TrOCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("base_model", "microsoft/trocr-large-printed")
        self.output_dir = config.get("output_dir", "models/trocr_finetuned")
        self.setup_model()
        
    def setup_model(self):
        """Initialise le modèle et le processeur"""
        logger.info(f"Initialisation du modèle TrOCR: {self.model_name}")
        
        # Chargement du processeur et du modèle
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        
        # Configuration du modèle
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Optimisations
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Modèle déplacé sur GPU")
        
        logger.info("✅ Modèle TrOCR initialisé")
    
    def prepare_datasets(self) -> DatasetDict:
        """Prépare les datasets d'entraînement"""
        logger.info("Préparation des datasets TrOCR...")
        
        # Charger les annotations
        annotations_dir = "Data/annotations"
        
        datasets = {}
        for split in ["train", "validation", "test"]:
            annotations_file = os.path.join(annotations_dir, f"{split}_annotations.json")
            
            if os.path.exists(annotations_file):
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Créer le dataset
                dataset = TrOCRDataset(
                    annotations, 
                    self.processor,
                    max_target_length=self.config.get("max_length", 512)
                )
                
                datasets[split] = dataset
                logger.info(f"Dataset {split}: {len(annotations)} échantillons")
            else:
                logger.warning(f"Fichier d'annotations manquant: {annotations_file}")
        
        if not datasets:
            raise ValueError("Aucun dataset trouvé. Exécutez d'abord la préparation des données.")
        
        return datasets
    
    def compute_metrics(self, eval_pred):
        """Calcule les métriques d'évaluation"""
        predictions, labels = eval_pred
        
        # Décoder les prédictions et labels
        decoded_preds = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Remplacer -100 par pad_token_id pour le décodage
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Nettoyer les textes
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculer les métriques
        exact_match = sum(pred == label for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        # BLEU score
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        
        # Character Error Rate
        total_chars = sum(len(label) for label in decoded_labels)
        char_errors = sum(
            sum(c1 != c2 for c1, c2 in zip(pred, label)) + abs(len(pred) - len(label))
            for pred, label in zip(decoded_preds, decoded_labels)
        )
        cer = char_errors / total_chars if total_chars > 0 else 1.0
        
        return {
            "exact_match": exact_match,
            "bleu": bleu_score["bleu"],
            "character_error_rate": cer,
            "accuracy": exact_match  # Alias pour compatibilité
        }
    
    def train(self) -> Dict[str, Any]:
        """Entraîne le modèle TrOCR"""
        logger.info("🚀 Démarrage de l'entraînement TrOCR")
        
        # Préparer les datasets
        datasets = self.prepare_datasets()
        
        if "train" not in datasets:
            raise ValueError("Dataset d'entraînement manquant")
        
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            num_train_epochs=self.config.get("epochs", 30),
            learning_rate=self.config.get("learning_rate", 5e-5),
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            warmup_steps=1000,
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 2),
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None  # Désactiver wandb par défaut
        )
        
        # Callbacks
        callbacks = []
        if "validation" in datasets:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
        
        # Créer le trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation", None),
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Entraînement
        start_time = datetime.now()
        logger.info("Début de l'entraînement...")
        
        try:
            train_result = trainer.train()
            
            # Sauvegarder le modèle final
            trainer.save_model()
            self.processor.save_pretrained(self.output_dir)
            
            # Évaluation finale
            eval_results = {}
            if "validation" in datasets:
                eval_results["validation"] = trainer.evaluate(datasets["validation"])
            if "test" in datasets:
                eval_results["test"] = trainer.evaluate(datasets["test"])
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Compilation des résultats
            results = {
                "model_type": "trocr",
                "base_model": self.model_name,
                "training_args": training_args.to_dict(),
                "train_results": {
                    "train_loss": train_result.training_loss,
                    "train_runtime": train_result.metrics.get("train_runtime", training_time),
                    "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                    "epochs_completed": train_result.metrics.get("epoch", self.config.get("epochs", 30))
                },
                "evaluation": eval_results,
                "model_path": self.output_dir,
                "training_completed": True,
                "training_time_seconds": training_time
            }
            
            # Sauvegarder les résultats
            results_file = os.path.join(self.output_dir, "training_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Entraînement TrOCR terminé en {training_time:.2f}s")
            logger.info(f"📁 Modèle sauvegardé: {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur durant l'entraînement TrOCR: {str(e)}")
            return {
                "model_type": "trocr",
                "error": str(e),
                "training_completed": False
            }
    
    def load_finetuned_model(self, model_path: str = None):
        """Charge un modèle fine-tuné"""
        if model_path is None:
            model_path = self.output_dir
        
        if not os.path.exists(model_path):
            raise ValueError(f"Modèle non trouvé: {model_path}")
        
        logger.info(f"Chargement du modèle fine-tuné: {model_path}")
        
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        logger.info("✅ Modèle fine-tuné chargé")
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Fait une prédiction sur une image"""
        # Charger l'image
        image = Image.open(image_path).convert("RGB")
        
        # Préprocessing
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Génération
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=512)
        
        # Décodage
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "image_path": image_path,
            "predicted_text": generated_text.strip(),
            "model_type": "trocr_finetuned"
        }
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Prédictions en batch"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur prédiction {image_path}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "model_type": "trocr_finetuned"
                })
        
        return results

def main():
    """Test du fine-tuning TrOCR"""
    config = {
        "base_model": "microsoft/trocr-large-printed",
        "output_dir": "models/trocr_finetuned",
        "epochs": 3,  # Test rapide
        "batch_size": 2,
        "learning_rate": 5e-5
    }
    
    trainer = TrOCRFineTuner(config)
    results = trainer.train()
    
    print("Résultats:", json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()