#!/usr/bin/env python3
"""
Fine-tuning EasyOCR pour les factures
Système complet d'entraînement personnalisé pour améliorer la précision OCR
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time

# EasyOCR et CRAFT imports
import easyocr
import craft_text_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceOCRDataset(Dataset):
    """Dataset personnalisé pour le fine-tuning EasyOCR"""
    
    def __init__(self, data_file: str, vocab: List[str] = None, transform=None, max_text_length: int = None):
        self.transform = transform
        
        # Use provided vocabulary or create a default one
        if vocab is not None:
            self.vocab = vocab
        else:
            # Default vocabulary (fallback)
            base_chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz'
            french_chars = 'àâäéèêëïîôöùûüÿç'
            special_chars = '€°§'
            vocab_chars = base_chars + french_chars + special_chars
            self.vocab = list(vocab_chars) + ['<PAD>', '<UNK>']
        
        # Construire le mapping caractère -> indice
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        
        # Calculer la longueur de séquence basée sur l'architecture du modèle
        # Image input: 256 width
        # Après CNN avec pooling: 256 -> 128 -> 64 -> 64 -> 32 -> 16 -> 16 -> 16
        # Le nombre de MaxPool2d (stride 2) en width: 256/2/2/2/2 = 16
        # Avec les MaxPool2d((2,1)): pas de réduction en width
        # Donc séquence finale ≈ 64 (en fonction de l'architecture CNN)
        self.max_text_length = max_text_length if max_text_length else 64
        
        # Charger les données
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Dataset chargé: {len(self.data)} échantillons")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger l'image
        image = cv2.imread(item['image_path'])
        if image is None:
            # Image par défaut si erreur
            image = np.zeros((64, 256, 3), dtype=np.uint8)
        
        # Convertir en RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extraire la région de texte si bbox disponible
        if 'bbox' in item and item['bbox']:
            bbox = item['bbox']
            if len(bbox) >= 4:
                # Extraire les coordonnées
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                # Cropper la région
                if x2 > x1 and y2 > y1:
                    image = image[y1:y2, x1:x2]
        
        # Redimensionner pour la cohérence
        image = cv2.resize(image, (256, 64))
        
        # Convertir en PIL pour les transformations
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Encoder le texte
        text = item.get('text', '')
        text_encoded = self.encode_text(text)
        
        return {
            'image': image,
            'text': text,
            'text_encoded': text_encoded,
            'confidence': item.get('confidence', 1.0)
        }
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode le texte en indices de caractères"""
        # Utiliser le vocabulaire défini dans le constructeur
        encoded = []
        for char in text[:self.max_text_length]:
            encoded.append(self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', len(self.vocab)-1)))
        
        # Padding
        pad_idx = self.char_to_idx.get('<PAD>', len(self.vocab)-2)
        while len(encoded) < self.max_text_length:
            encoded.append(pad_idx)
        
        return torch.tensor(encoded, dtype=torch.long)

class CRNN(nn.Module):
    """Modèle CRNN (CNN + RNN) pour la reconnaissance de texte"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super(CRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Partie CNN pour l'extraction de features
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv Block 4
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            # Conv Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Conv Block 6
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            # Adaptive pooling pour s'assurer que height=1
            nn.AdaptiveAvgPool2d((1, None))  # (batch, 512, 1, W')
        )
        
        # Partie RNN
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        
        # Couche de classification
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
        # Dropout pour la régularisation
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # CNN features
        cnn_features = self.cnn(x)  # (batch, 512, 1, W')
        
        # Reshape pour RNN - maintenant height=1, donc channels*height=512*1=512
        batch_size, channels, height, width = cnn_features.size()
        cnn_features = cnn_features.permute(0, 3, 1, 2)  # (batch, W', 512, 1)
        cnn_features = cnn_features.reshape(batch_size, width, channels * height)  # (batch, W', 512)
        
        # RNN
        rnn_out, _ = self.rnn(cnn_features)  # (batch, W', hidden_size*2)
        rnn_out = self.dropout(rnn_out)
        
        # Classification
        output = self.classifier(rnn_out)  # (batch, W', vocab_size)
        
        return output

class EasyOCRFineTuner:
    """Gestionnaire de fine-tuning pour EasyOCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Utilisation du device: {self.device}")
        
        # Initialiser EasyOCR de base
        self.base_reader = easyocr.Reader(['fr'], gpu=torch.cuda.is_available())
        
        # Vocabulaire étendu pour les factures
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # Modèle personnalisé
        self.model = CRNN(self.vocab_size, self.config.get('hidden_size', 256))
        self.model.to(self.device)
        
        # Optimiseur et fonction de perte
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001)
        )
        
        # Trouver l'index du token PAD pour l'ignorer dans la perte
        pad_idx = self.vocab.index('<PAD>') if '<PAD>' in self.vocab else self.vocab_size - 2
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        # Métriques
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _build_vocab(self) -> List[str]:
        """Construit le vocabulaire spécialisé pour les factures"""
        base_chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz'
        french_chars = 'àâäéèêëïîôöùûüÿç'
        special_chars = '€°§'
        
        vocab = list(base_chars + french_chars + special_chars)
        vocab.extend(['<PAD>', '<UNK>'])
        
        return vocab
    
    def prepare_data(self, dataset_file: str, batch_size: int = 8):
        """Prépare les DataLoaders"""
        # Transformations pour l'augmentation des données
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Charger les données et faire le split
        with open(dataset_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Split train/val
        split_idx = int(0.8 * len(all_data))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Sauvegarder temporairement les splits
        train_file = Path(dataset_file).parent / "temp_train.json"
        val_file = Path(dataset_file).parent / "temp_val.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False)
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False)
        
        # Créer les datasets avec la longueur de séquence correcte ET le vocabulaire
        # Calculer la longueur de séquence de sortie du modèle
        # Image input: (3, 64, 256)
        # Après CNN: (512, 1, W') où W' dépend des pooling operations
        # MaxPool2d(2, 2): 256 -> 128 (width réduit)
        # MaxPool2d(2, 2): 128 -> 64 (width réduit) 
        # MaxPool2d((2, 1)): 64 -> 64 (width inchangé)
        # MaxPool2d((2, 1)): 64 -> 64 (width inchangé)
        # AdaptiveAvgPool2d((1, None)): height -> 1, width inchangé
        # Donc la séquence finale devrait être de longueur 64
        model_seq_length = 64  # Basé sur l'architecture CNN
        
        train_dataset = InvoiceOCRDataset(str(train_file), 
                                        vocab=self.vocab,  # ✅ Passer le vocabulaire du fine-tuner
                                        transform=train_transform, 
                                        max_text_length=model_seq_length)
        val_dataset = InvoiceOCRDataset(str(val_file), 
                                      vocab=self.vocab,  # ✅ Passer le vocabulaire du fine-tuner
                                      transform=val_transform, 
                                      max_text_length=model_seq_length)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        logger.info(f"Données préparées: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Nettoyer les fichiers temporaires
        train_file.unlink()
        val_file.unlink()
    
    def train_epoch(self) -> Tuple[float, float]:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            texts_encoded = batch['text_encoded'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)  # (batch, seq_len, vocab_size)
            
            # Pour le calcul de la perte, on doit aligner les dimensions
            # outputs: (batch, seq_len, vocab_size)
            # texts_encoded: (batch, max_text_length)
            
            # Prendre seulement la longueur de séquence produite par le modèle
            seq_len = outputs.size(1)
            texts_encoded_aligned = texts_encoded[:, :seq_len]  # (batch, seq_len)
            
            # Reshape pour CrossEntropyLoss: (batch*seq_len, vocab_size) et (batch*seq_len,)
            outputs_flat = outputs.view(-1, outputs.size(-1))  # (batch*seq_len, vocab_size)
            texts_flat = texts_encoded_aligned.view(-1)  # (batch*seq_len,)
            
            # Calculer la perte
            loss = self.criterion(outputs_flat, texts_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculer l'accuracy
            predictions = torch.argmax(outputs, dim=-1)  # (batch, seq_len)
            correct = (predictions == texts_encoded_aligned).float()
            correct_predictions += correct.sum().item()
            total_predictions += correct.numel()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Valide le modèle"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                texts_encoded = batch['text_encoded'].to(self.device)
                
                outputs = self.model(images)  # (batch, seq_len, vocab_size)
                
                # Aligner les dimensions comme dans l'entraînement
                seq_len = outputs.size(1)
                texts_encoded_aligned = texts_encoded[:, :seq_len]  # (batch, seq_len)
                
                # Reshape pour CrossEntropyLoss
                outputs_flat = outputs.view(-1, outputs.size(-1))  # (batch*seq_len, vocab_size)
                texts_flat = texts_encoded_aligned.view(-1)  # (batch*seq_len,)
                
                loss = self.criterion(outputs_flat, texts_flat)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=-1)  # (batch, seq_len)
                correct = (predictions == texts_encoded_aligned).float()
                correct_predictions += correct.sum().item()
                total_predictions += correct.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, dataset_file: str, epochs: int = 50, batch_size: int = 8) -> Dict[str, Any]:
        """Lance l'entraînement complet"""
        logger.info("🚀 DÉMARRAGE DU FINE-TUNING EASYOCR")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Préparer les données
        self.prepare_data(dataset_file, batch_size)
        
        # Variables pour early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Entraînement
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Sauvegarder l'historique
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Époque {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'époque {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        
        # Sauvegarder le modèle final
        final_model_path = self.save_model("final_model.pth")
        
        # Sauvegarder l'historique d'entraînement
        history_path = self.save_training_history()
        
        logger.info("=" * 50)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ")
        logger.info(f"⏱️ Temps total: {total_time/60:.1f} minutes")
        logger.info(f"🎯 Meilleure val loss: {best_val_loss:.4f}")
        
        return {
            'model_path': final_model_path,
            'history_path': history_path,
            'best_val_loss': best_val_loss,
            'total_time': total_time,
            'training_history': self.training_history
        }
    
    def save_model(self, filename: str) -> str:
        """Sauvegarde le modèle"""
        models_dir = Path(self.config.get('output_dir', 'models/easyocr_finetuned'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        self.vocab_size = len(self.vocab)
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Modèle chargé depuis {model_path}")
    
    def save_training_history(self) -> str:
        """Sauvegarde l'historique d'entraînement"""
        output_dir = Path(self.config.get('output_dir', 'models/easyocr_finetuned'))
        
        # Sauvegarder en JSON
        history_file = output_dir / "training_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Créer des graphiques
        self.plot_training_curves(output_dir / "training_curves.png")
        
        return str(history_file)
    
    def plot_training_curves(self, save_path: str):
        """Crée des graphiques des courbes d'entraînement"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Perte
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Évolution de la perte')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Évolution de la précision')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Précision')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Courbes d'entraînement sauvegardées: {save_path}")
    
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """Fait une prédiction sur une image"""
        self.model.eval()
        
        # Charger et préprocesser l'image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Utiliser d'abord EasyOCR base pour détecter les zones de texte
        base_results = self.base_reader.readtext(image_path)
        
        predictions = []
        
        with torch.no_grad():
            for (bbox, text, confidence) in base_results:
                # Extraire la région
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                if x2 > x1 and y2 > y1:
                    region = image[y1:y2, x1:x2]
                    region = cv2.resize(region, (256, 64))
                    
                    # Convertir en tensor
                    region_pil = Image.fromarray(region)
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    region_tensor = transform(region_pil).unsqueeze(0).to(self.device)
                    
                    # Prédiction
                    output = self.model(region_tensor)
                    predicted_indices = torch.argmax(output, dim=2).squeeze().cpu().numpy()
                    
                    # Décoder le texte
                    predicted_text = self.decode_prediction(predicted_indices)
                    
                    predictions.append({
                        'bbox': bbox,
                        'original_text': text,
                        'predicted_text': predicted_text,
                        'original_confidence': confidence,
                        'enhanced': predicted_text != text
                    })
        
        return predictions
    
    def decode_prediction(self, indices: np.ndarray) -> str:
        """Décode les indices en texte"""
        text = ""
        for idx in indices:
            if idx < len(self.vocab) - 2:  # Exclure PAD et UNK
                char = self.vocab[idx]
                if char != '<PAD>':
                    text += char
        
        return text.strip()

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning EasyOCR pour factures")
    parser.add_argument('--dataset', required=True, help='Fichier dataset JSON')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille de batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Taux d\'apprentissage')
    parser.add_argument('--output_dir', default='fine-tuning-ocr/models/easyocr_finetuned', help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'hidden_size': 256,
        'patience': 10
    }
    
    # Créer le fine-tuner
    fine_tuner = EasyOCRFineTuner(config)
    
    # Lancer l'entraînement
    results = fine_tuner.train(
        dataset_file=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\n🎉 Fine-tuning terminé!")
    print(f"📁 Modèle sauvegardé: {results['model_path']}")
    print(f"📊 Historique: {results['history_path']}")
    print(f"🎯 Meilleure validation loss: {results['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()