#!/usr/bin/env python3
"""
Fine-tuning PaddleOCR pour les factures
Système d'entraînement personnalisé utilisant PaddlePaddle
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import yaml
from datetime import datetime
import time

# PaddleOCR imports
from paddleocr import PaddleOCR
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaddleInvoiceDataset(Dataset):
    """Dataset personnalisé pour PaddleOCR"""
    
    def __init__(self, data_file: str, transform=None, mode='train'):
        self.mode = mode
        self.transform = transform
        
        # Charger les données
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Dataset PaddleOCR chargé: {len(self.data)} échantillons ({mode})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Charger l'image
        image_path = item['image_path']
        image = cv2.imread(image_path)
        
        if image is None:
            # Image par défaut si erreur
            image = np.zeros((64, 256, 3), dtype=np.uint8)
        
        # Préparer les annotations pour PaddleOCR
        lines = item.get('lines', [])
        
        # Format PaddleOCR: liste de dictionnaires avec transcription et points
        annotations = []
        for line in lines:
            if 'transcription' in line and 'points' in line:
                annotations.append({
                    'transcription': line['transcription'],
                    'points': line['points'],
                    'difficult': False
                })
        
        sample = {
            'image': image,
            'annotations': annotations,
            'image_path': image_path
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class PaddleOCRConfig:
    """Configuration pour PaddleOCR fine-tuning"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dossiers de configuration
        self.config_dir = self.output_dir / "configs"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.config_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def create_detection_config(self, dataset_dir: str) -> str:
        """Crée la configuration pour la détection de texte"""
        config = {
            'Global': {
                'debug': False,
                'use_gpu': True,
                'epoch_num': 50,
                'log_smooth_window': 20,
                'print_batch_step': 10,
                'save_model_dir': str(self.checkpoints_dir / "det"),
                'save_epoch_step': 5,
                'eval_batch_step': [0, 400],
                'cal_metric_during_train': True,
                'pretrained_model': None,
                'checkpoints': None,
                'save_inference_dir': str(self.output_dir / "inference" / "det"),
                'use_visualdl': False,
                'infer_img': None,
                'save_res_path': str(self.logs_dir / "det_results.txt")
            },
            'Architecture': {
                'model_type': 'det',
                'algorithm': 'DB',
                'Transform': None,
                'Backbone': {
                    'name': 'ResNet',
                    'layers': 18,
                    'disable_se': True
                },
                'Neck': {
                    'name': 'DBFPN',
                    'out_channels': 256
                },
                'Head': {
                    'name': 'DBHead',
                    'k': 50
                }
            },
            'Loss': {
                'name': 'DBLoss',
                'balance_loss': True,
                'main_loss_type': 'DiceLoss',
                'alpha': 5,
                'beta': 10,
                'ohem_ratio': 3
            },
            'Optimizer': {
                'name': 'Adam',
                'beta1': 0.9,
                'beta2': 0.999,
                'lr': {
                    'name': 'Cosine',
                    'learning_rate': 0.001,
                    'warmup_epoch': 2
                },
                'regularizer': {
                    'name': 'L2',
                    'factor': 1e-4
                }
            },
            'PostProcess': {
                'name': 'DBPostProcess',
                'thresh': 0.3,
                'box_thresh': 0.6,
                'max_candidates': 1000,
                'unclip_ratio': 1.5
            },
            'Metric': {
                'name': 'DetMetric',
                'main_indicator': 'hmean'
            },
            'Train': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': dataset_dir,
                    'label_file_list': [f"{dataset_dir}/train_list.txt"],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'DetLabelEncode': None},
                        {'IaaAugment': {
                            'augmenter_args': [
                                {'type': 'Fliplr', 'args': {'p': 0.5}},
                                {'type': 'Affine', 'args': {'rotate': [-10, 10]}},
                                {'type': 'Resize', 'args': {'size': [0.5, 3]}}
                            ]
                        }},
                        {'EastRandomCropData': {'size': [640, 640], 'max_tries': 50, 'keep_ratio': True}},
                        {'MakeBorderMap': {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}},
                        {'MakeShrinkMap': {'shrink_ratio': 0.4, 'min_text_size': 8}},
                        {'NormalizeImage': {'scale': '1./255.', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'order': 'hwc'}},
                        {'ToCHWImage': None},
                        {'KeepKeys': {'keep_keys': ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']}}
                    ]
                },
                'loader': {
                    'shuffle': True,
                    'drop_last': False,
                    'batch_size_per_card': 8,
                    'num_workers': 4
                }
            },
            'Eval': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': dataset_dir,
                    'label_file_list': [f"{dataset_dir}/val_list.txt"],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'DetLabelEncode': None},
                        {'DetResizeForTest': {'limit_side_len': 736, 'limit_type': 'min'}},
                        {'NormalizeImage': {'scale': '1./255.', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'order': 'hwc'}},
                        {'ToCHWImage': None},
                        {'KeepKeys': {'keep_keys': ['image', 'shape', 'polys', 'ignore_tags']}}
                    ]
                },
                'loader': {
                    'shuffle': False,
                    'drop_last': False,
                    'batch_size_per_card': 1,
                    'num_workers': 2
                }
            }
        }
        
        config_file = self.config_dir / "det_config.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return str(config_file)
    
    def create_recognition_config(self, dataset_dir: str) -> str:
        """Crée la configuration pour la reconnaissance de texte"""
        config = {
            'Global': {
                'debug': False,
                'use_gpu': True,
                'epoch_num': 50,
                'log_smooth_window': 20,
                'print_batch_step': 10,
                'save_model_dir': str(self.checkpoints_dir / "rec"),
                'save_epoch_step': 5,
                'eval_batch_step': [0, 500],
                'cal_metric_during_train': True,
                'pretrained_model': None,
                'checkpoints': None,
                'save_inference_dir': str(self.output_dir / "inference" / "rec"),
                'use_visualdl': False,
                'infer_img': None,
                'save_res_path': str(self.logs_dir / "rec_results.txt"),
                'character_dict_path': str(self.create_character_dict()),
                'max_text_length': 100,
                'infer_mode': False,
                'use_space_char': True
            },
            'Architecture': {
                'model_type': 'rec',
                'algorithm': 'SVTR_LCNet',
                'Transform': None,
                'Backbone': {
                    'name': 'PPLCNetV3',
                    'scale': 0.95
                },
                'Head': {
                    'name': 'MultiHead',
                    'head_list': [
                        {'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 0.00001}}},
                        {'SARHead': {'enc_dim': 512, 'max_text_length': 70}}
                    ]
                }
            },
            'Loss': {
                'name': 'MultiLoss',
                'loss_config_list': [
                    {'CTCLoss': None},
                    {'SARLoss': None}
                ]
            },
            'Optimizer': {
                'name': 'AdamW',
                'beta1': 0.9,
                'beta2': 0.999,
                'lr': {
                    'name': 'Cosine',
                    'learning_rate': 0.001,
                    'warmup_epoch': 5
                },
                'regularizer': {
                    'name': 'L2',
                    'factor': 3e-05
                }
            },
            'PostProcess': {
                'name': 'CTCLabelDecode'
            },
            'Metric': {
                'name': 'RecMetric',
                'main_indicator': 'acc'
            },
            'Train': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': dataset_dir,
                    'label_file_list': [f"{dataset_dir}/rec_train_list.txt"],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'RecConAug': {'prob': 0.5, 'ext_data_num': 2, 'image_shape': [48, 320, 3], 'max_text_length': 100}},
                        {'RecAug': None},
                        {'MultiLabelEncode': None},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']}}
                    ]
                },
                'loader': {
                    'shuffle': True,
                    'batch_size_per_card': 128,
                    'drop_last': True,
                    'num_workers': 4
                }
            },
            'Eval': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': dataset_dir,
                    'label_file_list': [f"{dataset_dir}/rec_val_list.txt"],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'MultiLabelEncode': None},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']}}
                    ]
                },
                'loader': {
                    'shuffle': False,
                    'batch_size_per_card': 128,
                    'drop_last': False,
                    'num_workers': 4
                }
            }
        }
        
        config_file = self.config_dir / "rec_config.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return str(config_file)
    
    def create_character_dict(self) -> str:
        """Crée le dictionnaire de caractères pour la reconnaissance"""
        # Caractères spécialisés pour les factures françaises
        chars = []
        
        # Chiffres
        chars.extend([str(i) for i in range(10)])
        
        # Lettres minuscules
        chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        
        # Lettres majuscules
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        
        # Caractères spéciaux français
        chars.extend(['à', 'â', 'ä', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ö', 'ù', 'û', 'ü', 'ÿ', 'ç'])
        chars.extend(['À', 'Â', 'Ä', 'É', 'È', 'Ê', 'Ë', 'Î', 'Ï', 'Ô', 'Ö', 'Ù', 'Û', 'Ü', 'Ÿ', 'Ç'])
        
        # Ponctuation et symboles
        chars.extend([' ', '.', ',', ';', ':', '!', '?', '-', '_', '(', ')', '[', ']', '{', '}'])
        chars.extend(['/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '|'])
        chars.extend(['"', "'", '`', '~', '€', '°', '§'])
        
        # Enlever les doublons et trier
        chars = sorted(list(set(chars)))
        
        dict_file = self.config_dir / "character_dict.txt"
        with open(dict_file, 'w', encoding='utf-8') as f:
            for char in chars:
                f.write(char + '\n')
        
        logger.info(f"Dictionnaire de caractères créé: {len(chars)} caractères")
        return str(dict_file)

class PaddleOCRFineTuner:
    """Gestionnaire de fine-tuning pour PaddleOCR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'models/paddleocr_finetuned'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration PaddleOCR
        self.paddle_config = PaddleOCRConfig(str(self.output_dir))
        
        # Initialiser PaddleOCR de base
        self.base_reader = PaddleOCR(use_angle_cls=True, lang='fr')
        
        logger.info(f"PaddleOCR Fine-tuner initialisé - sortie: {self.output_dir}")
    
    def prepare_paddle_dataset(self, dataset_file: str) -> str:
        """Prépare le dataset au format PaddleOCR"""
        logger.info("📊 Préparation du dataset PaddleOCR...")
        
        # Créer le dossier dataset
        dataset_dir = self.output_dir / "paddle_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Charger les données
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Préparer les listes pour l'entraînement
        train_split = int(0.8 * len(data))
        train_data = data[:train_split]
        val_data = data[train_split:]
        
        # Créer les fichiers de liste pour la détection
        det_train_list = []
        det_val_list = []
        
        # Créer les fichiers de liste pour la reconnaissance
        rec_train_list = []
        rec_val_list = []
        
        for split_data, det_list, rec_list in [(train_data, det_train_list, rec_train_list), 
                                               (val_data, det_val_list, rec_val_list)]:
            for item in split_data:
                image_path = item['image_path']
                image_name = Path(image_path).name
                
                # Copier l'image dans le dossier dataset
                dest_image = images_dir / image_name
                if not dest_image.exists():
                    shutil.copy2(image_path, dest_image)
                
                # Format pour la détection
                lines = item.get('lines', [])
                if lines:
                    det_annotation = {
                        'transcription': [line['transcription'] for line in lines],
                        'points': [line['points'] for line in lines]
                    }
                    det_list.append(f"images/{image_name}\t{json.dumps(det_annotation, ensure_ascii=False)}")
                
                # Format pour la reconnaissance
                for line in lines:
                    rec_list.append(f"images/{image_name}\t{line['transcription']}")
        
        # Sauvegarder les listes
        list_files = {
            'train_list.txt': det_train_list,
            'val_list.txt': det_val_list,
            'rec_train_list.txt': rec_train_list,
            'rec_val_list.txt': rec_val_list
        }
        
        for filename, file_list in list_files.items():
            with open(dataset_dir / filename, 'w', encoding='utf-8') as f:
                for line in file_list:
                    f.write(line + '\n')
        
        logger.info(f"✅ Dataset PaddleOCR préparé: {len(train_data)} train, {len(val_data)} val")
        return str(dataset_dir)
    
    def train_detection_model(self, dataset_dir: str) -> str:
        """Entraîne le modèle de détection"""
        logger.info("🎯 Entraînement du modèle de détection...")
        
        # Créer la configuration
        config_file = self.paddle_config.create_detection_config(dataset_dir)
        
        # Commande d'entraînement
        train_script = f"""
import os
import sys
sys.path.append('/path/to/PaddleOCR')
from tools.train import main

if __name__ == '__main__':
    config_path = '{config_file}'
    main(config_path)
"""
        
        script_file = self.output_dir / "train_detection.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(train_script)
        
        logger.info(f"Script d'entraînement créé: {script_file}")
        logger.info("Pour lancer l'entraînement, exécutez:")
        logger.info(f"python {script_file}")
        
        return str(script_file)
    
    def train_recognition_model(self, dataset_dir: str) -> str:
        """Entraîne le modèle de reconnaissance"""
        logger.info("📝 Entraînement du modèle de reconnaissance...")
        
        # Créer la configuration
        config_file = self.paddle_config.create_recognition_config(dataset_dir)
        
        # Commande d'entraînement
        train_script = f"""
import os
import sys
sys.path.append('/path/to/PaddleOCR')
from tools.train import main

if __name__ == '__main__':
    config_path = '{config_file}'
    main(config_path)
"""
        
        script_file = self.output_dir / "train_recognition.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(train_script)
        
        logger.info(f"Script d'entraînement créé: {script_file}")
        logger.info("Pour lancer l'entraînement, exécutez:")
        logger.info(f"python {script_file}")
        
        return str(script_file)
    
    def create_inference_models(self) -> Dict[str, str]:
        """Crée les modèles d'inférence"""
        logger.info("🔄 Création des modèles d'inférence...")
        
        inference_scripts = {}
        
        # Script pour la détection
        det_script = f"""
import os
import sys
sys.path.append('/path/to/PaddleOCR')
from tools.export_model import main

if __name__ == '__main__':
    config_path = '{self.paddle_config.config_dir}/det_config.yml'
    checkpoints_path = '{self.paddle_config.checkpoints_dir}/det/best_accuracy'
    save_inference_dir = '{self.output_dir}/inference/det'
    
    main(config_path, checkpoints_path, save_inference_dir)
"""
        
        det_script_file = self.output_dir / "export_detection.py"
        with open(det_script_file, 'w', encoding='utf-8') as f:
            f.write(det_script)
        inference_scripts['detection'] = str(det_script_file)
        
        # Script pour la reconnaissance
        rec_script = f"""
import os
import sys
sys.path.append('/path/to/PaddleOCR')
from tools.export_model import main

if __name__ == '__main__':
    config_path = '{self.paddle_config.config_dir}/rec_config.yml'
    checkpoints_path = '{self.paddle_config.checkpoints_dir}/rec/best_accuracy'
    save_inference_dir = '{self.output_dir}/inference/rec'
    
    main(config_path, checkpoints_path, save_inference_dir)
"""
        
        rec_script_file = self.output_dir / "export_recognition.py"
        with open(rec_script_file, 'w', encoding='utf-8') as f:
            f.write(rec_script)
        inference_scripts['recognition'] = str(rec_script_file)
        
        return inference_scripts
    
    def test_finetuned_model(self, test_image: str) -> Dict[str, Any]:
        """Teste le modèle fine-tuné"""
        logger.info(f"🧪 Test du modèle sur: {test_image}")
        
        try:
            # Utiliser le modèle de base pour comparaison
            base_results = self.base_reader.ocr(test_image, cls=True)
            
            # TODO: Intégrer le modèle fine-tuné une fois entraîné
            # custom_results = self.custom_reader.ocr(test_image, cls=True)
            
            return {
                'image_path': test_image,
                'base_results': base_results,
                'custom_results': None,  # À implémenter après entraînement
                'comparison': None
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du test: {e}")
            return {'error': str(e)}
    
    def run_complete_training(self, dataset_file: str) -> Dict[str, Any]:
        """Lance l'entraînement complet"""
        logger.info("🚀 DÉMARRAGE DU FINE-TUNING PADDLEOCR")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # 1. Préparer le dataset
            dataset_dir = self.prepare_paddle_dataset(dataset_file)
            
            # 2. Créer les scripts d'entraînement
            det_script = self.train_detection_model(dataset_dir)
            rec_script = self.train_recognition_model(dataset_dir)
            
            # 3. Créer les scripts d'export
            inference_scripts = self.create_inference_models()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 50)
            logger.info("✅ PRÉPARATION TERMINÉE")
            logger.info(f"⏱️ Temps: {total_time:.1f}s")
            
            return {
                'dataset_dir': dataset_dir,
                'detection_script': det_script,
                'recognition_script': rec_script,
                'inference_scripts': inference_scripts,
                'config_dir': str(self.paddle_config.config_dir),
                'output_dir': str(self.output_dir),
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            return {'error': str(e)}

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning PaddleOCR pour factures")
    parser.add_argument('--dataset', required=True, help='Fichier dataset JSON')
    parser.add_argument('--output_dir', default='models/paddleocr_finetuned', help='Dossier de sortie')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'output_dir': args.output_dir
    }
    
    # Créer le fine-tuner
    fine_tuner = PaddleOCRFineTuner(config)
    
    # Lancer l'entraînement
    results = fine_tuner.run_complete_training(args.dataset)
    
    if 'error' not in results:
        print(f"\n🎉 Préparation terminée!")
        print(f"📁 Dataset: {results['dataset_dir']}")
        print(f"🎯 Script détection: {results['detection_script']}")
        print(f"📝 Script reconnaissance: {results['recognition_script']}")
        print("\n📋 Instructions d'utilisation:")
        print("1. Installez PaddleOCR depuis le repository officiel")
        print("2. Lancez les scripts d'entraînement dans l'ordre:")
        print(f"   python {results['detection_script']}")
        print(f"   python {results['recognition_script']}")
    else:
        print(f"❌ Erreur: {results['error']}")

if __name__ == "__main__":
    main()