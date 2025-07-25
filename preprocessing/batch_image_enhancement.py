#!/usr/bin/env python3
"""
Script de traitement batch pour l'amélioration d'images de factures
"""

import argparse
import glob
import json
import os
from datetime import datetime
from image_enhancement import InvoiceImageEnhancer

def main():
    parser = argparse.ArgumentParser(description='Traitement batch d\'amélioration d\'images')
    
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Dossier contenant les images à traiter')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Dossier de sortie pour les images améliorées')
    parser.add_argument('--extensions', '-e', nargs='+', 
                       default=['.png', '.jpg', '.jpeg'],
                       help='Extensions de fichiers autorisées')
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                       help='Traitement parallèle (par défaut)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Nombre de workers pour le parallélisme')
    parser.add_argument('--preserve-quality', action='store_true', default=True,
                       help='Mode conservateur pour préserver la qualité')
    parser.add_argument('--aggressive', action='store_true',
                       help='Mode agressif (désactive preserve-quality)')
    
    args = parser.parse_args()
    
    # Validation des arguments
    if not os.path.exists(args.input_dir):
        print(f"❌ Erreur: Dossier d'entrée non trouvé: {args.input_dir}")
        return
    
    if args.aggressive:
        args.preserve_quality = False
    
    print("🚀 TRAITEMENT BATCH D'IMAGES")
    print("="*50)
    print(f"📁 Dossier source    : {args.input_dir}")
    print(f"📁 Dossier sortie    : {args.output_dir}")
    print(f"🔧 Mode             : {'Conservateur' if args.preserve_quality else 'Agressif'}")
    print(f"⚡ Parallélisme     : {'Oui' if args.parallel else 'Non'} ({args.workers} workers)")
    print(f"📎 Extensions       : {args.extensions}")
    print("="*50)
    
    # Initialisation du enhancer
    enhancer = InvoiceImageEnhancer()
    
    # Lancement du traitement batch
    results = enhancer.enhance_batch_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_extensions=args.extensions,
        preserve_quality=args.preserve_quality,
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    print(f"\n🎉 Traitement terminé en {results['duration']:.1f} secondes")

if __name__ == "__main__":
    main()
#python preprocessing/batch_image_enhancement.py --input-dir "Data/images" --output-dir "Data/processed_images"