import fitz  # PyMuPDF
import os
from pathlib import Path
import argparse

def pdf_to_images(pdf_path, output_dir=None, dpi=300, image_format='PNG'):
    """
    Convertit un fichier PDF en images (une image par page)
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF
        output_dir (str): Dossier de sortie (optionnel)
        dpi (int): Résolution DPI pour les images
        image_format (str): Format d'image ('PNG', 'JPEG', 'WEBP')
    
    Returns:
        list: Liste des chemins des images créées
    """
    
    # Vérifier si le fichier PDF existe
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier PDF '{pdf_path}' n'existe pas.")
    
    # Créer le dossier de sortie
    if output_dir is None:
        output_dir = Path(pdf_path).stem + "_images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Ouvrir le PDF
    pdf_document = fitz.open(pdf_path)
    
    # Liste pour stocker les chemins des images créées
    created_images = []
    
    print(f"Conversion du PDF: {pdf_path}")
    print(f"Nombre de pages: {len(pdf_document)}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Format: {image_format}, DPI: {dpi}")
    print("-" * 50)
    
    # Convertir chaque page en image
    for page_num in range(len(pdf_document)):
        # Obtenir la page
        page = pdf_document.load_page(page_num)
        
        # Définir la matrice de transformation pour le DPI
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 est le DPI par défaut
        
        # Rendre la page en image
        pix = page.get_pixmap(matrix=mat)
        
        # Définir le nom du fichier de sortie
        output_filename = f"page_{page_num + 1:03d}.{image_format.lower()}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Sauvegarder l'image
        if image_format.upper() == 'PNG':
            pix.save(output_path)
        elif image_format.upper() == 'JPEG':
            pix.save(output_path)
        elif image_format.upper() == 'WEBP':
            pix.save(output_path)
        else:
            pix.save(output_path)  # PNG par défaut
        
        created_images.append(output_path)
        print(f"Page {page_num + 1}/{len(pdf_document)} → {output_filename}")
    
    # Fermer le document PDF
    pdf_document.close()
    
    print("-" * 50)
    print(f"Conversion terminée! {len(created_images)} images créées dans '{output_dir}'")
    
    return created_images

def main():
    parser = argparse.ArgumentParser(description='Convertir un PDF en images')
    parser.add_argument('pdf_path', help='Chemin vers le fichier PDF')
    parser.add_argument('-o', '--output', help='Dossier de sortie')
    parser.add_argument('-d', '--dpi', type=int, default=300, help='Résolution DPI (défaut: 300)')
    parser.add_argument('-f', '--format', choices=['PNG', 'JPEG', 'WEBP'], 
                       default='PNG', help='Format d\'image (défaut: PNG)')
    
    args = parser.parse_args()
    
    try:
        pdf_to_images(args.pdf_path, args.output, args.dpi, args.format)
    except Exception as e:
        print(f"Erreur: {e}")

# Exemple d'utilisation directe
if __name__ == "__main__":
    # Utilisation en ligne de commande
    #main()
    
    # Ou utilisation directe dans le code:
    pdf_to_images("2023-02-24.pdf", "images_output", dpi=300, image_format="PNG")