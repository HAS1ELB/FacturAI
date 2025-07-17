import os
from pdf2image import convert_from_path

# Chemins
INPUT_DIR = "Data/ids_factures"
OUTPUT_DIR = "Data/ids_images"
POPPLER_PATH = r"C:/tools/poppler/Library/bin" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_pdfs_to_images():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        for i, img in enumerate(images):
            base_name = os.path.splitext(pdf_file)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_page{i+1}.png")
            img.save(output_path)
            print(f"[✓] Sauvegardé : {output_path}")

if __name__ == "__main__":
    convert_pdfs_to_images()
