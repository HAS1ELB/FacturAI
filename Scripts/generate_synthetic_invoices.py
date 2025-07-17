import os
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import random

# Dossiers
TEMPLATE_DIR = "ids-factures"
OUTPUT_DIR = "synthetic_factures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Faker FR
fake = Faker("fr_FR")

# Générer des données aléatoires
def generate_fake_invoice_data():
    return {
        "date": fake.date(),
        "client": fake.name(),
        "adresse": fake.address().replace("\n", ", "),
        "numero": f"INV-{fake.random_int(1000, 9999)}",
        "montant": f"{round(random.uniform(100.0, 5000.0), 2)} MAD"
    }

# Générer un PDF avec contenu synthétique
def generate_synthetic_pdf(output_path, data, template_index, instance_index):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, f"FACTURE SYNTHÉTIQUE #{template_index}-{instance_index}")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Date : {data['date']}")
    c.drawString(50, height - 120, f"Numéro de facture : {data['numero']}")
    c.drawString(50, height - 140, f"Client : {data['client']}")
    c.drawString(50, height - 160, f"Adresse : {data['adresse']}")
    c.drawString(50, height - 180, f"Montant total : {data['montant']}")

    c.showPage()
    c.save()

# Pipeline
def main():
    templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".pdf")]
    
    for i, template_file in enumerate(templates):
        for j in range(5):  # Générer 5 versions par modèle
            data = generate_fake_invoice_data()
            output_filename = f"facture_synthetique_{i}_{j}.pdf"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            generate_synthetic_pdf(output_path, data, i, j)
    
    print(f"[✔] {len(templates) * 5} factures synthétiques générées dans le dossier '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
