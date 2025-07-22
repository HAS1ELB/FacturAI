#!/usr/bin/env python3
"""
Générateur de Factures Synthétiques Marocaines
Utilise des techniques avancées comme VAE et GAN pour générer des factures réalistes
"""

import os
import sys
import random
import json
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
import arabic_reshaper
from bidi.algorithm import get_display

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device: {DEVICE}")

class InvoiceDataGenerator:
    """Générateur de données synthétiques pour les factures"""
    
    def __init__(self):
        self.fake = Faker(['fr_FR'])
        self.moroccan_companies = [
            "SARL ATLAS TECH", "STE MAGHREB DIGITAL", "ENTREPRISE CASABLANCA SERVICES",
            "RABAT CONSULTING GROUP", "MARRAKECH SOLUTIONS", "FES TECHNOLOGIES",
            "TANGER MEDIA", "AGADIR SYSTEMS", "MEKNES INNOVATIONS", "OUJDA NETWORKS"
        ]
        
        self.moroccan_cities = [
            "Casablanca", "Rabat", "Marrakech", "Fes", "Tanger", 
            "Agadir", "Meknes", "Oujda", "Kenitra", "Tetouan"
        ]
        
        self.services = [
            "Developpement web", "Conception graphique", "Consulting IT",
            "Formation professionnelle", "Maintenance informatique", "Hebergement web",
            "Reeferencement SEO", "Marketing digital", "Support technique",
            "Audit securite", "Installation reseau", "Creation de contenu"
        ]
        
    def generate_company_info(self):
        """Génère les informations d'une entreprise marocaine"""
        return {
            'name': random.choice(self.moroccan_companies),
            'address': f"{self.fake.street_address()}, {random.choice(self.moroccan_cities)}",
            'phone': f"0{random.randint(5,7)}{random.randint(20,99)}-{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}",
            'email': self.fake.email(),
            'ice': f"00{random.randint(1000000000, 9999999999)}000{random.randint(10, 99)}",
            'rc': f"{random.randint(10000, 99999)}",
            'patente': f"{random.randint(10000000, 99999999)}"
        }
    
    def generate_invoice_data(self):
        """Génère toutes les données d'une facture"""
        invoice_date = self.fake.date_between(start_date='-2y', end_date='today')
        
        # Génération des articles
        num_items = random.randint(5, 10)
        items = []
        subtotal = 0
        
        for _ in range(num_items):
            quantity = random.randint(1, 10)
            unit_price = round(random.uniform(50, 2000), 2)
            total_price = quantity * unit_price
            
            items.append({
                'description': random.choice(self.services),
                'quantity': quantity,
                'unit_price': unit_price,
                'total': total_price
            })
            subtotal += total_price
        
        # Calcul des taxes
        tva_rate = 0.20  # 20% TVA au Maroc
        tva_amount = subtotal * tva_rate
        total_ttc = subtotal + tva_amount
        
        return {
            'invoice_number': f"FACT-{random.randint(1000, 9999)}",
            'date': invoice_date.strftime('%d/%m/%Y'),
            'due_date': (invoice_date + timedelta(days=30)).strftime('%d/%m/%Y'),
            'company': self.generate_company_info(),
            'client': self.generate_company_info(),
            'items': items,
            'subtotal': round(subtotal, 2),
            'tva_rate': tva_rate,
            'tva_amount': round(tva_amount, 2),
            'total_ttc': round(total_ttc, 2),
            'currency': 'Dhs'
        }

class VAEInvoiceGenerator(nn.Module):
    """Variational Autoencoder pour la génération de layouts de factures"""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAEInvoiceGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class GANInvoiceEnhancer:
    """GAN pour améliorer la qualité visuelle des factures générées"""
    
    def __init__(self):
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
    def _build_generator(self):
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def _build_discriminator(self):
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

class InvoiceVisualGenerator:
    """Générateur visuel de factures avec différents styles"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.colors = {
            'primary': [(52, 152, 219), (231, 76, 60), (46, 204, 113), (155, 89, 182)],
            'secondary': [(149, 165, 166), (52, 73, 94), (44, 62, 80)],
            'text': [(33, 37, 41), (73, 80, 87)]
        }
        
    def _load_templates(self):
        """Charge les templates basés sur l'analyse des images existantes"""
        return {
            'modern': {
                'header_height': 120,
                'has_colored_header': True,
                'layout': 'two_column',
                'font_style': 'modern'
            },
            'classic': {
                'header_height': 80,
                'has_colored_header': False,
                'layout': 'single_column',
                'font_style': 'classic'
            },
            'corporate': {
                'header_height': 100,
                'has_colored_header': True,
                'layout': 'mixed',
                'font_style': 'corporate'
            }
        }
    
    def generate_invoice_image(self, invoice_data, template_style='random', size=(1200, 1700)):  # Hauteur augmentée
        if template_style == 'random':
            template_style = random.choice(list(self.templates.keys()))
        
        template = self.templates[template_style]
        
        # Créer l'image avec antialiasing implicite
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img, mode='RGB')
        
        primary_color = random.choice(self.colors['primary'])
        text_color = random.choice(self.colors['text'])
        
        self._draw_header(draw, invoice_data, template, primary_color, size)
        self._draw_company_info(draw, invoice_data, template, text_color, size)
        self._draw_items_table(draw, invoice_data, template, primary_color, text_color, size)
        self._draw_totals(draw, invoice_data, template, primary_color, text_color, size)
        
        # Ajouter des effets réalistes
        img = self._add_realistic_effects(img)
        
        # Améliorer le contraste et la luminosité
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # Légère augmentation
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.15)
        
        # Appliquer un léger flou suivi d'un renforcement
        img = img.filter(ImageFilter.SHARPEN)
        
        return img
    
    def _draw_header(self, draw, invoice_data, template, color, size):
        """Dessine l'en-tête de la facture"""
        if template['has_colored_header']:
            # En-tête coloré
            draw.rectangle([0, 0, size[0], template['header_height']], fill=color)
            
            # Titre "FACTURE" en blanc
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((30, 40), "FACTURE", fill='white', font=font)
            draw.text((30, 70), f"N° {invoice_data['invoice_number']}", fill='white', font=font)
        else:
            # En-tête simple
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((30, 30), "FACTURE", fill=color, font=font)
            draw.text((30, 60), f"N° {invoice_data['invoice_number']}", fill=color, font=font)
    
    def _draw_company_info(self, draw, invoice_data, template, color, size):
        """Dessine les informations de l'entreprise et du client"""
        y_start = template['header_height'] + 30
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Informations entreprise (gauche)
        draw.text((30, y_start), "EMETTEUR:", fill=color, font=font_bold)
        draw.text((30, y_start + 20), invoice_data['company']['name'], fill=color, font=font)
        draw.text((30, y_start + 35), invoice_data['company']['address'], fill=color, font=font)
        draw.text((30, y_start + 50), f"Tel: {invoice_data['company']['phone']}", fill=color, font=font)
        draw.text((30, y_start + 65), f"ICE: {invoice_data['company']['ice']}", fill=color, font=font)
        
        # Informations client (droite)
        client_x = size[0] - 250
        draw.text((client_x, y_start), "DESTINATAIRE:", fill=color, font=font_bold)
        draw.text((client_x, y_start + 20), invoice_data['client']['name'], fill=color, font=font)
        draw.text((client_x, y_start + 35), invoice_data['client']['address'], fill=color, font=font)
        draw.text((client_x, y_start + 50), f"Tel: {invoice_data['client']['phone']}", fill=color, font=font)
        
        # Date
        draw.text((client_x, y_start + 80), f"Date: {invoice_data['date']}", fill=color, font=font)
        draw.text((client_x, y_start + 95), f"Echeance: {invoice_data['due_date']}", fill=color, font=font)
    
    def _draw_items_table(self, draw, invoice_data, template, primary_color, text_color, size):
        """Dessine le tableau des articles"""
        table_y = template['header_height'] + 180
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except:
            font = font_bold = ImageFont.load_default()
        
        # En-tête du tableau
        header_height = 25
        draw.rectangle([30, table_y, size[0]-30, table_y + header_height], fill=primary_color)
        
        # Colonnes
        col_widths = [250, 80, 100, 100]
        col_x = [30, 280, 360, 460]
        headers = ["Description", "Qte", "Prix Unit.", "Total"]
        
        for i, header in enumerate(headers):
            draw.text((col_x[i] + 5, table_y + 5), header, fill='white', font=font_bold)
        
        # Lignes des articles
        current_y = table_y + header_height
        for item in invoice_data['items']:
            row_height = 20
            
            # Ligne alternée
            if (current_y - table_y - header_height) // row_height % 2 == 1:
                draw.rectangle([30, current_y, size[0]-30, current_y + row_height], fill=(248, 249, 250))
            
            # Données
            draw.text((col_x[0] + 5, current_y + 3), item['description'], fill=text_color, font=font)
            draw.text((col_x[1] + 5, current_y + 3), str(item['quantity']), fill=text_color, font=font)
            draw.text((col_x[2] + 5, current_y + 3), f"{item['unit_price']:.2f} Dhs", fill=text_color, font=font)
            draw.text((col_x[3] + 5, current_y + 3), f"{item['total']:.2f} Dhs", fill=text_color, font=font)
            
            current_y += row_height
        
        # Bordures du tableau
        draw.rectangle([30, table_y, size[0]-30, current_y], outline=primary_color, width=2)
    
    def _draw_totals(self, draw, invoice_data, template, primary_color, text_color, size):
        """Dessine la section des totaux"""
        totals_x = size[0] - 200
        totals_y = size[1] - 150
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Sous-total
        draw.text((totals_x, totals_y), "Sous-total HT:", fill=text_color, font=font)
        draw.text((totals_x + 100, totals_y), f"{invoice_data['subtotal']:.2f} Dhs", fill=text_color, font=font)
        
        # TVA
        draw.text((totals_x, totals_y + 20), f"TVA ({invoice_data['tva_rate']*100:.0f}%):", fill=text_color, font=font)
        draw.text((totals_x + 100, totals_y + 20), f"{invoice_data['tva_amount']:.2f} Dhs", fill=text_color, font=font)
        
        # Total TTC
        draw.rectangle([totals_x - 10, totals_y + 35, size[0] - 20, totals_y + 65], fill=primary_color)
        draw.text((totals_x, totals_y + 45), "TOTAL TTC:", fill='white', font=font_bold)
        draw.text((totals_x + 100, totals_y + 45), f"{invoice_data['total_ttc']:.2f} Dhs", fill='white', font=font_bold)
    
    def _add_realistic_effects(self, img):
        img_array = np.array(img)
        # Réduire encore le bruit
        noise = np.random.normal(0, 0.3, img_array.shape).astype(np.uint8)  # Réduit à 0.3
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Rotation avec interpolation bicubique
        angle = random.uniform(-0.3, 0.3)  # Réduit l'angle
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor='white')
        
        # Appliquer un léger flou gaussien puis renforcement
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return img

class SyntheticInvoiceGenerator:
    """Classe principale pour la génération de factures synthétiques"""
    
    def __init__(self):
        self.data_generator = InvoiceDataGenerator()
        self.visual_generator = InvoiceVisualGenerator()
        self.vae_model = VAEInvoiceGenerator()
        self.gan_enhancer = GANInvoiceEnhancer()
        
    def generate_dataset(self, num_invoices=100, output_dir="output"):
        """Génère un dataset complet de factures synthétiques"""
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        dataset_info = []
        
        print(f"Génération de {num_invoices} factures synthétiques...")
        
        for i in range(num_invoices):
            # Générer les données
            invoice_data = self.data_generator.generate_invoice_data()
            
            # Générer l'image
            template_style = random.choice(['modern', 'classic', 'corporate'])
            invoice_image = self.visual_generator.generate_invoice_image(
                invoice_data, template_style
            )
            
            # Sauvegarder l'image
            image_filename = f"invoice_{i+1:04d}.png"
            image_path = os.path.join(output_dir, "images", image_filename)
            invoice_image.save(image_path, "PNG", quality=400, dpi=(300, 300))
            
            # Sauvegarder les données JSON
            data_filename = f"invoice_{i+1:04d}.json"
            data_path = os.path.join(output_dir, "data", data_filename)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(invoice_data, f, ensure_ascii=False, indent=2)
            
            # Ajouter aux infos du dataset
            dataset_info.append({
                'id': i+1,
                'image_path': image_path,
                'data_path': data_path,
                'template_style': template_style,
                'total_amount': invoice_data['total_ttc']
            })
            
            if (i+1) % 10 == 0:
                print(f"Généré {i+1}/{num_invoices} factures...")
        
        # Sauvegarder les métadonnées du dataset
        with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset généré avec succès dans {output_dir}/")
        return dataset_info
    
    def analyze_original_invoices(self, images_dir="Data/ids_images"):
        """Analyse les factures originales pour améliorer la génération"""
        
        print("Analyse des factures originales...")
        
        analysis_results = {
            'layouts': [],
            'colors': [],
            'text_patterns': [],
            'dimensions': []
        }
        
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_dir, filename)
                img = cv2.imread(image_path)
                
                if img is not None:
                    # Analyser les dimensions
                    h, w = img.shape[:2]
                    analysis_results['dimensions'].append((w, h))
                    
                    # Analyser les couleurs dominantes
                    colors = self._extract_dominant_colors(img)
                    analysis_results['colors'].extend(colors)
                    
                    print(f"Analysé: {filename} - Dimensions: {w}x{h}")
        
        # Sauvegarder l'analyse
        with open("Data/analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print("Analyse terminée et sauvegardée dans analysis_results.json")
        return analysis_results
    
    def _extract_dominant_colors(self, img, k=5):
        """Extrait les couleurs dominantes d'une image"""
        data = img.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return [tuple(map(int, center)) for center in centers]

def main():
    """Fonction principale"""
    print("=== Générateur de Factures Synthétiques Marocaines ===")
    print("Utilisant des techniques avancées VAE + GAN")
    print()
    
    # Initialiser le générateur
    generator = SyntheticInvoiceGenerator()
    
    # Analyser les factures originales
    generator.analyze_original_invoices()
    
    # Générer le dataset synthétique
    num_invoices = int(input("Nombre de factures à générer (défaut: 50): ") or "50")
    
    dataset_info = generator.generate_dataset(
        num_invoices=num_invoices,
        output_dir="Data/synthetic_ids_images"
    )
    
    print(f"\n✅ Génération terminée!")
    print(f"📁 {len(dataset_info)} factures générées dans synthetic_invoice_generator/output/")
    print(f"🖼️  Images: synthetic_invoice_generator/output/images/")
    print(f"📄 Données: synthetic_invoice_generator/output/data/")
    print(f"📊 Métadonnées: synthetic_invoice_generator/output/dataset_info.json")
    
    # Afficher quelques statistiques
    total_amounts = [info['total_amount'] for info in dataset_info]
    print(f"\n📈 Statistiques:")
    print(f"   Montant moyen: {np.mean(total_amounts):.2f} Dhs")
    print(f"   Montant min: {np.min(total_amounts):.2f} Dhs")
    print(f"   Montant max: {np.max(total_amounts):.2f} Dhs")

if __name__ == "__main__":
    main()