#!/usr/bin/env python3
"""
Générateur de Factures Synthétiques Marocaines Complexes
Génère des factures avec des layouts professionnels, logos, tableaux complexes, etc.
"""

import os
import sys
import random
import json
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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
from io import BytesIO
import base64

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device: {DEVICE}")

class ComplexInvoiceDataGenerator:
    """Générateur de données complexes pour les factures marocaines"""
    
    def __init__(self):
        self.fake = Faker(['fr_FR'])
        
        # Entreprises marocaines réalistes avec secteurs
        self.moroccan_companies = {
            'IT': [
                "ATLAS TECH SOLUTIONS SARL", "MAGHREB DIGITAL SERVICES", 
                "CASABLANCA IT CONSULTING", "RABAT SYSTEMS & NETWORKS",
                "MARRAKECH WEB SOLUTIONS", "FES CLOUD TECHNOLOGIES"
            ],
            'CONSTRUCTION': [
                "ENTREPRISE ATLAS CONSTRUCTION", "MAGHREB BATIMENT & TRAVAUX",
                "CASABLANCA CONSTRUCTION GROUP", "RABAT BUILDING SOLUTIONS",
                "MARRAKECH ARCHITECTURE & DESIGN", "FES CONSTRUCTION MODERNE"
            ],
            'COMMERCE': [
                "ATLAS TRADING COMPANY", "MAGHREB IMPORT EXPORT",
                "CASABLANCA COMMERCIAL CENTER", "RABAT BUSINESS SOLUTIONS",
                "MARRAKECH TRADING GROUP", "FES COMMERCIAL SERVICES"
            ],
            'SERVICES': [
                "ATLAS CONSULTING GROUP", "MAGHREB SERVICES PROFESSIONNELS",
                "CASABLANCA CONSEIL & EXPERTISE", "RABAT SERVICES TECHNIQUES",
                "MARRAKECH SOLUTIONS BUSINESS", "FES EXPERTISE CONSEIL"
            ]
        }
        
        self.moroccan_regions = {
            'CASABLANCA-SETTAT': ['Casablanca', 'Settat', 'Mohammedia', 'El Jadida', 'Berrechid'],
            'RABAT-SALE-KENITRA': ['Rabat', 'Sale', 'Kenitra', 'Temara', 'Skhirate'],
            'MARRAKECH-SAFI': ['Marrakech', 'Safi', 'Essaouira', 'Youssoufia', 'Kelaa des Sraghna'],
            'FES-MEKNES': ['Fes', 'Meknes', 'Taza', 'Ifrane', 'Khenifra'],
            'TANGER-TETOUAN-AL HOCEIMA': ['Tanger', 'Tétouan', 'Al Hoceima', 'Chefchaouen', 'Larache']
        }
        
        # Services complexes avec catégories
        self.complex_services = {
            'DEVELOPMENT': [
                "Developpement d'application web sur mesure",
                "Creation de plateforme e-commerce avec CMS",
                "Developpement d'API REST et integrations",
                "Application mobile native iOS/Android",
                "Systeme de gestion documentaire (GED)",
                "Plateforme de formation en ligne (LMS)"
            ],
            'CONSULTING': [
                "Audit de securite informatique complet",
                "Conseil en transformation digitale",
                "Etude de faisabilite technique",
                "Formation equipe developpement",
                "Accompagnement mise en place DevOps",
                "Conseil en architecture logicielle"
            ],
            'MAINTENANCE': [
                "Maintenance évolutive application",
                "Support technique niveau 2 et 3",
                "Supervision et monitoring 24/7",
                "Sauvegarde et archivage donnees",
                "Mise à jour securite systeme",
                "Optimisation performance serveurs"
            ],
            'INFRASTRUCTURE': [
                "Installation reseau entreprise",
                "Configuration serveurs dedies",
                "Mise en place solution cloud",
                "Deploiement infrastructure VPN",
                "Installation systeme telephonie IP",
                "Configuration firewall enterprise"
            ]
        }
        
        # Secteurs d'activité réalistes
        self.business_sectors = [
            "Technologies de l'information",
            "Conseil et services aux entreprises",
            "Construction et BTP",
            "Commerce et distribution",
            "Industrie et manufacturing",
            "Transport et logistique",
            "Sante et services medicaux",
            "Education et formation",
            "Tourisme et hotellerie",
            "Agriculture et agroalimentaire"
        ]
        
    def generate_complex_company_info(self, sector=None):
        """Génère des informations d'entreprise complexes"""
        if sector is None:
            sector = random.choice(list(self.moroccan_companies.keys()))
        
        region = random.choice(list(self.moroccan_regions.keys()))
        city = random.choice(self.moroccan_regions[region])
        
        # Numéro de téléphone réaliste marocain
        phone_prefix = random.choice(['05', '06', '07'])
        phone_number = f"+212 {phone_prefix}{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}"
        
        # Adresse plus détaillée
        street_types = ['Avenue', 'Boulevard', 'Rue', 'Place', 'Quartier']
        street_names = ['Mohammed V', 'Hassan II', 'Al Massira', 'Zerktouni', 'Moulay Youssef', 'Anfa']
        
        address = f"{random.choice(street_types)} {random.choice(street_names)}, {self.fake.building_number()}, {city} {random.randint(10000, 99999)}"
        
        return {
            'name': random.choice(self.moroccan_companies[sector]),
            'sector': random.choice(self.business_sectors),
            'address': address,
            'city': city,
            'region': region,
            'phone': phone_number,
            'fax': f"+212 5{random.randint(22,29)}-{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}",
            'email': self.fake.email(),
            'website': f"www.{self.fake.domain_name()}",
            'ice': f"00{random.randint(1000000000, 9999999999)}000{random.randint(10, 99)}",
            'rc': f"{random.randint(10000, 99999)}",
            'patente': f"{random.randint(10000000, 99999999)}",
            'if': f"{random.randint(10000000, 99999999)}",  # Identifiant fiscal
            'cnss': f"{random.randint(1000000, 9999999)}",  # CNSS
            'capital': f"{random.randint(100000, 10000000)} DH",
            'rib': f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(1000000000, 9999999999)} {random.randint(10, 99)}"
        }
    
    def generate_complex_invoice_data(self):
        """Génère des données de facture complexes"""
        invoice_date = self.fake.date_between(start_date='-2y', end_date='today')
        
        # Informations de base
        invoice_number = f"FACT-{random.randint(2024, 2025)}-{random.randint(1000, 9999)}"
        
        # Type de facture
        invoice_types = ['FACTURE', 'FACTURE PROFORMA', 'DEVIS', 'FACTURE D\'ACOMPTE', 'FACTURE DE SOLDE']
        invoice_type = random.choice(invoice_types)
        
        # Conditions de paiement
        payment_terms = [
            "Payable a 30 jours fin de mois",
            "Payable a réception",
            "Payable a 15 jours",
            "Payable a 60 jours",
            "Payable comptant",
            "Payable a 30 jours nets"
        ]
        
        # Modes de paiement
        payment_methods = [
            "Virement bancaire",
            "Cheque",
            "Especes",
            "Carte bancaire",
            "Lettre de change",
            "Traite"
        ]
        
        # Générer des articles complexes avec catégories
        categories = list(self.complex_services.keys())
        num_categories = random.randint(2, 4)
        selected_categories = random.sample(categories, num_categories)
        
        items = []
        subtotal = 0
        
        for category in selected_categories:
            num_items_in_category = random.randint(2, 5)
            services = random.sample(self.complex_services[category], 
                                   min(num_items_in_category, len(self.complex_services[category])))
            
            for service in services:
                # Unités variées
                units = ['H', 'Jour', 'Forfait', 'Mois', 'Licence', 'Unite']
                unit = random.choice(units)
                
                if unit in ['H', 'Jour']:
                    quantity = random.randint(8, 160)  # 1 à 20 jours
                    unit_price = random.uniform(200, 800)  # Prix horaire/jour
                elif unit == 'Mois':
                    quantity = random.randint(1, 12)
                    unit_price = random.uniform(5000, 25000)  # Prix mensuel
                elif unit == 'Forfait':
                    quantity = 1
                    unit_price = random.uniform(10000, 100000)  # Prix forfaitaire
                else:
                    quantity = random.randint(1, 10)
                    unit_price = random.uniform(500, 5000)
                
                # Remise parfois
                discount_rate = 0
                if random.random() < 0.3:  # 30% de chance d'avoir une remise
                    discount_rate = random.uniform(0.05, 0.20)  # 5% à 20%
                
                total_before_discount = quantity * unit_price
                discount_amount = total_before_discount * discount_rate
                total_after_discount = total_before_discount - discount_amount
                
                item = {
                    'category': category,
                    'description': service,
                    'reference': f"REF-{random.randint(1000, 9999)}",
                    'quantity': quantity,
                    'unit': unit,
                    'unit_price': round(unit_price, 2),
                    'discount_rate': round(discount_rate, 2),
                    'discount_amount': round(discount_amount, 2),
                    'total_before_discount': round(total_before_discount, 2),
                    'total': round(total_after_discount, 2)
                }
                
                items.append(item)
                subtotal += total_after_discount
        
        # Calculs complexes
        # Remise globale parfois
        global_discount_rate = 0
        if random.random() < 0.2:  # 20% de chance
            global_discount_rate = random.uniform(0.05, 0.15)
        
        global_discount_amount = subtotal * global_discount_rate
        subtotal_after_discount = subtotal - global_discount_amount
        
        # TVA (peut être différente selon les services)
        tva_rates = [0.20, 0.14, 0.10, 0.07]  # Différents taux de TVA au Maroc
        tva_rate = random.choice(tva_rates)
        tva_amount = subtotal_after_discount * tva_rate
        
        # Timbre fiscal
        timbre_fiscal = 20.00  # Timbre fiscal standard
        
        # Retenue à la source parfois
        retenue_source_rate = 0
        retenue_source_amount = 0
        if random.random() < 0.3:  # 30% de chance
            retenue_source_rate = random.choice([0.10, 0.15, 0.30])  # Taux de retenue
            retenue_source_amount = subtotal_after_discount * retenue_source_rate
        
        total_ttc = subtotal_after_discount + tva_amount + timbre_fiscal - retenue_source_amount
        
        # Informations bancaires pour le paiement
        bank_info = {
            'bank_name': random.choice(['Attijariwafa Bank', 'BMCE Bank', 'Banque Populaire', 'BMCI', 'CIH Bank']),
            'account_holder': None,  # Sera rempli avec le nom de l'entreprise
            'rib': f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(1000000000, 9999999999)} {random.randint(10, 99)}",
            'swift': f"BMCE{random.choice(['MACM', 'MAMC', 'MAXX'])}"
        }
        
        return {
            'invoice_type': invoice_type,
            'invoice_number': invoice_number,
            'date': invoice_date.strftime('%d/%m/%Y'),
            'due_date': (invoice_date + timedelta(days=random.randint(15, 60))).strftime('%d/%m/%Y'),
            'payment_terms': random.choice(payment_terms),
            'payment_method': random.choice(payment_methods),
            'company': self.generate_complex_company_info(),
            'client': self.generate_complex_company_info(),
            'items': items,
            'subtotal': round(subtotal, 2),
            'global_discount_rate': round(global_discount_rate, 2),
            'global_discount_amount': round(global_discount_amount, 2),
            'subtotal_after_discount': round(subtotal_after_discount, 2),
            'tva_rate': tva_rate,
            'tva_amount': round(tva_amount, 2),
            'timbre_fiscal': timbre_fiscal,
            'retenue_source_rate': retenue_source_rate,
            'retenue_source_amount': round(retenue_source_amount, 2),
            'total_ttc': round(total_ttc, 2),
            'currency': 'DH',
            'bank_info': bank_info,
            'notes': self.generate_invoice_notes(),
            'conditions': self.generate_terms_conditions()
        }
    
    def generate_invoice_notes(self):
        """Génère des notes pour la facture"""
        notes_options = [
            "Merci de votre confiance. Nous restons a votre disposition pour tout renseignement.",
            "Facture a regler selon les conditions convenues. Penalites de retard applicables.",
            "Prestations realisees conformement au cahier des charges valide.",
            "TVA recuperable selon la legislation en vigueur.",
            "Reglement par virement bancaire souhaite. Coordonnees ci-dessous.",
            "Facture etablie selon les conditions generales de vente.",
            "Delai de paiement conforme aux dispositions legales.",
            "Prestation de qualite conforme aux standards ISO."
        ]
        
        return random.choice(notes_options)
    
    def generate_terms_conditions(self):
        """Génère les conditions générales"""
        return [
            "Tout retard de paiement entraînera des penalites de 3% par mois.",
            "Nos conditions genrales de vente sont disponibles sur demande.",
            "Reclamations a formuler par ecrit dans les 8 jours.",
            "Competence juridique : Tribunaux de Casablanca.",
            "Prestations soumises a la TVA au taux en vigueur."
        ]

class LogoGenerator:
    """Générateur de logos simples pour les factures"""
    
    def __init__(self):
        self.colors = [
            (52, 152, 219), (231, 76, 60), (46, 204, 113), 
            (155, 89, 182), (241, 196, 15), (230, 126, 34)
        ]
    
    def generate_simple_logo(self, company_name, size=(100, 60)):
        """Génère un logo simple basé sur le nom de l'entreprise"""
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Couleur aléatoire
        color = random.choice(self.colors)
        
        # Forme géométrique simple
        shapes = ['circle', 'rectangle', 'triangle']
        shape = random.choice(shapes)
        
        # Dessiner la forme
        margin = 10
        if shape == 'circle':
            draw.ellipse([margin, margin, size[0]-margin, size[1]-margin], fill=color)
        elif shape == 'rectangle':
            draw.rectangle([margin, margin, size[0]-margin, size[1]-margin], fill=color)
        elif shape == 'triangle':
            points = [
                (size[0]//2, margin),
                (margin, size[1]-margin),
                (size[0]-margin, size[1]-margin)
            ]
            draw.polygon(points, fill=color)
        
        # Ajouter initiales
        initials = ''.join([word[0] for word in company_name.split()[:2]])
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Texte blanc au centre
        text_bbox = draw.textbbox((0, 0), initials, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (size[0] - text_width) // 2
        text_y = (size[1] - text_height) // 2
        
        draw.text((text_x, text_y), initials, fill='white', font=font)
        
        return img

class ComplexInvoiceVisualGenerator:
    """Générateur visuel de factures complexes"""
    
    def __init__(self):
        self.logo_generator = LogoGenerator()
        self.templates = self._load_complex_templates()
        self.color_schemes = self._load_color_schemes()
        
    def _load_complex_templates(self):
        """Charge les templates complexes"""
        return {
            'executive': {
                'has_logo': True,
                'has_watermark': True,
                'header_style': 'gradient',
                'table_style': 'modern',
                'footer_style': 'detailed',
                'color_scheme': 'professional'
            },
            'modern_corporate': {
                'has_logo': True,
                'has_watermark': False,
                'header_style': 'clean',
                'table_style': 'striped',
                'footer_style': 'compact',
                'color_scheme': 'modern'
            },
            'premium': {
                'has_logo': True,
                'has_watermark': True,
                'header_style': 'luxury',
                'table_style': 'premium',
                'footer_style': 'detailed',
                'color_scheme': 'premium'
            }
        }
    
    def _load_color_schemes(self):
        """Charge les schémas de couleurs"""
        return {
            'professional': {
                'primary': (52, 152, 219),
                'secondary': (44, 62, 80),
                'accent': (241, 196, 15),
                'text': (33, 37, 41),
                'light': (248, 249, 250),
                'border': (206, 212, 218)
            },
            'modern': {
                'primary': (46, 204, 113),
                'secondary': (39, 174, 96),
                'accent': (26, 188, 156),
                'text': (44, 62, 80),
                'light': (245, 246, 250),
                'border': (189, 195, 199)
            },
            'classic': {
                'primary': (52, 73, 94),
                'secondary': (44, 62, 80),
                'accent': (149, 165, 166),
                'text': (33, 37, 41),
                'light': (250, 250, 250),
                'border': (149, 165, 166)
            },
            'minimal': {
                'primary': (73, 80, 87),
                'secondary': (108, 117, 125),
                'accent': (206, 212, 218),
                'text': (33, 37, 41),
                'light': (255, 255, 255),
                'border': (222, 226, 230)
            },
            'premium': {
                'primary': (155, 89, 182),
                'secondary': (142, 68, 173),
                'accent': (231, 76, 60),
                'text': (44, 62, 80),
                'light': (253, 242, 248),
                'border': (195, 155, 211)
            }
        }
    
    def generate_complex_invoice_image(self, invoice_data, template_style='random', size=(1200, 2000)):
        """Génère une image de facture complexe"""
        if template_style == 'random':
            template_style = random.choice(list(self.templates.keys()))
        
        template = self.templates[template_style]
        colors = self.color_schemes[template['color_scheme']]
        
        # Créer l'image
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        
        # Variables de position
        current_y = 0
        margin = 40
        
        # Dessiner le filigrane si nécessaire
        if template['has_watermark']:
            self._draw_watermark(draw, size, colors)
        
        # Dessiner l'en-tête
        current_y = self._draw_complex_header(draw, invoice_data, template, colors, size, current_y)
        
        # Dessiner les informations des entreprises
        current_y = self._draw_company_details(draw, invoice_data, template, colors, size, current_y)
        
        # Dessiner les détails de la facture
        current_y = self._draw_invoice_details(draw, invoice_data, template, colors, size, current_y)
        
        # Dessiner le tableau des articles
        current_y = self._draw_complex_items_table(draw, invoice_data, template, colors, size, current_y)
        
        # Dessiner les totaux
        current_y = self._draw_complex_totals(draw, invoice_data, template, colors, size, current_y)
        
        # Dessiner le pied de page
        self._draw_complex_footer(draw, invoice_data, template, colors, size)
        
        # Appliquer des effets réalistes
        img = self._apply_realistic_effects(img)
        
        return img
    
    def _draw_watermark(self, draw, size, colors):
        """Dessine un filigrane"""
        # Créer une image pour le filigrane
        watermark = Image.new('RGBA', (300, 100), (255, 255, 255, 0))
        wm_draw = ImageDraw.Draw(watermark)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # Dessiner le texte du filigrane
        wm_draw.text((10, 30), "FACTURE", fill=(*colors['accent'], 30), font=font)
        
        # Faire une rotation
        watermark = watermark.rotate(45, expand=True)
        
        # Coller le filigrane sur l'image principale
        # Calculer la position pour centrer
        wm_width, wm_height = watermark.size
        x = (size[0] - wm_width) // 2
        y = (size[1] - wm_height) // 2
        
        # Créer une image temporaire pour le blend
        temp_img = Image.new('RGB', size, 'white')
        temp_img.paste(watermark, (x, y), watermark)
    
    def _draw_complex_header(self, draw, invoice_data, template, colors, size, current_y):
        """Dessine un en-tête complexe"""
        header_height = 140
        
        # Fond d'en-tête selon le style
        if template['header_style'] == 'gradient':
            # Simuler un gradient avec des rectangles
            for i in range(header_height):
                alpha = 1 - (i / header_height) * 0.3
                color = tuple(int(c * alpha) for c in colors['primary'])
                draw.rectangle([0, current_y + i, size[0], current_y + i + 1], fill=color)
        elif template['header_style'] == 'clean':
            draw.rectangle([0, current_y, size[0], current_y + header_height], fill=colors['primary'])
        elif template['header_style'] == 'luxury':
            # Bordure dorée
            draw.rectangle([0, current_y, size[0], current_y + header_height], fill=colors['primary'])
            draw.rectangle([0, current_y, size[0], current_y + 5], fill=colors['accent'])
            draw.rectangle([0, current_y + header_height - 5, size[0], current_y + header_height], fill=colors['accent'])
        
        # Logo si nécessaire
        if template['has_logo']:
            logo = self.logo_generator.generate_simple_logo(invoice_data['company']['name'])
            # Convertir PIL en format compatible
            logo_array = np.array(logo)
            logo_pil = Image.fromarray(logo_array)
            
            # Coller le logo (simulé ici par un rectangle)
            draw.rectangle([40, current_y + 20, 140, current_y + 80], fill='white')
            draw.rectangle([45, current_y + 25, 135, current_y + 75], fill=colors['accent'])
        
        # Titre de la facture
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            title_font = subtitle_font = ImageFont.load_default()
        
        title_x = 200 if template['has_logo'] else 40
        draw.text((title_x, current_y + 30), invoice_data['invoice_type'], fill='white', font=title_font)
        draw.text((title_x, current_y + 70), f"N° {invoice_data['invoice_number']}", fill='white', font=subtitle_font)
        
        # Date à droite
        date_x = size[0] - 250
        draw.text((date_x, current_y + 30), f"Date: {invoice_data['date']}", fill='white', font=subtitle_font)
        draw.text((date_x, current_y + 50), f"Echeance: {invoice_data['due_date']}", fill='white', font=subtitle_font)
        
        return current_y + header_height + 30
    
    def _draw_company_details(self, draw, invoice_data, template, colors, size, current_y):
        """Dessine les détails des entreprises"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Colonnes pour émetteur et destinataire
        left_col_width = size[0] // 2 - 60
        right_col_x = size[0] // 2 + 20
        
        # Cadres
        draw.rectangle([40, current_y, left_col_width, current_y + 200], outline=colors['border'], width=2)
        draw.rectangle([right_col_x, current_y, size[0] - 40, current_y + 200], outline=colors['border'], width=2)
        
        # En-têtes
        draw.rectangle([40, current_y, left_col_width, current_y + 30], fill=colors['light'])
        draw.rectangle([right_col_x, current_y, size[0] - 40, current_y + 30], fill=colors['light'])
        
        draw.text((50, current_y + 8), "ÉMETTEUR", fill=colors['primary'], font=font_bold)
        draw.text((right_col_x + 10, current_y + 8), "DESTINATAIRE", fill=colors['primary'], font=font_bold)
        
        # Informations émetteur
        y_offset = current_y + 40
        company = invoice_data['company']
        draw.text((50, y_offset), company['name'], fill=colors['text'], font=font_bold)
        draw.text((50, y_offset + 15), f"Secteur: {company['sector']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 30), company['address'], fill=colors['text'], font=font)
        draw.text((50, y_offset + 45), f"Tel: {company['phone']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 60), f"Email: {company['email']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 75), f"Site: {company['website']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 95), f"ICE: {company['ice']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 110), f"RC: {company['rc']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 125), f"Patente: {company['patente']}", fill=colors['text'], font=font)
        draw.text((50, y_offset + 140), f"Capital: {company['capital']}", fill=colors['text'], font=font)
        
        # Informations destinataire
        client = invoice_data['client']
        draw.text((right_col_x + 10, y_offset), client['name'], fill=colors['text'], font=font_bold)
        draw.text((right_col_x + 10, y_offset + 15), f"Secteur: {client['sector']}", fill=colors['text'], font=font)
        draw.text((right_col_x + 10, y_offset + 30), client['address'], fill=colors['text'], font=font)
        draw.text((right_col_x + 10, y_offset + 45), f"Tel: {client['phone']}", fill=colors['text'], font=font)
        draw.text((right_col_x + 10, y_offset + 60), f"Email: {client['email']}", fill=colors['text'], font=font)
        draw.text((right_col_x + 10, y_offset + 75), f"ICE: {client['ice']}", fill=colors['text'], font=font)
        draw.text((right_col_x + 10, y_offset + 95), f"RC: {client['rc']}", fill=colors['text'], font=font)
        
        return current_y + 220
    
    def _draw_invoice_details(self, draw, invoice_data, template, colors, size, current_y):
        """Dessine les détails de la facture"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Cadre pour les détails
        detail_height = 80
        draw.rectangle([40, current_y, size[0] - 40, current_y + detail_height], outline=colors['border'], width=2)
        draw.rectangle([40, current_y, size[0] - 40, current_y + 25], fill=colors['light'])
        
        draw.text((50, current_y + 5), "DETAILS DE LA FACTURE", fill=colors['primary'], font=font_bold)
        
        # Détails sur deux colonnes
        y_pos = current_y + 35
        draw.text((50, y_pos), f"Conditions de paiement: {invoice_data['payment_terms']}", fill=colors['text'], font=font)
        draw.text((50, y_pos + 15), f"Mode de paiement: {invoice_data['payment_method']}", fill=colors['text'], font=font)
        
        # Colonne droite
        right_x = size[0] - 300
        draw.text((right_x, y_pos), f"Devise: {invoice_data['currency']}", fill=colors['text'], font=font)
        draw.text((right_x, y_pos + 15), f"Taux TVA: {invoice_data['tva_rate']*100:.0f}%", fill=colors['text'], font=font)
        
        return current_y + detail_height + 20
    
    def _draw_complex_items_table(self, draw, invoice_data, template, colors, size, current_y):
        """Dessine un tableau d'articles complexe"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Définir les colonnes
        col_widths = [70, 300, 60, 50, 80, 60, 80]  # Référence, Description, Qté, Unité, Prix Unit., Remise, Total HT
        col_positions = [40]
        for width in col_widths[:-1]:
            col_positions.append(col_positions[-1] + width)
        
        headers = ["Ref", "Description", "Qte", "Unite", "Prix Unit.", "Remise", "Total HT"]
        
        # En-tête du tableau
        header_height = 30
        table_width = sum(col_widths)
        
        # Fond de l'en-tête
        draw.rectangle([40, current_y, 40 + table_width, current_y + header_height], fill=colors['primary'])
        
        # Texte des en-têtes
        for i, header in enumerate(headers):
            draw.text((col_positions[i] + 5, current_y + 8), header, fill='white', font=font_bold)
        
        # Lignes de séparation verticales dans l'en-tête
        for pos in col_positions[1:]:
            draw.line([pos, current_y, pos, current_y + header_height], fill='white', width=1)
        
        current_y += header_height
        
        # Regrouper les articles par catégorie
        categories = {}
        for item in invoice_data['items']:
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # Dessiner les articles par catégorie
        for category, items in categories.items():
            # En-tête de catégorie
            draw.rectangle([40, current_y, 40 + table_width, current_y + 25], fill=colors['light'])
            draw.text((45, current_y + 5), f"CATEGORIE: {category}", fill=colors['primary'], font=font_bold)
            current_y += 25
            
            # Articles de la catégorie
            for row_idx, item in enumerate(items):
                row_height = 35
                
                # Alternance des couleurs
                if row_idx % 2 == 1:
                    draw.rectangle([40, current_y, 40 + table_width, current_y + row_height], fill=(248, 249, 250))
                
                # Bordures
                draw.rectangle([40, current_y, 40 + table_width, current_y + row_height], outline=colors['border'], width=1)
                
                # Données
                y_text = current_y + 8
                draw.text((col_positions[0] + 2, y_text), item['reference'], fill=colors['text'], font=font)
                
                # Description avec retour à la ligne si nécessaire
                desc_lines = self._wrap_text(item['description'], 35)
                for i, line in enumerate(desc_lines[:2]):  # Max 2 lignes
                    draw.text((col_positions[1] + 2, y_text + i*12), line, fill=colors['text'], font=font)
                
                draw.text((col_positions[2] + 2, y_text), str(item['quantity']), fill=colors['text'], font=font)
                draw.text((col_positions[3] + 2, y_text), item['unit'], fill=colors['text'], font=font)
                draw.text((col_positions[4] + 2, y_text), f"{item['unit_price']:.2f}", fill=colors['text'], font=font)
                
                # Remise
                if item['discount_rate'] > 0:
                    draw.text((col_positions[5] + 2, y_text), f"{item['discount_rate']*100:.0f}%", fill=colors['accent'], font=font)
                else:
                    draw.text((col_positions[5] + 2, y_text), "-", fill=colors['text'], font=font)
                
                draw.text((col_positions[6] + 2, y_text), f"{item['total']:.2f}", fill=colors['text'], font=font)
                
                # Lignes de séparation verticales
                for pos in col_positions[1:]:
                    draw.line([pos, current_y, pos, current_y + row_height], fill=colors['border'], width=1)
                
                current_y += row_height
        
        return current_y + 20
    
    def _wrap_text(self, text, max_chars):
        """Découpe le texte en lignes"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line += (" " + word if current_line else word)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _draw_complex_totals(self, draw, invoice_data, template, colors, size, current_y):
        """Dessine la section des totaux complexe"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = font_bold = ImageFont.load_default()
        
        # Cadre pour les totaux
        totals_width = 350
        totals_x = size[0] - totals_width - 40
        totals_height = 200
        
        draw.rectangle([totals_x, current_y, size[0] - 40, current_y + totals_height], outline=colors['border'], width=2)
        draw.rectangle([totals_x, current_y, size[0] - 40, current_y + 25], fill=colors['light'])
        
        draw.text((totals_x + 10, current_y + 5), "RECAPITULATIF", fill=colors['primary'], font=font_bold)
        
        y_pos = current_y + 35
        line_height = 20
        
        # Sous-total
        draw.text((totals_x + 10, y_pos), "Sous-total HT:", fill=colors['text'], font=font)
        draw.text((totals_x + 200, y_pos), f"{invoice_data['subtotal']:.2f} {invoice_data['currency']}", fill=colors['text'], font=font)
        y_pos += line_height
        
        # Remise globale si applicable
        if invoice_data['global_discount_rate'] > 0:
            draw.text((totals_x + 10, y_pos), f"Remise globale ({invoice_data['global_discount_rate']*100:.0f}%):", fill=colors['accent'], font=font)
            draw.text((totals_x + 200, y_pos), f"-{invoice_data['global_discount_amount']:.2f} {invoice_data['currency']}", fill=colors['accent'], font=font)
            y_pos += line_height
            
            # Sous-total après remise
            draw.text((totals_x + 10, y_pos), "Sous-total apres remise:", fill=colors['text'], font=font)
            draw.text((totals_x + 200, y_pos), f"{invoice_data['subtotal_after_discount']:.2f} {invoice_data['currency']}", fill=colors['text'], font=font)
            y_pos += line_height
        
        # TVA
        draw.text((totals_x + 10, y_pos), f"TVA ({invoice_data['tva_rate']*100:.0f}%):", fill=colors['text'], font=font)
        draw.text((totals_x + 200, y_pos), f"{invoice_data['tva_amount']:.2f} {invoice_data['currency']}", fill=colors['text'], font=font)
        y_pos += line_height
        
        # Timbre fiscal
        draw.text((totals_x + 10, y_pos), "Timbre fiscal:", fill=colors['text'], font=font)
        draw.text((totals_x + 200, y_pos), f"{invoice_data['timbre_fiscal']:.2f} {invoice_data['currency']}", fill=colors['text'], font=font)
        y_pos += line_height
        
        # Retenue à la source si applicable
        if invoice_data['retenue_source_rate'] > 0:
            draw.text((totals_x + 10, y_pos), f"Retenue a la source ({invoice_data['retenue_source_rate']*100:.0f}%):", fill=colors['accent'], font=font)
            draw.text((totals_x + 200, y_pos), f"-{invoice_data['retenue_source_amount']:.2f} {invoice_data['currency']}", fill=colors['accent'], font=font)
            y_pos += line_height
        
        # Ligne de séparation
        draw.line([totals_x + 10, y_pos, size[0] - 50, y_pos], fill=colors['border'], width=2)
        y_pos += 10
        
        # Total TTC
        draw.rectangle([totals_x + 5, y_pos, size[0] - 45, y_pos + 30], fill=colors['primary'])
        draw.text((totals_x + 10, y_pos + 8), "TOTAL TTC:", fill='white', font=font_bold)
        draw.text((totals_x + 200, y_pos + 8), f"{invoice_data['total_ttc']:.2f} {invoice_data['currency']}", fill='white', font=font_bold)
        
        return current_y + totals_height + 30
    
    def _draw_complex_footer(self, draw, invoice_data, template, colors, size):
        """Dessine un pied de page complexe"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except:
            font = font_bold = ImageFont.load_default()
        
        footer_y = size[1] - 200
        
        # Informations bancaires
        bank_info = invoice_data['bank_info']
        draw.rectangle([40, footer_y, size[0] - 40, footer_y + 80], outline=colors['border'], width=1)
        draw.rectangle([40, footer_y, size[0] - 40, footer_y + 20], fill=colors['light'])
        
        draw.text((50, footer_y + 5), "INFORMATIONS BANCAIRES", fill=colors['primary'], font=font_bold)
        
        y_pos = footer_y + 25
        draw.text((50, y_pos), f"Banque: {bank_info['bank_name']}", fill=colors['text'], font=font)
        draw.text((50, y_pos + 15), f"Titulaire: {invoice_data['company']['name']}", fill=colors['text'], font=font)
        draw.text((50, y_pos + 30), f"RIB: {invoice_data['company']['rib']}", fill=colors['text'], font=font)
        draw.text((50, y_pos + 45), f"SWIFT: {bank_info['swift']}", fill=colors['text'], font=font)
        
        # Notes
        notes_y = footer_y + 90
        draw.text((50, notes_y), "NOTES:", fill=colors['primary'], font=font_bold)
        draw.text((50, notes_y + 15), invoice_data['notes'], fill=colors['text'], font=font)
        
        # Conditions générales
        conditions_y = notes_y + 40
        draw.text((50, conditions_y), "CONDITIONS GENERALES:", fill=colors['primary'], font=font_bold)
        for i, condition in enumerate(invoice_data['conditions'][:3]):  # Max 3 conditions
            draw.text((50, conditions_y + 15 + i*12), f"• {condition}", fill=colors['text'], font=font)
    
    def _apply_realistic_effects(self, img):
        """Applique des effets réalistes à l'image"""
        # Convertir en array numpy
        img_array = np.array(img)
        
        # Ajouter un léger bruit
        noise = np.random.normal(0, 0.5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Reconvertir en PIL
        img = Image.fromarray(img_array)
        
        # Rotation très légère
        angle = random.uniform(-0.5, 0.5)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor='white')
        
        # Améliorer le contraste
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Légère amélioration de la netteté
        img = img.filter(ImageFilter.SHARPEN)
        
        return img

class ComplexSyntheticInvoiceGenerator:
    """Classe principale pour la génération de factures synthétiques complexes"""
    
    def __init__(self):
        self.data_generator = ComplexInvoiceDataGenerator()
        self.visual_generator = ComplexInvoiceVisualGenerator()
        
    def generate_complex_dataset(self, num_invoices=100, output_dir="complex_output"):
        """Génère un dataset complet de factures synthétiques complexes"""
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        os.makedirs(f"{output_dir}/templates", exist_ok=True)
        
        dataset_info = []
        template_stats = {}
        
        print(f"Génération de {num_invoices} factures synthétiques complexes...")
        
        for i in range(num_invoices):
            # Générer les données complexes
            invoice_data = self.data_generator.generate_complex_invoice_data()
            
            # Choisir un template aléatoire
            template_style = random.choice(list(self.visual_generator.templates.keys()))
            
            # Générer l'image
            invoice_image = self.visual_generator.generate_complex_invoice_image(
                invoice_data, template_style, size=(1200, 2200)
            )
            
            # Sauvegarder l'image en haute qualité
            image_filename = f"complex_invoice_{i+1:04d}.png"
            image_path = os.path.join(output_dir, "images", image_filename)
            invoice_image.save(image_path, "PNG", quality=95, dpi=(300, 300))
            
            # Sauvegarder les données JSON
            data_filename = f"complex_invoice_{i+1:04d}.json"
            data_path = os.path.join(output_dir, "data", data_filename)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(invoice_data, f, ensure_ascii=False, indent=2)
            
            # Statistiques des templates
            if template_style not in template_stats:
                template_stats[template_style] = 0
            template_stats[template_style] += 1
            
            # Ajouter aux infos du dataset
            dataset_info.append({
                'id': i+1,
                'image_path': image_path,
                'data_path': data_path,
                'template_style': template_style,
                'invoice_type': invoice_data['invoice_type'],
                'total_amount': invoice_data['total_ttc'],
                'num_items': len(invoice_data['items']),
                'has_discount': invoice_data['global_discount_rate'] > 0,
                'has_tax_withholding': invoice_data['retenue_source_rate'] > 0,
                'company_sector': invoice_data['company']['sector'],
                'client_sector': invoice_data['client']['sector']
            })
            
            if (i+1) % 10 == 0:
                print(f"Généré {i+1}/{num_invoices} factures complexes...")
        
        # Sauvegarder les métadonnées du dataset
        metadata = {
            'dataset_info': dataset_info,
            'template_statistics': template_stats,
            'generation_date': datetime.now().isoformat(),
            'total_invoices': num_invoices,
            'templates_used': list(template_stats.keys())
        }
        
        with open(os.path.join(output_dir, "complex_dataset_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Créer un fichier de statistiques
        self._create_statistics_report(dataset_info, output_dir)
        
        print(f"Dataset complexe généré avec succès dans {output_dir}/")
        return dataset_info
    
    def _create_statistics_report(self, dataset_info, output_dir):
        """Crée un rapport de statistiques"""
        stats = {
            'total_invoices': len(dataset_info),
            'templates_distribution': {},
            'invoice_types_distribution': {},
            'amount_statistics': {},
            'sectors_distribution': {},
            'features_statistics': {}
        }
        
        # Statistiques par template
        for item in dataset_info:
            template = item['template_style']
            stats['templates_distribution'][template] = stats['templates_distribution'].get(template, 0) + 1
            
            # Types de factures
            inv_type = item['invoice_type']
            stats['invoice_types_distribution'][inv_type] = stats['invoice_types_distribution'].get(inv_type, 0) + 1
            
            # Secteurs
            sector = item['company_sector']
            stats['sectors_distribution'][sector] = stats['sectors_distribution'].get(sector, 0) + 1
        
        # Statistiques des montants
        amounts = [item['total_amount'] for item in dataset_info]
        stats['amount_statistics'] = {
            'min': min(amounts),
            'max': max(amounts),
            'mean': sum(amounts) / len(amounts),
            'median': sorted(amounts)[len(amounts)//2]
        }
        
        # Statistiques des fonctionnalités
        stats['features_statistics'] = {
            'with_discount': sum(1 for item in dataset_info if item['has_discount']),
            'with_tax_withholding': sum(1 for item in dataset_info if item['has_tax_withholding']),
            'average_items_per_invoice': sum(item['num_items'] for item in dataset_info) / len(dataset_info)
        }
        
        # Sauvegarder le rapport
        with open(os.path.join(output_dir, "statistics_report.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Rapport de statistiques sauvegardé dans {output_dir}/statistics_report.json")

def main():
    """Fonction principale"""
    print("=== Générateur de Factures Synthétiques Marocaines Complexes ===")
    print("Génération de factures avec layouts professionnels avancés")
    print("Templates disponibles: executive, modern_corporate, classic_business, minimalist, premium")
    print()
    
    # Initialiser le générateur
    generator = ComplexSyntheticInvoiceGenerator()
    
    # Paramètres de génération
    try:
        num_invoices = int(input("Nombre de factures à générer (défaut: 100): ") or "100")
        output_dir = input("Répertoire de sortie (défaut: Data/complex_output): ") or "Data/complex_output"
    except ValueError:
        num_invoices = 100
        output_dir = "Data/complex_output"
    
    # Générer le dataset
    dataset_info = generator.generate_complex_dataset(
        num_invoices=num_invoices,
        output_dir=output_dir
    )
    
    print(f"\n✅ Génération terminée!")
    print(f"📁 {len(dataset_info)} factures complexes générées dans {output_dir}/")
    print(f"🖼️  Images haute qualité: {output_dir}/images/")
    print(f"📄 Données JSON: {output_dir}/data/")
    print(f"📊 Métadonnées: {output_dir}/complex_dataset_metadata.json")
    print(f"📈 Statistiques: {output_dir}/statistics_report.json")
    
    # Afficher les statistiques
    amounts = [info['total_amount'] for info in dataset_info]
    templates = {}
    for info in dataset_info:
        template = info['template_style']
        templates[template] = templates.get(template, 0) + 1
    
    print(f"\n📈 Statistiques du dataset:")
    print(f"   Montant moyen: {sum(amounts)/len(amounts):.2f} DH")
    print(f"   Montant min: {min(amounts):.2f} DH")
    print(f"   Montant max: {max(amounts):.2f} DH")
    print(f"   Templates utilisés: {list(templates.keys())}")
    
    for template, count in templates.items():
        print(f"   - {template}: {count} factures ({count/len(dataset_info)*100:.1f}%)")

if __name__ == "__main__":
    main()