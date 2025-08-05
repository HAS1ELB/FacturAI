"""
Utilitaires géométriques pour le traitement des coordonnées et positions
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class GeometryUtils:
    """
    Utilitaires pour le traitement géométrique des éléments de factures
    
    Fonctionnalités :
    - Calculs de distances et positions
    - Détection d'alignement
    - Groupement spatial d'éléments
    - Conversion de coordonnées
    """
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calcule la distance euclidienne entre deux points
        
        Args:
            point1: Coordonnées du premier point (x, y)
            point2: Coordonnées du second point (x, y)
        
        Returns:
            Distance euclidienne
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    @staticmethod
    def calculate_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calcule le centre d'une bounding box
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Coordonnées du centre (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def calculate_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
        """
        Calcule l'aire d'une bounding box
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Aire de la bounding box
        """
        x1, y1, x2, y2 = bbox
        return abs(x2 - x1) * abs(y2 - y1)
    
    @staticmethod
    def calculate_overlap_ratio(bbox1: Tuple[float, float, float, float], 
                               bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calcule le ratio de chevauchement entre deux bounding boxes
        
        Args:
            bbox1: Première bounding box (x1, y1, x2, y2)
            bbox2: Seconde bounding box (x1, y1, x2, y2)
        
        Returns:
            Ratio de chevauchement (0.0 à 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcul de l'intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcul des aires
        area1 = GeometryUtils.calculate_bbox_area(bbox1)
        area2 = GeometryUtils.calculate_bbox_area(bbox2)
        
        # Union
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def is_horizontally_aligned(bbox1: Tuple[float, float, float, float], 
                               bbox2: Tuple[float, float, float, float], 
                               tolerance: float = 10.0) -> bool:
        """
        Vérifie si deux bounding boxes sont alignées horizontalement
        
        Args:
            bbox1: Première bounding box
            bbox2: Seconde bounding box
            tolerance: Tolérance en pixels
        
        Returns:
            True si alignées horizontalement
        """
        center1 = GeometryUtils.calculate_bbox_center(bbox1)
        center2 = GeometryUtils.calculate_bbox_center(bbox2)
        
        return abs(center1[1] - center2[1]) <= tolerance
    
    @staticmethod
    def is_vertically_aligned(bbox1: Tuple[float, float, float, float], 
                             bbox2: Tuple[float, float, float, float], 
                             tolerance: float = 10.0) -> bool:
        """
        Vérifie si deux bounding boxes sont alignées verticalement
        
        Args:
            bbox1: Première bounding box
            bbox2: Seconde bounding box
            tolerance: Tolérance en pixels
        
        Returns:
            True si alignées verticalement
        """
        center1 = GeometryUtils.calculate_bbox_center(bbox1)
        center2 = GeometryUtils.calculate_bbox_center(bbox2)
        
        return abs(center1[0] - center2[0]) <= tolerance
    
    @staticmethod
    def group_elements_by_proximity(elements: List[Dict[str, Any]], 
                                   distance_threshold: float = 50.0) -> List[List[Dict[str, Any]]]:
        """
        Groupe les éléments par proximité spatiale
        
        Args:
            elements: Liste d'éléments avec coordonnées
            distance_threshold: Seuil de distance pour le groupement
        
        Returns:
            Liste de groupes d'éléments
        """
        if not elements:
            return []
        
        groups = []
        remaining = elements.copy()
        
        while remaining:
            current_group = [remaining.pop(0)]
            
            i = 0
            while i < len(remaining):
                element = remaining[i]
                
                # Vérifier la proximité avec au moins un élément du groupe
                is_close = False
                for group_element in current_group:
                    if GeometryUtils._elements_are_close(element, group_element, distance_threshold):
                        is_close = True
                        break
                
                if is_close:
                    current_group.append(remaining.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    @staticmethod
    def _elements_are_close(element1: Dict[str, Any], element2: Dict[str, Any], 
                           threshold: float) -> bool:
        """
        Vérifie si deux éléments sont proches spatialement
        
        Args:
            element1: Premier élément
            element2: Second élément
            threshold: Seuil de distance
        
        Returns:
            True si les éléments sont proches
        """
        coords1 = element1.get("coordinates")
        coords2 = element2.get("coordinates")
        
        if not coords1 or not coords2:
            return False
        
        center1 = GeometryUtils.calculate_bbox_center(coords1)
        center2 = GeometryUtils.calculate_bbox_center(coords2)
        
        distance = GeometryUtils.calculate_distance(center1, center2)
        return distance <= threshold
    
    @staticmethod
    def detect_column_layout(elements: List[Dict[str, Any]], 
                            min_elements_per_column: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Détecte la disposition en colonnes des éléments
        
        Args:
            elements: Liste d'éléments avec coordonnées
            min_elements_per_column: Nombre minimum d'éléments par colonne
        
        Returns:
            Liste de colonnes (chaque colonne est une liste d'éléments)
        """
        if len(elements) < min_elements_per_column:
            return [elements] if elements else []
        
        # Trier par position X (gauche à droite)
        sorted_elements = sorted(elements, 
                               key=lambda x: GeometryUtils.calculate_bbox_center(
                                   x.get("coordinates", (0, 0, 0, 0)))[0])
        
        columns = []
        current_column = []
        last_x = None
        
        column_threshold = 100.0  # Seuil pour détecter une nouvelle colonne
        
        for element in sorted_elements:
            coords = element.get("coordinates")
            if not coords:
                continue
            
            center_x = GeometryUtils.calculate_bbox_center(coords)[0]
            
            if last_x is None or abs(center_x - last_x) > column_threshold:
                # Nouvelle colonne
                if len(current_column) >= min_elements_per_column:
                    columns.append(current_column)
                current_column = [element]
            else:
                current_column.append(element)
            
            last_x = center_x
        
        # Ajouter la dernière colonne
        if len(current_column) >= min_elements_per_column:
            columns.append(current_column)
        elif columns:
            # Fusionner avec la dernière colonne si trop petite
            columns[-1].extend(current_column)
        
        return columns
    
    @staticmethod
    def detect_table_structure(elements: List[Dict[str, Any]], 
                              row_tolerance: float = 15.0, 
                              col_tolerance: float = 20.0) -> Dict[str, Any]:
        """
        Détecte la structure tabulaire des éléments
        
        Args:
            elements: Liste d'éléments avec coordonnées
            row_tolerance: Tolérance pour l'alignement des lignes
            col_tolerance: Tolérance pour l'alignement des colonnes
        
        Returns:
            Structure de tableau détectée
        """
        if not elements:
            return {"rows": 0, "cols": 0, "structure": []}
        
        # Grouper par lignes (même Y approximatif)
        rows = []
        elements_copy = elements.copy()
        
        while elements_copy:
            current_element = elements_copy.pop(0)
            current_y = GeometryUtils.calculate_bbox_center(
                current_element.get("coordinates", (0, 0, 0, 0)))[1]
            
            current_row = [current_element]
            
            # Trouver tous les éléments à la même hauteur
            i = 0
            while i < len(elements_copy):
                element = elements_copy[i]
                element_y = GeometryUtils.calculate_bbox_center(
                    element.get("coordinates", (0, 0, 0, 0)))[1]
                
                if abs(element_y - current_y) <= row_tolerance:
                    current_row.append(elements_copy.pop(i))
                else:
                    i += 1
            
            # Trier la ligne par position X
            current_row.sort(
                key=lambda x: GeometryUtils.calculate_bbox_center(
                    x.get("coordinates", (0, 0, 0, 0)))[0])
            
            rows.append(current_row)
        
        # Trier les lignes par position Y
        rows.sort(key=lambda row: GeometryUtils.calculate_bbox_center(
            row[0].get("coordinates", (0, 0, 0, 0)))[1])
        
        # Déterminer le nombre de colonnes
        max_cols = max(len(row) for row in rows) if rows else 0
        
        return {
            "rows": len(rows),
            "cols": max_cols,
            "structure": rows,
            "is_regular": GeometryUtils._is_regular_table(rows)
        }
    
    @staticmethod
    def _is_regular_table(rows: List[List[Dict[str, Any]]]) -> bool:
        """
        Vérifie si la structure de tableau est régulière
        
        Args:
            rows: Structure de lignes
        
        Returns:
            True si le tableau est régulier
        """
        if not rows:
            return True
        
        expected_cols = len(rows[0])
        return all(len(row) == expected_cols for row in rows)
    
    @staticmethod
    def calculate_reading_order(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calcule l'ordre de lecture naturel des éléments
        
        Args:
            elements: Liste d'éléments avec coordonnées
        
        Returns:
            Éléments triés dans l'ordre de lecture
        """
        if not elements:
            return []
        
        # Tri par ligne puis par colonne (haut-bas, gauche-droite)
        def reading_order_key(element):
            coords = element.get("coordinates", (0, 0, 0, 0))
            center = GeometryUtils.calculate_bbox_center(coords)
            # Priorité Y puis X
            return (center[1], center[0])
        
        return sorted(elements, key=reading_order_key)
    
    @staticmethod
    def normalize_coordinates(coordinates: Tuple[float, float, float, float], 
                             image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """
        Normalise les coordonnées par rapport à la taille de l'image
        
        Args:
            coordinates: Coordonnées absolues (x1, y1, x2, y2)
            image_width: Largeur de l'image
            image_height: Hauteur de l'image
        
        Returns:
            Coordonnées normalisées (0.0 à 1.0)
        """
        x1, y1, x2, y2 = coordinates
        return (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        )
    
    @staticmethod
    def denormalize_coordinates(normalized_coords: Tuple[float, float, float, float], 
                               image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """
        Dénormalise les coordonnées
        
        Args:
            normalized_coords: Coordonnées normalisées (0.0 à 1.0)
            image_width: Largeur de l'image
            image_height: Hauteur de l'image
        
        Returns:
            Coordonnées absolues en pixels
        """
        x1, y1, x2, y2 = normalized_coords
        return (
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height
        )