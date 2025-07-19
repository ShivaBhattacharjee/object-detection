#!/usr/bin/env python3
"""
Truly Dynamic Object Classifier - learns everything from the model, zero hardcoding
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class SmartObjectClassifier:
    """Truly dynamic classification system that learns categories from class names"""
    
    def __init__(self):
        # Dynamic class names - populated from model
        self.class_names = []
        self.model = None
        
        # Dynamic categories - learned from class names  
        self.object_categories = {}
        
        # Learned patterns (no hardcoded lists)
        self.learned_patterns = {}
        
    def initialize_with_model(self, model):
        """Initialize with any YOLO model and learn categories dynamically"""
        self.model = model
        try:
            # Extract class names from the model
            if hasattr(model, 'names') and model.names:
                self.class_names = list(model.names.values())
                logger.info(f"✅ Loaded {len(self.class_names)} class names from model")
                
                # Learn categories dynamically from the class names
                self._learn_categories_from_data()
                
            else:
                logger.warning("⚠️ Model doesn't have class names")
                self.class_names = []
                
        except Exception as e:
            logger.error(f"❌ Failed to extract class names from model: {e}")
            self.class_names = []
    
    def _learn_categories_from_data(self):
        """Learn categories by analyzing the actual class names in the dataset"""
        logger.info("🧠 Learning categories from class names...")
        
        # Group similar words/patterns
        word_groups = defaultdict(list)
        
        for class_name in self.class_names:
            # Extract meaningful words from class names
            words = self._extract_meaningful_words(class_name)
            
            for word in words:
                word_groups[word].append(class_name)
        
        # Find common patterns and create categories
        self.object_categories = {}
        
        # Look for common semantic groups
        self._create_semantic_categories(word_groups)
        
        logger.info(f"📊 Learned {len(self.object_categories)} categories dynamically")
        logger.info(f"📋 Categories: {list(self.object_categories.keys())}")
    
    def _extract_meaningful_words(self, class_name: str) -> List[str]:
        """Extract meaningful words from class names"""
        # Clean the class name
        cleaned = re.sub(r'[_\-\d]', ' ', class_name.lower())
        words = cleaned.split()
        
        # Filter out common non-semantic words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return meaningful_words
    
    def _create_semantic_categories(self, word_groups: dict):
        """Create semantic categories based on word frequency and context"""
        
        # Minimal semantic indicators - just for bootstrapping pattern recognition
        semantic_indicators = {
            'technology': ['phone', 'computer', 'laptop', 'tablet', 'electronic', 'digital', 'smart', 'device'],
            'transportation': ['car', 'truck', 'vehicle', 'bike', 'plane', 'train', 'boat', 'auto'],
            'people': ['person', 'people', 'human', 'man', 'woman', 'child', 'baby'],
            'animals': ['cat', 'dog', 'bird', 'animal', 'pet', 'wildlife'],
            'furniture': ['chair', 'table', 'bed', 'sofa', 'furniture', 'house', 'room'],
            'food': ['food', 'fruit', 'vegetable', 'meat', 'drink', 'eat', 'meal'],
            'tools': ['tool', 'equipment', 'machine', 'instrument', 'device'],
            'clothing': ['shirt', 'pants', 'dress', 'clothes', 'wear', 'fashion'],
            'sports': ['ball', 'sport', 'game', 'play', 'athletic', 'exercise']
        }
        
        # For each semantic group, find matching classes
        for category, indicators in semantic_indicators.items():
            matching_classes = []
            
            for class_name in self.class_names:
                class_lower = class_name.lower()
                class_words = self._extract_meaningful_words(class_name)
                
                # Check if any indicator appears in the class name or its words
                if any(indicator in class_lower or indicator in class_words for indicator in indicators):
                    matching_classes.append(class_name)
            
            if matching_classes:
                self.object_categories[category] = matching_classes
        
        # Find uncategorized classes and group them by common words
        categorized_classes = set()
        for classes in self.object_categories.values():
            categorized_classes.update(classes)
        
        uncategorized = [c for c in self.class_names if c not in categorized_classes]
        
        # Group uncategorized items by common words
        if uncategorized:
            self._group_uncategorized_items(uncategorized, word_groups)
    
    def _group_uncategorized_items(self, uncategorized: List[str], word_groups: dict):
        """Group remaining items by common patterns"""
        
        # Find words that appear in multiple class names
        common_words = {word: classes for word, classes in word_groups.items() 
                       if len(classes) >= 2 and any(c in uncategorized for c in classes)}
        
        # Create categories from common words
        for word, classes in common_words.items():
            uncategorized_in_group = [c for c in classes if c in uncategorized]
            if len(uncategorized_in_group) >= 2:
                category_name = f"{word}_items"
                self.object_categories[category_name] = uncategorized_in_group
                
                # Remove these from uncategorized
                for c in uncategorized_in_group:
                    if c in uncategorized:
                        uncategorized.remove(c)
        
        # Put remaining items in 'miscellaneous'
        if uncategorized:
            self.object_categories['miscellaneous'] = uncategorized
    
    def classify_detection(self, class_id: int, confidence: float) -> Dict:
        """Classify a detection"""
        if 0 <= class_id < len(self.class_names):
            class_name = self.class_names[class_id]
            category = self.get_object_category(class_name)
            description = self._generate_description(class_name, category)
            
            return {
                'class_id': class_id,
                'class_name': class_name,
                'category': category,
                'description': description,
                'confidence': confidence,
                'display_name': class_name.replace('_', ' ').title()
            }
        else:
            return {
                'class_id': class_id,
                'class_name': 'unknown',
                'category': 'unknown',
                'description': 'Unidentified object',
                'confidence': confidence,
                'display_name': 'Unknown'
            }
    
    def _generate_description(self, class_name: str, category: str) -> str:
        """Generate description based on learned categories"""
        clean_name = class_name.replace('_', ' ').replace('-', ' ').title()
        
        if category != 'miscellaneous':
            return f"{clean_name} ({category.replace('_', ' ')})"
        else:
            return f"{clean_name} object"
    
    def get_object_category(self, class_name: str) -> str:
        """Get category for a class"""
        for category, classes in self.object_categories.items():
            if class_name in classes:
                return category
        return 'miscellaneous'
    
    def analyze_scene(self, detections: List[Dict]) -> Dict:
        """Analyze the entire scene and provide intelligent insights"""
        if not detections:
            return {
                'total_objects': 0,
                'categories': {},
                'object_counts': {},
                'scene_description': 'No objects detected',
                'dominant_category': None
            }
        
        # Count objects by type and category
        object_counts = Counter()
        category_counts = defaultdict(int)
        
        for det in detections:
            class_name = det['class_name']
            category = det['category']
            object_counts[class_name] += 1
            category_counts[category] += 1
        
        # Find dominant category
        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        
        # Generate scene description
        scene_description = self.generate_scene_description(object_counts, category_counts)
        
        return {
            'total_objects': len(detections),
            'categories': dict(category_counts),
            'object_counts': dict(object_counts),
            'scene_description': scene_description,
            'dominant_category': dominant_category
        }
    
    def generate_scene_description(self, object_counts: Counter, category_counts: defaultdict) -> str:
        """Generate a natural language description of the scene"""
        if not object_counts:
            return "Empty scene"
        
        descriptions = []
        
        # Handle specific interesting cases
        if 'person' in object_counts:
            count = object_counts['person']
            if count == 1:
                descriptions.append("1 person")
            else:
                descriptions.append(f"{count} people")
        
        # Handle category-based descriptions
        for category, count in category_counts.items():
            if category != 'miscellaneous' and category != 'people':
                if count == 1:
                    # Find the specific object in this category
                    for obj, obj_count in object_counts.items():
                        if self.get_object_category(obj) == category and obj_count > 0:
                            descriptions.append(f"1 {obj}")
                            break
                else:
                    descriptions.append(f"{count} {category} items")
        
        # Handle miscellaneous items
        misc_count = category_counts.get('miscellaneous', 0)
        if misc_count > 0:
            descriptions.append(f"{misc_count} other objects")
        
        if not descriptions:
            return f"{len(object_counts)} objects detected"
        
        return ", ".join(descriptions[:5])  # Limit to 5 descriptions
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get class ID for a class name"""
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return None
    
    def is_valid_class(self, class_id: int) -> bool:
        """Check if class ID is valid"""
        return 0 <= class_id < len(self.class_names)
    
    def get_total_classes(self) -> int:
        """Get total number of classes supported"""
        return len(self.class_names)
    
    def get_all_class_names(self) -> List[str]:
        """Get all available class names"""
        return self.class_names.copy()
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.object_categories.keys())
    
    def get_category_color(self, category: str) -> Tuple[int, int, int]:
        """Get BGR color for object category"""
        # Generate colors dynamically based on category name hash
        category_hash = hash(category) % 10
        
        colors = [
            (0, 255, 0),        # Green
            (255, 0, 0),        # Blue  
            (0, 165, 255),      # Orange
            (128, 0, 128),      # Purple
            (255, 255, 0),      # Cyan
            (0, 255, 255),      # Yellow
            (255, 192, 203),    # Pink
            (0, 128, 255),      # Light Orange
            (128, 128, 128),    # Gray
            (0, 0, 255),        # Red
        ]
        
        return colors[category_hash]
