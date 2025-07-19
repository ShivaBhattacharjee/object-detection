#!/usr/bin/env python3
"""
Smart Object Classifier using COCO dataset and intelligent classification
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class SmartObjectClassifier:
    """Intelligent object classification system using COCO dataset knowledge"""
    
    def __init__(self):
        # COCO dataset class names (80 classes) - this is the standard dataset YOLO is trained on
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Create object categories for better organization and counting
        self.object_categories = {
            'people': ['person'],
            'vehicles': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
            'animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'furniture': ['chair', 'couch', 'bed', 'dining table'],
            'electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster'],
            'food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
            'kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'sink', 'refrigerator'],
            'sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
            'accessories': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
            'household': ['potted plant', 'toilet', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        }
        
        # Enhanced descriptions for each object type
        self.object_descriptions = {
            # People
            'person': 'Human being detected',
            
            # Vehicles
            'bicycle': 'Two-wheeled pedal vehicle',
            'car': 'Four-wheeled motor vehicle',
            'motorcycle': 'Two-wheeled motor vehicle',
            'airplane': 'Aircraft for air travel',
            'bus': 'Large public transport vehicle',
            'train': 'Railway transport vehicle',
            'truck': 'Large goods transport vehicle',
            'boat': 'Water transport vessel',
            
            # Animals
            'bird': 'Flying avian creature',
            'cat': 'Domestic feline animal',
            'dog': 'Domestic canine animal',
            'horse': 'Large domesticated mammal',
            'sheep': 'Wool-producing livestock',
            'cow': 'Dairy/beef cattle',
            'elephant': 'Large pachyderm mammal',
            'bear': 'Large carnivorous mammal',
            'zebra': 'Striped equine animal',
            'giraffe': 'Tall-necked African mammal',
            
            # Electronics
            'tv': 'Television display screen',
            'laptop': 'Portable computer device',
            'mouse': 'Computer pointing device',
            'remote': 'Electronic control device',
            'keyboard': 'Computer input device',
            'cell phone': 'Mobile communication device',
            'microwave': 'Kitchen heating appliance',
            
            # Food items
            'banana': 'Yellow curved fruit',
            'apple': 'Round edible fruit',
            'orange': 'Citrus fruit',
            'pizza': 'Italian flatbread dish',
            'sandwich': 'Layered food item',
            
            # Default for others
        }
        
        # Add default descriptions for items not explicitly defined
        for class_name in self.coco_classes:
            if class_name not in self.object_descriptions:
                self.object_descriptions[class_name] = f"{class_name.replace('_', ' ').title()} object"
    
    def classify_detection(self, class_id: int, confidence: float) -> Dict:
        """Classify a YOLO detection into detailed object information"""
        if 0 <= class_id < len(self.coco_classes):
            class_name = self.coco_classes[class_id]
            category = self.get_object_category(class_name)
            description = self.object_descriptions.get(class_name, f"{class_name} object")
            
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
    
    def get_object_category(self, class_name: str) -> str:
        """Get the category for an object class"""
        for category, classes in self.object_categories.items():
            if class_name in classes:
                return category
        return 'other'
    
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
        
        # Handle animals specifically
        animal_counts = []
        for animal in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
            if animal in object_counts:
                count = object_counts[animal]
                if count == 1:
                    animal_counts.append(f"1 {animal}")
                else:
                    animal_counts.append(f"{count} {animal}s")
        
        if animal_counts:
            descriptions.append(", ".join(animal_counts))
        
        # Handle vehicles
        vehicle_counts = []
        for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
            if vehicle in object_counts:
                count = object_counts[vehicle]
                if count == 1:
                    vehicle_counts.append(f"1 {vehicle}")
                else:
                    vehicle_counts.append(f"{count} {vehicle}s")
        
        if vehicle_counts:
            descriptions.append(", ".join(vehicle_counts))
        
        # Handle other significant objects
        other_objects = []
        for obj, count in object_counts.most_common():
            if obj not in ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                          'bear', 'zebra', 'giraffe', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                if count > 1:
                    other_objects.append(f"{count} {obj}s")
                else:
                    other_objects.append(f"1 {obj}")
        
        if other_objects and len(other_objects) <= 3:
            descriptions.extend(other_objects[:3])
        elif other_objects:
            descriptions.append(f"{len(other_objects)} other objects")
        
        if not descriptions:
            return f"{len(object_counts)} objects detected"
        
        return ", ".join(descriptions)
    
    def get_class_id(self, class_name: str) -> Optional[int]:
        """Get COCO class ID for a class name"""
        try:
            return self.coco_classes.index(class_name)
        except ValueError:
            return None
    
    def is_valid_class(self, class_id: int) -> bool:
        """Check if class ID is valid in COCO dataset"""
        return 0 <= class_id < len(self.coco_classes)
    
    def get_category_color(self, category: str) -> Tuple[int, int, int]:
        """Get BGR color for object category"""
        category_colors = {
            'people': (0, 255, 0),      # Green
            'vehicles': (255, 0, 0),    # Blue
            'animals': (0, 165, 255),   # Orange
            'furniture': (128, 0, 128), # Purple
            'electronics': (255, 255, 0), # Cyan
            'food': (0, 255, 255),      # Yellow
            'kitchen': (255, 192, 203), # Pink
            'sports': (0, 128, 255),    # Light Orange
            'accessories': (128, 128, 128), # Gray
            'household': (0, 0, 255),   # Red
            'other': (255, 255, 255),   # White
            'unknown': (128, 128, 128)  # Gray
        }
        return category_colors.get(category, (0, 255, 0))  # Default green
