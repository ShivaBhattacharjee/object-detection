#!/usr/bin/env python3
"""
Fast Object Detection using YOLOv8 with Smart Classification
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, YOLO_MODEL
from smart_classifier import SmartObjectClassifier

logger = logging.getLogger(__name__)


class FastObjectDetector:
    """Fast object detection using YOLOv8 with intelligent classification"""
    
    def __init__(self):
        self.model = None
        self.classifier = SmartObjectClassifier()
        self.detection_times = []
        self.max_detection_history = 10
        
    def initialize(self) -> bool:
        """Initialize YOLO model"""
        try:
            logger.info(f"🚀 Loading YOLO model: {YOLO_MODEL}")
            # Use configurable YOLO model
            self.model = YOLO(YOLO_MODEL)  # This will download if not present
            
            # Initialize the classifier with the model to extract class names
            logger.info("📋 Initializing dynamic class names from model...")
            self.classifier.initialize_with_model(self.model)
            
            # Warm up the model
            logger.info("🔥 Warming up model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False)
            
            # Log model information
            total_classes = self.classifier.get_total_classes()
            categories = self.classifier.get_all_categories()
            logger.info(f"✅ YOLO model ready with {total_classes} classes")
            logger.info(f"📊 Available categories: {', '.join(categories)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame using YOLO + smart classification"""
        if self.model is None:
            return []
        
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Use smart classifier to get detailed object info
                        obj_info = self.classifier.classify_detection(class_id, float(conf))
                        
                        detection = {
                            'id': i,
                            'bbox': tuple(map(int, box)),  # (x1, y1, x2, y2)
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': obj_info['class_name'],
                            'category': obj_info['category'],
                            'description': obj_info['description'],
                            'display_name': obj_info['display_name']
                        }
                        
                        detections.append(detection)
            
            # Track performance
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            if len(self.detection_times) > self.max_detection_history:
                self.detection_times.pop(0)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], scene_analysis: Dict = None) -> np.ndarray:
        """Draw detection results with smart categorization"""
        result_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            display_name = detection['display_name']
            confidence = detection['confidence']
            category = detection['category']
            
            # Get category-specific color
            color = self.classifier.get_category_color(category)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Prepare label with category info
            label = f"{display_name}: {confidence:.0%}"
            if category != 'other':
                label = f"[{category.upper()}] {display_name}: {confidence:.0%}"
            
            # Calculate label size and position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 30
            
            # Draw label background
            cv2.rectangle(result_frame, 
                         (bbox[0], label_y - label_size[1] - 5),
                         (bbox[0] + label_size[0], label_y + 5), 
                         color, -1)
            
            # Draw label text
            cv2.putText(result_frame, label, 
                       (bbox[0], label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw scene analysis if provided
        if scene_analysis:
            self.draw_scene_analysis(result_frame, scene_analysis)
        
        return result_frame
    
    def draw_scene_analysis(self, frame: np.ndarray, analysis: Dict):
        """Draw scene analysis information on frame"""
        y_offset = 80
        line_height = 25
        
        # Scene description
        scene_desc = analysis.get('scene_description', '')
        if scene_desc:
            cv2.putText(frame, f"Scene: {scene_desc}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
        
        # Category breakdown
        categories = analysis.get('categories', {})
        if categories:
            category_text = "Categories: " + ", ".join([f"{cat}({count})" for cat, count in categories.items()])
            if len(category_text) > 60:  # Truncate if too long
                category_text = category_text[:57] + "..."
            cv2.putText(frame, category_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def analyze_scene(self, detections: List[Dict]) -> Dict:
        """Get scene analysis using smart classifier"""
        return self.classifier.analyze_scene(detections)
    
    def get_performance_stats(self) -> Dict:
        """Get detection performance statistics"""
        if not self.detection_times:
            return {'avg_detection_time': 0, 'fps': 0, 'device': 'unknown'}
        
        avg_time = sum(self.detection_times) / len(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Try to get device info
        device = 'CPU'
        if self.model and hasattr(self.model, 'device'):
            device = str(self.model.device)
        
        return {
            'avg_detection_time': avg_time,
            'fps': fps,
            'device': device,
            'total_detections': len(self.detection_times)
        }
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.model is not None
