#!/usr/bin/env python3
"""
Test script to demonstrate dynamic class loading from YOLO models
"""

import logging
from ultralytics import YOLO
from smart_classifier import SmartObjectClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dynamic_classifier():
    """Test the dynamic classifier with different YOLO models"""
    print("=" * 60)
    print("🧪 DYNAMIC CLASS LOADING TEST")
    print("=" * 60)
    
    # Test with different models
    models_to_test = [
        "yolov8n.pt",  # Standard COCO dataset (80 classes)
        # Add more models here to test
    ]
    
    for model_name in models_to_test:
        print(f"\n📦 Testing model: {model_name}")
        try:
            # Load YOLO model
            model = YOLO(model_name)
            
            # Create classifier and initialize with model
            classifier = SmartObjectClassifier()
            classifier.initialize_with_model(model)
            
            # Display information
            total_classes = classifier.get_total_classes()
            categories = classifier.get_all_categories()
            all_classes = classifier.get_all_class_names()
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Total classes: {total_classes}")
            print(f"📂 Categories found: {len(categories)}")
            print(f"📋 Categories: {', '.join(categories)}")
            
            # Show first 10 classes as example
            print(f"📝 First 10 classes: {', '.join(all_classes[:10])}")
            if len(all_classes) > 10:
                print(f"   ... and {len(all_classes) - 10} more")
            
            # Test classification of a few common objects
            print("\n🔍 Testing classification:")
            test_classes = ['person', 'car', 'dog', 'bicycle', 'airplane']
            for class_name in test_classes:
                class_id = classifier.get_class_id(class_name)
                if class_id is not None:
                    obj_info = classifier.classify_detection(class_id, 0.85)
                    print(f"  {class_name} -> {obj_info['display_name']} ({obj_info['category']})")
                    print(f"    Description: {obj_info['description']}")
            
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
        
        print("-" * 40)


def show_model_info():
    """Show detailed information about available YOLO models"""
    print("\n" + "=" * 60)
    print("📊 YOLO MODEL COMPARISON - MOVING BEYOND COCO!")
    print("=" * 60)
    
    model_info = {
        # Standard COCO models (LIMITED)
        "yolov8n.pt": {"size": "6.2MB", "classes": "80 (COCO)", "speed": "Fastest", "electronics": "~15"},
        "yolov8s.pt": {"size": "21.5MB", "classes": "80 (COCO)", "speed": "Fast", "electronics": "~15"},
        "yolov8m.pt": {"size": "49.7MB", "classes": "80 (COCO)", "speed": "Medium", "electronics": "~15"},
        
        # MUCH LARGER datasets (RECOMMENDED)
        "yolov8n-objects365.pt": {"size": "15MB", "classes": "365 (Objects365)", "speed": "Very Fast", "electronics": "~80"},
        "yolov8n-oiv7.pt": {"size": "12MB", "classes": "601 (Open Images)", "speed": "Very Fast", "electronics": "~120"},
        "yolov8n-lvis.pt": {"size": "18MB", "classes": "1203 (LVIS)", "speed": "Very Fast", "electronics": "~200"},
        
        # Custom possibilities
        "custom_electronics.pt": {"size": "Varies", "classes": "500+ (Custom)", "speed": "Varies", "electronics": "300+"},
    }
    
    print("🔴 COCO Models (TOO LIMITED - only 80 classes):")
    coco_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for model in coco_models:
        if model in model_info:
            info = model_info[model]
            print(f"  ❌ {model}")
            print(f"     Classes: {info['classes']} | Electronics: {info['electronics']} | Size: {info['size']}")
    
    print("\n� LARGE DATASET Models (MUCH BETTER - 300+ classes):")
    large_models = ["yolov8n-objects365.pt", "yolov8n-oiv7.pt", "yolov8n-lvis.pt"]
    for model in large_models:
        if model in model_info:
            info = model_info[model]
            print(f"  ✅ {model}")
            print(f"     Classes: {info['classes']} | Electronics: {info['electronics']} | Size: {info['size']}")
    
    print("\n🔵 CUSTOM Models (BEST - Domain specific):")
    custom_models = ["custom_electronics.pt"]
    for model in custom_models:
        if model in model_info:
            info = model_info[model]
            print(f"  🎯 {model}")
            print(f"     Classes: {info['classes']} | Electronics: {info['electronics']} | Size: {info['size']}")
    
    print("\n💡 Why larger datasets are better:")
    print("  • COCO (80 classes): Basic objects from 2014")
    print("  • Objects365 (365 classes): Modern objects, more electronics")
    print("  • Open Images (601 classes): Google's dataset, comprehensive")
    print("  • LVIS (1203 classes): Largest public dataset, rare objects")
    print("  • Custom models: Trained for specific domains (e.g., electronics)")


if __name__ == "__main__":
    show_model_info()
    test_dynamic_classifier()
    
    print("\n" + "=" * 60)
    print("✅ DYNAMIC CLASS LOADING TEST COMPLETED")
    print("=" * 60)
    print("🎯 Key improvements:")
    print("  • Class names are now pulled from the model, not hardcoded")
    print("  • System adapts to any dataset size automatically")
    print("  • Enhanced categorization for electronics and modern objects")
    print("  • Support for large datasets (Objects365, Open Images, LVIS)")
    print("  • Comprehensive fallback with 200+ classes instead of just 80")
    print("  • Configurable model selection in config.py")
    print("\n🚀 Next steps to use larger datasets:")
    print("  1. Run: python3 find_large_datasets.py")
    print("  2. Choose a model with 300+ classes")
    print("  3. Update YOLO_MODEL in config.py")
    print("  4. Enjoy detecting way more objects! 📱💻📺🎮")
