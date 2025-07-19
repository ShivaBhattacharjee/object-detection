#!/usr/bin/env python3
"""
Check which YOLO model is currently active and show its details
"""

import sys
import os

def check_current_model():
    """Check which model is configured and show details"""
    
    print("🔍 CHECKING CURRENT MODEL CONFIGURATION")
    print("=" * 60)
    
    # Read current config
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        # Find the active YOLO_MODEL line
        for line in content.split('\n'):
            if line.startswith('YOLO_MODEL = ') and not line.strip().startswith('#'):
                model_name = line.split('"')[1]
                print(f"📄 Config file says: {model_name}")
                break
        else:
            print("❌ No active YOLO_MODEL found in config.py")
            return
            
    except FileNotFoundError:
        print("❌ config.py not found")
        return
    
    # Test if the model works
    print(f"\n🧪 Testing model: {model_name}")
    try:
        from ultralytics import YOLO
        from smart_classifier import SmartObjectClassifier
        
        # Load the model
        print(f"🔄 Loading {model_name}...")
        model = YOLO(model_name)
        
        # Test with our classifier
        classifier = SmartObjectClassifier()
        classifier.initialize_with_model(model)
        
        # Show details
        print(f"✅ Model loaded successfully!")
        print(f"📊 Total classes: {classifier.get_total_classes()}")
        print(f"🎯 Model type: {'Large dataset' if classifier.get_total_classes() > 100 else 'Small dataset (COCO)'}")
        
        # Show some example classes
        all_classes = classifier.get_all_class_names()
        print(f"📋 First 10 classes: {all_classes[:10]}")
        
        # Look for electronics specifically
        electronics = [name for name in all_classes if any(word in name.lower() 
                      for word in ['phone', 'laptop', 'computer', 'gaming', 'smart', 'electronic', 'tablet', 'headphone'])]
        
        if electronics:
            print(f"🎮 Electronics found ({len(electronics)}): {electronics[:10]}")
        else:
            print("📱 No specific electronics found (might be in COCO format)")
        
        # Show categories
        categories = classifier.get_all_categories()
        print(f"📂 Categories ({len(categories)}): {categories[:10]}")
        
        return model_name, classifier.get_total_classes()
        
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return None, 0
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, 0

def show_model_info():
    """Show information about available models"""
    
    print(f"\n💡 AVAILABLE MODELS:")
    print("=" * 60)
    
    models_info = {
        "COCO Models (80 classes - small)": [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
        ],
        "Open Images (601 classes - large)": [
            "yolov8n-oiv7.pt", "yolov8s-oiv7.pt", "yolov8m-oiv7.pt"
        ],
        "LVIS Dataset (1203 classes - huge)": [
            "yolov8n-lvis.pt"
        ],
        "Objects365 (365 classes - medium)": [
            "yolov8n-obj365.pt"
        ]
    }
    
    for dataset, models in models_info.items():
        print(f"\n📊 {dataset}:")
        for model in models:
            print(f"  🔹 {model}")
    
    print(f"\n🛠️ TO CHANGE MODEL:")
    print("1. Run: python3 download_large_model.py <model_name>")
    print("2. Or manually edit YOLO_MODEL in config.py")
    print("3. Run your app - it will use the new model automatically!")

if __name__ == "__main__":
    model_name, num_classes = check_current_model()
    
    if model_name and num_classes > 0:
        print(f"\n🎯 SUMMARY:")
        print(f"✅ Currently using: {model_name}")
        print(f"📊 Classes available: {num_classes}")
        print(f"🎨 Dataset type: {'Large' if num_classes > 100 else 'Small (COCO)'}")
        
        if num_classes <= 80:
            print(f"\n💡 RECOMMENDATION:")
            print(f"Your current model only has {num_classes} classes.")
            print(f"For more electronics and objects, try:")
            print(f"  python3 download_large_model.py yolov8n-oiv7.pt  # 601 classes")
            print(f"  python3 download_large_model.py yolov8n-lvis.pt  # 1203 classes")
    
    show_model_info()
