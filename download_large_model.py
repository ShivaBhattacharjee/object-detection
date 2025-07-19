#!/usr/bin/env python3
"""
Script to find and download YOLO models trained on larger datasets
"""

import os
import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_available_models():
    """Check what YOLO models are available with larger datasets"""
    
    print("🔍 FINDING LARGER DATASET MODELS")
    print("=" * 60)
    
    # Available models with more classes than COCO
    larger_models = {
        "Open Images Dataset Models (600+ classes)": {
            "yolov8n-oiv7.pt": {
                "classes": "601 classes", 
                "size": "6MB",
                "description": "Open Images v7 - tons of electronics, tools, etc."
            },
            "yolov8s-oiv7.pt": {
                "classes": "601 classes",
                "size": "22MB", 
                "description": "Open Images v7 - better accuracy"
            },
            "yolov8m-oiv7.pt": {
                "classes": "601 classes",
                "size": "50MB",
                "description": "Open Images v7 - high accuracy"
            }
        },
        "Objects365 Dataset Models (365 classes)": {
            "yolov8n-obj365.pt": {
                "classes": "365 classes",
                "size": "6MB", 
                "description": "Objects365 - more everyday objects"
            }
        },
        "LVIS Dataset Models (1000+ classes)": {
            "yolov8n-lvis.pt": {
                "classes": "1203 classes",
                "size": "6MB",
                "description": "LVIS - huge variety of objects"
            }
        },
        "Custom/Community Models": {
            "yolov8n-worldv2.pt": {
                "classes": "10000+ classes",
                "size": "6MB",
                "description": "World model - massive dataset"
            }
        }
    }
    
    for dataset_name, models in larger_models.items():
        print(f"\n📊 {dataset_name}:")
        for model_name, info in models.items():
            print(f"  🔹 {model_name}")
            print(f"     Classes: {info['classes']}")
            print(f"     Size: {info['size']}")
            print(f"     Description: {info['description']}")
    
    return larger_models

def download_model(model_name):
    """Download a specific model"""
    print(f"\n📥 Downloading {model_name}...")
    
    try:
        # Try to import ultralytics and download
        from ultralytics import YOLO
        
        print(f"🔄 Loading {model_name} (will download if needed)...")
        model = YOLO(model_name)
        
        # Test the model
        if hasattr(model, 'names') and model.names:
            num_classes = len(model.names)
            print(f"✅ {model_name} loaded successfully!")
            print(f"📊 Classes: {num_classes}")
            print(f"📋 First 10 classes: {list(model.names.values())[:10]}")
            
            return True
        else:
            print(f"⚠️ Model loaded but no class names found")
            return False
            
    except ImportError:
        print("❌ ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def update_config(model_name):
    """Update config.py to use the new model"""
    config_path = "config.py"
    
    if not os.path.exists(config_path):
        print(f"❌ {config_path} not found")
        return False
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update YOLO_MODEL line
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            if line.startswith('YOLO_MODEL = '):
                updated_lines.append(f'YOLO_MODEL = "{model_name}"  # Updated to larger dataset')
            else:
                updated_lines.append(line)
        
        # Write back
        with open(config_path, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ Updated {config_path} to use {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def main():
    """Main function to guide user through model selection"""
    
    print("🚀 UPGRADE TO LARGER DATASETS")
    print("=" * 60)
    print("Current COCO dataset: 80 classes (too small!)")
    print("Let's get you a dataset with 300-1000+ classes!")
    
    # Show available models
    larger_models = check_available_models()
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    print("\n🥇 BEST FOR ELECTRONICS:")
    print("   yolov8n-oiv7.pt (Open Images - 601 classes)")
    print("   - Has lots of electronics, gadgets, tools")
    print("   - Gaming equipment, smart devices, etc.")
    
    print("\n🥈 MOST CLASSES:")  
    print("   yolov8n-lvis.pt (LVIS - 1203 classes)")
    print("   - Huge variety of objects")
    
    print("\n🥉 BALANCED:")
    print("   yolov8n-obj365.pt (Objects365 - 365 classes)")
    print("   - Good mix of everyday objects")
    
    print("\n" + "=" * 60)
    print("🛠️ TO UPGRADE:")
    print("1. Choose a model from above")
    print("2. Run: python3 download_large_model.py <model_name>")
    print("3. The system will automatically use the larger dataset!")
    
    # If ultralytics is available, offer to download
    try:
        from ultralytics import YOLO
        
        print("\n🎯 QUICK SETUP:")
        recommended = "yolov8n-oiv7.pt"
        response = input(f"Download recommended model {recommended}? (y/n): ").lower().strip()
        
        if response == 'y':
            if download_model(recommended):
                if update_config(recommended):
                    print(f"\n🎉 SUCCESS!")
                    print(f"✅ Downloaded {recommended} (601 classes)")
                    print(f"✅ Updated config.py")
                    print(f"✅ Run your app now - it will use the larger dataset!")
                    
                    # Test with our classifier
                    print(f"\n🧪 Testing with new model...")
                    test_new_model(recommended)
        
    except ImportError:
        print("\n📦 INSTALL REQUIRED:")
        print("pip install ultralytics")

def test_new_model(model_name):
    """Test the new model with our dynamic classifier"""
    try:
        from ultralytics import YOLO
        from smart_classifier import SmartObjectClassifier
        
        print(f"🔄 Loading {model_name}...")
        model = YOLO(model_name)
        
        classifier = SmartObjectClassifier()
        classifier.initialize_with_model(model)
        
        print(f"📊 Total classes: {classifier.get_total_classes()}")
        print(f"📋 Categories: {classifier.get_all_categories()}")
        
        # Show some electronics if available
        electronics_examples = []
        for i, class_name in enumerate(classifier.get_all_class_names()):
            if any(word in class_name.lower() for word in ['phone', 'laptop', 'computer', 'gaming', 'smart', 'electronic']):
                electronics_examples.append(class_name)
                if len(electronics_examples) >= 5:
                    break
        
        if electronics_examples:
            print(f"🎮 Electronics found: {electronics_examples}")
        else:
            print(f"📱 First 10 classes: {classifier.get_all_class_names()[:10]}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Check if model name provided as argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"🎯 DOWNLOADING SPECIFIC MODEL: {model_name}")
        print("=" * 60)
        
        # Download the specific model
        if download_model(model_name):
            if update_config(model_name):
                print(f"\n🎉 SUCCESS!")
                print(f"✅ Downloaded {model_name}")
                print(f"✅ Updated config.py")
                print(f"✅ Run your app now - it will use the larger dataset!")
                
                # Test with our classifier
                print(f"\n🧪 Testing with new model...")
                test_new_model(model_name)
            else:
                print("⚠️ Model downloaded but config update failed")
        else:
            print("❌ Download failed")
    else:
        # Show menu if no argument provided
        main()
