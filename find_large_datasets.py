#!/usr/bin/env python3
"""
Large Dataset Model Finder - Find and download YOLO models with more classes than COCO
"""

import logging
import os
import requests
from typing import Dict, List
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LargeDatasetModelFinder:
    """Find and download YOLO models trained on larger datasets"""
    
    def __init__(self):
        self.available_models = {
            # COCO models (baseline - 80 classes)
            "coco": {
                "yolov8n.pt": {"classes": 80, "size": "6.2MB", "dataset": "COCO", "speed": "Fastest"},
                "yolov8s.pt": {"classes": 80, "size": "21.5MB", "dataset": "COCO", "speed": "Fast"},
                "yolov8m.pt": {"classes": 80, "size": "49.7MB", "dataset": "COCO", "speed": "Medium"},
                "yolov8l.pt": {"classes": 80, "size": "83.7MB", "dataset": "COCO", "speed": "Slow"},
                "yolov8x.pt": {"classes": 80, "size": "136MB", "dataset": "COCO", "speed": "Slowest"},
            },
            
            # Open Images Dataset (600+ classes - much better for modern objects)
            "open_images": {
                "yolov5s-oiv7.pt": {"classes": 601, "size": "28MB", "dataset": "Open Images v7", "speed": "Fast"},
                "yolov8n-oiv7.pt": {"classes": 601, "size": "12MB", "dataset": "Open Images v7", "speed": "Very Fast"},
                "yolov8s-oiv7.pt": {"classes": 601, "size": "44MB", "dataset": "Open Images v7", "speed": "Fast"},
            },
            
            # Objects365 Dataset (365 classes - great for electronics and modern objects)
            "objects365": {
                "yolov8n-objects365.pt": {"classes": 365, "size": "15MB", "dataset": "Objects365", "speed": "Very Fast"},
                "yolov8s-objects365.pt": {"classes": 365, "size": "52MB", "dataset": "Objects365", "speed": "Fast"},
            },
            
            # LVIS Dataset (1200+ classes - largest public dataset)
            "lvis": {
                "yolov8n-lvis.pt": {"classes": 1203, "size": "18MB", "dataset": "LVIS", "speed": "Very Fast"},
                "yolov8s-lvis.pt": {"classes": 1203, "size": "58MB", "dataset": "LVIS", "speed": "Fast"},
            }
        }
        
        # Community model repositories
        self.community_sources = [
            "https://github.com/ultralytics/assets/releases/",
            "https://huggingface.co/models?library=ultralytics",
            "https://universe.roboflow.com/",
            "https://github.com/WongKinYiu/yolov7/releases/",
        ]
    
    def show_available_models(self):
        """Display all available models with their specifications"""
        print("=" * 80)
        print("🔍 AVAILABLE YOLO MODELS FOR LARGER DATASETS")
        print("=" * 80)
        
        for dataset_type, models in self.available_models.items():
            print(f"\n📊 {dataset_type.upper().replace('_', ' ')} MODELS:")
            print("-" * 50)
            
            for model_name, info in models.items():
                status = self._check_model_availability(model_name)
                print(f"🔹 {model_name}")
                print(f"   Classes: {info['classes']:,} | Size: {info['size']} | Speed: {info['speed']}")
                print(f"   Dataset: {info['dataset']} | Status: {status}")
                
                if info['classes'] > 80:
                    electronics_estimate = self._estimate_electronics_classes(info['classes'])
                    print(f"   📱 Estimated electronics classes: ~{electronics_estimate}")
                print()
    
    def _check_model_availability(self, model_name: str) -> str:
        """Check if a model is available locally or needs download"""
        if os.path.exists(model_name):
            return "✅ Available locally"
        else:
            return "📥 Needs download"
    
    def _estimate_electronics_classes(self, total_classes: int) -> int:
        """Estimate number of electronics classes based on total classes"""
        # Rough estimates based on dataset compositions
        if total_classes == 601:  # Open Images
            return 120  # ~20% electronics/tech
        elif total_classes == 365:  # Objects365
            return 80   # ~22% electronics/tech
        elif total_classes == 1203:  # LVIS
            return 200  # ~17% electronics/tech
        else:
            return int(total_classes * 0.15)  # Conservative 15%
    
    def test_model_classes(self, model_name: str):
        """Test a model and show its actual classes"""
        print(f"\n🧪 TESTING MODEL: {model_name}")
        print("=" * 50)
        
        try:
            logger.info(f"Loading model: {model_name}")
            model = YOLO(model_name)
            
            if hasattr(model, 'names') and model.names:
                class_names = list(model.names.values())
                total_classes = len(class_names)
                
                print(f"✅ Model loaded successfully!")
                print(f"📊 Total classes: {total_classes}")
                
                # Categorize classes
                electronics_classes = []
                for class_name in class_names:
                    if any(keyword in class_name.lower() for keyword in [
                        'phone', 'computer', 'laptop', 'tablet', 'camera', 'tv', 'monitor',
                        'speaker', 'headphone', 'mouse', 'keyboard', 'remote', 'gaming',
                        'electronic', 'device', 'smart', 'digital', 'tech'
                    ]):
                        electronics_classes.append(class_name)
                
                print(f"📱 Electronics/Tech classes found: {len(electronics_classes)}")
                
                if electronics_classes:
                    print("📋 Electronics classes sample:")
                    for i, cls in enumerate(electronics_classes[:15]):  # Show first 15
                        print(f"   • {cls}")
                    if len(electronics_classes) > 15:
                        print(f"   ... and {len(electronics_classes) - 15} more")
                
                # Show sample of all classes
                print(f"\n📝 All classes sample (first 20):")
                for i, cls in enumerate(class_names[:20]):
                    print(f"   {i:2d}: {cls}")
                if total_classes > 20:
                    print(f"   ... and {total_classes - 20} more classes")
                
                return True
                
            else:
                print("❌ Model doesn't expose class names")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def recommend_model(self, use_case: str = "electronics"):
        """Recommend the best model for a specific use case"""
        print(f"\n💡 MODEL RECOMMENDATIONS FOR: {use_case.upper()}")
        print("=" * 50)
        
        if use_case.lower() == "electronics":
            recommendations = [
                ("Objects365", "yolov8n-objects365.pt", "Best balance of electronics coverage and speed"),
                ("Open Images", "yolov8n-oiv7.pt", "Good electronics coverage, very fast"),
                ("LVIS", "yolov8n-lvis.pt", "Most comprehensive, includes rare electronics"),
            ]
        elif use_case.lower() == "speed":
            recommendations = [
                ("COCO", "yolov8n.pt", "Fastest, but limited to 80 classes"),
                ("Objects365", "yolov8n-objects365.pt", "Fast with 365 classes"),
                ("Open Images", "yolov8n-oiv7.pt", "Fast with 601 classes"),
            ]
        elif use_case.lower() == "comprehensive":
            recommendations = [
                ("LVIS", "yolov8s-lvis.pt", "1203 classes, best coverage"),
                ("Open Images", "yolov8s-oiv7.pt", "601 classes, good accuracy"),
                ("Objects365", "yolov8s-objects365.pt", "365 classes, balanced"),
            ]
        else:
            recommendations = [
                ("Balanced", "yolov8n-objects365.pt", "365 classes, good speed"),
                ("Coverage", "yolov8n-oiv7.pt", "601 classes, comprehensive"),
                ("Maximum", "yolov8n-lvis.pt", "1203 classes, most complete"),
            ]
        
        for i, (category, model, description) in enumerate(recommendations, 1):
            print(f"{i}. {category}: {model}")
            print(f"   {description}")
            
            # Show model specs
            for dataset_type, models in self.available_models.items():
                if model in models:
                    info = models[model]
                    print(f"   📊 {info['classes']} classes | {info['size']} | {info['speed']}")
                    break
            print()
    
    def show_community_resources(self):
        """Show where to find more models"""
        print("\n🌐 COMMUNITY RESOURCES FOR MORE MODELS")
        print("=" * 50)
        
        resources = [
            ("Ultralytics Hub", "https://hub.ultralytics.com/", "Official model repository"),
            ("Roboflow Universe", "https://universe.roboflow.com/", "Community trained models"),
            ("Hugging Face", "https://huggingface.co/models?library=ultralytics", "Open source models"),
            ("GitHub Releases", "https://github.com/ultralytics/assets/releases/", "Direct downloads"),
            ("Custom Training", "https://docs.ultralytics.com/modes/train/", "Train your own model"),
        ]
        
        for name, url, description in resources:
            print(f"🔗 {name}")
            print(f"   {url}")
            print(f"   {description}")
            print()
        
        print("💡 Tips for finding larger dataset models:")
        print("   • Search for 'Open Images YOLO' or 'LVIS YOLO'")
        print("   • Look for models with 300+ classes")
        print("   • Check model cards for class lists")
        print("   • Consider domain-specific models (e.g., 'electronics detection')")


def main():
    """Main function to run the model finder"""
    finder = LargeDatasetModelFinder()
    
    print("🔍 LARGE DATASET MODEL FINDER")
    print("Helping you move beyond COCO's limited 80 classes!")
    
    # Show available models
    finder.show_available_models()
    
    # Recommendations
    finder.recommend_model("electronics")
    finder.recommend_model("comprehensive")
    
    # Test a specific model
    print("\n" + "=" * 80)
    test_model = input("Enter a model name to test (or press Enter to skip): ").strip()
    if test_model:
        finder.test_model_classes(test_model)
    
    # Show community resources
    finder.show_community_resources()
    
    print("\n" + "=" * 80)
    print("✨ NEXT STEPS:")
    print("1. Choose a model with more classes than COCO (80)")
    print("2. Update YOLO_MODEL in config.py")
    print("3. Run the detection app - it will automatically adapt!")
    print("4. Enjoy detecting many more object types! 📱💻📺")


if __name__ == "__main__":
    main()
