#!/usr/bin/env python3
"""
Test Fast YOLO detection system
"""

import sys

def test_dependencies():
    """Test if all required dependencies are available"""
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not available - run: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available - run: pip install numpy")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLO) available")
    except ImportError:
        print("❌ Ultralytics not available - run: pip install ultralytics")
        return False
    
    try:
        import torch
        print("✅ PyTorch available")
        if torch.cuda.is_available():
            print("✅ CUDA available for GPU acceleration")
        else:
            print("ℹ️  CUDA not available, will use CPU")
    except ImportError:
        print("❌ PyTorch not available - run: pip install torch")
        return False
    
    return True

def test_fast_detector():
    """Test our FastObjectDetector"""
    try:
        from fast_detector import FastObjectDetector
        
        print("\n🚀 Testing Fast Object Detector...")
        
        detector = FastObjectDetector()
        if detector.initialize():
            print("✅ Fast detector initialized successfully")
            
            # Get performance stats
            perf_stats = detector.get_performance_stats()
            print(f"✅ Detector device: {perf_stats.get('device', 'Unknown')}")
            
            return True
        else:
            print("❌ Fast detector failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ Fast detector test failed: {e}")
        return False

def test_smart_classifier():
    """Test our SmartObjectClassifier"""
    try:
        from smart_classifier import SmartObjectClassifier
        
        print("\n🧠 Testing Smart Classifier...")
        
        classifier = SmartObjectClassifier()
        
        # Test a few classifications
        test_cases = [
            (0, 0.95),   # person
            (2, 0.85),   # car
            (16, 0.90),  # dog
            (64, 0.88),  # mouse
        ]
        
        print("✅ Smart classifier working:")
        for class_id, conf in test_cases:
            result = classifier.classify_detection(class_id, conf)
            print(f"  - Class {class_id}: {result['display_name']} ({result['category']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart classifier test failed: {e}")
        return False

def main():
    print("🧪 Testing Fast YOLO Detection System")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test fast detector
    detector_ok = test_fast_detector()
    
    # Test smart classifier
    classifier_ok = test_smart_classifier()
    
    print("\n" + "=" * 50)
    if deps_ok and detector_ok and classifier_ok:
        print("🎉 All tests passed! Fast YOLO system is ready.")
        print("🚀 Run: python app.py")
    else:
        print("❌ Some tests failed!")
        if not deps_ok:
            print("💡 Install dependencies: pip install -r requirements.txt")
        if not detector_ok:
            print("💡 Check fast_detector.py")
        if not classifier_ok:
            print("💡 Check smart_classifier.py")

if __name__ == "__main__":
    main()
