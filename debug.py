#!/usr/bin/env python3
"""
Debug script to test Fast YOLO detection with Smart Classification
"""

import logging
import sys
import cv2
import numpy as np

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_smart_classifier():
    """Test the smart object classifier"""
    try:
        from smart_classifier import SmartObjectClassifier
        
        print("🧠 Testing Smart Object Classifier...")
        
        classifier = SmartObjectClassifier()
        
        # Test classification of common objects
        test_objects = [
            (0, 0.95),   # person
            (2, 0.85),   # car
            (16, 0.90),  # dog
            (15, 0.80),  # cat
            (19, 0.75),  # cow
            (63, 0.88),  # laptop
            (64, 0.92),  # mouse
        ]
        
        print("Testing object classifications:")
        for class_id, confidence in test_objects:
            result = classifier.classify_detection(class_id, confidence)
            print(f"  Class {class_id}: {result['display_name']} ({result['category']}) - {result['description']}")
        
        # Test scene analysis
        mock_detections = [
            {'class_name': 'person', 'category': 'people', 'confidence': 0.95},
            {'class_name': 'person', 'category': 'people', 'confidence': 0.88},
            {'class_name': 'car', 'category': 'vehicles', 'confidence': 0.92},
            {'class_name': 'dog', 'category': 'animals', 'confidence': 0.85},
        ]
        
        scene_analysis = classifier.analyze_scene(mock_detections)
        print(f"\nScene Analysis Test:")
        print(f"  Description: {scene_analysis['scene_description']}")
        print(f"  Categories: {scene_analysis['categories']}")
        print(f"  Object counts: {scene_analysis['object_counts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart classifier test failed: {e}")
        return False

def test_fast_detector():
    """Test the fast object detector"""
    try:
        from fast_detector import FastObjectDetector
        
        print("\n🚀 Testing Fast Object Detector...")
        
        detector = FastObjectDetector()
        if not detector.initialize():
            print("❌ Failed to initialize detector")
            return False
        
        # Create a test image with simple shapes
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some recognizable shapes
        cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)  # White rectangle
        cv2.circle(test_frame, (500, 150), 50, (0, 255, 0), -1)  # Green circle
        cv2.putText(test_frame, "TEST", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        
        print("🔍 Running detection on test image...")
        detections = detector.detect_objects(test_frame)
        
        print(f"✅ Found {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['display_name']} ({det['category']}) - {det['confidence']:.1%}")
        
        # Test scene analysis
        if detections:
            scene_analysis = detector.analyze_scene(detections)
            print(f"\nScene: {scene_analysis['scene_description']}")
        
        # Save test result
        if detections:
            result_frame = detector.draw_detections(test_frame, detections)
            cv2.imwrite("debug_fast_detection.jpg", result_frame)
            print("💾 Test result saved as debug_fast_detection.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Fast detector test failed: {e}")
        return False

def test_performance():
    """Test detection performance"""
    try:
        from fast_detector import FastObjectDetector
        import time
        
        print("\n⚡ Testing Performance...")
        
        detector = FastObjectDetector()
        if not detector.initialize():
            print("❌ Failed to initialize detector")
            return False
        
        # Create test frames
        test_frames = []
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Time the detections
        start_time = time.time()
        total_detections = 0
        
        for frame in test_frames:
            detections = detector.detect_objects(frame)
            total_detections += len(detections)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(test_frames)
        fps = 1.0 / avg_time
        
        print(f"✅ Performance Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per frame: {avg_time:.3f}s")
        print(f"  Estimated FPS: {fps:.1f}")
        print(f"  Total detections: {total_detections}")
        
        # Get detector stats
        perf_stats = detector.get_performance_stats()
        print(f"  Detector FPS: {perf_stats.get('fps', 0):.1f}")
        print(f"  Device: {perf_stats.get('device', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    print("🔍 DEBUGGING FAST YOLO + SMART CLASSIFICATION")
    print("=" * 60)
    
    # Test smart classifier
    test_smart_classifier()
    
    print("\n" + "-" * 40)
    
    # Test fast detector
    test_fast_detector()
    
    print("\n" + "-" * 40)
    
    # Test performance
    test_performance()
    
    print("\n" + "=" * 60)
    print("💡 Fast YOLO + Smart Classification System")
    print("💡 Should be much faster than LLaVA!")
    print("💡 Try running: python app.py")

if __name__ == "__main__":
    main()
