#!/usr/bin/env python3
"""
Quick Multi-Object Detection Fix
Optimizes detection for holding phone + iPad together
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_multiobject_quick():
    """Quick test for multi-object detection"""
    print("🔧 QUICK MULTI-OBJECT DETECTION FIX")
    print("=" * 50)
    
    # Load model with the current configuration
    print("🔄 Loading current model...")
    try:
        # Try different models in order of preference
        models_to_try = [
            "yolov8s-oiv7.pt",   # Larger Open Images (downloading)
            "yolov8n-oiv7.pt",   # Current Open Images
            "yolov8m-oiv7.pt",   # Medium Open Images (if available)
            "yolov8n.pt"         # Fallback COCO
        ]
        
        model = None
        model_name = None
        
        for model_name in models_to_try:
            try:
                print(f"  Trying {model_name}...")
                model = YOLO(model_name)
                print(f"  ✅ Loaded {model_name} with {len(model.names)} classes")
                break
            except Exception as e:
                print(f"  ❌ Failed {model_name}: {e}")
                continue
        
        if model is None:
            print("❌ No model could be loaded")
            return
        
        # Find device-related classes
        device_classes = []
        for class_id, class_name in model.names.items():
            if any(keyword in class_name.lower() for keyword in [
                'phone', 'mobile', 'tablet', 'ipad', 'smartphone', 'cell',
                'computer', 'laptop', 'monitor', 'gaming'
            ]):
                device_classes.append((class_id, class_name))
        
        print(f"\n📱 Found {len(device_classes)} device classes:")
        for class_id, class_name in device_classes[:10]:  # Show first 10
            print(f"  • {class_name}")
        
        # Optimized detection settings for multi-object
        optimal_settings = {
            'conf': 0.25,        # Low confidence for more detections
            'iou': 0.2,          # Very low IOU for overlapping objects
            'max_det': 50,       # Allow many detections
            'agnostic_nms': True # Better NMS for multi-class detection
        }
        
        print(f"\n🎯 OPTIMAL SETTINGS FOR MULTI-OBJECT:")
        for key, value in optimal_settings.items():
            print(f"  {key}: {value}")
        
        # Test with camera
        print(f"\n🎥 TESTING WITH CAMERA...")
        print("Hold your phone and iPad together!")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not available")
            return
        
        detection_count = 0
        multi_device_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detection_count += 1
            
            # Run optimized detection
            results = model(frame, **optimal_settings, verbose=False)
            
            device_detections = []
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_name = model.names[class_id]
                    
                    # Check if it's a device
                    if any(class_id == device_id for device_id, _ in device_classes):
                        device_detections.append((box, conf, class_name))
                
                # Count multi-device frames
                if len(device_detections) >= 2:
                    multi_device_frames += 1
                
                # Draw all detections
                for box, conf, class_name in device_detections:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Use different colors for different devices
                    if 'phone' in class_name.lower() or 'mobile' in class_name.lower():
                        color = (0, 255, 0)  # Green for phones
                    elif 'tablet' in class_name.lower() or 'ipad' in class_name.lower():
                        color = (255, 0, 255)  # Magenta for tablets
                    else:
                        color = (0, 255, 255)  # Yellow for other devices
                    
                    # Draw thick bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with background
                    label = f"{class_name} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 0), 2)
            
            # Show status
            status_text = f"Devices: {len(device_detections)} | Multi-device frames: {multi_device_frames}/{detection_count}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, (0, 255, 255), 2)
            
            # Show success rate
            if detection_count > 0:
                success_rate = (multi_device_frames / detection_count) * 100
                rate_text = f"Multi-detection rate: {success_rate:.1f}%"
                cv2.putText(frame, rate_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (255, 255, 0), 2)
            
            cv2.imshow('Multi-Object Detection - FIXED', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final report
        print(f"\n📊 FINAL RESULTS:")
        print(f"  Total frames processed: {detection_count}")
        print(f"  Multi-device detections: {multi_device_frames}")
        if detection_count > 0:
            success_rate = (multi_device_frames / detection_count) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"1. Update your config.py with these settings:")
        print(f"   CONFIDENCE_THRESHOLD = 0.25")
        print(f"   IOU_THRESHOLD = 0.2")
        print(f"2. Use model: {model_name}")
        print(f"3. Hold objects clearly separated")
        print(f"4. Ensure good lighting conditions")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_multiobject_quick()
