#!/usr/bin/env python3
"""
Test Multi-Object Detection - Specifically for phone + iPad scenarios
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL
import time

class MultiObjectTester:
    """Test and improve multi-object detection"""
    
    def __init__(self):
        print("🔄 Loading model for multi-object testing...")
        self.model = YOLO(YOLO_MODEL)
        print(f"✅ Loaded {YOLO_MODEL} with {len(self.model.names)} classes")
        
        # Find relevant classes
        self.find_device_classes()
    
    def find_device_classes(self):
        """Find phone, tablet, and related device classes"""
        self.device_classes = {}
        
        # Keywords to search for
        device_keywords = {
            'phones': ['phone', 'mobile', 'smartphone', 'cell phone', 'cellular telephone'],
            'tablets': ['tablet', 'ipad', 'tablet computer'],
            'computers': ['computer', 'laptop', 'monitor', 'desktop'],
            'electronics': ['camera', 'gaming', 'electronic', 'device']
        }
        
        for category, keywords in device_keywords.items():
            self.device_classes[category] = []
            for class_id, class_name in self.model.names.items():
                if any(keyword in class_name.lower() for keyword in keywords):
                    self.device_classes[category].append((class_id, class_name))
        
        print("\n📱 DEVICE CLASSES FOUND:")
        for category, classes in self.device_classes.items():
            print(f"  {category.upper()}: {[name for _, name in classes]}")
    
    def test_detection_settings(self, frame):
        """Test different detection settings for multi-object scenarios"""
        print("\n🧪 TESTING DIFFERENT DETECTION SETTINGS:")
        
        # Test configurations
        test_configs = [
            {"conf": 0.5, "iou": 0.4, "max_det": 30, "name": "Default"},
            {"conf": 0.3, "iou": 0.3, "max_det": 50, "name": "Sensitive"},
            {"conf": 0.25, "iou": 0.25, "max_det": 50, "name": "Very Sensitive"},
            {"conf": 0.4, "iou": 0.2, "max_det": 50, "name": "Low IOU"},
        ]
        
        results_summary = []
        
        for config in test_configs:
            print(f"\n  🔍 Testing {config['name']} settings:")
            print(f"     Confidence: {config['conf']}, IOU: {config['iou']}, Max Det: {config['max_det']}")
            
            start_time = time.time()
            results = self.model(
                frame,
                conf=config['conf'],
                iou=config['iou'],
                max_det=config['max_det'],
                verbose=False
            )
            detection_time = time.time() - start_time
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                num_detections = len(boxes)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                
                # Count device types
                device_counts = {category: 0 for category in self.device_classes.keys()}
                detected_devices = []
                
                for class_id, conf in zip(class_ids, confidences):
                    class_name = self.model.names[class_id]
                    detected_devices.append(f"{class_name} ({conf:.2f})")
                    
                    # Categorize
                    for category, classes in self.device_classes.items():
                        if any(class_id == device_id for device_id, _ in classes):
                            device_counts[category] += 1
                
                print(f"     ✅ {num_detections} total detections in {detection_time:.3f}s")
                print(f"     📱 Devices: {dict(device_counts)}")
                print(f"     📋 Detected: {detected_devices}")
                
                results_summary.append({
                    'config': config['name'],
                    'total': num_detections,
                    'devices': sum(device_counts.values()),
                    'time': detection_time,
                    'details': device_counts
                })
            else:
                print(f"     ❌ No detections")
                results_summary.append({
                    'config': config['name'],
                    'total': 0,
                    'devices': 0,
                    'time': detection_time,
                    'details': {}
                })
        
        # Show best configuration
        print(f"\n📊 DETECTION RESULTS SUMMARY:")
        for result in results_summary:
            print(f"  {result['config']:15} | Total: {result['total']:2d} | Devices: {result['devices']:2d} | Time: {result['time']:.3f}s")
        
        # Recommend best setting
        best_config = max(results_summary, key=lambda x: x['devices'])
        print(f"\n💡 RECOMMENDATION: Use '{best_config['config']}' settings for best device detection")
        
        return results_summary
    
    def run_live_test(self):
        """Run live camera test for multi-object detection"""
        print("\n🎥 STARTING LIVE MULTI-OBJECT TEST")
        print("Hold your phone and iPad together to test detection!")
        print("Press 'q' to quit, 's' to save frame for analysis")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Use best settings for multi-object detection
        detection_params = {
            'conf': 0.3,   # Lower confidence for more detections
            'iou': 0.25,   # Lower IOU for overlapping objects
            'max_det': 50  # Allow many detections
        }
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, **detection_params, verbose=False)
            
            # Process results
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                device_count = 0
                
                # Draw detections
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_name = self.model.names[class_id]
                    
                    # Check if it's a device
                    is_device = False
                    for category, classes in self.device_classes.items():
                        if any(class_id == device_id for device_id, _ in classes):
                            is_device = True
                            device_count += 1
                            break
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if is_device else (255, 0, 0)  # Green for devices, blue for others
                    thickness = 3 if is_device else 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label
                    label = f"{class_name} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (255, 255, 255), 2)
                
                # Show device count
                info_text = f"Devices detected: {device_count} | Frame: {frame_count}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 255), 2)
            
            cv2.imshow('Multi-Object Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame for analysis
                filename = f"test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("🔧 MULTI-OBJECT DETECTION TESTER")
    print("=" * 50)
    
    tester = MultiObjectTester()
    
    # Test with camera
    print("\n1. Testing with live camera...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("📸 Captured test frame")
            tester.test_detection_settings(frame)
        cap.release()
        
        print("\n2. Starting live test...")
        time.sleep(2)
        tester.run_live_test()
    else:
        print("❌ Camera not available")
    
    print("\n✨ TIPS FOR BETTER MULTI-OBJECT DETECTION:")
    print("1. Hold objects clearly separated")
    print("2. Ensure good lighting")
    print("3. Keep objects at different distances from camera")
    print("4. Try different angles")
    print("5. Make sure objects are clearly visible (not obscured)")

if __name__ == "__main__":
    main()
