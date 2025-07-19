#!/usr/bin/env python3
"""
Real-time Object Detection with Fast YOLO + Smart Classification
"""

import cv2
import logging
import time
import signal
import sys

from config import WINDOW_NAME, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, ENABLE_SCENE_ANALYSIS
from fast_detector import FastObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeDetectionApp:
    """Real-time object detection with Fast YOLO + Smart Classification"""
    
    def __init__(self):
        self.detector = None
        self.cap = None
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def setup(self) -> bool:
        """Setup all components"""
        try:
            # Initialize fast detector
            logger.info("� Initializing Fast YOLO detector...")
            self.detector = FastObjectDetector()
            
            if not self.detector.initialize():
                logger.error("❌ Failed to initialize detector")
                return False
            
            logger.info("✅ Fast YOLO detector ready")
            
            # Initialize camera
            logger.info("📷 Initializing camera...")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            
            if not self.cap.isOpened():
                logger.error("❌ Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame with Fast YOLO detection"""
        # Detect objects using YOLO
        detections = self.detector.detect_objects(frame)
        
        # Get scene analysis if enabled
        scene_analysis = None
        if ENABLE_SCENE_ANALYSIS and detections:
            scene_analysis = self.detector.analyze_scene(detections)
        
        # Draw detections with smart classification
        result_frame = self.detector.draw_detections(frame, detections, scene_analysis)
        
        return result_frame, detections, scene_analysis
    
    def add_overlay_info(self, frame, detections, scene_analysis=None):
        """Add FPS and status info to frame"""
        # Update FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
        
        # Get detection performance
        perf_stats = self.detector.get_performance_stats()
        detection_fps = perf_stats.get('fps', 0)
        
        # Status text
        status = f"Display: {self.current_fps:.1f}fps | Detection: {detection_fps:.1f}fps | Objects: {len(detections)}"
        
        # Draw status
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls info
        controls = "Controls: 'q' quit | 's' save | 'i' info | 'c' toggle scene analysis"
        cv2.putText(frame, controls, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main application loop"""
        if not self.setup():
            logger.error("❌ Failed to setup application")
            return
        
        self.running = True
        logger.info("🚀 Starting real-time object detection with Fast YOLO")
        logger.info("Press 'q' to quit, 's' to save frame, 'i' for info, 'c' to toggle scene analysis")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame from camera")
                    break
                
                # Process frame (detect objects + analyze scene)
                processed_frame, detections, scene_analysis = self.process_frame(frame)
                
                # Add overlay information
                self.add_overlay_info(processed_frame, detections, scene_analysis)
                
                # Show frame
                cv2.imshow(WINDOW_NAME, processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("👋 Quit requested by user")
                    break
                elif key == ord('c'):
                    # Toggle scene analysis
                    import config
                    config.ENABLE_SCENE_ANALYSIS = not config.ENABLE_SCENE_ANALYSIS
                    status = "ON" if config.ENABLE_SCENE_ANALYSIS else "OFF"
                    logger.info(f"🔄 Scene analysis: {status}")
                elif key == ord('s'):
                    filename = f"detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"💾 Frame saved as {filename}")
                elif key == ord('i'):
                    self.print_info(detections, scene_analysis)
        
        except KeyboardInterrupt:
            logger.info("👋 Application interrupted")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def print_info(self, detections, scene_analysis=None):
        """Print detailed information"""
        logger.info("=" * 50)
        logger.info("📊 SYSTEM INFO")
        logger.info("=" * 50)
        logger.info(f"Current Display FPS: {self.current_fps:.2f}")
        logger.info(f"Frames processed: {self.fps_counter}")
        logger.info(f"Current detections: {len(detections)}")
        
        # Performance stats
        perf_stats = self.detector.get_performance_stats()
        logger.info(f"Detection FPS: {perf_stats.get('fps', 0):.2f}")
        logger.info(f"Avg detection time: {perf_stats.get('avg_detection_time', 0)*1000:.1f}ms")
        logger.info(f"Detection device: {perf_stats.get('device', 'Unknown')}")
        
        # Model information
        if hasattr(self.detector, 'classifier'):
            total_classes = self.detector.classifier.get_total_classes()
            categories = self.detector.classifier.get_all_categories()
            logger.info(f"Model classes: {total_classes}")
            logger.info(f"Categories: {', '.join(categories)}")
        
        if detections:
            logger.info("Current objects:")
            for det in detections:
                logger.info(f"  - {det['display_name']} ({det['category']}): {det['confidence']:.1%}")
        
        if scene_analysis:
            logger.info("Scene Analysis:")
            logger.info(f"  Description: {scene_analysis.get('scene_description', 'None')}")
            logger.info(f"  Categories: {scene_analysis.get('categories', {})}")
            logger.info(f"  Object counts: {scene_analysis.get('object_counts', {})}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup completed")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"📡 Received signal {signum}, shutting down...")
        self.running = False


def main():
    """Main entry point"""
    app = RealtimeDetectionApp()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)
    
    try:
        app.run()
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
