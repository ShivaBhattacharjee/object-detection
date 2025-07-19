"""
Configuration settings for Fast YOLO Object Detection with Smart Classification
"""

# Fast YOLO Detection settings
YOLO_MODEL = "yolov8n.pt"  # YOLOv8 nano for fastest performance
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Display settings
WINDOW_NAME = "Fast Object Detection with Smart Classification"
BORDER_COLOR = (0, 255, 0)  # Green color in BGR format
BORDER_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BACKGROUND_COLOR = (0, 0, 0)  # Black background

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Performance settings
TARGET_FPS = 30
SKIP_FRAMES = 0  # Process every frame for smooth detection
MAX_DETECTIONS_PER_FRAME = 20  # Allow more detections

# Smart classification settings
ENABLE_SCENE_ANALYSIS = True
SHOW_CATEGORY_COLORS = True
SHOW_OBJECT_COUNTS = True
