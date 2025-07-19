"""
Configuration settings for Fast YOLO Object Detection with Smart Classification
"""

# YOLO Model Selection - Choose based on your needs
# 
# COCO Models (80 classes - limited dataset):
# YOLO_MODEL = "yolov8n-oiv7.pt"  # Open Images v7, 601 classes
YOLO_MODEL = "yolov8s-oiv7.pt"  # Larger Open Images model, 601 classes, better for multi-object
# YOLO_MODEL = "yolov8n-lvis.pt"  # LVIS dataset, 1203 classes - try this later
# YOLO_MODEL = "yolov8s.pt"  # Small model, better accuracy, 80 classes
# YOLO_MODEL = "yolov8m.pt"  # Medium model, even better accuracy, 80 classes
# YOLO_MODEL = "yolov8l.pt"  # Large model, high accuracy, 80 classes
# YOLO_MODEL = "yolov8x.pt"  # Extra large model, best accuracy, 80 classes
#
# For MUCH LARGER datasets (1000+ classes), try these:
# 
# Open Images Dataset models (600+ classes):
# YOLO_MODEL = "yolov5s-oiv7.pt"     # Open Images v7, 601 classes
# YOLO_MODEL = "yolov8n-oiv7.pt"     # YOLOv8 nano on Open Images
#
# LVIS Dataset models (1200+ classes):
# YOLO_MODEL = "yolov8n-lvis.pt"     # LVIS dataset, 1203 classes
# YOLO_MODEL = "yolov8s-lvis.pt"     # Better accuracy on LVIS
#
# Objects365 Dataset (365 classes - more electronics):
# YOLO_MODEL = "yolov8n-objects365.pt"  # Objects365, good for electronics
#
# Custom models (train your own or download from community):
# YOLO_MODEL = "path/to/your/custom_model.pt"
# YOLO_MODEL = "electronics_detection.pt"  # Example custom electronics model
#
# To download larger dataset models, you may need:
# pip install ultralytics[export]
# Or find community models on Hugging Face, Roboflow, etc.

CONFIDENCE_THRESHOLD = 0.25  # Very low threshold for detecting both phone + iPad
IOU_THRESHOLD = 0.2         # Very low IOU for overlapping/close objects

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
MAX_DETECTIONS_PER_FRAME = 50  # Allow many more detections for multi-object scenes

# Smart classification settings
ENABLE_SCENE_ANALYSIS = True
SHOW_CATEGORY_COLORS = True
SHOW_OBJECT_COUNTS = True
