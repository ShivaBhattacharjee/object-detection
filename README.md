# Dynamic Object Detection with Large Dataset Support

Real-time object detection 

## 🚀 Key Features

- **Large Dataset Support**: Objects365 (365 classes), Open Images (601 classes), LVIS (1203 classes)
- **Electronics Focus**: 80-200+ electronics classes vs COCO's ~15
- **Dynamic Class Loading**: Automatically extracts class names from any YOLO model
- **Smart Categorization**: Enhanced categories for modern objects and electronics
- **Scalable**: Works with any dataset size (not limited to COCO's 80 classes)
- **Fast Detection**: Optimized for real-time performance
- **Zero Hardcoding**: No more hardcoded class lists!

## 📊 Dataset Comparison

| Dataset | Classes | Electronics | Use Case |
|---------|---------|-------------|----------|
| ❌ COCO | 80 | ~15 | **TOO LIMITED** |
| ✅ Objects365 | 365 | ~80 | Modern objects, good balance |
| ✅ Open Images | 601 | ~120 | Comprehensive Google dataset |
| ✅ LVIS | 1203 | ~200 | **MOST COMPLETE** public dataset |
| 🎯 Custom | Any | 300+ | Domain-specific training |

## Requirements

- Python 3.8+
- Webcam/camera
- CUDA (optional, for GPU acceleration)

## 🛠️ Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Find larger dataset models**:

   ```bash
   python3 find_large_datasets.py
   ```

3. **Test the dynamic class loading**:

   ```bash
   python3 test_dynamic_classes.py
   ```

4. **Run the application**:

   ```bash
   python3 app.py
   # or
   ./run.sh
   ```

## 🎯 Configuration for Larger Datasets

Edit `config.py` to use models with more classes:

```python
# ❌ COCO models (too limited - only 80 classes)
# YOLO_MODEL = "yolov8n.pt"  

# ✅ LARGER DATASETS (recommended)
YOLO_MODEL = "yolov8n-objects365.pt"  # 365 classes
# YOLO_MODEL = "yolov8n-oiv7.pt"      # 601 classes  
# YOLO_MODEL = "yolov8n-lvis.pt"      # 1203 classes
```

## � Electronics Detection Examples

With larger datasets, you can detect:

**COCO (limited)**: laptop, mouse, keyboard, cell phone, tv, microwave

**Objects365/Open Images/LVIS**: smartphone, tablet, gaming console, router, smart speaker, headphones, charger, drone, smartwatch, VR headset, projector, security camera, smart thermostat, and 100+ more!

### Smart Features

- **Automatic Counting**: "3 people, 2 cars, 1 dog"
- **Category Grouping**: Objects grouped by type (animals, vehicles, etc.)
- **Scene Understanding**: Intelligent scene descriptions
- **Color Coding**: Categories have different colored bounding boxes
- **Dynamic Adaptation**: Works with any YOLO model and dataset size

## 🎮 Usage

### Quick Start

```bash
./run.sh
```

### Manual Start

```bash
python3 app.py
```

### Debug Mode

```bash
python3 debug.py
```

## Controls

- `q` - Quit application
- `c` - Toggle scene analysis on/off
- `s` - Save current frame with detections
- `i` - Show detailed performance info

## Performance

- **Display FPS**: 25-30 FPS (smooth video)
- **Detection FPS**: 15-25 FPS (depends on hardware)
- **GPU Acceleration**: Automatic CUDA detection if available
- **Memory Usage**: Low (YOLOv8 nano model)

## 🏗️ Architecture

```text
Camera → YOLO Detection → Dynamic Classification → Scene Analysis → Display
```

The system now dynamically adapts to any YOLO model:

1. **Model Loading**: Loads any specified YOLO model
2. **Class Extraction**: Automatically extracts class names from model metadata
3. **Dynamic Categorization**: Creates categories based on class name patterns
4. **Smart Classification**: Intelligently classifies and describes objects
5. **Scene Analysis**: Generates natural language scene descriptions


## 📊 Example Output

### Object Detection

- Green boxes around people
- Blue boxes around vehicles  
- Orange boxes around animals
- Different colors for different categories

### Scene Analysis

- "2 people, 1 car, 3 birds"
- "1 person using laptop with mouse"
- "Kitchen scene: 1 person, microwave, refrigerator"

## 🔧 Troubleshooting

1. **Slow performance**:
   - Install CUDA for GPU acceleration
   - Reduce camera resolution in config.py

2. **"Import ultralytics" error**:
   - Run: `pip install ultralytics`

3. **Camera not found**:
   - Check CAMERA_INDEX in config.py
   - Verify camera permissions

4. **Low detection accuracy**:
   - Adjust CONFIDENCE_THRESHOLD in config.py
   - Ensure good lighting conditions

## ✨ Recent Improvements

- ✅ **Dynamic Class Loading**: No longer limited to hardcoded COCO classes
- ✅ **Model Flexibility**: Support for any YOLO model (YOLOv8, YOLOv9, YOLOv10, custom)
- ✅ **Auto-Categorization**: Smart categorization based on class name patterns
- ✅ **Scalable Design**: Works with datasets of any size
- ✅ **Better Configuration**: Easy model switching in config.py

## 📄 License

MIT License
