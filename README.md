# Fast Object Detection with YOLO + Smart Classification

Real-time object detection using YOLOv8 with intelligent object classification and scene analysis.

## Features

- **Fast Detection**: YOLOv8 nano for real-time performance (15-30 FPS)
- **Smart Classification**: 80 COCO object classes with intelligent categorization
- **Object Counting**: Automatic counting of detected objects (e.g., "2 people, 3 cars")
- **Scene Analysis**: Natural language scene descriptions
- **Category Colors**: Different colors for different object categories
- **No Hardcoding**: Uses comprehensive COCO dataset for object recognition

## Requirements

- Python 3.8+
- Webcam/camera
- CUDA (optional, for GPU acceleration)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the system**:
   ```bash
   python3 test.py
   ```

3. **Run the application**:
   ```bash
   python3 app.py
   # or
   ./run.sh
   ```

## Object Detection Capabilities

### Supported Objects (80 COCO classes):
- **People**: person detection with counting
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone, microwave
- **Food**: banana, apple, sandwich, orange, pizza, hot dog, cake
- **Furniture**: chair, couch, bed, dining table
- **And many more...**

### Smart Features:
- **Automatic Counting**: "3 people, 2 cars, 1 dog"
- **Category Grouping**: Objects grouped by type (animals, vehicles, etc.)
- **Scene Understanding**: Intelligent scene descriptions
- **Color Coding**: Categories have different colored bounding boxes

## Usage

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

## Architecture

```
Camera → YOLOv8 Detection → Smart Classification → Scene Analysis → Display
```

## Files

- `app.py` - Main application
- `fast_detector.py` - YOLOv8-based fast object detector
- `smart_classifier.py` - Intelligent object classification system
- `config.py` - Configuration settings
- `test.py` - System testing
- `debug.py` - Debug and performance testing
- `run.sh` - Startup script

## Example Output

### Object Detection:
- Green boxes around people
- Blue boxes around vehicles  
- Orange boxes around animals
- Different colors for different categories

### Scene Analysis:
- "2 people, 1 car, 3 birds"
- "1 person using laptop with mouse"
- "Kitchen scene: 1 person, microwave, refrigerator"

## Troubleshooting

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

## License

MIT License
