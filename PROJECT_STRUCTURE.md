# Object Detection - Clean Repository Structure

## 🚀 Core Application Files
- `app.py` - Main object detection application
- `fast_detector.py` - YOLO detection engine
- `smart_classifier.py` - Dynamic classification system
- `config.py` - Configuration settings

## 🛠️ Utility Scripts
- `download_large_model.py` - Download models with larger datasets
- `check_current_model.py` - Check which model is active
- `debug.py` - Debug and performance testing
- `test.py` - System testing
- `run.sh` - Quick startup script

## 📋 Configuration & Documentation
- `README.md` - Complete documentation
- `requirements.txt` - Python dependencies

## 🤖 AI Models
- `yolov8n.pt` - COCO dataset (80 classes) - backup
- `yolov8m-oiv7.pt` - Open Images v7 (601 classes) - **ACTIVE**

## 🎯 Quick Start
1. **Run detection:** `python3 app.py`
2. **Check model:** `python3 check_current_model.py`
3. **Download bigger models:** `python3 download_large_model.py yolov8n-lvis.pt`

Repository is now clean and ready for production! 🎉
