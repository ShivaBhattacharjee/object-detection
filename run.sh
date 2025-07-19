#!/bin/bash

#!/bin/bash

# Fast Object Detection with YOLO + Smart Classification - Startup Script
echo "🚀 Starting Fast Object Detection with YOLO + Smart Classification"
echo "================================================================="

# Check Python dependencies
echo "📦 Checking dependencies..."
python3 -c "import cv2, numpy, torch, ultralytics" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies available"
else
    echo "⚠️  Some dependencies missing. Installing..."
    pip3 install -r requirements.txt
fi

# Test the system
echo "🧪 Testing Fast YOLO system..."
python3 test.py

if [ $? -ne 0 ]; then
    echo "❌ System test failed!"
    echo "Please fix the issues above before running the app."
    exit 1
fi

echo ""
echo "🎯 Controls:"
echo "   'q' - Quit application"
echo "   'c' - Toggle scene analysis"
echo "   's' - Save current frame"
echo "   'i' - Show performance info"
echo ""
echo "Starting Fast YOLO detection..."
echo ""

# Run the application
echo "🚀 Launching real-time detection..."
python3 app.py
python app.py
