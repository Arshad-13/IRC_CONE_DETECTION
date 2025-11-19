# IRC Cone Detection using YOLOv8

Real-time cone detection system using YOLOv8 for detecting blue, red, and yellow cones.

## Features

- ✅ YOLOv8 model trained on custom cone dataset
- ✅ Real-time detection via webcam
- ✅ Image/video inference support
- ✅ 3 cone classes: blue, red, yellow
- ✅ Pre-trained model weights included

## Dataset

- **Training images**: 913
- **Validation images**: 259
- **Test images**: 37
- **Classes**: 3 (blue, red, yellow)
- **Format**: YOLOv8 format with train/valid/test splits

## Installation

```bash
# Clone the repository
git clone https://github.com/Arshad-13/IRC_CONE_DETECTION.git
cd IRC_CONE_DETECTION

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Live Camera Detection

```bash
python live_cone_detection.py
```

**Controls:**
- `q` - Quit
- `s` - Save current frame
- `p` - Pause/Resume

**Options:**
```bash
# Use different camera
python live_cone_detection.py --camera 1

# Adjust confidence threshold
python live_cone_detection.py --conf 0.5

# Hide FPS display
python live_cone_detection.py --no-fps
```

### 2. Image/Video Inference

```bash
# Single image
python inference_cone.py --source path/to/image.jpg

# Directory of images
python inference_cone.py --source path/to/images/

# Video file
python inference_cone.py --source path/to/video.mp4
```

**Advanced Options:**
```bash
python inference_cone.py \
    --model runs/detect/cone_detection/weights/best.pt \
    --source cone.v1i.yolov8/test/images \
    --output runs/detect/my_results \
    --conf 0.5 \
    --iou 0.45
```

### 3. Training (Optional)

To train on your own dataset:

```bash
python train_yolo_cone.py
```

**Training Configuration:**
- **Model**: YOLOv8n (nano) - fastest, good for real-time detection
- **Epochs**: 20
- **Image Size**: 640x640
- **Batch Size**: 8

You can modify these parameters in `train_yolo_cone.py`. For better accuracy, you can use larger models:
- `yolov8n.pt` - Nano (fastest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## Using the Model in Your Code

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/cone_detection/weights/best.pt')

# Run inference
results = model.predict('image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        
        # Get confidence and class
        conf = box.conf[0]
        cls = int(box.cls[0])
        class_name = model.names[cls]
        
        print(f"Detected {class_name} with confidence {conf:.2f}")
```

## Model Information

- **Architecture**: YOLOv8n (nano)
- **Parameters**: 3.01M
- **Training Epochs**: 20
- **Image Size**: 640x640
- **Model Location**: `runs/detect/cone_detection/weights/best.pt`

## Results

Training results and visualizations are available in:
- `runs/detect/cone_detection/` - Training metrics, curves, and confusion matrix
- `runs/detect/inference/` - Inference results with bounding boxes

## Project Structure

```
IRC_CONE_DETECTION/
├── cone.v1i.yolov8/              # Dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/
│   └── detect/
│       └── cone_detection/
│           └── weights/
│               └── best.pt        # Trained model
├── live_cone_detection.py         # Live camera detection
├── inference_cone.py              # Image/video inference
├── train_yolo_cone.py            # Training script
├── requirements.txt               # Dependencies
└── README.md
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `train_yolo_cone.py` (e.g., from 8 to 4)
- Use a smaller model (yolov8n.pt instead of yolov8s.pt)

### Slow Training
- Enable GPU acceleration if available
- Reduce number of workers if CPU is bottleneck
- Use a smaller image size (e.g., 416 instead of 640)

### Camera Not Working
- Check camera ID (try `--camera 1` or `--camera 2`)
- Ensure camera permissions are granted
- Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

## Performance

The model achieves good detection accuracy on cone detection tasks with real-time performance on CPU.

## License

Dataset: CC BY 4.0  
Code: MIT License

## Acknowledgments

- YOLOv8 by Ultralytics
- Dataset from Roboflow Universe

## Author

Arshad - [GitHub](https://github.com/Arshad-13)

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
