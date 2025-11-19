"""
YOLOv8 Cone Detection Inference Script
Use the trained model to detect cones in images or videos
"""

from ultralytics import YOLO
import cv2
import os
import argparse

def detect_cones(model_path, source, output_dir='runs/detect/inference', conf=0.25, iou=0.45):
    """
    Detect cones in images or videos using trained YOLOv8 model
    
    Args:
        model_path: Path to trained model weights
        source: Path to image, video, or directory
        output_dir: Directory to save results
        conf: Confidence threshold
        iou: IoU threshold for NMS
    """
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on: {source}")
    results = model.predict(
        source=source,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='results',
        conf=conf,
        iou=iou,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    print(f"\n✓ Detection complete! Results saved in: {output_dir}/results/")
    
    # Print detection summary
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"\nImage {i+1}:")
        print(f"  Total detections: {len(boxes)}")
        
        # Count detections by class
        if len(boxes) > 0:
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            for cls_name, count in class_counts.items():
                print(f"  - {cls_name}: {count}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Cone Detection Inference')
    parser.add_argument('--model', type=str, default='runs/detect/cone_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or directory')
    parser.add_argument('--output', type=str, default='runs/detect/inference',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (0-1)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first using train_yolo_cone.py")
        return
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"Error: Source not found at {args.source}")
        return
    
    # Run detection
    detect_cones(args.model, args.source, args.output, args.conf, args.iou)

if __name__ == "__main__":
    main()
