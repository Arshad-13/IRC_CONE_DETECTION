"""
Real-time Cone Detection using YOLOv8 and OpenCV
Detects blue, red, and yellow cones in live camera feed
"""

import cv2
from ultralytics import YOLO
import argparse
import time

def detect_cones_live(model_path, camera_id=0, conf_threshold=0.25, show_fps=True):
    """
    Detect cones in real-time from camera feed
    
    Args:
        model_path: Path to trained YOLOv8 model
        camera_id: Camera device ID (0 for default webcam)
        conf_threshold: Confidence threshold for detections
        show_fps: Whether to display FPS on screen
    """
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Initialize video capture
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("Real-time Cone Detection Started")
    print("="*60)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'p' to pause/resume")
    print("="*60 + "\n")
    
    # Define colors for each cone class (BGR format)
    colors = {
        'blue': (255, 0, 0),      # Blue
        'red': (0, 0, 255),       # Red
        'yellow': (0, 255, 255)   # Yellow
    }
    
    frame_count = 0
    fps = 0
    start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Run inference
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                # Draw bounding boxes and labels
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    # Get color for this class
                    color = colors.get(class_name, (0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label with class name and confidence
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Calculate label size for background
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            
            if show_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display the frame
            cv2.imshow('Cone Detection - Live Feed', frame)
        else:
            # Display paused message
            paused_frame = frame.copy()
            cv2.putText(
                paused_frame,
                "PAUSED - Press 'p' to resume",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            cv2.imshow('Cone Detection - Live Feed', paused_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"cone_detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as: {filename}")
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Video paused")
            else:
                print("Video resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed. Goodbye!")

def main():
    parser = argparse.ArgumentParser(description='Real-time YOLOv8 Cone Detection')
    parser.add_argument(
        '--model',
        type=str,
        default='runs/detect/cone_detection/weights/best.pt',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (0 for default webcam)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Hide FPS display'
    )
    
    args = parser.parse_args()
    
    # Run live detection
    detect_cones_live(
        model_path=args.model,
        camera_id=args.camera,
        conf_threshold=args.conf,
        show_fps=not args.no_fps
    )

if __name__ == "__main__":
    main()
