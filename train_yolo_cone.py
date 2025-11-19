"""
YOLOv8 Cone Detection Model Training Script
Trains a YOLOv8 model to detect blue, red, and yellow cones
"""

from ultralytics import YOLO
import os

def train_cone_detection_model():
    """
    Train YOLOv8 model for cone detection
    """
    # Initialize YOLOv8 model (using nano model for faster training)
    # You can change 'yolov8n.pt' to 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' for larger models
    model = YOLO('yolov8n.pt')
    
    # Path to data configuration file
    data_yaml = 'cone.v1i.yolov8/data.yaml'
    
    # Training parameters
    results = model.train(
        data=data_yaml,           # path to data.yaml
        epochs=20,                # number of training epochs
        imgsz=640,                # image size
        batch=8,                  # batch size (adjust based on your GPU memory)
        name='cone_detection',    # experiment name
        project='runs/detect',    # project directory
        patience=50,              # early stopping patience
        save=True,                # save checkpoints
        save_period=10,           # save checkpoint every N epochs
        device='cpu',             # GPU device (use 'cpu' if no GPU available)
        workers=4,                # number of worker threads
        optimizer='auto',         # optimizer (auto, SGD, Adam, AdamW)
        verbose=True,             # verbose output
        seed=42,                  # random seed for reproducibility
        pretrained=True,          # use pretrained weights
        lr0=0.01,                 # initial learning rate
        lrf=0.01,                 # final learning rate factor
        momentum=0.937,           # SGD momentum
        weight_decay=0.0005,      # weight decay
        warmup_epochs=3.0,        # warmup epochs
        box=7.5,                  # box loss gain
        cls=0.5,                  # cls loss gain
        dfl=1.5,                  # dfl loss gain
        plots=True,               # save plots
        val=True,                 # validate during training
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    # Get the best model path
    best_model_path = 'runs/detect/cone_detection/weights/best.pt'
    last_model_path = 'runs/detect/cone_detection/weights/last.pt'
    
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")
    
    # Load the best model for validation
    best_model = YOLO(best_model_path)
    
    # Validate the model
    print("\n" + "="*70)
    print("Validating the best model...")
    print("="*70)
    metrics = best_model.val()
    
    # Print validation metrics
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Export the model to different formats (optional)
    print("\n" + "="*70)
    print("Exporting model to ONNX format...")
    print("="*70)
    onnx_path = best_model.export(format='onnx')
    print(f"ONNX model saved at: {onnx_path}")
    
    # Test the model on test set
    print("\n" + "="*70)
    print("Testing model on test set...")
    print("="*70)
    test_results = best_model.predict(
        source='cone.v1i.yolov8/test/images',
        save=True,
        save_txt=True,
        save_conf=True,
        project='runs/detect',
        name='cone_detection_test',
        conf=0.25,
        iou=0.45,
    )
    
    print(f"\nTest predictions saved in: runs/detect/cone_detection_test/")
    
    print("\n" + "="*70)
    print("✓ Model training and evaluation complete!")
    print("="*70)
    print("\nModel files:")
    print(f"  - Best weights: {best_model_path}")
    print(f"  - Last weights: {last_model_path}")
    print(f"  - ONNX export: {onnx_path}")
    print("\nTo use the model for inference:")
    print(f"  model = YOLO('{best_model_path}')")
    print(f"  results = model.predict('path/to/image.jpg')")
    
    return best_model_path

if __name__ == "__main__":
    print("="*70)
    print("YOLOv8 Cone Detection Model Training")
    print("="*70)
    print("\nDataset: Cone Detection (blue, red, yellow cones)")
    print("Model: YOLOv8n (nano)")
    print("\nStarting training...\n")
    
    try:
        model_path = train_cone_detection_model()
        print(f"\n✓ Success! Model saved at: {model_path}")
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        raise
