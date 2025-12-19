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
    # Initialize YOLOv8 model (using medium model for better accuracy)
    model = YOLO('yolov8m.pt')
    
    # Path to data configuration file
    data_yaml = 'combined_data.yaml'
    
    # Training parameters
    print("Starting training on GPU...")
    results = model.train(
        data=data_yaml,           # path to data.yaml
        epochs=100,               # number of training epochs
        imgsz=640,                # image size
        batch=16,                 # batch size (adjust based on your GPU memory)
        name='cone_detection_combined', # experiment name
        project='runs/detect',    # project directory
        patience=20,              # early stopping patience
        save=True,                # save checkpoints
        save_period=10,           # save checkpoint every N epochs
        device=0,                 # GPU device index (0 for first GPU)
        workers=8,                # number of worker threads
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
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("="*70 + "\n")
    
    # Validate the model
    print("Validating model...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
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
