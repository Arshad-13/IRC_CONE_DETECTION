from ultralytics import YOLO
import time
import glob
import torch
import os

# Configuration
MODEL_PATH = 's:/cone/runs/detect/cone_detection_combined/weights/best.pt'
IMAGES_PATTERN = 's:/cone/cone.v1i.yolov8/test/images/*.jpg'
NUM_FRAMES_TO_TEST = 200  # Limit frames to avoid waiting too long

def benchmark():
    # Check device
    device = '0' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    
    print(f"Initializing Benchmark...")
    print(f"Device: {device} ({device_name})")
    print(f"Model: {MODEL_PATH}")
    
    # Load model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get images
    images = glob.glob(IMAGES_PATTERN)
    if not images:
        print(f"No images found at {IMAGES_PATTERN}")
        return
    
    print(f"Found {len(images)} images. Using {min(len(images), NUM_FRAMES_TO_TEST)} for test.")

    # Warmup
    print("Warming up GPU...")
    for _ in range(10):
        model.predict(images[0], device=device, verbose=False)

    # Benchmark Loop
    print("Starting benchmark run...")
    start_time = time.time()
    count = 0
    
    for img in images:
        if count >= NUM_FRAMES_TO_TEST:
            break
        
        # Run inference
        # verbose=False prevents printing to console for every frame
        results = model.predict(img, device=device, verbose=False)
        count += 1

    end_time = time.time()
    
    # Calculate stats
    total_time = end_time - start_time
    fps = count / total_time
    latency = (total_time / count) * 1000

    print("\n" + "="*40)
    print(f" BENCHMARK RESULTS")
    print("="*40)
    print(f" Processed:      {count} frames")
    print(f" Total Time:     {total_time:.2f} seconds")
    print(f" Average FPS:    {fps:.2f} FPS")
    print(f" Average Latency: {latency:.2f} ms")
    print("="*40)
    
    if fps > 30:
        print("✅ Status: REAL-TIME READY (>30 FPS)")
    else:
        print("⚠️ Status: BELOW REAL-TIME (<30 FPS)")

if __name__ == "__main__":
    benchmark()
