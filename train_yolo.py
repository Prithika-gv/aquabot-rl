from ultralytics import YOLO

# Load YOLOv8 nano — smallest and fastest, perfect for Pi 5
model = YOLO("yolov8n.pt")

# Train on marine debris dataset
results = model.train(
    data="C:/datasets/marine/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="aquabot_detector",
    device="cpu",         # laptop CPU for now
    patience=10,          # stop early if not improving
    save=True,
    plots=True            # generates accuracy graphs for paper
)

print("✅ Training complete!")
print(f"Best model saved to: runs/detect/aquabot_detector/weights/best.pt")