from ultralytics import YOLO

# Load YOLOv8 nano — smallest and fastest, perfect for Pi 5
model = YOLO("yolov8n.pt")

# Train on marine debris dataset
results = model.train(
    data="C:/datasets/aquabot_merged/aquabot_merged.yaml",
    epochs=100,
    batch=16,
    name="aquabot_v2",
    imgsz=640,
    device="cpu",         # laptop CPU for now
    patience=10,          # stop early if not improving
    save=True,
    plots=True            # generates accuracy graphs for paper
)

print("✅ Training complete!")
print(f"Best model saved to: runs/detect/aquabot_detector/weights/best.pt")