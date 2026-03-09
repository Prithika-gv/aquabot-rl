from ultralytics import YOLO

model = YOLO("runs/detect/aquabot_detector2/weights/best.pt")

# Run on test images
results = model.predict(
    source="C:/datasets/marine/test/images",
    save=True,          # saves images with bounding boxes
    conf=0.25,          # confidence threshold
    name="aquabot_test"
)

# Print metrics
metrics = model.val(data="C:/datasets/marine/data.yaml")
print(f"\n{'='*40}")
print(f"  YOLO RESULTS FOR PAPER")
print(f"{'='*40}")
print(f"  mAP@0.5      : {metrics.box.map50:.3f}")
print(f"  mAP@0.5-0.95 : {metrics.box.map:.3f}")
print(f"  Precision    : {metrics.box.p.mean():.3f}")
print(f"  Recall       : {metrics.box.r.mean():.3f}")
print(f"{'='*40}")
