import os
import shutil
from pathlib import Path

# Source datasets
datasets = [
    "C:/datasets/marine",
    "C:/datasets/river_fix",
    
]

# Output merged dataset
output = "C:/datasets/aquabot_merged"

for split in ["train", "valid", "test"]:
    os.makedirs(f"{output}/{split}/images", exist_ok=True)
    os.makedirs(f"{output}/{split}/labels", exist_ok=True)

total = 0
for dataset in datasets:
    for split in ["train", "valid", "test"]:
        img_dir = Path(f"{dataset}/{split}/images")
        lbl_dir = Path(f"{dataset}/{split}/labels")
        
        if not img_dir.exists():
            continue
            
        images = list(img_dir.glob("*"))
        for img in images:
            # Copy image
            shutil.copy2(img, f"{output}/{split}/images/{img.name}")
            # Copy label
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, f"{output}/{split}/labels/{lbl.name}")
        
        total += len(images)
        print(f"✅ {dataset.split('/')[-1]} {split}: {len(images)} images")

print(f"\nTotal images merged: {total}")