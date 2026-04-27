from roboflow import Roboflow

rf = Roboflow(api_key="21j2F5Dx7zstl5kxCZs2")

# Dataset 1 - River Trash Fix (2.2k images)
print("Downloading River Trash Fix...")
project1 = rf.workspace("kharisma-punya").project("river-trash-fix-meenc")
dataset1 = project1.version(9).download("yolov8", location="C:/datasets/river_fix")

# Dataset 2 - River Trash Final (1.7k images)  
# Find correct version for river_final
project2 = rf.workspace("kharisma-punya").project("river-trash-final-nqvzd")
for v in range(1, 10):
    try:
        dataset2 = project2.version(v).download("yolov8", location="C:/datasets/river_final")
        print(f"✅ Found at version {v}")
        break
    except:
        print(f"version {v} not found, trying next...")
print("Both downloaded!")