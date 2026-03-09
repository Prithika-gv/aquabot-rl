from roboflow import Roboflow

rf = Roboflow(api_key="tAUrTdNUEPNOCivdxjmu")
project = rf.workspace("ydieh").project("marine-lcfef")
dataset = project.version(1).download("yolov8", location="C:/datasets/marine")
print("Done! Dataset at:", dataset.location)