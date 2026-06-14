from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="IducOaHIU8ZFz1tf5ba2")
project = rf.workspace("osval-elias-montesinos-valladares").project("futbot-2026-8o5yc")
version = project.version(1)
dataset = version.download("yolov8")
model = YOLO("yolov8n-seg.pt") 

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=250,                  
    imgsz=640,        
    patience=25,          
    device='cpu'                  
)