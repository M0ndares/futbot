import cv2
from ultralytics import YOLO, SAM  # <--- Importamos ambos
import supervision as sv

VIDEO_PATH = "videos/prueba1.mov"
OUTPUT_PATH = "videos/resultado_segmentado.mov"

yolo_model = YOLO("runs/segment/train/weights/best.pt") 
sam_model = SAM("../notebooks/sam3.pt") 

box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=30) 
heat_map_annotator = sv.HeatMapAnnotator()

def process_frame(frame, frame_idx):
    results = yolo_model.track(frame, persist=True, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    if results.boxes.id is not None:
        detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
    
    labels = []
    if detections.tracker_id is not None:
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            name = yolo_model.names[class_id] 
            labels.append(f"{name} ID: {tracker_id}")
            
    annotated_frame = frame.copy()    
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)    
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = heat_map_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame

sv.process_video(
    source_path=VIDEO_PATH,
    target_path=OUTPUT_PATH,
    callback=process_frame
)
print(f"Video procesado guardado en: {OUTPUT_PATH}")