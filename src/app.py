import cv2
from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = "videos/prueba3.MOV"
OUTPUT_PATH = "videos/resultado_segmentado3.mov"

yolo_model = YOLO("runs/segment/train-2/weights/best.pt") 
tracker = sv.ByteTrack()

mask_annotator = sv.MaskAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=30) 
label_annotator = sv.LabelAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()

def process_frame(frame, frame_idx):
    results = yolo_model(frame, conf=0.6, classes=[0, 2, 5])[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    annotated_frame = frame.copy()        
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = heat_map_annotator.annotate(scene=annotated_frame, detections=detections)

    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            name = yolo_model.names[class_id] 
            labels.append(f"{name} ID: {tracker_id}")
            
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)    
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    else:
        labels = [yolo_model.names[class_id] for class_id in detections.class_id]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

# Ejecución del proceso
sv.process_video(
    source_path=VIDEO_PATH,
    target_path=OUTPUT_PATH,
    callback=process_frame
)