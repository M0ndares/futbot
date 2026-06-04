import supervision as sv
from ultralytics import YOLO, SAM
from trackers import ByteTrackTracker
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
model = YOLO(str(base_dir / 'yolov8n.pt'))
model_sam = SAM(str(base_dir / 'sam3.pt'))
tracker = sv.ByteTrack()

def capture():
    capture = cv2.VideoCapture(0)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator =sv.MaskAnnotator()
    tracker.reset()
    while True:
        ret, frame = capture.read()
        result = model(frame, verbose=False)[0]
        pre_tracker = sv.Detections.from_ultralytics(result)
        post_tracker = tracker.update_with_detections(pre_tracker)
        segmented = model_sam(frame, bboxes= post_tracker.xyxy, imgsz=320,)[0]
        post_sv = sv.Detections.from_ultralytics(segmented)
        labels = [model.names[class_id] for class_id in post_tracker.class_id]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=post_tracker)
        if segmented is not None:
            post_sv.class_id = post_tracker.class_id
        annotated = mask_annotator.annotate(scene=annotated, detections = post_sv)
        annotated = label_annotator.annotate(scene=annotated, detections=post_tracker, labels=labels)
        cv2.imshow('webcam', annotated)
        if cv2.waitKey(1) == ord('z'):
            capture.release()
            cv2.destroyAllWindows()
            break
capture()