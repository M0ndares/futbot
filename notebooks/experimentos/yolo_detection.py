import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
model = YOLO('yolov8n.pt')
tracker = sv.ByteTrack()

def capture():
    capture = cv2.VideoCapture(0)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker.reset()
    while True:
        ret, frame = capture.read()
        result = model(frame, verbose=False)[0]
        pre_tracker = sv.Detections.from_ultralytics(result)
        post_tracker = tracker.update_with_detections(pre_tracker)
        labels = [model.names[class_id] for class_id in post_tracker.class_id]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=post_tracker)
        annotated = label_annotator.annotate(scene=annotated, detections=post_tracker, labels=labels)
        cv2.imshow('webcam', annotated)
        if cv2.waitKey(1) == ord('z'):
            capture.release()
            cv2.destroyAllWindows()
capture()