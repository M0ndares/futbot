import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import supervision as sv
from pathlib import Path

CURRENT_PATH = Path.cwd()
VIDEO_PATH = CURRENT_PATH / "videos/prueba4.mp4" # prueba1.mp4, prueba2.mp4, prueba3.MOV
OUTPUT_PATH = CURRENT_PATH / "videos/resultado_segmentado4.mp4"
MATRIZ_PATH = CURRENT_PATH / "matriz_homografia.npy"
WINDOW_NAME = "Calibrador de Homografia"

ANCHO_REAL = 300
ALTO_REAL = 400

PUNTOS_REALES = np.array([
    [0, 0],              
    [ANCHO_REAL - 50, 0], 
    [ANCHO_REAL, ALTO_REAL - 100],     
    [ANCHO_REAL, ALTO_REAL], 
    [0, ALTO_REAL],       
], dtype=np.float32)

ESCALA = 0.3 
puntos_video = []
frame_calibracion = None
mapa_calor_acumulado = np.zeros((ALTO_REAL, ANCHO_REAL), dtype=np.float32)

def clic_mouse(event, x, y, flags, param):
    global puntos_video, frame_calibracion
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos_video) < len(PUNTOS_REALES):
            x_real = int(x / ESCALA)
            y_real = int(y / ESCALA)
            puntos_video.append([x_real, y_real])
            cv2.circle(frame_calibracion, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(frame_calibracion, f"P{len(puntos_video)}", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame_calibracion)

def ejecutar_calibracion():
    global frame_calibracion
    if os.path.exists(MATRIZ_PATH):
        return np.load(MATRIZ_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        exit()
    ancho_ajustado = int(frame.shape[1] * ESCALA)
    alto_ajustado = int(frame.shape[0] * ESCALA)
    frame_calibracion = cv2.resize(frame, (ancho_ajustado, alto_ajustado))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, clic_mouse)
    while True:
        cv2.imshow(WINDOW_NAME, frame_calibracion)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(puntos_video) == len(PUNTOS_REALES):
            break
    cv2.destroyAllWindows()
    if len(puntos_video) == len(PUNTOS_REALES):
        puntos_video_np = np.array(puntos_video, dtype=np.float32)
        H, _ = cv2.findHomography(puntos_video_np, PUNTOS_REALES, cv2.RANSAC, 5.0)
        np.save(MATRIZ_PATH, H)
        return H
    else:
        exit()

H = ejecutar_calibracion()
yolo_model = YOLO(str(CURRENT_PATH / "runs/segment/train/weights/best.pt"))
sam_model = SAM(str(CURRENT_PATH / "../notebooks/sam3.pt"))

tracker = sv.ByteTrack()

mask_annotator = sv.MaskAnnotator()              
trace_annotator = sv.TraceAnnotator(trace_length=30) 
label_annotator = sv.LabelAnnotator()             

def obtener_posiciones_reales(detections, matriz_h):
    if len(detections) == 0:
        return []
    puntos_video = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    puntos_reshaped = np.array([puntos_video], dtype=np.float32)
    puntos_proyectados = cv2.perspectiveTransform(puntos_reshaped, matriz_h)[0]
    return puntos_proyectados

def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    yolo_results = yolo_model(frame, conf=0.5, verbose=False)[0]
    
    if len(yolo_results.boxes) > 0:
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        confidences = yolo_results.boxes.conf.cpu().numpy()
        sam_results = sam_model.predict(frame, bboxes=boxes, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(sam_results)
        detections.class_id = class_ids
        detections.confidence = confidences
    else:
        detections = sv.Detections.empty()
    
    detections = tracker.update_with_detections(detections)
    annotated_frame = frame.copy()        
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    labels = []
    for idx, class_id in enumerate(detections.class_id):
        name = yolo_model.names[class_id].lower()
        labels.append(f"{yolo_model.names[class_id]} ID: {name}")


if __name__ == "__main__":        
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=OUTPUT_PATH,
        callback=process_frame
    )