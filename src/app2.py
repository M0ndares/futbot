import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM
import supervision as sv
from pathlib import Path

CURRENT_PATH = Path.cwd()
VIDEO_PATH = CURRENT_PATH / "videos/prueba4.mp4" 
OUTPUT_PATH = CURRENT_PATH / "videos/paso3_yolo_sam.mp4"
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

# ==========================================
# 📐 PASO 1: CALIBRACIÓN (ESQUINAS)
# ==========================================
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

# Cargar Matriz (Paso 1)
H = ejecutar_calibracion()

# Cargar Modelos de Redes Neuronales
yolo_model = YOLO(str(CURRENT_PATH / "runs/segment/train/weights/best.pt"))
sam_model = SAM(str(CURRENT_PATH / "../notebooks/sam3.pt"))

# Anotadores básicos para el entregable visual
mask_annotator = sv.MaskAnnotator() 
box_annotator = sv.BoxAnnotator() # Para pintar el cuadro de YOLO

# ==========================================
# 🧠 PASOS 2 Y 3: PROCESAMIENTO DE FRAMES
# ==========================================
def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    # 2. PASO 2: YOLO (Detección de Bounding Boxes)
    yolo_results = yolo_model(frame, conf=0.5, verbose=False)[0]
    
    if len(yolo_results.boxes) > 0:
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        confidences = yolo_results.boxes.conf.cpu().numpy()
        
        # 3. PASO 3: SAM (Segmentación de Polígonos con Máscaras)
        sam_results = sam_model.predict(frame, bboxes=boxes, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(sam_results)
        detections.class_id = class_ids
        detections.confidence = confidences
    else:
        detections = sv.Detections.empty()
    
    # Clonamos el frame original para pintarle encima
    annotated_frame = frame.copy() 
    
    if len(detections) > 0:
        # Pintamos el Paso 3 (SAM - Las máscaras de color traslúcido)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        # Pintamos el Paso 2 (YOLO - Las cajas de bordes sólidos)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame

if __name__ == "__main__": 
    print("🚀 Generando video del Paso 3 (YOLO + SAM)...")       
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=OUTPUT_PATH,
        callback=process_frame
    )
    print(f"🎯 ¡Listo! Video exportado en: {OUTPUT_PATH}")