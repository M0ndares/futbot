import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path

CURRENT_PATH = Path.cwd()
VIDEO_PATH = CURRENT_PATH / "videos/prueba1.mp4"
OUTPUT_PATH = CURRENT_PATH / "videos/resultado_segmentado1.mp4"
MATRIZ_PATH = CURRENT_PATH / "matriz_homografia.npy"
WINDOW_NAME = "Calibrador de Homografia"

ANCHO_REAL = 300
ALTO_REAL = 400

PUNTOS_REALES = np.array([
    [0, 0],              
    [ANCHO_REAL - 50, 0], 
    [ANCHO_REAL, 50],     
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
yolo_model = YOLO("runs/segment/train/weights/best.pt") 
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

def es_robot_aliado(frame, bbox):
    """ Corta el pedazo de robot y busca si contiene el verde acrílico del equipo """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
        
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 100, 100])
    verde_alto = np.array([80, 255, 255])
    
    mascara_verde = cv2.inRange(hsv_roi, verde_bajo, verde_alto)
    conteo_pixeles_verdes = cv2.countNonZero(mascara_verde)
    
    return conteo_pixeles_verdes > 1000

def dibujar_campo_tactico(detections, posiciones_reales, frame_original):
    global mapa_calor_acumulado
    mini_campo = np.zeros((ALTO_REAL, ANCHO_REAL, 3), dtype=np.uint8)
    mini_campo[:] = (34, 139, 34) 
    
    cv2.rectangle(mini_campo, (10, 10), (ANCHO_REAL - 10, ALTO_REAL - 10), (255, 255, 255), 2)
    cv2.line(mini_campo, (10, ALTO_REAL // 2), (ANCHO_REAL - 10, ALTO_REAL // 2), (255, 255, 255), 2)
    cv2.circle(mini_campo, (ANCHO_REAL // 2, ALTO_REAL // 2), 40, (255, 255, 255), 2)

    if len(posiciones_reales) > 0:
        for pt in posiciones_reales:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < ANCHO_REAL and 0 <= y < ALTO_REAL:
                cv2.circle(mapa_calor_acumulado, (x, y), 15, 2.0, -1)

    mapa_difuminado = cv2.GaussianBlur(mapa_calor_acumulado, (31, 31), 0)
    mapa_normalizado = cv2.normalize(mapa_difuminado, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mapa_color_termico = cv2.applyColorMap(mapa_normalizado, cv2.COLORMAP_JET)
    
    mascara_calor = mapa_normalizado > 10
    mini_campo[mascara_calor] = cv2.addWeighted(mini_campo, 0.4, mapa_color_termico, 0.6, 0)[mascara_calor]

    if len(detections) > 0 and len(posiciones_reales) > 0:
        for idx, pt in enumerate(posiciones_reales):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < ANCHO_REAL and 0 <= y < ALTO_REAL:
                class_id = detections.class_id[idx]
                name = yolo_model.names[class_id].lower()
                
                if 'ball' in name or 'pelota' in name:
                    cv2.circle(mini_campo, (x, y), 6, (0, 0, 255), -1) # Pelota: Roja
                else:
                    # Clasificar equipo por color dinámicamente para el radar táctico
                    es_aliado = es_robot_aliado(frame_original, detections.xyxy[idx])
                    color_nodo = (255, 255, 0) if es_aliado else (0, 255, 255) # Aliados Amarillos / Rivales Rojos
                    print(color_nodo)
                    cv2.circle(mini_campo, (x, y), 7, color_nodo, -1) 
                    if detections.tracker_id is not None:
                        tid = detections.tracker_id[idx]
                        cv2.putText(mini_campo, str(tid), (x - 5, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
    return mini_campo

def detectar_colisiones(posiciones_reales, detections):
    alertas = []
    if len(posiciones_reales) < 2 or detections.tracker_id is None:
        return alertas
    UMBRAL_DISTANCIA = 35.0 
    for i in range(len(posiciones_reales)):
        for j in range(i + 1, len(posiciones_reales)):
            clase_i = yolo_model.names[detections.class_id[i]].lower()
            clase_j = yolo_model.names[detections.class_id[j]].lower()
            if 'ball' in clase_i or 'ball' in clase_j or 'pelota' in clase_i or 'pelota' in clase_j:
                continue
            p1 = posiciones_reales[i]
            p2 = posiciones_reales[j]
            dist = np.linalg.norm(p1 - p2)
            if dist < UMBRAL_DISTANCIA:
                id1 = detections.tracker_id[i]
                id2 = detections.tracker_id[j]
                alertas.append(f"CHOQUE: Robot {id1} x Robot {id2}")
    return alertas

def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = yolo_model(frame, conf=0.4)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    annotated_frame = frame.copy()        
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # --- GENERAR ETIQUETAS DINÁMICAS POR COLOR ---
    labels = []
    for idx, class_id in enumerate(detections.class_id):
        name = yolo_model.names[class_id].lower()
        tid = detections.tracker_id[idx] if detections.tracker_id is not None else "X"
        
        if 'robot' in name:
            # Aquí ocurre la magia por software
            if es_robot_aliado(frame, detections.xyxy[idx]):
                labels.append(f"Aliado ID: {tid}")
            else:
                labels.append(f"Rival ID: {tid}")
        else:
            labels.append(f"{yolo_model.names[class_id]} ID: {tid}")

    if len(detections) > 0:
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)    
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    posiciones_reales = obtener_posiciones_reales(detections, H)
    mini_campo_dibujado = dibujar_campo_tactico(detections, posiciones_reales, frame)    
    colisiones = detectar_colisiones(posiciones_reales, detections)
    y_offset = 50

    for colision in colisiones:
        cv2.putText(annotated_frame, f"!! {colision} !!", (350, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30
    
    zona_y1, zona_y2 = 20, 20 + ALTO_REAL
    zona_x1, zona_x2 = 20, 20 + ANCHO_REAL
    
    h_f, w_f, _ = annotated_frame.shape
    if zona_y2 <= h_f and zona_x2 <= w_f:
        annotated_frame[zona_y1:zona_y2, zona_x1:zona_x2] = mini_campo_dibujado

    return annotated_frame

if __name__ == "__main__":        
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=OUTPUT_PATH,
        callback=process_frame
    )