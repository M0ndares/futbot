import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

VIDEO_PATH = "videos/prueba2.mp4"
OUTPUT_PATH = "videos/resultado_segmentado2.mp4"
MATRIZ_PATH = "matriz_homografia.npy"

PUNTOS_REALES = np.array([
    [0, 0],     
    [400, 0],   
    [400, 300], 
    [0, 300],    
], dtype=np.float32)

puntos_video = []
def clic_mouse(event, x, y, flags, param):
    global puntos_video, frame_calibracion
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_video.append([x, y])
        print(f"Punto {len(puntos_video)} registrado: [{x}, {y}]")
        cv2.circle(frame_calibracion, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame_calibracion, str(len(puntos_video)), (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Calibrador interactivo - Da Clics", frame_calibracion)

def ejecutar_calibracion():
    global frame_calibracion
    if os.path.exists(MATRIZ_PATH):
        return np.load(MATRIZ_PATH)
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        exit()
        
    frame_calibracion = frame.copy()
    cv2.namedWindow("Calibrador interactivo - Da Clics")
    cv2.setMouseCallback("Calibrador interactivo - Da Clics", clic_mouse)
    
    print(f"Por favor, da {len(PUNTOS_REALES)} clics en la imagen siguiendo el orden del plano real.")
    print("Presiona 'q' para salir cuando termines.")
    
    while True:
        cv2.imshow("Calibrador interactivo", frame_calibracion)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(puntos_video) == len(PUNTOS_REALES):
            break
            
    cv2.destroyAllWindows()
    
    if len(puntos_video) == len(PUNTOS_REALES):
        puntos_video_np = np.array(puntos_video, dtype=np.float32)
        H, _ = cv2.findHomography(puntos_video_np, PUNTOS_REALES, 0)
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
heat_map_annotator = sv.HeatMapAnnotator()         

def transformar_puntos_a_plano_real(detections, matriz_h):
    if len(detections) == 0:
        return detections

    puntos_video = detections.get_anchors(sv.Position.BOTTOM_CENTER)
    puntos_reshaped = np.array([puntos_video], dtype=np.float32)
    puntos_proyectados = cv2.perspectiveTransform(puntos_reshaped, matriz_h)[0]
    
    detections.xyxy[:, :2] = puntos_proyectados
    detections.xyxy[:, 2:] = puntos_proyectados
    
    return detections

def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = yolo_model(frame, conf=0.4)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    annotated_frame = frame.copy()        
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    detections_proyectadas = transformar_puntos_a_plano_real(detections, H)
    annotated_frame = heat_map_annotator.annotate(scene=annotated_frame, detections=detections_proyectadas)

    if len(detections_proyectadas) > 0 and detections_proyectadas.tracker_id is not None:
        labels = []
        for class_id, tracker_id in zip(detections_proyectadas.class_id, detections_proyectadas.tracker_id):
            name = yolo_model.names[class_id] 
            labels.append(f"{name} ID: {tracker_id}")
            
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections_proyectadas)    
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_proyectadas, labels=labels)
    else:
        labels = [yolo_model.names[class_id] for class_id in detections_proyectadas.class_id]
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_proyectadas, labels=labels)

    return annotated_frame

if __name__ == "__main__":
    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=OUTPUT_PATH,
        callback=process_frame
    )
