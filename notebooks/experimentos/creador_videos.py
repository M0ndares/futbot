import os
import cv2
import numpy as np
from pathlib import Path
# 1. Asegurar que exista la carpeta assets
current = Path(__file__).resolve().parent.parent
dir = current / 'assets/videos'
os.makedirs(dir, exist_ok=True)

# 2. Configuración del video ficticio (2 segundos a 30 FPS = 60 cuadros)
OUTPUT_PATH = dir / 'futbot-2s.mp4'
WIDTH, HEIGHT = 1080, 600  # Dimensiones de tu imagen original
FPS = 30
TOTAL_FRAMES = 60

# Definir el codec y el escritor de video (MP4 universal)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

# 3. Dibujar fondo estático (una simulación rápida de tu imagen de cámara)
# Un fondo gris con un trapecio verde que simula la cancha oblicua
fondo_base = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 200  # Gris claro

# Dibujamos un trapecio verde usando tus SOURCE_POINTS aproximados
# [250, 70], [335, 210], [338, 602], [255, 750] -> invertidos a formato estándar (X, Y)
# Ojo: En tu código pusiste filas/columnas. Dibujemos un trapecio verde representativo:
pts_cancha = np.array([[200, 100], [880, 100], [1000, 500], [80, 500]], np.int32)
cv2.fillPoly(fondo_base, [pts_cancha], (34, 139, 34))  # Verde pasto
cv2.polylines(fondo_base, [pts_cancha], True, (255, 255, 255), 5)  # Líneas blancas

# 4. Generar los cuadros con movimiento
for i in range(TOTAL_FRAMES):
    frame = fondo_base.copy()
    
    # Simular un balón naranja moviéndose por la cancha oblicua
    bx = int(300 + (i * 8))
    by = int(250 + (i * 2))
    cv2.circle(frame, (bx, by), 15, (0, 140, 255), -1)  # Balón naranja BGR
    cv2.putText(frame, "Ball", (bx - 15, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    
    # Simular un robot azul moviéndose
    rx = int(400 - (i * 2))
    ry = int(300 + (i * 3))
    cv2.rectangle(frame, (rx - 20, ry - 20), (rx + 20, ry + 20), (255, 0, 0), -1)  # Robot azul
    cv2.putText(frame, "Robot1", (rx - 25, ry - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Guardar el cuadro en el archivo de video
    video_writer.write(frame)

# Cerrar el archivo de video de forma segura
video_writer.release()
print(f"🎉 ¡Video sintético creado exitosamente en: {OUTPUT_PATH}!")