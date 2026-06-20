## **COPA FUTBOT 2026: CENTRO X META**
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue)
![Python](https://img.shields.io/badge/Language-Python-green)
![Supervision](https://img.shields.io/badge/Library-Supervision-orange)

Detección de objetos, mapas de calor y seguimiento en tiempo real para fútbol robótico.

---

### Arquitectura de la Solución
El sistema es capaz de procesar videos de la Copa FutBotMX para rastrear los objetos en campo, y traducir esa información visual a un plano 2D, calculando mapas de calor y notificando colisiones en tiempo real.

### Pipeline
La solución se estructuró de forma modular en cinco etapas críticas, ejecutadas secuencialmente cuadro por cuadro:

1.  **Calibración:** Al inicio, el sistema captura el primer frame y permite al usuario marcar las esquinas de la cancha para calcular la matriz de proyección homográfica ($H$)..
2.  **Identificación:** Un modelo YOLOv8n, entrenado con **2,840 imágenes** sobre tres clases (**'ball'**, **'goal'** y **'robot'**), detecta las *bounding boxes* de los elementos.
3.  **Segmentación:** El modelo **Segment Anything (SAM3)** toma las cajas de YOLO para segmentar los polígonos exactos, extrayendo el contorno de los objetos.
4.  **Consistencia:** Un algoritmo de rastreo asigna y mantiene IDs únicos y fijos para cada entidad.
5.  **Proyección homográfica:**
    * **Homografía ($H$):** Se aplica una transformación matemática proyectiva sobre el punto inferior de cada objeto para calcular su posición real en la cancha ($2D$) en centímetros.
    * **Análisis HSV:** Se analizan las áreas segmentadas de cada robot para clasificarlos por equipos de acuerdo a su color predominante.
    * **Mapas de Calor:** Se acumulan las posiciones en un lienzo para generar un mapa de calor  (`cv2.COLORMAP_JET`).
    * **Detección de Colisiones:** Se miden distancias euclidianas para alertar choques ($distancia < 35 \text{ cm}$).

---

### Requisitos de Software 

Este proyecto fue desarrollado en Python 3.11.15. Las librerías principales se pueden instalar vía `pip`.

* `opencv-python` 
* `ultralytics` 
* `supervision` 
* `numpy`
* `pathlib`
* `ultralytics`

*(Ver archivo `requirements.txt` para versiones exactas).*

### Requisitos de Hardware 

Debido a que el pipeline utiliza dos modelos de Deep Learning pesados (YOLOv8 + SAM) corriendo cuadro por cuadro, se recomienda encarecidamente el uso de GPU para una reproducción fluida.

* **Procesador (CPU):** Intel Core i7 / AMD Ryzen 7 o superior.
* **GPU (Recomendado):** NVIDIA GeForce RTX 3060 / RTX 4070 o superior, con al menos 8GB de VRAM.
* **RAM:** 16GB o más.

---

### Instalación y Reproducción 

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/M0ndares/futbot.git 
    cd futbot
    ```

2.  **Crear y activar el entorno:**
    ```bash
    conda create -n futbot python=3.11.15
    conda activate futbot
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar los pesos de los modelos:**
    Asegúrate de colocar los archivos `.pt` en las carpetas correctas según tu script (`/runs/segment/train/weights/best.pt` y `../notebooks/sam3.pt`).

5.  **Ejecutar el pipeline:**
    Si es la primera vez que lo corres con un video nuevo, borra el archivo `matriz_homografia.npy` para forzar la calibración.
    ```bash
    python src/app.py
    ```

---

### Resultados Obtenidos
El sistema logra integrar perfectamente las 5 fases, desplegando en tiempo real el mini-campo táctico, clasificando equipos por color e iluminando el mapa de calor acumulativo mientras alerta visualmente sobre los choques de los robots.

### Videos
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://www.instagram.com/reel/DZvCUDOAp7F/?igsh=ZmI2cm01c2phMnN2)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://youtube.com/shorts/xa6VbuJZ9_Y)

---

###  Licencia del Proyecto y Créditos
Este proyecto está licenciado bajo la **Licencia MIT**. Puedes consultar el texto completo en el archivo `LICENSE` adjunto en este repositorio.

## Uso de Código de Terceros 
Este proyecto utiliza las siguientes bibliotecas y herramientas de código abierto como dependencias clave:
* **Ultralytics**: Para la ejecución y detección mediante el modelo YOLOv8 y el backend de segmentación de SAM.
* **Roboflow Supervision**: Empleada para el algoritmo de rastreo `ByteTrack`, así como para la lógica de visualización mediante los anotadores avanzados (`MaskAnnotator`, `TraceAnnotator`, `LabelAnnotator`).
* **OpenCV**: Utilizada para la manipulación matricial de imágenes, conversión de espacios de color (HSV), cálculo de homografía y dibujo de mapas en tiempo real.