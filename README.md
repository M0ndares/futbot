## **COPA FUTBOT 2026: CENTRO X META**
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue)
![Python](https://img.shields.io/badge/Language-Python-green)
![Supervision](https://img.shields.io/badge/Library-Supervision-orange)

Object detection, heat maps and real-time monitoring for robotic soccer.

![Representación futbot](futbotReadme.gif)
---
### Solution Architecture
The system is capable of processing videos from the FutBotMX Cup to track objects on the field, and translate that visual information into a 2D plane, calculating heatmaps and notifying collisions in real time.

### Pipeline
The solution was structured modularly in five critical stages, executed sequentially frame by frame:

1. **Calibration:** At the beginning, the system captures the first frame and allows the user to mark the corners of the field to calculate the homography projection matrix ($H$).
2. **Identification:** A YOLOv8n model, trained on **2,840 images** across three classes (**'ball'**, **'goal'**, and **'robot'**), detects the *bounding boxes* of the elements.
3. **Segmentation:** The **Segment Anything (SAM3)** model takes the bounding boxes from YOLO to segment the exact polygons, extracting the outline of the objects.
4. **Consistency:** A tracking algorithm assigns and maintains unique, fixed IDs for each entity.
5. **Homographic Projection:**
    * **Homography ($H$):** A projective mathematical transformation is applied to the bottom point of each object to calculate its real position on the field ($2D$) in centimeters.
    * **HSV Analysis:** The segmented areas of each robot are analyzed to classify them into teams according to their predominant color.
    * **Heatmaps:** Positions are accumulated on a canvas to generate a heatmap (`cv2.COLORMAP_JET`).
    * **Collision Detection:** Euclidean distances are measured to alert about crashes ($distance < 35 \text{ cm}$).

---

### Software Requirements

This project was developed in Python 3.11.15. The main libraries can be installed via `pip`.

* `opencv-python` 
* `ultralytics` 
* `supervision` 
* `numpy`
* `pathlib`
* `ultralytics`

*(See `requirements.txt` file for exact versions).*

### Hardware Requirements

Since the pipeline uses two heavy Deep Learning models (YOLOv8 + SAM) running frame by frame, the use of a GPU is strongly recommended for smooth playback.

* **Processor (CPU):** Intel Core i7 / AMD Ryzen 7 or higher.
* **GPU (Recommended):** NVIDIA GeForce RTX 3060 / RTX 4070 or higher, with at least 8GB of VRAM.
* **RAM:** 16GB or more.

---

### Installation and Reproduction

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/M0ndares/futbot.git](https://github.com/M0ndares/futbot.git) 
    cd futbot
    ```

2. **Create and activate the environment:**
    ```bash
    conda create -n futbot python=3.11.15
    conda activate futbot
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download model weights:**
    Make sure to place the `.pt` files in the correct folders according to your script (`/runs/segment/train/weights/best.pt` and `../notebooks/sam3.pt`).

5. **Run the pipeline:**
    If it is the first time you are running it with a new video, delete the `matriz_homografia.npy` file to force calibration.
    ```bash
    python src/app.py
    ```

---

### Results Obtained
The system perfectly integrates all 5 phases, displaying the tactical mini-field in real time, classifying teams by color, and illuminating the cumulative heatmap while visually alerting about robot collisions.

### Videos
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://www.instagram.com/reel/DZvCUDOAp7F/?igsh=ZmI2cm01c2phMnN2)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://youtube.com/shorts/xa6VbuJZ9_Y)

---

### Project License and Credits
This project is licensed under the **MIT License**. You can view the full text in the attached `LICENSE` file in this repository.

## Third-Party Code Usage
This project uses the following open-source libraries and tools as key dependencies:
* **Ultralytics**: For execution and detection using the YOLOv8 model and the SAM segmentation backend.
* **Roboflow Supervision**: Used for the `ByteTrack` tracking algorithm, as well as for visualization logic using advanced annotators (`MaskAnnotator`, `TraceAnnotator`, `LabelAnnotator`).
* **OpenCV**: Used for matrix image manipulation, color space conversion (HSV), homography calculation, and real-time map drawing.