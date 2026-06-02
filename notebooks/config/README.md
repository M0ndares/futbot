# Centro X META AI: Copa FutBotMX 
> **Categoría:** Profesional  
> **Equipo:** mondares  
> **Desarrollador:** Osval Elías Montesinos Valladares

Esta carpeta contiene las 10 notebooks implementadas en el torneo **FutBotMX**. Este curso se centró en las tecnologías **YOLO** para la detección rápida y **SAM (Segment Anything Model)** para la segmentación precisa de los elementos en la cancha, unificados mediante la librería **Supervision**.

## Instalación y Replicabilidad 

Para garantizar que el proyecto se ejecute de manera idéntica al entorno de desarrollo original sin conflictos de dependencias, se fijaron las versiones críticas en el archivo requirements.txt, disponible en esta misma carpeta.

### Requisitos Previos
* Python 3.11.15
* Miniconda 3

### Pasos para Replicar
1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/M0ndares/futbot.git 
   ```

2. **Crear entorno virtual:**
   ```bash
   conda create -n supervision python=3.11.15 
   ```

3. **Activar entorno virtual:**
   ```bash
   conda activate supervision  
   ```

4. **Instalar dependencias:**
   ```bash
   pip install -r notebooks/info/requirements.txt  
   ```