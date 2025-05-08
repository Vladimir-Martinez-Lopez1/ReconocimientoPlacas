import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import numpy as np
from sort.sort import Sort 

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.2  # Confianza mínima
model.classes = [2, 3, 5, 7]  # Solo coches, motos, autobuses y camiones

# Inicializar SORT
tracker = Sort()

#captura de video por archivo o camara
#ingresar por teclado 0 para camara 1 para subir archivo 2 para 
video_source = int(input("Selecciona la fuente de video (1: subir video, 2 camara, 3: camaraIP 4: Transmitir desde Movil ): "))

if video_source == 1:

    # Configurar tkinter (ventana oculta)
    root = tk.Tk()
    root.withdraw()

    # Abrir ventana para seleccionar archivo de video
    video_path = filedialog.askopenfilename(
        title="Selecciona un archivo de video",
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
    )

    # Verificar si se seleccionó un archivo
    if not video_path:
        print("No se seleccionó ningún archivo. Saliendo...")
        exit()

    # Capturar video desde el archivo seleccionado
    video = cv2.VideoCapture(video_path)


elif video_source == 2:
    video = cv2.VideoCapture(0)  # Cámara en vivo

elif video_source == 3:
    video = cv2.VideoCapture("https://3.4.5.6/cam2")

elif video_source == 4:
    url = "http://192.168.1.100:8080/video"  # Reemplaza con la IP y puerto de tu teléfono
    video = cv2.VideoCapture(url) 


while True:
    ret, frame = video.read()
    if not ret:
        break

    # Inferencia
    results = model(frame)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        detections.append([x1, y1, x2, y2, float(conf)])

    # Convertir a np.array para SORT
    if len(detections) > 0:
        dets = np.array(detections)
    else:
        dets = np.empty((0, 5))

    tracks = tracker.update(dets)

    # Dibujar cajas e ID
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Seguimiento de Vehículos', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()