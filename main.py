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

# Captura de video
cap = cv2.VideoCapture('plates.mp4')  # O usa 0 para cámara en vivo

while True:
    ret, frame = cap.read()
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

cap.release()
cv2.destroyAllWindows()