import cv2
import time
import csv
from collections import defaultdict, Counter
from ultralytics import YOLO
import easyocr
import numpy as np
import sqlite3
from bd import create_database, insert_data, get_data
conn = create_database()
class VideoProcessor:
    def frame_generator(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)

            # Codificar a JPEG para enviar como stream
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    def __init__(self, video_path):
        self.video_path = video_path 
        self.model_path = '/home/hugocd/Documentos/programacion/ReconocimientoPlacas/yolo11n.pt'
        self.license_plate_model_path = '/home/hugocd/Documentos/programacion/ReconocimientoPlacas/runs/detect/license_plate_detector/weights/best.pt'
        self.output_video_path = 'output_video.mp4'
        self.csv_file_path = 'detection_tracking_log.csv'
        self.show_video = True
        self.target_fps = 30  # Objetivo intermedio entre 15-30 FPS
        self.classes_to_detect = [2, 3, 5]  # Solo coches, motos y autobuses para mayor eficiencia
    

        # Inicialización de modelos
        self.model = YOLO(self.model_path)
        self.license_plate_detector = YOLO(self.license_plate_model_path)
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        # Configuración de video
        self.cap = cv2.VideoCapture(self.video_path)
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ajustar FPS si es necesario
        self.process_every_n_frames = max(1, int(self.original_fps / self.target_fps))
        
        # Configuración del escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.target_fps, 
                                  (self.frame_width, self.frame_height))
        
        # Variables de seguimiento
        self.total_class_count = Counter()
        self.seen_ids = defaultdict(set)
        self.object_info = {}
        self.frame_number = 0
        self.blur_enabled = True
        
        # Configuración de clases
        self.class_names = {2: "car", 3: "motorbike", 5: "bus"}
        self.class_colors = {2: (0, 0, 255), 3: (255, 255, 0), 5: (0, 255, 255)}

    def process_frame(self, frame):
        start_time = time.time()
        self.frame_number += 1
        
        # Saltar frames para mantener FPS objetivo
        if self.frame_number % self.process_every_n_frames != 0:
            return frame
        
        # Detección y seguimiento optimizado
        results = self.model.track(frame, persist=True, classes=self.classes_to_detect, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                confidence = round(float(box.conf[0]), 2)
                
                if box.id is not None:
                    track_id = int(box.id[0].tolist())
                    self.update_tracking_info(cls, track_id, confidence, x1, y1, x2, y2)
                    
                    # Dibujar bounding box
                    color = self.class_colors.get(cls, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    self.put_text(frame, f"{self.class_names[cls]} {confidence}", (x1, y1 - 10))
                    self.put_text(frame, f"ID: {track_id}", (x1, y2 + 15))
                    
                    # Procesar matrículas solo para confianzas altas
                    if confidence > 0.7:
                        self.process_license_plate(frame, x1, y1, x2, y2, track_id)
        
        # Mostrar información de rendimiento
        self.display_performance_info(frame, start_time)
        return frame
    
    def update_tracking_info(self, cls, track_id, confidence, x1, y1, x2, y2):
        if track_id not in self.seen_ids[cls]:
            self.seen_ids[cls].add(track_id)
            self.total_class_count[self.class_names[cls]] += 1
        
        if track_id not in self.object_info:
            self.object_info[track_id] = {
                'class_name': self.class_names[cls],
                'max_confidence': confidence,
                'first_frame': self.frame_number,
                'last_frame': self.frame_number,
                'bounding_box': (x1, y1, x2, y2),
                'license_plate_text': '',
                'plate_confidence': None,
                'plate_bounding_box': (None, None, None, None)
            }
        else:
            info = self.object_info[track_id]
            info['last_frame'] = self.frame_number
            if confidence > info['max_confidence']:
                info['max_confidence'] = confidence
                info['bounding_box'] = (x1, y1, x2, y2)

    def process_license_plate(self, frame, x1, y1, x2, y2, track_id):
        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.shape[0] < 100 or vehicle_img.shape[1] < 100:
            return
        
        # Detección de matrícula optimizada
        plate_results = self.license_plate_detector.predict(vehicle_img, verbose=False)
        
        if plate_results and len(plate_results[0].boxes) > 0:
            plate_box = plate_results[0].boxes[0]  # Solo procesar la matrícula con mayor confianza
            px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
            px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
            
            if (px2 - px1) < 50 or (py2 - py1) < 15:
                return
            
            license_plate_roi = frame[py1:py2, px1:px2]
            
            # OCR directo sin super-resolución para mayor velocidad
            plate_ocr_results = self.reader.readtext(
                license_plate_roi, 
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                detail=1,
                paragraph=False
            )
            
            if plate_ocr_results:
                best_result = max(plate_ocr_results, key=lambda x: x[2])  # Seleccionar resultado con mayor confianza
                license_plate_text = best_result[1]
                plate_confidence = round(best_result[2], 2)
                
                # Actualizar información si es mejor
                current_info = self.object_info[track_id]
                if (current_info['plate_confidence'] is None or 
                    plate_confidence > current_info['plate_confidence']):
                    current_info.update({
                        'license_plate_text': license_plate_text,
                        'plate_confidence': plate_confidence,
                        'plate_bounding_box': (px1, py1, px2, py2)
                    })
                
                # Dibujar resultados
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 2)
                self.put_text(frame, f"Plate: {license_plate_text}", (px1, py2 + 20), 
                            color=(0, 0, 0), bg_color=(255, 255, 255))
            
            # Aplicar desenfoque si está activado
            if self.blur_enabled:
                blurred_plate = cv2.GaussianBlur(license_plate_roi, (31, 31), 15)  # Kernel más pequeño para mayor velocidad
                frame[py1:py2, px1:px2] = blurred_plate

    def put_text(self, frame, text, position, color=(255, 255, 255), bg_color=(0, 0, 0), 
                font_scale=0.5, thickness=1):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x, y = position
        cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), bg_color, -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def display_performance_info(self, frame, start_time):
        y_offset = 30
        for cls, count in self.total_class_count.items():
            self.put_text(frame, f"Total {cls}: {count}", (10, y_offset))
            y_offset += 20
        
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.put_text(frame, f"FPS: {min(current_fps, self.target_fps):.1f}", (10, y_offset))
        self.put_text(frame, f"Frame: {self.frame_number}", (10, y_offset + 20))

    def run(self):
        # Preparar archivo CSV
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'object_type', 'confidence', 'tracking_id', 
                            'x1', 'y1', 'x2', 'y2', 'license_plate_confidence', 
                            'mx1', 'my1', 'mx2', 'my2', 'license_plate_text'])
        
        last_frame_time = time.time()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Control de FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            wait_time = max(1, int(1000 * (1/self.target_fps - elapsed)))
            
            # Procesar frame
            processed_frame = self.process_frame(frame)
            self.out.write(processed_frame)
            
            # Mostrar frame
            if self.show_video:
                cv2.imshow('Vehicle Tracking', processed_frame)
                key = cv2.waitKey(wait_time) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # Espacio para pausa
                    cv2.waitKey(0)
                elif key == ord('b'):  # Alternar desenfoque
                    self.blur_enabled = not self.blur_enabled
            
            last_frame_time = current_time
        
        # Guardar datos en CSV
        self.save_to_csv()
        
        # Liberar recursos
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def save_to_csv(self):
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for track_id, info in self.object_info.items():
                writer.writerow([
                    info['first_frame'],
                    info['class_name'],
                    info['max_confidence'],
                    track_id,
                    *info['bounding_box'],
                    info['plate_confidence'] if info['plate_confidence'] else '',
                    *info['plate_bounding_box'],
                    info['license_plate_text'] if info['license_plate_text'] else ''
                ])
                if info['license_plate_text']:
                    insert_data(conn, info['license_plate_text'], info['class_name'], str(info['first_frame']))



if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()