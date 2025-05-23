import cv2
import time
import csv
import re
import os
from datetime import datetime
from collections import defaultdict, Counter
from ultralytics import YOLO
import easyocr
import numpy as np
from dotenv import load_dotenv
from bd import create_database, insert_data, get_data
conn = create_database()

load_dotenv()

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
    def __init__(self, video_source=None):
        # Configuración de rutas y parámetros
        self.model_path = os.getenv('MODEL_PATH')
        self.license_plate_model_path = os.getenv('LICENCE_PLATE_MODEL_PATH')
        self.output_video_path = 'output_video.mp4'
        self.csv_file_path = 'detection_tracking_log.csv'
        self.show_video = True
        self.target_fps = 30
        self.classes_to_detect = [2, 3, 5]
        self.plate_confidence_threshold = 0.85
        self.min_plate_width = 50
        self.min_plate_height = 15
        
        # Inicialización de modelos
        self.model = YOLO(self.model_path)
        self.license_plate_detector = YOLO(self.license_plate_model_path)
        self.reader = easyocr.Reader(['es'], gpu=True)
        
        # Variables de video
        self.set_video_source(video_source if video_source is not None else 0)
        
        # Variables de seguimiento
        self.total_class_count = Counter()
        self.seen_ids = defaultdict(set)
        self.object_info = {}
        self.frame_number = 0
        self.blur_enabled = True
        
        # Información de clases
        self.class_names = {2: "car", 3: "motorbike", 5: "bus"}
        self.class_colors = {2: (0, 0, 255), 3: (255, 255, 0), 5: (0, 255, 255)}

    def set_video_source(self, source):
        """Configura la fuente de video (archivo, cámara local o IP)"""
        if source == "0":
            source = 0
            
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.camera_index = int(source)
            self.video_path = None
        elif source.startswith(('rtsp://', 'http://', 'https://')):
            self.camera_index = source
            self.video_path = None
        else:
            self.video_path = source
            self.camera_index = None
        
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(self.video_path if self.video_path else self.camera_index)
        
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir la fuente de video")
        
        # Obtener propiedades del video
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Inicializar escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.target_fps, 
                                 (self.frame_width, self.frame_height))
        
        # Calcular cada cuántos frames procesar
        self.process_every_n_frames = max(1, int(self.original_fps / self.target_fps))

    def process_frame(self, frame):
        """Procesa un frame para detectar vehículos y placas"""
        self.frame_number += 1
        
        if self.frame_number % self.process_every_n_frames != 0:
            return frame
        
        start_time = time.time()
        
        try:
            results = self.model.track(frame, persist=True, classes=self.classes_to_detect, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    self.process_detection(frame, box)
                    
        except Exception as e:
            print(f"Error en el procesamiento del frame: {e}")
        
        self.display_performance_info(frame, start_time)
        return frame

    def process_detection(self, frame, box):
        """Procesa una detección individual de vehículo"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        confidence = round(float(box.conf[0]), 2)
        
        if box.id is None:
            return
            
        track_id = int(box.id[0].tolist())
        self.update_tracking_info(cls, track_id, confidence, x1, y1, x2, y2)
        
        # Dibujar bounding box del vehículo
        color = self.class_colors.get(cls, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.put_text(frame, f"{self.class_names[cls]} {confidence}", (x1, y1 - 10))
        self.put_text(frame, f"ID: {track_id}", (x1, y2 + 15))
        
        if confidence > 0.7:
            self.process_license_plate(frame, x1, y1, x2, y2, track_id)

    def update_tracking_info(self, cls, track_id, confidence, x1, y1, x2, y2):
        """Actualiza la información de seguimiento del objeto"""
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
                'plate_bounding_box': None
            }
        else:
            info = self.object_info[track_id]
            info['last_frame'] = self.frame_number
            if confidence > info['max_confidence']:
                info['max_confidence'] = confidence
                info['bounding_box'] = (x1, y1, x2, y2)

    def process_license_plate(self, frame, x1, y1, x2, y2, track_id):
        """Procesa la placa de un vehículo detectado"""
        current_info = self.object_info[track_id]
        
        # Si ya tenemos una placa con alta confianza, no procesar
        if current_info['plate_confidence'] and current_info['plate_confidence'] >= self.plate_confidence_threshold:
            return

        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.shape[0] < 100 or vehicle_img.shape[1] < 100:
            return
        
        # Detección de placa
        plate_results = self.license_plate_detector.predict(vehicle_img, verbose=False)
        
        if not plate_results or len(plate_results[0].boxes) == 0:
            return
            
        plate_box = plate_results[0].boxes[0]
        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
        px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
        
        # Validar tamaño mínimo de placa
        if (px2 - px1) < self.min_plate_width or (py2 - py1) < self.min_plate_height:
            return
            
        license_plate_roi = frame[py1:py2, px1:px2]
        
        # Reconocimiento de texto
        plate_ocr_results = self.reader.readtext(
            license_plate_roi,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            detail=1,
            paragraph=False
        )
        
        if not plate_ocr_results:
            return
            
        best_result = max(plate_ocr_results, key=lambda x: x[2])
        license_plate_text = best_result[1]
        plate_confidence = round(best_result[2], 2)

        # Validar formato de placa (ajustar según necesidades)
       # Cambiar a una expresión regular más flexible
        if not re.match(r'^[A-Z0-9]{4,8}$', license_plate_text):
            print(f"Formato de placa no válido: {license_plate_text}")
            return
            
        # Actualizar información si es mejor que la anterior
        if (current_info['plate_confidence'] is None or 
            plate_confidence > current_info['plate_confidence']):
            
            current_info.update({
                'license_plate_text': license_plate_text,
                'plate_confidence': plate_confidence,
                'plate_bounding_box': (px1, py1, px2, py2)
            })
            
            # Guardar imagen de la placa
            cv2.imwrite('temp_frames/last_plate.jpg', license_plate_roi)
            if license_plate_text:
                insert_data(
                    conn=conn,
                    placas=license_plate_text,
                    tipo_placa=current_info['class_name'],
                    frame=str(self.frame_number)
                )
            # Guardar en CSV inmediatamente
            self.save_to_csv()

        # Visualización
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 2)
        self.put_text(frame, f"{license_plate_text} ({plate_confidence:.2f})", 
                     (px1, py2 + 20), color=(0, 0, 0), bg_color=(255, 255, 255))
        
        if self.blur_enabled:
            frame[py1:py2, px1:px2] = cv2.GaussianBlur(license_plate_roi, (31, 31), 15)

    def put_text(self, frame, text, position, color=(255, 255, 255), bg_color=(0, 0, 0), 
                 font_scale=0.5, thickness=1):
        """Dibuja texto con fondo en un frame"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x, y = position
        cv2.rectangle(frame, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), bg_color, -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def display_performance_info(self, frame, start_time):
        """Muestra información de rendimiento en el frame"""
        y_offset = 30
        for cls, count in self.total_class_count.items():
            self.put_text(frame, f"Total {cls}: {count}", (10, y_offset))
            y_offset += 20
        
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.put_text(frame, f"FPS: {min(current_fps, self.target_fps):.1f}", (10, y_offset))
        self.put_text(frame, f"Frame: {self.frame_number}", (10, y_offset + 20))

    def run(self):
        """Ejecuta el procesamiento del video principal"""
        try:
            last_frame_time = time.time()
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if self.camera_index is not None:
                        time.sleep(0.1)
                        continue
                    break
                
                # Control de FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                wait_time = max(1, int(1000 * (1/self.target_fps - elapsed)))
                
                processed_frame = self.process_frame(frame)
                self.out.write(processed_frame)
                
                if self.show_video:
                    self.handle_key_controls(processed_frame)
                
                last_frame_time = current_time
                
        finally:
            self.cleanup()

    def handle_key_controls(self, frame):
        """Maneja las teclas presionadas durante la visualización"""
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.cleanup()
            exit()
        elif key == ord(' '):  # Espacio para pausar
            cv2.waitKey(0)
        elif key == ord('b'):  # Alternar difuminado
            self.blur_enabled = not self.blur_enabled
        elif key == ord('c') and isinstance(self.camera_index, int):  # Cambiar cámara
            self.cap.release()
            self.camera_index = 1 - self.camera_index
            self.cap = cv2.VideoCapture(self.camera_index)
        
        cv2.imshow('Vehicle Tracking', frame)

    def cleanup(self):
        """Libera recursos y guarda datos"""
        self.save_to_csv()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'out'):
            self.out.release()
        cv2.destroyAllWindows()

    def save_to_csv(self):
        """Guarda solo detecciones válidas de placas en CSV"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Verificar si el archivo existe y si está vacío para escribir encabezados
            file_exists = os.path.isfile(self.csv_file_path)
            
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Escribir encabezados solo si el archivo está vacío o no existe
                if not file_exists or os.stat(self.csv_file_path).st_size == 0:
                    writer.writerow(['timestamp', 'track_id', 'vehicle_type', 
                                   'license_plate', 'confidence', 'frame_number'])
                
                # Guardar todas las placas detectadas, no solo las nuevas
                for track_id, info in self.object_info.items():
                    if info['license_plate_text'] and info['plate_confidence']:
                        writer.writerow([
                            timestamp,
                            track_id,
                            info['class_name'],
                            info['license_plate_text'],
                            info['plate_confidence'],
                            info['first_frame']
                        ])
                file.flush()  # Forzar escritura inmediata
                
            print(f"Datos guardados en {self.csv_file_path}")  # Mensaje de depuración
            
        except Exception as e:
            print(f"Error al guardar en CSV: {str(e)}")