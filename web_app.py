from flask import Flask, render_template, request, jsonify, Response, redirect
import cv2
import numpy as np
import threading
from nuevo import VideoProcessor
import time
import os
import gc
from pyngrok import ngrok  # Importación nueva
from dotenv import load_dotenv

load_dotenv()
# Configuración del garbage collector
gc.enable()

app = Flask(__name__)

# Configuración global
UPLOAD_FOLDER = 'temp_frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ngrok.set_auth_token(os.getenv("NEGROK_KEY"))

# Variables de control
video_processor = None
processing_active = False
stop_event = threading.Event()
processing_lock = threading.Lock()
frame_count = 0  # Contador global de frames

@app.before_request
def before_request():
    """Redirige HTTP a HTTPS para compatibilidad con cámara móvil"""
    if not request.is_secure and not request.headers.get('X-Forwarded-Proto', 'http') == 'https':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global video_processor, processing_active
    
    with processing_lock:
        if video_processor is None:
            try:
                video_processor = VideoProcessor(0)  # Usar cámara 0 por defecto
                threading.Thread(target=process_frames, daemon=True).start()
                return jsonify({'status': 'processing_started'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return jsonify({'status': 'already_running'})

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    try:
        frame_file = request.files['frame']
        frame_path = os.path.join(UPLOAD_FOLDER, 'last_frame.jpg')
        frame_file.save(frame_path)
        return jsonify({'status': 'frame_received'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_processed_frame', methods=['GET'])
def get_processed_frame():
    """Devuelve solo placas detectadas, no frames completos"""
    plate_path = os.path.join(UPLOAD_FOLDER, 'last_plate.jpg')
    
    if os.path.exists(plate_path):
        with open(plate_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    
    return Response(status=204)

VIDEO_FOLDER = 'uploaded_videos'
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo de video'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        video_path = os.path.join(VIDEO_FOLDER, video_file.filename)
        video_file.save(video_path)
        return jsonify({'status': 'video_subido', 'filename': video_file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global video_processor, processing_active
    
    with processing_lock:
        if video_processor is not None:
            try:
                video_processor.cap.release()
                if video_processor.out is not None:
                    video_processor.out.release()
                # Guardar datos finales en CSV
                video_processor.save_to_csv()
            except Exception as e:
                print(f"Error al liberar recursos: {e}")
            finally:
                video_processor = None
        
        stop_event.set()
        processing_active = False
    
    return jsonify({'status': 'processing_stopped'})

def process_frames():
    global video_processor, processing_active, frame_count
    
    processing_active = True
    stop_event.clear()
    frame_count = 0
    
    try:
        while processing_active and not stop_event.is_set():
            frame_path = os.path.join(UPLOAD_FOLDER, 'last_frame.jpg')
            
            if os.path.exists(frame_path):
                try:
                    frame = cv2.imread(frame_path)
                    if frame is not None and video_processor is not None:
                        processed_frame = video_processor.process_frame(frame)
                        
                        # Solo guardar frame procesado si contiene detecciones
                        if video_processor.object_info:
                            cv2.imwrite(
                                os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg'), 
                                processed_frame
                            )
                        
                        # Limpieza periódica de memoria
                        frame_count += 1
                        if frame_count % 10 == 0:
                            gc.collect()
                            
                except Exception as e:
                    print(f"Error procesando frame: {e}")
            
            time.sleep(0.1)
    finally:
        processing_active = False
        # Limpieza final
        if os.path.exists(os.path.join(UPLOAD_FOLDER, 'last_frame.jpg')):
            os.remove(os.path.join(UPLOAD_FOLDER, 'last_frame.jpg'))
        if os.path.exists(os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg')):
            os.remove(os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg'))

def start_ngrok():
    """Inicia un túnel ngrok para HTTPS"""
    public_url = ngrok.connect(5000, bind_tls=True)
    print(f" * Ngrok tunnel running at: {public_url}")

if __name__ == '__main__':
    # Inicia ngrok cuando se ejecute la aplicación
    start_ngrok()
    
    # Ejecuta la aplicación Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)