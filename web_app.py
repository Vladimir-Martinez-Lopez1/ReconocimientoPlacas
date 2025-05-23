from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import threading
from nuevo import VideoProcessor
import time
import os
import gc
from dotenv import load_dotenv
from coni import VideoProcessor 

load_dotenv()
gc.enable()

app = Flask(__name__)

UPLOAD_FOLDER = 'temp_frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables de control
video_processor = None
processing_active = False
stop_event = threading.Event()
processing_lock = threading.Lock()
frame_count = 0

# --- ELIMINAMOS EL before_request PARA REDIRECCION HTTPS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global video_processor, processing_active
    
    with processing_lock:
        if video_processor is None:
            try:
                video_processor = VideoProcessor(0)
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
                        
                        if video_processor.object_info:
                            cv2.imwrite(
                                os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg'), 
                                processed_frame
                            )
                        
                        frame_count += 1
                        if frame_count % 10 == 0:
                            gc.collect()
                            
                except Exception as e:
                    print(f"Error procesando frame: {e}")
            
            time.sleep(0.1)
    finally:
        processing_active = False
        if os.path.exists(os.path.join(UPLOAD_FOLDER, 'last_frame.jpg')):
            os.remove(os.path.join(UPLOAD_FOLDER, 'last_frame.jpg'))
        if os.path.exists(os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg')):
            os.remove(os.path.join(UPLOAD_FOLDER, 'processed_frame.jpg'))


if __name__ == '__main__':
    # Ejecuta solo en localhost sin ngrok
    app.run(host='0.0.0.0', port=5500, debug=True)
