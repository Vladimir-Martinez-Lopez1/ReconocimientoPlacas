from flask import Flask, Response, request, jsonify
from nuevo import VideoProcessor
from dotenv import load_dotenv
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
load_dotenv()
# Variable global para el procesador de video
processor = None
path=0
def inicializar_video(path):
    global processor
    processor = VideoProcessor(path)

# Inicialización po
@app.route('/video_feed')
def video_feed():
    if processor is None:
        return "Video no inicializado", 500
    return Response(processor.frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para cambiar el path del video
@app.route('/set_video', methods=['POST'])
def set_video():
    data = request.json
    path = data.get('path')
    video_feed()
    if not path:
        return jsonify({'error': 'Falta el parámetro path'}), 400
    try:
        inicializar_video(path)
        #video_feed()
        return jsonify({'status': 'Video inicializado con éxito', 'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Inicia ngrok cuando se ejecute la aplicación
    #start_ngrok()
    
    app.run(host='0.0.0.0', port=8080)
