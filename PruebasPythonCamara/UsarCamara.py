import tkinter as tk
from tkinter import filedialog
import cv2

# Caso 1. Capturar video desde la cámara web
video = cv2.VideoCapture(0)

# Verificar si el video se abrió correctamente
if not video.isOpened():
    print("Error: No se pudo abrir el video. Verifica la ruta o el formato.")
    exit()


    

# Caso 2. Acceder a cualquier cámara basada en IP 
videoIP = cv2.VideoCapture("https://3.4.5.6/cam2")




# Caso 3. Acceder a camara de telefono.
#El telefono debe contar con IP Webcam:  Android o DroidCam : Android/iOS
# URL del stream de la cámara del teléfono
url = "http://192.168.1.100:8080/video"  # Reemplaza con la IP y puerto de tu teléfono
video = cv2.VideoCapture(url) # apturar el stream de video desde el teléfono




# Caso 4. Video almacenado.
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







while True:
    # Leer el cuadro actual del video
    ret, frame = video.read()
    
    # Mostrar el cuadro en una ventana
    cv2.imshow("Video", frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
