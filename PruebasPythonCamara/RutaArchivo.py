import tkinter as tk
from tkinter import filedialog
import cv2

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

