import tkinter as tk
from tkinter import filedialog, messagebox
from nuevo import VideoProcessor
import subprocess
import os

def seleccionar_archivo():
    ruta = filedialog.askopenfilename(
        title="Seleccionar archivo de video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv"), ("Todos los archivos", "*.*")]
    )
    if ruta:
        ruta_seleccionada.set(ruta)
        modo_seleccionado.set("Archivo")

def usar_camara_local():
    ruta_seleccionada.set("0")  # 0 es la cámara local por defecto
    modo_seleccionado.set("Cámara Local")
    messagebox.showinfo("Configuración", "Se usará la cámara web local (índice 0)")

def usar_camara_ip():
    # Crear ventana emergente para ingresar la URL/IP de la cámara
    ip_window = tk.Toplevel(root)
    ip_window.title("Configurar Cámara IP")
    ip_window.geometry("400x150")
    ip_window.configure(bg="#f0f0f0")
    
    tk.Label(ip_window, text="Ingrese la URL o dirección IP de la cámara:", 
             font=("Helvetica", 12), bg="#f0f0f0").pack(pady=10)
    
    ip_entry = tk.Entry(ip_window, font=("Helvetica", 12), width=30)
    ip_entry.pack(pady=5)
    ip_entry.insert(0, "rtsp://usuario:contraseña@ip:puerto/ruta")
    
    def confirmar_ip():
        ip = ip_entry.get()
        if ip:
            ruta_seleccionada.set(ip)
            modo_seleccionado.set("Cámara IP")
            ip_window.destroy()
            messagebox.showinfo("Configuración", f"Cámara IP configurada: {ip}")
    
    tk.Button(ip_window, text="Aceptar", command=confirmar_ip, 
              font=("Helvetica", 12), bg="#4CAF50", fg="white").pack(pady=10)

def crearobjeto():
    if not ruta_seleccionada.get():
        messagebox.showerror("Error", "Por favor seleccione una fuente de video")
        return
    
    try:
        source = ruta_seleccionada.get()
        
        # Determinar el tipo de fuente
        if modo_seleccionado.get() == "Archivo":
            processor = VideoProcessor(source)
        elif modo_seleccionado.get() == "Cámara Local":
            # Convertir a entero para índice de cámara
            processor = VideoProcessor(int(source))
        else:  # Cámara IP
            processor = VideoProcessor(source)
            
        processor.run()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo iniciar el procesamiento:\n{str(e)}")
def abrir_csv():
    ruta_csv = r"C:\Users\Vladimir\Documents\concurso\concurso\ReconocimientoPlacas\detection_tracking_log.csv"
    if os.path.exists(ruta_csv):
        try:
            subprocess.Popen(['notepad', ruta_csv])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el archivo CSV:\n{str(e)}")
    else:
        messagebox.showwarning("Archivo no encontrado", f"No se encontró el archivo:\n{ruta_csv}")


# Crear ventana principal
root = tk.Tk()
root.title("Procesador de Video")
root.geometry("700x300")  # Aumentamos el tamaño para los nuevos botones
root.configure(bg="#f0f0f0")

# Variables de control
ruta_seleccionada = tk.StringVar()
modo_seleccionado = tk.StringVar(value="No seleccionado")

# Fuentes
fuente_grande = ("Helvetica", 14)
fuente_mediana = ("Helvetica", 12)

# Título
titulo = tk.Label(root, text="Selecciona la fuente de video", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
titulo.pack(pady=15)

# Frame para botones de selección
frame_seleccion = tk.Frame(root, bg="#f0f0f0")
frame_seleccion.pack(pady=10)

# Botones de selección de fuente
boton_archivo = tk.Button(frame_seleccion, text="📁 Archivo de Video", command=seleccionar_archivo, 
                         font=fuente_mediana, width=20, bg="#2196F3", fg="white")
boton_archivo.grid(row=0, column=0, padx=10, pady=5)

boton_camara = tk.Button(frame_seleccion, text="📷 Cámara Local", command=usar_camara_local, 
                        font=fuente_mediana, width=20, bg="#FF9800", fg="white")
boton_camara.grid(row=0, column=1, padx=10, pady=5)

boton_ip = tk.Button(frame_seleccion, text="🌐 Cámara IP", command=usar_camara_ip, 
                    font=fuente_mediana, width=20, bg="#9C27B0", fg="white")
boton_ip.grid(row=0, column=2, padx=10, pady=5)

# Botón para abrir CSV
abrir_csv_button = tk.Button(root, text="📄 Ver Registro CSV", command=abrir_csv,
                             font=fuente_mediana, width=25, bg="#607D8B", fg="white")
abrir_csv_button.pack(pady=5)


# Frame para mostrar selección actual
frame_info = tk.Frame(root, bg="#f0f0f0")
frame_info.pack(pady=10)

tk.Label(frame_info, text="Modo seleccionado:", font=fuente_mediana, bg="#f0f0f0").grid(row=0, column=0)
tk.Label(frame_info, textvariable=modo_seleccionado, font=fuente_mediana, fg="#00796B", bg="#f0f0f0").grid(row=0, column=1)

tk.Label(frame_info, text="Fuente:", font=fuente_mediana, bg="#f0f0f0").grid(row=1, column=0)
tk.Label(frame_info, textvariable=ruta_seleccionada, font=fuente_mediana, width=50, 
         anchor='w', bg="#ffffff", relief="sunken").grid(row=1, column=1)

# Botón para procesar
procesar_button = tk.Button(root, text="▶ Iniciar Procesamiento", command=crearobjeto, 
                           font=fuente_grande, width=25, bg="#4CAF50", fg="white")
procesar_button.pack(pady=20)

# Ejecutar la aplicación
root.mainloop()