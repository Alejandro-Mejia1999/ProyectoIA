
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo
try:
    modelo = tf.keras.models.load_model("modelo_mobilenetv21.keras")
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Definir las clases (ajusta según tu dataset)
clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

# --- Variables Globales para el control de la transcripción y el procesamiento ---
last_transcribed_char = ""
CONFIDENCE_THRESHOLD = 0.75 # Umbral de confianza (75%).
cap = None
is_processing = False # Variable de estado general para el procesamiento

# --- Funciones Auxiliares ---

def dibujar_recuadro_deteccion(image_pil):
    """Dibuja un recuadro en la región central de la imagen PIL."""
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    box_size = int(min(width, height) * 0.7)
    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    draw.rectangle([x1, y1, x2, y2], outline="#2ECC71", width=3)
    return image_pil

def procesar_sena(imagen):
    """Procesa una imagen y devuelve la predicción con alta confianza."""
    global CONFIDENCE_THRESHOLD
    try:
        img = imagen.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediccion_prob = modelo.predict(img, verbose=0)[0]
        
        max_prob = np.max(prediccion_prob)
        predicted_index = np.argmax(prediccion_prob)
        
        if max_prob >= CONFIDENCE_THRESHOLD:
            return clases[predicted_index]
        else:
            return None
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

def actualizar_transcripcion(prediccion, force_new_char=False):
    """Añade una letra a la transcripción si es diferente a la última."""
    global last_transcribed_char
    
    if prediccion:
        if force_new_char or (prediccion != last_transcribed_char):
            transcripcion_texto.configure(state="normal")
            transcripcion_texto.insert(tk.END, prediccion + " ")
            transcripcion_texto.configure(state="disabled")
            transcripcion_texto.see(tk.END)
            
            last_transcribed_char = prediccion

def detener_proceso():
    """Detiene cualquier proceso de cámara o video."""
    global is_processing, cap, last_transcribed_char
    is_processing = False
    if cap is not None:
        cap.release()
        cap = None
    crear_placeholder()
    last_transcribed_char = ""
    resultado_label.config(text="Proceso detenido", foreground="#3498DB", font=("Segoe UI", 11, "bold"))

def crear_placeholder():
    """Crea una imagen placeholder cuando no hay imagen cargada."""
    placeholder = Image.new('RGB', (280, 280), color='#ECF0F1')
    draw = ImageDraw.Draw(placeholder)
    text = "Cargar Imagen, Video o Abrir Cámara"
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = None
    
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (placeholder.width - text_width) // 2
    y = (placeholder.height - text_height) // 2
    draw.text((x, y), text, fill="gray", font=font)

    placeholder_tk = ImageTk.PhotoImage(placeholder)
    imagen_label.configure(image=placeholder_tk)
    imagen_label.image = placeholder_tk


# --- Funciones de la Interfaz ---

def cargar_imagen():
    """Carga y procesa una imagen estática."""
    if is_processing:
        detener_proceso()

    ruta = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    if ruta:
        try:
            img = Image.open(ruta)
            img = img.resize((280, 280), Image.LANCZOS)
            img_with_box = dibujar_recuadro_deteccion(img.copy())
            img_tk = ImageTk.PhotoImage(img_with_box)
            imagen_label.configure(image=img_tk)
            imagen_label.image = img_tk
            
            resultado_label.config(text="✅ Imagen cargada con éxito", foreground="#27AE60", font=("Segoe UI", 11, "bold"))
            
            prediccion = procesar_sena(np.array(img.resize((224, 224))))
            
            global last_transcribed_char
            last_transcribed_char = ""
            
            if prediccion:
                actualizar_transcripcion(prediccion, force_new_char=True)
            else:
                resultado_label.config(text="⚠️ No se detectó una seña clara en la imagen", foreground="#F39C12", font=("Segoe UI", 11, "bold"))

        except Exception as e:
            resultado_label.config(text=f"❌ Error al cargar la imagen: {e}", foreground="#E74C3C", font=("Segoe UI", 11, "bold"))

def abrir_camara():
    """Abre el feed de la cámara en vivo."""
    global cap, is_processing
    if is_processing:
        detener_proceso()

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            resultado_label.config(text="❌ No se pudo abrir la cámara", foreground="#E74C3C", font=("Segoe UI", 11, "bold"))
            return
        
        is_processing = True
        resultado_label.config(text="📹 Cámara abierta - Reconocimiento activo", foreground="#27AE60", font=("Segoe UI", 11, "bold"))
        
        # Iniciar el bucle de la cámara
        update_feed()

    except Exception as e:
        resultado_label.config(text=f"❌ Error al abrir la cámara: {e}", foreground="#E74C3C", font=("Segoe UI", 11, "bold"))

def cargar_video():
    """Carga y procesa un archivo de video."""
    global cap, is_processing
    if is_processing:
        detener_proceso()
    
    ruta_video = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if ruta_video:
        try:
            cap = cv2.VideoCapture(ruta_video)
            if not cap.isOpened():
                resultado_label.config(text="❌ No se pudo abrir el archivo de video", foreground="#E74C3C", font=("Segoe UI", 11, "bold"))
                return
            
            is_processing = True
            resultado_label.config(text="🎥 Procesando video...", foreground="#3498DB", font=("Segoe UI", 11, "bold"))
            
            # Iniciar el bucle del video
            process_video_frames()

        except Exception as e:
            resultado_label.config(text=f"❌ Error al cargar el video: {e}", foreground="#E74C3C", font=("Segoe UI", 11, "bold"))

def update_feed():
    """Bucle para el feed de la cámara en vivo."""
    global cap, is_processing
    if not is_processing or cap is None:
        return
        
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display = cv2.resize(frame_rgb, (280, 280))
        img_pil = Image.fromarray(frame_display)
        img_with_box = dibujar_recuadro_deteccion(img_pil)
        img_tk = ImageTk.PhotoImage(img_with_box)
        imagen_label.configure(image=img_tk)
        imagen_label.image = img_tk
        
        frame_for_model = cv2.resize(frame_rgb, (224, 224))
        prediccion = procesar_sena(frame_for_model)
        
        actualizar_transcripcion(prediccion)
        
        ventana.after(100, update_feed)
    else:
        detener_proceso()

def process_video_frames():
    """Bucle para procesar un video pre-grabado."""
    global cap, is_processing
    if not is_processing or cap is None:
        return
        
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display = cv2.resize(frame_rgb, (280, 280))
        img_pil = Image.fromarray(frame_display)
        img_with_box = dibujar_recuadro_deteccion(img_pil)
        img_tk = ImageTk.PhotoImage(img_with_box)
        imagen_label.configure(image=img_tk)
        imagen_label.image = img_tk
        
        frame_for_model = cv2.resize(frame_rgb, (224, 224))
        prediccion = procesar_sena(frame_for_model)
        
        actualizar_transcripcion(prediccion)
        
        # Ajustar el delay para simular el ritmo del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        ventana.after(delay, process_video_frames)
    else:
        detener_proceso()
        resultado_label.config(text="✅ Procesamiento de video finalizado", foreground="#27AE60", font=("Segoe UI", 11, "bold"))


def limpiar_transcripcion():
    """Limpia el área de texto de la transcripción."""
    global last_transcribed_char
    transcripcion_texto.configure(state="normal")
    transcripcion_texto.delete(1.0, tk.END)
    transcripcion_texto.configure(state="disabled")
    last_transcribed_char = ""
    resultado_label.config(text="🧹 Transcripción limpiada", foreground="#3498DB", font=("Segoe UI", 11, "bold"))

def on_window_resize(event=None):
    """Maneja el redimensionamiento de la ventana."""
    if event and event.widget != ventana:
        return
    new_width = max(300, ventana.winfo_width() - 100)
    resultado_label.config(wraplength=new_width)

# Crear ventana principal con diseño responsive
ventana = tk.Tk()
ventana.title("TLS'UNAH")
ventana.geometry("600x900")
ventana.minsize(500, 700)
ventana.configure(bg="#2C3E50")
ventana.resizable(True, True)

ventana.bind('<Configure>', on_window_resize)

# Cargar iconos
def cargar_icono(ruta, tamaño=(20, 20)):
    try:
        icono_img = Image.open(ruta)
        icono_img = icono_img.resize(tamaño, Image.LANCZOS)
        return ImageTk.PhotoImage(icono_img)
    except Exception as e:
        return None

icono_subir_img = cargar_icono("./icon/subir-imagen.png")
icono_camara = cargar_icono("./icon/camara.png")
icono_tomar_foto = cargar_icono("./icon/Tomar-Foto.png")
icono_cerrar_proceso = cargar_icono("./icon/Cerrar-Camara.png")
icono_limpiar_transcripcion = cargar_icono("./icon/Limpiar-Transcripcion.png")
icono_salir = cargar_icono("./icon/Salir.png")
icono_acerca_de = cargar_icono("./icon/info.png")
icono_ayuda = cargar_icono("./icon/help.png")
icono_subir_video = cargar_icono("./icon/subir-video.png") # Nuevo icono para video

# --- Estilo Moderno y Profesional ---
style = ttk.Style()
style.theme_use("clam")

COLORS = {
    'primary': '#3498DB',
    'secondary': '#2ECC71',
    'accent': '#E74C3C',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'text': '#2C3E50',
    'success': '#27AE60',
    'warning': '#F39C12',
    'error': '#E74C3C'
}

style.configure("Menu.TButton",
                font=("Segoe UI", 10, "bold"),
                padding=(15, 8),
                background=COLORS['primary'],
                foreground="white",
                borderwidth=0,
                focuscolor="none")

style.map("Menu.TButton",
          background=[("active", "#2980B9"), ("pressed", "#21618C")])

style.configure("Title.TLabel",
                font=("Segoe UI", 28, "bold"),
                background=COLORS['dark'],
                foreground=COLORS['light'])

style.configure("Subtitle.TLabel",
                font=("Segoe UI", 14),
                background=COLORS['dark'],
                foreground="#BDC3C7")

style.configure("Section.TLabel",
                font=("Segoe UI", 14, "bold"),
                background=COLORS['light'],
                foreground=COLORS['text'])

style.configure("Main.TFrame", background=COLORS['dark'])
style.configure("Content.TFrame", background=COLORS['light'], relief="flat", borderwidth=2)
style.configure("Image.TFrame", background="white", relief="solid", borderwidth=3)

# --- Creación de la Barra de Menú Mejorada ---
menu_bar = tk.Menu(ventana, bg=COLORS['dark'], fg=COLORS['light'],
                   activebackground=COLORS['primary'], activeforeground="white",
                   font=("Segoe UI", 10))
ventana.config(menu=menu_bar)

opciones_menu = tk.Menu(menu_bar, tearoff=0, bg="white", fg=COLORS['text'],
                       activebackground=COLORS['primary'], activeforeground="white",
                       font=("Segoe UI", 10))
menu_bar.add_cascade(label="📋 Opciones", menu=opciones_menu)

opciones_menu.add_command(label="📁 Cargar Imagen", command=cargar_imagen,
                          image=icono_subir_img, compound="left")
opciones_menu.add_command(label="🎞️ Cargar Video", command=cargar_video,
                          image=icono_subir_video, compound="left")
opciones_menu.add_separator()
opciones_menu.add_command(label="📹 Abrir Cámara", command=abrir_camara,
                          image=icono_camara, compound="left")
opciones_menu.add_command(label="⏹️ Detener Proceso", command=detener_proceso,
                          image=icono_cerrar_proceso, compound="left")
opciones_menu.add_separator()
opciones_menu.add_command(label="🧹 Limpiar Transcripción", command=limpiar_transcripcion,
                          image=icono_limpiar_transcripcion, compound="left")
opciones_menu.add_separator()
opciones_menu.add_command(label="🚪 Salir", command=lambda: [detener_proceso(), ventana.quit()],
                          image=icono_salir, compound="left")

ayuda_menu = tk.Menu(menu_bar, tearoff=0, bg="white", fg=COLORS['text'],
                    activebackground=COLORS['primary'], activeforeground="white",
                    font=("Segoe UI", 10))
menu_bar.add_cascade(label="❓ Ayuda", menu=ayuda_menu)
ayuda_menu.add_command(label="ℹ️ Acerca de", command=lambda: resultado_label.config(
    text="TLS UNAH-IS",
    foreground=COLORS['primary'], font=("Segoe UI", 11, "bold")),
    image=icono_acerca_de, compound="left")
ayuda_menu.add_command(label="📖 Cómo usar", command=lambda: resultado_label.config(
    text="Instrucciones aquí...",
    foreground=COLORS['primary'], font=("Segoe UI", 11, "bold")),
    image=icono_ayuda, compound="left")

# --- CANVAS CON SCROLLBAR PARA CONTENIDO COMPLETO ---
main_canvas = tk.Canvas(ventana, bg=COLORS['dark'], highlightthickness=0)
main_canvas.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(ventana, orient="vertical", command=main_canvas.yview)
scrollbar.pack(side="right", fill="y")

main_canvas.configure(yscrollcommand=scrollbar.set)

content_container = ttk.Frame(main_canvas, style="Main.TFrame")
canvas_frame = main_canvas.create_window((0, 0), window=content_container, anchor="nw")

def configure_scroll_region(event=None):
    """Actualiza la región de scroll cuando cambia el contenido"""
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    canvas_width = main_canvas.winfo_width()
    main_canvas.itemconfig(canvas_frame, width=canvas_width)

def on_canvas_configure(event):
    """Maneja el redimensionamiento del canvas"""
    configure_scroll_region()

def on_mousewheel(event):
    """Permite scroll con la rueda del mouse"""
    main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

content_container.bind('<Configure>', configure_scroll_region)
main_canvas.bind('<Configure>', on_canvas_configure)
ventana.bind_all("<MouseWheel>", on_mousewheel)
ventana.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))
ventana.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))

# --- Estructura Principal Mejorada (dentro del content_container) ---
header_frame = ttk.Frame(content_container, style="Main.TFrame", padding=20)
header_frame.pack(fill="x", pady=(0, 10))
titulo = ttk.Label(header_frame, text="TLS-UNAH",style="Title.TLabel")
titulo.pack()
subtitulo = ttk.Label(header_frame, text="Sign Language Translator", style="Subtitle.TLabel")
subtitulo.pack(pady=(5, 0))

content_frame = ttk.Frame(content_container, style="Content.TFrame", padding=25)
content_frame.pack(expand=True, fill="both", padx=15, pady=(0, 15))

image_section = ttk.Frame(content_frame, style="Content.TFrame")
image_section.pack(pady=(0, 20))
image_title = ttk.Label(image_section, text="📷 Vista de Imagen", style="Section.TLabel")
image_title.pack(pady=(0, 10))
image_display_frame = ttk.Frame(image_section, style="Image.TFrame", padding=10)
image_display_frame.pack()
imagen_label = ttk.Label(image_display_frame, background="white", anchor="center")
imagen_label.pack()

crear_placeholder()

status_frame = ttk.Frame(content_frame, style="Content.TFrame", padding=10)
status_frame.pack(fill="x", pady=10)
resultado_label = ttk.Label(status_frame, text="Abrir el Menú",
                           font=("Segoe UI", 12, "bold"), foreground=COLORS['primary'],
                           background=COLORS['light'], wraplength=100, anchor="center")
resultado_label.pack()

transcription_frame = ttk.Frame(content_frame, style="Content.TFrame")
transcription_frame.pack(fill="both", expand=True, pady=(20, 0))
transcripcion_title = ttk.Label(transcription_frame, text="📝 Transcripción de Señas", style="Section.TLabel")
transcripcion_title.pack(anchor="w", pady=(0, 10))
text_container = ttk.Frame(transcription_frame, relief="solid", borderwidth=2, padding=2)
text_container.pack(fill="both", expand=True)

transcripcion_texto = scrolledtext.ScrolledText(
    text_container,
    height=12,
    width=50,
    font=("Consolas", 12),
    wrap=tk.WORD,
    state="disabled",
    bg="#FAFAFA",
    fg=COLORS['text'],
    selectbackground=COLORS['primary'],
    selectforeground="white",
    relief="flat",
    borderwidth=0,
    padx=10,
    pady=10
)
transcripcion_texto.pack(fill="both", expand=True)

footer_frame = ttk.Frame(content_container, style="Main.TFrame", padding=15)
footer_frame.pack(fill="x", side="bottom")
footer_label = ttk.Label(footer_frame,
                        font=("Segoe UI", 9),
                        background=COLORS['dark'],
                        foreground="#95A5A6",
                        justify="center")
footer_label.pack()

def on_closing():
    detener_proceso()
    ventana.destroy()

ventana.protocol("WM_DELETE_WINDOW", on_closing)

ventana.after(100, configure_scroll_region)

ventana.mainloop()
