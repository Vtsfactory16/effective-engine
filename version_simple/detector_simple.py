import cv2
import os
import random
import time
from datetime import datetime

class DetectorPersonajes:
    def __init__(self):
        # Cargar detector de rostros de OpenCV
        self.detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Definir personajes para cada tipo
        self.personajes = {
            'Feliz': ['Mickey Mouse', 'Pikachu', 'Bob Esponja'],
            'Triste': ['Eeyore', 'Sadness (Inside Out)', 'Dumbo'],
            'Sorprendido': ['Scooby Doo', 'Buzz Lightyear', 'Pikachu'],
            'Enojado': ['Hulk', 'Maléfica', 'Donald Duck'],
            'Normal': ['Mario Bros', 'Superman', 'Elsa']
        }
        
        # Personaje actual y temporizador
        self.personaje_actual = None
        self.expresion_actual = None
        self.tiempo_cambio = 0
        self.intervalo_cambio = 3  # segundos para cambiar de personaje
        
        # Cargar imágenes de personajes si están disponibles
        self.imagenes_personajes = {}
        self.cargar_imagenes()
    
    def cargar_imagenes(self):
        """Carga las imágenes de los personajes desde la carpeta 'personajes'"""
        if not os.path.exists("personajes"):
            os.makedirs("personajes")
            print("Se ha creado la carpeta 'personajes'. Añade imágenes de personajes ahí.")
        
        # Intentar cargar todas las imágenes posibles
        archivos = os.listdir("personajes") if os.path.exists("personajes") else []
        print(f"Archivos en carpeta 'personajes': {len(archivos)}")
        
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                nombre = os.path.splitext(archivo)[0]
                ruta_completa = os.path.join("personajes", archivo)
                try:
                    img = cv2.imread(ruta_completa, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.imagenes_personajes[nombre] = img
                        print(f"Cargada imagen de: {nombre}")
                except Exception as e:
                    print(f"Error al cargar {nombre}: {e}")
    
    def elegir_personaje_aleatorio(self):
        """Elige un personaje aleatorio basado en una expresión aleatoria"""
        expresiones = list(self.personajes.keys())
        expresion = random.choice(expresiones)
        personaje = random.choice(self.personajes[expresion])
        return expresion, personaje
    
    def mostrar_personaje(self, frame, personaje):
        """Muestra el personaje en el frame"""
        alto, ancho = frame.shape[:2]
        
        # Intentar mostrar la imagen del personaje si está disponible
        if personaje in self.imagenes_personajes:
            img_personaje = self.imagenes_personajes[personaje]
            
            # Redimensionar manteniendo relación de aspecto
            alto_img, ancho_img = img_personaje.shape[:2]
            factor = min(ancho/3 / ancho_img, alto/3 / alto_img)
            nuevo_ancho = int(ancho_img * factor)
            nuevo_alto = int(alto_img * factor)
            
            img_redim = cv2.resize(img_personaje, (nuevo_ancho, nuevo_alto))
            
            # Si la imagen tiene canal alfa (transparencia)
            if len(img_redim.shape) > 2 and img_redim.shape[2] == 4:
                # Separar la imagen y su máscara alfa
                bgr = img_redim[:, :, :3]
                alfa = img_redim[:, :, 3] / 255.0
                
                # Calcular posición (esquina superior derecha)
                y_offset = 10
                x_offset = ancho - nuevo_ancho - 10
                
                # Región del frame donde se colocará la imagen
                y1, y2 = y_offset, y_offset + nuevo_alto
                x1, x2 = x_offset, x_offset + nuevo_ancho
                
                # Asegurarse de que está dentro de los límites
                if y2 <= alto and x2 <= ancho:
                    # Región del frame
                    roi = frame[y1:y2, x1:x2]
                    
                    # Mezclar la imagen con el frame usando el canal alfa
                    for c in range(0, 3):
                        roi[:, :, c] = roi[:, :, c] * (1 - alfa) + bgr[:, :, c] * alfa
                    
                    # Colocar la región mezclada de vuelta en el frame
                    frame[y1:y2, x1:x2] = roi
            else:
                # Para imágenes sin transparencia, simplemente colocarlas
                y_offset = 10
                x_offset = ancho - nuevo_ancho - 10
                
                # Región del frame donde se colocará la imagen
                y1, y2 = y_offset, y_offset + nuevo_alto
                x1, x2 = x_offset, x_offset + nuevo_ancho
                
                # Asegurarse de que está dentro de los límites
                if y2 <= alto and x2 <= ancho:
                    frame[y1:y2, x1:x2] = img_redim
        else:
            # Si no hay imagen disponible, mostrar mensaje
            cv2.putText(frame, f"Imagen no disponible: {personaje}", 
                       (ancho - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar nombre del personaje y expresión
        texto_personaje = f"Personaje: {personaje}"
        texto_expresion = f"Expresión: {self.expresion_actual}" if self.expresion_actual else ""
        
        cv2.putText(frame, texto_personaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, texto_expresion, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def iniciar_camara(self):
        """Inicia la cámara y el proceso de detección"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("Cámara iniciada. Presiona 'q' para salir, 'c' para cambiar de personaje manualmente.")
        
        # Inicializar personaje aleatorio
        self.expresion_actual, self.personaje_actual = self.elegir_personaje_aleatorio()
        self.tiempo_cambio = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma.")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Convertir a escala de grises para detección
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            rostros = self.detector_rostro.detectMultiScale(gris, 1.3, 5)
            
            # Dibujar rectángulos alrededor de los rostros
            for (x, y, w, h) in rostros:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Verificar si hay que cambiar de personaje (si se detectan rostros)
            tiempo_actual = time.time()
            if len(rostros) > 0 and (tiempo_actual - self.tiempo_cambio) > self.intervalo_cambio:
                self.expresion_actual, self.personaje_actual = self.elegir_personaje_aleatorio()
                self.tiempo_cambio = tiempo_actual
                print(f"¡Nuevo personaje! {self.personaje_actual} ({self.expresion_actual})")
            
            # Mostrar el personaje actual
            if self.personaje_actual:
                self.mostrar_personaje(frame, self.personaje_actual)
            
            # Mostrar información de uso
            cv2.putText(frame, "Presiona 'q' para salir, 'c' para cambiar personaje", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mostrar el fotograma
            cv2.imshow('Detector de Personajes', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Cambiar personaje manualmente
                self.expresion_actual, self.personaje_actual = self.elegir_personaje_aleatorio()
                print(f"¡Nuevo personaje manual! {self.personaje_actual} ({self.expresion_actual})")
        
        cap.release()
        cv2.destroyAllWindows()

def descargar_personajes_ejemplo():
    """Descarga algunos personajes de ejemplo si no existen"""
    if not os.path.exists("personajes"):
        os.makedirs("personajes")
    
    # Lista de personajes para descargar (imágenes de ejemplo)
    personajes = {
        'Mickey Mouse': 'https://i.imgur.com/E0wWKxP.png',
        'Hulk': 'https://i.imgur.com/bPSECZf.png',
        'Pikachu': 'https://i.imgur.com/sXBxBbg.png',
        'Eeyore': 'https://i.imgur.com/F2jdHbS.png',
        'Mario Bros': 'https://i.imgur.com/JNgAGRt.png'
    }
    
    try:
        import requests
        for nombre, url in personajes.items():
            ruta_destino = os.path.join("personajes", f"{nombre}.png")
            
            # Verificar si ya existe
            if not os.path.exists(ruta_destino):
                print(f"Descargando imagen de {nombre}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(ruta_destino, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"  ✓ {nombre} descargado correctamente")
                else:
                    print(f"  ✗ Error al descargar {nombre}: {response.status_code}")
                    
        # Verificar las imágenes descargadas
        archivos = os.listdir("personajes")
        print(f"Imágenes disponibles después de la descarga: {len(archivos)}")
        for archivo in archivos:
            print(f"  - {archivo}")
            
    except Exception as e:
        print(f"Error al descargar personajes: {e}")
        print("Puedes añadir manualmente imágenes en la carpeta 'personajes'")

# Ejecutar el programa
if __name__ == "__main__":
    print("=== Detector de Personajes ===")
    print("Este programa detecta rostros y muestra personajes aleatorios.")
    
    # Intentar descargar personajes de ejemplo
    descargar_personajes_ejemplo()
    
    # Iniciar el detector
    detector = DetectorPersonajes()
    detector.iniciar_camara() 