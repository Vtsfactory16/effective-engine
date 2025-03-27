import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import os

class DetectorExpresiones:
    def __init__(self, ruta_modelo=None):
        # Inicializar detector de rostros de MediaPipe
        self.mp_rostro = mp.solutions.face_detection
        self.mp_dibujo = mp.solutions.drawing_utils
        self.detector_rostro = self.mp_rostro.FaceDetection(min_detection_confidence=0.5)
        
        # Cargar modelo de expresiones si existe
        self.modelo = None
        if ruta_modelo and os.path.exists(ruta_modelo):
            try:
                self.modelo = keras.models.load_model(ruta_modelo)
                print(f"Modelo cargado desde: {ruta_modelo}")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
        
        # Mapeo de expresiones a personajes (personalizable)
        self.expresiones = ['Enojo', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']
        self.personajes = {
            'Enojo': 'Hulk',
            'Disgusto': 'Grinch',
            'Miedo': 'Scooby Doo',
            'Felicidad': 'Mickey Mouse',
            'Tristeza': 'Eeyore',
            'Sorpresa': 'Pikachu',
            'Neutral': 'Data (Star Trek)'
        }
        
        # Cargar imágenes de personajes si están disponibles
        self.imagenes_personajes = {}
        for expresion, personaje in self.personajes.items():
            ruta_imagen = f"personajes/{personaje}.png"
            if os.path.exists(ruta_imagen):
                self.imagenes_personajes[expresion] = cv2.imread(ruta_imagen)
        
    def preprocesar_imagen(self, imagen, caja):
        """Preprocesa una imagen facial para la detección de expresiones"""
        # Extraer coordenadas de la caja del rostro
        xmin = int(caja[0])
        ymin = int(caja[1])
        ancho = int(caja[2])
        alto = int(caja[3])
        
        # Extraer rostro
        rostro = imagen[ymin:ymin+alto, xmin:xmin+ancho]
        
        # Redimensionar para el modelo
        try:
            rostro = cv2.resize(rostro, (48, 48))
            rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
            rostro = rostro / 255.0  # Normalizar
            return rostro.reshape(1, 48, 48, 1)
        except Exception as e:
            print(f"Error al preprocesar: {e}")
            return None

    def detectar_expresion(self, imagen):
        """Detecta expresiones faciales en una imagen"""
        if self.modelo is None:
            return "Modelo no cargado", 0.0
        
        # Ejecutar predicción
        prediccion = self.modelo.predict(imagen, verbose=0)
        indice_expresion = np.argmax(prediccion[0])
        confianza = prediccion[0][indice_expresion]
        
        if indice_expresion < len(self.expresiones):
            return self.expresiones[indice_expresion], confianza
        else:
            return "Desconocido", confianza

    def iniciar_camara(self):
        """Inicia la cámara y el proceso de detección"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        print("Cámara iniciada. Presiona 'q' para salir.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma.")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = self.detector_rostro.process(rgb_frame)
            
            if resultados.detections:
                for deteccion in resultados.detections:
                    # Dibujar cuadro de detección
                    caja_rel = deteccion.location_data.relative_bounding_box
                    alto, ancho, _ = frame.shape
                    
                    # Convertir a coordenadas absolutas
                    xmin = int(caja_rel.xmin * ancho)
                    ymin = int(caja_rel.ymin * alto)
                    ancho_caja = int(caja_rel.width * ancho)
                    alto_caja = int(caja_rel.height * alto)
                    
                    # Dibujar rectángulo
                    cv2.rectangle(frame, (xmin, ymin), (xmin + ancho_caja, ymin + alto_caja), (0, 255, 0), 2)
                    
                    # Preprocesar para el modelo
                    if self.modelo:
                        rostro_procesado = self.preprocesar_imagen(
                            rgb_frame, 
                            [xmin, ymin, ancho_caja, alto_caja]
                        )
                        
                        if rostro_procesado is not None:
                            # Detectar expresión
                            expresion, confianza = self.detectar_expresion(rostro_procesado)
                            personaje = self.personajes.get(expresion, "Desconocido")
                            
                            # Mostrar información
                            texto = f"{expresion}: {personaje} ({confianza:.2f})"
                            cv2.putText(frame, texto, (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Mostrar imagen del personaje si está disponible
                            if expresion in self.imagenes_personajes:
                                img_personaje = self.imagenes_personajes[expresion]
                                img_redim = cv2.resize(img_personaje, (200, 200))
                                
                                # Ubicar en la esquina superior derecha
                                h, w = img_redim.shape[:2]
                                frame[10:10+h, ancho-w-10:ancho-10] = img_redim
            
            # Mostrar el fotograma resultante
            cv2.imshow('Detector de Expresiones', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def crear_carpeta_personajes():
    """Crea la carpeta de personajes si no existe"""
    if not os.path.exists("personajes"):
        os.makedirs("personajes")
        print("Se ha creado la carpeta 'personajes'. Por favor, añade imágenes de personajes para cada expresión.")

# Ejecutar el programa
if __name__ == "__main__":
    crear_carpeta_personajes()
    
    # La ruta al modelo entrenado
    ruta_modelo = "modelo_expresiones.h5"
    
    detector = DetectorExpresiones(ruta_modelo)
    detector.iniciar_camara() 