import cv2
import os
import numpy as np
import time

def crear_directorios():
    """Crea la estructura de directorios para almacenar las imágenes"""
    base_dir = 'dataset'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for split in ['train', 'validation']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        for expresion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            expresion_dir = os.path.join(split_dir, expresion)
            if not os.path.exists(expresion_dir):
                os.makedirs(expresion_dir)
    
    return base_dir

def capturar_imagenes():
    """Captura imágenes desde la cámara para el conjunto de datos"""
    # Crear directorios
    base_dir = crear_directorios()
    
    # Inicializar detector de rostros
    detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    # Configuración
    expresiones = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    num_imagenes = 50  # Número de imágenes a capturar por expresión
    contador = 0
    expresion_actual = 0
    temporizador = 0
    capturas_realizadas = 0
    modo_captura = False
    
    print("Iniciando sistema de captura de imágenes...")
    print(f"Expresión actual: {expresiones[expresion_actual]}")
    print("Presiona 'c' para iniciar la captura de la expresión actual")
    print("Presiona 'n' para pasar a la siguiente expresión")
    print("Presiona 'q' para salir")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar la imagen")
            break
        
        # Voltear horizontalmente (efecto espejo)
        frame = cv2.flip(frame, 1)
        
        # Convertir a escala de grises para detección
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        rostros = detector_rostro.detectMultiScale(gris, 1.3, 5)
        
        # Variables para la interfaz
        info_texto = f"Expresión: {expresiones[expresion_actual]} | Capturas: {capturas_realizadas}/{num_imagenes}"
        
        # Dibujar rectángulos alrededor de los rostros
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Guardar si estamos en modo captura y hay un temporizador
            if modo_captura and temporizador > 0:
                # Mostrar temporizador
                cuenta_atras = f"Capturando en: {temporizador}"
                cv2.putText(frame, cuenta_atras, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Mostrar información
        cv2.putText(frame, info_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if modo_captura:
            if temporizador > 0:
                # Cuenta regresiva
                temporizador -= 1
            else:
                # Capturar imagen si hay rostros detectados
                if len(rostros) > 0:
                    for (x, y, w, h) in rostros:
                        # Extraer rostro
                        rostro = gris[y:y+h, x:x+w]
                        
                        # Redimensionar a 48x48
                        rostro_redim = cv2.resize(rostro, (48, 48))
                        
                        # Decidir si va a entrenamiento o validación (80/20)
                        if np.random.random() < 0.8:
                            conjunto = 'train'
                        else:
                            conjunto = 'validation'
                        
                        # Ruta para guardar
                        ruta_guardar = os.path.join(
                            base_dir, 
                            conjunto, 
                            expresiones[expresion_actual], 
                            f"{expresiones[expresion_actual]}_{contador}.jpg"
                        )
                        
                        # Guardar imagen
                        cv2.imwrite(ruta_guardar, rostro_redim)
                        
                        # Actualizar contador
                        contador += 1
                        capturas_realizadas += 1
                        
                        # Mostrar mensaje de captura
                        print(f"Imagen guardada: {ruta_guardar}")
                
                # Verificar si terminamos con esta expresión
                if capturas_realizadas >= num_imagenes:
                    print(f"Completadas {capturas_realizadas} capturas para {expresiones[expresion_actual]}")
                    
                    # Pasar a la siguiente expresión
                    expresion_actual = (expresion_actual + 1) % len(expresiones)
                    capturas_realizadas = 0
                    contador = 0
                    modo_captura = False
                    
                    if expresion_actual == 0:
                        print("¡Has completado todas las expresiones!")
                        print("Presiona 'q' para salir o continúa para hacer más capturas")
                    else:
                        print(f"Ahora capturando: {expresiones[expresion_actual]}")
                        print("Presiona 'c' para iniciar la captura")
                else:
                    # Continuar capturando después de una pausa
                    temporizador = 5  # 5 frames entre capturas
        
        # Mostrar la imagen
        cv2.imshow('Captura de Expresiones', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Salir
            break
        elif key == ord('c') and not modo_captura:
            # Iniciar captura
            print(f"Iniciando captura para: {expresiones[expresion_actual]}")
            modo_captura = True
            temporizador = 5  # Comenzar con 5 frames de temporizador
        elif key == ord('n') and not modo_captura:
            # Siguiente expresión
            expresion_actual = (expresion_actual + 1) % len(expresiones)
            capturas_realizadas = 0
            print(f"Cambiado a: {expresiones[expresion_actual]}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    print("Proceso de captura finalizado")
    print(f"Las imágenes se han guardado en la carpeta: {base_dir}")

if __name__ == "__main__":
    print("=== Sistema de Captura de Imágenes para Entrenamiento ===")
    print("Este programa te ayudará a capturar imágenes de tu rostro con diferentes expresiones")
    print("para entrenar el modelo de reconocimiento de expresiones faciales.")
    
    respuesta = input("¿Deseas comenzar la captura de imágenes? (s/n): ")
    
    if respuesta.lower() == 's':
        capturar_imagenes()
    else:
        print("Operación cancelada.") 