import requests
import os
import zipfile
import tensorflow as tf
from tensorflow import keras
import shutil
import sys

# URLs de modelos pre-entrenados (estas son ficticias, deberás sustituirlas por URLs reales)
MODELOS = {
    "fer2013": "https://example.com/models/fer2013_model.zip",
    "affectnet": "https://example.com/models/affectnet_model.zip"
}

def mostrar_progreso(bloque_num, tam_bloque, tam_total):
    """Muestra una barra de progreso de descarga"""
    porcentaje = min(100, int(100 * bloque_num * tam_bloque / tam_total))
    barra = '█' * int(porcentaje/5) + '░' * (20 - int(porcentaje/5))
    print(f"\r|{barra}| {porcentaje:.1f}%", end='')

def descargar_modelo(modelo="fer2013"):
    """Descarga un modelo pre-entrenado desde el repositorio"""
    if modelo not in MODELOS:
        print(f"Error: Modelo '{modelo}' no disponible. Opciones: {', '.join(MODELOS.keys())}")
        return False
    
    url = MODELOS[modelo]
    print(f"Descargando modelo {modelo}...")
    
    # Nombre del archivo zip temporal
    archivo_zip = "modelo_temp.zip"
    
    try:
        # Intentar descargar
        print("Conectando al servidor...")
        # En un escenario real, utilizaríamos:
        # urllib.request.urlretrieve(url, archivo_zip, mostrar_progreso)
        # Para esta demostración, mostraremos un mensaje
        print("Nota: Este es un script de demostración. Las URLs son ficticias.")
        print("En un entorno real, este script descargaría el modelo del servidor.")
        print("Por favor, proporciona una URL válida en el código.")
        
        # Comprobar si ya existe un modelo
        if os.path.exists("modelo_expresiones.h5"):
            print("Se encontró un modelo existente 'modelo_expresiones.h5'")
            respuesta = input("¿Deseas sobrescribirlo? (s/n): ")
            if respuesta.lower() != 's':
                print("Operación cancelada.")
                return False
            
        # Crear un modelo de demostración simple si no se puede descargar
        print("Creando un modelo de demostración simple...")
        crear_modelo_demo()
        print("\nModelo de demostración creado como 'modelo_expresiones.h5'")
        print("Este es un modelo muy básico solo para demostración.")
        print("Para resultados reales, entrena tu propio modelo con 'entrenar_modelo.py'")
        
        return True
        
    except Exception as e:
        print(f"\nError durante la descarga: {e}")
        return False
    
    # En un escenario real, extraeríamos el zip:
    # try:
    #     with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
    #         zip_ref.extractall(".")
    #     os.remove(archivo_zip)
    #     print("\nModelo descargado y extraído correctamente como 'modelo_expresiones.h5'")
    #     return True
    # except Exception as e:
    #     print(f"\nError al extraer el archivo: {e}")
    #     if os.path.exists(archivo_zip):
    #         os.remove(archivo_zip)
    #     return False

def crear_modelo_demo():
    """Crea un modelo simple de demostración"""
    modelo = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(7, activation='softmax')
    ])
    
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Guardar el modelo
    modelo.save('modelo_expresiones.h5')

def crear_carpeta_personajes():
    """Crea la carpeta de personajes si no existe"""
    if not os.path.exists("personajes"):
        os.makedirs("personajes")
        print("Se ha creado la carpeta 'personajes'. Por favor, añade imágenes de personajes para cada expresión.")

if __name__ == "__main__":
    print("=== Descargador de Modelo Pre-entrenado ===")
    print("Este script descargará un modelo pre-entrenado para la detección de expresiones faciales.")
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        modelo = sys.argv[1]
    else:
        print("\nModelos disponibles:")
        for clave in MODELOS.keys():
            print(f"- {clave}")
        modelo = input("\nSelecciona un modelo (o presiona Enter para usar fer2013): ") or "fer2013"
    
    # Descargar modelo
    if descargar_modelo(modelo):
        print("\nProceso completado.")
        
        # Crear carpeta para personajes
        crear_carpeta_personajes()
        
        print("\nPara usar el detector, ejecuta:")
        print("  python detector_expresiones.py")
    else:
        print("\nNo se pudo completar la descarga. Intenta entrenar tu propio modelo con:")
        print("  python entrenar_modelo.py") 