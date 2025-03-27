import requests
import os
import zipfile
import tensorflow as tf
from tensorflow import keras
import shutil
import sys
import gdown

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
    """Descarga un modelo pre-entrenado FER2013 desde Google Drive"""
    print(f"Descargando modelo {modelo}...")
    
    # Ruta del modelo descargado
    ruta_modelo = 'modelo_expresiones.h5'
    
    # Comprobar si ya existe un modelo
    if os.path.exists(ruta_modelo):
        print("Se encontró un modelo existente 'modelo_expresiones.h5'")
        respuesta = input("¿Deseas sobrescribirlo? (s/n): ")
        if respuesta.lower() != 's':
            print("Operación cancelada.")
            return False
    
    try:
        # URL de Google Drive para un modelo FER2013 entrenado
        # Este es un ID ficticio, se debe reemplazar por un ID real
        url = "https://drive.google.com/uc?id=1-L3LnxVXv4Ud73UuQXDSQaFfKxd4gnnf"
        
        print("Descargando modelo pre-entrenado...")
        gdown.download(url, ruta_modelo, quiet=False)
        
        if os.path.exists(ruta_modelo):
            print(f"\nModelo descargado correctamente como '{ruta_modelo}'")
            return True
        else:
            print("\nError: No se pudo descargar el modelo.")
            return False
            
    except Exception as e:
        print(f"\nError durante la descarga: {e}")
        
        print("\nCreando un modelo simple de demostración en su lugar...")
        crear_modelo_demo()
        print("Modelo de demostración creado. Ten en cuenta que este es un modelo simple para pruebas.")
        
        return True

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

def descargar_personajes_demo():
    """Descarga imágenes de personajes de demostración"""
    personajes = {
        'Enojo': 'Hulk',
        'Disgusto': 'Grinch',
        'Miedo': 'Scooby Doo',
        'Felicidad': 'Mickey Mouse',
        'Tristeza': 'Eeyore',
        'Sorpresa': 'Pikachu',
        'Neutral': 'Data (Star Trek)'
    }
    
    # Asegurarse de que la carpeta existe
    if not os.path.exists("personajes"):
        os.makedirs("personajes")
    
    print("Descargando imágenes de personajes de demostración...")
    
    # Enlaces de ejemplo para cada personaje (reemplazar con enlaces válidos)
    enlaces = {
        'Hulk': "https://i.imgur.com/bPSECZf.png",
        'Grinch': "https://i.imgur.com/3jdEO7o.png",
        'Scooby Doo': "https://i.imgur.com/V8wuPJV.png",
        'Mickey Mouse': "https://i.imgur.com/E0wWKxP.png",
        'Eeyore': "https://i.imgur.com/F2jdHbS.png",
        'Pikachu': "https://i.imgur.com/sXBxBbg.png",
        'Data (Star Trek)': "https://i.imgur.com/JAdamLW.png"
    }
    
    exito = True
    for personaje, url in enlaces.items():
        ruta_destino = os.path.join("personajes", f"{personaje}.png")
        try:
            print(f"Descargando imagen de {personaje}...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(ruta_destino, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"  ✓ {personaje} descargado correctamente")
            else:
                print(f"  ✗ Error al descargar {personaje}: {response.status_code}")
                exito = False
        except Exception as e:
            print(f"  ✗ Error al descargar {personaje}: {e}")
            exito = False
    
    return exito

if __name__ == "__main__":
    print("=== Preparación del Sistema de Reconocimiento de Expresiones ===")
    
    # Verificar si gdown está instalado, si no, instalarlo
    try:
        import gdown
    except ImportError:
        print("Instalando la biblioteca gdown para la descarga...")
        os.system("pip install gdown")
        import gdown
    
    # Descargar modelo
    if descargar_modelo():
        print("\nModelo listo para usar.")
        
        # Crear carpeta para personajes y descargar ejemplos
        print("\nPreparando imágenes de personajes...")
        if descargar_personajes_demo():
            print("Imágenes de personajes descargadas correctamente.")
        else:
            print("Algunas imágenes no pudieron descargarse. Puedes añadir tus propias imágenes en la carpeta 'personajes'.")
        
        print("\n¡Todo listo! Para iniciar el detector, ejecuta:")
        print("  python detector_expresiones.py")
    else:
        print("\nNo se pudo completar la preparación. Verifica tu conexión a internet e inténtalo nuevamente.") 