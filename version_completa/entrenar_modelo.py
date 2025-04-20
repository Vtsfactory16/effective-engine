import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os

def crear_modelo():
    """Crea un modelo CNN para reconocimiento de expresiones faciales"""
    modelo = Sequential()
    
    # Primera capa convolucional
    modelo.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(48, 48, 1)))
    modelo.add(BatchNormalization())    
    modelo.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.3))
    
    # Segunda capa convolucional
    modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.3))

    # Tercera capa convolucional
    modelo.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding="same"))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.3))
    
    # Capa flatten
    modelo.add(Flatten())
    
    # Capas densas
    modelo.add(Dense(1024, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(7, activation='softmax'))  # 7 emociones
    
    # Compilar modelo
    modelo.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return modelo

def entrenar_modelo(ruta_datos, batch_size=32, epocas=50):
    """Entrenamiento del modelo usando datos desde un directorio"""
    # Crear generadores de datos para entrenamiento y validación
    generador_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    generador_validacion = ImageDataGenerator(rescale=1./255)
    
    # Flujos de datos para entrenamiento y validación
    flujo_entrenamiento = generador_entrenamiento.flow_from_directory(
        os.path.join('version_completa', ruta_datos, 'train'),
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    flujo_validacion = generador_validacion.flow_from_directory(
        os.path.join('version_completa', ruta_datos, 'validation'),
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    # Muestro las clases y sus índices
    # class_indices = flujo_entrenamiento.class_indices
    # print("Clases: ", class_indices)
    
    # Crear modelo
    modelo = crear_modelo()
    print(modelo.summary())
    
    # Callbacks para guardar el mejor modelo
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'modelo_expresiones.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduccion_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        min_lr=0.00001
    )
    
    parada_temprana = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Entrenar modelo
    historial = modelo.fit(
        flujo_entrenamiento,
        steps_per_epoch=len(flujo_entrenamiento),
        epochs=epocas,
        validation_data=flujo_validacion,
        validation_steps=len(flujo_validacion),
        callbacks=[checkpoint, reduccion_lr, parada_temprana]
    )
    
    # Guardar modelo final
    modelo.save('modelo_expresiones_final.h5')
    
    return modelo, historial

def preparar_estructura_directorios(ruta_base='dataset'):
    """Crea la estructura de directorios para el entrenamiento"""
    expresiones = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Directorios principales
    for split in ['train', 'validation']:
        ruta_split = os.path.join(ruta_base, split)
        if not os.path.exists(ruta_split):
            os.makedirs(ruta_split)
        
        # Subdirectorios por expresión
        for expresion in expresiones:
            ruta_expresion = os.path.join(ruta_split, expresion)
            if not os.path.exists(ruta_expresion):
                os.makedirs(ruta_expresion)
    
    print(f"Estructura de directorios creada en '{ruta_base}'")
    print("Por favor, coloca las imágenes correspondientes en cada carpeta.")
    print("Estructura esperada:")
    print(f"{ruta_base}/")
    print("  ├── train/")
    print("  │   ├── angry/")
    print("  │   ├── disgust/")
    print("  │   ├── fear/")
    print("  │   ├── happy/")
    print("  │   ├── sad/")
    print("  │   ├── surprise/")
    print("  │   └── neutral/")
    print("  └── validation/")
    print("      ├── angry/")
    print("      ├── disgust/")
    print("      ├── fear/")
    print("      ├── happy/")
    print("      ├── sad/")
    print("      ├── surprise/")
    print("      └── neutral/")

if __name__ == "__main__":
    # Preparar directorios
    # preparar_estructura_directorios()
    
    # Preguntar si se desea entrenar el modelo
    respuesta = input("¿Deseas entrenar el modelo ahora? (s/n): ")
    
    if respuesta.lower() == 's':
        # Entrenar modelo
        ruta_datos = 'dataset'
        modelo, historial = entrenar_modelo(ruta_datos)
        print("Entrenamiento completado. Modelo guardado como 'modelo_expresiones.h5'")
        print("Entrenamiento completado. Modelo guardado como 'modelo_expresiones.keras'")
    else:
        print("Coloca las imágenes en las carpetas correspondientes y ejecuta este script nuevamente para entrenar el modelo.") 