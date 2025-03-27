# Detector de Expresiones Faciales con Asociación de Personajes

Este proyecto te permite detectar expresiones faciales en tiempo real a través de la cámara web y asociar cada expresión con un personaje conocido. Por ejemplo, si sonríes, podrías ver a Mickey Mouse, o si muestras enfado, podría aparecer Hulk.

## Requisitos

Asegúrate de tener Python 3.8 o superior instalado. Luego, instala las dependencias necesarias con:

```bash
pip install -r requirements.txt
```

## Configuración

El proyecto consta de tres componentes principales:

1. **Entrenamiento del modelo**: Puedes entrenar tu propio modelo de reconocimiento de expresiones o usar uno pre-entrenado.
2. **Imágenes de personajes**: Debes proporcionar imágenes de los personajes que quieres asociar con cada expresión.
3. **Ejecución del detector**: El programa principal que activará la cámara y detectará expresiones.

### 1. Entrenamiento del Modelo

Para entrenar tu propio modelo:

1. Ejecuta el script de preparación de estructura:
   ```bash
   python entrenar_modelo.py
   ```

2. Este script creará una estructura de directorios en una carpeta llamada `dataset` con subdirectorios para cada expresión facial.

3. Coloca imágenes faciales correspondientes a cada expresión en las carpetas apropiadas:
   - `dataset/train/angry/` - Imágenes de entrenamiento de rostros enojados
   - `dataset/train/disgust/` - Imágenes de entrenamiento de rostros con disgusto
   - `dataset/train/fear/` - Imágenes de entrenamiento de rostros con miedo
   - ... y así sucesivamente

4. Ejecuta nuevamente el script y selecciona 's' cuando te pregunte si deseas entrenar el modelo:
   ```bash
   python entrenar_modelo.py
   ```

5. El entrenamiento puede llevar tiempo, dependiendo de la cantidad de imágenes y la capacidad de tu computadora. El modelo entrenado se guardará como `modelo_expresiones.h5`.

### 2. Imágenes de Personajes

1. Crea una carpeta llamada `personajes` (el programa la creará automáticamente la primera vez que lo ejecutes).
2. Añade imágenes PNG de los personajes que deseas asociar con cada expresión, con los siguientes nombres:
   - `Hulk.png` - Para la expresión de enojo
   - `Grinch.png` - Para la expresión de disgusto
   - `Scooby Doo.png` - Para la expresión de miedo
   - `Mickey Mouse.png` - Para la expresión de felicidad
   - `Eeyore.png` - Para la expresión de tristeza
   - `Pikachu.png` - Para la expresión de sorpresa
   - `Data (Star Trek).png` - Para la expresión neutral

3. Puedes personalizar estas asociaciones editando el diccionario `self.personajes` en el archivo `detector_expresiones.py`.

### 3. Ejecución del Detector

Una vez que tengas el modelo entrenado y las imágenes de los personajes, puedes ejecutar el detector:

```bash
python detector_expresiones.py
```

Este comando activará tu cámara web, detectará tu rostro y mostrará el personaje asociado a la expresión que estés haciendo.

## Personalización

Puedes personalizar varios aspectos del programa:

- **Personajes**: Edita el diccionario `self.personajes` en `detector_expresiones.py` para cambiar los personajes asociados a cada expresión.
- **Modelo**: Si tienes un modelo pre-entrenado, puedes colocarlo en la carpeta raíz con el nombre `modelo_expresiones.h5`.
- **Umbrales de detección**: Puedes ajustar los parámetros de detección en `detector_expresiones.py` para mejorar la precisión en tu entorno.

## Notas

- Presiona 'q' para salir del programa cuando esté en ejecución.
- Para mejores resultados, asegúrate de tener buena iluminación al usar el detector.
- Si estás entrenando tu propio modelo, trata de tener al menos 100 imágenes de cada expresión para obtener buenos resultados.

## Solución de Problemas

- **Error al abrir la cámara**: Asegúrate de que tu cámara web esté conectada y funcionando correctamente.
- **Detección inexacta**: Prueba a mejorar la iluminación o entrenar el modelo con más imágenes.
- **Error al cargar el modelo**: Verifica que el archivo `modelo_expresiones.h5` exista en la carpeta correcta.

¡Diviértete detectando expresiones y asociándolas con tus personajes favoritos! 