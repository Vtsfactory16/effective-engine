# Detector de Personajes para Niños

Este programa detecta rostros a través de la cámara web y muestra automáticamente diferentes personajes animados en la pantalla. Es ideal para eventos con niños o actividades educativas.

## Características

- Detecta rostros en tiempo real usando la cámara
- Muestra personajes de diferentes categorías (Feliz, Triste, Sorprendido, etc.)
- Cambia automáticamente de personaje cada pocos segundos al detectar un rostro
- No requiere modelos de IA complejos ni configuraciones avanzadas

## Instalación

1. Asegúrate de tener Python instalado (recomendado Python 3.6 o superior)

2. Instala las dependencias necesarias:
   ```
   pip install -r requirements_simple.txt
   ```

## Uso

1. Ejecuta el programa:
   ```
   python detector_simple.py
   ```

2. El programa intentará descargar automáticamente algunas imágenes de personajes de ejemplo.

3. La cámara se activará y comenzará a detectar rostros.

4. Cada vez que se detecte un rostro, después de unos segundos cambiará automáticamente el personaje mostrado.

5. Controles:
   - Presiona 'c' para cambiar manualmente el personaje
   - Presiona 'q' para salir del programa

## Personalización

Puedes añadir tus propios personajes:

1. Coloca imágenes de personajes en la carpeta `personajes` (se creará automáticamente)
2. Usa formato PNG (preferiblemente con fondo transparente)
3. Nombra los archivos con el nombre del personaje (ej: `Mickey Mouse.png`)

También puedes modificar las categorías y personajes editando el archivo `detector_simple.py`:

```python
self.personajes = {
    'Feliz': ['Mickey Mouse', 'Pikachu', 'Bob Esponja'],
    'Triste': ['Eeyore', 'Sadness (Inside Out)', 'Dumbo'],
    # Añade o modifica categorías y personajes aquí
}
```

## Solución de problemas

- **No se abre la cámara**: Verifica que tu cámara web esté conectada y funcionando correctamente.
- **No se muestran los personajes**: Asegúrate de que las imágenes estén en la carpeta `personajes` con los nombres correctos.
- **Error al descargar personajes**: Puedes añadir manualmente imágenes en la carpeta `personajes`.

## Requisitos

- Python 3.6 o superior
- OpenCV
- Requests (para descargar las imágenes de ejemplo)
- Una cámara web funcional 