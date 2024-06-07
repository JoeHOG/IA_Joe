## Explicación del Código e Implementación

### Creación del Dataset

1. **Organización de Imágenes**:
    - Se creó un dataset completo de imágenes de Wally.
    - Las imágenes se colocaron en carpetas separadas:
        - **Negativas**: Imágenes donde no interactúa Wally.
        - **Positivas**: Imágenes donde Wally aparece completamente.

### Importación de Bibliotecas

- Se utilizaron las siguientes bibliotecas:
    - `numpy`
    - `cv2` (OpenCV)

### Uso del Cascade Classifier

1. **Importación del XML**:
    - Se utilizó un archivo XML generado previamente con el cascade classifier.

2. **Lectura y Procesamiento de Imágenes**:
    - Se leyó la imagen a comparar con el XML generado.
    - Se convirtió la imagen a escala de grises.

3. **Detección de Wally**:
    - Se utilizó un ciclo para detectar más de un Wally en la imagen.
    - Se imprimieron los resultados de la detección.
