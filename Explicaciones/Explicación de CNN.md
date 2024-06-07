```markdown
## Proceso de Entrenamiento del Modelo para Reconocimiento de Incidentes

Para este proceso, se requirió recopilar una gran cantidad de videos de diversos incidentes como incendios, inundaciones, robos, robos a casas y tornados. A continuación, se describe el flujo del trabajo realizado:

1. **Recopilación de Videos**:
   - Se utilizaron videos largos de aproximadamente 10 minutos cada uno.
   - Los videos fueron recortados frame a frame para que la red neuronal pudiera entender cada incidente de manera detallada.

2. **Reutilización de Código**:
   - Se recicló un código proporcionado por el docente.
   - Este código permitió importar los resultados obtenidos, generando una media de 12,000 fotos por carpeta.

3. **Instalación de Paqueterías Necesarias**:
   - Para que el código funcionara correctamente, se instalaron las siguientes librerías:
     ```sh
     pip install matplotlib
     pip install scikit-learn
     pip install tensorflow
     pip install scikit-image
     ```

4. **Lectura y Etiquetado de Carpetas**:
   - Se leyeron las carpetas de imágenes y se etiquetaron para diferenciarlas.
   - Se implementó la importación de `pickle`, lo que ayudó a extraer los nombres de las etiquetas y su mapeo, evitando la necesidad de correr el programa nuevamente.

5. **Procesamiento de Imágenes**:
   - Se imprimieron las salidas, que en este caso son 5 incidentes.
   - Se creó el dataset, asegurándose de que todas las imágenes estuvieran del mismo tamaño y dimensión.
   - Las imágenes se procesaron y se hizo un encoding para la red.

6. **Creación del Set de Entrenamiento y Validación**:
   - Se creó el modelo de la red neuronal convolucional (CNN) con las iteraciones necesarias para su entrenamiento.
   - El modelo se exportó con tamaños específicos y se realizaron más procesos.

7. **Visualización y Corrección del Entrenamiento**:
   - Se mostró la gráfica de entrenamiento mientras se guardaba la red en un archivo `.h5`.
   - Se hicieron correcciones del entrenamiento basadas en una graficación de los ejes X e Y.

8. **Evaluación del Modelo**:
   - Finalmente, se llamó al modelo y a los mapeos de clases, señalando la imagen que se compararía para determinar si era un incendio, inundación, tornado, robo o robo a casa.

Este proceso permitió crear un modelo robusto capaz de reconocer diferentes tipos de incidentes con precisión.
```