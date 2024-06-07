## Explicación del Código de Wally

En el código de Wally, se utilizó una lógica similar a la implementada en el ejemplo previo con Phaser. Aquí se detalla el flujo del proceso:

1. **Reconocimiento Facial con Redes Neuronales**:
   - Se utiliza un `facerecognizer` para reconocer rostros utilizando una red neuronal previamente entrenada.
   - El entrenamiento de la red neuronal se realizó utilizando un archivo XML que contiene los datos necesarios para reconocer diferentes personas.

2. **Carpeta de Emociones**:
   - Se creó una carpeta que contiene diferentes emociones: sorprendido, feliz y enojado.
   - Estas emociones se usan como referencia para el reconocimiento emocional de las personas en el video.

3. **Captura y Procesamiento de Video**:
   - El código captura un video desde la cámara web.
   - Durante la captura, el sistema reconoce los rostros y el entorno con el que la persona está interactuando.
   - El proceso incluye varias etapas: detectar, capturar y procesar el video registrado.

4. **Visualización de Resultados**:
   - Finalmente, los resultados del reconocimiento facial y emocional se visualizan, mostrando las emociones detectadas y la interacción de la persona con su entorno.
