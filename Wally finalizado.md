import numpy as np
import cv2 as cv

# Cargar el clasificador entrenado
rostro = cv.CascadeClassifier('C:\\Users\\joel_\\cascade10.xml')

# Cargar la imagen de Wally
img = cv.imread('C:\\Users\\joel_\\wally\\jj.png')

# Convertir la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
rostros = rostro.detectMultiScale(gray, 1.3, 5)

# Procesar cada rostro detectado
for (x, y, w, h) in rostros:
    # Dibujar un rectángulo verde alrededor del rostro detectado
    img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con los rostros detectados y los rectángulos
cv.imshow('rostros', img)
cv.waitKey(0)
cv.destroyAllWindows()
# Wally completo y adjunto el drive del entrenamiento usado como el modelo y las imagenes positivas y negativas
# https://drive.google.com/drive/folders/1Aw2zdNA-iPTmbphR-lky8Bk5UsNNVZyE