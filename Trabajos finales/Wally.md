```python
import numpy as np
import cv2 as cv
import math 

rostro = cv.CascadeClassifier('C:\\Users\\joel_\\cascade5.xml')
cap = cv.VideoCapture(0)
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
       frame2 = frame[ y:y+h, x:x+w]
        #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
       cv.imwrite('/home/likcos/pruebacaras/juan/juan'+str(i)+'.jpg', frame2)
       cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```


```python
#El chidooooooooooooooooooooooooooooooooooooooooooooooo
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



```


```python
import cv2 as cv
import numpy as np
import os

# Especificar el directorio de salida
output_dir = 'C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\p\\p'

# Crear el directorio si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar la imagen original
img_path = 'C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\p\\p\\wally31.png'
img = cv.imread(img_path)

if img is None:
    print(f"Error: no se pudo cargar la imagen desde {img_path}")
else:
    print(f"Imagen cargada correctamente desde {img_path}")

# Función para rotar la imagen
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Obtener la matriz de rotación
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    
    return rotated

# Lista de ángulos para rotar la imagen
angles = range(0, 360, 15)  # Rotar cada 15 grados

# Nombre base para los archivos guardados
base_name = 'wally_ro31'

# Procesar y guardar las imágenes rotadas
for angle in angles:
    rotated_img = rotate_image(img, angle)
    resized_img = cv.resize(rotated_img, (30, 30), interpolation=cv.INTER_AREA)
    output_path = os.path.join(output_dir, f'{base_name}_{angle}.png')
    cv.imwrite(output_path, resized_img)
    print(f'Saved: {output_path}')

# Mostrar confirmación de finalización
print("Todas las imágenes se han guardado correctamente.")

```

    Imagen cargada correctamente desde C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally31.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_0.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_15.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_30.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_45.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_60.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_75.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_90.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_105.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_120.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_135.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_150.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_165.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_180.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_195.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_210.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_225.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_240.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_255.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_270.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_285.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_300.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_315.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_330.png
    Saved: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_345.png
    Todas las imágenes se han guardado correctamente.
    


```python
import os
from PIL import Image

# Directorio de entrada
input_dir = r"C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si"
# Directorio de salida (puedes cambiar esto a donde quieras guardar las imágenes redimensionadas)
output_dir = r"C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\p"


# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Redimensionar las imágenes
for filename in os.listdir(input_dir):
    if filename.startswith('._'):
        print(f"Omitiendo archivo macOS oculto: {filename}")
        continue  # Omitir archivos ocultos de macOS
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Filtrar solo los tipos de archivo de imagen
        filepath = os.path.join(input_dir, filename)
        try:
            with Image.open(filepath) as img:
                print(f"Abriendo imagen: {filepath}")
                img_resized = img.resize((30, 30), Image.ANTIALIAS)
                output_filepath = os.path.join(output_dir, filename)
                img_resized.save(output_filepath)
                print(f"Imagen guardada: {output_filepath}")
        except Exception as e:
            print(f"No se pudo procesar el archivo {filepath}: {e}")
    else:
        print(f"Archivo no soportado: {filename}")


```

    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 1 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 1 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 10 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 10 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 100 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 100 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 101 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 101 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 102 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 102 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 103 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 103 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 104 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 104 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 105 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 105 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 106 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 106 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 107 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 107 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 108 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 108 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 109 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 109 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 11 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 11 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 110 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 110 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 112 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 112 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 113 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 113 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 114 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 114 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 115 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 115 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 116 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 116 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 117 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 117 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 118 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 118 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 119 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 119 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 120 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 120 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 121 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 121 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 122 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 122 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 124 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 124 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 126 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 126 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 127 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 127 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 128 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 128 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 129 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 129 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 13 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 13 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 130 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 130 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 131 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 131 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 132 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 132 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 133 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 133 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 134 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 134 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 135 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 135 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 136 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 136 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 138 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 138 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 139 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 139 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 14 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 14 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 140 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 140 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 141 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 141 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 142 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 142 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 143 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 143 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 144 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 144 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 145 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 145 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 146 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 146 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 147 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 147 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 148 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 148 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 15 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 15 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 150 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 150 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 151 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 151 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 152 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 152 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 153 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 153 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 154 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 154 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 155 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 155 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 156 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 156 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 157 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 157 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 158 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 158 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 159 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 159 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 16 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 16 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 160 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 160 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 161 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 161 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 162 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 162 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 163 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 163 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 164 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 164 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 165 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 165 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 166 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 166 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 167 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 167 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 168 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 168 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 169 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 169 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 17 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 17 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 170 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 170 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 171 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 171 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 172 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 172 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 173 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 173 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 174 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 174 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 175 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 175 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 176 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 176 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 177 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 177 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 178 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 178 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 179 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 179 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 18 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 18 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 180 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 180 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 181 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 181 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 182 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 182 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 183 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 183 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 184 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 184 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 185 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 185 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 186 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 186 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 187 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 187 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 188 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 188 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 189 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 189 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 19 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 19 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 190 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 190 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 191 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 191 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 192 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 192 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 194 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 194 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 195 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 195 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 196 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 196 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 198 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 198 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 2 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 2 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 20 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 20 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 200 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 200 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 202 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 202 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 204 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 204 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 206 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 206 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 207 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 207 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 208 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 208 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 21 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 21 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 210 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 210 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 212 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 212 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 214 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 214 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 216 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 216 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 217 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 217 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 218 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 218 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 22 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 22 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 220 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 220 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 222 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 222 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 223 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 223 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 225 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 225 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 226 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 226 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 227 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 227 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 228 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 228 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 229 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 229 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 23 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 23 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 230 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 230 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 232 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 232 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 233 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 233 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 234 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 234 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 235 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 235 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 236 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 236 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 237 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 237 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 238 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 238 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 24 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 24 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 240 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 240 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 242 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 242 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 243 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 243 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 244 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 244 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 245 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 245 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 246 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 246 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 247 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 247 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 248 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 248 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 249 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 249 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 25 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 25 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 250 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 250 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 252 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 252 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 254 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 254 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 256 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 256 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 257 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 257 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 258 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 258 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 26 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 26 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 260 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 260 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 261 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 261 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 262 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 262 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 265 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 265 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 266 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 266 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 27 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 27 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 270 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 270 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 274 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 274 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 277 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 277 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 278 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 278 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 28 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 28 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 289 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 289 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 29 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 29 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 290 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 290 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 298 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 298 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 299 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 299 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 3 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 3 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 30 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 30 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 300 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 300 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 301 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 301 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 302 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 302 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 303 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 303 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 304 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 304 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 305 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 305 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 309 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 309 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 31 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 31 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 311 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 311 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 32 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 32 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 33 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 33 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 34 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 34 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 35 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 35 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 36 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 36 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 37 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 37 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 38 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 38 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 39 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 39 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 4 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 4 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 40 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 40 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 41 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 41 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 42 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 42 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 43 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 43 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 44 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 44 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 45 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 45 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 46 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 46 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 47 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 47 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 48 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 48 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 49 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 49 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 5 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 5 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 50 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 50 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 51 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 51 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 52 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 52 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 53 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 53 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 54 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 54 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 58 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 58 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 6 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 6 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 64 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 64 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 67 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 67 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 7 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 7 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 70 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 70 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 71 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 71 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 72 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 72 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 73 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 73 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 74 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 74 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 75 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 75 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 76 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 76 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 77 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 77 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 78 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 78 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 79 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 79 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 8 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 8 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 80 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 80 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 81 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 81 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 82 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 82 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 83 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 83 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 84 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 84 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 85 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 85 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 86 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 86 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 88 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 88 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 89 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 89 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 9 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 9 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 90 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 90 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 91 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 91 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 92 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 92 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 93 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 93 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 94 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 94 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 95 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 95 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 96 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 96 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 97 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 97 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    Abriendo imagen: C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 98 eliminado.png
    No se pudo procesar el archivo C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\Si\Fondo de 98 eliminado.png: module 'PIL.Image' has no attribute 'ANTIALIAS'
    


```python
# Directorio de entrada
input_dir = r"C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\__MACOSX\\Si"
# Directorio de salida (puedes cambiar esto a donde quieras guardar las imágenes redimensionadas)
output_dir = r"C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\__MACOSX\\Si_resized"

```


```python
import os
import glob

def renombrar_imagenes(ruta):
    # Verificar si la ruta existe
    if not os.path.exists(ruta):
        print(f"La ruta {ruta} no existe.")
        return

    # Buscar todas las imágenes en la ruta (considerando las extensiones más comunes)
    imagenes = glob.glob(os.path.join(ruta, "*.png")) + \
               glob.glob(os.path.join(ruta, "*.jpg")) + \
               glob.glob(os.path.join(ruta, "*.jpeg")) + \
               glob.glob(os.path.join(ruta, "*.bmp")) + \
               glob.glob(os.path.join(ruta, "*.gif"))

    if not imagenes:
        print(f"No se encontraron imágenes en la ruta {ruta}.")
        return

    # Renombrar cada imagen
    for i, imagen in enumerate(imagenes):
        # Obtener la extensión del archivo
        extension = os.path.splitext(imagen)[1]
        # Crear el nuevo nombre
        nuevo_nombre = f"wallly{i}{extension}" #aqui colocar el nombre deseado de las imagenes
        # Obtener la ruta completa del nuevo nombre
        nueva_ruta = os.path.join(ruta, nuevo_nombre)
        # Renombrar el archivo
        os.rename(imagen, nueva_ruta)
        print(f"Renombrado: {imagen} -> {nueva_ruta}")

    print("Renombrado completado.")

# Forma de usar
ruta_imagenes = "C:\\Users\\joel_\\OneDrive\\Escritorio\\Willy\\p\\p"
renombrar_imagenes(ruta_imagenes)
```

    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\216.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly0.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\217.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly1.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\218.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly2.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\219.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly3.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\220.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly4.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\222.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly5.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\223.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly6.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\224.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly7.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly8.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\226.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly9.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\227.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly10.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\228.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly11.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\229.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly12.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\230.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly13.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\231.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly14.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\232.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly15.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\234.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly16.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\235.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly17.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\236.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly18.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\237.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly19.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\238.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly20.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\239.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly21.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly22.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\241.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly23.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\242.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly24.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\243.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly25.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\244.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly26.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\246.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly27.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\248.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly28.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\249.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly29.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\250.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly30.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\251.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly31.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\252.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly32.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\253.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly33.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\254.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly34.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly35.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\256.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly36.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\257.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly37.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\258.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly38.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\260.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly39.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\261.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly40.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\262.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly41.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\263.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly42.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\264.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly43.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\265.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly44.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\266.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly45.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\267.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly46.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\268.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly47.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\269.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly48.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly49.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\298.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly50.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\4.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly51.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\401.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly52.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\407.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly53.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\409.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly54.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\425.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly55.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\438.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly56.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\444.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly57.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\445.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly58.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\465.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly59.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\5.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly60.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\52.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly61.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\55.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly62.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\6.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly63.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\64.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly64.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\69.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly65.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\7.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly66.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\70.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly67.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\76.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly68.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\79.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly69.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\85.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly70.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\86.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly71.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\9.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly72.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly73.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally1.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly74.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally10.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly75.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally100.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly76.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally101.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly77.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally102.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly78.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally103.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly79.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally104.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly80.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly81.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally106.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly82.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally107.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly83.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally108.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly84.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally109.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly85.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally11.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly86.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally110.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly87.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally111.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly88.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally112.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly89.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally113.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly90.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally114.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly91.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally115.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly92.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally116.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly93.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally117.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly94.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally118.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly95.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally119.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly96.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally12.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly97.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly98.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally121.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly99.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally122.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly100.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally123.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly101.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally124.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly102.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally125.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly103.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally126.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly104.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally127.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly105.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally128.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly106.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally129.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly107.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally13.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly108.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally130.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly109.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally131.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly110.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally132.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly111.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally133.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly112.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally134.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly113.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly114.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally136.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly115.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally137.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly116.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally138.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly117.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally139.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly118.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally14.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly119.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally140.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly120.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally141.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly121.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally142.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly122.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally143.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly123.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally144.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly124.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally145.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly125.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally146.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly126.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally147.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly127.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally148.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly128.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally149.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly129.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly130.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly131.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally151.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly132.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally152.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly133.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally153.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly134.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally154.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly135.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally155.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly136.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally156.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly137.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally157.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly138.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally158.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly139.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally159.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly140.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally16.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly141.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally160.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly142.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally161.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly143.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally162.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly144.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally163.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly145.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally164.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly146.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly147.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally166.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly148.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally167.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly149.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally168.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly150.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally169.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly151.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally17.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly152.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally170.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly153.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally171.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly154.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally172.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly155.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally173.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly156.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally174.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly157.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally175.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly158.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally176.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly159.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally177.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly160.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally178.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly161.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally179.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly162.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally18.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly163.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly164.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally181.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly165.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally182.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly166.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally183.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly167.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally184.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly168.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally185.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly169.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally186.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly170.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally187.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly171.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally188.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly172.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally189.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly173.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally19.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly174.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally190.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly175.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally191.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly176.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally192.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly177.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally193.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly178.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally194.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly179.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly180.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally196.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly181.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally197.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly182.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally198.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly183.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally199.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly184.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally2.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly185.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally20.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly186.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally200.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly187.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally201.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly188.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally202.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly189.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally203.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly190.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally204.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly191.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally205.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly192.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally206.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly193.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally207.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly194.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally208.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly195.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally209.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly196.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally21.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly197.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly198.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally211.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly199.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally212.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly200.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally213.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly201.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally214.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly202.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally215.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly203.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally216.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly204.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally217.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly205.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally218.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly206.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally219.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly207.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally22.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly208.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally220.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly209.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally221.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly210.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally222.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly211.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally223.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly212.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally224.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly213.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly214.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally226.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly215.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally227.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly216.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally228.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly217.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally229.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly218.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally23.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly219.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally230.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly220.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally231.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly221.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally232.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly222.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally233.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly223.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally234.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly224.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally235.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly225.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally236.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly226.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally237.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly227.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally238.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly228.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally239.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly229.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally24.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly230.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally25.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly231.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally26.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly232.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally27.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly233.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally28.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly234.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally29.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly235.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally3.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly236.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly237.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally31.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly238.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally32.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly239.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally33.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly240.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally34.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly241.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally35.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly242.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally36.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly243.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally37.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly244.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally38.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly245.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally39.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly246.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally4.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly247.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally40.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly248.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally41.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly249.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally42.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly250.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally43.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly251.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally44.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly252.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly253.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally46.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly254.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally47.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly255.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally48.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly256.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally49.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly257.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally5.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly258.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally50.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly259.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally51.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly260.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally52.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly261.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally53.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly262.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally54.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly263.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally55.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly264.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally56.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly265.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally57.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly266.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally58.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly267.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally59.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly268.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally6.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly269.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly270.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally61.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly271.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally62.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly272.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally63.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly273.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally64.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly274.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally65.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly275.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally66.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly276.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally67.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly277.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally68.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly278.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally69.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly279.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally7.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly280.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally70.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly281.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally71.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly282.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally72.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly283.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally73.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly284.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally74.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly285.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly286.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally76.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly287.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally77.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly288.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally78.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly289.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally79.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly290.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally8.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly291.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally80.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly292.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally81.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly293.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally82.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly294.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally83.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly295.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally84.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly296.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally85.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly297.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally86.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly298.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally87.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly299.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally88.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly300.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally89.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly301.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally9.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly302.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly303.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally91.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly304.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally92.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly305.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally93.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly306.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally94.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly307.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally95.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly308.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally96.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly309.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally97.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly310.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally98.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly311.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally99.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly312.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly313.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly314.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly315.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly316.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly317.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly318.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly319.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly320.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly321.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly322.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly323.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly324.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly325.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly326.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly327.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly328.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly329.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly330.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly331.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly332.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly333.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly334.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly335.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro10_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly336.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly337.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly338.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly339.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly340.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly341.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly342.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly343.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly344.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly345.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly346.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly347.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly348.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly349.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly350.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly351.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly352.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly353.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly354.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly355.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly356.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly357.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly358.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly359.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro11_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly360.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly361.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly362.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly363.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly364.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly365.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly366.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly367.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly368.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly369.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly370.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly371.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly372.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly373.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly374.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly375.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly376.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly377.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly378.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly379.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly380.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly381.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly382.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly383.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro12_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly384.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly385.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly386.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly387.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly388.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly389.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly390.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly391.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly392.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly393.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly394.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly395.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly396.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly397.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly398.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly399.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly400.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly401.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly402.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly403.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly404.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly405.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly406.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly407.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro13_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly408.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly409.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly410.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly411.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly412.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly413.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly414.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly415.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly416.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly417.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly418.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly419.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly420.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly421.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly422.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly423.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly424.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly425.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly426.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly427.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly428.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly429.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly430.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly431.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro14_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly432.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly433.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly434.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly435.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly436.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly437.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly438.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly439.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly440.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly441.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly442.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly443.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly444.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly445.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly446.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly447.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly448.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly449.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly450.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly451.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly452.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly453.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly454.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly455.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro15_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly456.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly457.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly458.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly459.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly460.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly461.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly462.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly463.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly464.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly465.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly466.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly467.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly468.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly469.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly470.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly471.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly472.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly473.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly474.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly475.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly476.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly477.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly478.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly479.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro16_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly480.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly481.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly482.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly483.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly484.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly485.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly486.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly487.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly488.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly489.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly490.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly491.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly492.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly493.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly494.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly495.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly496.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly497.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly498.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly499.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly500.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly501.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly502.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly503.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro17_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly504.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly505.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly506.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly507.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly508.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly509.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly510.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly511.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly512.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly513.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly514.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly515.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly516.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly517.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly518.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly519.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly520.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly521.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly522.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly523.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly524.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly525.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly526.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly527.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro18_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly528.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly529.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly530.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly531.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly532.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly533.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly534.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly535.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly536.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly537.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly538.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly539.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly540.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly541.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly542.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly543.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly544.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly545.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly546.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly547.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly548.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly549.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly550.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly551.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro19_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly552.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly553.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly554.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly555.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly556.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly557.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly558.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly559.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly560.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly561.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly562.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly563.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly564.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly565.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly566.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly567.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly568.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly569.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly570.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly571.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly572.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly573.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly574.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly575.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro20_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly576.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly577.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly578.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly579.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly580.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly581.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly582.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly583.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly584.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly585.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly586.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly587.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly588.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly589.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly590.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly591.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly592.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly593.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly594.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly595.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly596.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly597.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly598.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly599.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro21_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly600.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly601.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly602.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly603.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly604.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly605.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly606.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly607.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly608.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly609.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly610.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly611.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly612.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly613.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly614.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly615.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly616.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly617.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly618.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly619.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly620.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly621.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly622.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly623.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro22_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly624.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly625.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly626.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly627.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly628.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly629.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly630.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly631.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly632.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly633.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly634.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly635.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly636.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly637.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly638.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly639.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly640.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly641.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly642.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly643.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly644.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly645.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly646.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly647.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro23_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly648.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly649.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly650.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly651.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly652.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly653.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly654.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly655.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly656.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly657.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly658.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly659.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly660.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly661.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly662.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly663.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly664.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly665.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly666.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly667.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly668.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly669.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly670.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly671.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro24_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly672.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly673.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly674.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly675.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly676.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly677.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly678.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly679.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly680.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly681.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly682.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly683.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly684.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly685.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly686.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly687.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly688.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly689.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly690.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly691.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly692.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly693.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly694.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly695.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro25_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly696.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly697.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly698.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly699.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly700.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly701.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly702.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly703.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly704.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly705.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly706.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly707.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly708.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly709.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly710.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly711.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly712.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly713.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly714.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly715.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly716.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly717.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly718.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly719.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro26_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly720.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly721.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly722.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly723.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly724.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly725.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly726.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly727.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly728.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly729.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly730.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly731.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly732.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly733.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly734.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly735.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly736.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly737.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly738.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly739.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly740.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly741.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly742.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly743.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro27_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly744.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly745.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly746.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly747.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly748.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly749.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly750.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly751.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly752.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly753.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly754.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly755.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly756.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly757.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly758.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly759.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly760.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly761.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly762.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly763.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly764.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly765.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly766.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly767.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro28_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly768.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly769.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly770.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly771.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly772.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly773.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly774.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly775.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly776.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly777.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly778.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly779.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly780.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly781.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly782.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly783.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly784.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly785.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly786.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly787.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly788.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly789.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly790.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly791.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro29_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly792.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly793.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly794.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly795.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly796.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly797.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly798.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly799.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly800.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly801.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly802.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly803.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly804.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly805.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly806.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly807.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly808.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly809.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly810.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly811.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly812.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly813.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly814.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly815.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro30_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly816.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly817.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly818.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly819.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly820.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly821.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly822.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly823.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly824.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly825.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly826.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly827.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly828.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly829.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly830.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly831.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly832.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly833.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly834.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly835.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly836.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly837.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly838.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly839.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro31_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly840.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly841.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly842.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly843.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly844.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly845.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly846.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly847.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly848.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly849.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly850.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly851.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly852.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly853.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly854.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly855.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly856.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly857.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly858.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly859.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly860.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly861.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly862.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly863.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro8_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly864.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_0.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly865.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_105.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly866.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_120.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly867.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_135.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly868.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_15.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly869.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_150.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly870.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_165.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly871.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_180.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly872.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_195.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly873.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_210.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly874.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_225.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly875.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_240.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly876.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_255.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly877.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_270.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly878.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_285.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly879.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_30.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly880.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_300.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly881.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_315.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly882.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_330.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly883.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_345.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly884.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_45.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly885.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_60.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly886.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_75.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly887.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wally_ro9_90.png -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly888.png
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\1.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly889.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\221.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly890.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\233.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly891.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\245.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly892.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\247.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly893.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\259.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly894.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\271.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly895.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\412.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly896.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\418.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly897.jpg
    Renombrado: C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\430.jpg -> C:\Users\joel_\OneDrive\Escritorio\Willy\p\p\wallly898.jpg
    Renombrado completado.
    


```python
import numpy as np
import cv2 as cv

# Cargar el clasificador entrenado
rostro = cv.CascadeClassifier('C:\\Users\\joel_\\30.xml')

# Cargar la imagen de Wally
img = cv.imread('C:\\Users\\joel_\\wally\\3.jpg')

# Convertir la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Ajustar los parámetros del detector para mejorar la precisión
# Puedes experimentar con scaleFactor y minNeighbors
rostros = rostro.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# Procesar cada rostro detectado
for (x, y, w, h) in rostros:
    # Dibujar un rectángulo verde alrededor del rostro detectado
    img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con los rostros detectados y los rectángulos
cv.imshow('rostros', img)
cv.waitKey(0)
cv.destroyAllWindows()

```


```python

```
