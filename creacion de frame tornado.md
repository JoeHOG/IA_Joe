import numpy as np
import cv2 as cv
import math

cap=cv.VideoCapture('C:\\Users\\joel_\\proyectovideo\\videos\\tornado2.mp4')
i=0
while True:
    ret, frame = cap.read()
    cv.imshow('img', frame)
    k=cv.waitKey(1)
    i=i+1
    cv.imwrite('C:\\Users\\joel_\\proyectovideo\\resultados\\fotos'+str(i)+'.jpg',frame)
    if k==27:
        break
cap.release()
cv.destroyAllWindows()