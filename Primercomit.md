import cv2 as cv
import numpy as np

img = cv.imread('tr.png',0)
w,h=img.shape
img2 = np.ones((w*2, h*2), dtype=np.uint8)*150
cv.imshow('marco1', img2)
for i in range(w):
    for j in range(h):
#          img[i,j]=255-img[i,j]
#        if(img[i,j]>150):
#            img[i,j]=255
#        else:
#            img[i,j]=0
cv.imshow('marco2', img)
cv.waitKey(0)
cv.destroyAllWindows()