img = cv.imread("man1.jpg",1)
img2 = np.ones((img.shape[0], img.shape[1],1), dtype=np.uint8)* 150
imgRGB  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        if(img[i][j]>190):
#            img2[i][j]=255
#        else:
#            img2[i][j]=0

umbral_bajo = (0,100,100)
umbral_alto = (15,255,255)

uba=(160, 100,100)
ubb=(180, 255,255)


# hacemos la mask y filtramos en la original
mask1 = cv.inRange(img_hsv, umbral_bajo, umbral_alto)
mask2 = cv.inRange(img_hsv, uba, ubb)

mask = mask2+mask1

res = cv.bitwise_and(img, img, mask=mask)

cv.imshow('mask', mask)

cv.imshow('ejemplo1', img)
cv.imshow('ejemplo2', img2)
cv.imshow('ejemplo3', imghsv)
cv.imshow('resultado', res)
cv.imshow('ejemplorgb', imgRGB)
print(img.shape)
cv.waitKey(0)
cv.destroyAllWindows()