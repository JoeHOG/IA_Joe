import cv2 as cv 
cap = cv.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret == True:
        img2= cv.cvtColor(img, cv.COLOR_BGR2HSV)
        ubb=(100,80,80)
        uba=(110,255,255)
        mask= cv.inRange(img2, ubb, uba)
        res=cv.bitwise_and (img, img, mask=mask)
        cv.imshow('img2',img)
        cv.imshow('res',res)
        cv.imshow('mask',mask)
        k =cv.waitKey(0) & 0xFF
        if k == 27 :
            break
cap.release()
cv.destroyAllWindows()