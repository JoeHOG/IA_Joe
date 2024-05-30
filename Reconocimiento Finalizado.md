import cv2 as cv
import os 

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('C:\\Users\\joel_\\OneDrive\\Documentos\\IA\\aa\\EmocionesLBPHZwei.xml')
dataSet = 'C:\\Users\\joel_\\OneDrive\\Documentos\\IA\\aa\\Emociones3'
faces  = os.listdir(dataSet)
cap = cv.VideoCapture(1)
rostro = cv.CascadeClassifier('C:\\Users\\joel_\\haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (255,255,0), 1, cv.LINE_AA)
        if result[1] < 100:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) 
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
# Adjunto el drive donde se encuentran todos los archivos generados
# https://drive.google.com/drive/folders/1dn8CmepxlzAXo89SnP5M5MkJwb_X4_Fh