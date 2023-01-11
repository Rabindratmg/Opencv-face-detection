import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #converting image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        #roi(region of intrest)
        roi_gray = gray[y:y+h,x:x+w]

        #Drawing a rectangle around face
        color = (0,0,255) #BGR
        stroke = 2 #thickness of line
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)



    cv2.imshow("Video capture",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()