import cv2
import numpy as np
import os
import pickle
from keras.models import load_model
import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

model = load_model('my_model.h5')


count = 0
output=[]

while True:
    ret, frame = cap.read()

    #converting image to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:

        #roi(region of intrest)
        roi_gray = gray[y:y+h,x:x+w]

        # Preprocess the face image (e.g., resize it to a consistent size)
        face_img = cv2.resize(roi_gray, (64, 64))

        face_img = face_img.reshape(64, 64, 1)  # Reshape to (50, 50, 1)
       
        
        # Make a prediction on the face image using the trained model
        prediction = model.predict(face_img.reshape(1, 64, 64, 1))
        probvalue = np.amax(prediction) 

        
            
        
        class_labels = np.argmax(prediction, axis=1)
        print(class_labels)
        print(int(probvalue*100))
        if int(probvalue*100)>=100:
            if np.argmax(prediction)==0:
                label = "Prabhat"
            
            elif np.argmax(prediction)==1:
                label = "Rabindra"
                
            elif np.argmax(prediction)==2:
                label = "Riju"
                
            elif np.argmax(prediction)==3:
                label = "Rojina"
                
            elif np.argmax(prediction)==4:
                label = "Yujan"
            count+=1
                    
        else:
            label="Unkown"



        


        #Drawing a rectangle around face
        color = (0,0,255) #BGR
        stroke = 2 #thickness of line
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,str(int(probvalue*100))+"%",(180,75),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        output.append(label)



    cv2.imshow("Video capture",frame)
    
    if count==20:
        break

    if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()