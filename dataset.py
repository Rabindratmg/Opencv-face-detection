import cv2

def generate_dataset():
    face_classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') 
    cap=cv2.VideoCapture(0)
    img_id=0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.5,5)
        for (x,y,w,h) in faces:
            roi_face= gray[y:y+h,x:x+w]
            img_id +=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,str(img_id),(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            face = cv2.resize(roi_face,(200,200))
            file_path = "dataset/" + "Yujan"+ str(img_id)+".jpg"
            cv2.imwrite(file_path,face)
            cv2.imshow("cropped_face",frame)
        if cv2.waitKey(20) == 13 or int(img_id)==1000:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("sample collected succesfully")

generate_dataset()


