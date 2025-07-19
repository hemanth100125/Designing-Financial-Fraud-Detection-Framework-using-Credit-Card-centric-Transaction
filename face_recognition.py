import cv2
import os
import time
# Import numpy for matrices calculations
import numpy as np
        
# Create Local Binary Patterns Histograms for face recognization
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face_LBPHFaceRecognizer.create()
#recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')
##recognizer.read('/home/pi/Desktop/face_recog_folder/Raspberry-Face-Recognition-master/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

count1=0
while True:

        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)

        # For each face in faces
        for(x,y,w,h) in faces:
            count1 += 1
            # Recognize the face belongs to which ID
            Id,i = recognizer.predict(gray[y:y+h,x:x+w])
            #id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            print(i)
            Id1=''
            # Check the ID if exist
            if i < 60:
                if Id == 1 :
                    Id1 = "Niharika"
                    print(Id1)
                if Id == 2 :
                    Id1 = "Ananys"
                    print(Id1)
                if Id == 3 :
                    Id1 = "fgds"
                    print(Id1)
                if Id == 4 :
                    Id1 = "fgjh 4"
                    print(Id1)
            else:               
                Id1 = "unknown"
                print(Id1)
            
            # Put text describe who is in the picture
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(im, str(Id1), (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im)
        # If 'q' is pressed, close program
        if cv2.waitKey(20) & count1 == 100: #if cv2.waitKey(10) & 0xFF == ord('q'):
            break
           
cam.release()
# Close all windows
cv2.destroyAllWindows()


