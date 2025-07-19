# from xml.sax.handler import feature_string_interning
from flask import Flask, render_template, request

import pickle
import telepot
pradeep = telepot.Bot('7912024658:AAHqD4q4gsjhEkKobWPtU2_mD-vBB52tOVc')
model = pickle.load(open('Naive_Bayes.pkl', 'rb'))

import cv2
import numpy as np
        
# Create Local Binary Patterns Histograms for face recognization
# recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face_LBPHFaceRecognizer.create()
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('FACE_RECOGNITION/trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "FACE_RECOGNITION/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pred', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        Data=[]
        for i in request.form.listvalues():
            for ii in i:
                Data.append(ii)
        print(Data)
        import random
        a=[]
        for i in range(0,28):
            a.append('{:.4f}'.format(random.uniform(0.0, 1.5)))
        return render_template('home.html', a=a)
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Data=[]
        for i in request.form.listvalues():
            for ii in i:
                Data.append(float(ii))
        print(Data)
        out = model.predict([Data])
        print(out)
        output=''
        if out[0] == 1 or Data[0] <= 110:
            
            output='Fraud Trancaction'
        else:
            
            output='Normal'

        print(output)
        import random
        num = random.randint(1111, 9999)
        print(num)
        
        pradeep.sendMessage('818591758', str(num))


        with open('file.txt', 'w') as f:
            f.write(str(num))
            f.close()
        return render_template('validation.html', msg = output, otp=num )
    return render_template('index.html')

@app.route('/verification', methods=['GET', 'POST'])
def verification():        
    if request.method == 'POST':
        OTP = int(request.form['OTP'])
        print(OTP)
        with open('file.txt', 'r') as f:
            OTP1 =  f.read()
            f.close()

        print(OTP1)
        OTP1 = int(OTP1)
        
        if OTP == OTP1:
            # Initialize and start the video frame capture
            cam = cv2.VideoCapture(0)
            # global count,count1,count2
            count = 0
            count1 = 0
            count2 = 0
            while True:

                    # Read the video frame
                    ret, im =cam.read()

                    # Convert the captured frame into grayscale
                    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

                    # Get all face from the video frame
                    faces = faceCascade.detectMultiScale(gray, 1.2,5)

                    # For each face in faces
                    for(x,y,w,h) in faces:
                        count += 1
                        # Recognize the face belongs to which ID
                        Id,i = recognizer.predict(gray[y:y+h,x:x+w])
                        #id = int(os.path.split(imagePath)[-1].split(".")[1])
                        
                        print(i)
                        Id1=''
                        # Check the ID if exist
                        if i < 70:
                            count1 += 1
                            if Id == 1 :
                                Id1 = "Romesh"
                                print(Id1)
                            if Id == 2 :
                                Id1 = "abc"
                                print(Id1)
                            if Id == 3 :
                                Id1 = "abc2"
                                print(Id1)
                            if Id == 4 :
                                Id1 = "abc3"
                                print(Id1)
                        else:
                            count2 += 1
                            Id1 = "unknown"
                            print(Id1)
                            
                        
                        # Put text describe who is in the picture
                        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                        cv2.putText(im, str(Id1), (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                        if Id1 == "unknown":
                            cv2.imwrite("out.jpg",im)
                    # Display the video frame with the bounded rectangle
                    cv2.imshow('im',im)
                    # If 'q' is pressed, close program
                    if cv2.waitKey(1) & count > 50: #if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                       
            cam.release()
            # Close all windows
            cv2.destroyAllWindows()

            if count1 > count2:
                return render_template('validation.html', msg1='face authenticated and transaction Successfull')
            else:
                
                pradeep.sendPhoto('818591758',photo=open("out.jpg","rb"))
                return render_template('index.html', msg2='face recognition faild')
        else:
            return render_template('index.html', msg2='Entered wrong otp')
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
