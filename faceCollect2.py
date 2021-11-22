import cv2
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
import os
from os import system

def SpeakText(command):  
    # engine = pyttsx3.init('dummy')
    # voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id)
    # engine.setProperty("rate", 150)
    # engine.say(command) 
    # engine.runAndWait()
    s = 'say ' + command
    system(s)

def getName(r):
    try:  
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            #SpeakText('adjusting noises for clear inputs')
            r.adjust_for_ambient_noise(source2, duration=1)
            print("speak now")
            SpeakText("speak the spelling of you name .")
            #listens for the user's input 
            audio2 = r.listen(source2,phrase_time_limit=10)
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
  
            print("Did you say "+MyText)

            name = ""

            for ch in MyText:
                if ch!=" ":
                    name += ch

            print("is your name " + name)
            SpeakText("is spelling of your name is " +MyText+ " . say yes to save  . say no to try again.")
            print('speak now')

            try:
                confirmation = ""
                audio = r.listen(source2,phrase_time_limit=5)
                confirmation = r.recognize_google(audio)
                confirmation = confirmation.lower()
                print("confirmation - " + confirmation)

                if(confirmation=='yes'):
                    nameLock = True
                    SpeakText('name loaded successfully.')
                    return (name,True)
                else:
                    SpeakText('alright , try again .')
                    return("",False)
            except:
                SpeakText("sorry did not get confirmation about the name . try again .")
                return ("",False)

    except:
        SpeakText("sorry no words recognized . please try again .")
        return ("",False)

def collect():
    #initialise camera
    r = sr.Recognizer()
    cap = cv2.VideoCapture(0)

    #face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    skip = 0

    nameLock = False
    askThreeTimes = 0

    while(askThreeTimes!=3):
        name,confirmation = getName(r)
        if(confirmation==True):
            break
        askThreeTimes += 1

    if(name==""):
        return

    SpeakText("smile please , now I will collect photos.")
    file_name = name
    face_data = []
    dataset_path = 'dataset/'
    threshold = 0
    iteration = 0
    while True:
        ret,frame = cap.read()
        
        if ret==False:
            continue
        
        #gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        
        # 5 - min neighbours
        # 1.3 - scaling factor
        #print("checking 1")
        if(len(face_data)==0):
            iteration += 1
            if(iteration==30):
                print("no face detected")
                break
        faces = face_cascade.detectMultiScale(frame,1.3,5)
        
        # sorted acc to area (inc order)
        faces = sorted(faces,key = lambda f: f[2]*f[3])
        
        for face in faces[-1:]:  
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
            #extract (crop out the required face) : region of interest
            offset = 10

            face_section = frame[y - offset : y+h+offset , x-offset : x+w+offset]
            face_section = cv2.resize(face_section,(100,100))
        
            #we are going to store only 10th frame

            skip += 1

            if skip%10==0:
                face_data.append(face_section)
                print(len(face_data))

                if(len(face_data)==15):
                    threshold = 1

            cv2.imshow('frame',frame)
            cv2.imshow("face_section",face_section)

        key = cv2.waitKey(1)&0xFF

        if(threshold==1):
            break
        
        if(key==ord('q')):
            break
    if(len(face_data)==0):
        print("try again")

    else:
        #convert our face list array into numpy array
        face_data = np.asarray(face_data)
        face_data = face_data.reshape((face_data.shape[0],-1))

        print(face_data.shape)

        #save the data into the file system
        np.save(dataset_path + file_name + '.npy',face_data)
        print("data saved successfully")

    cap.release()
    cv2.destroyAllWindows()
    print('face data collected successfully')
    SpeakText('face data saved successfully . entry filed successfully . thank you ')
collect()