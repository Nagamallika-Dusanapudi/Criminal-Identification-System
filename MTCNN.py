#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model,model_from_json


# In[3]:


import architecture 
weights_path = "facenet_keras_weights.h5"
enc_model=architecture.InceptionResNetV2()
enc_model.load_weights(weights_path)


# In[4]:

mtcnn_detector = MTCNN()


# In[5]:


def detect_face(filename, required_size=(160, 160),normalize = True):

    img = Image.open(filename)

    # convert to RGB
    img = img.convert('RGB')
 
    # convert to array
    pixels = np.asarray(img)
 
    # detect faces in the image
    results = mtcnn_detector.detect_faces(pixels)
 
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
  
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
 
    if normalize == True:

        mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
        std = np.std(face_array, axis=(0,1,2), keepdims=True)
        std_adj = np.maximum(std, 1.0)
        return (face_array - mean) / std

    else : 
        return face_array


# In[6]:


known_faces_encodings = []
known_faces_ids = []

known_faces_path = "dataset/"

for filename in os.listdir(known_faces_path):
  
  # Detect faces
  face = detect_face(known_faces_path+filename,normalize = True)

  # Compute face encodings

  feature_vector = enc_model.predict(face.reshape(1,160,160,3))
  feature_vector/= np.sqrt(np.sum(feature_vector**2))
  known_faces_encodings.append(feature_vector)

  # Save Person IDs
  label = filename.split('.')[0]
  known_faces_ids.append(label)


known_faces_encodings = np.array(known_faces_encodings).reshape(len(known_faces_encodings),128)
known_faces_ids = np.array(known_faces_ids)
count=0



def recognize(img,known_faces_encodings,known_faces_ids,threshold = 0.75):

  scores = np.zeros((len(known_faces_ids),1),dtype=float)

  enc = enc_model.predict(img.reshape(1,160,160,3))
  enc/= np.sqrt(np.sum(enc**2))

  scores = np.sqrt(np.sum((enc-known_faces_encodings)**2,axis=1))

  match = np.argmin(scores)

  if scores[match] > threshold :

    return ("UNKNOWN",0)
      
  else :

    return (known_faces_ids[match],scores[match])


# In[9]:


names={'4':'Mallika','3':'Pooja','UNKNOWN':'UNKNOWN'}



from geopy.geocoders import Nominatim
  
geoLoc = Nominatim(user_agent="GetLoc")
  
# passing the coordinates
locname = geoLoc.reverse("16.42374,81.12150")
  
# printing the address/location name
loc=locname.address



# In[16]:


'''import csv
class criminal_recognition:
    def face_recognition(self,mode,file_path,known_faces_encodings,known_faces_ids,
                             detector, threshold):
      count=0
      if detector == 'haar':
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('C://Users//dell//Anaconda3//lib//site-packages//cv2//data//haarcascade_frontalface_default.xml')

      if mode == 'webcam':
        # To capture webcam feed. Change argument for differnt webcams
        cap = cv2.VideoCapture(0)

      elif mode == 'video':
        # To capture video feed 
        cap = cv2.VideoCapture(file_path)

      while True:

        # Read the frame
        _, img = cap.read()

        # Stop if end of video file
        if _ == False:
            break;

        if detector == 'haar':

          #Convert to grayscale
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

          # Detect the faces
          faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        elif detector == 'mtcnn' :  

          results = mtcnn_detector.detect_faces(img)

          if(len(results)==0):
            continue

          faces = []

          for i in range(len(results)):

            x,y,w,h = results[i]['box']
            x, y = abs(x), abs(y)
            faces.append([x,y,w,h])
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:

            image = Image.fromarray(img[y:y+h, x:x+w])
            image = image.resize((160,160))
            face_array = asarray(image)

            # Normalize
            mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
            std = np.std(face_array, axis=(0,1,2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            face_array_normalized = (face_array - mean) / std

            # Recognize
            label = recognize(face_array_normalized,known_faces_encodings,known_faces_ids,threshold = 0.75)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            try:
                cv2.putText(img, names[label[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            except:
                cv2.putText(img, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Display
        cv2.imshow('Face_Recognition', img)
        if count==0 and label[0]!='UNKNOWN':
            line_number = int(label[0])
            myfilepath='C:/Users/dell/Desktop/main project/criminals.csv'
            with open(myfilepath, 'r') as f:
                mycsv = csv.reader(f)
                mycsv = list(mycsv)
                text = mycsv[line_number]
            l=["Name :",str(text[1]),"...Age :",str(text[2]),"...State :",str(text[3]),"...Crimes committed :",str(text[4]),"...criminal found at location :",loc]
            msg=''.join(l)
            count+=1

        # Stop if escape key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

      # Release the VideoCapture object
      cap.release()


# In[19]:


#obj1=criminal_recognition()
#obj1.face_recognition('webcam',None,known_faces_encodings,known_faces_ids,'mtcnn', 0.75)


# In[ ]:
'''



