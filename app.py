from tensorflow.python.keras.backend import set_session
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow import keras
import pickle
import MTCNN
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import csv
import time

id1=50
id2=id1
filename='files/criminals.csv'

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #face_cascade = cv2.CascadeClassifier('C://Users//dell//Anaconda3//lib//site-packages//cv2//data//haarcascade_frontalface_default.xml')
        results = MTCNN.mtcnn_detector.detect_faces(img)
        faces = []
        names={'4':'Mallika','3':'Pooja','UNKNOWN':'UNKNOWN'}
        for i in range(len(results)):

            x,y,w,h = results[i]['box']
            x, y = abs(x), abs(y)
            faces.append([x,y,w,h])
        # Draw the rectangle around each face
            if len(faces)==0:
                time.sleep(25)
                continue
        for (x, y, w, h) in faces:

            image = Image.fromarray(img[y:y+h, x:x+w])
            image = image.resize((160,160))
            face_array = asarray(image)
        mean = np.mean(face_array, axis=(0,1,2), keepdims=True)
        std = np.std(face_array, axis=(0,1,2), keepdims=True)
        std_adj = np.maximum(std, 1.0)
        face_array_normalized = (face_array - mean) / std
        label = MTCNN.recognize(face_array_normalized,MTCNN.known_faces_encodings,MTCNN.known_faces_ids,threshold = 0.75)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        try:
            b=cv2.putText(img, names[label[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        except:
            b=cv2.putText(img, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if MTCNN.count==0 and label[0]!='UNKNOWN':
            line_number = int(label[0])
            myfilepath='files/criminals.csv'
            with open(myfilepath, 'r') as f:
                mycsv = csv.reader(f)
                mycsv = list(mycsv)
                text = mycsv[line_number]
            l=["Name :",str(text[1]),"...Age :",str(text[2]),"...State :",str(text[3]),"...Crimes committed :",str(text[4]),"...criminal found at location :",MTCNN.loc]
            msg=''.join(l)
            MTCNN.count+=1

        return b




@st.cache(suppress_st_warning=True)
def load_data():
    data=pd.read_csv('files/20_victims_of_rape.csv')
    data1=pd.read_csv('files/10_Property_stolen_and_recovered.csv')
    data2=pd.read_csv('files/35_Human_rights_violation_by_police.csv')
    return data,data1,data2

st.title("Criminal Recognition System")
menu=["About App","Add Criminal Details","video surveillance","crime statistics"]
choice=st.sidebar.radio("Select Activity",menu)

if choice=='Add Criminal Details':
    st.text("Please login to add details")
    with st.beta_expander("Login"):
        with st.form(key='form1'):
            st.text("Please enter your credentials")
            username = st.text_input("Username")
            password = st.text_input("Password")
            login_button = st.form_submit_button(label='Login')
            if login_button:
                st.success("Hello {} you ve successfully logged in".format(username))
    st.text("Enter the details of criminals")
    form2 = st.form(key='form2')
    name = form2.text_input("Name")
    age=form2.text_input("Age")
    state = form2.text_input("State/Country")
    crimes = form2.text_input("Crimes committed")
    image=form2.file_uploader("Criminal image")
    submit_button = form2.form_submit_button(label='Add details')
    if submit_button:
        id1+=1
        with open(filename,'a')as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow([id,name,age,state,crimes])
            st.success("you ve successfully added details.")
elif choice=='video surveillance':
    
    st.subheader("criminal recognition")
    #form = st.form(key='my_form')
    #submit_button = form.form_submit_button(label='Capture')
    #cv=pickle.load(open('model.p',"rb"))
    #obj1=MTCNN.criminal_recognition(session,graph)
    #if submit_button:
    #cv.face_recognition('webcam',None,MTCNN.known_faces_encodings,MTCNN.known_faces_ids,'mtcnn', 0.75)
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

elif choice=='crime statistics':


    # In[2]:
    data,data1,data2=load_data()
    # In[12]:
    option=st.selectbox('select a crime',('rape','Property theft','human rights violation'))
    if option=='rape':
        #state,age=st.beta_columns(2)
        ch=st.radio('select criteria',('state-wise','age'))
        if ch=='state-wise':
            st.text('   State/UT wise statistics')
            g1 = pd.DataFrame(data.groupby(['Area_Name'])['Rape_Cases_Reported'].sum().reset_index())
            g1.columns = ['State/UT','Cases Reported']
            fig = px.bar(g1,x='State/UT',y='Cases Reported',color_discrete_sequence=['blue'])


            # In[13]:


            st.plotly_chart(fig)
            
        if ch=='age':
            st.text('     statistics based on age group')
            above_50 = data['Victims_Above_50_Yrs'].sum()
            ten_to_14 = data['Victims_Between_10-14_Yrs'].sum()
            fourteen_to_18 = data['Victims_Between_14-18_Yrs'].sum()
            eighteen_to_30 = data['Victims_Between_18-30_Yrs'].sum()
            thirty_to_50 = data['Victims_Between_30-50_Yrs'].sum()
            upto_10 = data['Victims_Upto_10_Yrs'].sum()

            age_grp = ['Upto 10','10 to 14','14 to 18','18 to 30','30 to 50','Above 50']
            age_group_vals = [upto_10,ten_to_14,fourteen_to_18,eighteen_to_30,thirty_to_50,above_50]

            fig = go.Figure(data=[go.Pie(labels=age_grp, values=age_group_vals,sort=False,
                                        marker=dict(colors=px.colors.qualitative.G10),textfont_size=12)])

            #fig.show()
            st.plotly_chart(fig)
    elif option=='Property theft':
        ch2=st.radio('select criteria',('state-wise','stolen/recovered'))
        if ch2=='state-wise':
            g2 = pd.DataFrame(data1.groupby(['Area_Name'])['Cases_Property_Stolen'].sum().reset_index())
            g2.columns = ['State/UT','Cases Reported']
            st.text('State-wise Property Stolen-Cases Reported',)
            fig = px.bar(g2,x='State/UT',y='Cases Reported',color_discrete_sequence=['green'])
            st.plotly_chart(fig)
        elif ch2=='stolen/recovered':
            prop_theft_recovered = data1['Cases_Property_Recovered'].sum()
            prop_theft_stolen = data1['Cases_Property_Stolen'].sum()

            prop_group = ['Property Stolen Cases','Property Recovered Cases']
            prop_vals = [prop_theft_stolen,prop_theft_recovered]

            colors = ['red','green']

            fig = go.Figure(data=[go.Pie(labels=prop_group, values=prop_vals,sort=False,
                                        marker=dict(colors=colors),textfont_size=12)])

            st.plotly_chart(fig)
    elif option=='human rights violation':
        g3 = pd.DataFrame(data2.groupby(['Year'])['Cases_Registered_under_Human_Rights_Violations'].sum().reset_index())
        g3.columns = ['Year','Cases Registered']

        fig = px.bar(g3,x='Year',y='Cases Registered',color_discrete_sequence=['purple'])
        st.plotly_chart(fig)





elif choice=='About App':
    st.subheader("Initiated for crime free India")
    #image=Image.open('C:/Users/dell/Desktop/main project/stop crime image.jfif')
    #st.image(image,use_column_width=True)



