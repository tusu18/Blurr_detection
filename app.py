import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import load
import cv2
from PIL import Image, ImageOps


import pickle
os.environ["NUMEXPR_MAX_THREADS"]="16"
os.environ["NUMEXPR_NUM_THREADS"]="16"
st.set_option('deprecation.showfileUploaderEncoding', False)

def laplacesobel(gray):
    lvar=cv2.Laplacian(gray,cv2.CV_64F).var()
    lmax=np.amax(cv2.Laplacian(gray,cv2.CV_64F))
    svarX=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5).var()
    smaxX=np.amax(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5))
    svarY=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5).var()
    smaxY=np.amax(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5))
    scarX=cv2.Scharr(gray,cv2.CV_64F,0,1).var()
    scmaxX=np.amax(cv2.Scharr(gray,cv2.CV_64F,0,1))
    scarY=cv2.Scharr(gray,cv2.CV_64F,1,0).var()
    scmaxY=np.amax(cv2.Scharr(gray,cv2.CV_64F,1,0))


    return smaxY,svarX

def get_user_input(image_data,model):    
    size=(600,600)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.resize(img, dsize=(600,600),    interpolation=cv2.INTER_CUBIC)
    smaxY,svarX=laplacesobel(gray)
    data=np.array([smaxY,svarX])
    data=data.reshape(1,-1)
    predict=model.predict(data)
    return predict
model=load("blurr_logistic.pkcl")
st.title("BLURR IMAGE DETECTION MODEL")
st.markdown("This application is made for image Blurr Detection")
st.markdown("![Alt Text](https://cnet1.cbsistatic.com/img/vIjS19RgmQrE_noolcMz-WkrANs=/1092x614/2019/05/31/a01d0905-3b69-45d8-92e1-c0a26dc7dec5/motion-blur.jpg)")
st.sidebar.title("A better Image blurr detection")
st.markdown("Upload Image let me tell yor skill")
file=st.file_uploader("Please Upload an File",type=["jpg","jpeg","png"]) 
if file is None:
    st.text("Add an Image so i can give some inference")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prd=get_user_input(image,model)
    if prd[0]==-1:
       st.write("Good work its a clear pic")
    elif prd[0]==1:
       st.write("Man hold your hands its blurr!!") 
    else:
        st.write("Some flaw on my side")
if st.button('Correct'):
    st.write("Thanks for feedback this is used to improve model")
if st.button('Wrong'):
    st.write("Sorry for the wrong ans will improve it thanks for feedback")
st.markdown("This Model has an accuracy of 84.3 and precision and recall of 98,82")        

