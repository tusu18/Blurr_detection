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
@st.cache(persist=True)
def laplacesobel(gray):
    lmax=np.log(np.amax(cv2.Laplacian(gray,cv2.CV_64F)))
    scmaxY=np.log(np.amax(cv2.Scharr(gray,cv2.CV_64F,1,0)))
    return lmax,scmaxY
@st.cache(persist=True)
def get_user_input(image_data,model):    
    size=(120,120)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.resize(img, dsize=(120,120),    interpolation=cv2.INTER_CUBIC)
    smaxY,svarX=laplacesobel(gray)
    data=np.array([smaxY,svarX])
    data=data.reshape(1,-1)
    predict=model.predict(data)
    return predict
model=load("Blurr_Model_Logit_19b.joblib")
st.title("BLURR IMAGE DETECTION MODEL")
st.markdown("This application is made for image Blurr Detection")
st.markdown("![Alt Text](https://cnet1.cbsistatic.com/img/vIjS19RgmQrE_noolcMz-WkrANs=/614x614/2019/05/31/a01d0905-3b69-45d8-92e1-c0a26dc7dec5/motion-blur.jpg)")
st.sidebar.title("A better Image blurr detection")
select=st.sidebar.selectbox("Details",["Inference","Dataset"],key="1")
if not st.sidebar.checkbox("Hide",True):
    st.markdown("Inference")
    if select=="Inference":
        st.write("Now i have taken account all the edge features using different types of filter just inorder to find  \nline") 
	    st.write("a match using the basic and most used Laplace filter,Sobel,Scharr i got Quite Heteroskedasticity    \nline")
	    st.write("in my scatter plot which was result in vast difference in variance of the the two images      \nline")
	    st.write("So i ploted using all different features and the best result i got using laplace max values of images        \nline")
        st.write("and scharr y derivatives which i the fitted in LogisticReg model after trying several models at C=0.01, with uniform weight as data was imbalanced quite")
        st.write("https://github.com/tusu18/Blurr_detection")
    elif select=="Dataset":
        st.write("The Training Set consists of: \nline")
        st.write("630 undistorted (clear) images \nline")
		st.write("220 naturally-blurred images    \nline")
		st.write("150 artificially-distorted images     \nline")
        st.write("The Evaluation Set consists of two individual data sets :     \nline")
	    st.write("The Natural Blur Set which consists of:      \nline")
		st.write("589 undistorted (clear) images      \nline")
		st.write("411 naturally-blurred images         \nline")
		st.write("The Digital Blur Set                  \nline")
	 	st.write("30 undistorted (clear) images          \nline")
		st.write("450 artificially-blurred images          \nline")    
st.markdown("Upload Image let me tell your skill")
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
st.markdown("This Model has an accuracy of 86")        

