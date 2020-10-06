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
st.title("BLUR IMAGE DETECTION MODEL")
st.markdown("This application is made for image Blur Detection")
st.markdown("![Alt Text](https://m-cdn.phonearena.com/images/article/114598-two_1600/Your-phone-is-taking-blurry-pictures-Heres-an-easy-fix.webp)")
st.sidebar.title("A better Image blur detection")
select=st.sidebar.selectbox("Details",["Inference","Dataset"],key="1")
if not st.sidebar.checkbox("Hide",True):
    if select=="Inference":
	st.markdown("Inference")
        st.write(
		"Now i have taken account all the edge features using different types of filter just inorder to finda match using the basic and most used Laplace filter Sobel Scharr\n" 
		"i got Quite Heteroskedasticity in my scatter plot which was result in vast difference in variance of the the two images So i ploted using all different features and the best result\n"
		"and scharr y derivatives which i the fitted in LogisticReg model after trying several models at C=0.01, with uniform weight as data was imbalanced quite\n"
		"https://github.com/tusu18/Blurr_detection"
	) 
    elif select=="Dataset":
	st.markdown("Dataset")
        st.write(
		"CERTH Image Blur Dataset!!!\n"
		"The Training Set consists of:\n"
		"630 undistorted (clear) images\n"
		"220 naturally-blurred images\n"
		"150 artificially-distorted images\n"
		"The Evaluation Set consists of two individual data sets :\n"
		"The Natural Blur Set which consists of:\n"
		"589 undistorted (clear) images\n"
		"411 naturally-blurred images\n"
		"The Digital Blur Set\n"
		"30 undistorted (clear) images\n"
		"450 artificially-blurred images."
	)
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
    st.write("Thanks for feedback this is used to improve model!")
if st.button('Wrong'):
    st.write("Sorry for the wrong ans will improve it thanks for feedback!")
st.markdown("This Model has an accuracy of 86 on eval set of CERT Dataset")  
st.markdown("https://github.com/tusu18/Blurr_detection")

