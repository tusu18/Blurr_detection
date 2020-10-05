import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import load
import cv2
from PIL import Image, ImageOps
os.environ["NUMEXPR_MAX_THREADS"]="16"
os.environ["NUMEXPR_NUM_THREADS"]="16"
st.set_option('deprecation.showfileUploaderEncoding', False)


def load_data(file):
    file=cv2.imread(file)
    file=cv2.resize(file,(120,120),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)
    svarx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5).var()
    smaxY=np.amax(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5))
    data=np.array([smaxY,svarx])
    data=data.reshape(1,-1)
    return data
model=load("blurr_class.joblib")
def predict(data):
    s=model.predict(data)
    return s
st.title("BLURR IMAGE DETECTION MODEL")
st.markdown("This application is made for image Blurr Detection")
st.markdown("![Alt Text](https://cnet1.cbsistatic.com/img/vIjS19RgmQrE_noolcMz-WkrANs=/1092x614/2019/05/31/a01d0905-3b69-45d8-92e1-c0a26dc7dec5/motion-blur.jpg)")
st.sidebar.title("A better Image blurr detection")
st.sidebar.markdown("Select from folder image")
st.markdown("Upload Image let me tell yor skill")
file=st.file_uploader("Please Upload an File",type=["jpg","jpeg","png"])  
if file is None:
    st.text("Add an Image so i can give some inference")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    data=load_data(file)
    prd=predict(data)
    if prd[0]==-1:
       st.write("Good work its a clear pic")
    else:
       st.write("Man hold your hands its blurr!!")        

