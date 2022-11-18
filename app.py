import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
plt = platform.system()
if plt == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath

#Title
st.title("CLASSIFIER OF TRANSPORTS")

file=st.file_uploader("Upload file!", type=['png','jpeg', 'jpg','gif','svg'])
model=load_learner("transports_model.pkl")
if file:
    st.image(file)
    img=PILImage.create(file)
    pred,pred_id, prob=model.predict(img)
    st.success(f"PREDICTION: {pred}")
    st.info(F"PROBABILITY: {prob[pred_id]*100:.1f}%")
    fig=px.bar(x=prob*100, y=model.dls.vocab)
    st.plotly_chart(fig)
