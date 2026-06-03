import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2
import os

st.set_page_config(
    page_title="Money Recognition",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #ffffff;}
    * {font-family: 'Courier New', 'SF Mono', monospace;}
    @keyframes blink {0%,50%{opacity:1}51%,100%{opacity:0}}
    .blinking-cursor {animation: blink 1s step-end infinite; display: inline-block; width: 10px;}
    .main-title {color: #ff69b4; font-size: 2rem; margin-bottom: 1rem; font-weight: normal;}
    .stButton > button {background: transparent; color: #ff69b4 !important; border: 1px solid #ff69b4 !important; border-radius: 0px !important; width: 100% !important;}
    .stButton > button:hover {background: #ff69b420 !important;}
    .stProgress > div > div > div {background-color: #ff69b4;}
    .stCaption {color: #ff69b4;}
    hr {border-color: #ff69b450;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> vietnamese money recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "vietnamese_money.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return ort.InferenceSession(MODEL_FILE)
    return None

session = load_model()

if session is None:
    st.error("> vietnamese_money.onnx not found")
    st.stop()

input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])

CLASS_NAMES = ['010000', '020000', '050000', '100000', '200000', '500000']

DISPLAY_NAMES = {
    '010000': '10.000 dong',
    '020000': '20.000 dong',
    '050000': '50.000 dong',
    '100000': '100.000 dong',
    '200000': '200.000 dong',
    '500000': '500.000 dong'
}

# HAM XU LY ANH GIONG HET KHI TRAIN
def preprocess_image(img):
    # Chuyen sang RGB neu can
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize ve 128x128 (giong train)
    img = img.resize(target_size)
    
    # Chuyen sang numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Rescale 1.0/255 (giong train)
    img_array = img_array / 255.0
    
    # Them batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

st.markdown("""
> nhan dien menh gia tien Viet Nam
> buoc 1: chup anh to tien (de trong khung hinh)
> buoc 2: nhan nut predict
> buoc 3: xem ket qua
""")

# SU DUNG CAMERA TREN DIEN THOAI
camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image:
    # Hien thi anh da chup
    st.image(camera_image, width=280)
    
    if st.button("> predict"):
        # Xu ly anh giong het khi train
        img_array = preprocess_image(camera_image)
        
        # Du doan
        input_name = input_info.name
        predictions = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        money_key = CLASS_NAMES[idx]
        
        st.markdown("---")
        st.markdown(f"### > {DISPLAY_NAMES[money_key]}")
        st.caption(f"do tin cay: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> top 5")
        top5_idx = np.argsort(predictions)[-5:][::-1]
        for i, idx in enumerate(top5_idx, 1):
            prob = float(predictions[idx])
            value = DISPLAY_NAMES[CLASS_NAMES[idx]]
            st.progress(prob, text=f"{i}. {value} - {prob:.2%}")
        
        if confidence < 0.6:
            st.warning("> do tin cay thap, vui long chup lai anh ro hon")

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition cnn")
