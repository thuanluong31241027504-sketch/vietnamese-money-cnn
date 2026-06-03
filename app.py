import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort
import io
import os
import cv2
from io import BytesIO

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

def preprocess_image(img):
    """Xu ly anh giong 100% khi train"""
    # Chuyen sang RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Tang do tuong phan (giong augmentation khi train)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Tang do sac net
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    # Resize ve dung kich thuoc
    img = img.resize(target_size)
    
    # Chuyen sang array va normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Them batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

st.markdown("""
> huong dan nhan dien tien Viet Nam
> 1. chup anh to tien (de trong khung hinh)
> 2. nhan nut predict
> 3. xem ket qua
""")

camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = Image.open(io.BytesIO(bytes_data))
    
    # Hien thi anh goc
    st.image(img, width=250, caption="Anh da chup")
    
    if st.button("> predict"):
        # Xu ly anh giong 100% train
        img_array = preprocess_image(img)
        
        # Du doan
        input_name = input_info.name
        predictions = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        money_key = CLASS_NAMES[idx]
        
        st.markdown("---")
        st.markdown(f"### > {DISPLAY_NAMES[money_key]}")
        
        if confidence > 0.7:
            st.success(f"do tin cay: {confidence:.2%}")
        elif confidence > 0.5:
            st.warning(f"do tin cay: {confidence:.2%}")
        else:
            st.error(f"do tin cay thap: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> top 3 du doan")
        top3_idx = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top3_idx, 1):
            prob = float(predictions[idx])
            value = DISPLAY_NAMES[CLASS_NAMES[idx]]
            st.progress(prob, text=f"{i}. {value} - {prob:.2%}")

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition cnn")
