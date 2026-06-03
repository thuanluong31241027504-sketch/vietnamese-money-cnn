import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os

st.set_page_config(
    page_title="Flower Recognition",
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

st.markdown('<div class="main-title">> flower recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "flowerpro.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return ort.InferenceSession(MODEL_FILE)
    return None

session = load_model()

if session is None:
    st.error("> flowerpro.onnx not found")
    st.stop()

input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])

CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.markdown("""
> huong dan:
> 1. chup anh hoa
> 2. nhan nut predict
> 3. xem ket qua
""")

camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image is not None:
    bytes_data = camera_image.getvalue()
    img = Image.open(io.BytesIO(bytes_data))
    
    st.image(img, width=250)
    
    if st.button("> predict"):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        input_name = input_info.name
        predictions = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        flower_name = CLASS_NAMES[idx]
        
        st.markdown("---")
        st.markdown(f"### > {flower_name}")
        st.caption(f"do tin cay: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> top 3")
        top3_idx = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top3_idx, 1):
            prob = float(predictions[idx])
            name = CLASS_NAMES[idx]
            st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

st.markdown("---")
st.caption("> version 1.0 | flower recognition cnn")
