import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
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
    .stApp {background-color: #fffbe6;}
    * {font-family: 'Courier New', monospace;}
    @keyframes blink {0%,50%{opacity:1}51%,100%{opacity:0}}
    .blinking-cursor {animation: blink 1s step-end infinite; display: inline-block; width: 10px;}
    .main-title {color: #8B4513; font-size: 2rem; text-align: center;}
    .stButton > button {background: #8B4513 !important; color: #fffbe6 !important; border: none !important; width: 100% !important;}
    .stButton > button:hover {background: #A0522D !important;}
    .stProgress > div > div > div {background-color: #8B4513;}
    hr {border-color: #8B4513; opacity: 0.3;}
    .result-box {border: 2px solid #8B4513; padding: 20px; margin-top: 20px; background-color: #fff8e7; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> vietnamese money recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "vietnamese_money.onnx"

@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_FILE)

if not os.path.exists(MODEL_FILE):
    st.error("> file not found")
    st.stop()

session = load_model()
input_info = session.get_inputs()[0]
target_size = (input_info.shape[1], input_info.shape[2])

CLASS_NAMES = ['10.000 dong', '20.000 dong', '50.000 dong', '100.000 dong', '200.000 dong', '500.000 dong']

camera = st.camera_input("", label_visibility="collapsed")

if camera:
    img = Image.open(io.BytesIO(camera.getvalue())).convert('RGB').resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("> predict"):
        pred = session.run(None, {input_info.name: img_array})[0][0]
        idx = np.argmax(pred)
        
        st.markdown(f"""
        <div class="result-box">
            <h2>{CLASS_NAMES[idx]}</h2>
            <p>do tin cay: {pred[idx]:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, name in enumerate(CLASS_NAMES):
            st.progress(float(pred[i]), text=f"{name}: {pred[i]:.2%}")
