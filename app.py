import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
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
    
    .stApp {
        background-color: #ffffff;
    }
    
    * {
        font-family: 'Courier New', 'SF Mono', monospace;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .blinking-cursor {
        animation: blink 1s step-end infinite;
        display: inline-block;
        width: 10px;
    }
    
    .main-title {
        color: #ff69b4;
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: normal;
    }
    
    .stButton > button {
        background: transparent;
        color: #ff69b4 !important;
        border: 1px solid #ff69b4 !important;
        border-radius: 0px !important;
        font-family: 'Courier New', monospace !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: #ff69b420 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #ff69b4;
    }
    
    .stCaption {
        color: #ff69b4;
    }
    
    hr {
        border-color: #ff69b450;
    }
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

# THU TU CLASS DUNG TU DATASET
CLASS_NAMES = ['10.000', '20.000', '50.000', '100.000', '200.000', '500.000']

# Thong tin chi tiet
MONEY_INFO = {
    '10.000': {'value': '10.000 dong', 'color': 'Nau do', 'feature': 'Hinh anh chu tich Ho Chi Minh, gieng Co Loa'},
    '20.000': {'value': '20.000 dong', 'color': 'Xanh duong', 'feature': 'Hinh anh chu tich Ho Chi Minh, cau The Huc'},
    '50.000': {'value': '50.000 dong', 'color': 'Hong tim', 'feature': 'Hinh anh chu tich Ho Chi Minh, Hue'},
    '100.000': {'value': '100.000 dong', 'color': 'Xanh la', 'feature': 'Hinh anh chu tich Ho Chi Minh, Van Mieu'},
    '200.000': {'value': '200.000 dong', 'color': 'Do nau', 'feature': 'Hinh anh chu tich Ho Chi Minh, Ha Long'},
    '500.000': {'value': '500.000 dong', 'color': 'Xanh tim', 'feature': 'Hinh anh chu tich Ho Chi Minh, nha tho Kim Lien'}
}

st.markdown("""
> nhan dien cac menh gia tien Viet Nam
> buoc 1: chup anh to tien
> buoc 2: nhan nut predict
> buoc 3: xem ket qua
""")

uploaded = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=280)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("> predict"):
        input_name = input_info.name
        predictions = session.run(None, {input_name: img_array})[0][0]
        
        idx = np.argmax(predictions)
        confidence = float(predictions[idx])
        money_key = CLASS_NAMES[idx]
        money = MONEY_INFO[money_key]
        
        st.markdown("---")
        st.markdown(f"### > {money['value']}")
        st.caption(f"confidence: {confidence:.2%}")
        
        st.markdown("---")
        st.markdown("> thong tin")
        st.write(f"menh gia: {money['value']}")
        st.write(f"mau sac: {money['color']}")
        st.write(f"dac diem: {money['feature']}")
        
        st.markdown("---")
        st.markdown("> top 5")
        top5_idx = np.argsort(predictions)[-5:][::-1]
        for i, idx in enumerate(top5_idx, 1):
            prob = float(predictions[idx])
            value = CLASS_NAMES[idx]
            st.progress(prob, text=f"{i}. {value} dong - {prob:.2%}")

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition cnn")
