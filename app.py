import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import random

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
    .main-title {
        color: #8B4513;
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: #8B4513 !important;
        color: #fffbe6 !important;
        border: none !important;
        border-radius: 0px !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    .stButton > button:hover {
        background: #A0522D !important;
    }
    .stProgress > div > div > div {
        background-color: #8B4513;
    }
    hr {
        border-color: #8B4513;
        opacity: 0.3;
    }
    .result-box {
        border: 2px solid #8B4513;
        padding: 15px;
        margin-top: 15px;
        background-color: #fff8e7;
        text-align: center;
    }
    .result-box h2 {
        color: #8B4513;
        font-size: 1.5rem;
        margin: 0;
    }
    .money-card {
        border: 1px solid #8B4513;
        padding: 10px;
        margin-bottom: 8px;
        background-color: #fff8e7;
    }
    .money-title {
        color: #8B4513;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .money-desc {
        color: #333;
        font-size: 0.65rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">vietnamese money recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "vietnamese_money.onnx"

# Fake model - chỉ để demo
class FakeSession:
    def __init__(self):
        self.counter = 0
    
    def get_inputs(self):
        return [type('obj', (object,), {'name': 'input', 'shape': [1, 128, 128, 3]})()]
    
    def run(self, output_names, input_feed):
        self.counter += 1
        # Luân phiên giữa 50k và 10k
        if self.counter % 2 == 1:
            # 50.000 dong
            return [np.array([[0.01, 0.01, 0.92, 0.03, 0.02, 0.01]], dtype=np.float32)]
        else:
            # 10.000 dong
            return [np.array([[0.92, 0.02, 0.01, 0.02, 0.02, 0.01]], dtype=np.float32)]

# Fake model
session = FakeSession()

input_info = session.get_inputs()[0]
input_shape = input_info.shape
target_size = (input_shape[1], input_shape[2])

CLASS_NAMES = ['010000', '020000', '050000', '100000', '200000', '500000']
DISPLAY_NAMES = ['10.000 dong', '20.000 dong', '50.000 dong', '100.000 dong', '200.000 dong', '500.000 dong']

MONEY_INFO = {
    '10.000 dong': {
        'color': 'Vang sam tren nen xanh luc',
        'feature': 'Mo dau Bach Ho (Ba Ria - Vung Tau)',
        'release': '30/08/2006'
    },
    '20.000 dong': {
        'color': 'Xanh lo',
        'feature': 'Chua Cau (Hoi An)',
        'release': '05/2006'
    },
    '50.000 dong': {
        'color': 'Do tim',
        'feature': 'Nghinh Luong Dinh va Phu Van Lau (Hue)',
        'release': '17/12/2003'
    },
    '100.000 dong': {
        'color': 'Xanh la cay',
        'feature': 'Van Mieu - Quoc Tu Giam (Ha Noi)',
        'release': '01/09/2004'
    },
    '200.000 dong': {
        'color': 'Do nau',
        'feature': 'Hon Dinh Huong tren vinh Ha Long',
        'release': '30/08/2006'
    },
    '500.000 dong': {
        'color': 'Xanh lo sam',
        'feature': 'Nha Chu tich Ho Chi Minh tai lang Sen (Nghe An)',
        'release': '17/12/2003'
    }
}

def preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### camera")
    camera_image = st.camera_input("", label_visibility="collapsed")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, width=250, caption="anh da chup")
        
        if st.button("predict"):
            img_array = preprocess_image(img)
            predictions = session.run(None, {input_info.name: img_array})[0][0]
            
            st.markdown("---")
            st.markdown("> xac suat tung menh gia")
            for i, name in enumerate(DISPLAY_NAMES):
                prob = float(predictions[i])
                st.progress(prob, text=f"{name}: {prob:.2%}")
            
            idx = np.argmax(predictions)
            confidence = float(predictions[idx])
            money_name = DISPLAY_NAMES[idx]
            money = MONEY_INFO[money_name]
            
            st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#8B4513;">{money_name}</h2>
                <p>do tin cay: {confidence:.2%}</p>
                <hr>
                <p><b>mau sac:</b> {money['color']}</p>
                <p><b>dac diem:</b> {money['feature']}</p>
                <p><b>ngay phat hanh:</b> {money['release']}</p>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("### thu vien tien")
    
    for name, money in MONEY_INFO.items():
        with st.expander(f"> {name}"):
            st.markdown(f"""
            <div class="money-card">
                <div class="money-title">{name}</div>
                <div class="money-desc"><b>mau sac:</b> {money['color']}</div>
                <div class="money-desc"><b>dac diem:</b> {money['feature']}</div>
                <div class="money-desc"><b>ngay phat hanh:</b> {money['release']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("version 1.0 | vietnam money recognition cnn")
