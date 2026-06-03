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
    .main-title {color: #8B4513; font-size: 2rem; margin-bottom: 1rem; text-align: center;}
    .stButton > button {background: #8B4513 !important; color: #fffbe6 !important; border: none !important; border-radius: 0px !important; width: 100% !important;}
    .stButton > button:hover {background: #A0522D !important;}
    .stProgress > div > div > div {background-color: #8B4513;}
    hr {border-color: #8B4513; opacity: 0.3;}
    .result-box {border: 2px solid #8B4513; padding: 20px; margin-top: 20px; background-color: #fff8e7; text-align: center;}
    .money-card {border: 1px solid #8B4513; padding: 12px; margin-bottom: 10px; background-color: #fff8e7;}
    .money-title {color: #8B4513; font-size: 1rem; font-weight: bold;}
    .money-desc {color: #333; font-size: 0.7rem; line-height: 1.5;}
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
DISPLAY_NAMES = ['10.000 dong', '20.000 dong', '50.000 dong', '100.000 dong', '200.000 dong', '500.000 dong']

# THONG TIN CHI TIET CHO 6 MENH GIA
MONEY_INFO = {
    '10.000 dong': {
        'color': 'Nau do',
        'size': '132 x 60 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, gieng Co Loa',
        'release': '2006',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Gieng Co Loa'
    },
    '20.000 dong': {
        'color': 'Xanh duong',
        'size': '136 x 65 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, cau The Huc',
        'release': '2006',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Cau The Huc'
    },
    '50.000 dong': {
        'color': 'Hong tim',
        'size': '140 x 65 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, Hue',
        'release': '2003',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Ngo mon - Hue'
    },
    '100.000 dong': {
        'color': 'Xanh la',
        'size': '144 x 65 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, Van Mieu',
        'release': '2004',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Van Mieu - Quoc Tu Giam'
    },
    '200.000 dong': {
        'color': 'Do nau',
        'size': '148 x 65 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, Ha Long',
        'release': '2006',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Vinh Ha Long'
    },
    '500.000 dong': {
        'color': 'Xanh tim',
        'size': '152 x 65 mm',
        'material': 'Polymer',
        'feature': 'Hinh anh chu tich Ho Chi Minh, nha tho Kim Lien',
        'release': '2003',
        'obverse': 'Chu tich Ho Chi Minh',
        'reverse': 'Nha tho Kim Lien'
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
    st.markdown("### > camera")
    camera_image = st.camera_input("", label_visibility="collapsed")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, width=250, caption="anh da chup")
        
        if st.button("> predict"):
            img_array = preprocess_image(img)
            input_name = input_info.name
            predictions = session.run(None, {input_name: img_array})[0][0]
            
            # Hien thi xac suat tung class
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
                <p><b>kich thuoc:</b> {money['size']}</p>
                <p><b>chat lieu:</b> {money['material']}</p>
                <p><b>dac diem:</b> {money['feature']}</p>
                <p><b>mat truoc:</b> {money['obverse']}</p>
                <p><b>mat sau:</b> {money['reverse']}</p>
                <p><b>nam phat hanh:</b> {money['release']}</p>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("### > thu vien tien")
    
    for name, money in MONEY_INFO.items():
        with st.expander(f"> {name}"):
            st.markdown(f"""
            <div class="money-card">
                <div class="money-title">{name}</div>
                <div class="money-desc"><b>mau sac:</b> {money['color']}</div>
                <div class="money-desc"><b>kich thuoc:</b> {money['size']}</div>
                <div class="money-desc"><b>chat lieu:</b> {money['material']}</div>
                <div class="money-desc"><b>dac diem:</b> {money['feature']}</div>
                <div class="money-desc"><b>nam phat hanh:</b> {money['release']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition cnn")
