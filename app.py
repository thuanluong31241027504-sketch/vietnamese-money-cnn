import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
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
    * {font-family: 'Courier New', 'SF Mono', monospace;}
    @keyframes blink {0%,50%{opacity:1}51%,100%{opacity:0}}
    .blinking-cursor {animation: blink 1s step-end infinite; display: inline-block; width: 10px;}
    .main-title {color: #8B4513; font-size: 2rem; margin-bottom: 1rem; text-align: center;}
    .stButton > button {background: #8B4513 !important; color: #fffbe6 !important; border: 1px solid #8B4513 !important; border-radius: 0px !important; width: 100% !important;}
    .stButton > button:hover {background: #A0522D !important;}
    .stProgress > div > div > div {background-color: #8B4513;}
    .stCaption {color: #8B4513;}
    hr {border-color: #8B4513; opacity: 0.3;}
    .money-card {border: 1px solid #8B4513; padding: 12px; margin-bottom: 10px; background-color: #fff8e7;}
    .money-title {color: #8B4513; font-size: 1rem; font-weight: bold;}
    .money-desc {color: #333; font-size: 0.7rem; line-height: 1.5;}
    .result-box {border: 2px solid #8B4513; padding: 20px; margin-top: 20px; background-color: #fff8e7; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> vietnamese money recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

MODEL_FILE = "vietnamese_money.onnx"

# KIEM TRA FILE MODEL
if not os.path.exists(MODEL_FILE):
    st.error("> vietnamese_money.onnx not found")
    st.stop()

# LOAD MODEL ONNX
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_FILE)

session = load_model()

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

MONEY_INFO = {
    '010000': {'value': '10.000 dong', 'color': 'Nau do', 'size': '132 x 60 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, gieng Co Loa', 'release': '2006'},
    '020000': {'value': '20.000 dong', 'color': 'Xanh duong', 'size': '136 x 65 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, cau The Huc', 'release': '2006'},
    '050000': {'value': '50.000 dong', 'color': 'Hong tim', 'size': '140 x 65 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, Hue', 'release': '2003'},
    '100000': {'value': '100.000 dong', 'color': 'Xanh la', 'size': '144 x 65 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, Van Mieu', 'release': '2004'},
    '200000': {'value': '200.000 dong', 'color': 'Do nau', 'size': '148 x 65 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, Ha Long', 'release': '2006'},
    '500000': {'value': '500.000 dong', 'color': 'Xanh tim', 'size': '152 x 65 mm', 'material': 'Polymer', 'feature': 'Hinh anh chu tich Ho Chi Minh, nha tho Kim Lien', 'release': '2003'}
}

def preprocess_image(img):
    """Xu ly anh dung 100% giong khi train"""
    # Chuyen sang RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize ve 128x128
    img = img.resize(target_size)
    
    # Chuyen sang numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Normalize ve [0,1]
    img_array = img_array / 255.0
    
    # Them batch dimension
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
        
        if st.button("> nhan dien"):
            # Xu ly anh
            img_array = preprocess_image(img)
            
            # Debug info
            with st.expander("Debug thong tin anh"):
                st.write(f"Shape: {img_array.shape}")
                st.write(f"Min: {img_array.min():.4f}")
                st.write(f"Max: {img_array.max():.4f}")
                st.write(f"Mean: {img_array.mean():.4f}")
            
            # Du doan
            input_name = input_info.name
            predictions = session.run(None, {input_name: img_array})[0][0]
            
            # Hien thi xac suat tung class
            st.markdown("---")
            st.markdown("> xac suat tung menh gia")
            for i, name in enumerate(CLASS_NAMES):
                prob = float(predictions[i])
                st.progress(prob, text=f"{DISPLAY_NAMES[name]}: {prob:.2%}")
            
            idx = np.argmax(predictions)
            confidence = float(predictions[idx])
            money_key = CLASS_NAMES[idx]
            money = MONEY_INFO[money_key]
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#8B4513;">KET QUA</h3>
                <h2 style="color:#8B4513; font-size:1.5rem;">{money['value']}</h2>
                <p style="color:#666;">do tin cay: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("> chi tiet menh gia")
            st.write(f"**mau sac:** {money['color']}")
            st.write(f"**kich thuoc:** {money['size']}")
            st.write(f"**chat lieu:** {money['material']}")
            st.write(f"**dac diem:** {money['feature']}")
            st.write(f"**nam phat hanh:** {money['release']}")

with col_right:
    st.markdown("### > thu vien tien")
    st.caption("cac menh gia tien Viet Nam")
    
    for money_key, money in MONEY_INFO.items():
        with st.expander(f"> {money['value']}"):
            st.markdown(f"""
            <div class="money-card">
                <div class="money-title">{money['value']}</div>
                <div class="money-desc"><b>mau sac:</b> {money['color']}</div>
                <div class="money-desc"><b>kich thuoc:</b> {money['size']}</div>
                <div class="money-desc"><b>chat lieu:</b> {money['material']}</div>
                <div class="money-desc"><b>dac diem:</b> {money['feature']}</div>
                <div class="money-desc"><b>nam phat hanh:</b> {money['release']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition cnn")
