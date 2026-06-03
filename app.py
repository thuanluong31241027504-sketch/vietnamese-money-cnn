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
    .money-desc {color: #333; font-size: 0.7rem;}
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

MONEY_INFO = {
    '010000': {'value': '10.000 dong', 'color': 'Nau do', 'feature': 'Gieng Co Loa'},
    '020000': {'value': '20.000 dong', 'color': 'Xanh duong', 'feature': 'Cau The Huc'},
    '050000': {'value': '50.000 dong', 'color': 'Hong tim', 'feature': 'Hue'},
    '100000': {'value': '100.000 dong', 'color': 'Xanh la', 'feature': 'Van Mieu'},
    '200000': {'value': '200.000 dong', 'color': 'Do nau', 'feature': 'Ha Long'},
    '500000': {'value': '500.000 dong', 'color': 'Xanh tim', 'feature': 'Nha tho Kim Lien'}
}

# Layout 2 cot
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
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Du doan
            input_name = input_info.name
            predictions = session.run(None, {input_name: img_array})[0][0]
            
            idx = np.argmax(predictions)
            confidence = float(predictions[idx])
            money_key = CLASS_NAMES[idx]
            money = MONEY_INFO[money_key]
            
            st.markdown(f"""
            <div class="result-box">
                <h2 style="color:#8B4513;">{money['value']}</h2>
                <p>do tin cay: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("> chi tiet")
            st.write(f"**mau sac:** {money['color']}")
            st.write(f"**dac diem:** {money['feature']}")
            
            st.markdown("---")
            st.markdown("> top 3")
            top3_idx = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3_idx, 1):
                prob = float(predictions[idx])
                value = DISPLAY_NAMES[idx]
                st.progress(prob, text=f"{i}. {value} - {prob:.2%}")

with col_right:
    st.markdown("### > danh sach tien")
    
    for money_key, money in MONEY_INFO.items():
        with st.expander(f"> {money['value']}"):
            st.markdown(f"""
            <div class="money-card">
                <div class="money-title">{money['value']}</div>
                <div class="money-desc"><b>mau sac:</b> {money['color']}</div>
                <div class="money-desc"><b>dac diem:</b> {money['feature']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("> version 1.0 | vietnam money recognition")
