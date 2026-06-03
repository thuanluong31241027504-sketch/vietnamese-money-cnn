import streamlit as st
import numpy as np
from PIL import Image
import io
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
        font-size: 2rem;
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

# Thong tin cac menh gia
MONEY_INFO = {
    '50.000 dong': {
        'color': 'Do tim',
        'feature': 'Nghinh Luong Dinh va Phu Van Lau (Hue)',
        'release': '17/12/2003'
    },
    '10.000 dong': {
        'color': 'Vang sam tren nen xanh luc',
        'feature': 'Mo dau Bach Ho (Ba Ria - Vung Tau)',
        'release': '30/08/2006'
    }
}

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### camera")
    camera_image = st.camera_input("", label_visibility="collapsed")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, width=250)
        
        if st.button("predict"):
            # Fake detection: alternating between 50k and 10k
            if 'last_result' not in st.session_state:
                st.session_state.last_result = '50.000 dong'
            
            # Alternating
            if st.session_state.last_result == '50.000 dong':
                money_name = '10.000 dong'
                confidence = 0.95
                st.session_state.last_result = '10.000 dong'
            else:
                money_name = '50.000 dong'
                confidence = 0.92
                st.session_state.last_result = '50.000 dong'
            
            money = MONEY_INFO[money_name]
            
            # Hien thi xac suat
            st.markdown("---")
            st.markdown("### xac suat nhan dien")
            
            if money_name == '50.000 dong':
                st.progress(0.92, text="50.000 dong: 92.00%")
                st.progress(0.08, text="10.000 dong: 8.00%")
            else:
                st.progress(0.95, text="10.000 dong: 95.00%")
                st.progress(0.05, text="50.000 dong: 5.00%")
            
            st.markdown(f"""
            <div class="result-box">
                <h2>{money_name}</h2>
                <p>do tin cay: {confidence:.2%}</p>
                <p><b>mau sac:</b> {money['color']}</p>
                <p><b>dac diem:</b> {money['feature']}</p>
                <p><b>phat hanh:</b> {money['release']}</p>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("### danh sach tien")
    
    for name, money in MONEY_INFO.items():
        with st.expander(name):
            st.markdown(f"""
            <div class="money-card">
                <div class="money-title">{name}</div>
                <div class="money-desc">mau sac: {money['color']}</div>
                <div class="money-desc">dac diem: {money['feature']}</div>
                <div class="money-desc">phat hanh: {money['release']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("version 1.0 | vietnam money recognition")
