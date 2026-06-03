import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from collections import Counter

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

# Thông tin các mệnh giá
MONEY_INFO = {
    '10.000 dong': {
        'color': 'Vang sam tren nen xanh luc',
        'feature': 'Mo dau Bach Ho (Ba Ria - Vung Tau)',
        'release': '30/08/2006',
        'dominant': [100, 120, 80]  # mau vang xanh
    },
    '20.000 dong': {
        'color': 'Xanh lo',
        'feature': 'Chua Cau (Hoi An)',
        'release': '05/2006',
        'dominant': [80, 150, 200]  # mau xanh lo
    },
    '50.000 dong': {
        'color': 'Do tim',
        'feature': 'Nghinh Luong Dinh va Phu Van Lau (Hue)',
        'release': '17/12/2003',
        'dominant': [200, 80, 150]  # mau do tim
    },
    '100.000 dong': {
        'color': 'Xanh la cay',
        'feature': 'Van Mieu - Quoc Tu Giam (Ha Noi)',
        'release': '01/09/2004',
        'dominant': [50, 150, 50]  # mau xanh la
    },
    '200.000 dong': {
        'color': 'Do nau',
        'feature': 'Hon Dinh Huong tren vinh Ha Long',
        'release': '30/08/2006',
        'dominant': [150, 80, 60]  # mau do nau
    },
    '500.000 dong': {
        'color': 'Xanh lo sam',
        'feature': 'Nha Chu tich Ho Chi Minh tai lang Sen (Nghe An)',
        'release': '17/12/2003',
        'dominant': [60, 100, 150]  # mau xanh sam
    }
}

def get_dominant_color(img):
    """Lay mau chu dao cua anh"""
    # Resize nho de tinh nhanh
    img_small = img.resize((50, 50))
    img_array = np.array(img_small)
    
    # Tinh trung binh mau RGB
    avg_color = np.mean(img_array, axis=(0, 1))
    return avg_color

def detect_money_by_color(img):
    """Nhan dien menh gia dua tren mau sac"""
    dominant = get_dominant_color(img)
    
    # Tinh khoang cach den tung menh gia
    distances = {}
    for name, info in MONEY_INFO.items():
        target = info['dominant']
        distance = np.sqrt(np.sum((dominant - target)**2))
        distances[name] = distance
    
    # Chon menh gia co khoang cach nho nhat
    best_match = min(distances, key=distances.get)
    confidence = 1.0 - (distances[best_match] / 500)  # Chuan hoa do tin cay
    
    # Gioi han confidence
    confidence = max(0.3, min(0.98, confidence))
    
    return best_match, confidence, distances

col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### camera")
    camera_image = st.camera_input("", label_visibility="collapsed")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        st.image(img, width=250)
        
        if st.button("predict"):
            # Nhan dien bang mau sac
            money_name, confidence, distances = detect_money_by_color(img)
            money = MONEY_INFO[money_name]
            
            # Hien thi xac suat tat ca menh gia
            st.markdown("---")
            st.markdown("### xac suat nhan dien")
            
            # Sap xep theo distance (gan nhat den xa nhat)
            sorted_money = sorted(distances.items(), key=lambda x: x[1])
            for name, dist in sorted_money:
                prob = 1.0 - (dist / 500)
                prob = max(0.05, min(0.98, prob))
                st.progress(prob, text=f"{name}: {prob:.2%}")
            
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
st.caption("version 1.0 | vietnam money recognition | fake mode")
