import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os
import time

st.set_page_config(page_title="Money Recognition", layout="wide")

# Custom CSS
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
    .success-box {border: 2px solid #2e7d32; padding: 15px; background-color: #e8f5e9; text-align: center;}
    .error-box {border: 2px solid #c62828; padding: 15px; background-color: #ffebee; text-align: center;}
    .warning-box {border: 2px solid #ed6c02; padding: 15px; background-color: #fff4e5; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">> vietnamese money recognition<span class="blinking-cursor">_</span></div>', unsafe_allow_html=True)

# ========== 1. KIEM TRA FILE MODEL ==========
MODEL_FILE = "vietnamese_money.onnx"

st.write("### 🔍 Kiem tra model")

if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Khong tim thay file: {MODEL_FILE}")
    st.write("**Danh sach file trong thu muc:**")
    for f in os.listdir('.'):
        st.write(f"   - {f}")
    st.stop()
else:
    file_size = os.path.getsize(MODEL_FILE) / 1024 / 1024
    st.success(f"✅ Tim thay file: {MODEL_FILE} ({file_size:.2f} MB)")

# ========== 2. LOAD MODEL ONNX ==========
st.write("### 🔄 Load model")

try:
    session = ort.InferenceSession(MODEL_FILE)
    st.success("✅ Model loaded thanh cong!")
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    st.write(f"**Input:** {input_info.name} | shape: {input_info.shape} | type: {input_info.type}")
    st.write(f"**Output:** {output_info.name} | shape: {output_info.shape} | type: {output_info.type}")
    
    target_size = (input_info.shape[1], input_info.shape[2])
    num_classes = output_info.shape[1]
    st.write(f"**Kich thuoc anh:** {target_size[0]}x{target_size[1]}")
    st.write(f"**So luong class:** {num_classes}")
    
except Exception as e:
    st.error(f"❌ Load model that bai: {e}")
    st.stop()

# ========== 3. CLASS NAMES ==========
CLASS_NAMES = ['010000', '020000', '050000', '100000', '200000', '500000']
DISPLAY_NAMES = ['10.000 dong', '20.000 dong', '50.000 dong', '100.000 dong', '200.000 dong', '500.000 dong']

MONEY_INFO = {
    '010000': {'color': 'Nau do', 'feature': 'Gieng Co Loa'},
    '020000': {'color': 'Xanh duong', 'feature': 'Cau The Huc'},
    '050000': {'color': 'Hong tim', 'feature': 'Hue'},
    '100000': {'color': 'Xanh la', 'feature': 'Van Mieu'},
    '200000': {'color': 'Do nau', 'feature': 'Ha Long'},
    '500000': {'color': 'Xanh tim', 'feature': 'Nha tho Kim Lien'}
}

# ========== 4. TEST THU MODEL ==========
st.write("### 🧪 Test model voi dummy input")

try:
    dummy = np.random.randn(1, target_size[0], target_size[1], 3).astype(np.float32)
    test_output = session.run(None, {input_info.name: dummy})[0]
    st.success(f"✅ Test thanh cong! Output shape: {test_output.shape}")
    st.write(f"**Sample output:** {test_output[0][:3]}...")
except Exception as e:
    st.error(f"❌ Test that bai: {e}")

st.markdown("---")

# ========== 5. FUNCTION XU LY ANH ==========
def preprocess_image(img_data):
    """Xu ly anh chuan 100%"""
    # Doc anh
    img = Image.open(io.BytesIO(img_data))
    
    # Debug
    st.write(f"**Anh goc:** mode={img.mode}, size={img.size}")
    
    # Chuyen RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        st.write(f"**Sau convert:** mode={img.mode}")
    
    # Resize
    img = img.resize(target_size)
    st.write(f"**Sau resize:** size={img.size}")
    
    # Convert to array
    img_array = np.array(img).astype(np.float32)
    st.write(f"**Array shape:** {img_array.shape}, range: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    # Normalize
    img_array = img_array / 255.0
    st.write(f"**Sau normalize:** range: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    # Add batch
    img_array = np.expand_dims(img_array, axis=0)
    st.write(f"**Final shape:** {img_array.shape}")
    
    return img_array, img

# ========== 6. GIAO DIEN CHINH ==========
st.write("### 📸 Nhan dien tien")

camera_image = st.camera_input("", label_visibility="collapsed")

if camera_image is not None:
    # Lay du lieu anh
    bytes_data = camera_image.getvalue()
    
    # Hien thi anh
    img_preview = Image.open(io.BytesIO(bytes_data))
    st.image(img_preview, width=250, caption="Anh da chup")
    
    if st.button("> NHAN DIEN", type="primary", use_container_width=True):
        with st.spinner("Dang xu ly..."):
            # Xu ly anh
            img_array, img_processed = preprocess_image(bytes_data)
            
            # Du doan
            start_time = time.time()
            predictions = session.run(None, {input_info.name: img_array})[0][0]
            inference_time = time.time() - start_time
            
            st.write(f"**Thoi gian du doan:** {inference_time*1000:.2f} ms")
            
            # Hien thi xac suat tung class
            st.write("### 📊 Xac suat tung menh gia:")
            for i, (name, display) in enumerate(zip(CLASS_NAMES, DISPLAY_NAMES)):
                prob = float(predictions[i])
                st.progress(prob, text=f"{display}: {prob:.2%}")
            
            # Lay ket qua
            idx = np.argmax(predictions)
            confidence = float(predictions[idx])
            money_key = CLASS_NAMES[idx]
            money_name = DISPLAY_NAMES[idx]
            money_info = MONEY_INFO[money_key]
            
            # Hien thi ket qua
            st.markdown("---")
            
            if confidence > 0.8:
                box_class = "success-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <h2 style="color:#2e7d32; margin:0;">🎯 {money_name}</h2>
                    <p style="color:#2e7d32;">Do tin cay: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            elif confidence > 0.6:
                box_class = "warning-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <h2 style="color:#ed6c02; margin:0;">⚠️ {money_name}</h2>
                    <p style="color:#ed6c02;">Do tin cay: {confidence:.2%} (trung binh)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                box_class = "error-box"
                st.markdown(f"""
                <div class="{box_class}">
                    <h2 style="color:#c62828; margin:0;">❌ Khong xac dinh</h2>
                    <p style="color:#c62828;">Do tin cay qua thap ({confidence:.2%})</p>
                    <p>Vui long chup lai anh ro hon</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Thong tin chi tiet
            st.write("### 📋 Thong tin menh gia")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Mau sac:** {money_info['color']}")
                st.write(f"**Dac diem:** {money_info['feature']}")
            with col2:
                st.write(f"**Menh gia:** {money_name}")
                st.write(f"**Do tin cay:** {confidence:.2%}")
            
            # Top 3 du doan
            st.write("### 🏆 Top 3 du doan")
            top3_idx = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top3_idx, 1):
                prob = float(predictions[idx])
                name = DISPLAY_NAMES[idx]
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

st.markdown("---")
st.caption("> version 2.0 | vietnam money recognition cnn | debug mode")
