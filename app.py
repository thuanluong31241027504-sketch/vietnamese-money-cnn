import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(layout="wide")
st.title("Test thứ tự class")

session = ort.InferenceSession("vietnamese_money.onnx")
input_info = session.get_inputs()[0]
target_size = (input_info.shape[1], input_info.shape[2])

# Các cách sắp xếp class để thử
class_options = {
    "Cach 1": ['1000', '10000', '2000', '20000', '5000', '50000'],
    "Cach 2": ['1000', '2000', '5000', '10000', '20000', '50000'],
    "Cach 3": ['1000', '2000', '10000', '20000', '5000', '50000'],
    "Cach 4": ['1000', '5000', '10000', '20000', '2000', '50000'],
}

selected = st.selectbox("Chon cach sap xep class", list(class_options.keys()))
CLASS_NAMES = class_options[selected]

st.write(f"Dang dung: {CLASS_NAMES}")

uploaded = st.file_uploader("Chon anh tien", type=['jpg', 'png'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=200)
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        pred = session.run(None, {input_info.name: img_array})[0][0]
        idx = np.argmax(pred)
        
        st.write("### Chi tiet tung class:")
        for i, name in enumerate(CLASS_NAMES):
            st.progress(float(pred[i]), text=f"{name}: {pred[i]:.2%}")
        
        st.success(f"### Du doan: {CLASS_NAMES[idx]}")
