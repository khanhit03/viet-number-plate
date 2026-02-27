# app.py - Giao diện web
import streamlit as st
from detector import detect_plate
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Phát hiện Xe Việt Nam", layout="centered")
st.title("Phát hiện Xe (Dự án AI đơn giản cho beginner)")

st.write("Upload ảnh xe máy/ô tô Việt Nam để AI phát hiện!")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh từ upload
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Lưu tạm
    temp_path = "temp_upload.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    with st.spinner("AI đang phân tích ảnh..."):
        result_img, message = detect_plate(temp_path)

    st.subheader("Kết quả:")
    st.write(message)

    # Hiển thị ảnh gốc và ảnh kết quả (đã sửa use_column_width)
    col1, col2 = st.columns(2)
    col1.image(image, caption="Ảnh gốc", use_column_width=None)
    if result_img is not None:
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        col2.image(result_rgb, caption="Ảnh phát hiện", use_column_width=None)