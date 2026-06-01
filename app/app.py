import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import sys

# Cập nhật đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_loader import get_device, load_unet, load_swin
from src.evaluation.ensemble_inference import preprocess_image, post_process_mask, predict_ensemble, create_overlay

# ==========================================
# CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="AI Da Liễu", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .hospital-header { background-color: #005b96; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px; }
    .patient-card { background-color: #f1f7fd; padding: 15px; border-left: 5px solid #03396c; border-radius: 5px; margin-bottom: 20px; color: #03396c; }
    .stButton>button { background-color: #d9534f; color: white; font-weight: bold; height: 50px; border-radius: 8px;}
    .stButton>button:hover { background-color: #c9302c; color: white; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# KHỞI TẠO AI
# ==========================================
@st.cache_resource
def init_ai_system():
    unet_path = os.path.join(project_root, 'models', 'best_unet_resnet50.pth')
    swin_path = os.path.join(project_root, 'models', 'best_swin_unet.pth')
    device = get_device()
    return load_unet(unet_path, device), load_swin(swin_path, device), device

# ==========================================
# HỆ THỐNG ĐIỀU PHỐI (MAIN)
# ==========================================
def main():
    st.markdown("<div class='hospital-header'><h1>🏥 BỆNH VIỆN DA LIỄU TRUNG ƯƠNG</h1><h4>Hệ Thống Hội Chẩn Trí Tuệ Nhân Tạo (Ensemble AI)</h4></div>", unsafe_allow_html=True)
    
    model_unet, model_swin, device = init_ai_system()
    col_sidebar, col_main = st.columns([1, 3])
    
    with col_sidebar:
        st.markdown("<div class='patient-card'><b>👤 THÔNG TIN BỆNH NHÂN</b></div>", unsafe_allow_html=True)
        patient_id = st.text_input("Mã Bệnh Án (*Bắt buộc):", placeholder="VD: BN-2026")
        uploaded_file = st.file_uploader("Tải ảnh lâm sàng (*Bắt buộc):", type=["jpg", "jpeg", "png"])
        
        st.write("") 
        start_button = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN", use_container_width=True)
    
    with col_main:
        if not start_button:
            st.info("👈 Vui lòng nhập Mã Bệnh Án, tải ảnh lên ở cột bên trái và bấm nút 'Bắt đầu chẩn đoán'.")
        
        if start_button:
            if not patient_id.strip():
                st.error("⚠️ Lỗi: Vui lòng nhập Mã Bệnh Án trước khi tiến hành hội chẩn!")
            elif uploaded_file is None:
                st.error("⚠️ Lỗi: Vui lòng tải ảnh lâm sàng lên trước khi tiến hành hội chẩn!")
            else:
                st.subheader(f"🖥️ KẾT QUẢ PHÂN TÍCH HÌNH ẢNH (Bệnh án: {patient_id.strip().upper()})")
                
                image_pil = Image.open(uploaded_file).convert('RGB')
                orig_width, orig_height = image_pil.size
                orig_np = np.array(image_pil)
                
                with st.spinner("🤖 Đang tiến hành hội chẩn. Vui lòng đợi..."):
                    img_tensor = preprocess_image(image_pil)
                    
                    # NHẬN 2 BIẾN NHƯ CŨ
                    raw_mask, confidence = predict_ensemble(img_tensor, model_unet, model_swin, device)
                    
                    final_mask = post_process_mask(raw_mask)
                    final_mask_resized = cv2.resize(final_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                    overlay_img = create_overlay(orig_np, final_mask_resized)
                    
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.markdown("**1. Ảnh Gốc**")
                    st.image(image_pil, use_container_width=True)
                with img_col2:
                    st.markdown("**2. Mặt Nạ AI**")
                    st.image(final_mask_resized, use_container_width=True, clamp=True)
                with img_col3:
                    st.markdown("**3. Bản Đồ Khoanh Vùng**")
                    st.image(overlay_img, use_container_width=True)
                    
                # CHỈ HIỂN THỊ 1 DÒNG KẾT LUẬN DUY NHẤT
                st.success(f"✅ Báo cáo AI: Độ tự tin hội chẩn tổng hợp đạt **{confidence*100:.1f}%**.")

if __name__ == "__main__":
    main()