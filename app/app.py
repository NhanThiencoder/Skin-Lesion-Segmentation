import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import sys
import datetime
import tempfile
from fpdf import FPDF

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
# KHỞI TẠO AI VÀ CÁC HÀM XỬ LÝ PDF
# ==========================================
@st.cache_resource
def init_ai_system():
    unet_path = os.path.join(project_root, 'models', 'best_unet_resnet50.pth')
    swin_path = os.path.join(project_root, 'models', 'best_swin_unet.pth')
    device = get_device()
    return load_unet(unet_path, device), load_swin(swin_path, device), device

def get_next_pdf_filename(base_dir, patient_id):
    """Đếm số file trong folder để tạo tên tăng dần bằng tiếng Anh"""
    os.makedirs(base_dir, exist_ok=True)
    existing_pdfs = [f for f in os.listdir(base_dir) if f.lower().endswith('.pdf')]
    next_idx = len(existing_pdfs) + 1
    
    safe_patient_id = "".join([c for c in patient_id if c.isalnum() or c in ('-', '_')]).strip()
    file_name = f"Diagnostic_Report_{next_idx:04d}_{safe_patient_id}.pdf"
    
    return file_name, os.path.join(base_dir, file_name)

def generate_pdf_report(patient_id, results_data, save_path):
    """Sinh file PDF chứa thông tin và hình ảnh kết quả (Tiếng Anh)"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Tiêu đề báo cáo
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, f"AI DIAGNOSTIC REPORT - PATIENT ID: {patient_id.upper()}", ln=True, align='C')
    
    # Thời gian
    pdf.set_font("Helvetica", '', 12)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f"Analysis Time: {current_time}", ln=True, align='C')
    pdf.ln(5)

    for res in results_data:
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, f"Image File: {res['image_name']}", ln=True)

        pdf.set_font("Helvetica", '', 12)
        status = "Lesion Detected (Abnormal)" if res['has_lesion'] else "No Lesion Detected (Normal)"
        pdf.cell(0, 8, f"Status: {status}", ln=True)
        if res['has_lesion']:
            pdf.cell(0, 8, f"AI Confidence Score: {res['confidence']*100:.1f}%", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
            Image.fromarray(res['orig_np']).save(tmp_orig, format="JPEG")
            orig_path = tmp_orig.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_overlay:
            Image.fromarray(res['overlay_img']).save(tmp_overlay, format="JPEG")
            overlay_path = tmp_overlay.name

        y_before_img = pdf.get_y() + 5
        
        # Thêm text chú thích cho ảnh gốc
        pdf.set_font("Helvetica", 'I', 10)
        pdf.text(x=10, y=y_before_img - 2, txt="Original Image")
        pdf.image(orig_path, x=10, y=y_before_img, w=90)
        
        # Thêm text chú thích cho ảnh mặt nạ
        pdf.text(x=110, y=y_before_img - 2, txt="AI Segmentation Overlay")
        pdf.image(overlay_path, x=110, y=y_before_img, w=90)
        
        pdf.set_y(y_before_img + 95)
        pdf.ln(5)

        os.remove(orig_path)
        os.remove(overlay_path)

    pdf.output(save_path)

# ==========================================
# HỆ THỐNG ĐIỀU PHỐI (MAIN)
# ==========================================
def main():
    st.markdown("<div class='hospital-header'><h1>🏥Hệ Thống Hội Chẩn Trí Tuệ Nhân Tạo Ensemble AI</h1></div>", unsafe_allow_html=True)
    model_unet, model_swin, device = init_ai_system()
    col_sidebar, col_main = st.columns([1, 3])
    
    with col_sidebar:
        st.markdown("<div class='patient-card'><b>👤 THÔNG TIN BỆNH NHÂN</b></div>", unsafe_allow_html=True)
        patient_id = st.text_input("Mã Bệnh Án (*Bắt buộc):", placeholder="VD: BN-2026")
        uploaded_files = st.file_uploader("Tải ảnh lâm sàng (*Bắt buộc, hỗ trợ nhiều ảnh):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        st.write("") 
        start_button = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN", use_container_width=True)
    
    with col_main:
        if not start_button:
            st.info("👈 Vui lòng nhập Mã Bệnh Án, tải ảnh lên ở cột bên trái và bấm nút 'Bắt đầu chẩn đoán'.")
        
        if start_button:
            if not patient_id.strip():
                st.error("⚠️ Lỗi: Vui lòng nhập Mã Bệnh Án trước khi tiến hành hội chẩn!")
            elif not uploaded_files:
                st.error("⚠️ Lỗi: Vui lòng tải ít nhất 1 ảnh lâm sàng lên trước khi tiến hành hội chẩn!")
            else:
                st.subheader(f"🖥️ KẾT QUẢ PHÂN TÍCH HÌNH ẢNH (Bệnh án: {patient_id.strip().upper()})")
                
                tabs = st.tabs([f"Ảnh {i+1}" for i in range(len(uploaded_files))])
                
                analysis_results = []
                
                for i, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
                    with tab:
                        image_pil = Image.open(uploaded_file).convert('RGB')
                        orig_width, orig_height = image_pil.size
                        orig_np = np.array(image_pil)
                        
                        with st.spinner(f"🤖 Đang phân tích ảnh {i+1}..."):
                            img_tensor = preprocess_image(image_pil)
                            raw_mask, confidence = predict_ensemble(img_tensor, model_unet, model_swin, device)
                            
                            final_mask = post_process_mask(raw_mask)
                            final_mask_resized = cv2.resize(final_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                            overlay_img = create_overlay(orig_np, final_mask_resized)
                        
                        has_lesion = (final_mask_resized == 255).any()
                        
                        analysis_results.append({
                            "image_name": uploaded_file.name,
                            "orig_np": orig_np,
                            "overlay_img": overlay_img,
                            "confidence": confidence,
                            "has_lesion": has_lesion
                        })
                        
                        if has_lesion:
                            st.metric(label="🎯 Độ tự tin AI phát hiện tổn thương", value=f"{confidence*100:.1f}%", delta="Bất thường", delta_color="inverse")
                        else:
                            st.metric(label="🎯 Trạng thái", value="Bình thường", delta="Không phát hiện tổn thương", delta_color="normal")
                            st.success("✅ AI nhận định vùng da bình thường, không có dấu hiệu bệnh lý nổi bật.")

                        img_col1, img_col2, img_col3 = st.columns(3)
                        with img_col1:
                            st.markdown("##### 1. Ảnh Gốc")
                            st.image(image_pil, use_container_width=True)
                        with img_col2:
                            st.markdown("##### 2. Mặt Nạ AI")
                            if has_lesion:
                                st.image(final_mask_resized, use_container_width=True, clamp=True)
                            else:
                                st.info("Không có mask tổn thương")
                        with img_col3:
                            st.markdown("##### 3. Bản Đồ Khoanh Vùng")
                            if has_lesion:
                                st.image(overlay_img, use_container_width=True)
                            else:
                                st.image(image_pil, use_container_width=True)

                # ===== PHẦN XUẤT PDF =====
                st.markdown("---")
                pdf_dir = os.path.join(project_root, 'reports', 'pdf_report')
                pdf_filename, pdf_filepath = get_next_pdf_filename(pdf_dir, patient_id)
                
                generate_pdf_report(patient_id, analysis_results, pdf_filepath)
                
                st.success(f"🖨️ Đã tự động tạo báo cáo: **{pdf_filename}**")
                
                with open(pdf_filepath, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Tải Báo Cáo Chẩn Đoán (PDF)",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()