import sys
import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_architectures.unet_resnet50 import UNetResNet50
from src.model_architectures.swin_unet import SwinUNet 

# ==========================================
# CÁC HÀM LÕI (ĐỂ APP.PY GỌI SANG DÙNG)
# ==========================================
def preprocess_image(image_pil):
    """Tiền xử lý ảnh (Dùng chung cho cả Test và Web)"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transform(image_pil).unsqueeze(0)

def post_process_mask(predicted_mask):
    """Lọc nhiễu, giữ lại đốm bệnh lớn nhất"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(predicted_mask, connectivity=8)
    if num_labels <= 1: return predicted_mask
    largest_label = 1 + np.argmax(stats[1:, 4]) 
    cleaned_mask = np.zeros_like(predicted_mask)
    cleaned_mask[labels == largest_label] = 255
    return cleaned_mask

def predict_ensemble(img_tensor, model_unet, model_swin, device):
    """Thuật toán Max Voting Ensemble"""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prob_u = torch.sigmoid(model_unet(img_tensor))
        prob_s = torch.sigmoid(model_swin(img_tensor))
        
        ensemble_prob = torch.max(prob_u, prob_s)
        mean_conf = ensemble_prob[ensemble_prob > 0.1].mean().item() if (ensemble_prob > 0.1).any() else 0.0
        
        mask = (ensemble_prob > 0.4).float().squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
    return mask, mean_conf

def create_overlay(orig_np, mask_np):
    """Tạo lớp phủ màu xanh y tế"""
    colored_mask = np.zeros_like(orig_np)
    colored_mask[mask_np == 255] = [0, 255, 0] 
    return cv2.addWeighted(orig_np, 0.7, colored_mask, 0.3, 0)

# ==========================================
# CHẾ ĐỘ CHẠY ĐỘC LẬP (XUẤT BÁO CÁO TRỰC QUAN)
# ==========================================
def main():
    print("🚀 BẮT ĐẦU CHẾ ĐỘ XUẤT BÁO CÁO TRỰC QUAN (VISUAL REPORT)...")
    
    # Số lượng ảnh muốn xuất báo cáo
    NUM_SAMPLES = 5 
    
    report_dir = os.path.join(project_root, 'reports', 'ensemble')
    os.makedirs(report_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. NẠP MÔ HÌNH VÀ FIX NAN NHƯ BÌNH THƯỜNG
    print("⏳ Đang nạp hệ thống AI...")
    model_unet = UNetResNet50().to(device)
    chk_u = torch.load(os.path.join(project_root, 'models', 'best_unet_resnet50.pth'), map_location=device, weights_only=False)
    st_u = chk_u.get("model_state_dict", chk_u)
    
    clean_u = {}
    for k, v in st_u.items():
        new_k = k.replace("module.", "")
        if not new_k.startswith("net."): new_k = "net." + new_k
        if torch.isnan(v).any(): v = torch.nan_to_num(v, nan=1.0 if 'var' in k else 0.0)
        clean_u[new_k] = v
    model_unet.load_state_dict(clean_u, strict=False)
    model_unet.eval()

    model_swin = SwinUNet().to(device)
    chk_s = torch.load(os.path.join(project_root, 'models', 'best_swin_unet.pth'), map_location=device, weights_only=False)
    model_swin.load_state_dict(chk_s.get("model_state_dict", chk_s), strict=False)
    model_swin.eval()
    
    # 2. ĐỌC DỮ LIỆU & CHỌN NGẪU NHIÊN VÀI ẢNH
    test_json = os.path.join(project_root, 'data', 'processed', 'manifests', 'test.json')
    with open(test_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_list = data.get('items', []) if isinstance(data, dict) else data
    
    # Trộn ngẫu nhiên và lấy đúng NUM_SAMPLES ảnh
    import random
    random.shuffle(image_list)
    selected_images = image_list[:NUM_SAMPLES]
    
    print(f"\n📸 Đã chọn {NUM_SAMPLES} ảnh mẫu. Đang tạo báo cáo...")
    
    # 3. TIẾN HÀNH DỰ ĐOÁN VÀ VẼ BÁO CÁO
    for i, item in enumerate(selected_images):
        img_rel_path = item if isinstance(item, str) else (item.get('image_path') or item.get('image'))
        file_name = os.path.basename(img_rel_path)
        img_path = os.path.join(project_root, 'data', 'processed', 'images', 'test', file_name)
        
        if not os.path.exists(img_path) or not img_path.endswith('.npy'):
            continue
            
        # Tiền xử lý
        img_arr = np.load(img_path)
        if img_arr.shape[-1] == 3 or img_arr.shape[-1] == 1:
            img_arr = np.transpose(img_arr, (2, 0, 1))
        img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0)
        
        # Để vẽ báo cáo, cần chuyển mảng float đã normalize về ảnh RGB xem bằng mắt thường
        inv_tensor = img_tensor.clone().squeeze()
        for c, m, s in zip(range(3), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            inv_tensor[c] = inv_tensor[c] * s + m
        orig_img = (inv_tensor.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        
        # Dự đoán
        raw_mask, conf = predict_ensemble(img_tensor, model_unet, model_swin, device)
        final_mask = post_process_mask(raw_mask)
        
        # Chuyển mask trắng đen thành ảnh 3 kênh (RGB) để lát ghép chung
        mask_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
        
        # Tạo overlay
        overlay_img = create_overlay(orig_img, final_mask)
        
        # Ghép 3 ảnh thành 1 dải ngang (Orig | Mask | Overlay)
        combined = np.hstack((orig_img, mask_rgb, overlay_img))
        
        # Lưu file
        save_name = f"Sample_{i+1}_{file_name.replace('.npy', '.png')}"
        cv2.imwrite(os.path.join(report_dir, save_name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  + Đã lưu báo cáo: {save_name} (Tự tin: {conf*100:.1f}%)")

    print(f"\n✅ Xong! Hãy mở thư mục {report_dir} để xem các tấm ảnh ghép báo cáo.")

if __name__ == '__main__':
    main()