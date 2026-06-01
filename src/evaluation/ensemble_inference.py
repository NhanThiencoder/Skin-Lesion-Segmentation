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
# INFERENCE THRESHOLDS (tune for specificity)
# ==========================================
# Các ngưỡng dưới đây không cần train lại model.
# Tăng ngưỡng => giảm khoanh nhầm da lành, nhưng có thể bỏ sót tổn thương nhỏ/mờ.
FUSION_MODE = "mean"         # "mean" (khuyến nghị) | "min" (gắt hơn)

# Threshold chính để tạo mask từ map xác suất (sau fusion)
MASK_THRESHOLD = 0.62

# Ngưỡng diện tích tối thiểu của mask (trên ảnh 256x256)
MIN_AREA_RATIO = 0.0040      # 0.40% diện tích ~ 262 px

# Ngưỡng chắc chắn (sau fusion)
MIN_MEAN_PROB = 0.75         # mean prob trong vùng mask
MIN_MAX_PROB = 0.90          # max prob toàn ảnh

# Chốt "lõi" xác suất cao: nếu không có core đủ lớn => coi như không có tổn thương
CORE_THRESHOLD = 0.85
MIN_CORE_AREA_RATIO = 0.0007  # ~46 px

# Lọc bỏ CC nhỏ sau khi threshold
POST_MIN_AREA_RATIO = 0.0040


def _find_processed_dataset_root() -> str:
    """Find the processed dataset folder that contains manifests/images.

    Supports both layouts:
    - data/processed/manifests/*.json
    - data/processed/<dataset_name>/manifests/*.json
    """
    processed_root = os.path.join(project_root, 'data', 'processed')

    # Layout A: directly under processed/
    direct_manifest = os.path.join(processed_root, 'manifests', 'test.json')
    if os.path.exists(direct_manifest):
        return processed_root

    # Layout B: under a dataset subfolder
    if os.path.isdir(processed_root):
        for name in os.listdir(processed_root):
            candidate = os.path.join(processed_root, name)
            if not os.path.isdir(candidate):
                continue
            candidate_manifest = os.path.join(candidate, 'manifests', 'test.json')
            if os.path.exists(candidate_manifest):
                return candidate

    # Last resort: return base (will error with a clearer path)
    return processed_root

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
    """Lọc nhiễu, giữ lại đốm bệnh lớn nhất.

    Notes
    -----
    - Input/Output: mask uint8 với giá trị {0, 255}.
    - Nếu vùng lớn nhất quá nhỏ, trả về mask rỗng để tránh khoanh vùng da lành.
    """
    if predicted_mask is None:
        return predicted_mask

    if predicted_mask.dtype != np.uint8:
        predicted_mask = predicted_mask.astype(np.uint8)

    if predicted_mask.ndim == 3:
        predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_BGR2GRAY)

    # Nhị phân hoá để connected components ổn định
    bin_mask = (predicted_mask > 0).astype(np.uint8) * 255
    if not (bin_mask > 0).any():
        return bin_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return bin_mask

    largest_label = 1 + np.argmax(stats[1:, 4])
    largest_area = int(stats[largest_label, 4])

    # 256x256 => 65536 px. Mặc định 0.30% ~ 196px.
    min_area = max(128, int(POST_MIN_AREA_RATIO * bin_mask.shape[0] * bin_mask.shape[1]))
    if largest_area < min_area:
        return np.zeros_like(bin_mask)

    cleaned_mask = np.zeros_like(bin_mask)
    cleaned_mask[labels == largest_label] = 255
    return cleaned_mask

def predict_ensemble(img_tensor, model_unet, model_swin, device):
    """Thuật toán Ensemble (conservative).

    Có thêm cơ chế chặn false-positive: nếu vùng dự đoán quá nhỏ
    hoặc độ chắc chắn trung bình trong vùng quá thấp, trả về mask rỗng.
    """
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prob_u = torch.sigmoid(model_unet(img_tensor))
        prob_s = torch.sigmoid(model_swin(img_tensor))
        
        if FUSION_MODE == "min":
            ensemble_prob = torch.minimum(prob_u, prob_s)
        else:
            ensemble_prob = (prob_u + prob_s) / 2.0

        max_prob = ensemble_prob.max().item()
        if max_prob < MIN_MAX_PROB:
            empty = np.zeros(ensemble_prob.squeeze().shape, dtype=np.uint8)
            return empty, float(max_prob)

        core_mask = (ensemble_prob > CORE_THRESHOLD)
        core_area_ratio = core_mask.float().mean().item()
        if not core_mask.any() or core_area_ratio < MIN_CORE_AREA_RATIO:
            empty = np.zeros(ensemble_prob.squeeze().shape, dtype=np.uint8)
            return empty, float(max_prob)

        mask_bin = (ensemble_prob > MASK_THRESHOLD)
        area_ratio = mask_bin.float().mean().item()  # mask_bin shape (B,1,H,W)

        if not mask_bin.any() or area_ratio < MIN_AREA_RATIO:
            empty = np.zeros(ensemble_prob.squeeze().shape, dtype=np.uint8)
            return empty, 0.0

        mean_prob_in_mask = ensemble_prob[mask_bin].mean().item()
        if mean_prob_in_mask < MIN_MEAN_PROB:
            empty = np.zeros(ensemble_prob.squeeze().shape, dtype=np.uint8)
            return empty, float(mean_prob_in_mask)

        mask = mask_bin.float().squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

    return mask, float(mean_prob_in_mask)

def create_overlay(orig_np, mask_np):
    """Tạo lớp phủ màu xanh y tế"""
    if mask_np is None or not (mask_np == 255).any():
        return orig_np
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
    processed_ds_root = _find_processed_dataset_root()
    test_json = os.path.join(processed_ds_root, 'manifests', 'test.json')
    if not os.path.exists(test_json):
        raise FileNotFoundError(
            f"Không tìm thấy {test_json}. "
            f"Hãy kiểm tra thư mục data/processed hoặc manifests/test.json."
        )
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
        file_name = os.path.basename(img_rel_path) if img_rel_path else None

        # Try resolve path in a few common formats
        img_path = None
        if isinstance(img_rel_path, str) and img_rel_path:
            candidate = os.path.join(processed_ds_root, img_rel_path)
            if os.path.exists(candidate):
                img_path = candidate

        if img_path is None and file_name:
            candidate = os.path.join(processed_ds_root, 'images', 'test', file_name)
            if os.path.exists(candidate):
                img_path = candidate
        
        if not img_path or not os.path.exists(img_path) or not img_path.endswith('.npy'):
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