import os
import sys
import torch

# --- CẬP NHẬT ĐƯỜNG DẪN ---
# current_dir đang là thư mục app/
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lùi lại 1 cấp để ra thư mục gốc (Skin-Lesion-Segmentation)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import thư trúc mô hình từ thư mục src
from src.model_architectures.unet_resnet50 import UNetResNet50
from src.model_architectures.swin_unet import SwinUNet

def get_device():
    """Lấy thiết bị tính toán khả dụng"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_unet(weight_path, device):
    """Nạp U-Net và đặc trị lỗi NaN"""
    model = UNetResNet50().to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        if not new_k.startswith("net."): 
            new_k = "net." + new_k
            
        if torch.isnan(v).any():
            fill_value = 1.0 if 'var' in k else 0.0
            v = torch.nan_to_num(v, nan=fill_value)
            
        clean_dict[new_k] = v
        
    model.load_state_dict(clean_dict, strict=False)
    model.eval()
    return model

def load_swin(weight_path, device):
    """Nạp mô hình Swin-Unet"""
    model = SwinUNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model