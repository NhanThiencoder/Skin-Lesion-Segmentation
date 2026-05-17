import cv2
import os
import zipfile
import shutil
import glob

def apply_dullrazor(image):
    """
    Hàm áp dụng thuật toán DullRazor để xóa lông trên ảnh tổn thương da.
    """
    # Bước 1: Chuyển đổi ảnh sang ảnh xám (Grayscale)
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bước 2: Tạo kernel hình chữ nhật kích thước 17x17 để quét các cấu trúc dạng sợi (lông)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Bước 3: Dùng phép toán hình thái học Blackhat để làm nổi bật các sợi lông sẫm màu
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # Bước 4: Tạo mặt nạ nhị phân (Mask) tách biệt sợi lông (trắng) và nền (đen)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Bước 5: Inpainting - Lấp đầy các phần bị che bởi lông bằng thuật toán Telea
    image_inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return image_inpainted

def pipeline_extract_and_process(zip_path, temp_extract_dir, final_output_dir):
    """
    Tự động giải nén file zip, xử lý DullRazor từng ảnh, lưu kết quả 
    và tự động nén thư mục kết quả cuối cùng thành file ZIP mới.
    """
    # Kiểm tra sự tồn tại của file ZIP đầu vào
    if not os.path.exists(zip_path):
        print(f"❌ Không tìm thấy file ZIP gốc tại đường dẫn: {zip_path}")
        print("Vui lòng đảm bảo file zip được đặt đúng vị trí trong thư mục data/processed/")
        return

    # 1. Tự động giải nén file ZIP đầu vào vào thư mục tạm thời
    print(f"📦 Đang giải nén tự động file nguồn: {os.path.basename(zip_path)}...")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir) 
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    print("✅ Giải nén file nguồn hoàn tất!")

    # 2. Tạo thư mục kết quả cuối cùng bên trong thư mục processed
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir)
    print(f"📁 Đang tạo thư mục lưu kết quả tại: {final_output_dir}")

    # 3. Quét tất cả các file ảnh trong thư mục tạm vừa giải nén
    image_paths = []
    for ext in ('/**/*.jpg', '/**/*.png', '/**/*.jpeg'):
        image_paths.extend(glob.glob(temp_extract_dir + ext, recursive=True))
        
    total_images = len(image_paths)
    if total_images == 0:
        print("⚠️ Không tìm thấy ảnh nào (.jpg, .png) trong file zip sau khi giải nén.")
        shutil.rmtree(temp_extract_dir)
        return

    print(f"🚀 Tìm thấy {total_images} ảnh. Bắt đầu chạy tiền xử lý DullRazor...")

    # 4. Duyệt qua từng ảnh để xử lý và lưu trực tiếp vào thư mục đích
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f" Lỗi không đọc được ảnh: {filename}")
            continue
            
        # Tẩy lông bằng DullRazor
        clean_img = apply_dullrazor(img)
        
        # Đường dẫn lưu ảnh sạch trực tiếp vào thư mục đích
        save_path = os.path.join(final_output_dir, filename)
        cv2.imwrite(save_path, clean_img)
        
        # In tiến độ xử lý
        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f" ⏳ Đã xử lý thành công: {idx + 1}/{total_images} ảnh")

    # 5. DỌN DẸP: Xóa thư mục giải nén tạm thời của file nguồn để tiết kiệm bộ nhớ
    print("🧹 Đang dọn dẹp các tệp giải nén tạm thời...")
    shutil.rmtree(temp_extract_dir)
    
    # 6. TIẾN HÀNH ZIP THƯ MỤC KẾT QUẢ CUỐI CÙNG
    print(f"🗜️ Đang tự động nén thư mục kết quả thành file ZIP...")
    shutil.make_archive(final_output_dir, 'zip', final_output_dir)
    print(f"🤐 Đã tạo file ZIP kết quả thành công!")

    # 7. DỌN DẸP: Xóa thư mục unzipped kết quả, chỉ giữ lại file ảnh sạch dạng .zip
    print("🧹 Đang xóa thư mục ảnh rời để tối ưu không gian đĩa (chỉ giữ lại file ZIP kết quả)...")
    shutil.rmtree(final_output_dir)

    print(f"\n✅ TẤT CẢ HOÀN TẤT VÀ THÀNH CÔNG ĐỒNG BỘ!")
    print(f" File ZIP ảnh sạch của cả nhóm nằm tại: {final_output_dir}.zip")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN TƯƠNG ĐỐI (Từ src/data_preprocessing/ lùi ra root) ---
    
    # Đường dẫn tới file ZIP chứa ảnh đã resize/chuẩn hóa của bạn
    ZIP_FILE_PATH = "../../data/processed/ISIC2018_256_imagenet.zip"
    
    # Thư mục giải nén tạm thời (sẽ tự động xóa sau khi chạy xong)
    TEMP_DIR = "../../data/processed/temp_extracted"
    
    # Đường dẫn thư mục kết quả cuối cùng
    FINAL_OUTPUT_DIR = "../../data/processed/ISIC2018_256_cleaned"
    
    # Thực thi hệ thống pipeline tự động
    pipeline_extract_and_process(ZIP_FILE_PATH, TEMP_DIR, FINAL_OUTPUT_DIR)