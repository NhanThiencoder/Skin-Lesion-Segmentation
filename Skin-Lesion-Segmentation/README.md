# Đồ án Deep Learning: Phân vùng tổn thương da (Skin Lesion Segmentation)

Dự án này tập trung vào việc xây dựng và đánh giá các mô hình Deep Learning để giải quyết bài toán phân vùng tổn thương da từ hình ảnh y tế, sử dụng bộ dữ liệu từ ISIC 2018 Challenge.

## Mục tiêu dự án

1.  **Tiền xử lý dữ liệu:** Áp dụng các kỹ thuật tăng cường và chuẩn hóa dữ liệu để cải thiện hiệu suất mô hình.
2.  **Xây dựng mô hình:**
    *   Sử dụng ít nhất một kiến trúc CNN kinh điển (VGG, ResNet, U-Net).
    *   Sử dụng ít nhất một kiến trúc có cơ chế Attention (Transformer, ViT, Attention U-Net).
3.  **Huấn luyện mô hình:**
    *   Áp dụng các phương pháp huấn luyện nâng cao như Transfer Learning hoặc Ensemble Learning.
    *   Tinh chỉnh tham số (Parameter Tuning) để tìm ra bộ tham số tối ưu.
4.  **Đánh giá mô hình:**
    *   Sử dụng các chỉ số đo lường phù hợp cho bài toán segmentation (ví dụ: Dice Coefficient, Jaccard Index/IoU, Accuracy).
    *   Trực quan hóa kết quả dự đoán của mô hình.
5.  **Triển khai ứng dụng:** Xây dựng một ứng dụng đơn giản (ví dụ: web app) để cho phép người dùng tải lên hình ảnh và nhận về kết quả phân vùng.

## Bộ dữ liệu

Dự án sử dụng bộ dữ liệu từ [ISIC Challenge 2018: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.isic-archive.com/data/#2018). Cụ thể là Task 1: Lesion Boundary Segmentation.

## Cấu trúc thư mục

```
Skin-Lesion-Segmentation/
├── data/                     # Chứa dữ liệu gốc và đã qua xử lý
├── models/                   # Lưu các mô hình đã huấn luyện (.pth, .h5)
├── notebooks/                # Chứa các file Jupyter Notebook để thử nghiệm, khám phá
├── src/                      # Mã nguồn chính của dự án
│   ├── data_preprocessing/   # Các scripts tiền xử lý dữ liệu
│   ├── model_architectures/  # Định nghĩa kiến trúc các mô hình
│   ├── training/             # Các scripts để huấn luyện mô hình
│   ├── evaluation/           # Các scripts để đánh giá mô hình
│   └── deployment/           # Mã nguồn cho ứng dụng triển khai
├── requirements.txt          # Các thư viện cần thiết
└── README.md                 # Mô tả dự án
```

## Thành viên nhóm

*   Nhân Thiện
*   Tấn Phát
*   Minh Đức
*   Quốc Bảo

## Cài đặt

1.  Clone repository:
    ```bash
    git clone https://github.com/NhanThiencoder/Skin-Lesion-Segmentation.git
    cd Skin-Lesion-Segmentation
    ```

2.  Tạo môi trường ảo (khuyến khích):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

## Cách thực thi

*(Hướng dẫn chi tiết sẽ được cập nhật sau khi hoàn thành các module)*

1.  **Tiền xử lý dữ liệu:**
    ```bash
    python src/data_preprocessing/preprocess.py
    ```

2.  **Huấn luyện mô hình:**
    ```bash
    python src/training/train.py --model unet --epochs 50
    ```
