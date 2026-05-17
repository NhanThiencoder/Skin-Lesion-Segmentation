Dưới đây là mẫu các issue cho các thành viên trong nhóm. Bạn hãy sao chép nội dung cho từng issue, tạo mới trên tab "Issues" của repository và gán cho thành viên tương ứng.

---

### **Issue 1: Thiết lập dự án và tiền xử lý dữ liệu**

**Người được giao:** Nhân Thiện

**Mô tả:**
Công việc này bao gồm các bước đầu tiên để thiết lập môi trường và chuẩn bị dữ liệu cho việc huấn luyện.

**Danh sách công việc:**
- [ ] Tải và giải nén bộ dữ liệu ISIC 2018 Task 1 vào thư mục `data/`.
- [ ] Viết script trong `src/data_preprocessing/` để thực hiện:
  - [ ] Đọc dữ liệu hình ảnh và mặt nạ (mask).
  - [ ] Resize tất cả hình ảnh và mặt nạ về một kích thước thống nhất (ví dụ: 256x256).
  - [ ] Chuẩn hóa giá trị pixel của hình ảnh (ví dụ: về khoảng [0, 1]).
  - [ ] Phân chia dữ liệu thành các tập train, validation và test.
- [ ] Viết script để áp dụng các kỹ thuật tăng cường dữ liệu (Data Augmentation) như lật, xoay, thay đổi độ sáng/tương phản. Sử dụng thư viện `albumentations`.
- [ ] Cập nhật file `README.md` với hướng dẫn chi tiết về cách chạy các script tiền xử lý.

---

### **Issue 2: Xây dựng và Huấn luyện mô hình CNN (VGG/ResNet/U-Net)**

**Người được giao:** Tấn Phát

**Mô tả:**
Nhiệm vụ này tập trung vào việc triển khai và huấn luyện một kiến trúc CNN kinh điển cho bài toán phân vùng. U-Net là một lựa chọn phổ biến và hiệu quả.

**Danh sách công việc:**
- [ ] Nghiên cứu và lựa chọn một kiến trúc CNN phù hợp (ví dụ: U-Net).
- [ ] Định nghĩa kiến trúc mô hình đã chọn trong `src/model_architectures/`.
- [ ] Viết script huấn luyện trong `src/training/train.py`:
  - [ ] Tải dữ liệu đã được tiền xử lý.
  - [ ] Xây dựng vòng lặp huấn luyện (training loop).
  - [ ] Định nghĩa hàm mất mát (loss function) phù hợp (ví dụ: Dice Loss, Binary Cross-Entropy).
  - [ ] Lựa chọn bộ tối ưu hóa (optimizer) như Adam hoặc SGD.
  - [ ] Lưu lại mô hình có kết quả tốt nhất trên tập validation vào thư mục `models/`.
- [ ] Áp dụng **Transfer Learning**: sử dụng một backbone (VGG hoặc ResNet) đã được pre-train làm encoder để trích xuất đặc trưng.
- [ ] Thực hiện **Parameter Tuning** (tinh chỉnh learning rate, batch size, epoch) để tối ưu hóa mô hình.

---

### **Issue 3: Xây dựng và Huấn luyện mô hình có cơ chế Attention**

**Người được giao:** Minh Đức

**Mô tả:**
Nhiệm vụ này yêu cầu nghiên cứu và áp dụng một kiến trúc có sử dụng cơ chế Attention để cải thiện khả năng tập trung của mô hình vào các vùng quan trọng.

**Danh sách công việc:**
- [ ] Nghiên cứu và lựa chọn kiến trúc có cơ chế **Attention / Transformer / ViT** (Ví dụ: TransUNet - kết hợp U-Net và Vision Transformer cho phân vùng ảnh).
- [ ] Lựa chọn và triển khai kiến trúc mô hình trong `src/model_architectures/`.
- [ ] Tái sử dụng hoặc điều chỉnh script huấn luyện từ `Issue 2` để huấn luyện mô hình mới này.
- [ ] So sánh hiệu suất ban đầu của mô hình này với mô hình CNN từ `Issue 2`.
- [ ] Ghi lại các kết quả và nhận xét vào một file markdown trong thư mục `notebooks/`.

---

### **Issue 4: Đánh giá mô hình và Triển khai ứng dụng**

**Người được giao:** Quốc Bảo

**Mô tả:**
Công việc này tập trung vào việc đánh giá chi tiết các mô hình đã huấn luyện và xây dựng một giao diện đơn giản để trình diễn kết quả.

**Danh sách công việc:**
- [ ] Viết script trong `src/evaluation/evaluate.py` để:
  - [ ] Tải 2 mô hình (CNN và Attention) đã huấn luyện từ thư mục `models/`.
  - [ ] Áp dụng **Ensemble Learning** kết hợp kết quả dự đoán của cả 2 mô hình để tạo ra mặt nạ dự đoán cuối cùng (nhằm nâng cao độ chính xác).
  - [ ] Chạy dự đoán trên tập test.
  - [ ] In ra các chỉ số đánh giá: Dice Coefficient, Jaccard Index (IoU), Accuracy, Precision, Recall.
  - [ ] Trực quan hóa kết quả: hiển thị hình ảnh gốc, mặt nạ thực tế và mặt nạ dự đoán cạnh nhau.
- [ ] Tạo một Jupyter Notebook trong `notebooks/` để trình bày và so sánh kết quả của các mô hình đơn lẻ và mô hình Ensemble.
- [ ] Xây dựng một ứng dụng web đơn giản (sử dụng Streamlit hoặc Flask) trong `src/deployment/`:
  - [ ] Giao diện cho phép người dùng tải lên một hình ảnh tổn thương da.
  - [ ] Ứng dụng sẽ xử lý ảnh, dùng mô hình tốt nhất để dự đoán và hiển thị ảnh gốc cùng với vùng tổn thương được phân vùng.
- [ ] Cập nhật `README.md` với hướng dẫn chạy script đánh giá và khởi động ứng dụng.
