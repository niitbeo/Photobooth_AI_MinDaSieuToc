# Photobooth AI MinDa Siêu Tốc (ONNX Runtime)

Đây là phiên bản AI CodeFormer làm mịn da, xóa mụn, làm nét khuôn mặt **Siêu Nhẹ - Siêu Tốc** dành riêng cho các dòng máy tính cấu hình thấp (không có VGA, chỉ chạy CPU Core i3/i5) chuyên dùng cho buồng chụp ảnh Photobooth.

> **Tác giả:** Nguyễn Lê Trường (Photobooth AI MinDa)
> 
> ☕ **Bonus ly cafe sáng:** Nếu source code siêu việt này giúp hệ thống Photobooth của bạn chạy mượt mà và kiếm được nhiều tiền, hãy mời tác giả một ly cafe sáng nhé! Cảm ơn bạn rất nhiều!
> 
> <div align="center">
>   <img src="nganhag.png" width="300" alt="Mã QR Ủng hộ Cafe Sáng">
> </div>

## 📸 Hình ảnh Thực tế (Trước - Sau)
Bản siêu tốc này tích hợp thuật toán **Lõi kép**: Càn quét mụn rỗ bằng **Máy Cà Phấn OpenCV** trước khi đưa cho **Lõi AI CodeFormer** vẽ lại nét siêu thực (tránh hoàn toàn hiện tượng ảo giác rỗ mặt của AI gốc).
<div align="center">
  <img src="SoSanh_HienTuong.jpg" width="800" alt="Kết quả xóa mụn 2">
</div>

## 🚀 Tính năng nổi bật
* **Không cần PyTorch/CUDA**: Chạy độc quyền bằng ONNX Runtime dựa vào sức mạnh thuần của CPU đa luồng.
* **Tốc độ ánh sáng (Real-time)**: Xử lý 1 ảnh với hệ CPU Core i3 chỉ tốn ~13 giây (Tốc độ khoảng 6.5s/1 mặt), đủ sức gánh quy trình In ảnh của Photobooth.
* **YuNet Face Detector (2023)**: Thay vì dùng HaarCascade cũ, phiên bản này được nhúng `YuNet` chuyên nghiệp giúp tỷ lệ nhận diện mặt người trống góc tối lên đến 99,99% (Không nhận diện nhầm áo/vách tường).
* **CodeFormer ONNX (360MB)**: Trái tim bộ lọc giữ nguyên khả năng tẩy điểm mù lỗi lõm xuất sắc.

---

## 💻 Cài đặt
Do Github giới hạn file >100MB, file `codeformer.onnx` nặng 360MB KHÔNG được bao gồm trong Source code này. Bạn cần làm theo 2 bước sau:

**Bước 1: Tải bộ source code**
```bash
git clone https://github.com/niitbeo/Photobooth_AI_MinDaSieuToc.git
cd Photobooth_AI_MinDaSieuToc
```

**Bước 2: Cài đặt thư viện Python (Máy cần có Python 3.8 - 3.11)**
```bash
python -m venv venv
# Tuỳ hệ điều hành Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

**Bước 3: Tải Model `codeformer.onnx` AI Cốt lõi**
Bạn phải tải file AI nặng 360MB về bỏ vào trong cùng thư mục này (Ngang hàng với file `photobooth_onnx.py`).
* Link tải từ HuggingFace (Siêu Tốc): [Tải codeformer.onnx (360MB)](https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx)

*File `face_detection_yunet_2023mar.onnx` (1.5MB - Model tìm bộ mặt) đã được đi kèm siêu nhẹ sẵn trong source code.*

---

## 📸 Hướng dẫn sử dụng cho Photobooth

Bây giờ bạn chỉ cần gọi dòng lệnh Command Line (CLI) từ phần mềm Photobooth của bạn (dslrBooth, Sparkbooth...) vào lúc "Chụp xong 1 tấm ảnh trước khi in":

**Cú pháp:**
```bash
python photobooth_onnx.py -i <đường_dẫn_ảnh_khi_chụp> -o <đường_dẫn_ảnh_xuất_ra_in> -w <độ_ảo_cà_mặt>
```

**Ví dụ:**
```bash
.\venv\Scripts\python.exe photobooth_onnx.py -i "C:\DSLR_Booth\guest_01.jpg" -o "C:\DSLR_Booth\guest_01_AI.jpg" -w 0.7
```

* **`-w` (Fidelity weight):** Số này điều chỉnh mức độ làm mịn. Dao động từ `0.0` đến `1.0`. Khuyên dùng cho Photobooth là `0.7` để giữ lại góc cạnh tự nhiên và không biến khách thành người lạ.

## 🤝 Liên hệ
*Bản quyền và tối ưu luồng AI bởi Tác Giả: Nguyễn Lê Trường.*
Mọi thắc mắc kỹ thuật vui lòng tạo issue hoặc liên hệ trực tiếp.
