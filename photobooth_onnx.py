import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import time
import psutil

class CodeFormerONNX:
    def __init__(self, modelpath):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # Tối ưu hóa cho CPU đa luồng
        so.intra_op_num_threads = psutil.cpu_count(logical=True)
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(modelpath, so, providers=['CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_name0 = model_inputs[0].name
        self.input_name1 = model_inputs[1].name
        self.inpheight = model_inputs[0].shape[2]
        self.inpwidth = model_inputs[0].shape[3]

    def post_processing(self, tensor, min_max=(-1, 1)):
        _tensor = tensor[0]
        _tensor = _tensor.clip(min_max[0], min_max[1])
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        img_np = _tensor.transpose(1, 2, 0)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = (img_np * 255.0).round().astype(np.uint8)
        return img_np

    def process_face(self, srcimg, weight=0.7, beauty_level=0.85):
        # TIỀN XỬ LÝ: Tẩy mụn siêu tốc bằng thuật toán Nhiếp Ảnh (App Filter 360)
        # Giúp làm nhẵn các ổ mụn rỗ và lỗ chân lông to, tránh việc AI nhìn nhầm thành nếp nhăn
        smooth = srcimg.copy()
        
        if beauty_level > 0:
            h, w = srcimg.shape[:2]
            
            # Tính toán kích thước cọ mụn tỷ lệ thuận với kích thước khuôn mặt (chống bị wash-out mặt nhỏ)
            k_size = max(3, int(w * 0.03))
            if k_size % 2 == 0: k_size += 1
            d_size = max(5, int(w * 0.05))

            # 1. TIÊU DIỆT MỤN TO
            smooth = cv2.medianBlur(smooth, k_size) 
            
            # 2. BÀO PHẲNG DA
            for _ in range(3):
                smooth = cv2.bilateralFilter(smooth, d_size, 75, 75)
            
            # 3. Kéo lại nét viền (mắt, mũi, môi) bớt lờ mờ do cà phấn
            gaussian = cv2.GaussianBlur(smooth, (0,0), 3.0)
            sharpened = cv2.addWeighted(smooth, 1.5, gaussian, -0.5, 0)
            
            # 4. Tính toán độ mix mụn và nền:
            blend_original = max(0.0, 1.0 - beauty_level)
            if blend_original > 0:
                beauty_face = cv2.addWeighted(sharpened, 1.0 - blend_original, srcimg, blend_original, 0)
            else:
                beauty_face = sharpened
                
            # 5. KHẮC PHỤC QUẦNG MỜ (HALO BLUR): 
            # Chỉ đánh phấn vùng Elip (trung tâm mặt). 
            # Áo, viền tóc, cảnh vật background bên trong box phải chừa lại y nguyên 100% không bị làm mờ.
            face_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(face_mask, (w//2, h//2), (int(w*0.35), int(h*0.42)), 0, 0, 360, 1.0, -1)
            blur_r = max(3, int(w * 0.1))
            if blur_r % 2 == 0: blur_r += 1
            face_mask = cv2.GaussianBlur(face_mask, (blur_r, blur_r), 0)
            face_mask = np.stack([face_mask]*3, axis=2)
            
            pre_processed_img = (srcimg * (1.0 - face_mask) + beauty_face * face_mask).astype(np.uint8)
        else:
            pre_processed_img = srcimg

        # Bây giờ mới ném khuôn mặt đã "đánh phấn mịn" này cho CodeFormer để nó làm cực sắc nét lại mắt/mũi/miệng
        dstimg = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2RGB)
        dstimg = cv2.resize(dstimg, (self.inpwidth, self.inpheight), interpolation=cv2.INTER_AREA)
        dstimg = (dstimg.astype(np.float32)/255.0 - 0.5) / 0.5
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0).astype(np.float32)

        # Inference siêu tốc
        output = self.session.run(None, {self.input_name0: input_image, self.input_name1: np.array([weight], dtype=np.float64)})[0]
        restored_img = self.post_processing(output, min_max=(-1, 1))
        return restored_img

def get_feathered_mask(size, blur_radius=15):
    """Tạo mask viền mờ để ghép mặt mượt mà, không bị lộ viền"""
    mask = np.zeros((size[1], size[0]), dtype=np.float32)
    # Vẽ hình elip trứng (khuôn mặt)
    center = (size[0] // 2, size[1] // 2)
    axes = (int(size[0] * 0.4), int(size[1] * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    # Làm mờ viền
    mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
    return np.stack([mask, mask, mask], axis=2)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Photobooth CodeFormer ONNX (No PyTorch)")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Ảnh gốc từ máy ảnh')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Ảnh sau khi làm nét')
    parser.add_argument('-w', '--weight', type=float, default=0.7, help='Fidelity weight')
    parser.add_argument('-b', '--beauty', type=float, default=0.85, help='Mức độ đánh phấn làm mịn mụn (0.0 đến 1.0. Tắt là 0)')
    args = parser.parse_args()

    print(f"Đang xử lý bằng AI ONNX siêu thanh: {args.input_path}")

    # Load hình ảnh
    img = cv2.imread(args.input_path)
    if img is None:
        print("Không đọc được ảnh!")
        return
    
    # Sử dụng bộ dò tìm khuôn mặt YuNet (Siêu nhẹ, độ chính xác cực cao, không bỏ sót mặt)
    detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (img.shape[1], img.shape[0]),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    
    # Tìm khuôn mặt
    _, faces_result = detector.detect(img)
    faces = []
    
    if faces_result is not None:
        for face in faces_result:
            box = face[0:4].astype(np.int32)
            # box = [x, y, w, h]
            faces.append(box)
    
    print(f"Phát hiện {len(faces)} khuôn mặt.")

    if len(faces) == 0:
        cv2.imwrite(args.output_path, img)
        print("Không thấy khuôn mặt. Trả về ảnh gốc.")
        return

    # Load Model ONNX (Chỉ load một lần nếu làm app, ở đây tool chạy 1 lần)
    net = CodeFormerONNX('codeformer.onnx')

    restored_img = img.copy()

    for (x, y, w, h) in faces:
        # Parding khuôn mặt lớn hơn một chút để lấy cả viền tóc
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img.shape[1], x + w + margin_x)
        y2 = min(img.shape[0], y + h + margin_y)

        # Cắt khuôn mặt
        face_crop = restored_img[y1:y2, x1:x2]
        crop_h, crop_w = face_crop.shape[:2]

        # Xử lý làm nét
        restored_face = net.process_face(face_crop, weight=args.weight, beauty_level=args.beauty)
        
        # Scale về kích thước gốc
        restored_face_resized = cv2.resize(restored_face, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # Lấy mask và blend
        mask = get_feathered_mask((crop_w, crop_h), blur_radius=int(crop_w*0.08))
        
        # Ghép ảnh
        blended = face_crop * (1.0 - mask) + restored_face_resized * mask
        restored_img[y1:y2, x1:x2] = blended.astype(np.uint8)

    cv2.imwrite(args.output_path, restored_img)
    print(f"Hoàn thành xuất tới: {args.output_path}")

    # Thống kê
    process = psutil.Process()
    mem_info = process.memory_info()
    print("========================================")
    print("THỐNG KÊ (ONNX C++ TRÊN CPU):")
    print(f"- Thời gian chạy: {time.time() - start_time:.2f} giây")
    print(f"- RAM tiêu thụ: {mem_info.rss / 1024 / 1024:.2f} MB")
    print("========================================")

if __name__ == '__main__':
    main()
