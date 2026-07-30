# Jetson Inspector

Hệ thống kiểm tra chất lượng sản phẩm bằng xử lý ảnh,
tối ưu cho NVIDIA Jetson Orin Nano với JetPack 6.1.

---

## Tính năng

- Giao diện GUI trực quan với Tkinter
- Hỗ trợ nhiều loại camera: USB, CSI, IP/RTSP, IMV (iRAYple)
- Phân tích lỗi bề mặt bằng gradient + morphology
- Zone ROI: vẽ vùng phân tích, gán tool cho từng vùng
- Nhiều tool: Defect Detection, Shape Search, Blob Analysis, Edge Detection, Binary Check
- Kết nối PLC qua Modbus TCP
- Thread-safe: fix race condition cho multithread
- Model management: lưu/load thông số

---

## Cấu trúc thư mục

```
SR/
├── main.py                 # GUI chính
├── inspector.py            # Core xử lý ảnh
├── shape_search.py         # Tool tìm hình dạng
├── zone_processor.py       # Xử lý vùng ROI
├── zone_tool_dialog.py     # Dialog cấu hình zone
├── imv_camera.py           # Camera IMV iRAYple
├── modbus_plc.py           # PLC Modbus TCP
├── models/                 # Model thông số
├── shape_templates/        # Ảnh mẫu Shape Search
├── input_images/           # Ảnh đầu vào (tự tạo)
└── output_results/         # Kết quả OK/NG (tự tạo)
    ├── OK/
    └── NG/
```

---

## Yêu cầu

### Hardware
- NVIDIA Jetson Orin Nano
- Camera: USB / CSI / IP / IMV iRAYple

### Software
- Ubuntu 20.04 / 22.04
- JetPack 6.1
- Python 3.10+
- OpenCV 4.10+ với CUDA

---

## Cài đặt

```bash
git clone https://github.com/nhat5320000/KeoSR.git
cd jetson-inspector
pip install pillow numpy
cp -r /path/to/MVSDK ./MVSDK
pip install pymodbus
```

---

## Chạy

```bash
python3 main.py
```

---

## Hướng dẫn nhanh

```
Bước 1: Config → Camera → Test Camera
Bước 2: [R] ROI Setup → Vẽ vùng → Gán Tool → Lưu
Bước 3: Config → Điều chỉnh thông số → Áp dụng
Bước 4: Model → Nhập tên → Lưu Model
Bước 5: [C] Chụp → [T] Dừng + xử lý → Xem OK/NG
```

---

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `C` | Bắt đầu chụp |
| `T` | Dừng + xử lý |
| `R` | ROI Setup |
| `I` | Config |
| `M` | Model |
| `F1` | Help |
| `←` `→` | Xem ảnh kết quả |
| `Q` | Thoát |

---

## Camera

| Loại | camera_type | Ghi chú |
|------|-------------|---------|
| USB Webcam | `USB` | camera_id = 0,1,2 |
| Jetson CSI | `CSI` | camera_csi_id = 0,1 |
| IP / RTSP | `IP` | camera_ip_url = rtsp://... |
| IMV iRAYple | `IMV` | Cần MVSDK |

### IMV Camera

| Thông số | Đơn vị | Ghi chú |
|----------|--------|---------|
| Exposure | microseconds (µs) | 10000 = 10ms, -1 = auto |
| Gain | dB | 0.0 ~ 24.0, -1 = auto |

---

## Thông số xử lý ảnh

### Tiền xử lý

| Thông số | Mô tả | Khuyến nghị |
|----------|-------|-------------|
| `blur_ksize` | Gaussian blur kernel (số lẻ) | 3-7 |
| `blur_sigma` | Sigma blur, 0 = tự tính | 0 |
| `bilateral_enable` | Bilateral filter | True |
| `bilateral_d` | Đường kính bilateral | 5-11 |
| `bilateral_sC` | Sigma color | 50-100 |
| `bilateral_sS` | Sigma space | 50-100 |
| `preprocess_order` | Thứ tự xử lý | blur_bilateral |
| `clahe_enable` | Tăng tương phản cục bộ | True |
| `clahe_clip` | Giới hạn tương phản | 2-4 |
| `clahe_grid_X/Y` | Lưới CLAHE | 8-16 |

### Gradient

| Thông số | Mô tả | Khuyến nghị |
|----------|-------|-------------|
| `grad_method` | sobel/scharr/canny/laplacian | sobel |
| `sobel_ksize` | Kernel Sobel | 3 |
| `grad_thr` | Ngưỡng phát hiện lỗi | 15-30 |
| `grad_thresh_type` | binary_inv/otsu/adaptive | binary_inv |
| `canny_thr1` | Canny ngưỡng thấp | 50 |
| `canny_thr2` | Canny ngưỡng cao | 150 |
| `adaptive_block` | Block adaptive threshold | 11 |
| `adaptive_c` | Hằng số adaptive | 2 |

### Morphology

| Thông số | Mô tả | Khuyến nghị |
|----------|-------|-------------|
| `morph_shape` | ellipse/rect/cross | ellipse |
| `morph_open_iter` | Số lần OPEN - xóa nhiễu | 1-3 |
| `morph_close_iter` | Số lần CLOSE - lấp lỗ | 2-4 |
| `morph_extra_enable` | Bật thêm bước morph | False |
| `morph_extra_type` | dilate/erode/gradient | dilate |

### ROI Mask

| Thông số | Mô tả | Khuyến nghị |
|----------|-------|-------------|
| `roi_thresh` | Ngưỡng tìm vật thể | 100-200 |
| `roi_thresh_type` | binary/binary_inv/otsu | binary |
| `roi_offset` | Độ rộng vùng phân tích (px) | 50-100 |
| `roi_min_area` | Diện tích tối thiểu ROI (px²) | 1000-5000 |
| `roi_contour_approx` | Làm mịn contour | False |

### Classify

| Thông số | Mô tả | Ghi chú |
|----------|-------|---------|
| `ok_ratio` | Tỉ lệ pixel OK tối thiểu | 0.95-0.99 |
| `noise_max_area` | Bỏ qua lỗi nhỏ trong ROI (px²) | 500-2500 |
| `border_noise_max_area` | Bỏ qua lỗi nhỏ ở biên (px²) | 500-2500 |
| `min_defect_area` | Bỏ qua lỗi cực nhỏ (px²) | 5-50 |
| `green_dilate_k` | Mở rộng vùng OK | Lớn = dễ OK |
| `ring_dilate_k` | Vòng kiểm tra lỗi | Lớn = dễ OK |
| `sep_open_k` | Tách các vùng lỗi | 3-5 |
| `max_defect_count` | Số lỗi tối đa kiểm tra | -1 = tất cả |

---

## Tool phân tích

### Defect Detection
Phát hiện lỗi bề mặt bằng gradient analysis.
Phù hợp: vết xước, lỗ khuyết, bề mặt không đồng đều.

### Shape Search
Tìm hình dạng khớp ảnh mẫu (template matching).
Phù hợp: kiểm tra hình dạng, vị trí linh kiện.

### Blob Analysis
Đếm và phân tích vùng blob.
Phù hợp: đếm linh kiện, kiểm tra kết nối.

### Edge Detection
Phát hiện biên cạnh bằng Canny.
Phù hợp: khe hở, gap, mép cắt.

### Binary Check
Kiểm tra tỉ lệ sáng/tối.
Phù hợp: độ phủ, mức độ lấp đầy.

---

## Điều chỉnh theo vấn đề

| Vấn đề | Thông số cần chỉnh |
|--------|-------------------|
| Quá nhiều NG sai | Tăng `noise_max_area`, giảm `ok_ratio` |
| Bỏ sót lỗi thật | Giảm `noise_max_area`, tăng `ok_ratio` |
| Vùng đỏ nhỏ li ti | Tăng `min_defect_area`, tăng `morph_open_iter` |
| Vùng đỏ → xanh nhiều | Giảm `green_dilate_k`, giảm `ring_dilate_k` |
| ROI không đúng | Điều chỉnh `roi_thresh`, `roi_thresh_type` |
| Xử lý chậm | Giảm `n_threads`, tắt `bilateral` |
| NG 0% ngẫu nhiên | Race condition - đã fix trong version này |
| 2 ảnh giống nhau | Race condition - đã fix trong version này |

---

## Kết nối PLC Modbus TCP

```
Config → Tab PLC
plc_enable  = True
plc_host    = 192.168.x.x
plc_port    = 502
plc_unit_id = 1
```

**Coil - PLC ghi vào Camera:**

| Địa chỉ | Tên | Mô tả |
|---------|-----|-------|
| M1000 | Camera ON | 1 = bật camera, 0 = tắt |
| M1001 | Trigger | 1 = bắt đầu chụp, 0 = dừng + xử lý |

**Register - Camera ghi vào PLC:**

| Địa chỉ | Tên | Mô tả |
|---------|-----|-------|
| D2000 | frame_count | Số ảnh chụp |
| D2002 | ok_count | Số ảnh OK |
| D2004 | ng_count | Số ảnh NG |
| D2006 | proc_ms | Thời gian xử lý (ms) |
| D2008 | status | 0=Idle 1=Chụp 2=XửLý 3=Xong 9=Lỗi |

---

## Tối ưu Jetson

```python
n_threads   = 4       # Tối đa 4
camera_fps  = 30      # Tối đa 30
blur_ksize  = 3       # Kernel nhỏ
bilateral   = False   # Tắt nếu cần nhanh
clahe_grid  = 8       # Grid nhỏ hơn
```

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| Segmentation fault | Emoji trong Tkinter | Xóa emoji khỏi button text |
| IMV SDK not found | Thiếu MVSDK | Copy MVSDK vào thư mục code |
| Camera không mở | Driver lỗi | Kiểm tra `CAP_V4L2` |
| Kết quả không nhất quán | Race condition | Đã fix - local objects |
| NG 0% ngẫu nhiên | Race condition kernel | Đã fix - cfg_snap |
| 2 ảnh giống nhau | Race condition CFG | Đã fix - local kernels |

---

## License

MIT License
