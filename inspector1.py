import cv2
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

# ════════════════════════════════════════════════════════════════
# CẤU HÌNH - ĐẦY ĐỦ & LINH HOẠT
# ════════════════════════════════════════════════════════════════
CFG = {
    # ── Crop ─────────────────────────────────────────────────
    "crop_mode":             "none",

    # ── Tiền xử lý (Pre-processing) ──────────────────────────
    "blur_ksize":            5,       # Gaussian blur kernel (lẻ 1-21)
    "blur_sigma":            0,       # 0 = tự tính từ ksize
    "bilateral_d":           11,      # Bilateral diameter
    "bilateral_sc":          75,      # Bilateral sigma color
    "bilateral_ss":          75,      # Bilateral sigma space
    "bilateral_enable":      True,    # ✅ Bật/tắt bilateral
    "preprocess_order":      "blur_bilateral",
                                      # ✅ "blur_bilateral" | "bilateral_blur"
                                      #    | "blur_only" | "bilateral_only"

    # ── CLAHE ─────────────────────────────────────────────────
    "clahe_clip":            4.0,     # Clip limit (0.5-8.0)
    "clahe_grid_x":          14,      # Grid X
    "clahe_grid_y":          14,      # Grid Y
    "clahe_enable":          True,    # ✅ Bật/tắt CLAHE

    # ── Gradient ─────────────────────────────────────────────
    "grad_thr":              18,      # Ngưỡng gradient (1-100)
    "grad_method":           "sobel", # ✅ "sobel"|"scharr"|"laplacian"|"canny"
    "sobel_ksize":           3,       # ✅ Kích thước kernel Sobel (1,3,5,7)
    "canny_thr1":            50,      # ✅ Canny threshold 1
    "canny_thr2":            150,     # ✅ Canny threshold 2
    "grad_thresh_type":      "binary_inv",
                                      # ✅ "binary_inv"|"otsu"|"adaptive"
    "adaptive_block":        11,      # ✅ Block size cho adaptive threshold
    "adaptive_c":            2,       # ✅ Hằng số C cho adaptive threshold

    # ── Morphology ───────────────────────────────────────────
    "morph_k":               5,       # Kernel size morphology chính
    "morph_open_iter":       2,       # ✅ Số lần OPEN
    "morph_close_iter":      3,       # ✅ Số lần CLOSE
    "morph_shape":           "ellipse",
                                      # ✅ "ellipse"|"rect"|"cross"
    "morph_extra_enable":    False,   # ✅ Bật thêm 1 bước morph
    "morph_extra_type":      "dilate",# ✅ "dilate"|"erode"|"gradient"
    "morph_extra_iter":      1,       # ✅ Số lần morph thêm

    # ── ROI Mask ─────────────────────────────────────────────
    "roi_thresh":            180,     # Ngưỡng tìm ROI (50-255)
    "roi_thresh_type":       "binary",# ✅ "binary"|"binary_inv"|"otsu"
    "roi_offset":            75,      # Mở rộng biên ROI (px)
    "roi_min_area":          1000,    # ✅ Diện tích tối thiểu vùng ROI
    "roi_contour_approx":    False,   # ✅ Làm mịn contour ROI

    # ── Classify ─────────────────────────────────────────────
    "ok_ratio":              0.98,    # Tỉ lệ OK (0.5-1.0)
    "noise_max_area":        2500,    # Nhiễu trong: max area (px²)
    "border_noise_max_area": 2500,    # Nhiễu biên: max area (px²)
    "min_defect_area":       10,      # ✅ Bỏ qua lỗi nhỏ hơn ngưỡng này
    "green_dilate_k":        7,       # ✅ Kernel dilate vùng xanh (3-15)
    "sep_open_k":            3,       # ✅ Kernel tách nhiễu K_SEP
    "ring_dilate_k":         3,       # ✅ Kernel dilate để tạo ring K1
    "max_defect_count":      -1,      # ✅ -1=không giới hạn số lỗi tối đa

    # ── Output ───────────────────────────────────────────────
    "out_q":                 85,      # JPEG quality (50-100)
    "max_w":                 1600,    # Max width ảnh kết quả
    "save_panels":           [1,2,3,4,5,6],
                                      # ✅ Chọn panel nào lưu (1-6)
    "colormap_grad":         cv2.COLORMAP_HOT,
                                      # ✅ COLORMAP_HOT|JET|VIRIDIS|INFERNO

    # ── Hệ thống ─────────────────────────────────────────────
    "n_threads":             6,
}


# ════════════════════════════════════════════════════════════════
# HELPER: Tạo kernel morphology
# ════════════════════════════════════════════════════════════════
def _make_kernel(size, shape_str=None):
    shape_str = shape_str or CFG.get("morph_shape", "ellipse")
    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect":    cv2.MORPH_RECT,
        "cross":   cv2.MORPH_CROSS,
    }
    shape = shape_map.get(shape_str, cv2.MORPH_ELLIPSE)
    s     = max(1, size)
    return cv2.getStructuringElement(shape, (s, s))


# Kernels động (tạo lại khi CFG thay đổi)
def _rebuild_kernels():
    global KERNEL, K_SEP, K1, K_GREEN
    KERNEL  = _make_kernel(CFG["morph_k"])
    K_SEP   = _make_kernel(CFG["sep_open_k"])
    K1      = _make_kernel(CFG["ring_dilate_k"])
    K_GREEN = _make_kernel(CFG["green_dilate_k"])


KERNEL  = _make_kernel(5)
K_SEP   = _make_kernel(3)
K1      = _make_kernel(3)
K_GREEN = _make_kernel(7)

# GPU objects
_clahe           = None
_gpu_ready       = False
_gpu_blur        = None
_gpu_sobelx      = None
_gpu_sobely      = None
_gpu_morph_open  = None
_gpu_morph_close = None


# ════════════════════════════════════════════════════════════════
# INIT GPU
# ════════════════════════════════════════════════════════════════
def init_gpu():
    global _clahe, _gpu_ready
    global _gpu_blur, _gpu_sobelx, _gpu_sobely
    global _gpu_morph_open, _gpu_morph_close

    _rebuild_kernels()

    _clahe = cv2.createCLAHE(
        clipLimit    = CFG["clahe_clip"],
        tileGridSize = (CFG["clahe_grid_x"], CFG["clahe_grid_y"]))

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        k = CFG["blur_ksize"]
        sg = CFG["blur_sigma"]
        _gpu_blur = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (k, k), sg)
        _gpu_sobelx = cv2.cuda.createSobelFilter(
            cv2.CV_8UC1, cv2.CV_32F, 1, 0,
            ksize=CFG["sobel_ksize"])
        _gpu_sobely = cv2.cuda.createSobelFilter(
            cv2.CV_8UC1, cv2.CV_32F, 0, 1,
            ksize=CFG["sobel_ksize"])
        _gpu_morph_open = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN,  cv2.CV_8UC1, KERNEL,
            iterations=CFG["morph_open_iter"])
        _gpu_morph_close = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, KERNEL,
            iterations=CFG["morph_close_iter"])
        _gpu_ready = True
        print("✅ GPU CUDA sẵn sàng")
    else:
        _gpu_ready = False
        print("⚠️  Không có GPU, dùng CPU")


# ════════════════════════════════════════════════════════════════
# ROI MASK
# ════════════════════════════════════════════════════════════════
def get_roi_mask(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    t = CFG["roi_thresh_type"]
    if t == "otsu":
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif t == "binary_inv":
        _, binary = cv2.threshold(
            gray, CFG["roi_thresh"], 255,
            cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(
            gray, CFG["roi_thresh"], 255,
            cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    contours = [c for c in contours
                if cv2.contourArea(c) >=
                CFG["roi_min_area"]]
    if not contours:
        return None, None, None

    c_orig = max(contours, key=cv2.contourArea)

    if CFG["roi_contour_approx"]:
        eps    = 0.005 * cv2.arcLength(c_orig, True)
        c_orig = cv2.approxPolyDP(c_orig, eps, True)

    h, w      = img_bgr.shape[:2]
    mask_orig = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask_orig, [c_orig], 255)

    # ════════════════════════════════════════
    # FIX: Dung kernel nho dilate nhieu lan
    # thay vi kernel lon 1 lan
    # roi_offset=75 → kernel 151x151 = CHAM
    # Thay bang: kernel 5x5 dilate 15 lan = NHANH
    # ════════════════════════════════════════
    offset  = CFG["roi_offset"]
    k_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))
    # Tinh so lan lap: offset/2 vi moi lan mo rong ~2px
    n_iter  = max(1, offset // 2)
    mask_dila = cv2.dilate(
        mask_orig, k_small,
        iterations=n_iter)

    between = cv2.bitwise_and(
        mask_dila, cv2.bitwise_not(mask_orig))
    # ── FIX: Neu between rong → dung mask_dila ──────
    if np.count_nonzero(between) == 0:
        print("[WARN] get_roi_mask: between=empty "
              f"→ fallback dung mask_dila")
        between = mask_dila.copy()
    # ────────────────────────────────────────────────

    cdil, _ = cv2.findContours(
        mask_dila, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not cdil:
        return between, c_orig, None
    return between, c_orig, \
        max(cdil, key=cv2.contourArea)


# ════════════════════════════════════════════════════════════════
# GRADIENT HELPER
# ════════════════════════════════════════════════════════════════
def _compute_gradient_cpu(gray: np.ndarray):
    """
    ✅ Tính gradient theo method trong CFG
    """
    method = CFG["grad_method"]
    try:
        if method == "scharr":
            gx   = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy   = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            grad = cv2.magnitude(gx, gy)

        elif method == "laplacian":
            lap  = cv2.Laplacian(gray, cv2.CV_32F,
                                ksize=CFG["sobel_ksize"])
            grad = np.abs(lap)

        elif method == "canny":
            # Canny trả về binary edge → dùng làm grad trực tiếp
            edges = cv2.Canny(gray,
                            CFG["canny_thr1"],
                            CFG["canny_thr2"])
            return edges.astype(np.uint8), edges

        else:  # sobel (default)
            gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0,
                            ksize=CFG["sobel_ksize"])
            gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1,
                            ksize=CFG["sobel_ksize"])
            grad = cv2.magnitude(gx, gy)

        #cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)
        #grad = grad.astype(np.uint8)
        #return grad, None   # (grad_map, edge_or_None)
        if grad is None or grad.size == 0:
            grad = np.zeros_like(gray, dtype=np.float32)

        grad_norm = np.zeros_like(grad)
        cv2.normalize(grad, grad_norm, 0, 255,
                    cv2.NORM_MINMAX)
        grad_out = grad_norm.astype(np.uint8)
        return grad_out, None

    except Exception as e:
        print(f"[WARN] _compute_gradient_cpu: {e}")
        return np.zeros_like(gray, dtype=np.uint8), None

def _apply_threshold_cpu(grad: np.ndarray):
    """
    ✅ Áp threshold theo grad_thresh_type
    """
    t = CFG["grad_thresh_type"]

    if t == "otsu":
        _, low_mask = cv2.threshold(
            grad, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elif t == "adaptive":
        low_mask = cv2.adaptiveThreshold(
            grad, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            CFG["adaptive_block"],
            CFG["adaptive_c"])

    else:  # binary_inv (default)
        _, low_mask = cv2.threshold(
            grad, CFG["grad_thr"], 255,
            cv2.THRESH_BINARY_INV)

    return low_mask


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════
def pipeline(img: np.ndarray):
    if _gpu_ready:
        return _pipeline_gpu(img)
    else:
        return _pipeline_cpu(img)


def _pipeline_gpu(img: np.ndarray):
    # ── Pre-process theo order ──────────────────────────────
    order = CFG["preprocess_order"]

    if order == "blur_only":
        gpu_src     = cv2.cuda_GpuMat(); gpu_src.upload(img)
        gpu_blurred = _gpu_blur.apply(gpu_src)
        denoised    = gpu_blurred.download()

    elif order == "bilateral_only":
        denoised = cv2.bilateralFilter(
            img,
            CFG["bilateral_d"],
            CFG["bilateral_sc"],
            CFG["bilateral_ss"])

    elif order == "bilateral_blur":
        denoised = cv2.bilateralFilter(
            img,
            CFG["bilateral_d"],
            CFG["bilateral_sc"],
            CFG["bilateral_ss"])
        gpu_src     = cv2.cuda_GpuMat()
        gpu_src.upload(denoised)
        gpu_blurred = _gpu_blur.apply(gpu_src)
        denoised    = gpu_blurred.download()

    else:  # blur_bilateral (default)
        gpu_src     = cv2.cuda_GpuMat(); gpu_src.upload(img)
        gpu_blurred = _gpu_blur.apply(gpu_src)
        blurred_cpu = gpu_blurred.download()
        denoised    = cv2.bilateralFilter(
            blurred_cpu,
            CFG["bilateral_d"],
            CFG["bilateral_sc"],
            CFG["bilateral_ss"]) \
            if CFG["bilateral_enable"] else blurred_cpu

    # ── Gray + CLAHE ─────────────────────────────────────────
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    if CFG["clahe_enable"]:
        gray = _clahe.apply(gray)

    # ── Gradient (GPU Sobel hoặc CPU fallback) ───────────────
    method = CFG["grad_method"]
    if method == "sobel":
        gpu_gray = cv2.cuda_GpuMat(); gpu_gray.upload(gray)
        gpu_gx   = _gpu_sobelx.apply(gpu_gray)
        gpu_gy   = _gpu_sobely.apply(gpu_gray)

        gx_cpu = gpu_gx.download()
        gy_cpu = gpu_gy.download()
        mag_cpu = cv2.magnitude(gx_cpu, gy_cpu)
        cv2.normalize(mag_cpu, mag_cpu, 0, 255, cv2.NORM_MINMAX)
        grad = mag_cpu.astype(np.uint8)

        # Threshold tren CPU
        _, low_mask = cv2.threshold(
            grad, CFG["grad_thr"], 255,
            cv2.THRESH_BINARY_INV)

        # Morphology tren GPU
        gpu_low = cv2.cuda_GpuMat()
        gpu_low.upload(low_mask)
        gpu_open  = _gpu_morph_open.apply(gpu_low)
        gpu_close = _gpu_morph_close.apply(gpu_open)
        cleaned   = gpu_close.download()
    else:
        # Fallback CPU cho method khác
        grad, edge = _compute_gradient_cpu(gray)
        low_mask   = edge if edge is not None \
                     else _apply_threshold_cpu(grad)
        cleaned    = _apply_morphology_cpu(low_mask)

    # ── Extra morphology ─────────────────────────────────────
    if CFG["morph_extra_enable"]:
        cleaned = _apply_extra_morph(cleaned)

    return denoised, gray, grad, low_mask, cleaned


def _pipeline_cpu(img: np.ndarray):
    # ── Pre-process theo order ──────────────────────────────
    order = CFG["preprocess_order"]
    k     = CFG["blur_ksize"]
    sg    = CFG["blur_sigma"]

    if order == "blur_only":
        img = cv2.GaussianBlur(img, (k, k), sg)

    elif order == "bilateral_only":
        if CFG["bilateral_enable"]:
            img = cv2.bilateralFilter(
                img,
                CFG["bilateral_d"],
                CFG["bilateral_sc"],
                CFG["bilateral_ss"])

    elif order == "bilateral_blur":
        if CFG["bilateral_enable"]:
            img = cv2.bilateralFilter(
                img,
                CFG["bilateral_d"],
                CFG["bilateral_sc"],
                CFG["bilateral_ss"])
        img = cv2.GaussianBlur(img, (k, k), sg)

    else:  # blur_bilateral
        img = cv2.GaussianBlur(img, (k, k), sg)
        if CFG["bilateral_enable"]:
            img = cv2.bilateralFilter(
                img,
                CFG["bilateral_d"],
                CFG["bilateral_sc"],
                CFG["bilateral_ss"])

    # ── Gray + CLAHE ─────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if CFG["clahe_enable"]:
        gray = _clahe.apply(gray)

    # ── Gradient ─────────────────────────────────────────────
    grad, edge = _compute_gradient_cpu(gray)

    # ── Threshold ────────────────────────────────────────────
    low_mask = edge if edge is not None \
               else _apply_threshold_cpu(grad)

    # ── Morphology ───────────────────────────────────────────
    cleaned = _apply_morphology_cpu(low_mask)

    # ── Extra morphology ─────────────────────────────────────
    if CFG["morph_extra_enable"]:
        cleaned = _apply_extra_morph(cleaned)

    return img, gray, grad, low_mask, cleaned


def _apply_morphology_cpu(mask: np.ndarray):
    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,  KERNEL,
        iterations=CFG["morph_open_iter"])
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, KERNEL,
        iterations=CFG["morph_close_iter"])
    return cleaned


def _apply_extra_morph(mask: np.ndarray):
    """✅ Bước morphology thêm tùy chọn"""
    t    = CFG["morph_extra_type"]
    it   = CFG["morph_extra_iter"]
    ops  = {
        "dilate":   cv2.MORPH_DILATE,
        "erode":    cv2.MORPH_ERODE,
        "gradient": cv2.MORPH_GRADIENT,
    }
    op = ops.get(t, cv2.MORPH_DILATE)
    return cv2.morphologyEx(mask, op, KERNEL, iterations=it)


# ════════════════════════════════════════════════════════════════
# CLASSIFY
# ════════════════════════════════════════════════════════════════
def classify(cleaned: np.ndarray, roi_mask: np.ndarray):
    px    = (roi_mask > 0) if roi_mask is not None \
            else np.ones_like(cleaned, bool)
    total = int(px.sum())
    _e    = {"cond":"A","red_count":0,"red_max_area":0,
             "details":[],"largest_green_px":0}
    if total == 0:
        return "OK", 1.0, _e

    ratio = np.count_nonzero(cleaned[px]) / total
    if ratio >= CFG["ok_ratio"]:
        return "OK", round(ratio, 4), _e

    green_roi = np.zeros_like(cleaned)
    green_roi[px & (cleaned > 0)] = 255
    high_roi  = np.zeros_like(cleaned)
    high_roi[px & (cleaned == 0)] = 255

    n_g, _, g_stats, _ = cv2.connectedComponentsWithStats(
        green_roi, connectivity=8)
    lg_px = int(g_stats[1:, cv2.CC_STAT_AREA].max()) \
            if n_g > 1 else 0

    # ✅ Dùng kernel sep_open_k từ CFG
    high_sep = cv2.morphologyEx(
        high_roi, cv2.MORPH_OPEN, K_SEP)
    n_red, red_lbl, red_stats, _ = cv2.connectedComponentsWithStats(
        high_sep, connectivity=8)
    red_areas = red_stats[1:, cv2.CC_STAT_AREA]

    base = {"cond":"B_fail","largest_green_px":lg_px,
            "details":[], "red_count":len(red_areas),
            "red_max_area":int(red_areas.max())
                           if len(red_areas) else 0}

    if len(red_areas) == 0:
        base["cond"] = "B"
        return "OK", round(ratio, 4), base

    # ✅ Dùng kernel green_dilate_k từ CFG
    green_dil = cv2.dilate(green_roi, K_GREEN)
    all_pass  = True
    details   = []
    max_defect = CFG["max_defect_count"]

    for i in range(1, n_red):
        area = int(red_stats[i, cv2.CC_STAT_AREA])

        # ✅ Bỏ qua lỗi quá nhỏ
        if area < CFG["min_defect_area"]:
            continue

        dot  = (red_lbl == i).astype(np.uint8) * 255
        # ✅ Dùng kernel ring_dilate_k từ CFG
        ring = cv2.bitwise_xor(cv2.dilate(dot, K1), dot)
        fi   = not np.any((ring > 0) & (green_dil == 0))
        or_  = roi_mask is not None and \
               np.any((ring > 0) & (roi_mask == 0))
        thr  = CFG["noise_max_area"] if fi \
               else CFG["border_noise_max_area"]
        loc  = "IN" if fi else ("BR" if or_ else "MZ")
        ok   = area < thr
        details.append({
            "id":i, "area":area, "fully_inside":fi,
            "outside_roi":or_, "pass":ok,
            "reason":f"{loc}({area}px"
                     f"{'<' if ok else '>='}{thr}px)"
                     f" {'✓' if ok else '✗'}"
        })
        if not ok:
            all_pass = False

        # ✅ Giới hạn số lỗi tối đa
        if max_defect > 0 and len(details) >= max_defect:
            break

    base.update({
        "details": details,
        "red_count": len(details),
        "red_max_area": max(
            (d["area"] for d in details), default=0)
    })
    if all_pass:
        base["cond"] = "B"
        return "OK", round(ratio, 4), base
    return "NG", round(ratio, 4), base


# ════════════════════════════════════════════════════════════════
# VẼ CONTOUR  (giữ nguyên)
# ════════════════════════════════════════════════════════════════
def draw_roi_lines(panel, c_orig, c_dila, label, ratio):
    out = panel.copy()
    clr = (0,255,0) if label=="OK" else (0,0,255)
    if c_orig is not None and c_dila is not None:
        h, w  = out.shape[:2]
        tmp   = np.zeros((h,w), np.uint8)
        mo    = np.zeros((h,w), np.uint8)
        cv2.fillPoly(tmp, [c_dila], 255)
        cv2.fillPoly(mo,  [c_orig], 255)
        shade = np.zeros_like(out)
        shade[cv2.bitwise_and(
            tmp, cv2.bitwise_not(mo)) > 0] = (0,165,255)
        out = cv2.addWeighted(out, 0.75, shade, 0.25, 0)
    if c_orig is not None:
        cv2.drawContours(out, [c_orig], -1, (0,255,0), 2)
        pt = tuple(c_orig[c_orig[:,:,1].argmin()][0])
        cv2.putText(out, "Inner", (pt[0]+4, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    if c_dila is not None:
        cv2.drawContours(out, [c_dila], -1, (0,0,255), 2)
        pt = tuple(c_dila[c_dila[:,:,1].argmin()][0])
        cv2.putText(out, "Outer", (pt[0]+4, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    cv2.putText(out, f"{label}  {ratio*100:.1f}%",
                (10,38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, clr, 3)
    return out


# ════════════════════════════════════════════════════════════════
# LƯU ẢNH KẾT QUẢ
# ════════════════════════════════════════════════════════════════
def save_result(out_path, label, ratio, noise_info,
                img_crop, gray, grad, low_mask, cleaned,
                roi_mask, c_orig, c_dila):
    h, w = gray.shape
    clr  = (0,255,0) if label=="OK" else (0,0,255)
    B    = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    RL   = lambda p: draw_roi_lines(
        p, c_orig, c_dila, label, ratio)

    # ✅ Dùng colormap từ CFG
    cmap = CFG.get("colormap_grad", cv2.COLORMAP_HOT)

    all_panels = {
        1: img_crop,
        2: RL(cv2.applyColorMap(grad, cmap)),
        3: RL(B(low_mask)),
        4: RL(B(cleaned)),
        5: RL(B(cv2.bitwise_and(gray, gray, mask=cleaned))),
        6: None,  # Xây dưới
    }

    p6 = np.zeros((h,w,3), np.uint8)
    if roi_mask is not None:
        p6[roi_mask > 0]                = (0,165,255)
        p6[(roi_mask>0)&(cleaned>0)]    = (0,255,  0)
        p6[(roi_mask>0)&(cleaned==0)]   = (0,  0,255)

    details = noise_info.get("details", [])
    if details and roi_mask is not None:
        hr  = np.zeros_like(cleaned)
        hr[(roi_mask>0)&(cleaned==0)] = 255
        hs  = cv2.morphologyEx(hr, cv2.MORPH_OPEN, K_SEP)
        _, rl, _, _ = cv2.connectedComponentsWithStats(
            hs, connectivity=8)
        for d in details:
            p6[rl==d["id"]] = (0,220,220) if d["pass"] \
                               else (0,0,255)

    if c_orig is not None:
        cv2.drawContours(p6, [c_orig], -1, (0,255,0), 2)
    if c_dila is not None:
        cv2.drawContours(p6, [c_dila], -1, (0,0,255), 2)

    np_ = sum(1 for d in details if d["pass"])
    nf  = sum(1 for d in details if not d["pass"])
    lns = [
        (f"[{label}] cond=[{noise_info.get('cond','?')}]", clr),
        (f"[A] low={ratio*100:.1f}%>="
         f"{CFG['ok_ratio']*100:.0f}%"
         f" {'✓' if ratio>=CFG['ok_ratio'] else '✗'}",
         (0,255,0) if ratio>=CFG["ok_ratio"] else (80,80,80)),
        (f"[B] green={noise_info.get('largest_green_px',0)}px | "
         f"dots:{len(details)} ✓{np_} ✗{nf}",
         (0,255,0) if nf==0 else (0,0,255)),
    ]
    for d in ([x for x in details if not x["pass"]][:4] +
              [x for x in details if  x["pass"]][:3]):
        lns.append((
            f"  {'✓' if d['pass'] else '✗'} "
            f"#{d['id']} {d['reason']}",
            (0,220,220) if d["pass"] else (0,0,255)))
    lns.append(("Yellow=ignored  Red=defect", (180,180,180)))
    for i, (txt, c) in enumerate(lns):
        cv2.putText(p6, txt, (8, 26+i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 2)

    all_panels[6] = p6

    # ✅ Chỉ lưu các panel được chọn
    panels_to_save = CFG.get("save_panels", [1,2,3,4,5,6])
    selected = [all_panels[i] for i in panels_to_save
                if i in all_panels]

    if not selected:
        selected = list(all_panels.values())

    titles_map = {
        1:"1.Original+ROI", 2:"2.Gradient+ROI",
        3:"3.LowMask+ROI",  4:"4.Cleaned+ROI",
        5:"5.Result+ROI",   6:"6.ROI Analysis"
    }
    for i, p in zip(panels_to_save, selected):
        t = titles_map.get(i, "")
        p_r = cv2.resize(p, (w,h),
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(p_r, t, (6,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255,255,255), 2)
        cv2.putText(p_r, t, (6,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0,0,0), 1)
        selected[panels_to_save.index(i)] = p_r

    # Xếp grid
    n = len(selected)
    if n <= 3:
        grid = np.hstack(selected)
    else:
        row1 = np.hstack(selected[:3])
        row2 = np.hstack(
            (selected[3:] +
             [np.zeros_like(selected[0])] * (3 - len(selected[3:]))
             )[:3])
        grid = np.vstack([row1, row2])

    if grid.shape[1] > CFG["max_w"]:
        sc   = CFG["max_w"] / grid.shape[1]
        grid = cv2.resize(grid, None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), grid,
                [cv2.IMWRITE_JPEG_QUALITY, CFG["out_q"]])


# ════════════════════════════════════════════════════════════════
# WORKER & BATCH  (giữ nguyên)
# ════════════════════════════════════════════════════════════════
def process_one(args):
    img_path, out_ok, out_ng = args
    img = cv2.imread(str(img_path))
    if img is None:
        return img_path.name, "ERROR", 0, 0.0

    t0 = time.perf_counter()
    _rebuild_kernels()  # Cập nhật kernel nếu CFG thay đổi

    img_crop, gray, grad, low_mask, cleaned = pipeline(img)
    roi_mask, c_orig, c_dila = get_roi_mask(img_crop)
    label, ratio, noise_info = classify(cleaned, roi_mask)

    out_path = (out_ok if label=="OK" else out_ng) / \
               f"{img_path.stem}_{label}_{ratio*100:.0f}pct.jpg"

    save_result(out_path, label, ratio, noise_info,
                img_crop, gray, grad, low_mask, cleaned,
                roi_mask, c_orig, c_dila)

    return img_path.name, label, \
           (time.perf_counter()-t0)*1000, ratio


def batch_process(input_dir="input_images",
                  output_dir="output_results"):
    ok = Path(output_dir) / "OK"
    ng = Path(output_dir) / "NG"
    ok.mkdir(parents=True, exist_ok=True)
    ng.mkdir(parents=True, exist_ok=True)

    exts   = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    images = sorted(p for p in Path(input_dir).iterdir()
                    if p.suffix.lower() in exts)
    if not images:
        print("⚠️  Không tìm thấy ảnh!"); return

    print(f"📂 {len(images)} ảnh")
    print(f"⚙️  GPU: {'✅ CUDA' if _gpu_ready else '❌ CPU'}")
    print(f"⚙️  grad_method = {CFG['grad_method']}")
    print(f"⚙️  thresh_type = {CFG['grad_thresh_type']}")
    print(f"⚙️  ok_ratio    = {CFG['ok_ratio']*100:.0f}%\n")

    args    = [(p, ok, ng) for p in images]
    counts  = {"OK":0,"NG":0,"ERROR":0}
    ms_list = []
    t0      = time.perf_counter()

    with ThreadPoolExecutor(
            max_workers=CFG["n_threads"]) as exe:
        for name, label, ms, ratio in exe.map(
                process_one, args):
            counts[label] = counts.get(label, 0) + 1
            if ms: ms_list.append(ms)
            icon = "✅" if label=="OK" else \
                   ("❌" if label=="NG" else "⚠️")
            print(f"  {icon} {name:<35} {label:<3} "
                  f"{ratio*100:5.1f}%  {ms:.0f}ms")

    total = time.perf_counter() - t0
    print(f"\n{'='*50}")
    print(f"  ✅ OK   : {counts['OK']} ảnh")
    print(f"  ❌ NG   : {counts['NG']} ảnh")
    print(f"  ⏱ Tổng : {total:.2f}s")
    if ms_list:
        print(f"  ⏱ TB   : {np.mean(ms_list):.0f}ms/ảnh")
    print(f"{'='*50}")


if __name__ == "__main__":
    init_gpu()
    batch_process("input_images", "output_results")