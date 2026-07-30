# ════════════════════════════════════════════════════════════════
# zone_processor.py  –  Xử lý từng vùng ROI bằng Tool đã gán
# ════════════════════════════════════════════════════════════════
import cv2
import numpy as np
import math
from pathlib import Path


class ZoneProcessor:
    """Áp dụng tool đã cấu hình cho từng vùng ROI"""
 
    COLORS = {
        "rect":     (0, 255, 0),
        "rot_rect": (0, 200, 255),
        "circle":   (255, 100, 0),
        "polygon":  (255, 0, 200),
    }

    def _gray_bgr(self, g):
        if g is None or g.size == 0:
            return np.zeros((10,10,3), np.uint8)
        if len(g.shape) == 3:
            return g
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def _apply_colormap(self, g,
                        cmap=cv2.COLORMAP_HOT):
        if g is None or g.size == 0:
            return np.zeros((10,10,3), np.uint8)
        if len(g.shape) == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(g, cmap)

    def _make_analysis_grid(self, panels, titles,
                             label, detail,
                             target_h=200):
        import math
        if not panels:
            return np.zeros((80,300,3), np.uint8)

        col = (0,220,0) if label=="OK" else \
              (0,80,220) if label=="NG" else \
              (180,180,0)

        resized = []
        for p in panels:
            if p is None or p.size == 0:
                p = np.zeros(
                    (target_h,target_h,3),np.uint8)
            if len(p.shape) == 2:
                p = cv2.cvtColor(
                    p, cv2.COLOR_GRAY2BGR)
            h,w  = p.shape[:2]
            sc   = target_h / max(h,1)
            rp   = cv2.resize(
                p, (max(1,int(w*sc)), target_h),
                interpolation=cv2.INTER_AREA)
            resized.append(rp)

        titled = []
        for i, rp in enumerate(resized):
            out = rp.copy()
            t   = titles[i] \
                  if i < len(titles) else f"#{i+1}"
            cv2.rectangle(out,(0,0),
                           (out.shape[1],20),
                           (0,0,0),-1)
            cv2.putText(out, t, (4,14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, col, 1,
                        cv2.LINE_AA)
            titled.append(out)

        COLS   = 3
        n_rows = math.ceil(len(titled)/COLS)
        blank  = np.zeros_like(titled[0])
        while len(titled) < n_rows*COLS:
            titled.append(blank.copy())

        rows = [np.hstack(titled[r*COLS:(r+1)*COLS])
                for r in range(n_rows)]
        grid = np.vstack(rows)

        fh     = 26
        footer = np.zeros(
            (fh, grid.shape[1], 3), np.uint8)
        footer[:] = (20,20,35)
        cv2.putText(footer,
                    f"  [{label}]  {detail}",
                    (6,18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, col, 1, cv2.LINE_AA)
        return np.vstack([grid, footer])

    @staticmethod
    def draw_zone_overlay(frame: np.ndarray,
                        zone:  dict,
                        result: dict) -> np.ndarray:
        """Vẽ viền vùng + label kết quả lên frame"""
        out   = frame.copy()
        label = result.get("label", "?")
        score = result.get("score", 0)
        col   = (0, 220, 0)  if label == "OK" else \
                (0,  60, 220) if label == "NG"  else \
                (180, 180, 0)
        t = zone.get("type")
        d = zone.get("data", [])
        h, w = frame.shape[:2]

        try:
            if t == "rect" and len(d) == 4:
                cv2.rectangle(out,
                            (d[0], d[1]),
                            (d[2], d[3]),
                            col, 2)
                tx, ty = d[0] + 4, d[1] + 20

            elif t == "circle" and len(d) == 3:
                cv2.circle(out,
                        (d[0], d[1]), d[2],
                        col, 2)
                tx = max(0, d[0] - d[2])
                ty = max(0, d[1] - d[2] - 6)

            elif t == "rot_rect" and len(d) == 5:
                box = cv2.boxPoints(
                    ((d[0], d[1]),
                    (d[2], d[3]),
                    d[4]))
                pts = box.astype(np.int32)
                cv2.polylines(out, [pts],
                            True, col, 2)
                tx = int(box[:, 0].min())
                ty = max(0, int(box[:, 1].min()) - 6)

            elif t == "polygon" and len(d) >= 3:
                pts = np.array(d, dtype=np.int32)
                cv2.polylines(out, [pts],
                            True, col, 2)
                tx = int(pts[:, 0].min())
                ty = max(0, int(pts[:, 1].min()) - 6)
            else:
                return out

            # Vẽ label + score
            name     = zone.get("name", "")
            tool     = zone.get("tool", "")
            lbl_text = (f"{name}: {label} "
                        f"{score*100:.0f}%")

            # Background text
            (tw, th), _ = cv2.getTextSize(
                lbl_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 1)
            cv2.rectangle(out,
                        (tx - 2,  ty - th - 4),
                        (tx + tw + 4, ty + 2),
                        (0, 0, 0), -1)
            cv2.putText(out, lbl_text,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"draw_zone_overlay error: {e}")

        return out

    def process_zone(self, frame: np.ndarray,
                     zone: dict) -> dict:
        """
        Cắt vùng từ frame → chạy tool → trả kết quả
        """
        tool    = zone.get("tool", "DefectDetection")
        cfg     = zone.get("tool_cfg", {})
        enabled = zone.get("enabled", True)

        if not enabled:
            return {"label":"SKIP","score":1.0,
                    "detail":"Vùng bị tắt"}

        # Cắt vùng từ frame
        crop = self._extract_crop(frame, zone)
        if crop is None or crop.size == 0:
            return {"label":"ERROR","score":0.0,
                    "detail":"Không cắt được vùng"}

        # Gọi tool tương ứng
        dispatcher = {
            "DefectDetection": self._run_defect,
            "ShapeSearch":     self._run_shape_search,
            "BlobAnalysis":    self._run_blob,
            "EdgeDetection":   self._run_edge,
            "BinaryCheck":     self._run_binary,
        }
        fn = dispatcher.get(tool, self._run_defect)
        return fn(crop, cfg)

    # ── Extract vùng từ frame ────────────────────────────
    def _extract_crop(self, frame: np.ndarray,
                      zone: dict):
        t = zone.get("type")
        d = zone.get("data", [])
        h, w = frame.shape[:2]
        try:
            if t == "rect":
                x0,y0 = max(0,d[0]),max(0,d[1])
                x1,y1 = min(w,d[2]),min(h,d[3])
                if x1>x0 and y1>y0:
                    return frame[y0:y1, x0:x1]
            elif t == "circle":
                cx,cy,r = d[0],d[1],d[2]
                x0 = max(0,cx-r); y0 = max(0,cy-r)
                x1 = min(w,cx+r); y1 = min(h,cy+r)
                crop = frame[y0:y1, x0:x1].copy()
                # Áp mask tròn
                mask = np.zeros(crop.shape[:2], np.uint8)
                ch, cw = crop.shape[:2]
                cv2.circle(mask, (cw//2,ch//2),
                           min(cw,ch)//2, 255, -1)
                return cv2.bitwise_and(
                    crop, crop, mask=mask)
            elif t == "rot_rect":
                # Crop bounding box của rotated rect
                box = cv2.boxPoints(
                    ((d[0],d[1]),(d[2],d[3]),d[4]))
                xmin = max(0, int(box[:,0].min()))
                ymin = max(0, int(box[:,1].min()))
                xmax = min(w, int(box[:,0].max()))
                ymax = min(h, int(box[:,1].max()))
                return frame[ymin:ymax, xmin:xmax]
            elif t == "polygon":
                pts  = np.array(d, dtype=np.int32)
                xmin = max(0, pts[:,0].min())
                ymin = max(0, pts[:,1].min())
                xmax = min(w, pts[:,0].max())
                ymax = min(h, pts[:,1].max())
                crop = frame[ymin:ymax,
                             xmin:xmax].copy()
                # Áp polygon mask
                mask = np.zeros(crop.shape[:2], np.uint8)
                shifted = pts - np.array([xmin, ymin])
                cv2.fillPoly(mask, [shifted], 255)
                return cv2.bitwise_and(
                    crop, crop, mask=mask)
        except Exception:
            pass
        return None

    # ── DefectDetection ──────────────────────────────────
    def _run_defect(self, crop: np.ndarray,
                    cfg: dict) -> dict:
        """Phát hiện lỗi bề mặt bằng gradient"""
        try:
            k = int(cfg.get("blur_ksize", 5))
            k = k if k % 2 == 1 else k + 1
            gray = cv2.cvtColor(
                crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(
                gray, (k, k), 0)

            # CLAHE
            clip = float(cfg.get("clahe_clip", 4.0))
            clahe = cv2.createCLAHE(
                clipLimit=clip,
                tileGridSize=(8,8))
            gray = clahe.apply(gray)

            # Gradient
            method = cfg.get("grad_method", "sobel")
            if method == "sobel":
                gx   = cv2.Sobel(
                    gray, cv2.CV_32F, 1, 0, ksize=3)
                gy   = cv2.Sobel(
                    gray, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(gx, gy)
            elif method == "canny":
                t1 = int(cfg.get("canny_thr1", 50))
                t2 = int(cfg.get("canny_thr2", 150))
                grad = cv2.Canny(
                    gray, t1, t2).astype(np.float32)
            else:
                gx   = cv2.Sobel(
                    gray, cv2.CV_32F, 1, 0, ksize=3)
                gy   = cv2.Sobel(
                    gray, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(gx, gy)

            cv2.normalize(grad, grad, 0, 255,
                          cv2.NORM_MINMAX)
            grad = grad.astype(np.uint8)

            thr = int(cfg.get("grad_thr", 18))
            _, low = cv2.threshold(
                grad, thr, 255,
                cv2.THRESH_BINARY_INV)

            # Morphology
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5,5))
            oi = int(cfg.get("morph_open_iter", 2))
            ci = int(cfg.get("morph_close_iter", 3))
            cleaned = cv2.morphologyEx(
                low, cv2.MORPH_OPEN, kernel,
                iterations=oi)
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_CLOSE, kernel,
                iterations=ci)

            total = cleaned.size
            ok_px = np.count_nonzero(cleaned)
            ratio = ok_px / max(total, 1)

            ok_r = float(cfg.get("ok_ratio", 0.98))
            label = "OK" if ratio >= ok_r else "NG"

            return {
                "label":  label,
                "score":  ratio,
                "detail": f"ratio={ratio:.3f} "
                          f">= {ok_r} → {label}",
            }
        except Exception as e:
            return {"label":"ERROR","score":0.0,
                    "detail":str(e)}

    # ── ShapeSearch ──────────────────────────────────────
    def _run_shape_search(self, crop: np.ndarray,
                          cfg: dict) -> dict:
        """Tìm kiếm mẫu trong vùng"""
        try:
            tpl_path = cfg.get("template_path", "")
            if not tpl_path or \
                    not Path(tpl_path).exists():
                return {"label":"NG","score":0.0,
                        "detail":"Chưa có mẫu template"}

            template = cv2.imread(tpl_path)
            if template is None:
                return {"label":"NG","score":0.0,
                        "detail":"Không đọc được mẫu"}

            # Dùng ShapeMatcher từ shape_search.py
            from shape_search import ShapeMatcher
            matcher = ShapeMatcher(cfg)
            result  = matcher.match(template, crop)
            return {
                "label":  result["label"],
                "score":  result["score"],
                "detail": result["details"],
            }
        except Exception as e:
            return {"label":"ERROR","score":0.0,
                    "detail":str(e)}

    # ── BlobAnalysis ─────────────────────────────────────
    def _run_blob(self, crop: np.ndarray,
                  cfg: dict) -> dict:
        """Đếm và phân tích blob"""
        try:
            gray = cv2.cvtColor(
                crop, cv2.COLOR_BGR2GRAY)
            thr   = int(cfg.get("thresh", 128))
            ttype = cfg.get("thresh_type", "binary")

            if ttype == "otsu":
                _, binary = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif ttype == "binary_inv":
                _, binary = cv2.threshold(
                    gray, thr, 255,
                    cv2.THRESH_BINARY_INV)
            else:
                _, binary = cv2.threshold(
                    gray, thr, 255,
                    cv2.THRESH_BINARY)

            n, _, stats, _ = \
                cv2.connectedComponentsWithStats(
                    binary, connectivity=8)

            min_a = int(cfg.get("min_area", 100))
            max_a = int(cfg.get("max_area", 50000))

            valid = [
                i for i in range(1, n)
                if min_a <= stats[i,
                                  cv2.CC_STAT_AREA]
                         <= max_a
            ]
            count = len(valid)

            min_c = int(cfg.get("min_count", 1))
            max_c = int(cfg.get("max_count", -1))

            ok = count >= min_c
            if max_c >= 0:
                ok = ok and count <= max_c

            score = min(count / max(min_c, 1), 1.0) \
                    if ok else 0.0
            return {
                "label":  "OK" if ok else "NG",
                "score":  score,
                "detail": f"blobs={count} "
                          f"[{min_c}~"
                          f"{'∞' if max_c<0 else max_c}]",
            }
        except Exception as e:
            return {"label":"ERROR","score":0.0,
                    "detail":str(e)}

    # ── EdgeDetection ────────────────────────────────────
    def _run_edge(self, crop: np.ndarray,
                  cfg: dict) -> dict:
        """Phát hiện biên"""
        try:
            gray = cv2.cvtColor(
                crop, cv2.COLOR_BGR2GRAY)
            t1   = int(cfg.get("canny_thr1", 50))
            t2   = int(cfg.get("canny_thr2", 150))
            edges = cv2.Canny(gray, t1, t2)

            total      = edges.size
            edge_px    = np.count_nonzero(edges)
            ratio      = edge_px / max(total, 1)
            ok_ratio   = float(
                cfg.get("ok_edge_ratio", 0.8))
            label      = "OK" \
                         if ratio >= ok_ratio \
                         else "NG"
            return {
                "label":  label,
                "score":  ratio,
                "detail": f"edge_ratio={ratio:.3f}",
            }
        except Exception as e:
            return {"label":"ERROR","score":0.0,
                    "detail":str(e)}

    # ── BinaryCheck ──────────────────────────────────────
    
    
    def _run_binary(self, crop: np.ndarray,
                cfg: dict) -> dict:
        """Kiểm tra tỉ lệ nhị phân"""
        try:
            gray  = cv2.cvtColor(
                crop, cv2.COLOR_BGR2GRAY)
            thr   = int(cfg.get("thresh", 128))
            ttype = cfg.get("thresh_type", "otsu")
            inv   = bool(cfg.get("invert", False))

            # ── Threshold ────────────────────────────
            if ttype == "otsu":
                _, binary = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif ttype == "binary_inv":
                _, binary = cv2.threshold(
                    gray, thr, 255,
                    cv2.THRESH_BINARY_INV)
            else:
                _, binary = cv2.threshold(
                    gray, thr, 255,
                    cv2.THRESH_BINARY)

            if inv:
                binary = cv2.bitwise_not(binary)

            # ── Tính ratio ───────────────────────────
            total = binary.size
            white = np.count_nonzero(binary)
            black = total - white
            ratio = white / max(total, 1)
            ok_r  = float(cfg.get("ok_ratio", 0.9))
            label = "OK" if ratio >= ok_r else "NG"
            col   = (0, 220, 0) \
                if label == "OK" else (0, 60, 220)

            # ── Panel 3: Binary colormap ─────────────
            # Trắng = white pixel (xanh lá)
            # Đen   = black pixel (đỏ)
            p_color = np.zeros(
                (binary.shape[0],
                binary.shape[1], 3),
                np.uint8)
            p_color[binary >  0] = (80, 200,  80)  # trắng → xanh lá
            p_color[binary == 0] = (60,  60, 200)  # đen   → xanh dương

            # ── Panel 4: Overlay tỉ lệ trên ảnh gốc ─
            p_overlay = crop.copy()
            p_overlay[binary >  0] = (
                p_overlay[binary >  0] * 0.5 +
                np.array([80, 200, 80]) * 0.5
            ).astype(np.uint8)
            p_overlay[binary == 0] = (
                p_overlay[binary == 0] * 0.5 +
                np.array([60, 60, 200]) * 0.5
            ).astype(np.uint8)
            cv2.putText(p_overlay,
                        f"{label} {ratio*100:.1f}%",
                        (4, p_overlay.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, 1, cv2.LINE_AA)

            # ── Panel 5: Histogram bar ───────────────
            bar_w, bar_h = 200, 80
            bar = np.zeros((bar_h, bar_w, 3), np.uint8)
            bar[:] = (28, 28, 40)

            # White bar
            w_fill = int(ratio * bar_w)
            cv2.rectangle(bar,
                        (0,     bar_h//4),
                        (w_fill, 3*bar_h//4),
                        (80, 200, 80), -1)
            # Threshold line
            thr_x = int(ok_r * bar_w)
            cv2.line(bar,
                    (thr_x, 0),
                    (thr_x, bar_h),
                    (255, 255, 0), 2)
            # Text
            cv2.putText(bar,
                        f"white: {ratio*100:.1f}%",
                        (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)
            cv2.putText(bar,
                        f"thr:   {ok_r*100:.0f}%",
                        (4, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 0), 1)
            cv2.putText(bar,
                        f"black: {black/max(total,1)*100:.1f}%",
                        (4, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 200), 1)
            cv2.putText(bar,
                        f"[{label}]",
                        (4, bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 2)

            # ── Panel 6: Histogram equalized ─────────
            eq = cv2.equalizeHist(gray)

            # ── Danh sách panels ─────────────────────
            panels = [
                crop.copy(),          # 1. Original
                self._gray_bgr(gray), # 2. Gray
                self._gray_bgr(binary),# 3. Binary B&W
                p_overlay,            # 4. Overlay ← CUỐI = hiển thị bên phải
            ]
            titles = [
                "1.Original",
                "2.Gray",
                "3.Binary",
                "4.Overlay",
            ]
            detail = (f"white={ratio*100:.1f}%"
                    f">={ok_r*100:.0f}%"
                    f"  black={black/max(total,1)*100:.1f}%")

            # ── Tạo grid phân tích ───────────────────
            grid = self._make_analysis_grid(
                panels, titles, label, detail)

            return {
                "label":   label,
                "score":   ratio,
                "detail":  detail,
                "panels":  panels,   # ✅ Danh sách ảnh
                "grid":    grid,     # ✅ Ảnh grid 3×2
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "label":   "ERROR",
                "score":   0.0,
                "detail":  str(e),
                "panels":  None,
                "grid":    None,
            }