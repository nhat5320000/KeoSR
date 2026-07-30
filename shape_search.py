# ════════════════════════════════════════════════════════════════
# shape_search.py  -  Shape Search Tool
# Tìm kiếm vùng ảnh khớp với mẫu (kể cả khi mẫu bị xoay)
# ════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import tkinter.simpledialog

# ════════════════════════════════════════════════════════════════
# CAU HINH MAC DINH
# ════════════════════════════════════════════════════════════════
DEFAULT_SS_CFG = {
    "match_ratio":        0.75,
    "match_method":       "ORB",
    "rotate_check":       True,
    "rotate_step":        15,
    "scale_check":        False,
    "scale_min":          0.8,
    "scale_max":          1.2,
    "scale_step":         0.1,
    "orb_nfeatures":      500,
    "min_match_count":    10,
    "template_threshold": 0.8,
    "blur_ksize":         3,
    "preprocess":         "gray",
    "output_dir":         "shape_search_results",
    "template_dir":       "shape_templates",
    "n_threads":          4,
    "camera_id":          0,
    "camera_fps":         30,
}

SS_CFG_FILE  = "shape_search_config.json"
TEMPLATE_DIR = Path("shape_templates")
TEMPLATE_DIR.mkdir(exist_ok=True)


def load_ss_cfg():
    if Path(SS_CFG_FILE).exists():
        with open(SS_CFG_FILE) as f:
            return {**DEFAULT_SS_CFG, **json.load(f)}
    return DEFAULT_SS_CFG.copy()


def save_ss_cfg(cfg):
    with open(SS_CFG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ════════════════════════════════════════════════════════════════
# SHAPE MATCHER
# ════════════════════════════════════════════════════════════════
class ShapeMatcher:
    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self._orb     = None
        self._sift    = None
        self._matcher = None
        self._init_detectors()

    def _init_detectors(self):
        method = self.cfg.get("match_method", "ORB")
        if method == "ORB":
            nf = int(self.cfg.get("orb_nfeatures", 500))
            self._orb     = cv2.ORB_create(nfeatures=nf)
            self._matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING, crossCheck=True)
        elif method == "SIFT":
            try:
                self._sift    = cv2.SIFT_create()
                self._matcher = cv2.BFMatcher(
                    cv2.NORM_L2, crossCheck=False)
            except Exception:
                nf = int(self.cfg.get("orb_nfeatures", 500))
                self._orb     = cv2.ORB_create(nfeatures=nf)
                self._matcher = cv2.BFMatcher(
                    cv2.NORM_HAMMING, crossCheck=True)
                self.cfg["match_method"] = "ORB"

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        pp = self.cfg.get("preprocess", "gray")
        k  = int(self.cfg.get("blur_ksize", 3))
        k  = k if k % 2 == 1 else k + 1
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if pp == "edge":
            return cv2.Canny(gray, 50, 150)
        return gray

    def _rotate_image(self, img: np.ndarray,
                      angle: float) -> np.ndarray:
        h, w   = img.shape[:2]
        cx, cy = w // 2, h // 2
        M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _scale_image(self, img: np.ndarray,
                     scale: float) -> np.ndarray:
        h, w = img.shape[:2]
        nw   = max(1, int(w * scale))
        nh   = max(1, int(h * scale))
        return cv2.resize(img, (nw, nh),
                          interpolation=cv2.INTER_LINEAR)

    def _match_orb_sift(self,
                        tpl_gray:   np.ndarray,
                        scene_gray: np.ndarray
                        ) -> tuple:
        detector = self._sift if self._sift else self._orb
        kp1, des1 = detector.detectAndCompute(tpl_gray,   None)
        kp2, des2 = detector.detectAndCompute(scene_gray, None)

        if des1 is None or des2 is None or len(kp1) == 0:
            return 0.0, []

        if (self.cfg.get("match_method") == "SIFT"
                and self._sift is not None):
            matches = self._matcher.knnMatch(des1, des2, k=2)
            good    = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
        else:
            raw     = self._matcher.match(des1, des2)
            good    = sorted(raw, key=lambda x: x.distance)

        min_match = int(self.cfg.get("min_match_count", 10))
        if len(good) < min_match:
            return 0.0, good

        score = len(good) / max(len(kp1), 1)
        return min(score, 1.0), good

    def _match_template(self,
                        tpl_gray:   np.ndarray,
                        scene_gray: np.ndarray
                        ) -> tuple:
        th, tw = tpl_gray.shape[:2]
        sh, sw = scene_gray.shape[:2]
        if th > sh or tw > sw:
            return 0.0, None
        res = cv2.matchTemplate(
            scene_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return float(max_val), max_loc

    def match(self,
              template: np.ndarray,
              scene:    np.ndarray) -> dict:
        method    = self.cfg.get("match_method", "ORB")
        do_rotate = bool(self.cfg.get("rotate_check", True))
        do_scale  = bool(self.cfg.get("scale_check",  False))
        threshold = float(self.cfg.get("match_ratio",  0.75))

        tpl_gray   = self._preprocess(template)
        scene_gray = self._preprocess(scene)

        best_score = 0.0
        best_angle = 0.0
        best_scale = 1.0
        best_loc   = None

        angles = [0.0]
        if do_rotate:
            step   = float(self.cfg.get("rotate_step", 15))
            angles = list(np.arange(0, 360, step))

        scales = [1.0]
        if do_scale:
            s_min  = float(self.cfg.get("scale_min",  0.8))
            s_max  = float(self.cfg.get("scale_max",  1.2))
            s_step = float(self.cfg.get("scale_step", 0.1))
            scales = list(np.arange(s_min,
                                    s_max + 0.001, s_step))

        for scale in scales:
            tpl_s = (self._scale_image(tpl_gray, scale)
                     if scale != 1.0 else tpl_gray)
            for angle in angles:
                tpl_r = (self._rotate_image(tpl_s, angle)
                         if angle != 0.0 else tpl_s)
                if method == "TEMPLATE":
                    score, loc = self._match_template(
                        tpl_r, scene_gray)
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                        best_scale = scale
                        best_loc   = loc
                else:
                    score, _ = self._match_orb_sift(
                        tpl_r, scene_gray)
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                        best_scale = scale

        found   = best_score >= threshold
        details = (f"Method={method}  "
                   f"Score={best_score:.3f}  "
                   f"Thr={threshold:.2f}  "
                   f"Angle={best_angle:.0f}  "
                   f"Scale={best_scale:.2f}")
        return {
            "found":    found,
            "score":    best_score,
            "angle":    best_angle,
            "scale":    best_scale,
            "location": best_loc,
            "details":  details,
            "label":    "OK" if found else "NG",
        }


# ════════════════════════════════════════════════════════════════
# CAMERA THREAD
# ════════════════════════════════════════════════════════════════
class SSCameraThread(threading.Thread):
    def __init__(self, cam_id: int, fps: int):
        super().__init__(daemon=True)
        self.cam_id     = cam_id
        self.fps        = fps
        self.buffer     = []
        self.running    = False
        self.capturing  = False
        self._lock      = threading.Lock()
        self.last_frame = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True
        interval     = 1.0 / max(self.fps, 1)
        while self.running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame.copy()
                if self.capturing:
                    with self._lock:
                        self.buffer.append(frame.copy())
            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        cap.release()

    def start_capture(self):
        with self._lock:
            self.buffer.clear()
        self.capturing = True

    def stop_capture(self):
        self.capturing = False
        with self._lock:
            frames = self.buffer.copy()
            self.buffer.clear()
        return frames

    def stop(self):
        self.running = self.capturing = False


# ════════════════════════════════════════════════════════════════
# SHAPE SEARCH APP
# ════════════════════════════════════════════════════════════════
class ShapeSearchApp(tk.Toplevel):
    BG       = "#1a1a2e"
    FG       = "#cdd6f4"
    ACC      = "#89b4fa"
    ENTRY_BG = "#313244"

    def __init__(self, parent=None, cam_thread=None):
        if parent is not None:
            super().__init__(parent)
        else:
            root = tk.Tk()
            root.withdraw()
            super().__init__(root)

        self.title("Shape Search - Tim kiem hinh dang")
        self.configure(bg=self.BG)
        self.geometry("1200x720")
        self.resizable(True, True)

        self.cfg           = load_ss_cfg()
        self.cam_thread    = cam_thread
        self._own_camera   = cam_thread is None
        self._cam          = None
        self.template      = None
        self.template_name = ""
        self.matcher       = ShapeMatcher(self.cfg)
        self._running      = True
        self._processing   = False
        self._session_dt   = datetime.now()
        self._ok_count     = 0
        self._ng_count     = 0
        self._results      = []
        self._cur_idx      = 0

        self.var_status   = tk.StringVar(value="San sang")
        self.var_template = tk.StringVar(value="-- chua co mau --")
        self.var_ok       = tk.StringVar(value="OK: 0")
        self.var_ng       = tk.StringVar(value="NG: 0")
        self.var_ratio    = tk.StringVar(
            value=str(self.cfg.get("match_ratio", 0.75)))

        self._build_ui()
        self._start_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if self._own_camera:
            self._start_own_camera()

    # ════════════════════════════════════════════════════════
    # BUILD UI
    # ════════════════════════════════════════════════════════
    def _build_ui(self):
        BG, FG, ACC = self.BG, self.FG, self.ACC

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel",      background=BG, foreground=FG)
        style.configure("TFrame",      background=BG)
        style.configure("TLabelframe", background=BG, foreground=ACC)
        style.configure("TLabelframe.Label",
                        background=BG, foreground=ACC,
                        font=("Consolas", 10, "bold"))

        root_f = ttk.Frame(self)
        root_f.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(root_f)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(root_f, bg=BG, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        cf = ttk.LabelFrame(left, text=" Camera Preview ")
        cf.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.lbl_preview = tk.Label(cf, bg="black")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True,
                              padx=2, pady=2)

        bar = tk.Frame(left, bg="#252535",
                       relief="groove", bd=1)
        bar.pack(fill=tk.X, pady=2)
        tk.Label(bar, text=" Ket qua ",
                 bg="#252535", fg=ACC,
                 font=("Consolas", 9, "bold")
                 ).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=2, pady=3)
        tk.Button(bar, text="<",
                  font=("Consolas", 10, "bold"),
                  bg=self.ENTRY_BG, fg=FG,
                  relief="flat", width=2,
                  command=self._prev_result
                  ).pack(side=tk.LEFT, padx=1, pady=2)
        tk.Button(bar, text=">",
                  font=("Consolas", 10, "bold"),
                  bg=self.ENTRY_BG, fg=FG,
                  relief="flat", width=2,
                  command=self._next_result
                  ).pack(side=tk.LEFT, padx=(1, 4), pady=2)

        self.lbl_ok = tk.Label(bar,
                               textvariable=self.var_ok,
                               bg="#252535", fg="#a6e3a1",
                               font=("Consolas", 10, "bold"),
                               padx=6)
        self.lbl_ok.pack(side=tk.LEFT, pady=2)

        self.lbl_ng = tk.Label(bar,
                               textvariable=self.var_ng,
                               bg="#252535", fg="#f38ba8",
                               font=("Consolas", 10, "bold"),
                               padx=6)
        self.lbl_ng.pack(side=tk.LEFT, pady=2)

        self.verdict_box = tk.Label(
            bar, text=" - ",
            font=("Consolas", 14, "bold"),
            bg="#252535", fg=FG,
            width=6, padx=8, pady=4,
            relief="flat", anchor="center")
        self.verdict_box.pack(side=tk.RIGHT, padx=4, pady=2)

        rf = tk.Frame(left, bg="black",
                      relief="groove", bd=1)
        rf.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.lbl_result = tk.Label(rf, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)

        tk.Label(left, textvariable=self.var_status,
                 bg=BG, fg=ACC,
                 font=("Consolas", 9), anchor="w"
                 ).pack(fill=tk.X)

        self._build_right(right)

    def _build_right(self, parent):
        BG, FG, ACC = self.BG, self.FG, self.ACC

        # ── Template section ─────────────────────────────
        tpl_lf = ttk.LabelFrame(parent, text=" Template ")
        tpl_lf.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(tpl_lf, textvariable=self.var_template,
                 bg=BG, fg="#f9e2af",
                 font=("Consolas", 8),
                 anchor="w", wraplength=270
                 ).pack(fill=tk.X, padx=4, pady=2)

        self.lbl_template = tk.Label(
            tpl_lf, bg="#0d0d1a",
            width=280, height=100)
        self.lbl_template.pack(padx=4, pady=4)

        fr_tpl = tk.Frame(tpl_lf, bg=BG)
        fr_tpl.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(fr_tpl, text="Chup mau",
                  bg="#89dceb", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", pady=5,
                  command=self._capture_template
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)
        tk.Button(fr_tpl, text="Tai mau",
                  bg="#cba6f7", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", pady=5,
                  command=self._load_template
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)
        tk.Button(fr_tpl, text="Luu mau",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", pady=5,
                  command=self._save_template
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)

        tk.Label(tpl_lf, text="Mau da luu:",
                 bg=BG, fg=ACC,
                 font=("Consolas", 8, "bold"),
                 anchor="w"
                 ).pack(fill=tk.X, padx=4)

        fr_list = tk.Frame(tpl_lf, bg=BG)
        fr_list.pack(fill=tk.X, padx=4, pady=(0, 4))
        sb_tpl  = ttk.Scrollbar(fr_list, orient="vertical")
        self.lb_templates = tk.Listbox(
            fr_list, bg="#181825", fg=FG,
            font=("Consolas", 8),
            selectbackground="#89b4fa",
            selectforeground="black",
            relief="flat", height=4,
            yscrollcommand=sb_tpl.set,
            activestyle="dotbox")
        sb_tpl.config(command=self.lb_templates.yview)
        sb_tpl.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_templates.pack(side=tk.LEFT,
                               fill=tk.BOTH, expand=True)
        self.lb_templates.bind(
            "<Double-Button-1>",
            lambda e: self._load_template_from_list())

        # ── Config section ────────────────────────────────
        cfg_lf = ttk.LabelFrame(parent, text=" Cai dat ")
        cfg_lf.pack(fill=tk.X, padx=4, pady=4)

        self._var_method    = tk.StringVar(
            value=self.cfg.get("match_method", "ORB"))
        self._var_ratio     = tk.DoubleVar(
            value=self.cfg.get("match_ratio", 0.75))
        self._var_rotate    = tk.BooleanVar(
            value=self.cfg.get("rotate_check", True))
        self._var_r_step    = tk.IntVar(
            value=self.cfg.get("rotate_step", 15))
        self._var_scale     = tk.BooleanVar(
            value=self.cfg.get("scale_check", False))
        self._var_preproc   = tk.StringVar(
            value=self.cfg.get("preprocess", "gray"))
        self._var_min_match = tk.IntVar(
            value=self.cfg.get("min_match_count", 10))
        self._var_nfeat     = tk.IntVar(
            value=self.cfg.get("orb_nfeatures", 500))

        # Combobox method
        f = tk.Frame(cfg_lf, bg=BG)
        f.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f, text="Phuong phap:",
                 bg=BG, fg=FG,
                 font=("Consolas", 8),
                 width=16, anchor="w"
                 ).pack(side=tk.LEFT)
        ttk.Combobox(f, textvariable=self._var_method,
                     values=["ORB", "SIFT", "TEMPLATE"],
                     state="readonly",
                     font=("Consolas", 8), width=12
                     ).pack(side=tk.LEFT)

        # Entry rows
        for lbl, var in [
            ("Match ratio",  self._var_ratio),
            ("Buoc xoay",    self._var_r_step),
            ("Min matches",  self._var_min_match),
            ("ORB features", self._var_nfeat),
        ]:
            f2 = tk.Frame(cfg_lf, bg=BG)
            f2.pack(fill=tk.X, padx=4, pady=2)
            tk.Label(f2, text=f"{lbl}:",
                     bg=BG, fg=FG,
                     font=("Consolas", 8),
                     width=16, anchor="w"
                     ).pack(side=tk.LEFT)
            tk.Entry(f2, textvariable=var,
                     bg=self.ENTRY_BG, fg=FG,
                     insertbackground=FG,
                     font=("Consolas", 8),
                     width=8, relief="flat"
                     ).pack(side=tk.LEFT)

        # Checkbutton rows
        for lbl, var in [
            ("Kiem tra xoay",  self._var_rotate),
            ("Kiem tra scale", self._var_scale),
        ]:
            f3 = tk.Frame(cfg_lf, bg=BG)
            f3.pack(fill=tk.X, padx=4, pady=2)
            tk.Label(f3, text=f"{lbl}:",
                     bg=BG, fg=FG,
                     font=("Consolas", 8),
                     width=16, anchor="w"
                     ).pack(side=tk.LEFT)
            tk.Checkbutton(f3, variable=var,
                           bg=BG, fg=ACC,
                           selectcolor="#313244",
                           activebackground=BG,
                           relief="flat"
                           ).pack(side=tk.LEFT)

        # Combobox preprocess
        f4 = tk.Frame(cfg_lf, bg=BG)
        f4.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f4, text="Tien xu ly:",
                 bg=BG, fg=FG,
                 font=("Consolas", 8),
                 width=16, anchor="w"
                 ).pack(side=tk.LEFT)
        ttk.Combobox(f4, textvariable=self._var_preproc,
                     values=["gray", "edge", "none"],
                     state="readonly",
                     font=("Consolas", 8), width=8
                     ).pack(side=tk.LEFT)

        tk.Button(cfg_lf, text="Luu cai dat",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", pady=4,
                  command=self._save_settings
                  ).pack(fill=tk.X, padx=4, pady=4)

        # ── Control section ───────────────────────────────
        ctrl_lf = ttk.LabelFrame(parent, text=" Dieu khien ")
        ctrl_lf.pack(fill=tk.X, padx=4, pady=4)

        for txt, bg, cmd in [
            ("BAT DAU CHUP", "#89dceb", self._start_capture),
            ("DUNG + XU LY", "#f38ba8", self._stop_and_process),
            ("TEST 1 ANH",   "#cba6f7", self._test_single),
            ("Xoa ket qua",  "#313244", self._clear_results),
        ]:
            tk.Button(ctrl_lf, text=txt, bg=bg, fg="black",
                      font=("Consolas", 10, "bold"),
                      relief="flat", padx=6, pady=6,
                      command=cmd
                      ).pack(fill=tk.X, padx=4, pady=2)

        tk.Label(parent,
                 text=str(self.cfg.get("output_dir", "")),
                 bg=BG, fg="#585b70",
                 font=("Consolas", 7),
                 anchor="w"
                 ).pack(fill=tk.X, padx=8, pady=2)

    # ════════════════════════════════════════════════════════
    # CAMERA PREVIEW
    # ════════════════════════════════════════════════════════
    def _start_preview(self):
        self._preview_running = True
        self.after(100, self._update_preview)

    def _update_preview(self):
        if not self._preview_running:
            return
        cam = self.cam_thread if not self._own_camera \
              else self._cam
        if cam and cam.last_frame is not None:
            frame = cam.last_frame.copy()
        else:
            frame = np.zeros((360, 480, 3), np.uint8)
            cv2.putText(frame, "No Camera", (140, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (80, 80, 80), 2)
        if cam and getattr(cam, "capturing", False):
            n = len(cam.buffer)
            cv2.putText(frame, f"REC {n} frames",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        self._show_frame(self.lbl_preview, frame,
                         max_w=700, max_h=300)
        self.after(40, self._update_preview)

    def _show_frame(self, widget, frame,
                    max_w=640, max_h=360):
        if frame is None:
            return
        h, w = frame.shape[:2]
        sc   = min(max_w / w, max_h / h, 1.0)
        if sc < 1.0:
            frame = cv2.resize(
                frame, None, fx=sc, fy=sc,
                interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        widget.configure(image=img)
        widget.image = img

    # ════════════════════════════════════════════════════════
    # TEMPLATE MANAGEMENT
    # ════════════════════════════════════════════════════════
    def _capture_template(self):
        cam = self.cam_thread if not self._own_camera \
              else self._cam
        if cam is None or cam.last_frame is None:
            messagebox.showwarning(
                "Chu y", "Camera chua san sang!",
                parent=self)
            return
        self.template      = cam.last_frame.copy()
        self.template_name = \
            f"captured_{datetime.now():%Y%m%d_%H%M%S}"
        self.var_template.set(f"[Chup] {self.template_name}")
        self._show_template_preview()
        self._set_status("Da chup mau tu camera")

    def _load_template(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Chon anh mau",
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror(
                "Loi", "Khong doc duoc anh!", parent=self)
            return
        self.template      = img
        self.template_name = Path(path).stem
        self.var_template.set(f"[File] {self.template_name}")
        self._show_template_preview()
        self._set_status(f"Da tai mau: {self.template_name}")

    def _save_template(self):
        if self.template is None:
            messagebox.showwarning(
                "Chu y", "Chua co mau de luu!", parent=self)
            return
        name = tkinter.simpledialog.askstring(
            "Ten mau", "Nhap ten mau:",
            initialvalue=self.template_name,
            parent=self)
        if not name:
            return
        path = TEMPLATE_DIR / f"{name}.jpg"
        cv2.imwrite(str(path), self.template,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.template_name = name
        self.var_template.set(f"[Luu] {name}")
        self._refresh_template_list()
        self._set_status(f"Da luu mau: {name}")

    def _load_template_from_list(self):
        sel = self.lb_templates.curselection()
        if not sel:
            return
        name = self.lb_templates.get(sel[0]).strip()
        path = TEMPLATE_DIR / f"{name}.jpg"
        if not path.exists():
            messagebox.showerror(
                "Loi", f"File khong ton tai: {path}",
                parent=self)
            return
        img = cv2.imread(str(path))
        if img is None:
            return
        self.template      = img
        self.template_name = name
        self.var_template.set(f"[DS] {name}")
        self._show_template_preview()
        self._set_status(f"Da load mau: {name}")

    def _refresh_template_list(self):
        self.lb_templates.delete(0, tk.END)
        for p in sorted(TEMPLATE_DIR.glob("*.jpg")):
            self.lb_templates.insert(tk.END, p.stem)

    def _show_template_preview(self):
        if self.template is None:
            return
        h, w = self.template.shape[:2]
        sc   = min(270 / w, 95 / h, 1.0)
        disp = cv2.resize(self.template, None,
                          fx=sc, fy=sc,
                          interpolation=cv2.INTER_AREA)
        rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pimg = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.lbl_template.configure(image=pimg)
        self.lbl_template._pimg = pimg

    # ════════════════════════════════════════════════════════
    # SETTINGS
    # ════════════════════════════════════════════════════════
    def _save_settings(self):
        try:
            self.cfg["match_method"]    = \
                self._var_method.get()
            self.cfg["match_ratio"]     = \
                float(self._var_ratio.get())
            self.cfg["rotate_check"]    = \
                bool(self._var_rotate.get())
            self.cfg["rotate_step"]     = \
                int(self._var_r_step.get())
            self.cfg["scale_check"]     = \
                bool(self._var_scale.get())
            self.cfg["preprocess"]      = \
                self._var_preproc.get()
            self.cfg["min_match_count"] = \
                int(self._var_min_match.get())
            self.cfg["orb_nfeatures"]   = \
                int(self._var_nfeat.get())
            save_ss_cfg(self.cfg)
            self.matcher = ShapeMatcher(self.cfg)
            self._set_status("Da luu cai dat")
        except Exception as e:
            messagebox.showerror(
                "Loi", f"Luu that bai: {e}", parent=self)

    # ════════════════════════════════════════════════════════
    # CAPTURE & PROCESS
    # ════════════════════════════════════════════════════════
    def _get_cam(self):
        return self.cam_thread \
               if not self._own_camera else self._cam

    def _start_capture(self):
        if self.template is None:
            messagebox.showwarning(
                "Chu y",
                "Chua co mau!\nHay chup hoac tai mau truoc.",
                parent=self)
            return
        cam = self._get_cam()
        if cam is None:
            messagebox.showwarning(
                "Chu y", "Camera chua san sang!",
                parent=self)
            return
        self._session_dt = datetime.now()
        cam.start_capture()
        self._set_status("DANG CHUP... Nhan DUNG de dung")

    def _stop_and_process(self):
        cam = self._get_cam()
        if cam is None or not cam.capturing:
            self._set_status("Chua bat dau chup!")
            return
        frames = cam.stop_capture()
        if not frames:
            messagebox.showwarning(
                "Chu y", "Khong co anh!", parent=self)
            return
        self._set_status(
            f"Dung: {len(frames)} anh - dang xu ly...")
        threading.Thread(
            target=self._process_frames,
            args=(frames,), daemon=True).start()

    def _test_single(self):
        if self.template is None:
            messagebox.showwarning(
                "Chu y", "Chua co mau!", parent=self)
            return
        cam = self._get_cam()
        if cam is None or cam.last_frame is None:
            messagebox.showwarning(
                "Chu y", "Camera chua san sang!",
                parent=self)
            return
        frame = cam.last_frame.copy()
        threading.Thread(
            target=self._process_frames,
            args=([frame],), daemon=True).start()

    def _process_frames(self, frames: list):
        if self._processing:
            return
        self._processing = True

        out_dir = Path(self.cfg["output_dir"])
        ok_dir  = out_dir / "OK"
        ng_dir  = out_dir / "NG"
        ok_dir.mkdir(parents=True, exist_ok=True)
        ng_dir.mkdir(parents=True, exist_ok=True)

        t0       = time.perf_counter()
        ok_count = 0
        ng_count = 0
        results  = []
        date     = self._session_dt.strftime("%y%m%d")
        hhmm     = self._session_dt.strftime("%H%M")

        def _process_one(args):
            i, frame = args
            result   = self.matcher.match(
                self.template, frame)
            annotated = self._annotate(frame.copy(), result)
            label     = result["label"]
            fname     = (f"shape_{date}_{hhmm}_"
                         f"{label}_{i+1:03d}.jpg")
            save_dir  = ok_dir if label == "OK" else ng_dir
            cv2.imwrite(str(save_dir / fname), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                **result,
                "frame":    annotated,
                "filename": fname,
                "index":    i,
            }

        n_threads = int(self.cfg.get("n_threads", 4))
        with ThreadPoolExecutor(
                max_workers=n_threads) as exe:
            for res in exe.map(_process_one,
                               enumerate(frames)):
                results.append(res)
                if res["label"] == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
                self.after(0, lambda r=res:
                           self._on_result(r))

        total = time.perf_counter() - t0
        self._ok_count += ok_count
        self._ng_count += ng_count
        self._results  += results

        self.after(0, self._update_counts)
        self.after(0, lambda: self._show_result(
            len(self._results) - 1))
        self.after(0, lambda: self._set_status(
            f"Xong! OK:{ok_count} NG:{ng_count} "
            f"Tong:{total:.2f}s"))
        self._processing = False

    def _on_result(self, result: dict):
        label = result["label"]
        score = result["score"]
        self._set_status(
            f"{'OK' if label == 'OK' else 'NG'} "
            f"score={score:.3f} "
            f"{result['details'][:60]}")

    def _update_counts(self):
        self.var_ok.set(f"OK: {self._ok_count}")
        self.var_ng.set(f"NG: {self._ng_count}")
        self.lbl_ng.configure(
            bg="#4a1a1a" if self._ng_count > 0
            else "#252535",
            fg="#ff6b6b" if self._ng_count > 0
            else "#f38ba8")
        if self._ng_count > 0:
            self.verdict_box.configure(
                text=" NG ", bg="#f38ba8", fg="#1e1e2e")
        elif self._ok_count > 0:
            self.verdict_box.configure(
                text=" OK ", bg="#a6e3a1", fg="#1e1e2e")
        else:
            self.verdict_box.configure(
                text="  -  ", bg="#252535", fg=self.FG)

    def _annotate(self, frame: np.ndarray,
                  result: dict) -> np.ndarray:
        label = result["label"]
        score = result["score"]
        angle = result["angle"]
        loc   = result.get("location")
        color = (0, 255, 0) if label == "OK" \
                else (0, 0, 255)
        h, w  = frame.shape[:2]
        if loc is not None and self.template is not None:
            th, tw = self.template.shape[:2]
            x, y   = loc
            cv2.rectangle(frame,
                          (x, y), (x + tw, y + th),
                          color, 3)
        lines = [
            f"{label}  {score*100:.1f}%",
            f"Angle={angle:.0f}  "
            f"Method={self.cfg.get('match_method','ORB')}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                        (10, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 4)
        return frame

    # ════════════════════════════════════════════════════════
    # NAVIGATION
    # ════════════════════════════════════════════════════════
    def _show_result(self, idx: int):
        if not self._results:
            return
        idx = max(0, min(idx, len(self._results) - 1))
        self._cur_idx = idx
        frame = self._results[idx].get("frame")
        if frame is not None:
            self._show_frame(self.lbl_result, frame,
                             max_w=800, max_h=400)

    def _prev_result(self):
        if self._results:
            self._show_result(self._cur_idx - 1)

    def _next_result(self):
        if self._results:
            self._show_result(self._cur_idx + 1)

    def _clear_results(self):
        self._results  = []
        self._ok_count = 0
        self._ng_count = 0
        self._cur_idx  = 0
        self._update_counts()
        self.lbl_result.configure(image="")
        self.lbl_result.image = None
        self._set_status("Da xoa ket qua")

    # ════════════════════════════════════════════════════════
    # OWN CAMERA
    # ════════════════════════════════════════════════════════
    def _start_own_camera(self):
        cam_id    = int(self.cfg.get("camera_id", 0))
        fps       = int(self.cfg.get("camera_fps", 30))
        self._cam = SSCameraThread(cam_id, fps)
        self._cam.start()
        self._set_status(f"Camera {cam_id} khoi dong...")

    # ════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════
    def _set_status(self, msg: str):
        self.after(0, lambda: self.var_status.set(msg))

    def _on_close(self):
        self._preview_running = False
        if self._own_camera and self._cam:
            self._cam.stop()
        self.destroy()


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════
def run_standalone():
    root = tk.Tk()
    root.withdraw()
    app  = ShapeSearchApp(parent=root)
    app.mainloop()


if __name__ == "__main__":
    run_standalone()