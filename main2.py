# ════════════════════════════════════════════════════════════════
# main.py  -  GUI + Camera + Inspector + ROI + Config + Tools
# Optimized for Jetson Orin Nano (Ubuntu 20.04/22.04 + JetPack)
# ════════════════════════════════════════════════════════════════
import os
import sys

# ════════════════════════════════════════════════════════════════
# ENV SETUP TRUOC KHI IMPORT BAT CU THU GI
# ════════════════════════════════════════════════════════════════
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS",      "2")
os.environ.setdefault("MKL_NUM_THREADS",      "2")
os.environ.setdefault("CUDA_MODULE_LOADING",  "EAGER")
os.environ.setdefault("GST_DEBUG",            "0")

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
import math
import gc
import signal
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime
import inspector as insp
from shape_search import ShapeSearchApp

DEFAULT_CFG = {
    "camera_id":          0,
    "camera_fps":         30,
    "camera_type":        "IMV",
    "camera_width":       1280,
    "camera_height":      720,
    "camera_exposure":    -1,
    "camera_gain":        -1,
    "camera_brightness":  0,
    "camera_contrast":    0,
    "camera_saturation":  0,
    "camera_autofocus":   True,
    "camera_flip":        0,
    "camera_ip_url":      "rtsp://",
    "camera_csi_id":      0,
    "camera_buffer_size": 1,
    "blur_ksize":            5,
    "blur_sigma":            0,
    "bilateral_enable":      True,
    "bilateral_d":           11,
    "bilateral_sc":          75,
    "bilateral_ss":          75,
    "preprocess_order":      "blur_bilateral",
    "clahe_enable":          True,
    "clahe_clip":            4.0,
    "clahe_grid_x":          14,
    "clahe_grid_y":          14,
    "grad_thr":              18,
    "grad_method":           "sobel",
    "sobel_ksize":           3,
    "canny_thr1":            50,
    "canny_thr2":            150,
    "grad_thresh_type":      "binary_inv",
    "adaptive_block":        11,
    "adaptive_c":            2,
    "min_grad_area":         50,
    "morph_open_iter":       2,
    "morph_close_iter":      3,
    "morph_shape":           "ellipse",
    "morph_extra_enable":    False,
    "morph_extra_type":      "dilate",
    "morph_extra_iter":      1,
    "roi_thresh":            180,
    "roi_thresh_type":       "binary",
    "roi_offset":            75,
    "roi_min_area":          1000,
    "roi_contour_approx":    False,
    "ok_ratio":              0.98,
    "noise_max_area":        2500,
    "border_noise_max_area": 2500,
    "min_defect_area":       10,
    "green_dilate_k":        7,
    "sep_open_k":            3,
    "ring_dilate_k":         3,
    "max_defect_count":      -1,
    "out_q":                 85,
    "max_w":                 1600,
    "n_threads":             4,
    "input_dir":             "input_images",
    "output_dir":            "output_results",
    "roi_shapes":            [],
    "roi_crop":              None,
    "roi_brightness":        0,
    "roi_contrast":          1.0,
    "plc_enable":   True,          # Bat/tat PLC
    "plc_host":     "192.168.4.20", # IP PLC
    "plc_port":     502,            # Port Modbus
    "plc_unit_id":  1,              # Unit ID
    "camera_max_frames": 300,   # Max frames capture
    "save_ok_images": True,   # Luu anh OK
    "save_ng_images": True,   # Luu anh NG
    "capture_interval_ms": 0,
}

CFG_FILE   = "config.json"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_cfg():
    if Path(CFG_FILE).exists():
        try:
            with open(CFG_FILE) as f:
                return {**DEFAULT_CFG, **json.load(f)}
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] load_cfg error: {e}")
    return DEFAULT_CFG.copy()


def save_cfg(cfg):
    try:
        with open(CFG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except IOError as e:
        print(f"[WARN] save_cfg error: {e}")


def save_model(model_name, cfg):
    path = MODELS_DIR / f"{model_name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_model(model_name):
    path = MODELS_DIR / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' khong ton tai!")
    with open(path) as f:
        return {**DEFAULT_CFG, **json.load(f)}


def list_models():
    return [p.stem for p in sorted(MODELS_DIR.glob("*.json"))]


# ════════════════════════════════════════════════════════════════
# CAMERA THREAD
# ════════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    MAX_BUFFER = 300

    def __init__(self, cfg):
        super().__init__(daemon=True, name="CameraThread")
        self.cfg         = cfg
        self._buffer     = []
        self.running     = False
        self.capturing   = False
        self._lock       = threading.Lock()
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._cap        = None

    @property
    def last_frame(self):
        with self._frame_lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    @last_frame.setter
    def last_frame(self, frame):
        with self._frame_lock:
            self._last_frame = frame

    def _open_cap(self):
        cam_type = self.cfg.get("camera_type", "USB")
        cam_id   = int(self.cfg.get("camera_id", 0))
        w        = int(self.cfg.get("camera_width",  1280))
        h        = int(self.cfg.get("camera_height",  720))
        fps      = int(self.cfg.get("camera_fps",      30))
        cap      = None

        try:
            if cam_type == "IMV":
                try:
                    from imv_camera import IMVCamera
                    #imv = IMVCamera(cam_index=cam_id)
                    imv = IMVCamera(
                        cam_index=int(
                            self.cfg.get("camera_id", 0)))
                    if imv.open():
                        # Set FPS
                        fps = int(self.cfg.get("camera_fps", 30))
                        imv.set(cv2.CAP_PROP_FPS, fps)
                        # Set Exposure
                        exp = int(self.cfg.get(
                            "camera_exposure", -1))
                        imv.set(cv2.CAP_PROP_EXPOSURE, exp)
                        # Set Gain
                        gain = float(self.cfg.get(
                            "camera_gain", -1))
                        imv.set(cv2.CAP_PROP_GAIN, gain)
                        # FIX: Set Width/Height
                        w = int(self.cfg.get("camera_width", 1440))
                        h = int(self.cfg.get("camera_height", 1080))
                        imv.set_parameter("Resolution", (w, h))

                        # FIX: Set Flip
                        flip_raw = self.cfg.get(
                            "camera_flip", 0)
                        try:
                            flip = int(str(flip_raw).split(" ")[0])
                        except (ValueError, TypeError):
                            flip = 0
                        imv.set_parameter(
                            "ReverseX", flip in (1, 3))
                        imv.set_parameter(
                            "ReverseY", flip in (2, 3))
                        return imv
                    else:
                        print("[CameraThread] IMV open that bai")
                        return None
                except Exception as e:
                    print(f"[CameraThread] IMV loi: {e}")
                    return None

            elif cam_type == "IP":
                url = self.cfg.get("camera_ip_url", "rtsp://")
                cap = cv2.VideoCapture(url)

            elif cam_type == "CSI":
                csi_id   = int(self.cfg.get("camera_csi_id", 0))
                flip_val = int(str(self.cfg.get("camera_flip", 0)).split(" ")[0])
                pipeline = (
                    f"nvarguscamerasrc sensor-id={csi_id} ! "
                    f"video/x-raw(memory:NVMM),"
                    f"width={w},height={h},"
                    f"format=NV12,framerate={fps}/1 ! "
                    f"nvvidconv flip-method={flip_val} ! "
                    f"video/x-raw,width={w},height={h},"
                    f"format=BGRx ! videoconvert ! "
                    f"video/x-raw,format=BGR ! "
                    f"appsink drop=1 max-buffers=2 sync=false"
                )
                cap = cv2.VideoCapture(
                    pipeline, cv2.CAP_GSTREAMER)

            elif cam_type == "File":
                url = self.cfg.get("camera_ip_url", "")
                cap = cv2.VideoCapture(url)

            else:  # USB
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cam_id)

            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                exp = int(self.cfg.get("camera_exposure", -1))
                if exp >= 0:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
                else:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

                gain = int(self.cfg.get("camera_gain", -1))
                if gain >= 0:
                    cap.set(cv2.CAP_PROP_GAIN, gain)

                brightness = int(self.cfg.get(
                    "camera_brightness", 0))
                if brightness != 0:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

                contrast = int(self.cfg.get(
                    "camera_contrast", 0))
                if contrast != 0:
                    cap.set(cv2.CAP_PROP_CONTRAST, contrast)

                sat = int(self.cfg.get("camera_saturation", 0))
                if sat != 0:
                    cap.set(cv2.CAP_PROP_SATURATION, sat)

                autofocus = bool(self.cfg.get(
                    "camera_autofocus", True))
                cap.set(cv2.CAP_PROP_AUTOFOCUS,
                        1 if autofocus else 0)

        except Exception as e:
            print(f"[CameraThread] _open_cap error: {e}")

        return cap

    def run(self):
        self.running = True
        fps         = int(self.cfg.get("camera_fps", 30))
        flip        = int(str(self.cfg.get("camera_flip", 0)).split(" ")[0])
        interval    = 1.0 / max(fps, 1)
        retry_count = 0
        MAX_RETRY   = 5

        while self.running:
            try:
                self._cap = self._open_cap()
                if self._cap is None or \
                        not self._cap.isOpened():
                    retry_count += 1
                    print(f"[CameraThread] Khong mo duoc "
                          f"({retry_count}/{MAX_RETRY})")
                    if retry_count >= MAX_RETRY:
                        break
                    time.sleep(2.0)
                    continue

                retry_count = 0
                print("[CameraThread] Camera mo thanh cong")

                while self.running:
                    t0 = time.perf_counter()
                    ret, frame = self._cap.read()

                    if not ret or frame is None:
                        print("[CameraThread] Mat frame...")
                        break

                    cam_type = self.cfg.get(
                        "camera_type", "USB")
                    if cam_type not in ("CSI", "IMV"):
                        if flip == 1:
                            frame = cv2.flip(frame, 1)
                        elif flip == 2:
                            frame = cv2.flip(frame, 0)
                        elif flip == 3:
                            frame = cv2.flip(frame, -1)

                    self.last_frame = frame

                    if self.capturing:
                        now = time.perf_counter()
                        interval = getattr(
                            self,
                            '_capture_interval',
                            0.0)

                        if interval <= 0:
                            should_add = True
                        else:
                            last_t = getattr(
                                self,
                                '_last_capture_t',
                                0.0)
                            should_add = (
                                now - last_t >= interval)
                            if should_add:
                                self._last_capture_t = now

                        if should_add:
                            with self._lock:
                                if len(self._buffer) < \
                                        self.MAX_BUFFER:
                                    self._buffer.append(
                                        frame.copy())
                    elapsed = time.perf_counter() - t0
                    wait    = interval - elapsed
                    if wait > 0:
                        time.sleep(wait)

            except Exception as e:
                print(f"[CameraThread] Exception: {e}")
                retry_count += 1
                if retry_count >= MAX_RETRY:
                    break
                time.sleep(1.0)

            finally:
                if self._cap:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None

        print("[CameraThread] Dung")

    def start_capture(self):
        with self._lock:
            self._buffer.clear()
        self.MAX_BUFFER = int(
            self.cfg.get("camera_max_frames", 300))

        # Interval sampling
        interval_ms = float(self.cfg.get(
            "capture_interval_ms", 0))
        self._capture_interval = interval_ms / 1000.0
        self._last_capture_t   = 0.0

        self.capturing = True

    def stop_capture(self):
        self.capturing = False
        time.sleep(0.05)
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    def stop(self):
        self.running   = False
        self.capturing = False


# ════════════════════════════════════════════════════════════════
# BASE TOOL
# ════════════════════════════════════════════════════════════════
class BaseTool:
    name  = "Base Tool"
    icon  = "[fix]"
    color = "#89b4fa"
    desc  = "Mo ta tool"

    def __init__(self, app):
        self.app = app

    def run(self):
        messagebox.showinfo(
            self.name,
            f"{self.name}\n{self.desc}\n\n(Dang phat trien)",
            parent=self.app)

    def build_button(self, parent):
        return tk.Button(
            parent,
            text=f"{self.icon}\n{self.name}",
            bg=self.color, fg="black",
            font=("Consolas", 8, "bold"),
            relief="flat", padx=4, pady=6,
            wraplength=80, justify="center",
            command=self.run)


# ════════════════════════════════════════════════════════════════
# TOOL CLASSES
# ════════════════════════════════════════════════════════════════
class PositionAdjustTool(BaseTool):
    name = "Position\nAdjust"; icon = "[grd]"; color = "#89b4fa"
    desc = "Phat hien va can chinh vi tri doi tuong"

class BinaryTool(BaseTool):
    name = "Binary\nTool"; icon = "[bin]"; color = "#89b4fa"
    desc = "Nhi phan hoa anh"

class BlobAnalysisTool(BaseTool):
    name = "Blob\nAnalysis"; icon = "[blb]"; color = "#89b4fa"
    desc = "Phan tich vung blob"

class EdgeDetectionTool(BaseTool):
    name = "Edge\nDetection"; icon = "[edg]"; color = "#89b4fa"
    desc = "Phat hien bien Sobel/Canny"

class LineDetectionTool(BaseTool):
    name = "Line\nDetection"; icon = "[lin]"; color = "#89b4fa"
    desc = "Phat hien duong thang"

class CircleDetectionTool(BaseTool):
    name = "Circle\nDetection"; icon = "[cir]"; color = "#89b4fa"
    desc = "Phat hien hinh tron"

class ShapeSearchTool(BaseTool):
    name  = "Shape\nSearch"
    icon  = "[shp]"
    color = "#89b4fa"
    desc  = "Tim kiem hinh dang khop mau"

    def run(self):
        ShapeSearchApp(
            parent=self.app,
            cam_thread=self.app.cam_thread)

class DefectDetectionTool(BaseTool):
    name = "Defect\nDetection"; icon = "[roi]"; color = "#89b4fa"
    desc = "Phat hien loi be mat gradient"

    def run(self):
        self.app._manual_process()


# ════════════════════════════════════════════════════════════════
# TOOL MANAGER
# ════════════════════════════════════════════════════════════════
class ToolManager:
    TOOLS = [
        PositionAdjustTool, BinaryTool, BlobAnalysisTool,
        EdgeDetectionTool,  LineDetectionTool,
        CircleDetectionTool, ShapeSearchTool,
        DefectDetectionTool,
    ]
    COLS = 2

    def __init__(self, app):
        self.app       = app
        self.instances = [T(app) for T in self.TOOLS]

    def build_panel(self, parent):
        lf = ttk.LabelFrame(parent, text=" [tool] Tool xu ly ")
        lf.pack(fill=tk.X, padx=4, pady=4)
        grid = tk.Frame(lf, bg="#1e1e2e")
        grid.pack(fill=tk.X, padx=4, pady=4)
        for i, tool in enumerate(self.instances):
            r, c = i // self.COLS, i % self.COLS
            btn  = tool.build_button(grid)
            btn.grid(row=r, column=c,
                     padx=3, pady=3, sticky="nsew")
            grid.columnconfigure(c, weight=1)


# ════════════════════════════════════════════════════════════════
# CONFIG DIALOG
# ════════════════════════════════════════════════════════════════
class ConfigDialog(tk.Toplevel):
    BG       = "#1e1e2e"
    FG       = "#cdd6f4"
    ACC      = "#89b4fa"
    ENTRY_BG = "#313244"

    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.app = parent_app
        self.title("[cfg] Cai Dat Thong So")
        self.configure(bg=self.BG)
        self.geometry("920x700")
        self.resizable(True, True)
        self.transient(parent_app)
        self.grab_set()
        self._edit_cfg = {k: v
                          for k, v in parent_app.cfg.items()}
        self._vars     = {}
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    #
    def _tab_plc(self, parent):
        f = self._scrollable(parent)
        self._section(f, "PLC Modbus TCP", "#89dceb")
        self._make_row(f, "plc_enable",
                    "Bat PLC",
                    "bool",
                    tooltip="Bat/tat ket noi PLC")
        self._make_row(f, "plc_host",
                    "PLC IP",
                    "str",
                    tooltip="IP address cua PLC")
        self._make_row(f, "plc_port",
                    "PLC Port",
                    "int", 1, 65535,
                    tooltip="Modbus TCP port (mac dinh 502)")
        self._make_row(f, "plc_unit_id",
                    "Unit ID",
                    "int", 1, 255,
                    tooltip="Modbus Unit ID (mac dinh 1)")

        self._section(f, "Dia chi Modbus", "#89dceb")
        # Hien thi thong tin dia chi (read-only)
        info_frame = tk.Frame(f, bg=self.BG)
        info_frame.pack(fill=tk.X, padx=8, pady=4)
        info_text = (
            "Coil (PLC → Camera):\n"
            "  M1000 (addr=1000): Bat camera\n"
            "  M1001 (addr=1001): Trigger chup\n\n"
            "Register (Camera → PLC):\n"
            "  D2000: So anh chup\n"
            "  D2002: So anh OK\n"
            "  D2004: So anh NG\n"
            "  D2006: Thoi gian xu ly (ms)\n"
            "  D2008: Trang thai\n"
            "    0=Idle 1=Chup 2=XuLy 3=Xong 9=Loi"
        )
        tk.Label(f, text=info_text,
                bg=self.BG, fg="#89dceb",
                font=("Consolas", 8),
                justify="left", anchor="w"
                ).pack(fill=tk.X, padx=16, pady=4)

        self._section(f, "Kiem tra ket noi", "#89dceb")
        btn_plc = tk.Frame(f, bg=self.BG)
        btn_plc.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(btn_plc,
                text="Test PLC Connect",
                bg="#89dceb", fg="black",
                font=("Consolas", 9, "bold"),
                relief="flat", padx=12, pady=6,
                command=self._test_plc
                ).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_plc,
                text="Restart PLC Monitor",
                bg="#cba6f7", fg="black",
                font=("Consolas", 9, "bold"),
                relief="flat", padx=12, pady=6,
                command=self._restart_plc
                ).pack(side=tk.LEFT, padx=4)

        self.lbl_plc_info = tk.Label(
            f, text="",
            bg=self.BG, fg="#a6e3a1",
            font=("Consolas", 8), anchor="w",
            wraplength=600, justify="left")
        self.lbl_plc_info.pack(
            fill=tk.X, padx=16, pady=4)

    def _test_plc(self):
        """Test ket noi PLC"""
        self._collect()
        host    = self._edit_cfg.get(
            "plc_host", "192.168.4.20")
        port    = int(self._edit_cfg.get("plc_port", 502))
        unit_id = int(self._edit_cfg.get("plc_unit_id", 1))

        self.lbl_plc_info.configure(
            text="Dang ket noi...", fg="#89b4fa")
        self.update_idletasks()

        def _do_test():
            try:
                from modbus_plc import PLCClient
                plc = PLCClient(
                    host=host, port=port,
                    unit_id=unit_id, timeout=3.0)
                if plc.connect():
                    coils = plc.read_coils(1000, 2)
                    msg = (f"[ok] PLC OK!\n"
                        f"  {host}:{port} "
                        f"unit={unit_id}\n"
                        f"  M1000={coils[0]} "
                        f"M1001={coils[1]}")
                    plc.disconnect()
                    self.after(0,
                        lambda: self.lbl_plc_info.configure(
                            text=msg, fg="#a6e3a1"))
                else:
                    self.after(0,
                        lambda: self.lbl_plc_info.configure(
                            text=f"[ng] Ket noi FAIL "
                                f"{host}:{port}",
                            fg="#f38ba8"))
            except ImportError:
                self.after(0,
                    lambda: self.lbl_plc_info.configure(
                        text="[ng] Khong tim thay "
                            "modbus_plc.py!",
                        fg="#f38ba8"))
            except Exception as e:
                err = str(e)
                self.after(0,
                    lambda: self.lbl_plc_info.configure(
                        text=f"[ng] Loi: {err}",
                        fg="#f38ba8"))

        threading.Thread(
            target=_do_test, daemon=True).start()

    def _restart_plc(self):
        """Restart PLC monitor"""
        def _do():
            try:
                # Dung monitor cu
                if self.app.plc_monitor:
                    self.app.plc_monitor.stop()
                    self.app.plc_monitor = None
                if self.app.plc_client:
                    self.app.plc_client.disconnect()
                    self.app.plc_client = None
                # Khoi dong lai
                self.app._plc_init()
                self.after(0,
                    lambda: self.lbl_plc_info.configure(
                        text="[ok] Da restart PLC monitor",
                        fg="#a6e3a1"))
            except Exception as e:
                err = str(e)
                self.after(0,
                    lambda: self.lbl_plc_info.configure(
                        text=f"[ng] Loi: {err}",
                        fg="#f38ba8"))
        threading.Thread(target=_do, daemon=True).start()
    #
    def _build_ui(self):
        BG, FG, ACC = self.BG, self.FG, self.ACC
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#252535",
                        foreground=FG,
                        font=("Consolas", 9, "bold"),
                        padding=[8, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", "#313244")],
                  foreground=[("selected", ACC)])
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG,
                        foreground=FG)

        tk.Label(self,
                 text="[cfg]  CAI DAT THONG SO INSPECTOR",
                 bg=BG, fg=ACC,
                 font=("Consolas", 12, "bold")
                 ).pack(pady=(10, 4))
        ttk.Separator(self, orient="horizontal"
                      ).pack(fill=tk.X, padx=12)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        for name, builder in [
            ("[cam] Camera",     self._tab_camera),
            ("[pre] Tien xu ly", self._tab_preprocess),
            ("[grd] Gradient",   self._tab_gradient),
            ("[mph] Morphology", self._tab_morphology),
            ("[roi] ROI Mask",   self._tab_roi),
            ("[cls] Classify",   self._tab_classify),
            ("[sav] Output",     self._tab_output),
            ("[plc] PLC",         self._tab_plc),
        ]:
            frame = ttk.Frame(nb)
            nb.add(frame, text=name)
            builder(frame)

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill=tk.X, padx=8, pady=(0, 8))
        for txt, bg, cmd in [
            ("[ok]  Ap dung & Dong", "#a6e3a1",
             self._apply_and_close),
            ("[eye]  Xem truoc",     "#89b4fa",
             self._preview_apply),
            ("[rst]  Reset",         "#fab387",
             self._reset_defaults),
            ("[x]  Huy",            "#f38ba8",
             self.destroy),
        ]:
            tk.Button(bot, text=txt, bg=bg, fg="black",
                      font=("Consolas", 11, "bold"),
                      relief="flat", padx=14, pady=7,
                      command=cmd
                      ).pack(side=tk.LEFT, fill=tk.X,
                             expand=True, padx=2)

    def _scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=self.BG,
                           highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical",
                           command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, bg=self.BG)
        wid   = canvas.create_window(
            (0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(
                        wid, width=e.width))

        def _on_mousewheel(event):
            try:
                if event.delta:
                    canvas.yview_scroll(
                        -1 * (event.delta // 120), "units")
                elif event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
            except tk.TclError:
                pass

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>",   _on_mousewheel)
        canvas.bind("<Button-5>",   _on_mousewheel)
        inner.bind("<MouseWheel>",  _on_mousewheel)
        inner.bind("<Button-4>",    _on_mousewheel)
        inner.bind("<Button-5>",    _on_mousewheel)
        return inner

    def _make_row(self, parent, key, label, typ,
                  mn=None, mx=None, choices=None,
                  tooltip=""):
        BG, FG = self.BG, self.FG
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(row, text=f"{label}:", bg=BG, fg=FG,
                 font=("Consolas", 9), width=24, anchor="w"
                 ).pack(side=tk.LEFT)
        val = self._edit_cfg.get(
            key, DEFAULT_CFG.get(key, ""))

        if typ == "bool":
            var = tk.BooleanVar(value=bool(val))
            tk.Checkbutton(row, variable=var,
                           bg=BG, fg=self.ACC,
                           selectcolor="#313244",
                           activebackground=BG,
                           font=("Consolas", 9),
                           relief="flat"
                           ).pack(side=tk.LEFT)
            self._vars[key] = (var, typ)

        elif typ == "choice":
            var = tk.StringVar(value=str(val))
            ttk.Combobox(row, textvariable=var,
                         values=choices,
                         state="readonly",
                         font=("Consolas", 9), width=18
                         ).pack(side=tk.LEFT)
            self._vars[key] = (var, typ)

        elif typ in ("int", "float"):
            var = tk.StringVar(value=str(val))
            ent = tk.Entry(row, textvariable=var,
                           bg=self.ENTRY_BG, fg=FG,
                           insertbackground=FG,
                           font=("Consolas", 9),
                           width=8, relief="flat")
            ent.pack(side=tk.LEFT, padx=(0, 6))
            self._vars[key] = (var, typ)

            if mn is not None and mx is not None:
                res = 1 if typ == "int" else 0.05

                def _sl_cmd(v, k=key, sv=var, t=typ):
                    vv = int(float(v)) if t == "int" \
                         else round(float(v), 3)
                    sv.set(str(vv))
                    self._edit_cfg[k] = vv

                sl = tk.Scale(row, from_=mn, to=mx,
                              orient="horizontal",
                              resolution=res,
                              command=_sl_cmd,
                              bg=BG, fg=FG,
                              troughcolor="#313244",
                              highlightthickness=0,
                              font=("Consolas", 7),
                              length=180,
                              showvalue=False)
                try:
                    sl.set(float(val) if val != "" else mn)
                except Exception:
                    sl.set(mn)
                sl.pack(side=tk.LEFT, fill=tk.X,
                        expand=True, padx=(0, 4))

                def _entry_cmd(e, k=key, slider=sl,
                               t=typ, sv=var):
                    try:
                        vv = int(float(sv.get())) \
                            if t == "int" \
                            else round(float(sv.get()), 3)
                        slider.set(vv)
                        self._edit_cfg[k] = vv
                    except (ValueError, tk.TclError):
                        pass

                ent.bind("<Return>",   _entry_cmd)
                ent.bind("<FocusOut>", _entry_cmd)

        else:  # str
            var = tk.StringVar(value=str(val))
            tk.Entry(row, textvariable=var,
                     bg=self.ENTRY_BG, fg=FG,
                     insertbackground=FG,
                     font=("Consolas", 9),
                     width=26, relief="flat"
                     ).pack(side=tk.LEFT, fill=tk.X,
                            expand=True)
            self._vars[key] = (var, typ)

        if tooltip:
            tk.Label(row, text=f"[i] {tooltip}",
                     bg=BG, fg="#585b70",
                     font=("Consolas", 7), anchor="w"
                     ).pack(side=tk.LEFT, padx=4)

    def _section(self, parent, title, color="#89b4fa"):
        tk.Label(parent, text=f"  {title}",
                 bg=color, fg="black",
                 font=("Consolas", 9, "bold"),
                 anchor="w", pady=3
                 ).pack(fill=tk.X, padx=8, pady=(10, 4))

    def _tab_camera(self, parent):
        f = self._scrollable(parent)
        self._section(f, "[cam] Loai Camera", "#cba6f7")
        self._make_row(f, "camera_type", "Camera type",
                       "choice",
                       choices=["USB", "CSI", "IP",
                                "File", "IMV"],
                       tooltip="USB  CSI  IP=RTSP  "
                               "File  IMV=iRAYple")
        self._make_row(f, "camera_id", "Camera ID",
                       "int", 0, 10,
                       tooltip="Index camera (0,1,2...)")
        self._make_row(f, "camera_csi_id", "CSI sensor ID",
                       "int", 0, 3)
        self._make_row(f, "camera_ip_url", "IP/File URL",
                       "str",
                       tooltip="rtsp://... hoac duong dan file")

        self._section(f, "[grd] Do phan giai & FPS", "#cba6f7")
        self._make_row(f, "camera_width",  "Width (px)",
                       "int", 320, 4096)
        self._make_row(f, "camera_height", "Height (px)",
                       "int", 240, 2160)
        self._make_row(f, "camera_fps", "FPS", "int", 1, 240)
        self._make_row(f, "camera_buffer_size",
                       "Buffer size", "int", 1, 10,
                       tooltip="1=it tre nhat")

        self._section(f, "Pho Sang & Gain", "#cba6f7")
        #self._make_row(f, "camera_brightness",
                       #"Brightness", "int", -64, 64)
        #self._make_row(f, "camera_contrast",
                       #"Contrast", "int", -64, 64)
        self._make_row(f, "camera_brightness",
                        "Brightness (USB only)", "int", -64, 64,
                        tooltip="Chi co tac dung voi USB/CSI camera")
        self._make_row(f, "camera_contrast",
                        "Contrast (USB only)", "int", -64, 64,
                        tooltip="Chi co tac dung voi USB/CSI camera")
        self._make_row(f, "camera_saturation",
                       "Saturation", "int", -100, 100)
        self._make_row(f, "camera_autofocus",
                       "Autofocus", "bool")

        self._section(f, "Huong Anh", "#cba6f7")
        self._make_row(f, "camera_flip", "Flip mode",
                       "choice",
                       choices=["0 - Khong flip",
                                "1 - Lat ngang",
                                "2 - Lat doc",
                                "3 - Xoay 180"])

        self._section(f, "[fix] Kiem Tra USB/IP/CSI",
                      "#cba6f7")
        btn_frame = tk.Frame(f, bg=self.BG)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(btn_frame, text="Test Camera",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", padx=12, pady=6,
                  command=self._test_camera
                  ).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="List Cameras",
                  bg="#89dceb", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", padx=12, pady=6,
                  command=self._list_cameras
                  ).pack(side=tk.LEFT, padx=4)
        self.lbl_cam_info = tk.Label(
            f, text="",
            bg=self.BG, fg="#a6e3a1",
            font=("Consolas", 8), anchor="w",
            wraplength=600, justify="left")
        self.lbl_cam_info.pack(
            fill=tk.X, padx=16, pady=4)

        self._section(f, "IMV Camera (iRAYple)", "#89b4fa")
        self._make_row(f, "camera_exposure",
                       "Exposure (us)",
                       "int", -1, 500000,
                       tooltip="-1=auto | us | 10000=10ms")
        self._make_row(f, "camera_gain",
                       "Gain (dB)",
                       "int", -1, 24,
                       tooltip="-1=auto | 0~24 dB")

        btn_imv = tk.Frame(f, bg=self.BG)
        btn_imv.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(btn_imv,
                  text="Read IMV Params",
                  bg="#89dceb", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", padx=12, pady=6,
                  command=self._read_imv_params
                  ).pack(side=tk.LEFT, padx=4)

        self.lbl_imv_info = tk.Label(
            f, text="",
            bg=self.BG, fg="#a6e3a1",
            font=("Consolas", 8), anchor="w",
            wraplength=600, justify="left")
        self.lbl_imv_info.pack(
            fill=tk.X, padx=16, pady=4)

    def _read_imv_params(self):
        """Doc thong so hien tai tu IMV camera"""
        cam_thread = self.app.cam_thread
        if cam_thread is None or \
                cam_thread._cap is None:
            self.lbl_imv_info.configure(
                text="Camera chua mo!",
                fg="#f38ba8")
            return

        cap = cam_thread._cap

        if not hasattr(cap, 'get_current_value'):
            self.lbl_imv_info.configure(
                text="Khong phai IMV camera!",
                fg="#f9e2af")
            return

        def _do_read():
            try:
                exp  = cap.get_current_value("ExposureTime")
                gain = cap.get_current_value("Gain")
                fps  = cap.get_current_value(
                    "AcquisitionFrameRate")
                ae   = cap.get_current_value("ExposureAuto")
                ag   = cap.get_current_value("GainAuto")
                #w    = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                #h    = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                w   = cap.get_current_value("Width")
                h   = cap.get_current_value("Height")
                rx  = cap.get_current_value("ReverseX")
                ry  = cap.get_current_value("ReverseY")

                try:
                    exp_ms = f"({float(exp)/1000:.1f}ms)"
                except Exception:
                    exp_ms = ""

                # Canh bao neu cfg khac thuc te
                cfg_w = self.app.cfg.get("camera_width", 0)
                cfg_h = self.app.cfg.get("camera_height", 0)
                size_warn = ""
                if cfg_w != w or cfg_h != h:
                    size_warn = (
                                f"\n  [!] CFG={cfg_w}x{cfg_h}"
                                f" != Camera={w}x{h}")
                msg  = (f"Size    : {w} x {h}"
                        f"  (256~1440 x 64~1080 inc=4)"
                        f"{size_warn}\n"
                        f"Exposure: {exp}us {exp_ms}"
                        f"  [Auto:{ae}]\n"
                        f"Gain    : {gain}dB"
                        f"  [Auto:{ag}]\n"
                        f"FPS     : {fps}\n"
                        f"FlipX   : {rx}"
                        f"  FlipY: {ry}")
                self.after(
                    0, lambda:
                    self.lbl_imv_info.configure(
                        text=msg, fg="#a6e3a1"))
            except Exception as e:
                err = str(e)
                self.after(
                    0, lambda:
                    self.lbl_imv_info.configure(
                        text=f"Loi: {err}",
                        fg="#f38ba8"))

        threading.Thread(
            target=_do_read, daemon=True).start()

    def _test_camera(self):
        self._collect()
        cam_type = self._edit_cfg.get("camera_type", "USB")
        cam_id   = int(self._edit_cfg.get("camera_id", 0))
        w        = int(self._edit_cfg.get(
            "camera_width", 1280))
        h        = int(self._edit_cfg.get(
            "camera_height", 720))
        fps      = int(self._edit_cfg.get("camera_fps", 30))

        self.lbl_cam_info.configure(
            text="Dang ket noi...", fg="#89b4fa")
        self.update_idletasks()

        def _do_test():
            try:
                if cam_type == "IMV":
                    from imv_camera import (IMVCamera,
                                            IMV_AVAILABLE)
                    if not IMV_AVAILABLE:
                        self.after(
                            0, lambda:
                            self.lbl_cam_info.configure(
                                text="IMV SDK khong co!",
                                fg="#f38ba8"))
                        return
                    imv = IMVCamera(cam_index=cam_id)
                    if imv.open():
                        ret, frame = imv.read()
                        imv.release()
                        if ret and frame is not None:
                            fh2, fw2 = frame.shape[:2]
                            msg = (f"IMV Camera OK!\n"
                                   f"  ID={cam_id}\n"
                                   f"  {fw2}x{fh2}")
                            self.after(
                                0, lambda:
                                self.lbl_cam_info.configure(
                                    text=msg,
                                    fg="#a6e3a1"))
                        else:
                            self.after(
                                0, lambda:
                                self.lbl_cam_info.configure(
                                    text="IMV mo OK "
                                         "nhung khong doc frame",
                                    fg="#f9e2af"))
                    else:
                        self.after(
                            0, lambda:
                            self.lbl_cam_info.configure(
                                text=f"Khong mo duoc "
                                     f"IMV ID={cam_id}",
                                fg="#f38ba8"))
                    return

                elif cam_type == "IP":
                    cap = cv2.VideoCapture(
                        self._edit_cfg.get(
                            "camera_ip_url", ""))
                elif cam_type == "File":
                    cap = cv2.VideoCapture(
                        self._edit_cfg.get(
                            "camera_ip_url", ""))
                else:
                    cap = cv2.VideoCapture(
                        cam_id, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = cv2.VideoCapture(cam_id)

                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        fh2, fw2 = frame.shape[:2]
                        msg = (f"Camera OK!\n"
                               f"  Type={cam_type} "
                               f"ID={cam_id}\n"
                               f"  {fw2}x{fh2}  "
                               f"FPS:{fps}")
                        self.after(
                            0, lambda:
                            self.lbl_cam_info.configure(
                                text=msg, fg="#a6e3a1"))
                    else:
                        self.after(
                            0, lambda:
                            self.lbl_cam_info.configure(
                                text="Camera mo OK "
                                     "nhung khong doc frame",
                                fg="#f9e2af"))
                else:
                    cap.release()
                    self.after(
                        0, lambda:
                        self.lbl_cam_info.configure(
                            text=f"Khong mo duoc camera\n"
                                 f"Type={cam_type} "
                                 f"ID={cam_id}",
                            fg="#f38ba8"))
            except Exception as e:
                err = str(e)
                self.after(
                    0, lambda:
                    self.lbl_cam_info.configure(
                        text=f"Loi: {err}",
                        fg="#f38ba8"))

        threading.Thread(
            target=_do_test, daemon=True).start()

    def _list_cameras(self):
        self.lbl_cam_info.configure(
            text="Dang tim cameras...", fg="#89b4fa")
        self.update_idletasks()

        def _do_list():
            found = []
            for i in range(6):
                try:
                    cap = cv2.VideoCapture(
                        i, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cw = int(cap.get(
                            cv2.CAP_PROP_FRAME_WIDTH))
                        ch = int(cap.get(
                            cv2.CAP_PROP_FRAME_HEIGHT))
                        found.append(
                            f"  Camera {i}: {cw}x{ch}")
                        cap.release()
                except Exception:
                    pass
            if found:
                msg = (f"Tim thay {len(found)} camera:\n"
                       + "\n".join(found))
                self.after(
                    0, lambda:
                    self.lbl_cam_info.configure(
                        text=msg, fg="#a6e3a1"))
            else:
                self.after(
                    0, lambda:
                    self.lbl_cam_info.configure(
                        text="Khong tim thay camera USB!",
                        fg="#f9e2af"))

        threading.Thread(
            target=_do_list, daemon=True).start()

    def _tab_preprocess(self, parent):
        f = self._scrollable(parent)
        self._section(f, "Gaussian Blur", "#89dceb")
        self._make_row(f, "blur_ksize", "Blur ksize (le)",
                       "int", 1, 21,
                       tooltip="Kernel lam mo, phai la so le")
        self._make_row(f, "blur_sigma", "Blur sigma",
                       "float", 0, 10,
                       tooltip="0 = tu tinh tu ksize")
        self._section(f, "Bilateral Filter", "#89dceb")
        self._make_row(f, "bilateral_enable",
                       "Bat bilateral", "bool")
        self._make_row(f, "bilateral_d",
                       "Bilateral d", "int", 1, 20)
        self._make_row(f, "bilateral_sc",
                       "Bilateral sC", "int", 1, 200)
        self._make_row(f, "bilateral_ss",
                       "Bilateral sS", "int", 1, 200)
        self._section(f, "Thu tu xu ly", "#89dceb")
        self._make_row(f, "preprocess_order",
                       "Preprocess order", "choice",
                       choices=["blur_bilateral",
                                "bilateral_blur",
                                "blur_only",
                                "bilateral_only"])
        self._section(f, "CLAHE", "#89dceb")
        self._make_row(f, "clahe_enable",
                       "Bat CLAHE", "bool")
        self._make_row(f, "clahe_clip",
                       "CLAHE clip limit",
                       "float", 0.5, 8.0)
        self._make_row(f, "clahe_grid_x",
                       "CLAHE grid X", "int", 2, 32)
        self._make_row(f, "clahe_grid_y",
                       "CLAHE grid Y", "int", 2, 32)

    def _tab_gradient(self, parent):
        f = self._scrollable(parent)
        self._section(f, "Phuong phap Gradient", "#f9e2af")
        self._make_row(f, "grad_method",
                       "Grad method", "choice",
                       choices=["sobel", "scharr",
                                "laplacian", "canny"])
        self._make_row(f, "sobel_ksize",
                       "Sobel ksize", "int", 1, 7)
        self._make_row(f, "grad_thr",
                       "Grad threshold", "int", 1, 100)
        self._section(f, "Canny", "#f9e2af")
        self._make_row(f, "canny_thr1",
                       "Canny thr1", "int", 1, 300)
        self._make_row(f, "canny_thr2",
                       "Canny thr2", "int", 1, 300)
        self._section(f, "Loai Threshold", "#f9e2af")
        self._make_row(f, "grad_thresh_type",
                       "Thresh type", "choice",
                       choices=["binary_inv", "otsu",
                                "adaptive"])
        self._make_row(f, "adaptive_block",
                       "Adaptive block", "int", 3, 51)
        self._make_row(f, "adaptive_c",
                       "Adaptive C", "int", -20, 20)

    def _tab_morphology(self, parent):
        f = self._scrollable(parent)
        self._section(f, "Morphology chinh", "#cba6f7")
        self._make_row(f, "morph_shape",
                       "Kernel shape", "choice",
                       choices=["ellipse", "rect", "cross"])
        self._make_row(f, "morph_open_iter",
                       "Open iterations", "int", 1, 10)
        self._make_row(f, "morph_close_iter",
                       "Close iterations", "int", 1, 10)
        self._section(f, "Morphology them", "#cba6f7")
        self._make_row(f, "morph_extra_enable",
                       "Bat morph them", "bool")
        self._make_row(f, "morph_extra_type",
                       "Loai morph them", "choice",
                       choices=["dilate", "erode",
                                "gradient"])
        self._make_row(f, "morph_extra_iter",
                       "Morph them iter", "int", 1, 5)

    def _tab_roi(self, parent):
        f = self._scrollable(parent)
        self._section(f, "ROI Detection", "#a6e3a1")
        self._make_row(f, "roi_thresh",
                       "ROI threshold", "int", 50, 255)
        self._make_row(f, "roi_thresh_type",
                       "ROI thresh type", "choice",
                       choices=["binary", "binary_inv",
                                "otsu"])
        self._make_row(f, "roi_offset",
                       "ROI offset (px)", "int", 5, 300)
        self._make_row(f, "roi_min_area",
                       "ROI min area (px)", "int",
                       100, 50000)
        self._make_row(f, "roi_contour_approx",
                       "Lam min contour", "bool")

    def _tab_classify(self, parent):
        f = self._scrollable(parent)
        self._section(f, "Danh gia OK/NG", "#fab387")
        self._make_row(f, "ok_ratio",
                       "OK ratio (0-1)",
                       "float", 0.5, 1.0,
                       tooltip="% pixel OK toi thieu")
        self._section(f, "Phan tich loi", "#fab387")
        self._make_row(f, "noise_max_area",
                       "Noise max (px)", "int", 50, 50000)
        self._make_row(f, "border_noise_max_area",
                       "Border noise (px)",
                       "int", 50, 50000)
        self._make_row(f, "min_defect_area",
                       "Min defect (px)", "int", 0, 500)
        self._make_row(f, "min_grad_area",
                       "Min grad area (px)", "int", 0, 5000)
        self._make_row(f, "max_defect_count",
                       "Max defect count", "int", -1, 100)
        self._section(f, "Kernel phan tich", "#fab387")
        self._make_row(f, "green_dilate_k",
                       "Green dilate k", "int", 3, 21)
        self._make_row(f, "sep_open_k",
                       "Sep open k", "int", 1, 11)
        self._make_row(f, "ring_dilate_k",
                       "Ring dilate k", "int", 1, 11)

    def _tab_output(self, parent):
        f = self._scrollable(parent)
        self._section(f, "Luu anh ket qua", "#f38ba8")
        self._make_row(f, "save_ok_images",
                        "Luu anh OK",
                        "bool",
                        tooltip="Luu anh ket qua OK vao folder")
        self._make_row(f, "save_ng_images",
                        "Luu anh NG",
                        "bool",
                        tooltip="Luu anh ket qua NG vao folder")

        self._make_row(f, "out_q", "JPEG quality",
                        "int", 50, 100)
        self._make_row(f, "max_w", "Max width (px)",
                        "int", 400, 4000)
        self._make_row(f, "out_q",
                       "JPEG quality", "int", 50, 100)
        self._make_row(f, "max_w",
                       "Max width (px)", "int", 400, 4000)
        self._section(f, "He thong", "#f38ba8")
        self._make_row(f, "n_threads", "So threads",
                       "int", 1, 8,
                       tooltip="Jetson: 2-4 la toi uu")
        self._make_row(f, "input_dir",
                       "Input folder", "str")
        self._make_row(f, "output_dir",
                       "Output folder", "str")
        # Trong _tab_output, them:
        self._section(f, "Chup anh", "#f38ba8")
        self._make_row(f, "camera_max_frames",
                    "Max frames",
                    "int", 1, 9999,
                    tooltip="So frame toi da khi chup (mac dinh 300)")

    def _collect(self):
        for key, (var, typ) in self._vars.items():
            try:
                raw = var.get()
                if typ == "int":
                    if isinstance(raw, str) and raw:
                        raw = raw.split(" ")[0]
                    self._edit_cfg[key] = \
                        int(float(str(raw)))
                elif typ == "float":
                    self._edit_cfg[key] = \
                        round(float(str(raw)), 4)
                elif typ == "bool":
                    self._edit_cfg[key] = bool(raw)
                else:
                    self._edit_cfg[key] = str(raw)
            except (ValueError, tk.TclError):
                pass

    def _preview_apply(self):
        self._collect()
        self.app.cfg.update(self._edit_cfg)
        self.app._apply_cfg_to_inspector()
        self.app._set_status(
            "[eye] Da ap dung tam (chua luu file)")

    def _apply_and_close(self):
        self._collect()
        self.app.cfg.update(self._edit_cfg)
        save_cfg(self.app.cfg)
        self.app._apply_cfg_to_inspector()
        self.app._restart_camera_if_needed()
        self.app._set_status("[ok] Da luu cau hinh")
        self.destroy()

    def _reset_defaults(self):
        if messagebox.askyesno(
                "Xac nhan",
                "Reset tat ca ve mac dinh?",
                parent=self):
            self._edit_cfg = DEFAULT_CFG.copy()
            for key, (var, typ) in self._vars.items():
                val = DEFAULT_CFG.get(key, "")
                try:
                    var.set(bool(val)
                            if typ == "bool"
                            else str(val))
                except Exception:
                    pass
            self.app._set_status("[rst] Reset ve mac dinh")
# ════════════════════════════════════════════════════════════════
# ROI SETUP WINDOW
# ════════════════════════════════════════════════════════════════
class ROISetupWindow(tk.Toplevel):
    COLORS = {
        "rect":     (0, 255, 0),
        "rot_rect": (0, 200, 255),
        "circle":   (255, 100, 0),
        "polygon":  (255, 0, 200),
    }
    TOOL_COLORS = {
        "DefectDetection": (0,   255,   0),
        "ShapeSearch":     (255, 200,   0),
        "BlobAnalysis":    (0,   200, 255),
        "EdgeDetection":   (200,   0, 255),
        "BinaryCheck":     (0,   255, 200),
        None:              (128, 128, 128),
    }
    DISP_W             = 900
    DISP_H             = 540
    UPDATE_INTERVAL_MS = 50

    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.app   = parent_app
        self.title("[roi] ROI Setup - Ve & Gan Tool")
        self.configure(bg="#1a1a2e")
        self.geometry(
            f"{self.DISP_W + 260}x{self.DISP_H + 160}")
        self.resizable(True, True)
        self.transient(parent_app)

        self.tool          = tk.StringVar(value="none")
        self.shapes        = []
        self._drawing      = False
        self._start        = None
        self._current      = None
        self._poly_pts     = []
        self._rot_pts      = []
        self._drag_idx     = -1
        self._drag_off     = (0, 0)
        self._selected_idx = -1
        self._img_w        = self.DISP_W
        self._img_h        = self.DISP_H
        self._fr_w         = 640
        self._fr_h         = 480
        self._off_x        = 0
        self._off_y        = 0
        self._pimg_cache   = None

        self.var_bright   = tk.IntVar(
            value=int(self.app.cfg.get(
                "roi_brightness", 0)))
        self.var_contrast = tk.DoubleVar(
            value=float(self.app.cfg.get(
                "roi_contrast", 1.0)))

        import copy
        self.shapes = copy.deepcopy(
            self.app.cfg.get("roi_shapes", []))
        self._crop  = self.app.cfg.get("roi_crop", None)

        self._build_ui()
        self._running = True
        self.after(self.UPDATE_INTERVAL_MS,
                   self._update_loop)
        self.protocol("WM_DELETE_WINDOW",
                      self._on_close_btn)

    def _build_ui(self):
        BG  = "#1a1a2e"
        FG  = "#cdd6f4"
        ACC = "#89b4fa"

        tb = tk.Frame(self, bg="#252535",
                      relief="groove", bd=1)
        tb.pack(fill=tk.X, padx=4, pady=(4, 0))

        tk.Label(tb, text="Ve:", bg="#252535", fg=ACC,
                 font=("Consolas", 9, "bold")
                 ).pack(side=tk.LEFT, padx=(6, 2))
        self._tool_btns = {}
        for lbl, t, col in [
            ("[non] None",  "none",     "#313244"),
            ("[rec] Rect",  "rect",     "#a6e3a1"),
            ("[rot] Rot",   "rot_rect", "#89dceb"),
            ("[cir] Circ",  "circle",   "#cba6f7"),
            ("[hex] Poly",  "polygon",  "#f9e2af"),
            ("[mov] Move",  "move",     "#89b4fa"),
        ]:
            btn = tk.Button(
                tb, text=lbl, bg=col, fg="black",
                font=("Consolas", 9, "bold"),
                relief="flat", padx=5, pady=2,
                command=lambda x=t: self._set_tool(x))
            btn.pack(side=tk.LEFT, padx=1, pady=3)
            self._tool_btns[t] = btn

        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=3)

        for txt, cmd in [
            ("[und] Undo", self._undo),
            ("[del] Clear", self._clear),
        ]:
            tk.Button(tb, text=txt,
                      bg="#f38ba8", fg="black",
                      font=("Consolas", 9, "bold"),
                      relief="flat", padx=5, pady=2,
                      command=cmd
                      ).pack(side=tk.LEFT, padx=1, pady=3)

        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=3)

        for txt, bg, cmd in [
            ("[crp] Set Crop", "#fab387",
             self._set_crop_from_rect),
            ("[x] No Crop",   "#313244",
             self._clear_crop),
        ]:
            tk.Button(tb, text=txt, bg=bg, fg=FG
                      if bg == "#313244" else "black",
                      font=("Consolas", 9, "bold"),
                      relief="flat", padx=5, pady=2,
                      command=cmd
                      ).pack(side=tk.LEFT, padx=1, pady=3)

        ttk.Separator(tb, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=3)

        tk.Button(tb, text="[cfg] Gan Tool",
                  bg="#f9e2af", fg="black",
                  font=("Consolas", 9, "bold"),
                  relief="flat", padx=6, pady=2,
                  command=self._open_zone_tool_config
                  ).pack(side=tk.LEFT, padx=1, pady=3)

        self.var_tool_lbl = tk.StringVar(value="Tool: NONE")
        tk.Label(tb, textvariable=self.var_tool_lbl,
                 bg="#252535", fg=ACC,
                 font=("Consolas", 9)
                 ).pack(side=tk.LEFT, padx=6)

        self.var_roi_cnt = tk.StringVar(value="ROI: 0")
        tk.Label(tb, textvariable=self.var_roi_cnt,
                 bg="#252535", fg="#a6e3a1",
                 font=("Consolas", 9, "bold")
                 ).pack(side=tk.RIGHT, padx=8)

        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(
            main, bg="black",
            width=self.DISP_W, height=self.DISP_H,
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground=ACC)
        self.canvas.pack(side=tk.LEFT,
                         fill=tk.BOTH, expand=True)

        for ev, cb in [
            ("<ButtonPress-1>",   self._on_press),
            ("<B1-Motion>",       self._on_drag),
            ("<ButtonRelease-1>", self._on_release),
            ("<Double-Button-1>", self._on_dbl),
            ("<Motion>",          self._on_hover),
            ("<Escape>",
             lambda e: self._cancel_drawing()),
        ]:
            self.canvas.bind(ev, cb)

        rp = tk.Frame(main, bg=BG, width=255)
        rp.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        rp.pack_propagate(False)

        def lbl(text, fg=FG):
            tk.Label(rp, text=text, bg=BG, fg=fg,
                     font=("Consolas", 9, "bold"),
                     anchor="w"
                     ).pack(fill=tk.X, padx=8,
                            pady=(8, 2))

        lbl("[brt] Brightness")
        tk.Scale(rp, from_=-100, to=100,
                 orient="horizontal",
                 variable=self.var_bright,
                 bg=BG, fg=FG, troughcolor="#313244",
                 highlightthickness=0,
                 font=("Consolas", 8)
                 ).pack(fill=tk.X, padx=8)

        lbl("[con] Contrast")
        tk.Scale(rp, from_=0.5, to=3.0,
                 orient="horizontal",
                 variable=self.var_contrast,
                 resolution=0.05,
                 bg=BG, fg=FG, troughcolor="#313244",
                 highlightthickness=0,
                 font=("Consolas", 8)
                 ).pack(fill=tk.X, padx=8)

        ttk.Separator(rp, orient="horizontal").pack(
            fill=tk.X, padx=8, pady=6)

        lbl("[crp] Crop vung:", fg="#fab387")
        self.lbl_crop_info = tk.Label(
            rp,
            text="Chua co crop\n(dung toan bo anh)",
            bg=BG, fg="#f9e2af",
            font=("Consolas", 8),
            wraplength=230, justify="left")
        self.lbl_crop_info.pack(padx=8, anchor="w")

        ttk.Separator(rp, orient="horizontal").pack(
            fill=tk.X, padx=8, pady=6)

        lbl("[lst] Vung & Tool:", fg="#89dceb")
        list_frame = tk.Frame(rp, bg=BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4)

        sb = ttk.Scrollbar(list_frame, orient="vertical")
        self.lb_zones = tk.Listbox(
            list_frame,
            bg="#181825", fg=FG,
            font=("Consolas", 8),
            selectbackground="#89b4fa",
            selectforeground="black",
            relief="flat", height=6,
            yscrollcommand=sb.set,
            activestyle="dotbox")
        sb.config(command=self.lb_zones.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_zones.pack(side=tk.LEFT,
                           fill=tk.BOTH, expand=True)
        self.lb_zones.bind(
            "<Double-Button-1>",
            lambda e:
            self._open_zone_tool_config_by_list())
        self.lb_zones.bind(
            "<<ListboxSelect>>", self._on_list_select)

        ttk.Separator(rp, orient="horizontal").pack(
            fill=tk.X, padx=8, pady=4)

        lbl("[img] Crop Preview:", fg="#a6e3a1")
        self.lbl_crop_prev = tk.Label(
            rp, bg="#0d0d1a", width=245, height=100)
        self.lbl_crop_prev.pack(padx=4, pady=2)

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill=tk.X, padx=4, pady=(0, 6))
        tk.Button(bot, text="[sav]  Luu & Dong",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 11, "bold"),
                  relief="flat", padx=20, pady=8,
                  command=self._save_and_close
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(0, 4))
        tk.Button(bot, text="[x]  Huy",
                  bg="#f38ba8", fg="black",
                  font=("Consolas", 11, "bold"),
                  relief="flat", padx=20, pady=8,
                  command=self._on_close_btn
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(4, 0))

        self._update_crop_info()
        self._update_roi_cnt()

    def _open_zone_tool_config(self):
        if not self.shapes:
            messagebox.showwarning(
                "Chu y",
                "Chua co vung nao!\n"
                "Ve it nhat 1 vung truoc.",
                parent=self)
            return
        sel = self.lb_zones.curselection()
        idx = sel[0] if sel else len(self.shapes) - 1
        self._open_zone_dialog(idx)

    def _open_zone_tool_config_by_list(self):
        sel = self.lb_zones.curselection()
        if not sel:
            return
        self._open_zone_dialog(sel[0])

    def _open_zone_dialog(self, idx: int):
        if idx < 0 or idx >= len(self.shapes):
            return
        try:
            from zone_tool_dialog import ZoneToolDialog
        except ImportError:
            messagebox.showerror(
                "Loi",
                "Khong tim thay zone_tool_dialog.py!",
                parent=self)
            return
        dlg = ZoneToolDialog(
            parent=self,
            zone_data=self.shapes[idx],
            cam_thread=self.app.cam_thread,
            zone_index=idx)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.shapes[idx] = dlg.result
            self._selected_idx = idx
            self._update_roi_cnt()
            self._update_zone_list()

    def _on_list_select(self, event):
        sel = self.lb_zones.curselection()
        if sel:
            self._selected_idx = sel[0]

    def _update_zone_list(self):
        self.lb_zones.delete(0, tk.END)
        for i, s in enumerate(self.shapes):
            t    = s.get("type", "?")
            name = s.get("name", f"Vung {i+1}")
            tool = s.get("tool", "—")
            ena  = "+" if s.get("enabled", True) else "-"
            self.lb_zones.insert(
                tk.END,
                f"{ena} #{i+1} [{t}] {name} > {tool}")
            tool_col = {
                "DefectDetection": "#f38ba8",
                "ShapeSearch":     "#f9e2af",
                "BlobAnalysis":    "#89dceb",
                "EdgeDetection":   "#cba6f7",
                "BinaryCheck":     "#a6e3a1",
            }.get(tool, "#cdd6f4")
            self.lb_zones.itemconfigure(i, fg=tool_col)

    def _update_loop(self):
        if not self._running:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        try:
            cam   = self.app.cam_thread
            frame = None
            if cam and cam.last_frame is not None:
                frame = cam.last_frame

            if frame is None:
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "No Camera",
                            (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (80, 80, 80), 2)

            alpha = float(self.var_contrast.get())
            beta  = int(self.var_bright.get())
            if alpha != 1.0 or beta != 0:
                frame = cv2.convertScaleAbs(
                    frame, alpha=alpha, beta=beta)

            fh, fw = frame.shape[:2]
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                self.after(self.UPDATE_INTERVAL_MS,
                           self._update_loop)
                return

            sc = min(cw / fw, ch / fh)
            dw = int(fw * sc)
            dh = int(fh * sc)
            ox = (cw - dw) // 2
            oy = (ch - dh) // 2

            self._img_w = dw; self._img_h = dh
            self._off_x = ox; self._off_y = oy
            self._fr_w  = fw; self._fr_h  = fh

            disp = cv2.resize(frame, (dw, dh),
                              interpolation=cv2.INTER_LINEAR)
            disp = self._draw_shapes(disp)

            bg = np.zeros((ch, cw, 3), np.uint8)
            bg[oy:oy + dh, ox:ox + dw] = disp

            rgb  = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            pimg = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(
                0, 0, anchor="nw", image=pimg)
            self.canvas._pimg = pimg

            if (self._crop and cam and
                    cam.last_frame is not None):
                try:
                    orig = cam.last_frame
                    if alpha != 1.0 or beta != 0:
                        orig = cv2.convertScaleAbs(
                            orig, alpha=alpha, beta=beta)
                    x0, y0, x1, y1 = self._crop
                    x0 = max(0, x0); y0 = max(0, y0)
                    x1 = min(fw, x1); y1 = min(fh, y1)
                    if x1 > x0 and y1 > y0:
                        cp  = orig[y0:y1, x0:x1]
                        ph, pw2 = cp.shape[:2]
                        sc2 = min(245/max(pw2,1),
                                  95/max(ph,1), 1.0)
                        if sc2 < 1.0:
                            cp = cv2.resize(
                                cp, None,
                                fx=sc2, fy=sc2,
                                interpolation=
                                cv2.INTER_LINEAR)
                        rgb2  = cv2.cvtColor(
                            cp, cv2.COLOR_BGR2RGB)
                        pimg2 = ImageTk.PhotoImage(
                            Image.fromarray(rgb2))
                        self.lbl_crop_prev.configure(
                            image=pimg2)
                        self.lbl_crop_prev._pimg = pimg2
                except Exception:
                    pass

        except Exception as e:
            print(f"[ROISetup] _update_loop: {e}")

        self.after(self.UPDATE_INTERVAL_MS,
                   self._update_loop)

    def _draw_shapes(self, disp):
        if not self.shapes and not self._crop:
            return disp
        ov = disp.copy()

        def r2i(rx, ry):
            return (
                int(rx * self._img_w /
                    max(self._fr_w, 1)),
                int(ry * self._img_h /
                    max(self._fr_h, 1))
            )

        for i, s in enumerate(self.shapes):
            t, d = s["type"], s["data"]
            tool = s.get("tool", None)
            col  = self.TOOL_COLORS.get(
                tool, self.COLORS.get(t, (0, 255, 0)))
            thick = 3 if i == self._selected_idx else 2
            try:
                if t == "rect":
                    p1 = r2i(d[0], d[1])
                    p2 = r2i(d[2], d[3])
                    cv2.rectangle(ov, p1, p2, col, thick)
                    fill = np.zeros_like(ov)
                    cv2.rectangle(fill, p1, p2, col, -1)
                    cv2.addWeighted(
                        ov, 1, fill, 0.12, 0, ov)
                elif t == "circle":
                    cx, cy = r2i(d[0], d[1])
                    pr = max(1, int(
                        d[2] * self._img_w /
                        max(self._fr_w, 1)))
                    cv2.circle(ov, (cx, cy),
                               pr, col, thick)
                elif t == "rot_rect":
                    box = cv2.boxPoints(
                        ((d[0], d[1]),
                         (d[2], d[3]), d[4]))
                    pts = np.array(
                        [r2i(x, y) for x, y in box],
                        dtype=np.int32)
                    cv2.polylines(
                        ov, [pts], True, col, thick)
                elif t == "polygon":
                    pts = np.array(
                        [r2i(x, y) for x, y in d],
                        dtype=np.int32)
                    cv2.polylines(
                        ov, [pts], True, col, thick)

                name      = s.get("name", f"#{i+1}")
                tool_s    = (tool or "—")[:12]
                enabled   = s.get("enabled", True)
                label_txt = (f"{name}:{tool_s}"
                             if enabled
                             else f"[OFF]{name}")

                if t == "rect":
                    tx, ty = r2i(d[0], d[1])
                elif t == "circle":
                    tx, ty = r2i(d[0], d[1] - d[2] - 5)
                elif t in ("rot_rect", "polygon"):
                    tx, ty = r2i(d[0], d[1])
                else:
                    tx, ty = 10, 10 + i * 20

                (tw, th), _ = cv2.getTextSize(
                    label_txt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(ov,
                              (tx, ty - th - 2),
                              (tx + tw + 4, ty + 2),
                              (0, 0, 0), -1)
                cv2.putText(ov, label_txt, (tx + 2, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1)
            except Exception:
                pass

        if self._crop:
            cx0, cy0 = r2i(self._crop[0], self._crop[1])
            cx1, cy1 = r2i(self._crop[2], self._crop[3])
            cv2.rectangle(ov, (cx0, cy0), (cx1, cy1),
                          (255, 140, 0), 3)
            cv2.putText(ov, "CROP",
                        (cx0 + 4, cy0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 140, 0), 2)

        cv2.addWeighted(ov, 0.75, disp, 0.25, 0, disp)

        col_t = (200, 200, 50)
        if self._drawing and self._current:
            ix, iy = self._current
            if self.tool.get() == "rect" and self._start:
                cv2.rectangle(disp, self._start,
                              (ix, iy), col_t, 1)
            elif (self.tool.get() == "circle"
                  and self._start):
                r = int(math.hypot(
                    ix - self._start[0],
                    iy - self._start[1]))
                cv2.circle(disp, self._start,
                           r, col_t, 1)
            elif (self.tool.get() == "rot_rect"
                  and len(self._rot_pts) >= 1):
                cv2.line(disp, self._rot_pts[0],
                         (ix, iy), col_t, 1)

        if (self.tool.get() == "polygon"
                and self._poly_pts):
            for k in range(1, len(self._poly_pts)):
                cv2.line(disp, self._poly_pts[k-1],
                         self._poly_pts[k], col_t, 1)
            if self._current:
                cv2.line(disp, self._poly_pts[-1],
                         self._current, col_t, 1)
            for pt in self._poly_pts:
                cv2.circle(disp, pt, 4, col_t, -1)

        return disp

    def _set_tool(self, t):
        self.tool.set(t)
        self.var_tool_lbl.set(f"Tool: {t.upper()}")
        colors = {
            "none":     "#313244",
            "rect":     "#a6e3a1",
            "rot_rect": "#89dceb",
            "circle":   "#cba6f7",
            "polygon":  "#f9e2af",
            "move":     "#89b4fa",
        }
        for k, btn in self._tool_btns.items():
            btn.configure(
                relief="sunken" if k == t else "flat",
                bg=colors.get(k, "#313244"))
        self.canvas.configure(
            cursor="crosshair"
            if t != "none" else "arrow")

    def _undo(self):
        if self.shapes:
            self.shapes.pop()
            self._selected_idx = len(self.shapes) - 1
            self._update_roi_cnt()
            self._update_zone_list()

    def _clear(self):
        self.shapes        = []
        self._poly_pts     = []
        self._rot_pts      = []
        self._drawing      = False
        self._selected_idx = -1
        self._update_roi_cnt()
        self._update_zone_list()

    def _cancel_drawing(self):
        self._drawing  = False
        self._poly_pts = []
        self._rot_pts  = []
        self._start    = None
        self._current  = None

    def _update_roi_cnt(self):
        self.var_roi_cnt.set(f"ROI: {len(self.shapes)}")

    def _set_crop_from_rect(self):
        for s in self.shapes:
            if s["type"] == "rect":
                d = s["data"]
                self._crop = [d[0], d[1], d[2], d[3]]
                self._update_crop_info()
                return
        messagebox.showwarning(
            "Chu y", "Ve it nhat 1 Rect!", parent=self)

    def _clear_crop(self):
        self._crop = None
        self._update_crop_info()

    def _update_crop_info(self):
        if self._crop:
            x0, y0, x1, y1 = self._crop
            w, h = x1 - x0, y1 - y0
            self.lbl_crop_info.configure(
                text=f"x:{x0} y:{y0}\nw:{w} h:{h}")
        else:
            self.lbl_crop_info.configure(
                text="Chua co crop\n(dung toan bo anh)")

    def _c2i(self, cx, cy):
        return (
            max(0, min(self._img_w-1, cx-self._off_x)),
            max(0, min(self._img_h-1, cy-self._off_y))
        )

    def _i2r(self, ix, iy):
        return (
            int(ix * self._fr_w / max(self._img_w, 1)),
            int(iy * self._fr_h / max(self._img_h, 1))
        )

    def _r2i(self, rx, ry):
        return (
            int(rx * self._img_w / max(self._fr_w, 1)),
            int(ry * self._img_h / max(self._fr_h, 1))
        )

    def _on_press(self, e):
        t = self.tool.get()
        if t == "none":
            return
        ix, iy = self._c2i(e.x, e.y)
        if t == "move":
            self._drag_idx = self._hit_test(ix, iy)
            if self._drag_idx >= 0:
                cx, cy = self._shape_center(
                    self._drag_idx)
                self._drag_off = (ix-cx, iy-cy)
                self._selected_idx = self._drag_idx
                self._update_zone_list()
                self.lb_zones.selection_clear(0, tk.END)
                self.lb_zones.selection_set(
                    self._drag_idx)
            return
        if t == "polygon":
            self._poly_pts.append((ix, iy))
            return
        if t == "rot_rect":
            self._rot_pts.append((ix, iy))
            if len(self._rot_pts) == 1:
                self._drawing = True
                self._current = (ix, iy)
            return
        self._drawing = True
        self._start   = (ix, iy)
        self._current = (ix, iy)

    def _on_drag(self, e):
        ix, iy = self._c2i(e.x, e.y)
        self._current = (ix, iy)
        if (self.tool.get() == "move"
                and self._drag_idx >= 0):
            self._move_shape(
                self._drag_idx,
                ix - self._drag_off[0],
                iy - self._drag_off[1])

    def _on_release(self, e):
        t = self.tool.get()
        ix, iy = self._c2i(e.x, e.y)
        if t == "move":
            self._drag_idx = -1
            return
        if t in ("rect", "circle") and self._drawing:
            self._drawing = False
            rx0, ry0 = self._i2r(*self._start)
            rx1, ry1 = self._i2r(ix, iy)
            if t == "rect":
                x0 = min(rx0, rx1); y0 = min(ry0, ry1)
                x1 = max(rx0, rx1); y1 = max(ry0, ry1)
                if abs(x1-x0) > 5 and abs(y1-y0) > 5:
                    self.shapes.append({
                        "type": "rect",
                        "data": [x0, y0, x1, y1],
                        "tool": None, "tool_cfg": {},
                        "name": f"Vung {len(self.shapes)+1}",
                        "enabled": True, "template": "",
                    })
            elif t == "circle":
                r = int(math.hypot(rx1-rx0, ry1-ry0))
                if r > 5:
                    self.shapes.append({
                        "type": "circle",
                        "data": [rx0, ry0, r],
                        "tool": None, "tool_cfg": {},
                        "name": f"Vung {len(self.shapes)+1}",
                        "enabled": True, "template": "",
                    })
            self._start = self._current = None
            self._selected_idx = len(self.shapes) - 1
            self._update_roi_cnt()
            self._update_zone_list()

        if t == "rot_rect" and len(self._rot_pts) == 2:
            self._drawing = False
            p1 = self._i2r(*self._rot_pts[0])
            p2 = self._i2r(*self._rot_pts[1])
            p3 = self._i2r(ix, iy)
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            L  = max(math.hypot(dx, dy), 1)
            nx, ny = -dy/L, dx/L
            h_val = abs((p3[0]-p1[0])*nx +
                        (p3[1]-p1[1])*ny)
            ang   = math.degrees(math.atan2(dy, dx))
            cx_v  = (p1[0]+p2[0])//2
            cy_v  = (p1[1]+p2[1])//2
            w_v   = int(L)
            h_v   = max(int(h_val), 5)
            if w_v > 5:
                self.shapes.append({
                    "type": "rot_rect",
                    "data": [cx_v, cy_v, w_v, h_v, ang],
                    "tool": None, "tool_cfg": {},
                    "name": f"Vung {len(self.shapes)+1}",
                    "enabled": True, "template": "",
                })
            self._rot_pts  = []
            self._drawing  = False
            self._selected_idx = len(self.shapes) - 1
            self._update_roi_cnt()
            self._update_zone_list()

    def _on_dbl(self, e):
        if (self.tool.get() == "polygon"
                and len(self._poly_pts) >= 3):
            pts = [self._i2r(x, y)
                   for x, y in self._poly_pts]
            self.shapes.append({
                "type": "polygon",
                "data": pts,
                "tool": None, "tool_cfg": {},
                "name": f"Vung {len(self.shapes)+1}",
                "enabled": True, "template": "",
            })
            self._poly_pts     = []
            self._selected_idx = len(self.shapes) - 1
            self._update_roi_cnt()
            self._update_zone_list()

    def _on_hover(self, e):
        ix, iy = self._c2i(e.x, e.y)
        self._current = (ix, iy)

    def _hit_test(self, ix, iy):
        rx, ry = self._i2r(ix, iy)
        for i, s in reversed(list(enumerate(self.shapes))):
            t, d = s["type"], s["data"]
            try:
                if t == "rect":
                    if (d[0] <= rx <= d[2]
                            and d[1] <= ry <= d[3]):
                        return i
                elif t == "circle":
                    if math.hypot(
                            rx-d[0], ry-d[1]) <= d[2]:
                        return i
                elif t in ("polygon", "rot_rect"):
                    pts = self._get_contour(i)
                    if (pts is not None and
                            cv2.pointPolygonTest(
                                pts,
                                (float(rx), float(ry)),
                                False) >= 0):
                        return i
            except Exception:
                pass
        return -1

    def _shape_center(self, idx):
        s    = self.shapes[idx]
        d, t = s["data"], s["type"]
        if t == "rect":
            rx = (d[0]+d[2])//2; ry = (d[1]+d[3])//2
        elif t == "circle":
            rx, ry = d[0], d[1]
        elif t == "rot_rect":
            rx, ry = d[0], d[1]
        elif t == "polygon":
            rx = sum(p[0] for p in d)//len(d)
            ry = sum(p[1] for p in d)//len(d)
        else:
            rx = ry = 0
        return self._r2i(rx, ry)

    def _get_contour(self, idx):
        s    = self.shapes[idx]
        t, d = s["type"], s["data"]
        if t == "rect":
            return np.array(
                [[d[0],d[1]], [d[2],d[1]],
                 [d[2],d[3]], [d[0],d[3]]],
                dtype=np.float32)
        elif t == "rot_rect":
            return cv2.boxPoints(
                ((d[0],d[1]), (d[2],d[3]), d[4])
            ).astype(np.float32)
        elif t == "polygon":
            return np.array(d, dtype=np.float32)
        return None

    def _move_shape(self, idx, ix, iy):
        rx, ry = self._i2r(ix, iy)
        s      = self.shapes[idx]
        t, d   = s["type"], s["data"]
        if t == "rect":
            hw = (d[2]-d[0])//2; hh = (d[3]-d[1])//2
            s["data"] = [rx-hw, ry-hh, rx+hw, ry+hh]
        elif t == "circle":
            s["data"] = [rx, ry, d[2]]
        elif t == "rot_rect":
            s["data"] = [rx, ry, d[2], d[3], d[4]]
        elif t == "polygon":
            ocx = sum(p[0] for p in d)//len(d)
            ocy = sum(p[1] for p in d)//len(d)
            dx2, dy2 = rx-ocx, ry-ocy
            s["data"] = [[p[0]+dx2, p[1]+dy2]
                         for p in d]

    def _save_and_close(self):
        import json as _json
        self.app.cfg["roi_shapes"] = _json.loads(
            _json.dumps(self.shapes))
        self.app.cfg["roi_crop"]       = self._crop
        self.app.cfg["roi_brightness"] = \
            int(self.var_bright.get())
        self.app.cfg["roi_contrast"]   = round(
            float(self.var_contrast.get()), 2)
        self.app.cfg["roi_frame_w"]    = self._fr_w
        self.app.cfg["roi_frame_h"]    = self._fr_h
        save_cfg(self.app.cfg)

        n         = len(self.shapes)
        crop      = "Y" if self._crop else "N"
        b         = int(self.var_bright.get())
        c         = float(self.var_contrast.get())
        tools_set = set(s.get("tool") or "—"
                        for s in self.shapes)
        tools_str = ",".join(
            t for t in tools_set if t != "—")

        self.app.var_roi_info.set(
            f"ROI:{n} Crop:{crop} "
            f"B:{b} C:{c:.1f} "
            f"Tools:[{tools_str or '—'}]")

        self._running = False
        self.destroy()

    def _on_close_btn(self):
        self._running = False
        self.destroy()


# ════════════════════════════════════════════════════════════════
# HELP DIALOG
# ════════════════════════════════════════════════════════════════
class HelpDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("[hlp] Huong Dan Su Dung")
        self.configure(bg="#1e1e2e")
        self.geometry("900x700")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        self._build_ui()

    def _build_ui(self):
        BG  = "#1e1e2e"
        FG  = "#cdd6f4"
        ACC = "#89b4fa"

        tk.Label(self,
                 text="HUONG DAN SU DUNG - JETSON INSPECTOR",
                 bg=BG, fg=ACC,
                 font=("Consolas", 13, "bold")
                 ).pack(pady=(12, 4))
        ttk.Separator(self, orient="horizontal"
                      ).pack(fill=tk.X, padx=12)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        style = ttk.Style(self)
        style.configure("TNotebook",     background=BG)
        style.configure("TNotebook.Tab",
                        background="#252535",
                        foreground=FG,
                        font=("Consolas", 9, "bold"),
                        padding=[8, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", "#313244")],
                  foreground=[("selected", ACC)])

        for name, builder in [
            ("[run] Bat dau",    self._tab_quickstart),
            ("[key] Phim tat",   self._tab_shortcuts),
            ("[cam] Camera",     self._tab_camera),
            ("[pre] Tien xu ly", self._tab_preprocess),
            ("[grd] Gradient",   self._tab_gradient),
            ("[mph] Morphology", self._tab_morphology),
            ("[roi] ROI",        self._tab_roi),
            ("[cls] Classify",   self._tab_classify),
            ("[sav] Output",     self._tab_output),
            ("[tool] Tools",     self._tab_tools),
        ]:
            frame = ttk.Frame(nb)
            nb.add(frame, text=name)
            builder(frame)

        tk.Button(self, text="[x]  Dong",
                  bg="#f38ba8", fg="black",
                  font=("Consolas", 10, "bold"),
                  command=self.destroy,
                  relief="flat", padx=20, pady=6
                  ).pack(pady=(0, 10))

    def _make_text(self, parent):
        frame = tk.Frame(parent, bg="#1e1e2e")
        frame.pack(fill=tk.BOTH, expand=True,
                   padx=4, pady=4)
        sb  = ttk.Scrollbar(frame, orient="vertical")
        txt = tk.Text(frame,
                      bg="#181825", fg="#cdd6f4",
                      font=("Consolas", 10),
                      yscrollcommand=sb.set,
                      wrap=tk.WORD, relief="flat",
                      padx=12, pady=8,
                      cursor="arrow",
                      state=tk.NORMAL)
        sb.config(command=txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, fg_c, font_s in [
            ("h1",     "#cba6f7", ("Consolas", 12, "bold")),
            ("h2",     "#89b4fa", ("Consolas", 10, "bold")),
            ("h3",     "#89dceb", ("Consolas", 10, "bold")),
            ("key",    "#a6e3a1", ("Consolas", 10, "bold")),
            ("val",    "#f9e2af", ("Consolas", 10)),
            ("normal", "#cdd6f4", ("Consolas", 10)),
            ("note",   "#f38ba8", ("Consolas", 10, "italic")),
            ("tip",    "#a6e3a1", ("Consolas", 10, "italic")),
            ("code",   "#cdd6f4", ("Consolas", 9)),
        ]:
            kw = {"foreground": fg_c, "font": font_s}
            if tag == "code":
                kw["background"] = "#313244"
            txt.tag_config(tag, **kw)
        return txt

    def _write(self, txt, content):
        txt.config(state=tk.NORMAL)
        for tag, text in content:
            txt.insert(tk.END, text, tag)
        txt.config(state=tk.DISABLED)

    # ════════════════════════════════════════════════
    def _tab_quickstart(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "HUONG DAN BAT DAU NHANH\n\n"),

            ("h2",    "Buoc 1: Cai dat Camera\n"),
            ("normal","  Mo Config > Tab Camera\n"),
            ("normal","  Chon loai camera: USB / CSI / IMV\n"),
            ("normal","  Dat FPS <= 30 (Jetson)\n"),
            ("normal","  Nhan 'Test Camera' de kiem tra\n\n"),

            ("h2",    "Buoc 2: Thiet lap ROI\n"),
            ("normal","  Nhan [R] hoac nut ROI Setup\n"),
            ("normal","  Ve vung phan tich tren anh\n"),
            ("normal","  Gan Tool cho tung vung\n"),
            ("normal","  Luu va Dong\n\n"),

            ("h2",    "Buoc 3: Chinh thong so\n"),
            ("normal","  Mo Config > Tung tab\n"),
            ("normal","  Chinh Gradient, Morphology, Classify\n"),
            ("normal","  Nhan 'Xem truoc' de kiem tra\n\n"),

            ("h2",    "Buoc 4: Luu Model\n"),
            ("normal","  Nhan nut Model\n"),
            ("normal","  Nhap ten model\n"),
            ("normal","  Nhan 'Luu Model'\n\n"),

            ("h2",    "Buoc 5: Chup va xu ly\n"),
            ("normal","  Nhan [C] de bat dau chup\n"),
            ("normal","  Nhan [T] de dung + xu ly\n"),
            ("normal","  Xem ket qua OK/NG\n\n"),

            ("note",  "  Jetson toi uu:\n"
                      "  - n_threads: 2-4\n"
                      "  - blur_ksize: 3-5\n"
                      "  - FPS: <= 30\n"
                      "  - bilateral: tat neu can nhanh\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_shortcuts(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "PHIM TAT\n\n"),
            ("h2",    "Dieu khien chinh:\n"),
            ("key",   "  [C]      "),
            ("normal","Bat dau chup anh\n"),
            ("key",   "  [T]      "),
            ("normal","Dung chup + xu ly\n"),
            ("key",   "  [Q]      "),
            ("normal","Thoat chuong trinh\n\n"),
            ("h2",    "Chuyen man hinh:\n"),
            ("key",   "  [R]      "),
            ("normal","Mo ROI Setup\n"),
            ("key",   "  [I]      "),
            ("normal","Mo Config\n"),
            ("key",   "  [M]      "),
            ("normal","Mo quan ly Model\n"),
            ("key",   "  [F1]     "),
            ("normal","Mo Help\n\n"),
            ("h2",    "Xem ket qua:\n"),
            ("key",   "  [<-]     "),
            ("normal","Anh truoc\n"),
            ("key",   "  [->]     "),
            ("normal","Anh sau\n\n"),
            ("h2",    "Loc ket qua:\n"),
            ("normal","  Nhan nut [ok]OK  : chi hien anh OK\n"),
            ("normal","  Nhan nut [ng]NG  : chi hien anh NG\n"),
            ("normal","  Nhan nut [dir]ALL: hien tat ca\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_camera(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "CAI DAT CAMERA\n\n"),

            ("h2",    "Loai Camera:\n"),
            ("key",   "  USB      "),
            ("normal","Webcam USB thong thuong\n"),
            ("normal","           camera_id = 0,1,2...\n\n"),
            ("key",   "  CSI      "),
            ("normal","Camera Jetson (MIPI CSI)\n"),
            ("normal","           camera_csi_id = 0 hoac 1\n\n"),
            ("key",   "  IP       "),
            ("normal","Camera mang RTSP\n"),
            ("normal","           camera_ip_url = rtsp://...\n\n"),
            ("key",   "  IMV      "),
            ("normal","Camera iRAYple (cong nghiep)\n"),
            ("normal","           camera_id = 0\n\n"),

            ("h2",    "Do phan giai & FPS:\n"),
            ("key",   "  Width/Height "),
            ("normal","Kich thuoc frame\n"),
            ("normal","               IMV: 256~1440 x 64~1080\n"),
            ("key",   "  FPS         "),
            ("normal","Toc do khung hinh\n"),
            ("note",  "               Jetson: nen <= 30 FPS\n\n"),

            ("h2",    "IMV Camera - Exposure & Gain:\n"),
            ("key",   "  Exposure (us) "),
            ("normal","Thoi gian phoi sang\n"),
            ("val",   "               -1     "),
            ("normal","= tu dong (Auto)\n"),
            ("val",   "               1000   "),
            ("normal","= 1ms (rat nhanh, toi)\n"),
            ("val",   "               10000  "),
            ("normal","= 10ms (binh thuong)\n"),
            ("val",   "               100000 "),
            ("normal","= 100ms (cham, sang)\n\n"),
            ("key",   "  Gain (dB)    "),
            ("normal","Do khuech dai tin hieu\n"),
            ("val",   "               -1     "),
            ("normal","= tu dong (Auto)\n"),
            ("val",   "               0.0    "),
            ("normal","= khong khuech dai\n"),
            ("val",   "               12.0   "),
            ("normal","= khuech dai vua\n"),
            ("val",   "               24.0   "),
            ("normal","= khuech dai toi da (nhieu nhieu)\n\n"),

            ("tip",   "  Meo: Bat dau voi Auto ON,\n"
                      "  sau do tat Auto va chinh thu cong\n"
                      "  de on dinh ket qua phan tich\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_preprocess(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "TIEN XU LY ANH\n\n"),
            ("normal","Muc dich: Lam giam nhieu truoc khi\n"
                      "phat hien loi. Anh sach = phat hien\n"
                      "chinh xac hon.\n\n"),

            ("h2",    "1. Gaussian Blur:\n"),
            ("normal","   Lam mo anh de giam nhieu\n\n"),
            ("key",   "  blur_ksize  "),
            ("normal","Kich thuoc kernel (so le)\n"),
            ("val",   "               1      "),
            ("normal","= khong blur\n"),
            ("val",   "               3-5    "),
            ("normal","= blur nhe (khuyen nghi)\n"),
            ("val",   "               7-11   "),
            ("normal","= blur manh\n"),
            ("val",   "               13+    "),
            ("normal","= blur rat manh, mat chi tiet\n\n"),
            ("key",   "  blur_sigma  "),
            ("normal","Do manh blur\n"),
            ("val",   "               0      "),
            ("normal","= tu tinh tu ksize (khuyen nghi)\n\n"),

            ("h2",    "2. Bilateral Filter:\n"),
            ("normal","   Lam mo NHUNG GIU LAI bien canh\n"
                      "   Cham hon Gaussian, chat luong tot hon\n\n"),
            ("key",   "  bilateral_enable "),
            ("normal","Bat/tat\n"),
            ("note",  "                   Tat de tang toc do\n\n"),
            ("key",   "  bilateral_d      "),
            ("normal","Duong kinh vung loc\n"),
            ("val",   "                   5-9  "),
            ("normal","= nhanh\n"),
            ("val",   "                   11+  "),
            ("normal","= cham hon, tot hon\n\n"),
            ("key",   "  bilateral_sC     "),
            ("normal","Nguong mau\n"),
            ("normal","                   Cao = blur manh giua cac mau khac nhau\n"),
            ("val",   "                   50-100 "),
            ("normal","thuong dung\n\n"),
            ("key",   "  bilateral_sS     "),
            ("normal","Nguong khong gian\n"),
            ("normal","                   Cao = lay pixel xa hon\n"),
            ("val",   "                   50-100 "),
            ("normal","thuong dung\n\n"),

            ("h2",    "3. Thu tu xu ly:\n"),
            ("val",   "  blur_bilateral  "),
            ("normal","Gaussian truoc, Bilateral sau\n"),
            ("normal","                  (mac dinh, tot nhat)\n"),
            ("val",   "  bilateral_blur  "),
            ("normal","Bilateral truoc, Gaussian sau\n"),
            ("val",   "  blur_only       "),
            ("normal","Chi Gaussian (nhanh nhat)\n"),
            ("val",   "  bilateral_only  "),
            ("normal","Chi Bilateral\n\n"),

            ("h2",    "4. CLAHE:\n"),
            ("normal","   Tang tuong phan cuc bo\n"
                      "   Lam noi bat chi tiet be mat\n\n"),
            ("key",   "  clahe_enable  "),
            ("normal","Bat/tat CLAHE\n\n"),
            ("key",   "  clahe_clip    "),
            ("normal","Gioi han tang tuong phan\n"),
            ("val",   "                1-2    "),
            ("normal","= tang nhe\n"),
            ("val",   "                3-4    "),
            ("normal","= tang vua (khuyen nghi)\n"),
            ("val",   "                6-8    "),
            ("normal","= tang manh, co the tao nhieu\n\n"),
            ("key",   "  clahe_grid_X/Y "),
            ("normal","Chia anh thanh luoi X x Y o\n"),
            ("val",   "                4-8    "),
            ("normal","= tuong phan dong deu\n"),
            ("val",   "                8-16   "),
            ("normal","= tuong phan cuc bo (khuyen nghi)\n"),
            ("val",   "                20+    "),
            ("normal","= cuc bo rat chi tiet\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_gradient(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "PHAT HIEN GRADIENT (LOI)\n\n"),
            ("normal","Phat hien vung co su thay doi\n"
                      "do sang dot ngot = bien canh = loi\n\n"),

            ("h2",    "1. Phuong phap Gradient:\n\n"),
            ("key",   "  sobel     "),
            ("normal","Can bang toc do va chat luong\n"),
            ("normal","            Khuyen nghi cho da so truong hop\n\n"),
            ("key",   "  scharr    "),
            ("normal","Chinh xac hon Sobel voi kernel nho\n"),
            ("normal","            Tot cho loi nho, bien mong\n\n"),
            ("key",   "  laplacian "),
            ("normal","Phat hien bien 2 chieu\n"),
            ("normal","            Nhay hon, dung khi loi co bien ro\n\n"),
            ("key",   "  canny     "),
            ("normal","Phat hien bien sac net, it nhieu\n"),
            ("normal","            Tot nhat cho anh co nhieu nhieu\n\n"),

            ("key",   "  sobel_ksize "),
            ("normal","Kich thuoc kernel Sobel\n"),
            ("val",   "               1      "),
            ("normal","= bien rat mong, nhay\n"),
            ("val",   "               3      "),
            ("normal","= binh thuong (khuyen nghi)\n"),
            ("val",   "               5-7    "),
            ("normal","= bien day, it nhieu hon\n\n"),

            ("h2",    "2. Nguong Gradient (grad_thr):\n\n"),
            ("normal","   Phan biet vung OK va loi\n\n"),
            ("normal","   Pixel gradient > grad_thr = DO (loi)\n"),
            ("normal","   Pixel gradient < grad_thr = XANH (OK)\n\n"),
            ("val",   "  grad_thr thap (5-15)  "),
            ("normal","= rat nhay\n"),
            ("normal","                         Phat hien nhieu loi hon\n"),
            ("note",  "                         De bao nham\n\n"),
            ("val",   "  grad_thr vua  (15-30) "),
            ("normal","= can bang (khuyen nghi)\n\n"),
            ("val",   "  grad_thr cao  (30-60) "),
            ("normal","= it nhay\n"),
            ("normal","                         Bo qua loi nho\n"),
            ("tip",   "                         It bao nham\n\n"),

            ("h2",    "3. Canny (chi khi method=canny):\n\n"),
            ("key",   "  canny_thr1 "),
            ("normal","Nguong thap - bien yeu\n"),
            ("key",   "  canny_thr2 "),
            ("normal","Nguong cao  - bien manh\n"),
            ("val",   "              Ti le thr2/thr1 "),
            ("normal","nen = 2~3\n"),
            ("val",   "              Vi du: 50/150, 100/200\n\n"),

            ("h2",    "4. Loai Threshold:\n\n"),
            ("key",   "  binary_inv "),
            ("normal","Gradient cao = DO (loi)\n"),
            ("normal","             Pho bien nhat\n\n"),
            ("key",   "  otsu       "),
            ("normal","Tu dong tim nguong toi uu\n"),
            ("normal","             Khong can chinh grad_thr\n"),
            ("normal","             Tot khi do sang thay doi\n\n"),
            ("key",   "  adaptive   "),
            ("normal","Nguong thay doi theo vung\n"),
            ("normal","             Tot cho be mat khong dong deu\n\n"),
            ("key",   "  adaptive_block "),
            ("normal","Kich thuoc vung tinh nguong\n"),
            ("val",   "                  Le, 3-51\n"),
            ("key",   "  adaptive_c     "),
            ("normal","Hang so hieu chinh (-20 ~ 20)\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_morphology(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "XU LY MORPHOLOGY\n\n"),
            ("normal","Lam sach ket qua sau threshold:\n"
                      "- Xoa nhieu nho\n"
                      "- Lap day lo hong\n"
                      "- Ket noi vung roi rac\n\n"),

            ("h2",    "1. Hinh dang Kernel:\n\n"),
            ("key",   "  ellipse  "),
            ("normal","Hinh ellipse - tu nhien nhat\n"),
            ("normal","             Khuyen nghi cho be mat tron\n\n"),
            ("key",   "  rect     "),
            ("normal","Hinh vuong - phu hop goc can\n\n"),
            ("key",   "  cross    "),
            ("normal","Hinh thap - nhe hon\n\n"),

            ("h2",    "2. Morphology Open (morph_open_iter):\n\n"),
            ("normal","   Quy trinh: EROSION → DILATION\n"),
            ("normal","   Tac dung : XOA nhieu nho (cham do nho)\n\n"),
            ("val",   "  1-2 lan  "),
            ("normal","Xoa nhieu nho (khuyen nghi)\n"),
            ("val",   "  3-5 lan  "),
            ("normal","Xoa nhieu lon hon\n"),
            ("note",  "           Qua nhieu = mat chi tiet loi\n\n"),

            ("h2",    "3. Morphology Close (morph_close_iter):\n\n"),
            ("normal","   Quy trinh: DILATION → EROSION\n"),
            ("normal","   Tac dung : LAP DAY lo hong nho\n\n"),
            ("val",   "  1-2 lan  "),
            ("normal","Lap lo nho\n"),
            ("val",   "  3-5 lan  "),
            ("normal","Lap lo lon hon (khuyen nghi 3)\n"),
            ("note",  "           Qua nhieu = ket noi vung khong lien quan\n\n"),

            ("h2",    "4. Morphology Them (tuy chon):\n\n"),
            ("key",   "  morph_extra_enable "),
            ("normal","Bat them 1 buoc xu ly\n\n"),
            ("key",   "  morph_extra_type:\n"),
            ("val",   "    dilate   "),
            ("normal","Mo rong vung loi\n"),
            ("normal","              Dung khi loi bi mat sau open\n\n"),
            ("val",   "    erode    "),
            ("normal","Thu hep vung loi\n"),
            ("normal","              Dung khi qua nhieu nhieu\n\n"),
            ("val",   "    gradient "),
            ("normal","Lay vien cua vung loi\n"),
            ("normal","              Hien thi duong bien loi\n\n"),
            ("key",   "  morph_extra_iter "),
            ("normal","So lan lap (1-5)\n\n"),

            ("tip",   "  Meo thuc te:\n"
                      "  - Anh nhieu nhieu: tang open_iter\n"
                      "  - Loi bi ngat quang: tang close_iter\n"
                      "  - Mat loi nho: giam open_iter\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_roi(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "ROI MASK - VUNG PHAN TICH\n\n"),
            ("normal","Tim vung vat the trong anh\n"
                      "Chi phan tich trong vung nay\n\n"),

            ("h2",    "1. Nguong tim vat the:\n\n"),
            ("key",   "  roi_thresh      "),
            ("normal","Nguong sang/toi\n"),
            ("val",   "                   50-100  "),
            ("normal","= lay vung sang\n"),
            ("val",   "                   150-200 "),
            ("normal","= chi lay vung rat sang\n\n"),
            ("key",   "  roi_thresh_type:\n"),
            ("val",   "    binary     "),
            ("normal","Lay vung SANG hon nguong\n"),
            ("normal","               Dung khi vat the sang, nen toi\n\n"),
            ("val",   "    binary_inv "),
            ("normal","Lay vung TOI hon nguong\n"),
            ("normal","               Dung khi vat the toi, nen sang\n\n"),
            ("val",   "    otsu       "),
            ("normal","Tu dong tim nguong\n"),
            ("normal","               Tot khi do sang khong on dinh\n\n"),

            ("h2",    "2. Mo rong bien ROI:\n\n"),
            ("key",   "  roi_offset "),
            ("normal","Mo rong vien phan tich ra ngoai (px)\n"),
            ("normal","             = Do day vung kiem tra xung quanh vat\n\n"),
            ("val",   "  30-50   "),
            ("normal","Kiem tra vung hep xung quanh\n"),
            ("val",   "  50-100  "),
            ("normal","Kiem tra vung rong hon (khuyen nghi)\n"),
            ("val",   "  100+    "),
            ("normal","Kiem tra rat rong\n\n"),
            ("note",  "  Luu y: roi_offset qua lon se\n"
                      "  cham xu ly tren Jetson\n\n"),

            ("h2",    "3. Loc vung ROI:\n\n"),
            ("key",   "  roi_min_area    "),
            ("normal","Dien tich toi thieu vung ROI (px2)\n"),
            ("normal","                   Loai vung qua nho (nhieu nen)\n\n"),
            ("val",   "  500-1000  "),
            ("normal","Giu cac vung kha lon\n"),
            ("val",   "  1000-5000 "),
            ("normal","Chi giu vung lon (khuyen nghi)\n\n"),
            ("key",   "  roi_contour_approx "),
            ("normal","Lam min duong vien ROI\n"),
            ("val",   "                      Bat  "),
            ("normal","= vien muot hon\n"),
            ("val",   "                      Tat  "),
            ("normal","= vien chinh xac hon\n\n"),

            ("h2",    "4. Cong cu ve ROI (ROI Setup):\n\n"),
            ("key",   "  [rec] Rect   "),
            ("normal","Ve hinh chu nhat\n"),
            ("key",   "  [cir] Circle "),
            ("normal","Ve hinh tron\n"),
            ("key",   "  [hex] Polygon"),
            ("normal","Ve da giac (DblClick ket thuc)\n"),
            ("key",   "  [rot] Rot    "),
            ("normal","Ve hinh chu nhat xoay\n"),
            ("key",   "  [mov] Move   "),
            ("normal","Di chuyen vung\n\n"),
            ("key",   "  [crp] Set Crop "),
            ("normal","Dat vung crop tren anh\n"),
            ("normal","                  Anh luu = anh da crop\n"),
            ("key",   "  [cfg] Gan Tool "),
            ("normal","Gan Tool phan tich cho vung\n\n"),

            ("tip",   "  Meo: Ve Rect truoc, dat Crop,\n"
                      "  roi gan Tool de phan tich\n"
                      "  chinh xac vung can thiet\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_classify(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "DANH GIA OK/NG\n\n"),
            ("normal","2 buoc danh gia:\n"),
            ("normal","  Buoc A: Kiem tra ty le pixel OK\n"),
            ("normal","  Buoc B: Phan tich chi tiet vung loi\n\n"),

            ("h2",    "BUOC A - Ty le OK:\n\n"),
            ("key",   "  ok_ratio "),
            ("normal","Ty le pixel XANH toi thieu de pass\n\n"),
            ("val",   "  0.85-0.90 "),
            ("normal","= de tinh (bao quat)\n"),
            ("val",   "  0.95-0.98 "),
            ("normal","= binh thuong (khuyen nghi)\n"),
            ("val",   "  0.99      "),
            ("normal","= rat chat che\n\n"),
            ("normal","  Neu ratio >= ok_ratio → OK\n"),
            ("normal","  Neu ratio <  ok_ratio → tiep tuc Buoc B\n\n"),

            ("h2",    "BUOC B - Phan tich loi chi tiet:\n\n"),

            ("h3",    "  Loc loi:\n"),
            ("key",   "  min_defect_area      "),
            ("normal","Bo qua loi qua nho\n"),
            ("val",   "                        1-10  "),
            ("normal","= giu loi rat nho\n"),
            ("val",   "                        10-50 "),
            ("normal","= bo qua nhieu nho (khuyen nghi)\n\n"),

            ("h3",    "  Phan loai vi tri loi:\n"),
            ("normal","  Moi loi duoc kiem tra o dau trong ROI:\n\n"),
            ("key",   "  IN  "),
            ("normal","= Loi nam TRONG ROI (xa bien)\n"),
            ("key",   "  BR  "),
            ("normal","= Loi nam O BIEN ROI\n"),
            ("key",   "  MZ  "),
            ("normal","= Loi nam VUNG TRUNG GIAN\n\n"),

            ("h3",    "  Nguong dien tich bo qua:\n"),
            ("key",   "  noise_max_area        "),
            ("normal","Max dien tich loi IN de bo qua\n\n"),
            ("val",   "  100-500   "),
            ("normal","= chat che (it bo qua)\n"),
            ("val",   "  500-2500  "),
            ("normal","= binh thuong (khuyen nghi)\n"),
            ("val",   "  2500+     "),
            ("normal","= de tinh (bo qua nhieu)\n\n"),
            ("key",   "  border_noise_max_area "),
            ("normal","Max dien tich loi BR/MZ de bo qua\n"),
            ("normal","                        Thuong >= noise_max_area\n\n"),

            ("h3",    "  Kernel phan tich:\n"),
            ("key",   "  green_dilate_k "),
            ("normal","Mo rong vung XANH\n\n"),
            ("val",   "  3-5     "),
            ("normal","= kenh chat che voi loi\n"),
            ("val",   "  7-9     "),
            ("normal","= vung xanh rong hon\n"),
            ("note",  "           Lon = loi do de bi coi la 'trong xanh'\n"),
            ("note",  "                → de OK hon\n\n"),

            ("key",   "  ring_dilate_k  "),
            ("normal","Do day vong quanh loi khi kiem tra\n\n"),
            ("val",   "  1-2     "),
            ("normal","= vong mong, chat che\n"),
            ("val",   "  3-5     "),
            ("normal","= vong day hon\n"),
            ("note",  "           Lon = loi de duoc coi la 'trong xanh'\n"),
            ("note",  "                → de OK hon\n\n"),

            ("key",   "  sep_open_k     "),
            ("normal","Tach cac vung loi roi nhau\n"),
            ("val",   "  1-3     "),
            ("normal","= tach nhe\n"),
            ("val",   "  3-5     "),
            ("normal","= tach manh hon\n\n"),

            ("key",   "  max_defect_count "),
            ("normal","So loi toi da kiem tra\n"),
            ("val",   "  -1      "),
            ("normal","= kiem tra tat ca (mac dinh)\n"),
            ("val",   "  5-10    "),
            ("normal","= dung khi du N loi\n\n"),

            ("h2",    "BANG TOM TAT DIEU CHINH:\n\n"),
            ("normal","  Qua nhieu NG sai:\n"),
            ("tip",   "    Tang noise_max_area\n"
                      "    Tang green_dilate_k\n"
                      "    Giam ok_ratio\n\n"),
            ("normal","  Bo sot loi that:\n"),
            ("tip",   "    Giam noise_max_area\n"
                      "    Giam green_dilate_k\n"
                      "    Tang ok_ratio\n\n"),
            ("normal","  Vung do chuyen xanh nhieu:\n"),
            ("tip",   "    Giam green_dilate_k (tu 7 xuong 3)\n"
                      "    Giam ring_dilate_k  (tu 3 xuong 1)\n"
                      "    Giam noise_max_area\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_output(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "CAI DAT OUTPUT\n\n"),

            ("h2",    "Luu anh ket qua:\n\n"),
            ("key",   "  save_ok_images "),
            ("normal","Luu anh OK vao folder output/OK\n"),
            ("key",   "  save_ng_images "),
            ("normal","Luu anh NG vao folder output/NG\n\n"),
            ("tip",   "  Meo: Tat save_ok de tiet kiem disk\n"
                      "  Giu save_ng de xem lai loi\n\n"),

            ("key",   "  out_q    "),
            ("normal","Chat luong anh JPEG (50-100)\n"),
            ("val",   "            70-80  "),
            ("normal","= file nho, du xem\n"),
            ("val",   "            85-95  "),
            ("normal","= can bang (khuyen nghi)\n"),
            ("val",   "            100    "),
            ("normal","= file lon, ro nhat\n\n"),

            ("key",   "  max_w    "),
            ("normal","Chieu rong toi da anh ket qua (px)\n"),
            ("val",   "            800-1200  "),
            ("normal","= anh nho, xem co ban\n"),
            ("val",   "            1400-1600 "),
            ("normal","= anh vua (khuyen nghi)\n"),
            ("val",   "            2000+     "),
            ("normal","= anh lon, ro chi tiet\n\n"),

            ("h2",    "He thong:\n\n"),
            ("key",   "  n_threads "),
            ("normal","So luong xu ly song song\n\n"),
            ("val",   "             1-2   "),
            ("normal","= an toan, it RAM\n"),
            ("val",   "             4     "),
            ("normal","= Jetson toi uu (khuyen nghi)\n"),
            ("val",   "             6-8   "),
            ("normal","= nhanh hon nhung ton RAM\n"),
            ("note",  "             Jetson: khong qua 4!\n\n"),

            ("key",   "  input_dir  "),
            ("normal","Thu muc chua anh dau vao\n"),
            ("key",   "  output_dir "),
            ("normal","Thu muc luu ket qua OK/NG\n\n"),

            ("h2",    "Chup anh:\n\n"),
            ("key",   "  Max frames   "),
            ("normal","So frame toi da moi lan chup burst\n"),
            ("val",   "                100-300 "),
            ("normal","= binh thuong\n"),
            ("val",   "                500+    "),
            ("normal","= chup nhieu (can RAM)\n\n"),
            ("key",   "  Interval(ms) "),
            ("normal","Khoang cach giua cac frame\n"),
            ("val",   "                0       "),
            ("normal","= chup nhanh nhat co the\n"),
            ("val",   "                33      "),
            ("normal","= ~30 FPS\n"),
            ("val",   "                100     "),
            ("normal","= 10 FPS\n"),
            ("val",   "                1000    "),
            ("normal","= 1 FPS (chup cham)\n"),
        ])

    # ════════════════════════════════════════════════
    def _tab_tools(self, parent):
        txt = self._make_text(parent)
        self._write(txt, [
            ("h1",    "TOOL XU LY\n\n"),

            ("h2",    "1. Defect Detection:\n"),
            ("normal","   Phat hien loi be mat bang gradient\n"),
            ("normal","   Dung cho: vet xuoc, lo khuyet, ban\n"),
            ("normal","   Thong so: grad_thr, ok_ratio,\n"
                      "            noise_max_area\n\n"),

            ("h2",    "2. Shape Search:\n"),
            ("normal","   Tim kiem hinh dang khop voi mau\n"),
            ("normal","   Dung cho: kiem tra hinh dang, vi tri\n"),
            ("normal","   Can chup mau truoc khi dung\n\n"),

            ("h2",    "3. Blob Analysis:\n"),
            ("normal","   Dem va phan tich vung blob\n"),
            ("normal","   Dung cho: dem linh kien, lien ket\n"),
            ("normal","   Thong so: min_area, max_area,\n"
                      "            min_count, max_count\n\n"),

            ("h2",    "4. Edge Detection:\n"),
            ("normal","   Phat hien bien canh\n"),
            ("normal","   Dung cho: kiem tra khe ho, gap mep\n"),
            ("normal","   Thong so: canny_thr1, canny_thr2\n\n"),

            ("h2",    "5. Binary Check:\n"),
            ("normal","   Kiem tra ty le sang/toi\n"),
            ("normal","   Dung cho: kiem tra do phu, muc do\n"),
            ("normal","   Thong so: thresh, ok_ratio\n\n"),

            ("h2",    "Cach su dung Zone Tool:\n\n"),
            ("normal","  1. Mo ROI Setup [R]\n"),
            ("normal","  2. Ve vung can kiem tra\n"),
            ("normal","  3. Nhan 'Gan Tool'\n"),
            ("normal","  4. Chon tool phu hop\n"),
            ("normal","  5. Chinh thong so\n"),
            ("normal","  6. Nhan 'Xem truoc' de test\n"),
            ("normal","  7. Luu & Dong\n\n"),

            ("tip",   "  Meo: Co the dat nhieu vung voi\n"
                      "  nhieu tool khac nhau tren 1 anh\n"
                      "  Tat ca phai OK moi la OK tong the\n"),
        ])
# ════════════════════════════════════════════════════════════════
# MODEL DIALOG
# ════════════════════════════════════════════════════════════════
class ModelDialog(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("[mdl] Quan Ly Model")
        self.configure(bg="#1e1e2e")
        self.geometry("420x380")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        BG, FG, ACC = "#1e1e2e", "#cdd6f4", "#89b4fa"

        tk.Label(self, text="QUAN LY MODEL",
                 bg=BG, fg=ACC,
                 font=("Consolas", 13, "bold")
                 ).pack(pady=(14, 6))
        ttk.Separator(self, orient="horizontal"
                      ).pack(fill=tk.X, padx=16)

        fr1 = tk.Frame(self, bg=BG)
        fr1.pack(fill=tk.X, padx=16, pady=(12, 4))
        tk.Label(fr1, text="Ten Model:", bg=BG, fg=FG,
                 font=("Consolas", 10, "bold"),
                 width=12, anchor="w"
                 ).pack(side=tk.LEFT)
        self.var_model = tk.StringVar(
            value=app.cfg.get("current_model", ""))
        tk.Entry(fr1, textvariable=self.var_model,
                 bg="#313244", fg=FG,
                 insertbackground=FG,
                 font=("Consolas", 11), relief="flat"
                 ).pack(side=tk.LEFT, fill=tk.X,
                        expand=True, padx=(6, 0))

        fr2 = tk.Frame(self, bg=BG)
        fr2.pack(fill=tk.X, padx=16, pady=4)
        tk.Button(fr2, text="[sav]  Luu Model",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 10, "bold"),
                  command=self._save_model,
                  relief="flat", padx=10, pady=6
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(0, 4))
        tk.Button(fr2, text="[dir]  Load Model",
                  bg="#89b4fa", fg="black",
                  font=("Consolas", 10, "bold"),
                  command=self._load_model,
                  relief="flat", padx=10, pady=6
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(4, 0))

        ttk.Separator(self, orient="horizontal"
                      ).pack(fill=tk.X, padx=16, pady=8)
        tk.Label(self, text="Danh sach Model:",
                 bg=BG, fg=ACC,
                 font=("Consolas", 10, "bold"), anchor="w"
                 ).pack(fill=tk.X, padx=16)

        list_fr = tk.Frame(self, bg=BG)
        list_fr.pack(fill=tk.BOTH, expand=True,
                     padx=16, pady=(4, 8))
        sb = ttk.Scrollbar(list_fr, orient="vertical")
        self.listbox = tk.Listbox(
            list_fr, bg="#181825", fg=FG,
            font=("Consolas", 10),
            selectbackground="#89b4fa",
            selectforeground="black",
            relief="flat",
            yscrollcommand=sb.set,
            activestyle="dotbox")
        sb.config(command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT,
                          fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>",
                          self._on_select)
        self.listbox.bind("<Double-Button-1>",
                          lambda e: self._load_model())
        self._refresh_list()

        fr3 = tk.Frame(self, bg=BG)
        fr3.pack(fill=tk.X, padx=16, pady=(0, 12))
        tk.Button(fr3, text="[del] Xoa",
                  bg="#f38ba8", fg="black",
                  font=("Consolas", 10, "bold"),
                  command=self._delete_model,
                  relief="flat", padx=10, pady=5
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(0, 4))
        tk.Button(fr3, text="[x]  Dong",
                  bg="#313244", fg=FG,
                  font=("Consolas", 10, "bold"),
                  command=self.destroy,
                  relief="flat", padx=10, pady=5
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=(4, 0))

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for m in list_models():
            self.listbox.insert(tk.END, f"  {m}")

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            self.var_model.set(
                self.listbox.get(sel[0]).strip())

    def _save_model(self):
        name = self.var_model.get().strip()
        if not name:
            messagebox.showwarning(
                "Chu y", "Nhap ten!", parent=self)
            return
        if any(c in set('/\\:*?"<>|') for c in name):
            messagebox.showerror(
                "Loi", "Ten khong hop le!", parent=self)
            return
        self.app._save_params()
        save_model(name, self.app.cfg)
        self.app.cfg["current_model"] = name
        save_cfg(self.app.cfg)
        self.app.var_model_display.set(f"Model: {name}")
        self._refresh_list()
        messagebox.showinfo(
            "OK", f"Da luu: {name}", parent=self)

    def _load_model(self):
        name = self.var_model.get().strip()
        if not name:
            messagebox.showwarning(
                "Chu y", "Nhap ten!", parent=self)
            return
        try:
            cfg = load_model(name)
            self.app.cfg.update(cfg)
            self.app.cfg["current_model"] = name
            self.app._apply_cfg_to_inspector()
            save_cfg(self.app.cfg)
            self.app.var_model_display.set(
                f"Model: {name}")
            n    = len(self.app.cfg.get(
                "roi_shapes", []))
            crop = "Y" if self.app.cfg.get(
                "roi_crop") else "N"
            b    = self.app.cfg.get("roi_brightness", 0)
            c    = self.app.cfg.get("roi_contrast", 1.0)
            self.app.var_roi_info.set(
                f"ROI:{n} Crop:{crop} "
                f"B:{b} C:{c:.1f}")
            messagebox.showinfo(
                "OK", f"Da load: {name}", parent=self)
        except FileNotFoundError as e:
            messagebox.showerror(
                "Loi", str(e), parent=self)

    def _delete_model(self):
        name = self.var_model.get().strip()
        if not name:
            messagebox.showwarning(
                "Chu y", "Chon model!", parent=self)
            return
        if messagebox.askyesno(
                "Xac nhan", f"Xoa '{name}'?",
                parent=self):
            path = MODELS_DIR / f"{name}.json"
            if path.exists():
                path.unlink()
                self._refresh_list()
                messagebox.showinfo(
                    "OK", f"Da xoa: {name}", parent=self)
            else:
                messagebox.showerror(
                    "Loi", "Khong ton tai!", parent=self)


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════
class App(tk.Tk):
    BG                  = "#1e1e2e"
    FG                  = "#cdd6f4"
    ACC                 = "#89b4fa"
    ENTRY_BG            = "#313244"
    PREVIEW_INTERVAL_MS = 40

    def __init__(self):
        super().__init__()
        self.title("Jetson Inspector - Camera + Analyze")
        self.resizable(True, True)
        self.configure(bg=self.BG)

        self.cfg         = load_cfg()
        self.cam_thread  = None
        self._all_imgs   = []
        self._ok_imgs    = []
        self._ng_imgs    = []
        self._cur_list   = []
        self._cur_idx    = 0
        self._filter     = "ALL"
        self._run_count  = 0
        self._session_dt = datetime.now()
        self._gui_lock   = threading.Lock()

        # ← THEM 2 DONG NAY
        self._last_ok    = 0
        self._last_ng    = 0

        self.var_model_display = tk.StringVar(
            value=f"Model: "
                  f"{self.cfg.get('current_model', '—')}")
        n    = len(self.cfg.get("roi_shapes", []))
        crop = "Y" if self.cfg.get("roi_crop") else "N"
        b    = self.cfg.get("roi_brightness", 0)
        c    = self.cfg.get("roi_contrast", 1.0)
        self.var_roi_info = tk.StringVar(
            value=f"ROI:{n} Crop:{crop} "
                  f"B:{b} C:{c:.1f}")

        self.tool_manager = ToolManager(self)
        self._build_ui()
        self._apply_cfg_to_inspector()

        self.bind("<KeyPress>", self._on_key)
        self.focus_set()

        self._preview_running = True
        self._update_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.plc_client  = None
        self.plc_monitor = None
        self._plc_init()  # Khoi dong PLC neu enabled

    def _plc_init(self):
        """Khoi dong PLC neu duoc bat"""
        if not self.cfg.get("plc_enable", False):
            print("[PLC] PLC disabled trong config")
            return
        try:
            from modbus_plc import (PLCClient, PLCMonitor,
                                    STATUS_IDLE,
                                    STATUS_CAPTURING,
                                    STATUS_PROCESSING,
                                    STATUS_DONE,
                                    STATUS_ERROR)

            host    = self.cfg.get("plc_host", "192.168.4.20")
            port    = int(self.cfg.get("plc_port", 502))
            unit_id = int(self.cfg.get("plc_unit_id", 1))

            self.plc_client = PLCClient(
                host    = host,
                port    = port,
                unit_id = unit_id)

            if self.plc_client.connect():
                self._set_status(
                    f"[PLC] Ket noi OK {host}:{port}")
                # Set trang thai ban dau
                self.plc_client.set_status(STATUS_IDLE)

                # Khoi dong monitor
                self.plc_monitor = PLCMonitor(
                    self.plc_client,
                    on_camera_on   = self._plc_on_camera_on,
                    on_camera_off  = self._plc_on_camera_off,
                    on_trigger_on  = self._plc_on_trigger_on,
                    on_trigger_off = self._plc_on_trigger_off)
                self.plc_monitor.start()
                print(f"[PLC] Monitor started")
            else:
                self._set_status(
                    f"[PLC] Ket noi FAIL {host}:{port}")
                self.plc_client = None

        except ImportError:
            print("[PLC] Khong tim thay modbus_plc.py!")
        except Exception as e:
            print(f"[PLC] _plc_init loi: {e}")
            self.plc_client = None    
    def _build_ui(self):
        BG, FG, ACC = self.BG, self.FG, self.ACC
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel",
                        background=BG, foreground=FG)
        style.configure("TFrame",      background=BG)
        style.configure("TLabelframe",
                        background=BG, foreground=ACC)
        style.configure(
            "TLabelframe.Label",
            background=BG, foreground=ACC,
            font=("Consolas", 10, "bold"))

        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True,
                  padx=6, pady=6)

        left  = ttk.Frame(root)
        left.pack(side=tk.LEFT,
                  fill=tk.BOTH, expand=True)
        right = tk.Frame(root, bg=BG, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y,
                   padx=(6, 0))
        right.pack_propagate(False)

        # Camera preview
        cf = ttk.LabelFrame(left,
                            text=" [cam] Camera Preview ")
        cf.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.lbl_preview = tk.Label(cf, bg="black")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True,
                              padx=2, pady=2)

        # Navigation bar
        bar = tk.Frame(left, bg="#252535",
                       relief="groove", bd=1)
        bar.pack(fill=tk.X, pady=2)

        tk.Label(bar, text=" Ket qua ",
                 bg="#252535", fg=ACC,
                 font=("Consolas", 9, "bold")
                 ).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Separator(bar, orient="vertical").pack(
            side=tk.LEFT, fill=tk.Y, padx=2, pady=3)

        for txt, cmd in [("◀", self._prev_img),
                         ("▶", self._next_img)]:
            tk.Button(bar, text=txt,
                      font=("Consolas", 10, "bold"),
                      bg=self.ENTRY_BG, fg=FG,
                      relief="flat", width=2,
                      command=cmd
                      ).pack(side=tk.LEFT,
                             padx=1, pady=2)

        for txt, bg, f in [
            ("[ok]OK",  "#a6e3a1", "OK"),
            ("[ng]NG",  "#f38ba8", "NG"),
            ("[dir]ALL", self.ENTRY_BG, "ALL"),
        ]:
            tk.Button(bar, text=txt,
                      font=("Consolas", 9, "bold"),
                      bg=bg,
                      fg="black"
                      if bg != self.ENTRY_BG else FG,
                      relief="flat",
                      command=lambda x=f:
                      self._set_filter(x)
                      ).pack(side=tk.LEFT,
                             padx=1, pady=2)

        self.var_nav = tk.StringVar(
            value="— chua co anh —")
        tk.Label(bar, textvariable=self.var_nav,
                 bg="#252535", fg=ACC,
                 font=("Consolas", 9)
                 ).pack(side=tk.LEFT, fill=tk.X,
                        expand=True, padx=4)

        self.var_ok = tk.StringVar(value="OK: —")
        self.lbl_ok = tk.Label(
            bar, textvariable=self.var_ok,
            bg="#252535", fg="#a6e3a1",
            font=("Consolas", 10, "bold"), padx=6)
        self.lbl_ok.pack(side=tk.LEFT, pady=2)

        self.var_ng = tk.StringVar(value="NG: —")
        self.lbl_ng = tk.Label(
            bar, textvariable=self.var_ng,
            bg="#252535", fg="#f38ba8",
            font=("Consolas", 10, "bold"), padx=6)
        self.lbl_ng.pack(side=tk.LEFT, pady=2)

        self.var_tgian = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.var_tgian,
                 bg="#252535", fg="#89b4fa",
                 font=("Consolas", 9), padx=4
                 ).pack(side=tk.LEFT, pady=2)

        self.verdict_box = tk.Label(
            bar, text=" — ",
            font=("Consolas", 14, "bold"),
            bg="#252535", fg=FG,
            width=6, padx=8, pady=4,
            relief="flat", anchor="center")
        self.verdict_box.pack(
            side=tk.RIGHT, padx=4, pady=2)

        # Result display
        rf = tk.Frame(left, bg="black",
                      relief="groove", bd=1)
        rf.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.lbl_result = tk.Label(rf, bg="black")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.var_status = tk.StringVar(
            value="San sang | C=Chup T=Xu ly "
                  "R=ROI I=Config Q=Thoat")
        tk.Label(left, textvariable=self.var_status,
                 bg=BG, fg=ACC,
                 font=("Consolas", 9), anchor="w"
                 ).pack(fill=tk.X)

        self._build_right_panel(right)
    #
    def _set_max_frames(self):
        """Cap nhat so frame toi da"""
        try:
            val = int(self.var_max_frames.get())
            if val < 1:
                val = 1
            if val > 9999:
                val = 9999
            self.cfg["camera_max_frames"] = val
            # Cap nhat CameraThread neu dang chay
            if self.cam_thread:
                self.cam_thread.MAX_BUFFER = val
            save_cfg(self.cfg)
            self._set_status(
                f"[ok] Max frames = {val}")
        except ValueError:
            self._set_status(
                "[wrn] Max frames khong hop le!")
    #
    def _set_capture_interval(self):
        """Cap nhat interval giua cac frame chup"""
        try:
            val = int(self.var_interval.get())
            if val < 0:
                val = 0
            self.cfg["capture_interval_ms"] = val
            save_cfg(self.cfg)

            # Cap nhat CameraThread neu dang chay
            if self.cam_thread:
                self.cam_thread._capture_interval = \
                    val / 1000.0

            fps_eff = 1000/val if val > 0 else 999
            self._set_status(
                f"[ok] Interval={val}ms "
                f"({fps_eff:.0f} FPS effective)")
        except ValueError:
            self._set_status(
                "[wrn] Interval khong hop le!")

    def _build_right_panel(self, parent):
        BG, FG, ACC = self.BG, self.FG, self.ACC

        # Model row
        row1 = tk.Frame(parent, bg=BG)
        row1.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Button(row1, text="[mdl] Model",
                  bg="#cba6f7", fg="black",
                  font=("Consolas", 9, "bold"),
                  command=self._open_model_dialog,
                  relief="flat", padx=8, pady=4
                  ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(row1, textvariable=self.var_model_display,
                 bg="#313244", fg="#f9e2af",
                 font=("Consolas", 9, "bold"),
                 padx=8, pady=4, relief="flat", anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X,
                        expand=True)

        # Settings buttons
        tk.Label(parent, text="  cai dat",
                 bg=BG, fg=ACC,
                 font=("Consolas", 9, "bold"), anchor="w"
                 ).pack(fill=tk.X, padx=4, pady=(6, 2))

        row3 = tk.Frame(parent, bg=BG)
        row3.pack(fill=tk.X, padx=4, pady=2)
        for txt, bg, cmd in [
            ("[roi] ROI", "#fab387", self._open_roi_setup),
            ("[cfg] Cfg", "#f9e2af", self._open_config),
            ("? Help",   "#89dceb", self._open_help),
        ]:
            tk.Button(row3, text=txt, bg=bg, fg="black",
                      font=("Consolas", 8, "bold"),
                      command=cmd, relief="flat",
                      padx=4, pady=4
                      ).pack(side=tk.LEFT, fill=tk.X,
                             expand=True, padx=1)

        ttk.Separator(parent, orient="horizontal"
                      ).pack(fill=tk.X, padx=4, pady=4)
        tk.Label(parent, textvariable=self.var_roi_info,
                 bg="#252535", fg="#a6e3a1",
                 font=("Consolas", 8),
                 padx=6, pady=2, anchor="w"
                 ).pack(fill=tk.X, padx=4, pady=(0, 4))

        self.tool_manager.build_panel(parent)

        ttk.Separator(parent, orient="horizontal"
                      ).pack(fill=tk.X, padx=4, pady=4)

        # Control buttons
        ctrl_lf = ttk.LabelFrame(
            parent, text=" [ctl] Dieu khien ")
        ctrl_lf.pack(fill=tk.X, padx=4, pady=(0, 4))
        for txt, bg, cmd in [
            ("[cam][C] BAT DAU CHUP",
             "#89dceb", self._start_capture),
            ("[stp][T] DUNG + XU LY",
             "#f38ba8", self._stop_and_process),
            ("[run] XU LY THU CONG",
             "#a6e3a1", self._manual_process),
        ]:
            tk.Button(ctrl_lf, text=txt, bg=bg,
                      fg="black",
                      font=("Consolas", 10, "bold"),
                      command=cmd, relief="flat",
                      padx=6, pady=6
                      ).pack(fill=tk.X, padx=4, pady=3)

        # ── IMV Camera Realtime Control ───────────────────
        imv_lf = ttk.LabelFrame(
            parent, text=" IMV Camera ")
        imv_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        # Exposure row
        f_exp = tk.Frame(imv_lf, bg=BG)
        f_exp.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f_exp, text="Exp(us):",
                 bg=BG, fg=FG,
                 font=("Consolas", 8),
                 width=9, anchor="w"
                 ).pack(side=tk.LEFT)
        #self.var_imv_exp = tk.StringVar(value="10000")
        self.var_imv_exp  = tk.StringVar(
            value=str(self.cfg.get("camera_exposure", 10000))
            if self.cfg.get("camera_exposure", -1) >= 0
            else "10000")
        tk.Entry(f_exp, textvariable=self.var_imv_exp,
                 bg=self.ENTRY_BG, fg=FG,
                 font=("Consolas", 8),
                 width=7, relief="flat"
                 ).pack(side=tk.LEFT, padx=2)
        tk.Button(f_exp, text="Set",
                  bg="#89dceb", fg="black",
                  font=("Consolas", 8, "bold"),
                  relief="flat", padx=4,
                  command=self._set_imv_exposure
                  ).pack(side=tk.LEFT)

        # Gain row
        f_gain = tk.Frame(imv_lf, bg=BG)
        f_gain.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f_gain, text="Gainraw:",
                 bg=BG, fg=FG,
                 font=("Consolas", 8),
                 width=9, anchor="w"
                 ).pack(side=tk.LEFT)
        #self.var_imv_gain = tk.StringVar(value="1")
        self.var_imv_gain = tk.StringVar(
            value=str(self.cfg.get("camera_gain", 1))
            if self.cfg.get("camera_gain", -1) >= 0
            else "1")
        tk.Entry(f_gain, textvariable=self.var_imv_gain,
                 bg=self.ENTRY_BG, fg=FG,
                 font=("Consolas", 8),
                 width=7, relief="flat"
                 ).pack(side=tk.LEFT, padx=2)
        tk.Button(f_gain, text="Set",
                  bg="#89dceb", fg="black",
                  font=("Consolas", 8, "bold"),
                  relief="flat", padx=4,
                  command=self._set_imv_gain
                  ).pack(side=tk.LEFT)

        # Auto buttons row 1
        f_a1 = tk.Frame(imv_lf, bg=BG)
        f_a1.pack(fill=tk.X, padx=4, pady=1)
        tk.Button(f_a1, text="AutoExp ON",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 7, "bold"),
                  relief="flat", padx=2, pady=2,
                  command=lambda: self._set_imv_auto(
                      "AutoExposure", True)
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)
        tk.Button(f_a1, text="AutoExp OFF",
                  bg="#f38ba8", fg="black",
                  font=("Consolas", 7, "bold"),
                  relief="flat", padx=2, pady=2,
                  command=lambda: self._set_imv_auto(
                      "AutoExposure", False)
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)

        # Auto buttons row 2
        f_a2 = tk.Frame(imv_lf, bg=BG)
        f_a2.pack(fill=tk.X, padx=4, pady=1)
        tk.Button(f_a2, text="AutoGain ON",
                  bg="#a6e3a1", fg="black",
                  font=("Consolas", 7, "bold"),
                  relief="flat", padx=2, pady=2,
                  command=lambda: self._set_imv_auto(
                      "AutoGain", True)
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)
        tk.Button(f_a2, text="AutoGain OFF",
                  bg="#f38ba8", fg="black",
                  font=("Consolas", 7, "bold"),
                  relief="flat", padx=2, pady=2,
                  command=lambda: self._set_imv_auto(
                      "AutoGain", False)
                  ).pack(side=tk.LEFT, fill=tk.X,
                         expand=True, padx=1)

        # Read params button
        tk.Button(imv_lf, text="Read Params",
                  bg="#cba6f7", fg="black",
                  font=("Consolas", 8, "bold"),
                  relief="flat", padx=4, pady=2,
                  command=self._read_imv_params_realtime
                  ).pack(fill=tk.X, padx=4, pady=(0, 2))

        # IMV status label
        self.lbl_imv_status = tk.Label(
            imv_lf, text="",
            bg=BG, fg="#a6e3a1",
            font=("Consolas", 7),
            anchor="w", wraplength=260)
        self.lbl_imv_status.pack(
            fill=tk.X, padx=4, pady=(0, 4))

        # Log frame
        log_frame = tk.Frame(parent, bg="#0d0d1a",
                             relief="groove", bd=1)
        log_frame.pack(fill=tk.BOTH, expand=True,
                       padx=4, pady=4)
        self.lbl_log = tk.Label(
            log_frame, text="",
            bg="#0d0d1a", fg=ACC,
            font=("Consolas", 8),
            anchor="nw", justify="left",
            wraplength=260)
        self.lbl_log.pack(fill=tk.BOTH, expand=True,
                          padx=4, pady=4)
        #
        # Thêm vào ctrl_lf, SAU các button điều khiển
        # Max frames row
        f_mf = tk.Frame(ctrl_lf, bg=BG)
        f_mf.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f_mf, text="Max frames:",
                bg=BG, fg=FG,
                font=("Consolas", 8),
                width=11, anchor="w"
                ).pack(side=tk.LEFT)
        self.var_max_frames = tk.IntVar(
            value=int(self.cfg.get(
                "camera_max_frames", 300)))
        tk.Entry(f_mf,
                textvariable=self.var_max_frames,
                bg=self.ENTRY_BG, fg=FG,
                font=("Consolas", 8),
                width=5, relief="flat"
                ).pack(side=tk.LEFT, padx=2)
        tk.Button(f_mf, text="Set",
                bg="#89dceb", fg="black",
                font=("Consolas", 8, "bold"),
                relief="flat", padx=4,
                command=self._set_max_frames
                ).pack(side=tk.LEFT)
        tk.Label(f_mf, text="frames",
                bg=BG, fg="#585b70",
                font=("Consolas", 7)
                ).pack(side=tk.LEFT, padx=2)
        # Them vao ctrl_lf sau max_frames row
        f_interval = tk.Frame(ctrl_lf, bg=BG)
        f_interval.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f_interval, text="Interval(ms):",
                bg=BG, fg=FG,
                font=("Consolas", 8),
                width=11, anchor="w"
                ).pack(side=tk.LEFT)

        self.var_interval = tk.IntVar(
            value=int(self.cfg.get(
                "capture_interval_ms", 0)))
        tk.Entry(f_interval,
                textvariable=self.var_interval,
                bg=self.ENTRY_BG, fg=FG,
                font=("Consolas", 8),
                width=5, relief="flat"
                ).pack(side=tk.LEFT, padx=2)
        tk.Button(f_interval, text="Set",
                bg="#89dceb", fg="black",
                font=("Consolas", 8, "bold"),
                relief="flat", padx=4,
                command=self._set_capture_interval
                ).pack(side=tk.LEFT)
        tk.Label(f_interval, text="ms (0=all)",
                bg=BG, fg="#585b70",
                font=("Consolas", 7)
                ).pack(side=tk.LEFT, padx=2)
    # PLC
    def _plc_on_camera_on(self):
        """M1000=1 → Bat camera"""
        print("[PLC] Camera ON")
        self.after(0, self._ensure_camera)
        self.after(0, lambda: self._set_status(
            "[PLC] M1000=1 → Camera ON"))

    def _plc_on_camera_off(self):
        """M1000=0 → Tat camera"""
        print("[PLC] Camera OFF")
        def _do():
            if self.cam_thread:
                self.cam_thread.stop()
                self.cam_thread = None
            self._set_status("[PLC] M1000=0 → Camera OFF")
        self.after(0, _do)

    def _plc_on_trigger_on(self):
        """M1001=1 → Bat dau chup"""
        print("[PLC] Trigger ON → Start capture")
        try:
            from modbus_plc import STATUS_CAPTURING
            if self.plc_client:
                self.plc_client.set_status(STATUS_CAPTURING)
        except Exception:
            pass
        self.after(0, self._start_capture)
        self.after(0, lambda: self._set_status(
            "[PLC] M1001=1 → Dang chup..."))

    def _plc_on_trigger_off(self):
        """M1001=0 → Dung chup + xu ly"""
        print("[PLC] Trigger OFF → Stop + Process")
        try:
            from modbus_plc import STATUS_PROCESSING
            if self.plc_client:
                self.plc_client.set_status(STATUS_PROCESSING)
        except Exception:
            pass
        self.after(0, self._plc_stop_and_process)
        self.after(0, lambda: self._set_status(
            "[PLC] M1001=0 → Dang xu ly..."))

    def _plc_stop_and_process(self):
        """Stop capture + process + gui ket qua ve PLC"""
        import time as _time

        if (self.cam_thread is None or
                not self.cam_thread.capturing):
            return

        frames      = self.cam_thread.stop_capture()
        frame_count = len(frames)
        if not frames:
            return

        t0 = _time.perf_counter()

        def _after_process():
            try:
                from modbus_plc import STATUS_DONE
                if self.plc_client:
                    elapsed_ms = int(
                        (_time.perf_counter() - t0) * 1000)
                    self.plc_client.send_result(
                        frame_count = frame_count,
                        ok_count    = self._last_ok,
                        ng_count    = self._last_ng,
                        proc_ms     = elapsed_ms,
                        status      = STATUS_DONE)
                    print(f"[PLC] Gui ket qua: "
                        f"frames={frame_count} "
                        f"OK={self._last_ok} "
                        f"NG={self._last_ng} "
                        f"t={elapsed_ms}ms")
            except Exception as e:
                print(f"[PLC] _after_process loi: {e}")

        def _run():
            # ← DUNG DIRECT MODE thay vi _save_and_process
            self._process_frames_direct(frames)
            self.after(0, _after_process)

        threading.Thread(
            target=_run, daemon=True).start()


    # ════════════════════════════════════════════════════════
    # OPEN DIALOGS
    # ════════════════════════════════════════════════════════
    def _open_roi_setup(self):
        self._ensure_camera()
        ROISetupWindow(self)

    def _open_config(self):
        ConfigDialog(self)

    def _open_help(self):
        HelpDialog(self)

    def _open_model_dialog(self):
        ModelDialog(self, self)

    # ════════════════════════════════════════════════════════
    # IMV REALTIME CONTROL
    # ════════════════════════════════════════════════════════
    def _get_imv_cap(self):
        """Lay IMVCamera object neu dang dung IMV"""
        if (self.cam_thread and
                self.cam_thread._cap is not None and
                hasattr(self.cam_thread._cap,
                        'set_parameter')):
            return self.cam_thread._cap
        return None

    def _set_imv_exposure(self):
        cap = self._get_imv_cap()
        if cap is None:
            self._set_imv_status("Khong phai IMV!")
            return
        try:
            val = float(self.var_imv_exp.get())
            ok  = cap.set_parameter("ExposureTime", val)
            if ok:
                actual = cap.get_current_value("ExposureTime")
                self._set_imv_status(f"Exp={actual}us OK")

                # ✅ Dong bo vao cfg de khong bi mat khi restart
                self.cfg["camera_exposure"] = int(val)
                save_cfg(self.cfg)
            else:
                self._set_imv_status("Set Exp FAIL!")
        except ValueError:
            self._set_imv_status("Gia tri khong hop le!")

    def _set_imv_gain(self):
        cap = self._get_imv_cap()
        if cap is None:
            self._set_imv_status("Khong phai IMV!")
            return
        try:
            val = float(self.var_imv_gain.get())
            ok  = cap.set_parameter("Gain", val)
            if ok:
                actual = cap.get_current_value("Gain")
                self._set_imv_status(f"Gain={actual}dB OK")

                # ✅ Dong bo vao cfg
                self.cfg["camera_gain"] = float(val)
                save_cfg(self.cfg)
            else:
                self._set_imv_status("Set Gain FAIL!")
        except ValueError:
            self._set_imv_status("Gia tri khong hop le!")

    def _set_imv_auto(self, param: str, auto: bool):
        cap = self._get_imv_cap()
        if cap is None:
            self._set_imv_status("Khong phai IMV!")
            return
        ok = cap.set_parameter(param, auto)
        state = "ON" if auto else "OFF"
        self._set_imv_status(
            f"{param}={state} {'OK' if ok else 'FAIL'}")

        # ✅ Dong bo vao cfg
        if ok:
            if param == "AutoExposure":
                # auto = True → cfg exposure = -1
                if auto:
                    self.cfg["camera_exposure"] = -1
                    save_cfg(self.cfg)
            elif param == "AutoGain":
                if auto:
                    self.cfg["camera_gain"] = -1
                    save_cfg(self.cfg)
    def _read_imv_params_realtime(self):
        cap = self._get_imv_cap()
        if cap is None:
            self._set_imv_status("Khong phai IMV!")
            return

        def _do():
            try:
                exp  = cap.get_current_value("ExposureTime")
                gain = cap.get_current_value("Gain")
                fps  = cap.get_current_value(
                    "AcquisitionFrameRate")
                ae   = cap.get_current_value("ExposureAuto")
                ag   = cap.get_current_value("GainAuto")
                msg  = (f"Exp:{exp}us[{ae}]\n"
                        f"Gain:{gain}dB[{ag}]\n"
                        f"FPS:{fps}")

                # ✅ Cap nhat entry + cfg
                try:
                    self.var_imv_exp.set(str(exp))
                    self.var_imv_gain.set(str(gain))
                    # Dong bo vao cfg
                    self.cfg["camera_exposure"] = int(
                        float(str(exp))) \
                        if ae == "OFF" else -1
                    self.cfg["camera_gain"] = float(
                        str(gain)) \
                        if ag == "OFF" else -1
                except Exception:
                    pass
                self._set_imv_status(msg)
            except Exception as e:
                self._set_imv_status(f"Loi: {e}")

        threading.Thread(
            target=_do, daemon=True).start()


    def _set_imv_status(self, msg: str):
        """Update IMV status label thread-safe"""
        try:
            self.after(
                0, lambda m=msg:
                self.lbl_imv_status.configure(text=m))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════
    # VERDICT / RESULTS
    # ════════════════════════════════════════════════════════
    def _update_verdict(self, ok_n, ng_n,
                        total_s, tb_ms):
        # ← THEM 2 DONG NAY DAU TIEN
        self._last_ok = ok_n
        self._last_ng = ng_n

        self.var_ok.set(f"OK: {ok_n}")
        self.var_ng.set(f"NG: {ng_n}")
        self.lbl_ng.configure(
            bg="#4a1a1a" if ng_n > 0 else "#252535",
            fg="#ff6b6b" if ng_n > 0 else "#f38ba8")
        self.lbl_ok.configure(
            bg="#1a3a1a"
            if (ok_n > 0 and ng_n == 0)
            else "#252535")
        self.var_tgian.set(
            f"{total_s:.1f}s/{tb_ms:.0f}ms"
            if total_s > 0 else "")
        if ng_n == 0 and ok_n > 0:
            self.verdict_box.configure(
                text=" OK ", bg="#a6e3a1", fg="#1e1e2e")
        elif ng_n > 0:
            self.verdict_box.configure(
                text=" NG ", bg="#f38ba8", fg="#1e1e2e")
        else:
            self.verdict_box.configure(
                text="  —  ", bg="#252535", fg=self.FG)
        if total_s > 0:
            self.lbl_log.configure(
                text=f"OK: {ok_n}  NG: {ng_n}\n"
                     f"Tong: {total_s:.2f}s\n"
                     f"TB: {tb_ms:.0f}ms/anh")
        else:
            self.lbl_log.configure(
                text=f"OK: {ok_n}  NG: {ng_n}")

    def _reload_result_list(self):
        try:
            out    = Path(self.cfg["output_dir"])
            ok_dir = out / "OK"
            ng_dir = out / "NG"
            ok_dir.mkdir(parents=True, exist_ok=True)
            ng_dir.mkdir(parents=True, exist_ok=True)
            self._ok_imgs  = sorted(
                ok_dir.glob("*.jpg"),
                key=lambda p: p.stat().st_mtime)
            self._ng_imgs  = sorted(
                ng_dir.glob("*.jpg"),
                key=lambda p: p.stat().st_mtime)
            self._all_imgs = sorted(
                list(self._ok_imgs) +
                list(self._ng_imgs),
                key=lambda p: p.stat().st_mtime)
            self._apply_filter()
        except Exception as e:
            print(f"[WARN] _reload_result_list: {e}")

    def _apply_filter(self):
        if self._filter == "OK":
            self._cur_list = list(self._ok_imgs)
        elif self._filter == "NG":
            self._cur_list = list(self._ng_imgs)
        else:
            self._cur_list = list(self._all_imgs)
        self._cur_idx = max(0, len(self._cur_list) - 1)
        self._show_current()

    def _set_filter(self, f):
        self._filter = f
        self._apply_filter()

    def _show_current(self):
        if not self._cur_list:
            self.var_nav.set("— chua co anh —")
            return
        p = self._cur_list[self._cur_idx]
        try:
            img = cv2.imread(str(p))
            if img is not None:
                self._show_frame(
                    self.lbl_result, img,
                    max_w=1200, max_h=500)
            tag = "[ok]" \
                if p.parent.name == "OK" else "[ng]"
            self.var_nav.set(
                f"{tag}[{self._filter}] "
                f"{self._cur_idx+1}/"
                f"{len(self._cur_list)}  {p.name}")
        except Exception as e:
            print(f"[WARN] _show_current: {e}")

    def _prev_img(self):
        if not self._cur_list:
            return
        self._cur_idx = (
            (self._cur_idx - 1) % len(self._cur_list))
        self._show_current()

    def _next_img(self):
        if not self._cur_list:
            return
        self._cur_idx = (
            (self._cur_idx + 1) % len(self._cur_list))
        self._show_current()

    # ════════════════════════════════════════════════════════
    # PREVIEW
    # ════════════════════════════════════════════════════════
    def _update_preview(self):
        if not self._preview_running:
            return
        try:
            frame = None
            if (self.cam_thread and
                    self.cam_thread.last_frame
                    is not None):
                frame = self.cam_thread.last_frame

            if frame is None:
                frame = np.zeros(
                    (360, 480, 3), np.uint8)
                cv2.putText(frame, "No Camera",
                            (140, 190),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (80, 80, 80), 2)
            else:
                if (self.cam_thread and
                        self.cam_thread.capturing):
                    n     = len(self.cam_thread._buffer)
                    frame = frame.copy()
                    cv2.putText(
                        frame,
                        f"REC  {n} frames",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

                crop = self.cfg.get("roi_crop")
                if crop:
                    frame = frame.copy()
                    cv2.rectangle(
                        frame,
                        (crop[0], crop[1]),
                        (crop[2], crop[3]),
                        (255, 140, 0), 2)
                    cv2.putText(
                        frame, "CROP",
                        (crop[0]+4, crop[1]+18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 140, 0), 1)

            self._show_frame(self.lbl_preview, frame,
                             max_w=900, max_h=400)
        except Exception as e:
            print(f"[WARN] _update_preview: {e}")

        self.after(self.PREVIEW_INTERVAL_MS,
                   self._update_preview)

    def _show_frame(self, widget, frame,
                    max_w=640, max_h=360):
        if frame is None:
            return
        try:
            h, w = frame.shape[:2]
            sc   = min(max_w/w, max_h/h, 1.0)
            if sc < 0.99:
                frame = cv2.resize(
                    frame,
                    (int(w*sc), int(h*sc)),
                    interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(
                Image.fromarray(rgb))
            widget.configure(image=img)
            widget.image = img
        except Exception as e:
            print(f"[WARN] _show_frame: {e}")

    # ════════════════════════════════════════════════════════
    # KEY HANDLER
    # ════════════════════════════════════════════════════════
    def _on_key(self, event):
        k = event.keysym.upper()
        if k == "C":
            self._start_capture()
        elif k == "T":
            self._stop_and_process()
        elif k == "Q":
            self._on_close()
        elif k == "LEFT":
            self._prev_img()
        elif k == "RIGHT":
            self._next_img()
        elif k == "F1":
            self._open_help()
        elif k == "M":
            self._open_model_dialog()
        elif k == "R":
            self._open_roi_setup()
        elif k == "I":
            self._open_config()

    # ════════════════════════════════════════════════════════
    # CAMERA CONTROL
    # ════════════════════════════════════════════════════════
    def _ensure_camera(self):
        if (self.cam_thread is None or
                not self.cam_thread.is_alive()):
            self.cam_thread = CameraThread(self.cfg)
            self.cam_thread.start()
            cam_type = self.cfg.get("camera_type", "USB")
            cam_id   = self.cfg.get("camera_id", 0)
            self._set_status(
                f"[cam] Camera {cam_type} "
                f"[{cam_id}] khoi dong...")

    def _restart_camera_if_needed(self):
        if (self.cam_thread and
                self.cam_thread.is_alive()):
            self.cam_thread.stop()
            time.sleep(0.5)
        self._ensure_camera()


    def _start_capture(self):
        self._save_params()
        self._ensure_camera()
        self._session_dt = datetime.now()
        self.cam_thread.start_capture()
        self._run_count += 1
        self._set_status(
            "[rec] DANG CHUP... Nhan T de dung")
        self.after(0, lambda: self._update_verdict(
            0, 0, 0, 0))
    def _stop_and_process(self):
        if (self.cam_thread is None or
                not self.cam_thread.capturing):
            self._set_status("[wrn] Chua bat dau chup!")
            return
        frames = self.cam_thread.stop_capture()
        self._set_status(f"[stp] {len(frames)} anh")
        if not frames:
            messagebox.showwarning("Chu y", "Khong co anh!")
            return

        # FIX: Xu ly TRUC TIEP tu buffer
        # KHONG luu xuong disk roi doc lai
        threading.Thread(
            target=self._process_frames_direct,
            args=(frames,),
            daemon=True).start()
    def _process_frames_direct(self, frames):
        """
        Xu ly truc tiep tu buffer RAM
        Bo qua buoc luu/doc disk
        """
        try:
            brightness = int(self.cfg.get("roi_brightness", 0))
            contrast   = float(self.cfg.get("roi_contrast", 1.0))
            crop       = self.cfg.get("roi_crop", None)
            model      = self.cfg.get("current_model", "NOMODEL")
            date       = self._session_dt.strftime("%y%m%d")
            hhmm       = self._session_dt.strftime("%H%M")

            # Validate crop
            crop_valid = False
            cx0 = cy0 = cx1 = cy1 = 0
            if crop and len(crop) == 4:
                cx0,cy0,cx1,cy1 = [int(x) for x in crop]
                if cx1 > cx0+10 and cy1 > cy0+10:
                    crop_valid = True

            self._set_status(
                f"Xu ly {len(frames)} anh truc tiep...")

            # Xu ly tung frame trong RAM
            processed = []
            for i, frame in enumerate(frames):
                try:
                    # Brightness/contrast
                    if brightness != 0 or contrast != 1.0:
                        frame = cv2.convertScaleAbs(
                            frame,
                            alpha=contrast,
                            beta=brightness)

                    # Crop
                    if crop_valid:
                        fh, fw = frame.shape[:2]
                        x0 = max(0, min(cx0, fw-1))
                        y0 = max(0, min(cy0, fh-1))
                        x1 = max(x0+1, min(cx1, fw))
                        y1 = max(y0+1, min(cy1, fh))
                        frame = frame[y0:y1, x0:x1].copy()

                    processed.append(frame)
                except Exception as e:
                    print(f"[WARN] prep frame {i}: {e}")

            frames.clear()
            gc.collect()

            if not processed:
                self._set_status("[wrn] Khong co anh!")
                return

            # Kiem tra co zone khong
            roi_shapes = self.cfg.get("roi_shapes", [])
            has_zone   = any(
                s.get("tool") and s.get("enabled", True)
                for s in roi_shapes)

            if has_zone:
                self._run_zone_direct(processed)
            else:
                self._run_inspector_direct(processed)

        except Exception as e:
            self._set_status(f"[ng] Loi: {e}")
            import traceback
            traceback.print_exc()


    def _run_inspector_direct(self, frames):
        """
        Chay inspector TRUC TIEP tu list frames
        Khong doc/ghi disk cho input
        """
        from concurrent.futures import ThreadPoolExecutor

        out_dir = Path(self.cfg["output_dir"])
        ok_dir  = out_dir / "OK"
        ng_dir  = out_dir / "NG"
        ok_dir.mkdir(parents=True, exist_ok=True)
        ng_dir.mkdir(parents=True, exist_ok=True)

        # Xoa anh cu
        for d in [ok_dir, ng_dir]:
            for f in d.glob("*.jpg"):
                try: f.unlink()
                except: pass

        save_ok = bool(self.cfg.get("save_ok_images", True))
        save_ng = bool(self.cfg.get("save_ng_images", True))
        model   = self.cfg.get("current_model", "NOMODEL")
        date    = self._session_dt.strftime("%y%m%d")
        hhmm    = self._session_dt.strftime("%H%M")

        counts   = {"OK": 0, "NG": 0}
        ms_list  = []
        _lock    = threading.Lock()
        t0_batch = time.perf_counter()
        
        def _process_one(args):
            idx, frame = args
            try:
                t0 = time.perf_counter()

                frame_orig = frame.copy()

                # FIX race condition: local objects
                cfg_snap      = dict(insp.CFG)
                local_kernel  = insp._make_kernel(
                    cfg_snap["morph_k"],
                    cfg_snap["morph_shape"])
                local_k_sep   = insp._make_kernel(
                    cfg_snap["sep_open_k"],
                    cfg_snap["morph_shape"])
                local_k1      = insp._make_kernel(
                    cfg_snap["ring_dilate_k"],
                    cfg_snap["morph_shape"])
                local_k_green = insp._make_kernel(
                    cfg_snap["green_dilate_k"],
                    cfg_snap["morph_shape"])
                _grid       = cfg_snap.get(
                    "clahe_grid", (14, 14))
                local_clahe = cv2.createCLAHE(
                    clipLimit    = cfg_snap["clahe_clip"],
                    tileGridSize = (
                        cfg_snap.get("clahe_grid_x", _grid[0]),
                        cfg_snap.get("clahe_grid_y", _grid[1]))
                ) if cfg_snap.get("clahe_enable", True) else None

                img_c, gray, grad, lm, cl = \
                    insp._pipeline_cpu_local(
                        frame.copy(), cfg_snap,
                        local_clahe, local_kernel)
                roi_mask, c_orig, c_dila = \
                    insp.get_roi_mask_local(img_c, cfg_snap)
                label, ratio, noise_info = \
                    insp.classify_local(
                        cl, roi_mask, cfg_snap,
                        local_k_sep, local_k1, local_k_green)
                # ── DEBUG: Luu anh khi gap NG 0% ──
                if label == "NG" and ratio == 0.0:
                    import pathlib
                    dbg = pathlib.Path("debug_ng0")
                    dbg.mkdir(exist_ok=True)
                    t_str = time.strftime("%H%M%S")
                    cv2.imwrite(
                        str(dbg/f"frame_{idx}_{t_str}.jpg"),
                        frame_orig)
                    cv2.imwrite(
                        str(dbg/f"img_c_{idx}_{t_str}.jpg"),
                        img_c)
                    cv2.imwrite(
                        str(dbg/f"cl_{idx}_{t_str}.jpg"),
                        cl)
                    if roi_mask is not None:
                        cv2.imwrite(
                            str(dbg/f"roi_{idx}_{t_str}.jpg"),
                            roi_mask)
                    print(f"[DEBUG] NG 0% frame={idx} "
                          f"saved to debug_ng0/")
                # ─────────────────────────────────

                ms = (time.perf_counter()-t0)*1000

                with _lock:
                    counts[label] = \
                        counts.get(label, 0) + 1
                    count_idx = counts[label]

                # Luu ket qua neu can
                should_save = (
                    (label=="OK" and save_ok) or
                    (label=="NG" and save_ng))

                if should_save:
                    fname = (f"{model}_{date}_{hhmm}_"
                            f"{label}_{count_idx:03d}.jpg")
                    dest  = ok_dir \
                        if label=="OK" else ng_dir
                    insp.save_result_local(
                        dest/fname, label, ratio,
                        noise_info, frame_orig, gray, grad,
                        lm, cl, roi_mask, c_orig, c_dila,
                        cfg_snap, local_k_sep)

                return label, ratio, ms

            except Exception as e:
                print(f"[WARN] _process_one {idx}: {e}")
                return "ERROR", 0, 0

        n_workers = min(
            int(self.cfg.get("n_threads", 4)),
            len(frames), 8)

        with ThreadPoolExecutor(
                max_workers=n_workers) as exe:
            for label, ratio, ms in exe.map(
                    _process_one,
                    enumerate(frames)):
                if ms > 0:
                    ms_list.append(ms)
                self._set_status(
                    f"[proc] OK:{counts['OK']}"
                    f" NG:{counts['NG']}")

        total  = time.perf_counter() - t0_batch
        avg_ms = np.mean(ms_list) if ms_list else 0

        self._set_status(
            f"[ok] Xong! OK:{counts['OK']}"
            f" NG:{counts['NG']}"
            f" Tong:{total:.2f}s"
            f" TB:{avg_ms:.0f}ms/anh")

        self.after(
            0,
            lambda ok=counts['OK'],
                ng=counts['NG'],
                t=total, tb=avg_ms:
            self._update_verdict(ok, ng, t, tb))

        self._reload_result_list()
        self.after(0, self._show_current)    
    

    def _run_zone_direct(self, frames):
        """Zone direct - luu tam roi goi zone inspector"""
        inp = Path(self.cfg["input_dir"])
        inp.mkdir(parents=True, exist_ok=True)
        for f in inp.glob("*.jpg"):
            try: f.unlink()
            except: pass

        model = self.cfg.get("current_model", "NOMODEL")
        date  = self._session_dt.strftime("%y%m%d")
        hhmm  = self._session_dt.strftime("%H%M")

        for i, frame in enumerate(frames):
            fname = f"{model}_{date}_{hhmm}_{i+1:02d}.jpg"
            cv2.imwrite(
                str(inp/fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

        frames.clear()
        gc.collect()
        self._run_zone_inspector()

    def _save_and_process(self, frames):
        try:
            inp = Path(self.cfg["input_dir"])
            inp.mkdir(parents=True, exist_ok=True)

            for f in inp.glob("*.jpg"):
                try:
                    f.unlink()
                except Exception:
                    pass

            brightness = int(self.cfg.get(
                "roi_brightness", 0))
            contrast   = float(self.cfg.get(
                "roi_contrast", 1.0))
            crop       = self.cfg.get("roi_crop", None)
            model      = self.cfg.get(
                "current_model", "NOMODEL")
            date       = self._session_dt.strftime(
                "%y%m%d")
            hhmm       = self._session_dt.strftime(
                "%H%M")

            crop_valid = False
            cx0 = cy0 = cx1 = cy1 = 0
            if crop and len(crop) == 4:
                cx0 = int(crop[0]); cy0 = int(crop[1])
                cx1 = int(crop[2]); cy1 = int(crop[3])
                if cx1 > cx0 + 10 and cy1 > cy0 + 10:
                    crop_valid = True

            mode_str = (f"crop({cx0},{cy0},"
                        f"{cx1},{cy1})"
                        if crop_valid else "full_frame")
            self._set_status(
                f"[sav] Luu {len(frames)} "
                f"anh [{mode_str}]...")

            saved_count = 0
            for i, frame in enumerate(frames):
                try:
                    if brightness != 0 or contrast != 1.0:
                        frame = cv2.convertScaleAbs(
                            frame,
                            alpha=contrast,
                            beta=brightness)
                    if crop_valid:
                        fh2, fw2 = frame.shape[:2]
                        x0 = max(0, min(cx0, fw2-1))
                        y0 = max(0, min(cy0, fh2-1))
                        x1 = max(x0+1, min(cx1, fw2))
                        y1 = max(y0+1, min(cy1, fh2))
                        if x1 > x0 and y1 > y0:
                            frame = frame[
                                y0:y1, x0:x1].copy()
                    fname = (f"{model}_{date}_"
                             f"{hhmm}_{i+1:02d}.jpg")
                    ok = cv2.imwrite(
                        str(inp / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ok:
                        saved_count += 1
                except Exception as e:
                    print(f"[WARN] save frame {i}: {e}")

            frames.clear()
            gc.collect()

            self._set_status(
                f"[ok] Da luu {saved_count} anh"
                f" -> xu ly...")
            self._run_inspector()

        except Exception as e:
            self._set_status(f"[ng] Loi luu anh: {e}")
            import traceback
            traceback.print_exc()

    def _manual_process(self):
        self._save_params()
        self._session_dt = datetime.now()
        threading.Thread(
            target=self._run_inspector,
            daemon=True).start()

    def _run_inspector(self):
        self._set_status("Dang xu ly anh...")
        t0 = time.perf_counter()

        # Kiem tra co anh khong truoc
        inp_dir = Path(self.cfg["input_dir"])
        images  = list(inp_dir.glob("*.jpg"))
        print(f"[Inspector] Tim thay {len(images)} "
          f"anh trong {inp_dir}")

        if not images:
            self._set_status(
                "[wrn] Khong co anh trong "
                f"{inp_dir}!")
            return
        roi_shapes = self.cfg.get("roi_shapes", [])
        has_zone   = any(
            s.get("tool") and s.get("enabled", True)
            for s in roi_shapes)

        if has_zone:
            self._run_zone_inspector()
        else:
            try:
                counts, ms_list = _patched_batch(
                    input_dir  = self.cfg["input_dir"],
                    output_dir = self.cfg["output_dir"],
                    return_stats = True,
                    name_fn    = self._make_image_name)
                total = time.perf_counter() - t0
                ok_n  = counts.get("OK", 0)
                ng_n  = counts.get("NG", 0)
                tb    = (float(np.mean(ms_list))
                         if ms_list else 0.0)
                print(f"[Inspector] Xong: "
                    f"OK={ok_n} NG={ng_n} "
                    f"t={total:.2f}s")
                self._set_status(
                    f"[ok] Xong! OK:{ok_n} NG:{ng_n} "
                    f"Tong:{total:.2f}s "
                    f"TB:{tb:.0f}ms")
                self.after(
                    0,
                    lambda ok=ok_n, ng=ng_n,
                           t=total, tb2=tb:
                    self._update_verdict(ok, ng, t, tb2))
                self._reload_result_list()
                self.after(0, self._show_current)
            except Exception as e:
                self._set_status(f"[ng] Loi: {e}")
                print(f"[ERROR] _run_inspector: {e}")
                import traceback
                traceback.print_exc()

    def _run_zone_inspector(self):
        try:
            from zone_processor2 import ZoneProcessor
        except ImportError:
            try:
                from zone_processor import ZoneProcessor
            except ImportError:
                self._set_status(
                    "[ng] Khong tim thay "
                    "zone_processor.py!")
                return

        inp_dir = Path(self.cfg["input_dir"])
        out_dir = Path(self.cfg["output_dir"])
        images  = sorted(inp_dir.glob("*.jpg"))

        if not images:
            self._set_status("[wrn] Khong co anh input!")
            return

        roi_shapes   = self.cfg.get("roi_shapes", [])
        active_zones = [
            s for s in roi_shapes
            if s.get("tool") and s.get("enabled", True)]

        if not active_zones:
            self._set_status(
                "[wrn] Chua gan Tool cho vung nao!")
            return

        crop       = self.cfg.get("roi_crop", None)
        crop_ox    = 0
        crop_oy    = 0
        crop_valid = False

        if crop and len(crop) == 4:
            cx0 = int(crop[0]); cy0 = int(crop[1])
            cx1 = int(crop[2]); cy1 = int(crop[3])
            if cx1 > cx0 + 10 and cy1 > cy0 + 10:
                crop_ox    = cx0
                crop_oy    = cy0
                crop_valid = True

        for sub in ["OK", "NG"]:
            d = out_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.jpg"):
                try:
                    f.unlink()
                except Exception:
                    pass

        processor = ZoneProcessor()
        model     = self.cfg.get(
            "current_model", "NOMODEL")
        date      = self._session_dt.strftime("%y%m%d")
        hhmm      = self._session_dt.strftime("%H%M")
        ok_count  = 0
        ng_count  = 0
        t0        = time.perf_counter()
        roi_fw    = int(self.cfg.get("roi_frame_w", 0))
        roi_fh    = int(self.cfg.get("roi_frame_h", 0))

        for i, img_path in enumerate(images):
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                img_h, img_w = frame.shape[:2]

                if crop_valid:
                    crop_w  = cx1 - cx0
                    crop_h  = cy1 - cy0
                    sx_crop = img_w / max(crop_w, 1)
                    sy_crop = img_h / max(crop_h, 1)
                    _roi_fw = roi_fw if roi_fw > 0 \
                        else (cx1 if cx1 > 0 else img_w)
                    _roi_fh = roi_fh if roi_fh > 0 \
                        else (cy1 if cy1 > 0 else img_h)

                    def map_zone(zone: dict) -> dict:
                        import copy as _copy
                        z  = _copy.deepcopy(zone)
                        tp = z.get("type")
                        d  = z.get("data", [])
                        if not d:
                            return z

                        def mx_full(x):
                            x_in = (x - crop_ox) * sx_crop
                            return max(0, min(
                                img_w-1, int(x_in)))

                        def my_full(y):
                            y_in = (y - crop_oy) * sy_crop
                            return max(0, min(
                                img_h-1, int(y_in)))

                        def mr_full(r):
                            return max(5, int(
                                r * (sx_crop+sy_crop)/2))

                        if tp == "rect" and len(d) == 4:
                            x0n = mx_full(d[0])
                            y0n = my_full(d[1])
                            x1n = min(img_w, max(
                                x0n+5, mx_full(d[2])))
                            y1n = min(img_h, max(
                                y0n+5, my_full(d[3])))
                            z["data"] = [x0n,y0n,x1n,y1n]
                        elif tp == "circle" \
                                and len(d) == 3:
                            z["data"] = [mx_full(d[0]),
                                         my_full(d[1]),
                                         mr_full(d[2])]
                        elif tp == "rot_rect" \
                                and len(d) == 5:
                            z["data"] = [mx_full(d[0]),
                                         my_full(d[1]),
                                         mr_full(d[2]),
                                         mr_full(d[3]),
                                         d[4]]
                        elif tp == "polygon":
                            z["data"] = [
                                [mx_full(p[0]),
                                 my_full(p[1])]
                                for p in d]
                        return z

                else:
                    _roi_fw = roi_fw if roi_fw > 0 \
                        else img_w
                    _roi_fh = roi_fh if roi_fh > 0 \
                        else img_h
                    sx = img_w / max(_roi_fw, 1)
                    sy = img_h / max(_roi_fh, 1)

                    def map_zone(zone: dict) -> dict:
                        import copy as _copy
                        z  = _copy.deepcopy(zone)
                        tp = z.get("type")
                        d  = z.get("data", [])
                        if not d:
                            return z

                        def mx(x):
                            return max(0, min(
                                img_w-1, int(x*sx)))

                        def my(y):
                            return max(0, min(
                                img_h-1, int(y*sy)))

                        def mr(r):
                            return max(5, int(
                                r*(sx+sy)/2))

                        if tp == "rect" and len(d) == 4:
                            x0n = mx(d[0])
                            y0n = my(d[1])
                            x1n = min(img_w, max(
                                x0n+10, int(d[2]*sx)))
                            y1n = min(img_h, max(
                                y0n+10, int(d[3]*sy)))
                            z["data"] = [x0n,y0n,x1n,y1n]
                        elif tp == "circle" \
                                and len(d) == 3:
                            z["data"] = [mx(d[0]),
                                         my(d[1]),
                                         mr(d[2])]
                        elif tp == "rot_rect" \
                                and len(d) == 5:
                            z["data"] = [mx(d[0]),
                                         my(d[1]),
                                         mr(d[2]),
                                         mr(d[3]), d[4]]
                        elif tp == "polygon":
                            z["data"] = [
                                [mx(p[0]), my(p[1])]
                                for p in d]
                        return z

                mapped_zones = [
                    map_zone(z) for z in active_zones]

                zone_results = []
                for _, zone_mapped in zip(
                        active_zones, mapped_zones):
                    result = processor.process_zone(
                        frame, zone_mapped)
                    zone_results.append(result)

                all_ok  = all(
                    r.get("label") == "OK"
                    for r in zone_results)
                overall = "OK" if all_ok else "NG"
                if all_ok:
                    ok_count += 1
                else:
                    ng_count += 1

                ov_col    = (0,220,0) \
                    if all_ok else (0,60,220)
                frame_ann = frame.copy()
                for zm, zr in zip(
                        mapped_zones, zone_results):
                    frame_ann = \
                        processor.draw_zone_overlay(
                            frame_ann, zm, zr)

                txt2 = f"OVERALL: {overall}"
                (tw, th), _ = cv2.getTextSize(
                    txt2, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 2)
                cv2.rectangle(
                    frame_ann,
                    (4, 4), (tw+14, th+14),
                    (0, 0, 0), -1)
                cv2.putText(
                    frame_ann, txt2, (8, th+8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, ov_col, 2, cv2.LINE_AA)

                max_w_cfg = int(self.cfg.get("max_w",
                                             1600))
                fh_a, fw_a = frame_ann.shape[:2]
                sc_f = min(max_w_cfg/fw_a, 1.0)
                if sc_f < 1.0:
                    frame_ann = cv2.resize(
                        frame_ann,
                        (int(fw_a*sc_f),
                         int(fh_a*sc_f)),
                        interpolation=cv2.INTER_LINEAR)

                rows = []
                for zi, (zone_orig, zr, zm) in enumerate(
                        zip(active_zones,
                            zone_results,
                            mapped_zones)):
                    name_z  = zone_orig.get(
                        "name", f"#{zi+1}")
                    tool_z  = zone_orig.get("tool", "—")
                    label_z = zr.get("label", "?")
                    score_z = zr.get("score", 0)
                    det_z   = zr.get("detail", "")
                    col_z   = (0,220,0) \
                        if label_z == "OK" \
                        else (0,60,220)
                    zone_w  = frame_ann.shape[1]

                    hdr = np.zeros(
                        (28, zone_w, 3), np.uint8)
                    hdr[:] = (18,38,18) \
                        if label_z == "OK" \
                        else (38,18,18)
                    cv2.rectangle(
                        hdr, (0,0), (5,28), col_z, -1)
                    cv2.putText(
                        hdr,
                        f"  [{label_z}] {name_z}"
                        f"  | {tool_z}"
                        f"  | {score_z*100:.1f}%"
                        f"  | {det_z}",
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, col_z, 1, cv2.LINE_AA)
                    rows.append(hdr)

                    panels = zr.get("panels")
                    if panels:
                        PANEL_H = 200
                        COLS    = 3
                        titled  = []
                        for pi, p in enumerate(panels):
                            if p is None or p.size == 0:
                                continue
                            if len(p.shape) == 2:
                                p = cv2.cvtColor(
                                    p,
                                    cv2.COLOR_GRAY2BGR)
                            ph, pw = p.shape[:2]
                            sc_p = PANEL_H/max(ph,1)
                            p_d  = cv2.resize(
                                p,
                                (max(1,int(pw*sc_p)),
                                 PANEL_H),
                                interpolation=
                                cv2.INTER_LINEAR)
                            cv2.rectangle(
                                p_d, (0,0),
                                (p_d.shape[1],22),
                                (0,0,0), -1)
                            cv2.putText(
                                p_d, f"#{pi+1}",
                                (4, 16),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.46, col_z, 1,
                                cv2.LINE_AA)
                            titled.append(p_d)

                        if titled:
                            n_r   = math.ceil(
                                len(titled)/COLS)
                            blank = np.zeros_like(
                                titled[0])
                            while len(titled) < n_r*COLS:
                                titled.append(
                                    blank.copy())
                            g_rows = []
                            for rr in range(n_r):
                                chunk = titled[
                                    rr*COLS:(rr+1)*COLS]
                                g_rows.append(
                                    np.hstack(chunk))
                            mw  = max(
                                r.shape[1]
                                for r in g_rows)
                            pad = []
                            for r in g_rows:
                                dw = mw - r.shape[1]
                                if dw > 0:
                                    pb = np.zeros(
                                        (r.shape[0],
                                         dw, 3),
                                        np.uint8)
                                    r = np.hstack([r,pb])
                                pad.append(r)
                            grid_img = np.vstack(pad)
                            gh, gw   = grid_img.shape[:2]
                            if gw != zone_w:
                                sc_g = zone_w/max(gw,1)
                                grid_img = cv2.resize(
                                    grid_img,
                                    (zone_w,
                                     max(1,int(gh*sc_g))),
                                    interpolation=
                                    cv2.INTER_LINEAR)
                            rows.append(grid_img)
                            zr["panels"] = None

                if rows:
                    rows.insert(0, frame_ann)
                    final_w  = max(
                        r.shape[1] for r in rows)
                    padded_r = []
                    for r in rows:
                        dw = final_w - r.shape[1]
                        if dw > 0:
                            pb = np.zeros(
                                (r.shape[0], dw, 3),
                                np.uint8)
                            r = np.hstack([r, pb])
                        padded_r.append(r)
                    analysis_img = np.vstack(padded_r)
                else:
                    analysis_img = frame_ann

                if analysis_img.shape[1] > max_w_cfg:
                    sc_t = max_w_cfg / \
                        analysis_img.shape[1]
                    analysis_img = cv2.resize(
                        analysis_img, None,
                        fx=sc_t, fy=sc_t,
                        interpolation=cv2.INTER_LINEAR)
                                # ── Luu anh don gian: chi crop + chu OK/NG ──

                fname    = (f"{model}_{date}_{hhmm}"
                            f"_{overall}_{i+1:02d}.jpg")
                save_dir = out_dir / overall
                save_dir.mkdir(parents=True,
                               exist_ok=True)
                cv2.imwrite(
                    str(save_dir / fname),
                    analysis_img,
                    [cv2.IMWRITE_JPEG_QUALITY,
                     int(self.cfg.get("out_q", 85))])

                _img_copy = analysis_img.copy()
                self.after(
                    0,
                    lambda img=_img_copy:
                    self._show_frame(
                        self.lbl_result, img,
                        max_w=1200, max_h=500))

                # ← Xoa analysis_img neu con ton tai
                if 'analysis_img' in dir():
                    del analysis_img
                del frame_ann, frame, rows
                gc.collect()

                self._set_status(
                    f"[proc] {i+1}/{len(images)} "
                    f"OK:{ok_count} NG:{ng_count}")

            except Exception as e:
                print(
                    f"[ERROR] zone img {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        total  = time.perf_counter() - t0
        avg_ms = total * 1000 / max(len(images), 1)
        self._set_status(
            f"[ok] Xong! OK:{ok_count} NG:{ng_count} "
            f"Tong:{total:.2f}s "
            f"TB:{avg_ms:.0f}ms/anh")
        self.after(
            0,
            lambda ok=ok_count, ng=ng_count,
                   t=total, tb=avg_ms:
            self._update_verdict(ok, ng, t, tb))
        self._reload_result_list()
        self.after(0, self._show_current)

    def _make_image_name(self, index, label):
        model = self.cfg.get("current_model", "NOMODEL")
        date  = self._session_dt.strftime("%y%m%d")
        hhmm  = self._session_dt.strftime("%H%M")
        return (f"{model}_{date}_{hhmm}_"
                f"{label}_{index:02d}.jpg")

    def _save_params(self):
        save_cfg(self.cfg)
        self._apply_cfg_to_inspector()

    def _reset_params(self):
        self.cfg = DEFAULT_CFG.copy()
        self._apply_cfg_to_inspector()
        self._set_status("[rst] Reset mac dinh")

    def _apply_cfg_to_inspector(self):
        try:
            insp.CFG.update({
                "save_ok_images": bool(
                    self.cfg.get("save_ok_images", True)),
                "save_ng_images": bool(
                    self.cfg.get("save_ng_images", True)),
                "blur_ksize":
                    int(self.cfg.get("blur_ksize", 5)),
                "blur_sigma":
                    float(self.cfg.get("blur_sigma", 0)),
                "bilateral_enable":
                    bool(self.cfg.get(
                        "bilateral_enable", True)),
                "bilateral_d":
                    int(self.cfg.get("bilateral_d", 11)),
                "bilateral_sc":
                    int(self.cfg.get("bilateral_sc", 75)),
                "bilateral_ss":
                    int(self.cfg.get("bilateral_ss", 75)),
                "preprocess_order":
                    str(self.cfg.get(
                        "preprocess_order",
                        "blur_bilateral")),
                "clahe_enable":
                    bool(self.cfg.get(
                        "clahe_enable", True)),
                "clahe_clip":
                    float(self.cfg.get(
                        "clahe_clip", 4.0)),
                "clahe_grid":
                    (int(self.cfg.get(
                         "clahe_grid_x", 14)),
                     int(self.cfg.get(
                         "clahe_grid_y", 14))),
                "grad_thr":
                    int(self.cfg.get("grad_thr", 18)),
                "grad_method":
                    str(self.cfg.get(
                        "grad_method", "sobel")),
                "sobel_ksize":
                    int(self.cfg.get("sobel_ksize", 3)),
                "canny_thr1":
                    int(self.cfg.get("canny_thr1", 50)),
                "canny_thr2":
                    int(self.cfg.get("canny_thr2", 150)),
                "grad_thresh_type":
                    str(self.cfg.get(
                        "grad_thresh_type",
                        "binary_inv")),
                "adaptive_block":
                    int(self.cfg.get(
                        "adaptive_block", 11)),
                "adaptive_c":
                    int(self.cfg.get("adaptive_c", 2)),
                "min_grad_area":
                    int(self.cfg.get(
                        "min_grad_area", 50)),
                "morph_open_iter":
                    int(self.cfg.get(
                        "morph_open_iter", 2)),
                "morph_close_iter":
                    int(self.cfg.get(
                        "morph_close_iter", 3)),
                "morph_shape":
                    str(self.cfg.get(
                        "morph_shape", "ellipse")),
                "morph_extra_enable":
                    bool(self.cfg.get(
                        "morph_extra_enable", False)),
                "morph_extra_type":
                    str(self.cfg.get(
                        "morph_extra_type", "dilate")),
                "morph_extra_iter":
                    int(self.cfg.get(
                        "morph_extra_iter", 1)),
                "roi_thresh":
                    int(self.cfg.get("roi_thresh", 180)),
                "roi_thresh_type":
                    str(self.cfg.get(
                        "roi_thresh_type", "binary")),
                "roi_offset":
                    int(self.cfg.get("roi_offset", 75)),
                "roi_min_area":
                    int(self.cfg.get(
                        "roi_min_area", 1000)),
                "roi_contour_approx":
                    bool(self.cfg.get(
                        "roi_contour_approx", False)),
                "ok_ratio":
                    float(self.cfg.get(
                        "ok_ratio", 0.98)),
                "noise_max_area":
                    int(self.cfg.get(
                        "noise_max_area", 2500)),
                "border_noise_max_area":
                    int(self.cfg.get(
                        "border_noise_max_area", 2500)),
                "min_defect_area":
                    int(self.cfg.get(
                        "min_defect_area", 10)),
                "green_dilate_k":
                    int(self.cfg.get(
                        "green_dilate_k", 7)),
                "sep_open_k":
                    int(self.cfg.get("sep_open_k", 3)),
                "ring_dilate_k":
                    int(self.cfg.get(
                        "ring_dilate_k", 3)),
                "max_defect_count":
                    int(self.cfg.get(
                        "max_defect_count", -1)),
                "out_q":
                    int(self.cfg.get("out_q", 85)),
                "max_w":
                    int(self.cfg.get("max_w", 1600)),
                "n_threads":
                    int(self.cfg.get("n_threads", 4)),
            })
            insp._clahe = cv2.createCLAHE(
                clipLimit=insp.CFG["clahe_clip"],
                tileGridSize=insp.CFG["clahe_grid"])
        except Exception as e:
            print(f"[WARN] _apply_cfg_to_inspector: {e}")

    def _set_status(self, msg):
        try:
            self.after(
                0, lambda m=msg:
                self.var_status.set(m))
        except Exception:
            pass

    def _on_close(self):
        print("[App] Dang dong...")
        self._preview_running = False
        # Dung PLC monitor
        if self.plc_monitor:
            self.plc_monitor.stop()
            self.plc_monitor = None

        # Ngat ket noi PLC
        if self.plc_client:
            try:
                from modbus_plc import STATUS_IDLE
                self.plc_client.set_status(STATUS_IDLE)
            except Exception:
                pass
            self.plc_client.disconnect()
            self.plc_client = None

        if self.cam_thread:
            self.cam_thread.stop()
        try:
            save_cfg(self.cfg)
        except Exception as e:
            print(f"[WARN] save_cfg on close: {e}")
        self.after(200, self._do_destroy)

    def _do_destroy(self):
        try:
            self.destroy()
        except Exception:
            pass
        os._exit(0)


# ════════════════════════════════════════════════════════════════
# PATCHED BATCH - NGOAI class App
# ════════════════════════════════════════════════════════════════
def _patched_batch(input_dir="input_images",
                   output_dir="output_results",
                   return_stats=False,
                   name_fn=None):
    from concurrent.futures import ThreadPoolExecutor

    save_ok = bool(insp.CFG.get("save_ok_images", True))
    save_ng = bool(insp.CFG.get("save_ng_images", True))

    ok_dir = Path(output_dir) / "OK"
    ng_dir = Path(output_dir) / "NG"
    ok_dir.mkdir(parents=True, exist_ok=True)
    ng_dir.mkdir(parents=True, exist_ok=True)

    exts   = {".png", ".jpg", ".jpeg",
              ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in exts)
    if not images:
        return {"OK": 0, "NG": 0}, []

    counts   = {"OK": 0, "NG": 0, "ERROR": 0}
    ms_list  = []
    ok_count = 0
    ng_count = 0
    _lock    = threading.Lock()

    def process_with_options(args):
        nonlocal ok_count, ng_count
        p, o_dir, n_dir = args
        try:
            img = cv2.imread(str(p))
            if img is None:
                return p.name, "ERROR", 0, 0.0

            t0 = time.perf_counter()

            # FIX race condition: local objects
            cfg_snap      = dict(insp.CFG)
            local_kernel  = insp._make_kernel(
                cfg_snap["morph_k"],
                cfg_snap["morph_shape"])
            local_k_sep   = insp._make_kernel(
                cfg_snap["sep_open_k"],
                cfg_snap["morph_shape"])
            local_k1      = insp._make_kernel(
                cfg_snap["ring_dilate_k"],
                cfg_snap["morph_shape"])
            local_k_green = insp._make_kernel(
                cfg_snap["green_dilate_k"],
                cfg_snap["morph_shape"])
            _grid       = cfg_snap.get("clahe_grid", (14, 14))
            local_clahe = cv2.createCLAHE(
                clipLimit    = cfg_snap["clahe_clip"],
                tileGridSize = (
                    cfg_snap.get("clahe_grid_x", _grid[0]),
                    cfg_snap.get("clahe_grid_y", _grid[1]))
            ) if cfg_snap.get("clahe_enable", True) else None

            img_crop, gray, grad, low_mask, cleaned = \
                insp._pipeline_cpu_local(
                    img, cfg_snap,
                    local_clahe, local_kernel)
            roi_mask, c_orig, c_dila = \
                insp.get_roi_mask_local(img_crop, cfg_snap)
            label, ratio, noise_info = \
                insp.classify_local(
                    cleaned, roi_mask, cfg_snap,
                    local_k_sep, local_k1, local_k_green)

            ms = (time.perf_counter() - t0) * 1000

            should_save = (
                (label == "OK" and save_ok) or
                (label == "NG" and save_ng))

            if should_save:
                with _lock:
                    if label == "OK":
                        ok_count += 1
                        idx = ok_count
                    else:
                        ng_count += 1
                        idx = ng_count

                if name_fn is not None:
                    fname = name_fn(idx, label)
                else:
                    fname = (f"{p.stem}_{label}"
                             f"_{ratio*100:.0f}pct.jpg")

                dest_dir = o_dir \
                    if label == "OK" else n_dir
                out_path = dest_dir / fname

                insp.save_result_local(
                    out_path, label, ratio,
                    noise_info, img_crop, gray,
                    grad, low_mask, cleaned,
                    roi_mask, c_orig, c_dila,
                    cfg_snap, local_k_sep)
            else:
                with _lock:
                    if label == "OK":
                        ok_count += 1
                    else:
                        ng_count += 1

            return p.name, label, ms, ratio

        except Exception as e:
            print(f"[WARN] process {p.name}: {e}")
            return p.name, "ERROR", 0, 0

    n_workers = min(
        int(insp.CFG.get("n_threads", 4)),
        len(images), 4)
    print(f"[batch] {len(images)} anh"
          f" | {n_workers} workers"
          f" | save_ok={save_ok}"
          f" save_ng={save_ng}")

    with ThreadPoolExecutor(
            max_workers=n_workers) as exe:
        for name, label, ms, ratio in exe.map(
                process_with_options,
                [(p, ok_dir, ng_dir)
                 for p in images]):
            counts[label] = \
                counts.get(label, 0) + 1
            if ms:
                ms_list.append(ms)

    print(f"[batch] Ket qua: {counts}")
    return counts, ms_list


insp.batch_process = _patched_batch


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    def _signal_handler(sig, frame):
        print(f"\n[Signal] {sig} nhan duoc, thoat...")
        os._exit(0)

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        insp.init_gpu()
    except Exception as e:
        print(f"[WARN] init_gpu: {e}")

    os.environ["TCL_LIBRARY"] = os.environ.get(
        "TCL_LIBRARY", "")

    app = App()

    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[App] Ctrl+C")
    finally:
        os._exit(0)