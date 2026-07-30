# ════════════════════════════════════════════════════════════════
# zone_tool_dialog.py  –  Dialog cấu hình Tool cho từng vùng ROI
# ════════════════════════════════════════════════════════════════
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import math
from pathlib import Path


class ZoneToolDialog(tk.Toplevel):
    """
    Dialog để gán và cấu hình Tool cho 1 vùng ROI
    Hỗ trợ: DefectDetection / ShapeSearch /
             BlobAnalysis / EdgeDetection / BinaryCheck
    """

    BG       = "#1e1e2e"
    FG       = "#cdd6f4"
    ACC      = "#89b4fa"
    ENTRY_BG = "#313244"

    TOOLS = [
        "DefectDetection",
        "ShapeSearch",
        "BlobAnalysis",
        "EdgeDetection",
        "BinaryCheck",
    ]

    def __init__(self, parent, zone_data,
                 cam_thread, zone_index=0):
        super().__init__(parent)
        self.cam_thread  = cam_thread
        self.zone_index  = zone_index
        self.result      = None
        self._running    = True

        import copy
        self.zone_data = copy.deepcopy(zone_data)
        cfg = self.zone_data.get("tool_cfg", {})

        # ════════════════════════════════════════
        # Khởi tạo TẤT CẢ var_ TRƯỚC _build_ui()
        # ════════════════════════════════════════

        # Thông tin vùng
        self.var_name    = tk.StringVar(
            value=self.zone_data.get(
                "name", f"Vùng {zone_index+1}"))
        self.var_enabled = tk.BooleanVar(
            value=bool(
                self.zone_data.get("enabled", True)))
        self.var_tool    = tk.StringVar(
            value=self.zone_data.get(
                "tool") or "DefectDetection")

        # DefectDetection
        self.var_blur_ksize  = tk.IntVar(
            value=int(cfg.get("blur_ksize", 5)))
        self.var_clahe_clip  = tk.DoubleVar(
            value=float(cfg.get("clahe_clip", 4.0)))
        self.var_grad_method = tk.StringVar(
            value=cfg.get("grad_method", "sobel"))
        self.var_grad_thr    = tk.IntVar(
            value=int(cfg.get("grad_thr", 18)))
        self.var_morph_open  = tk.IntVar(
            value=int(cfg.get("morph_open_iter", 2)))
        self.var_morph_close = tk.IntVar(
            value=int(cfg.get("morph_close_iter", 3)))
        self.var_ok_ratio    = tk.DoubleVar(
            value=float(cfg.get("ok_ratio", 0.98)))
        self.var_canny_thr1  = tk.IntVar(
            value=int(cfg.get("canny_thr1", 50)))
        self.var_canny_thr2  = tk.IntVar(
            value=int(cfg.get("canny_thr2", 150)))

        # ShapeSearch
        self.var_tpl_path    = tk.StringVar(
            value=cfg.get("template_path", ""))
        self.var_match_thr   = tk.DoubleVar(
            value=float(cfg.get("match_thr", 0.8)))
        self.var_match_method = tk.StringVar(
            value=cfg.get("match_method",
                          "TM_CCOEFF_NORMED"))

        # BlobAnalysis
        self.var_blob_thresh      = tk.IntVar(
            value=int(cfg.get("thresh", 128)))
        self.var_blob_thresh_type = tk.StringVar(
            value=cfg.get("thresh_type", "otsu"))
        self.var_blob_min_area    = tk.IntVar(
            value=int(cfg.get("min_area", 100)))
        self.var_blob_max_area    = tk.IntVar(
            value=int(cfg.get("max_area", 50000)))
        self.var_blob_min_count   = tk.IntVar(
            value=int(cfg.get("min_count", 1)))
        self.var_blob_max_count   = tk.IntVar(
            value=int(cfg.get("max_count", -1)))

        # EdgeDetection
        self.var_edge_thr1     = tk.IntVar(
            value=int(cfg.get("canny_thr1", 50)))
        self.var_edge_thr2     = tk.IntVar(
            value=int(cfg.get("canny_thr2", 150)))
        self.var_edge_ok_ratio = tk.DoubleVar(
            value=float(
                cfg.get("ok_edge_ratio", 0.8)))
        self.var_edge_blur     = tk.IntVar(
            value=int(cfg.get("blur_ksize", 3)))

        # BinaryCheck
        self.var_bin_thresh      = tk.IntVar(
            value=int(cfg.get("thresh", 128)))
        self.var_bin_thresh_type = tk.StringVar(
            value=cfg.get("thresh_type", "otsu"))
        self.var_bin_ok_ratio    = tk.DoubleVar(
            value=float(cfg.get("ok_ratio", 0.9)))
        self.var_bin_invert      = tk.BooleanVar(
            value=bool(cfg.get("invert", False)))

        # ════════════════════════════════════════
        # Build UI sau khi init xong tất cả vars
        # ════════════════════════════════════════
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW",
                      self._on_cancel)

    # ════════════════════════════════════════════
    # BUILD UI
    # ════════════════════════════════════════════
    def _build_ui(self):
        BG, FG, ACC = self.BG, self.FG, self.ACC

        self.title(
            f"⚙️ Cấu hình Tool – "
            f"Vùng {self.zone_index+1}")
        self.configure(bg=BG)
        self.geometry("680x640")
        self.resizable(True, True)
#       self.grab_set()  # disabled for Jetson

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TNotebook",     background=BG)
        style.configure(
            "TNotebook.Tab", background="#252535",
            foreground=FG,
            font=("Consolas", 9, "bold"),
            padding=[8, 4])
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#313244")],
            foreground=[("selected", ACC)])
        style.configure("TFrame", background=BG)
        style.configure("TLabel",
                        background=BG, foreground=FG)
        style.configure(
            "TLabelframe",
            background=BG, foreground=ACC)
        style.configure(
            "TLabelframe.Label",
            background=BG, foreground=ACC,
            font=("Consolas", 9, "bold"))

        # ── Title ────────────────────────────────
        tk.Label(self,
                 text="⚙️  CẤU HÌNH TOOL CHO VÙNG",
                 bg=BG, fg=ACC,
                 font=("Consolas", 12, "bold")
                 ).pack(pady=(10, 4))
        ttk.Separator(self, orient="horizontal"
                      ).pack(fill=tk.X, padx=12)

        # ── Zone info bar ────────────────────────
        info_bar = tk.Frame(self, bg="#252535",
                            relief="groove", bd=1)
        info_bar.pack(fill=tk.X, padx=8,
                      pady=(6, 0))

        tk.Label(info_bar, text="Tên vùng:",
                 bg="#252535", fg=FG,
                 font=("Consolas", 9)
                 ).pack(side=tk.LEFT, padx=(8, 2))
        tk.Entry(info_bar,
                 textvariable=self.var_name,
                 bg=self.ENTRY_BG, fg=FG,
                 insertbackground=FG,
                 font=("Consolas", 9),
                 width=16, relief="flat"
                 ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Checkbutton(
            info_bar, text="Bật",
            variable=self.var_enabled,
            bg="#252535", fg="#a6e3a1",
            selectcolor="#313244",
            activebackground="#252535",
            font=("Consolas", 9, "bold")
        ).pack(side=tk.LEFT, padx=4)

        tk.Label(info_bar, text="Tool:",
                 bg="#252535", fg=FG,
                 font=("Consolas", 9)
                 ).pack(side=tk.LEFT, padx=(12, 2))

        tool_cb = ttk.Combobox(
            info_bar,
            textvariable=self.var_tool,
            values=self.TOOLS,
            state="readonly",
            font=("Consolas", 9),
            width=18)
        tool_cb.pack(side=tk.LEFT, padx=(0, 8))
        # Chuyển tab khi đổi tool
        tool_cb.bind("<<ComboboxSelected>>",
                     self._on_tool_changed)

        # ── Notebook ─────────────────────────────
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True,
                     padx=8, pady=4)

        # Tạo tất cả tabs
        self._tab_frames = {}
        tab_defs = [
            ("🔍 Defect",   "DefectDetection"),
            ("🔷 Shape",    "ShapeSearch"),
            ("🔵 Blob",     "BlobAnalysis"),
            ("📏 Edge",     "EdgeDetection"),
            ("⬛ Binary",   "BinaryCheck"),
            ("👁 Preview",  "preview"),
        ]
        builders = {
            "DefectDetection": self._tab_defect,
            "ShapeSearch":     self._tab_shape,
            "BlobAnalysis":    self._tab_blob,
            "EdgeDetection":   self._tab_edge,
            "BinaryCheck":     self._tab_binary,
            "preview":         self._tab_preview,
        }
        for tab_name, key in tab_defs:
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=tab_name)
            self._tab_frames[key] = frame
            if key in builders:
                builders[key](frame)

        # Chọn tab theo tool hiện tại
        self._select_tab_for_tool(
            self.var_tool.get())

        # ── Bottom buttons ────────────────────────
        bot = tk.Frame(self, bg=BG)
        bot.pack(fill=tk.X, padx=8,
                 pady=(0, 8))

        tk.Button(
            bot, text="💾  Lưu & Đóng",
            bg="#a6e3a1", fg="black",
            font=("Consolas", 11, "bold"),
            relief="flat", padx=16, pady=7,
            command=self._on_save
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True, padx=(0, 4))

        tk.Button(
            bot, text="👁  Xem trước",
            bg="#89b4fa", fg="black",
            font=("Consolas", 11, "bold"),
            relief="flat", padx=16, pady=7,
            command=self._run_preview
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True, padx=(4, 4))

        tk.Button(
            bot, text="✖  Hủy",
            bg="#f38ba8", fg="black",
            font=("Consolas", 11, "bold"),
            relief="flat", padx=16, pady=7,
            command=self._on_cancel
        ).pack(side=tk.LEFT, fill=tk.X,
               expand=True, padx=(4, 0))

    # ════════════════════════════════════════════
    # HELPER: Scrollable frame
    # ════════════════════════════════════════════
    def _scrollable(self, parent):
        canvas = tk.Canvas(
            parent, bg=self.BG,
            highlightthickness=0)
        sb = ttk.Scrollbar(
            parent, orient="vertical",
            command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT,
                    fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, bg=self.BG)
        wid   = canvas.create_window(
            (0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(
                wid, width=e.width))
        return inner

    # ════════════════════════════════════════════
    # HELPER: Tạo row param
    # ════════════════════════════════════════════
    def _row(self, parent, label, widget_fn,
             tooltip=""):
        BG, FG = self.BG, self.FG
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(
            row, text=f"{label}:",
            bg=BG, fg=FG,
            font=("Consolas", 9),
            width=20, anchor="w"
        ).pack(side=tk.LEFT)
        widget_fn(row)
        if tooltip:
            tk.Label(
                row,
                text=f"ℹ {tooltip}",
                bg=BG, fg="#585b70",
                font=("Consolas", 7)
            ).pack(side=tk.LEFT, padx=4)

    def _entry(self, parent, var, width=8):
        tk.Entry(
            parent, textvariable=var,
            bg=self.ENTRY_BG, fg=self.FG,
            insertbackground=self.FG,
            font=("Consolas", 9),
            width=width, relief="flat"
        ).pack(side=tk.LEFT, padx=(0, 4))

    def _slider(self, parent, var,
                mn, mx, res=1):
        def _cmd(v):
            try:
                vv = int(float(v)) \
                    if res == 1 \
                    else round(float(v), 3)
                var.set(vv)
            except Exception:
                pass
        sl = tk.Scale(
            parent, from_=mn, to=mx,
            orient="horizontal",
            resolution=res,
            command=_cmd,
            bg=self.BG, fg=self.FG,
            troughcolor="#313244",
            highlightthickness=0,
            font=("Consolas", 7),
            length=200, showvalue=False)
        try:
            sl.set(float(var.get()))
        except Exception:
            sl.set(mn)
        sl.pack(side=tk.LEFT, fill=tk.X,
                expand=True, padx=(0, 4))

    def _combo(self, parent, var, values,
               width=16):
        ttk.Combobox(
            parent, textvariable=var,
            values=values,
            state="readonly",
            font=("Consolas", 9),
            width=width
        ).pack(side=tk.LEFT)

    def _section(self, parent, title,
                 color="#89b4fa"):
        tk.Label(
            parent, text=f"  {title}",
            bg=color, fg="black",
            font=("Consolas", 9, "bold"),
            anchor="w", pady=3
        ).pack(fill=tk.X, padx=8, pady=(10, 4))

    # ════════════════════════════════════════════
    # TAB: DefectDetection
    # ════════════════════════════════════════════
    def _tab_defect(self, parent):
        f = self._scrollable(parent)

        self._section(f, "🌀 Tiền xử lý",
                      "#89dceb")
        self._row(f, "Blur ksize",
                  lambda p: (self._entry(
                      p, self.var_blur_ksize),
                      self._slider(
                          p, self.var_blur_ksize,
                          1, 21)),
                  tooltip="Kernel làm mờ (lẻ)")
        self._row(f, "CLAHE clip",
                  lambda p: (self._entry(
                      p, self.var_clahe_clip, 6),
                      self._slider(
                          p, self.var_clahe_clip,
                          0.5, 8.0, 0.1)),
                  tooltip="Tương phản cục bộ")

        self._section(f, "📐 Gradient", "#f9e2af")
        self._row(f, "Grad method",
                  lambda p: self._combo(
                      p, self.var_grad_method,
                      ["sobel", "scharr",
                       "laplacian", "canny"]))
        self._row(f, "Grad threshold",
                  lambda p: (self._entry(
                      p, self.var_grad_thr),
                      self._slider(
                          p, self.var_grad_thr,
                          1, 100)),
                  tooltip="Ngưỡng gradient (1-100)")
        self._row(f, "Canny thr1",
                  lambda p: (self._entry(
                      p, self.var_canny_thr1),
                      self._slider(
                          p, self.var_canny_thr1,
                          1, 300)))
        self._row(f, "Canny thr2",
                  lambda p: (self._entry(
                      p, self.var_canny_thr2),
                      self._slider(
                          p, self.var_canny_thr2,
                          1, 300)))

        self._section(f, "🔶 Morphology",
                      "#cba6f7")
        self._row(f, "Open iterations",
                  lambda p: (self._entry(
                      p, self.var_morph_open),
                      self._slider(
                          p, self.var_morph_open,
                          1, 10)))
        self._row(f, "Close iterations",
                  lambda p: (self._entry(
                      p, self.var_morph_close),
                      self._slider(
                          p, self.var_morph_close,
                          1, 10)))

        self._section(f, "⚖️ Đánh giá", "#fab387")
        self._row(f, "OK ratio",
                  lambda p: (self._entry(
                      p, self.var_ok_ratio, 6),
                      self._slider(
                          p, self.var_ok_ratio,
                          0.5, 1.0, 0.01)),
                  tooltip="Tỉ lệ pixel OK tối thiểu")

    # ════════════════════════════════════════════
    # TAB: ShapeSearch
    # ════════════════════════════════════════════
    def _tab_shape(self, parent):
        f = self._scrollable(parent)

        self._section(f, "🔷 Template Matching",
                      "#89b4fa")

        # Template path
        def _tpl_row(p):
            tk.Entry(
                p, textvariable=self.var_tpl_path,
                bg=self.ENTRY_BG, fg=self.FG,
                insertbackground=self.FG,
                font=("Consolas", 9),
                width=24, relief="flat"
            ).pack(side=tk.LEFT,
                   fill=tk.X, expand=True)
            tk.Button(
                p, text="...",
                bg="#313244", fg=self.FG,
                font=("Consolas", 9),
                relief="flat", padx=4,
                command=self._browse_template
            ).pack(side=tk.LEFT, padx=2)

        self._row(f, "Template file", _tpl_row,
                  tooltip="Ảnh mẫu để so sánh")

        # Preview template
        self.lbl_tpl_preview = tk.Label(
            f, bg="#0d0d1a",
            text="(Chưa có template)",
            fg="#585b70",
            font=("Consolas", 8))
        self.lbl_tpl_preview.pack(
            padx=16, pady=4, anchor="w")

        # Nút chụp từ camera
        tk.Button(
            f, text="Chup tu Camera lam Template",
            bg="#a6e3a1", fg="black",
            font=("Consolas", 9, "bold"),
            relief="flat", padx=10, pady=4,
            command=self._capture_template
        ).pack(padx=16, pady=4, anchor="w")

        self._section(f, "⚙️ Thông số", "#89b4fa")
        self._row(f, "Match threshold",
                  lambda p: (self._entry(
                      p, self.var_match_thr, 6),
                      self._slider(
                          p, self.var_match_thr,
                          0.1, 1.0, 0.01)),
                  tooltip="Ngưỡng khớp mẫu (0-1)")
        self._row(f, "Match method",
                  lambda p: self._combo(
                      p, self.var_match_method,
                      ["TM_CCOEFF_NORMED",
                       "TM_CCORR_NORMED",
                       "TM_SQDIFF_NORMED"]))

        # Load preview nếu có file
        self._load_tpl_preview()

    def _browse_template(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh template",
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ])
        if path:
            self.var_tpl_path.set(path)
            self._load_tpl_preview()

    def _load_tpl_preview(self):
        if not hasattr(self, "lbl_tpl_preview"):
            return
        path = self.var_tpl_path.get()
        if path and Path(path).exists():
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                sc = min(200/w, 100/h, 1.0)
                img = cv2.resize(
                    img, None, fx=sc, fy=sc,
                    interpolation=cv2.INTER_AREA)
                rgb  = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB)
                pimg = ImageTk.PhotoImage(
                    Image.fromarray(rgb))
                self.lbl_tpl_preview.configure(
                    image=pimg, text="")
                self.lbl_tpl_preview._pimg = pimg
                return
        self.lbl_tpl_preview.configure(
            image="",
            text="(Chưa có template)")

    def _capture_template(self):
        if self.cam_thread is None or \
                self.cam_thread.last_frame is None:
            messagebox.showwarning(
                "Chú ý",
                "Camera chưa hoạt động!",
                parent=self)
            return
        frame = self.cam_thread.last_frame.copy()
        # Crop theo zone nếu có
        zone = self.zone_data
        try:
            t, d = zone["type"], zone["data"]
            h, w = frame.shape[:2]
            if t == "rect":
                crop = frame[
                    max(0,d[1]):min(h,d[3]),
                    max(0,d[0]):min(w,d[2])]
            elif t == "circle":
                crop = frame[
                    max(0,d[1]-d[2]):min(h,d[1]+d[2]),
                    max(0,d[0]-d[2]):min(w,d[0]+d[2])]
            else:
                crop = frame
        except Exception:
            crop = frame

        # Lưu file
        tpl_dir = Path("templates")
        tpl_dir.mkdir(exist_ok=True)
        fname = (f"tpl_zone{self.zone_index}"
                 f"_{self.var_name.get()}.jpg")
        save_path = tpl_dir / fname
        cv2.imwrite(str(save_path), crop)
        self.var_tpl_path.set(str(save_path))
        self._load_tpl_preview()
        messagebox.showinfo(
            "OK",
            f"✅ Đã lưu template:\n{save_path}",
            parent=self)

    # ════════════════════════════════════════════
    # TAB: BlobAnalysis
    # ════════════════════════════════════════════
    def _tab_blob(self, parent):
        f = self._scrollable(parent)

        self._section(f, "🎯 Threshold", "#89dceb")
        self._row(f, "Threshold",
                  lambda p: (self._entry(
                      p, self.var_blob_thresh),
                      self._slider(
                          p, self.var_blob_thresh,
                          0, 255)))
        self._row(f, "Thresh type",
                  lambda p: self._combo(
                      p, self.var_blob_thresh_type,
                      ["binary", "binary_inv",
                       "otsu"]))

        self._section(f, "📐 Lọc Blob", "#89dceb")
        self._row(f, "Min area (px²)",
                  lambda p: (self._entry(
                      p, self.var_blob_min_area),
                      self._slider(
                          p, self.var_blob_min_area,
                          1, 10000)))
        self._row(f, "Max area (px²)",
                  lambda p: (self._entry(
                      p, self.var_blob_max_area),
                      self._slider(
                          p, self.var_blob_max_area,
                          100, 100000)))
        self._row(f, "Min count",
                  lambda p: (self._entry(
                      p, self.var_blob_min_count),
                      self._slider(
                          p, self.var_blob_min_count,
                          0, 50)),
                  tooltip="Số blob tối thiểu")
        self._row(f, "Max count (-1=∞)",
                  lambda p: (self._entry(
                      p, self.var_blob_max_count),
                      self._slider(
                          p, self.var_blob_max_count,
                          -1, 50)),
                  tooltip="-1 = không giới hạn")

    # ════════════════════════════════════════════
    # TAB: EdgeDetection
    # ════════════════════════════════════════════
    def _tab_edge(self, parent):
        f = self._scrollable(parent)

        self._section(f, "📏 Canny Edge",
                      "#f9e2af")
        self._row(f, "Blur ksize",
                  lambda p: (self._entry(
                      p, self.var_edge_blur),
                      self._slider(
                          p, self.var_edge_blur,
                          1, 21)))
        self._row(f, "Canny thr1",
                  lambda p: (self._entry(
                      p, self.var_edge_thr1),
                      self._slider(
                          p, self.var_edge_thr1,
                          1, 300)))
        self._row(f, "Canny thr2",
                  lambda p: (self._entry(
                      p, self.var_edge_thr2),
                      self._slider(
                          p, self.var_edge_thr2,
                          1, 300)))

        self._section(f, "⚖️ Đánh giá", "#fab387")
        self._row(f, "OK edge ratio",
                  lambda p: (self._entry(
                      p, self.var_edge_ok_ratio,
                      6),
                      self._slider(
                          p, self.var_edge_ok_ratio,
                          0.1, 1.0, 0.01)),
                  tooltip="Tỉ lệ pixel biên tối thiểu")

    # ════════════════════════════════════════════
    # TAB: BinaryCheck
    # ════════════════════════════════════════════
    def _tab_binary(self, parent):
        f = self._scrollable(parent)

        self._section(f, "⬛ Binary Check",
                      "#a6e3a1")
        self._row(f, "Threshold",
                  lambda p: (self._entry(
                      p, self.var_bin_thresh),
                      self._slider(
                          p, self.var_bin_thresh,
                          0, 255)))
        self._row(f, "Thresh type",
                  lambda p: self._combo(
                      p, self.var_bin_thresh_type,
                      ["otsu", "binary",
                       "binary_inv"]))

        def _inv_row(p):
            tk.Checkbutton(
                p,
                variable=self.var_bin_invert,
                text="Đảo ngược binary",
                bg=self.BG, fg=self.FG,
                selectcolor="#313244",
                activebackground=self.BG,
                font=("Consolas", 9)
            ).pack(side=tk.LEFT)

        self._row(f, "Invert", _inv_row)

        self._section(f, "⚖️ Đánh giá", "#fab387")
        self._row(f, "OK ratio",
                  lambda p: (self._entry(
                      p, self.var_bin_ok_ratio,
                      6),
                      self._slider(
                          p, self.var_bin_ok_ratio,
                          0.1, 1.0, 0.01)),
                  tooltip="Tỉ lệ pixel trắng tối thiểu")

    # ════════════════════════════════════════════
    # TAB: Preview (live camera)
    # ════════════════════════════════════════════
    def _tab_preview(self, parent):
        BG = self.BG

        # Canvas preview
        self.canvas_prev = tk.Canvas(
            parent, bg="black",
            highlightthickness=1,
            highlightbackground=self.ACC)
        self.canvas_prev.pack(
            fill=tk.BOTH, expand=True,
            padx=6, pady=4)

        # Result label
        self.lbl_prev_result = tk.Label(
            parent,
            text="Nhấn 👁 Xem trước để chạy",
            bg=BG, fg=self.ACC,
            font=("Consolas", 9),
            anchor="w", wraplength=600)
        self.lbl_prev_result.pack(
            fill=tk.X, padx=8, pady=(0, 4))

        # Auto preview button
        tk.Button(
            parent,
            text="🔄 Chạy Preview",
            bg="#89b4fa", fg="black",
            font=("Consolas", 9, "bold"),
            relief="flat", padx=8, pady=4,
            command=self._run_preview
        ).pack(padx=8, pady=4)

    # ════════════════════════════════════════════
    # PREVIEW
    # ════════════════════════════════════════════
    def _run_preview(self):
        """Chạy tool trên frame hiện tại"""
        if self.cam_thread is None or \
                self.cam_thread.last_frame is None:
            messagebox.showwarning(
                "Chú ý",
                "Camera chưa hoạt động!",
                parent=self)
            return

        try:
            from zone_processor import ZoneProcessor
        except ImportError:
            messagebox.showerror(
                "Lỗi",
                "Không tìm thấy zone_processor.py!",
                parent=self)
            return

        # Thu thập cfg hiện tại
        self._collect_cfg()
        frame = self.cam_thread.last_frame.copy()

        processor = ZoneProcessor()
        result    = processor.process_zone(
            frame, self.zone_data)

        label  = result["label"]
        score  = result.get("score", 0)
        detail = result.get("detail", "")
        grid   = result.get("grid")

        # Hiển thị kết quả text
        col_txt = "✅" if label == "OK" else "❌"
        self.lbl_prev_result.configure(
            text=f"{col_txt} {label}  "
                 f"score={score*100:.1f}%  |  "
                 f"{detail}")

        # Hiển thị grid ảnh phân tích
        if grid is not None:
            self._show_preview_grid(grid)

        # Chuyển sang tab Preview
        try:
            self.nb.select(5)
        except Exception:
            pass

    def _show_preview_grid(self,
                            grid: np.ndarray):
        """Hiển thị grid phân tích lên canvas"""
        if not hasattr(self, "canvas_prev"):
            return
        cw = self.canvas_prev.winfo_width()
        ch = self.canvas_prev.winfo_height()
        if cw < 10: cw = 640
        if ch < 10: ch = 300

        gh, gw = grid.shape[:2]
        sc = min(cw/gw, ch/gh, 1.0)
        disp = cv2.resize(
            grid,
            (int(gw*sc), int(gh*sc)),
            interpolation=cv2.INTER_AREA)

        rgb  = cv2.cvtColor(
            disp, cv2.COLOR_BGR2RGB)
        pimg = ImageTk.PhotoImage(
            Image.fromarray(rgb))

        self.canvas_prev.delete("all")
        self.canvas_prev.create_image(
            0, 0, anchor="nw", image=pimg)
        self.canvas_prev._pimg = pimg

    # ════════════════════════════════════════════
    # COLLECT CFG từ vars → zone_data
    # ════════════════════════════════════════════
    def _collect_cfg(self):
        """Thu thập thông số → cập nhật zone_data"""
        tool = self.var_tool.get()
        cfg  = {}

        if tool == "DefectDetection":
            k = int(self.var_blur_ksize.get())
            cfg = {
                "blur_ksize":       k if k%2==1 else k+1,
                "clahe_clip":       round(float(
                    self.var_clahe_clip.get()), 2),
                "grad_method":      self.var_grad_method.get(),
                "grad_thr":         int(self.var_grad_thr.get()),
                "canny_thr1":       int(self.var_canny_thr1.get()),
                "canny_thr2":       int(self.var_canny_thr2.get()),
                "morph_open_iter":  int(self.var_morph_open.get()),
                "morph_close_iter": int(self.var_morph_close.get()),
                "ok_ratio":         round(float(
                    self.var_ok_ratio.get()), 3),
            }
        elif tool == "ShapeSearch":
            cfg = {
                "template_path": self.var_tpl_path.get(),
                "match_thr":     round(float(
                    self.var_match_thr.get()), 3),
                "match_method":  self.var_match_method.get(),
            }
        elif tool == "BlobAnalysis":
            cfg = {
                "thresh":      int(self.var_blob_thresh.get()),
                "thresh_type": self.var_blob_thresh_type.get(),
                "min_area":    int(self.var_blob_min_area.get()),
                "max_area":    int(self.var_blob_max_area.get()),
                "min_count":   int(self.var_blob_min_count.get()),
                "max_count":   int(self.var_blob_max_count.get()),
            }
        elif tool == "EdgeDetection":
            k = int(self.var_edge_blur.get())
            cfg = {
                "canny_thr1":    int(self.var_edge_thr1.get()),
                "canny_thr2":    int(self.var_edge_thr2.get()),
                "ok_edge_ratio": round(float(
                    self.var_edge_ok_ratio.get()), 3),
                "blur_ksize":    k if k%2==1 else k+1,
            }
        elif tool == "BinaryCheck":
            cfg = {
                "thresh":      int(self.var_bin_thresh.get()),
                "thresh_type": self.var_bin_thresh_type.get(),
                "ok_ratio":    round(float(
                    self.var_bin_ok_ratio.get()), 3),
                "invert":      bool(self.var_bin_invert.get()),
            }

        self.zone_data["tool"]     = tool
        self.zone_data["tool_cfg"] = cfg
        self.zone_data["name"]     = \
            self.var_name.get().strip() or \
            f"Vùng {self.zone_index+1}"
        self.zone_data["enabled"]  = \
            bool(self.var_enabled.get())

    # ════════════════════════════════════════════
    # TOOL CHANGED → chuyển tab
    # ════════════════════════════════════════════
    def _on_tool_changed(self, event=None):
        self._select_tab_for_tool(
            self.var_tool.get())

    def _select_tab_for_tool(self, tool):
        """Chuyển sang tab tương ứng với tool"""
        tab_map = {
            "DefectDetection": 0,
            "ShapeSearch":     1,
            "BlobAnalysis":    2,
            "EdgeDetection":   3,
            "BinaryCheck":     4,
        }
        idx = tab_map.get(tool, 0)
        try:
            self.nb.select(idx)
        except Exception:
            pass

    # ════════════════════════════════════════════
    # SAVE / CANCEL
    # ════════════════════════════════════════════
    def _on_save(self):
        self._collect_cfg()
        self.result = self.zone_data
        self._running = False
        self.destroy()

    def _on_cancel(self):
        self.result   = None
        self._running = False
        self.destroy()