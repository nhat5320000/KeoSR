# imv_camera.py
# IMV Camera wrapper - GainRaw + Resolution + Flip
import sys
import os
import cv2
import numpy as np
import ctypes
from ctypes import c_void_p, byref, c_double, c_uint64, c_bool

# ════════════════════════════════════════════════════════════════
# Tim SDK path
# ════════════════════════════════════════════════════════════════
def _find_sdk():
    candidates = [
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), "MVSDK"),
        "/home/vvqn/MVSDK",
        os.environ.get("MVSDK_PATH", ""),
    ]
    for p in candidates:
        if p and os.path.exists(p) \
                and os.path.exists(
                    os.path.join(p, "IMVApi.py")):
            return p
    return None

_SDK_PATH = _find_sdk()
if _SDK_PATH:
    if _SDK_PATH not in sys.path:
        sys.path.insert(0, _SDK_PATH)
    print(f"[IMVCamera] SDK found: {_SDK_PATH}")
else:
    print("[IMVCamera] KHONG tim thay MVSDK!")

try:
    from IMVApi import *
    from IMVDefines import *
    IMV_AVAILABLE = True
    print("[IMVCamera] SDK load OK")
except ImportError as e:
    IMV_AVAILABLE = False
    print(f"[IMVCamera] SDK khong co: {e}")


# ════════════════════════════════════════════════════════════════
# IMVCamera Class
# ════════════════════════════════════════════════════════════════
class IMVCamera:
    """
    Wrapper cho IMV camera
    - GainRaw  : 1, 2, 4, 8, 16
    - Exposure : microseconds
    - Width    : 256~1440 inc=4
    - Height   : 64~1080  inc=4
    - Flip     : ReverseX / ReverseY
    """

    GAIN_RAW_VALUES = [1, 2, 4, 8, 16]
    WIDTH_MIN  = 256;  WIDTH_MAX  = 1440; WIDTH_INC  = 4
    HEIGHT_MIN = 64;   HEIGHT_MAX = 1080; HEIGHT_INC = 4

    def __init__(self, cam_index: int = 0):
        self.cam_index  = cam_index
        self._cam       = None
        self.available  = False
        self._width     = 0
        self._height    = 0
        self._grabbing  = False

    # ════════════════════════════════════════════════════════
    # OPEN / CLOSE
    # ════════════════════════════════════════════════════════
    def open(self) -> bool:
        if not IMV_AVAILABLE:
            print("[IMVCamera] SDK khong co!")
            return False
        try:
            self._cam = MvCamera()

            deviceList = IMV_DeviceList()
            nRet = MvCamera.IMV_EnumDevices(
                deviceList,
                IMV_EInterfaceType.interfaceTypeAll)

            if nRet != IMV_OK or \
                    deviceList.nDevNum == 0:
                print("[IMVCamera] Khong tim thay camera!")
                return False

            print(f"[IMVCamera] Tim thay "
                  f"{deviceList.nDevNum} camera")

            index    = self.cam_index
            indexPtr = c_void_p(index)
            nRet     = self._cam.IMV_CreateHandle(
                IMV_ECreateHandleMode.modeByIndex,
                byref(indexPtr))
            if nRet != IMV_OK:
                print(f"[IMVCamera] CreateHandle "
                      f"loi: {nRet}")
                return False

            nRet = self._cam.IMV_Open()
            if nRet != IMV_OK:
                print(f"[IMVCamera] Open loi: {nRet}")
                return False

            nRet = self._cam.IMV_StartGrabbing()
            if nRet != IMV_OK:
                print(f"[IMVCamera] StartGrabbing "
                      f"loi: {nRet}")
                return False

            self.available = True
            self._grabbing = True

            # In thong so hien tai
            exp  = self.get_current_value("ExposureTime")
            gain = self.get_current_value("GainRaw")
            fps  = self.get_current_value(
                "AcquisitionFrameRate")
            ae   = self.get_current_value("ExposureAuto")
            ag   = self.get_current_value("GainAuto")
            w    = self.get_current_value("Width")
            h    = self.get_current_value("Height")
            rx   = self.get_current_value("ReverseX")
            ry   = self.get_current_value("ReverseY")

            self._width  = int(w) \
                if isinstance(w, (int, float)) else 0
            self._height = int(h) \
                if isinstance(h, (int, float)) else 0

            print(f"[IMVCamera] Camera [{index}] san sang")
            print(f"  Size    : {self._width}"
                  f"x{self._height}")
            print(f"  Exposure: {exp}us  [Auto={ae}]")
            print(f"  GainRaw : {gain}  "
                  f"(valid={self.GAIN_RAW_VALUES})"
                  f"  [Auto={ag}]")
            print(f"  FPS     : {fps}")
            print(f"  FlipX   : {rx}  FlipY: {ry}")
            return True

        except Exception as e:
            print(f"[IMVCamera] open() loi: {e}")
            import traceback
            traceback.print_exc()
            return False

    def isOpened(self) -> bool:
        return self.available

    def release(self):
        if self._cam and self.available:
            try:
                self._cam.IMV_StopGrabbing()
                self._cam.IMV_Close()
                self._cam.IMV_DestroyHandle()
                print("[IMVCamera] Da giai phong camera")
            except Exception as e:
                print(f"[IMVCamera] release() loi: {e}")
        self.available = False
        self._grabbing = False
        self._cam      = None

    # ════════════════════════════════════════════════════════
    # READ FRAME - COPY NGUYEN TU CODE GOC
    # ════════════════════════════════════════════════════════
    def read(self):
        if not self.available or self._cam is None:
            return False, None
        try:
            frame = IMV_Frame()
            nRet  = self._cam.IMV_GetFrame(frame, 500)
            if nRet != IMV_OK:
                return False, None

            width  = frame.frameInfo.width
            height = frame.frameInfo.height
            size   = frame.frameInfo.size

            self._width  = width
            self._height = height

            if (frame.frameInfo.pixelFormat ==
                    IMV_EPixelType.gvspPixelMono8):
                buf = (ctypes.c_ubyte * size)()
                ctypes.memmove(buf, frame.pData, size)
                img = np.frombuffer(
                    buf, dtype=np.uint8
                ).reshape(height, width)
                img = cv2.cvtColor(
                    img, cv2.COLOR_GRAY2BGR)
            else:
                dst_size     = width * height * 3
                buf          = \
                    (ctypes.c_ubyte * dst_size)()
                convertParam = IMV_PixelConvertParam()
                convertParam.nWidth          = width
                convertParam.nHeight         = height
                convertParam.ePixelFormat    = \
                    frame.frameInfo.pixelFormat
                convertParam.pSrcData        = frame.pData
                convertParam.nSrcDataLen     = size
                convertParam.eDstPixelFormat = \
                    IMV_EPixelType.gvspPixelBGR8
                convertParam.pDstBuf         = buf
                convertParam.nDstBufSize     = dst_size
                self._cam.IMV_PixelConvert(convertParam)
                img = np.frombuffer(
                    buf, dtype=np.uint8
                ).reshape(height, width, 3)

            self._cam.IMV_ReleaseFrame(frame)
            return True, img.copy()

        except Exception as e:
            print(f"[IMVCamera] read() loi: {e}")
            return False, None

    # ════════════════════════════════════════════════════════
    # GET CURRENT VALUE
    # ════════════════════════════════════════════════════════
    def get_current_value(self, param_name: str):
        if not self.available:
            return "Camera khong san sang"
        try:
            # ── Double ────────────────────────────────────
            if param_name == "ExposureTime":
                val  = c_double(0.0)
                nRet = self._cam\
                    .IMV_GetDoubleFeatureValue(
                        "ExposureTime", val)
                return round(val.value, 2) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            elif param_name in ("GainRaw", "Gain"):
                # Camera nay dung GainRaw
                val  = c_double(0.0)
                nRet = self._cam\
                    .IMV_GetDoubleFeatureValue(
                        "GainRaw", val)
                return round(val.value, 2) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            elif param_name in (
                    "AcquisitionFrameRate", "FrameRate"):
                val  = c_double(0.0)
                nRet = self._cam\
                    .IMV_GetDoubleFeatureValue(
                        "AcquisitionFrameRate", val)
                return round(val.value, 2) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            # ── Enum ──────────────────────────────────────
            elif param_name == "ExposureAuto":
                val  = c_uint64(0)
                nRet = self._cam\
                    .IMV_GetEnumFeatureValue(
                        "ExposureAuto", val)
                return "ON" \
                    if (nRet == IMV_OK and
                        val.value == 2) \
                    else "OFF"

            elif param_name == "GainAuto":
                val  = c_uint64(0)
                nRet = self._cam\
                    .IMV_GetEnumFeatureValue(
                        "GainAuto", val)
                return "ON" \
                    if (nRet == IMV_OK and
                        val.value == 2) \
                    else "OFF"

            # ── Int: Width / Height ───────────────────────
            elif param_name == "Width":
                val  = c_uint64(0)
                nRet = self._cam\
                    .IMV_GetIntFeatureValue(
                        "Width", val)
                return int(val.value) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            elif param_name == "Height":
                val  = c_uint64(0)
                nRet = self._cam\
                    .IMV_GetIntFeatureValue(
                        "Height", val)
                return int(val.value) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            # ── Bool: Flip ────────────────────────────────
            elif param_name == "ReverseX":
                val  = c_bool(False)
                nRet = self._cam\
                    .IMV_GetBoolFeatureValue(
                        "ReverseX", val)
                return bool(val.value) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

            elif param_name == "ReverseY":
                val  = c_bool(False)
                nRet = self._cam\
                    .IMV_GetBoolFeatureValue(
                        "ReverseY", val)
                return bool(val.value) \
                    if nRet == IMV_OK \
                    else f"Loi {nRet}"

        except Exception as e:
            return f"Loi: {str(e)}"
        return "Khong ho tro"

    # ════════════════════════════════════════════════════════
    # SET PARAMETER
    # ════════════════════════════════════════════════════════
    def set_parameter(self, param_name: str,
                      value) -> bool:
        if not self.available:
            return False
        try:
            # ── Double ────────────────────────────────────
            if param_name == "ExposureTime":
                self._cam.IMV_SetEnumFeatureValue(
                    "ExposureAuto", 0)
                nRet = self._cam\
                    .IMV_SetDoubleFeatureValue(
                        "ExposureTime", float(value))

            elif param_name in ("GainRaw", "Gain"):
                self._cam.IMV_SetEnumFeatureValue(
                    "GainAuto", 0)
                raw_val = self._nearest_gain_raw(
                    float(value))
                print(f"[IMVCamera] GainRaw={raw_val} "
                      f"(input={value})")
                nRet = self._cam\
                    .IMV_SetDoubleFeatureValue(
                        "GainRaw", float(raw_val))

            elif param_name == "FrameRate":
                self._cam.IMV_SetBoolFeatureValue(
                    "AcquisitionFrameRateEnable", True)
                nRet = self._cam\
                    .IMV_SetDoubleFeatureValue(
                        "AcquisitionFrameRate",
                        float(value))

            # ── Enum: Auto ────────────────────────────────
            elif param_name == "AutoExposure":
                mode = 2 if value else 0
                nRet = self._cam\
                    .IMV_SetEnumFeatureValue(
                        "ExposureAuto", mode)

            elif param_name == "AutoGain":
                mode = 2 if value else 0
                nRet = self._cam\
                    .IMV_SetEnumFeatureValue(
                        "GainAuto", mode)

            # ── Int: Width / Height ───────────────────────
            # CAN DUNG GRAB TRUOC!
            elif param_name == "Width":
                w = int(value)
                w = max(self.WIDTH_MIN,
                        min(self.WIDTH_MAX, w))
                w = (w // self.WIDTH_INC) \
                    * self.WIDTH_INC
                ok = self._set_resolution(
                    width=w, height=None)
                if ok:
                    self._width = w
                return ok

            elif param_name == "Height":
                h = int(value)
                h = max(self.HEIGHT_MIN,
                        min(self.HEIGHT_MAX, h))
                h = (h // self.HEIGHT_INC) \
                    * self.HEIGHT_INC
                ok = self._set_resolution(
                    width=None, height=h)
                if ok:
                    self._height = h
                return ok

            elif param_name == "Resolution":
                # value = (width, height)
                w = int(value[0])
                h = int(value[1])
                w = max(self.WIDTH_MIN,
                        min(self.WIDTH_MAX, w))
                w = (w // self.WIDTH_INC) \
                    * self.WIDTH_INC
                h = max(self.HEIGHT_MIN,
                        min(self.HEIGHT_MAX, h))
                h = (h // self.HEIGHT_INC) \
                    * self.HEIGHT_INC
                ok = self._set_resolution(
                    width=w, height=h)
                if ok:
                    self._width  = w
                    self._height = h
                return ok

            # ── Bool: Flip ────────────────────────────────
            elif param_name == "ReverseX":
                nRet = self._cam\
                    .IMV_SetBoolFeatureValue(
                        "ReverseX", bool(value))

            elif param_name == "ReverseY":
                nRet = self._cam\
                    .IMV_SetBoolFeatureValue(
                        "ReverseY", bool(value))

            else:
                print(f"[IMVCamera] unknown "
                      f"'{param_name}'")
                return False

            ok = (nRet == IMV_OK)
            if ok:
                print(f"[IMVCamera] Set {param_name}"
                      f"={value} OK")
            else:
                print(f"[IMVCamera] Set {param_name}"
                      f"={value} FAIL nRet={nRet}")
            return ok

        except Exception as e:
            print(f"[IMVCamera] set_parameter loi: {e}")
            return False

    def _set_resolution(self,
                        width=None,
                        height=None) -> bool:
        """Set Width/Height - PHAI dung grab truoc!"""
        if not self.available:
            return False
        try:
            print("[IMVCamera] Dung grab "
                  "de set resolution...")
            self._cam.IMV_StopGrabbing()
            self._grabbing = False

            ok = True
            if width is not None:
                nRet = self._cam\
                    .IMV_SetIntFeatureValue(
                        "Width", width)
                if nRet == IMV_OK:
                    print(f"[IMVCamera] Width={width} OK")
                else:
                    print(f"[IMVCamera] Width={width} "
                          f"FAIL: {nRet}")
                    ok = False

            if height is not None:
                nRet = self._cam\
                    .IMV_SetIntFeatureValue(
                        "Height", height)
                if nRet == IMV_OK:
                    print(f"[IMVCamera] Height={height} OK")
                else:
                    print(f"[IMVCamera] Height={height} "
                          f"FAIL: {nRet}")
                    ok = False

            # Bat lai grab
            nRet = self._cam.IMV_StartGrabbing()
            if nRet == IMV_OK:
                self._grabbing = True
                print("[IMVCamera] StartGrabbing lai OK")
            else:
                print(f"[IMVCamera] StartGrabbing "
                      f"loi: {nRet}")
                ok = False
            return ok

        except Exception as e:
            print(f"[IMVCamera] _set_resolution "
                  f"loi: {e}")
            try:
                self._cam.IMV_StartGrabbing()
                self._grabbing = True
            except Exception:
                pass
            return False

    # ════════════════════════════════════════════════════════
    # GET/SET - Interface giong cv2.VideoCapture
    # ════════════════════════════════════════════════════════
    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            v = self.get_current_value(
                "AcquisitionFrameRate")
            try:
                return float(v)
            except Exception:
                return 0.0
        elif prop_id == cv2.CAP_PROP_EXPOSURE:
            v = self.get_current_value("ExposureTime")
            try:
                return float(v)
            except Exception:
                return 0.0
        elif prop_id == cv2.CAP_PROP_GAIN:
            v = self.get_current_value("GainRaw")
            try:
                return float(v)
            except Exception:
                return 0.0
        return 0.0

    def set(self, prop_id: int,
            value: float) -> bool:
        if not self.available:
            return False

        if prop_id == cv2.CAP_PROP_EXPOSURE:
            if value < 0:
                return self.set_parameter(
                    "AutoExposure", True)
            else:
                self.set_parameter(
                    "AutoExposure", False)
                return self.set_parameter(
                    "ExposureTime", float(value))

        elif prop_id == cv2.CAP_PROP_AUTO_EXPOSURE:
            return self.set_parameter(
                "AutoExposure", value > 0.5)

        elif prop_id == cv2.CAP_PROP_GAIN:
            if value < 0:
                return self.set_parameter(
                    "AutoGain", True)
            else:
                self.set_parameter("AutoGain", False)
                return self.set_parameter(
                    "GainRaw", float(value))

        elif prop_id == cv2.CAP_PROP_FPS:
            return self.set_parameter(
                "FrameRate", float(value))

        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.set_parameter(
                "Width", int(value))

        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.set_parameter(
                "Height", int(value))

        elif prop_id in (
            cv2.CAP_PROP_BUFFERSIZE,
            cv2.CAP_PROP_BRIGHTNESS,
            cv2.CAP_PROP_CONTRAST,
            cv2.CAP_PROP_SATURATION,
            cv2.CAP_PROP_AUTOFOCUS,
        ):
            return True

        return False

    # ════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════
    def _nearest_gain_raw(self, value: float) -> int:
        """
        Lam tron ve GainRaw gan nhat hop le
        Valid: 1, 2, 4, 8, 16
        vd: 3→4, 5→4, 10→8
        """
        return min(self.GAIN_RAW_VALUES,
                   key=lambda x: abs(x - value))

    def print_params(self):
        if not self.available:
            print("[IMVCamera] Camera chua san sang")
            return
        w   = self.get_current_value("Width")
        h   = self.get_current_value("Height")
        exp = self.get_current_value("ExposureTime")
        ae  = self.get_current_value("ExposureAuto")
        gain= self.get_current_value("GainRaw")
        ag  = self.get_current_value("GainAuto")
        fps = self.get_current_value(
            "AcquisitionFrameRate")
        rx  = self.get_current_value("ReverseX")
        ry  = self.get_current_value("ReverseY")
        print("[IMVCamera] Current params:")
        print(f"  Size    : {w}x{h}")
        print(f"  Exposure: {exp}us  [Auto={ae}]")
        print(f"  GainRaw : {gain}  "
              f"(valid={self.GAIN_RAW_VALUES})"
              f"  [Auto={ag}]")
        print(f"  FPS     : {fps}")
        print(f"  FlipX   : {rx}  FlipY: {ry}")


# ════════════════════════════════════════════════════════════════
# TEST STANDALONE
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("IMV CAMERA TEST - GainRaw + Resolution + Flip")
    print("=" * 50)

    if not IMV_AVAILABLE:
        print("SDK khong co, thoat")
        exit(1)

    cam = IMVCamera(cam_index=0)
    if not cam.open():
        print("Khong mo duoc!")
        exit(1)

    print("\n[1] Thong so hien tai:")
    cam.print_params()

    print("\n[2] Test Exposure = 20000us...")
    cam.set_parameter("ExposureTime", 20000)
    print(f"    = {cam.get_current_value('ExposureTime')}")

    print("\n[3] Test GainRaw = 4...")
    cam.set_parameter("GainRaw", 4)
    print(f"    = {cam.get_current_value('GainRaw')}")

    print("\n[4] Test Resolution = 1280x1024...")
    cam.set_parameter("Resolution", (1280, 1024))
    print(f"    W={cam.get_current_value('Width')}"
          f" H={cam.get_current_value('Height')}")

    print("\n[5] Test FlipX = True...")
    cam.set_parameter("ReverseX", True)
    print(f"    ReverseX = "
          f"{cam.get_current_value('ReverseX')}")

    print("\n[6] Test FlipX = False...")
    cam.set_parameter("ReverseX", False)
    print(f"    ReverseX = "
          f"{cam.get_current_value('ReverseX')}")

    print("\n[7] Doc 3 frames...")
    for i in range(3):
        ret, frame = cam.read()
        s = f"{frame.shape}" \
            if (ret and frame is not None) else "FAIL"
        print(f"    Frame {i+1}: {s}")

    print("\n[8] Thong so cuoi:")
    cam.print_params()

    print("\n[9] Giai phong...")
    cam.release()
    print("DONE")