# modbus_plc.py
# Modbus TCP - Ket noi PLC
# M1000: Bat camera (Coil)
# M1001: Trigger chup (Coil) - HIGH=chup LOW=dung+xu ly
# D2000: So anh chup (Holding Register)
# D2002: So anh OK
# D2004: So anh NG
# D2006: Thoi gian xu ly (ms)
# D2008: Trang thai (0=idle 1=chup 2=xu ly 3=xong)

import threading
import time
from pymodbus.client import ModbusTcpClient


# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════

# Coil address (PLC → Camera)
COIL_CAMERA_ON   = 1000   # M1000
COIL_TRIGGER     = 1001   # M1001

# Holding Register address (Camera → PLC)
REG_FRAME_COUNT  = 2000   # D2000
REG_OK_COUNT     = 2002   # D2002
REG_NG_COUNT     = 2004   # D2004
REG_PROC_TIME_MS = 2006   # D2006
REG_STATUS       = 2008   # D2008
REG_VERDICT      = 2010   # D2010 - Ket qua OK/NG

# Status values
STATUS_IDLE      = 0
STATUS_CAPTURING = 1
STATUS_PROCESSING= 2
STATUS_DONE      = 3
STATUS_ERROR     = 9


# ════════════════════════════════════════════════════════════════
# PLCClient
# ════════════════════════════════════════════════════════════════
class PLCClient:
    """
    Modbus TCP Client
    Doc Coil tu PLC, ghi Holding Register ve PLC
    """

    def __init__(self,
                 host:    str = "192.168.4.20",
                 port:    int = 502,
                 unit_id: int = 1,
                 timeout: float = 3.0):
        self.host    = host
        self.port    = port
        self.unit_id = unit_id
        self.timeout = timeout
        self._client = None
        self._lock   = threading.Lock()
        self.connected = False

    # ════════════════════════════════════════════════
    # CONNECT / DISCONNECT
    # ════════════════════════════════════════════════
    def connect(self) -> bool:
        try:
            self._client = ModbusTcpClient(
                host    = self.host,
                port    = self.port,
                timeout = self.timeout)
            ok = self._client.connect()
            self.connected = ok
            if ok:
                print(f"[PLC] Ket noi OK "
                      f"{self.host}:{self.port}")
            else:
                print(f"[PLC] Ket noi FAIL "
                      f"{self.host}:{self.port}")
            return ok
        except Exception as e:
            print(f"[PLC] connect() loi: {e}")
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self._client:
                self._client.close()
            self.connected = False
            print("[PLC] Da ngat ket noi")
        except Exception as e:
            print(f"[PLC] disconnect() loi: {e}")

    def reconnect(self) -> bool:
        self.disconnect()
        time.sleep(0.5)
        return self.connect()

    # ════════════════════════════════════════════════
    # READ COIL
    # ════════════════════════════════════════════════
    def read_coil(self, address: int) -> bool:
        """Doc 1 coil, tra ve True/False"""
        with self._lock:
            try:
                result = self._client.read_coils(
                    address  = address,
                    count    = 1,
                    slave    = self.unit_id)
                if result.isError():
                    return False
                return bool(result.bits[0])
            except Exception as e:
                print(f"[PLC] read_coil({address}) "
                      f"loi: {e}")
                self.connected = False
                return False

    def read_coils(self,
                   address: int,
                   count:   int) -> list:
        """Doc nhieu coil"""
        with self._lock:
            try:
                result = self._client.read_coils(
                    address = address,
                    count   = count,
                    slave   = self.unit_id)
                if result.isError():
                    return [False] * count
                return list(result.bits[:count])
            except Exception as e:
                print(f"[PLC] read_coils loi: {e}")
                self.connected = False
                return [False] * count

    # ════════════════════════════════════════════════
    # WRITE REGISTER
    # ════════════════════════════════════════════════
    def write_register(self,
                       address: int,
                       value:   int) -> bool:
        """Ghi 1 holding register (16-bit)"""
        with self._lock:
            try:
                # Clamp 0~65535
                value = max(0, min(65535, int(value)))
                result = self._client\
                    .write_register(
                        address = address,
                        value   = value,
                        slave   = self.unit_id)
                return not result.isError()
            except Exception as e:
                print(f"[PLC] write_register"
                      f"({address}={value}) loi: {e}")
                self.connected = False
                return False

    def write_registers(self,
                        address: int,
                        values:  list) -> bool:
        """Ghi nhieu holding register lien tiep"""
        with self._lock:
            try:
                vals = [max(0, min(65535, int(v)))
                        for v in values]
                result = self._client\
                    .write_registers(
                        address = address,
                        values  = vals,
                        slave   = self.unit_id)
                return not result.isError()
            except Exception as e:
                print(f"[PLC] write_registers loi: {e}")
                self.connected = False
                return False

    # ════════════════════════════════════════════════
    # HELPER: Ghi ket qua ve PLC
    # ════════════════════════════════════════════════
    def send_result(self,
                    frame_count: int,
                    ok_count:    int,
                    ng_count:    int,
                    proc_ms:     int,
                    status:      int = STATUS_DONE):
        if ok_count > 0 and ng_count == 0:
            verdict = 1   # OK
        elif ng_count > 0:
            verdict = 2   # NG
        else:
            verdict = 0   # IDLE
        values = [
            frame_count,   # D2000
            0,             # D2001 (padding)
            ok_count,      # D2002
            0,             # D2003 (padding)
            ng_count,      # D2004
            0,             # D2005 (padding)
            proc_ms,       # D2006
            0,             # D2007 (padding)
            status,        # D2008
            0,             # D2009
            verdict,       # D2010
        ]
        ok = self.write_registers(
            REG_FRAME_COUNT, values)
        if ok:
            v_str = ('OK' if verdict == 1
                 else 'NG' if verdict == 2
                 else 'IDLE')
            print(f"[PLC] Send: "
                f"frames={frame_count} "
                f"OK={ok_count} NG={ng_count} "
                f"t={proc_ms}ms "
                f"status={status} "
                f"verdict={v_str}")
        return ok

    def set_status(self, status: int):
        """Cap nhat trang thai"""
        return self.write_register(
            REG_STATUS, status)


# ════════════════════════════════════════════════════════════════
# PLCMonitor - Thread theo doi tín hiệu PLC
# ════════════════════════════════════════════════════════════════
class PLCMonitor(threading.Thread):
    """
    Thread lien tuc doc Coil tu PLC
    Goi callback khi co thay doi
    """

    POLL_INTERVAL = 0.05   # 50ms = 20Hz

    def __init__(self, plc_client: PLCClient,
                 on_camera_on:   callable = None,
                 on_camera_off:  callable = None,
                 on_trigger_on:  callable = None,
                 on_trigger_off: callable = None):
        super().__init__(daemon=True,
                         name="PLCMonitor")
        self.plc             = plc_client
        self.on_camera_on    = on_camera_on
        self.on_camera_off   = on_camera_off
        self.on_trigger_on   = on_trigger_on
        self.on_trigger_off  = on_trigger_off
        self.running         = False

        # Trang thai truoc do de detect thay doi
        self._prev_camera_on = False
        self._prev_trigger   = False

    def run(self):
        self.running = True
        print("[PLCMonitor] Bat dau theo doi PLC...")

        while self.running:
            try:
                if not self.plc.connected:
                    print("[PLCMonitor] Mat ket noi, "
                          "thu lai...")
                    self.plc.reconnect()
                    time.sleep(1.0)
                    continue

                # Doc 2 coil: M1000, M1001
                coils = self.plc.read_coils(
                    COIL_CAMERA_ON, 2)

                camera_on = coils[0]   # M1000
                trigger   = coils[1]   # M1001

                # ── Detect thay doi M1000 ─────────────
                if camera_on != self._prev_camera_on:
                    if camera_on:
                        print("[PLC] M1000=1 "
                              "→ BAT CAMERA")
                        if self.on_camera_on:
                            self.on_camera_on()
                    else:
                        print("[PLC] M1000=0 "
                              "→ TAT CAMERA")
                        if self.on_camera_off:
                            self.on_camera_off()
                    self._prev_camera_on = camera_on

                # ── Detect thay doi M1001 ─────────────
                if trigger != self._prev_trigger:
                    if trigger:
                        print("[PLC] M1001=1 "
                              "→ BAT DAU CHUP")
                        if self.on_trigger_on:
                            self.on_trigger_on()
                    else:
                        print("[PLC] M1001=0 "
                              "→ DUNG CHUP + XU LY")
                        if self.on_trigger_off:
                            self.on_trigger_off()
                    self._prev_trigger = trigger

            except Exception as e:
                print(f"[PLCMonitor] Exception: {e}")
                self.plc.connected = False
                time.sleep(1.0)

            time.sleep(self.POLL_INTERVAL)

        print("[PLCMonitor] Dung")

    def stop(self):
        self.running = False


# ════════════════════════════════════════════════════════════════
# TEST STANDALONE
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("MODBUS PLC TEST")
    print("=" * 50)

    plc = PLCClient(
        host    = "192.168.4.20",
        port    = 502,
        unit_id = 1)

    print("\n[1] Ket noi PLC...")
    if not plc.connect():
        print("Khong ket noi duoc!")
        exit(1)

    print("\n[2] Doc M1000, M1001...")
    coils = plc.read_coils(COIL_CAMERA_ON, 2)
    print(f"    M1000 (camera_on) = {coils[0]}")
    print(f"    M1001 (trigger)   = {coils[1]}")

    print("\n[3] Ghi test result ve PLC...")
    plc.send_result(
        frame_count = 10,
        ok_count    = 8,
        ng_count    = 2,
        proc_ms     = 1500,
        status      = STATUS_DONE)

    print("\n[4] Set status IDLE...")
    plc.set_status(STATUS_IDLE)

    print("\n[5] Monitor 10 giay...")
    def on_cam_on():
        print("  >>> CAMERA ON!")
    def on_cam_off():
        print("  >>> CAMERA OFF!")
    def on_trig_on():
        print("  >>> TRIGGER ON - BAT DAU CHUP!")
    def on_trig_off():
        print("  >>> TRIGGER OFF - DUNG CHUP!")

    monitor = PLCMonitor(
        plc,
        on_camera_on   = on_cam_on,
        on_camera_off  = on_cam_off,
        on_trigger_on  = on_trig_on,
        on_trigger_off = on_trig_off)
    monitor.start()

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass

    monitor.stop()
    plc.disconnect()
    print("DONE")