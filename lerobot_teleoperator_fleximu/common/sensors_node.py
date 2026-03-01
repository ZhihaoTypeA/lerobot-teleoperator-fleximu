# IMU data handling node (threaded)

import socket
import struct
import time
import threading
from scipy.spatial.transform import Rotation as R

from .config import HOST, IMU_PORT, FLEX_PORT, IMU_MAPPING


class SensorsNode:
    def __init__(self, rcvbuf_bytes: int = 1 << 20):
        #IMU socket
        self.imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.imu_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
        self.imu_sock.bind((HOST, IMU_PORT))
        self.imu_sock.settimeout(0.2)

        #FLEX socket
        self.flex_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.flex_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.flex_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf_bytes)
        self.flex_sock.bind((HOST, FLEX_PORT))
        self.flex_sock.settimeout(0.2)

        self._lock = threading.Lock()

        self.current_euler = {}
        self.offsets = {}
        self.is_calibrated = False
        self.abs_rotations = {k: R.identity() for k in IMU_MAPPING.values()}

        self.flex_raw = 0.0
        self.flex_last_time = 0.0
        self.flex_min = 4095
        self.flex_max = 0
        self.flex_calibrated = False

        self._stop_evt = threading.Event()
        self._imu_thread = threading.Thread(target=self._imu_loop, daemon=True)
        self._flex_thread = threading.Thread(target=self._flex_loop, daemon=True)
        self._imu_thread.start()
        self._flex_thread.start()

    #Packet parsing
    @staticmethod
    def _parse_imu_packet(data: bytes):
        # WT55 packet format check
        if len(data) < 44:
            return None, None
        if data[0:4].decode("ascii", errors="ignore") != "WT55":
            return None, None

        try:
            dev_id = data[4:12].decode("ascii", errors="ignore")
            raw = struct.unpack("<hhh", data[38:44])  # roll/pitch/yaw
            euler_deg = [x / 32768.0 * 180.0 for x in raw]  # [x,y,z] in deg
            return dev_id, euler_deg
        except Exception:
            return None, None

    @staticmethod
    def _parse_flex_packet(data: bytes):
        if not data.startswith(b"FLEX:"):
            return None
        try:
            content = data.decode("ascii", errors="ignore").split(":", 1)[1].strip()
            return int(content)
        except Exception:
            return None

    #Loops
    def _imu_loop(self):
        while not self._stop_evt.is_set():
            try:
                data, _ = self.imu_sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            dev_id, payload = self._parse_imu_packet(data)
            if dev_id is None:
                continue
            if dev_id not in IMU_MAPPING:
                continue

            name = IMU_MAPPING[dev_id]
            r_raw = R.from_euler("ZYX", [payload[2], payload[1], payload[0]], degrees=True)

            with self._lock:
                if self.is_calibrated and name in self.offsets:
                    self.abs_rotations[name] = self.offsets[name].inv() * r_raw
                else:
                    self.current_euler[name] = r_raw

    def _flex_loop(self):
        while not self._stop_evt.is_set():
            try:
                data, _ = self.flex_sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            val = self._parse_flex_packet(data)
            if val is None:
                continue

            with self._lock:
                self.flex_raw = val
                self.flex_last_time = time.time()

    def close(self):
        self._stop_evt.set()
        try:
            self.imu_sock.close()
        except Exception:
            pass
        try:
            self.flex_sock.close()
        except Exception:
            pass

    #Calibration
    def calibrate(self, flex_duration=5.0):
        print("[Sensors] Calibrating IMU...")
        with self._lock:
            for name, r_raw in self.current_euler.items():
                self.offsets[name] = r_raw
        print("[Sensors] IMU Calibration Complete.")

        print("[Sensors] Calibrating FLEX sensor...")
        start_time = time.time()
        temp_min = 4095
        temp_max = 0

        while True:
            elapsed = time.time() - start_time
            remaining = flex_duration - elapsed
            if remaining <= 0:
                break

            with self._lock:
                val = self.flex_raw
                last_t = self.flex_last_time

            if last_t > 0 and (time.time() - last_t) < 0.2:
                if val < temp_min:
                    temp_min = val
                if val > temp_max:
                    temp_max = val

                print(
                    f"\rRemaining: {remaining:.1f}s | Current FLEX: {val} | Min: {temp_min} | Max: {temp_max}",
                    end="",
                )

            time.sleep(0.02)

        print("\n[Sensors] FLEX Calibration Complete.\n")
        with self._lock:
            self.flex_min = temp_min
            self.flex_max = temp_max
            self.is_calibrated = True
            self.flex_calibrated = True


    def get_flex_ratio(self):
        with self._lock:
            if not self.flex_calibrated:
                return 0.0
            val = float(self.flex_raw)
            fmin = float(self.flex_min)
            fmax = float(self.flex_max)

        val = max(fmin, min(fmax, val))
        if fmax == fmin:
            return 0.0
        return (val - fmin) / (fmax - fmin)
