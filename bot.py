"""
BlueStacks Bot Engine
Dùng ADB để chụp màn hình + OpenCV để nhận dạng + ADB để thao tác.
"""

import subprocess
import threading
import contextlib
from bot_config import CONFIG, resolve
import time
import io
import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional

ADB = CONFIG["adb"]
TEMPLATES_DIR = str(resolve(CONFIG["templates_dir"]))


@dataclass
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2


class BlueStacksBot:
    def __init__(self, port: int = CONFIG["port"], adb: str = ADB):
        self.adb  = adb
        self.port = port
        self.device = CONFIG.get("device") or f"localhost:{port}"
        self.stop_event = threading.Event()
        self.last_reason = "ready"
        self._task_depth = threading.local()
        self._sc_tmp = os.path.join(os.path.dirname(__file__), "_sc.png")

    def check_stop(self):
        if self.stop_event.is_set():
            from verified_workflow import Stopped
            raise Stopped("stopped")

    @contextlib.contextmanager
    def task(self):
        # Named kernel mutex also excludes other CLI/GUI processes.
        import ctypes
        depth = getattr(self._task_depth, "value", 0)
        handle = None
        if depth == 0:
            kernel = ctypes.windll.kernel32
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            handle = kernel.CreateMutexW(None, False, "Local\\HayDayBot_" + self.device.replace(":", "_"))
            if not handle:
                raise RuntimeError("Cannot create device mutex")
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            if kernel.WaitForSingleObject(handle, 0) not in (0, 0x80):
                kernel.CloseHandle(handle)
                raise RuntimeError("BlueStacks session is busy")
        self._task_depth.value = depth + 1
        try:
            self.check_stop()
            yield
        finally:
            self._task_depth.value = depth
            if handle:
                kernel.ReleaseMutex(handle)
                kernel.CloseHandle(handle)

    # ── ADB helpers ───────────────────────────────────────────────────────────

    def _run(self, *args, check=True, timeout: int = CONFIG["timeout"]):
        self.check_stop()
        if args and args[0] == "shell" and getattr(self._task_depth, "value", 0) == 0:
            with self.task():
                return self._run(*args, check=check, timeout=timeout)
        cmd = [self.adb, "-s", self.device] + list(args)
        try:
            result = subprocess.run(cmd, capture_output=True, check=check, timeout=timeout)
            from verified_workflow import Workflow
            Workflow(self).log("adb", arguments=list(args), returncode=result.returncode)
            return result
        except subprocess.CalledProcessError as error:
            from verified_workflow import Workflow
            Workflow(self).log("adb", arguments=list(args), returncode=error.returncode,
                               error=(error.stderr or b"").decode("utf-8", errors="replace"))
            raise
        except subprocess.TimeoutExpired:
            print(f"[bot] ADB timeout ({timeout}s): {' '.join(str(a) for a in args[:3])}")
            raise RuntimeError(f"ADB timeout after {timeout}s for {self.device}: {args[0]}")

    def connect(self) -> bool:
        """Validate the configured executable, serial and Android screenshot."""
        self.adb = str(resolve(self.adb))
        try:
            if not os.path.isfile(self.adb):
                raise RuntimeError(f"ADB executable not found: {self.adb} (bot_config.json)")
            state = self._run("get-state", check=False)
            if state.stdout.strip() != b"device" and ":" in self.device:
                self.check_stop()
                subprocess.run([self.adb, "connect", self.device], capture_output=True,
                               timeout=CONFIG["timeout"])
                state = self._run("get-state", check=False)
            if state.stdout.strip() != b"device":
                detail = (state.stderr or state.stdout).decode(errors="replace").strip()
                raise RuntimeError(f"ADB device {self.device} is not connected/ready: {detail}")
            self.screenshot()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.last_reason = str(error)
            print(f"[bot] {self.last_reason}")
            return False
        self.last_reason = "ready"
        print(f"[bot] Connected to BlueStacks @ {self.device}; screenshot verified")
        return True

    def auto_connect(self) -> bool:
        """Thử các port phổ biến của BlueStacks."""
        for p in [5555, 5556, 5557, 5558, 5565, 5575]:
            self.port = p
            self.device = f"localhost:{p}"
            if self.connect():
                return True
        print("[bot] Không tìm thấy BlueStacks. Kiểm tra ADB trong cài đặt BlueStacks.")
        return False

    def get_screen_size(self) -> tuple[int, int]:
        """Lấy độ phân giải thực của emulator từ ADB thay vì hardcode."""
        import re
        r = self._run("shell", "wm", "size")
        out = r.stdout.decode("utf-8", errors="ignore")
        m = re.search(r"(\d+)x(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
        return 1600, 900

    def _find_hwnd(self):
        """Tìm handle cửa sổ BlueStacks/Hay Day (dùng chung cho win_tap và win_harvest_swipe)."""
        import ctypes, ctypes.wintypes
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        found = []
        def cb(h, _):
            if user32.IsWindowVisible(h):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(h, buf, 256)
                if any(k in buf.value.lower() for k in ("hay day", "hayday", "bluestack")):
                    found.append(h)
            return True
        user32.EnumWindows(WNDENUMPROC(cb), 0)
        return found[0] if found else None

    # ── Screenshot ────────────────────────────────────────────────────────────

    def screenshot(self) -> np.ndarray:
        """Chụp màn hình BlueStacks, trả về ảnh BGR numpy array."""
        try:
            data = self._run("exec-out", "screencap", "-p").stdout
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(f"ADB screenshot failed for {self.device}: {detail}") from error
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Invalid Android screenshot PNG ({len(data)} bytes)")
        try:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as error:
            raise RuntimeError("Invalid Android screenshot: PNG decode failed") from error
        if img is None:
            raise RuntimeError("Invalid Android screenshot: PNG decode failed")
        return img

    def save_screenshot(self, path: str) -> str:
        img = self.screenshot()
        cv2.imwrite(path, img)
        print(f"[bot] Screenshot lưu tại: {path}")
        return path

    def screenshot_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        full = self.screenshot()
        return full[y:y+h, x:x+w]

    # ── Template matching ─────────────────────────────────────────────────────

    def find(
        self,
        template_name: str,
        screen: Optional[np.ndarray] = None,
        threshold: float = 0.80,
        max_results: int = 50,
        scale_range: tuple = (0.9, 1.1),
    ) -> list[Match]:
        """
        Tìm template trong màn hình. template_name là tên file (không có .png)
        trong thư mục templates/.
        Hỗ trợ multi-scale để xử lý độ phân giải khác nhau.
        """
        tpl_path = os.path.join(TEMPLATES_DIR, f"{template_name}.png")
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"Template không tồn tại: {tpl_path}")

        if screen is None:
            screen = self.screenshot()

        tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
        if tpl is None:
            raise ValueError(f"Không đọc được template: {tpl_path}")

        matches: list[Match] = []
        th, tw = tpl.shape[:2]

        for scale in np.linspace(scale_range[0], scale_range[1], 5):
            resized = cv2.resize(tpl, (int(tw * scale), int(th * scale)))
            rh, rw = resized.shape[:2]
            if rh > screen.shape[0] or rw > screen.shape[1]:
                continue

            result = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(result >= threshold)
            for pt in zip(*locs[::-1]):
                score = float(result[pt[1], pt[0]])
                matches.append(Match(x=int(pt[0]), y=int(pt[1]), w=rw, h=rh, score=score))

        # Non-maximum suppression: loại bỏ duplicate gần nhau
        matches = _nms(matches, iou_thresh=0.3)
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:max_results]

    def find_color(
        self,
        rgb: tuple,
        screen: Optional[np.ndarray] = None,
        tolerance: int = 20,
    ) -> list[tuple[int, int]]:
        """Tìm tất cả pixel có màu gần với rgb."""
        if screen is None:
            screen = self.screenshot()
        r, g, b = rgb
        lo = np.array([max(0, b-tolerance), max(0, g-tolerance), max(0, r-tolerance)])
        hi = np.array([min(255, b+tolerance), min(255, g+tolerance), min(255, r+tolerance)])
        mask = cv2.inRange(screen, lo, hi)
        points = np.column_stack(np.where(mask > 0))
        return [(int(p[1]), int(p[0])) for p in points]

    # ── Actions ───────────────────────────────────────────────────────────────

    def tap(self, x: int, y: int, delay: float = 0.3):
        self._run("shell", "input", "tap", str(x), str(y))
        time.sleep(delay)

    def tap_match(self, m: Match, delay: float = 0.3):
        cx, cy = m.center
        self.tap(cx, cy, delay)

    def tap_all(self, matches: list[Match], delay: float = 0.4):
        for m in matches:
            self.tap_match(m, delay)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Swipe in Android screenshot coordinates."""
        r = self._run("shell", "input", "swipe",
                      str(x1), str(y1), str(x2), str(y2), str(duration_ms))
        if r.returncode != 0 or b"error" in r.stderr:
            raise RuntimeError("ADB swipe failed")
        else:
            time.sleep(0.2)

    def win_swipe_post(self, x1, y1, x2, y2, duration_ms=1200, **kwargs):
        return self.swipe(x1, y1, x2, y2, duration_ms)

    def long_press(self, x: int, y: int, duration_ms: int = 800):
        self.swipe(x, y, x, y, duration_ms)

    def win_harvest_swipe(self, x1, y1, x2, y2, hold_s=.5, drag_s=.9, **kwargs):
        # Use screenshot-space Android input; no desktop mouse button can remain held.
        self._run("shell", "input", "draganddrop", str(x1), str(y1), str(x2), str(y2), str(int(drag_s*1000)))

    def win_tap(self, x, y, **kwargs):
        return self.tap(x, y)

    def drag(self, x1, y1, x2, y2, duration_ms=700):
        self._run("shell", "input", "draganddrop", *map(str, (x1, y1, x2, y2, duration_ms)))

    def back(self):
        self._run("shell", "input", "keyevent", "4")
        time.sleep(0.3)

    def adb_shell(self, cmd: str):
        """Chạy lệnh shell ADB tuỳ ý, ví dụ input text/keyevent."""
        self._run("shell", *cmd.split())

    def input_text(self, text: str):
        """Nhập văn bản qua ADB (không dùng được ký tự đặc biệt)."""
        self._run("shell", "input", "text", str(text))

    def keyevent(self, code: int):
        """Gửi keyevent ADB. Ví dụ 67=BACKSPACE, 61=TAB, 66=ENTER."""
        self._run("shell", "input", "keyevent", str(code))

    def clear_and_type(self, text: str, field_x: int, field_y: int):
        """Tap vào field, xoá hết, rồi nhập text mới."""
        self.tap(field_x, field_y, delay=0.3)
        self._run("shell", "input", "keyevent", "123")  # KEYCODE_MOVE_END
        time.sleep(0.1)
        # Giữ Shift+Home để select all, rồi xoá
        self._run("shell", "input", "keyevent", "--longpress", "67")  # long BACKSPACE
        time.sleep(0.3)
        self.input_text(str(text))

    # ── Debug ─────────────────────────────────────────────────────────────────

    def show_matches(self, screen: np.ndarray, matches: list[Match], window: str = "matches"):
        vis = screen.copy()
        for m in matches:
            cv2.rectangle(vis, (m.x, m.y), (m.x+m.w, m.y+m.h), (0, 255, 0), 2)
            cx, cy = m.center
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{m.score:.2f}", (m.x, m.y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        cv2.imshow(window, cv2.resize(vis, (vis.shape[1]//2, vis.shape[0]//2)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_region_as_template(self, x: int, y: int, w: int, h: int, name: str):
        """Cắt vùng màn hình và lưu làm template mới."""
        region = self.screenshot_region(x, y, w, h)
        path = os.path.join(TEMPLATES_DIR, f"{name}.png")
        os.makedirs(TEMPLATES_DIR, exist_ok=True)
        cv2.imwrite(path, region)
        print(f"[bot] Template '{name}' lưu tại: {path}")
        return path


def _nms(matches: list[Match], iou_thresh: float = 0.3) -> list[Match]:
    """Non-maximum suppression: loại bỏ các match chồng lên nhau."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m.score, reverse=True)
    kept = []
    for m in matches:
        overlap = False
        for k in kept:
            if _iou(m, k) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return kept


def _iou(a: Match, b: Match) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x+a.w, a.y+a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x+b.w, b.y+b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union else 0
