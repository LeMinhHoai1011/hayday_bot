"""
auto_calibrate.py — Tự động phát hiện và cập nhật tọa độ UI trong Hay Day

Cách hoạt động:
  - Mỗi lần gọi, chụp màn hình và dùng màu + hình dạng để tìm element
  - Lưu kết quả vào positions.json
  - Lần sau load lại từ file nếu scan nhanh không cần thiết
  - Hỗ trợ 2 mode:
    * FARM mode   → tìm shop_stand, barn, silo trong màn hình farm
    * DIALOG mode → tìm wheat_icon, btn_>>, btn_sell trong dialog shop

Bắt đầu dùng:
  from auto_calibrate import Positions
  pos = Positions(bot)
  shop = pos.get("shop_stand")   # → (x, y), tự detect nếu chưa có
  pos.scan_farm()                # force quét lại farm
"""

import os
import json
import time
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass, asdict

_HERE          = os.path.dirname(os.path.abspath(__file__))
POSITIONS_FILE = os.path.join(_HERE, "positions.json")
TEMPLATES_DIR  = os.path.join(_HERE, "templates")


# ═══════════════════════════════════════════════════════════════════════════════
#  GIÁ TRỊ MẶC ĐỊNH (dự phòng khi auto-detect thất bại)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    # Farm elements
    "shop_stand":    (250,  635),
    "barn":          (1314, 442),
    "silo":          (1200, 160),
    # Shop dialog
    "shop_slot_1":   (155,  345),
    "wheat_icon":    (280,  295),
    "sugar_icon":    (490,  295),
    "corn_icon":     (680,  295),
    "btn_qty_plus":  (900,  220),
    "btn_qty_minus": (695,  220),
    "btn_price_max": (875,  415),
    "btn_price_min": (730,  415),
    "btn_put_sale":  (785,  760),
    # Upgrade dialog
    "btn_upgrade":   None,   # dynamic only
    "btn_confirm":   None,   # dynamic only
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR: Phát hiện element bằng màu + hình dạng
# ═══════════════════════════════════════════════════════════════════════════════

class ElementDetector:
    """Phát hiện các UI element bằng màu HSV và hình dạng."""

    # ── Màu HSV đặc trưng của từng element ───────────────────────────────────

    # Shop stand: mái đỏ sọc (hue đỏ)
    SHOP_RED_LO  = np.array([0,  150, 150])
    SHOP_RED_HI  = np.array([10, 255, 255])
    SHOP_RED2_LO = np.array([170,150, 150])
    SHOP_RED2_HI = np.array([180,255, 255])

    # Barn: vùng đỏ lớn (mái + thân nhà đỏ)
    BARN_LO  = np.array([0,  120, 100])
    BARN_HI  = np.array([12, 255, 255])
    BARN2_LO = np.array([168,120, 100])
    BARN2_HI = np.array([180,255, 255])

    # Silo: trụ đỏ hẹp + cao
    SILO_LO  = np.array([0,  130, 120])
    SILO_HI  = np.array([11, 255, 255])
    SILO2_LO = np.array([168,130, 120])
    SILO2_HI = np.array([180,255, 255])

    # Lúa mỳ (vàng đậm) trong inventory
    WHEAT_LO = np.array([18, 150, 160])
    WHEAT_HI = np.array([38, 255, 255])

    # Nút ">>" (cam-vàng)
    ARROW_LO = np.array([18, 150, 180])
    ARROW_HI = np.array([40, 255, 255])

    # Nút "Đưa ra bán" active (vàng-gold sáng)
    SALE_LO  = np.array([32, 180, 200])
    SALE_HI  = np.array([52, 255, 255])

    # Nút upgrade (xanh lá)
    UPG_LO   = np.array([38,  90, 130])
    UPG_HI   = np.array([88, 255, 255])

    # Ô shop trống (gỗ nâu-vàng)
    SLOT_LO  = np.array([18,  40, 140])
    SLOT_HI  = np.array([38, 130, 220])

    def __init__(self, sc: np.ndarray):
        self.sc = sc
        self.h, self.w = sc.shape[:2]

    def _mask(self, lo, hi, lo2=None, hi2=None):
        hsv = cv2.cvtColor(self.sc, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, lo, hi)
        if lo2 is not None:
            m |= cv2.inRange(hsv, lo2, hi2)
        return m

    def _mask_roi(self, lo, hi, lo2, hi2, x0f, y0f, x1f, y1f):
        x0, y0 = int(self.w*x0f), int(self.h*y0f)
        x1, y1 = int(self.w*x1f), int(self.h*y1f)
        roi = self.sc[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, lo, hi)
        if lo2 is not None:
            m |= cv2.inRange(hsv, lo2, hi2)
        return m, x0, y0

    def _contours_in_roi(self, mask, x0, y0, min_area, max_area=None):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            if max_area and a > max_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"]) + x0
            cy = int(M["m01"]/M["m00"]) + y0
            x, y, bw, bh = cv2.boundingRect(c)
            result.append({
                "cx": cx, "cy": cy,
                "area": a,
                "bbox": (x+x0, y+y0, bw, bh),
                "aspect": bw/bh if bh > 0 else 0,
            })
        result.sort(key=lambda r: r["area"], reverse=True)
        return result

    # ── Farm elements ────────────────────────────────────────────────────────

    def detect_shop_stand(self) -> Optional[tuple]:
        """
        Tìm cửa hàng bên đường (mái đỏ-trắng sọc).
        Vị trí: góc trái-dưới của farm (x:0-35%, y:45-95%).
        Đặc điểm: vùng đỏ nhỏ-vừa (1000-25000px²), nằm trong góc trái dưới.
        """
        mask, x0, y0 = self._mask_roi(
            self.SHOP_RED_LO, self.SHOP_RED_HI,
            self.SHOP_RED2_LO, self.SHOP_RED2_HI,
            0.0, 0.45, 0.35, 0.95
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=800, max_area=20000)
        if items:
            # Lấy vùng đỏ nằm thấp nhất (y lớn nhất) trong góc trái
            # Shop stand thường ở thấp hơn so với các vùng đỏ khác (cây lá)
            best = max(items, key=lambda r: r["cy"])
            return (best["cx"], best["cy"])
        return None

    def detect_barn(self) -> Optional[tuple]:
        """
        Tìm Barn (kho đỏ, nhà đỏ lớn).
        Vị trí: nửa phải màn hình, giữa-dưới (x:55-100%, y:30-65%).
        Đặc điểm: vùng đỏ lớn (>8000px²), tỉ lệ chiều rộng/cao ≈ 1-2.
        """
        mask, x0, y0 = self._mask_roi(
            self.BARN_LO, self.BARN_HI,
            self.BARN2_LO, self.BARN2_HI,
            0.55, 0.30, 1.0, 0.65
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=6000)
        if not items:
            # Thử vùng rộng hơn
            mask, x0, y0 = self._mask_roi(
                self.BARN_LO, self.BARN_HI,
                self.BARN2_LO, self.BARN2_HI,
                0.50, 0.25, 1.0, 0.75
            )
            items = self._contours_in_roi(mask, x0, y0, min_area=6000)
        # Lọc theo tỉ lệ (barn rộng hơn cao hoặc gần vuông)
        candidates = [r for r in items if 0.6 < r["aspect"] < 3.0]
        if candidates:
            # Barn thường ở bên phải nhất
            return max(candidates, key=lambda r: r["cx"])["cx"], \
                   max(candidates, key=lambda r: r["cx"])["cy"]
        return None

    def detect_silo(self) -> Optional[tuple]:
        """
        Tìm Silo (trụ đỏ cao).
        Vị trí: góc phải trên (x:60-95%, y:5-40%).
        Đặc điểm: hình trụ → aspect ratio < 0.6 (cao hơn rộng).
        """
        mask, x0, y0 = self._mask_roi(
            self.SILO_LO, self.SILO_HI,
            self.SILO2_LO, self.SILO2_HI,
            0.60, 0.05, 0.95, 0.40
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=1500)
        # Lọc: silo cao hơn rộng (aspect < 0.8)
        candidates = [r for r in items if r["aspect"] < 0.9]
        if candidates:
            best = candidates[0]
            return (best["cx"], best["cy"])
        return None

    # ── Dialog elements ──────────────────────────────────────────────────────

    def detect_wheat_in_inventory(self) -> Optional[tuple]:
        """
        Tìm icon lúa mỳ (vàng đậm) trong inventory panel (trái dialog).
        Chỉ tìm trong x:3-45%, y:15-70%.
        """
        mask, x0, y0 = self._mask_roi(
            self.WHEAT_LO, self.WHEAT_HI,
            None, None,
            0.03, 0.15, 0.45, 0.70
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=300, max_area=20000)
        if items:
            return (items[0]["cx"], items[0]["cy"])
        return None

    def detect_btn_price_max(self) -> Optional[tuple]:
        """
        Tìm nút ">>" (cam-vàng, nửa phải, giữa dialog).
        Tìm trong x:50-95%, y:35-60%.
        """
        mask, x0, y0 = self._mask_roi(
            self.ARROW_LO, self.ARROW_HI,
            None, None,
            0.50, 0.35, 0.95, 0.60
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=1000, max_area=15000)
        if items:
            # Lấy nút nằm ở bên phải hơn (">>" ở phải hơn "<<")
            return max(items, key=lambda r: r["cx"])["cx"], \
                   max(items, key=lambda r: r["cx"])["cy"]
        return None

    def detect_btn_put_sale(self) -> Optional[tuple]:
        """
        Tìm nút 'Đưa ra bán' (vàng-gold to, nằm cuối dialog).
        Tìm trong x:25-85%, y:72-95%.
        """
        mask, x0, y0 = self._mask_roi(
            self.SALE_LO, self.SALE_HI,
            None, None,
            0.25, 0.72, 0.85, 0.95
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=4000)
        if items:
            return (items[0]["cx"], items[0]["cy"])
        return None

    def detect_btn_upgrade(self) -> Optional[tuple]:
        """
        Tìm nút nâng cấp (xanh lá) trong dialog barn/silo.
        Tìm toàn màn hình.
        """
        mask, x0, y0 = self._mask_roi(
            self.UPG_LO, self.UPG_HI,
            None, None,
            0.0, 0.1, 1.0, 1.0
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=1200)
        if not items:
            return None
        # Lấy vùng xanh lớn nhất
        return (items[0]["cx"], items[0]["cy"])

    def detect_btn_confirm(self) -> Optional[tuple]:
        """Tìm nút xác nhận (vàng-cam lớn ở giữa-dưới)."""
        CONF_LO = np.array([15, 140, 180])
        CONF_HI = np.array([38, 255, 255])
        mask, x0, y0 = self._mask_roi(
            CONF_LO, CONF_HI,
            None, None,
            0.20, 0.40, 0.80, 0.90
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=4000)
        if items:
            return (items[0]["cx"], items[0]["cy"])
        return None

    def detect_empty_shop_slot(self) -> Optional[tuple]:
        """
        Tìm ô trống đầu tiên trong shop (màu gỗ nâu-vàng).
        """
        mask, x0, y0 = self._mask_roi(
            self.SLOT_LO, self.SLOT_HI,
            None, None,
            0.03, 0.25, 0.73, 0.65
        )
        items = self._contours_in_roi(mask, x0, y0, min_area=2500, max_area=60000)
        # Lọc theo tỉ lệ (ô gần vuông hoặc ngang)
        candidates = [r for r in items if 0.3 < r["aspect"] < 3.0]
        if candidates:
            # Lấy ô trái nhất (trên-trái)
            return min(candidates, key=lambda r: r["cx"] + r["cy"])["cx"], \
                   min(candidates, key=lambda r: r["cx"] + r["cy"])["cy"]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITIONS: Quản lý tọa độ động, lưu/đọc JSON
# ═══════════════════════════════════════════════════════════════════════════════

class Positions:
    """
    Quản lý tọa độ UI tự động.

    Dùng:
        pos = Positions(bot)
        shop = pos.get("shop_stand")   # → (x, y), auto-detect nếu cần
        pos.scan_farm()                # force quét lại farm
        pos.scan_dialog()              # quét khi dialog shop đang mở
    """

    SCAN_COOLDOWN = 30   # giây: không quét lại trong 30s trừ khi force

    def __init__(self, bot):
        self.bot = bot
        self._data = {}       # cache tọa độ hiện tại
        self._timestamps = {} # lần cuối detect thành công
        self._load_file()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_file(self):
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k == "_timestamps":
                        continue
                    if v is not None:
                        self._data[k] = tuple(v)
                ts = saved.get("_timestamps", {})
                for k, v in ts.items():
                    self._timestamps[k] = v
                print(f"[pos] Loaded {len(self._data)} positions from file")
            except Exception as e:
                print(f"[pos] Load error: {e}")

    def _save_file(self):
        try:
            out = {k: list(v) for k, v in self._data.items() if v is not None}
            out["_timestamps"] = self._timestamps
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[pos] Save error: {e}")

    # ── Core get/set ──────────────────────────────────────────────────────────

    def set(self, key: str, pos: Optional[tuple], save: bool = True):
        """Cập nhật vị trí, lưu file nếu save=True."""
        if pos is not None:
            self._data[key] = pos
            self._timestamps[key] = time.time()
            if save:
                self._save_file()

    def get(self, key: str, auto_scan: bool = True) -> tuple:
        """
        Lấy tọa độ. Nếu chưa có hoặc quá cũ → tự động scan.
        Trả về default nếu scan thất bại.
        """
        if auto_scan:
            age = time.time() - self._timestamps.get(key, 0)
            if key not in self._data or age > self.SCAN_COOLDOWN * 10:
                self._auto_scan_key(key)

        pos = self._data.get(key)
        if pos is not None:
            return pos

        default = DEFAULTS.get(key)
        if default:
            print(f"[pos] '{key}' dùng default {default}")
            return default
        raise KeyError(f"Không có tọa độ cho '{key}'")

    def _auto_scan_key(self, key: str):
        """Scan key cụ thể nếu thuộc farm hoặc dialog."""
        farm_keys = {"shop_stand", "barn", "silo"}
        dialog_keys = {"wheat_icon", "btn_price_max", "btn_put_sale",
                       "btn_upgrade", "btn_confirm", "shop_slot_1"}
        if key in farm_keys:
            self.scan_farm(force=False)
        elif key in dialog_keys:
            self.scan_dialog(force=False)

    # ── Scanner methods ───────────────────────────────────────────────────────

    def scan_farm(self, force: bool = True) -> dict:
        """
        Quét màn hình farm để tìm shop_stand, barn, silo.
        Chỉ chạy nếu force=True hoặc đã quá cooldown.
        """
        last = max(
            (self._timestamps.get(k, 0) for k in ("shop_stand","barn","silo")),
            default=0
        )
        if not force and (time.time() - last) < self.SCAN_COOLDOWN:
            return {}

        print("[pos] Quét farm...")
        sc = self.bot.screenshot()
        det = ElementDetector(sc)
        found = {}

        shop = det.detect_shop_stand()
        barn = det.detect_barn()
        silo = det.detect_silo()

        for key, val in [("shop_stand", shop), ("barn", barn), ("silo", silo)]:
            if val:
                old = self._data.get(key)
                self.set(key, val, save=False)
                if old and old != val:
                    print(f"[pos] {key}: {old} → {val}")
                else:
                    print(f"[pos] {key}: {val}")
                found[key] = val
            else:
                print(f"[pos] {key}: không phát hiện được, giữ {'(' + str(self._data.get(key)) + ')' if key in self._data else 'default'}")

        self._save_file()
        return found

    def scan_dialog(self, force: bool = True) -> dict:
        """
        Quét dialog đang mở để tìm wheat_icon, btn_price_max, btn_put_sale.
        Gọi khi shop dialog đang hiển thị.
        """
        last = max(
            (self._timestamps.get(k, 0) for k in ("wheat_icon","btn_price_max","btn_put_sale")),
            default=0
        )
        if not force and (time.time() - last) < self.SCAN_COOLDOWN:
            return {}

        print("[pos] Quét dialog shop...")
        sc = self.bot.screenshot()
        det = ElementDetector(sc)
        found = {}

        wheat    = det.detect_wheat_in_inventory()
        price_max = det.detect_btn_price_max()
        put_sale  = det.detect_btn_put_sale()
        slot     = det.detect_empty_shop_slot()

        mapping = {
            "wheat_icon":    wheat,
            "btn_price_max": price_max,
            "btn_put_sale":  put_sale,
            "shop_slot_1":   slot,
        }
        for key, val in mapping.items():
            if val:
                old = self._data.get(key)
                self.set(key, val, save=False)
                if old and old != val:
                    print(f"[pos] {key}: {old} → {val}")
                else:
                    print(f"[pos] {key}: {val}")
                found[key] = val
            else:
                print(f"[pos] {key}: không phát hiện, dùng {'(' + str(self._data.get(key)) + ')' if key in self._data else 'default'}")

        self._save_file()
        return found

    def scan_upgrade_dialog(self) -> dict:
        """Quét dialog upgrade để tìm nút nâng cấp."""
        sc = self.bot.screenshot()
        det = ElementDetector(sc)
        found = {}
        btn = det.detect_btn_upgrade()
        confirm = det.detect_btn_confirm()
        for key, val in [("btn_upgrade", btn), ("btn_confirm", confirm)]:
            if val:
                self.set(key, val, save=False)
                found[key] = val
                print(f"[pos] {key}: {val}")
        self._save_file()
        return found

    def scan_all(self) -> dict:
        """Quét tất cả: farm + dialog (nếu dialog đang mở)."""
        farm = self.scan_farm(force=True)
        return farm

    def print_all(self):
        """In ra tất cả tọa độ hiện tại."""
        print("\n[pos] ── Tọa độ hiện tại ──────────────────────────")
        for key, default in DEFAULTS.items():
            current = self._data.get(key, default)
            age = time.time() - self._timestamps.get(key, 0)
            age_str = f"{int(age)}s" if key in self._timestamps else "default"
            print(f"  {key:<18} = {str(current):<18} [{age_str}]")
        print()

    def draw_debug(self, sc: np.ndarray = None, label: str = "positions") -> str:
        """
        Vẽ tất cả tọa độ lên ảnh, lưu ra eco_positions_<label>.png.
        Màu xanh = từ auto-detect, đỏ = default.
        """
        if sc is None:
            sc = self.bot.screenshot()
        vis = sc.copy()

        farm_keys   = ["shop_stand", "barn", "silo"]
        dialog_keys = ["wheat_icon", "btn_price_max", "btn_put_sale",
                       "shop_slot_1", "btn_qty_plus", "btn_qty_minus"]

        for key in farm_keys + dialog_keys:
            pos = self._data.get(key) or DEFAULTS.get(key)
            if pos is None:
                continue
            detected = key in self._data
            color = (0, 200, 0) if detected else (0, 0, 200)
            x, y = pos
            cv2.circle(vis, (x, y), 20, color, -1)
            cv2.circle(vis, (x, y), 20, (255, 255, 255), 2)
            cv2.putText(vis, key[:14], (x + 25, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Legend
        cv2.putText(vis, "XANH=detected   DO=default",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        from bot_config import CONFIG, resolve
        directory = resolve(CONFIG["debug_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / f"eco_positions_{label}.png")
        cv2.imwrite(path, vis)
        print(f"[pos] Debug image: {path}")
        return path


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI — chạy trực tiếp để kiểm tra
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\tools\hayday_bot")
    from bot import BlueStacksBot

    b = BlueStacksBot(port=5555)
    b.connect()

    pos = Positions(b)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"

    if cmd == "scan":
        pos.scan_farm(force=True)
        pos.print_all()
        path = pos.draw_debug(label="farm")
        import subprocess
        subprocess.Popen(["explorer", path])

    elif cmd == "dialog":
        print("Giả định shop dialog đang mở...")
        pos.scan_dialog(force=True)
        pos.print_all()
        path = pos.draw_debug(label="dialog")
        import subprocess
        subprocess.Popen(["explorer", path])

    elif cmd == "all":
        pos.print_all()

    elif cmd == "clear":
        if os.path.exists(POSITIONS_FILE):
            os.remove(POSITIONS_FILE)
            print(f"Đã xóa {POSITIONS_FILE}")
