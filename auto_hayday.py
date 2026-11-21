"""
Hay Day Auto Bot — thao tác theo ảnh mẫu đã xác nhận.

Cách dùng:
    python auto_hayday.py           # chạy menu
    python auto_hayday.py loop      # chạy loop liên tục
    python auto_hayday.py debug     # chụp + hiển thị vùng phát hiện
"""

import sys
import time
import cv2
import numpy as np
from bot import BlueStacksBot, ADB
from bot_config import CONFIG
from verified_workflow import (FarmWorkflow, HARVEST_TEMPLATES, PLANT_TEMPLATES,
                               EvidenceError, WORKFLOW_ERRORS)
from auto_calibrate import Positions, ElementDetector

# ── Màu HSV đo thực tế từ game ────────────────────────────────────────────────
# Lúa mì / ngô / cây CHÍN (vàng óng — đo thực tế: S≥230, V≥168)
RIPE_LO  = np.array([20, 225, 168])
RIPE_HI  = np.array([42, 255, 255])

# Ruộng TRỐNG (đất nâu tối)
EMPTY_LO = np.array([ 8,  55,  50])
EMPTY_HI = np.array([22, 200, 170])

# Cây đang MỌC (xanh lá nhạt)
GROWING_LO = np.array([35,  40,  60])
GROWING_HI = np.array([90, 230, 210])

# Vùng menu chọn cây (xuất hiện phía dưới sau khi tap ruộng trống)
# Wheat là item đầu tiên bên trái trong menu
MENU_Y_FRAC  = 0.78   # menu xuất hiện ở ~78% chiều cao màn hình
WHEAT_X_FRAC = 0.30   # nút lúa mì ở khoảng 30% chiều ngang

MIN_FIELD_AREA = 5000  # px²: lọc noise, chỉ lấy vùng đủ lớn

# Vùng màn hình có thể có ruộng (tránh UI viền)
# fraction: (left, top, right, bottom) của kích thước màn hình
FARM_ROI = (0.04, 0.12, 0.96, 0.88)

# ── P3 Grid Sampling ──────────────────────────────────────────────────────────
SAMPLE_R = 12   # bán kính vùng sample (25×25px)

# HSV ngưỡng sample TẠI ĐIỂM trong lưới (looser vì đã biết vị trí trong đồng)
PT_RIPE_LO  = np.array([18, 160, 140])
PT_RIPE_HI  = np.array([46, 255, 255])
PT_EMPTY_LO = np.array([ 8, 170, 100])
PT_EMPTY_HI = np.array([17, 220, 220])
PT_GROW_LO  = np.array([32,  50,  70])
PT_GROW_HI  = np.array([90, 255, 220])

# Phát hiện biên cánh đồng (loose, chỉ dùng để tìm bbox tổng thể)
FIELD_DETECT_LO = np.array([15, 100, 100])
FIELD_DETECT_HI = np.array([50, 255, 255])
FIELD_MIN_AREA  = 15000  # px² tối thiểu để là cánh đồng (loại bỏ cây nhỏ)


class HayDayAutoBot(BlueStacksBot):

    def __init__(self, port: int = __import__("bot_config").CONFIG["port"], adb: str = ADB):
        super().__init__(port, adb)
        self._pos = None

    def _get_pos(self) -> Positions:
        if self._pos is None:
            self._pos = Positions(self)
        return self._pos

    # ── P3: Grid-based field detection ───────────────────────────────────────

    def _find_field_bbox(self, sc: np.ndarray) -> tuple | None:
        """
        Trả về (x, y, w, h) bbox cánh đồng lúa.
        Ưu tiên cache từ positions.json['field_bbox'].
        Nếu chưa có: thử auto-detect, nếu không chắc thì yêu cầu calibrate.
        """
        pos = self._get_pos()
        cached = pos._data.get("field_bbox")
        if cached:
            return tuple(cached)
        # Thử auto-detect
        result = self._detect_field_bbox(sc)
        if result:
            return result
        # Fallback: hỏi user calibrate
        print("  [grid] Cần calibrate vùng cánh đồng.")
        print("  [grid] Chạy menu 9 hoặc gọi bot.calibrate_field_region()")
        return None

    def _detect_field_bbox(self, sc: np.ndarray) -> tuple | None:
        """
        Phát hiện bbox cánh đồng lúa chính:
        - Dùng RIPE_LO/HI (tight) để lọc cây mùa thu (V<168)
        - Lấy contour LỚN NHẤT + fill ratio > 0.40 = cánh đồng lúa
        (cánh đồng là vùng vàng DẶC và LỚN nhất trên map)
        """
        h, w = sc.shape[:2]
        x0 = int(w * FARM_ROI[0]); y0 = int(h * FARM_ROI[1])
        x1 = int(w * FARM_ROI[2]); y1 = int(h * FARM_ROI[3])
        roi = sc[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Tight threshold: lọc cây (V<168), giữ lại lúa + cỏ vàng
        mask = cv2.inRange(hsv, RIPE_LO, RIPE_HI)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < FIELD_MIN_AREA:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            aspect = rw / rh if rh > 0 else 99
            if aspect > 7 or aspect < 0.3:
                continue
            # Fill ratio: cánh đồng lúa là vùng đặc, cỏ rải rác thưa hơn
            fill = area / (rw * rh) if rw * rh > 0 else 0
            if fill < 0.40:
                continue
            candidates.append((area, fill, (rx + x0, ry + y0, rw, rh)))
        if not candidates:
            print("  [grid] Không tìm thấy cánh đồng lúa.")
            return None
        # Lấy vùng lớn nhất (cánh đồng lúa >> cỏ/đất)
        best_area, best_fill, best = max(candidates, key=lambda t: t[0])
        pos = self._get_pos()
        pos.set("field_bbox", best)
        print(f"  [grid] Phát hiện cánh đồng: bbox={best}, area={best_area:.0f}, fill={best_fill:.2f}")
        return best

    def calibrate_field_region(self):
        """
        Cho user kéo chọn vùng cánh đồng lúa trên screenshot.
        Lưu bbox vào positions.json['field_bbox'].
        Chạy 1 lần, dùng mãi đến khi farm thay đổi (menu 8 để reset).
        """
        sc = self.screenshot()
        half = cv2.resize(sc, (sc.shape[1] // 2, sc.shape[0] // 2))
        print("  Kéo chọn vùng CÁnh đồng lúa (hình chữ nhật vàng), Enter xác nhận, C hủy.")
        roi = cv2.selectROI("Kéo chọn CÁNH ĐỒNG LÚA → Enter", half, fromCenter=False)
        cv2.destroyAllWindows()
        if roi[2] == 0 or roi[3] == 0:
            print("  Hủy calibrate.")
            return
        x, y, rw, rh = [v * 2 for v in roi]   # scale lại x2 (vì hiển thị ½)
        bbox = (x, y, rw, rh)
        pos = self._get_pos()
        with self.task():
            farm = FarmWorkflow(self)
            farm.v.point('farm_screen', sc)
            fresh = farm.stable()
            farm.zones = [(x, y, x+rw, y+rh)]
            farm.reference = sc
            farm.verify_frame(fresh)
            farm.save_calibration(sc, farm.zones[0])
        print(f"Đã lưu vùng ruộng {bbox} ở ảnh Android {sc.shape[1]}x{sc.shape[0]}.")

    def reset_field_bbox(self):
        path = FarmWorkflow(self).calibration_path
        path.unlink(missing_ok=True)
        print("Đã xóa vùng hiệu chỉnh. Dùng menu 9 để chọn lại.")

    def _field_grid_points(self, sc: np.ndarray) -> list[tuple[int, int]]:
        """
        Tạo lưới điểm mẫu bên trong cánh đồng.
        Mỗi điểm = tâm một ô ruộng ước lượng (~110×65px tại 1600×900).
        """
        bbox = self._find_field_bbox(sc)
        if not bbox:
            return []
        rx, ry, rw, rh = bbox
        _, w = sc.shape[:2]
        # Scale ô ruộng theo độ phân giải — 70×45 cho 1600×900 (~5×4=20 điểm)
        scale   = w / 1600.0
        plot_w  = max(30, int(70 * scale))
        plot_h  = max(20, int(45 * scale))
        cols    = max(1, rw // plot_w)
        rows    = max(1, rh // plot_h)
        cell_w  = rw / cols
        cell_h  = rh / rows
        points  = []
        for r in range(rows):
            for c in range(cols):
                # Isometric correction: mỗi hàng lùi trái ~35% chiều ngang ô
                iso_dx = -int(r * cell_w * 0.35)
                px = rx + int((c + 0.5) * cell_w) + iso_dx
                py = ry + int((r + 0.5) * cell_h)
                # Clamp trong bbox
                px = max(rx + SAMPLE_R, min(rx + rw - SAMPLE_R, px))
                py = max(ry + SAMPLE_R, min(ry + rh - SAMPLE_R, py))
                points.append((px, py))
        return points

    def _sample_plot(self, sc: np.ndarray, x: int, y: int) -> str:
        """
        Sample 25×25px tại (x,y) → phân loại 'ripe'/'empty'/'growing'/'unknown'.
        Dùng ngưỡng looser (PT_*) vì chỉ sample trong vùng đã biết là cánh đồng.
        """
        h, w = sc.shape[:2]
        x0, y0 = max(0, x - SAMPLE_R), max(0, y - SAMPLE_R)
        x1, y1 = min(w, x + SAMPLE_R), min(h, y + SAMPLE_R)
        patch = sc[y0:y1, x0:x1]
        if patch.size == 0:
            return "unknown"
        hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        ripe_n    = int(cv2.inRange(hsv, PT_RIPE_LO,  PT_RIPE_HI).sum())  // 255
        growing_n = int(cv2.inRange(hsv, PT_GROW_LO,  PT_GROW_HI).sum())  // 255
        # Nếu không phải ripe/growing → empty (mọi điểm trong bbox đều là plot)
        if ripe_n >= total * 0.25:   return "ripe"
        if growing_n >= total * 0.25: return "growing"
        empty_n = cv2.countNonZero(cv2.inRange(hsv, PT_EMPTY_LO, PT_EMPTY_HI))
        return "empty" if empty_n >= total * .5 else "unknown"

    def _adapt_hsv(self, sc: np.ndarray):
        """Trả về (ripe_lo, ripe_hi, empty_lo, empty_hi) phù hợp brightness màn hình."""
        gray = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 85:
            return (
                np.array([12, 110, 110]), np.array([40, 255, 220]),
                np.array([ 6,  40,  30]), np.array([20, 180, 130]),
            )
        return RIPE_LO, RIPE_HI, EMPTY_LO, EMPTY_HI

    def _menu_visible(self, sc: np.ndarray) -> bool:
        """True nếu menu chọn hạt giống đang hiển thị (vùng dưới có màu kem/vàng nhạt)."""
        h, w = sc.shape[:2]
        roi = sc[int(h * 0.65):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 25, 155]), np.array([45, 130, 255]))
        return int(mask.sum()) > 40000

    # ── Phát hiện vùng theo màu ───────────────────────────────────────────────

    def _detect_regions(
        self, sc: np.ndarray, lo: np.ndarray, hi: np.ndarray,
        min_area: int = MIN_FIELD_AREA, label: str = ""
    ) -> list[dict]:
        """
        Tìm các vùng liên thông trong mask màu.
        Chỉ tìm trong vùng FARM_ROI để tránh nhiễu từ UI.
        """
        h, w = sc.shape[:2]
        x0 = int(w * FARM_ROI[0]); y0 = int(h * FARM_ROI[1])
        x1 = int(w * FARM_ROI[2]); y1 = int(h * FARM_ROI[3])

        roi_sc = sc[y0:y1, x0:x1]
        hsv    = cv2.cvtColor(roi_sc, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv, lo, hi)

        # Làm sạch noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            rx, ry, rw, rh = cv2.boundingRect(c)
            # Lọc: tỉ lệ khung không quá dài/mỏng (tránh đường viền, thanh HP...)
            aspect = rw / rh if rh > 0 else 99
            if aspect > 8 or aspect < 0.12:
                continue
            # Lọc: ruộng lúa là hình chữ nhật convex; tán cây mùa thu không đều
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) < 0.50:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            # Chuyển tọa độ từ ROI về màn hình đầy đủ
            cx = int(M["m10"] / M["m00"]) + x0
            cy = int(M["m01"] / M["m00"]) + y0
            regions.append({"center": (cx, cy), "area": area,
                             "bbox": (rx+x0, ry+y0, rw, rh), "label": label})
        return regions

    def find_ripe_fields(self, sc=None) -> list[dict]:
        """P3: sample lưới điểm trong bbox cánh đồng, trả về điểm CHÍN."""
        if sc is None: sc = self.screenshot()
        pts = self._field_grid_points(sc)
        if not pts:
            rlo, rhi, _, _ = self._adapt_hsv(sc)
            return self._detect_regions(sc, rlo, rhi, label="CHÍN")
        r = SAMPLE_R
        return [{"center": (x, y), "area": r*r*4,
                 "bbox": (x-r, y-r, r*2, r*2), "label": "CHÍN"}
                for x, y in pts if self._sample_plot(sc, x, y) == "ripe"]

    def find_empty_fields(self, sc=None) -> list[dict]:
        """P3: sample lưới điểm trong bbox cánh đồng, trả về điểm TRỐNG."""
        if sc is None: sc = self.screenshot()
        pts = self._field_grid_points(sc)
        if not pts:
            _, _, elo, ehi = self._adapt_hsv(sc)
            return self._detect_regions(sc, elo, ehi, min_area=1500, label="TRỐNG")
        r = SAMPLE_R
        return [{"center": (x, y), "area": r*r*4,
                 "bbox": (x-r, y-r, r*2, r*2), "label": "TRỐNG"}
                for x, y in pts if self._sample_plot(sc, x, y) == "empty"]

    def find_growing_fields(self, sc=None) -> list[dict]:
        """P3: sample lưới điểm trong bbox cánh đồng, trả về điểm ĐANG MỌC."""
        if sc is None: sc = self.screenshot()
        pts = self._field_grid_points(sc)
        if not pts:
            return self._detect_regions(sc, GROWING_LO, GROWING_HI, label="ĐANG MỌC")
        r = SAMPLE_R
        return [{"center": (x, y), "area": r*r*4,
                 "bbox": (x-r, y-r, r*2, r*2), "label": "ĐANG MỌC"}
                for x, y in pts if self._sample_plot(sc, x, y) == "growing"]

    # ── Hành động ─────────────────────────────────────────────────────────────

    # ── Camera scroll ─────────────────────────────────────────────────────────

    def scroll_view(self, dy: int = 150, duration_ms: int = 600):
        """
        Cuộn farm view: dy>0 kéo tay từ dưới lên trên (camera dịch lên, thấy phần trên farm).
        dy<0 kéo ngược lại.
        """
        h_px = 900; w_px = 1600
        cx = w_px // 2
        y_mid = h_px // 2
        y_start = y_mid + abs(dy) // 2 if dy > 0 else y_mid - abs(dy) // 2
        y_end   = y_start - dy
        # Dùng swipe thường (không draganddrop) để pan camera, không trigger harvest
        self._run("shell", "input", "swipe",
                  str(cx), str(y_start), str(cx), str(y_end), str(duration_ms))
        time.sleep(0.4)

    def recenter_camera(self, up_times: int = 3):
        """
        Cuộn farm về phía trên để wheat field về vùng hiển thị.
        Gọi trước mỗi chu kỳ harvest để bù camera drift.
        """
        print("  [camera] Cuộn farm về trên...")
        for _ in range(up_times):
            self.scroll_view(dy=120, duration_ms=500)
        time.sleep(0.5)
        print("  [camera] Xong. Nhớ recalibrate bbox nếu cần (menu 9).")

    def harvest_ripe(self, sc=None):
        from verified_workflow import FarmWorkflow
        return FarmWorkflow(self).run()

    def plant_empty(self, sc=None, delay=.5):
        from verified_workflow import FarmWorkflow
        return FarmWorkflow(self).run(plant=True)

    # ── Debug view ────────────────────────────────────────────────────────────

    def debug_view(self, save_path="debug_detect.png"):
        farm = FarmWorkflow(self)
        try:
            vis, points = farm.preview(farm.stable())
            if not cv2.imwrite(save_path, vis):
                raise OSError(f"Cannot save {save_path}")
            print(f"{len(points)} ripe fields; next tap: {points[0][:2] if points else None}")
            return vis, points
        except WORKFLOW_ERRORS as error:
            farm.fail(error)
            return None

    def calibrate_wheat_button(self):
        from hayday import HayDayBot

        print("Mở menu gieo trong BlueStacks rồi cắt mẫu crop_menu_wheat.")
        HayDayBot.capture_template_interactive(self)

    def dismiss_popups(self) -> bool:
        """Đóng popup phổ biến (event, tin tức, level up). Trả về True nếu đã đóng."""
        sc = self.screenshot()
        h, w = sc.shape[:2]
        for tpl in ("close_btn", "x_btn", "ok_btn"):
            try:
                ms = self.find(tpl, sc, threshold=0.75)
                if ms:
                    self.tap_match(ms[0])
                    print(f"  [popup] Đóng popup bằng template '{tpl}'")
                    return True
            except FileNotFoundError:
                pass
        for fx, fy in ((0.92, 0.08), (0.88, 0.12)):
            x, y = int(w * fx), int(h * fy)
            px = sc[max(0, y-12):y+12, max(0, x-12):x+12]
            if px.size == 0:
                continue
            hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)
            red = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([12, 255, 255]))
            if int(red.sum()) > 400:
                self.tap(x, y)
                print(f"  [popup] Tap nút đỏ tại ({x},{y})")
                return True
        return False

    # ── Bán hàng trong shop ───────────────────────────────────────────────────

    def _shop_open(self, sc: np.ndarray) -> bool:
        """True nếu shop dialog đang mở (có ô gỗ hoặc nút bán)."""
        det = ElementDetector(sc)
        return det.detect_empty_shop_slot() is not None or det.detect_btn_put_sale() is not None

    def _inventory_open(self, sc: np.ndarray) -> bool:
        """True nếu dialog chọn vật phẩm đang mở (có wheat icon hoặc nút >>)."""
        det = ElementDetector(sc)
        return det.detect_wheat_in_inventory() is not None or det.detect_btn_price_max() is not None

    def sell_wheat(self, count=1):
        from sell_upgrade import ShopSeller
        return ShopSeller(self, self._get_pos()).run(slots=count)

    def run_loop(self, interval=120.0, max_cycles=0, limit=None):
        from sell_upgrade import ShopSeller

        farm = FarmWorkflow(self)
        cycle = 0
        reserve = CONFIG.get('wheat_seed_reserve', 0)
        try:
            if interval <= 0 or max_cycles < 0:
                raise EvidenceError('interval must be positive and max_cycles cannot be negative')
            with self.task():
                farm.v.require(tuple(dict.fromkeys(HARVEST_TEMPLATES + PLANT_TEMPLATES)))
                while True:
                    image = farm.stable()
                    farm.verify_frame(image)
                    ripe = farm.fields('field_ripe', image)
                    if not ripe:
                        farm.log('ripeness_check', confirmed=0)
                        self.stop_event.wait(interval)
                        farm.check()
                        continue
                    farm.log('ripeness_check', confirmed=len(ripe))
                    harvested = farm.run(limit=limit)
                    if farm.reason == 'no_confirmed_fields':
                        self.stop_event.wait(interval)
                        farm.check()
                        continue
                    if farm.reason != 'complete':
                        return
                    targets = farm.confirmed.copy()
                    planted = farm.run(plant=True, limit=harvested, targets=targets)
                    if farm.reason != 'complete' or planted != harvested:
                        if farm.reason in ('complete', 'no_confirmed_fields'):
                            farm.fail(EvidenceError('replant_incomplete'))
                        return
                    cycle += 1
                    deadline = time.monotonic() + interval
                    reserve = max(reserve, planted, len(ripe))
                    seller = ShopSeller(self, self._get_pos())
                    sold = seller.run(slots=10, reserve=reserve, deadline=deadline)
                    if seller.reason not in ('complete', 'shop_full', 'shop_slots_unconfirmed',
                                             'no_surplus', 'out_of_wheat'):
                        if not seller.reason.startswith(('stock_unconfirmed:', 'missing_template:')):
                            return
                    # Before continuing, check that closing the shop restored the same farm view.
                    image = farm.stable()
                    farm.verify_frame(image)
                    farm.log('cycle', harvested=harvested, planted=planted, sold_slots=sold)
                    if max_cycles and cycle >= max_cycles:
                        farm.log('stop', reason='cycle_limit')
                        return
                    self.stop_event.wait(max(0, deadline-time.monotonic()))
                    farm.check()
        except KeyboardInterrupt:
            self.stop_event.set()
            farm.fail(EvidenceError('keyboard_interrupt'))
        except WORKFLOW_ERRORS as error:
            farm.fail(error)


def main():
    bot = HayDayAutoBot()
    if not bot.connect():
        return

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "loop":
            bot.run_loop(max_cycles=int(sys.argv[2]) if len(sys.argv) > 2 else 0,
                         limit=int(sys.argv[3]) if len(sys.argv) > 3 else None)
            return
        if cmd == "debug":
            if bot.debug_view() is not None:
                import subprocess
                subprocess.Popen(["explorer", "debug_detect.png"])
            return
        if cmd == "harvest":
            FarmWorkflow(bot).run(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
            return
        if cmd == "plant":
            FarmWorkflow(bot).run(plant=True, limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
            return

    # Menu tương tác
    while True:
        print("\n" + "="*45)
        print("  🌾 HAY DAY AUTO BOT")
        print("="*45)
        print("  1. Debug: xem vùng phát hiện (chụp + tô màu)")
        print("  2. Cắt mẫu nút lúa mì từ ảnh BlueStacks")
        print("  3. Thu hoạch tất cả ruộng chín")
        print("  4. Trồng lại tất cả ruộng trống")
        print("  5. 1 lần: harvest + plant")
        print("  6. Auto loop (cào → gieo → bán dư → kiểm tra chín)")
        print("  7. Bán lúa mì trong shop (giá MAX)")
        print("  8. Reset field bbox (khi farm thay đổi)")
        print("  9. Calibrate field: kéo chọn vùng cánh đồng lúa")
        print(" 10. Cuộn camera về trên (sửa drift)")
        print("  0. Thoát")
        print("-"*45)
        c = input("  Chọn: ").strip()

        if c == "1":
            if bot.debug_view() is not None:
                import subprocess
                subprocess.Popen(["explorer", "debug_detect.png"])

        elif c == "2":
            bot.calibrate_wheat_button()

        elif c == "3":
            bot.harvest_ripe()

        elif c == "4":
            bot.plant_empty()

        elif c == "5":
            farm = FarmWorkflow(bot)
            try:
                with bot.task():
                    farm.v.require(tuple(dict.fromkeys(HARVEST_TEMPLATES + PLANT_TEMPLATES)))
                    harvested = farm.run()
                    if harvested and farm.reason == 'complete':
                        farm.run(plant=True, limit=harvested, targets=farm.confirmed)
            except WORKFLOW_ERRORS as error:
                farm.fail(error)

        elif c == "6":
            iv = input("  Chờ bao nhiêu giây giữa mỗi lần? (mặc định 120 cho lúa mì): ").strip()
            bot.run_loop(interval=float(iv) if iv else 120.0)

        elif c == "7":
            n = input("  Bán bao nhiêu lần? (mặc định 1): ").strip()
            bot.sell_wheat(count=int(n) if n else 1)

        elif c == "8":
            bot.reset_field_bbox()

        elif c == "9":
            try:
                bot.calibrate_field_region()
            except WORKFLOW_ERRORS as error:
                print(error)

        elif c == "10":
            n = input("  Cuộn lên mấy lần? (mặc định 3): ").strip()
            bot.recenter_camera(up_times=int(n) if n else 3)
            print("  Xong. Nếu wheat field về đúng vị trí → chạy 9 để recalibrate bbox.")

        elif c == "0":
            break


if __name__ == "__main__":
    main()
