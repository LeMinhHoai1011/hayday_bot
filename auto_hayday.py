"""
Hay Day Auto Bot — phát hiện bằng màu HSV, không cần template thủ công.

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
        pos.set("field_bbox", bbox)
        print(f"  [grid] Đã lưu field_bbox={bbox}")

    def reset_field_bbox(self):
        """Xóa cache bbox để detect lại — gọi khi farm mở rộng hoặc thay đổi."""
        pos = self._get_pos()
        if "field_bbox" in pos._data:
            del pos._data["field_bbox"]
            pos._save_file()
            print("  [grid] Đã xóa cache field_bbox.")
            print("  [grid] Chạy menu 9 (Calibrate field) để mark lại vùng mới.")
        else:
            print("  [grid] Không có cache field_bbox.")

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

    def debug_view(self, save_path: str = "debug_detect.png"):
        """Chụp + vẽ tất cả vùng phát hiện lên ảnh để kiểm tra."""
        sc  = self.screenshot()
        vis = sc.copy()
        h, w = sc.shape[:2]

        color_map = {
            "CHÍN":      (0,   220,  255),   # vàng
            "TRỐNG":     (0,   120,  255),   # cam
            "ĐANG MỌC":  (0,   255,   80),   # xanh
        }
        all_regions = (
            self.find_ripe_fields(sc) +
            self.find_empty_fields(sc) +
            self.find_growing_fields(sc)
        )
        for r in all_regions:
            x, y, rw, rh = r["bbox"]
            cx, cy = r["center"]
            col = color_map.get(r["label"], (255, 255, 255))
            cv2.rectangle(vis, (x,y), (x+rw, y+rh), col, 2)
            cv2.circle(vis, (cx, cy), 6, col, -1)
            cv2.putText(vis, f"{r['label']} {r['area']:.0f}",
                        (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # Đánh dấu vị trí nút lúa mì trong menu
        wx, wy = int(w * WHEAT_X_FRAC), int(h * MENU_Y_FRAC)
        cv2.drawMarker(vis, (wx, wy), (255,0,255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis, "WHEAT BTN", (wx+15, wy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,255), 1)

        # Vẽ field bbox (trắng) + tất cả grid points (xanh lam)
        bbox = self._find_field_bbox(sc)
        if bbox:
            bx, by, bw, bh = bbox
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
            cv2.putText(vis, "FIELD", (bx, by-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        pts = self._field_grid_points(sc)
        for px, py in pts:
            state = self._sample_plot(sc, px, py)
            dot_col = {
                "ripe":    (0, 220, 255),
                "empty":   (0, 120, 255),
                "growing": (0, 255, 80),
            }.get(state, (160, 160, 160))
            cv2.circle(vis, (px, py), SAMPLE_R, dot_col, 1)
            cv2.circle(vis, (px, py), 4, dot_col, -1)

        cv2.imwrite(save_path, vis)
        print(f"  Debug ảnh: {save_path}")
        print(f"  Chín: {sum(1 for r in all_regions if r['label']=='CHÍN')}  "
              f"Trống: {sum(1 for r in all_regions if r['label']=='TRỐNG')}  "
              f"Đang mọc: {sum(1 for r in all_regions if r['label']=='ĐANG MỌC')}")
        if pts:
            print(f"  Grid: {len(pts)} điểm trong bbox cánh đồng")
        return vis, all_regions

    # ── Tinh chỉnh vị trí nút lúa mì ─────────────────────────────────────────

    def calibrate_wheat_button(self):
        """
        Tap vào một ruộng trống, chụp màn hình, rồi dùng ROI để xác định
        chính xác vị trí nút lúa mì trong menu.
        """
        sc = self.screenshot()
        empty = self.find_empty_fields(sc)
        if not empty:
            print("  Cần có ruộng trống để calibrate.")
            return
        cx, cy = empty[0]["center"]
        print(f"  Tap ruộng trống ({cx},{cy})...")
        self.tap(cx, cy, delay=0.8)
        sc2 = self.screenshot()
        half = cv2.resize(sc2, (sc2.shape[1]//2, sc2.shape[0]//2))
        roi = cv2.selectROI(
            "Chọn NÚT LÚA MÌ trong menu (Enter xác nhận, C hủy)", half
        )
        cv2.destroyAllWindows()
        self.back()
        if roi[2] == 0: return
        x, y, rw, rh = [v*2 for v in roi]
        bx, by = x + rw//2, y + rh//2
        h, w = sc2.shape[:2]
        global WHEAT_X_FRAC, MENU_Y_FRAC
        WHEAT_X_FRAC = bx / w
        MENU_Y_FRAC  = by / h
        print(f"  Nút lúa mì: pixel=({bx},{by})  "
              f"fraction=({WHEAT_X_FRAC:.3f}, {MENU_Y_FRAC:.3f})")
        print(f"  Cập nhật WHEAT_X_FRAC={WHEAT_X_FRAC:.3f}, MENU_Y_FRAC={MENU_Y_FRAC:.3f} vào file nếu muốn lưu.")

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

    def run_loop(self, interval: float = 65.0, max_cycles: int = 0):
        """
        Vòng lặp chính:
          1. Thu hoạch tất cả ruộng chín
          2. Trồng lại tất cả ruộng trống
          3. Chờ interval giây (đủ cho lúa mì 1p + buffer)
        """
        print("=" * 50)
        print("  🌾 HAY DAY AUTO LOOP — Ctrl+C để dừng")
        print(f"  Interval: {interval}s")
        print("=" * 50)
        cycle = 0
        try:
            while True:
                cycle += 1
                ts = time.strftime("%H:%M:%S")
                print(f"\n[Cycle {cycle}] {ts}")

                with self.task():
                    self.harvest_ripe()
                    if self.last_reason not in ("complete", "no_confirmed_fields"):
                        break
                    self.plant_empty()
                    if self.last_reason not in ("complete", "no_confirmed_fields"):
                        break

                if max_cycles and cycle >= max_cycles:
                    print(f"  Đủ {max_cycles} lần, dừng.")
                    break

                print(f"  Chờ {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Dừng loop.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    bot = HayDayAutoBot()
    if not bot.connect():
        if not bot.auto_connect():
            print("Không kết nối được BlueStacks. Bật ADB trong Settings > Advanced.")
            return

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "loop":
            bot.run_loop()
            return
        if cmd == "debug":
            bot.debug_view()
            import subprocess
            subprocess.Popen(["explorer", "debug_detect.png"])
            return
        if cmd == "harvest":
            bot.harvest_ripe()
            return
        if cmd == "plant":
            bot.plant_empty()
            return

    # Menu tương tác
    while True:
        print("\n" + "="*45)
        print("  🌾 HAY DAY AUTO BOT")
        print("="*45)
        print("  1. Debug: xem vùng phát hiện (chụp + tô màu)")
        print("  2. Tinh chỉnh vị trí nút lúa mì (calibrate)")
        print("  3. Thu hoạch tất cả ruộng chín")
        print("  4. Trồng lại tất cả ruộng trống")
        print("  5. 1 lần: harvest + plant")
        print("  6. Auto loop liên tục (harvest → plant → chờ)")
        print("  7. Bán lúa mì trong shop (giá MAX)")
        print("  8. Reset field bbox (khi farm thay đổi)")
        print("  9. Calibrate field: kéo chọn vùng cánh đồng lúa")
        print(" 10. Cuộn camera về trên (sửa drift)")
        print("  0. Thoát")
        print("-"*45)
        c = input("  Chọn: ").strip()

        if c == "1":
            bot.debug_view()
            import subprocess
            subprocess.Popen(["explorer", "debug_detect.png"])

        elif c == "2":
            bot.calibrate_wheat_button()

        elif c == "3":
            bot.harvest_ripe()

        elif c == "4":
            bot.plant_empty()

        elif c == "5":
            sc = bot.screenshot()
            bot.harvest_ripe(sc)
            time.sleep(1)
            sc = bot.screenshot()
            bot.plant_empty(sc)

        elif c == "6":
            iv = input("  Chờ bao nhiêu giây giữa mỗi lần? (mặc định 65 cho lúa mì): ").strip()
            bot.run_loop(interval=float(iv) if iv else 65.0)

        elif c == "7":
            n = input("  Bán bao nhiêu lần? (mặc định 1): ").strip()
            bot.sell_wheat(count=int(n) if n else 1)

        elif c == "8":
            bot.reset_field_bbox()

        elif c == "9":
            bot.calibrate_field_region()

        elif c == "10":
            n = input("  Cuộn lên mấy lần? (mặc định 3): ").strip()
            bot.recenter_camera(up_times=int(n) if n else 3)
            print("  Xong. Nếu wheat field về đúng vị trí → chạy 9 để recalibrate bbox.")

        elif c == "0":
            break


if __name__ == "__main__":
    main()
