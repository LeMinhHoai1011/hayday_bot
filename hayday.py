"""
Hay Day Bot — các hành động game cụ thể.
Dùng template matching để nhận dạng + ADB để thao tác.
"""

import time
import os
import cv2
import numpy as np
from bot import BlueStacksBot, Match, TEMPLATES_DIR


class HayDayBot(BlueStacksBot):

    # ── Thu hoạch ─────────────────────────────────────────────────────────────

    def harvest_all(self, threshold: float = 0.75) -> int:
        from verified_workflow import FarmWorkflow
        return FarmWorkflow(self).run()

    def plant_all(self, crop_template="crop_menu_wheat", threshold=.75):
        from verified_workflow import FarmWorkflow
        return FarmWorkflow(self).run(plant=True)

    # ── Thu thập sản phẩm ─────────────────────────────────────────────────────

    def collect_products(self, threshold: float = 0.75) -> int:
        """
        Thu thập từ động vật / máy móc / cây ăn quả đã có sản phẩm.
        Cần template: templates/product_ready.png
        (icon bong bóng sản phẩm xuất hiện trên gà/bò/máy...)
        """
        print("[hayday] Đang tìm sản phẩm sẵn sàng...")
        sc = self.screenshot()
        matches = self.find("product_ready", screen=sc, threshold=threshold)
        if not matches:
            print("[hayday] Không có sản phẩm nào sẵn sàng.")
            return 0
        print(f"[hayday] Tìm thấy {len(matches)} sản phẩm → đang thu thập...")
        self.tap_all(matches, delay=0.4)
        print(f"[hayday] Thu thập xong {len(matches)} sản phẩm.")
        return len(matches)

    # ── Bán hàng ──────────────────────────────────────────────────────────────

    def sell_items(self, item_template: str, price: int = None, threshold: float = 0.75) -> bool:
        """
        Mở shop, tìm mặt hàng và đặt bán.
        item_template: tên template mặt hàng (vd: 'item_bread')
        """
        print(f"[hayday] Mở roadside shop...")
        sc = self.screenshot()
        shop = self.find("shop_button", screen=sc, threshold=threshold)
        if not shop:
            print("[hayday] Không tìm thấy nút shop.")
            return False
        self.tap_match(shop[0], delay=0.8)
        # Tìm slot trống trong shop
        sc2 = self.screenshot()
        slot = self.find("shop_empty_slot", screen=sc2, threshold=threshold)
        if not slot:
            print("[hayday] Shop đầy rồi.")
            return False
        self.tap_match(slot[0], delay=0.5)
        # Tìm mặt hàng trong kho
        sc3 = self.screenshot()
        item = self.find(item_template, screen=sc3, threshold=threshold)
        if not item:
            print(f"[hayday] Không có {item_template} trong kho.")
            return False
        self.tap_match(item[0], delay=0.5)
        print(f"[hayday] Đã đặt {item_template} vào shop.")
        return True

    # ── Loop tự động ──────────────────────────────────────────────────────────

    def auto_loop(
        self,
        interval: float = 60.0,
        do_harvest: bool = True,
        do_collect: bool = True,
        do_plant: bool = False,
        crop: str = "crop_menu_wheat",
        threshold: float = 0.75,
        max_cycles: int = 0,
    ):
        """
        Chạy vòng lặp tự động: thu hoạch → thu thập → (trồng lại).
        interval: số giây giữa các lần quét (mặc định 60s)
        max_cycles: 0 = chạy mãi, >0 = dừng sau n lần
        """
        print("=" * 50)
        print("[hayday] Bắt đầu auto loop. Nhấn Ctrl+C để dừng.")
        print(f"  Interval: {interval}s | Harvest: {do_harvest} | Collect: {do_collect} | Plant: {do_plant}")
        print("=" * 50)
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n[cycle {cycle}] {time.strftime('%H:%M:%S')}")
                if do_harvest:
                    self.harvest_all(threshold)
                if do_collect:
                    self.collect_products(threshold)
                if do_plant:
                    self.plant_all(crop, threshold)
                if max_cycles and cycle >= max_cycles:
                    print(f"[hayday] Đã chạy đủ {max_cycles} lần, dừng.")
                    break
                print(f"[hayday] Chờ {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[hayday] Dừng loop.")

    # ── Tiện ích ──────────────────────────────────────────────────────────────

    def capture_template_interactive(self):
        """
        Chụp màn hình hiện tại, hiện lên để chọn vùng template bằng chuột.
        Dùng để tạo template mới dễ dàng.
        """
        print("[hayday] Chụp màn hình BlueStacks...")
        sc = self.screenshot()
        half = cv2.resize(sc, (sc.shape[1]//2, sc.shape[0]//2))

        roi = cv2.selectROI("Kéo chọn vùng template, Enter để xác nhận, C để hủy", half)
        cv2.destroyAllWindows()
        if roi[2] == 0 or roi[3] == 0:
            print("[hayday] Hủy.")
            return
        x, y, w, h = [v*2 for v in roi]  # scale lại vì đã resize về 1/2
        name = input("Nhập tên template (vd: harvest_ready): ").strip()
        if name:
            self.save_region_as_template(x, y, w, h, name)

    def debug_scan(self, template_name: str, threshold: float = 0.75):
        """Chụp màn hình và hiện kết quả nhận dạng để debug."""
        sc = self.screenshot()
        try:
            matches = self.find(template_name, screen=sc, threshold=threshold)
        except FileNotFoundError as e:
            print(f"[debug] {e}")
            return
        print(f"[debug] '{template_name}': tìm thấy {len(matches)} kết quả")
        for i, m in enumerate(matches):
            print(f"  {i+1}. center={m.center} score={m.score:.3f}")
        self.show_matches(sc, matches, window=f"scan: {template_name}")
