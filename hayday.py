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
        import re
        from bot_config import CONFIG, resolve

        image = self.screenshot()
        h, w = image.shape[:2]
        preview = cv2.resize(image, (w // 2, h // 2))
        ph, pw = preview.shape[:2]
        x, y, rw, rh = cv2.selectROI("Select template - Enter / C to cancel", preview,
                                     fromCenter=False)
        cv2.destroyAllWindows()
        if not rw or not rh:
            return
        name = input("Tên ảnh mẫu (vd. field_ripe): ").strip().removesuffix(".png")
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            print("Tên ảnh mẫu chỉ gồm chữ, số và dấu gạch dưới.")
            return
        x0, x1 = round(x*w/pw), round((x+rw)*w/pw)
        y0, y1 = round(y*h/ph), round((y+rh)*h/ph)
        directory = resolve(CONFIG["templates_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + ".png")
        if not cv2.imwrite(str(path), image[y0:y1, x0:x1]):
            raise OSError(f"Cannot save {path}")
        print(f"Đã lưu {path} từ ảnh Android {w}x{h}")

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
