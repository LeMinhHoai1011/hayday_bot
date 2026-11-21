"""Shared verified wheat selling and explicitly requested Barn/Silo upgrades.

python sell_upgrade.py sell [slots]  (default command)
python sell_upgrade.py barn | silo | all | calibrate | calibrate_shop
See VERIFIED_WORKFLOW.md for required visual evidence and configuration.
"""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, r"D:\tools\hayday_bot")
from bot import BlueStacksBot
from auto_calibrate import Positions, ElementDetector

DIALOG_THRESH = 6.0   # mean pixel diff tối thiểu để xác nhận dialog đã mở


def _save(sc, name):
    from bot_config import CONFIG, resolve
    directory = resolve(CONFIG["debug_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / f"eco_{name}.png"), sc)


def _diff(a, b):
    return float(cv2.absdiff(a, b).mean())


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE BÁN LÚA MÌ
# ═══════════════════════════════════════════════════════════════════════════════

from verified_workflow import VerifiedShopSeller as ShopSeller, Workflow, EvidenceError


class UpgradeManager:
    """
    Nâng cấp Barn (kho vật phẩm) và Silo (tháp nông sản).
    Tọa độ Barn/Silo được phát hiện động.
    Nút upgrade được tìm real-time sau khi dialog mở.
    """

    def __init__(self, bot: BlueStacksBot, pos: Positions):
        self.b = bot
        self.pos = pos

    def _sc(self):
        return self.b.screenshot()

    def _tap_building(self, key, label):
        flow = Workflow(self.b)
        image = flow.stable()
        flow.v.point("farm_screen", image)
        flow.tap(flow.v.point(key, image), key)
        after = flow.stable()
        flow.v.point(label + "_screen", after)
        return True, after

    def _do_upgrade(self, label):
        flow = Workflow(self.b)
        before = flow.stable()
        flow.v.point(label + "_screen", before)
        if flow.v.state("missing_materials", before):
            return "missing_materials"
        flow.tap(flow.v.point("btn_upgrade", before), "btn_upgrade")
        image = flow.stable()
        if flow.v.state("upgrade_confirm", image):
            flow.tap(flow.v.point("btn_confirm", image), "upgrade_confirm")
            image = flow.stable()
        if not flow.v.state(label + "_upgraded", image):
            flow.fail(EvidenceError("upgrade_unconfirmed"), before)
            return "unconfirmed"
        flow.log("upgrade", before=label+"_screen", after=label+"_upgraded")
        return "upgraded"

    def upgrade_barn(self) -> str:
        with self.b.task():
            print("[upgrade] Tap Barn (kho vật phẩm)...")
            opened, _ = self._tap_building("barn", "barn")
            if not opened:
                return "failed"
            result = self._do_upgrade("barn")
            self.b.back(); time.sleep(0.4)
            print(f"[upgrade] Barn → {result}")
            return result

    def upgrade_silo(self) -> str:
        with self.b.task():
            print("[upgrade] Tap Silo (tháp nông sản)...")
            opened, _ = self._tap_building("silo", "silo")
            if not opened:
                return "failed"
            result = self._do_upgrade("silo")
            self.b.back(); time.sleep(0.4)
            print(f"[upgrade] Silo → {result}")
            return result

    def upgrade_all(self) -> dict:
        with self.b.task():
            barn = self.upgrade_barn()
            time.sleep(0.5)
            silo = self.upgrade_silo()
            return {"barn": barn, "silo": silo}


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATE HELPERS (gọi từ run.py)
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(bot: BlueStacksBot):
    """Scan farm + lưu debug image."""
    pos = Positions(bot)
    pos.scan_farm(force=True)
    pos.print_all()
    path = pos.draw_debug(label="farm")
    return path


def calibrate_shop(bot: BlueStacksBot):
    """
    Mở shop → scan dialog → lưu debug image → đóng shop.
    """
    with bot.task():
        pos = Positions(bot)
        seller = ShopSeller(bot, pos)
        seller.open_shop()
        seller.tap_empty_slot()
        sc = bot.screenshot()
        pos.scan_dialog(force=True)
        path = pos.draw_debug(sc=sc, label="shop_dialog")
        bot.back()
        bot.back()
        return path


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    b = BlueStacksBot()
    if not b.connect():
        return

    pos = Positions(b)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sell"

    if cmd == "calibrate":
        path = calibrate(b)
        import subprocess
        subprocess.Popen(["explorer", path])

    elif cmd == "calibrate_shop":
        path = calibrate_shop(b)
        import subprocess
        subprocess.Popen(["explorer", path])

    elif cmd in ("sell", "all"):
        print("\n── BÁN LÚA MÌ ─────────────────────────────")
        slots = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        seller = ShopSeller(b, pos)
        sold = seller.run(slots=slots)
        print(f"Kết quả: {sold} slot(s) đã đưa ra bán")

    if cmd in ("barn", "all"):
        print("\n── NÂNG CẤP BARN ───────────────────────────")
        mgr = UpgradeManager(b, pos)
        print(f"Barn: {mgr.upgrade_barn()}")

    if cmd in ("silo", "all"):
        print("\n── NÂNG CẤP SILO ───────────────────────────")
        mgr = UpgradeManager(b, pos)
        print(f"Silo: {mgr.upgrade_silo()}")

    print("\nHoàn thành. Xem eco_*.png để debug.")


if __name__ == "__main__":
    main()
