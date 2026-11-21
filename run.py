"""
Chạy Hay Day Bot — menu tương tác.
Chạy: python run.py
"""

import sys
import os
from auto_calibrate import Positions
from hayday import HayDayBot
from sell_upgrade import ShopSeller, UpgradeManager, calibrate, calibrate_shop

from bot_config import CONFIG

ADB_PORT = CONFIG["port"]  # Đổi nếu BlueStacks dùng port khác


def menu(bot: HayDayBot):
    while True:
        print("\n" + "="*50)
        print("  HAY DAY BOT")
        print("="*50)
        print("  --- FARM ---")
        print("  1. Chup man hinh (test ket noi)")
        print("  2. Tao template moi")
        print("  3. Debug template")
        print("  4. Thu hoach ruong chin")
        print("  5. Thu san pham (ga/bo/may)")
        print("  6. Trong lai o trong")
        print("  7. Auto loop (harvest + collect)")
        print("")
        print("  --- BAN HANG & NANG CAP ---")
        print("  8. Ban lua mi (shop ben duong)")
        print("  9. Nang cap Barn")
        print("  10. Nang cap Silo")
        print("  11. Ban + nang cap (ca 3 buoc)")
        print("  12. Calibrate: kiem tra toa do farm (shop/barn/silo)")
        print("  13. Calibrate shop: kiem tra toa do dialog ban hang")
        print("")
        print("  0. Thoat")
        print("-"*50)
        choice = input("  Chon: ").strip()

        if choice == "1":
            path = os.path.join(os.path.dirname(__file__), "debug_sc.png")
            bot.save_screenshot(path)
            import subprocess
            subprocess.Popen(["explorer", path])

        elif choice == "2":
            bot.capture_template_interactive()

        elif choice == "3":
            name = input("  Ten template: ").strip()
            thr  = input("  Threshold (mac dinh 0.75): ").strip()
            bot.debug_scan(name, float(thr) if thr else 0.75)

        elif choice == "4":
            thr = input("  Threshold (mac dinh 0.75): ").strip()
            bot.harvest_all(float(thr) if thr else 0.75)

        elif choice == "5":
            thr = input("  Threshold (mac dinh 0.75): ").strip()
            bot.collect_products(float(thr) if thr else 0.75)

        elif choice == "6":
            crop = input("  Template cay trong (mac dinh: crop_menu_wheat): ").strip()
            thr  = input("  Threshold (mac dinh 0.75): ").strip()
            bot.plant_all(crop or "crop_menu_wheat", float(thr) if thr else 0.75)

        elif choice == "7":
            interval = input("  Quet moi bao nhieu giay? (mac dinh 60): ").strip()
            cycles   = input("  So lan lap (0 = mai mai): ").strip()
            bot.auto_loop(
                interval=float(interval) if interval else 60,
                do_harvest=True,
                do_collect=True,
                do_plant=False,
                max_cycles=int(cycles) if cycles else 0,
            )

        elif choice == "8":
            qty = input("  So slot dang ban gia toi da (mac dinh 1): ").strip()
            seller = ShopSeller(bot, Positions(bot))
            seller.run(slots=int(qty) if qty else 1)

        elif choice == "9":
            mgr = UpgradeManager(bot, Positions(bot))
            result = mgr.upgrade_barn()
            print(f"  Barn: {result}")

        elif choice == "10":
            mgr = UpgradeManager(bot, Positions(bot))
            result = mgr.upgrade_silo()
            print(f"  Silo: {result}")

        elif choice == "11":
            qty = input("  So slot dang ban gia toi da (mac dinh 1): ").strip()
            seller = ShopSeller(bot, Positions(bot))
            sold = seller.run(slots=int(qty) if qty else 1)
            mgr = UpgradeManager(bot, Positions(bot))
            barn = mgr.upgrade_barn()
            silo = mgr.upgrade_silo()
            print(f"\n  Ket qua: ban={sold}, barn={barn}, silo={silo}")

        elif choice == "12":
            path = calibrate(bot)
            import subprocess
            subprocess.Popen(["explorer", path])

        elif choice == "13":
            path = calibrate_shop(bot)
            import subprocess
            subprocess.Popen(["explorer", path])

        elif choice == "0":
            print("Thoat.")
            break
        else:
            print("  Khong hop le, thu lai.")


def main():
    print("[bot] Khởi động...")
    bot = HayDayBot(port=ADB_PORT)

    if not bot.connect():
        sys.exit(1)

    menu(bot)


if __name__ == "__main__":
    main()
