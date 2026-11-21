"""
Hay Day Bot GUI — Tkinter desktop app
- Hien thi screenshot BlueStacks
- Ve vung thu hoach bang chuot
- Nut: Thu hoach, Ban lua, Dung, Quet lai
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from bot import BlueStacksBot
from auto_calibrate import ElementDetector

# ─── Cài đặt ─────────────────────────────────
SCALE = 0.55          # thu nho anh hien thi
REFRESH_MS = 2000     # cap nhat anh moi 2s
SWIPE_ROW_STEP = 30   # khoang cach giua cac hang keo (px tren man hinh goc)
SWIPE_SPEED_MS = 600  # thoi gian moi lan keo (ms)
SWIPE_DELAY = 0.25    # delay giua cac lan keo (s)

# ─────────────────────────────────────────────


class HayDayGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hay Day Bot")
        self.root.resizable(False, False)

        self.bot: BlueStacksBot | None = None
        self.zones: list[tuple[int,int,int,int]] = []  # danh sach vung (x0,y0,x1,y1) tren anh goc
        self._drag_start: tuple[int,int] | None = None
        self._rect_id = None
        self._running = False
        self._stop_flag = False
        self._photo = None
        self._sc: np.ndarray | None = None

        self._build_ui()
        self._connect()
        self._refresh_loop()

    # ─── UI ──────────────────────────────────
    def _build_ui(self):
        # Thanh trang thai
        top = tk.Frame(self.root, bg="#2b2b2b")
        top.pack(fill=tk.X)
        self._status = tk.StringVar(value="Dang ket noi...")
        tk.Label(top, textvariable=self._status, bg="#2b2b2b", fg="#aaffaa",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=8, pady=4)
        self._live_badge = tk.Label(top, text="●", bg="#2b2b2b", fg="#ff4444",
                                    font=("Consolas", 14))
        self._live_badge.pack(side=tk.RIGHT, padx=6)

        # Canvas hien thi anh
        W = int(1600 * SCALE)
        H = int(900  * SCALE)
        self._canvas = tk.Canvas(self.root, width=W, height=H,
                                  bg="#111", cursor="crosshair",
                                  highlightthickness=0)
        self._canvas.pack()
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<ButtonPress-3>",   self._on_right_click)

        # Thanh nut
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, pady=4)

        btn_cfg = dict(font=("Segoe UI", 10, "bold"), width=14, pady=6, relief=tk.FLAT, bd=0)

        self._btn_harvest = tk.Button(btn_frame, text="⚡ Thu Hoach",
                                      bg="#e67e22", fg="white",
                                      command=self._harvest_zones, **btn_cfg)
        self._btn_harvest.pack(side=tk.LEFT, padx=6, pady=4)

        self._btn_sell = tk.Button(btn_frame, text="💰 Ban Lua",
                                   bg="#27ae60", fg="white",
                                   command=self._sell, **btn_cfg)
        self._btn_sell.pack(side=tk.LEFT, padx=6)

        self._btn_harvsell = tk.Button(btn_frame, text="🔄 Cao + Ban",
                                        bg="#8e44ad", fg="white",
                                        command=self._harvest_then_sell, **btn_cfg)
        self._btn_harvsell.pack(side=tk.LEFT, padx=6)

        self._btn_rescan = tk.Button(btn_frame, text="🔍 Quet Lai",
                                     bg="#2980b9", fg="white",
                                     command=self._rescan, **btn_cfg)
        self._btn_rescan.pack(side=tk.LEFT, padx=6)

        self._btn_clear = tk.Button(btn_frame, text="✕ Xoa Vung",
                                    bg="#555", fg="white",
                                    command=self._clear_zones, **btn_cfg)
        self._btn_clear.pack(side=tk.LEFT, padx=6)

        self._btn_stop = tk.Button(btn_frame, text="⏹ Dung",
                                   bg="#c0392b", fg="white",
                                   command=self._stop, **btn_cfg)
        self._btn_stop.pack(side=tk.RIGHT, padx=6)

        self._btn_refresh = tk.Button(btn_frame, text="📷 Cap nhat",
                                      bg="#34495e", fg="white",
                                      command=self._force_refresh, **btn_cfg)
        self._btn_refresh.pack(side=tk.RIGHT, padx=6)

        # Log
        log_frame = tk.Frame(self.root, bg="#1e1e1e")
        log_frame.pack(fill=tk.X)
        self._log = tk.Text(log_frame, height=5, bg="#111", fg="#ddd",
                            font=("Consolas", 9), state=tk.DISABLED,
                            wrap=tk.WORD, padx=4)
        self._log.pack(fill=tk.X, padx=4, pady=4)

        # Huong dan
        hint = ("Chuot trai: ve vung thu hoach  |  Chuot phai: xoa vung vua ve  "
                "|  Nhan 'Thu Hoach' de keo liem qua cac vung da danh dau")
        tk.Label(self.root, text=hint, bg="#1e1e1e", fg="#888",
                 font=("Segoe UI", 8)).pack(pady=(0,4))

    # ─── Ket noi ─────────────────────────────
    def _connect(self):
        def worker():
            try:
                self.bot = BlueStacksBot()
                ok = self.bot.connect()
                self._set_status("Đã kết nối" if ok else "Không kết nối được BlueStacks")
                self.root.after(0, lambda: self._live_badge.config(fg="#44ff44" if ok else "#ff4444"))
            except Exception as error:
                self._log_msg(str(error))
        threading.Thread(target=worker, daemon=True).start()

    # ─── Ve vung ─────────────────────────────
    def _screen_to_game(self, cx, cy):
        """Chuyen toa do canvas -> toa do man hinh game."""
        return int(cx / SCALE), int(cy / SCALE)

    def _game_to_canvas(self, gx, gy):
        return int(gx * SCALE), int(gy * SCALE)

    def _on_press(self, e):
        self._drag_start = (e.x, e.y)
        if self._rect_id:
            self._canvas.delete(self._rect_id)
        self._rect_id = self._canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#ff0", width=2, dash=(4,2))

    def _on_drag(self, e):
        if self._drag_start and self._rect_id:
            x0, y0 = self._drag_start
            self._canvas.coords(self._rect_id, x0, y0, e.x, e.y)

    def _on_release(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = e.x, e.y
        if abs(x1-x0) < 10 or abs(y1-y0) < 10:
            if self._rect_id:
                self._canvas.delete(self._rect_id)
            self._drag_start = None
            return
        # Luu vung (toa do game)
        gx0, gy0 = self._screen_to_game(min(x0,x1), min(y0,y1))
        gx1, gy1 = self._screen_to_game(max(x0,x1), max(y0,y1))
        self._add_zone((gx0, gy0, gx1, gy1))
        self._drag_start = None

    def _on_right_click(self, e):
        """Xoa vung gan nhat khi click phai."""
        if self.zones:
            removed = self.zones.pop()
            self._redraw_zones()
            self._log_msg(f"Xoa vung {removed}")

    def _add_zone(self, zone):
        self.zones.append(zone)
        self._redraw_zones()
        self._log_msg(f"Vung #{len(self.zones)}: game ({zone[0]},{zone[1]})→({zone[2]},{zone[3]})")

    def _redraw_zones(self):
        self._canvas.delete("zone")
        colors = ["#ff0", "#0ff", "#0f0", "#f80", "#f0f"]
        for i, (gx0,gy0,gx1,gy1) in enumerate(self.zones):
            cx0,cy0 = self._game_to_canvas(gx0,gy0)
            cx1,cy1 = self._game_to_canvas(gx1,gy1)
            c = colors[i % len(colors)]
            self._canvas.create_rectangle(cx0,cy0,cx1,cy1,
                outline=c, width=2, tags="zone")
            self._canvas.create_text(cx0+4, cy0+4,
                text=f"#{i+1}", fill=c, anchor=tk.NW,
                font=("Consolas", 9, "bold"), tags="zone")

    def _clear_zones(self):
        self.zones.clear()
        self._canvas.delete("zone")
        self._log_msg("Da xoa tat ca vung.")

    # ─── Cap nhat anh ────────────────────────
    def _refresh_loop(self):
        if not self._running:
            self._update_screenshot()
        self.root.after(REFRESH_MS, self._refresh_loop)

    def _force_refresh(self):
        self._update_screenshot()

    def _update_screenshot(self):
        if self.bot is None:
            return
        def _do():
            try:
                sc = self.bot.screenshot()
                self._sc = sc
                self.root.after(0, lambda: self._show_frame(sc))
            except Exception as ex:
                self._log_msg(f"Chup anh loi: {ex}")
        threading.Thread(target=_do, daemon=True).start()

    def _show_frame(self, sc: np.ndarray):
        rgb = cv2.cvtColor(sc, cv2.COLOR_BGR2RGB)
        h_sc, w_sc = rgb.shape[:2]
        new_w, new_h = int(w_sc * SCALE), int(h_sc * SCALE)
        small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(small)
        photo = ImageTk.PhotoImage(img)
        self._photo = photo  # giu reference
        self._canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._redraw_zones()  # ve vung len tren

    # ─── Log ─────────────────────────────────
    def _log_msg(self, msg: str):
        def _do():
            self._log.config(state=tk.NORMAL)
            self._log.insert(tk.END, f"› {msg}\n")
            self._log.see(tk.END)
            self._log.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self._status.set(msg))

    # ─── Thu hoach ───────────────────────────
    def _start_task(self, mode):
        if self._running or self.bot is None:
            return
        # Set synchronously on Tk's thread to close the double-click race.
        self._running = True
        self._stop_flag = False
        self.bot.stop_event.clear()
        zones = list(self.zones)
        self._set_status("Đang thực hiện " + mode)
        def worker():
            from verified_workflow import FarmWorkflow
            from sell_upgrade import ShopSeller
            from auto_calibrate import Positions
            result_status = "Hoàn tất"
            try:
                with self.bot.task():
                    if mode in ("harvest", "harvest_sell"):
                        farm = FarmWorkflow(self.bot)
                        count = farm.run(zones=zones or None)
                        self._log_msg(f"Đã xác nhận thu {count} ô; {farm.reason}")
                        if farm.reason != "complete" or count == 0:
                            result_status = "Dừng: " + farm.reason
                            return
                    self.bot.check_stop()
                    if mode in ("sell", "harvest_sell"):
                        seller = ShopSeller(self.bot, Positions(self.bot))
                        sold = seller.run(slots=1)
                        self._log_msg(f"{sold} slot đã đăng; {seller.reason}")
                        if seller.reason != "complete":
                            result_status = "Dừng: " + seller.reason
            except Exception as error:
                self._log_msg(str(error))
                result_status = "Lỗi: " + str(error)
            finally:
                if self.bot.stop_event.is_set():
                    result_status = "Đã dừng"
                self._set_status(result_status)
                self.root.after(0, lambda: setattr(self, "_running", False))
        threading.Thread(target=worker, daemon=True).start()

    def _harvest_zones(self):
        self._start_task("harvest")

    def _sell(self):
        self._start_task("sell")

    def _harvest_then_sell(self):
        self._start_task("harvest_sell")

    # ─── Quet lai ────────────────────────────
    def _rescan(self):
        if self._running or self.bot is None:
            return
        def worker():
            try:
                from verified_workflow import Vision
                image = self.bot.screenshot()
                fields = Vision(self.bot).matches("field_ripe", image)
                zones = [(x-25,y-25,x+25,y+25) for x,y,_ in fields]
                def update():
                    if self._running:
                        return
                    self.zones = zones
                    self._show_frame(image)
                self.root.after(0, update)
                self._log_msg(f"Nhận diện {len(fields)} ô chín trên ảnh mới")
            except Exception as error:
                self._log_msg(str(error))
        threading.Thread(target=worker, daemon=True).start()

    def _merge_zones(self, zones, gap=50):
        """Gop cac zone co overlap hoac gan nhau."""
        if not zones:
            return []
        merged = list(zones)
        changed = True
        while changed:
            changed = False
            result = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                x0,y0,x1,y1 = merged[i]
                for j in range(i+1, len(merged)):
                    if used[j]:
                        continue
                    ax0,ay0,ax1,ay1 = merged[j]
                    # Kiem tra co overlap/sat nhau
                    if (x0-gap < ax1 and ax0-gap < x1 and
                        y0-gap < ay1 and ay0-gap < y1):
                        x0 = min(x0,ax0); y0 = min(y0,ay0)
                        x1 = max(x1,ax1); y1 = max(y1,ay1)
                        used[j] = True
                        changed = True
                result.append((x0,y0,x1,y1))
                used[i] = True
            merged = result
        return merged

    # ─── Dung ────────────────────────────────
    def _stop(self):
        self._stop_flag = True
        self._log_msg("Dang dung...")
        if self.bot is not None:
            self.bot.stop_event.set()
        self._set_status("Đang dừng" if self._running else "Đã dừng")


def main():
    root = tk.Tk()
    root.configure(bg="#1e1e1e")
    app = HayDayGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
