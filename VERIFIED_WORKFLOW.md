# Luồng xác nhận Hay Day

GUI (`python gui_bot.py`), `run.py` và `auto_hayday.py` gọi chung `FarmWorkflow` / `ShopSeller`. Menu 8 và 11 hỏi số slot, giá tối đa. Chỉ các lệnh barn/silo/all và menu nâng cấp mới nâng kho; mặc định `sell_upgrade.py` chỉ bán.

## Cấu hình

Sửa `bot_config.json`: adb, port hoặc device (serial ADB; null dùng localhost:port), debug_dir, timeout. Đường dẫn tương đối tính từ thư mục tool. `rois` ánh xạ tên ảnh mẫu sang `[left, top, right, bottom]` theo tỷ lệ 0–1 của screenshot Android để giới hạn vùng tìm kiếm. Ví dụ `"inventory_wheat": [0.1, 0.2, 0.7, 0.8]`. Chọn ROI theo màn hình thực tế, không dùng tọa độ cửa sổ Windows.

Input dùng ADB có kiểm tra mã trả về, timeout; các hàm win_tap/win_harvest_swipe cũ vẫn có API tương thích nhưng chuyển sang Android input. Không giữ chuột Windows. Nếu BlueStacks không hỗ trợ input ADB, tool dừng và báo lỗi; chưa có backend chuột Windows với nhận diện viewport đã kiểm chứng.

## Ảnh mẫu cần bổ sung trước khi dùng

Bộ ảnh có sẵn chỉ có `harvest.png`; **chưa đủ để chạy luồng mới**. Tool dừng với `missing_template:<tên>` và không click suy đoán. Chụp ảnh Android ở đúng độ phân giải/zoom; dùng menu 2 của `run.py` cắt mẫu, đặt tên sau (không thêm .png ở ô nhập tên). Chọn mẫu có hoa văn rõ, tránh ảnh đồng màu, tránh số lượng thay đổi, tránh nền farm nếu là trạng thái dialog. Cần kiểm tra template/ROI trên cả ảnh đúng và sai trước khi chạy tự động.

- Farm: `farm_screen` (dấu hiệu chỉ có ở farm), `field_ripe` (một ô lúa mì chín, tâm nằm trong ô), `field_empty` (một ô đất trống), `harvest_tool` (tâm công cụ để kéo về ô), `seed_menu`, `crop_menu_wheat` (trong menu hạt giống), `field_growing`.
- Shop: `shop_stand` (mái sọc/quầy có cả chi tiết trắng và đỏ), `shop_screen`, `shop_slot_empty`, `inventory_screen`, `inventory_wheat`, `wheat_selected` (chi tiết xác nhận wheat đang được chọn), `btn_price_max`, `price_max_reached` (giá tối đa cho số lượng thực tế), `btn_put_sale`, `shop_slot_wheat` (lúa đang đăng trong slot), `shop_full`, `out_of_wheat` (chỉ báo hết lúa).
- Nâng kho: `barn`, `silo`, `barn_screen`, `silo_screen`, `missing_materials`, `btn_upgrade`, `upgrade_confirm`, `btn_confirm`, `barn_upgraded`, `silo_upgraded`. Dấu hiệu upgraded phải xác nhận dung lượng tăng; không dùng nút đóng/cửa sổ biến mất làm bằng chứng. Chưa hỗ trợ đọc dung lượng bằng OCR.

Mẫu field_empty/field_growing dùng xác nhận nên đủ nhỏ để nằm trong vùng 50×50 quanh tâm ô. Mẫu shop_slot_wheat nằm trong vùng 120×110 quanh tâm slot; đặt tâm mẫu slot_empty trùng tâm slot có lúa. Các mẫu không tự scale: thay độ phân giải/zoom phải chụp lại mẫu. Các mẫu price_max_reached/upgraded phụ thuộc giá/dung lượng đã hiệu chuẩn; cần cập nhật khi thay cấu hình đó.

## Hành vi và giới hạn

Mỗi lượt hiện xử lý một ô, lấy ảnh mới và nhận diện lại trước lượt kế; không kéo xuyên qua ô unknown/nền. Count là số chuyển trạng thái tại tâm ô đã xác nhận. Không dùng lưới ước lượng màu để đếm nữa. Camera dùng feature tracking + RANSAC ngoài vùng ruộng mục tiêu, loại biên HUD; thiếu ít nhất 8 mốc đồng thuận thì dừng và lưu ảnh. Mốc còn có thể nhầm cây động; cần kiểm tra trên farm thật. Khi camera trôi với vùng GUI do người dùng chọn, bot dừng yêu cầu chọn lại vùng; khi tự tìm ô thì nhận diện lại ảnh mới.

Cào + Bán giữ khóa xuyên suốt hai bước, chỉ bán nếu thu được ít nhất một ô và thu hoạch kết thúc với `complete`. Shop luôn quét lại bằng mẫu mái sọc trên ảnh farm mới, không dùng cache. Chưa xác nhận đăng thì chỉ kiểm tra lại đúng slot, không bấm đăng lần nữa. Shop đầy/hết lúa chỉ kết luận khi có dấu hiệu riêng; thiếu dấu hiệu trả `unconfirmed`.

Khóa Windows named mutex theo device loại trừ các tác vụ giữa GUI/CLI; cần dùng cùng serial device cho cùng emulator. Dừng ngăn input kế tiếp; ADB hiện tại được phép kết thúc hoặc timeout rồi mới báo Đã dừng. `debug/steps.jsonl` lưu thao tác, nguồn tọa độ, trạng thái, kết quả ADB, dịch camera và lý do dừng; ảnh before/after theo timestamp khi không xác nhận được.

## Kiểm chứng

`python -m unittest -v test_workflows` chạy kiểm thử offline bằng ảnh tổng hợp và bot giả, không thao tác BlueStacks. `python -m compileall -q .` kiểm tra cú pháp. Các kiểm thử này không thay thế nghiệm thu trên game thật. Phải bổ sung ảnh mẫu và chạy từng tình huống trong yêu cầu nghiệm thu trước khi coi phiên bản sẵn sàng dùng.
