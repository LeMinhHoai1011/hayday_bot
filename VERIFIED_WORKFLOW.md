# Vòng lúa mì trên BlueStacks 5

## Kết nối và chạy

Bật Android Debug Bridge trong **BlueStacks 5 → Settings → Advanced**. Mở đúng farm,
giữ nguyên độ phân giải và zoom. Sửa \`adb\` và \`device\` trong \`bot_config.json\`
(\`null\` dùng \`localhost:<port>\`). Ba điểm vào chính kiểm tra ADB, serial và ảnh PNG
Android ngay khi kết nối. Cài các thư viện Python của dự án nếu môi trường chưa có
\`opencv-python\`, \`numpy\` và \`Pillow\`.

1. Chạy \`python auto_hayday.py\`, chọn **9** và khoanh cụm ruộng. Vùng và ảnh tham
   chiếu được lưu trong \`debug/field_calibration_*.npz\` theo thiết bị. Chọn lại sau
   khi đổi độ phân giải, zoom hoặc kéo farm bằng tay.
2. Chạy \`python auto_hayday.py debug\`. Ảnh \`debug_detect.png\` đánh dấu vùng tìm
   màu xanh, tâm ô màu vàng và điểm chạm đầu tiên khoanh đỏ. Trong GUI, kéo vùng
   ruộng trên ảnh rồi bấm **Quét lại** để dùng cùng bộ dò.
3. Chạy \`python auto_hayday.py harvest 1\` hoặc \`python auto_hayday.py plant 1\`
   để thử một thao tác. Lệnh \`python auto_hayday.py loop 1 1\` cào tối đa một ô,
   gieo chính ô đó và thử bán dư. Dùng \`python auto_hayday.py loop\` cho vòng liên
   tục. Hai tham số tùy chọn của \`loop\` là số vòng và số ô tối đa mỗi vòng.
4. Với vòng liên tục, bot kiểm tra ảnh lúa chín trước khi bắt đầu lượt mới. Menu
   **6** của \`auto_hayday.py\` cho đổi thời gian chờ mặc định 120 giây.

## Cào và gieo một cụm đã đánh dấu

1. Mở đúng farm và đưa cả cụm vào khung ảnh Android ở mức zoom dùng để chạy.
2. Chạy `python auto_hayday.py calibrate-plots`. Bấm vào tâm mặt đất của từng ô
   trong cụm, kể cả ô đang trống hoặc đang mọc. Chuột phải bỏ điểm cuối, Enter
   lưu và Esc hủy. Số thứ tự bấm là mã ô và thứ tự xử lý. Hiệu chỉnh lại khi
   đổi farm, zoom, độ phân giải hoặc bố trí ruộng.
3. Chạy `python auto_hayday.py debug` để xem tâm và trạng thái từng ô. Ảnh dùng
   cùng bộ dò với `python auto_hayday.py harvest-plant`. Chạy
   `python auto_hayday.py harvest-plant 1` để thử giới hạn một ô đã xác nhận.
   Một đường kéo có thể chạm thêm ô trong cụm. Bot ghi nhận những ô đã quan sát
   chuyển từ chín sang trống và đưa chúng vào danh sách gieo.
4. Xem dòng `harvest_plant_result` trong `debug/steps.jsonl`. Trường
   `remaining` đếm các ô chín hoặc chưa rõ lúc bắt đầu mà chưa xác nhận cào,
   kể cả ô chưa đến lượt theo giới hạn. `unplanted` đếm riêng số ô đã cào mà
   chưa xác nhận gieo. Bot thử kéo liềm tối đa hai lần khi ảnh vẫn xác nhận ô
   chín. Nếu màn hình và mốc camera còn xác định được, bot lưu ảnh ô lỗi và
   làm tiếp. Khi mất mốc hoặc gặp màn hình lạ, bot dừng.

`drag_anchors.harvest_tool` trong `bot_config.json` là tọa độ tương đối từ góc
trái trên của ảnh mẫu, theo tỷ lệ từ 0 đến 1. `debug/last_drag.png` đánh dấu tâm
ảnh mẫu bằng vòng xanh, điểm bắt đầu kéo bằng dấu đỏ và tâm ô đích bằng dấu
vàng. `debug/last_harvest_drag.png` giữ riêng lượt kéo liềm gần nhất. Với đường gieo đi qua ô trống hoặc chưa rõ ngoài danh sách đã cào trong
cụm, bot ghi lý do và bỏ lượt gieo đó.

## Ảnh mẫu

Các mẫu đã cắt trong \`templates/\` lấy từ ảnh Android 1600×900 trên BlueStacks ở
mức zoom dùng để thử. Khi thay zoom/độ phân giải/giao diện, mở trạng thái tương
ứng rồi chạy \`python run.py\`, chọn **2**, kéo vùng, nhấn Enter và nhập tên mẫu.
Menu lưu ảnh gốc đã dùng để chọn, với độ phân giải gốc và đường dẫn
\`templates_dir\` trong cấu hình. Căn giữa mẫu theo **mặt ô đất**, kể cả khi
bông lúa cao hơn tâm ô. \`rois\` dùng tọa độ chuẩn hóa \`[trái, trên, phải, dưới]\`
theo ảnh Android. Dùng vùng đủ rộng cho toàn bộ mẫu, kể cả vùng quả sau thao tác.

| Chức năng | Tên mẫu (đuôi \`.png\`) |
| --- | --- |
| Cào | \`farm_screen\`, \`field_ripe\`, \`harvest_tool\`, \`field_empty\`, \`field_selected_ripe\` (viền sáng của ô khi liềm mở) |
| Gieo | \`farm_screen\`, \`field_empty\`, \`seed_menu\`, \`crop_menu_wheat\`, \`field_growing\`, \`field_selected_empty\` (viền sáng của ô trong menu gieo) |
| Bán | \`shop_stand\`, \`shop_screen\`, \`shop_slot_empty\`, \`shop_slot_wheat\`, \`shop_close\`, \`inventory_screen\`, \`inventory_wheat\`, \`inventory_close\`, \`wheat_selected\`, \`btn_price_max\`, \`price_max_reached\`, \`btn_put_sale\` |
| Bán dư với số lượng điều chỉnh | \`btn_quantity_minus\`, \`price_max_reached_any_quantity\` |
| Trạng thái shop cần ảnh bổ sung | \`shop_full\` chụp khi toàn bộ slot đang dùng đã đầy, với dấu hiệu phân biệt riêng. \`out_of_wheat\` chụp dấu hiệu hết lúa trong kho. Nếu không có ảnh phân biệt, bot ghi thiếu bằng chứng và đóng màn hình đã xác nhận khi có thể. |

Sau khi chạm ô, bot xác nhận công cụ và dùng các mốc ngoài ruộng để bù dịch
camera. Với vùng GUI đã khoanh, bot yêu cầu chọn lại khi ảnh lệch. Sau khi kéo,
bot chờ hình ảnh đích và chỉ đếm ô khi thấy trạng thái mới tại chính tâm ô đó.
\`debug/last_drag.png\` đánh dấu nguồn và đích của lượt kéo. Khi dừng vì thiếu bằng
chứng, xem \`debug/steps.jsonl\` và các ảnh \`*_before.png\` / \`*_after.png\`.

## Bán lúa dư

Đặt \`wheat_seed_reserve\` bằng số hạt cần giữ cho toàn bộ vùng định gieo. Bản cấu
hình này giữ tối thiểu 30 hạt. Bot còn giữ ít nhất số ô đã xác nhận trong lượt
gieo. Để vòng lặp đọc số lượng, cấu hình \`tesseract\` thành đường dẫn đến
\`tesseract.exe\` hoặc để \`tesseract\` trong PATH và hiệu chỉnh hai ROI chỉ chứa
số \`inventory_wheat_count\` và \`sale_wheat_quantity\`. ROI thứ nhất phải bao số
ngay cạnh ảnh lúa mì, ROI thứ hai bao lượng bán trên phiếu. Nếu chữ số không
đọc được hoặc hai ảnh cho kết quả khác nhau, bot bỏ qua lượt bán và ghi lý do.
Khi cần, bot hạ lượng đăng đến mức dư đã xác nhận rồi đọc lại trước khi gửi.
Nếu gửi thành công nhưng ảnh trả về chưa rõ, bot chỉ kiểm tra lại đúng slot
đã dùng. Khi quầy đầy có bằng chứng, bot đóng quầy và xác nhận trở lại farm.

Menu **8** của \`run.py\` và \`python sell_upgrade.py sell 1\` là lệnh bán theo
số slot người dùng chọn. Vòng tự động mới tự tính phần dư. GUI và CLI dùng mutex
Windows theo serial để chỉ một tác vụ điều khiển thiết bị. Nút **Dừng** chặn
lệnh ADB tiếp theo sau lệnh đang chạy.

## Kiểm thử

\`python -m compileall -q .\` kiểm tra cú pháp.
\`python -m unittest -v test_workflows\` chạy hồi quy offline.
