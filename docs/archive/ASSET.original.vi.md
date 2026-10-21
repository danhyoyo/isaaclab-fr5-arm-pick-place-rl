# FR5 với gripper — bộ asset riêng cho Isaac Lab

Đây là bản sao từ `D:\FR5_Reinforcement-learning\fr5_description`.
Bộ nguồn được giữ nguyên. Chỉ chọn bản `fr5_withgripper` cùng mesh cần thiết;
các biến thể FR5 khác và bản `new_gripper.urdf` chưa hoàn chỉnh không được đưa vào.

## File nên mở

`fr5_description/urdf/fr5_withgripper/fr5_arm.usda`

File này chứa các ghi đè vật lý lên USD nhập từ URDF. Giữ nguyên toàn bộ thư mục
`fr5_description` khi di chuyển. `fr5_withgripper.usd` là lớp nhập bên dưới, chưa
chứa đầy đủ thông số sửa trong lớp ghi đè.

## Những thay đổi

- Khối lượng 7 link tay tham khảo URDF chính thức FR5 V6 của FAIRINO:
  https://github.com/FAIR-INNOVATION/frcobot_ros2/blob/main/fairino_description/urdf/fairino5_v6.urdf
  Đọc ngày 21/10/2026. Lần lượt: 1.6185, 4.64, 10.08, 2.71, 1.56, 1.56, 0.36 kg.
- Giữ khối lượng gripper khai báo ban đầu: thân 1.4156404425 kg, mỗi ngón
  0.0589543897 kg. Tổng khối lượng được gán khoảng 24.06205 kg.
- Tính lại tâm khối lượng và tensor quán tính từ convex hull kín của collision mesh
  từng link, giả định mật độ đồng đều và chuẩn hóa theo khối lượng đã gán.
  URDF và USD dùng cùng tensor, USD lưu dưới dạng quán tính chính và trục chính.
- Drive của hai ngón: stiffness 2000 N/m, damping 20 N·s/m, lực tối đa 40 N,
  tốc độ tối đa 0.1 m/s. Mở = 0 m; đóng = 0.04 m cho **cả fj1 và fj2**.
- Drive tay: j1–j3 dùng stiffness/damping 2000/100; j4–j6 dùng 500/25.
  Mô-men tối đa 150 N·m và 28 N·m tương ứng. Đây là giá trị khởi đầu cho mô phỏng.
- Collider gripper được đặt trên mesh, dùng convex decomposition thay convex hull.
  Ma sát tĩnh/động gripper 0.8/0.6, restitution 0; cần đánh giá lại với vật cần gắp.
- Bỏ tham chiếu hình ảnh world rỗng, chuyển vật liệu OmniPBR sang UsdPreviewSurface
  để asset không phụ thuộc file MDL ngoài. Giữ màu diffuse; các hiệu ứng vật liệu khác
  có thể khác bản gốc.
- URDF sửa dùng đường dẫn mesh tương đối; bỏ axis/limit thừa trên khớp fixed.
  Drive, ma sát và cách phân rã collider là cấu hình riêng của USD/Isaac Lab, không
  được mang nguyên vẹn sang simulator khác bằng URDF.

## Giới hạn của thông số

**Đây là bộ thông số ước lượng cho mô phỏng, chưa phải mô hình động lực học đã
hiệu chuẩn của robot thật.** Hình học và hệ tọa độ bản này khác URDF FR5 V6 chính thức,
nên không chép trực tiếp tâm khối lượng và quán tính của V6. Các giá trị từ convex
hull bỏ qua kết cấu rỗng, motor, hộp số và phân bố vật liệu. Khối lượng gripper chưa
được cân thực tế. Có thể thay bằng số liệu CAD/nhà sản xuất trong tương lai.

Self-collision đang tắt như asset gốc. Cần xử lý cặp va chạm/link và kiểm chứng riêng
trước khi bật. Chưa đánh giá tỷ lệ gắp thành công, chưa tích hợp task RL pick-and-place
và không dùng checkpoint của cánh tay cũ.

## Chạy bằng môi trường Isaac Lab có sẵn

Trong PowerShell:

```powershell
Set-Location D:\IsaacProjects\fr5_arm
& D:\IsaacLab\.venv\Scripts\python.exe .\validate_asset.py
& D:\IsaacLab\.venv\Scripts\python.exe .\gripper_scene.py --viz none --cycles 2
& D:\IsaacLab\.venv\Scripts\python.exe .\gripper_scene.py --viz none --cycles 2 --gravity
& D:\IsaacLab\.venv\Scripts\python.exe .\check_contact.py --viz none
```

Xem robot đóng/mở liên tục trong cửa sổ:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\gripper_scene.py --viz kit --cycles 0
```

Scene nâng robot lên 0.6 m, không có vật gắp. Mặc định tắt trọng lực để đo độc lập
chuyển động ngón; `--gravity` bật lại. Kết quả hữu hạn được lưu dưới `verification/`.

`fr5_cfg.py` cung cấp `FR5_CFG` để dùng với `Articulation` của Isaac Lab trên máy này.
Đây là cấu hình robot, chưa phải môi trường học tăng cường.

## Kiểm chứng và nguồn gốc

- `parameters.json`: giá trị trước/sau và phương pháp ước lượng.
- `source_manifest.json`: SHA-256 các file nguồn đã sử dụng.
- `verification/static.json`: kiểm tra phụ thuộc, tên khớp, tensor quán tính và khe đầu ngón.
- `verification/motion*.json`: chỉ xuất hiện sau khi bài kiểm tra mô phỏng tương ứng thành công.
- `verification/contact.json`: hai ngón dừng khi ép vào khối hộp cố định 36 mm và mở lại.
- `fr5_description/urdf/fr5_withgripper.original.urdf`: bản URDF gốc để đối chiếu.

Khe đầu ngón tính tĩnh khoảng 80.08 / 40.08 / 0.08 mm khi hai khớp cùng ở
0 / 0.02 / 0.04 m. Đây là phép đo vùng đầu ngón, không phải chứng nhận tiếp xúc collider.

## Kết quả đã chạy ngày 21/10/2026

- Kiểm tra tĩnh: đạt; không còn file phụ thuộc chưa resolve.
- Hai chu kỳ đóng/mở không trọng lực: đạt; vị trí đóng mỗi ngón gần 0.04 m.
- Hai chu kỳ đóng/mở có trọng lực: đạt; sai số lớn nhất khoảng 0.289 mm.
- Va chạm với hộp cố định 36 mm: đạt; hai khớp dừng tại khoảng
  0.02068 / 0.02048 m thay vì đi xuyên tới 0.04 m, sau đó mở lại gần 0.
- Kiểm tra SHA-256: 43 file nguồn đã dùng không thay đổi.

Thử va chạm với hộp cố định chỉ xác nhận các ngón bị chặn bởi collider; chưa đo
lực tiếp xúc và chưa chứng minh gắp, nâng, giữ vật tự do thành công.
