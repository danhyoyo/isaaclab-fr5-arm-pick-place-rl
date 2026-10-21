# Custom arm: pick and place

Dự án chạy riêng tại `D:\IsaacProjects\pick_and_place`, dùng Python và thư viện
trong `D:\IsaacLab\.venv`. Không chép hoặc thay thế file trong mã nguồn IsaacLab.
Không cần cài lại IsaacLab, tạo thêm venv, hay kích hoạt venv trước khi chạy.

## Phiên bản trên máy

- `D:\IsaacLab\VERSION`: 3.0.0 (metadata package trong venv: 27.2.0).
- Isaac Sim 6.2.0.0, Python 3.12, RSL-RL 5.5.1.
- Physics: Isaac Sim PhysX; quaternion trong Python dùng `(x, y, z, w)`.

Code đã được chuyển sang API của checkout hiện tại; không bảo đảm tương thích
ngược với IsaacLab 2.x trong tutorial.

## Chạy trong PowerShell

Có thể dùng `uv run` với venv đã có của IsaacLab. Cú pháp sau đã được kiểm tra:

```powershell
Set-Location D:\IsaacProjects\pick_and_place
uv run --project D:\IsaacLab --no-sync python .\run.py check
uv run --project D:\IsaacLab --no-sync python .\run.py train --num_envs 256 --max_iterations 4000 --headless
uv run --project D:\IsaacLab --no-sync python .\run.py play --num_envs 8 --checkpoint .\logs\rsl_rl\franka_lift_place_custom_arm\2026-05-25_18-19-11\model_3999.pt
```

`--project D:\IsaacLab` chọn môi trường của IsaacLab; không di chuyển code project.
`--no-sync` dùng nguyên các thư viện đang cài, không đồng bộ lại dependencies.
Các lệnh gọi trực tiếp `python.exe` bên dưới là cách tương đương.

```powershell
Set-Location D:\IsaacProjects\pick_and_place
```

Kiểm tra task, cấu hình và đường dẫn asset, không khởi động simulator:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py check
```

Chạy thử với 8 môi trường, 2 vòng cập nhật PPO:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py train --num_envs 8 --max_iterations 2 --headless
```

Huấn luyện dài hơn, bắt đầu với 256 môi trường:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py train --num_envs 256 --max_iterations 4000 --headless
```

Tăng `--num_envs` sau khi kiểm tra tốc độ và VRAM. Giá trị 4096 trong video
không phải yêu cầu bắt buộc. Thay `--headless` bằng `--viz kit` để xem cửa sổ
mô phỏng khi train.

Chạy policy đi kèm tutorial với 8 môi trường:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py play --num_envs 8 --checkpoint .\logs\rsl_rl\franka_lift_place_custom_arm\2026-05-25_18-19-11\model_3999.pt
```

`play` mặc định mở cửa sổ Isaac Sim (`--viz kit`). Nhấn Ctrl+C trong terminal
để dừng. Dùng `--headless` nếu muốn tắt cửa sổ.

Để chạy policy vừa train, thay `--checkpoint` bằng đường dẫn `model_*.pt`
trong thư mục log mới. Không cần di chuyển thư mục `logs`.

## Cấu trúc

- `run.py`: nạp package riêng, đăng ký task, gọi train/play của IsaacLab.
- `lift_place_env_cfg.py`: scene, observations, rewards, resets, PhysX.
- `custom_arm/joint_pos_place_env_cfg.py`: robot, gripper, cube, vùng đích.
- `custom_arm/agents/rsl_rl_ppo_cfg.py`: cấu hình PPO.
- `mdp/`: các hàm riêng của task, không thay thế `mdp` của IsaacLab.
- `custom_arm/cumotion_arm/`, `cube_5cm/`: các asset USD. Giữ cả file USD phụ
  và textures khi sao chép project.

Asset được tìm theo vị trí code, không dùng biến môi trường `HOME`.
Log và checkpoint mới luôn nằm dưới `logs/rsl_rl/franka_lift_place_custom_arm/`
trong project, kể cả khi gọi `run.py` từ thư mục khác. Khi `play`, IsaacLab có
thể cập nhật file policy xuất ra trong thư mục `exported` của checkpoint.

## Môi trường chạy

Isaac Sim cần quyền ghi các thư mục cache/cấu hình của nó. Chạy từ PowerShell
thông thường nếu sandbox chặn quyền này. Lần đầu có thể mất lâu để tải asset,
extension và biên dịch GPU kernels.

Launcher chuyển `--headless` thành `--viz none` của IsaacLab hiện tại.
Không dùng lệnh Linux `source .../bin/activate` hay đường dẫn
`scripts/reinforcement_learning/rsl_rl/train.py` trong video.

## Đã kiểm tra trên máy này

- Hai task train/play nạp và kiểm tra cấu hình thành công.
- Train 8 môi trường trên GPU qua 2 vòng PPO, lưu `model_0.pt`, `model_1.pt`.
- Policy tutorial nạp, export và chạy 300 bước với 8 môi trường; kiểm tra gồm
  reset sau một episode 250 bước. Đây là kiểm tra chạy được, chưa phải đánh giá
  tỷ lệ gắp/đặt thành công sau khi chuyển phiên bản.

Chạy lại kiểm tra playback hữu hạn (tự đóng sau 300 bước, không mở cửa sổ):

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\tests\smoke_play.py
```

Bài kiểm tra dùng bản sao checkpoint trong `logs/verification/pretrained/`,
giữ nguyên checkpoint và các file export gốc của tutorial.

## Gripper: đã sửa góc đóng sai dấu

### Scene chỉ để xem đóng/mở

```powershell
uv run --project D:\IsaacLab --no-sync python .\gripper_scene.py
```

Scene này chỉ có robot, sàn và đèn, không có cube hay RL task. Camera nhìn gần
gripper. Các khớp tay giữ nguyên vị trí; gripper luân phiên đóng -60 độ trong
2 giây rồi mở 0 độ trong 2 giây, lặp lại cho tới khi đóng cửa sổ hoặc nhấn Ctrl+C.
Trọng lực được tắt trong scene xem thử để tách riêng chuyển động gripper.
Thêm `--cycles 2` để tự dừng sau hai chu kỳ hoặc `--viz none --cycles 2` để kiểm
tra tự động. Góc đóng dùng chung hằng số với task train; góc mở là 0 độ.

### Lỗi góc đóng trong cấu hình cũ

USD gốc giới hạn `cumotion_arm_joint_hand` từ 0 đến +60 độ, nhưng cấu hình
tutorial ra lệnh đóng -60 độ. Sửa lệnh thành +60 độ chỉ làm khớp chuyển động,
nhưng sai hướng kẹp: đo vị trí hai tâm đầu càng trong PhysX thấy khoảng cách
tăng từ 64.6 mm ở 0 độ lên 97.2 mm ở +60 độ. Kiểm tra chuyển động ban đầu đã
bỏ sót điểm này.

Bản sửa hiện tại dùng `cumotion_arm_gripper_fixed.usda` làm lớp ghi đè nhẹ
lên robot gốc, đổi giới hạn thành [-60, 0] độ. File USD gốc và các mesh được
giữ nguyên. Góc đóng là -60 độ; đo thực tế đạt gần -60 độ và khoảng cách tâm
hai đầu càng giảm xuống 18.3 mm. Mở lại 0 độ trở về khoảng cách 64.6 mm.
Các số này là khoảng cách giữa tâm hai vùng đầu càng, không phải chiều rộng
khe hở chính xác hoặc phép đo khả năng giữ vật. Không tăng hành trình ngoài
60 độ khi chưa kiểm tra va chạm/hình học.

Kiểm tra đóng/mở độc lập với policy, có cửa sổ mô phỏng:

```powershell
uv run --project D:\IsaacLab --no-sync python .\check_gripper.py --viz kit
```

Đổi `--viz kit` thành `--viz none` để chạy tự động không mở cửa sổ. Bài kiểm
tra nâng robot khỏi sàn để đo riêng chuyển động khớp, thử lệnh âm/dương/mở,
ghi kết quả vào `logs/verification/gripper.json`, rồi tự kết thúc.
Kiểm tra có điều kiện bắt buộc: góc đóng trong giới hạn, khớp đạt góc đóng,
khoảng cách hai đầu càng giảm hơn một nửa, và khớp mở lại về 0 độ. Đây chưa
phải phép đo tỷ lệ gắp cube thành công.

Các checkpoint train trước khi sửa đã học với lệnh đóng ngoài giới hạn hoặc
với hướng đóng/mở bị đảo;
không thể coi đó là policy đã học gắp bằng gripper hoạt động đúng. Nên bắt đầu
lần train mới, không resume checkpoint cũ:

```powershell
uv run --project D:\IsaacLab --no-sync python .\run.py train --num_envs 256 --max_iterations 4000 --headless --run_name gripper_geometry_fixed
```

Lần train mới lưu vào thư mục mới có hậu tố `gripper_geometry_fixed`; checkpoint cũ
được giữ nguyên. Kiểm tra mô phỏng thành công trước đây chỉ chứng minh chương
trình chạy được, không phát hiện lỗi dấu góc gripper này.
