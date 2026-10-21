# Huấn luyện FR5 gắp và đặt hộp

Dự án độc lập tại `D:\IsaacProjects\fr5_arm`. Code, asset, log và checkpoint đều nằm
trong thư mục này; không import hoặc đọc file từ dự án `pick_and_place`.
Chỉ dùng chung môi trường Python/Isaac Lab đã cài trong `D:\IsaacLab`.

## Chạy nhanh

Trong PowerShell:

```powershell
cd D:\IsaacProjects\fr5_arm
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py check
& D:\IsaacLab\.venv\Scripts\python.exe .\demo.py --viz kit --num_envs 1
```

`demo.py` dùng các waypoint để kiểm tra gắp–nâng–chuyển–hạ–thả–rút đầu kẹp bằng
chính giao diện hành động của RL. Chỉ reset vật lúc bắt đầu episode, không gắn vật
vào tay hoặc dịch chuyển vật trong lúc thực hiện. Đây là bộ điều khiển kiểm tra,
không phải policy đã huấn luyện. Demo kết thúc hữu hạn và báo lỗi nếu không thành công.

Huấn luyện thử ngắn để kiểm tra toàn bộ luồng:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py train --num_envs 8 --max_iterations 2 --headless --run_name smoke
```

Bắt đầu một lần huấn luyện mới:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py train --num_envs 128 --max_iterations 5000 --headless --run_name fr5_cube30
```

Tăng số môi trường sau khi theo dõi VRAM và tốc độ. Bỏ `--headless` để mở cửa sổ
mô phỏng. Có thể dùng `uv run --project D:\IsaacLab --no-sync python ...` thay cho
đường dẫn Python trực tiếp. Không cần cài lại thư viện hoặc tạo môi trường mới.

## Chạy policy và đánh giá

Chạy checkpoint mới nhất của FR5:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py play --num_envs 4 --checkpoint latest
```

Hoặc chỉ định một checkpoint cụ thể:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py play --num_envs 4 --checkpoint .\logs\rsl_rl\fr5_pick_place\TEN_LAN_CHAY\model_4999.pt
```

Thay `TEN_LAN_CHAY` và tên model bằng file thật đã sinh ra. Thêm `--steps 650 --headless`
để chạy hữu hạn. Policy cũ của cánh tay cumotion không tương thích với task này.

Đánh giá tỷ lệ thành công trên các episode hoàn chỉnh:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\evaluate.py --checkpoint .\logs\rsl_rl\fr5_pick_place\TEN_LAN_CHAY\model_4999.pt --episodes 32 --num_envs 8 --seed 123 --viz none
```

Kết quả ghi vào `evaluation/latest.json`, gồm số thành công, tỷ lệ thành công,
số lần rơi vật và hết thời gian. Nên đánh giá nhiều seed sau khi train đủ lâu;
checkpoint của lần smoke chỉ chứng minh chương trình chạy được.

Tiếp tục huấn luyện từ checkpoint FR5:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe .\run.py train --num_envs 128 --max_iterations 2000 --headless --checkpoint .\logs\rsl_rl\fr5_pick_place\TEN_LAN_CHAY\model_4999.pt
```

Tham số số vòng cập nhật khi resume tuân theo backend RSL-RL đang cài; kiểm tra
iteration trong log để xác định tiến độ thực tế.

## Bài toán đang dùng

- Robot: USD FR5 có hai ngón trượt đã sửa ở bước chuẩn bị asset trước đó.
- Vật: hộp 30 × 30 × 30 mm, khối lượng 30 g, tạo bằng Cuboid trong code.
  Không cần file USD hoặc texture mới. Khe mở đầu ngón khoảng 80 mm.
- Bàn: mặt bàn ở z = 0.10 m; robot được cố định với gốc cao 0.10 m để tránh
  collider của link vai chạm sàn. Tư thế bắt đầu là khuỷu tay hướng lên.
- Vị trí hộp quanh x = -0.45, y = -0.12 m; đích quanh x = -0.45, y = +0.12 m.
  Mỗi lần reset ngẫu nhiên x ± 0.04 m và y ± 0.02 m, hai vùng tách nhau.
- 4 hành động: dx/dy/dz cập nhật mục tiêu đầu kẹp tối đa 6 mm mỗi bước,
  và một lệnh gripper (`>= 0` mở, `< 0` đóng) điều khiển cả `fj1`, `fj2`.
- Hướng kẹp được IK giữ từ trên xuống, yaw 90 độ. Task này không học xoay vật tùy ý.
- Điều khiển 50 Hz, vật lý 200 Hz; episode tối đa 12 giây.
- PPO: MLP actor/critic 256–128–64, chuẩn hóa observation, 32 bước rollout/môi trường.
- Quan sát dùng trạng thái mô phỏng: khớp, vị trí/hướng đầu kẹp, hộp, đích,
  vận tốc hộp, sai lệch mục tiêu điều khiển, lịch sử nâng và thời gian ổn định.
  Đây là policy dùng trạng thái, chưa dùng camera.

Reward hỗ trợ tiếp cận, kẹp, nâng, vận chuyển, hạ, thả, rút đầu kẹp và thành công.
Phạt thay đổi hành động và vận tốc khớp. Điều kiện thành công bắt buộc:

1. Hộp từng được nâng trên vị trí nghỉ ít nhất 45 mm trong lúc ở gần đầu kẹp đang đóng.
2. Tâm hộp cách đích dưới 20 mm theo XY, sai lệch độ cao dưới 8 mm.
3. Tốc độ hộp dưới 0.06 m/s, cả hai ngón đã mở dưới 4 mm.
4. Đầu kẹp cách hộp trên 55 mm và toàn bộ điều kiện giữ liên tục 0.30 giây.

Đẩy hộp trên bàn tới đích không được tính thành công. Kết thúc episode khi thành
công, vật ra khỏi vùng cho phép hoặc hết thời gian. Mọi tọa độ tính theo môi trường
riêng, không theo vị trí toàn cục của từng bản sao.

## Các file chính

```text
fr5_arm/
├── run.py                       # check / train / play, log luôn theo thư mục dự án
├── demo.py                      # Kiểm tra gắp–đặt vật lý bằng waypoint
├── evaluate.py                  # Đánh giá checkpoint theo episode
├── fr5_cfg.py                   # Cấu hình robot cơ sở
├── fr5_description/             # Asset FR5 và mesh độc lập
├── fr5_pick_place/
│   ├── __init__.py              # Đăng ký hai task Gymnasium
│   ├── constants.py             # Kích thước, vị trí, tư thế, giới hạn
│   ├── env_cfg.py               # Scene, observation, reward, reset, termination
│   ├── actions_cfg.py           # Cấu hình hành động, nạp implementation sau simulator
│   ├── actions.py               # Cartesian target + IK giữ hướng kẹp
│   ├── mdp.py                   # Logic task với tensor GPU
│   ├── task_logic.py            # Lịch sử episode và kiểm tra thành công
│   └── ppo_cfg.py               # Tham số PPO
├── tests/
│   ├── test_task_logic.py       # Chống thành công giả, reset từng môi trường
│   └── smoke_env.py             # Chạy qua timeout/reset của một episode
├── logs/rsl_rl/fr5_pick_place/   # Thư mục riêng cho mỗi lần train
├── evaluation/                  # Kết quả đánh giá policy
└── verification/                # Kết quả kiểm tra asset, demo và môi trường
```

`env_cfg.py` tăng độ cứng/giảm chấn tay lên 8000/200 và cổ tay 2000/80 cho task
chịu trọng lực; giới hạn mô-men vẫn 150 N·m cho j1–j3, 28 N·m cho j4–j6. Không
đổi cấu hình scene gripper độc lập có sẵn. Thông số vật lý của asset vẫn là ước lượng
như ghi trong `README.md` và `parameters.json`, chưa hiệu chuẩn với robot thật.

Kiểm tra bổ sung:

```powershell
& D:\IsaacLab\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_*.py
& D:\IsaacLab\.venv\Scripts\python.exe .\tests\smoke_env.py --viz none
& D:\IsaacLab\.venv\Scripts\python.exe .\demo.py --viz none --num_envs 4 --seed 42
```

TensorBoard có thể đọc `logs/rsl_rl/fr5_pick_place`. Theo dõi các mục reward và
`Episode_Termination/success`, sau đó dùng `evaluate.py` để đo thành công trên một
số episode xác định. Reward tăng không tự chứng minh policy đã học gắp–đặt.

## Đã kiểm tra ngày 21/10/2026

- Hai task train/play nạp cấu hình thành công; 3 kiểm tra logic đạt.
- Demo vật lý seed 42: **4/4 môi trường thành công**, hộp được nâng khoảng 14 cm,
  chuyển sang đích, thả và rút đầu kẹp; không gắn/teleport vật trong thao tác.
- Môi trường chạy 650 bước: 4 môi trường, 4 lần reset, 47 observation và 4 action;
  observation/reward hữu hạn, không báo thành công giả khi robot đứng yên mở kẹp.
- PPO: 8 môi trường, 2 vòng cập nhật, lưu `model_0.pt` và `model_1.pt` trong
  `logs/rsl_rl/fr5_pick_place/2026-10-21_09-24-14_smoke_fr5/`.
- `play --checkpoint latest --steps 650`: nạp checkpoint, xuất TorchScript/ONNX
  và chạy policy qua 650 bước thành công.
- Đánh giá checkpoint smoke với seed 123: **0/2 episode thành công**, 2 timeout.
  Đây chưa phải policy đã học được nhiệm vụ; cần chạy huấn luyện dài hạn.

Kết quả chi tiết: `verification/training_checks.json`,
`verification/scripted_pick_place_seed42.json`, `verification/environment.json`
và `evaluation/latest.json`.
