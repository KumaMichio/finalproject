# Train YOLO held-out (Xã Đàn) — hướng dẫn chuyển máy GPU

Mục tiêu: train lại YOLO11s NHƯNG loại hẳn ngã tư **Xã Đàn** khỏi tập train,
rồi đo mAP trên 30 ảnh nhãn tay Xã Đàn (đã có sẵn) → con số **test
location-disjoint** (ngã tư model chưa từng thấy), thay cho số 93,2% in-domain
hiện tại.

Chạy MỌI lệnh dưới đây từ **thư mục gốc** `finalproject/`.

---

## 1. Cần copy sang máy GPU

| Đường dẫn | Dung lượng | Bắt buộc |
|---|---|---|
| `custom_tracking_system/scripts/` | nhỏ | ✅ (chứa script đã sửa) |
| `custom_tracking_system/data/auto_label/` (images + labels) | **4,05 GB** | ✅ dữ liệu train |
| `custom_tracking_system/data/eval_xadan_manual/` | ~vài MB | ✅ nhãn tay để eval |
| `requirements.txt` | nhỏ | ✅ |

KHÔNG cần copy: `weights/` (nặng, model nền `yolo11s.pt` tự tải ~18MB),
CARLA, các dataset khác.

---

## 2. Môi trường trên máy GPU (GTX 1050 Ti / RTX 4060)

Cài PyTorch bản CUDA (KHÔNG dùng requirements_cpu.txt):

```bash
python -m venv venv_tracking
venv_tracking\Scripts\activate            # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
```
Kiểm tra GPU nhận đúng:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Ba bước chạy

### B1 — Dựng dataset đã loại Xã Đàn (chỉ copy file, không cần GPU, ~1-2 phút)
```bash
python custom_tracking_system/scripts/train/merge_detection_datasets.py \
    --datasets auto=custom_tracking_system/data/auto_label \
    --exclude-prefix "Ngã_tư_Xã_Đàn" \
    --out custom_tracking_system/data/auto_label_heldout
```
Kỳ vọng in ra: `train ~13346`, `val ~2355` (tổng 15701 = 15957 − 256 ảnh Xã Đàn).

### B2 — Train lại
GTX 1050 Ti (4GB) — cấu hình tiết kiệm, ~5–10 giờ (để chạy qua đêm):
```bash
python custom_tracking_system/scripts/train/train_yolo_detector.py \
    --data custom_tracking_system/data/auto_label_heldout.yaml \
    --epochs 30 --imgsz 512 --batch 8 --half --device 0 \
    --out custom_tracking_system/weights/yolo11s_vn_heldout.pt
```
- Nếu OOM (hết VRAM): hạ `--batch 4`.
- RTX 4060: có thể `--epochs 50 --imgsz 640 --batch 16` (~1–2 giờ).

### B3 — Eval trên nhãn tay Xã Đàn (ngã tư held-out)
```bash
python custom_tracking_system/scripts/run_hanoi_manual_val.py \
    --eval-dir custom_tracking_system/data/eval_xadan_manual \
    --weights custom_tracking_system/weights/yolo11s_vn_heldout.pt \
    --name yolo11s_vn_heldout_xadan \
    --note "Xã Đàn HELD-OUT khỏi train — test location-disjoint, nhãn người"
```
Kết quả (mAP@0.5, confusion matrix, PR) ghi ra:
`docs/assets/yolo11s_vn_heldout_xadan/metrics.json`

---

## 4. Đọc kết quả

- Con số mAP@0.5 mới = độ chính xác trên **ngã tư model CHƯA thấy khi train**.
- So sánh với 93,2% cũ (in-domain, dính train) để nói về mức tổng quát hóa.
- `yolo11s_vn_heldout.pt` CHỈ để lấy số báo cáo. Demo cuối vẫn dùng
  `yolo11s_vn.pt` (train full data) — giữ 2 checkpoint tách biệt.
