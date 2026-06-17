# QA Ảnh Auto-Label (YOLO Training Data)

> Cập nhật: 2026-06-17  
> Dataset: `custom_tracking_system/data/auto_label/`  
> Script QA: `qa_autolabel.py` (project root)

---

## Tổng quan dataset

| Thống kê | Giá trị |
|---------|---------|
| Tổng ảnh | 15,957 |
| Tổng bbox | 623,329 |
| Nguồn | CCTV thực tế Hà Nội (Phố Huế, Giảng Võ, Cửa Nam, Lê Duẩn...) |
| Format label | YOLO txt (cx cy w h normalized) |
| Classes | person, car, motorcycle, bus, truck |
| Videos gốc | 124 video (120 ok, 4 skip do cannot_open) |

---

## Bước 1 — Kiểm tra format label files ✅ HOÀN THÀNH

**Script**: `qa_autolabel.py`  
**Chạy**: `python qa_autolabel.py` từ project root

### Kết quả

| Chỉ số | Kết quả | Đánh giá |
|--------|---------|---------|
| Empty files | 0 | Tốt |
| Format errors (wrong cols) | 0 | Tốt |
| Bbox out-of-bounds | 0 | Tốt |
| Parse errors | 0 | Tốt |
| Invalid class id | 0 | Tốt |
| Tiny bbox (area < 0.0005) | 21,388 (3.4%) | Chấp nhận được |

**Kết luận bước 1**: Dataset **sạch hoàn toàn về format**. Không cần xử lý thêm về mặt cấu trúc file.

### Phân tích tiny bbox

| Kích thước | Số lượng | Ghi chú |
|-----------|---------|--------|
| area < 0.0001 | 1 | Loại bỏ khi train (quá nhỏ, nhiễu) |
| 0.0001 – 0.0003 | 3,595 | Giữ — xe/người ở xa, CCTV góc rộng |
| 0.0003 – 0.0005 | 17,792 | Giữ — bình thường |

Tiny bbox 3.4% là **bình thường** với camera CCTV góc rộng — phương tiện ở xa tự nhiên nhỏ. Không cần lọc thêm (YOLO tự bỏ qua anchor quá nhỏ trong quá trình train).

### Class distribution

| Class | Số bbox | Tỉ lệ |
|-------|---------|--------|
| motorcycle | 435,118 | 69.8% |
| car | 153,233 | 24.6% |
| person | 20,318 | 3.3% |
| truck | 8,837 | 1.4% |
| bus | 5,823 | 0.9% |

**Imbalance đáng chú ý**: motorcycle >> bus/truck (~75:1).  
**Giải pháp khi train**: thêm `--cls 1.5` hoặc dùng weighted sampling — không cần sửa dataset.

---

## Bước 2 — Spot-check ảnh bằng mắt ⬜ CẦN LÀM

**Mục đích**: Phát hiện bbox sai nghiêm trọng (label nhầm class, bbox lệch hoàn toàn).  
**Số lượng**: Kiểm tra ~30–50 ảnh ngẫu nhiên từ 3–4 nguồn video khác nhau.

**Cách làm nhanh**: Mở thư mục `data/auto_label/preview/` (có preview grids sẵn, mỗi grid ~133 ảnh có bbox vẽ sẵn) và xem qua.

**Những gì cần chú ý**:
- [ ] Bbox có bám sát xe/người không, hay lệch nhiều?
- [ ] Xe máy bị nhầm thành car và ngược lại?
- [ ] Người đi bộ bị miss (không có bbox dù có người trong ảnh)?
- [ ] Bbox bị cắt ở mép ảnh chiếm >50% diện tích?

**Tiêu chí pass**: Nếu >85% bbox trông hợp lệ trong spot-check → proceed to train.

---

## Bước 3 — Xử lý class imbalance ⬜ QUYẾT ĐỊNH KHI TRAIN

Không cần sửa dataset. Xử lý tại thời điểm train YOLO:

```bash
# Trong train_yolo_detector.py, thêm tham số:
# cls=1.5  (tăng class loss weight cho bus/truck)
# hoặc dùng augmentation mạnh hơn cho minority classes
python scripts/train/train_yolo_detector.py \
    --data data/visdrone_carla_vn.yaml \
    --model yolo11m.pt \
    --epochs 50 --batch 16 --device 0 --half
```

Nếu sau train mAP của bus/truck < 0.3: cân nhắc undersample motorcycle xuống ~150K.

---

## Bước 4 — Kiểm tra 4 video bị skip ⬜ LOW PRIORITY

4 video `NB-LC` có `status: "skip", reason: "cannot_open"`. Đây là ~3% tổng số video.  
**Tác động**: Không đáng kể — 120/124 video đã được xử lý thành công.  
**Hành động**: Bỏ qua, không cần xử lý thêm.

---

## Tóm tắt trạng thái QA

| Bước | Trạng thái | Kết quả |
|------|-----------|---------|
| 1. Kiểm tra format label | ✅ Done | Sạch, 0 lỗi |
| 2. Spot-check ảnh | ⬜ Pending | Cần ~30 phút thủ công |
| 3. Xử lý class imbalance | ⬜ Khi train | Dùng cls weight |
| 4. Video bị skip | ✅ Bỏ qua | 4/124, không đáng kể |

**Quyết định**: Dataset đủ điều kiện để upload Kaggle và train sau khi hoàn thành bước 2.
