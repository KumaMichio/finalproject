# post_accident_prediction.md — Điều chỉnh sau nghiên cứu: Dự đoán hướng đi của xe sau tai nạn/sự cố

> Tài liệu này tổng kết các thay đổi đề xuất cho project sau một mạch nghiên cứu
> (15/06/2026) về ứng dụng trajectory prediction, dự đoán dài hạn, cơ sở xác
> suất, đặc thù giao thông Việt Nam, và các dataset/model SOTA. Đây là tài liệu
> ĐỊNH HƯỚNG — bổ sung cho `MUST.md`, chưa code.
>
> Mục tiêu mới làm rõ: **dự đoán hướng đi của các vehicle SAU khi gây tai nạn
> hoặc sự cố** (không chỉ luồng bình thường như MUST.md mô tả).

---

## 0. Bối cảnh & nguyên tắc lọc

Nghiên cứu đẩy theo hướng SOTA (Trajectron++, AgentFormer, diffusion) + dataset
thật quy mô lớn (Argoverse/nuScenes/Waymo). Project bị ràng buộc bởi:

- Môi trường **CARLA-lite hanoi_district3** (OpenDRIVE standalone, đã có).
- Mục tiêu **"dễ chuyển từ sim sang đời thực"**.

### Cấu hình phần cứng (cập nhật) — 3 option, demo chính trên RTX 4060

| GPU | VRAM / Hệ | Vai trò |
|---|---|---|
| **RTX 4060** | 8GB GDDR6, Ada, CUDA+Tensor | **CHÍNH** — cả train lẫn demo. Hợp stack PyTorch/CUDA/CARLA nhất |
| GTX 1050Ti | 4GB, Pascal, CUDA | Fallback / chứng minh "chạy được trên máy yếu" |
| RX 6700XT | 12GB, AMD RDNA2 | VRAM nhiều nhất nhưng kẹt ecosystem (ROCm Windows non, DirectML chậm) → **không dùng cho AI stack**; chỉ CARLA render được |

**Hệ quả của RTX 4060 (so với giả định cũ 1050Ti trong MUST §7):**

1. **Mở khóa blocker** — fine-tune YOLO local được (project_context.md ghi
   "cần GPU NVIDIA, máy hiện tại chỉ có 1050Ti"); 4060 chính là GPU đó → công
   thức §3.1 chạy được.
2. **Predictor nâng từ "chỉ nhẹ" → "trung bình"**: Trajectron++ /
   Social-STGCNN inference + train offline khả thi trên 4060.
3. **CARLA + pipeline chạy đồng thời mượt hơn** (CARLA Low ~2–3GB, còn ~5GB
   cho AI).

→ Nguyên tắc 3 mức (theo budget VRAM 8GB):
- **On-device demo** (4060 + CARLA chạy cùng): pipeline nhẹ + nhánh vật lý +
  Trajectron++/Social-STGCNN inference (đủ ~5GB còn lại).
- **Offline train/benchmark** (4060, tắt CARLA, full 8GB): Trajectron++,
  AgentFormer, YOLO fine-tune.
- **Để dành / KHÔNG real-time cùng CARLA**: diffusion (MID/LED — denoise lặp,
  chậm), AgentFormer đầy đủ → chỉ benchmark offline.
- **1050Ti fallback**: chỉ pipeline nhẹ goal-conditioned (kế hoạch MUST gốc).

---

## 1. Điểm khái niệm cốt lõi — vì sao kế hoạch MUST cần điều chỉnh

Kế hoạch MUST.md §3.2 dùng predictor **goal-conditioned + snap lane centerline**
(dựa vào lane graph từ OpenDRIVE). Nghiên cứu chỉ ra 2 chỗ kế hoạch này **gãy**:

1. **Xe máy VN "chảy như chất lỏng"** — lane gần như vô nghĩa, luồn lách tự do.
   Snap lane *quá gò bó* cho xe máy.
2. **Xe sau tai nạn** chuyển động theo **vật lý/quán tính** (văng, mất lái, xoay)
   — ra khỏi lane. Snap lane *sai hẳn*. Prior "luồng bình thường" học từ dữ liệu
   normal sẽ dự đoán sai chuyển động erratic này.

Chỗ (2) chính là **lõi mục tiêu mới**. Vì vậy không thể dùng MỘT predictor
lane-snapping cho mọi trường hợp — cần tách nhánh theo ngữ cảnh.

---

## 2. Về model AI

Giữ nguyên xương sống: YOLO11m → ByteTrack → OSNet → global ID →
`incident_detector`. Thay đổi nằm ở **tầng dự đoán**, tách 3 nhánh theo ngữ cảnh:

| Ngữ cảnh | Model đề xuất | Lý do |
|---|---|---|
| Xe **bình thường** (ô tô/bus/tải), horizon dài | Goal-conditioned + snap lane (kế hoạch MUST §3.2) | Bám lane đúng với xe 4 bánh tuân luật |
| **Xe máy** trong luồng dày | Goal-conditioned nhưng **giảm trọng số lane**, tăng nhánh tự do/tương tác | Lane gần vô nghĩa, đa phương thức cao |
| **Xe sau tai nạn** (lõi mới) | **Nhánh vật lý/động học ngắn hạn 2–3s**, ưu tiên động lực gần nhất, **KHÔNG** snap lane | Chuyển động erratic do quán tính/mất lái |
| Xe **lân cận** tai nạn | Goal-conditioned + vị trí tai nạn làm **obstacle** | Vẫn trên đường, chỉ né/dừng |

Bổ sung cụ thể:

- **Nhánh hybrid vật lý + học máy cho hậu tai nạn (MỚI)** — kích hoạt khi
  `incident_detector` báo va chạm. Đây là điểm khác biệt chính của project so
  với mọi predictor "luồng bình thường".
- **Tương tác (interaction)** — `modules/trajectory_gru.py` đã có sẵn feature
  K-nearest-neighbor (`NUM_NEIGHBORS=3`); tận dụng/mở rộng nhẹ (social-pooling)
  thay vì nhét GNN/Transformer nặng.
- **Accident trigger** — giữ **rule-based** (bbox chồng lấn + thay đổi vận
  tốc/gia tốc đột ngột), vốn đã có trong `modules/incident_detector.py`
  (PREDICTED_COLLISION, SUDDEN_STOP). Accident-LSTM / DoTA-anomaly để dành làm
  phần nâng cao.
- **Xác suất có hiệu chỉnh (calibration)** — mạng nơ-ron thường overconfident;
  thêm temperature scaling cho softmax goal. Rẻ, nên làm.
- **Đa phương thức** — output top-K path kèm xác suất, mỗi điểm convert lat/lon.
- **Khả thi trên RTX 4060 (8GB)**: Trajectron++, Social-STGCNN — inference
  on-device cùng CARLA + train offline. Cân nhắc dùng làm nhánh predictor
  "trung bình" thay vì chỉ benchmark.
- **Chỉ benchmark offline / KHÔNG real-time cùng CARLA**: AgentFormer đầy đủ,
  diffusion (MID/LED — denoise lặp, chậm), occupancy-grid cảnh đông xe máy,
  TrafficPredict/MCENET.

---

## 3. Về dataset để fine-tune

### 3.1 YOLO detector — đổi hướng quan trọng

Detector hiện fine-tune trên **ảnh render CARLA** (carla6, `yolo11m_vn.pt`) →
đây chính là nguyên nhân domain gap (Recall≈0% trên ảnh thật, xem `error.md`).
Để đạt mục tiêu sim→real, phải fine-tune trên **dataset camera giám sát THẬT**.

**Quyết định biến thể (để tăng FPS) — `yolo11m` → `yolo11s`:**
- Chọn **`yolo11s`** (KHÔNG nano): ~2× FPS, mAP chỉ giảm ~4 điểm. Nano nhanh
  ~3× nhưng giảm ~12 điểm → dễ sót xe máy nhỏ/bị che, mà detection là **nút
  thắt** của cả pipeline (track + predict chỉ tốt bằng detection đầu vào).
- **Đã CHỐT `yolo11s`** (không cần đo nữa). Lưu ý nguyên nhân FPS: demo trước
  đây chỉ đạt **10 FPS**, nhưng đó là **trần CARLA synchronous mode** (`config/
  camera_config.yaml`: `synchronous_mode: true`, `fixed_delta_seconds: 0.1` →
  10 Hz), KHÔNG phải do `yolo11m` chậm. Vì vậy lợi ích thật của `yolo11s`
  không phải "vượt 10 FPS trong CARLA" mà là:
  1. **Video thật** (không có trần 10 Hz) → FPS thật tăng thực sự.
  2. **Giải phóng VRAM/compute** trên 4060 cho 2 tải MỚI chạy đồng thời:
     probabilistic predictor + nhánh vật lý hậu tai nạn — đây là lợi ích chính.
  3. Cho phép hạ `fixed_delta_seconds` → 0.05 (20 Hz) mà pipeline vẫn kịp, nếu
     cần sim mượt hơn.
- **GỘP 1 lần train**: đổi biến thể KHÔNG transfer weight từ `yolo11m_vn.pt`
  được → phải train lại từ pretrained `yolo11s.pt`. Vì §3.1 cũng cần fine-tune
  lại trên data thật → làm chung 1 lần: `yolo11s.pt` (COCO) → fine-tune theo
  công thức 4 tầng dưới đây. ĐỪNG fine-tune `m` trên data thật rồi mới hạ `s`
  (phí công + mất chất lượng qua 2 lần).

**Mâu thuẫn cốt lõi — góc nhìn ↔ mật độ xe máy** (không bộ công khai nào có cả 2):

| Trục | Dataset khớp | Vấn đề |
|---|---|---|
| Góc nhìn CCTV trên cao (giống Mỗ Lao z=6) | VisDrone, MIO-TCD, UA-DETRAC, AI City/CityFlow, **ITD** | Phương Tây/TQ/Ấn → ít/khác xe VN |
| Mật độ xe máy VN-like | DriveIndia, IDD | Góc **dashcam**, không phải CCTV |

→ Giải: mix nhiều nguồn + **fine-tune cuối trên footage Mỗ Lao tự quay**.

**Lưu ý: demo chạy TRONG CARLA** → detector phải nhận diện được CẢ xe render
CARLA (cho demo sim) LẪN xe thật (cho sim→real). Vậy bộ train phải **giữ cả 2
domain** (render hanoi_district3 + dataset thật) — KHÔNG bỏ hẳn CARLA.

Công thức theo tầng (khởi từ `yolo11s.pt`):
1. **COCO pretrain** (giữ generic person/bicycle).
2. **Góc CCTV + vật thể nhỏ dày**: VisDrone (mạnh nhất — trên cao, nhỏ, dày,
   châu Á) + MIO-TCD + UA-DETRAC (chủ yếu lấy car/bus/truck + viewpoint).
3. **Mật độ xe máy VN-like**: DriveIndia (sẵn YOLO format, ~67k ảnh) + IDD;
   augmentation mạnh bù lệch viewpoint dashcam.
4. **Domain CARLA**: render từ hanoi_district3 (sinh tự động) — giữ một phần để
   detector chạy được trong demo sim.
5. **Cảnh tai nạn**: CADP + TU-DAT.
6. **CCTV Mỗ Lao tự quay — bước QUYẾT ĐỊNH cuối** (đóng đồng thời viewpoint +
   density + ngoại hình xe VN).

→ Việc tốn công nhất: **hợp nhất taxonomy** về bộ 5 lớp project
(person / car / motorcycle / bus / truck).

#### ITD — đã nhận MODEL (không phải dataset). File: `AI_custom/best_xl_ITD_v1.2.pt`

Sau khi request, nhận được **trọng số `yolo11x`** train sẵn trên ITD (Indian
Traffic Dataset, camera tĩnh, 8 lớp: `two wheeler, autorickshaw, car, bus, LCV,
truck, bicycle, pedestrian`) — **KHÔNG có ảnh+nhãn gốc** → không trộn ảnh ITD
vào tập train được. Hệ quả & cách dùng:

- ⚠️ `yolo11x` là biến thể **NẶNG NHẤT** → KHÔNG deploy làm detector runtime
  (mâu thuẫn mục tiêu FPS). Chỉ dùng **offline**.
- ⭐ **Auto-label / pseudo-label**: chạy model này lên footage Mỗ Lao + render
  CARLA (chưa nhãn) → sinh bbox → QA tay → train `yolo11s`. Đây là cách thay
  thế (một phần) việc thiếu ảnh ITD: model đã "gói" kiến thức fixed-cam + xe
  máy dày, áp lên data của chính project.
- **Knowledge distillation**: yolo11x (teacher) → yolo11s (student).
- **Baseline/benchmark** trong báo cáo.
- ⚠️ Train trên camera **Ấn Độ** → vẫn gap với VN (xe/góc) và gap LỚN với render
  CARLA → pseudo-label CÓ lỗi, **phải QA tay**. License **CC BY-NC 4.0** (chỉ
  nghiên cứu, phải cite — Agarwal et al., COMSNETS 2024).
- Remap 8→5: `two wheeler|autorickshaw → motorcycle`; `car → car`; `bus → bus`;
  `LCV|truck → truck`; `pedestrian → person`; `bicycle → bỏ`.
- **Smoke-test trước**: chạy model lên 1 clip Hà Nội để đo nhanh độ transfer
  sang VN trước khi đầu tư auto-label cả bộ.

### 3.2 Trajectory predictor — 3 tầng

1. **Xương sống (luồng bình thường)**: inD/rounD/highD (BEV top-down — *khớp
   hoàn hảo* với MapView top-down + tọa độ world của project) + INTERACTION +
   Apolloscape/TrafficPredict (Trung Quốc, đa loại xe, gần tính hỗn hợp VN).
2. **Hậu tai nạn**: DoTA (giá trị nhất — chia sẵn 3 pha tiền sự cố / khung tai
   nạn / hậu tai nạn) + DAD (có xe máy đâm xe máy) + TU-DAT (camera ven đường +
   sẵn quỹ đạo + loại va chạm, trộn thật/BeamNG — sát use-case nhất).
3. **Synthetic của chính project**: CARLA hanoi_district3 (đã có map) + CCTV VN.

---

## 4. Về dữ liệu cần thu thập thêm

Xếp theo ưu tiên với mục tiêu hậu-tai-nạn:

1. **Quỹ đạo hậu va chạm synthetic (MỚI, quan trọng nhất — dữ liệu thật gần như
   không có)**: project đã có sẵn `modules/scenario_controller.py` với 5 kịch
   bản tai nạn. Mở rộng `collect_trajectory_data.py` để **chạy các kịch bản này
   trên hanoi_district3 và log riêng quỹ đạo erratic sau va chạm** → cách "chế
   tạo" hàng loạt tình huống hậu tai nạn. Bổ sung BeamNG/CARLA physics nếu cần
   chính xác vật lý hơn.
2. **Synthetic luồng bình thường** trên hanoi_district3: 4 class
   (motorcycle/car/bus/truck), có **lane_id + goal**, horizon dài (MUST §6.1).
   Collector hiện chỉ log `x,y,vx,vy`, 3 class, map cũ → cần nâng cấp.
3. **Video thật tại ngã tư Mỗ Lao** (MUST §6.2): 2–3h, nhiều khung giờ, camera
   trên cao. Dùng cho (a) calibrate goal classifier — phân phối rẽ thật của xe
   VN; (b) fine-tune cuối YOLO — đóng domain gap.
4. **Clip bất thường/near-miss thật**: hiếm → cần module anomaly lọc từ footage
   dài (tham khảo NIDB near-miss).
5. **Calibration pixel→lat/lon**: ~4 điểm mốc/vị trí camera, thủ công 1 lần
   (`Map/georeference.py` world↔lat/lon đã có).
6. (Nếu mở rộng incident) nhãn **mũ bảo hiểm/vi phạm** — sinh từ IDD.

---

## 5. Về cách demo

Tận dụng hạ tầng đã có (dashboard React, scenario panel, evidence 30s, MapView
Leaflet trong kế hoạch). 5 lớp demo từ dễ → thuyết phục:

1. **Demo lõi (CARLA hanoi_district3, tái lập 100%)**: trigger 1 kịch bản tai
   nạn → phát hiện va chạm → đánh dấu accident marker → **dự đoán quỹ đạo hậu
   tai nạn của xe gây sự cố (nhánh vật lý)** + **đường né/đổi hướng của xe lân
   cận (goal-conditioned + obstacle)** → vẽ **multi-modal paths kèm xác suất**
   lên MapView (tile OSM thật Mỗ Lao qua georeference). Demo "đinh".
2. **Demo trên video thật**: chạy pipeline trên footage Mỗ Lao (hoặc clip ngã
   tư Hà Nội có sẵn) → chứng minh transfer sim→real.
3. **Demo "đường tẩu thoát hit-and-run"**: xe gây tai nạn bỏ chạy → evidence
   package (ring buffer 30s) + predicted escape route overlay.
4. **Side-by-side sim ↔ thật cùng 1 ngã tư**: georeference làm tọa độ so sánh
   được → minh chứng "dễ chuyển sang đời thực".
5. **Benchmark offline (cho báo cáo)**: đánh giá predictor trên
   inD/INTERACTION/DoTA bằng **ADE/FDE, minADE_k/minFDE_k, miss rate** + cột so
   sánh với Trajectron++/SOTA trong literature; với tai nạn báo thêm
   **time-to-detect + độ chính xác hướng tẩu thoát**; kèm biểu đồ **calibration**
   xác suất.

---

## 6. Một dòng định hướng

Đừng "bê nguyên" SOTA — máy không tải nổi và VN phá vỡ giả định lane. Hướng đi:
**giữ pipeline nhẹ goal-conditioned, tách riêng nhánh vật lý cho hậu-tai-nạn,
fine-tune lại bằng dataset camera giám sát THẬT (không phải render CARLA), và
"chế" dữ liệu hậu va chạm bằng `scenario_controller` có sẵn**. SOTA nặng chỉ
xuất hiện ở cột benchmark trong báo cáo.

---

## 7. Liên hệ tài liệu khác

- `MUST.md` — tài liệu pivot gốc (map Hà Nội + dự đoán xác suất). Tài liệu này
  bổ sung góc "hậu tai nạn" + dataset thật cho §3.2/§6 của MUST.
- `error.md` — domain gap detector (lý do cần đổi sang dataset thật ở §3.1).
- `project_context.md` — kiến trúc pipeline hiện tại.
- `Map/` — hanoi_district3.xodr, georeference.py, traffic_light_sim.py (Giai
  đoạn 1 đã xong).
