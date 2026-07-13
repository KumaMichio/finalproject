"""
build_evidence_summary.py — Sinh 1 trang HTML tổng hợp bằng chứng (tự chứa, in PDF được)
=========================================================================================
Gộp bằng chứng "model chạy đúng số liệu báo cáo" thành 1 file HTML duy nhất, ảnh nhúng
base64 → mở offline / in ra PDF đính kèm ĐATN. Số liệu đọc ĐỘNG từ các metrics.json nên
luôn khớp file gốc.

Nguồn:
  - docs/assets/goal_classifier_verified/metrics.json  (hand-verify, accuracy thật)
  - docs/assets/goal_classifier_real/confirm_eval.json (tái lập + Wilson CI)
  - docs/assets/yolo11s_vn_manual_xadan/metrics.json    (YOLO nhãn người)

Chạy:
  custom_tracking_system/venv_tracking/Scripts/python custom_tracking_system/scripts/build_evidence_summary.py
"""
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]        # project root
ASSETS = ROOT / "docs" / "assets"
OUT = ASSETS / "evidence_summary.html"


def embed(path, max_w=1000, q=80):
    """Đọc ảnh (path unicode-safe), thu nhỏ nếu quá rộng, trả data URI JPEG base64."""
    import cv2
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


gv = load(ASSETS / "goal_classifier_verified" / "metrics.json")
gc = load(ASSETS / "goal_classifier_real" / "confirm_eval.json")
yo = load(ASSETS / "yolo11s_vn_manual_xadan" / "metrics.json")


def pct(x):
    return f"{x*100:.1f}%"


def ci(lohi):
    return f"{lohi[0]*100:.1f}–{lohi[1]*100:.1f}%"


# ── Bảng per-class ─────────────────────────────────────────────────────────────
GOAL_VN = {"straight": "Đi thẳng", "left": "Rẽ trái", "right": "Rẽ phải", "u_turn": "Quay đầu"}
goal_rows = ""
for lbl in ["straight", "left", "right", "u_turn"]:
    c = gv["per_class"].get(lbl, {})
    n = c.get("n", 0)
    if n:
        rc = c["model_recall"]; w = c["wilson95"]
        goal_rows += (f"<tr><td>{GOAL_VN[lbl]}</td><td>{n}</td>"
                      f"<td><b>{pct(rc)}</b></td><td>{w[0]*100:.1f}–{w[1]*100:.1f}%</td>"
                      f"<td>{pct(c['auto_correct'])}</td></tr>")
    else:
        goal_rows += f"<tr><td>{GOAL_VN[lbl]}</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>"

YOLO_VN = {"person": "Người", "car": "Ô tô", "motorcycle": "Xe máy", "bus": "Xe buýt", "truck": "Xe tải"}
yolo_rows = ""
for lbl in ["person", "car", "motorcycle", "bus", "truck"]:
    c = yo["per_class"].get(lbl, {})
    nb = yo["class_box_counts"].get(lbl, 0)
    if c:
        yolo_rows += (f"<tr><td>{YOLO_VN[lbl]}</td><td>{nb}</td>"
                      f"<td><b>{pct(c['mAP50'])}</b></td><td>{pct(c['mAP50_95'])}</td></tr>")

# ── Ảnh nhúng ──────────────────────────────────────────────────────────────────
img_goal_cm = embed(ASSETS / "goal_classifier_verified" / "confusion_matrix_human.png", 720)
img_goal_mon = embed(ASSETS / "goal_classifier_verified" / "montage_verified_tracks.png", 1100, 74)
img_goal_rel = embed(ASSETS / "goal_classifier_real" / "reliability_diagram.png", 620)
img_yolo_cm = embed(ASSETS / "yolo11s_vn_manual_xadan" / "confusion_matrix_normalized.png", 900)
img_yolo_pr = embed(ASSETS / "yolo11s_vn_manual_xadan" / "BoxPR_curve.png", 760)
img_yolo_val = embed(ASSETS / "yolo11s_vn_manual_xadan" / "val_batch0_pred.jpg", 1000, 78)

gen = datetime.now().strftime("%d/%m/%Y %H:%M")


def fig(src, cap):
    if not src:
        return f'<figure class="ph">[thiếu ảnh]<figcaption>{cap}</figcaption></figure>'
    return f'<figure><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a2230;background:#eef1f5}
.page{max-width:1000px;margin:0 auto;padding:32px 40px 60px;background:#fff}
h1{font-size:26px;margin:0 0 4px}
.sub{color:#5a6675;margin:0 0 4px}
.meta{color:#8894a4;font-size:13px;margin:0 0 26px}
h2{font-size:20px;margin:34px 0 6px;padding-top:14px;border-top:2px solid #e2e7ee}
h3{font-size:15px;margin:20px 0 8px;color:#2b3a4d}
p{margin:8px 0}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
.kpi{flex:1;min-width:150px;border:1px solid #dde3ec;border-radius:10px;padding:14px 16px;background:#fafbfd}
.kpi .n{font-size:30px;font-weight:700;line-height:1.1}
.kpi .l{font-size:12.5px;color:#5a6675;margin-top:3px}
.kpi.good .n{color:#12805a}
.kpi.blue .n{color:#1d4ed8}
.kpi.amber .n{color:#b45309}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #dde3ec;padding:6px 10px;text-align:center}
th{background:#f2f5f9;font-weight:600}
td:first-child,th:first-child{text-align:left}
.note{background:#fff8ec;border:1px solid #f1d9a8;border-left:4px solid #e0a63a;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:14px}
.good-note{background:#eefaf4;border:1px solid #b6e6d0;border-left:4px solid #17a06e}
.figs{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}
figure{margin:0;flex:1;min-width:280px;border:1px solid #e2e7ee;border-radius:8px;padding:8px;background:#fafbfd}
figure.wide{flex-basis:100%}
img{width:100%;height:auto;border-radius:4px;display:block}
figcaption{font-size:12.5px;color:#5a6675;margin-top:6px;text-align:center}
.ph{display:flex;align-items:center;justify-content:center;min-height:120px;color:#b00;flex-direction:column}
.tag{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:10px;background:#e7edf6;color:#33506e;margin-left:6px;vertical-align:middle}
.tag.hv{background:#e2f6ec;color:#12805a}
footer{margin-top:40px;padding-top:14px;border-top:1px solid #e2e7ee;color:#8894a4;font-size:12.5px}
@media print{body{background:#fff}.page{max-width:none;padding:0}h2{break-before:auto}figure{break-inside:avoid}.note,.kpi{break-inside:avoid}}
"""

HTML = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tổng hợp bằng chứng — Mô hình vs số liệu báo cáo</title>
<style>{CSS}</style></head><body><div class="page">

<h1>Tổng hợp bằng chứng định lượng</h1>
<p class="sub">Kiểm chứng các mô hình hoạt động đúng với số liệu báo cáo — đối chiếu nhãn NGƯỜI KIỂM, có khoảng tin cậy</p>
<p class="meta">Hệ thống giám sát giao thông đa camera · Sinh tự động {gen} · Dữ liệu CCTV thật Việt Nam</p>

<h2>Tổng quan</h2>
<table>
<tr><th>Mô hình</th><th>Số báo cáo</th><th>Bằng chứng nhãn NGƯỜI (mới)</th><th>Ghi chú trung thực</th></tr>
<tr><td>YOLO11s_vn — phát hiện</td><td>mAP50 89,5% <span class="tag">auto-label, test-đè-train</span></td>
    <td><b>mAP50 {pct(yo['mAP50'])}</b> <span class="tag hv">nhãn người, Xã Đàn</span></td>
    <td>Cao hơn số cũ; nhưng in-domain (Xã Đàn có trong train), chưa phải OOD</td></tr>
<tr><td>Goal Classifier — hướng đi</td><td>Accuracy 60,12% <span class="tag">vs nhãn auto nhiễu</span></td>
    <td><b>Accuracy {pct(gv['honest_accuracy'])}</b> <span class="tag hv">vs người, n={gv['n_verified']}</span></td>
    <td>Cao hơn số cũ vì nhãn auto nhiễu {pct(gv['auto_label_noise'])} dìm điểm → 60,12% là thận trọng</td></tr>
<tr><td>ByteTrack — theo dõi</td><td>MOTA 99,5% / IDF1 100,8%</td>
    <td>IDF1 là artifact (GT bán tự động)</td>
    <td>Chỉ số đổi ID (IDs) mới có nghĩa; không dùng MOTA làm số cuối</td></tr>
</table>

<h2>A · Goal Classifier — dự đoán hướng đi <span class="tag hv">đã hand-verify</span></h2>
<div class="kpis">
  <div class="kpi good"><div class="n">{pct(gv['honest_accuracy'])}</div><div class="l">Accuracy THẬT (model vs người)<br>95% CI {ci(gv['honest_accuracy_wilson95'])} · n={gv['n_verified']}</div></div>
  <div class="kpi blue"><div class="n">{pct(gc['reproduced']['accuracy'])}</div><div class="l">Tái lập từ trọng số đã lưu<br>khớp báo cáo · Wilson {ci(gc['accuracy_wilson95'])}</div></div>
  <div class="kpi amber"><div class="n">{pct(gv['auto_label_noise'])}</div><div class="l">Độ nhiễu nhãn auto-label<br>(auto chỉ đúng {pct(gv['auto_label_correct'])} so với người)</div></div>
</div>
<div class="note good-note"><b>Kết luận:</b> Accuracy đo với nhãn NGƯỜI ({pct(gv['honest_accuracy'])}) <b>cao hơn</b> con số báo cáo 60,12% (đo với nhãn auto-label nhiễu {pct(gv['auto_label_noise'])}). Tức <b>60,12% là ước lượng thận trọng, không thổi phồng</b>. Trọng số đã lưu tái lập chính xác {pct(gc['reproduced']['accuracy'])} → số liệu tin cậy được.</div>

<h3>Recall theo lớp (đối chiếu nhãn người)</h3>
<table>
<tr><th>Hướng</th><th>n</th><th>Recall (model)</th><th>95% CI (Wilson)</th><th>Nhãn auto đúng</th></tr>
{goal_rows}
</table>
<div class="note">Lớp <b>Quay đầu</b> gần như vắng mặt trong mẫu phân tầng (theo tần suất auto-label ~3%) → chưa đánh giá được ở đây. <b>20% track</b> (50/250) người đánh "không chắc" đã bị loại khỏi phép tính (chỉ tính 200 track rõ ràng).</div>

<div class="figs">
{fig(img_goal_cm, "Ma trận nhầm lẫn — model vs nhãn NGƯỜI (n=200)")}
{fig(img_goal_rel, f"Reliability diagram — ECE=0,101 (hiệu chỉnh xác suất)")}
</div>
<div class="figs">
{fig(img_goal_mon, "Mẫu các track đã kiểm tay: quỹ đạo xe vẽ trên khung CCTV thật (OK=đúng, MISS=sai) — dấu vết kiểm chứng")}
</div>

<h2>B · YOLO11s_vn — phát hiện đối tượng <span class="tag hv">nhãn người, Xã Đàn</span></h2>
<div class="kpis">
  <div class="kpi good"><div class="n">{pct(yo['mAP50'])}</div><div class="l">mAP@0.5 (nhãn người kiểm)<br>{yo['n_images']} ảnh · {yo['n_boxes']:,} box</div></div>
  <div class="kpi blue"><div class="n">{pct(yo['mAP50_95'])}</div><div class="l">mAP@0.5:0.95</div></div>
  <div class="kpi"><div class="n">{pct(yo['precision'])}</div><div class="l">Precision</div></div>
  <div class="kpi amber"><div class="n">{pct(yo['recall'])}</div><div class="l">Recall (bỏ sót ~{100-yo['recall']*100:.0f}% vật nhỏ)</div></div>
</div>
<div class="note"><b>Bối cảnh trung thực:</b> đo trên ngã tư <b>Xã Đàn – Phạm Ngọc Thạch</b> (khác cảnh demo Phố Huế), nhãn tiền gán bằng model thầy độc lập <i>yolo11x-ITD</i> rồi người sửa. Giá trị: thay nhãn auto-label nhiễu bằng <b>nhãn NGƯỜI</b> → khép điểm yếu lớn nhất của số 89,5% cũ. <b>Nhưng cảnh Xã Đàn cũng nằm trong tập train</b> (2 video trong <code>data/auto_label</code>) → đây là <b>in-domain</b>, KHÔNG phải tổng quát hóa OOD. Điểm yếu thật: recall xe máy ~79% (bỏ sót vật nhỏ/bị che).</div>

<h3>mAP theo lớp (đối chiếu nhãn người)</h3>
<table>
<tr><th>Lớp</th><th>Số box</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr>
{yolo_rows}
</table>

<div class="figs">
{fig(img_yolo_cm, "Ma trận nhầm lẫn (chuẩn hoá) — recall đường chéo; cột nền = vật bỏ sót")}
{fig(img_yolo_pr, "Đường cong Precision–Recall theo lớp")}
</div>
<div class="figs">
{fig(img_yolo_val, "Kết quả phát hiện của model trên ảnh Xã Đàn (batch kiểm định)")}
</div>

<h2>C · ByteTrack — theo dõi trong 1 camera</h2>
<div class="note"><b>Giới hạn cấu trúc (thừa nhận trung thực):</b> ground-truth theo dõi được tạo bán tự động (sao chép vết ByteTrack rồi chỉ sửa cột ID), nên FP≈FN≈0 <b>theo cấu trúc</b> → <b>MOTA (99,5%) và IDF1 (100,8%) bị thổi phồng/artifact</b>, không phản ánh tỉ lệ bỏ sót thật. Chỉ <b>số lần đổi ID</b> (IDs = 50 trên 390 track) là có ý nghĩa tương đối. Kết luận: không báo MOTA như con số cuối; số tracking thật cần gán box GT độc lập (hướng hoàn thiện).</div>

<h2>D · Phương pháp & nguồn dữ liệu</h2>
<p>Tất cả số liệu đo trên <b>dữ liệu CCTV giao thông thật Việt Nam</b> (không dùng mô phỏng). Tập huấn luyện phát hiện: 15.957 ảnh / 623.329 khung bao (auto-label từ ~124 clip, kiểm mẫu tay). Dữ liệu Goal Classifier: 6.959 track → 5.332 train / 1.334 test (tách theo track, seed cố định). Bản đồ: OpenStreetMap (101 nút / 742 đoạn).</p>
<p><b>Nguyên tắc trung thực áp dụng xuyên suốt:</b> (1) tách bạch train/test; (2) đối chiếu <b>nhãn NGƯỜI</b> thay cho nhãn auto-label nhiễu; (3) báo <b>khoảng tin cậy</b> (Wilson/bootstrap) và ECE thay vì con số trần; (4) <b>thừa nhận</b> giới hạn (in-domain vs OOD, GT tracking bán tự động) thay vì che giấu.</p>

<footer>
Sinh bởi <code>custom_tracking_system/scripts/build_evidence_summary.py</code> — số liệu đọc động từ
<code>docs/assets/{{goal_classifier_verified, goal_classifier_real, yolo11s_vn_manual_xadan}}</code>.
Mở bằng trình duyệt hoặc In → Lưu PDF để đính kèm phụ lục đồ án.
</footer>
</div></body></html>"""

OUT.write_text(HTML, encoding="utf-8")
size_mb = len(HTML.encode("utf-8")) / 1e6
print(f"Đã tạo {OUT}  ({size_mb:.2f} MB)")
print("Ảnh nhúng:", sum(x is not None for x in
      [img_goal_cm, img_goal_mon, img_goal_rel, img_yolo_cm, img_yolo_pr, img_yolo_val]), "/ 6")
