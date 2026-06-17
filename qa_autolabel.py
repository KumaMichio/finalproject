from pathlib import Path
from collections import defaultdict

label_dir = Path("custom_tracking_system/data/auto_label/labels")
CLASS_NAMES = ["person", "car", "motorcycle", "bus", "truck"]

errors = []
tiny_bbox = []
empty_files = []
class_counts = defaultdict(int)
total_boxes = 0
files_checked = 0

for f in label_dir.glob("*.txt"):
    files_checked += 1
    lines = f.read_text(encoding="utf-8").strip().splitlines()

    if not lines:
        empty_files.append(f.name)
        continue

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append((f.name, f"wrong_cols({len(parts)})", line[:60]))
            continue
        try:
            cls_id, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append((f.name, "parse_error", line[:60]))
            continue

        total_boxes += 1
        cls_id_int = int(cls_id)
        if 0 <= cls_id_int < len(CLASS_NAMES):
            class_counts[CLASS_NAMES[cls_id_int]] += 1
        else:
            errors.append((f.name, f"invalid_class({cls_id_int})", line[:60]))
            continue

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append((f.name, "out_of_bounds", line[:60]))

        area = w * h
        if area < 0.0005:
            tiny_bbox.append((f.name, round(w, 4), round(h, 4), round(area, 6)))

print("=" * 55)
print(f"Files checked : {files_checked:,}")
print(f"Total boxes   : {total_boxes:,}")
print(f"Empty files   : {len(empty_files)}")
print(f"Format errors : {len(errors)}")
print(f"Tiny bbox     : {len(tiny_bbox)}  (w*h < 0.0005)")
print()
print("Class distribution:")
for cls in CLASS_NAMES:
    n = class_counts[cls]
    pct = n / total_boxes * 100 if total_boxes else 0
    bar = "#" * int(pct / 2)
    print(f"  {cls:<12} {n:>7,}  {pct:5.1f}%  {bar}")
print()
if errors:
    print(f"First 10 errors:")
    for name, issue, line in errors[:10]:
        print(f"  [{issue}] {name}: {line}")
else:
    print("No format errors found.")
print()
if tiny_bbox:
    print(f"Tiny bbox examples (first 10):")
    for name, w, h, area in tiny_bbox[:10]:
        print(f"  {name}  w={w} h={h} area={area}")
    # breakdown by size bucket
    under_100 = sum(1 for _, w, h, a in tiny_bbox if a < 0.0001)
    print(f"\n  Breakdown:")
    print(f"    area < 0.0001 (rất nhỏ): {under_100:,}")
    print(f"    area 0.0001-0.0005     : {len(tiny_bbox)-under_100:,}")
print("=" * 55)
