"""
Chuan bi video demo cho run_demo.bat.

Video CCTV that (Pho Hue - Tran Khat Chan) nam trong thu muc co dau tieng Viet
(vd ".../Phố Huế Trần Khát Chân/12h.10.9.22.mp4"). cmd.exe xu ly path unicode
rat kem, nen ta copy san video sang mot path ASCII on dinh:

    data/demo_pho_hue.mp4

Python doc/ghi path unicode chuan xac, khong phu thuoc codepage cua cmd, nen buoc
copy nay lam trong Python thay vi trong .bat.

Chay:  python scripts/prepare_demo_video.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Path co dau tieng Viet -> tranh UnicodeEncodeError khi in ra console cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "demo_pho_hue.mp4"

# Uu tien video da dung khop voi bao cao (Hinh 4.6/4.7): Pho Hue 12h.10.9.22.mp4
PREFERRED_NAME = "12h.10.9.22.mp4"
SEARCH_DIR = (
    ROOT
    / "custom_tracking_system"
    / "data"
    / "datasets"
    / "traffic_intersection"
)


def find_demo_video() -> Path | None:
    if not SEARCH_DIR.is_dir():
        return None
    # 1) Uu tien dung ten video Pho Hue trong bao cao, trong thu muc co "Huế"
    for p in SEARCH_DIR.rglob(PREFERRED_NAME):
        if "Hu" in p.parent.name:  # "Phố Huế ..."
            return p
    # 2) Bat ky video nao trong thu muc Pho Hue
    for folder in SEARCH_DIR.rglob("*"):
        if folder.is_dir() and "Hu" in folder.name:
            vids = sorted(folder.glob("*.mp4"))
            if vids:
                return vids[0]
    # 3) Fallback: video .mp4 dau tien tim thay
    vids = sorted(SEARCH_DIR.rglob("*.mp4"))
    return vids[0] if vids else None


def main() -> int:
    DST.parent.mkdir(parents=True, exist_ok=True)

    if DST.exists() and DST.stat().st_size > 0:
        print(f"[OK] Video demo da san sang: {DST}")
        return 0

    src = find_demo_video()
    if src is None:
        print(
            "[MISSING] Khong tim thay video CCTV that trong\n"
            f"          {SEARCH_DIR}\n"
            "          Dat mot file .mp4 vao data/demo_pho_hue.mp4 roi chay lai.",
            file=sys.stderr,
        )
        return 1

    print(f"[COPY] {src}\n    -> {DST}")
    shutil.copy(src, DST)
    print("[OK] Xong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
